use anyhow::Result;
use rand::rngs::StdRng;

use super::ActiveRequestState;
use super::PendingRequest;
use crate::executor::DecodePlan;
use crate::executor::DecodeResult;
use crate::executor::DecodeStepItem;
use crate::executor::ModelExecutor;
use crate::executor::PrefillPlan;
use crate::executor::PrefillResult;
use crate::executor::PrefillStepItem;
use crate::executor::UnifiedPlan;
use crate::executor::UnifiedResult;
use crate::speculative::DraftPlan;
use crate::speculative::DraftRequestResult;
use crate::speculative::DraftStepItem;
use crate::speculative::VerifyPlan;
use crate::speculative::VerifyResult;
use crate::speculative::VerifyStepItem;

pub(crate) enum ExecutionPlan {
    Prefill {
        pending: Vec<PendingRequest>,
    },
    Decode,
    /// Draft + verify the whole active batch (all requests are draft-ready).
    SpeculativeDecode,
    /// Verify the whole active batch over a one-token span — the load gate
    /// turned speculation off for this step. No draft is proposed, so the
    /// target commits one token per request, but the DFlash capture and the
    /// draft-ready state survive: the next round under the threshold
    /// speculates again. A plain decode would drop the capture, which only a
    /// fresh prefill rebuilds.
    GatedSpeculativeDecode,
    Unified {
        pending: Vec<PendingRequest>,
    },
}

pub(crate) enum ExecutionArtifacts {
    Prefill {
        pending: Vec<PendingRequest>,
        result: PrefillResult,
    },
    Decode {
        result: DecodeResult,
    },
    SpeculativeDecode {
        verify: VerifyResult,
    },
    Unified {
        pending: Vec<PendingRequest>,
        result: UnifiedResult,
    },
}

pub(crate) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
    speculative: bool,
) -> Option<ExecutionPlan> {
    // Prompt-logprobs requests need all-position logits, which the unified
    // forward does not compute (it passes all_position_logits=None). And under
    // DFlash speculation, an eligible request must capture its target hidden
    // context during prefill — the unified forward skips that capture, so a
    // request prefilled via Unified would never become draft-ready and DFlash
    // would silently no-op for it forever. Either way, route pending through a
    // dedicated prefill step instead of degrading silently.
    let needs_prompt_logprobs = pending.iter().any(|req| req.prompt_logprobs.is_some());
    // Deliberately a loose superset of the real capture eligibility
    // (`dflash_prefill_supported`, which also needs `cached_tokens == 0 && no
    // prompt logprobs`): over-routing an ineligible request to a dedicated
    // prefill only costs one fusion, but under-routing a capture-eligible one
    // into Unified would silently break its readiness. Never tighten this into
    // the dangerous direction.
    let needs_dflash_capture = speculative
        && pending
            .iter()
            .any(|r| r.lora_adapter.is_none() && r.logprobs.is_none());
    if !pending.is_empty() && have_active && !needs_prompt_logprobs && !needs_dflash_capture {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(crate) fn execute_plan(
    executor: &mut impl ModelExecutor,
    active: &mut [ActiveRequestState],
    plan: ExecutionPlan,
    rng: &mut StdRng,
) -> Result<ExecutionArtifacts> {
    match plan {
        ExecutionPlan::Prefill { pending } => {
            let indices: Vec<usize> = (0..pending.len()).collect();
            let requests = build_prefill_items(&pending, &indices);
            let mut result = executor.execute_prefill(PrefillPlan {
                requests: &requests,
                sample_seed: rand::RngExt::random(rng),
            })?;
            sort_prefill_results(&mut result.requests);
            Ok(ExecutionArtifacts::Prefill { pending, result })
        }
        ExecutionPlan::Decode => {
            let indices: Vec<usize> = (0..active.len()).collect();
            let requests = build_decode_items(active, &indices);
            let mut result = executor.execute_decode(DecodePlan {
                requests: &requests,
                sample_seed: rand::RngExt::random(rng),
            })?;
            sort_decode_results(&mut result.requests);
            Ok(ExecutionArtifacts::Decode { result })
        }
        ExecutionPlan::SpeculativeDecode => {
            // Two executor calls per step: draft proposes K tokens per request,
            // then a single target forward verifies the K+1 span. Both index by
            // request_id; sorting keeps draft and verify results aligned.
            let draft_requests = build_speculative_draft_items(active);
            let mut draft = executor.execute_speculative_draft(DraftPlan {
                requests: &draft_requests,
            })?;
            draft.requests.sort_by_key(|result| result.request_id);
            let verify_requests = build_speculative_verify_items(active, &draft.requests);
            let mut verify = executor.execute_speculative_verify(VerifyPlan {
                requests: &verify_requests,
                sample_seed: rand::RngExt::random(rng),
                gated: false,
            })?;
            verify.requests.sort_by_key(|result| result.request_id);
            Ok(ExecutionArtifacts::SpeculativeDecode { verify })
        }
        ExecutionPlan::GatedSpeculativeDecode => {
            let verify_requests = build_gated_verify_items(active);
            let mut verify = executor.execute_speculative_verify(VerifyPlan {
                requests: &verify_requests,
                sample_seed: rand::RngExt::random(rng),
                gated: true,
            })?;
            verify.requests.sort_by_key(|result| result.request_id);
            Ok(ExecutionArtifacts::SpeculativeDecode { verify })
        }
        ExecutionPlan::Unified { pending } => {
            let pending_indices: Vec<usize> = (0..pending.len()).collect();
            let active_indices: Vec<usize> = (0..active.len()).collect();
            let prefill_requests = build_prefill_items(&pending, &pending_indices);
            let decode_requests = build_decode_items(active, &active_indices);
            let mut result = executor.execute_unified(UnifiedPlan {
                prefill_requests: &prefill_requests,
                decode_requests: &decode_requests,
                sample_seed: rand::RngExt::random(rng),
            })?;
            sort_prefill_results(&mut result.prefill_requests);
            sort_decode_results(&mut result.decode_requests);
            Ok(ExecutionArtifacts::Unified { pending, result })
        }
    }
}

/// `PEGAINFER_SPEC_MAX_BATCH=<n>` turns on the load-aware speculation gate: once
/// the active batch reaches `<n>` requests, the engine stops proposing drafts
/// for that step. `0`/unset = off. Read once; a malformed value refuses to
/// start (see [`crate::spec_env`]).
pub(crate) fn spec_max_batch() -> usize {
    static MAX_BATCH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_BATCH.get_or_init(|| crate::spec_env::read(crate::spec_env::MAX_BATCH))
}

/// What the engine does with the active batch this step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeculationDecision {
    /// Every request is draft-ready and the batch is under the load gate:
    /// draft + verify.
    Speculate,
    /// Every request is draft-ready but the batch reached the load gate.
    Gated,
    /// Some request is not draft-ready (or no drafter is loaded): plain
    /// decode.
    NotReady,
}

/// All-or-nothing: speculate the whole active batch only when every request is
/// draft-ready (no LoRA, no logprobs). A single non-ready request falls the
/// batch back to plain decode rather than running a mixed step. Sampling
/// params are NOT a gate: sampled-verify (#512) runs the regular sampler over
/// the verify rows, so the full surface — temperature/top_k/top_p/min_p/seed —
/// rides the speculative path.
pub(crate) fn speculation_decision_at(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    max_batch: usize,
) -> SpeculationDecision {
    if !executor.speculative_enabled() || active.is_empty() {
        return SpeculationDecision::NotReady;
    }
    let all_ready = active.iter().all(|req| {
        executor.speculative_request_ready(req.request_id)
            && req.lora_adapter.is_none()
            && req.logprobs.is_none()
    });
    if !all_ready {
        return SpeculationDecision::NotReady;
    }
    if max_batch > 0 && active.len() >= max_batch {
        SpeculationDecision::Gated
    } else {
        SpeculationDecision::Speculate
    }
}

fn build_speculative_draft_items(active: &[ActiveRequestState]) -> Vec<DraftStepItem> {
    active
        .iter()
        .map(|r| {
            // Only greedy requests may hedge; the draft lane turns the
            // remaining budget into eligibility against its effective branch
            // positions (a chain must survive the verify-side span clamp).
            let budget = if r.params.is_greedy() {
                r.max_tokens.saturating_sub(r.generated_count)
            } else {
                0
            };
            DraftStepItem::new(r.request_id, r.last_token, budget)
        })
        .collect()
}

fn build_speculative_verify_items(
    active: &[ActiveRequestState],
    draft_results: &[DraftRequestResult],
) -> Vec<VerifyStepItem> {
    let mut requests = Vec::with_capacity(draft_results.len());
    for draft in draft_results {
        let active = active
            .iter()
            .find(|req| req.request_id == draft.request_id)
            .expect("draft request_id must exist in active set");
        // Clamp the verify span to the request's remaining output budget so
        // a long accepted run can't overshoot max_tokens.
        let remaining = active.max_tokens.saturating_sub(active.generated_count);
        // A continuing active request always has budget left (resolve emits
        // FinishMany the moment generated_count hits max_tokens), so
        // this is a true invariant, not a runtime condition — don't crash the
        // scheduler thread in release on a state we've proven unreachable.
        debug_assert!(remaining > 0, "active request must have output budget");
        let mut token_ids = draft.token_ids.clone();
        token_ids.truncate(remaining);
        requests.push(VerifyStepItem::new(
            draft.request_id,
            token_ids,
            active.params,
            active.stop_policy.clone(),
        ));
    }
    requests
}

/// One-token verify spans for a gated round: each request's confirmed last
/// token and nothing else.
fn build_gated_verify_items(active: &[ActiveRequestState]) -> Vec<VerifyStepItem> {
    active
        .iter()
        .map(|req| {
            VerifyStepItem::new(
                req.request_id,
                vec![req.last_token],
                req.params,
                req.stop_policy.clone(),
            )
        })
        .collect()
}

fn build_prefill_items(pending: &[PendingRequest], indices: &[usize]) -> Vec<PrefillStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &pending[index];
            PrefillStepItem {
                request_id: r.request_id,
                prompt_tokens: r.prompt_tokens.clone(),
                max_output_tokens: r.max_tokens,
                params: r.params,
                logprobs: r.logprobs,
                prompt_logprobs: r.prompt_logprobs,
                lora_adapter: r.lora_adapter.clone(),
                cached_tokens: r.cached_tokens,
                chunk_budget: r.step_chunk,
                chunk_start: 0,
                chunk_tokens: 0,
            }
        })
        .collect()
}

fn build_decode_items(active: &[ActiveRequestState], indices: &[usize]) -> Vec<DecodeStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &active[index];
            DecodeStepItem {
                request_id: r.request_id,
                token_id: r.last_token,
                params: r.params,
                logprobs: r.logprobs,
                lora_adapter: r.lora_adapter.clone(),
            }
        })
        .collect()
}

fn sort_prefill_results(results: &mut [crate::executor::PrefillRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

fn sort_decode_results(results: &mut [crate::executor::DecodeRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use pegainfer_frontend::engine::StopPolicy;
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;
    use crate::executor::RequestId;
    use crate::scheduler::test_support::FakeExecutor;

    fn pending() -> PendingRequest {
        PendingRequest {
            request_id: RequestId::new(0),
            lora_adapter: None,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            stop_policy: StopPolicy::default(),
            max_tokens: 8,
            logprobs: None,
            prompt_logprobs: None,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 3,
            cached_tokens: 0,
        }
    }

    fn active(generated_count: usize, max_tokens: usize) -> ActiveRequestState {
        ActiveRequestState {
            request_id: RequestId::new(7),
            lora_adapter: None,
            last_token: 42,
            generated_count,
            max_tokens,
            prompt_len: 10,
            params: SamplingParams::default(),
            stop_policy: StopPolicy::default(),
            logprobs: None,
        }
    }

    #[test]
    fn draft_items_carry_the_remaining_budget_for_greedy_requests_only() {
        let exhausted = active(31, 32);
        let fresh = active(0, 32);
        let mut sampled = active(0, 32);
        sampled.params.temperature = 0.7;
        let items = build_speculative_draft_items(&[exhausted, fresh, sampled]);
        assert_eq!(items[0].hedge_budget, 1);
        assert_eq!(items[1].hedge_budget, 32);
        assert_eq!(items[2].hedge_budget, 0);
    }

    fn speculative_batch(count: usize) -> Vec<ActiveRequestState> {
        (0..count)
            .map(|i| ActiveRequestState {
                request_id: RequestId::new(i as u64 + 1),
                lora_adapter: None,
                last_token: 42,
                generated_count: 0,
                max_tokens: 32,
                prompt_len: 10,
                params: SamplingParams::default(),
                stop_policy: StopPolicy::default(),
                logprobs: None,
            })
            .collect()
    }

    #[test]
    fn a_request_without_draft_context_is_never_gated() {
        // A request that never captured its context cannot run a verify round,
        // so the batch decodes plainly; the gate must not paper over that.
        let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new())))
            .with_speculative()
            .with_request_not_ready(RequestId::new(3));
        let batch = speculative_batch(8);
        assert_eq!(
            speculation_decision_at(&exec, &batch, 4),
            SpeculationDecision::NotReady
        );
    }

    #[test]
    fn speculative_verify_items_clamp_to_remaining_output_budget() {
        let active = [active(24, 32)];
        let draft = DraftRequestResult {
            request_id: RequestId::new(7),
            token_ids: (0..16).collect(),
        };

        let verify = build_speculative_verify_items(&active, &[draft]);

        assert_eq!(verify.len(), 1);
        // 32 - 24 = 8 remaining → the 16-token span truncates to 8.
        assert_eq!(verify[0].as_slice().len(), 8);
        assert_eq!(verify[0].as_slice(), (0..8).collect::<Vec<_>>());
        assert_eq!(verify[0].stop_policy, StopPolicy::default());
    }

    // The plan selector is the whole batch-formation policy: what the scheduler
    // does each tick is fully determined by (have_active, has_pending). Pin the
    // 2×2 truth table so a policy regression can't slip through silently.
    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan(false, vec![], false).is_none(),
            "idle scheduler (no active, no pending) produces no plan"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![], false),
                Some(ExecutionPlan::Decode)
            ),
            "active-only ticks decode the running batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending()], false),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "pending-only prefills the new arrivals"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending()], false),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending fuses prefill and decode into one unified step"
        );
        // Prompt-logprobs requests need all-position logits; route to Prefill
        // even when decodes are active so prompt logprobs are not silently lost.
        let mut scored_req = pending();
        scored_req.prompt_logprobs = Some(5);
        assert!(
            matches!(
                build_next_plan(true, vec![scored_req], false),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "active + pending prompt_logprobs request routes to prefill not unified"
        );
        // prompt_logprobs=0 still needs every position's logits (the scored
        // token's logprob is a full-row reduction), so it routes to Prefill too.
        let mut zero_count_req = pending();
        zero_count_req.prompt_logprobs = Some(0);
        assert!(
            matches!(
                build_next_plan(true, vec![zero_count_req], false),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "active + pending prompt_logprobs=0 request routes to prefill not unified"
        );
        // No prompt logprobs requested → the unified fusion is fine.
        let plain = pending();
        assert!(
            matches!(
                build_next_plan(true, vec![plain], false),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending plain request (no prompt logprobs) can use unified"
        );
        // Under DFlash speculation, an eligible pending must capture its
        // target context during prefill — the unified forward skips that capture,
        // so route it to a dedicated prefill step rather than let DFlash silently
        // no-op for it.
        assert!(
            matches!(
                build_next_plan(true, vec![pending()], true),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "spec + active + eligible pending routes to prefill so the drafter context is captured"
        );
        // Sampled-verify (#512): a non-greedy pending speculates too, so it
        // needs the same capture — Prefill, not Unified (which would silently
        // never make it draft-ready). This covers min_p and seeded params as
        // well; nothing on the sampling surface opts a request out.
        for params in [
            SamplingParams {
                temperature: 1.0,
                ..SamplingParams::default()
            },
            SamplingParams {
                temperature: 0.8,
                min_p: 0.05,
                ..SamplingParams::default()
            },
            SamplingParams {
                temperature: 0.8,
                seed: Some(42),
                ..SamplingParams::default()
            },
        ] {
            let mut sampled = pending();
            sampled.params = params;
            assert!(
                matches!(
                    build_next_plan(true, vec![sampled], true),
                    Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
                ),
                "spec + active + sampled pending routes to prefill for draft capture"
            );
        }
        // A LoRA pending never speculates, so no capture is needed and unified
        // fusion is still fine under speculation.
        let mut lora = pending();
        lora.lora_adapter = Some("adapter".to_string());
        assert!(
            matches!(
                build_next_plan(true, vec![lora], true),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "spec + active + LoRA pending (never speculates, no capture) can use unified"
        );
    }
}
