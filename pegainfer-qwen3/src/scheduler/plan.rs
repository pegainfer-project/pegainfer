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
    // echo+logprobs requests need all-position logits, which the unified forward
    // does not compute (it passes all_position_logits=None). And under DFlash
    // speculation, an eligible request must capture its target hidden context
    // during prefill — the unified forward skips that capture, so a request
    // prefilled via Unified would never become draft-ready and DFlash would
    // silently no-op for it forever. Either way, route pending through a
    // dedicated prefill step instead of degrading silently.
    let needs_prompt_logprobs = pending.iter().any(|r| r.echo && r.logprobs > 0);
    // Deliberately a loose superset of the real capture eligibility
    // (`dflash_prefill_supported`, which also needs `cached_tokens == 0 && !echo`):
    // over-routing an ineligible request to a dedicated prefill only costs one
    // fusion, but under-routing a capture-eligible one into Unified would silently
    // break its readiness. Never tighten this into the dangerous direction.
    let needs_dflash_capture = speculative
        && pending
            .iter()
            .any(|r| r.lora_adapter.is_none() && r.logprobs == 0);
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
            let any_echo = pending.iter().any(|req| req.echo);
            let mut result = executor.execute_prefill(PrefillPlan {
                requests: &requests,
                echo: any_echo,
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

/// All-or-nothing: speculate the whole active batch only when every request is
/// draft-ready (no LoRA, no logprobs). A single non-ready request falls the
/// batch back to plain decode rather than running a mixed step. Sampling
/// params are NOT a gate: sampled-verify (#512) runs the regular sampler over
/// the verify rows, so the full surface — temperature/top_k/top_p/min_p/seed —
/// rides the speculative path.
pub(crate) fn should_speculative_decode(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
) -> bool {
    executor.speculative_enabled()
        && !active.is_empty()
        && active.iter().all(|req| {
            executor.speculative_request_ready(req.request_id)
                && req.lora_adapter.is_none()
                && req.logprobs == 0
        })
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
    draft_results
        .iter()
        .map(|draft| {
            let active = active
                .iter()
                .find(|req| req.request_id == draft.request_id)
                .expect("draft request_id must exist in active set");
            // Clamp the verify span to the request's remaining output budget so
            // a long accepted run can't overshoot max_tokens.
            let remaining = active.max_tokens.saturating_sub(active.generated_count);
            // A continuing active request always has budget left (resolve emits
            // EmitManyAndFinish the moment generated_count hits max_tokens), so
            // this is a true invariant, not a runtime condition — don't crash the
            // scheduler thread in release on a state we've proven unreachable.
            debug_assert!(remaining > 0, "active request must have output budget");
            let mut token_ids = draft.token_ids.clone();
            token_ids.truncate(remaining);
            VerifyStepItem::new(draft.request_id, token_ids, active.params)
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
                echo: r.echo,
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
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;
    use crate::executor::RequestId;

    fn pending() -> PendingRequest {
        PendingRequest {
            request_id: RequestId::new(0),
            lora_adapter: None,
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            max_tokens: 8,
            logprobs: 0,
            echo: false,
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
            logprobs: 0,
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
        // echo+logprobs requests need all-position logits; route to Prefill
        // even when decodes are active so prompt logprobs are not silently lost.
        let mut echo_req = pending();
        echo_req.echo = true;
        echo_req.logprobs = 5;
        assert!(
            matches!(
                build_next_plan(true, vec![echo_req], false),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "active + pending echo+logprobs request routes to prefill not unified"
        );
        // echo without logprobs (no prompt logprobs needed) can still use unified.
        let mut echo_no_lp = pending();
        echo_no_lp.echo = true;
        assert!(
            matches!(
                build_next_plan(true, vec![echo_no_lp], false),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending echo-only request (no logprobs) can use unified"
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
