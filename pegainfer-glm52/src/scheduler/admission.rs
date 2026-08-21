//! Request validation and slot placement for one rank: [`validate_request`]
//! fast-rejects at the door (past it, a bad value only surfaces inside a
//! collective and tears the engine down), then [`admit_from_queue`] fills the
//! rank's free slots from its own FIFO queue at step boundaries under the
//! full-lifetime KV budget. Requests arrive pre-bound to this rank — the
//! `EngineHandle` routes by `data_parallel_rank` (the vLLM frontend's DP
//! choice) and least-load-places unbound ones — so admission never moves a
//! request and metrics/KV ownership agree with the frontend's engine index.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Context as _;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopCause;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::unix_now_s;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::KvPrefix;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::RequestKv;
use pegainfer_kv_store::SaveCursor;

use super::ActiveRequest;
use super::BoundaryCopy;
use super::PAGE;
use super::RankSlots;
use super::offload::Resolved;
use super::offload::{self};
use super::slot::Glm52SlotState;

pub(super) fn validate_request(
    req: &GenerateRequest,
    max_model_len: usize,
    prefill_only: bool,
    native_mtp_prefill: bool,
) -> Result<(), String> {
    if req.prompt_tokens.is_empty() {
        return Err("GLM5.2 requires a non-empty prompt".to_owned());
    }
    if req.max_tokens == 0 {
        return Err("GLM5.2 requires max_tokens > 0".to_owned());
    }
    if prefill_only && req.max_tokens != 1 {
        return Err(format!(
            "GLM5.2 prefill-only mode requires max_tokens=1, got {}",
            req.max_tokens
        ));
    }
    // Highest position any forward step can touch: the (max_tokens-1)-th
    // generated token is fed at position prompt+max_tokens-2, so requiring
    // prompt+max_tokens-1 <= cap keeps every step strictly below the cap.
    let last_position = req.prompt_tokens.len() + req.max_tokens - 1;
    if last_position > max_model_len {
        return Err(format!(
            "GLM5.2 context cap: prompt {} + max_tokens {} exceeds max_model_len {max_model_len}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    if native_mtp_prefill
        && req.prompt_tokens.len() + crate::mtp::glm52_mtp_draft_len() - 1 > max_model_len
    {
        return Err(format!(
            "GLM5.2 native-MTP prefill requires {} positions of proposal headroom: \
             prompt {} exceeds max_model_len {max_model_len}",
            crate::mtp::glm52_mtp_draft_len() - 1,
            req.prompt_tokens.len()
        ));
    }
    // Mirror the sampler kernel's parameter ensures HERE: past intake a bad
    // value only surfaces as a failed step, and a failed step is fatal to the
    // engine — user input must be rejected at the door, never inside a
    // collective.
    if !req.params.is_greedy() {
        let p = &req.params;
        if !p.temperature.is_finite() {
            return Err(format!(
                "GLM5.2 sampling requires a finite temperature, got {}",
                p.temperature
            ));
        }
        if !(p.top_p > 0.0 && p.top_p <= 1.0) {
            return Err(format!(
                "GLM5.2 sampling requires top_p in (0, 1], got {}",
                p.top_p
            ));
        }
        if !(p.min_p.is_finite() && (0.0..1.0).contains(&p.min_p)) {
            return Err(format!(
                "GLM5.2 sampling requires min_p in [0, 1), got {}",
                p.min_p
            ));
        }
    }
    if req.logprobs > 0 || req.echo {
        return Err("GLM5.2 bring-up does not support logprobs/echo".to_owned());
    }
    if req.lora_adapter.is_some() {
        return Err("GLM5.2 does not support LoRA adapters".to_owned());
    }
    Ok(())
}

/// Pool pages a request draws over its whole lifetime, reserved at
/// admission. One more token than the last KV-written position: kvbm appends
/// the final generated token to the sequence and provisions its page even
/// though its KV is never written (the dangling-token contract — the same
/// off-by-one Kimi's admission had to learn empirically).
pub(super) fn lifetime_blocks(prompt_tokens: usize, max_tokens: usize) -> usize {
    (prompt_tokens + max_tokens).div_ceil(PAGE)
}

fn admission_lifetime_blocks(
    req: &GenerateRequest,
    handoff: Option<&offload::NativeMtpHandoff>,
) -> usize {
    let (input_tokens, max_output_tokens) = match handoff {
        Some(handoff) => {
            let shape = offload::native_kv_shape(req, handoff);
            (shape.input_tokens, shape.max_output_tokens)
        }
        None => (req.prompt_tokens.len(), req.max_tokens),
    };
    lifetime_blocks(input_tokens, max_output_tokens)
}

pub(super) fn reject(req: &GenerateRequest, message: String) {
    let prompt_tokens = req.prompt_tokens.len();
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s,
        scheduled_at_unix_s: unix_now_s(),
        prompt_tokens,
        cached_tokens: 0,
    });
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens,
        completion_tokens: 0,
    });
}

pub(super) fn admit_from_queue(
    rank: usize,
    pending: &mut VecDeque<Resolved>,
    slots: &mut RankSlots,
    pool: &Arc<BlockPool>,
    usable_blocks: usize,
    store: &KvStore,
    prefix_cache_enabled: bool,
    drafter_enabled: bool,
    native_mtp_prefill: bool,
    pending_resets: &mut Vec<usize>,
) -> anyhow::Result<()> {
    // Zero-capacity natives finish at intake, before the slot and budget
    // gates — a saturated rank must never delay a reply that needs no
    // capacity. Two forms: P consumed a typed stop token, and the
    // replayed anchor exhausting max_tokens (→ Length); neither restores KV.
    pending.retain(|entry| {
        let Resolved::Native { req, handoff, .. } = entry else {
            return true;
        };
        let (terminal_token, finish_reason, stop_cause) =
            match (handoff.stop_cause, handoff.anchor_token_id) {
                (Some(cause), _) => {
                    let cause = StopCause::from(cause);
                    let token = match cause {
                        StopCause::Eos(id) | StopCause::Token(id) => id,
                    };
                    (Some(token), FinishReason::Stop, Some(cause))
                }
                // Compatibility for a v5 envelope that omitted the optional
                // cause. Its fingerprint prevents mixing with older peers,
                // but retaining the fallback makes malformed metadata fail in
                // the same non-allocating way as the old EOS marker.
                (None, None) => (None, FinishReason::Stop, None),
                (None, Some(anchor)) if req.max_tokens == 1 => {
                    (Some(anchor), FinishReason::Length, None)
                }
                (None, Some(_)) => return true,
            };
        let prompt_tokens = req.prompt_tokens.len();
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s: req.queued_at_unix_s.unwrap_or_else(unix_now_s),
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: terminal_token.map_or(0, |_| handoff.committed_len),
        });
        // P sampled the terminal/anchor token; replay it to this request once.
        if let Some(token) = terminal_token {
            if req
                .token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: None,
                })
                .is_err()
            {
                return false;
            }
        }
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason,
            stop_cause,
            prompt_tokens,
            completion_tokens: 1,
        });
        false
    });
    let mut committed: usize = slots
        .iter()
        .flatten()
        .map(|active| active.kv.lifetime_blocks())
        .sum();
    // Pages pinned by in-flight release saves are physically unallocatable
    // until their D2H lands. Hide them from the rank's full-lifetime budget
    // so admission defers instead of promising pages a later schedule cannot
    // get (which would fail the whole engine).
    let usable = usable_blocks.saturating_sub(store.pinned_blocks(rank));

    // Admission fills only the configured slot count; the fixed array's tail
    // beyond `glm52_decode_slots()` stays permanently empty. The queue holds
    // only RESOLVED intakes — restore waiting happened off-thread, so a
    // front here is never parked on storage.
    while let Some(slot) = slots[..crate::model::glm52_decode_slots()]
        .iter()
        .position(Option::is_none)
    {
        let Some(front) = pending.front() else {
            break;
        };
        // Drop a disconnected FIFO front before it can block valid work
        // behind an admission budget it will never consume (any resolved
        // state — a built KV, a prefix hold — releases via RAII).
        let front_req = match front {
            Resolved::Plain { req, .. }
            | Resolved::Native { req, .. }
            | Resolved::Failed { req, .. } => req,
        };
        if front_req.token_tx.is_closed() {
            drop(pending.pop_front());
            continue;
        }
        if matches!(front, Resolved::Failed { .. }) {
            let Some(Resolved::Failed { req, message }) = pending.pop_front() else {
                unreachable!("front matched Failed");
            };
            reject(&req, message);
            continue;
        }

        // Full-lifetime budget, honor-or-reject. `usable` accounts for the
        // block classes the scheduler knows about; the allocator is the
        // final authority, so add back pages held by active requests and by
        // the front's own prefix hold (already out of the free pool) and
        // defer if the physical lifetime budget is smaller.
        // A native hold credits FULL pages only: an unaligned handoff's
        // padded boundary page is pinned as the copy-on-restore source and
        // never folds into the request's resident set, so counting it would
        // over-admit by one page at exact capacity.
        let (need_blocks, front_held) = match front {
            Resolved::Plain { req, prefix } => (
                admission_lifetime_blocks(req, None),
                prefix.hit_tokens() / PAGE,
            ),
            Resolved::Native { req, handoff, .. } => (
                admission_lifetime_blocks(req, Some(handoff)),
                handoff.committed_len / PAGE,
            ),
            Resolved::Failed { .. } => unreachable!("handled above"),
        };
        let active_resident: usize = slots
            .iter()
            .flatten()
            .map(|active| active.kv.resident_blocks())
            .sum();
        let physical_usable = pool
            .available_blocks()
            .saturating_add(active_resident)
            .saturating_add(front_held);
        if committed + need_blocks > usable.min(physical_usable) {
            // Holds pinned BEHIND a budget-stalled front shrink the free
            // pool with no release path of their own — shed the rearmost
            // Plain one (blocks fall to the evictable, still-matchable
            // pool) and retry. Native holds never shed: eviction of a
            // restored page turns a wait into a terminal reject.
            let mut shed = false;
            for entry in pending.iter_mut().skip(1).rev() {
                if let Resolved::Plain { prefix, .. } = entry {
                    if prefix.hit_tokens() > 0 {
                        *prefix = KvPrefix::none();
                        shed = true;
                        break;
                    }
                }
            }
            if shed {
                continue;
            }
            // A native queued behind the stalled front pins its restored
            // pages until its own admission — the only transition that ever
            // releases them (eviction of a restored page would be a terminal
            // reject). Bounded FIFO exception: admit the first one whose own
            // lifetime fits the current budget.
            let bypass = pending.iter().enumerate().skip(1).find_map(|(idx, entry)| {
                let Resolved::Native { req, handoff, .. } = entry else {
                    return None;
                };
                if req.token_tx.is_closed() {
                    return None;
                }
                let need = admission_lifetime_blocks(req, Some(handoff));
                let physical = pool
                    .available_blocks()
                    .saturating_add(active_resident)
                    .saturating_add(handoff.committed_len / PAGE);
                (committed + need <= usable.min(physical)).then_some((idx, need))
            });
            let Some((idx, need_blocks)) = bypass else {
                break;
            };
            let Some(Resolved::Native {
                req,
                prefix,
                handoff,
            }) = pending.remove(idx)
            else {
                unreachable!("bypass index selected a Native entry");
            };
            if let Some(deferred) = admit_native(
                rank,
                slot,
                req,
                prefix,
                handoff,
                need_blocks,
                drafter_enabled,
                pool,
                slots,
                pending_resets,
                &mut committed,
            )? {
                pending.push_front(deferred);
                break;
            }
            continue;
        }

        match pending.pop_front().expect("checked non-empty") {
            Resolved::Failed { .. } => unreachable!("handled above"),
            Resolved::Plain { req, prefix } => {
                let client_prompt_tokens = req.prompt_tokens.len();
                let mut kv = if native_mtp_prefill {
                    pool.new_request_with_cache_salt(
                        req.prompt_tokens.clone(),
                        req.max_tokens,
                        Some(super::native_mtp_cache_salt()),
                        None,
                    )
                } else {
                    pool.new_request(req.prompt_tokens.clone(), req.max_tokens, None)
                };
                // The allocator is the final authority: entitlement is
                // declared under the reserve gate, atomically with the
                // physical re-check. A concurrent resolve either sees the
                // new floor or won these pages first — then the front
                // defers, holds re-queue intact, and retries next tick.
                if !kv.try_admit(pool, front_held) {
                    pending.push_front(Resolved::Plain { req, prefix });
                    break;
                }
                let cached_tokens = if prefix_cache_enabled {
                    match kv.match_and_add_prefix(pool) {
                        Ok(cached) => cached,
                        Err(err) => {
                            let err = err.context("GLM5.2 prefix match at admission");
                            let _ = req.token_tx.send(TokenEvent::Error {
                                message: format!("{err:#}"),
                                prompt_tokens: req.prompt_tokens.len(),
                                completion_tokens: 0,
                            });
                            return Err(err);
                        }
                    }
                } else {
                    0
                };
                // The resolve's anti-eviction hold has served its purpose:
                // the match above re-pinned whatever it restored.
                drop(prefix);
                let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
                let _ = req.token_tx.send(TokenEvent::Scheduled {
                    queued_at_unix_s,
                    scheduled_at_unix_s: unix_now_s(),
                    prompt_tokens: client_prompt_tokens,
                    cached_tokens,
                });
                let state = Glm52SlotState::new(
                    req.prompt_tokens.clone(),
                    req.max_tokens,
                    req.stop_policy.clone(),
                    cached_tokens,
                );
                if drafter_enabled {
                    pending_resets.push(slot);
                }
                anyhow::ensure!(
                    kv.lifetime_blocks() == need_blocks,
                    "GLM5.2 admission budget drift: planned {need_blocks} blocks, RequestKv \
                     owns lifetime capacity for {}",
                    kv.lifetime_blocks()
                );
                slots[slot] = Some(ActiveRequest {
                    req,
                    state,
                    client_prompt_tokens,
                    kv,
                    save_cursor: SaveCursor::new(),
                    boundary_copy: None,
                });
                committed += need_blocks;
            }
            Resolved::Native {
                req,
                prefix,
                handoff,
            } => {
                if let Some(deferred) = admit_native(
                    rank,
                    slot,
                    req,
                    prefix,
                    handoff,
                    need_blocks,
                    drafter_enabled,
                    pool,
                    slots,
                    pending_resets,
                    &mut committed,
                )? {
                    pending.push_front(deferred);
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Slot a resolved native P/D intake. All authoritative allocation happens
/// here: build the request KV, re-pin the restored chain, adopt the anchor,
/// seed the verify state. A client-gone finish leaves the slot empty and
/// `committed` untouched; zero-capacity forms (EOS-only, anchored Length)
/// never reach here (intake sweep).
///
/// Entitlement is declared before the first client event, so a lost
/// `try_admit` race returns `Some(deferred)` with nothing sent — the caller
/// re-queues and the retry replays cleanly.
#[allow(clippy::too_many_arguments)]
fn admit_native(
    rank: usize,
    slot: usize,
    mut req: GenerateRequest,
    prefix: KvPrefix,
    handoff: offload::NativeMtpHandoff,
    need_blocks: usize,
    drafter_enabled: bool,
    pool: &Arc<BlockPool>,
    slots: &mut RankSlots,
    pending_resets: &mut Vec<usize>,
    committed: &mut usize,
) -> anyhow::Result<Option<Resolved>> {
    let client_prompt_tokens = req.prompt_tokens.len();
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
    let anchor = handoff
        .anchor_token_id
        .context("GLM5.2 zero-capacity handoffs finish at the intake sweep, never at a slot")?;
    anyhow::ensure!(
        req.max_tokens > 1,
        "GLM5.2 anchored-Length handoffs finish at the intake sweep, never at a slot"
    );
    req.prompt_tokens.push(anchor);
    let mut kv = pool.new_request_with_cache_salt(
        req.prompt_tokens.clone(),
        req.max_tokens,
        Some(super::native_mtp_cache_salt()),
        None,
    );
    // Credit FULL pages only: an unaligned chain's padded boundary page is
    // the pinned copy source, never part of the resident set.
    if !kv.try_admit(pool, handoff.committed_len / PAGE) {
        req.prompt_tokens.pop();
        return Ok(Some(Resolved::Native {
            req,
            prefix,
            handoff,
        }));
    }
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s,
        scheduled_at_unix_s: unix_now_s(),
        prompt_tokens: client_prompt_tokens,
        cached_tokens: handoff.committed_len,
    });
    // P sampled the anchor but never sent it; it reaches the client from here.
    if req
        .token_tx
        .send(TokenEvent::Token {
            id: anchor,
            logprob: None,
        })
        .is_err()
    {
        return Ok(None);
    }
    let boundary_copy = match restore_native_kv(&mut kv, prefix, &handoff, pool) {
        Ok(copy) => copy,
        Err(err) => {
            let err = err.context("GLM5.2 native P/D restore at admission");
            let _ = req.token_tx.send(TokenEvent::Error {
                message: format!("{err:#}"),
                prompt_tokens: client_prompt_tokens,
                completion_tokens: 1,
            });
            if let Err(release) = kv.release() {
                log::warn!("GLM5.2 native P/D failed-restore release: {release:#}");
            }
            return Err(err);
        }
    };
    let mut state = Glm52SlotState::new(
        req.prompt_tokens.clone(),
        req.max_tokens,
        req.stop_policy.clone(),
        handoff.committed_len,
    );
    state.seed_native_pd_replayed_anchor();
    let verify_drafts = state.set_drafts(
        handoff.draft_tokens.clone(),
        crate::mtp::glm52_mtp_draft_len(),
    );
    log::info!(
        "GLM5.2 native P/D admitted: rank={rank} slot={slot} \
         committed_len={} envelope_drafts={} verify_drafts={verify_drafts} \
         boundary_copy={} first_step=verify",
        handoff.committed_len,
        handoff.draft_tokens.len(),
        boundary_copy.is_some(),
    );
    if drafter_enabled {
        pending_resets.push(slot);
    }
    anyhow::ensure!(
        kv.lifetime_blocks() == need_blocks,
        "GLM5.2 native admission budget drift: planned {need_blocks}, KV owns {}",
        kv.lifetime_blocks()
    );
    slots[slot] = Some(ActiveRequest {
        req,
        state,
        client_prompt_tokens,
        kv,
        save_cursor: SaveCursor::new(),
        boundary_copy,
    });
    *committed += need_blocks;
    Ok(None)
}

/// Rebuild the restored chain under the request's own KV. Full pages match
/// straight from the radix — the state machine's >=1-uncached-token cap
/// keeps the boundary page out even when `committed_len + 1` is page-aligned
/// and the padded name coincides with the request chain (that page carries a
/// never-written row at the anchor position). The boundary rows, if any, get
/// a private page; its bytes arrive via a first-step D2D copy from the
/// restored padded-name page, which the returned [`BoundaryCopy`] keeps
/// pinned until then.
fn restore_native_kv(
    kv: &mut RequestKv,
    prefix: KvPrefix,
    handoff: &offload::NativeMtpHandoff,
    pool: &BlockPool,
) -> anyhow::Result<Option<BoundaryCopy>> {
    let full_page_tokens = (handoff.committed_len / PAGE) * PAGE;
    let cached = kv.match_and_add_prefix(pool)?;
    anyhow::ensure!(
        cached == full_page_tokens,
        "native restore matched {cached} tokens, expected {full_page_tokens} \
         (the resolve hold pins every full page)"
    );
    let boundary_rows = handoff.committed_len - full_page_tokens;
    let boundary_copy = if boundary_rows > 0 {
        kv.schedule_prefill(boundary_rows, pool)
            .map_err(|e| anyhow::anyhow!("boundary page schedule: {e}"))?;
        let dst_page = kv
            .step_page_indices(boundary_rows)
            .last()
            .copied()
            .context("scheduled boundary page has an index")?;
        kv.apply_prefill_chunk(pool)?;
        let src_page = pegainfer_kv_store::resolved_page_ids(&prefix)
            .last()
            .copied()
            .context("resolved chain has pages")?;
        Some(BoundaryCopy {
            src_page,
            dst_page,
            _prefix: prefix,
        })
    } else {
        None
    };
    anyhow::ensure!(
        kv.kv_position() == handoff.committed_len,
        "native restore left kv_position at {}, expected committed_len {}",
        kv.kv_position(),
        handoff.committed_len
    );
    kv.adopt_external_prefill_anchor()?;
    Ok(boundary_copy)
}

#[cfg(test)]
mod tests {
    use pegainfer_sample::SamplingParams;

    use super::*;
    use crate::scheduler::testkit::request;
    use crate::scheduler::testkit::sampled;

    #[test]
    fn malformed_sampling_params_die_at_intake() {
        // Values the sampler kernel would reject with an `ensure!` — which
        // past intake means a failed step and a fatal engine exit.
        let cases = [
            pegainfer_sample::SamplingParams {
                top_p: 0.0,
                ..sampled(0.8)
            },
            pegainfer_sample::SamplingParams {
                top_p: 1.5,
                ..sampled(0.8)
            },
            pegainfer_sample::SamplingParams {
                top_p: f32::NAN,
                ..sampled(0.8)
            },
            sampled(f32::INFINITY),
            sampled(f32::NAN),
            pegainfer_sample::SamplingParams {
                min_p: 1.0,
                ..sampled(0.8)
            },
            pegainfer_sample::SamplingParams {
                min_p: -0.1,
                ..sampled(0.8)
            },
        ];
        for params in cases {
            let req = request(vec![10], params, 4);
            assert!(
                validate_request(&req, 4096, false, false).is_err(),
                "params must be rejected at intake: {params:?}"
            );
        }
        // The greedy path never reaches the sampler: out-of-range values that
        // ride a greedy request stay accepted (temperature 0 ignores top_p).
        let req = request(
            vec![10],
            pegainfer_sample::SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            4,
        );
        assert!(validate_request(&req, 4096, false, false).is_ok());
    }

    #[test]
    fn lifetime_blocks_counts_the_dangling_token() {
        // 64 prompt + 1 max_tokens: the generated token is appended to the
        // sequence (dangling) and provisions page 2 even though its KV is
        // never written.
        assert_eq!(lifetime_blocks(64, 1), 2);
        assert_eq!(lifetime_blocks(63, 1), 1);
        assert_eq!(lifetime_blocks(64, 64), 2);
        assert_eq!(lifetime_blocks(64, 65), 3);
    }

    #[test]
    fn native_pd_admission_counts_the_internal_anchor_position() {
        let handoff = offload::NativeMtpHandoff {
            fingerprint: offload::handoff_fingerprint(),
            committed_len: PAGE,
            anchor_token_id: Some(11),
            stop_cause: None,
            draft_tokens: vec![1, 2],
        };
        let req = request(vec![10; PAGE], SamplingParams::default(), PAGE);
        assert_eq!(
            admission_lifetime_blocks(&req, Some(&handoff)),
            3,
            "the replayed anchor appends one input position beyond the prompt"
        );
        assert_eq!(
            admission_lifetime_blocks(&req, None),
            2,
            "ordinary requests retain their existing lifetime geometry"
        );
    }

    #[test]
    fn prefill_only_accepts_exactly_one_output_token() {
        let one = request(vec![10, 11], SamplingParams::default(), 1);
        assert!(validate_request(&one, 4096, true, false).is_ok());

        let many = request(vec![10, 11], SamplingParams::default(), 2);
        let error =
            validate_request(&many, 4096, true, false).expect_err("decode must be rejected");
        assert!(error.contains("requires max_tokens=1"), "{error}");
    }

    #[test]
    fn native_mtp_prefill_reserves_the_fixed_proposal_positions() {
        // Headroom tracks the configured draft span, not the compile ceiling.
        let headroom = crate::mtp::glm52_mtp_draft_len() - 1;
        let fits = request(vec![10; 4096 - headroom], SamplingParams::default(), 1);
        assert!(validate_request(&fits, 4096, true, true).is_ok());

        let overflows = request(vec![10; 4096 - headroom + 1], SamplingParams::default(), 1);
        let error = validate_request(&overflows, 4096, true, true)
            .expect_err("fixed MTP proposal must fit inside the context cap");
        assert!(
            error.contains(&format!("{headroom} positions of proposal headroom")),
            "{error}"
        );

        assert!(
            validate_request(&overflows, 4096, true, false).is_ok(),
            "plain TP4 prefill does not execute the native-MTP proposal loop"
        );
    }
}
