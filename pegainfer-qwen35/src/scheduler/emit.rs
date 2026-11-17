//! Request-emission concerns: pruning closed requests, refusing unsupported
//! request features, and delivering admission rejections to callers.

use super::*;

pub(super) fn prune_closed_requests<B>(
    backend: &mut B,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    pending: &mut Vec<QueuedRequest>,
    ledger: &mut RequestLedger,
) -> Result<()>
where
    B: DecodeDispatchBackend + PrefillPromoteBackend,
{
    pending.retain(|req| {
        if ledger.is_aborted(req.id) {
            ledger.retire(req.id);
            false
        } else {
            true
        }
    });
    for idx in (0..active.len()).rev() {
        if ledger.is_aborted(active[idx].id) {
            debug!(
                "request pruned before scheduling: request_id={:?} phase=decode tokens_generated={}",
                active[idx].client_label,
                ledger.completion_tokens(active[idx].id)
            );
            let removed = backend.take_active_request(active, idx);
            backend.drop_active_state(&removed.backend_state)?;
            ledger.retire(removed.id);
        }
    }
    for idx in (0..prefilling.len()).rev() {
        if ledger.is_aborted(prefilling[idx].req.id) {
            let removed = prefilling.remove(idx);
            debug!(
                "request pruned before scheduling: request_id={:?} phase=prefill cursor={}",
                removed.req.request.client_label, removed.cursor
            );
            // Admission creates rank-local recurrent state, even on a cold miss.
            let expectation = DropExpectation::MustExist;
            backend.drop_prefill_state(&removed.backend_state, expectation)?;
            ledger.retire(removed.req.id);
        }
    }
    Ok(())
}

pub(super) fn reject_unsupported_prompt_logprobs(
    pending: &mut Vec<QueuedRequest>,
    ledger: &mut RequestLedger,
) {
    pending.retain(|req| {
        if req.request.prompt_logprobs.is_none() {
            return true;
        }
        reject_or_retire(
            req.id,
            EngineRejectReason::Unsupported {
                feature: "prompt_logprobs".to_owned(),
            },
            ledger,
        );
        false
    });
}

pub(super) fn send_rejection(
    req: &QueuedRequest,
    reason: RejectReason,
    ledger: &mut RequestLedger,
) {
    let reason = match reason {
        RejectReason::ZeroMaxTokens => EngineRejectReason::Unsupported {
            feature: "max_tokens=0".to_owned(),
        },
        RejectReason::ContextLength { limit } => EngineRejectReason::ContextLength {
            prompt_tokens: req.request.prompt_tokens.len(),
            max_tokens: req.request.max_tokens,
            limit,
        },
        RejectReason::KvBudget => EngineRejectReason::KvBudget {
            prompt_tokens: req.request.prompt_tokens.len(),
            worst_case_tokens: plan::request_lifetime_tokens(
                req.request.prompt_tokens.len(),
                req.request.max_tokens,
            ),
        },
    };
    reject_or_retire(req.id, reason, ledger);
}
