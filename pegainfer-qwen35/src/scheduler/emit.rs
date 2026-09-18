//! Terminal request emission: shutdown fan-out, closed-request pruning, and
//! rejection / error delivery to callers.

use super::*;

pub(super) fn terminal_scheduler_shutdown(
    submit_rx: &mut mpsc::UnboundedReceiver<SubmittedRequest>,
    load_tx: &watch::Sender<SchedulerMetrics>,
    kv_total_blocks: u64,
    active: Vec<ActiveRequest35>,
    prefilling: Vec<PrefillingRequest35>,
    pending: Vec<SchedulerRequest>,
    deferred: Vec<SchedulerRequest>,
    inflight_prefill: Option<InflightPrefill>,
    failure: FatalSchedulerError,
) {
    submit_rx.close();

    let mut requests = failure.transient;
    requests.extend(active.into_iter().map(Into::into));
    requests.extend(prefilling.into_iter().map(Into::into));
    requests.extend(pending.into_iter().map(Into::into));
    requests.extend(deferred.into_iter().map(Into::into));
    if let Some(InflightPrefill { output, chunk, .. }) = inflight_prefill {
        // The stream must drain before the chunk's KV/recurrent/conv state is
        // released or transferred into terminal request ownership.
        drop(output);
        requests.extend(chunk.reqs.into_iter().map(Into::into));
    }
    while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
        requests.push(req.into());
    }

    warn!(
        "Qwen3.5 TP scheduler terminating after replica failure: {}",
        failure.message
    );
    for request in requests {
        request.send_error(&failure.message);
    }
    load_tx.send_replace(SchedulerMetrics {
        kv_used_blocks: 0,
        kv_total_blocks,
        num_running_reqs: 0,
        num_waiting_reqs: 0,
        spec_decode: None,
    });
}

pub(super) fn prune_closed_requests<B>(
    backend: &mut B,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    pending: &mut Vec<SchedulerRequest>,
) -> std::result::Result<(), FatalSchedulerError>
where
    B: DecodeDispatchBackend + PrefillPromoteBackend,
{
    pending.retain(|req| !req.token_tx.is_closed());

    for idx in (0..active.len()).rev() {
        if active[idx].token_tx.is_closed() {
            debug!(
                "request pruned before scheduling: request_id={:?} phase=decode tokens_generated={}",
                active[idx].request_id, active[idx].generated_count
            );
            let removed = backend.take_active_request(active, idx);
            if let Err(err) = backend.drop_active_state(&removed.backend_state) {
                return Err(FatalSchedulerError::new(err.to_string()).with_request(removed));
            }
        }
    }

    for idx in (0..prefilling.len()).rev() {
        if prefilling[idx].req.token_tx.is_closed() {
            let removed = prefilling.remove(idx);
            debug!(
                "request pruned before scheduling: request_id={:?} phase=prefill cursor={}",
                removed.req.request_id, removed.cursor
            );
            let expectation = if removed.cursor == 0 {
                DropExpectation::MustBeAbsent
            } else {
                DropExpectation::MustExist
            };
            if let Err(err) = backend.drop_prefill_state(&removed.backend_state, expectation) {
                return Err(FatalSchedulerError::new(err.to_string()).with_request(removed));
            }
        }
    }
    Ok(())
}

pub(super) const UNSUPPORTED_PROMPT_LOGPROBS_MESSAGE: &str =
    "prompt_logprobs is unsupported by the Qwen3.5 serving contract";

pub(super) fn reject_unsupported_prompt_logprobs(pending: &mut Vec<SchedulerRequest>) {
    pending.retain(|req| {
        if req.prompt_logprobs.is_none() {
            return true;
        }
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: UNSUPPORTED_PROMPT_LOGPROBS_MESSAGE.to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        false
    });
}

pub(super) fn send_rejection(req: &SchedulerRequest, reason: RejectReason) {
    let message = match reason {
        RejectReason::ContextLength { limit } => format!(
            "request exceeds this model's maximum context length of {limit} tokens: requested {} (prompt={} + max_tokens={})",
            req.prompt_tokens.len().saturating_add(req.max_tokens),
            req.prompt_tokens.len(),
            req.max_tokens
        ),
        RejectReason::KvBudget => {
            let max_request_tokens = max_kv_tokens(req.prompt_tokens.len(), req.max_tokens);
            format!(
                "request requires more KV pages than this model instance can provide: prompt_tokens={}, max_request_tokens={max_request_tokens}",
                req.prompt_tokens.len()
            )
        }
    };
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}
