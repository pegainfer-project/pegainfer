use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::unix_now_s;

use crate::runner::worker::KIMI_MAX_REQUEST_TOKENS;

/// KV tokens a request can write over its lifetime. The final generated
/// token is returned but never fed back, so its KV is never written (the
/// same dangling-token contract as the qwen3 admission).
fn request_max_kv_tokens(req: &GenerateRequest) -> usize {
    req.prompt_tokens.len() + req.max_tokens.saturating_sub(1)
}

/// Pool blocks a request may draw over its lifetime. One token more than
/// `request_max_kv_tokens`: kvbm appends the final generated token to the
/// sequence and provisions its block even though its KV is never written,
/// so at boundary alignments the peak draw is `ceil((prompt + max) / bs)`.
pub(in crate::runner) fn request_lifetime_blocks(
    req: &GenerateRequest,
    block_size: usize,
) -> usize {
    (req.prompt_tokens.len() + req.max_tokens)
        .div_ceil(block_size)
        .max(1)
}

/// Honor-or-reject (#239): a request that can never fit — per-request KV
/// capacity, pool size, or a path-specific prompt cap — is rejected at
/// admission with the limit spelled out, instead of poisoning the batch
/// mid-decode when the KV write finally lands out of bounds.
pub(in crate::runner) fn validate_kv_capacity(
    req: &GenerateRequest,
    block_size: usize,
    max_request_blocks: usize,
    max_prompt_tokens: Option<usize>,
) -> Result<(), String> {
    if let Some(max_prompt) = max_prompt_tokens
        && req.prompt_tokens.len() > max_prompt
    {
        return Err(format!(
            "prompt of {} tokens exceeds the per-request prompt cap of {max_prompt} \
             tokens on this serving path",
            req.prompt_tokens.len()
        ));
    }
    let max_kv_tokens = request_max_kv_tokens(req);
    if max_kv_tokens > KIMI_MAX_REQUEST_TOKENS {
        return Err(format!(
            "prompt_tokens ({}) + max_tokens ({}) needs {max_kv_tokens} KV tokens, \
             exceeding the per-request capacity of {KIMI_MAX_REQUEST_TOKENS} tokens",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    let blocks = request_lifetime_blocks(req, block_size);
    if blocks > max_request_blocks {
        return Err(format!(
            "request needs {blocks} KV blocks ({max_kv_tokens} KV tokens), exceeding \
             the pool capacity of {max_request_blocks} blocks"
        ));
    }
    Ok(())
}

pub(in crate::runner) fn preflight_prefill_candidate(
    req: GenerateRequest,
) -> Option<GenerateRequest> {
    let Some(verdict) = unschedulable_verdict(&req) else {
        return Some(req);
    };
    // Scheduled first, terminal event last — consumers stop at the terminal.
    send_scheduled(&req);
    match verdict {
        UnschedulableVerdict::Finish => {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                stop_cause: None,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
        UnschedulableVerdict::Reject(message) => {
            let _ = req.token_tx.send(TokenEvent::Rejected {
                message,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
    }
    None
}

pub(in crate::runner) fn send_scheduled(req: &GenerateRequest) {
    let scheduled_at = unix_now_s();
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at),
        scheduled_at_unix_s: scheduled_at,
        prompt_tokens: req.prompt_tokens.len(),
        // Emitted at admission, before the KV prefix match runs — the real
        // hit count is not known yet (kimi prefix-cache usage reporting is a
        // follow-up).
        cached_tokens: 0,
    });
}

enum UnschedulableVerdict {
    Finish,
    Reject(String),
}

fn unschedulable_verdict(req: &GenerateRequest) -> Option<UnschedulableVerdict> {
    if req.max_tokens == 0 {
        return Some(UnschedulableVerdict::Finish);
    }
    if req.prompt_tokens.is_empty() {
        return Some(UnschedulableVerdict::Reject(
            "Kimi-K2 forward requires at least one prompt token".to_string(),
        ));
    }
    // Honor-or-reject (#236): prompt echo needs per-position prompt logprobs,
    // which the prefill path does not compute (lm_head runs on the last
    // position only). Reject instead of silently returning a response with
    // the echo stripped.
    if req.echo {
        return Some(UnschedulableVerdict::Reject(
            "echo is not supported on the Kimi-K2 serving path: prompt \
             logprobs are not computed; set echo=false"
                .to_string(),
        ));
    }
    if let Err(message) = validate_sampling_params(req) {
        return Some(UnschedulableVerdict::Reject(message));
    }
    None
}

/// Honor-or-reject (#237): a request whose sampling parameters cannot be
/// honored exactly is rejected here, before any forward work.
fn validate_sampling_params(req: &GenerateRequest) -> Result<(), String> {
    let p = &req.params;
    if !p.temperature.is_finite() || p.temperature < 0.0 {
        return Err(format!(
            "temperature must be finite and >= 0, got {}",
            p.temperature
        ));
    }
    if !p.top_p.is_finite() || p.top_p <= 0.0 || p.top_p > 1.0 {
        return Err(format!("top_p must be in (0, 1], got {}", p.top_p));
    }
    if p.top_k < -1 || p.top_k == 0 {
        return Err(format!(
            "top_k must be -1 (disabled) or >= 1, got {}",
            p.top_k
        ));
    }
    Ok(())
}
