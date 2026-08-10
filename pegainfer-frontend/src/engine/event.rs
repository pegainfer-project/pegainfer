use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
    Error,
}

#[derive(Debug)]
pub enum TokenEvent {
    Scheduled {
        queued_at_unix_s: f64,
        scheduled_at_unix_s: f64,
        prompt_tokens: usize,
        /// Prompt tokens served from the prefix cache (0 when the engine has
        /// no prefix cache or the value is not known at emit time).
        cached_tokens: usize,
    },
    Token {
        id: u32,
        logprob: Option<TokenLogprob>,
    },
    PromptTokens {
        ids: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
    /// Opaque P/D handoff metadata forwarded through the vLLM-compatible
    /// `kv_transfer_params` response field.
    KvTransfer { params: serde_json::Value },
    Finished {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error {
        message: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Rejected {
        message: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}

/// The tag that routes a [`TokenEvent`] back to its request on the shared
/// output channel — the external request id (vLLM's `request_id`). `Arc<str>`
/// keeps per-event tagging to a refcount bump instead of a string copy.
pub type RequestTag = Arc<str>;

/// Seconds since `UNIX_EPOCH` as `f64` — the clock base for `TokenEvent`
/// timestamps.
pub fn unix_now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs_f64()
}
