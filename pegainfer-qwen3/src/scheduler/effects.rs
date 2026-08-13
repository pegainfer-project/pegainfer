//! Pure per-step effect data produced by [`super::resolve::resolve_step`].
//!
//! Effects reference requests by the scheduler's internal [`RequestId`] only;
//! translating them into the engine contract (emitter calls on the typestate
//! handles) is `crate::frontend_adapter`'s job. Keeping this layer sink-free
//! is what lets the decode-overlap path clone `PendingRequest`s and lets the
//! resolve logic stay a pure function of executor results.

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::TokenLogprob;

use super::ActiveRequestState;
use super::PendingRequest;
use crate::executor::RequestId;

/// The request's prefix-cache hit count, learned when its first prefill chunk
/// lands (#246). Reported once per request.
pub(crate) struct CachedTokensEffect {
    pub(crate) request_id: RequestId,
    pub(crate) cached_tokens: usize,
}

pub(crate) struct PromptEchoEffect {
    pub(crate) request_id: RequestId,
    pub(crate) ids: Vec<u32>,
    pub(crate) logprobs: Vec<Option<TokenLogprob>>,
}

pub(crate) enum PendingEffect {
    Finish {
        request_id: RequestId,
        finish_reason: FinishReason,
    },
    EmitAndFinish {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
    },
    Promote {
        state: ActiveRequestState,
        first_token: u32,
        logprob: Option<TokenLogprob>,
    },
    /// A non-final prefill chunk ran; the request goes back to the front of
    /// the prefilling queue with its progress updated.
    ContinuePrefill { req: PendingRequest },
}

pub(crate) enum DecodeEffect {
    Finish {
        request_id: RequestId,
        finish_reason: FinishReason,
    },
    EmitAndFinish {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        finish_reason: FinishReason,
    },
    EmitAndContinue {
        request_id: RequestId,
        token: u32,
        logprob: Option<TokenLogprob>,
        /// The request's running completion count after this token, for the
        /// scheduler's own `generated_count`/`max_tokens` bookkeeping (the
        /// wire-visible counts are tallied by the emitter).
        completion_tokens: usize,
    },
    /// Commit several accepted speculative tokens and keep the request running.
    EmitManyAndContinue {
        request_id: RequestId,
        tokens: Vec<u32>,
        completion_tokens: usize,
    },
    /// Commit several accepted speculative tokens, then finish — a stop token or
    /// the max-output budget was hit partway through the accepted span.
    EmitManyAndFinish {
        request_id: RequestId,
        tokens: Vec<u32>,
        finish_reason: FinishReason,
    },
}

pub(crate) struct StepEffects {
    pub(crate) cached: Vec<CachedTokensEffect>,
    pub(crate) prompt_echoes: Vec<PromptEchoEffect>,
    pub(crate) pending: Vec<PendingEffect>,
    pub(crate) decode: Vec<DecodeEffect>,
}

impl StepEffects {
    pub(crate) fn empty() -> Self {
        Self {
            cached: Vec::new(),
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: Vec::new(),
        }
    }
}
