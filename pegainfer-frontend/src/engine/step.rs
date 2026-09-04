//! The step-batched wire contract between a model scheduler and a frontend.
//!
//! One scheduler step produces one [`StepOutputs`] message. Per request the
//! step carries at most one [`RequestUpdate`] — a flat record rather than an
//! event sequence, so the intra-step ordering rules of the old `TokenEvent`
//! protocol (`Scheduled` before tokens before a single terminal) are structure,
//! not convention: a `RequestUpdate` cannot express a token after its terminal.
//! Cross-step ordering is enforced on the producer side by the
//! [`super::RequestLedger`]: a request's account closes at its terminal, and
//! writes against a closed account panic.

use std::fmt;
use std::time::Instant;

use super::event::FinishReason;
use super::event::TokenLogprob;

/// In-process routing id for one generate request, minted by
/// [`super::SchedulerHandle::submit`] from a per-scheduler counter. `Copy` and
/// integer-keyed on purpose: the external protocol's string request id stays in
/// the protocol stack, which maps it to this id at its own boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u64);

impl RequestId {
    /// Minting ids is [`super::SchedulerHandle::submit`]'s job in production
    /// (a per-scheduler counter). Public so scheduler unit tests and tools
    /// can fabricate requests.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw counter value, for logs and protocol-side maps.
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req-{}", self.0)
    }
}

/// One generate request as a frontend submits it. Identity (`RequestId`),
/// queue timestamp, and the abort flag are minted at submit time and live on
/// the request's ledger account, not here.
pub struct Request {
    pub prompt_tokens: Vec<u32>,
    pub params: crate::sampler::SamplingParams,
    pub max_tokens: usize,
    pub lora_adapter: Option<String>,
    /// Opaque router/P-D metadata from the request's
    /// `vllm_xargs.kv_transfer_params`.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Number of top logprobs to return per token (0 = disabled).
    pub logprobs: usize,
    /// Return prompt tokens (and their logprobs) ahead of the completion.
    pub echo: bool,
    /// Trace context of the caller's request span, when tracing is on. The
    /// model scheduler opens its queue/prefill/decode spans as children of
    /// this. `None` when tracing is disabled.
    pub trace_parent: Option<fastrace::collector::SpanContext>,
    /// External request name for scheduler-side logs only; never used for
    /// routing (that is [`RequestId`]).
    pub client_label: Option<std::sync::Arc<str>>,
}

/// One submitted request as the scheduler receives it: the id of its open
/// ledger account, and the payload. The id is minted by
/// [`super::SchedulerHandle::submit`]; a payload never carries one of its own.
pub struct QueuedRequest {
    pub id: RequestId,
    pub request: Request,
}

/// Everything one scheduler step produced, in one message. The scheduler-side
/// ledger sends exactly one per step that touched any request; an idle step
/// sends nothing.
#[derive(Debug, Default)]
pub struct StepOutputs {
    pub updates: Vec<RequestUpdate>,
}

/// Everything that happened to one request within one step, flattened. A
/// consumer folds the sections in field order: admission facts, then new
/// tokens, then the terminal.
#[derive(Debug)]
pub struct RequestUpdate {
    pub id: RequestId,
    /// Present exactly once per request lifetime, in the step whose iteration
    /// admitted it.
    pub scheduled: Option<ScheduledInfo>,
    /// Tokens committed this step. Multi-token spans (speculative/MTP commits)
    /// arrive together; a pure-prefill step contributes none.
    pub tokens: Vec<u32>,
    /// Parallel to `tokens` (same length whenever non-empty is possible; all
    /// `None` when logprobs were not requested).
    pub logprobs: Vec<Option<TokenLogprob>>,
    /// Prompt tokens served from a prefix cache, reported in the step where
    /// the scheduler learns it (first prefill chunk), not at admission —
    /// admission-time `Scheduled` cannot know it yet.
    pub cached_tokens: Option<usize>,
    /// Echoed prompt, once, when the request's prefill completes.
    pub prompt_echo: Option<PromptEcho>,
    /// Opaque P/D handoff metadata forwarded through the protocol stack's
    /// `kv_transfer_params` response field.
    pub kv_transfer: Option<serde_json::Value>,
    /// End of the request's lifetime. No later step mentions this id again.
    pub terminal: Option<Terminal>,
}

impl RequestUpdate {
    pub(crate) fn empty(id: RequestId) -> Self {
        Self {
            id,
            scheduled: None,
            tokens: Vec::new(),
            logprobs: Vec::new(),
            cached_tokens: None,
            prompt_echo: None,
            kv_transfer: None,
            terminal: None,
        }
    }

    /// An update that carries no observable fact (possible when a request's
    /// only activity this step was bookkeeping). The ledger drops these
    /// rather than shipping empty records.
    pub(crate) fn is_vacant(&self) -> bool {
        self.scheduled.is_none()
            && self.tokens.is_empty()
            && self.cached_tokens.is_none()
            && self.prompt_echo.is_none()
            && self.kv_transfer.is_none()
            && self.terminal.is_none()
    }
}

/// Admission facts stamped by the contract layer, never by model code:
/// `queued_at` at [`super::SchedulerHandle::submit`], `scheduled_at` at
/// [`super::RequestLedger::admit`]. Monotonic `Instant`s — the wall-clock
/// rendering some protocols need (vLLM's unix floats) is the protocol stack's
/// translation, done against its own anchor.
#[derive(Clone, Copy, Debug)]
pub struct ScheduledInfo {
    pub queued_at: Instant,
    pub scheduled_at: Instant,
    pub prompt_tokens: usize,
}

#[derive(Debug)]
pub struct PromptEcho {
    pub ids: Vec<u32>,
    /// Parallel to `ids`; `None` entries where the model produced no logprob
    /// (the first position, or logprobs disabled).
    pub logprobs: Vec<Option<TokenLogprob>>,
}

/// Why a request was refused at admission. Typed so a frontend can map the
/// class onto its own error surface (HTTP status, retry policy) instead of
/// pattern-matching rendered text; `Display` is the client-facing message.
/// `#[non_exhaustive]`: model lines grow new refusals (frontends keep a
/// wildcard arm), and a variant only exists once a scheduler produces it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RejectReason {
    /// Worst-case length (`prompt + max_tokens`) exceeds the model's
    /// position-encoding window.
    ContextLength {
        prompt_tokens: usize,
        max_tokens: usize,
        limit: usize,
    },
    /// Echo needs all-position logits in one forward pass, so the prompt must
    /// fit the profiled prefill bound.
    EchoPrefillTokens { prompt_tokens: usize, limit: usize },
    /// Worst-case length needs more KV blocks than this instance can ever
    /// provide to one request.
    KvBudget {
        prompt_tokens: usize,
        worst_case_tokens: usize,
    },
    /// The named adapter is not loaded on this engine.
    UnknownLoraAdapter { name: String },
    /// The request asks for a feature this engine does not implement.
    Unsupported { feature: String },
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextLength {
                prompt_tokens,
                max_tokens,
                limit,
            } => write!(
                f,
                "request exceeds this model's maximum context length of {limit} tokens: \
                 requested {} (prompt={prompt_tokens} + max_tokens={max_tokens})",
                prompt_tokens.saturating_add(*max_tokens)
            ),
            Self::EchoPrefillTokens {
                prompt_tokens,
                limit,
            } => write!(
                f,
                "echo request prompt exceeds the profiled prefill limit of {limit} tokens: \
                 prompt_tokens={prompt_tokens}"
            ),
            Self::KvBudget {
                prompt_tokens,
                worst_case_tokens,
            } => write!(
                f,
                "request requires more KV blocks than this model instance can provide: \
                 prompt_tokens={prompt_tokens}, max_request_tokens={worst_case_tokens}"
            ),
            Self::UnknownLoraAdapter { name } => {
                write!(f, "LoRA adapter is not loaded: {name}")
            }
            Self::Unsupported { feature } => {
                write!(f, "this engine does not support {feature}")
            }
        }
    }
}

/// Why and how a request's lifetime ended. Token counts are tallied by the
/// ledger from what actually shipped, not hand-maintained by model code.
#[derive(Debug)]
pub enum Terminal {
    Finished {
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Refused at admission: the request never held a slot and produced no
    /// tokens.
    Rejected {
        reason: RejectReason,
        prompt_tokens: usize,
    },
    /// Died to an engine-side error (execution failure, scheduler bug, engine
    /// teardown).
    Failed {
        message: String,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_rejection_names_the_engine_feature() {
        let reason = RejectReason::Unsupported {
            feature: "frontend-resolved prefix".to_string(),
        };
        assert_eq!(
            reason.to_string(),
            "this engine does not support frontend-resolved prefix"
        );
    }
}
