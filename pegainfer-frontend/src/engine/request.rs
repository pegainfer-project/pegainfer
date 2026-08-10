use super::sink::TokenSink;
use crate::sampler::SamplingParams;

pub struct GenerateRequest {
    pub request_id: Option<String>,
    pub queued_at_unix_s: Option<f64>,
    /// Trace context of the caller's request span, when tracing is on. The
    /// model scheduler opens its queue/prefill/decode spans as children of this
    /// so the host-side phase breakdown attaches to the same trace the frontend
    /// started. `None` when tracing is disabled — the scheduler then skips span
    /// work entirely. `SpanContext` is `Copy`, so it rides through the
    /// scheduler's `Clone` request state without holding a live (non-`Clone`)
    /// `Span`.
    pub trace_parent: Option<fastrace::collector::SpanContext>,
    /// Logical data-parallel rank selected by the frontend. `None` lets the
    /// handle place the request on its least-loaded partition (waiting
    /// requests weigh 4x).
    pub data_parallel_rank: Option<usize>,
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub lora_adapter: Option<String>,
    /// Opaque router/P-D metadata from the request's
    /// `vllm_xargs.kv_transfer_params`.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Where the scheduler emits this request's `TokenEvent`s. All requests on
    /// one engine share a single tagged output channel behind this sink (see
    /// [`TokenSink`]); the frontend demuxes by tag.
    pub token_tx: TokenSink,
    pub logprobs: usize,
    pub echo: bool,
}
