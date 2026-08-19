//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod plan;

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::sys;
use log::debug;
use log::info;
use log::warn;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::EngineHandle as SchedulerHandle;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest as SchedulerRequest;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::engine::panic_message;
use pegainfer_frontend::sampler::SamplingParams;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;
use tokio::sync::watch;

use self::plan::ActiveDecodeState;
use self::plan::ActiveKvBudget;
use self::plan::ExecutionPlan;
use self::plan::PrefillKvBudget;
use self::plan::PrefillQueueState;
use self::plan::RejectReason;
use self::plan::admit_pending_requests;
use self::plan::choose_prefill_budget;
use self::plan::compaction_after_retire;
use self::plan::max_kv_tokens;
use self::plan::plan_prefill_chunks;
use self::plan::prefilling_future_pages;
use self::plan::slot_for_new_request;
use crate::Qwen35DecodeOverlap;
use crate::Qwen35SchedulerPolicy;
use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::RequestId;
use crate::logprobs::snapshot_requested_logprobs;
use crate::recurrent_state::RecurrentState;
use crate::tp_executor::DropExpectation;
use crate::tp_executor::Qwen35TpExecutor;
use crate::tp_executor::TpDecodeStepItem;
use crate::tp_executor::TpPrefillChunkItem;
use crate::tp_executor::TpUnifiedPlan;
use crate::weights::Qwen35Model;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    request_id: Option<String>,
    token_tx: TokenSink,
    backend_state: ActiveBackendState,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    logprobs: usize,
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
/// It owns its growing KV and recurrent state until the prompt is exhausted,
/// at which point it is promoted into the decode batch.
struct PrefillingRequest35 {
    req: SchedulerRequest,
    backend_state: PrefillBackendState,
    /// Prompt tokens prefilled so far.
    cursor: usize,
    /// Tokens to prefill in the step currently scheduled (set by `take_prefill_chunks`).
    step_chunk: usize,
}

enum ActiveBackendState {
    Single {
        kv: KvState,
        /// Index into `BatchDecodeGraphState.slot_states`.
        graph_slot_idx: usize,
    },
    Tp {
        request_id: RequestId,
    },
}

enum PrefillBackendState {
    Single { kv: KvState, rec: RecurrentState },
    Tp { request_id: RequestId },
}

struct TerminalRequest {
    token_tx: TokenSink,
    prompt_tokens: usize,
    completion_tokens: usize,
}

impl TerminalRequest {
    fn send_error(self, message: &str) {
        let _ = self.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
        });
    }
}

impl From<SchedulerRequest> for TerminalRequest {
    fn from(req: SchedulerRequest) -> Self {
        Self {
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
            token_tx: req.token_tx,
        }
    }
}

impl From<ActiveRequest35> for TerminalRequest {
    fn from(req: ActiveRequest35) -> Self {
        Self {
            token_tx: req.token_tx,
            prompt_tokens: req.prompt_len,
            completion_tokens: req.generated_count,
        }
    }
}

impl From<PrefillingRequest35> for TerminalRequest {
    fn from(req: PrefillingRequest35) -> Self {
        req.req.into()
    }
}

struct PrefillCompletionRequest {
    req: SchedulerRequest,
    backend_state: PrefillBackendState,
}

trait CompletionRequest {
    fn token_tx(&self) -> &TokenSink;
    fn into_terminal(self) -> TerminalRequest;
}

impl CompletionRequest for ActiveRequest35 {
    fn token_tx(&self) -> &TokenSink {
        &self.token_tx
    }

    fn into_terminal(self) -> TerminalRequest {
        self.into()
    }
}

impl CompletionRequest for PrefillCompletionRequest {
    fn token_tx(&self) -> &TokenSink {
        &self.req.token_tx
    }

    fn into_terminal(self) -> TerminalRequest {
        self.req.into()
    }
}

struct CompletionCandidate<R> {
    request: R,
    final_events: Vec<TokenEvent>,
}

impl<R: CompletionRequest> CompletionCandidate<R> {
    fn commit(self) {
        for event in self.final_events {
            let _ = self.request.token_tx().send(event);
        }
    }

    fn into_terminal(self) -> TerminalRequest {
        self.request.into_terminal()
    }
}

struct FatalSchedulerError {
    message: String,
    transient: Vec<TerminalRequest>,
}

#[derive(Clone, Debug, PartialEq)]
struct PrefillArtifact {
    token: u32,
    logprob: Option<TokenLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
struct DecodeArtifact {
    token: u32,
    logprob: Option<TokenLogprob>,
}

struct AlignedUnifiedArtifacts {
    prefill: Vec<Option<PrefillArtifact>>,
    decode: Vec<DecodeArtifact>,
}

enum PrefillStepArtifacts {
    Single {
        tokens: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
    Tp(Vec<Option<PrefillArtifact>>),
}

impl PrefillStepArtifacts {
    fn final_artifact(&self, idx: usize) -> PrefillArtifact {
        match self {
            Self::Single { tokens, logprobs } => PrefillArtifact {
                token: tokens[idx],
                logprob: logprobs[idx].clone(),
            },
            Self::Tp(artifacts) => artifacts[idx]
                .clone()
                .expect("validated TP final-prefill row must contain an artifact"),
        }
    }
}

impl FatalSchedulerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            transient: Vec::new(),
        }
    }

    fn with_request(mut self, request: impl Into<TerminalRequest>) -> Self {
        self.transient.push(request.into());
        self
    }

    fn with_requests<I, R>(mut self, requests: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<TerminalRequest>,
    {
        self.transient.extend(requests.into_iter().map(Into::into));
        self
    }
}

pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

/// Env-gated per-step ITL diagnostics (issue #470). When `PEGAINFER_ITL_DEBUG`
/// is set, the scheduler emits one `ITL_STEP` line per executed step, tagging
/// the plan kind, the *actual* prefill-chunk token count associated with the
/// action, the active decode width, and the CPU wall-time. This lets the
/// mixed-load bench separate serial Unified stalls from overlap launch,
/// decode, completion, and wait actions instead of relying on the coarse
/// `[submit, last-token]` injection window. Off by default: no cost on the
/// normal bench path.
fn itl_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEGAINFER_ITL_DEBUG").is_some())
}

/// Monotonic microseconds since the first ITL step, so `ITL_STEP` timestamps
/// are correlatable within one process run (paired with wall-clock epoch us).
fn itl_debug_mono_us() -> u128 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_micros()
}

fn log_itl_step(
    step_start: Option<Instant>,
    plan: &str,
    prefill_tokens: usize,
    prefill_reqs: usize,
    decode_n: usize,
) {
    let Some(step_start) = step_start else {
        return;
    };
    let dur_us = step_start.elapsed().as_micros();
    let epoch_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    info!(
        "ITL_STEP mono_us={} epoch_us={} plan={} prefill_tok={} prefill_reqs={} decode_n={} dur_us={}",
        itl_debug_mono_us(),
        epoch_us,
        plan,
        prefill_tokens,
        prefill_reqs,
        decode_n,
        dur_us
    );
}

// ── Entry point ─────────────────────────────────────────────────────────

pub fn start_with_capacity(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<SchedulerHandle> {
    start_with_capacity_and_policy(
        model,
        seed,
        max_batch,
        max_prefill_tokens,
        Qwen35SchedulerPolicy::Off,
        Qwen35DecodeOverlap::Off,
    )
}

pub(crate) fn start_with_capacity_and_policy(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    decode_overlap: Qwen35DecodeOverlap,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    // Static instance cap for the vLLM bridge's max_model_len. Live admission
    // still uses the current page budget inside the scheduler loop.
    let total_blocks = model.kv_pool().capacity_pages().saturating_sub(1);
    let kv_total_blocks = total_blocks as u64;
    let block_size = model.kv_pool().layout().page_size;
    let servable = servable_len(
        model.config().max_position_embeddings,
        total_blocks,
        block_size,
    );
    let backend = SingleGpuBackend::new(model, max_batch, decode_overlap)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (startup_tx, startup_rx) = std_mpsc::channel();
    let (load_tx, load_rx) = watch::channel(SchedulerMetrics {
        kv_total_blocks,
        ..SchedulerMetrics::default()
    });

    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(backend.model()) {
            Ok(_guard) => {
                let _ = startup_tx.send(Ok(()));
                scheduler_loop(
                    SchedulerBackend::Single(backend),
                    submit_rx,
                    seed,
                    max_prefill_tokens,
                    scheduler_policy,
                    load_tx,
                );
            }
            Err(err) => {
                let _ = startup_tx.send(Err(err));
            }
        })
        .expect("failed to spawn Qwen3.5 scheduler thread");

    let Ok(startup) = startup_rx.recv() else {
        let panic_note = match join_handle.join() {
            Err(panic) => format!(" (thread panicked: {})", panic_message(panic.as_ref())),
            Ok(()) => String::new(),
        };
        anyhow::bail!("Qwen3.5 scheduler exited during startup{panic_note}");
    };
    if let Err(err) = startup {
        let _ = join_handle.join();
        return Err(err);
    }
    Ok(
        SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
            .with_servable_len(servable)
            .with_kv_capacity(KvCapacity {
                total_blocks,
                block_size,
            })
            .with_metrics_watch(load_rx),
    )
}

pub(crate) fn start_tp_with_capacity(
    model_path: &str,
    seed: u64,
    device_ordinals: &[usize],
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let backend =
        TpSchedulerBackend::new(model_path, device_ordinals, max_batch, max_prefill_tokens)?;
    let servable = servable_len(
        backend.max_position_embeddings(),
        backend.capacity_pages_for_requests(),
        backend.page_size(),
    );
    let kv_capacity = KvCapacity {
        total_blocks: backend.capacity_pages_for_requests(),
        block_size: backend.page_size(),
    };

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = watch::channel(SchedulerMetrics {
        kv_total_blocks: kv_capacity.total_blocks as u64,
        ..SchedulerMetrics::default()
    });
    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35-tp".into())
        .spawn(move || {
            scheduler_loop(
                SchedulerBackend::Tp(backend),
                submit_rx,
                seed,
                max_prefill_tokens,
                Qwen35SchedulerPolicy::Off,
                load_tx,
            );
        })
        .expect("failed to spawn Qwen3.5 TP scheduler thread");

    Ok(
        SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
            .with_servable_len(servable)
            .with_kv_capacity(kv_capacity)
            .with_metrics_watch(load_rx),
    )
}

struct SingleGpuBackend {
    model: Qwen35Model,
    graph_state: BatchDecodeGraphState,
    prefill_stream: Option<Arc<CudaStream>>,
}

// One instance per scheduler; the size asymmetry costs nothing here.
#[allow(clippy::large_enum_variant)]
enum SchedulerBackend {
    Single(SingleGpuBackend),
    Tp(TpSchedulerBackend),
}

struct AsyncPrefillOutput {
    logits: Option<HiddenStates>,
    done: CudaEvent,
    stream: Arc<CudaStream>,
    completed: bool,
}

impl AsyncPrefillOutput {
    fn is_ready(&mut self) -> bool {
        match unsafe { sys::cuEventQuery(self.done.cu_event()) } {
            sys::CUresult::CUDA_SUCCESS => {
                self.completed = true;
                true
            }
            sys::CUresult::CUDA_ERROR_NOT_READY => false,
            err => fatal_cuda_lifecycle(&format!(
                "query Qwen3.5 async prefill event failed: {err:?}"
            )),
        }
    }

    fn into_logits(mut self) -> HiddenStates {
        if !self.completed {
            if let Err(err) = self.done.synchronize() {
                fatal_cuda_lifecycle(&format!("wait for Qwen3.5 async prefill failed: {err}"));
            }
            self.completed = true;
        }
        self.logits
            .take()
            .expect("async prefill logits must be consumed exactly once")
    }
}

impl Drop for AsyncPrefillOutput {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Err(err) = self.stream.synchronize() {
            fatal_cuda_lifecycle(&format!(
                "drain Qwen3.5 async prefill during cleanup failed: {err}"
            ));
        }
    }
}

fn fatal_cuda_lifecycle(message: &str) -> ! {
    log::error!("FATAL: {message}; aborting before CUDA-referenced state is released");
    std::process::abort();
}

struct TpSchedulerBackend {
    executor: Qwen35TpExecutor,
    next_request_id: u64,
}

impl SingleGpuBackend {
    fn new(
        model: Qwen35Model,
        max_batch: usize,
        decode_overlap: Qwen35DecodeOverlap,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "Qwen3.5 max_batch must be > 0");
        let graph_capacity = crate::batch_decode_graph::bucket_for(max_batch);
        let graph_state = model.create_batch_decode_graph_state_with_capacity(graph_capacity)?;
        let prefill_stream = match decode_overlap {
            Qwen35DecodeOverlap::Off => None,
            Qwen35DecodeOverlap::SharedSm => Some(
                model
                    .device_ctx()
                    .ctx
                    .new_stream()
                    .map_err(|err| anyhow::anyhow!("create Qwen3.5 prefill stream: {err}"))?,
            ),
        };
        Ok(Self {
            model,
            graph_state,
            prefill_stream,
        })
    }

    fn model(&self) -> &Qwen35Model {
        &self.model
    }

    fn max_batch(&self) -> usize {
        // #470: admit the requested `--max-batch`, which may sit below the loaded
        // graph bucket (e.g. 5 on bucket 8); never exceed the physical slots.
        self.model
            .decode_admission_batch
            .min(self.graph_state.slot_states.len())
            .max(1)
    }

    fn page_size(&self) -> usize {
        self.model.kv_pool().layout().page_size
    }

    fn available_pages(&self) -> usize {
        self.model.kv_pool().available_pages()
    }

    fn capacity_pages_for_requests(&self) -> usize {
        self.model.kv_pool().capacity_pages().saturating_sub(1)
    }

    fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
    }

    fn alloc_recurrent(&self) -> Result<RecurrentState> {
        RecurrentState::new(self.model.device_ctx(), self.model.config())
    }

    fn batch_prefill_logits(&self, chunk: &mut ScheduledChunk) -> Result<HiddenStates> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU prefill received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        self.model
            .batch_prefill_logits(&window_refs, kvs, &mut rec_refs)
    }

    fn overlap_enabled(&self) -> bool {
        self.prefill_stream.is_some()
    }

    fn launch_async_prefill(&mut self, chunk: &mut ScheduledChunk) -> Result<AsyncPrefillOutput> {
        let prefill_stream = self
            .prefill_stream
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 decode overlap is disabled"))?;

        // Request KV/recurrent state was allocated on the model stream. Order
        // those producers before the prefill stream without blocking the host.
        prefill_stream
            .join(&self.model.device_ctx().stream)
            .map_err(|err| anyhow::anyhow!("join Qwen3.5 prefill stream: {err}"))?;

        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU async prefill received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let logits = match self.model.batch_prefill_logits_on_stream(
            Arc::clone(&prefill_stream),
            &window_refs,
            kvs,
            &mut rec_refs,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                if let Err(sync_err) = prefill_stream.synchronize() {
                    fatal_cuda_lifecycle(&format!(
                        "Qwen3.5 async prefill failed ({err}); stream drain failed: {sync_err}"
                    ));
                }
                return Err(err);
            }
        };
        let done = match prefill_stream.record_event(None) {
            Ok(done) => done,
            Err(err) => {
                if let Err(sync_err) = prefill_stream.synchronize() {
                    fatal_cuda_lifecycle(&format!(
                        "record Qwen3.5 async prefill event failed ({err}); stream drain failed: {sync_err}"
                    ));
                }
                return Err(anyhow::anyhow!("record Qwen3.5 async prefill event: {err}"));
            }
        };
        Ok(AsyncPrefillOutput {
            logits: Some(logits),
            done,
            stream: prefill_stream,
            completed: false,
        })
    }

    fn unified_step(
        &mut self,
        chunk: &mut ScheduledChunk,
        active: &mut [ActiveRequest35],
    ) -> Result<crate::unified_forward::UnifiedStepOutput> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU unified step received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut decode_kv_refs: Vec<&mut KvState> = active
            .iter_mut()
            .map(|r| match &mut r.backend_state {
                ActiveBackendState::Single { kv, .. } => kv,
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU unified step received TP active state")
                }
            })
            .collect();
        self.model.unified_step(
            &window_refs,
            kvs,
            &mut rec_refs,
            &decode_tokens,
            &mut decode_kv_refs,
            &mut self.graph_state,
        )
    }

    fn decode_graph(&mut self, active: &mut [ActiveRequest35]) -> Result<()> {
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut kv_refs: Vec<&mut KvState> = active
            .iter_mut()
            .map(|r| match &mut r.backend_state {
                ActiveBackendState::Single { kv, .. } => kv,
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU decode received TP active state")
                }
            })
            .collect();
        self.model
            .batch_decode_graph(&token_ids, &mut kv_refs, &mut self.graph_state)
    }

    fn sample_prefill_logits(
        &mut self,
        pending: &[SchedulerRequest],
        logits: &HiddenStates,
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        debug_assert_eq!(
            logits.seq_len,
            pending.len(),
            "Qwen3.5 prefill logits rows must preserve pending request order"
        );
        let requested_logprobs: Vec<usize> = pending.iter().map(|r| r.logprobs).collect();
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &requested_logprobs)?;
        let params_refs: Vec<&SamplingParams> = pending.iter().map(|r| &r.params).collect();
        let tokens = self.model.select_tokens_from_logits_varied(
            logits,
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        pending[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    fn sample_decode_logits(
        &mut self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let requested_logprobs: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
        let cpu_logits = snapshot_requested_logprobs(
            self.model.device_ctx(),
            &self.graph_state.buffers.logits,
            &requested_logprobs,
        )?;
        let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
        let tokens = self.model.select_tokens_batch_varied(
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        active[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.model.is_stop_token(token)
    }

    fn copy_recurrent_to_slot(
        &mut self,
        recurrent: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        self.graph_state
            .copy_state_to_slot(self.model.device_ctx(), recurrent, slot_idx)
    }

    fn compact_slot(&mut self, active: &mut [ActiveRequest35], compaction: plan::SlotCompaction) {
        let src_slot = match active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => graph_slot_idx,
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        };
        debug_assert_eq!(src_slot, compaction.moved_from);

        let ctx = self.model.device_ctx();
        let src = &self.graph_state.slot_states[compaction.moved_from];
        for layer_idx in 0..src.layers.len() {
            let (src_part, dst_part) = if compaction.moved_to < compaction.moved_from {
                let (left, right) = self
                    .graph_state
                    .slot_states
                    .split_at_mut(compaction.moved_from);
                (
                    &right[0].layers[layer_idx],
                    &mut left[compaction.moved_to].layers[layer_idx],
                )
            } else {
                unreachable!("idx < active.len() <= last");
            };

            ctx.stream
                .memcpy_dtod(&src_part.state, &mut dst_part.state)
                .expect("compact slot state copy failed");
            ctx.stream
                .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                .expect("compact slot conv_state copy failed");
        }
        self.graph_state.slot_states[compaction.moved_to].seq_len =
            self.graph_state.slot_states[compaction.moved_from].seq_len;

        match &mut active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => {
                *graph_slot_idx = compaction.moved_to;
            }
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        }
    }
}

impl TpSchedulerBackend {
    fn new(
        model_path: &str,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        let executor = Qwen35TpExecutor::from_runtime_with_limits(
            model_path,
            false,
            device_ordinals,
            max_batch,
            max_prefill_tokens,
        )?;
        Ok(Self {
            executor,
            next_request_id: 1,
        })
    }

    fn alloc_request_id(&mut self) -> RequestId {
        let id = RequestId::new(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        id
    }

    fn max_batch(&self) -> usize {
        self.executor.max_batch()
    }

    fn page_size(&self) -> usize {
        self.executor.page_size()
    }

    fn capacity_pages_for_requests(&self) -> usize {
        self.executor.capacity_pages_for_requests()
    }

    fn max_position_embeddings(&self) -> usize {
        self.executor.max_position_embeddings()
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.executor.is_stop_token(token)
    }

    fn available_pages(
        &self,
        active: &[ActiveRequest35],
        prefilling: &[PrefillingRequest35],
    ) -> usize {
        let page_size = self.page_size();
        let active_pages: usize = active
            .iter()
            .map(|req| pages_needed(current_active_tokens(req), page_size))
            .sum();
        let prefilling_pages: usize = prefilling
            .iter()
            .map(|req| pages_needed(req.cursor, page_size))
            .sum();
        self.capacity_pages_for_requests()
            .saturating_sub(active_pages.saturating_add(prefilling_pages))
    }

    fn execute_prefill_chunk(
        &self,
        chunk: &ScheduledChunk,
        sample_seed: u64,
    ) -> Result<Vec<Option<PrefillArtifact>>> {
        let items = tp_prefill_items(chunk)?;
        let result = self
            .executor
            .execute_prefill_chunks_with_seed(&items, sample_seed)?;
        align_prefill_results(chunk, &result)
            .map_err(|err| self.executor.poison_artifact_contract("prefill", &err))
    }

    fn execute_decode(
        &self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<Vec<DecodeArtifact>> {
        let items = tp_decode_items(active)?;
        let result = self.executor.execute_decode_items(&items, sample_seed)?;
        align_decode_results(active, &result)
            .map_err(|err| self.executor.poison_artifact_contract("decode", &err))
    }

    fn execute_unified(
        &self,
        chunk: &ScheduledChunk,
        active: &[ActiveRequest35],
        decode_sample_seed: u64,
        prefill_sample_seed: u64,
    ) -> Result<AlignedUnifiedArtifacts> {
        let plan = TpUnifiedPlan {
            prefill: tp_prefill_items(chunk)?,
            decode: tp_decode_items(active)?,
            prefill_sample_seed,
            decode_sample_seed,
        };
        let result = self.executor.execute_unified(&plan)?;
        let prefill = align_prefill_results(chunk, &result.prefill).map_err(|err| {
            self.executor
                .poison_artifact_contract("unified prefill", &err)
        })?;
        let decode = align_decode_results(active, &result.decode).map_err(|err| {
            self.executor
                .poison_artifact_contract("unified decode", &err)
        })?;
        Ok(AlignedUnifiedArtifacts { prefill, decode })
    }

    fn drop_request(&self, request_id: RequestId, expectation: DropExpectation) -> Result<()> {
        self.executor.drop_request(request_id, expectation)
    }
}

impl SchedulerBackend {
    fn max_batch(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_batch(),
            Self::Tp(backend) => backend.max_batch(),
        }
    }

    fn page_size(&self) -> usize {
        match self {
            Self::Single(backend) => backend.page_size(),
            Self::Tp(backend) => backend.page_size(),
        }
    }

    fn available_pages(
        &self,
        active: &[ActiveRequest35],
        prefilling: &[PrefillingRequest35],
    ) -> usize {
        match self {
            Self::Single(backend) => backend.available_pages(),
            Self::Tp(backend) => backend.available_pages(active, prefilling),
        }
    }

    fn capacity_pages_for_requests(&self) -> usize {
        match self {
            Self::Single(backend) => backend.capacity_pages_for_requests(),
            Self::Tp(backend) => backend.capacity_pages_for_requests(),
        }
    }

    fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_position_embeddings(),
            Self::Tp(backend) => backend.max_position_embeddings(),
        }
    }

    fn alloc_prefill_state(&mut self) -> Result<PrefillBackendState> {
        match self {
            Self::Single(backend) => Ok(PrefillBackendState::Single {
                kv: backend.alloc_kv(),
                rec: backend.alloc_recurrent()?,
            }),
            Self::Tp(backend) => Ok(PrefillBackendState::Tp {
                request_id: backend.alloc_request_id(),
            }),
        }
    }

    fn is_stop_token(&self, token: u32) -> bool {
        match self {
            Self::Single(backend) => backend.is_stop_token(token),
            Self::Tp(backend) => backend.is_stop_token(token),
        }
    }
}

fn current_active_tokens(req: &ActiveRequest35) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

fn tp_prefill_items(chunk: &ScheduledChunk) -> Result<Vec<TpPrefillChunkItem>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("TP prefill received single-GPU chunk state");
    };
    anyhow::ensure!(
        chunk.reqs.len() == request_ids.len()
            && chunk.reqs.len() == chunk.windows.len()
            && chunk.reqs.len() == chunk.ends.len(),
        "Qwen3.5 TP scheduled prefill vectors are misaligned"
    );
    Ok(chunk
        .reqs
        .iter()
        .zip(request_ids)
        .zip(&chunk.windows)
        .zip(&chunk.ends)
        .map(|(((req, request_id), window), end)| {
            TpPrefillChunkItem::new_with_sampling(
                *request_id,
                window.clone(),
                req.logprobs,
                req.params,
                *end == req.prompt_tokens.len(),
            )
        })
        .collect())
}

fn tp_decode_items(active: &[ActiveRequest35]) -> Result<Vec<TpDecodeStepItem>> {
    active
        .iter()
        .map(|req| {
            let ActiveBackendState::Tp { request_id } = &req.backend_state else {
                anyhow::bail!("TP decode received single-GPU active state");
            };
            Ok(TpDecodeStepItem::new(
                *request_id,
                req.last_token,
                req.logprobs,
                req.params,
            ))
        })
        .collect()
}

fn align_prefill_results(
    chunk: &ScheduledChunk,
    result: &PrefillResult,
) -> Result<Vec<Option<PrefillArtifact>>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("align_prefill_results requires TP chunk state");
    };
    anyhow::ensure!(
        request_ids.len() == chunk.reqs.len() && chunk.ends.len() == chunk.reqs.len(),
        "Qwen3.5 TP prefill alignment vectors are misaligned"
    );
    let expected: HashSet<RequestId> = request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .filter_map(|((&request_id, req), &end)| {
            (end == req.prompt_tokens.len()).then_some(request_id)
        })
        .collect();
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for PrefillRequestResult {
        request_id,
        first_token,
        first_token_logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected.contains(request_id),
            "Qwen3.5 TP prefill returned unknown or non-final request id {}",
            request_id.get()
        );
        let artifact = PrefillArtifact {
            token: *first_token,
            logprob: first_token_logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP prefill returned duplicate request id {}",
            request_id.get()
        );
    }
    anyhow::ensure!(
        by_id.len() == expected.len(),
        "Qwen3.5 TP prefill result is missing final request IDs"
    );

    request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .map(|((&request_id, req), &end)| {
            if end == req.prompt_tokens.len() {
                by_id.remove(&request_id).map(Some).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.5 TP prefill result is missing final request id {}",
                        request_id.get()
                    )
                })
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn align_decode_results(
    active: &[ActiveRequest35],
    result: &DecodeResult,
) -> Result<Vec<DecodeArtifact>> {
    let expected: Vec<RequestId> = active
        .iter()
        .map(|active_req| {
            let ActiveBackendState::Tp { request_id } = active_req.backend_state else {
                anyhow::bail!("align_decode_results requires TP active state");
            };
            Ok(request_id)
        })
        .collect::<Result<_>>()?;
    let expected_set: HashSet<_> = expected.iter().copied().collect();
    anyhow::ensure!(
        expected_set.len() == expected.len(),
        "Qwen3.5 TP active decode IDs contain duplicates"
    );
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for DecodeRequestResult {
        request_id,
        token,
        logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected_set.contains(request_id),
            "Qwen3.5 TP decode returned unknown request id {}",
            request_id.get()
        );
        let artifact = DecodeArtifact {
            token: *token,
            logprob: logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP decode returned duplicate request id {}",
            request_id.get()
        );
    }
    expected
        .into_iter()
        .map(|request_id| {
            by_id.remove(&request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode result is missing request id {}",
                    request_id.get()
                )
            })
        })
        .collect()
}

fn split_decode_artifacts(artifacts: &[DecodeArtifact]) -> (Vec<u32>, Vec<Option<TokenLogprob>>) {
    artifacts
        .iter()
        .map(|artifact| (artifact.token, artifact.logprob.clone()))
        .unzip()
}

fn servable_len(max_context: usize, max_pages: usize, page_size: usize) -> u32 {
    max_context
        .min(max_pages.saturating_mul(page_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_model_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 scheduler thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 scheduler thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    model.tune_decode_gemm_algos()?;
    Ok(CublasThreadGuard)
}

// ── Main loop ───────────────────────────────────────────────────────────

fn publish_load(
    load_tx: &watch::Sender<SchedulerMetrics>,
    backend: &SchedulerBackend,
    active: &[ActiveRequest35],
    prefilling: &[PrefillingRequest35],
    inflight_prefill_reqs: usize,
    num_waiting_reqs: usize,
) {
    let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
    let (num_running_reqs, num_waiting_reqs) =
        logical_load_counts(active, prefilling, inflight_prefill_reqs, num_waiting_reqs);
    load_tx.send_replace(SchedulerMetrics {
        kv_used_blocks: kv_total_blocks
            .saturating_sub(backend.available_pages(active, prefilling) as u64),
        kv_total_blocks,
        num_running_reqs,
        num_waiting_reqs,
        spec_decode: None,
    });
}

fn logical_load_counts(
    active: &[ActiveRequest35],
    prefilling: &[PrefillingRequest35],
    inflight_prefill_reqs: usize,
    num_waiting_reqs: usize,
) -> (u64, u64) {
    (
        (active.len() + prefilling.len() + inflight_prefill_reqs) as u64,
        num_waiting_reqs as u64,
    )
}

fn should_block_on_submit(owned_work_empty: bool, inflight_prefill: bool) -> bool {
    owned_work_empty && !inflight_prefill
}

fn terminal_scheduler_shutdown(
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

fn prune_closed_requests<B>(
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

const UNSUPPORTED_ECHO_MESSAGE: &str = "echo=true is unsupported by the Qwen3.5 serving contract";

fn reject_unsupported_echo(pending: &mut Vec<SchedulerRequest>) {
    pending.retain(|req| {
        if !req.echo {
            return true;
        }
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message: UNSUPPORTED_ECHO_MESSAGE.to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        false
    });
}

#[allow(clippy::needless_pass_by_value)]
fn scheduler_loop(
    mut backend: SchedulerBackend,
    mut submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    seed: u64,
    prefill_budget: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    load_tx: watch::Sender<SchedulerMetrics>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();
    let mut prefilling: Vec<PrefillingRequest35> = Vec::new();
    let mut inflight_prefill: Option<InflightPrefill> = None;
    let max_batch = backend.max_batch();

    info!("scheduler ready (max_batch={})", max_batch);

    loop {
        if inflight_prefill
            .as_mut()
            .is_some_and(|prefill| prefill.output.is_ready())
        {
            let (prefill_tokens, prefill_reqs) =
                inflight_prefill.as_ref().map_or((0, 0), |prefill| {
                    (
                        prefill.chunk.windows.iter().map(Vec::len).sum(),
                        prefill.chunk.reqs.len(),
                    )
                });
            let decode_n = active.len();
            let step_start = itl_debug_enabled().then(Instant::now);
            let finish_result = finish_async_prefill(
                &mut backend,
                &mut active,
                &mut prefilling,
                inflight_prefill
                    .take()
                    .expect("ready async prefill must still be present"),
            );
            log_itl_step(
                step_start,
                "overlap_complete",
                prefill_tokens,
                prefill_reqs,
                decode_n,
            );
            if let Err(failure) = finish_result {
                let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
                terminal_scheduler_shutdown(
                    &mut submit_rx,
                    &load_tx,
                    kv_total_blocks,
                    active,
                    prefilling,
                    Vec::new(),
                    deferred,
                    inflight_prefill.take(),
                    failure,
                );
                return;
            }
        }

        // 1. Merge deferred work with every submission currently available.
        let mut pending = std::mem::take(&mut deferred);
        while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Remove closed work before metrics, admission, or planning. Active
        // and prefilling cleanup goes through the backend's normal retirement
        // paths so graph slots and TP request state are released consistently.
        if let Err(failure) =
            prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending)
        {
            let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
            terminal_scheduler_shutdown(
                &mut submit_rx,
                &load_tx,
                kv_total_blocks,
                active,
                prefilling,
                pending,
                deferred,
                inflight_prefill.take(),
                failure,
            );
            return;
        }
        reject_unsupported_echo(&mut pending);

        // 3. Publish the settled post-prune state. Requests accepted from the
        // channel are waiting until admission below; closed requests never
        // appear in this snapshot or consume its KV/slot accounting.
        publish_load(
            &load_tx,
            &backend,
            &active,
            &prefilling,
            inflight_prefill
                .as_ref()
                .map_or(0, |prefill| prefill.chunk.reqs.len()),
            pending.len(),
        );

        // 4. Nothing in flight and nothing pending: the idle snapshot above is
        // already visible, so block until work arrives. Drain and prune again
        // after wakeup because the first request may already be closed and more
        // submissions may have raced with the blocking receive.
        if should_block_on_submit(
            active.is_empty() && prefilling.is_empty() && pending.is_empty(),
            inflight_prefill.is_some(),
        ) {
            if let Some((req, _kv_prefix)) = submit_rx.blocking_recv() {
                pending.push(req);
            } else {
                info!("scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
                pending.push(req);
            }
            if let Err(failure) =
                prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending)
            {
                let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
                terminal_scheduler_shutdown(
                    &mut submit_rx,
                    &load_tx,
                    kv_total_blocks,
                    active,
                    prefilling,
                    pending,
                    deferred,
                    inflight_prefill.take(),
                    failure,
                );
                return;
            }
            reject_unsupported_echo(&mut pending);
            publish_load(&load_tx, &backend, &active, &prefilling, 0, pending.len());
            if pending.is_empty() {
                continue;
            }
        }

        // One async prefill owns its scheduled request state. Do not admit or
        // launch a second chunk until it resolves. Active decode keeps moving;
        // if it retires first, wait on the event instead of blocking on submit.
        if inflight_prefill.is_some() {
            deferred = pending;
            let itl_step_start = itl_debug_enabled().then(Instant::now);
            let (itl_prefill_tokens, itl_prefill_reqs) =
                inflight_prefill.as_ref().map_or((0, 0), |prefill| {
                    (
                        prefill.chunk.windows.iter().map(Vec::len).sum(),
                        prefill.chunk.reqs.len(),
                    )
                });
            let itl_decode_n = active.len();
            let (itl_plan_kind, step_result) = if active.is_empty() {
                let result = finish_async_prefill(
                    &mut backend,
                    &mut active,
                    &mut prefilling,
                    inflight_prefill
                        .take()
                        .expect("async prefill must be present before blocking wait"),
                );
                ("overlap_wait", result)
            } else {
                let result = decode_step(&mut backend, &mut active, &mut rng);
                ("overlap_decode", result)
            };
            log_itl_step(
                itl_step_start,
                itl_plan_kind,
                itl_prefill_tokens,
                itl_prefill_reqs,
                itl_decode_n,
            );
            if let Err(failure) = step_result {
                let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
                terminal_scheduler_shutdown(
                    &mut submit_rx,
                    &load_tx,
                    kv_total_blocks,
                    active,
                    prefilling,
                    Vec::new(),
                    deferred,
                    inflight_prefill.take(),
                    failure,
                );
                return;
            }
            continue;
        }

        // 5. Admit new prompts. In-flight prefills reserve their promotion slot
        //    and future KV growth, so shrink the slot/page budgets accordingly
        let active_budget: Vec<ActiveKvBudget> = active
            .iter()
            .map(|req| ActiveKvBudget {
                prompt_len: req.prompt_len,
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let page_size = backend.page_size();
        let prefilling_budget: Vec<PrefillKvBudget> = prefilling
            .iter()
            .map(|p| PrefillKvBudget {
                current_tokens: p.cursor,
                prompt_len: p.req.prompt_tokens.len(),
                max_tokens: p.req.max_tokens,
            })
            .collect();
        let page_budget = backend
            .available_pages(&active, &prefilling)
            .saturating_sub(prefilling_future_pages(&prefilling_budget, page_size));
        let decode_batching_slot = max_batch.saturating_sub(prefilling.len());
        let admission = admit_pending_requests(
            pending,
            &active_budget,
            decode_batching_slot,
            page_size,
            page_budget,
            // KvPool capacity includes the CUDA Graph padding page reserved at
            // construction, so a real request can use at most the remaining pages.
            backend.capacity_pages_for_requests(),
            backend.max_position_embeddings(),
            |req| req.prompt_tokens.len(),
            |req| req.max_tokens,
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
        }

        // 6. Move freshly admitted prompts into the chunked-prefill queue.
        for req in admission.pending {
            debug!(
                "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                req.request_id,
                req.prompt_tokens.len(),
                req.max_tokens
            );
            match backend.alloc_prefill_state() {
                Ok(backend_state) => prefilling.push(PrefillingRequest35 {
                    backend_state,
                    cursor: 0,
                    step_chunk: 0,
                    req,
                }),
                Err(e) => {
                    warn!("failed to allocate recurrent state for new request: {e}");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: e.to_string(),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
            }
        }

        deferred = admission.deferred;

        // 7. Choose this tick's prefill budget, take that chunk off the front of
        //    the queue, then dispatch by plan. Auto can return 0 for a short
        //    decode-priority tick; the next iteration reconsiders the same FIFO
        //    prefill without reordering it.
        let active_decode: Vec<ActiveDecodeState> = active
            .iter()
            .map(|req| ActiveDecodeState {
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let prefill_queue: Vec<PrefillQueueState> = prefilling
            .iter()
            .map(|req| PrefillQueueState {
                remaining_tokens: req.req.prompt_tokens.len().saturating_sub(req.cursor),
            })
            .collect();
        let step_prefill_budget = choose_prefill_budget(
            scheduler_policy,
            prefill_budget,
            &active_decode,
            &prefill_queue,
        );
        let scheduled = take_prefill_chunks(&mut prefilling, step_prefill_budget);
        // ITL diagnostics (#470): capture the *actual* prefill-chunk token count
        // and the frozen decode width for this step before the plan consumes the
        // scheduled set. Off unless PEGAINFER_ITL_DEBUG is set.
        let itl_debug = itl_debug_enabled();
        let itl_prefill_tokens: usize = scheduled.iter().map(|p| p.step_chunk).sum();
        let itl_prefill_reqs = scheduled.len();
        let itl_decode_n = active.len();
        let plan = plan::build_next_plan(!active.is_empty(), scheduled);
        if let Some(plan) = plan {
            let itl_plan_kind = match &plan {
                ExecutionPlan::Unified { .. } if matches!(&backend, SchedulerBackend::Single(single) if single.overlap_enabled()) => {
                    "overlap_launch"
                }
                ExecutionPlan::Unified { .. } => "unified",
                ExecutionPlan::Prefill { .. } => "prefill",
                ExecutionPlan::Decode => "decode",
            };
            let itl_step_start = itl_debug.then(Instant::now);
            let step_result = match plan {
                ExecutionPlan::Unified { pending } => {
                    if matches!(&backend, SchedulerBackend::Single(single) if single.overlap_enabled())
                    {
                        launch_overlap_step(
                            &mut backend,
                            &mut active,
                            pending,
                            &mut inflight_prefill,
                            &mut rng,
                        )
                    } else {
                        unified_step_sched(
                            &mut backend,
                            &mut active,
                            pending,
                            &mut prefilling,
                            &mut rng,
                        )
                    }
                }
                ExecutionPlan::Prefill { pending } => prefill_batch(
                    &mut backend,
                    &mut active,
                    pending,
                    &mut prefilling,
                    &mut rng,
                ),
                ExecutionPlan::Decode => decode_step(&mut backend, &mut active, &mut rng),
            };
            log_itl_step(
                itl_step_start,
                itl_plan_kind,
                itl_prefill_tokens,
                itl_prefill_reqs,
                itl_decode_n,
            );
            if let Err(failure) = step_result {
                let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
                terminal_scheduler_shutdown(
                    &mut submit_rx,
                    &load_tx,
                    kv_total_blocks,
                    active,
                    prefilling,
                    Vec::new(),
                    deferred,
                    inflight_prefill.take(),
                    failure,
                );
                return;
            }
        }
    }
}

fn send_rejection(req: &SchedulerRequest, reason: RejectReason) {
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

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
) -> std::result::Result<(), FatalSchedulerError> {
    let mut chunk = ScheduledChunk::from(scheduled);
    let sample_seed = rand::RngExt::random(rng);
    let artifacts = match backend {
        SchedulerBackend::Single(single) => {
            // Scope the borrows of `chunk` to the executor call so the error path can
            // move `chunk` into `fail_chunk`.
            let logits = match single.batch_prefill_logits(&mut chunk) {
                Ok(v) => v,
                Err(e) => {
                    warn!("batch prefill failed: {e}");
                    fail_chunk(chunk, &e.to_string());
                    return Ok(());
                }
            };
            let prefill_sample_seed = rand::RngExt::random(rng);
            match single.sample_prefill_logits(&chunk.reqs, &logits, prefill_sample_seed) {
                Ok((tokens, logprobs)) => PrefillStepArtifacts::Single { tokens, logprobs },
                Err(e) => {
                    warn!("prefill sampling failed: {e}");
                    fail_chunk(chunk, &e.to_string());
                    return Ok(());
                }
            }
        }
        SchedulerBackend::Tp(tp) => match tp.execute_prefill_chunk(&chunk, sample_seed) {
            Ok(v) => PrefillStepArtifacts::Tp(v),
            Err(e) => {
                warn!("TP prefill chunk failed: {e}");
                return Err(FatalSchedulerError::new(e.to_string()).with_requests(chunk.reqs));
            }
        },
    };

    promote_or_requeue(backend, active, prefilling, chunk, &artifacts)
}

fn launch_overlap_step(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    inflight_prefill: &mut Option<InflightPrefill>,
    rng: &mut StdRng,
) -> std::result::Result<(), FatalSchedulerError> {
    debug_assert!(inflight_prefill.is_none());
    let mut chunk = ScheduledChunk::from(scheduled);
    let decode_seed = rand::RngExt::random(rng);
    let prefill_seed = rand::RngExt::random(rng);
    let output = match backend {
        SchedulerBackend::Single(single) => single.launch_async_prefill(&mut chunk),
        SchedulerBackend::Tp(_) => unreachable!("Qwen3.5 TP cannot launch async prefill"),
    };
    match output {
        Ok(output) => {
            *inflight_prefill = Some(InflightPrefill {
                chunk,
                output,
                sample_seed: prefill_seed,
            });
        }
        Err(err) => {
            warn!("async prefill launch failed: {err}");
            fail_chunk(chunk, &err.to_string());
        }
    }
    decode_step_with_seed(backend, active, decode_seed)
}

fn finish_async_prefill(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    inflight: InflightPrefill,
) -> std::result::Result<(), FatalSchedulerError> {
    let InflightPrefill {
        chunk,
        output,
        sample_seed,
    } = inflight;
    let logits = output.into_logits();
    let SchedulerBackend::Single(single) = backend else {
        unreachable!("Qwen3.5 TP cannot finish async prefill");
    };
    let (tokens, logprobs) = match single.sample_prefill_logits(&chunk.reqs, &logits, sample_seed) {
        Ok(result) => result,
        Err(err) => {
            warn!("async prefill sampling failed: {err}");
            fail_chunk(chunk, &err.to_string());
            return Ok(());
        }
    };
    let artifacts = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(single, active, prefilling, chunk, &artifacts)
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

fn unified_step_sched(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
) -> std::result::Result<(), FatalSchedulerError> {
    let mut chunk = ScheduledChunk::from(scheduled);
    if matches!(backend, SchedulerBackend::Tp(_)) {
        // Preserve the established scheduler RNG order: decode seed first,
        // prefill seed second. Workers execute the forwards in the opposite
        // (prefill-then-decode) order using these preselected seeds.
        let decode_sample_seed = rand::RngExt::random(rng);
        let prefill_sample_seed = rand::RngExt::random(rng);
        let result = {
            let SchedulerBackend::Tp(tp) = backend else {
                unreachable!()
            };
            tp.execute_unified(&chunk, active, decode_sample_seed, prefill_sample_seed)
        };
        let artifacts = match result {
            Ok(artifacts) => artifacts,
            Err(err) => {
                warn!("TP unified step failed: {err}");
                return Err(FatalSchedulerError::new(err.to_string()).with_requests(chunk.reqs));
            }
        };

        let (decode_tokens, decode_logprobs) = split_decode_artifacts(&artifacts.decode);
        if let Err(failure) =
            dispatch_decode_tokens(backend, active, &decode_tokens, &decode_logprobs)
        {
            return Err(failure.with_requests(chunk.reqs));
        }

        let prefill = PrefillStepArtifacts::Tp(artifacts.prefill);
        return promote_or_requeue(backend, active, prefilling, chunk, &prefill);
    }

    let SchedulerBackend::Single(backend) = backend else {
        unreachable!()
    };
    // Scope the borrows of `chunk` / `active` to the executor call so the error
    // and decode-processing paths can use them afterwards.
    let result = backend.unified_step(&mut chunk, active);
    let output = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("unified step failed: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            fail_chunk(chunk, &message);
            return Ok(());
        }
    };
    let decode_seed = rand::RngExt::random(rng);
    let prefill_seed = rand::RngExt::random(rng);

    // Process decode results FIRST (it may retire requests and free graph slots
    // that promotion then fills densely).
    if output.decoded {
        process_decode_logits(backend, active, decode_seed)?;
    }

    let prefill_logits = output
        .prefill_logits
        .as_ref()
        .expect("scheduled prefill chunk must return prefill logits");
    let (tokens, logprobs) =
        match backend.sample_prefill_logits(&chunk.reqs, prefill_logits, prefill_seed) {
            Ok(v) => v,
            Err(e) => {
                warn!("unified prefill sampling failed: {e}");
                fail_chunk(chunk, &e.to_string());
                return Ok(());
            }
        };
    let prefill = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(backend, active, prefilling, chunk, &prefill)
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    rng: &mut StdRng,
) -> std::result::Result<(), FatalSchedulerError> {
    // Preserve the historical scheduler RNG sequence: TP consumes the first
    // seed, while single-GPU decode consumed a second seed inside sampling.
    let first_seed = rand::RngExt::random(rng);
    let sample_seed = if matches!(backend, SchedulerBackend::Single(_)) {
        rand::RngExt::random(rng)
    } else {
        first_seed
    };
    decode_step_with_seed(backend, active, sample_seed)
}

fn decode_step_with_seed(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
) -> std::result::Result<(), FatalSchedulerError> {
    let (tokens, logprobs_vec) = match backend {
        SchedulerBackend::Single(single) => {
            if let Err(e) = single.decode_graph(active) {
                warn!("batch_decode_graph error: {e}");
                let message = e.to_string();
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return Ok(());
            }
            // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
            match single.sample_decode_logits(active, sample_seed) {
                Ok(v) => v,
                Err(e) => {
                    warn!("decode sampling/logprobs error: {e}");
                    let message = e.to_string();
                    for req in active.drain(..) {
                        let _ = req.token_tx.send(TokenEvent::Error {
                            message: message.clone(),
                            prompt_tokens: req.prompt_len,
                            completion_tokens: req.generated_count,
                        });
                    }
                    return Ok(());
                }
            }
        }
        SchedulerBackend::Tp(tp) => match tp.execute_decode(active, sample_seed) {
            Ok(v) => split_decode_artifacts(&v),
            Err(e) => {
                warn!("TP eager decode error: {e}");
                return Err(FatalSchedulerError::new(e.to_string()));
            }
        },
    };

    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec)
}

/// Process decode logits from unified step: sample, extract logprobs, dispatch.
fn process_decode_logits(
    backend: &mut SingleGpuBackend,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
) -> std::result::Result<(), FatalSchedulerError> {
    let (tokens, logprobs_vec) = match backend.sample_decode_logits(active, sample_seed) {
        Ok(v) => v,
        Err(e) => {
            warn!("decode sampling/logprobs error: {e}");
            let message = e.to_string();
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            return Ok(());
        }
    };

    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec)
}

/// Dispatch sampled decode tokens: send events, check EOS/limits, retire finished.
///
/// `tokens` and `logprobs` are indexed by original position in `active`.
/// Retirements collected first, then compacted in reverse order.
fn dispatch_decode_tokens(
    backend: &mut impl DecodeDispatchBackend,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
) -> std::result::Result<(), FatalSchedulerError> {
    enum Retirement {
        Completion(Vec<TokenEvent>),
        CleanupOnly,
        Disconnected,
    }

    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let token = tokens[i];
        let logprob = logprobs[i].clone();
        let req = &mut active[i];
        req.generated_count += 1;

        let is_eos = !req.params.ignore_eos && backend.is_stop_token(token);
        let at_limit = req.generated_count >= req.max_tokens;

        if is_eos {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Stop
            );
            let event = TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            };
            if backend.completion_requires_drop_ack() {
                to_retire.push((i, Retirement::Completion(vec![event])));
            } else {
                let _ = req.token_tx.send(event);
                to_retire.push((i, Retirement::CleanupOnly));
            }
        } else if at_limit {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Length
            );
            let events = vec![
                TokenEvent::Token { id: token, logprob },
                TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                },
            ];
            if backend.completion_requires_drop_ack() {
                to_retire.push((i, Retirement::Completion(events)));
            } else {
                for event in events {
                    let _ = req.token_tx.send(event);
                }
                to_retire.push((i, Retirement::CleanupOnly));
            }
        } else if req
            .token_tx
            .send(TokenEvent::Token { id: token, logprob })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, req.generated_count
            );
            to_retire.push((i, Retirement::Disconnected));
        } else {
            req.last_token = token;
        }
    }

    // Remove in reverse order so compact_slot indices stay valid
    for (i, retirement) in to_retire.into_iter().rev() {
        let request = backend.take_active_request(active, i);
        match retirement {
            Retirement::Completion(final_events) => {
                let candidate = CompletionCandidate {
                    request,
                    final_events,
                };
                if let Err(err) = backend.drop_active_state(&candidate.request.backend_state) {
                    return Err(FatalSchedulerError::new(err.to_string())
                        .with_request(candidate.into_terminal()));
                }
                candidate.commit();
            }
            Retirement::CleanupOnly | Retirement::Disconnected => {
                if let Err(err) = backend.drop_active_state(&request.backend_state) {
                    return Err(FatalSchedulerError::new(err.to_string()).with_request(request));
                }
            }
        }
    }
    Ok(())
}

trait DecodeDispatchBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn completion_requires_drop_ack(&self) -> bool;
    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35;
    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()>;
}

impl DecodeDispatchBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn completion_requires_drop_ack(&self) -> bool {
        false
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        compact_single_slot(self, active, idx)
    }

    fn drop_active_state(&mut self, _state: &ActiveBackendState) -> Result<()> {
        Ok(())
    }
}

impl DecodeDispatchBackend for SchedulerBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn completion_requires_drop_ack(&self) -> bool {
        matches!(self, SchedulerBackend::Tp(_))
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        match self {
            SchedulerBackend::Single(backend) => compact_single_slot(backend, active, idx),
            SchedulerBackend::Tp(_) => active.swap_remove(idx),
        }
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        match (self, state) {
            (SchedulerBackend::Single(_), ActiveBackendState::Single { .. }) => Ok(()),
            (SchedulerBackend::Tp(backend), ActiveBackendState::Tp { request_id }) => {
                backend.drop_request(*request_id, DropExpectation::MustExist)
            }
            _ => anyhow::bail!("mismatched Qwen3.5 scheduler backend state during retirement"),
        }
    }
}

/// Remove single-GPU request at `idx` via swap_remove and compact graph slots.
///
/// After swap_remove, the element that was at `active.len()-1` (before remove)
/// now sits at `idx`. Its graph slot must be copied into the vacated slot so
/// that slots 0..active.len() remain dense.
fn compact_single_slot(
    backend: &mut SingleGpuBackend,
    active: &mut Vec<ActiveRequest35>,
    idx: usize,
) -> ActiveRequest35 {
    let compaction = compaction_after_retire(active.len(), idx);
    let removed = active.swap_remove(idx);

    if let Some(compaction) = compaction {
        backend.compact_slot(active, compaction);
    }
    removed
}

// ── Chunked-prefill helpers ────────────────────────────────────────────────

/// Step's scheduled prefill set
struct ScheduledChunk {
    reqs: Vec<SchedulerRequest>,
    backend_state: ScheduledChunkBackendState,
    /// Prompt cursor after this step's chunk
    ends: Vec<usize>,
    /// This step's chunked token slice per request
    windows: Vec<Vec<u32>>,
}

struct InflightPrefill {
    // Fields drop in declaration order. Drain the stream before request state
    // can return KV pages or release recurrent/convolution buffers on unwind.
    output: AsyncPrefillOutput,
    chunk: ScheduledChunk,
    sample_seed: u64,
}

enum ScheduledChunkBackendState {
    Single {
        kvs: Vec<KvState>,
        recs: Vec<RecurrentState>,
    },
    Tp {
        request_ids: Vec<RequestId>,
    },
}

impl From<Vec<PrefillingRequest35>> for ScheduledChunk {
    fn from(scheduled: Vec<PrefillingRequest35>) -> Self {
        let n = scheduled.len();
        let is_tp = scheduled
            .first()
            .is_some_and(|p| matches!(p.backend_state, PrefillBackendState::Tp { .. }));
        let mut chunk = ScheduledChunk {
            reqs: Vec::with_capacity(n),
            backend_state: if is_tp {
                ScheduledChunkBackendState::Tp {
                    request_ids: Vec::with_capacity(n),
                }
            } else {
                ScheduledChunkBackendState::Single {
                    kvs: Vec::with_capacity(n),
                    recs: Vec::with_capacity(n),
                }
            },
            ends: Vec::with_capacity(n),
            windows: Vec::with_capacity(n),
        };
        for p in scheduled {
            let end = p.cursor + p.step_chunk;
            chunk
                .windows
                .push(p.req.prompt_tokens[p.cursor..end].to_vec());
            chunk.ends.push(end);
            chunk.reqs.push(p.req);
            match (&mut chunk.backend_state, p.backend_state) {
                (
                    ScheduledChunkBackendState::Single { kvs, recs },
                    PrefillBackendState::Single { kv, rec },
                ) => {
                    kvs.push(kv);
                    recs.push(rec);
                }
                (
                    ScheduledChunkBackendState::Tp { request_ids },
                    PrefillBackendState::Tp { request_id },
                ) => request_ids.push(request_id),
                _ => unreachable!("mixed Qwen3.5 scheduler backend states in one chunk"),
            }
        }
        chunk
    }
}

/// Pull this step's prefill set off the FRONT of `prefilling`, capping the
/// step's total forwarded prompt tokens at `prefill_budget`.
fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    prefill_budget: usize,
) -> Vec<PrefillingRequest35> {
    let remaining: Vec<usize> = prefilling
        .iter()
        .map(|p| p.req.prompt_tokens.len() - p.cursor)
        .collect();
    let chunks = plan_prefill_chunks(&remaining, prefill_budget);
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..chunks.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&chunks) {
        p.step_chunk = *chunk;
    }
    scheduled
}

/// Report a forward/sampling failure to every request in the failed chunk.
fn fail_chunk(chunk: ScheduledChunk, message: &str) {
    for req in chunk.reqs {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
    }
}

/// For each request in the just-prefilled chunk: if its prompt is now exhausted,
/// sample its first token, emit events, and move it into the decode batch;
/// otherwise re-queue it (with an advanced cursor) at the FRONT of `prefilling`.
/// `artifacts` are indexed by request order in `chunk`.
fn promote_or_requeue(
    backend: &mut impl PrefillPromoteBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    chunk: ScheduledChunk,
    artifacts: &PrefillStepArtifacts,
) -> std::result::Result<(), FatalSchedulerError> {
    let ScheduledChunk {
        reqs,
        backend_state,
        ends,
        ..
    } = chunk;
    let mut still_prefilling: Vec<PrefillingRequest35> = Vec::new();
    let backend_states = split_scheduled_backend_state(backend_state);
    let mut entries: VecDeque<_> = reqs
        .into_iter()
        .zip(backend_states)
        .zip(ends)
        .enumerate()
        .map(|(i, ((req, backend_state), end))| (i, req, backend_state, end))
        .collect();

    while let Some((i, req, backend_state, end)) = entries.pop_front() {
        // Not finished: re-queue with the advanced cursor
        if end < req.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                req,
                backend_state,
                cursor: end,
                step_chunk: 0,
            });
            continue;
        }

        let prompt_len = req.prompt_tokens.len();
        let artifact = artifacts.final_artifact(i);
        let first_token = artifact.token;
        let logprob = artifact.logprob;

        if !req.params.ignore_eos && backend.is_stop_token(first_token) {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                0,
                FinishReason::Stop
            );
            let candidate = CompletionCandidate {
                request: PrefillCompletionRequest { req, backend_state },
                final_events: vec![TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                }],
            };
            if let Err(err) = backend
                .drop_prefill_state(&candidate.request.backend_state, DropExpectation::MustExist)
            {
                return Err(prefill_lifecycle_failure(
                    err.to_string(),
                    candidate.into_terminal(),
                    still_prefilling,
                    entries,
                ));
            }
            candidate.commit();
            continue;
        }

        if req.max_tokens <= 1 {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                1,
                FinishReason::Length
            );
            let candidate = CompletionCandidate {
                request: PrefillCompletionRequest { req, backend_state },
                final_events: vec![
                    TokenEvent::Token {
                        id: first_token,
                        logprob,
                    },
                    TokenEvent::Finished {
                        finish_reason: FinishReason::Length,
                        prompt_tokens: prompt_len,
                        completion_tokens: 1,
                    },
                ],
            };
            if let Err(err) = backend
                .drop_prefill_state(&candidate.request.backend_state, DropExpectation::MustExist)
            {
                return Err(prefill_lifecycle_failure(
                    err.to_string(),
                    candidate.into_terminal(),
                    still_prefilling,
                    entries,
                ));
            }
            candidate.commit();
            continue;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: first_token,
                logprob,
            })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, 0
            );
            let removed = PrefillCompletionRequest { req, backend_state };
            if let Err(err) =
                backend.drop_prefill_state(&removed.backend_state, DropExpectation::MustExist)
            {
                return Err(prefill_lifecycle_failure(
                    err.to_string(),
                    removed.into_terminal(),
                    still_prefilling,
                    entries,
                ));
            }
            continue;
        }

        let active_backend_state = backend.promote_prefill_state(active.len(), backend_state);
        active.push(ActiveRequest35 {
            request_id: req.request_id,
            token_tx: req.token_tx,
            backend_state: active_backend_state,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
            logprobs: req.logprobs,
        });
    }

    prefilling.splice(0..0, still_prefilling);
    Ok(())
}

fn prefill_lifecycle_failure(
    message: String,
    current: TerminalRequest,
    still_prefilling: Vec<PrefillingRequest35>,
    remaining: VecDeque<(usize, SchedulerRequest, PrefillBackendState, usize)>,
) -> FatalSchedulerError {
    FatalSchedulerError::new(message)
        .with_request(current)
        .with_requests(still_prefilling)
        .with_requests(remaining.into_iter().map(|(_, req, _, _)| req))
}

trait PrefillPromoteBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState;
    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()>;
}

impl PrefillPromoteBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        let PrefillBackendState::Single { kv, rec } = state else {
            panic!("single-GPU promotion received TP prefill state");
        };
        let slot_idx = slot_for_new_request(active_len, self.max_batch())
            .expect("admission must reserve a graph slot");
        self.copy_recurrent_to_slot(&rec, slot_idx)
            .expect("copy recurrent state to slot failed");
        ActiveBackendState::Single {
            kv,
            graph_slot_idx: slot_idx,
        }
    }

    fn drop_prefill_state(
        &mut self,
        _state: &PrefillBackendState,
        _expectation: DropExpectation,
    ) -> Result<()> {
        Ok(())
    }
}

impl PrefillPromoteBackend for SchedulerBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        match (self, state) {
            (SchedulerBackend::Single(single), PrefillBackendState::Single { kv, rec }) => {
                let slot_idx = slot_for_new_request(active_len, single.max_batch())
                    .expect("admission must reserve a graph slot");
                single
                    .copy_recurrent_to_slot(&rec, slot_idx)
                    .expect("copy recurrent state to slot failed");
                ActiveBackendState::Single {
                    kv,
                    graph_slot_idx: slot_idx,
                }
            }
            (SchedulerBackend::Tp(_), PrefillBackendState::Tp { request_id }) => {
                ActiveBackendState::Tp { request_id }
            }
            _ => panic!("mismatched Qwen3.5 scheduler backend state during promotion"),
        }
    }

    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()> {
        match (self, state) {
            (SchedulerBackend::Single(_), PrefillBackendState::Single { .. }) => Ok(()),
            (SchedulerBackend::Tp(backend), PrefillBackendState::Tp { request_id }) => {
                backend.drop_request(*request_id, expectation)
            }
            _ => anyhow::bail!("mismatched Qwen3.5 scheduler backend state during prefill drop"),
        }
    }
}

fn split_scheduled_backend_state(
    backend_state: ScheduledChunkBackendState,
) -> Vec<PrefillBackendState> {
    match backend_state {
        ScheduledChunkBackendState::Single { kvs, recs } => kvs
            .into_iter()
            .zip(recs)
            .map(|(kv, rec)| PrefillBackendState::Single { kv, rec })
            .collect(),
        ScheduledChunkBackendState::Tp { request_ids } => request_ids
            .into_iter()
            .map(|request_id| PrefillBackendState::Tp { request_id })
            .collect(),
    }
}

#[cfg(test)]
mod tests;
