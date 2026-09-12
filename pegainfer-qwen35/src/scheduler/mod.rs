//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod backend;
mod emit;
mod plan;
mod steps;
mod telemetry;
mod tp;
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

use self::backend::*;
use self::emit::*;
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
use self::steps::*;
use self::telemetry::*;
use self::tp::*;
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
use crate::tp_executor::TpSlotCompaction;
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
    /// Optional top-logprob count; Some(0) scores only the chosen token.
    logprobs: Option<usize>,
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
        /// Dense decode slot (`active` position). Graph-mode workers assert
        /// `slot_idx == row` on every decode command; eager workers ignore it.
        slot_idx: usize,
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
        .spawn(move || {
            match crate::cublas_thread::bind_model_thread(backend.model(), "scheduler") {
                Ok(_guard) => {
                    if let Err(err) = backend.model().tune_decode_gemm_algos() {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
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
    enable_cuda_graph: bool,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let backend = TpSchedulerBackend::new(
        model_path,
        device_ordinals,
        max_batch,
        max_prefill_tokens,
        enable_cuda_graph,
    )?;
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

fn current_active_tokens(req: &ActiveRequest35) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

fn servable_len(max_context: usize, max_pages: usize, page_size: usize) -> u32 {
    max_context
        .min(max_pages.saturating_mul(page_size))
        .try_into()
        .unwrap_or(u32::MAX)
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
    let decode_overlap = matches!(
        &backend,
        SchedulerBackend::Single(single) if single.overlap_enabled()
    );

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
        reject_unsupported_prompt_logprobs(&mut pending);

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
            reject_unsupported_prompt_logprobs(&mut pending);
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
            decode_overlap,
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
                ExecutionPlan::Unified { .. } if decode_overlap => "overlap_launch",
                ExecutionPlan::Unified { .. } => "unified",
                ExecutionPlan::Prefill { .. } => "prefill",
                ExecutionPlan::Decode => "decode",
            };
            let itl_step_start = itl_debug.then(Instant::now);
            let step_result = match plan {
                ExecutionPlan::Unified { pending } => {
                    if decode_overlap {
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

#[cfg(test)]
mod tests;
