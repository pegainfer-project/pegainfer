//! Qwen3.5 step scheduler: batches requests on a dedicated GPU owner thread.
//!
//! Retains model-specific state alongside the frontend request ledger:
//! - `RecurrentState` alongside content-hashed `RequestKv` (hybrid attention)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod backend;
mod emit;
mod plan;
mod steps;
mod telemetry;
mod tp;
use std::collections::HashMap;
use std::collections::HashSet;
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
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::LiveScheduler;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason as EngineRejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId as FrontendRequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::drive;
use pegainfer_frontend::engine::panic_message;
use pegainfer_frontend::engine::scheduler_pair;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::KvView;
use pegainfer_kv_cache::RequestKv;
use rand::SeedableRng;
use rand::rngs::StdRng;

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
use crate::prefix_cache::Qwen35PrefixCache;
use crate::prefix_cache::RecurrentStateStore;
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
    id: FrontendRequestId,
    client_label: Option<Arc<str>>,
    backend_state: ActiveBackendState,
    last_token: u32,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
    stop_policy: StopPolicy,
    /// Optional top-logprob count; Some(0) scores only the chosen token.
    logprobs: Option<usize>,
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
/// It owns its growing KV and recurrent state until the prompt is exhausted,
/// at which point it is promoted into the decode batch.
struct PrefillingRequest35 {
    req: QueuedRequest,
    backend_state: PrefillBackendState,
    /// Prompt tokens prefilled so far.
    cursor: usize,
    /// Tokens to prefill in the step currently scheduled (set by `take_prefill_chunks`).
    step_chunk: usize,
}

enum ActiveBackendState {
    Single {
        kv: Box<RequestKv>,
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
    Single {
        kv: Box<RequestKv>,
        rec: RecurrentState,
    },
    Tp {
        request_id: RequestId,
    },
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

pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

// ── Entry point ─────────────────────────────────────────────────────────

pub fn start_with_capacity(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<Engine> {
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
) -> Result<Engine> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    // Static instance cap for the vLLM bridge's max_model_len. Live admission
    // still uses the current page budget inside the scheduler loop.
    let backend = SingleGpuBackend::new(model, max_batch, decode_overlap)?;
    let total_blocks = backend.capacity_pages_for_requests();
    let block_size = backend.page_size();
    let servable = servable_len(
        backend.model().config().max_position_embeddings,
        total_blocks,
        block_size,
    );

    let (handle, wiring) = scheduler_pair();
    let (startup_tx, startup_rx) = std_mpsc::channel();

    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(backend.model()) {
            Ok(_guard) => {
                let _ = startup_tx.send(Ok(()));
                drive(
                    Qwen35Scheduler::new(
                        SchedulerBackend::Single(backend),
                        seed,
                        max_prefill_tokens,
                        scheduler_policy,
                    ),
                    wiring,
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
    Ok(Engine {
        schedulers: vec![LiveScheduler {
            handle,
            join: join_handle,
        }],
        info: EngineInfo {
            servable_len: Some(servable),
            kv_capacity: Some(KvCapacity {
                total_blocks,
                block_size,
            }),
        },
        lora: None,
    })
}

pub(crate) fn start_tp_with_capacity(
    model_path: &str,
    seed: u64,
    device_ordinals: &[usize],
    max_batch: usize,
    max_prefill_tokens: usize,
    enable_cuda_graph: bool,
    prefix_snapshot_bytes: usize,
) -> Result<Engine> {
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
        prefix_snapshot_bytes,
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

    Ok(Engine {
        schedulers: vec![spawn_scheduler(
            "scheduler-qwen35-tp",
            Qwen35Scheduler::new(
                SchedulerBackend::Tp(backend),
                seed,
                max_prefill_tokens,
                Qwen35SchedulerPolicy::Off,
            ),
        )],
        info: EngineInfo {
            servable_len: Some(servable),
            kv_capacity: Some(kv_capacity),
        },
        lora: None,
    })
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
    let guard = CublasThreadGuard;
    model.tune_decode_gemm_algos()?;
    Ok(guard)
}

// ── Main loop ───────────────────────────────────────────────────────────

fn reject_or_retire(id: FrontendRequestId, reason: EngineRejectReason, ledger: &mut RequestLedger) {
    if ledger.is_aborted(id) {
        ledger.retire(id);
    } else {
        ledger.reject(id, reason);
    }
}

fn fail_or_retire(id: FrontendRequestId, message: impl Into<String>, ledger: &mut RequestLedger) {
    if ledger.is_aborted(id) {
        ledger.retire(id);
    } else {
        ledger.fail(id, message);
    }
}

struct Qwen35Scheduler {
    // Drain asynchronous work before releasing its request allocations or backend.
    inflight_prefill: Option<InflightPrefill>,
    active: Vec<ActiveRequest35>,
    prefilling: Vec<PrefillingRequest35>,
    pending: Vec<QueuedRequest>,
    rng: StdRng,
    prefill_budget: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    failed: bool,
    backend: SchedulerBackend,
}

impl Qwen35Scheduler {
    fn new(
        backend: SchedulerBackend,
        seed: u64,
        prefill_budget: usize,
        scheduler_policy: Qwen35SchedulerPolicy,
    ) -> Self {
        info!("scheduler ready (max_batch={})", backend.max_batch());
        Self {
            inflight_prefill: None,
            active: Vec::new(),
            prefilling: Vec::new(),
            pending: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            prefill_budget,
            scheduler_policy,
            failed: false,
            backend,
        }
    }

    fn advance(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let Self {
            inflight_prefill,
            active,
            prefilling,
            pending,
            rng,
            prefill_budget,
            scheduler_policy,
            backend,
            ..
        } = self;
        let decode_overlap = matches!(
            backend,
            SchedulerBackend::Single(single) if single.overlap_enabled()
        );
        let max_batch = backend.max_batch();

        let prefill_ready = inflight_prefill
            .as_mut()
            .is_some_and(|prefill| prefill.output.is_ready());
        if prefill_ready {
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
                backend,
                active,
                prefilling,
                inflight_prefill
                    .take()
                    .expect("ready async prefill must still be present"),
                ledger,
            );
            log_itl_step(
                step_start,
                "overlap_complete",
                prefill_tokens,
                prefill_reqs,
                decode_n,
            );
            finish_result?;
        }

        prune_closed_requests(backend, active, prefilling, pending, ledger)?;
        reject_unsupported_prompt_logprobs(pending, ledger);

        if prefill_ready {
            // Commit completed prefill updates before starting another prefill or decode.
            return Ok(());
        }

        // One async prefill owns its scheduled request state. Do not admit or
        // launch a second chunk until it resolves. Active decode keeps moving;
        // if it retires first, wait for the prefill event to complete.
        if inflight_prefill.is_some() {
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
                    backend,
                    active,
                    prefilling,
                    inflight_prefill
                        .take()
                        .expect("async prefill must be present before blocking wait"),
                    ledger,
                );
                ("overlap_wait", result)
            } else {
                let result = decode_step(backend, active, rng, ledger);
                ("overlap_decode", result)
            };
            log_itl_step(
                itl_step_start,
                itl_plan_kind,
                itl_prefill_tokens,
                itl_prefill_reqs,
                itl_decode_n,
            );
            return step_result;
        }

        if pending.is_empty() && active.is_empty() && prefilling.is_empty() {
            return Ok(());
        }

        // Admit new prompts. In-flight prefills reserve their promotion slot
        // and future KV growth, so shrink the slot/page budgets accordingly.
        let active_budget: Vec<ActiveKvBudget> = active
            .iter()
            .map(|req| ActiveKvBudget {
                prompt_len: req.prompt_len,
                generated_count: ledger.completion_tokens(req.id),
                max_tokens: req.max_tokens,
            })
            .collect();
        let page_size = backend.page_size();
        let mut admission_pending = std::mem::take(pending);
        loop {
            let prefilling_budget: Vec<PrefillKvBudget> = prefilling
                .iter()
                .map(|p| PrefillKvBudget {
                    current_tokens: p.cursor,
                    prompt_len: p.req.request.prompt_tokens.len(),
                    max_tokens: p.req.request.max_tokens,
                })
                .collect();
            let page_budget = backend
                .available_pages()
                .saturating_sub(prefilling_future_pages(&prefilling_budget, page_size));
            let decode_batching_slot = max_batch.saturating_sub(prefilling.len());
            let admission = admit_pending_requests(
                admission_pending,
                &active_budget,
                decode_batching_slot,
                page_size,
                page_budget,
                // KvPool capacity includes the CUDA Graph padding page reserved at
                // construction, so a real request can use at most the remaining pages.
                backend.capacity_pages_for_requests(),
                backend.max_position_embeddings(),
                |req| req.request.prompt_tokens.len(),
                |req| req.request.max_tokens,
                |req| backend.active_joint_prefix_pages(&req.request),
            );
            for (rejected, reason) in &admission.rejected {
                send_rejection(rejected, *reason, ledger);
            }

            // Materialize admitted prompts. A restore can move cached KV out
            // of the available pool, so re-run admission for FIFO followers.
            let mut restored_prefix = false;
            for req in admission.pending {
                debug!(
                    "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                    req.request.client_label,
                    req.request.prompt_tokens.len(),
                    req.request.max_tokens
                );
                match backend.alloc_prefill_state(&req.request) {
                    Ok((backend_state, cached_tokens)) => {
                        restored_prefix |= cached_tokens > 0;
                        ledger.admit(req.id);
                        ledger.set_cached_tokens(req.id, cached_tokens);
                        prefilling.push(PrefillingRequest35 {
                            req,
                            backend_state,
                            cursor: cached_tokens,
                            step_chunk: 0,
                        });
                    }
                    Err(AdmissionError::Recoverable(error)) => {
                        warn!("failed to admit new request: {error}");
                        fail_or_retire(req.id, error.to_string(), ledger);
                    }
                    Err(AdmissionError::Fatal(error)) => return Err(error),
                }
            }
            if admission.deferred.is_empty() || !restored_prefix {
                *pending = admission.deferred;
                break;
            }
            admission_pending = admission.deferred;
        }

        // Choose this tick's prefill budget, take that chunk off the front of
        // the queue, then dispatch by plan. Auto can return 0 for a short
        // decode-priority tick; the next iteration reconsiders the same FIFO
        // prefill without reordering it.
        let active_decode: Vec<ActiveDecodeState> = active
            .iter()
            .map(|req| ActiveDecodeState {
                generated_count: ledger.completion_tokens(req.id),
                max_tokens: req.max_tokens,
            })
            .collect();
        let prefill_queue: Vec<PrefillQueueState> = prefilling
            .iter()
            .map(|req| PrefillQueueState {
                remaining_tokens: req
                    .req
                    .request
                    .prompt_tokens
                    .len()
                    .saturating_sub(req.cursor),
            })
            .collect();
        let step_prefill_budget = choose_prefill_budget(
            *scheduler_policy,
            *prefill_budget,
            &active_decode,
            &prefill_queue,
            decode_overlap,
        );
        let scheduled =
            take_prefill_chunks(prefilling, step_prefill_budget, backend.snapshot_stride());
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
                        launch_overlap_step(backend, active, pending, inflight_prefill, rng, ledger)
                    } else {
                        unified_step_sched(backend, active, pending, prefilling, rng, ledger)
                    }
                }
                ExecutionPlan::Prefill { pending } => {
                    prefill_batch(backend, active, pending, prefilling, rng, ledger)
                }
                ExecutionPlan::Decode => decode_step(backend, active, rng, ledger),
            };
            log_itl_step(
                itl_step_start,
                itl_plan_kind,
                itl_prefill_tokens,
                itl_prefill_reqs,
                itl_decode_n,
            );
            step_result?;
        }
        Ok(())
    }
}

impl Scheduler for Qwen35Scheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.pending.push(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let result = self.advance(ledger);
        if result.is_err() {
            // The driver answers every open account after this returns.
            drop(self.inflight_prefill.take());
            self.active.clear();
            self.prefilling.clear();
            self.pending.clear();
            self.failed = true;
        }
        result
    }

    fn metrics(&self) -> SchedulerMetrics {
        let kv_total_blocks = self.backend.capacity_pages_for_requests() as u64;
        let inflight_prefill_reqs = self
            .inflight_prefill
            .as_ref()
            .map_or(0, |prefill| prefill.chunk.reqs.len());
        SchedulerMetrics {
            kv_used_blocks: if self.failed {
                // TP pages stay owned until backend teardown; zero is display-only.
                0
            } else {
                kv_total_blocks.saturating_sub(self.backend.available_pages() as u64)
            },
            kv_total_blocks,
            num_running_reqs: (self.active.len() + self.prefilling.len() + inflight_prefill_reqs)
                as u64,
            num_waiting_reqs: self.pending.len() as u64,
            spec_decode: None,
            prefix_cache: None,
        }
    }
}

#[cfg(test)]
mod tests;

fn active_request_kv(request: &mut ActiveRequest35) -> Option<&mut RequestKv> {
    match &mut request.backend_state {
        ActiveBackendState::Single { kv, .. } => Some(kv),
        ActiveBackendState::Tp { .. } => None,
    }
}
fn clamp_prefill_chunk(cursor: usize, chunk: usize, snapshot_stride: Option<usize>) -> usize {
    snapshot_stride.map_or(chunk, |stride| {
        debug_assert!(stride > 0);
        chunk.min(stride - cursor % stride)
    })
}
