//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! One canonical eager unified command per step. Linear-attention/GDR weights
//! and state are sharded per rank; decode rows run as one batched forward per
//! rank plus one batched rank-0 sampling pass.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::thread::{self};
use std::time::Instant;

use anyhow::Result;
use pegainfer_core::kv_pool::KvState;
use pegainfer_frontend::sampler::SamplingParams;

use crate::batch_decode::DecodeGraphUse;
use crate::batch_decode_graph::BATCH_BUCKETS;
use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::batch_decode_graph::bucket_for;
use crate::config::TensorParallelConfig;
use crate::decode_buffers::BatchDecodeBuffers35;
use crate::executor::DecodePlan;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
#[cfg(test)]
use crate::executor::DecodeStepItem;
use crate::executor::PrefillPlan;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::PrefillStepItem;
use crate::executor::RequestId;
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefill::PREFILL_CHUNK_LEN;
use crate::prefill_buffers::GdrChunkwiseScratch35;
use crate::recurrent_state::LinearStatePointerTables;
use crate::recurrent_state::RecurrentState;
use crate::weights::ModelRuntimeConfig;
use crate::weights::Qwen35Model;

mod responses;
mod worker;

use responses::*;
use worker::*;

/// The pre-capture sweep records every decode bucket per rank; the 60 s NCCL
/// startup budget is far too small for that.
const TP_PRECAPTURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);
const TRITON_AOT_DEVICE_TABLE_LEN: usize = 16;

/// One controller-barriered phase of the TP decode-graph pre-capture sweep.
///
/// Capture and launch are separate phases because a captured collective's
/// first launch blocks on its peers: overlapping that with a peer still in
/// capture/instantiate/upload (which contend driver locks and allocate device
/// memory) deadlocks the driver. So every rank finishes capturing a bucket
/// before any rank launches it.
#[derive(Clone, Copy, Debug)]
enum PrecapturePhase {
    /// One eager all-reduce per bucket message size, so the size-selected NCCL
    /// algorithm connects before any `cuStreamBeginCapture` records it.
    Warmup,
    /// Record + instantiate + upload one bucket; no launch, no cross-rank dependency.
    Capture { bucket_idx: usize },
    /// Launch one bucket (pure enqueue after `Capture`) + sync; collectives pair across ranks.
    Launch { bucket_idx: usize },
    /// Verify every reachable bucket captured.
    Finalize,
}

/// Scheduler-owned slot move for TP graph decode: when the request at slot
/// `to` retires mid-batch, the request at slot `from` (the last occupied slot)
/// takes over slot `to` so decode rows stay dense. Workers apply the move and
/// fail (poisoning the executor) if slot occupancy does not match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TpSlotCompaction {
    pub(crate) moved_request_id: RequestId,
    pub(crate) from: usize,
    pub(crate) to: usize,
}

#[allow(dead_code)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunPrefillChunks {
        chunks: Vec<TpPrefillChunkItem>,
        sample_seed: u64,
        start: Arc<TpGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunDecodeStep {
        requests: Vec<TpDecodeStepItem>,
        sample_seed: u64,
        start: Arc<TpGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunUnifiedStep {
        plan: TpUnifiedPlan,
        start: Arc<TpGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    DropRequest {
        request_id: RequestId,
        /// Slot move the scheduler already applied to its own bookkeeping;
        /// `Some` only when the dropped request held a decode slot that a
        /// still-active request now takes over. Eager workers ignore it.
        compaction: Option<TpSlotCompaction>,
        start: Arc<TpGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    /// Startup-only (graph-enabled TP): one phase of the decode-graph
    /// pre-capture sweep, barriered across ranks by the controller.
    Precapture {
        phase: PrecapturePhase,
        start: Arc<TpGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    #[cfg(test)]
    SnapshotState {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    #[cfg(test)]
    RemoveRequestStateForTest {
        request_id: RequestId,
        resp: mpsc::Sender<bool>,
    },
    #[cfg(test)]
    DisconnectForTest {
        ready: mpsc::SyncSender<()>,
    },
    Shutdown,
}

/// Scheduler-owned lifecycle proof required from every TP rank during cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropExpectation {
    MustBeAbsent,
    MustExist,
}

/// One-shot go/cancel decision broadcast to every rank's thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TpGateDecision {
    #[default]
    Pending,
    Go,
    Cancel,
}

/// A gate the dispatcher resolves exactly once; waiters block until the
/// decision leaves `Pending`. Used for per-command starts and for startup.
#[derive(Default)]
struct TpGate {
    decision: Mutex<TpGateDecision>,
    changed: Condvar,
}

impl TpGate {
    /// Resolve the decision; returns false if someone resolved it first.
    fn resolve(&self, next: TpGateDecision) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if *decision != TpGateDecision::Pending {
            return false;
        }
        *decision = next;
        self.changed.notify_all();
        true
    }

    /// Block until the gate is resolved, then return the decision.
    fn wait(&self) -> TpGateDecision {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        while *decision == TpGateDecision::Pending {
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *decision
    }
}

#[derive(Default)]
struct TpRuntimePoison {
    reason: Mutex<Option<String>>,
}

impl TpRuntimePoison {
    fn poison(&self, reason: String) -> String {
        let mut current = self.reason.lock().unwrap_or_else(PoisonError::into_inner);
        current.get_or_insert(reason).clone()
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(reason) = self
            .reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
        {
            anyhow::bail!("Qwen3.5 TP executor is poisoned: {reason}");
        }
        Ok(())
    }
}

/// TP executor. Rank 0 is the primary worker and returns scheduler-visible
/// artifacts; every rank runs the same ordered state-mutating commands.
pub struct Qwen35TpExecutor {
    workers: Vec<TpWorker>,
    poison: Arc<TpRuntimePoison>,
    world_size: usize,
    max_batch: usize,
    page_size: usize,
    capacity_pages_for_requests: usize,
    max_position_embeddings: usize,
    eos_token_id: u32,
    /// Whether decode steps replay pre-captured CUDA Graphs that record NCCL
    /// collectives (`enable_cuda_graph` AND a compiled TP-local decode GQA
    /// group; see the P2c gate in `tp-design.md`).
    graph_enabled: bool,
    /// Slot tracker for the convenience `execute_prefill`/`execute_decode`/
    /// `drop_request` API (model-local tests), mirroring the single-GPU
    /// `Qwen35Executor`: prefill completion appends, decode plans must cover
    /// every tracked request in slot order, drop swap-removes and derives the
    /// slot compaction. Scheduler-driven flows bypass it entirely — they pass
    /// explicit slots via `execute_decode_items` and
    /// `drop_request_with_compaction`. Never mix the two flows on one executor.
    active_slots: Mutex<Vec<RequestId>>,
}

#[derive(Clone)]
pub(crate) struct TpPrefillChunkItem {
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    logprobs: Option<usize>,
    sampling_params: SamplingParams,
    finish_prefill: bool,
}

impl TpPrefillChunkItem {
    fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: Option<usize>,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params: SamplingParams::default(),
            finish_prefill,
        }
    }

    pub(crate) fn new_with_sampling(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: Option<usize>,
        sampling_params: SamplingParams,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params,
            finish_prefill,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TpDecodeStepItem {
    request_id: RequestId,
    token_id: u32,
    logprobs: Option<usize>,
    sampling_params: SamplingParams,
    /// Scheduler-assigned decode slot under CUDA Graph TP. Rows must arrive in
    /// dense slot order (`slot_idx == row`); on the request's first decode row
    /// the worker D2D-copies its prefill recurrent state into the slot and
    /// drops the per-request allocation. `None` on the slot-free eager path.
    slot_idx: Option<usize>,
}

impl TpDecodeStepItem {
    pub(crate) fn new(
        request_id: RequestId,
        token_id: u32,
        logprobs: Option<usize>,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
            sampling_params,
            slot_idx: None,
        }
    }

    pub(crate) fn new_with_slot(
        request_id: RequestId,
        token_id: u32,
        logprobs: Option<usize>,
        sampling_params: SamplingParams,
        slot_idx: usize,
    ) -> Self {
        Self {
            slot_idx: Some(slot_idx),
            ..Self::new(request_id, token_id, logprobs, sampling_params)
        }
    }
}

#[derive(Clone)]
pub(crate) struct TpUnifiedPlan {
    pub(crate) prefill: Vec<TpPrefillChunkItem>,
    pub(crate) decode: Vec<TpDecodeStepItem>,
    pub(crate) prefill_sample_seed: u64,
    pub(crate) decode_sample_seed: u64,
}

#[derive(Debug)]
pub(crate) struct TpUnifiedResult {
    pub(crate) prefill: PrefillResult,
    pub(crate) decode: DecodeResult,
}

impl Qwen35TpExecutor {
    pub fn from_runtime_with_capacity(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        max_batch: usize,
    ) -> Result<Self> {
        Self::from_runtime_with_limits(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            max_batch,
            PREFILL_CHUNK_LEN,
        )
    }

    pub(crate) fn from_runtime_with_limits(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        validate_cuda_ordinals(device_ordinals)?;
        anyhow::ensure!(
            device_ordinals.len() > 1,
            "Qwen3.5 TP executor requires at least two CUDA devices, got {}",
            device_ordinals.len()
        );
        anyhow::ensure!(
            max_prefill_tokens > 0,
            "Qwen3.5 TP max_prefill_tokens must be positive"
        );

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen35Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig::try_from((rank, world_size))?),
                    device_ordinal,
                },
            )?);
        }
        let first = models
            .first()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP executor loaded no models"))?;
        // P2c gate: graph decode under TP requires a compiled batch-decode
        // kernel for the rank-local GQA group. The group ratio is TP-invariant,
        // so every rank decides identically; an uncompiled group (e.g. 27B's
        // group 6) keeps the batched eager path byte-for-byte.
        let geometry = first.geometry;
        let graph_enabled = enable_cuda_graph && first.config().decode_group_is_compiled();
        if enable_cuda_graph && !graph_enabled {
            static LOG_GRAPH_GATE: std::sync::Once = std::sync::Once::new();
            LOG_GRAPH_GATE.call_once(|| {
                log::info!(
                    "Qwen3.5 TP decode GQA group {} ({} q heads / {} kv heads per rank) has no compiled batch-decode kernel; CUDA Graph requested but decode stays on the batched eager path",
                    geometry.local_num_attention_heads() / geometry.local_num_key_value_heads(),
                    geometry.local_num_attention_heads(),
                    geometry.local_num_key_value_heads(),
                );
            });
        }
        let page_size = first.kv_pool().layout().page_size;
        let mut min_capacity_pages = usize::MAX;
        for (rank, model) in models.iter().enumerate() {
            let rank_page_size = model.kv_pool().layout().page_size;
            anyhow::ensure!(
                rank_page_size == page_size,
                "Qwen3.5 TP rank {rank} KV page size {rank_page_size} does not match rank 0 page size {page_size}"
            );
            min_capacity_pages = min_capacity_pages.min(model.kv_pool().capacity_pages());
        }
        let capacity_pages_for_requests = min_capacity_pages.saturating_sub(1);
        let max_position_embeddings = first.config().max_position_embeddings;
        let eos_token_id = first.config().eos_token_id;

        let nccl_id = cudarc::nccl::safe::Id::new()
            .map_err(|e| anyhow::anyhow!("failed to create Qwen3.5 TP NCCL id: {e:?}"))?;
        let startup_gate = Arc::new(TpGate::default());
        let effective_max_batch = Arc::new(AtomicUsize::new(0));
        let poison = Arc::new(TpRuntimePoison::default());
        let mut workers = Vec::with_capacity(world_size);
        let mut preflights = Vec::with_capacity(world_size);
        let mut startups = Vec::with_capacity(world_size);
        for (rank, model) in models.into_iter().enumerate() {
            match TpWorker::spawn(
                rank,
                world_size,
                model,
                max_batch,
                max_prefill_tokens,
                graph_enabled,
                nccl_id,
                Arc::clone(&startup_gate),
                Arc::clone(&effective_max_batch),
                Arc::clone(&poison),
            ) {
                Ok((worker, preflight, startup)) => {
                    workers.push(worker);
                    preflights.push(preflight);
                    startups.push(startup);
                }
                Err(err) => {
                    startup_gate.resolve(TpGateDecision::Cancel);
                    return Err(err);
                }
            }
        }
        let mut min_rank_max_batch = max_batch;
        for (rank, preflight) in preflights.into_iter().enumerate() {
            match preflight.recv() {
                Ok(Ok(rank_max_batch)) => {
                    min_rank_max_batch = min_rank_max_batch.min(rank_max_batch);
                }
                Ok(Err(err)) => {
                    startup_gate.resolve(TpGateDecision::Cancel);
                    return Err(err);
                }
                Err(_) => {
                    startup_gate.resolve(TpGateDecision::Cancel);
                    return Err(anyhow::anyhow!(
                        "Qwen3.5 TP worker {rank} exited during pre-NCCL startup"
                    ));
                }
            }
        }
        anyhow::ensure!(
            min_rank_max_batch > 0,
            "Qwen3.5 TP has no memory capacity for one recurrent request state"
        );
        effective_max_batch.store(min_rank_max_batch, Ordering::Release);
        if min_rank_max_batch < max_batch {
            log::warn!(
                "Qwen3.5 TP max_batch reduced from {max_batch} to {min_rank_max_batch} by rank-local recurrent-state memory capacity"
            );
        }
        let (watchdog_done, watchdog) = match spawn_nccl_startup_watchdog() {
            Ok(watchdog) => watchdog,
            Err(err) => {
                startup_gate.resolve(TpGateDecision::Cancel);
                return Err(err);
            }
        };
        startup_gate.resolve(TpGateDecision::Go);
        let startup_result = startups
            .into_iter()
            .enumerate()
            .try_for_each(|(rank, startup)| {
                startup.recv().map_err(|_| {
                    anyhow::anyhow!("Qwen3.5 TP worker {rank} exited during startup")
                })?
            });
        if let Err(err) = startup_result {
            drop(workers);
            disarm_nccl_startup_watchdog(watchdog_done, watchdog)?;
            return Err(err);
        }
        disarm_nccl_startup_watchdog(watchdog_done, watchdog)?;

        let executor = Self {
            workers,
            poison,
            world_size,
            max_batch: min_rank_max_batch,
            page_size,
            capacity_pages_for_requests,
            max_position_embeddings,
            eos_token_id,
            graph_enabled,
            active_slots: Mutex::new(Vec::new()),
        };
        // Pre-capture every reachable decode graph now: after NCCL connect,
        // exactly once, before serving. A mid-serving capture on one rank while
        // a peer replays would desync the recorded collectives, so serve time
        // is replay-only.
        if graph_enabled {
            executor.run_decode_graph_precapture_sweep()?;
            log::info!(
                "Qwen3.5 TP decode CUDA Graph enabled: {} bucket(s) up to batch {} captured per rank",
                BATCH_BUCKETS
                    .iter()
                    .take_while(|&&b| b <= bucket_for(executor.max_batch))
                    .count(),
                bucket_for(executor.max_batch),
            );
        }
        Ok(executor)
    }

    /// Whether decode replays pre-captured CUDA Graphs (P2c gate: requested
    /// AND the TP-local decode GQA group has a compiled kernel).
    pub fn graph_enabled(&self) -> bool {
        self.graph_enabled
    }

    #[cfg(test)]
    fn world_size(&self) -> usize {
        self.world_size
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    pub(crate) fn capacity_pages_for_requests(&self) -> usize {
        self.capacity_pages_for_requests
    }

    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    /// Pre-capture every reachable decode bucket on every rank, phase-by-phase
    /// and barriered by the controller so a captured collective's first launch
    /// never overlaps a peer's capture (qwen3 sweep precedent).
    fn run_decode_graph_precapture_sweep(&self) -> Result<()> {
        // NCCL has no device timeout, so a desynced sweep wedges forever; this
        // watchdog aborts on the deadline. abort() not exit() — exit's cudart
        // atexit teardown takes the same wedged driver lock — and it disarms
        // only on the explicit success send (drop-on-error stays armed).
        let (sweep_done_tx, sweep_done_rx) = mpsc::sync_channel::<()>(1);
        let deadline = Instant::now() + TP_PRECAPTURE_TIMEOUT;
        let watchdog = thread::Builder::new()
            .name("qwen35-tp-precapture-watchdog".into())
            .spawn(move || {
                if sweep_done_rx.recv_timeout(TP_PRECAPTURE_TIMEOUT).is_ok() {
                    return;
                }
                std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                eprintln!(
                    "Qwen3.5 TP decode graph pre-capture did not complete within {}s — NCCL wedge suspected, aborting",
                    TP_PRECAPTURE_TIMEOUT.as_secs()
                );
                log::error!(
                    "Qwen3.5 TP decode graph pre-capture did not complete within {}s — NCCL wedge suspected, aborting",
                    TP_PRECAPTURE_TIMEOUT.as_secs()
                );
                std::process::abort();
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3.5 TP pre-capture watchdog: {e}"))?;

        let started = Instant::now();
        let max_bucket = bucket_for(self.max_batch);
        let sweep = (|| {
            self.run_precapture_phase(PrecapturePhase::Warmup)?;
            for (bucket_idx, &bucket) in BATCH_BUCKETS.iter().enumerate() {
                if bucket > max_bucket {
                    break;
                }
                self.run_precapture_phase(PrecapturePhase::Capture { bucket_idx })?;
                self.run_precapture_phase(PrecapturePhase::Launch { bucket_idx })?;
            }
            self.run_precapture_phase(PrecapturePhase::Finalize)
        })();
        sweep?;
        sweep_done_tx
            .send(())
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP pre-capture watchdog exited"))?;
        watchdog
            .join()
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP pre-capture watchdog panicked"))?;
        log::info!(
            "Qwen3.5 TP decode graph pre-capture: buckets up to {max_bucket} captured per rank in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn run_precapture_phase(&self, phase: PrecapturePhase) -> Result<()> {
        self.poison.ensure_healthy()?;
        let resp_rx = self.dispatch_mutating("decode graph precapture", |start, resp| {
            TpWorkerCommand::Precapture { phase, start, resp }
        })?;
        let responses = recv_runtime_responses(
            &resp_rx,
            self.world_size,
            "decode graph precapture",
            &self.poison,
        )?;
        validate_dispatched_responses(
            validate_ack_responses(responses, self.world_size, "decode graph precapture"),
            "decode graph precapture",
            &self.poison,
        )
    }

    #[cfg(test)]
    fn ping_all(&self) -> Result<()> {
        self.poison.ensure_healthy()?;
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::Ping {
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        let responses = recv_runtime_responses(&resp_rx, self.world_size, "ping", &self.poison)?;
        validate_dispatched_responses(
            validate_ack_responses(responses, self.world_size, "ping"),
            "ping",
            &self.poison,
        )
    }

    pub fn execute_prefill(&self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP prefill plan requires at least one request"
        );
        let chunks: Vec<TpPrefillChunkItem> = plan
            .requests
            .iter()
            .cloned()
            .map(TpPrefillChunkItem::from)
            .collect();
        let result = self.execute_prefill_chunks(&chunks)?;
        if self.graph_enabled {
            // Convenience-API slot tracking: every prefill plan item finishes
            // prefill (TpPrefillChunkItem::from sets finish_prefill), so each
            // request takes the next dense decode slot.
            let mut active = self
                .active_slots
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for chunk in &chunks {
                active.push(chunk.request_id);
            }
        }
        Ok(result)
    }

    fn execute_prefill_chunks(&self, chunks: &[TpPrefillChunkItem]) -> Result<PrefillResult> {
        self.execute_prefill_chunks_with_seed(chunks, 0)
    }

    pub(crate) fn execute_prefill_chunks_with_seed(
        &self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<PrefillResult> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        validate_prefill_chunks(chunks)?;
        let chunks = chunks.to_vec();
        let resp_rx = self.dispatch_mutating("prefill chunks", |start, resp| {
            TpWorkerCommand::RunPrefillChunks {
                chunks: chunks.clone(),
                sample_seed,
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "prefill chunks", &self.poison)?;
        validate_dispatched_responses(
            validate_prefill_responses(responses, self.world_size),
            "prefill chunks",
            &self.poison,
        )
    }

    pub fn execute_decode(&self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        let requests: Vec<TpDecodeStepItem> = if self.graph_enabled {
            let active = self
                .active_slots
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            anyhow::ensure!(
                plan.requests.len() == active.len(),
                "Qwen3.5 TP graph decode must cover all {} active requests in slot order, got {}",
                active.len(),
                plan.requests.len()
            );
            plan.requests
                .iter()
                .enumerate()
                .map(|(slot, request)| {
                    anyhow::ensure!(
                        active[slot] == request.request_id,
                        "Qwen3.5 TP graph decode slot {slot} holds request {} but the plan carries {}",
                        active[slot].get(),
                        request.request_id.get()
                    );
                    Ok(TpDecodeStepItem::new_with_slot(
                        request.request_id,
                        request.token_id,
                        request.logprobs,
                        SamplingParams::default(),
                        slot,
                    ))
                })
                .collect::<Result<_>>()?
        } else {
            plan.requests
                .iter()
                .map(|request| {
                    TpDecodeStepItem::new(
                        request.request_id,
                        request.token_id,
                        request.logprobs,
                        SamplingParams::default(),
                    )
                })
                .collect()
        };
        self.execute_decode_items(&requests, 0)
    }

    pub(crate) fn execute_decode_items(
        &self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<DecodeResult> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        validate_decode_requests(requests)?;
        let requests = requests.to_vec();
        let resp_rx = self.dispatch_mutating("decode step", |start, resp| {
            TpWorkerCommand::RunDecodeStep {
                requests: requests.clone(),
                sample_seed,
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "decode step", &self.poison)?;
        validate_dispatched_responses(
            validate_decode_responses(responses, self.world_size),
            "decode step",
            &self.poison,
        )
    }

    pub(crate) fn execute_unified(&self, plan: &TpUnifiedPlan) -> Result<TpUnifiedResult> {
        self.poison.ensure_healthy()?;
        validate_unified_plan(plan, self.max_batch)?;
        let resp_rx = self.dispatch_mutating("unified step", |start, resp| {
            TpWorkerCommand::RunUnifiedStep {
                plan: plan.clone(),
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "unified step", &self.poison)?;
        validate_dispatched_responses(
            validate_unified_responses(responses, self.world_size),
            "unified step",
            &self.poison,
        )
    }

    pub(crate) fn poison_artifact_contract(
        &self,
        operation: &'static str,
        err: &anyhow::Error,
    ) -> anyhow::Error {
        let reason = self.poison.poison(format!(
            "invalid Qwen3.5 TP {operation} artifact set: {err:#}"
        ));
        anyhow::anyhow!(reason)
    }

    pub fn drop_request(&self, request_id: RequestId, expectation: DropExpectation) -> Result<()> {
        let compaction = self.track_retired_slot(request_id);
        self.drop_request_with_compaction(request_id, expectation, compaction)
    }

    /// Retire a request, attaching the slot compaction the caller (scheduler)
    /// already applied to its own dense-slot bookkeeping. Workers apply the
    /// move and poison on occupancy mismatch; eager workers ignore it.
    pub(crate) fn drop_request_with_compaction(
        &self,
        request_id: RequestId,
        expectation: DropExpectation,
        compaction: Option<TpSlotCompaction>,
    ) -> Result<()> {
        self.poison.ensure_healthy()?;
        let resp_rx =
            self.dispatch_mutating("drop request", |start, resp| TpWorkerCommand::DropRequest {
                request_id,
                compaction,
                start,
                resp,
            })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "drop request", &self.poison)?;
        validate_dispatched_responses(
            validate_drop_responses(responses, self.world_size, expectation),
            "drop request",
            &self.poison,
        )
    }

    /// Convenience-API tracker: swap-remove the retired request and derive the
    /// slot compaction (last occupied slot moves into the vacated one).
    /// Returns `None` on the eager path and for untracked requests.
    fn track_retired_slot(&self, request_id: RequestId) -> Option<TpSlotCompaction> {
        if !self.graph_enabled {
            return None;
        }
        let mut active = self
            .active_slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let idx = active.iter().position(|&id| id == request_id)?;
        let last = active.len() - 1;
        active.swap_remove(idx);
        // `then`, not `then_some`: the moved request only exists when the
        // retired request was not the tail slot.
        (idx < active.len()).then(|| TpSlotCompaction {
            moved_request_id: active[idx],
            from: last,
            to: idx,
        })
    }

    #[cfg(test)]
    fn snapshot_workers(&self) -> Result<Vec<WorkerStateSnapshot>> {
        self.poison.ensure_healthy()?;
        self.snapshot_workers_unchecked_for_test()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn snapshot_workers_unchecked_for_test(&self) -> Result<Vec<WorkerStateSnapshot>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::SnapshotState {
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        wait_for_worker_snapshots(&resp_rx, self.world_size, &self.poison)
    }

    #[cfg(test)]
    fn inject_prefill_dispatch_failure_for_test(
        &self,
        chunks: &[TpPrefillChunkItem],
        fail_rank: usize,
    ) -> Result<()> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            fail_rank < self.world_size,
            "injected TP dispatch failure rank {fail_rank} is outside world size {}",
            self.world_size
        );
        validate_prefill_chunks(chunks)?;
        let chunks = chunks.to_vec();
        dispatch_mutating_commands(
            self.world_size,
            "injected prefill chunks",
            &self.poison,
            |start, resp| TpWorkerCommand::RunPrefillChunks {
                chunks: chunks.clone(),
                sample_seed: 0,
                start,
                resp,
            },
            |rank, command| {
                if rank == fail_rank {
                    anyhow::bail!("injected dispatch failure at rank {rank}");
                }
                self.workers[rank].send(command)
            },
        )?;
        Ok(())
    }

    #[cfg(test)]
    fn remove_worker_request_state_for_test(
        &self,
        rank: usize,
        request_id: RequestId,
    ) -> Result<bool> {
        self.poison.ensure_healthy()?;
        let worker = self
            .workers
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("test worker rank {rank} is out of range"))?;
        let (resp_tx, resp_rx) = mpsc::channel();
        worker.send(TpWorkerCommand::RemoveRequestStateForTest {
            request_id,
            resp: resp_tx,
        })?;
        resp_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|err| anyhow::anyhow!("test worker rank {rank} did not remove state: {err}"))
    }

    #[cfg(test)]
    fn disconnect_worker_receiver_for_test(&self, rank: usize) -> Result<()> {
        self.poison.ensure_healthy()?;
        let worker = self
            .workers
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("test worker rank {rank} is out of range"))?;
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        worker.send(TpWorkerCommand::DisconnectForTest { ready: ready_tx })?;
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|err| anyhow::anyhow!("test worker rank {rank} did not disconnect: {err}"))?;

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let (resp_tx, _resp_rx) = mpsc::channel();
            if worker
                .send(TpWorkerCommand::Ping { resp: resp_tx })
                .is_err()
            {
                return Ok(());
            }
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "test worker rank {rank} receiver remained connected"
            );
            std::thread::yield_now();
        }
    }

    fn dispatch_mutating(
        &self,
        operation: &'static str,
        build: impl Fn(Arc<TpGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
    ) -> Result<mpsc::Receiver<TpWorkerResponse>> {
        dispatch_mutating_commands(
            self.world_size,
            operation,
            &self.poison,
            build,
            |rank, command| self.workers[rank].send(command),
        )
    }

    #[cfg(test)]
    fn send_or_poison(&self, worker: &TpWorker, command: TpWorkerCommand) -> Result<()> {
        worker.send(command).map_err(|err| {
            let reason = self
                .poison
                .poison(format!("failed to dispatch TP worker command: {err:#}"));
            anyhow::anyhow!(reason)
        })
    }
}

fn dispatch_mutating_commands(
    world_size: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
    build: impl Fn(Arc<TpGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
    mut send: impl FnMut(usize, TpWorkerCommand) -> Result<()>,
) -> Result<mpsc::Receiver<TpWorkerResponse>> {
    let start = Arc::new(TpGate::default());
    let (resp_tx, resp_rx) = mpsc::channel();
    for rank in 0..world_size {
        let command = build(Arc::clone(&start), resp_tx.clone());
        if let Err(err) = send(rank, command) {
            start.resolve(TpGateDecision::Cancel);
            let reason = poison.poison(format!(
                "failed to dispatch {operation} to TP worker rank {rank}: {err:#}"
            ));
            return Err(anyhow::anyhow!(reason));
        }
    }
    drop(resp_tx);
    let resolved = start.resolve(TpGateDecision::Go);
    debug_assert!(resolved, "fresh TP command gate resolved more than once");
    Ok(resp_rx)
}

impl Drop for Qwen35TpExecutor {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.tx.send(TpWorkerCommand::Shutdown);
        }
        for worker in &mut self.workers {
            worker.join_bounded();
        }
    }
}

fn validate_prefill_chunks(chunks: &[TpPrefillChunkItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(chunks.len());
    for chunk in chunks {
        anyhow::ensure!(
            !chunk.prompt_tokens.is_empty(),
            "Qwen3.5 TP prefill chunk for request {} is empty",
            chunk.request_id.get()
        );
        anyhow::ensure!(
            seen.insert(chunk.request_id),
            "duplicate Qwen3.5 TP request id {} in one prefill chunk command",
            chunk.request_id.get()
        );
    }
    Ok(())
}

/// Worker request states in decode-row order: `row_of_state[i]` is the row
/// `states[i]` occupies in this command, `None` when it is not part of it.
fn validate_decode_requests(requests: &[TpDecodeStepItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(requests.len());
    for request in requests {
        anyhow::ensure!(
            seen.insert(request.request_id),
            "duplicate Qwen3.5 TP request id {} in one decode command",
            request.request_id.get()
        );
    }
    Ok(())
}

fn validate_cuda_ordinals(device_ordinals: &[usize]) -> Result<()> {
    let mut seen = HashSet::with_capacity(device_ordinals.len());
    for &ordinal in device_ordinals {
        anyhow::ensure!(
            ordinal < TRITON_AOT_DEVICE_TABLE_LEN,
            "Qwen3.5 TP CUDA ordinal {ordinal} exceeds the Triton AOT device table bound {TRITON_AOT_DEVICE_TABLE_LEN}"
        );
        anyhow::ensure!(
            seen.insert(ordinal),
            "Qwen3.5 TP CUDA ordinals must be distinct; ordinal {ordinal} appears more than once"
        );
    }
    Ok(())
}

fn validate_unified_plan(plan: &TpUnifiedPlan, max_batch: usize) -> Result<()> {
    anyhow::ensure!(
        !plan.prefill.is_empty(),
        "Qwen3.5 TP unified plan requires at least one prefill chunk"
    );
    anyhow::ensure!(
        !plan.decode.is_empty(),
        "Qwen3.5 TP unified plan requires at least one decode request"
    );
    validate_prefill_chunks(&plan.prefill)?;
    validate_decode_requests(&plan.decode)?;
    anyhow::ensure!(
        plan.prefill.len().saturating_add(plan.decode.len()) <= max_batch,
        "Qwen3.5 TP unified plan has {} rows, exceeding scheduler capacity {max_batch}",
        plan.prefill.len().saturating_add(plan.decode.len())
    );

    let prefill_ids: HashSet<_> = plan.prefill.iter().map(|item| item.request_id).collect();
    for decode in &plan.decode {
        anyhow::ensure!(
            !prefill_ids.contains(&decode.request_id),
            "Qwen3.5 TP unified plan request id {} appears in both prefill and decode",
            decode.request_id.get()
        );
    }
    Ok(())
}

fn validate_unified_worker_state(state: &TpWorkerState, plan: &TpUnifiedPlan) -> Result<()> {
    validate_unified_worker_layout(plan, state.max_batch, state.requests.len(), |request_id| {
        state
            .request_index(request_id)
            .map(|idx| state.requests[idx].phase)
    })
}

fn validate_unified_worker_layout(
    plan: &TpUnifiedPlan,
    max_batch: usize,
    resident_count: usize,
    mut phase_for: impl FnMut(RequestId) -> Option<TpRequestPhase>,
) -> Result<()> {
    validate_unified_plan(plan, max_batch)?;

    let new_prefill_count = plan
        .prefill
        .iter()
        .filter(|item| phase_for(item.request_id).is_none())
        .count();
    anyhow::ensure!(
        resident_count.saturating_add(new_prefill_count) <= max_batch,
        "Qwen3.5 TP unified plan would exceed worker capacity {}",
        max_batch
    );

    for item in &plan.prefill {
        if let Some(phase) = phase_for(item.request_id) {
            anyhow::ensure!(
                phase == TpRequestPhase::Prefilling,
                "Qwen3.5 TP unified prefill request {} is already in decode state",
                item.request_id.get()
            );
        }
    }
    for item in &plan.decode {
        let phase = phase_for(item.request_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Qwen3.5 TP unified decode request {} has no worker state",
                item.request_id.get()
            )
        })?;
        anyhow::ensure!(
            phase == TpRequestPhase::Decoding,
            "Qwen3.5 TP unified request {} is not ready for decode",
            item.request_id.get()
        );
    }
    Ok(())
}

impl From<PrefillStepItem> for TpPrefillChunkItem {
    fn from(request: PrefillStepItem) -> Self {
        Self::new(
            request.request_id,
            request.prompt_tokens,
            request.logprobs,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_gate_cancel_releases_waiting_workers() {
        let gate = Arc::new(TpGate::default());
        let worker_gate = Arc::clone(&gate);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let _ = done_tx.send(worker_gate.wait());
        });

        gate.resolve(TpGateDecision::Cancel);

        assert_eq!(
            done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cancelled startup gate should release workers within one second"),
            TpGateDecision::Cancel,
        );
        waiter.join().unwrap();
    }

    #[test]
    fn nccl_startup_watchdog_disarms_after_success() {
        let (done_tx, watchdog) = spawn_nccl_startup_watchdog().unwrap();
        disarm_nccl_startup_watchdog(done_tx, watchdog).unwrap();
    }

    #[test]
    fn runtime_poison_preserves_first_failure() {
        let poison = TpRuntimePoison::default();
        assert_eq!(poison.poison("rank 1 OOM".into()), "rank 1 OOM");
        assert_eq!(poison.poison("rank 0 NCCL error".into()), "rank 1 OOM");
        let err = poison.ensure_healthy().unwrap_err().to_string();
        assert!(err.contains("rank 1 OOM"));
        assert!(!err.contains("rank 0 NCCL error"));
    }

    #[test]
    fn mutating_partial_dispatch_cancels_delivered_prefix_and_poisons() {
        let poison = TpRuntimePoison::default();
        let (rank0_tx, rank0_rx) = mpsc::channel();
        let (rank1_tx, rank1_rx) = mpsc::channel::<TpWorkerCommand>();
        let senders = [rank0_tx, rank1_tx];
        let err = dispatch_mutating_commands(
            2,
            "test prefill",
            &poison,
            |start, resp| TpWorkerCommand::RunPrefillChunks {
                chunks: vec![TpPrefillChunkItem::new(
                    RequestId::new(2),
                    vec![9707],
                    None,
                    true,
                )],
                sample_seed: 0,
                start,
                resp,
            },
            |rank, command| {
                if rank == 1 {
                    anyhow::bail!("injected prefix-only dispatch failure");
                }
                senders[rank]
                    .send(command)
                    .map_err(|_| anyhow::anyhow!("test receiver disconnected"))
            },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("injected prefix-only dispatch failure"));
        let TpWorkerCommand::RunPrefillChunks { start, .. } = rank0_rx.recv().unwrap() else {
            panic!("expected prefill command")
        };
        assert_eq!(start.wait(), TpGateDecision::Cancel);
        assert!(matches!(
            rank1_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn limits_constructor_rejects_zero_prefill_budget_before_loading() {
        let err = match Qwen35TpExecutor::from_runtime_with_limits("unused", false, &[0, 1], 1, 0) {
            Ok(_) => panic!("zero TP prefill budget should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("max_prefill_tokens must be positive"));
    }

    #[test]
    fn rejects_single_device_topology() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", false, &[0], 1) {
            Ok(_) => panic!("single-device TP topology should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires at least two CUDA devices"));
    }

    #[test]
    fn slot_map_admit_release_and_compact() {
        let id = |value: u64| RequestId::new(value);
        let mut owners = vec![None, None, None, None];

        slot_admit(&mut owners, 0, id(1)).unwrap();
        slot_admit(&mut owners, 1, id(2)).unwrap();
        slot_admit(&mut owners, 2, id(3)).unwrap();

        let err = slot_admit(&mut owners, 1, id(9)).unwrap_err().to_string();
        assert!(err.contains("still owned by request 2"));

        // Retire slot 1: last occupied slot (2, request 3) moves into it.
        let needs_move = slot_compact(
            &mut owners,
            id(2),
            TpSlotCompaction {
                moved_request_id: id(3),
                from: 2,
                to: 1,
            },
        )
        .unwrap();
        assert!(needs_move, "materialized moved request needs the GPU move");
        assert_eq!(owners, vec![Some(id(1)), Some(id(3)), None, None]);

        // Retire the tail slot: release without compaction.
        assert_eq!(slot_release(&mut owners, id(3)), Some(1));
        assert_eq!(owners, vec![Some(id(1)), None, None, None]);

        // Releasing a request that never materialized a slot is not an error.
        assert_eq!(slot_release(&mut owners, id(77)), None);
    }

    #[test]
    fn slot_map_compact_tolerates_unmaterialized_requests() {
        let id = |value: u64| RequestId::new(value);
        let mut owners = vec![None, None, None];

        // Dropped request materialized, moved request not yet admitted to its
        // slot (retired between promotion and its first decode row): clear
        // only, no GPU move.
        slot_admit(&mut owners, 0, id(1)).unwrap();
        let needs_move = slot_compact(
            &mut owners,
            id(1),
            TpSlotCompaction {
                moved_request_id: id(2),
                from: 2,
                to: 0,
            },
        )
        .unwrap();
        assert!(!needs_move);
        assert_eq!(owners, vec![None, None, None]);

        // Moved request materialized, dropped request not: the move is needed
        // and the moved request takes over the vacated slot.
        slot_admit(&mut owners, 2, id(3)).unwrap();
        let needs_move = slot_compact(
            &mut owners,
            id(4),
            TpSlotCompaction {
                moved_request_id: id(3),
                from: 2,
                to: 0,
            },
        )
        .unwrap();
        assert!(needs_move);
        assert_eq!(owners, vec![Some(id(3)), None, None]);
    }

    #[test]
    fn slot_map_compact_poisons_on_occupancy_mismatch() {
        let id = |value: u64| RequestId::new(value);
        let mut owners = vec![Some(id(1)), Some(id(2))];

        let err = slot_compact(
            &mut owners,
            id(9),
            TpSlotCompaction {
                moved_request_id: id(2),
                from: 1,
                to: 0,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("slot 0 holds request 1"));

        let err = slot_compact(
            &mut owners,
            id(1),
            TpSlotCompaction {
                moved_request_id: id(9),
                from: 1,
                to: 0,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("slot 1 holds request 2"));

        let err = slot_admit(&mut owners, 5, id(1)).unwrap_err().to_string();
        assert!(err.contains("exceeds worker slot map"));
    }

    #[test]
    fn validates_prefill_chunk_shape() {
        let empty = [TpPrefillChunkItem::new(
            RequestId::new(1),
            vec![],
            None,
            false,
        )];
        let err = validate_prefill_chunks(&empty).unwrap_err().to_string();
        assert!(err.contains("is empty"));

        let duplicate = [
            TpPrefillChunkItem::new(RequestId::new(1), vec![151_646], None, false),
            TpPrefillChunkItem::new(RequestId::new(1), vec![9707], None, true),
        ];
        let err = validate_prefill_chunks(&duplicate).unwrap_err().to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn validates_decode_request_shape() {
        validate_decode_requests(&[TpDecodeStepItem::new(
            RequestId::new(1),
            9707,
            None,
            SamplingParams::default(),
        )])
        .expect("single decode request is valid");

        let duplicate = [
            TpDecodeStepItem::new(RequestId::new(1), 9707, None, SamplingParams::default()),
            TpDecodeStepItem::new(RequestId::new(1), 560, None, SamplingParams::default()),
        ];
        let err = validate_decode_requests(&duplicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"));
    }

    fn assert_workers_empty(executor: &Qwen35TpExecutor) {
        let snapshots = executor
            .snapshot_workers()
            .expect("snapshot healthy TP workers");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    fn assert_snapshots_empty(snapshots: &[WorkerStateSnapshot], world_size: usize) {
        assert_eq!(snapshots.len(), world_size);
        for (rank, snapshot) in snapshots.iter().enumerate() {
            assert_eq!(snapshot.rank, rank);
            assert_eq!(snapshot.request_count, 0, "rank {rank} retained requests");
            assert!(
                snapshot.requests.is_empty(),
                "rank {rank} retained request IDs"
            );
        }
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_drop_expectations_detect_rank_lifecycle_divergence() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_drop_expectations_detect_rank_lifecycle_divergence",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");

        executor
            .drop_request(RequestId::new(400), DropExpectation::MustBeAbsent)
            .expect("pre-materialization drop should observe all ranks absent");
        executor.ping_all().expect("absent drop preserves health");

        let clean_id = RequestId::new(401);
        executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(clean_id, vec![151_646, 9707], None)],
            })
            .expect("materialize clean request");
        executor
            .drop_request(clean_id, DropExpectation::MustExist)
            .expect("materialized drop should observe all ranks present");
        assert_workers_empty(&executor);

        let divergent_id = RequestId::new(402);
        executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(
                    divergent_id,
                    vec![151_646, 9707],
                    None,
                )],
            })
            .expect("materialize divergent request");
        assert!(
            executor
                .remove_worker_request_state_for_test(1, divergent_id)
                .expect("remove rank-1 request state")
        );
        let err = executor
            .drop_request(divergent_id, DropExpectation::MustExist)
            .unwrap_err()
            .to_string();
        assert!(err.contains("MustExist"));
        assert!(executor.ping_all().is_err());
        let snapshots = executor
            .snapshot_workers_unchecked_for_test()
            .expect("snapshot workers after mixed drop poison");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_partial_dispatch_gate_prevents_rank_local_mutation() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_partial_dispatch_gate_prevents_rank_local_mutation",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let chunk = TpPrefillChunkItem::new(RequestId::new(410), vec![151_646, 9707], None, true);

        let err = executor
            .inject_prefill_dispatch_failure_for_test(&[chunk], 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("injected dispatch failure at rank 1"));
        assert!(executor.ping_all().is_err());
        let snapshots = executor
            .snapshot_workers_unchecked_for_test()
            .expect("snapshot workers after cancelled prefix dispatch");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_worker_receiver_disconnect_poisons_without_snapshot_claim() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_worker_receiver_disconnect_poisons_without_snapshot_claim",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        executor
            .disconnect_worker_receiver_for_test(1)
            .expect("disconnect rank-1 worker receiver");

        let err = executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(
                    RequestId::new(420),
                    vec![151_646, 9707],
                    None,
                )],
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to dispatch prefill chunks to TP worker rank 1"));
        assert!(executor.ping_all().is_err());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_unified_step_advances_prefill_and_decode_together() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_unified_step_advances_prefill_and_decode_together",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 2)
            .expect("start TP2 executor");
        let decode_id = RequestId::new(30);
        let decode_prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(
                    decode_id,
                    vec![151_646, 9707],
                    Some(1),
                )],
            })
            .expect("materialize TP2 decode request");
        let prefill_id = RequestId::new(31);
        let unified = executor
            .execute_unified(&TpUnifiedPlan {
                prefill: vec![TpPrefillChunkItem::new(
                    prefill_id,
                    vec![151_646, 9707],
                    Some(1),
                    true,
                )],
                decode: vec![TpDecodeStepItem::new(
                    decode_id,
                    decode_prefill.requests[0].first_token,
                    Some(1),
                    SamplingParams::default(),
                )],
                prefill_sample_seed: 102,
                decode_sample_seed: 101,
            })
            .expect("run TP2 unified step");

        assert_eq!(unified.prefill.requests.len(), 1);
        assert_eq!(unified.prefill.requests[0].request_id, prefill_id);
        assert!(unified.prefill.requests[0].first_token_logprob.is_some());
        assert_eq!(unified.decode.requests.len(), 1);
        assert_eq!(unified.decode.requests[0].request_id, decode_id);
        assert!(unified.decode.requests[0].logprob.is_some());
        for snapshot in executor.snapshot_workers().expect("snapshot unified state") {
            assert_eq!(snapshot.request_count, 2);
            assert!(
                snapshot
                    .requests
                    .iter()
                    .all(|(_, phase)| *phase == TpRequestPhase::Decoding)
            );
        }

        for request_id in [decode_id, prefill_id] {
            executor
                .drop_request(request_id, DropExpectation::MustExist)
                .expect("drop unified request");
        }
        assert_workers_empty(&executor);
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_drop_all_restores_complete_request_capacity() {
        const CONFIGURED_MAX_BATCH: usize = 2;

        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_drop_all_restores_complete_request_capacity",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(
            &model_path,
            false,
            &[0, 1],
            CONFIGURED_MAX_BATCH,
        )
        .expect("start TP2 executor");
        assert_eq!(executor.max_batch(), CONFIGURED_MAX_BATCH);
        assert_workers_empty(&executor);

        let first_ids: Vec<_> = (100..100 + CONFIGURED_MAX_BATCH as u64)
            .map(RequestId::new)
            .collect();
        let first_requests: Vec<_> = first_ids
            .iter()
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], None))
            .collect();
        let first_results = executor
            .execute_prefill(PrefillPlan {
                requests: &first_requests,
            })
            .expect("fill complete TP2 request capacity");
        assert_eq!(first_results.requests.len(), CONFIGURED_MAX_BATCH);
        let expected_ids: HashSet<_> = first_ids.iter().copied().collect();
        for snapshot in executor
            .snapshot_workers()
            .expect("snapshot full TP2 request capacity")
        {
            assert_eq!(snapshot.request_count, CONFIGURED_MAX_BATCH);
            assert_eq!(
                snapshot
                    .requests
                    .iter()
                    .map(|(request_id, _)| *request_id)
                    .collect::<HashSet<_>>(),
                expected_ids
            );
            assert!(
                snapshot
                    .requests
                    .iter()
                    .all(|(_, phase)| *phase == TpRequestPhase::Decoding),
                "rank {} retained a non-decoding request after final prefill",
                snapshot.rank
            );
        }
        for request_id in &first_ids {
            executor
                .drop_request(*request_id, DropExpectation::MustExist)
                .expect("drop first-pass TP2 request");
        }
        assert_workers_empty(&executor);

        let second_ids: Vec<_> = (200..200 + CONFIGURED_MAX_BATCH as u64)
            .map(RequestId::new)
            .collect();
        let second_requests: Vec<_> = second_ids
            .iter()
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], None))
            .collect();
        let second_prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &second_requests,
            })
            .expect("refill complete TP2 request capacity");
        assert_eq!(second_prefill.requests.len(), CONFIGURED_MAX_BATCH);
        let decode_requests: Vec<_> = second_prefill
            .requests
            .iter()
            .map(|result| DecodeStepItem::new(result.request_id, result.first_token, None))
            .collect();
        let decode = executor
            .execute_decode(DecodePlan {
                requests: &decode_requests,
            })
            .expect("complete one decode step after TP2 capacity refill");
        assert_eq!(decode.requests.len(), CONFIGURED_MAX_BATCH);
        for request_id in &second_ids {
            executor
                .drop_request(*request_id, DropExpectation::MustExist)
                .expect("drop second-pass TP2 request");
        }
        assert_workers_empty(&executor);
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_readmission_matches_clean_first_token_artifact() {
        const REQUESTED_LOGPROBS: usize = 5;

        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_readmission_matches_clean_first_token_artifact",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let prompt = vec![151_646, 9707];

        let clean_id = RequestId::new(300);
        let clean_request =
            PrefillStepItem::new(clean_id, prompt.clone(), Some(REQUESTED_LOGPROBS));
        let clean = executor
            .execute_prefill(PrefillPlan {
                requests: &[clean_request],
            })
            .expect("run clean TP2 prefill");
        assert_eq!(clean.requests.len(), 1);
        assert!(clean.requests[0].first_token_logprob.is_some());
        let clean_artifact = (
            clean.requests[0].first_token,
            clean.requests[0].first_token_logprob.clone(),
        );
        executor
            .drop_request(clean_id, DropExpectation::MustExist)
            .expect("drop clean TP2 request");
        assert_workers_empty(&executor);

        let readmitted_id = RequestId::new(301);
        let readmitted_request =
            PrefillStepItem::new(readmitted_id, prompt, Some(REQUESTED_LOGPROBS));
        let readmitted = executor
            .execute_prefill(PrefillPlan {
                requests: &[readmitted_request],
            })
            .expect("run readmitted TP2 prefill");
        assert_eq!(readmitted.requests.len(), 1);
        let readmitted_artifact = (
            readmitted.requests[0].first_token,
            readmitted.requests[0].first_token_logprob.clone(),
        );
        assert_eq!(readmitted_artifact, clean_artifact);
        executor
            .drop_request(readmitted_id, DropExpectation::MustExist)
            .expect("drop readmitted TP2 request");
        assert_workers_empty(&executor);
    }
}
