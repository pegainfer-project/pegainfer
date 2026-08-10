//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! Phase 1 supports eager dense TP prefill and decode. Unified execution still
//! fails closed until the scheduler path can drive ordered eager decode.

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

use anyhow::Result;
use pegainfer_core::kv_pool::KvState;
use pegainfer_frontend::sampler::SamplingParams;

#[cfg(test)]
use crate::batch_decode_graph::MAX_BATCH;
use crate::config::TensorParallelConfig;
use crate::decode_buffers::BatchDecodeBuffers35;
use crate::executor::DecodePlan;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
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

const TP_NCCL_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const TP_RUNTIME_STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
const TP_WORKER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const TP_RUNTIME_MEMORY_RESERVE_BYTES: usize = 512 * 1024 * 1024;

#[allow(dead_code)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunPrefillChunks {
        chunks: Vec<TpPrefillChunkItem>,
        sample_seed: u64,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunDecodeStep {
        requests: Vec<TpDecodeStepItem>,
        sample_seed: u64,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunUnifiedStep {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    DropRequest {
        request_id: RequestId,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    Shutdown,
}

#[derive(Debug)]
enum TpWorkerReply {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
}

#[derive(Debug)]
struct TpWorkerResponse {
    rank: usize,
    result: Result<TpWorkerReply>,
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
    #[cfg(test)]
    world_size: usize,
    max_batch: usize,
    page_size: usize,
    capacity_pages_for_requests: usize,
    max_position_embeddings: usize,
    eos_token_id: u32,
}

#[derive(Clone)]
pub(crate) struct TpPrefillChunkItem {
    request_id: RequestId,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    sampling_params: SamplingParams,
    finish_prefill: bool,
}

impl TpPrefillChunkItem {
    fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: usize,
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
        logprobs: usize,
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
    logprobs: usize,
    sampling_params: SamplingParams,
}

impl TpDecodeStepItem {
    pub(crate) fn new(
        request_id: RequestId,
        token_id: u32,
        logprobs: usize,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
            sampling_params,
        }
    }
}

impl Qwen35TpExecutor {
    #[cfg(test)]
    fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Self::from_runtime_with_capacity(model_path, enable_cuda_graph, device_ordinals, MAX_BATCH)
    }

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
        anyhow::ensure!(
            device_ordinals.len() > 1,
            "Qwen3.5 TP executor requires at least two CUDA devices, got {}",
            device_ordinals.len()
        );
        anyhow::ensure!(
            !enable_cuda_graph,
            "Qwen3.5 TP Phase 1 supports eager execution only; disable CUDA Graph"
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
                    enable_cuda_graph: false,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                },
            )?);
        }
        let first = models
            .first()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP executor loaded no models"))?;
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
        let startup_gate = Arc::new(TpStartupGate::default());
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
                    startup_gate.cancel();
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
                    startup_gate.cancel();
                    return Err(err);
                }
                Err(_) => {
                    startup_gate.cancel();
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
                startup_gate.cancel();
                return Err(err);
            }
        };
        startup_gate.connect();
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

        Ok(Self {
            workers,
            poison,
            #[cfg(test)]
            world_size,
            max_batch: min_rank_max_batch,
            page_size,
            capacity_pages_for_requests,
            max_position_embeddings,
            eos_token_id,
        })
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
        wait_for_acks(resp_rx, self.workers.len(), "ping", &self.poison)
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
        self.execute_prefill_chunks(&chunks)
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
        let chunks = chunks.to_vec();
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::RunPrefillChunks {
                    chunks: chunks.clone(),
                    sample_seed,
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        wait_for_prefill(resp_rx, self.workers.len(), &self.poison)
    }

    pub fn execute_decode(&self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        let requests: Vec<TpDecodeStepItem> = plan
            .requests
            .iter()
            .map(|request| {
                TpDecodeStepItem::new(
                    request.request_id,
                    request.token_id,
                    request.logprobs,
                    SamplingParams::default(),
                )
            })
            .collect();
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
        let requests = requests.to_vec();
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::RunDecodeStep {
                    requests: requests.clone(),
                    sample_seed,
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        wait_for_decode(resp_rx, self.workers.len(), &self.poison)
    }

    pub fn drop_request(&self, request_id: RequestId) -> Result<()> {
        self.poison.ensure_healthy()?;
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::DropRequest {
                    request_id,
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        wait_for_acks(resp_rx, self.workers.len(), "drop request", &self.poison)
    }

    fn send_or_poison(&self, worker: &TpWorker, command: TpWorkerCommand) -> Result<()> {
        worker.send(command).map_err(|err| {
            let reason = self
                .poison
                .poison(format!("failed to dispatch TP worker command: {err:#}"));
            anyhow::anyhow!(reason)
        })
    }
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

fn spawn_nccl_startup_watchdog() -> Result<(mpsc::SyncSender<()>, JoinHandle<()>)> {
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let watchdog = thread::Builder::new()
        .name("qwen35-tp-nccl-startup-watchdog".into())
        .spawn(move || {
            if done_rx.recv_timeout(TP_NCCL_STARTUP_TIMEOUT).is_ok() {
                return;
            }
            eprintln!(
                "Qwen3.5 TP NCCL startup did not complete within {}s; aborting",
                TP_NCCL_STARTUP_TIMEOUT.as_secs()
            );
            log::error!(
                "Qwen3.5 TP NCCL startup did not complete within {}s; aborting",
                TP_NCCL_STARTUP_TIMEOUT.as_secs()
            );
            std::process::abort();
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Qwen3.5 TP NCCL watchdog: {err}"))?;
    Ok((done_tx, watchdog))
}

#[allow(clippy::needless_pass_by_value)]
fn disarm_nccl_startup_watchdog(
    done_tx: mpsc::SyncSender<()>,
    watchdog: JoinHandle<()>,
) -> Result<()> {
    done_tx
        .send(())
        .map_err(|_| anyhow::anyhow!("Qwen3.5 TP NCCL watchdog exited unexpectedly"))?;
    watchdog
        .join()
        .map_err(|_| anyhow::anyhow!("Qwen3.5 TP NCCL watchdog panicked"))
}

struct TpWorker {
    tx: mpsc::Sender<TpWorkerCommand>,
    handle: Option<JoinHandle<()>>,
    done: mpsc::Receiver<()>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum TpStartupDecision {
    #[default]
    Pending,
    Connect,
    Cancel,
}

#[derive(Default)]
struct TpStartupGate {
    decision: Mutex<TpStartupDecision>,
    changed: Condvar,
}

impl TpStartupGate {
    fn connect(&self) {
        self.set(TpStartupDecision::Connect);
    }

    fn cancel(&self) {
        self.set(TpStartupDecision::Cancel);
    }

    fn wait(&self) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        while *decision == TpStartupDecision::Pending {
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *decision == TpStartupDecision::Connect
    }

    fn set(&self, next: TpStartupDecision) {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if *decision == TpStartupDecision::Pending {
            *decision = next;
            self.changed.notify_all();
        }
    }
}

impl TpWorker {
    #[allow(clippy::type_complexity)]
    fn spawn(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        max_batch: usize,
        max_prefill_tokens: usize,
        nccl_id: cudarc::nccl::safe::Id,
        startup_gate: Arc<TpStartupGate>,
        effective_max_batch: Arc<AtomicUsize>,
        poison: Arc<TpRuntimePoison>,
    ) -> Result<(
        Self,
        mpsc::Receiver<Result<usize>>,
        mpsc::Receiver<Result<()>>,
    )> {
        let (tx, rx) = mpsc::channel();
        let (preflight_tx, preflight_rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let panic_poison = Arc::clone(&poison);
        let handle = thread::Builder::new()
            .name(format!("qwen35-tp-rank-{rank}"))
            .spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let prepared = TpWorkerPrepared::new(
                        rank,
                        world_size,
                        model,
                        max_batch,
                        max_prefill_tokens,
                    );
                    let prepared = match prepared {
                        Ok((prepared, rank_max_batch)) => {
                            let _ = preflight_tx.send(Ok(rank_max_batch));
                            prepared
                        }
                        Err(err) => {
                            let _ = preflight_tx.send(Err(err));
                            return;
                        }
                    };
                    if !startup_gate.wait() {
                        return;
                    }
                    let max_batch = effective_max_batch.load(Ordering::Acquire);
                    match prepared.connect(nccl_id, max_batch, poison) {
                        Ok(mut state) => {
                            let _ = startup_tx.send(Ok(()));
                            state.run(rx);
                        }
                        Err(err) => {
                            let _ = startup_tx.send(Err(err));
                        }
                    }
                }));
                if outcome.is_err() {
                    panic_poison.poison(format!("worker rank {rank} panicked"));
                }
                let _ = done_tx.send(());
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3.5 TP worker {rank}: {e}"))?;

        Ok((
            Self {
                tx,
                handle: Some(handle),
                done: done_rx,
            },
            preflight_rx,
            startup_rx,
        ))
    }

    fn send(&self, command: TpWorkerCommand) -> Result<()> {
        self.tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker channel closed"))
    }

    fn join_bounded(&mut self) {
        if self.handle.is_none() {
            return;
        }
        if self.done.recv_timeout(TP_WORKER_SHUTDOWN_TIMEOUT).is_err() {
            fatal_tp_abort("Qwen3.5 TP worker did not exit during bounded shutdown");
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TpWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(TpWorkerCommand::Shutdown);
        self.join_bounded();
    }
}

struct TpWorkerState {
    rank: usize,
    _world_size: usize,
    max_batch: usize,
    model: Qwen35Model,
    requests: Vec<TpRequestState>,
    decode_buffers: BatchDecodeBuffers35,
    sample_scratch: pegainfer_sample::SampleScratch,
    _cublas_guard: CublasThreadGuard,
    poison: Arc<TpRuntimePoison>,
}

struct TpWorkerPrepared {
    rank: usize,
    world_size: usize,
    max_batch: usize,
    model: Qwen35Model,
    decode_buffers: BatchDecodeBuffers35,
    sample_scratch: pegainfer_sample::SampleScratch,
    cublas_guard: CublasThreadGuard,
}

struct TpRequestState {
    request_id: RequestId,
    phase: TpRequestPhase,
    kv: KvState,
    recurrent: RecurrentState,
    linear_pointer_tables: LinearStatePointerTables,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpRequestPhase {
    Prefilling,
    Decoding,
}

impl TpWorkerPrepared {
    fn new(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        requested_max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<(Self, usize)> {
        let cublas_guard = bind_worker_thread(&model)?;
        let (free_bytes, total_bytes) = model
            .device_ctx()
            .ctx
            .mem_get_info()
            .map_err(|err| anyhow::anyhow!("failed to query TP rank {rank} memory: {err}"))?;
        let recurrent_bytes = RecurrentState::allocation_bytes(model.config());
        let prefill_scratch_tokens = prefill_scratch_tokens(max_prefill_tokens);
        let prefill_scratch_bytes =
            GdrChunkwiseScratch35::estimate_bytes(model.config(), prefill_scratch_tokens);
        let max_batch = effective_recurrent_capacity(
            requested_max_batch,
            free_bytes,
            recurrent_bytes,
            TP_RUNTIME_MEMORY_RESERVE_BYTES,
            prefill_scratch_bytes,
        );
        anyhow::ensure!(
            max_batch > 0,
            "Qwen3.5 TP rank {rank} has {} MiB free after fixed buffers, but one recurrent request needs {} MiB plus {} MiB runtime reserve and {} MiB prefill scratch for {} tokens",
            free_bytes / (1024 * 1024),
            recurrent_bytes / (1024 * 1024),
            TP_RUNTIME_MEMORY_RESERVE_BYTES / (1024 * 1024),
            prefill_scratch_bytes / (1024 * 1024),
            prefill_scratch_tokens,
        );
        log::info!(
            "Qwen3.5 TP rank {rank} recurrent capacity: requested={requested_max_batch}, effective={max_batch}, per_request={:.3} MiB, free={:.0} MiB/{:.0} MiB, runtime_reserve={} MiB, prefill_tokens={}, prefill_scratch={:.0} MiB",
            recurrent_bytes as f64 / 1024.0 / 1024.0,
            free_bytes as f64 / 1024.0 / 1024.0,
            total_bytes as f64 / 1024.0 / 1024.0,
            TP_RUNTIME_MEMORY_RESERVE_BYTES / (1024 * 1024),
            prefill_scratch_tokens,
            prefill_scratch_bytes as f64 / 1024.0 / 1024.0,
        );
        let decode_buffers = model.create_batch_decode_buffers_with_capacity(max_batch)?;
        let sample_scratch = pegainfer_sample::SampleScratch::new(
            model.device_ctx(),
            model.config().selection_vocab,
            max_batch,
        )?;
        Ok((
            Self {
                rank,
                world_size,
                max_batch,
                model,
                decode_buffers,
                sample_scratch,
                cublas_guard,
            },
            max_batch,
        ))
    }

    fn connect(
        self,
        nccl_id: cudarc::nccl::safe::Id,
        effective_max_batch: usize,
        poison: Arc<TpRuntimePoison>,
    ) -> Result<TpWorkerState> {
        let Self {
            rank,
            world_size,
            max_batch,
            mut model,
            decode_buffers,
            sample_scratch,
            cublas_guard,
        } = self;
        anyhow::ensure!(
            effective_max_batch > 0 && effective_max_batch <= max_batch,
            "Qwen3.5 TP rank {rank} effective max_batch {effective_max_batch} exceeds local capacity {max_batch}"
        );
        let comm = cudarc::nccl::safe::Comm::from_rank(
            model.device_ctx().stream.clone(),
            rank,
            world_size,
            nccl_id,
        )
        .map_err(|e| anyhow::anyhow!("failed to initialize Qwen3.5 TP NCCL rank {rank}: {e:?}"))?;
        model.attach_tp_comm(comm);
        Ok(TpWorkerState {
            rank,
            _world_size: world_size,
            max_batch: effective_max_batch,
            model,
            requests: Vec::new(),
            decode_buffers,
            sample_scratch,
            _cublas_guard: cublas_guard,
            poison,
        })
    }
}

fn prefill_scratch_tokens(max_prefill_tokens: usize) -> usize {
    max_prefill_tokens.min(PREFILL_CHUNK_LEN)
}

fn effective_recurrent_capacity(
    requested_max_batch: usize,
    free_bytes: usize,
    recurrent_bytes_per_request: usize,
    runtime_reserve_bytes: usize,
    prefill_scratch_bytes: usize,
) -> usize {
    if recurrent_bytes_per_request == 0 {
        return requested_max_batch;
    }
    requested_max_batch.min(
        free_bytes
            .saturating_sub(runtime_reserve_bytes)
            .saturating_sub(prefill_scratch_bytes)
            / recurrent_bytes_per_request,
    )
}

impl TpWorkerState {
    #[allow(clippy::needless_pass_by_value)]
    fn run(&mut self, rx: mpsc::Receiver<TpWorkerCommand>) {
        while let Ok(command) = rx.recv() {
            let fatal = match command {
                TpWorkerCommand::Ping { resp } => {
                    self.respond(resp, "ping", Ok(TpWorkerReply::Ack))
                }
                TpWorkerCommand::RunPrefillChunks {
                    chunks,
                    sample_seed,
                    resp,
                } => {
                    let result = self.execute_prefill_chunks(&chunks, sample_seed);
                    self.respond(resp, "prefill", result)
                }
                TpWorkerCommand::RunDecodeStep {
                    requests,
                    sample_seed,
                    resp,
                } => {
                    let result = self.execute_decode(&requests, sample_seed);
                    self.respond(resp, "decode", result)
                }
                TpWorkerCommand::RunUnifiedStep { resp } => {
                    let rank = self.rank;
                    self.respond(
                        resp,
                        "unified step",
                        Err(anyhow::anyhow!(
                            "Qwen3.5 TP worker rank {rank} has no TP unified implementation yet"
                        )),
                    )
                }
                TpWorkerCommand::DropRequest { request_id, resp } => {
                    self.drop_request(request_id);
                    self.respond(resp, "drop request", Ok(TpWorkerReply::Ack))
                }
                TpWorkerCommand::Shutdown => break,
            };
            if fatal {
                break;
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn respond(
        &self,
        resp: mpsc::Sender<TpWorkerResponse>,
        operation: &'static str,
        result: Result<TpWorkerReply>,
    ) -> bool {
        match result {
            Ok(reply) => {
                let _ = resp.send(TpWorkerResponse {
                    rank: self.rank,
                    result: Ok(reply),
                });
                false
            }
            Err(err) => {
                let reason = self.poison.poison(format!(
                    "rank {} failed during {operation}: {err:#}",
                    self.rank
                ));
                let _ = resp.send(TpWorkerResponse {
                    rank: self.rank,
                    result: Err(anyhow::anyhow!(reason)),
                });
                true
            }
        }
    }

    fn execute_prefill_chunks(
        &mut self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        validate_prefill_chunks(chunks)?;
        let new_requests = chunks
            .iter()
            .filter(|chunk| self.request_index(chunk.request_id).is_none())
            .count();
        anyhow::ensure!(
            self.requests.len() + new_requests <= self.max_batch,
            "Qwen3.5 TP prefill chunks would exceed worker capacity {}",
            self.max_batch
        );

        let mut primary_results = Vec::new();
        let mut final_row_idx = 0usize;
        for chunk in chunks {
            let state_idx = self.ensure_prefill_state(chunk.request_id)?;
            let state = &mut self.requests[state_idx];
            anyhow::ensure!(
                state.phase == TpRequestPhase::Prefilling,
                "Qwen3.5 TP request {} is already in decode state",
                chunk.request_id.get()
            );

            let prompt = [chunk.prompt_tokens.as_slice()];
            let mut recurrent_refs = vec![&mut state.recurrent];
            let logits = self.model.batch_prefill_logits(
                &prompt,
                std::slice::from_mut(&mut state.kv),
                &mut recurrent_refs,
            )?;

            if chunk.finish_prefill {
                if self.rank == 0 {
                    // TP prefill samples final chunks one row at a time. Offset
                    // by the final-row index so rows from the same command do
                    // not reuse the same sampling stream.
                    let row_seed = sample_seed.wrapping_add(final_row_idx as u64);
                    let result = self.sample_final_prefill_chunk(chunk, &logits, row_seed)?;
                    primary_results.push(result);
                }
                final_row_idx += 1;
                self.requests[state_idx].phase = TpRequestPhase::Decoding;
            }
        }

        if self.rank == 0 {
            Ok(TpWorkerReply::Prefill(PrefillResult {
                requests: primary_results,
            }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn sample_final_prefill_chunk(
        &mut self,
        chunk: &TpPrefillChunkItem,
        logits: &pegainfer_core::tensor::HiddenStates,
        sample_seed: u64,
    ) -> Result<PrefillRequestResult> {
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &[chunk.logprobs])?;
        let params_refs = [&chunk.sampling_params];
        let tokens = pegainfer_sample::select_batch(
            self.model.device_ctx(),
            logits,
            &params_refs,
            &[0],
            sample_seed,
            &mut self.sample_scratch,
        )?;
        let first_token = tokens[0];
        let first_token_logprob = cpu_logits[0].as_ref().and_then(|row| {
            pegainfer_sample::token_logprob_from_row(row, first_token, chunk.logprobs)
        });
        Ok(PrefillRequestResult {
            request_id: chunk.request_id,
            first_token,
            first_token_logprob,
        })
    }

    fn execute_decode(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode command requires at least one request"
        );
        validate_decode_requests(requests)?;
        anyhow::ensure!(
            requests.len() <= self.max_batch,
            "Qwen3.5 TP decode batch {} exceeds worker capacity {}",
            requests.len(),
            self.max_batch
        );

        let mut primary_results =
            Vec::with_capacity(if self.rank == 0 { requests.len() } else { 0 });
        for (row_idx, request) in requests.iter().enumerate() {
            let state_idx = self.request_index(request.request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode request {} has no worker state",
                    request.request_id.get()
                )
            })?;
            anyhow::ensure!(
                self.requests[state_idx].phase == TpRequestPhase::Decoding,
                "Qwen3.5 TP request {} is not ready for decode",
                request.request_id.get()
            );

            {
                let state = &mut self.requests[state_idx];
                let mut kv_refs = [&mut state.kv];
                let mut recurrent_refs = [&mut state.recurrent];
                self.model.batch_decode_eager_logits(
                    &[request.token_id],
                    &mut kv_refs,
                    &mut recurrent_refs,
                    &state.linear_pointer_tables,
                    &mut self.decode_buffers,
                )?;
            }

            if self.rank == 0 {
                let cpu_logits = snapshot_requested_logprobs(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &[request.logprobs],
                )?;
                let params_refs = [&request.sampling_params];
                let tokens = pegainfer_sample::select_batch(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &params_refs,
                    &[0],
                    sample_seed.wrapping_add(row_idx as u64),
                    &mut self.sample_scratch,
                )?;
                let token = tokens[0];
                let logprob = cpu_logits[0].as_ref().and_then(|row| {
                    pegainfer_sample::token_logprob_from_row(row, token, request.logprobs)
                });
                primary_results.push(DecodeRequestResult {
                    request_id: request.request_id,
                    token,
                    logprob,
                });
            }
        }

        if self.rank == 0 {
            Ok(TpWorkerReply::Decode(DecodeResult {
                requests: primary_results,
            }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn ensure_prefill_state(&mut self, request_id: RequestId) -> Result<usize> {
        if let Some(idx) = self.request_index(request_id) {
            return Ok(idx);
        }
        let mut recurrent = RecurrentState::new(self.model.device_ctx(), self.model.config())?;
        let linear_pointer_tables = {
            let mut recurrent_refs = [&mut recurrent];
            LinearStatePointerTables::from_recurrent_refs(
                self.model.device_ctx(),
                self.model.config(),
                &mut recurrent_refs,
                1,
                "Qwen3.5 TP eager",
            )?
        };
        let state = TpRequestState {
            request_id,
            phase: TpRequestPhase::Prefilling,
            kv: self.model.alloc_kv(),
            recurrent,
            linear_pointer_tables,
        };
        self.requests.push(state);
        Ok(self.requests.len() - 1)
    }

    fn request_index(&self, request_id: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|state| state.request_id == request_id)
    }

    fn drop_request(&mut self, request_id: RequestId) {
        if let Some(idx) = self.request_index(request_id) {
            self.requests.swap_remove(idx);
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

impl From<DecodeStepItem> for TpDecodeStepItem {
    fn from(request: DecodeStepItem) -> Self {
        Self::new(
            request.request_id,
            request.token_id,
            request.logprobs,
            SamplingParams::default(),
        )
    }
}

#[allow(clippy::needless_pass_by_value)]
fn wait_for_acks(
    responses: mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    op_name: &'static str,
    poison: &TpRuntimePoison,
) -> Result<()> {
    for _ in 0..expected {
        let response = recv_runtime_response(&responses, op_name, poison)?;
        match response.result? {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP {op_name} unexpectedly returned prefill result")
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP {op_name} unexpectedly returned decode result")
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn wait_for_prefill(
    responses: mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    poison: &TpRuntimePoison,
) -> Result<PrefillResult> {
    let mut result = None;
    for _ in 0..expected {
        let response = recv_runtime_response(&responses, "prefill", poison)?;
        match response.result? {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Prefill(prefill) => {
                anyhow::ensure!(
                    response.rank == 0,
                    "Qwen3.5 TP prefill returned a primary result from rank {}",
                    response.rank
                );
                anyhow::ensure!(
                    result.is_none(),
                    "Qwen3.5 TP prefill returned multiple primary results"
                );
                result = Some(prefill);
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP prefill unexpectedly returned decode result")
            }
        }
    }
    result.ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP prefill returned no primary result"))
}

#[allow(clippy::needless_pass_by_value)]
fn wait_for_decode(
    responses: mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    poison: &TpRuntimePoison,
) -> Result<DecodeResult> {
    let mut result = None;
    for _ in 0..expected {
        let response = recv_runtime_response(&responses, "decode", poison)?;
        match response.result? {
            TpWorkerReply::Ack => {}
            TpWorkerReply::Decode(decode) => {
                anyhow::ensure!(
                    response.rank == 0,
                    "Qwen3.5 TP decode returned a primary result from rank {}",
                    response.rank
                );
                anyhow::ensure!(
                    result.is_none(),
                    "Qwen3.5 TP decode returned multiple primary results"
                );
                result = Some(decode);
            }
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP decode unexpectedly returned prefill result")
            }
        }
    }
    result.ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP decode returned no primary result"))
}

fn recv_runtime_response(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<TpWorkerResponse> {
    match responses.recv_timeout(TP_RUNTIME_STEP_TIMEOUT) {
        Ok(response) => Ok(response),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let reason = poison.poison(format!("response channel disconnected during {operation}"));
            Err(anyhow::anyhow!(reason))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => fatal_tp_abort(&format!(
            "Qwen3.5 TP {operation} did not complete within {}s",
            TP_RUNTIME_STEP_TIMEOUT.as_secs()
        )),
    }
}

fn fatal_tp_abort(message: &str) -> ! {
    eprintln!("{message}; aborting");
    log::error!("{message}; aborting");
    std::process::abort();
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_worker_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 TP worker thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 TP worker thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(CublasThreadGuard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_gate_cancel_releases_waiting_workers() {
        let gate = Arc::new(TpStartupGate::default());
        let worker_gate = Arc::clone(&gate);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let _ = done_tx.send(worker_gate.wait());
        });

        gate.cancel();

        assert!(
            !done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cancelled startup gate should release workers within one second")
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
    fn runtime_response_reports_any_rank_failure_immediately() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        tx.send(TpWorkerResponse {
            rank: 1,
            result: Err(anyhow::anyhow!("rank 1 failed")),
        })
        .unwrap();

        let err = wait_for_acks(rx, 2, "test", &poison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank 1 failed"));
    }

    #[test]
    fn disconnected_runtime_response_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        drop(tx);

        let err = recv_runtime_response(&rx, "test", &poison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("response channel disconnected during test"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn prefill_scratch_tokens_follow_budget_and_chunk_cap() {
        assert_eq!(prefill_scratch_tokens(1_024), 1_024);
        assert_eq!(prefill_scratch_tokens(PREFILL_CHUNK_LEN), 20_000);
        assert_eq!(prefill_scratch_tokens(40_000), 20_000);
    }

    #[test]
    fn recurrent_capacity_reserves_runtime_and_prefill_headroom() {
        const MIB: usize = 1024 * 1024;
        assert_eq!(
            effective_recurrent_capacity(64, 10_000 * MIB, 50 * MIB, 512 * MIB, 1_000 * MIB,),
            64
        );
        assert_eq!(
            effective_recurrent_capacity(64, 2_061 * MIB, 50 * MIB, 512 * MIB, 1_000 * MIB,),
            10
        );
        assert_eq!(
            effective_recurrent_capacity(64, 1_511 * MIB, 50 * MIB, 512 * MIB, 1_000 * MIB,),
            0
        );
    }

    #[test]
    fn zero_sized_recurrent_state_keeps_requested_capacity() {
        assert_eq!(
            effective_recurrent_capacity(64, 0, 0, usize::MAX, usize::MAX),
            64
        );
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
    fn rejects_tensor_parallel_cuda_graph() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", true, &[0, 1], 1) {
            Ok(_) => panic!("TP CUDA Graph should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("eager execution only"));
    }

    #[test]
    fn validates_prefill_chunk_shape() {
        let empty = [TpPrefillChunkItem::new(RequestId::new(1), vec![], 0, false)];
        let err = validate_prefill_chunks(&empty).unwrap_err().to_string();
        assert!(err.contains("is empty"));

        let duplicate = [
            TpPrefillChunkItem::new(RequestId::new(1), vec![151_646], 0, false),
            TpPrefillChunkItem::new(RequestId::new(1), vec![9707], 0, true),
        ];
        let err = validate_prefill_chunks(&duplicate).unwrap_err().to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn validates_decode_request_shape() {
        validate_decode_requests(&[TpDecodeStepItem::new(
            RequestId::new(1),
            9707,
            0,
            SamplingParams::default(),
        )])
        .expect("single decode request is valid");

        let duplicate = [
            TpDecodeStepItem::new(RequestId::new(1), 9707, 0, SamplingParams::default()),
            TpDecodeStepItem::new(RequestId::new(1), 560, 0, SamplingParams::default()),
        ];
        let err = validate_decode_requests(&duplicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn starts_tp2_workers_and_broadcasts_lifecycle_commands() {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        assert_eq!(executor.world_size(), 2);
        assert_eq!(executor.max_batch(), 1);
        executor.ping_all().expect("ping all workers");
        executor
            .drop_request(RequestId::new(7))
            .expect("drop request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_default_capacity_is_memory_safe() {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime(&model_path, false, &[0, 1])
            .expect("start TP2 executor with memory-derived capacity");
        eprintln!(
            "Qwen3.5 TP2 memory-derived max_batch={}",
            executor.max_batch()
        );
        assert!(executor.max_batch() > 0);
        assert!(executor.max_batch() <= MAX_BATCH);
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_prefill_runs_and_returns_primary_result() {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(11);
        let request = PrefillStepItem::new(request_id, vec![151_646, 9707], 0);
        let result = executor
            .execute_prefill(PrefillPlan {
                requests: &[request],
            })
            .expect("run TP2 prefill");
        assert_eq!(result.requests.len(), 1);
        assert_eq!(result.requests[0].request_id, request_id);
        executor
            .drop_request(request_id)
            .expect("drop prefetched request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_chunked_prefill_advances_existing_request_state() {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(13);
        let first = TpPrefillChunkItem::new(request_id, vec![151_646], 0, false);
        let first_result = executor
            .execute_prefill_chunks(&[first])
            .expect("run non-final TP2 prefill chunk");
        assert!(first_result.requests.is_empty());

        let final_chunk = TpPrefillChunkItem::new(request_id, vec![9707], 0, true);
        let final_result = executor
            .execute_prefill_chunks(&[final_chunk])
            .expect("run final TP2 prefill chunk");
        assert_eq!(final_result.requests.len(), 1);
        assert_eq!(final_result.requests[0].request_id, request_id);

        executor
            .drop_request(request_id)
            .expect("drop chunk-prefilled request");
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_decode_runs_after_prefill() {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let request_id = RequestId::new(17);
        let request = PrefillStepItem::new(request_id, vec![151_646, 9707], 0);
        let prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &[request],
            })
            .expect("run TP2 prefill");
        assert_eq!(prefill.requests.len(), 1);
        assert_eq!(prefill.requests[0].request_id, request_id);

        let decode_request = DecodeStepItem::new(request_id, prefill.requests[0].first_token, 0);
        let decode = executor
            .execute_decode(DecodePlan {
                requests: &[decode_request],
            })
            .expect("run TP2 eager decode");
        assert_eq!(decode.requests.len(), 1);
        assert_eq!(decode.requests[0].request_id, request_id);

        executor
            .drop_request(request_id)
            .expect("drop decoded request");
    }
}
