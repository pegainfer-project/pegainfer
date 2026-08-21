//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! Phase 2A adds one canonical eager unified command while retaining the
//! replicated linear-attention state layout from Phase 1.

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
const TRITON_AOT_DEVICE_TABLE_LEN: usize = 16;

#[allow(dead_code)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunPrefillChunks {
        chunks: Vec<TpPrefillChunkItem>,
        sample_seed: u64,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunDecodeStep {
        requests: Vec<TpDecodeStepItem>,
        sample_seed: u64,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunUnifiedStep {
        plan: TpUnifiedPlan,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    DropRequest {
        request_id: RequestId,
        start: Arc<TpCommandStartGate>,
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

#[derive(Debug)]
enum TpWorkerReply {
    Ack,
    DropAck {
        existed: bool,
    },
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(TpUnifiedResult),
    #[cfg(test)]
    Snapshot(WorkerStateSnapshot),
}

#[derive(Debug)]
struct TpWorkerResponse {
    rank: usize,
    result: Result<TpWorkerReply>,
}

/// Scheduler-owned lifecycle proof required from every TP rank during cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropExpectation {
    MustBeAbsent,
    MustExist,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TpCommandDecision {
    #[default]
    Pending,
    Execute,
    Cancel,
}

#[derive(Default)]
struct TpCommandStartGate {
    decision: Mutex<TpCommandDecision>,
    changed: Condvar,
}

impl TpCommandStartGate {
    fn execute(&self) -> bool {
        self.resolve(TpCommandDecision::Execute)
    }

    fn cancel(&self) -> bool {
        self.resolve(TpCommandDecision::Cancel)
    }

    fn wait(&self) -> TpCommandDecision {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        while *decision == TpCommandDecision::Pending {
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *decision
    }

    fn resolve(&self, next: TpCommandDecision) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if *decision != TpCommandDecision::Pending {
            return false;
        }
        *decision = next;
        self.changed.notify_all();
        true
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
        self.poison.ensure_healthy()?;
        let resp_rx =
            self.dispatch_mutating("drop request", |start, resp| TpWorkerCommand::DropRequest {
                request_id,
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
        build: impl Fn(Arc<TpCommandStartGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
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
    build: impl Fn(Arc<TpCommandStartGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
    mut send: impl FnMut(usize, TpWorkerCommand) -> Result<()>,
) -> Result<mpsc::Receiver<TpWorkerResponse>> {
    let start = Arc::new(TpCommandStartGate::default());
    let (resp_tx, resp_rx) = mpsc::channel();
    for rank in 0..world_size {
        let command = build(Arc::clone(&start), resp_tx.clone());
        if let Err(err) = send(rank, command) {
            start.cancel();
            let reason = poison.poison(format!(
                "failed to dispatch {operation} to TP worker rank {rank}: {err:#}"
            ));
            return Err(anyhow::anyhow!(reason));
        }
    }
    drop(resp_tx);
    let resolved = start.execute();
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

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkerStateSnapshot {
    rank: usize,
    request_count: usize,
    requests: Vec<(RequestId, TpRequestPhase)>,
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
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_prefill_chunks(&chunks, sample_seed);
                        self.respond(resp, "prefill", result)
                    }
                }
                TpWorkerCommand::RunDecodeStep {
                    requests,
                    sample_seed,
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_decode(&requests, sample_seed);
                        self.respond(resp, "decode", result)
                    }
                }
                TpWorkerCommand::RunUnifiedStep { plan, start, resp } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_unified(&plan);
                        self.respond(resp, "unified step", result)
                    }
                }
                TpWorkerCommand::DropRequest {
                    request_id,
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let existed = self.drop_request(request_id);
                        self.respond(resp, "drop request", Ok(TpWorkerReply::DropAck { existed }))
                    }
                }
                #[cfg(test)]
                TpWorkerCommand::SnapshotState { resp } => {
                    let snapshot = WorkerStateSnapshot {
                        rank: self.rank,
                        request_count: self.requests.len(),
                        requests: self
                            .requests
                            .iter()
                            .map(|state| (state.request_id, state.phase))
                            .collect(),
                    };
                    self.respond(
                        resp,
                        "snapshot state",
                        Ok(TpWorkerReply::Snapshot(snapshot)),
                    )
                }
                #[cfg(test)]
                TpWorkerCommand::RemoveRequestStateForTest { request_id, resp } => {
                    let _ = resp.send(self.drop_request(request_id));
                    false
                }
                #[cfg(test)]
                TpWorkerCommand::DisconnectForTest { ready } => {
                    let _ = ready.send(());
                    break;
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
        let requests = self.execute_prefill_rows(chunks, sample_seed)?;
        if self.rank == 0 {
            Ok(TpWorkerReply::Prefill(PrefillResult { requests }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn execute_prefill_rows(
        &mut self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<Vec<PrefillRequestResult>> {
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

        Ok(primary_results)
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
        let requests = self.execute_decode_rows(requests, sample_seed)?;
        if self.rank == 0 {
            Ok(TpWorkerReply::Decode(DecodeResult { requests }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    fn execute_decode_rows(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<Vec<DecodeRequestResult>> {
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

        Ok(primary_results)
    }

    fn execute_unified(&mut self, plan: &TpUnifiedPlan) -> Result<TpWorkerReply> {
        validate_unified_worker_state(self, plan)?;

        // The command order is canonical across ranks. Sampling seeds are
        // selected by the scheduler in decode-then-prefill order, independent
        // of this forward order.
        let prefill_requests =
            self.execute_prefill_rows(&plan.prefill, plan.prefill_sample_seed)?;
        let decode_requests = self.execute_decode_rows(&plan.decode, plan.decode_sample_seed)?;

        if self.rank == 0 {
            Ok(TpWorkerReply::Unified(TpUnifiedResult {
                prefill: PrefillResult {
                    requests: prefill_requests,
                },
                decode: DecodeResult {
                    requests: decode_requests,
                },
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

    fn drop_request(&mut self, request_id: RequestId) -> bool {
        if let Some(idx) = self.request_index(request_id) {
            self.requests.swap_remove(idx);
            true
        } else {
            false
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

fn recv_runtime_responses(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<Vec<TpWorkerResponse>> {
    collect_runtime_responses(expected, operation, poison, || {
        recv_runtime_response(responses, operation, poison)
    })
}

fn collect_runtime_responses(
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
    mut recv_next: impl FnMut() -> Result<TpWorkerResponse>,
) -> Result<Vec<TpWorkerResponse>> {
    let mut collected = Vec::with_capacity(expected);
    for _ in 0..expected {
        let response = recv_next()?;
        if let Err(err) = &response.result {
            // A failed rank may leave peers blocked in a collective, so response-set
            // completeness is no longer recoverable or useful.
            let reason = poison.poison(format!(
                "rank {} failed during {operation}: {err:#}",
                response.rank
            ));
            return Err(anyhow::anyhow!(reason));
        }
        collected.push(response);
    }
    Ok(collected)
}

fn validate_dispatched_responses<T>(
    result: Result<T>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<T> {
    result.map_err(|err| {
        let reason = poison.poison(format!(
            "invalid Qwen3.5 TP {operation} response set: {err:#}"
        ));
        anyhow::anyhow!(reason)
    })
}

fn validate_exact_rank_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<Vec<(usize, TpWorkerReply)>> {
    anyhow::ensure!(
        responses.len() == world_size,
        "{operation} expected {world_size} responses, got {}",
        responses.len()
    );
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut replies = Vec::with_capacity(world_size);
    for response in responses {
        anyhow::ensure!(
            response.rank < world_size,
            "{operation} returned out-of-range rank {} for world size {world_size}",
            response.rank
        );
        anyhow::ensure!(
            seen_ranks.insert(response.rank),
            "{operation} returned duplicate rank {}",
            response.rank
        );
        replies.push((response.rank, response.result?));
    }
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "{operation} response set did not contain every rank"
    );
    replies.sort_unstable_by_key(|(rank, _)| *rank);
    Ok(replies)
}

#[cfg(test)]
fn validate_ack_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<()> {
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, operation)? {
        anyhow::ensure!(
            matches!(reply, TpWorkerReply::Ack),
            "{operation} rank {rank} returned {} instead of acknowledgement",
            reply_name(&reply)
        );
    }
    Ok(())
}

fn validate_drop_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    expectation: DropExpectation,
) -> Result<()> {
    let mut existence = Vec::with_capacity(world_size);
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "drop request")? {
        let TpWorkerReply::DropAck { existed } = reply else {
            anyhow::bail!(
                "drop request rank {rank} returned {} instead of drop acknowledgement",
                reply_name(&reply)
            );
        };
        existence.push((rank, existed));
    }
    let expected = expectation == DropExpectation::MustExist;
    anyhow::ensure!(
        existence.iter().all(|(_, existed)| *existed == expected),
        "drop request expected {expectation:?}, got rank existence {existence:?}"
    );
    Ok(())
}

fn validate_prefill_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<PrefillResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "prefill")? {
        match (rank, reply) {
            (0, TpWorkerReply::Prefill(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "prefill rank 0 returned {} instead of primary prefill result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "prefill non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("prefill returned no primary result"))
}

fn validate_decode_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<DecodeResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "decode")? {
        match (rank, reply) {
            (0, TpWorkerReply::Decode(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "decode rank 0 returned {} instead of primary decode result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "decode non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("decode returned no primary result"))
}

fn validate_unified_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<TpUnifiedResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "unified step")? {
        match (rank, reply) {
            (0, TpWorkerReply::Unified(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "unified step rank 0 returned {} instead of primary unified result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "unified step non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("unified step returned no primary result"))
}

fn reply_name(reply: &TpWorkerReply) -> &'static str {
    match reply {
        TpWorkerReply::Ack => "acknowledgement",
        TpWorkerReply::DropAck { .. } => "drop acknowledgement",
        TpWorkerReply::Prefill(_) => "prefill result",
        TpWorkerReply::Decode(_) => "decode result",
        TpWorkerReply::Unified(_) => "unified result",
        #[cfg(test)]
        TpWorkerReply::Snapshot(_) => "worker snapshot",
    }
}

#[cfg(test)]
fn wait_for_worker_snapshots(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    world_size: usize,
    poison: &TpRuntimePoison,
) -> Result<Vec<WorkerStateSnapshot>> {
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut snapshots = Vec::with_capacity(world_size);
    for _ in 0..world_size {
        let response = recv_runtime_response(responses, "snapshot state", poison)?;
        anyhow::ensure!(
            response.rank < world_size,
            "Qwen3.5 TP snapshot returned out-of-range rank {} for world size {world_size}",
            response.rank
        );
        anyhow::ensure!(
            seen_ranks.insert(response.rank),
            "Qwen3.5 TP snapshot returned duplicate rank {}",
            response.rank
        );
        match response.result? {
            TpWorkerReply::Snapshot(snapshot) => {
                anyhow::ensure!(
                    snapshot.rank == response.rank,
                    "Qwen3.5 TP snapshot payload rank {} does not match response rank {}",
                    snapshot.rank,
                    response.rank
                );
                anyhow::ensure!(
                    snapshot.request_count == snapshot.requests.len(),
                    "Qwen3.5 TP rank {} snapshot count {} does not match {} request entries",
                    snapshot.rank,
                    snapshot.request_count,
                    snapshot.requests.len()
                );
                snapshots.push(snapshot);
            }
            TpWorkerReply::Ack => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned acknowledgement")
            }
            TpWorkerReply::DropAck { .. } => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned drop acknowledgement")
            }
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned prefill result")
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned decode result")
            }
            TpWorkerReply::Unified(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned unified result")
            }
        }
    }
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "Qwen3.5 TP snapshot response set did not contain every rank"
    );
    snapshots.sort_unstable_by_key(|snapshot| snapshot.rank);
    Ok(snapshots)
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
    fn runtime_response_failure_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let responses = vec![
            reply(0, TpWorkerReply::Ack),
            TpWorkerResponse {
                rank: 1,
                result: Err(anyhow::anyhow!("rank 1 failed")),
            },
        ];

        let err = validate_dispatched_responses(
            validate_ack_responses(responses, 2, "test"),
            "test",
            &poison,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank 1 failed"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn runtime_response_collection_fails_fast_when_peer_never_responds() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        tx.send(TpWorkerResponse {
            rank: 0,
            result: Err(anyhow::anyhow!("rank 0 failed")),
        })
        .unwrap();
        let _keep_peer_channel_connected = tx;
        let mut receive_attempts = 0;

        let err = collect_runtime_responses(2, "test", &poison, || {
            receive_attempts += 1;
            rx.recv_timeout(std::time::Duration::from_millis(50))
                .map_err(|err| anyhow::anyhow!("waited for nonresponding rank: {err}"))
        })
        .unwrap_err()
        .to_string();

        assert_eq!(receive_attempts, 1, "collector waited for the missing rank");
        assert!(err.contains("rank 0 failed"));
        assert!(!err.contains("waited for nonresponding rank"));
        assert!(poison.ensure_healthy().is_err());
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

    fn reply(rank: usize, reply: TpWorkerReply) -> TpWorkerResponse {
        TpWorkerResponse {
            rank,
            result: Ok(reply),
        }
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
                    0,
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
        assert_eq!(start.wait(), TpCommandDecision::Cancel);
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
                requests: &[PrefillStepItem::new(clean_id, vec![151_646, 9707], 0)],
            })
            .expect("materialize clean request");
        executor
            .drop_request(clean_id, DropExpectation::MustExist)
            .expect("materialized drop should observe all ranks present");
        assert_workers_empty(&executor);

        let divergent_id = RequestId::new(402);
        executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(divergent_id, vec![151_646, 9707], 0)],
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
        let chunk = TpPrefillChunkItem::new(RequestId::new(410), vec![151_646, 9707], 0, true);

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
                    0,
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
                requests: &[PrefillStepItem::new(decode_id, vec![151_646, 9707], 1)],
            })
            .expect("materialize TP2 decode request");
        let prefill_id = RequestId::new(31);
        let unified = executor
            .execute_unified(&TpUnifiedPlan {
                prefill: vec![TpPrefillChunkItem::new(
                    prefill_id,
                    vec![151_646, 9707],
                    1,
                    true,
                )],
                decode: vec![TpDecodeStepItem::new(
                    decode_id,
                    decode_prefill.requests[0].first_token,
                    1,
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
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], 0))
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
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], 0))
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
            .map(|result| DecodeStepItem::new(result.request_id, result.first_token, 0))
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
        let clean_request = PrefillStepItem::new(clean_id, prompt.clone(), REQUESTED_LOGPROBS);
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
        let readmitted_request = PrefillStepItem::new(readmitted_id, prompt, REQUESTED_LOGPROBS);
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
