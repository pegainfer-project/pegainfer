//! Tensor-parallel worker runtime: the per-rank worker thread, startup
//! gating, the NCCL startup watchdog, and the per-rank command loop.

use super::*;
use crate::cublas_thread::CublasThreadGuard;

const TP_NCCL_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const TP_WORKER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const TP_RUNTIME_MEMORY_RESERVE_BYTES: usize = 512 * 1024 * 1024;

pub(super) fn spawn_nccl_startup_watchdog() -> Result<(mpsc::SyncSender<()>, JoinHandle<()>)> {
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
pub(super) fn disarm_nccl_startup_watchdog(
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

pub(super) struct TpWorker {
    pub(super) tx: mpsc::Sender<TpWorkerCommand>,
    handle: Option<JoinHandle<()>>,
    done: mpsc::Receiver<()>,
}

impl TpWorker {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(super) fn spawn(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        max_batch: usize,
        max_prefill_tokens: usize,
        graph_enabled: bool,
        nccl_id: cudarc::nccl::safe::Id,
        startup_gate: Arc<TpGate>,
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
                        graph_enabled,
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
                    if startup_gate.wait() != TpGateDecision::Go {
                        return;
                    }
                    let max_batch = effective_max_batch.load(Ordering::Acquire);
                    match prepared.connect(nccl_id, max_batch, graph_enabled, poison) {
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

    pub(super) fn send(&self, command: TpWorkerCommand) -> Result<()> {
        self.tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker channel closed"))
    }

    pub(super) fn join_bounded(&mut self) {
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

pub(super) struct TpWorkerState {
    rank: usize,
    _world_size: usize,
    pub(super) max_batch: usize,
    /// Before `model` on purpose: NCCL comm teardown polls until every graph
    /// that recorded its collectives is destroyed, so the decode graphs must
    /// drop before `model.tp_comm` (qwen3 teardown-hang precedent).
    graph_state: Option<BatchDecodeGraphState>,
    model: Qwen35Model,
    pub(super) requests: Vec<TpRequestState>,
    /// Graph-mode slot ownership: `slot_map[i]` is the request whose recurrent
    /// state lives in `graph_state.slot_states[i]`. The scheduler owns slot
    /// assignment and compaction; the worker only applies and checks them.
    /// Empty in eager mode.
    slot_map: Vec<Option<RequestId>>,
    decode_buffers: BatchDecodeBuffers35,
    /// Eager decode GDR pointer tables: allocated once at capacity, refilled
    /// with the live rows every step.
    decode_pointer_tables: LinearStatePointerTables,
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

pub(super) struct TpRequestState {
    pub(super) request_id: RequestId,
    pub(super) phase: TpRequestPhase,
    kv: KvState,
    /// Prefill-owned recurrent state. Graph mode moves it into the decode slot
    /// on the request's first decode row (`None` afterwards); the eager path
    /// keeps it for the request's whole lifetime.
    recurrent: Option<RecurrentState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TpRequestPhase {
    Prefilling,
    Decoding,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WorkerStateSnapshot {
    pub(super) rank: usize,
    pub(super) request_count: usize,
    pub(super) requests: Vec<(RequestId, TpRequestPhase)>,
}

impl TpWorkerPrepared {
    fn new(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        requested_max_batch: usize,
        max_prefill_tokens: usize,
        graph_enabled: bool,
    ) -> Result<(Self, usize)> {
        let cublas_guard = crate::cublas_thread::bind_model_thread(&model, "TP worker")?;
        let (free_bytes, total_bytes) = model
            .device_ctx()
            .ctx
            .mem_get_info()
            .map_err(|err| anyhow::anyhow!("failed to query TP rank {rank} memory: {err}"))?;
        // Recurrent state is rank-local, so worker capacity math uses the
        // local value-head/qkv sizes.
        let recurrent_bytes = RecurrentState::allocation_bytes(model.config(), model.geometry);
        let prefill_scratch_tokens = prefill_scratch_tokens(max_prefill_tokens);
        let prefill_scratch_bytes = GdrChunkwiseScratch35::estimate_bytes(
            model.config(),
            model.geometry,
            prefill_scratch_tokens,
        );
        // Graph mode pre-allocates one fixed-address slot state per decode
        // bucket position up front; reserve that before sizing per-request
        // (prefill-transient) state capacity. The reserve must track the
        // bucket of the *effective* batch, not the requested one: reserving
        // for `bucket_for(requested)` can starve a tight-memory rank down to
        // zero capacity. Iterate the bucket downward until it stabilises —
        // the bucket only shrinks, so this converges — and clamp the fitted
        // batch to the reserved bucket so the later `bucket_for(effective)`
        // graph allocation never exceeds the reserve.
        let max_batch = if graph_enabled {
            let mut slot_bucket = bucket_for(requested_max_batch);
            loop {
                let reserve = slot_bucket * recurrent_bytes;
                let candidate = effective_recurrent_capacity(
                    requested_max_batch,
                    free_bytes.saturating_sub(reserve),
                    recurrent_bytes,
                    TP_RUNTIME_MEMORY_RESERVE_BYTES,
                    prefill_scratch_bytes,
                );
                let fitted = candidate.min(slot_bucket);
                let next = bucket_for(fitted);
                if next >= slot_bucket {
                    break fitted;
                }
                slot_bucket = next;
            }
        } else {
            effective_recurrent_capacity(
                requested_max_batch,
                free_bytes,
                recurrent_bytes,
                TP_RUNTIME_MEMORY_RESERVE_BYTES,
                prefill_scratch_bytes,
            )
        };
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
        graph_enabled: bool,
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
        let decode_pointer_tables = LinearStatePointerTables::with_capacity(
            model.device_ctx(),
            model.config(),
            effective_max_batch,
            "Qwen3.5 TP eager decode",
        )?;
        let (graph_state, slot_map) = if graph_enabled {
            // cuBLASLt plans are thread-local: tune the decode bucket GEMMs on
            // this worker thread now so plan selection never runs inside
            // cuStreamBeginCapture during the pre-capture sweep.
            model.tune_decode_gemm_algos()?;
            let slots = bucket_for(effective_max_batch);
            let graph_state = model.create_batch_decode_graph_state_with_capacity(slots)?;
            (Some(graph_state), vec![None; slots])
        } else {
            (None, Vec::new())
        };
        Ok(TpWorkerState {
            rank,
            _world_size: world_size,
            max_batch: effective_max_batch,
            graph_state,
            model,
            requests: Vec::new(),
            slot_map,
            decode_buffers,
            decode_pointer_tables,
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
                    if start.wait() == TpGateDecision::Cancel {
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
                    if start.wait() == TpGateDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_decode(&requests, sample_seed);
                        self.respond(resp, "decode", result)
                    }
                }
                TpWorkerCommand::RunUnifiedStep { plan, start, resp } => {
                    if start.wait() == TpGateDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_unified(&plan);
                        self.respond(resp, "unified step", result)
                    }
                }
                TpWorkerCommand::DropRequest {
                    request_id,
                    compaction,
                    start,
                    resp,
                } => {
                    if start.wait() == TpGateDecision::Cancel {
                        false
                    } else {
                        let result = self
                            .drop_request(request_id, compaction)
                            .map(|existed| TpWorkerReply::DropAck { existed });
                        self.respond(resp, "drop request", result)
                    }
                }
                TpWorkerCommand::Precapture { phase, start, resp } => {
                    if start.wait() == TpGateDecision::Cancel {
                        false
                    } else {
                        let result = self.precapture_phase(phase).map(|()| TpWorkerReply::Ack);
                        self.respond(resp, "decode graph precapture", result)
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
                    let _ = resp.send(self.drop_request(request_id, None).unwrap_or(false));
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
            let mut recurrent_refs = vec![
                state
                    .recurrent
                    .as_mut()
                    .expect("prefill-phase TP request owns its recurrent state"),
            ];
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

    /// Run one batched eager decode step over all rows in command order: a
    /// single forward for the whole batch on every rank, then (rank 0 only)
    /// one batched sampling pass over the per-row sampling params. Returns one
    /// result row per request in command order on rank 0, empty elsewhere.
    fn run_decode_batch(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<Vec<DecodeRequestResult>> {
        let bs = requests.len();
        if bs == 0 {
            return Ok(Vec::new());
        }
        if self.graph_state.is_some() {
            return self.run_decode_batch_graph(requests, sample_seed);
        }

        // Resolve the worker state slot of every row in command order.
        // Decode request ids are unique within one command
        // (validate_decode_requests), so each slot is borrowed at most once.
        let mut row_of_state: Vec<Option<usize>> = vec![None; self.requests.len()];
        for (row, request) in requests.iter().enumerate() {
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
            debug_assert!(row_of_state[state_idx].is_none());
            row_of_state[state_idx] = Some(row);
        }
        let mut kv_refs: Vec<&mut KvState> = Vec::with_capacity(bs);
        let mut recurrent_refs: Vec<&mut RecurrentState> = Vec::with_capacity(bs);
        for state in states_in_row_order(&mut self.requests, &row_of_state) {
            let TpRequestState { kv, recurrent, .. } = state;
            kv_refs.push(kv);
            recurrent_refs.push(
                recurrent
                    .as_mut()
                    .expect("eager TP decode request owns its recurrent state"),
            );
        }

        // GDR pointer tables over the full decode batch: allocated once at
        // capacity, refilled from the live rows every step (H2D only), so
        // swap_remove retirement between steps can never leave a stale row
        // addressed and no step pays for device allocations.
        self.decode_pointer_tables.refill_from_recurrent_refs(
            self.model.device_ctx(),
            &mut recurrent_refs,
            bs,
            "Qwen3.5 TP eager decode",
        )?;
        let token_ids: Vec<u32> = requests.iter().map(|request| request.token_id).collect();
        self.model.batch_decode_eager_logits(
            &token_ids,
            &mut kv_refs,
            &mut recurrent_refs,
            &self.decode_pointer_tables,
            &mut self.decode_buffers,
        )?;

        if self.rank != 0 {
            return Ok(Vec::new());
        }
        sample_decode_rows(
            self.model.device_ctx(),
            &self.decode_buffers.logits,
            requests,
            sample_seed,
            &mut self.sample_scratch,
        )
    }

    /// CUDA Graph decode step under TP: replay-only (every bucket was recorded
    /// by the startup pre-capture sweep), one forward for the whole batch on
    /// every rank, then (rank 0 only) the same batched host-side sampling pass
    /// as the eager path.
    ///
    /// Rows must arrive in the scheduler-owned dense slot order
    /// (`slot_idx == row`). On a request's first decode row its prefill-owned
    /// recurrent state is D2D-copied into `graph_state.slot_states[slot]` and
    /// the per-request allocation is dropped; the persistent linear-state
    /// pointer tables then keep every replay reading the fixed slot addresses.
    fn run_decode_batch_graph(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<Vec<DecodeRequestResult>> {
        let bs = requests.len();
        let graph_state = self
            .graph_state
            .as_mut()
            .expect("graph decode arm requires graph state");
        let ctx = self.model.device_ctx();

        // Resolve the worker state of every row, enforce dense slot order, and
        // admit first-decode rows into their slots. Decode request ids are
        // unique within one command (validate_decode_requests), so each slot
        // is borrowed at most once.
        let mut row_of_state: Vec<Option<usize>> = vec![None; self.requests.len()];
        for (row, request) in requests.iter().enumerate() {
            anyhow::ensure!(
                request.slot_idx == Some(row),
                "Qwen3.5 TP graph decode row {row} carries slot {:?}; rows must arrive in dense slot order 0..{bs}",
                request.slot_idx
            );
            let state_idx = self
                .requests
                .iter()
                .position(|state| state.request_id == request.request_id)
                .ok_or_else(|| {
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
            debug_assert!(row_of_state[state_idx].is_none());
            row_of_state[state_idx] = Some(row);

            if self.slot_map.get(row).copied().flatten() == Some(request.request_id) {
                anyhow::ensure!(
                    self.requests[state_idx].recurrent.is_none(),
                    "Qwen3.5 TP request {} was admitted to slot {row} but still owns prefill recurrent state",
                    request.request_id.get()
                );
            } else {
                slot_admit(&mut self.slot_map, row, request.request_id)?;
                let recurrent = self.requests[state_idx].recurrent.take().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.5 TP request {} lost its prefill recurrent state before slot admission",
                        request.request_id.get()
                    )
                })?;
                graph_state.copy_state_to_slot(ctx, &recurrent, row)?;
            }
        }

        // KV refs in row (slot) order; page tables stay per-step H2D via
        // sync_paged_meta inside batch_decode_graph.
        let mut kv_refs: Vec<&mut KvState> = states_in_row_order(&mut self.requests, &row_of_state)
            .into_iter()
            .map(|state| &mut state.kv)
            .collect();
        let token_ids: Vec<u32> = requests.iter().map(|request| request.token_id).collect();
        self.model.batch_decode_graph(
            &token_ids,
            &mut kv_refs,
            graph_state,
            DecodeGraphUse::Replay,
        )?;

        if self.rank != 0 {
            return Ok(Vec::new());
        }
        sample_decode_rows(
            ctx,
            &graph_state.buffers.logits,
            requests,
            sample_seed,
            &mut self.sample_scratch,
        )
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
        let first_token_logprob = cpu_logits[0].as_ref().and_then(|(row, top_k)| {
            pegainfer_sample::token_logprob_from_row(row, first_token, *top_k)
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

        self.run_decode_batch(requests, sample_seed)
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
        let recurrent = RecurrentState::new(
            self.model.device_ctx(),
            self.model.config(),
            self.model.geometry,
        )?;
        let state = TpRequestState {
            request_id,
            phase: TpRequestPhase::Prefilling,
            kv: self.model.alloc_kv(),
            recurrent: Some(recurrent),
        };
        self.requests.push(state);
        Ok(self.requests.len() - 1)
    }

    pub(super) fn request_index(&self, request_id: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|state| state.request_id == request_id)
    }

    /// One phase of the startup pre-capture sweep (graph mode only).
    fn precapture_phase(&mut self, phase: PrecapturePhase) -> Result<()> {
        match phase {
            PrecapturePhase::Warmup => self.model.warmup_tp_collective(),
            PrecapturePhase::Capture { bucket_idx } => {
                self.precapture_bucket(bucket_idx, DecodeGraphUse::CaptureOnly)
            }
            PrecapturePhase::Launch { bucket_idx } => {
                self.precapture_bucket(bucket_idx, DecodeGraphUse::Replay)
            }
            PrecapturePhase::Finalize => {
                let graph_state = self.graph_state.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Qwen3.5 TP pre-capture Finalize without graph state")
                })?;
                for (bucket_idx, &bucket) in BATCH_BUCKETS.iter().enumerate() {
                    if bucket > graph_state.slot_states.len() {
                        break;
                    }
                    anyhow::ensure!(
                        graph_state.graphs[bucket_idx].is_captured(),
                        "Qwen3.5 TP decode graph pre-capture left bucket {bucket} uncaptured"
                    );
                }
                Ok(())
            }
        }
    }

    /// Capture or launch one bucket with synthetic rows. Outputs are
    /// discarded; the rows exist only to give the recorded kernels valid
    /// addresses. One real row (token 0 at position 0 over a freshly
    /// allocated one-page KV state) selects nothing — the bucket is passed
    /// explicitly — and every other row is padding on the pool's reserved
    /// padding page, exactly as when serving. The sweep therefore holds one
    /// KV page at a time regardless of pool size or bucket.
    fn precapture_bucket(&mut self, bucket_idx: usize, graph_use: DecodeGraphUse) -> Result<()> {
        let bucket = BATCH_BUCKETS[bucket_idx];
        let graph_state = self.graph_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Qwen3.5 TP pre-capture on a worker without graph state")
        })?;
        anyhow::ensure!(
            bucket <= graph_state.slot_states.len(),
            "Qwen3.5 TP pre-capture bucket {bucket} exceeds {} slots",
            graph_state.slot_states.len()
        );
        let mut synthetic_kv = self.model.alloc_kv();
        let mut kv_refs = [&mut synthetic_kv];
        self.model.batch_decode_graph_padded(
            &[0u32],
            &mut kv_refs,
            graph_state,
            graph_use,
            bucket,
        )?;
        // Capture acks only after the async cuGraphUpload lands; Launch acks
        // only after the collectives drained.
        self.model
            .device_ctx()
            .stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Qwen3.5 TP pre-capture bucket {bucket} sync: {e}"))?;
        Ok(())
    }

    /// Retire a request. Graph mode also applies the scheduler's slot
    /// compaction (D2D move + occupancy assertions) so the slot layout stays
    /// dense; any mismatch between the scheduler's claim and the worker's slot
    /// map is a divergence and fails the command (poisoning the executor).
    fn drop_request(
        &mut self,
        request_id: RequestId,
        compaction: Option<TpSlotCompaction>,
    ) -> Result<bool> {
        let Some(idx) = self.request_index(request_id) else {
            anyhow::ensure!(
                compaction.is_none(),
                "Qwen3.5 TP drop of absent request {} carries a slot compaction",
                request_id.get()
            );
            return Ok(false);
        };
        if let Some(graph_state) = self.graph_state.as_mut() {
            match compaction {
                Some(compaction) => {
                    let needs_move = slot_compact(&mut self.slot_map, request_id, compaction)?;
                    if needs_move {
                        graph_state.move_slot_within(
                            self.model.device_ctx(),
                            compaction.from,
                            compaction.to,
                        )?;
                    }
                }
                None => {
                    slot_release(&mut self.slot_map, request_id);
                }
            }
        }
        self.requests.swap_remove(idx);
        Ok(true)
    }
}

/// Admit `request_id` to decode `slot`: the slot must be free (retirement and
/// compaction keep the map dense, so an occupied slot here is a scheduler
/// divergence).
pub(super) fn slot_admit(
    owners: &mut [Option<RequestId>],
    slot: usize,
    request_id: RequestId,
) -> Result<()> {
    let slot_count = owners.len();
    let owner = owners.get_mut(slot).ok_or_else(|| {
        anyhow::anyhow!("Qwen3.5 TP decode slot {slot} exceeds worker slot map {slot_count}")
    })?;
    anyhow::ensure!(
        owner.is_none(),
        "Qwen3.5 TP decode slot {slot} still owned by request {} at admission of request {}",
        owner.expect("checked").get(),
        request_id.get()
    );
    *owner = Some(request_id);
    Ok(())
}

/// Clear `request_id`'s slot if it held one. Requests retired before their
/// first decode row never materialized a slot; that is not an error.
pub(super) fn slot_release(
    owners: &mut [Option<RequestId>],
    request_id: RequestId,
) -> Option<usize> {
    let slot = owners.iter().position(|owner| *owner == Some(request_id))?;
    owners[slot] = None;
    Some(slot)
}

/// Apply the scheduler's slot compaction to the worker's slot map and report
/// whether a GPU state move is needed. Both requests may legitimately be
/// unmaterialized (retired/compacted before their first decode row), but a
/// materialized slot must hold exactly the request the scheduler claims.
pub(super) fn slot_compact(
    owners: &mut [Option<RequestId>],
    dropped: RequestId,
    compaction: TpSlotCompaction,
) -> Result<bool> {
    let TpSlotCompaction {
        moved_request_id,
        from,
        to,
    } = compaction;
    anyhow::ensure!(
        from < owners.len() && to < owners.len(),
        "Qwen3.5 TP slot compaction {from} -> {to} exceeds worker slot map {}",
        owners.len()
    );
    let dropped_owner = owners[to];
    let moved_owner = owners[from];
    if let Some(owner) = dropped_owner {
        anyhow::ensure!(
            owner == dropped,
            "Qwen3.5 TP slot {to} holds request {} where the scheduler dropped request {}",
            owner.get(),
            dropped.get()
        );
    }
    if let Some(owner) = moved_owner {
        anyhow::ensure!(
            owner == moved_request_id,
            "Qwen3.5 TP slot {from} holds request {} where the scheduler moved request {}",
            owner.get(),
            moved_request_id.get()
        );
    }
    owners[to] = moved_owner;
    owners[from] = None;
    Ok(moved_owner.is_some())
}

fn states_in_row_order<'a>(
    states: &'a mut [TpRequestState],
    row_of_state: &[Option<usize>],
) -> Vec<&'a mut TpRequestState> {
    let mut rows: Vec<(usize, &'a mut TpRequestState)> = states
        .iter_mut()
        .zip(row_of_state)
        .filter_map(|(state, row)| row.map(|row| (row, state)))
        .collect();
    rows.sort_unstable_by_key(|(row, _)| *row);
    rows.into_iter().map(|(_, state)| state).collect()
}

/// Rank-0 sampling pass over one decode batch: snapshot the requested logprob
/// rows, select one token per row, and pair each token with its logprob.
fn sample_decode_rows(
    ctx: &pegainfer_core::tensor::DeviceContext,
    logits: &pegainfer_core::tensor::HiddenStates,
    requests: &[TpDecodeStepItem],
    sample_seed: u64,
    scratch: &mut pegainfer_sample::SampleScratch,
) -> Result<Vec<DecodeRequestResult>> {
    let bs = requests.len();
    let requested_logprobs: Vec<Option<usize>> =
        requests.iter().map(|request| request.logprobs).collect();
    let cpu_logits = snapshot_requested_logprobs(ctx, logits, &requested_logprobs)?;
    let params_refs: Vec<&SamplingParams> = requests
        .iter()
        .map(|request| &request.sampling_params)
        .collect();
    let steps = vec![0u64; bs];
    let tokens =
        pegainfer_sample::select_batch(ctx, logits, &params_refs, &steps, sample_seed, scratch)?;
    anyhow::ensure!(
        tokens.len() == bs,
        "Qwen3.5 TP decode sampling returned {} tokens for {bs} rows",
        tokens.len()
    );
    Ok(requests
        .iter()
        .enumerate()
        .map(|(row, request)| {
            let logprob = cpu_logits[row].as_ref().and_then(|(logits_row, top_k)| {
                pegainfer_sample::token_logprob_from_row(logits_row, tokens[row], *top_k)
            });
            DecodeRequestResult {
                request_id: request.request_id,
                token: tokens[row],
                logprob,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_startup_watchdog_disarms_after_success() {
        let (done_tx, watchdog) = spawn_nccl_startup_watchdog().unwrap();
        disarm_nccl_startup_watchdog(done_tx, watchdog).unwrap();
    }
}
