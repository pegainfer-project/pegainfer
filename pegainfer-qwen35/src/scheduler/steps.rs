//! Scheduler step drivers: batch prefill, overlap launch, unified
//! prefill+decode, pure decode, token dispatch/retirement, and the
//! chunked-prefill vocabulary shared with the backend.

use super::*;

// ── Batch prefill ───────────────────────────────────────────────────────

pub(super) fn prefill_batch(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
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
                    fail_chunk(chunk, &e.to_string(), ledger);
                    return Ok(());
                }
            };
            let prefill_sample_seed = rand::RngExt::random(rng);
            match single.sample_prefill_logits(&chunk.reqs, &logits, prefill_sample_seed) {
                Ok((tokens, logprobs)) => {
                    single.apply_prefill(&mut chunk, &tokens)?;
                    PrefillStepArtifacts::Single { tokens, logprobs }
                }
                Err(e) => {
                    warn!("prefill sampling failed: {e}");
                    fail_chunk(chunk, &e.to_string(), ledger);
                    return Ok(());
                }
            }
        }
        SchedulerBackend::Tp(tp) => match tp.execute_prefill_chunk(&chunk, sample_seed) {
            Ok(v) => PrefillStepArtifacts::Tp(v),
            Err(e) => {
                warn!("TP prefill chunk failed: {e}");
                return Err(e);
            }
        },
    };

    promote_or_requeue(backend, active, prefilling, chunk, &artifacts, ledger)
}

pub(super) fn launch_overlap_step(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    inflight_prefill: &mut Option<InflightPrefill>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
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
            fail_chunk(chunk, &err.to_string(), ledger);
        }
    }
    decode_step_with_seed(backend, active, decode_seed, ledger)
}

pub(super) fn finish_async_prefill(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    inflight: InflightPrefill,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let InflightPrefill {
        mut chunk,
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
            fail_chunk(chunk, &err.to_string(), ledger);
            return Ok(());
        }
    };
    single.apply_prefill(&mut chunk, &tokens)?;
    let artifacts = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(single, active, prefilling, chunk, &artifacts, ledger)
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

pub(super) fn unified_step_sched(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
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
                return Err(err);
            }
        };

        let (decode_tokens, decode_logprobs) = split_decode_artifacts(&artifacts.decode);
        dispatch_decode_tokens(backend, active, &decode_tokens, &decode_logprobs, ledger)?;

        let prefill = PrefillStepArtifacts::Tp(artifacts.prefill);
        return promote_or_requeue(backend, active, prefilling, chunk, &prefill, ledger);
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
            fail_active(active, &message, ledger);
            fail_chunk(chunk, &message, ledger);
            return Ok(());
        }
    };
    let decode_seed = rand::RngExt::random(rng);
    let prefill_seed = rand::RngExt::random(rng);

    // Process decode results FIRST (it may retire requests and free graph slots
    // that promotion then fills densely).
    if output.decoded {
        process_decode_logits(backend, active, decode_seed, ledger)?;
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
                fail_chunk(chunk, &e.to_string(), ledger);
                return Ok(());
            }
        };
    backend.apply_prefill(&mut chunk, &tokens)?;
    let prefill = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(backend, active, prefilling, chunk, &prefill, ledger)
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

pub(super) fn decode_step(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    // Preserve the historical scheduler RNG sequence: TP consumes the first
    // seed, while single-GPU decode consumed a second seed inside sampling.
    let first_seed = rand::RngExt::random(rng);
    let sample_seed = if matches!(backend, SchedulerBackend::Single(_)) {
        rand::RngExt::random(rng)
    } else {
        first_seed
    };
    decode_step_with_seed(backend, active, sample_seed, ledger)
}

fn decode_step_with_seed(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let (tokens, logprobs_vec) = match backend {
        SchedulerBackend::Single(single) => {
            if let Err(e) = single.decode_graph(active) {
                warn!("batch_decode_graph error: {e}");
                let message = e.to_string();
                fail_active(active, &message, ledger);
                return Ok(());
            }
            // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
            match single.sample_decode_logits(active, sample_seed) {
                Ok(v) => v,
                Err(e) => {
                    warn!("decode sampling/logprobs error: {e}");
                    let message = e.to_string();
                    fail_active(active, &message, ledger);
                    return Ok(());
                }
            }
        }
        SchedulerBackend::Tp(tp) => match tp.execute_decode(active, sample_seed) {
            Ok(v) => split_decode_artifacts(&v),
            Err(e) => {
                warn!("TP eager decode error: {e}");
                return Err(e);
            }
        },
    };

    if let SchedulerBackend::Single(single) = backend {
        single.apply_decode(active, &tokens)?;
    }
    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec, ledger)
}

/// Process decode logits from unified step: sample, extract logprobs, dispatch.
fn process_decode_logits(
    backend: &mut SingleGpuBackend,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let (tokens, logprobs_vec) = match backend.sample_decode_logits(active, sample_seed) {
        Ok(v) => v,
        Err(e) => {
            warn!("decode sampling/logprobs error: {e}");
            let message = e.to_string();
            fail_active(active, &message, ledger);
            return Ok(());
        }
    };

    backend.apply_decode(active, &tokens)?;
    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec, ledger)
}

/// Dispatch sampled decode tokens and retire completed backend state.
///
/// `tokens` and `logprobs` are indexed by original position in `active`.
/// Retirements collected first, then compacted in reverse order.
pub(super) fn dispatch_decode_tokens(
    backend: &mut impl DecodeDispatchBackend,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
    ledger: &mut RequestLedger,
) -> Result<()> {
    let mut to_retire = Vec::new();
    for (i, req) in active.iter_mut().enumerate() {
        if ledger.is_aborted(req.id) {
            to_retire.push((i, None));
            continue;
        }
        let token = tokens[i];
        let finish = if let Some(cause) = req
            .stop_policy
            .classify(token, |id| backend.is_stop_token(id))
        {
            Some((FinishReason::Stop, Some(cause)))
        } else if ledger.completion_tokens(req.id) + 1 >= req.max_tokens {
            Some((FinishReason::Length, None))
        } else {
            None
        };
        if finish.is_some() {
            to_retire.push((i, finish));
        } else {
            ledger.push_tokens(req.id, &[token], std::slice::from_ref(&logprobs[i]));
            req.last_token = token;
        }
    }

    // Keep slots dense and wait for retirement ACK before publishing completion.
    for (i, finish) in to_retire.into_iter().rev() {
        let request = backend.take_active_request(active, i);
        backend.drop_active_state(&request.backend_state)?;
        if ledger.is_aborted(request.id) {
            ledger.retire(request.id);
        } else if let Some((reason, stop_cause)) = finish {
            ledger.push_tokens(request.id, &[tokens[i]], std::slice::from_ref(&logprobs[i]));
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                request.client_label,
                request.prompt_len,
                ledger.completion_tokens(request.id),
                reason
            );
            ledger.finish_with_cause(request.id, reason, stop_cause);
        }
    }
    Ok(())
}

pub(super) trait DecodeDispatchBackend {
    fn is_stop_token(&self, token: u32) -> bool;
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

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        match self {
            SchedulerBackend::Single(backend) => compact_single_slot(backend, active, idx),
            SchedulerBackend::Tp(backend) => backend.take_active_request(active, idx),
        }
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        match (self, state) {
            (SchedulerBackend::Single(_), ActiveBackendState::Single { .. }) => Ok(()),
            (SchedulerBackend::Tp(backend), ActiveBackendState::Tp { request_id, .. }) => {
                let compaction = backend.pending_compaction.take();
                backend.executor.drop_request_with_compaction(
                    *request_id,
                    DropExpectation::MustExist,
                    compaction,
                )
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
pub(super) struct ScheduledChunk {
    pub(super) reqs: Vec<QueuedRequest>,
    pub(super) backend_state: ScheduledChunkBackendState,
    /// Prompt cursor after this step's chunk
    pub(super) ends: Vec<usize>,
    /// This step's chunked token slice per request
    pub(super) windows: Vec<Vec<u32>>,
}

pub(super) struct InflightPrefill {
    // Fields drop in declaration order. Drain the stream before request state
    // can return KV pages or release recurrent/convolution buffers on unwind.
    pub(super) output: AsyncPrefillOutput,
    pub(super) chunk: ScheduledChunk,
    sample_seed: u64,
}

pub(super) enum ScheduledChunkBackendState {
    Single {
        // Keep request allocations stable while an async prefill owns the chunk.
        #[allow(clippy::vec_box)]
        kvs: Vec<Box<RequestKv>>,
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
                .push(p.req.request.prompt_tokens[p.cursor..end].to_vec());
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
pub(super) fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    prefill_budget: usize,
    snapshot_stride: Option<usize>,
) -> Vec<PrefillingRequest35> {
    let remaining: Vec<usize> = prefilling
        .iter()
        .map(|p| p.req.request.prompt_tokens.len() - p.cursor)
        .collect();
    let chunks = plan_prefill_chunks(&remaining, prefill_budget);
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..chunks.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&chunks) {
        p.step_chunk = clamp_prefill_chunk(p.cursor, *chunk, snapshot_stride);
    }
    scheduled
}

/// Report a forward/sampling failure to every request in the failed chunk.
fn fail_chunk(chunk: ScheduledChunk, message: &str, ledger: &mut RequestLedger) {
    for req in chunk.reqs {
        fail_or_retire(req.id, message, ledger);
    }
}

fn fail_active(active: &mut Vec<ActiveRequest35>, message: &str, ledger: &mut RequestLedger) {
    for req in active.drain(..) {
        fail_or_retire(req.id, message, ledger);
    }
}

/// Commit immediate completions after retirement, and put unfinished chunks
/// back at the front of the prefill queue without changing their FIFO order.
pub(super) fn promote_or_requeue(
    backend: &mut impl PrefillPromoteBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    chunk: ScheduledChunk,
    artifacts: &PrefillStepArtifacts,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let ScheduledChunk {
        reqs,
        backend_state,
        ends,
        ..
    } = chunk;
    let mut still_prefilling = Vec::new();
    let backend_states = split_scheduled_backend_state(backend_state);
    for (i, ((req, backend_state), end)) in
        reqs.into_iter().zip(backend_states).zip(ends).enumerate()
    {
        if ledger.is_aborted(req.id) {
            backend.drop_prefill_state(&backend_state, DropExpectation::MustExist)?;
            ledger.retire(req.id);
            continue;
        }
        if end < req.request.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                req,
                backend_state,
                cursor: end,
                step_chunk: 0,
            });
            continue;
        }

        let artifact = artifacts.final_artifact(i);
        let first_token = artifact.token;
        let finish = if let Some(cause) = req
            .request
            .stop_policy
            .classify(first_token, |id| backend.is_stop_token(id))
        {
            Some((FinishReason::Stop, Some(cause)))
        } else if req.request.max_tokens <= 1 {
            Some((FinishReason::Length, None))
        } else {
            None
        };
        if let Some((reason, stop_cause)) = finish {
            backend.drop_prefill_state(&backend_state, DropExpectation::MustExist)?;
            if ledger.is_aborted(req.id) {
                ledger.retire(req.id);
            } else {
                ledger.push_tokens(req.id, &[first_token], &[artifact.logprob]);
                ledger.finish_with_cause(req.id, reason, stop_cause);
            }
            continue;
        }

        let active_backend_state = backend.promote_prefill_state(active.len(), backend_state);
        ledger.push_tokens(req.id, &[first_token], &[artifact.logprob]);
        active.push(ActiveRequest35 {
            id: req.id,
            client_label: req.request.client_label,
            backend_state: active_backend_state,
            last_token: first_token,
            max_tokens: req.request.max_tokens,
            prompt_len: req.request.prompt_tokens.len(),
            params: req.request.params,
            stop_policy: req.request.stop_policy,
            logprobs: req.request.logprobs,
        });
    }
    prefilling.splice(0..0, still_prefilling);
    Ok(())
}

pub(super) trait PrefillPromoteBackend {
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
            (SchedulerBackend::Single(single), state @ PrefillBackendState::Single { .. }) => {
                single.promote_prefill_state(active_len, state)
            }
            (SchedulerBackend::Tp(backend), PrefillBackendState::Tp { request_id }) => {
                let slot_idx = slot_for_new_request(active_len, backend.max_batch())
                    .expect("admission must reserve a TP decode slot");
                ActiveBackendState::Tp {
                    request_id,
                    slot_idx,
                }
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
