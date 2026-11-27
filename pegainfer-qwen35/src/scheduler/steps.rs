//! Scheduler step drivers: batch prefill, overlap launch, unified
//! prefill+decode, pure decode, token dispatch/retirement, and the
//! chunked-prefill vocabulary shared with the backend.

use super::*;

// ── Batch prefill ───────────────────────────────────────────────────────

pub(super) fn prefill_batch(
    backend: &mut SchedulerBackend,
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let mut chunk = ScheduledChunk::from(scheduled);
    let sample_seed = rand::RngExt::random(rng);
    let artifacts = match backend {
        SchedulerBackend::Single(single) => {
            let logits = match single.batch_prefill_logits(&mut chunk) {
                Ok(v) => v,
                Err(e) => {
                    warn!("batch prefill failed: {e}");
                    single.release_chunk_slots(&chunk);
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
                    single.release_chunk_slots(&chunk);
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

    promote_or_requeue(
        backend,
        active_cohorts,
        prefilling,
        chunk,
        &artifacts,
        ledger,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_overlap_step(
    backend: &mut SchedulerBackend,
    decode_cohort: CohortId,
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
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
            backend.release_chunk_slots(&chunk);
            fail_chunk(chunk, &err.to_string(), ledger);
        }
    }
    decode_step_with_seed(
        backend,
        decode_cohort,
        &mut active_cohorts[decode_cohort.index()],
        decode_seed,
        ledger,
    )
}

pub(super) fn finish_async_prefill(
    backend: &mut SchedulerBackend,
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
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
            single.release_chunk_slots(&chunk);
            fail_chunk(chunk, &err.to_string(), ledger);
            return Ok(());
        }
    };
    single.apply_prefill(&mut chunk, &tokens)?;
    let artifacts = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(
        single,
        active_cohorts,
        prefilling,
        chunk,
        &artifacts,
        ledger,
    )
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn unified_step_sched(
    backend: &mut SchedulerBackend,
    decode_cohort: CohortId,
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let mut chunk = ScheduledChunk::from(scheduled);
    if matches!(backend, SchedulerBackend::Tp(_)) {
        debug_assert_eq!(decode_cohort, CohortId::A);
        let decode_sample_seed = rand::RngExt::random(rng);
        let prefill_sample_seed = rand::RngExt::random(rng);
        let result = {
            let SchedulerBackend::Tp(tp) = backend else {
                unreachable!()
            };
            tp.execute_unified(
                &chunk,
                &active_cohorts[CohortId::A.index()],
                decode_sample_seed,
                prefill_sample_seed,
            )
        };
        let artifacts = match result {
            Ok(artifacts) => artifacts,
            Err(err) => {
                warn!("TP unified step failed: {err}");
                return Err(err);
            }
        };

        let (decode_tokens, decode_logprobs) = split_decode_artifacts(&artifacts.decode);
        dispatch_decode_tokens(
            backend,
            &mut active_cohorts[CohortId::A.index()],
            &decode_tokens,
            &decode_logprobs,
            ledger,
        )?;

        let prefill = PrefillStepArtifacts::Tp(artifacts.prefill);
        return promote_or_requeue(backend, active_cohorts, prefilling, chunk, &prefill, ledger);
    }

    let SchedulerBackend::Single(backend) = backend else {
        unreachable!()
    };
    let result = backend.unified_step(
        &mut chunk,
        decode_cohort,
        &mut active_cohorts[decode_cohort.index()],
    );
    let output = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("unified step failed: {e}");
            let message = e.to_string();
            backend.release_active_slots(&active_cohorts[decode_cohort.index()]);
            fail_active(&mut active_cohorts[decode_cohort.index()], &message, ledger);
            backend.release_chunk_slots(&chunk);
            fail_chunk(chunk, &message, ledger);
            return Ok(());
        }
    };
    let decode_seed = rand::RngExt::random(rng);
    let prefill_seed = rand::RngExt::random(rng);

    if output.decoded {
        process_decode_logits(
            backend,
            decode_cohort,
            &mut active_cohorts[decode_cohort.index()],
            decode_seed,
            ledger,
        )?;
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
                backend.release_chunk_slots(&chunk);
                fail_chunk(chunk, &e.to_string(), ledger);
                return Ok(());
            }
        };
    backend.apply_prefill(&mut chunk, &tokens)?;
    let prefill = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(backend, active_cohorts, prefilling, chunk, &prefill, ledger)
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

pub(super) fn decode_step(
    backend: &mut SchedulerBackend,
    decode_cohort: CohortId,
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
    decode_step_with_seed(backend, decode_cohort, active, sample_seed, ledger)
}

fn decode_step_with_seed(
    backend: &mut SchedulerBackend,
    decode_cohort: CohortId,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let (tokens, logprobs_vec) = match backend {
        SchedulerBackend::Single(single) => {
            if let Err(e) = single.decode_graph(decode_cohort, active) {
                warn!("batch_decode_graph error: {e}");
                let message = e.to_string();
                single.release_active_slots(active);
                fail_active(active, &message, ledger);
                return Ok(());
            }
            // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
            match single.sample_decode_logits(decode_cohort, active, sample_seed) {
                Ok(v) => v,
                Err(e) => {
                    warn!("decode sampling/logprobs error: {e}");
                    let message = e.to_string();
                    single.release_active_slots(active);
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
    decode_cohort: CohortId,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let (tokens, logprobs_vec) =
        match backend.sample_decode_logits(decode_cohort, active, sample_seed) {
            Ok(v) => v,
            Err(e) => {
                warn!("decode sampling/logprobs error: {e}");
                let message = e.to_string();
                backend.release_active_slots(active);
                fail_active(active, &message, ledger);
                return Ok(());
            }
        };

    backend.apply_decode(active, &tokens)?;
    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec, ledger)
}

/// Dispatch sampled decode tokens and retire completed backend state.
///
/// `tokens` and `logprobs` are indexed by original position in the decoded cohort's `active`;
/// retirements are collected first and removed in reverse order.
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

    // Remove retired requests from the active cohort in reverse order and release their slots.
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
        // Single GPU maps active slots dynamically via pointer tables; `swap_remove` needs no D2D copy.
        active.swap_remove(idx)
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        let ActiveBackendState::Single {
            cohort,
            graph_slot_idx,
            ..
        } = state
        else {
            anyhow::bail!("single-GPU retirement received TP active state")
        };
        self.release_slot(*cohort, *graph_slot_idx);
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
            SchedulerBackend::Single(backend) => backend.take_active_request(active, idx),
            SchedulerBackend::Tp(backend) => backend.take_active_request(active, idx),
        }
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        match (self, state) {
            (
                SchedulerBackend::Single(backend),
                ActiveBackendState::Single {
                    cohort,
                    graph_slot_idx,
                    ..
                },
            ) => {
                backend.release_slot(*cohort, *graph_slot_idx);
                Ok(())
            }
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
        /// Pre-assigned `(cohort, slot_idx)` target per request; prefill writes in place.
        slots: Vec<(CohortId, usize)>,
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
                    slots: Vec::with_capacity(n),
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
                    ScheduledChunkBackendState::Single { kvs, slots },
                    PrefillBackendState::Single {
                        kv,
                        cohort,
                        slot_idx,
                    },
                ) => {
                    kvs.push(kv);
                    slots.push((cohort, slot_idx));
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

/// Drain the planned FIFO prefix, aligning chunks to snapshot boundaries.
pub(super) fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    plan: &StepPlan,
    snapshot_stride: Option<usize>,
) -> Vec<PrefillingRequest35> {
    debug_assert!(
        plan.prefill.len() <= prefilling.len(),
        "step plan cannot schedule more prompts than are queued"
    );
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..plan.prefill.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&plan.prefill) {
        let request = &p.req.request;
        p.step_chunk = prefill_step_chunk(
            p.cursor,
            *chunk,
            request.prompt_tokens.len(),
            request.history_tokens,
            snapshot_stride,
        );
    }
    scheduled
}

/// Run the plan and return the prompt rows forwarded after snapshot alignment.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_step(
    backend: &mut SchedulerBackend,
    decode_cohort: Option<CohortId>,
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
    prefilling: &mut Vec<PrefillingRequest35>,
    plan: &StepPlan,
    inflight_prefill: &mut Option<InflightPrefill>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
    snapshot_stride: Option<usize>,
) -> Result<usize> {
    let scheduled = take_prefill_chunks(prefilling, plan, snapshot_stride);
    if scheduled.is_empty() {
        debug_assert!(
            plan.prefill.is_empty(),
            "a planned prefill row must schedule a prompt"
        );
        if plan.decode {
            let cohort = decode_cohort.expect("decode step requires an active cohort");
            decode_step(
                backend,
                cohort,
                &mut active_cohorts[cohort.index()],
                rng,
                ledger,
            )?;
        }
        return Ok(0);
    }
    let forwarded: usize = scheduled.iter().map(|p| p.step_chunk).sum();
    let overlap = backend.overlap_enabled();
    if !plan.decode {
        prefill_batch(backend, active_cohorts, scheduled, prefilling, rng, ledger)?;
    } else if overlap {
        let cohort = decode_cohort.expect("overlap decode step requires an active cohort");
        launch_overlap_step(
            backend,
            cohort,
            active_cohorts,
            scheduled,
            inflight_prefill,
            rng,
            ledger,
        )?;
    } else {
        let cohort = decode_cohort.expect("unified decode step requires an active cohort");
        unified_step_sched(
            backend,
            cohort,
            active_cohorts,
            scheduled,
            prefilling,
            rng,
            ledger,
        )?;
    }
    Ok(forwarded)
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
    active_cohorts: &mut [Vec<ActiveRequest35>; 2],
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

        let target_cohort = match &backend_state {
            PrefillBackendState::Single { cohort, .. } => *cohort,
            PrefillBackendState::Tp { .. } => CohortId::A,
        };
        let active_len = active_cohorts[target_cohort.index()].len();
        let active_backend_state = backend.promote_prefill_state(active_len, backend_state);
        ledger.push_tokens(req.id, &[first_token], &[artifact.logprob]);
        active_cohorts[target_cohort.index()].push(ActiveRequest35 {
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
        _active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        let PrefillBackendState::Single {
            kv,
            cohort,
            slot_idx,
        } = state
        else {
            panic!("single-GPU promotion received TP prefill state");
        };
        // Prefill already wrote directly into the target cohort slot; no D2D copy needed.
        ActiveBackendState::Single {
            kv,
            cohort,
            graph_slot_idx: slot_idx,
        }
    }

    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        _expectation: DropExpectation,
    ) -> Result<()> {
        let PrefillBackendState::Single {
            cohort, slot_idx, ..
        } = state
        else {
            anyhow::bail!("single-GPU prefill drop received TP state")
        };
        self.release_slot(*cohort, *slot_idx);
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
            (SchedulerBackend::Single(backend), state @ PrefillBackendState::Single { .. }) => {
                backend.drop_prefill_state(state, expectation)
            }
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
        ScheduledChunkBackendState::Single { kvs, slots } => kvs
            .into_iter()
            .zip(slots)
            .map(|(kv, (cohort, slot_idx))| PrefillBackendState::Single {
                kv,
                cohort,
                slot_idx,
            })
            .collect(),
        ScheduledChunkBackendState::Tp { request_ids } => request_ids
            .into_iter()
            .map(|request_id| PrefillBackendState::Tp { request_id })
            .collect(),
    }
}
