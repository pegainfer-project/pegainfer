//! Scheduler for Qwen3.5: a [`Scheduler`] on the contract driver thread.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)
//!
//! Shape matches K3: this module implements `submit` / `step` / `metrics`,
//! writes the ledger, and returns [`Engine`] from `start_*`. GPU execute and
//! overlap live in [`backend`].

mod backend;
mod plan;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use log::debug;
use log::info;
use log::warn;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason as ContractRejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId as EngineRequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::spawn_scheduler;
use rand::SeedableRng;
use rand::rngs::StdRng;

use self::backend::ActiveRequest35;
use self::backend::CublasThreadGuard;
use self::backend::DecodeDispatchBackend;
use self::backend::InflightPrefill;
use self::backend::PrefillPromoteBackend;
use self::backend::PrefillStepArtifacts;
use self::backend::PrefillingRequest35;
use self::backend::Qwen35Backend;
use self::backend::ScheduledChunk;
use self::backend::SingleGpuBackend;
use self::backend::TpSchedulerBackend;
use self::backend::bind_model_thread;
use self::backend::servable_len;
use self::backend::split_decode_artifacts;
use self::backend::split_scheduled_backend_state;
use self::plan::ActiveDecodeState;
use self::plan::ActiveKvBudget;
use self::plan::ExecutionPlan;
use self::plan::PrefillKvBudget;
use self::plan::PrefillQueueState;
use self::plan::RejectReason;
use self::plan::admit_pending_requests;
use self::plan::choose_prefill_budget;
use self::plan::max_kv_tokens;
use self::plan::plan_prefill_chunks;
use self::plan::prefilling_future_pages;
use crate::Qwen35DecodeOverlap;
use crate::Qwen35SchedulerPolicy;
use crate::tp_executor::DropExpectation;
use crate::weights::Qwen35Model;

struct PendingRequest {
    id: EngineRequestId,
    request: Request,
}

pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

/// Env-gated per-step ITL diagnostics (issue #470). When `PEGAINFER_ITL_DEBUG`
/// is set, the scheduler emits one `ITL_STEP` line per executed step, tagging
/// the plan kind, the *actual* prefill-chunk token count associated with the
/// action, the active decode width, and the CPU wall-time. Off by default.
fn itl_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEGAINFER_ITL_DEBUG").is_some())
}

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

// ── Engine assembly ─────────────────────────────────────────────────────

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
    let total_blocks = model.kv_pool().capacity_pages().saturating_sub(1);
    let block_size = model.kv_pool().layout().page_size;
    let servable = servable_len(
        model.config().max_position_embeddings,
        total_blocks,
        block_size,
    );
    let backend = SingleGpuBackend::new(model, max_batch, decode_overlap)?;
    let scheduler = Qwen35Scheduler::new(
        Qwen35Backend::Single(backend),
        seed,
        max_prefill_tokens,
        scheduler_policy,
    );
    Ok(Engine {
        schedulers: vec![spawn_scheduler("qwen35-scheduler", scheduler)],
        info: EngineInfo {
            kv_capacity: Some(KvCapacity {
                total_blocks,
                block_size,
            }),
            servable_len: Some(servable),
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
) -> Result<Engine> {
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
    let scheduler = Qwen35Scheduler::new(
        Qwen35Backend::Tp(backend),
        seed,
        max_prefill_tokens,
        Qwen35SchedulerPolicy::Off,
    );
    Ok(Engine {
        schedulers: vec![spawn_scheduler("qwen35-scheduler-tp", scheduler)],
        info: EngineInfo {
            kv_capacity: Some(kv_capacity),
            servable_len: Some(servable),
        },
        lora: None,
    })
}

// ── The Scheduler implementation ────────────────────────────────────────

fn finish_or_retire(id: EngineRequestId, reason: FinishReason, ledger: &mut RequestLedger) {
    if ledger.is_aborted(id) {
        ledger.retire(id);
    } else {
        ledger.finish(id, reason);
    }
}

struct Qwen35Scheduler {
    backend: Qwen35Backend,
    rng: StdRng,
    prefill_budget: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    pending: Vec<PendingRequest>,
    active: Vec<ActiveRequest35>,
    prefilling: Vec<PrefillingRequest35>,
    inflight_prefill: Option<InflightPrefill>,
    /// Lives for the driver-thread lifetime. Single-GPU only; TP workers bind
    /// themselves.
    cublas_guard: Option<CublasThreadGuard>,
    ready_logged: bool,
}

impl Qwen35Scheduler {
    fn new(
        backend: Qwen35Backend,
        seed: u64,
        prefill_budget: usize,
        scheduler_policy: Qwen35SchedulerPolicy,
    ) -> Self {
        Self {
            backend,
            rng: StdRng::seed_from_u64(seed),
            prefill_budget,
            scheduler_policy,
            pending: Vec::new(),
            active: Vec::new(),
            prefilling: Vec::new(),
            inflight_prefill: None,
            cublas_guard: None,
            ready_logged: false,
        }
    }

    /// Bind CUDA + init thread-local cuBLAS on the first `step` of the
    /// driver thread (single-GPU only). Stored on `self` so the guard
    /// outlives every later step.
    fn bind_if_needed(&mut self) -> Result<()> {
        if self.cublas_guard.is_some() {
            return Ok(());
        }
        if let Qwen35Backend::Single(single) = &self.backend {
            let guard = bind_model_thread(single.model())?;
            self.cublas_guard = Some(guard);
        }
        if !self.ready_logged {
            info!("scheduler ready (max_batch={})", self.backend.max_batch());
            self.ready_logged = true;
        }
        Ok(())
    }

    fn prune_aborted(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let mut index = 0;
        while index < self.pending.len() {
            if ledger.is_aborted(self.pending[index].id) {
                let removed = self.pending.remove(index);
                ledger.retire(removed.id);
            } else {
                index += 1;
            }
        }

        for idx in (0..self.active.len()).rev() {
            if ledger.is_aborted(self.active[idx].id) {
                let removed = self.backend.take_active_request(&mut self.active, idx);
                self.backend.drop_active_state(&removed.backend_state)?;
                ledger.retire(removed.id);
            }
        }

        for idx in (0..self.prefilling.len()).rev() {
            if ledger.is_aborted(self.prefilling[idx].id) {
                let removed = self.prefilling.remove(idx);
                let expectation = prefill_drop_expectation(removed.cursor);
                self.backend
                    .drop_prefill_state(&removed.backend_state, expectation)?;
                ledger.retire(removed.id);
            }
        }
        Ok(())
    }

    fn reject_echo(&mut self, ledger: &mut RequestLedger) {
        let mut index = 0;
        while index < self.pending.len() {
            let Some(reason) = echo_refusal(&self.pending[index].request) else {
                index += 1;
                continue;
            };
            let removed = self.pending.remove(index);
            if ledger.is_aborted(removed.id) {
                ledger.retire(removed.id);
            } else {
                ledger.reject(removed.id, reason);
            }
        }
    }

    fn admit_pending(&mut self, ledger: &mut RequestLedger) {
        let pending = std::mem::take(&mut self.pending);
        let active_budget: Vec<ActiveKvBudget> = self
            .active
            .iter()
            .map(|req| ActiveKvBudget {
                prompt_len: req.prompt_len,
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let page_size = self.backend.page_size();
        let prefilling_budget: Vec<PrefillKvBudget> = self
            .prefilling
            .iter()
            .map(|p| PrefillKvBudget {
                current_tokens: p.cursor,
                prompt_len: p.request.prompt_tokens.len(),
                max_tokens: p.request.max_tokens,
            })
            .collect();
        let page_budget = self
            .backend
            .available_pages(&self.active, &self.prefilling)
            .saturating_sub(prefilling_future_pages(&prefilling_budget, page_size));
        let decode_batching_slot = self
            .backend
            .max_batch()
            .saturating_sub(self.prefilling.len());
        let admission = admit_pending_requests(
            pending,
            &active_budget,
            decode_batching_slot,
            page_size,
            page_budget,
            self.backend.capacity_pages_for_requests(),
            self.backend.max_position_embeddings(),
            |req| req.request.prompt_tokens.len(),
            |req| req.request.max_tokens,
        );
        for (rejected, reason) in admission.rejected {
            if ledger.is_aborted(rejected.id) {
                ledger.retire(rejected.id);
            } else {
                ledger.reject(
                    rejected.id,
                    contract_reject_reason(
                        rejected.request.prompt_tokens.len(),
                        rejected.request.max_tokens,
                        reason,
                    ),
                );
            }
        }
        for req in admission.pending {
            if ledger.is_aborted(req.id) {
                ledger.retire(req.id);
                continue;
            }
            ledger.admit(req.id);
            debug!(
                "request admitted: request_id={} prompt_len={} max_tokens={}",
                req.id,
                req.request.prompt_tokens.len(),
                req.request.max_tokens
            );
            match self.backend.alloc_prefill_state() {
                Ok(backend_state) => self.prefilling.push(PrefillingRequest35 {
                    id: req.id,
                    request: req.request,
                    backend_state,
                    cursor: 0,
                    step_chunk: 0,
                }),
                Err(error) => {
                    warn!("failed to allocate recurrent state for new request: {error}");
                    ledger.fail(req.id, error.to_string());
                }
            }
        }
        self.pending = admission.deferred;
    }
}

impl Scheduler for Qwen35Scheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.pending.push(PendingRequest {
            id: request.id,
            request: request.request,
        });
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.bind_if_needed()?;

        if self
            .inflight_prefill
            .as_mut()
            .is_some_and(|prefill| prefill.output.is_ready())
        {
            let (prefill_tokens, prefill_reqs) =
                self.inflight_prefill.as_ref().map_or((0, 0), |prefill| {
                    (
                        prefill.chunk.windows.iter().map(Vec::len).sum(),
                        prefill.chunk.reqs.len(),
                    )
                });
            let decode_n = self.active.len();
            let step_start = itl_debug_enabled().then(Instant::now);
            finish_async_prefill(
                &mut self.backend,
                &mut self.active,
                &mut self.prefilling,
                self.inflight_prefill
                    .take()
                    .expect("ready async prefill must still be present"),
                ledger,
            )?;
            log_itl_step(
                step_start,
                "overlap_complete",
                prefill_tokens,
                prefill_reqs,
                decode_n,
            );
        }

        self.prune_aborted(ledger)?;
        self.reject_echo(ledger);

        // One async prefill owns its scheduled request state. Do not admit or
        // launch a second chunk until it resolves. Active decode keeps moving;
        // if it retires first, wait on the event inside `step` instead of
        // returning idle to the driver.
        if self.inflight_prefill.is_some() {
            let itl_step_start = itl_debug_enabled().then(Instant::now);
            let (itl_prefill_tokens, itl_prefill_reqs) =
                self.inflight_prefill.as_ref().map_or((0, 0), |prefill| {
                    (
                        prefill.chunk.windows.iter().map(Vec::len).sum(),
                        prefill.chunk.reqs.len(),
                    )
                });
            let itl_decode_n = self.active.len();
            let itl_plan_kind = if self.active.is_empty() {
                finish_async_prefill(
                    &mut self.backend,
                    &mut self.active,
                    &mut self.prefilling,
                    self.inflight_prefill
                        .take()
                        .expect("async prefill must be present before blocking wait"),
                    ledger,
                )?;
                "overlap_wait"
            } else {
                decode_step(&mut self.backend, &mut self.active, &mut self.rng, ledger)?;
                "overlap_decode"
            };
            log_itl_step(
                itl_step_start,
                itl_plan_kind,
                itl_prefill_tokens,
                itl_prefill_reqs,
                itl_decode_n,
            );
            return Ok(());
        }

        self.admit_pending(ledger);

        let active_decode: Vec<ActiveDecodeState> = self
            .active
            .iter()
            .map(|req| ActiveDecodeState {
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let prefill_queue: Vec<PrefillQueueState> = self
            .prefilling
            .iter()
            .map(|req| PrefillQueueState {
                remaining_tokens: req.request.prompt_tokens.len().saturating_sub(req.cursor),
            })
            .collect();
        let step_prefill_budget = choose_prefill_budget(
            self.scheduler_policy,
            self.prefill_budget,
            &active_decode,
            &prefill_queue,
        );
        let scheduled = take_prefill_chunks(&mut self.prefilling, step_prefill_budget);
        let itl_debug = itl_debug_enabled();
        let itl_prefill_tokens: usize = scheduled.iter().map(|p| p.step_chunk).sum();
        let itl_prefill_reqs = scheduled.len();
        let itl_decode_n = self.active.len();
        let Some(plan) = plan::build_next_plan(!self.active.is_empty(), scheduled) else {
            return Ok(());
        };
        let itl_plan_kind = match &plan {
            ExecutionPlan::Unified { .. } if matches!(&self.backend, Qwen35Backend::Single(single) if single.overlap_enabled()) => {
                "overlap_launch"
            }
            ExecutionPlan::Unified { .. } => "unified",
            ExecutionPlan::Prefill { .. } => "prefill",
            ExecutionPlan::Decode => "decode",
        };
        let itl_step_start = itl_debug.then(Instant::now);
        match plan {
            ExecutionPlan::Unified { pending } => {
                if matches!(&self.backend, Qwen35Backend::Single(single) if single.overlap_enabled())
                {
                    launch_overlap_step(
                        &mut self.backend,
                        &mut self.active,
                        pending,
                        &mut self.inflight_prefill,
                        &mut self.rng,
                        ledger,
                    )?;
                } else {
                    unified_step_sched(
                        &mut self.backend,
                        &mut self.active,
                        pending,
                        &mut self.prefilling,
                        &mut self.rng,
                        ledger,
                    )?;
                }
            }
            ExecutionPlan::Prefill { pending } => {
                prefill_batch(
                    &mut self.backend,
                    &mut self.active,
                    pending,
                    &mut self.prefilling,
                    &mut self.rng,
                    ledger,
                )?;
            }
            ExecutionPlan::Decode => {
                decode_step(&mut self.backend, &mut self.active, &mut self.rng, ledger)?;
            }
        }
        log_itl_step(
            itl_step_start,
            itl_plan_kind,
            itl_prefill_tokens,
            itl_prefill_reqs,
            itl_decode_n,
        );
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        let kv_total_blocks = self.backend.capacity_pages_for_requests() as u64;
        let (num_running_reqs, num_waiting_reqs) = logical_load_counts(
            self.active.len(),
            self.prefilling.len(),
            self.inflight_prefill
                .as_ref()
                .map_or(0, |prefill| prefill.chunk.reqs.len()),
            self.pending.len(),
        );
        SchedulerMetrics {
            kv_used_blocks:
                kv_total_blocks.saturating_sub(
                    self.backend.available_pages(&self.active, &self.prefilling) as u64,
                ),
            kv_total_blocks,
            num_running_reqs,
            num_waiting_reqs,
            spec_decode: None,
        }
    }
}

/// Running = decode + resident prefill + in-flight overlap prefill. Waiting
/// is the queued/deferred count the caller already computed. Inflight must
/// stay in the running tally so an overlap wait inside `step` is not
/// published as idle.
fn logical_load_counts(
    active_len: usize,
    prefilling_len: usize,
    inflight_prefill_reqs: usize,
    num_waiting_reqs: usize,
) -> (u64, u64) {
    (
        (active_len + prefilling_len + inflight_prefill_reqs) as u64,
        num_waiting_reqs as u64,
    )
}

/// Cursor 0 never materialized worker state; a later cursor did. TP drop
/// must match that, or a cancelled first chunk looks like a missing request
/// on the worker and a cancelled mid-prompt chunk looks like a leak.
fn prefill_drop_expectation(cursor: usize) -> DropExpectation {
    if cursor == 0 {
        DropExpectation::MustBeAbsent
    } else {
        DropExpectation::MustExist
    }
}

/// Echo is unsupported on Qwen3.5: refuse before KV admission with a zero
/// prefill bound so the contract `Display` is the client message.
fn echo_refusal(request: &Request) -> Option<ContractRejectReason> {
    request
        .echo
        .then_some(ContractRejectReason::EchoPrefillTokens {
            prompt_tokens: request.prompt_tokens.len(),
            limit: 0,
        })
}

/// Widen a plan-level admission verdict into the contract's typed reason.
fn contract_reject_reason(
    prompt_tokens: usize,
    max_tokens: usize,
    reason: RejectReason,
) -> ContractRejectReason {
    match reason {
        RejectReason::ContextLength { limit } => ContractRejectReason::ContextLength {
            prompt_tokens,
            max_tokens,
            limit,
        },
        RejectReason::KvBudget => ContractRejectReason::KvBudget {
            prompt_tokens,
            worst_case_tokens: max_kv_tokens(prompt_tokens, max_tokens),
        },
    }
}

fn fail_ids(
    ids: impl IntoIterator<Item = EngineRequestId>,
    ledger: &mut RequestLedger,
    message: &str,
) {
    for id in ids {
        if ledger.is_aborted(id) {
            ledger.retire(id);
        } else {
            ledger.fail(id, message);
        }
    }
}

fn fail_active(
    backend: &mut impl DecodeDispatchBackend,
    active: &mut Vec<ActiveRequest35>,
    ledger: &mut RequestLedger,
    message: &str,
) -> Result<()> {
    while !active.is_empty() {
        let req = backend.take_active_request(active, active.len() - 1);
        backend.drop_active_state(&req.backend_state)?;
        if ledger.is_aborted(req.id) {
            ledger.retire(req.id);
        } else {
            ledger.fail(req.id, message);
        }
    }
    Ok(())
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    backend: &mut Qwen35Backend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let mut chunk = ScheduledChunk::from(scheduled);
    let sample_seed = rand::RngExt::random(rng);
    let artifacts = match backend {
        Qwen35Backend::Single(single) => {
            let logits = match single.batch_prefill_logits(&mut chunk) {
                Ok(v) => v,
                Err(e) => {
                    warn!("batch prefill failed: {e}");
                    fail_ids(chunk.ids, ledger, &e.to_string());
                    return Ok(());
                }
            };
            let prefill_sample_seed = rand::RngExt::random(rng);
            match single.sample_prefill_logits(&chunk.reqs, &logits, prefill_sample_seed) {
                Ok((tokens, logprobs)) => PrefillStepArtifacts::Single { tokens, logprobs },
                Err(e) => {
                    warn!("prefill sampling failed: {e}");
                    fail_ids(chunk.ids, ledger, &e.to_string());
                    return Ok(());
                }
            }
        }
        Qwen35Backend::Tp(tp) => match tp.execute_prefill_chunk(&chunk, sample_seed) {
            Ok(v) => PrefillStepArtifacts::Tp(v),
            Err(e) => {
                warn!("TP prefill chunk failed: {e}");
                return Err(e);
            }
        },
    };

    promote_or_requeue(backend, active, prefilling, chunk, &artifacts, ledger)
}

fn launch_overlap_step(
    backend: &mut Qwen35Backend,
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
        Qwen35Backend::Single(single) => single.launch_async_prefill(&mut chunk),
        Qwen35Backend::Tp(_) => unreachable!("Qwen3.5 TP cannot launch async prefill"),
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
            fail_ids(chunk.ids, ledger, &err.to_string());
        }
    }
    decode_step_with_seed(backend, active, decode_seed, ledger)
}

fn finish_async_prefill(
    backend: &mut Qwen35Backend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    inflight: InflightPrefill,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let InflightPrefill {
        chunk,
        output,
        sample_seed,
    } = inflight;
    let logits = output.into_logits();
    let Qwen35Backend::Single(single) = backend else {
        unreachable!("Qwen3.5 TP cannot finish async prefill");
    };
    let (tokens, logprobs) = match single.sample_prefill_logits(&chunk.reqs, &logits, sample_seed) {
        Ok(result) => result,
        Err(err) => {
            warn!("async prefill sampling failed: {err}");
            fail_ids(chunk.ids, ledger, &err.to_string());
            return Ok(());
        }
    };
    let artifacts = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(single, active, prefilling, chunk, &artifacts, ledger)
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

fn unified_step_sched(
    backend: &mut Qwen35Backend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let mut chunk = ScheduledChunk::from(scheduled);
    if matches!(backend, Qwen35Backend::Tp(_)) {
        let decode_sample_seed = rand::RngExt::random(rng);
        let prefill_sample_seed = rand::RngExt::random(rng);
        let result = {
            let Qwen35Backend::Tp(tp) = backend else {
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

    let Qwen35Backend::Single(backend) = backend else {
        unreachable!()
    };
    let result = backend.unified_step(&mut chunk, active);
    let output = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("unified step failed: {e}");
            let message = e.to_string();
            fail_active(backend, active, ledger, &message)?;
            fail_ids(chunk.ids, ledger, &message);
            return Ok(());
        }
    };
    let decode_seed = rand::RngExt::random(rng);
    let prefill_seed = rand::RngExt::random(rng);

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
                fail_ids(chunk.ids, ledger, &e.to_string());
                return Ok(());
            }
        };
    let prefill = PrefillStepArtifacts::Single { tokens, logprobs };
    promote_or_requeue(backend, active, prefilling, chunk, &prefill, ledger)
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    backend: &mut Qwen35Backend,
    active: &mut Vec<ActiveRequest35>,
    rng: &mut StdRng,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let first_seed = rand::RngExt::random(rng);
    let sample_seed = if matches!(backend, Qwen35Backend::Single(_)) {
        rand::RngExt::random(rng)
    } else {
        first_seed
    };
    decode_step_with_seed(backend, active, sample_seed, ledger)
}

fn decode_step_with_seed(
    backend: &mut Qwen35Backend,
    active: &mut Vec<ActiveRequest35>,
    sample_seed: u64,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let (tokens, logprobs_vec) = match backend {
        Qwen35Backend::Single(single) => {
            if let Err(e) = single.decode_graph(active) {
                warn!("batch_decode_graph error: {e}");
                fail_active(single, active, ledger, &e.to_string())?;
                return Ok(());
            }
            match single.sample_decode_logits(active, sample_seed) {
                Ok(v) => v,
                Err(e) => {
                    warn!("decode sampling/logprobs error: {e}");
                    fail_active(single, active, ledger, &e.to_string())?;
                    return Ok(());
                }
            }
        }
        Qwen35Backend::Tp(tp) => match tp.execute_decode(active, sample_seed) {
            Ok(v) => split_decode_artifacts(&v),
            Err(e) => {
                warn!("TP eager decode error: {e}");
                return Err(e);
            }
        },
    };

    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec, ledger)
}

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
            fail_active(backend, active, ledger, &e.to_string())?;
            return Ok(());
        }
    };

    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec, ledger)
}

enum Retirement {
    Stop,
    Length {
        token: u32,
        logprob: Option<TokenLogprob>,
    },
    Aborted,
}

/// Dispatch sampled decode tokens. Stop tokens are not pushed (the bridge
/// appends EOS for usage). Drop backend state before the ledger terminal so
/// TP drop-then-finish is structurally drop-before-visible-terminal.
fn dispatch_decode_tokens(
    backend: &mut impl DecodeDispatchBackend,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
    ledger: &mut RequestLedger,
) -> Result<()> {
    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let token = tokens[i];
        let logprob = logprobs[i].clone();
        let req = &mut active[i];
        if ledger.is_aborted(req.id) {
            to_retire.push((i, Retirement::Aborted));
            continue;
        }
        req.generated_count += 1;

        let is_eos = !req.params.ignore_eos && backend.is_stop_token(token);
        let at_limit = req.generated_count >= req.max_tokens;

        if is_eos {
            debug!(
                "request finished: request_id={} prompt_tokens={} finish_reason={:?}",
                req.id,
                req.prompt_len,
                FinishReason::Stop
            );
            to_retire.push((i, Retirement::Stop));
        } else if at_limit {
            debug!(
                "request finished: request_id={} prompt_tokens={} finish_reason={:?}",
                req.id,
                req.prompt_len,
                FinishReason::Length
            );
            to_retire.push((i, Retirement::Length { token, logprob }));
        } else {
            req.last_token = token;
            ledger.push_tokens(req.id, &[token], &[logprob]);
        }
    }

    for (i, retirement) in to_retire.into_iter().rev() {
        match retirement {
            Retirement::Stop => {
                let request = backend.take_active_request(active, i);
                backend.drop_active_state(&request.backend_state)?;
                finish_or_retire(request.id, FinishReason::Stop, ledger);
            }
            Retirement::Length { token, logprob } => {
                let request = backend.take_active_request(active, i);
                backend.drop_active_state(&request.backend_state)?;
                if !ledger.is_aborted(request.id) {
                    ledger.push_tokens(request.id, &[token], &[logprob]);
                }
                finish_or_retire(request.id, FinishReason::Length, ledger);
            }
            Retirement::Aborted => {
                let request = backend.take_active_request(active, i);
                backend.drop_active_state(&request.backend_state)?;
                ledger.retire(request.id);
            }
        }
    }
    Ok(())
}

// ── Chunked-prefill helpers ────────────────────────────────────────────────

fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    prefill_budget: usize,
) -> Vec<PrefillingRequest35> {
    let remaining: Vec<usize> = prefilling
        .iter()
        .map(|p| p.request.prompt_tokens.len() - p.cursor)
        .collect();
    let chunks = plan_prefill_chunks(&remaining, prefill_budget);
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..chunks.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&chunks) {
        p.step_chunk = *chunk;
    }
    scheduled
}

/// For each request in the just-prefilled chunk: if its prompt is now exhausted,
/// sample its first token and move it into the decode batch; otherwise re-queue
/// it (with an advanced cursor) at the FRONT of `prefilling`.
fn promote_or_requeue(
    backend: &mut impl PrefillPromoteBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    chunk: ScheduledChunk,
    artifacts: &PrefillStepArtifacts,
    ledger: &mut RequestLedger,
) -> Result<()> {
    let ScheduledChunk {
        ids,
        reqs,
        backend_state,
        ends,
        ..
    } = chunk;
    let mut still_prefilling: Vec<PrefillingRequest35> = Vec::new();
    let backend_states = split_scheduled_backend_state(backend_state);
    let mut entries: VecDeque<_> = ids
        .into_iter()
        .zip(reqs)
        .zip(backend_states)
        .zip(ends)
        .enumerate()
        .map(|(i, (((id, request), backend_state), end))| (i, id, request, backend_state, end))
        .collect();

    while let Some((i, id, request, backend_state, end)) = entries.pop_front() {
        if ledger.is_aborted(id) {
            let expectation = prefill_drop_expectation(end);
            backend.drop_prefill_state(&backend_state, expectation)?;
            ledger.retire(id);
            continue;
        }

        if end < request.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                id,
                request,
                backend_state,
                cursor: end,
                step_chunk: 0,
            });
            continue;
        }

        let prompt_len = request.prompt_tokens.len();
        let artifact = artifacts.final_artifact(i);
        let first_token = artifact.token;
        let logprob = artifact.logprob;

        if !request.params.ignore_eos && backend.is_stop_token(first_token) {
            debug!(
                "request finished: request_id={} prompt_tokens={} finish_reason={:?}",
                id,
                prompt_len,
                FinishReason::Stop
            );
            backend.drop_prefill_state(&backend_state, DropExpectation::MustExist)?;
            finish_or_retire(id, FinishReason::Stop, ledger);
            continue;
        }

        if request.max_tokens <= 1 {
            debug!(
                "request finished: request_id={} prompt_tokens={} finish_reason={:?}",
                id,
                prompt_len,
                FinishReason::Length
            );
            backend.drop_prefill_state(&backend_state, DropExpectation::MustExist)?;
            ledger.push_tokens(id, &[first_token], &[logprob]);
            finish_or_retire(id, FinishReason::Length, ledger);
            continue;
        }

        let active_backend_state = backend.promote_prefill_state(active.len(), backend_state);
        ledger.push_tokens(id, &[first_token], &[logprob]);
        active.push(ActiveRequest35 {
            id,
            backend_state: active_backend_state,
            last_token: first_token,
            generated_count: 1,
            max_tokens: request.max_tokens,
            prompt_len,
            params: request.params,
            logprobs: request.logprobs,
        });
    }

    prefilling.splice(0..0, still_prefilling);
    Ok(())
}
