//! How Qwen3 plugs into the engine contract — the whole adaptation, in one
//! file.
//!
//! [`Qwen3Scheduler`] implements [`pegainfer_frontend::engine::Scheduler`]:
//! the contract's driver polls `submit`/`step`/`metrics`; everything
//! south of here ([`crate::scheduler`], the executor) deals in the contract's
//! [`RequestId`] and pure effect data. This file owns the only two
//! contract-facing responsibilities:
//!
//! - the **ledger writes**: translating each step's effects into verdicts and
//!   tokens recorded on the [`RequestLedger`] by id;
//! - **engine assembly**: building the executor and returning the
//!   [`Engine`] bundle from `launch`.
//!
//! Cancellation is reactive: the frontend flips a request's abort flag and
//! this adapter retires it on its next touch — the scheduler mechanics never
//! see aborts. Finishes ride the committed step unless the executor withholds
//! them ([`ModelExecutor::withholds_finishes`]): a P/D prefill executor takes
//! them as [`RequestLedger::defer_finish`] tokens and delivers each request's
//! final record once its KV saves are peer-visible.

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Result;
use log::info;
use log::warn;
use pegainfer_frontend::engine::DeferredFinish;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::LoraClient;
use pegainfer_frontend::engine::LoraControl;
use pegainfer_frontend::engine::LoraControlReceiver;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::Qwen3LoraOptions;
use crate::Qwen3OffloadOptions;
use crate::executor::ModelExecutor;
use crate::executor::Qwen3Executor;
use crate::executor::RequestId;
use crate::scheduler::ActiveRequestState;
use crate::scheduler::PendingRequest;
use crate::scheduler::RejectReason as AdmissionReject;
use crate::scheduler::admit_deferred_requests;
use crate::scheduler::admitted_future_blocks;
use crate::scheduler::block_on_loading;
use crate::scheduler::effects::DecodeEffect;
use crate::scheduler::effects::PendingEffect;
use crate::scheduler::effects::StepEffects;
use crate::scheduler::failure_target_ids;
use crate::scheduler::max_request_tokens;
use crate::scheduler::offer_prefetch;
use crate::scheduler::phase_trace::PhaseTracker;
use crate::scheduler::plan::ExecutionArtifacts;
use crate::scheduler::plan::ExecutionPlan;
use crate::scheduler::plan::execute_plan;
use crate::scheduler::reclaim_ready_prefetch;
use crate::scheduler::reject_unknown_lora_requests;
use crate::scheduler::release_rejected;
use crate::scheduler::resolve::resolve_step;
use crate::scheduler::runtime_plan;
use crate::scheduler::servable_len;
use crate::scheduler::take_prefill_chunks;
use crate::weights::Qwen3MemoryOptions;

// ── Engine assembly ─────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn start_qwen3(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: crate::DecodeOverlap,
    dflash_draft_model_path: Option<&str>,
    dump_graph_png: Option<&Path>,
) -> Result<Engine> {
    let mut executor = Qwen3Executor::from_runtime_with_decode_environment(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        Qwen3LoraOptions::default(),
        offload_options,
        max_prefill_tokens,
        dflash_draft_model_path,
        memory_options,
        decode_overlap,
    )?;
    if let Some(path) = dump_graph_png {
        let summary = executor.dump_decode_graph_png(path)?;
        info!(
            "Qwen3 decode CUDA Graph exported: nodes={}, kernels={}, edges={}, dot={}, png={}",
            summary.nodes,
            summary.kernels,
            summary.edges,
            summary.dot_path.display(),
            summary.png_path.display()
        );
    }
    executor.set_no_prefix_cache(no_prefix_cache);
    executor.enable_decode_overlap(decode_overlap)?;
    // Speculative decoding loads its draft model after the target is up (the
    // draft is built against the target's embeddings/head) and forces the
    // prefix cache off, so it must follow set_no_prefix_cache. Its GPU footprint
    // was already reserved during profiling from the draft path passed above.
    if let Some(draft_path) = dflash_draft_model_path {
        executor.load_dflash_draft_model(draft_path)?;
    }

    Ok(start_with_executor(
        executor,
        seed,
        max_prefill_tokens,
        false,
    ))
}

pub(crate) fn start_qwen3_with_lora_control(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
    lora_options: Qwen3LoraOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
) -> Result<Engine> {
    let mut executor = Qwen3Executor::from_runtime_with_lora_options(
        model_path,
        enable_cuda_graph,
        device_ordinals,
        lora_options,
        offload_options,
        max_prefill_tokens,
        None,
        memory_options,
    )?;
    executor.set_no_prefix_cache(no_prefix_cache);
    Ok(start_with_executor(
        executor,
        seed,
        max_prefill_tokens,
        true,
    ))
}

/// Wrap a ready executor in the scheduler and hand the whole thing to the
/// contract: one scheduler, one driver thread, required metadata filled in.
pub(crate) fn start_with_executor<E>(
    executor: E,
    seed: u64,
    max_prefill_tokens: usize,
    lora_control: bool,
) -> Engine
where
    E: ModelExecutor + 'static,
{
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let servable = servable_len(
        executor.max_context_tokens(),
        executor.max_request_blocks(),
        executor.block_size(),
    );
    // Executor just built: the only committed block is the leaked CUDA-graph
    // padding slot, so available_blocks() is total − 1. Conservative by one
    // block, which is the right side to err on for a capacity ceiling.
    let kv_total = executor.available_blocks() as u64;
    let kv_capacity = KvCapacity {
        total_blocks: kv_total as usize,
        block_size: executor.block_size(),
    };
    // The LoRA channel is qwen3's private wiring, minted before the thread
    // swallows the scheduler; the contract only carries the client.
    let (lora, lora_rx) = if lora_control {
        let (client, rx) = LoraClient::channel();
        (Some(client), Some(rx))
    } else {
        (None, None)
    };
    let scheduler = Qwen3Scheduler::new(executor, seed, max_prefill_tokens, kv_total, lora_rx);
    Engine {
        schedulers: vec![spawn_scheduler("qwen3-scheduler", scheduler)],
        info: EngineInfo {
            kv_capacity: Some(kv_capacity),
            servable_len: Some(servable),
        },
        lora,
    }
}

// ── The Scheduler implementation ────────────────────────────────────────

pub(crate) struct Qwen3Scheduler<E: ModelExecutor> {
    executor: E,
    rng: StdRng,
    max_prefill_tokens: usize,
    kv_total: u64,
    /// Host-side phase tracer (queue/prefill/decode spans). No-op when
    /// tracing is off — requests then carry no trace parent.
    tracker: PhaseTracker,
    /// Requests that could not be admitted yet (KV budget pressure).
    deferred: Vec<PendingRequest>,
    /// Requests parked while their async CPU-tier KV prefetch loads.
    loading: Vec<PendingRequest>,
    /// Admitted requests whose prompts are not fully prefilled yet (chunked
    /// prefill). FIFO by request id; each step takes chunks off the front.
    prefilling: Vec<PendingRequest>,
    /// An in-flight request being decoded.
    active: Vec<ActiveRequestState>,
    /// Decode-overlap async prefill: pending requests whose prefill is
    /// in-flight on the prefill overlap stream.
    inflight_prefill_pending: Option<Vec<PendingRequest>>,
    /// The engine's private LoRA channel, minted before spawn; `None` when
    /// this engine serves no adapter control. Drained at the top of every
    /// step into `pending_control`.
    lora_rx: Option<LoraControlReceiver>,
    /// LoRA commands wait here until the scheduler drains idle; generation
    /// submitted behind a pending command waits in `post_control_deferred` so
    /// it cannot run against an adapter set the command was about to change.
    pending_control: VecDeque<LoraControl>,
    post_control_deferred: Vec<PendingRequest>,
}

impl<E: ModelExecutor> Qwen3Scheduler<E> {
    pub(crate) fn new(
        executor: E,
        seed: u64,
        max_prefill_tokens: usize,
        kv_total: u64,
        lora_rx: Option<LoraControlReceiver>,
    ) -> Self {
        Self {
            executor,
            rng: StdRng::seed_from_u64(seed),
            max_prefill_tokens,
            kv_total,
            tracker: PhaseTracker::default(),
            deferred: Vec::new(),
            loading: Vec::new(),
            prefilling: Vec::new(),
            active: Vec::new(),
            inflight_prefill_pending: None,
            lora_rx,
            pending_control: VecDeque::new(),
            post_control_deferred: Vec::new(),
        }
    }

    fn idle(&self) -> bool {
        self.active.is_empty()
            && self.deferred.is_empty()
            && self.prefilling.is_empty()
            && self.inflight_prefill_pending.is_none()
    }

    /// Admit the request on the ledger. `false` (and full cleanup) when the
    /// frontend aborted it while it waited.
    fn admit_or_retire(&mut self, req: &PendingRequest, ledger: &mut RequestLedger) -> bool {
        if ledger.is_aborted(req.request_id) {
            ledger.retire(req.request_id);
            release_rejected(&mut self.executor, &mut self.tracker, req);
            return false;
        }
        ledger.admit(req.request_id);
        true
    }

    fn reject_pending(
        &mut self,
        req: &PendingRequest,
        reason: RejectReason,
        ledger: &mut RequestLedger,
    ) {
        ledger.reject(req.request_id, reason);
        release_rejected(&mut self.executor, &mut self.tracker, req);
    }

    /// Retire one request whose frontend aborted it: silent on the wire,
    /// full cleanup on the engine side.
    fn retire_aborted(&mut self, request_id: RequestId, ledger: &mut RequestLedger) {
        ledger.retire(request_id);
        self.tracker.finish(request_id);
        let _ = self.executor.drop_request(request_id);
    }

    /// Fail the batch a step execution error touched, then clear the running
    /// set — the same requests the plan was about to advance.
    fn fail_touched_requests(
        &mut self,
        targets: Vec<RequestId>,
        message: &str,
        ledger: &mut RequestLedger,
    ) {
        for request_id in targets {
            if ledger.is_active(request_id) {
                ledger.fail(request_id, message);
            }
            // Close the request's open phase span; an execution error is a
            // termination path like any other finish.
            self.tracker.finish(request_id);
            if let Err(error) = self.executor.drop_request(request_id) {
                warn!(
                    "failed to drop request state after execution error for {request_id:?}: {error}"
                );
            }
        }
        self.active.clear();
    }

    /// Translate one step's pure effects into ledger writes, executor
    /// bookkeeping, and state transitions. The old per-token send-failure
    /// signal is replaced by explicit abort probes on the ledger.
    fn apply_effects(&mut self, effects: StepEffects, ledger: &mut RequestLedger) {
        // Finishes are collected and resolved at the end of the step. A
        // withholding executor (P/D prefill: `Finished` may leave only once
        // this step's KV saves are peer-visible) takes them as
        // [`DeferredFinish`] tokens and delivers off the scheduler thread —
        // each carries the request's whole buffered record, so late delivery
        // cannot reorder. Everyone else finishes through the ledger, so the
        // terminal rides the committed step, which the driver ships after
        // publishing metrics — the finishing batch's send-time stats then
        // read the drained occupancy instead of racing the publish.
        let mut finishes: Vec<(RequestId, FinishReason)> = Vec::new();

        for cached in effects.cached {
            if ledger.is_active(cached.request_id) {
                ledger.set_cached_tokens(cached.request_id, cached.cached_tokens);
            }
        }

        for echo in effects.prompt_echoes {
            if ledger.is_active(echo.request_id) {
                ledger.echo_prompt(
                    echo.request_id,
                    PromptEcho {
                        ids: echo.ids,
                        logprobs: echo.logprobs,
                    },
                );
            }
        }

        let mut to_retire = Vec::new();
        for effect in effects.decode {
            match effect {
                DecodeEffect::Finish {
                    request_id,
                    finish_reason,
                } => {
                    let Some(index) = self
                        .active
                        .iter()
                        .position(|req| req.request_id == request_id)
                    else {
                        continue;
                    };
                    if ledger.is_active(request_id) {
                        if ledger.is_aborted(request_id) {
                            ledger.retire(request_id);
                        } else {
                            finishes.push((request_id, finish_reason));
                        }
                    }
                    self.tracker.finish(request_id);
                    let _ = self.executor.drop_request(request_id);
                    to_retire.push(index);
                }
                DecodeEffect::EmitAndFinish {
                    request_id,
                    token,
                    logprob,
                    finish_reason,
                } => {
                    let Some(index) = self
                        .active
                        .iter()
                        .position(|req| req.request_id == request_id)
                    else {
                        continue;
                    };
                    if ledger.is_active(request_id) {
                        if ledger.is_aborted(request_id) {
                            ledger.retire(request_id);
                        } else {
                            ledger.push_tokens(request_id, &[token], &[logprob]);
                            finishes.push((request_id, finish_reason));
                        }
                    }
                    self.tracker.finish(request_id);
                    let _ = self.executor.drop_request(request_id);
                    to_retire.push(index);
                }
                DecodeEffect::EmitAndContinue {
                    request_id,
                    token,
                    logprob,
                    completion_tokens,
                } => {
                    let Some(index) = self
                        .active
                        .iter()
                        .position(|req| req.request_id == request_id)
                    else {
                        continue;
                    };
                    if ledger.is_aborted(request_id) {
                        self.retire_aborted(request_id, ledger);
                        to_retire.push(index);
                    } else {
                        ledger.push_tokens(request_id, &[token], &[logprob]);
                        let req = &mut self.active[index];
                        req.last_token = token;
                        req.generated_count = completion_tokens;
                    }
                }
                DecodeEffect::EmitManyAndContinue {
                    request_id,
                    tokens,
                    completion_tokens,
                } => {
                    let Some(index) = self
                        .active
                        .iter()
                        .position(|req| req.request_id == request_id)
                    else {
                        continue;
                    };
                    if ledger.is_aborted(request_id) {
                        self.retire_aborted(request_id, ledger);
                        to_retire.push(index);
                    } else {
                        ledger.push_tokens(request_id, &tokens, &[]);
                        if let Some(&last) = tokens.last() {
                            let req = &mut self.active[index];
                            req.last_token = last;
                            req.generated_count = completion_tokens;
                        }
                    }
                }
                DecodeEffect::EmitManyAndFinish {
                    request_id,
                    tokens,
                    finish_reason,
                } => {
                    let Some(index) = self
                        .active
                        .iter()
                        .position(|req| req.request_id == request_id)
                    else {
                        continue;
                    };
                    if ledger.is_active(request_id) {
                        if ledger.is_aborted(request_id) {
                            ledger.retire(request_id);
                        } else {
                            ledger.push_tokens(request_id, &tokens, &[]);
                            finishes.push((request_id, finish_reason));
                        }
                    }
                    self.tracker.finish(request_id);
                    let _ = self.executor.drop_request(request_id);
                    to_retire.push(index);
                }
            }
        }
        to_retire.sort_unstable();
        to_retire.dedup();
        for &i in to_retire.iter().rev() {
            self.active.swap_remove(i);
        }

        // Requests that ran a non-final chunk this step came off the front of
        // the prefilling queue; splicing them back at the front (in step
        // order, which is request-id order) keeps the queue FIFO so chunked
        // prompts finish before newer arrivals start.
        let mut continued: Vec<PendingRequest> = Vec::new();
        for effect in effects.pending {
            match effect {
                PendingEffect::ContinuePrefill { req } => {
                    if ledger.is_aborted(req.request_id) {
                        let request_id = req.request_id;
                        self.retire_aborted(request_id, ledger);
                    } else {
                        continued.push(req);
                    }
                }
                PendingEffect::Finish {
                    request_id,
                    finish_reason,
                } => {
                    if ledger.is_active(request_id) {
                        if ledger.is_aborted(request_id) {
                            ledger.retire(request_id);
                        } else {
                            finishes.push((request_id, finish_reason));
                        }
                    }
                    self.tracker.finish(request_id);
                    let _ = self.executor.drop_request(request_id);
                }
                PendingEffect::EmitAndFinish {
                    request_id,
                    token,
                    logprob,
                    finish_reason,
                } => {
                    if ledger.is_active(request_id) {
                        if ledger.is_aborted(request_id) {
                            ledger.retire(request_id);
                        } else {
                            ledger.push_tokens(request_id, &[token], &[logprob]);
                            finishes.push((request_id, finish_reason));
                        }
                    }
                    self.tracker.finish(request_id);
                    let _ = self.executor.drop_request(request_id);
                }
                PendingEffect::Promote {
                    state,
                    first_token,
                    logprob,
                } => {
                    let request_id = state.request_id;
                    if ledger.is_aborted(request_id) {
                        self.retire_aborted(request_id, ledger);
                    } else {
                        ledger.push_tokens(request_id, &[first_token], &[logprob]);
                        self.tracker.enter_decode(request_id);
                        self.active.push(state);
                    }
                }
            }
        }
        self.prefilling.splice(0..0, continued);

        if !finishes.is_empty() {
            if self.executor.withholds_finishes() {
                let withheld: Vec<DeferredFinish> = finishes
                    .into_iter()
                    .map(|(request_id, reason)| ledger.defer_finish(request_id, reason))
                    .collect();
                self.executor.release_finished_events(withheld);
            } else {
                for (request_id, reason) in finishes {
                    ledger.finish(request_id, reason);
                }
            }
        }
    }

    fn drain_idle_control(&mut self) {
        while let Some(control) = self.pending_control.pop_front() {
            handle_control_request(&mut self.executor, control);
        }
    }
}

impl<E: ModelExecutor> Scheduler for Qwen3Scheduler<E> {
    fn submit(&mut self, request: QueuedRequest) {
        // Already-aborted requests ride the normal path: admission re-checks
        // the abort flag and retires them (`admit_or_retire`).
        let QueuedRequest { id, request } = request;
        self.tracker.enter_queue(id, request.trace_parent);
        let pending = PendingRequest::from_request(id, request);
        if self.pending_control.is_empty() {
            self.deferred.push(pending);
        } else {
            self.post_control_deferred.push(pending);
        }
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        // 0. Pull queued LoRA commands off the engine's private channel; they
        // apply only once the scheduler drains idle (`drain_idle_control`).
        if let Some(lora_rx) = &self.lora_rx {
            while let Ok(control) = lora_rx.try_recv() {
                self.pending_control.push_back(control);
            }
        }

        // 1. Poll in-flight async prefill (decode-overlap mode).
        if self.inflight_prefill_pending.is_some()
            && let Some(prefill_result) = self.executor.poll_async_prefill()
        {
            let pending = self
                .inflight_prefill_pending
                .take()
                .expect("checked in-flight prefill above");
            info!(
                "decode-overlap: async prefill completed ({} reqs)",
                pending.len()
            );
            let artifacts = ExecutionArtifacts::Prefill {
                pending,
                result: prefill_result,
            };
            let effects = resolve_step(&self.executor, &self.active, artifacts);
            self.apply_effects(effects, ledger);
        }

        // 2. Reclaim settled prefetches, then offer fresh requests to
        // prefetch. Control commands gate generation, so only offer once no
        // control is pending (a prefetch must not race ahead of an adapter
        // load it depends on).
        let reserve_floor = admitted_future_blocks(&self.executor, &self.active, &self.prefilling);
        reclaim_ready_prefetch(
            &mut self.executor,
            &mut self.deferred,
            &mut self.loading,
            reserve_floor,
        );
        if self.pending_control.is_empty() {
            offer_prefetch(
                &mut self.executor,
                &mut self.deferred,
                &mut self.loading,
                reserve_floor,
            );
        }

        // 3. Once idle, apply pending control commands before admitting newer
        // generation requests that arrived behind them.
        if self.idle() {
            self.drain_idle_control();
            if self.pending_control.is_empty() && !self.post_control_deferred.is_empty() {
                self.deferred.append(&mut self.post_control_deferred);
            }
        }

        // 4. Still nothing runnable → wait on an in-flight prefetch DMA (its
        // request prefills next), else hand control back to the polling
        // driver.
        if self.idle() {
            if !self.loading.is_empty() {
                let reserve_floor =
                    admitted_future_blocks(&self.executor, &self.active, &self.prefilling);
                block_on_loading(
                    &mut self.executor,
                    &mut self.deferred,
                    &mut self.loading,
                    reserve_floor,
                );
            }
            return Ok(());
        }

        // 5. Validate + admit; every verdict closes the request's queued
        //    ledger account.
        let lora_validation =
            reject_unknown_lora_requests(std::mem::take(&mut self.deferred), &self.executor);
        for rejected in &lora_validation.rejected {
            let reason = RejectReason::UnknownLoraAdapter {
                name: rejected
                    .lora_adapter
                    .clone()
                    .expect("only requests naming an adapter can fail LoRA validation"),
            };
            self.reject_pending(rejected, reason, ledger);
        }
        let admission = admit_deferred_requests(
            lora_validation.accepted,
            &self.active,
            &self.prefilling,
            self.executor.block_size(),
            self.executor.available_blocks(),
            self.executor.max_request_blocks(),
            self.executor.max_context_tokens(),
            self.executor.max_decode_batch_size(),
            self.max_prefill_tokens,
            |id| self.executor.prefetched_blocks(id),
        );
        for (rejected, reason) in &admission.rejected {
            self.reject_pending(rejected, contract_reject_reason(rejected, *reason), ledger);
        }
        self.deferred = admission.deferred;
        for req in admission.pending {
            if self.admit_or_retire(&req, ledger) {
                self.prefilling.push(req);
            }
        }

        // 6. Chunk selection. While an async prefill is in flight, only
        // decode runs.
        let pending = if self.inflight_prefill_pending.is_some() {
            Vec::new()
        } else {
            take_prefill_chunks(
                &mut self.prefilling,
                self.max_prefill_tokens,
                matches!(
                    numeric_policy(),
                    NumericPolicy::Pin | NumericPolicy::PerToken
                ),
            )
        };
        // These requests' prompt work is about to hit the GPU this step:
        // close their queue span, open prefill. Idempotent, so chunked
        // prefill across steps opens the prefill span once (on the first
        // chunk).
        for req in &pending {
            self.tracker.enter_prefill(req.request_id);
        }

        let Some(plan) = runtime_plan(&self.executor, &self.active, pending) else {
            return Ok(());
        };

        // 7. Decode-overlap path: when a Unified plan appears and overlap
        // streams are active (and no async prefill is already in-flight),
        // execute_unified internally uses SplitConcurrent which only syncs
        // decode and defers the prefill sync. The prefill result is polled at
        // the top of a later step.
        if self.executor.has_decode_overlap()
            && self.inflight_prefill_pending.is_none()
            && let ExecutionPlan::Unified { pending } = plan
        {
            let prefill_tokens: usize = pending.iter().map(|r| r.step_chunk).sum();
            info!(
                "decode-overlap: unified step with async prefill ({} reqs, ~{} tokens)",
                pending.len(),
                prefill_tokens
            );
            let pending_for_poll = pending.clone();
            let unified_plan = ExecutionPlan::Unified { pending };
            let targets = failure_target_ids(&self.active, &unified_plan);
            let artifacts = match execute_plan(
                &mut self.executor,
                &mut self.active,
                unified_plan,
                &mut self.rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    warn!("Execution step failed: {e}");
                    self.fail_touched_requests(targets, &e.to_string(), ledger);
                    return Ok(());
                }
            };
            // Only decode effects come out of the unified result (its prefill
            // request list is empty until the poll resolves).
            let effects = resolve_step(&self.executor, &self.active, artifacts);
            self.apply_effects(effects, ledger);
            self.inflight_prefill_pending = Some(pending_for_poll);
            return Ok(());
        }

        // 8. Execute, resolve, apply. A step failure kills the touched batch
        // but not the engine.
        let targets = failure_target_ids(&self.active, &plan);
        let artifacts =
            match execute_plan(&mut self.executor, &mut self.active, plan, &mut self.rng) {
                Ok(v) => v,
                Err(e) => {
                    warn!("Execution step failed: {e}");
                    self.fail_touched_requests(targets, &e.to_string(), ledger);
                    return Ok(());
                }
            };
        let effects = resolve_step(&self.executor, &self.active, artifacts);
        self.apply_effects(effects, ledger);
        Ok(())
    }

    /// Live KV occupancy between steps — the steady-state load a router
    /// wants, not a transient in-step peak. `num_waiting_reqs` folds every
    /// not-yet-running queue (KV-deferred, prefetch-loading, post-control)
    /// into one number.
    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self
                .kv_total
                .saturating_sub(self.executor.available_blocks() as u64),
            kv_total_blocks: self.kv_total,
            num_running_reqs: (self.active.len()
                + self.prefilling.len()
                + self.inflight_prefill_pending.as_ref().map_or(0, Vec::len))
                as u64,
            num_waiting_reqs: (self.deferred.len()
                + self.loading.len()
                + self.post_control_deferred.len()) as u64,
            spec_decode: self.executor.spec_decode_counters(),
        }
    }
}

/// Widen a mechanics-side admission verdict into the contract's typed reason:
/// the mechanics carry only what the decision needed, the request at hand
/// supplies the rest.
fn contract_reject_reason(req: &PendingRequest, reason: AdmissionReject) -> RejectReason {
    match reason {
        AdmissionReject::ContextLength { limit } => RejectReason::ContextLength {
            prompt_tokens: req.prompt_tokens.len(),
            max_tokens: req.max_tokens,
            limit,
        },
        AdmissionReject::EchoPrefillTokens { limit } => RejectReason::EchoPrefillTokens {
            prompt_tokens: req.prompt_tokens.len(),
            limit,
        },
        AdmissionReject::KvBudget => RejectReason::KvBudget {
            prompt_tokens: req.prompt_tokens.len(),
            worst_case_tokens: max_request_tokens(req),
        },
    }
}

fn handle_control_request(executor: &mut impl ModelExecutor, control: LoraControl) {
    match control {
        LoraControl::Load { request, reply } => {
            info!(
                "LoRA adapter load applied while scheduler is idle: name={}, path={}",
                request.lora_name,
                request.lora_path.display()
            );
            let _ = reply.send(
                executor
                    .load_lora_adapter(&request)
                    .map_err(|error| error.to_string()),
            );
        }
        LoraControl::Unload { request, reply } => {
            info!(
                "LoRA adapter unload applied while scheduler is idle: name={}",
                request.lora_name
            );
            let _ = reply.send(
                executor
                    .unload_lora_adapter(&request)
                    .map_err(|error| error.to_string()),
            );
        }
        LoraControl::List { reply } => {
            let _ = reply.send(executor.list_lora_adapters());
        }
    }
}
