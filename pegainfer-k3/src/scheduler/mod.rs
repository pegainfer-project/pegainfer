//! How K3 plugs into the step-batched engine contract.
//!
//! [`K3Scheduler`] implements [`Scheduler`]: the contract's driver polls
//! `intake` / `step` / `load`, and everything model-side sits behind
//! [`StepExecutor`]. This module owns the two contract-facing jobs:
//!
//! - the **handle registry**: each request's typestate handle
//!   ([`IntakeTicket`] until admission, [`ActiveRequest`] after), keyed by the
//!   contract's [`RequestId`], plus every emitter call that moves one along;
//! - **engine assembly**: wrapping executors in schedulers and returning the
//!   [`Engine`] bundle a model line hands back from `launch`.
//!
//! Scope, deliberately: no drafter and no prefix cache (K3's KDA recurrent
//! state is not reconstructible from tokens, so a prefix hit would be a lie —
//! `set_cached_tokens` is never reported). Cancellation is reactive: the
//! frontend flips a request's abort flag and the scheduler retires it, in
//! silence, on its next touch.

mod executor;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::collections::VecDeque;

use anyhow::Result;
use pegainfer_frontend::engine::ActiveRequest;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::IntakeTicket;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::LoadSnapshot;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::StepEmitter;
use pegainfer_frontend::engine::spawn_scheduler;

pub use self::executor::DecodeSlot;
pub use self::executor::SlotId;
pub use self::executor::StepExecutor;
pub use self::executor::UNWIRED_MESSAGE;
pub use self::executor::UnwiredExecutor;

// ── Engine assembly ─────────────────────────────────────────────────────

/// Scheduler facts that come from the model line rather than the executor.
#[derive(Clone, Debug, Default)]
pub struct K3SchedulerConfig {
    /// Token ids that end a stream with [`FinishReason::Stop`]. Requests that
    /// set `ignore_eos` opt out.
    pub eos_token_ids: Vec<u32>,
    /// KV pool capacity to advertise, or `None` from an engine that does not
    /// own a pool yet. Injected rather than asked of the executor so the
    /// number the frontend plans against is decided in one place.
    pub kv_capacity: Option<KvCapacity>,
}

/// Wrap ready executors in schedulers and hand the whole thing to the
/// contract: one scheduler thread per executor, required metadata filled in.
///
/// The partition count is the caller's decision and must equal the count its
/// `serve_plan` already promised the frontend — the frontend registers one
/// engine identity per partition while the engine is still loading.
pub fn start_with_executors<E>(executors: Vec<E>, config: &K3SchedulerConfig) -> Engine
where
    E: StepExecutor + 'static,
{
    let schedulers = executors
        .into_iter()
        .enumerate()
        .map(|(rank, executor)| {
            spawn_scheduler(
                &format!("k3-scheduler-{rank}"),
                K3Scheduler::new(executor, config.clone()),
            )
        })
        .collect();
    Engine {
        schedulers,
        info: EngineInfo {
            kv_capacity: config.kv_capacity,
            // No servable ceiling of K3's own yet: the protocol stack keeps
            // its max-length validation at the model context window.
            servable_len: None,
        },
        // K3 serves no adapters.
        lora: None,
    }
}

/// `ep_size` scheduler partitions over placeholder executors: every request is
/// admitted and then failed with [`UNWIRED_MESSAGE`], so the serving path is
/// real end to end and no client is answered with invented tokens. Serving now
/// goes through the GPU executor instead; what is left here is the vehicle for
/// exercising the request protocol on a box with no GPU and no weights.
#[must_use]
pub fn launch_unwired(ep_size: usize, eos_token_ids: Vec<u32>) -> Engine {
    start_with_executors(
        vec![UnwiredExecutor; ep_size],
        &K3SchedulerConfig {
            eos_token_ids,
            kv_capacity: None,
        },
    )
}

// ── The Scheduler implementation ────────────────────────────────────────

/// Answer a request that just reached its end: silence if the frontend
/// abandoned it since the scheduler last looked, the finish otherwise. Abort
/// can land at any moment, so the check belongs on every finish path.
fn finish_or_retire(active: ActiveRequest, reason: FinishReason, emitter: &mut StepEmitter) {
    if active.is_aborted() {
        emitter.retire(active);
    } else {
        emitter.finish(active, reason);
    }
}

/// One request's contract handle, in whichever lifecycle state it holds.
enum HandleSlot {
    /// Submitted, not yet admitted: reject/retire consume the ticket.
    Queued(IntakeTicket),
    /// Admitted: token pushes go through it; finish/fail/retire consume it.
    Streaming(ActiveRequest),
}

/// A submitted request waiting for a slot.
struct PendingRequest {
    id: RequestId,
    request: Request,
}

/// An admitted request occupying an execution slot.
struct RunningRequest {
    id: RequestId,
    slot: SlotId,
    /// This request's most recent committed token — next step's input.
    last_token: u32,
    max_tokens: usize,
    ignore_eos: bool,
}

pub struct K3Scheduler<E: StepExecutor> {
    executor: E,
    eos_token_ids: Vec<u32>,
    kv_capacity: Option<KvCapacity>,
    /// Contract handle per live request. Entries leave exactly at terminal
    /// transitions, so a request is in here iff the scheduler still owes it
    /// an answer.
    handles: HashMap<RequestId, HandleSlot>,
    /// Requests waiting for a slot, in submission order. A full batch is a
    /// wait, not a verdict — only permanently unservable requests are refused
    /// (see [`K3Scheduler::admission_refusal`]).
    queued: VecDeque<PendingRequest>,
    running: Vec<RunningRequest>,
    /// Slots not currently held by a running request.
    free_slots: Vec<SlotId>,
}

impl<E: StepExecutor> K3Scheduler<E> {
    pub fn new(executor: E, config: K3SchedulerConfig) -> Self {
        let free_slots = (0..executor.max_batch()).rev().collect();
        Self {
            executor,
            eos_token_ids: config.eos_token_ids,
            kv_capacity: config.kv_capacity,
            handles: HashMap::new(),
            queued: VecDeque::new(),
            running: Vec::new(),
            free_slots,
        }
    }

    /// Take a streaming handle out of the registry for a terminal transition.
    fn take_streaming(&mut self, id: RequestId) -> ActiveRequest {
        match self.handles.remove(&id) {
            Some(HandleSlot::Streaming(handle)) => handle,
            _ => unreachable!("running request {id} must hold a streaming handle"),
        }
    }

    /// Hand a slot back: the executor drops its state, the scheduler its
    /// reservation. Every terminal path goes through here.
    fn release_slot(&mut self, slot: SlotId) {
        self.executor.release(slot);
        self.free_slots.push(slot);
    }

    /// The refusal this request earns before it ever runs, if any. Only
    /// permanent misfits are refused — a full batch is a wait, not a verdict.
    fn admission_refusal(&self, request: &Request) -> Option<RejectReason> {
        let prompt_tokens = request.prompt_tokens.len();
        let limit = self.executor.max_context_tokens();
        (prompt_tokens.saturating_add(request.max_tokens) > limit).then_some(
            RejectReason::ContextLength {
                prompt_tokens,
                max_tokens: request.max_tokens,
                limit,
            },
        )
    }

    /// Whether `token` ends this request's stream by end-of-sequence.
    fn is_stop_token(&self, token: u32, ignore_eos: bool) -> bool {
        !ignore_eos && self.eos_token_ids.contains(&token)
    }

    /// Fill free slots from the queue: retire what the frontend abandoned,
    /// refuse what can never fit, prefill the rest.
    fn admit_queued(&mut self, emitter: &mut StepEmitter) {
        while !self.free_slots.is_empty() {
            let Some(pending) = self.queued.pop_front() else {
                break;
            };
            let HandleSlot::Queued(ticket) = self
                .handles
                .remove(&pending.id)
                .expect("queued request holds its ticket until admission")
            else {
                unreachable!("queued request {} must hold a ticket", pending.id);
            };
            if ticket.is_aborted() {
                emitter.retire_ticket(ticket);
                continue;
            }
            if let Some(reason) = self.admission_refusal(&pending.request) {
                emitter.reject(ticket, reason);
                continue;
            }
            let mut active = emitter.admit(ticket);
            if pending.request.max_tokens == 0 {
                // Nothing to generate: answer without occupying a slot.
                finish_or_retire(active, FinishReason::Length, emitter);
                continue;
            }
            let slot = self
                .free_slots
                .pop()
                .expect("loop runs only while a slot is free");
            let first = self.executor.prefill(
                slot,
                &pending.request.prompt_tokens,
                &pending.request.params,
            );
            let first = match first {
                Ok(token) => token,
                Err(error) => {
                    // A prefill failure is this request's problem, not the
                    // engine's: answer it and keep serving.
                    self.release_slot(slot);
                    emitter.fail(active, format!("{error:#}"));
                    continue;
                }
            };
            let state = RunningRequest {
                id: pending.id,
                slot,
                last_token: first,
                max_tokens: pending.request.max_tokens,
                ignore_eos: pending.request.params.ignore_eos,
            };
            if self.is_stop_token(first, state.ignore_eos) {
                // The stop token itself is not part of the completion.
                self.release_slot(slot);
                finish_or_retire(active, FinishReason::Stop, emitter);
                continue;
            }
            emitter.push_tokens(&mut active, &[first], &[]);
            if active.completion_tokens() >= state.max_tokens {
                self.release_slot(slot);
                finish_or_retire(active, FinishReason::Length, emitter);
                continue;
            }
            self.handles
                .insert(pending.id, HandleSlot::Streaming(active));
            self.running.push(state);
        }
    }

    /// One decode step over every running request, minus the ones the
    /// frontend abandoned since the last step.
    fn decode_running(&mut self, emitter: &mut StepEmitter) {
        self.retire_aborted(emitter);
        let batch: Vec<DecodeSlot> = self
            .running
            .iter()
            .map(|state| DecodeSlot {
                slot: state.slot,
                last_token: state.last_token,
            })
            .collect();
        // The step is unconditional: an empty batch still reaches the
        // executor, which is what keeps EP ranks free-running — a rank with
        // nothing to serve must still launch the step's fixed per-layer MoE
        // kernels (padding rows in place of live ones), or the device-side
        // barriers inside them would pair against a peer's wrong step.
        // Single-rank executors simply return an empty token list without
        // touching the device.
        let tokens = match self.executor.decode(&batch) {
            Ok(tokens) => tokens,
            Err(error) => {
                // The step touched the whole running batch, so the whole
                // batch is what dies. The scheduler stays up.
                self.fail_running(&format!("{error:#}"), emitter);
                return;
            }
        };
        assert_eq!(
            tokens.len(),
            batch.len(),
            "executor returned {} tokens for a batch of {}",
            tokens.len(),
            batch.len()
        );
        let mut still_running = Vec::with_capacity(self.running.len());
        for (mut state, token) in std::mem::take(&mut self.running).into_iter().zip(tokens) {
            state.last_token = token;
            if self.is_stop_token(token, state.ignore_eos) {
                let active = self.take_streaming(state.id);
                self.release_slot(state.slot);
                finish_or_retire(active, FinishReason::Stop, emitter);
                continue;
            }
            let Some(HandleSlot::Streaming(active)) = self.handles.get_mut(&state.id) else {
                unreachable!("running request {} must hold a streaming handle", state.id);
            };
            emitter.push_tokens(active, &[token], &[]);
            if active.completion_tokens() >= state.max_tokens {
                let active = self.take_streaming(state.id);
                self.release_slot(state.slot);
                finish_or_retire(active, FinishReason::Length, emitter);
            } else {
                still_running.push(state);
            }
        }
        self.running = still_running;
    }

    /// Drop every running request the frontend gave up on. Silent on the
    /// wire — the frontend already dropped its state for these ids.
    fn retire_aborted(&mut self, emitter: &mut StepEmitter) {
        let aborted: Vec<usize> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, state)| match self.handles.get(&state.id) {
                Some(HandleSlot::Streaming(handle)) => handle.is_aborted(),
                _ => unreachable!("running request {} must hold a streaming handle", state.id),
            })
            .map(|(index, _)| index)
            .collect();
        for index in aborted.into_iter().rev() {
            let state = self.running.swap_remove(index);
            let handle = self.take_streaming(state.id);
            emitter.retire(handle);
            self.release_slot(state.slot);
        }
    }

    /// Fail every running request with one message and free their slots.
    fn fail_running(&mut self, message: &str, emitter: &mut StepEmitter) {
        for state in std::mem::take(&mut self.running) {
            let handle = self.take_streaming(state.id);
            emitter.fail(handle, message);
            self.release_slot(state.slot);
        }
    }
}

impl<E: StepExecutor> Scheduler for K3Scheduler<E> {
    fn intake(&mut self, mut ticket: IntakeTicket) {
        // Ownership transfer only; every verdict is emitted from `step`.
        // Tickets already aborted at intake ride the normal path — admission
        // re-checks the flag and retires them.
        let id = ticket.id();
        let request = ticket
            .take_request()
            .expect("intake receives tickets with their payload");
        self.handles.insert(id, HandleSlot::Queued(ticket));
        self.queued.push_back(PendingRequest { id, request });
    }

    fn step(&mut self, emitter: &mut StepEmitter) -> Result<()> {
        self.admit_queued(emitter);
        self.decode_running(emitter);
        // No `Err` path yet: prefill and decode failures are per-request and
        // absorbed above. `Err` here would mean the engine is beyond use.
        Ok(())
    }

    fn load(&self) -> LoadSnapshot {
        LoadSnapshot {
            // K3 owns no KV pool of its own yet, so occupancy is honestly
            // zero; the advertised total is whatever the line injected.
            kv_used_blocks: 0,
            kv_total_blocks: self
                .kv_capacity
                .map_or(0, |capacity| capacity.total_blocks as u64),
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
        }
    }
}
