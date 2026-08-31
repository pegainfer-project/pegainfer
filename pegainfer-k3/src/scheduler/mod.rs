//! How K3 plugs into the step-batched engine contract.
//!
//! [`K3Scheduler`] implements [`Scheduler`]: the contract's driver polls
//! `submit` / `step` / `metrics`, and everything model-side sits behind
//! [`StepExecutor`]. This module owns the two contract-facing jobs:
//!
//! - the **ledger writes**: every verdict and token for a request, recorded
//!   against the contract's [`RequestId`] on the step's [`RequestLedger`];
//! - **engine assembly**: wrapping executors in schedulers and returning the
//!   [`Engine`] bundle a model line hands back from `launch`.
//!
//! Scope, deliberately: no drafter and no prefix cache (K3's KDA recurrent
//! state is not reconstructible from tokens, so a prefix hit would be a lie —
//! `set_cached_tokens` is never reported). Cancellation is reactive: the
//! frontend flips a request's abort flag and the scheduler retires it, in
//! silence, on its next touch.

mod executor;
mod gang;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::spawn_scheduler;

pub use self::executor::DecodeSlot;
pub use self::executor::SlotId;
pub use self::executor::StepExecutor;
pub use self::gang::K3CpGang;
use crate::executor::cp::k3_cp_admits;

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
    /// Context-parallel prefill serving, when armed. `None` = every prompt
    /// prefills on its own partition.
    pub cp: Option<K3CpServing>,
}

/// The armed CP prefill lane: the gang handle plus the admission window that
/// decides which prompts are worth gang-prefilling.
#[derive(Clone, Debug)]
pub struct K3CpServing {
    pub gang: Arc<K3CpGang>,
    /// Prompts shorter than this prefill locally — below it the gang's
    /// coordination overhead outweighs the split.
    pub min_tokens: usize,
    /// The executors' prefill chunk cap. A CP segment must fit one chunk
    /// (M0: one chunk step per rank), so this bounds eligibility from above.
    pub chunk_tokens: usize,
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
                K3Scheduler::new(executor, rank, config.clone()),
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

// ── The Scheduler implementation ────────────────────────────────────────

/// Answer a request that just reached its end: silence if the frontend
/// abandoned it since the scheduler last looked, the finish otherwise. Abort
/// can land at any moment, so the check belongs on every finish path.
fn finish_or_retire(id: RequestId, reason: FinishReason, ledger: &mut RequestLedger) {
    if ledger.is_aborted(id) {
        ledger.retire(id);
    } else {
        ledger.finish(id, reason);
    }
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
    /// This scheduler's partition index — its identity in the CP gang.
    partition: usize,
    eos_token_ids: Vec<u32>,
    kv_capacity: Option<KvCapacity>,
    /// The CP prefill lane, when the engine armed one.
    cp: Option<K3CpServing>,
    /// Requests waiting for a slot, in submit order. A full batch is a
    /// wait, not a verdict — only permanently unservable requests are refused
    /// (see [`K3Scheduler::admission_refusal`]).
    queued: VecDeque<QueuedRequest>,
    running: Vec<RunningRequest>,
    /// Slots not currently held by a running request.
    free_slots: Vec<SlotId>,
}

impl<E: StepExecutor> K3Scheduler<E> {
    pub fn new(executor: E, partition: usize, config: K3SchedulerConfig) -> Self {
        if let Some(cp) = &config.cp {
            assert!(
                partition < cp.gang.size(),
                "partition {partition} outside the CP gang of {}",
                cp.gang.size()
            );
        }
        let free_slots = (0..executor.max_batch()).rev().collect();
        Self {
            executor,
            partition,
            eos_token_ids: config.eos_token_ids,
            kv_capacity: config.kv_capacity,
            cp: config.cp,
            queued: VecDeque::new(),
            running: Vec::new(),
            free_slots,
        }
    }

    /// The gang to CP-prefill `prompt_len` tokens with, if the lane is armed
    /// and the prompt sits in its window: long enough to be worth the gang,
    /// and admissible under the executors' chunk cap (M0).
    fn cp_gang_for(&self, prompt_len: usize) -> Option<Arc<K3CpGang>> {
        let cp = self.cp.as_ref()?;
        (prompt_len >= cp.min_tokens && k3_cp_admits(prompt_len, cp.gang.size(), cp.chunk_tokens))
            .then(|| cp.gang.clone())
    }

    /// Serve pending CP gang jobs posted by peer partitions. Runs at the top
    /// of every step so a posted gang assembles within one step time.
    fn serve_gang(&mut self) -> Result<()> {
        let Some(gang) = self.cp.as_ref().map(|cp| cp.gang.clone()) else {
            return Ok(());
        };
        gang.serve(self.partition, &mut self.executor)
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
    /// refuse what can never fit, prefill the rest. `Err` means a gang
    /// prefill failed — never a local one, which is the request's problem.
    fn admit_queued(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        while !self.free_slots.is_empty() {
            let Some(pending) = self.queued.pop_front() else {
                break;
            };
            let id = pending.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            if let Some(reason) = self.admission_refusal(&pending.request) {
                ledger.reject(id, reason);
                continue;
            }
            ledger.admit(id);
            if pending.request.max_tokens == 0 {
                // Nothing to generate: answer without occupying a slot.
                finish_or_retire(id, FinishReason::Length, ledger);
                continue;
            }
            let slot = self
                .free_slots
                .pop()
                .expect("loop runs only while a slot is free");
            let first = match self.cp_gang_for(pending.request.prompt_tokens.len()) {
                Some(gang) => {
                    match gang.post_and_run(
                        self.partition,
                        slot,
                        Arc::from(pending.request.prompt_tokens.as_slice()),
                        &mut self.executor,
                    ) {
                        Ok(token) => token,
                        Err(error) => {
                            // A gang failure is never per-request: this
                            // partition may have left its peers mid-protocol,
                            // so the engine is beyond use.
                            self.release_slot(slot);
                            ledger.fail(id, format!("{error:#}"));
                            return Err(error);
                        }
                    }
                }
                None => {
                    match self.executor.prefill(
                        slot,
                        &pending.request.prompt_tokens,
                        &pending.request.params,
                    ) {
                        Ok(token) => token,
                        Err(error) => {
                            // A local prefill failure is this request's
                            // problem, not the engine's: answer it and keep
                            // serving.
                            self.release_slot(slot);
                            ledger.fail(id, format!("{error:#}"));
                            continue;
                        }
                    }
                }
            };
            let state = RunningRequest {
                id,
                slot,
                last_token: first,
                max_tokens: pending.request.max_tokens,
                ignore_eos: pending.request.params.ignore_eos,
            };
            if self.is_stop_token(first, state.ignore_eos) {
                // The stop token itself is not part of the completion.
                self.release_slot(slot);
                finish_or_retire(id, FinishReason::Stop, ledger);
                continue;
            }
            ledger.push_tokens(id, &[first], &[]);
            if ledger.completion_tokens(id) >= state.max_tokens {
                self.release_slot(slot);
                finish_or_retire(id, FinishReason::Length, ledger);
                continue;
            }
            self.running.push(state);
        }
        Ok(())
    }

    /// One decode step over every running request, minus the ones the
    /// frontend abandoned since the last step.
    fn decode_running(&mut self, ledger: &mut RequestLedger) {
        self.retire_aborted(ledger);
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
        let token_lists = match self.executor.decode_many(&batch) {
            Ok(token_lists) => token_lists,
            Err(error) => {
                // The step touched the whole running batch, so the whole
                // batch is what dies. The scheduler stays up.
                self.fail_running(&format!("{error:#}"), ledger);
                return;
            }
        };
        assert_eq!(
            token_lists.len(),
            batch.len(),
            "executor returned {} token lists for a batch of {}",
            token_lists.len(),
            batch.len()
        );
        let mut still_running = Vec::with_capacity(self.running.len());
        for (mut state, committed) in std::mem::take(&mut self.running)
            .into_iter()
            .zip(token_lists)
        {
            assert!(
                !committed.is_empty(),
                "a round must commit at least one token"
            );
            // A speculative round commits several tokens at once; walk them
            // in order and stop at the first terminal one. Tokens past a
            // stop/length cut are computed-but-dead, like a rejected draft.
            let already = ledger.completion_tokens(state.id);
            let mut kept: Vec<u32> = Vec::with_capacity(committed.len());
            let mut finished = None;
            for &token in &committed {
                if self.is_stop_token(token, state.ignore_eos) {
                    // The stop token itself is not part of the completion.
                    finished = Some(FinishReason::Stop);
                    break;
                }
                kept.push(token);
                state.last_token = token;
                if already + kept.len() >= state.max_tokens {
                    finished = Some(FinishReason::Length);
                    break;
                }
            }
            if !kept.is_empty() {
                ledger.push_tokens(state.id, &kept, &[]);
            }
            match finished {
                Some(reason) => {
                    self.release_slot(state.slot);
                    finish_or_retire(state.id, reason, ledger);
                }
                None => still_running.push(state),
            }
        }
        self.running = still_running;
    }

    /// Drop every running request the frontend gave up on. Silent on the
    /// wire — the frontend already dropped its state for these ids.
    fn retire_aborted(&mut self, ledger: &mut RequestLedger) {
        let mut index = 0;
        while index < self.running.len() {
            if ledger.is_aborted(self.running[index].id) {
                let state = self.running.swap_remove(index);
                ledger.retire(state.id);
                self.release_slot(state.slot);
            } else {
                index += 1;
            }
        }
    }

    /// Fail every running request with one message and free their slots.
    fn fail_running(&mut self, message: &str, ledger: &mut RequestLedger) {
        for state in std::mem::take(&mut self.running) {
            ledger.fail(state.id, message);
            self.release_slot(state.slot);
        }
    }
}

impl<E: StepExecutor> Scheduler for K3Scheduler<E> {
    fn submit(&mut self, request: QueuedRequest) {
        // Ownership transfer only; every verdict is written to the ledger
        // from `step`. Requests already aborted when submitted ride the
        // normal path — admission re-checks the flag and retires them.
        self.queued.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        // Gang jobs first: a peer partition may be waiting at a CP prefill
        // rendezvous, and this step's decode would keep it pumping for a
        // whole extra step time. A serve error propagates: this partition can
        // no longer hold up its end of the gang, so the engine is beyond use.
        self.serve_gang()?;
        self.admit_queued(ledger)?;
        self.decode_running(ledger);
        // Local prefill and decode failures are per-request and absorbed
        // above; gang failures propagate the same way a serve error does.
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            // K3 owns no KV pool of its own yet, so occupancy is honestly
            // zero; the advertised total is whatever the line injected.
            kv_used_blocks: 0,
            kv_total_blocks: self
                .kv_capacity
                .map_or(0, |capacity| capacity.total_blocks as u64),
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
            ..SchedulerMetrics::default()
        }
    }
}
