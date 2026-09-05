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
pub mod whale;
pub mod whale_hub;

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
use pegainfer_frontend::sampler::SamplingParams;

pub use self::executor::DecodeSlot;
pub use self::executor::SlotId;
pub use self::executor::StepExecutor;
pub use self::gang::K3CpGang;
use self::whale::CommittedWhale;
use self::whale::GlobalRank;
use self::whale::WhaleDuty;
use self::whale::WhaleMember;
use self::whale::WhaleSeq;
use self::whale::WhaleToMember;
use self::whale::WhaleToSequencer;
pub use self::whale_hub::K3WhaleHub;
use crate::executor::cp::k3_cp_admits;
use crate::executor::cp::k3_whale_width;

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
    /// The fleet whale lane, when armed. Mutually exclusive with `cp`: the
    /// in-process gang and the fleet rendezvous are two coordination layers
    /// for the same superstep, and a deployment runs exactly one.
    pub whale: Option<K3WhaleServing>,
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

/// The armed whale lane: fleet-wide CP prefill through the whale rendezvous
/// ([`whale`]). One hub per process; each scheduler partition is one global
/// rank in the fleet's world.
#[derive(Clone, Debug)]
pub struct K3WhaleServing {
    /// The rendezvous transport. The process hosting global rank 0 also
    /// hosts the sequencer inside its hub.
    pub hub: K3WhaleHub,
    /// Total ranks in the fleet.
    pub world: usize,
    /// The global rank of this process's partition 0; partition `p` serves
    /// global rank `first_local + p`.
    pub first_local: GlobalRank,
    /// Prompts shorter than this prefill locally.
    pub min_tokens: usize,
    /// The executors' prefill chunk cap — bounds a whale segment from above
    /// (one superstep per rank).
    pub chunk_tokens: usize,
}

/// A whale request this poster sent to the sequencer and is waiting to hear
/// back about. Its slot is reserved the whole time, so the commit can never
/// arrive to a full batch. `id.raw()` rides the descriptor's `request` field,
/// pairing gathers and commits with this admission.
struct PostedWhale {
    id: RequestId,
    slot: SlotId,
    /// The sequence this whale got, learned from our own gather broadcast.
    /// `None` until the gather arrives; a cancel for an unknown sequence can
    /// only be the width refusal of this one outstanding post.
    seq: Option<WhaleSeq>,
    prompt: Arc<[u32]>,
    params: SamplingParams,
    max_tokens: usize,
}

/// One partition's live whale state: the protocol member plus the poster-side
/// bookkeeping. At most one whale post is outstanding per rank at a time —
/// whales are sparse, and one-at-a-time keeps the cancel pairing trivial.
struct WhaleLane {
    serving: K3WhaleServing,
    member: WhaleMember,
    rank: GlobalRank,
    /// The outstanding post, if any (awaiting commit or cancel).
    posted: Option<PostedWhale>,
    /// A cancelled post falling back to a local prefill, run at the next
    /// unrestricted launch.
    fallback: Option<PostedWhale>,
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
    /// The whale lane, when the engine armed one.
    whale: Option<WhaleLane>,
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
        assert!(
            config.cp.is_none() || config.whale.is_none(),
            "the in-process CP gang and the fleet whale lane are exclusive"
        );
        let whale = config.whale.map(|serving| {
            let rank = serving.first_local + partition;
            assert!(
                rank < serving.world,
                "partition {partition} maps to global rank {rank}, outside the {}-rank world",
                serving.world
            );
            WhaleLane {
                member: WhaleMember::new(rank),
                rank,
                posted: None,
                fallback: None,
                serving,
            }
        });
        let free_slots = (0..executor.max_batch()).rev().collect();
        Self {
            executor,
            partition,
            eos_token_ids: config.eos_token_ids,
            kv_capacity: config.kv_capacity,
            cp: config.cp,
            whale,
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

    /// Serve the whale lane at this launch boundary: drain the rendezvous,
    /// answer gathers, enter any whale committed for this exact launch, and
    /// say whether multi-launch admissions are allowed this step.
    ///
    /// `false` (an armed or committed whale is pending) restricts the step to
    /// single-launch work: the scheduler then only decodes, the launch count
    /// advances exactly one per step, and this member hits the committed
    /// launch dead on. That is the scheduler's half of the arming contract in
    /// [`whale`] — the member's replies stay exact floors on reachable
    /// launches because nothing multi-launch starts while one is pending.
    fn serve_whale(&mut self, ledger: &mut RequestLedger) -> Result<bool> {
        let Some(mut lane) = self.whale.take() else {
            return Ok(true);
        };
        let verdict = self.serve_whale_lane(&mut lane, ledger);
        self.whale = Some(lane);
        verdict
    }

    fn serve_whale_lane(
        &mut self,
        lane: &mut WhaleLane,
        ledger: &mut RequestLedger,
    ) -> Result<bool> {
        loop {
            for message in lane.serving.hub.drain(lane.rank) {
                Self::track_own_post(lane, &message);
                let count = self.executor.step_count();
                if let Some(reply) = lane.member.on_message(message, count)? {
                    lane.serving.hub.send(reply)?;
                }
            }
            match lane.member.at_launch(self.executor.step_count())? {
                // The superstep advanced the count; consult the next boundary
                // before leaving — a second whale may sit right behind it.
                WhaleDuty::Enter(whale) => self.enter_whale(lane, *whale, ledger)?,
                WhaleDuty::Free => {
                    if !lane.member.is_quiet() {
                        return Ok(false);
                    }
                    self.run_whale_fallback(lane, ledger);
                    return Ok(true);
                }
            }
        }
    }

    /// Pair an inbound rendezvous message with this rank's outstanding post,
    /// before the member consumes it. A gather for our own request records
    /// its sequence; a cancel routes the post to the local-prefill fallback —
    /// either by its recorded sequence, or, for a post that never gathered
    /// (unknown sequence, not armed here), as the sequencer's width refusal.
    fn track_own_post(lane: &mut WhaleLane, message: &WhaleToMember) {
        match message {
            WhaleToMember::Gather { descriptor } if descriptor.poster == lane.rank => {
                if let Some(posted) = lane.posted.as_mut()
                    && posted.id.raw() == descriptor.request
                {
                    posted.seq = Some(descriptor.seq);
                }
            }
            WhaleToMember::Cancel { seq } => {
                let refused = lane.posted.as_ref().is_some_and(|posted| {
                    posted.seq == Some(*seq)
                        || (posted.seq.is_none() && !lane.member.is_armed(*seq))
                });
                if refused {
                    lane.fallback = lane.posted.take();
                }
            }
            _ => {}
        }
    }

    /// Enter a committed whale's superstep. Every gang member calls the
    /// executor at this exact launch, unconditionally — a committed whale
    /// cannot be cancelled, and an unwanted result is dropped, not prevented.
    /// An executor error here is engine-fatal: this rank just left its gang
    /// mid-superstep.
    fn enter_whale(
        &mut self,
        lane: &mut WhaleLane,
        whale: CommittedWhale,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        let posted = if whale.descriptor.poster == lane.rank {
            let posted = lane.posted.take().ok_or_else(|| {
                anyhow::anyhow!(
                    "whale {} committed for this poster, but no post is outstanding",
                    whale.descriptor.seq
                )
            })?;
            anyhow::ensure!(
                posted.id.raw() == whale.descriptor.request,
                "whale {} answers request {}, but the outstanding post is {}",
                whale.descriptor.seq,
                whale.descriptor.request,
                posted.id.raw()
            );
            Some(posted)
        } else {
            None
        };
        let first = self
            .executor
            .prefill_whale(&whale, posted.as_ref().map(|posted| posted.slot))?;
        let Some(posted) = posted else {
            return Ok(());
        };
        let first = first.ok_or_else(|| {
            anyhow::anyhow!(
                "whale {} owner sampled no boundary token",
                whale.descriptor.seq
            )
        })?;
        if ledger.is_aborted(posted.id) {
            // The whale ran regardless (unanimity is the contract); only the
            // answer is dropped.
            ledger.retire(posted.id);
            self.release_slot(posted.slot);
            return Ok(());
        }
        self.commit_first_token(
            posted.id,
            posted.slot,
            first,
            posted.max_tokens,
            posted.params.ignore_eos,
            ledger,
        );
        Ok(())
    }

    /// A cancelled whale post answers its request with a plain local prefill,
    /// run at the first unrestricted launch after the cancel.
    fn run_whale_fallback(&mut self, lane: &mut WhaleLane, ledger: &mut RequestLedger) {
        let Some(posted) = lane.fallback.take() else {
            return;
        };
        if ledger.is_aborted(posted.id) {
            ledger.retire(posted.id);
            self.release_slot(posted.slot);
            return;
        }
        match self
            .executor
            .prefill(posted.slot, &posted.prompt, &posted.params)
        {
            Ok(first) => self.commit_first_token(
                posted.id,
                posted.slot,
                first,
                posted.max_tokens,
                posted.params.ignore_eos,
                ledger,
            ),
            Err(error) => {
                // A local prefill failure is this request's problem, exactly
                // as on the ordinary admission path.
                self.release_slot(posted.slot);
                ledger.fail(posted.id, format!("{error:#}"));
            }
        }
    }

    /// Record a freshly prefilled request's first token: finish it on the
    /// spot when the token is terminal, seat it in the running batch
    /// otherwise.
    fn commit_first_token(
        &mut self,
        id: RequestId,
        slot: SlotId,
        first: u32,
        max_tokens: usize,
        ignore_eos: bool,
        ledger: &mut RequestLedger,
    ) {
        if self.is_stop_token(first, ignore_eos) {
            // The stop token itself is not part of the completion.
            self.release_slot(slot);
            finish_or_retire(id, FinishReason::Stop, ledger);
            return;
        }
        ledger.push_tokens(id, &[first], &[]);
        if ledger.completion_tokens(id) >= max_tokens {
            self.release_slot(slot);
            finish_or_retire(id, FinishReason::Length, ledger);
            return;
        }
        self.running.push(RunningRequest {
            id,
            slot,
            last_token: first,
            max_tokens,
            ignore_eos,
        });
    }

    /// Whether the whale lane should carry this prompt: lane armed, prompt in
    /// its window, and some gang width admits it.
    fn whale_eligible(&self, prompt_len: usize) -> bool {
        self.whale.as_ref().is_some_and(|lane| {
            prompt_len >= lane.serving.min_tokens
                && k3_whale_width(prompt_len, lane.serving.world, lane.serving.chunk_tokens)
                    .is_some()
        })
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
    /// refuse what can never fit, post whale-window prompts to the
    /// rendezvous, prefill the rest. `Err` means a gang prefill or the
    /// rendezvous transport failed — never a local prefill, which is the
    /// request's problem.
    ///
    /// `may_prefill = false` (an armed or committed whale pins this rank's
    /// launch count) defers admissions that would launch — a multi-launch
    /// local prefill would overshoot the committed superstep. Verdicts that
    /// never touch the device (aborts, rejections, empty completions) flow
    /// regardless.
    fn admit_queued(&mut self, ledger: &mut RequestLedger, may_prefill: bool) -> Result<()> {
        while !self.free_slots.is_empty() {
            // Peek: a wait verdict must leave the request untouched (no
            // ledger writes) for a later step.
            let Some(next) = self.queued.front() else {
                break;
            };
            let id = next.id;
            let prompt_len = next.request.prompt_tokens.len();
            let refusal = self.admission_refusal(&next.request);
            let empty_completion = next.request.max_tokens == 0;
            if ledger.is_aborted(id) {
                self.queued.pop_front();
                ledger.retire(id);
                continue;
            }
            if let Some(reason) = refusal {
                self.queued.pop_front();
                ledger.reject(id, reason);
                continue;
            }
            if empty_completion {
                // Nothing to generate: answer without occupying a slot.
                self.queued.pop_front();
                ledger.admit(id);
                finish_or_retire(id, FinishReason::Length, ledger);
                continue;
            }
            if self.whale_eligible(prompt_len) {
                let lane = self.whale.as_ref().expect("eligibility implies the lane");
                if lane.posted.is_some() || lane.fallback.is_some() {
                    // One outstanding whale per rank keeps the cancel pairing
                    // trivial; whales are sparse and the rendezvous is a few
                    // launches, so the head of the queue waits.
                    break;
                }
                let (poster, hub) = (lane.rank, lane.serving.hub.clone());
                let queued = self.queued.pop_front().expect("front just observed");
                ledger.admit(id);
                let slot = self
                    .free_slots
                    .pop()
                    .expect("loop runs only while a slot is free");
                let prompt: Arc<[u32]> = Arc::from(queued.request.prompt_tokens.as_slice());
                if let Err(error) = hub.send(WhaleToSequencer::Request {
                    request: id.raw(),
                    poster,
                    prompt: prompt.clone(),
                }) {
                    // The rendezvous transport is fleet infrastructure; its
                    // loss is the engine's, not the request's.
                    self.release_slot(slot);
                    ledger.fail(id, format!("{error:#}"));
                    return Err(error);
                }
                let lane = self.whale.as_mut().expect("eligibility implies the lane");
                lane.posted = Some(PostedWhale {
                    id,
                    slot,
                    seq: None,
                    prompt,
                    params: queued.request.params,
                    max_tokens: queued.request.max_tokens,
                });
                continue;
            }
            if !may_prefill {
                break;
            }
            let pending = self.queued.pop_front().expect("front just observed");
            ledger.admit(id);
            let slot = self
                .free_slots
                .pop()
                .expect("loop runs only while a slot is free");
            let first = match self.cp_gang_for(prompt_len) {
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
            self.commit_first_token(
                id,
                slot,
                first,
                pending.request.max_tokens,
                pending.request.params.ignore_eos,
                ledger,
            );
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
        // Then the whale lane, at what is a launch boundary by construction:
        // it drains the rendezvous, enters a committed superstep when this is
        // its launch, and its verdict caps this step's admissions at
        // single-launch work while a whale is pending.
        let may_prefill = self.serve_whale(ledger)?;
        self.admit_queued(ledger, may_prefill)?;
        self.decode_running(ledger);
        // Local prefill and decode failures are per-request and absorbed
        // above; gang, whale, and transport failures propagate the same way a
        // serve error does.
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
