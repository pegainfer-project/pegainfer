//! Contract tests: drive a real [`K3Scheduler`] through the engine contract
//! (submit → step stream → terminal) with a fake executor. These pin the
//! protocol a frontend can rely on, not scheduler internals — every one of
//! them survives the GPU executor landing.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LiveScheduler;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;

use super::DecodeSlot;
use super::K3CpGang;
use super::K3CpServing;
use super::K3SchedulerConfig;
use super::K3WhaleHub;
use super::K3WhaleServing;
use super::SlotId;
use super::StepExecutor;
use super::start_with_executors;
use super::whale::CommittedWhale;
use super::whale_hub::LocalWhaleHub;
use crate::executor::cp::K3CpGroup;

const EOS_TOKEN: u32 = 99;

// ── Fake executor ───────────────────────────────────────────────────────

/// Scripted stand-in for the GPU model: slot `s` answers prefill with
/// `10 * (s + 1)` and its `n`-th decode with `10 * (s + 1) + n`, so a test can
/// assert the exact stream a request receives. It also asserts the
/// scheduler's slot discipline — no slot is ever handed out twice or decoded
/// after release.
struct FakeExecutor {
    max_batch: usize,
    max_context: usize,
    /// Decode steps served per live slot.
    steps: HashMap<SlotId, u32>,
    live: HashSet<SlotId>,
    /// Emit [`EOS_TOKEN`] instead of the scripted token at this step index
    /// (0 = at prefill).
    eos_at: Option<u32>,
    fail_next_prefill: bool,
    fail_next_decode: bool,
    decode_delay: Duration,
    released: Arc<Mutex<Vec<SlotId>>>,
}

impl FakeExecutor {
    fn new(max_batch: usize, released: Arc<Mutex<Vec<SlotId>>>) -> Self {
        Self {
            max_batch,
            max_context: 4096,
            steps: HashMap::new(),
            live: HashSet::new(),
            eos_at: None,
            fail_next_prefill: false,
            fail_next_decode: false,
            decode_delay: Duration::ZERO,
            released,
        }
    }

    fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = max_context;
        self
    }

    fn with_eos_at(mut self, step: u32) -> Self {
        self.eos_at = Some(step);
        self
    }

    fn with_one_prefill_failure(mut self) -> Self {
        self.fail_next_prefill = true;
        self
    }

    fn with_one_decode_failure(mut self) -> Self {
        self.fail_next_decode = true;
        self
    }

    fn with_decode_delay(mut self, delay: Duration) -> Self {
        self.decode_delay = delay;
        self
    }

    fn token(&self, slot: SlotId, step: u32) -> u32 {
        if self.eos_at == Some(step) {
            EOS_TOKEN
        } else {
            10 * (slot as u32 + 1) + step
        }
    }
}

impl StepExecutor for FakeExecutor {
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn prefill(&mut self, slot: SlotId, _prompt: &[u32], _params: &SamplingParams) -> Result<u32> {
        assert!(self.live.insert(slot), "slot {slot} was handed out twice");
        if std::mem::take(&mut self.fail_next_prefill) {
            // The scheduler already reserved the slot and will `release` it
            // on this error, so the seat is live until then.
            anyhow::bail!("fake prefill failure");
        }
        self.steps.insert(slot, 0);
        Ok(self.token(slot, 0))
    }

    fn decode(&mut self, batch: &[DecodeSlot]) -> Result<Vec<u32>> {
        // Idle steps reach the executor by contract (free-running EP ranks
        // pad them); a fake with nothing to do steps in silence, without
        // consuming the injected failure.
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        std::thread::sleep(self.decode_delay);
        if std::mem::take(&mut self.fail_next_decode) {
            anyhow::bail!("fake decode failure");
        }
        Ok(batch
            .iter()
            .map(|entry| {
                assert!(
                    self.live.contains(&entry.slot),
                    "slot {} decoded after release",
                    entry.slot
                );
                let step = self
                    .steps
                    .get_mut(&entry.slot)
                    .expect("live slot was prefilled");
                *step += 1;
                let step = *step;
                self.token(entry.slot, step)
            })
            .collect())
    }

    fn release(&mut self, slot: SlotId) {
        assert!(self.live.remove(&slot), "slot {slot} released while free");
        self.steps.remove(&slot);
        self.released.lock().expect("released log").push(slot);
    }
}

// ── Harness ─────────────────────────────────────────────────────────────

fn request(prompt_len: usize, max_tokens: usize) -> Request {
    Request {
        prompt_tokens: vec![7; prompt_len],
        params: SamplingParams::default(),
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: 0,
        echo: false,
        trace_parent: None,
        client_label: None,
    }
}

fn launch(executor: FakeExecutor) -> (LiveScheduler, StepCollector) {
    let config = K3SchedulerConfig {
        eos_token_ids: vec![EOS_TOKEN],
        kv_capacity: None,
        cp: None,
        whale: None,
    };
    partition(start_with_executors(vec![executor], &config))
}

fn partition(mut engine: Engine) -> (LiveScheduler, StepCollector) {
    let mut scheduler = engine.schedulers.remove(0);
    let steps = scheduler
        .handle
        .take_steps()
        .expect("a fresh scheduler yields its step stream once");
    (scheduler, StepCollector::new(steps))
}

/// Demultiplex the step stream per request, preserving each request's update
/// order, so a test can await one request without dropping another's updates.
struct StepCollector {
    steps: StepReceiver,
    buffered: HashMap<RequestId, VecDeque<RequestUpdate>>,
}

impl StepCollector {
    fn new(steps: StepReceiver) -> Self {
        Self {
            steps,
            buffered: HashMap::new(),
        }
    }

    fn next_for(&mut self, id: RequestId) -> RequestUpdate {
        loop {
            if let Some(update) = self.buffered.get_mut(&id).and_then(VecDeque::pop_front) {
                return update;
            }
            let step = self
                .steps
                .blocking_recv()
                .expect("step stream closed while awaiting an update");
            for update in step.updates {
                self.buffered
                    .entry(update.id)
                    .or_default()
                    .push_back(update);
            }
        }
    }

    /// Fold this request's stream to its end: every token in order, plus the
    /// terminal.
    fn collect_terminal(&mut self, id: RequestId) -> (Vec<u32>, Terminal) {
        let mut tokens = Vec::new();
        loop {
            let update = self.next_for(id);
            tokens.extend_from_slice(&update.tokens);
            assert!(
                update.cached_tokens.is_none(),
                "K3 has no prefix cache and must never report cached tokens"
            );
            if let Some(terminal) = update.terminal {
                return (tokens, terminal);
            }
        }
    }

    /// Drain the stream to engine shutdown and return every terminal seen for
    /// `id`. For asserting silence after an abort.
    fn drain_terminals_for(&mut self, id: RequestId) -> Vec<Terminal> {
        let mut terminals: Vec<Terminal> = self
            .buffered
            .remove(&id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|update| update.terminal)
            .collect();
        while let Some(step) = self.steps.blocking_recv() {
            for update in step.updates {
                if update.id == id
                    && let Some(terminal) = update.terminal
                {
                    terminals.push(terminal);
                }
            }
        }
        terminals
    }
}

fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn admitted_request_streams_its_tokens_and_finishes_at_max_tokens() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let (partition, mut steps) = launch(FakeExecutor::new(4, Arc::clone(&released)));

    let control = partition.handle.submit(request(4, 3));
    let first = steps.next_for(control.id());
    assert!(
        first.scheduled.is_some(),
        "admission facts ride the request's first update: {first:?}"
    );
    assert_eq!(first.scheduled.expect("admitted").prompt_tokens, 4);

    let (mut tokens, terminal) = steps.collect_terminal(control.id());
    tokens.splice(0..0, first.tokens);
    assert_eq!(tokens, vec![10, 11, 12]);
    assert!(
        matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 4,
                completion_tokens: 3,
            }
        ),
        "{terminal:?}"
    );
    assert!(
        wait_until(Duration::from_secs(1), || !released
            .lock()
            .expect("released log")
            .is_empty()),
        "a finished request must give its slot back"
    );
}

#[test]
fn a_zero_length_completion_finishes_without_taking_a_slot() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let (partition, mut steps) = launch(FakeExecutor::new(1, Arc::clone(&released)));

    let empty = partition.handle.submit(request(4, 0));
    let (tokens, terminal) = steps.collect_terminal(empty.id());
    assert!(tokens.is_empty());
    assert!(
        matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 0,
                ..
            }
        ),
        "{terminal:?}"
    );
    assert!(
        released.lock().expect("released log").is_empty(),
        "no slot was taken, so none is released"
    );

    // The one slot is still free for the next request.
    let next = partition.handle.submit(request(4, 1));
    let (tokens, _) = steps.collect_terminal(next.id());
    assert_eq!(tokens, vec![10]);
}

#[test]
fn eos_finishes_with_stop_and_is_not_streamed() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, released).with_eos_at(2);
    let (partition, mut steps) = launch(executor);

    let control = partition.handle.submit(request(4, 64));
    let (tokens, terminal) = steps.collect_terminal(control.id());
    assert_eq!(
        tokens,
        vec![10, 11],
        "the stop token itself is not part of the completion"
    );
    assert!(
        matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Stop,
                completion_tokens: 2,
                ..
            }
        ),
        "{terminal:?}"
    );
}

#[test]
fn oversized_request_is_rejected_without_taking_a_slot() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(1, released).with_max_context(16);
    let (partition, mut steps) = launch(executor);

    let too_long = partition.handle.submit(request(12, 8));
    let (tokens, terminal) = steps.collect_terminal(too_long.id());
    assert!(tokens.is_empty(), "a rejected request produces no tokens");
    match terminal {
        Terminal::Rejected {
            reason:
                RejectReason::ContextLength {
                    prompt_tokens,
                    max_tokens,
                    limit,
                },
            prompt_tokens: reported,
        } => {
            assert_eq!((prompt_tokens, max_tokens, limit), (12, 8, 16));
            assert_eq!(reported, 12);
        }
        other => panic!("an unservable request must be rejected, got {other:?}"),
    }

    // The single slot was never occupied, so the next request runs at once.
    let fits = partition.handle.submit(request(4, 1));
    let (tokens, terminal) = steps.collect_terminal(fits.id());
    assert_eq!(tokens, vec![10]);
    assert!(
        matches!(terminal, Terminal::Finished { .. }),
        "{terminal:?}"
    );
}

#[test]
fn aborted_request_retires_silently_and_frees_its_slot() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let executor =
        FakeExecutor::new(4, Arc::clone(&released)).with_decode_delay(Duration::from_millis(5));
    let (partition, mut steps) = launch(executor);

    // Long enough to still be decoding when the abort lands, short enough to
    // stay inside the executor's context window.
    let control = partition.handle.submit(request(4, 1_000));
    let first = steps.next_for(control.id());
    assert!(
        first.scheduled.is_some(),
        "request must be admitted before the abort: {first:?}"
    );

    control.abort();
    assert!(
        wait_until(Duration::from_secs(2), || released
            .lock()
            .expect("released log")
            .contains(&0)),
        "an aborted request must give its slot back"
    );

    // The abort is the frontend's own act; the scheduler answers with
    // silence, not a terminal. Drain to engine shutdown to prove it.
    drop(partition.handle);
    let terminals = steps.drain_terminals_for(control.id());
    assert!(
        terminals.is_empty(),
        "aborted request must not receive a terminal: {terminals:?}"
    );
    partition.join.join().expect("scheduler thread exits");
}

#[test]
fn prefill_failure_fails_the_request_and_the_scheduler_keeps_serving() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, Arc::clone(&released)).with_one_prefill_failure();
    let (partition, mut steps) = launch(executor);

    let doomed = partition.handle.submit(request(4, 8));
    let (tokens, terminal) = steps.collect_terminal(doomed.id());
    assert!(tokens.is_empty(), "a prefill failure produces no tokens");
    match terminal {
        Terminal::Failed {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert!(message.contains("fake prefill failure"), "{message}");
            assert_eq!((prompt_tokens, completion_tokens), (4, 0));
        }
        other => panic!("a prefill failure must surface as Failed, got {other:?}"),
    }
    assert!(
        wait_until(Duration::from_secs(1), || released
            .lock()
            .expect("released log")
            .contains(&0)),
        "a failed prefill must give its slot back"
    );

    let after = partition.handle.submit(request(4, 2));
    let (tokens, terminal) = steps.collect_terminal(after.id());
    assert_eq!(tokens, vec![10, 11]);
    assert!(
        matches!(terminal, Terminal::Finished { .. }),
        "{terminal:?}"
    );
}

#[test]
fn decode_failure_fails_the_batch_and_the_scheduler_keeps_serving() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, Arc::clone(&released)).with_one_decode_failure();
    let (partition, mut steps) = launch(executor);

    let doomed = partition.handle.submit(request(4, 8));
    let (tokens, terminal) = steps.collect_terminal(doomed.id());
    assert_eq!(
        tokens,
        vec![10],
        "the prefill token ships before the decode failure"
    );
    match terminal {
        Terminal::Failed {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert!(message.contains("fake decode failure"), "{message}");
            assert_eq!((prompt_tokens, completion_tokens), (4, 1));
        }
        other => panic!("a decode failure must surface as Failed, got {other:?}"),
    }

    let after = partition.handle.submit(request(4, 2));
    let (tokens, terminal) = steps.collect_terminal(after.id());
    assert_eq!(tokens, vec![10, 11]);
    assert!(
        matches!(terminal, Terminal::Finished { .. }),
        "{terminal:?}"
    );
}

// ── CP gang protocol ────────────────────────────────────────────────────
//
// These tests pin the gang's one hard invariant on a fake EP world: every
// partition enters `prefill_cp` for a job at the same launch count, without
// anyone ever waiting quietly. The world models the mega launches' two-sided
// pairing — a step completes only once every partition has launched a step
// with the same index — with a timeout panic standing in for the device
// watchdog, so a protocol regression fails as a deadlock report, not a hang.

const CP_TIMEOUT: Duration = Duration::from_secs(10);

/// The shared fake EP world: per-partition launch counts plus the protocol
/// log the assertions read.
struct CpWorld {
    counts: Vec<AtomicU64>,
    /// Partitions whose scheduler thread has exited (executor dropped) — a
    /// gone peer no longer holds back anyone's step pairing. This is where
    /// the model is kinder than the device: a real EP rank that stops
    /// launching wedges its peers in the mega kernel until the watchdog
    /// fires. The flag only lets the *teardown* of a passed test drain;
    /// while a test runs, every live partition must keep pairing.
    finished: Vec<AtomicBool>,
    /// Per job (keyed by the prompt's first token): each partition's launch
    /// count on entering `prefill_cp`.
    entries: Mutex<HashMap<u32, Vec<(usize, u64)>>>,
    /// Per partition: job keys in execution order.
    orders: Mutex<Vec<Vec<u32>>>,
}

impl CpWorld {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            counts: (0..size).map(|_| AtomicU64::new(0)).collect(),
            finished: (0..size).map(|_| AtomicBool::new(false)).collect(),
            entries: Mutex::new(HashMap::new()),
            orders: Mutex::new(vec![Vec::new(); size]),
        })
    }

    /// Launch one step as `partition` and wait for it to pair: blocks until
    /// every live peer has launched a step with the same index.
    fn step(&self, partition: usize) {
        self.burst(partition, 1);
    }

    /// Launch `n` steps back-to-back, waiting only for the last to pair —
    /// the way a local multi-chunk prefill queues its whole ladder before
    /// its one sync. This is the cross-partition skew the leveling loop
    /// must absorb.
    fn burst(&self, partition: usize, n: u64) {
        let count = self.counts[partition].fetch_add(n, Ordering::SeqCst) + n;
        let deadline = Instant::now() + CP_TIMEOUT;
        loop {
            let paired = self.counts.iter().zip(&self.finished).all(|(peer, gone)| {
                peer.load(Ordering::SeqCst) >= count || gone.load(Ordering::SeqCst)
            });
            if paired {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "partition {partition} deadlocked: step {count} never paired"
            );
            std::thread::yield_now();
        }
    }

    /// Wait until every partition has logged its `prefill_cp` entry for
    /// `key` — the stand-in for the chunk step's exchange windows, which no
    /// member leaves before the whole gang has arrived.
    fn await_gang(&self, key: u32, size: usize) {
        let deadline = Instant::now() + CP_TIMEOUT;
        loop {
            if self.entries.lock().expect("entry log")[&key].len() == size {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "gang for job {key} never fully entered prefill_cp"
            );
            std::thread::yield_now();
        }
    }
}

/// One partition of the fake world, serving whichever lane the test armed
/// (CP gang or whale). Every stepping method launches through
/// [`CpWorld::step`], so the scheduler's own cadence (decode padding, gang
/// pumps, the chunk step) is what drives the pairing.
struct FakeCpExecutor {
    world: Arc<CpWorld>,
    /// Per-whale role log; stays empty when only the CP lane is armed.
    roles: WhaleRoles,
    partition: usize,
    /// The armed lane's admission floor: a prompt at or past it taking the
    /// plain local prefill path is the failure under test.
    local_floor: usize,
}

impl Drop for FakeCpExecutor {
    fn drop(&mut self) {
        self.world.finished[self.partition].store(true, Ordering::SeqCst);
    }
}

impl StepExecutor for FakeCpExecutor {
    fn max_batch(&self) -> usize {
        2
    }

    fn max_context_tokens(&self) -> usize {
        65536
    }

    fn prefill(&mut self, _slot: SlotId, prompt: &[u32], _params: &SamplingParams) -> Result<u32> {
        anyhow::ensure!(
            prompt.len() < self.local_floor,
            "a gang-eligible prompt of {} tokens took the local prefill path",
            prompt.len()
        );
        self.world.burst(self.partition, 4);
        Ok(500)
    }

    fn decode(&mut self, batch: &[DecodeSlot]) -> Result<Vec<u32>> {
        self.world.step(self.partition);
        Ok(vec![7; batch.len()])
    }

    fn prefill_cp(
        &mut self,
        _slot: SlotId,
        prompt: &[u32],
        group: &Arc<K3CpGroup>,
        cp_rank: usize,
    ) -> Result<Option<u32>> {
        let key = prompt[0];
        {
            let mut entries = self.world.entries.lock().expect("entry log");
            entries.entry(key).or_default().push((
                self.partition,
                self.world.counts[self.partition].load(Ordering::SeqCst),
            ));
            self.world.orders.lock().expect("order log")[self.partition].push(key);
        }
        self.world.await_gang(key, group.cp_size());
        self.world.step(self.partition);
        Ok((cp_rank == group.cp_size() - 1).then_some(1000 + key))
    }

    fn prefill_whale(
        &mut self,
        whale: &CommittedWhale,
        slot: Option<SlotId>,
    ) -> Result<Option<u32>> {
        let key = whale.descriptor.prompt[0];
        {
            let mut entries = self.world.entries.lock().expect("entry log");
            entries.entry(key).or_default().push((
                self.partition,
                self.world.counts[self.partition].load(Ordering::SeqCst),
            ));
            self.world.orders.lock().expect("order log")[self.partition].push(key);
            self.roles
                .lock()
                .expect("role log")
                .entry(key)
                .or_default()
                .push((self.partition, whale.cp_rank, slot.is_some()));
        }
        self.world.await_gang(key, whale.descriptor.gang.len());
        self.world.step(self.partition);
        Ok(slot.map(|_| 2000 + key))
    }

    fn pump_step(&mut self) -> Result<()> {
        self.world.step(self.partition);
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.world.counts[self.partition].load(Ordering::SeqCst)
    }

    fn release(&mut self, _slot: SlotId) {}
}

/// A `size`-partition engine over one fake world, CP lane armed.
fn launch_cp(size: usize) -> (Arc<CpWorld>, Vec<(LiveScheduler, StepCollector)>) {
    let world = CpWorld::new(size);
    let executors = (0..size)
        .map(|partition| FakeCpExecutor {
            world: Arc::clone(&world),
            roles: Arc::default(),
            partition,
            local_floor: 32,
        })
        .collect();
    let config = K3SchedulerConfig {
        eos_token_ids: vec![EOS_TOKEN],
        kv_capacity: None,
        cp: Some(K3CpServing {
            gang: K3CpGang::new(K3CpGroup::new(size).expect("CP group")),
            min_tokens: 32,
            chunk_tokens: 4096,
        }),
        whale: None,
    };
    let mut engine = start_with_executors(executors, &config);
    let partitions = engine
        .schedulers
        .drain(..)
        .map(|mut scheduler| {
            let steps = scheduler
                .handle
                .take_steps()
                .expect("a fresh scheduler yields its step stream once");
            (scheduler, StepCollector::new(steps))
        })
        .collect();
    (world, partitions)
}

/// A CP-eligible request whose prompt is keyed by its first token.
fn cp_request(key: u32, max_tokens: usize) -> Request {
    Request {
        prompt_tokens: vec![key; 60],
        ..request(0, max_tokens)
    }
}

#[test]
fn cp_gang_enters_prefill_cp_at_one_launch_count() {
    let (world, mut partitions) = launch_cp(3);

    let control = partitions[0].0.handle.submit(cp_request(5, 2));
    let (tokens, terminal) = partitions[0].1.collect_terminal(control.id());
    assert_eq!(
        tokens,
        vec![1005, 7],
        "the owner streams the boundary token, then decodes"
    );
    assert!(
        matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ),
        "{terminal:?}"
    );

    let entries = world.entries.lock().expect("entry log");
    let job = &entries[&5];
    assert_eq!(job.len(), 3, "every partition ran the job exactly once");
    assert!(
        job.iter().all(|&(_, count)| count == job[0].1),
        "the gang must enter prefill_cp at one launch count: {job:?}"
    );

    drain_partitions(partitions);
}

#[test]
fn concurrent_cp_posters_run_in_one_order_on_every_partition() {
    let (world, mut partitions) = launch_cp(3);

    // Two posters race: whatever order the board assigns, every partition
    // must execute the jobs in that same order — interleaved gangs would
    // cross their exchange windows.
    let first = partitions[0].0.handle.submit(cp_request(5, 1));
    let second = partitions[1].0.handle.submit(cp_request(6, 1));
    let (tokens, _) = partitions[0].1.collect_terminal(first.id());
    assert_eq!(tokens, vec![1005]);
    let (tokens, _) = partitions[1].1.collect_terminal(second.id());
    assert_eq!(tokens, vec![1006]);

    {
        let entries = world.entries.lock().expect("entry log");
        for key in [5, 6] {
            let job = &entries[&key];
            assert_eq!(job.len(), 3);
            assert!(
                job.iter().all(|&(_, count)| count == job[0].1),
                "job {key} entered unlevel: {job:?}"
            );
        }
        let orders = world.orders.lock().expect("order log");
        assert!(
            orders.iter().all(|order| *order == orders[0]),
            "partitions disagree on the job order: {orders:?}"
        );
    }

    drain_partitions(partitions);
}

#[test]
fn a_local_prefill_burst_does_not_unlevel_the_gang() {
    let (world, mut partitions) = launch_cp(3);

    // Partition 0 runs a local multi-chunk prefill — a 4-launch burst that
    // puts it several launches ahead — while partition 1 posts a gang job.
    // The laggards must pump up to the burst's count before anyone computes.
    let local = partitions[0].0.handle.submit(request(4, 1));
    let gang = partitions[1].0.handle.submit(cp_request(5, 1));
    let (tokens, _) = partitions[0].1.collect_terminal(local.id());
    assert_eq!(tokens, vec![500]);
    let (tokens, _) = partitions[1].1.collect_terminal(gang.id());
    assert_eq!(tokens, vec![1005]);

    let entries = world.entries.lock().expect("entry log");
    let job = &entries[&5];
    assert_eq!(job.len(), 3);
    assert!(
        job.iter().all(|&(_, count)| count == job[0].1),
        "the gang entered unlevel across a local burst: {job:?}"
    );
    drop(entries);

    drain_partitions(partitions);
}

// ── The whale lane ──────────────────────────────────────────────────────
//
// Same fake world and executor as the CP gang — the launch counts and their
// pairing are the physics under test — but coordination runs through the
// whale rendezvous ([`super::whale`]) over a [`LocalWhaleHub`], the
// in-process degenerate fleet. The cancel → local-fallback path has no
// in-process trigger (the poster's eligibility check and the sequencer's
// admission are the same deterministic predicate, and gather timeouts live
// in the TCP hub), so it is pinned at the state-machine level in
// [`super::whale`]'s tests instead.

const WHALE_CHUNK: usize = 4096;
const WHALE_MIN: usize = 2048;

/// Per whale (keyed by the prompt's first token): each member's view on
/// entry — partition, CP rank, and whether it owns the result slot.
type WhaleRoles = Arc<Mutex<HashMap<u32, Vec<(usize, usize, bool)>>>>;

/// A `size`-partition engine over one fake world, whale lane armed through a
/// local hub (`world == size`, global rank == partition).
fn launch_whale(
    size: usize,
) -> (
    Arc<CpWorld>,
    WhaleRoles,
    Vec<(LiveScheduler, StepCollector)>,
) {
    let world = CpWorld::new(size);
    let roles: WhaleRoles = Arc::default();
    let executors = (0..size)
        .map(|partition| FakeCpExecutor {
            world: Arc::clone(&world),
            roles: Arc::clone(&roles),
            partition,
            local_floor: WHALE_MIN,
        })
        .collect();
    let config = K3SchedulerConfig {
        eos_token_ids: vec![EOS_TOKEN],
        kv_capacity: None,
        cp: None,
        whale: Some(K3WhaleServing {
            hub: K3WhaleHub::Local(LocalWhaleHub::new(size, WHALE_CHUNK)),
            world: size,
            first_local: 0,
            min_tokens: WHALE_MIN,
            chunk_tokens: WHALE_CHUNK,
        }),
    };
    let mut engine = start_with_executors(executors, &config);
    let partitions = engine
        .schedulers
        .drain(..)
        .map(|mut scheduler| {
            let steps = scheduler
                .handle
                .take_steps()
                .expect("a fresh scheduler yields its step stream once");
            (scheduler, StepCollector::new(steps))
        })
        .collect();
    (world, roles, partitions)
}

/// A whale-window request whose prompt is keyed by its first token.
fn whale_request(key: u32, prompt_len: usize, max_tokens: usize) -> Request {
    Request {
        prompt_tokens: vec![key; prompt_len],
        ..request(0, max_tokens)
    }
}

/// Tear the engine down: drop every handle and join the scheduler threads.
fn drain_partitions(partitions: Vec<(LiveScheduler, StepCollector)>) {
    for (partition, _) in partitions {
        drop(partition.handle);
        partition.join.join().expect("scheduler thread exits");
    }
}

#[test]
fn a_committed_whale_enters_every_rank_at_one_launch_count() {
    let (world, roles, mut partitions) = launch_whale(4);

    // 12288 tokens over chunk 4096: only width 4 admits, so the whole world
    // is the gang.
    let control = partitions[2].0.handle.submit(whale_request(5, 12288, 2));
    let (tokens, terminal) = partitions[2].1.collect_terminal(control.id());
    assert_eq!(
        tokens,
        vec![2005, 7],
        "the poster streams the boundary token, then decodes"
    );
    assert!(
        matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ),
        "{terminal:?}"
    );

    let entries = world.entries.lock().expect("entry log");
    let job = &entries[&5];
    assert_eq!(job.len(), 4, "every rank ran the whale exactly once");
    assert!(
        job.iter().all(|&(_, count)| count == job[0].1),
        "the gang must enter prefill_whale at one launch count: {job:?}"
    );
    let roles = roles.lock().expect("role log");
    let job = &roles[&5];
    let cp_ranks: HashSet<usize> = job.iter().map(|&(_, cp_rank, _)| cp_rank).collect();
    assert_eq!(cp_ranks, (0..4).collect(), "CP ranks must be a permutation");
    for &(partition, cp_rank, owns_slot) in job {
        assert_eq!(
            owns_slot,
            partition == 2,
            "only the poster owns the result slot: {job:?}"
        );
        if partition == 2 {
            assert_eq!(cp_rank, 3, "the poster serves the last CP rank");
        }
    }
    drop((entries, roles));

    drain_partitions(partitions);
}

#[test]
fn ranks_outside_the_gang_never_hear_of_the_whale() {
    let (world, roles, mut partitions) = launch_whale(4);

    // 6144 tokens: width 4 makes 1536-token segments (under the floor), so
    // the whale runs on a width-2 gang and two ranks stay out entirely.
    let control = partitions[0].0.handle.submit(whale_request(6, 6144, 1));
    let (tokens, _) = partitions[0].1.collect_terminal(control.id());
    assert_eq!(tokens, vec![2006]);

    let entries = world.entries.lock().expect("entry log");
    let job = &entries[&6];
    assert_eq!(
        job.len(),
        2,
        "exactly the gang enters; the other ranks only ever pad: {job:?}"
    );
    assert!(
        job.iter().all(|&(_, count)| count == job[0].1),
        "the gang must enter at one launch count: {job:?}"
    );
    let roles = roles.lock().expect("role log");
    let job = &roles[&6];
    let poster = job
        .iter()
        .find(|&&(partition, _, _)| partition == 0)
        .expect("the poster is in its own gang");
    assert_eq!(
        (poster.1, poster.2),
        (1, true),
        "the poster serves the last CP rank and owns the slot"
    );
    let helper = job
        .iter()
        .find(|&&(partition, _, _)| partition != 0)
        .expect("a width-2 gang has one helper");
    assert_eq!((helper.1, helper.2), (0, false), "{job:?}");
    drop((entries, roles));

    drain_partitions(partitions);
}

#[test]
fn concurrent_whales_serialize_in_one_order_on_every_rank() {
    let (world, _roles, mut partitions) = launch_whale(4);

    // Two posters race. The sequencer gathers one whale at a time and every
    // committed launch is strictly later than the previous, so all ranks run
    // the two supersteps in the same order at two distinct counts.
    let first = partitions[0].0.handle.submit(whale_request(5, 12288, 1));
    let second = partitions[3].0.handle.submit(whale_request(6, 12288, 1));
    let (tokens, _) = partitions[0].1.collect_terminal(first.id());
    assert_eq!(tokens, vec![2005]);
    let (tokens, _) = partitions[3].1.collect_terminal(second.id());
    assert_eq!(tokens, vec![2006]);

    let entries = world.entries.lock().expect("entry log");
    for key in [5, 6] {
        let job = &entries[&key];
        assert_eq!(job.len(), 4);
        assert!(
            job.iter().all(|&(_, count)| count == job[0].1),
            "whale {key} entered unlevel: {job:?}"
        );
    }
    assert_ne!(
        entries[&5][0].1, entries[&6][0].1,
        "two whales cannot share a superstep"
    );
    let orders = world.orders.lock().expect("order log");
    assert!(
        orders.iter().all(|order| *order == orders[0]),
        "ranks disagree on the whale order: {orders:?}"
    );
    drop((entries, orders));

    drain_partitions(partitions);
}

#[test]
fn a_second_whale_on_one_rank_waits_for_the_outstanding_post() {
    let (world, _roles, mut partitions) = launch_whale(4);

    // One rank posts two whales back-to-back. The lane keeps one post
    // outstanding, so the second waits at the head of the queue and both
    // still complete, on distinct supersteps.
    let first = partitions[1].0.handle.submit(whale_request(5, 12288, 1));
    let second = partitions[1].0.handle.submit(whale_request(6, 12288, 1));
    let (tokens, _) = partitions[1].1.collect_terminal(first.id());
    assert_eq!(tokens, vec![2005]);
    let (tokens, _) = partitions[1].1.collect_terminal(second.id());
    assert_eq!(tokens, vec![2006]);

    let entries = world.entries.lock().expect("entry log");
    assert_eq!(entries[&5].len(), 4);
    assert_eq!(entries[&6].len(), 4);
    assert!(
        entries[&5][0].1 < entries[&6][0].1,
        "the queued whale must run strictly after the outstanding one: {entries:?}"
    );
    drop(entries);

    drain_partitions(partitions);
}

#[test]
fn short_prompts_prefill_locally_under_the_whale_lane() {
    let (world, roles, mut partitions) = launch_whale(4);

    let control = partitions[0].0.handle.submit(request(60, 1));
    let (tokens, _) = partitions[0].1.collect_terminal(control.id());
    assert_eq!(tokens, vec![500], "a short prompt takes the local prefill");
    assert!(
        world.entries.lock().expect("entry log").is_empty(),
        "no whale ran"
    );
    assert!(roles.lock().expect("role log").is_empty());

    drain_partitions(partitions);
}

#[test]
fn requests_beyond_the_slot_budget_wait_instead_of_being_refused() {
    let released = Arc::new(Mutex::new(Vec::new()));
    let (partition, mut steps) = launch(FakeExecutor::new(1, released));

    let controls: Vec<_> = (0..3)
        .map(|_| partition.handle.submit(request(4, 2)))
        .collect();
    for control in &controls {
        let (tokens, terminal) = steps.collect_terminal(control.id());
        assert_eq!(
            tokens,
            vec![10, 11],
            "every queued request runs on the one slot in turn"
        );
        assert!(
            matches!(
                terminal,
                Terminal::Finished {
                    reason: FinishReason::Length,
                    ..
                }
            ),
            "{terminal:?}"
        );
    }
}
