//! Contract tests: drive a real [`K3Scheduler`] through the engine contract
//! (submit → step stream → terminal) with a fake executor. These pin the
//! protocol a frontend can rely on, not scheduler internals — every one of
//! them survives the GPU executor landing.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
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
use super::K3SchedulerConfig;
use super::SlotId;
use super::StepExecutor;
use super::UNWIRED_MESSAGE;
use super::launch_unwired;
use super::start_with_executors;

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

#[test]
fn unwired_engine_spawns_one_scheduler_per_partition_and_fails_honestly() {
    let engine = launch_unwired(4, vec![EOS_TOKEN]);
    assert_eq!(engine.schedulers.len(), 4);
    assert!(
        engine.info.kv_capacity.is_none() && engine.lora.is_none(),
        "the phase-1 engine reports no KV pool and no adapters"
    );

    let (partition, mut steps) = partition(engine);
    let control = partition.handle.submit(request(4, 8));
    let (tokens, terminal) = steps.collect_terminal(control.id());
    assert!(
        tokens.is_empty(),
        "an engine without a model must not invent tokens"
    );
    match terminal {
        Terminal::Failed { message, .. } => {
            assert_eq!(message, UNWIRED_MESSAGE);
        }
        other => panic!("the placeholder engine must fail requests, got {other:?}"),
    }
}
