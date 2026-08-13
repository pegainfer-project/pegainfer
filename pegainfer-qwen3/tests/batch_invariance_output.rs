//! Engine-level Pin gate: a request's emitted token ids and per-token logprob bits must be
//! identical alone and co-batched. The harness folds the raw step stream into one global fact
//! order (admission, then tokens, per `RequestUpdate`), which carries no step composition — so the
//! per-phase guards below only assert the composition under test happened at all; step and chunk
//! assignment is gated in the scheduler's unit tests, which read the assignment itself.
//!
//! Needs one CUDA GPU, `PEGAINFER_TEST_MODEL_PATH`, and `--test-threads=1`.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::SchedulerHandle;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use tokio::sync::mpsc::error::TryRecvError;

#[allow(dead_code)]
mod common;

use common::harness::request;

const A_TOKENS: usize = 64;
const LOAD_COUNT: usize = 40;
const BURST_DEPTH: usize = 3;
const MIN_BURSTS_IN_WINDOW: usize = 8;
type Sample = (u32, Option<f32>);

static POLICY_LOCK: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct Trace {
    /// The submit-time tag, for diagnostics only — routing is by [`RequestId`].
    label: String,
    samples: Vec<Sample>,
    token_orders: Vec<usize>,
    scheduled_orders: Vec<usize>,
    terminal: bool,
}

impl Trace {
    fn token_before(&self, order: usize) -> bool {
        self.token_orders.iter().any(|&candidate| candidate < order)
    }

    fn token_after(&self, order: usize) -> bool {
        self.token_orders.iter().any(|&candidate| candidate > order)
    }

    fn scheduled_between(&self, start: usize, end: usize) -> bool {
        self.scheduled_orders
            .iter()
            .any(|&candidate| candidate > start && candidate < end)
    }
}

/// Drives the engine's step stream directly (instead of the per-request
/// demultiplexing in `common::harness`): the phase guards need one global
/// order over every request's admission and token facts, which only the
/// undemultiplexed stream provides.
struct Harness {
    handle: Option<SchedulerHandle>,
    scheduler_join: Option<std::thread::JoinHandle<()>>,
    rx: StepReceiver,
    traces: HashMap<RequestId, Trace>,
    order: usize,
}

impl Harness {
    fn new(mut engine: Engine) -> Self {
        assert_eq!(
            engine.schedulers.len(),
            1,
            "test drives a single-scheduler engine"
        );
        let mut scheduler = engine.schedulers.remove(0);
        let rx = scheduler
            .handle
            .take_steps()
            .expect("a fresh scheduler yields its step stream once");
        Self {
            handle: Some(scheduler.handle),
            scheduler_join: Some(scheduler.join),
            rx,
            traces: HashMap::new(),
            order: 0,
        }
    }

    fn submit(
        &mut self,
        tag: impl Into<String>,
        prompt_tokens: Vec<u32>,
        output: (usize, usize),
    ) -> (RequestId, usize) {
        self.drain();
        let cutoff = self.order;
        let label = tag.into();
        let mut req = request(
            prompt_tokens,
            SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            output.0,
        );
        req.logprobs = output.1;
        req.client_label = Some(Arc::from(label.as_str()));
        let control = self
            .handle
            .as_ref()
            .expect("harness handle lives until drop")
            .submit(req);
        let id = control.id();
        assert!(
            self.traces
                .insert(
                    id,
                    Trace {
                        label,
                        ..Trace::default()
                    }
                )
                .is_none()
        );
        (id, cutoff)
    }

    /// Fold one request's step record into its trace, assigning each fact
    /// (admission, then every token) the next global order number.
    fn dispatch(&mut self, update: RequestUpdate) {
        let mut order = self.order;
        let trace = self
            .traces
            .get_mut(&update.id)
            .expect("update for unknown request id");
        if update.scheduled.is_some() {
            order += 1;
            trace.scheduled_orders.push(order);
        }
        for (i, &id) in update.tokens.iter().enumerate() {
            order += 1;
            let logprob = update
                .logprobs
                .get(i)
                .and_then(Option::as_ref)
                .map(|lp| lp.logprob);
            trace.samples.push((id, logprob));
            trace.token_orders.push(order);
        }
        match update.terminal {
            Some(Terminal::Finished { .. }) => trace.terminal = true,
            Some(Terminal::Failed { message, .. }) => {
                panic!("request {} failed: {message}", trace.label)
            }
            Some(Terminal::Rejected { reason, .. }) => {
                panic!("request {} rejected: {reason}", trace.label)
            }
            None => {}
        }
        self.order = order;
    }

    /// Block for the next step's outputs and fold every update in it.
    fn recv(&mut self) {
        let step = self.rx.blocking_recv().expect("engine step stream closed");
        for update in step.updates {
            self.dispatch(update);
        }
    }

    fn drain(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(step) => {
                    for update in step.updates {
                        self.dispatch(update);
                    }
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => panic!("engine step stream closed"),
            }
        }
    }

    fn barrier(&mut self) {
        self.drain();
        while self.traces.values().any(|trace| !trace.terminal) {
            self.recv();
        }
    }

    fn collect(&mut self, id: RequestId) -> Vec<Sample> {
        while !self.trace(id).terminal {
            self.recv();
        }
        self.trace(id).samples.clone()
    }

    fn trace(&self, id: RequestId) -> &Trace {
        self.traces.get(&id).expect("missing request trace")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        // Closing the intake lets the scheduler drain and exit.
        drop(self.handle.take());
        if let Some(join) = self.scheduler_join.take() {
            let _ = join.join();
        }
    }
}

fn model_path_or_skip() -> Option<String> {
    if let Ok(path) = std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Some(path)
    } else {
        eprintln!("skip batch_invariance_output: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        None
    }
}

fn start_pin_engine(model_path: &str) -> Engine {
    let engine = pegainfer_qwen3::start_engine_with_offload(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        pegainfer_qwen3::Qwen3OffloadOptions::disabled(),
        true,
        pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        pegainfer_qwen3::Qwen3MemoryOptions::default(),
        pegainfer_qwen3::DecodeOverlap::Off,
        true,
        None,
    )
    .expect("failed to start engine");
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Pin,
        "--batch-invariant did not select Pin"
    );
    engine
}

fn prompt(len: usize, row: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i + row) % 1000 + 10).collect()
}

fn first_divergence(expected: &[Sample], actual: &[Sample]) -> Option<usize> {
    let common = expected.len().min(actual.len());
    (0..common)
        .find(|&i| {
            expected[i].0 != actual[i].0
                || expected[i].1.map(f32::to_bits) != actual[i].1.map(f32::to_bits)
        })
        .or((expected.len() != actual.len()).then_some(common))
}

fn status(expected: &[Sample], actual: &[Sample]) -> String {
    first_divergence(expected, actual).map_or_else(
        || "identical".into(),
        |index| format!("first-divergence={index}"),
    )
}

fn report_divergence(label: &str, expected: &[Sample], actual: &[Sample]) {
    let Some(index) = first_divergence(expected, actual) else {
        return;
    };
    let start = index.saturating_sub(4);
    let end = (start + 8).min(expected.len().max(actual.len()));
    let expected_window = &expected[start.min(expected.len())..end.min(expected.len())];
    let actual_window = &actual[start.min(actual.len())..end.min(actual.len())];
    let expected_lp = expected.get(index).and_then(|sample| sample.1);
    let actual_lp = actual.get(index).and_then(|sample| sample.1);
    let delta = expected_lp.zip(actual_lp).map(|(left, right)| right - left);
    eprintln!(
        "[{label}] first-divergence={index} expected-window={expected_window:?} actual-window={actual_window:?} logprob-delta={delta:?} expected-logprob={expected_lp:?} actual-logprob={actual_lp:?}"
    );
}

fn assert_same(label: &str, expected: &[Sample], actual: &[Sample]) {
    report_divergence(label, expected, actual);
    let expected_ids: Vec<_> = expected.iter().map(|sample| sample.0).collect();
    let actual_ids: Vec<_> = actual.iter().map(|sample| sample.0).collect();
    assert_eq!(
        actual_ids, expected_ids,
        "{label}: token-id sequence drifted"
    );
    let expected_bits: Vec<_> = expected
        .iter()
        .map(|sample| sample.1.map(f32::to_bits))
        .collect();
    let actual_bits: Vec<_> = actual
        .iter()
        .map(|sample| sample.1.map(f32::to_bits))
        .collect();
    assert_eq!(
        actual_bits, expected_bits,
        "{label}: per-token logprob bits drifted"
    );
}

fn assert_probe_shape(label: &str, samples: &[Sample]) {
    assert_eq!(samples.len(), A_TOKENS, "{label}: wrong output length");
    assert!(
        samples.iter().all(|sample| sample.1.is_some()),
        "{label}: requested logprob is missing"
    );
}

fn phase_zero(harness: &mut Harness) -> Vec<Sample> {
    let (first_id, _) = harness.submit("p0-a-1", prompt(600, 1), (A_TOKENS, 1));
    let first = harness.collect(first_id);
    let (second_id, _) = harness.submit("p0-a-2", prompt(600, 1), (A_TOKENS, 1));
    let second = harness.collect(second_id);
    assert_probe_shape("phase 0 first run", &first);
    assert_probe_shape("phase 0 second run", &second);
    let divergence = first_divergence(&first, &second);
    eprintln!(
        "[phase 0 determinism] seq_len={} {}",
        second.len(),
        status(&first, &second)
    );
    if divergence.is_some() {
        report_divergence("phase 0 determinism", &first, &second);
        panic!("phase 0: output changed across identical runs; the gate is ill-defined in-process");
    }
    first
}

fn wait_for_load_tokens(harness: &mut Harness, loads: &[RequestId]) {
    while loads.iter().any(|&id| harness.trace(id).samples.is_empty()) {
        harness.recv();
    }
}

fn load_span(harness: &mut Harness, loads: &[RequestId], window: (usize, usize)) -> (usize, usize) {
    while loads.iter().any(|&id| {
        let trace = harness.trace(id);
        !trace.terminal && !trace.token_after(window.1)
    }) {
        harness.recv();
    }
    let before = loads
        .iter()
        .filter(|&&id| harness.trace(id).token_before(window.0))
        .count();
    let after = loads
        .iter()
        .filter(|&&id| harness.trace(id).token_after(window.1))
        .count();
    (before, after)
}

fn phase_one(harness: &mut Harness, baseline: &[Sample]) {
    let loads: Vec<_> = (0..LOAD_COUNT)
        .map(|i| {
            harness
                .submit(
                    format!("p1-load-{i}"),
                    prompt(550 + i * 2, 100 + i as u32),
                    (192, 0),
                )
                .0
        })
        .collect();
    wait_for_load_tokens(harness, &loads);
    let (a_id, _) = harness.submit("p1-a", prompt(600, 1), (A_TOKENS, 1));
    let actual = harness.collect(a_id);
    assert_probe_shape("phase 1 A", &actual);
    let first = harness.trace(a_id).token_orders[0];
    let last = *harness.trace(a_id).token_orders.last().unwrap();
    let (before, after) = load_span(harness, &loads, (first, last));
    assert_eq!(
        before, LOAD_COUNT,
        "phase 1 non-vacuity: only {before}/{LOAD_COUNT} loads emitted before A's first token"
    );
    assert_eq!(
        after, LOAD_COUNT,
        "phase 1 non-vacuity: only {after}/{LOAD_COUNT} loads emitted after A's last token"
    );
    eprintln!(
        "[phase 1 decode-heavy] seq_len={} {} loads-before={before} loads-after={after}",
        actual.len(),
        status(baseline, &actual)
    );
    assert_same(
        "phase 1 unified-prefill first token",
        &baseline[..1],
        &actual[..1],
    );
    assert_same("phase 1 decode suffix", &baseline[1..], &actual[1..]);
}

fn submit_burst(harness: &mut Harness, id: usize) -> RequestId {
    harness
        .submit(
            format!("p2-burst-{id}"),
            prompt(120 + id, 300 + id as u32),
            (4, 0),
        )
        .0
}

fn phase_two(harness: &mut Harness, baseline: &[Sample]) {
    let (a_id, _) = harness.submit("p2-a", prompt(600, 1), (A_TOKENS, 1));
    while harness.trace(a_id).samples.is_empty() {
        harness.recv();
    }
    let mut bursts: Vec<_> = (0..BURST_DEPTH)
        .map(|id| submit_burst(harness, id))
        .collect();
    let mut advanced = HashSet::new();
    let mut next = BURST_DEPTH;
    while !harness.trace(a_id).terminal {
        harness.recv();
        let ready = bursts.iter().position(|&id| {
            !advanced.contains(&id)
                && (!harness.trace(id).scheduled_orders.is_empty() || harness.trace(id).terminal)
        });
        if let Some(index) = ready {
            let id = bursts[index];
            advanced.insert(id);
            harness.drain();
            if !harness.trace(a_id).terminal {
                bursts.push(submit_burst(harness, next));
                next += 1;
            }
        }
    }
    let actual = harness.trace(a_id).samples.clone();
    assert_probe_shape("phase 2 A", &actual);
    let first = harness.trace(a_id).token_orders[0];
    let last = *harness.trace(a_id).token_orders.last().unwrap();
    let inside = bursts
        .iter()
        .flat_map(|&id| &harness.trace(id).scheduled_orders)
        .filter(|&&order| order > first && order < last)
        .count();
    assert!(
        inside >= MIN_BURSTS_IN_WINDOW,
        "phase 2 liveness: only {inside} burst prefills ran inside A's decode span — the \
         decode-under-prefill-load composition never materialized here"
    );
    eprintln!(
        "[phase 2 prefill-burst] seq_len={} {} scheduled-inside={inside}",
        actual.len(),
        status(baseline, &actual)
    );
    assert_same("phase 2 output", baseline, &actual);
}

fn phase_three(harness: &mut Harness) {
    // Phase 0's control prompt prefills in one chunk; this one does not. Re-establish determinism on
    // the chunked shape, or a wobble in the shape itself would read as batch-composition drift.
    let (base_id, _) = harness.submit("p3-a-long-alone", prompt(1500, 77), (A_TOKENS, 1));
    let baseline = harness.collect(base_id);
    assert_probe_shape("phase 3 A_long baseline", &baseline);
    harness.barrier();
    let (repeat_id, _) = harness.submit("p3-a-long-alone-2", prompt(1500, 77), (A_TOKENS, 1));
    let repeat = harness.collect(repeat_id);
    assert_probe_shape("phase 3 A_long control", &repeat);
    eprintln!(
        "[phase 3 control] A_long alone x2: {}",
        status(&baseline, &repeat)
    );
    assert_same("phase 3 A_long determinism control", &baseline, &repeat);
    harness.barrier();

    let (b1, _) = harness.submit("p3-b1", prompt(900, 91), (4, 0));
    let (b2, _) = harness.submit("p3-b2", prompt(900, 92), (4, 0));
    let (a_id, cutoff) = harness.submit("p3-a-long-batched", prompt(1500, 77), (A_TOKENS, 1));
    let actual = harness.collect(a_id);
    assert_probe_shape("phase 3 A_long batched", &actual);
    // A's first token lands only once its prefill completes, so this window is A's queue+prefill
    // span: a B scheduled inside it prefilled while A was queued or prefilling — necessary for the
    // two to contend for one step's budget, not proof of it.
    let first = harness.trace(a_id).token_orders[0];
    let b_scheduled = [b1, b2]
        .into_iter()
        .filter(|&id| harness.trace(id).scheduled_between(cutoff, first))
        .count();
    assert_eq!(
        b_scheduled, 2,
        "phase 3 liveness: only {b_scheduled}/2 B prefills overlapped A_long's prefill span — A_long \
         ran alone, so this phase exercised nothing"
    );
    eprintln!(
        "[phase 3 chunked-prefill] seq_len={} {} b-scheduled-before-first={b_scheduled}",
        actual.len(),
        status(&baseline, &actual)
    );
    assert_same("phase 3 output", &baseline, &actual);
}

#[test]
fn output_sequence_batch_invariant_under_pin() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _guard = POLICY_LOCK.lock().unwrap();
    let mut harness = Harness::new(start_pin_engine(&model_path));
    let baseline = phase_zero(&mut harness);
    phase_one(&mut harness, &baseline);
    harness.barrier();
    phase_two(&mut harness, &baseline);
    harness.barrier();
    phase_three(&mut harness);
    harness.barrier();
}
