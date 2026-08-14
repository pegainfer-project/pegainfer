//! Kimi-K3 decode executor gates against the certified reference engine.
//!
//! The fixture in `tests/fixtures/k3_4l_greedy.json` is a greedy replay of the
//! reference engine over a layer-truncated model: a fixed 16-token prompt fed
//! one token per step, then 24 steps feeding the argmax back. It records the
//! argmax and the top-5 (id, logit) of every step, so a mismatch localizes to
//! the step that first diverged instead of only to the continuation.
//!
//! Manual gates: CI compiles them, a Blackwell box with the checkpoint runs
//! them with `--ignored`. Point `PEGAINFER_K3_TEST_224` at the 224-expert
//! checkpoint directory (`PEGAINFER_K3_TEST_DEVICE` picks the GPU).
//!
//! The reference and this executor differ in exactly two places — the dense
//! projections merge one cuBLASLt partial instead of eight split-K segments,
//! and the routed experts run a quantized grouped GEMM instead of per-row MXFP4
//! GEMVs — so the gate is a token match, not a bit match.
//!
//! Both routed-expert transports are held to this fixture. Production is the
//! fused MegaMoE kernel; the masked chain is the anchor it was A/B'd against
//! and is reachable only from test code, so gate 1 runs it explicitly. They are
//! not bit-equal to each other (the fused kernel folds the routing weight in
//! before the down projection and mid-quantizes per 32 rather than per 128), so
//! each is compared to the fixture rather than to the other.
//!
//! The second difference sets a measured noise floor. Quantizing the expert
//! activations to FP8 with power-of-two group scales costs about 0.7% on the
//! routed-expert output, which reaches the logits as a median deviation of two
//! bf16 ULP. A step whose reference top-1 leads top-2 by no more than that is a
//! coin flip for any implementation that is not bit-identical, so the fixture
//! comparison spends exactly that much tolerance and no more: a step may
//! disagree only when the reference's own margin is inside the noise floor, and
//! even then the sampled token must be one the reference ranked in its top 5.
//! Every other step must match exactly.
//!
//! Row independence and graph equivalence are held to a stricter standard,
//! because they are about the executor agreeing with *itself*: same bucket,
//! same shapes, so nothing may move at all. Bucket width is the one exception —
//! it changes the N the dense projections hand cuBLASLt, which changes the tile
//! shapes and their summation order, so two buckets separate by the same noise
//! floor and are compared through the fixture rather than to each other.

use std::path::PathBuf;

use pegainfer_k3::DecodeSlot;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::K3MoeTransport;
use pegainfer_k3::StepExecutor;

/// The noise floor, in bf16 ULP at the logit's magnitude. Measured by replaying
/// the whole fixture and comparing every logit the reference published: the
/// median step lands within two ULP of the reference, and the worst within
/// eleven. Two is what the fixture comparison spends on a step the reference
/// itself decided by less; everything above it must match exactly.
const NOISE_FLOOR_ULP: f32 = 2.0;

const FIXTURE: &str = include_str!("fixtures/k3_4l_greedy.json");
const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";
const DEVICE_ENV: &str = "PEGAINFER_K3_TEST_DEVICE";

struct Golden {
    prompt: Vec<u32>,
    /// Argmax after every step, prompt steps included.
    argmax: Vec<u32>,
    top5_ids: Vec<Vec<u32>>,
    top5_logits: Vec<Vec<f32>>,
    num_layers: usize,
    max_ctx: usize,
}

fn golden() -> Golden {
    let json: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let ids = |value: &serde_json::Value| -> Vec<u32> {
        value
            .as_array()
            .expect("array")
            .iter()
            .map(|entry| entry.as_u64().expect("token id") as u32)
            .collect()
    };
    let steps = json["steps"].as_array().expect("steps array");
    Golden {
        prompt: ids(&json["prompt"]),
        argmax: steps
            .iter()
            .map(|step| step["argmax"].as_u64().expect("argmax") as u32)
            .collect(),
        top5_ids: steps.iter().map(|step| ids(&step["top5_ids"])).collect(),
        top5_logits: steps
            .iter()
            .map(|step| {
                step["top5_logits"]
                    .as_array()
                    .expect("array")
                    .iter()
                    .map(|entry| entry.as_f64().expect("logit") as f32)
                    .collect()
            })
            .collect(),
        num_layers: json["num_layers"].as_u64().expect("num_layers") as usize,
        max_ctx: json["max_ctx"].as_u64().expect("max_ctx") as usize,
    }
}

/// The checkpoint directory, or `None` when this box does not have it mounted.
fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(CHECKPOINT_ENV).ok()?);
    path.join("config.json").exists().then_some(path)
}

fn device() -> usize {
    std::env::var(DEVICE_ENV)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0)
}

/// The two routed-expert transports, named for the gate reports.
const TRANSPORTS: [(&str, K3MoeTransport); 2] = [
    ("mega", K3MoeTransport::MEGA),
    ("masked chain", K3MoeTransport::masked_chain_for_tests()),
];

/// Build a single-rank executor over the truncated model the fixture pins.
/// All routed experts stay local, which is what the fixture was produced with.
fn executor(
    golden: &Golden,
    max_batch: usize,
    cuda_graph: bool,
    moe_transport: K3MoeTransport,
) -> Option<K3Executor> {
    let path = checkpoint()?;
    let config = K3ExecutorConfig {
        max_batch,
        max_ctx: golden.max_ctx,
        num_layers: golden.num_layers,
        cuda_graph,
        moe_transport,
    };
    Some(
        K3Executor::load(&path, device(), 0, 1, config)
            .expect("the truncated rank model should load"),
    )
}

impl Golden {
    /// The token fed at every step of the fixture: the prompt one token at a
    /// time, then the fixture's own argmax. Feeding this instead of our own
    /// sample keeps a step's inputs independent of the previous step's outcome,
    /// so a disagreement localizes to the step that produced it.
    fn feed(&self) -> Vec<u32> {
        let mut feed = self.prompt.clone();
        feed.extend(
            self.argmax[self.prompt.len() - 1..self.argmax.len() - 1]
                .iter()
                .copied(),
        );
        feed
    }

    /// How far the reference's top-1 leads its top-2 at this step, in units of
    /// the bf16 spacing at that magnitude.
    fn margin_ulp(&self, step: usize) -> f32 {
        let top = self.top5_logits[step][0];
        let ulp = f32::from_bits((top.abs().to_bits() & 0x7f80_0000).max(1)) / 128.0;
        (top - self.top5_logits[step][1]) / ulp
    }

    /// A step the reference itself decided by no more than the measured noise
    /// floor. No implementation that is not bit-identical can be held to these.
    fn is_coin_flip(&self, step: usize) -> bool {
        self.margin_ulp(step) <= NOISE_FLOOR_ULP
    }
}

/// Compare a replay against the fixture step by step. Exact everywhere the
/// reference had a real margin; inside the noise floor the sampled token only
/// has to be one the reference ranked in its top 5. Reports every step that
/// spends tolerance, and the first hard failure with the reference's top-5.
fn assert_fixture_match(golden: &Golden, sampled: &[u32], what: &str) {
    assert_eq!(
        sampled.len(),
        golden.argmax.len(),
        "{what}: replayed {} steps, fixture has {}",
        sampled.len(),
        golden.argmax.len()
    );
    let feed = golden.feed();
    let mut spent = 0usize;
    for (step, (&got, &want)) in sampled.iter().zip(&golden.argmax).enumerate() {
        if got == want {
            continue;
        }
        let excused = golden.is_coin_flip(step) && golden.top5_ids[step].contains(&got);
        assert!(
            excused,
            "{what}: step {step} (fed {}) sampled {got}, fixture says {want}; \
             reference margin {:.2} bf16 ULP, top-5 ids {:?} logits {:?}",
            feed[step],
            golden.margin_ulp(step),
            golden.top5_ids[step],
            golden.top5_logits[step]
        );
        spent += 1;
        eprintln!(
            "{what}: step {step} is a coin flip (reference margin {:.2} bf16 ULP): \
             sampled {got}, fixture says {want}, reference top-5 {:?}",
            golden.margin_ulp(step),
            golden.top5_ids[step]
        );
    }
    eprintln!(
        "{what}: {}/{} steps match the fixture exactly, {spent} inside the noise floor",
        sampled.len() - spent,
        sampled.len()
    );
}

/// Gates 2 to 4 hold the executor against itself, so they get no tolerance.
fn assert_same_trajectory(baseline: &[u32], sampled: &[u32], what: &str) {
    assert_eq!(
        sampled.len(),
        baseline.len(),
        "{what}: produced {} steps, the reference run has {}",
        sampled.len(),
        baseline.len()
    );
    for (step, (&got, &want)) in sampled.iter().zip(baseline).enumerate() {
        assert_eq!(
            got, want,
            "{what}: step {step} sampled {got}, the reference run sampled {want}"
        );
    }
    eprintln!(
        "{what}: all {} steps match the reference run",
        sampled.len()
    );
}

/// The bucket-1 eager greedy trajectory — the executor's own reference for the
/// gates that test it against itself. Built on a throwaway executor so the
/// batched, captured or shared-bucket run under test cannot influence it.
fn baseline_trajectory(golden: &Golden) -> Option<Vec<u32>> {
    let mut executor = executor(golden, 1, false, K3MoeTransport::MEGA)?;
    let steps = golden.argmax.len() - golden.prompt.len();
    Some(
        executor
            .greedy_replay(0, &golden.prompt, steps)
            .expect("the baseline replay should run"),
    )
}

/// Run the fixture's greedy replay in every listed slot at once, one lockstep
/// batch per step. The highest listed slot picks the bucket, so listing only
/// slot 7 leaves seven padding rows and listing 0..8 leaves none — same shapes,
/// different neighbours, which is exactly the contrast row independence is.
fn replica_run(executor: &mut K3Executor, golden: &Golden, slots: &[usize]) -> Vec<Vec<u32>> {
    for slot in slots.iter().copied() {
        executor.release(slot);
    }
    let mut produced: Vec<Vec<u32>> = vec![Vec::new(); slots.len()];
    for step in 0..golden.argmax.len() {
        let batch: Vec<DecodeSlot> = slots
            .iter()
            .copied()
            .enumerate()
            .map(|(index, slot)| DecodeSlot {
                slot,
                last_token: if step < golden.prompt.len() {
                    golden.prompt[step]
                } else {
                    produced[index][step - 1]
                },
            })
            .collect();
        let tokens = executor
            .decode(&batch)
            .expect("the batched step should run");
        for (stream, token) in produced.iter_mut().zip(tokens) {
            stream.push(token);
        }
    }
    produced
}

/// Gate 1 — the end-to-end token match against the reference engine at bucket
/// 1, eagerly. Every one of the fixture's 40 steps is replayed on its own
/// inputs, so this is 40 independent verdicts on the whole 4-layer forward
/// pass rather than one verdict on a continuation.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn golden_replay_matches_the_reference_at_bucket_one() {
    let golden = golden();
    for (name, transport) in TRANSPORTS {
        let Some(mut executor) = executor(&golden, 1, false, transport) else {
            eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
            return;
        };
        let sampled = executor
            .forced_replay(0, &golden.feed())
            .expect("the replay should run");
        assert_fixture_match(&golden, &sampled, &format!("bucket 1, eager, {name}"));
    }
}

/// Gate 1b — the same fixture under greedy feedback, where a single flip
/// re-roots the continuation. It must hold up to the first step the reference
/// decided inside the noise floor; past that the two engines are answering
/// different questions and comparing them says nothing.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn golden_greedy_replay_matches_up_to_the_first_coin_flip() {
    let golden = golden();
    let Some(mut executor) = executor(&golden, 1, false, K3MoeTransport::MEGA) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let steps = golden.argmax.len() - golden.prompt.len();
    let sampled = executor
        .greedy_replay(0, &golden.prompt, steps)
        .expect("the replay should run");
    let shared = sampled
        .iter()
        .zip(&golden.argmax)
        .take_while(|(got, want)| got == want)
        .count();
    if shared < sampled.len() {
        let step = shared;
        assert!(
            golden.is_coin_flip(step) && golden.top5_ids[step].contains(&sampled[step]),
            "greedy step {step} sampled {}, fixture says {}; reference margin {:.2} bf16 ULP, \
             top-5 ids {:?} logits {:?}",
            sampled[step],
            golden.argmax[step],
            golden.margin_ulp(step),
            golden.top5_ids[step],
            golden.top5_logits[step]
        );
    }
    eprintln!(
        "greedy: {shared} of {} leading steps match the fixture{}",
        golden.argmax.len(),
        if shared < sampled.len() {
            format!(
                "; step {shared} is a coin flip (reference margin {:.2} bf16 ULP) and re-roots \
                 the continuation",
                golden.margin_ulp(shared)
            )
        } else {
            String::new()
        }
    );
}

/// Gate 2 — row independence inside a bucket. The same replay runs three ways
/// in bucket 8: eight times over with no padding at all, four times in the top
/// half, and once alone in slot 7 behind seven padding rows. Neighbours change
/// which rows the batched kernels touch and which experts the masked chain
/// activates; the seat's own tokens must not notice. All three runs keep slot 7
/// as the highest seat and so share the bucket, which makes this a verdict on
/// rows leaking rather than on batch-size numerics.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn golden_greedy_replay_is_row_independent() {
    let golden = golden();
    let Some(mut executor) = executor(&golden, 8, false, K3MoeTransport::MEGA) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let all_eight = replica_run(&mut executor, &golden, &[0, 1, 2, 3, 4, 5, 6, 7]);
    for (index, stream) in all_eight.iter().enumerate().skip(1) {
        assert_same_trajectory(
            &all_eight[0],
            stream,
            &format!("bucket 8, row {index} of 8"),
        );
    }
    // Half the rows real, half padding.
    let top_half = replica_run(&mut executor, &golden, &[4, 5, 6, 7]);
    for (index, slot) in [4usize, 5, 6, 7].into_iter().enumerate() {
        assert_same_trajectory(
            &all_eight[slot],
            &top_half[index],
            &format!("bucket 8, slot {slot} of four"),
        );
    }
    // Slot 7 still picks bucket 8; rows 0..7 are now padding rows carrying a
    // token nobody asked for.
    let lone_seven = replica_run(&mut executor, &golden, &[7]);
    assert_same_trajectory(&all_eight[7], &lone_seven[0], "bucket 8, slot 7 alone");
}

/// Gate 2b — how far apart two buckets land on the same sequence. Not a row
/// independence question: the dense projections hand cuBLASLt a different N per
/// bucket, so the tiles and their summation order change and the trajectories
/// separate by the same noise floor the reference comparison spends. Held to
/// the fixture with that tolerance rather than to each other exactly.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn a_wider_bucket_stays_inside_the_noise_floor() {
    let golden = golden();
    let Some(mut executor) = executor(&golden, 8, false, K3MoeTransport::MEGA) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let sampled = executor
        .forced_replay(7, &golden.feed())
        .expect("the replay should run");
    assert_fixture_match(&golden, &sampled, "bucket 8, eager");
}

/// Gate 3 — graphs change nothing. A captured step must produce the very same
/// tokens as the eager step it was captured from, at the same bucket, and must
/// keep producing them once there is no capture left to do. Each bucket holds
/// two graphs (one per state parity), so a 40-step replay exercises both.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn golden_greedy_replay_matches_through_cuda_graphs() {
    let golden = golden();
    let steps = golden.argmax.len() - golden.prompt.len();
    for (max_batch, slot) in [(1usize, 0usize), (8, 7)] {
        let Some(mut eager) = executor(&golden, max_batch, false, K3MoeTransport::MEGA) else {
            eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
            return;
        };
        let reference = eager
            .greedy_replay(slot, &golden.prompt, steps)
            .expect("the eager replay should run");
        drop(eager);

        let mut executor = executor(&golden, max_batch, true, K3MoeTransport::MEGA)
            .expect("the checkpoint was there a moment ago");
        let sampled = executor
            .greedy_replay(slot, &golden.prompt, steps)
            .expect("the replay should run");
        assert_same_trajectory(&reference, &sampled, &format!("bucket {max_batch}, graphs"));
        // A second replay on the same executor runs entirely off the captured
        // graphs, with no capture left to do.
        let again = executor
            .greedy_replay(slot, &golden.prompt, steps)
            .expect("the replayed graph should run again");
        assert_same_trajectory(
            &reference,
            &again,
            &format!("bucket {max_batch}, graph replay"),
        );
    }
}

/// Gate 3b — capture and replay every bucket up to the executor's width, so a
/// bucket whose kernels were never instantiated fails here rather than in
/// production. Tokens are not checked; this is a liveness and corruption gate.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn cuda_graphs_capture_and_replay_every_bucket() {
    let golden = golden();
    let Some(mut executor) = executor(&golden, 128, true, K3MoeTransport::MEGA) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    for highest in [0usize, 1, 3, 7, 15, 31, 47, 63, 95, 127] {
        let batch: Vec<DecodeSlot> = (0..=highest)
            .map(|slot| DecodeSlot {
                slot,
                last_token: golden.prompt[slot % golden.prompt.len()],
            })
            .collect();
        for slot in 0..=highest {
            executor.release(slot);
        }
        // Two steps: the first captures the parity-0 graph, the second the
        // parity-1 graph, and both must produce a token for every row.
        for _ in 0..2 {
            let tokens = executor.decode(&batch).expect("the bucket should step");
            assert_eq!(tokens.len(), batch.len(), "bucket covering slot {highest}");
            assert!(
                tokens.iter().all(|token| (*token as usize) < 163_840),
                "bucket covering slot {highest} sampled an out-of-vocab id"
            );
        }
    }
    eprintln!("every bucket captured and replayed");
}

/// Gate 4 — four sequences at staggered starts in one bucket. Each stream must
/// reproduce the single-slot replay, which is only possible if the per-slot
/// context lengths, cache windows and recurrent states stay separated.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn four_concurrent_sequences_each_match_the_baseline() {
    let golden = golden();
    let Some(baseline) = baseline_trajectory(&golden) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut executor = executor(&golden, 4, true, K3MoeTransport::MEGA)
        .expect("the checkpoint was there a moment ago");
    let total = golden.argmax.len();
    // Slot i starts i steps after slot 0.
    let stagger = [0usize, 3, 7, 11];
    let mut produced: Vec<Vec<u32>> = vec![Vec::new(); stagger.len()];

    let horizon = total + stagger.iter().copied().max().expect("non-empty");
    for tick in 0..horizon {
        let mut batch = Vec::new();
        for (slot, start) in stagger.iter().copied().enumerate() {
            if tick < start {
                continue;
            }
            if tick == start {
                // A seat is only clean at the moment a sequence takes it: until
                // then it is a padding row, and padding rows are stepped like
                // any other. This is the same handover the serving path makes,
                // where `prefill` resets the seat before its first decode.
                executor.release(slot);
            }
            let index = tick - start;
            if index >= total {
                continue;
            }
            let last_token = if index < golden.prompt.len() {
                golden.prompt[index]
            } else {
                produced[slot][index - 1]
            };
            batch.push((slot, DecodeSlot { slot, last_token }));
        }
        if batch.is_empty() {
            continue;
        }
        let slots: Vec<usize> = batch.iter().map(|(slot, _)| *slot).collect();
        let entries: Vec<DecodeSlot> = batch.into_iter().map(|(_, entry)| entry).collect();
        let tokens = executor
            .decode(&entries)
            .expect("the batched step should run");
        for (slot, token) in slots.into_iter().zip(tokens) {
            produced[slot].push(token);
        }
    }

    for (slot, stream) in produced.iter().enumerate() {
        assert_same_trajectory(&baseline, stream, &format!("slot {slot} of four"));
    }
}

/// The serving path: `prefill` ingests the prompt and returns the request's
/// first token, then `decode` continues it. Both must agree with the fixture.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn prefill_then_decode_serves_the_baseline_continuation() {
    let golden = golden();
    let Some(baseline) = baseline_trajectory(&golden) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut executor = executor(&golden, 1, true, K3MoeTransport::MEGA)
        .expect("the checkpoint was there a moment ago");
    let params = pegainfer_frontend::sampler::SamplingParams::default();
    let first = executor
        .prefill(0, &golden.prompt, &params)
        .expect("prefill should run");
    let boundary = golden.prompt.len() - 1;
    assert_eq!(
        first, baseline[boundary],
        "prefill's first token: got {first}, the decode path sampled {}",
        baseline[boundary]
    );
    let mut last = first;
    for (step, &want) in baseline.iter().enumerate().skip(golden.prompt.len()) {
        let tokens = executor
            .decode(&[DecodeSlot {
                slot: 0,
                last_token: last,
            }])
            .expect("decode should run");
        last = tokens[0];
        assert_eq!(
            last, want,
            "decode step {step}: got {last}, the decode path sampled {want}"
        );
    }
    eprintln!("prefill + decode reproduced the decode-only continuation");
}

/// Localization aid, not a gate: feed the fixture's own token sequence at every
/// step so a step's inputs never depend on the previous step's sample, and
/// report each step's argmax and top-5 next to the reference's. A run that
/// agrees here but diverges under greedy feedback split on a near-tie; a run
/// that disagrees here has drifted.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn forced_replay_reports_per_step_agreement() {
    let golden = golden();
    let Some(mut executor) = executor(&golden, 1, false, K3MoeTransport::MEGA) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let feed = golden.feed();
    let mut disagreements = 0usize;
    for (step, token) in feed.iter().copied().enumerate() {
        let sampled = if step == 0 {
            executor
                .forced_replay(0, &feed[..1])
                .expect("replay should run")[0]
        } else {
            executor
                .decode(&[DecodeSlot {
                    slot: 0,
                    last_token: token,
                }])
                .expect("decode should run")[0]
        };
        let logits = executor.last_logits(0).expect("logits readback");
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_by(|a, b| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        });
        let top5: Vec<(u32, f32)> = order[..5]
            .iter()
            .map(|id| (*id, logits[*id as usize]))
            .collect();
        let want = golden.argmax[step];
        let agree = sampled == want;
        if !agree {
            disagreements += 1;
        }
        eprintln!(
            "step {step:>2} fed {token:>6} -> {sampled:>6} (fixture {want:>6}) {} \
             margin {:.2} ULP\n  ours  {top5:?}\n  ref   ids {:?} logits {:?}",
            if agree { "ok" } else { "MISMATCH" },
            golden.margin_ulp(step),
            golden.top5_ids[step],
            golden.top5_logits[step]
        );
    }
    eprintln!(
        "forced replay: {}/{} steps agree",
        golden.argmax.len() - disagreements,
        golden.argmax.len()
    );
}

/// A step-time snapshot, not a gate: eager against captured, at the narrowest,
/// a mid and the widest bucket. Run over the truncated model the fixture pins, so the
/// numbers say what the launch sequence costs per layer and what capture buys
/// back, not what the whole model will cost.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn step_time_snapshot() {
    let golden = golden();
    for max_batch in [1usize, 16, 128] {
        for cuda_graph in [false, true] {
            let Some(mut executor) = executor(&golden, max_batch, cuda_graph, K3MoeTransport::MEGA)
            else {
                eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
                return;
            };
            let batch: Vec<DecodeSlot> = (0..max_batch)
                .map(|slot| DecodeSlot {
                    slot,
                    last_token: golden.prompt[slot % golden.prompt.len()],
                })
                .collect();
            for slot in 0..max_batch {
                executor.release(slot);
            }
            // Warm up: both parities' graphs get captured, and the first
            // eager step pays for whatever the driver lazily builds.
            for _ in 0..8 {
                executor.decode(&batch).expect("warmup step");
            }
            let rounds = 64;
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                executor.decode(&batch).expect("timed step");
            }
            let per_step = started.elapsed().as_secs_f64() * 1e3 / rounds as f64;
            eprintln!(
                "bucket {max_batch:>3}, {:<7}: {per_step:.3} ms/step over {} layers, \
                 {:.3} ms per layer, {:.1} tok/s",
                if cuda_graph { "graphs" } else { "eager" },
                golden.num_layers,
                per_step / golden.num_layers as f64,
                max_batch as f64 * 1e3 / per_step
            );
        }
    }
}
