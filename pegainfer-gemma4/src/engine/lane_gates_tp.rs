//! Tensor-parallel gates: a two-rank engine must agree with one rank on the
//! same checkpoint, prompts and batch. This is the check that the sharded
//! loader, the two reductions and the per-rank KV stay in step; it drives the
//! real launch path, not a hand-built stack.
//!
//! It compares *distributions*, not greedy tokens. A two-rank reduction sums
//! the same products in a different order than one rank does, and NCCL writes
//! bf16 back, so the logits differ in their last bits and a near-tie can flip
//! the greedy pick. The engine's own docs say as much — greedy is reproducible
//! for a fixed workload, not across two arithmetic orders — so the gate holds
//! the top-k logprobs to a line and requires a differing pick to be a genuine
//! near-tie (each pick inside the other run's top-k).
//!
//! Two GPUs are mandatory, so these run one at a time on a box that has them.

use std::collections::HashMap;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::parallel::ParallelConfig;

use super::lane_tests::Drained;
use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch_with;

/// How many top logprobs each run keeps, and the largest absolute logprob gap
/// two runs may show on a token they both kept. The gap is the bf16
/// reduction-order drift accumulated over the tower, not a shape difference.
const TOP_K: usize = 8;
const LOGBROB_LINE: f32 = 0.5;

/// The two devices a TP2 gate runs across, defaulting to 0 and 1.
fn devices() -> (usize, usize) {
    let read = |name: &str, fallback: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    };
    (read("PEGAINFER_TP_DEVICE", 0), read("PEGAINFER_TP_PEER", 1))
}

/// One rank, one device, eager.
fn single_options(device: usize) -> EngineLoadOptions {
    EngineLoadOptions {
        enable_cuda_graph: false,
        device_ordinals: vec![device],
        ..EngineLoadOptions::default()
    }
}

/// Two ranks, two devices. Graphs are on when `PEGAINFER_TP_GRAPH` is set, which
/// turns the gate into a parity check: one rank eager against two ranks
/// captured.
fn tp2_options(a: usize, b: usize) -> EngineLoadOptions {
    EngineLoadOptions {
        enable_cuda_graph: std::env::var("PEGAINFER_TP_GRAPH").is_ok(),
        device_ordinals: vec![a, b],
        parallel_config: Some(ParallelConfig::new(2, 1)),
        ..EngineLoadOptions::default()
    }
}

/// Serve the first prompt on its own, then the rest as one concurrent batch,
/// and shut the engine down once.
///
/// A single prompt in flight is the **solo** admission path (`step` plus
/// `prefill_extra_ranks`), which a batch never takes; draining it before the
/// others are submitted is what makes it a lone arrival. Both runs drive this
/// same sequence, so the per-prompt comparison stays index-aligned.
fn serve_batch(
    harness: &mut Harness,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    logprobs: usize,
) -> Vec<Drained> {
    let mut controls = Vec::with_capacity(prompts.len());
    let mut drained = Vec::with_capacity(prompts.len());
    if let Some(first) = prompts.first() {
        let control = harness.submit_scored(first.clone(), max_tokens, Some(logprobs), None);
        drained.push(harness.steps.drain(control.id(), "greedy solo"));
        controls.push(control);
    }
    for prompt in &prompts[1..] {
        controls.push(harness.submit_scored(prompt.clone(), max_tokens, Some(logprobs), None));
    }
    for control in &controls[1..] {
        drained.push(harness.steps.drain(control.id(), "greedy"));
    }
    let controls: Vec<&_> = controls.iter().collect();
    harness.shutdown(&controls);
    drained
}

fn prompts() -> Vec<Vec<u32>> {
    // `PEGAINFER_TP_PROMPT_TOKENS` widens or narrows the prompts; the default
    // is the three-prompt set the line compares at. `PEGAINFER_TP_PROMPTS`
    // narrows the count to exercise the solo admission path.
    let base: usize = std::env::var("PEGAINFER_TP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let count: usize = std::env::var("PEGAINFER_TP_PROMPTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    (0..count)
        .map(|i| ids(base + i * 7, 11 + i as u32))
        .collect()
}

fn top_of(lp: &TokenLogprob) -> HashMap<u32, f32> {
    lp.top_logprobs
        .iter()
        .map(|(token, value)| (*token, *value))
        .collect()
}

/// The worst shared top-k logprob gap over the run, and how many steps kept
/// the same pick. `Detail` names the step that produced the gap so a failure
/// can be read without a re-run.
struct Detail {
    request: usize,
    step: usize,
    token: u32,
    one: f32,
    two: f32,
    one_top: Vec<(u32, f32)>,
    two_top: Vec<(u32, f32)>,
}

struct Gaps {
    /// Largest gap over tokens both runs kept in their top-k. Inflated by a
    /// flattened distribution, where a shared token can sit at very different
    /// ranks in the two lists — read it beside `worst_pick`.
    worst_shared: f32,
    /// Largest gap on a token *both* runs picked: the like-for-like measure of
    /// how far the two distributions moved.
    worst_pick: f32,
    same_pick: usize,
    compared: usize,
    detail: Detail,
}

fn distribution_gap(one: &[Drained], two: &[Drained], what: &str) -> Gaps {
    let mut worst_shared = 0.0f32;
    let mut worst_pick = 0.0f32;
    let mut same_pick = 0usize;
    let mut compared = 0usize;
    let mut detail = Detail {
        request: 0,
        step: 0,
        token: 0,
        one: 0.0,
        two: 0.0,
        one_top: Vec::new(),
        two_top: Vec::new(),
    };
    for (index, (a, b)) in one.iter().zip(two).enumerate() {
        assert_eq!(
            a.logprobs.len(),
            b.logprobs.len(),
            "{what}: request {index} produced {} vs {} scored steps",
            a.logprobs.len(),
            b.logprobs.len()
        );
        for (step, (a, b)) in a.logprobs.iter().zip(&b.logprobs).enumerate() {
            let (Some(a), Some(b)) = (a, b) else {
                continue;
            };
            compared += 1;
            let (left, right) = (top_of(a), top_of(b));
            for (token, value) in &left {
                if let Some(other) = right.get(token) {
                    let gap = (value - other).abs();
                    if gap > worst_shared {
                        worst_shared = gap;
                        detail = Detail {
                            request: index,
                            step,
                            token: *token,
                            one: *value,
                            two: *other,
                            one_top: a.top_logprobs.clone(),
                            two_top: b.top_logprobs.clone(),
                        };
                    }
                }
            }
            let (pa, pb) = (a.top_logprobs[0].0, b.top_logprobs[0].0);
            if pa == pb {
                same_pick += 1;
                // The same token in both runs: the clean like-for-like measure
                // of how far the two distributions moved.
                if let (Some(va), Some(vb)) = (left.get(&pa), right.get(&pa)) {
                    worst_pick = worst_pick.max((va - vb).abs());
                }
            } else {
                // A different pick has to be a near-tie: each pick inside the
                // other run's top-k. Otherwise the two runs disagree about the
                // distribution, not about its last bits.
                assert!(
                    right.contains_key(&pa),
                    "{what}: request {index} step {step}: one-rank pick is outside the two-rank top-{TOP_K}"
                );
                assert!(
                    left.contains_key(&pb),
                    "{what}: request {index} step {step}: two-rank pick is outside the one-rank top-{TOP_K}"
                );
            }
        }
    }
    Gaps {
        worst_shared,
        worst_pick,
        same_pick,
        compared,
        detail,
    }
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, two GPUs, and --test-threads=1"]
fn the_two_rank_engine_matches_one_rank() {
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let prompts = prompts();
    // Serving knobs may be forced from the environment so the same gate runs
    // at a reduced envelope as well as at the default point.
    let slots_env = std::env::var("PEGAINFER_TP_SLOTS").ok();
    let ctx_env = std::env::var("PEGAINFER_TP_CTX").ok();
    let mut overrides: Vec<(&str, &str)> = Vec::new();
    if let Some(value) = slots_env.as_deref() {
        overrides.push((super::DECODE_SLOTS_ENV, value));
    }
    if let Some(value) = ctx_env.as_deref() {
        overrides.push((super::MAX_CONTEXT_ENV, value));
    }
    eprintln!("tp2: knob overrides {overrides:?}");

    // One rank first, and torn down before the two-rank engine takes the same
    // devices.
    let mut one = launch_with(&single_options(device), &overrides);
    let single = serve_batch(&mut one, &prompts, 16, TOP_K);
    drop(one);
    // The control: the same one-rank run again. If this is not bit-identical
    // the comparison below measures harness noise, not tensor parallelism.
    let mut one_again = launch_with(&single_options(device), &overrides);
    let repeat = serve_batch(&mut one_again, &prompts, 16, TOP_K);
    drop(one_again);
    let control = distribution_gap(&single, &repeat, "tp1-repeat");
    assert_eq!(
        control.worst_pick.to_bits(),
        0.0f32.to_bits(),
        "two one-rank runs disagree ({} on a picked token), so the two-rank comparison is \
         measuring nondeterminism",
        control.worst_pick
    );

    let mut two = launch_with(&tp2_options(device, peer), &[]);
    let tp2 = serve_batch(&mut two, &prompts, 16, TOP_K);
    drop(two);

    let gaps = distribution_gap(&single, &tp2, "tp2");
    eprintln!(
        "tp2: {}/{} steps keep the one-rank pick; worst picked-token logprob gap {:.4}",
        gaps.same_pick, gaps.compared, gaps.worst_pick
    );
    eprintln!(
        "tp2: worst shared-token {:.4} at request {} step {} token {} ({:.4} vs {:.4})",
        gaps.worst_shared,
        gaps.detail.request,
        gaps.detail.step,
        gaps.detail.token,
        gaps.detail.one,
        gaps.detail.two
    );
    eprintln!("tp2: one-rank top-{TOP_K} {:?}", gaps.detail.one_top);
    eprintln!("tp2: two-rank top-{TOP_K} {:?}", gaps.detail.two_top);
    assert!(
        gaps.worst_pick < LOGBROB_LINE,
        "two-rank and one-rank picked-token logprobs differ by {} (line {LOGBROB_LINE})",
        gaps.worst_pick
    );
}
