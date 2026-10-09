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
//! near-tie (each pick inside the other run's top-k *and* the two picks within
//! `NEAR_TIE_WIDTH` of each other in both readings).
//!
//! Two GPUs are mandatory, so these run one at a time on a box that has them.
//! The near-tie rule's own halves are the exception: they read nothing but the
//! two rows, so they carry unit tests that need no device and run with the
//! library's ordinary suite.

use std::collections::HashMap;
use std::time::Duration;

use pegainfer_core::tensor::DeviceContext;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::parallel::ParallelConfig;

use super::lane_tests::Drained;
use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch_with;
use super::lane_tests::wait_until;
use crate::testkit::f32_tensor;
use crate::testkit::golden_bytes;
use crate::testkit::i32_tensor;
use crate::testkit::u32_tensor;

/// How many top logprobs each run keeps, and the largest absolute logprob gap
/// two runs may show on a token they both kept. The gap is the bf16
/// reduction-order drift accumulated over the tower, not a shape difference.
///
/// The line is set from this gate's own readings — one procedure, one head:
/// four runs of the 12B on two L20s (`NCCL_PROTO=LL128`) each held
/// `worst_pick` to 0.5738, the same to four decimals every run, and one run
/// with the `o_proj` reduction skipped — the structural error this line is for
/// — drove the one-rank pick out of the two-rank top-8 (the near-tie rule fires
/// before the line is reached) and read `worst_top` 2.1895 with that rule
/// relaxed; `worst_pick` read 0 there, the two lists sharing no token, which is
/// the case the near-tie rule exists for. 1.0 is ~1.7x over the floor and ~2.2x
/// under the fault, and is the value `DRIFT_LINE` below already uses. The shape
/// follows the repo's one precedent for calibrating a quantity of this kind,
/// `serve_oracle`'s `neutral_scale`: two algorithms over one context, 0.31..5.75
/// observed and a line at 12.0.
///
/// What the line is for: a structural tensor-parallel error — a wrong shard, a
/// missing reduction, a rank out of step — moves a logprob by many nats and
/// flips most picks, not 2 of 24. Reduction noise is what the near-tie rule
/// beside it polices.
const TOP_K: usize = 8;
const LOGBROB_LINE: f32 = 1.0;

/// How far apart the two runs' picks may read **within the same run** and still
/// be a tie: that run's own two entries, read in both runs.
///
/// Top-`TOP_K` membership alone is not a tie — `[6, 0, …]` against `[0, 6, …]`
/// passes it while each pick sits six nats from its own runner-up — but the
/// bound folded in beside this rule, `worst_pick < LOGBROB_LINE`, *does* catch
/// that shape: both picks' own gaps fold in, and a flip that wide has moved one
/// of them. What the width adds is the band between the two: a margin just over
/// the width riding a near-zero same-token mean (0.9 and 1.05 apart with 0.975
/// gaps passes the line and fails this), a tie test that stands on its own, and
/// a failure that names the two picks and both margins.
///
/// The margin is a difference inside one row, so it is the same quantity on
/// log_softmax (here) as on raw logits, which is where `serve_oracle`'s rule
/// reads it — and that side has only a whole-row 12.0 line, so there the width
/// is what catches the six-nat shape as well.
const NEAR_TIE_WIDTH: f32 = 1.0;

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

/// Two ranks, two devices. Eager, because a tensor-parallel launch refuses
/// graphs: NCCL's abort waits for a captured graph that references the
/// communicator to be destroyed, and a rank stopped inside a collective cannot
/// release its own — so a captured collective would make the failure path
/// unresolvable. The engine refuses them at launch
/// (`engine.rs`, "CUDA graphs are unsupported under tensor parallelism"), which
/// is what this arm has to match.
fn tp2_options(a: usize, b: usize) -> EngineLoadOptions {
    EngineLoadOptions {
        enable_cuda_graph: false,
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

/// Every value a gap is taken from has to be finite. One NaN logit folds the
/// row's logsumexp — and with it every logprob on that row — into NaN while the
/// top-k *ids* stay right, and both `f32::max` and `gap > worst` step over a NaN
/// without moving the running bound, so a gate would pass on a row it exists to
/// fail. `serve_oracle::compare_row` asserts the same over both arms.
pub(super) fn assert_finite(lp: &TokenLogprob, at: &str) {
    assert!(
        lp.logprob.is_finite() && lp.top_logprobs.iter().all(|(_, value)| value.is_finite()),
        "{at} scored a non-finite logprob: {} with top {:?}",
        lp.logprob,
        lp.top_logprobs
    );
}

/// Two runs of the same one-rank workload must come out the same token for token
/// and bit for bit. This is the control's bar: a gap metric that only bounds a
/// tolerance can pass while two tokens trade places, so the control compares the
/// runs themselves and the two-rank comparison below is left measuring tensor
/// parallelism rather than harness noise.
fn assert_identical(one: &[Drained], two: &[Drained], what: &str) {
    assert_eq!(
        one.len(),
        two.len(),
        "{what}: {} vs {} requests",
        one.len(),
        two.len()
    );
    for (index, (a, b)) in one.iter().zip(two).enumerate() {
        assert_eq!(
            a.ids, b.ids,
            "{what}: request {index} decoded a different token"
        );
        assert_eq!(
            a.logprobs, b.logprobs,
            "{what}: request {index} scored different logprobs"
        );
    }
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
    /// Largest gap on a token *both* runs picked, and on either token when the
    /// picks differ: the like-for-like measure of how far the two distributions
    /// moved. A flip is where a large move hides — the two tops can still read
    /// equal — so the flipped picks are measured too, not skipped.
    worst_pick: f32,
    /// Largest gap between the two runs' *top* logprobs, whichever token holds
    /// the top. A flip keeps this at zero, so it is a bound beside `worst_pick`,
    /// not what catches one.
    worst_top: f32,
    same_pick: usize,
    compared: usize,
    detail: Detail,
}

fn distribution_gap(one: &[Drained], two: &[Drained], what: &str) -> Gaps {
    let mut worst_shared = 0.0f32;
    let mut worst_pick = 0.0f32;
    let mut worst_top = 0.0f32;
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
            assert_finite(
                a,
                &format!("{what}: request {index} step {step} (one rank)"),
            );
            assert_finite(
                b,
                &format!("{what}: request {index} step {step} (two rank)"),
            );
            let (left, right) = (top_of(a), top_of(b));
            worst_top = worst_top.max((a.top_logprobs[0].1 - b.top_logprobs[0].1).abs());
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
                // A different pick has to be a near-tie, in both halves: each
                // pick inside the other run's top-k *and* the two picks within
                // `NEAR_TIE_WIDTH` of each other in each run's own reading.
                // Otherwise the two runs disagree about the distribution, not
                // about its last bits.
                assert!(
                    right.contains_key(&pa),
                    "{what}: request {index} step {step}: one-rank pick is outside the two-rank top-{TOP_K}"
                );
                assert!(
                    left.contains_key(&pb),
                    "{what}: request {index} step {step}: two-rank pick is outside the one-rank top-{TOP_K}"
                );
                // Both picks now sit in both lists, so each one's own gap is
                // measurable, and a flip is exactly where a large move hides:
                // the two tops can read equal while the same token moved far.
                // Fold both in before the break, or the flip is the one step
                // that leaves the run with no bound at all.
                for token in [pa, pb] {
                    if let (Some(va), Some(vb)) = (left.get(&token), right.get(&token)) {
                        worst_pick = worst_pick.max((va - vb).abs());
                    }
                }
                // Membership is not a tie on its own: `[6, 0, …]` against
                // `[0, 6, …]` has each pick as the other's runner-up while both
                // sit six nats from their own runner-up. The bound folded in
                // above catches that shape (a flip that wide has moved a pick),
                // but not a margin just over the width riding a near-zero
                // same-token mean — so read the margin in each run's own list.
                let (one_margin, two_margin) = (
                    (left[&pa] - left[&pb]).abs(),
                    (right[&pa] - right[&pb]).abs(),
                );
                assert!(
                    one_margin <= NEAR_TIE_WIDTH && two_margin <= NEAR_TIE_WIDTH,
                    "{what}: request {index} step {step}: the picks {pa} and {pb} sit \
                     {one_margin} and {two_margin} apart in the one-rank and two-rank \
                     readings, wider than the {NEAR_TIE_WIDTH} a reduction-order flip moves \
                     them (the picked-token bound, both picks folded in, reads {worst_pick} \
                     against {LOGBROB_LINE}) — each pick is inside the other run's \
                     top-{TOP_K}, so this margin is what tells a tie from a different \
                     distribution"
                );
                eprintln!(
                    "{what}: near-tie flip {pa} vs {pb}, margins {one_margin} / {two_margin}"
                );
                // The two runs now decode from different prefixes, so every later
                // step of this request compares different text; stop here.
                break;
            }
        }
    }
    Gaps {
        worst_shared,
        worst_pick,
        worst_top,
        same_pick,
        compared,
        detail,
    }
}

/// A run of hand-written top-k rows, one per scored step: the rule reads nothing
/// but `logprobs`, so nothing else about the rows has to be real.
fn drained_run(rows: &[&[(u32, f32)]]) -> Drained {
    Drained {
        tokens: rows.len(),
        cached: 0,
        finish: FinishReason::Length,
        ids: rows.iter().map(|row| row[0].0).collect(),
        logprobs: rows.iter().map(|row| Some(scored(row))).collect(),
        prompt_echo: None,
    }
}

/// One scored row, its top entry taken as the row's own logprob.
fn scored(row: &[(u32, f32)]) -> TokenLogprob {
    TokenLogprob {
        logprob: row[0].1,
        rank: 1,
        top_logprobs: row.to_vec(),
    }
}

/// A flip six nats wide: each pick is the other run's runner-up, so mutual
/// top-`TOP_K` membership holds. The gate's own bound catches this shape as well
/// — the folded `worst_pick` reads 6 against `LOGBROB_LINE` — so this test is
/// the membership half's failure, and the one below is what the width adds.
#[test]
#[should_panic(expected = "apart in the one-rank and two-rank readings")]
fn a_flip_six_nats_wide_is_not_a_near_tie() {
    let one = vec![drained_run(&[&[(100, 6.0), (200, 0.0)]])];
    let two = vec![drained_run(&[&[(200, 6.0), (100, 0.0)]])];
    distribution_gap(&one, &two, "six nats");
}

/// The band only the width rejects: the margins are 0.9 and 1.05 apart while both
/// picks' own gaps fold to 0.975 — inside `LOGBROB_LINE`, so the gate's line
/// passes this shape and the margin is the whole rule.
#[test]
#[should_panic(expected = "apart in the one-rank and two-rank readings")]
fn a_flip_the_line_cannot_see_is_still_not_a_tie() {
    let one = vec![drained_run(&[&[(100, 0.9), (200, 0.0)]])];
    let two = vec![drained_run(&[&[(200, 0.975), (100, -0.075)]])];
    distribution_gap(&one, &two, "the band");
}

/// A tie in one run only: the one-rank row's top two sit half a nat apart, the
/// two-rank row prefers `200` by six and a half. Membership holds on both sides
/// again, so the width has to be read in both rows for this to fail.
#[test]
#[should_panic(expected = "apart in the one-rank and two-rank readings")]
fn a_tie_in_only_one_of_the_two_runs_is_not_a_near_tie() {
    let one = vec![drained_run(&[&[(100, 0.0), (200, -0.5)]])];
    let two = vec![drained_run(&[&[(200, 6.0), (100, -0.5)]])];
    distribution_gap(&one, &two, "one run");
}

/// The shape the rule is for, and the bound a flip must not escape: both picks'
/// own gaps fold in before the request stops, so the step that flipped is
/// measured rather than skipped — and the request's later steps, where the two
/// runs are decoding different text, are not compared at all.
#[test]
fn a_flip_inside_the_width_still_measures_its_own_step() {
    let one = vec![drained_run(&[
        &[(100, 0.0), (200, -0.2)],
        &[(100, 9.0), (200, 8.0)],
    ])];
    let two = vec![drained_run(&[
        &[(200, 0.1), (100, -0.3)],
        &[(100, -9.0), (200, -8.0)],
    ])];
    let gaps = distribution_gap(&one, &two, "unit");
    assert_eq!(gaps.compared, 1, "the request stops at the flip");
    assert_eq!(
        gaps.same_pick, 0,
        "the step that flipped is not a shared pick"
    );
    assert!(
        (gaps.worst_pick - 0.3).abs() < 1e-6,
        "both picks' own gaps fold in (0.3 each), read {}",
        gaps.worst_pick
    );
}

/// Two ranks must agree with one rank on the same checkpoint. Which
/// tensor-parallel branch it exercises is the checkpoint's business: the shipped
/// 12B (a single global KV head) takes the **replicate** branch, a
/// 31B-geometry checkpoint (`G % P == 0`) takes the **shard** branch that
/// production serves. Point `PEGAINFER_TEST_MODEL_PATH` at whichever is under
/// test, and note that a full-depth 31B cannot be gated here at all: the
/// single-rank control has to fit one card, which 57 GiB of weights do not.
#[test]
#[ignore = "needs two GPUs and --test-threads=1; checkpoint from PEGAINFER_TEST_MODEL_PATH"]
fn the_two_rank_engine_matches_one_rank() {
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let prompts = prompts();
    // Serving knobs may be forced from the environment so the same gate runs
    // at a reduced envelope as well as at the default point.
    let overrides = envelope_overrides();
    let overrides = as_refs(&overrides);
    eprintln!("tp2: knob overrides {overrides:?}");

    // One rank first, and torn down before the two-rank engine takes the same
    // devices.
    let mut one = launch_with(&single_options(device), &overrides);
    let single = serve_batch(&mut one, &prompts, 16, TOP_K);
    drop(one);
    // The control: the same one-rank run again, and it must come out
    // **bit-identical**. A metric that only holds a tolerance can pass while a
    // token trades places with another, so compare the runs themselves.
    let mut one_again = launch_with(&single_options(device), &overrides);
    let repeat = serve_batch(&mut one_again, &prompts, 16, TOP_K);
    drop(one_again);
    assert_identical(&single, &repeat, "tp1-repeat");

    let mut two = launch_with(&tp2_options(device, peer), &overrides);
    let tp2 = serve_batch(&mut two, &prompts, 16, TOP_K);
    drop(two);

    // `single == repeat` bit for bit, so comparing tp2 against the repeat run
    // would repeat this comparison; one of them is enough.
    let gaps = distribution_gap(&single, &tp2, "tp2");
    // A reading is only comparable with the build and pair it came from, so
    // name both.
    eprintln!(
        "tp2: NCCL {}, devices {device},{peer}",
        super::nccl_version()
    );
    eprintln!(
        "tp2: {}/{} steps keep the one-rank pick; worst picked-token logprob gap {:.4} (top {:.4})",
        gaps.same_pick, gaps.compared, gaps.worst_pick, gaps.worst_top
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
        gaps.worst_pick < LOGBROB_LINE && gaps.worst_top < LOGBROB_LINE,
        "two-rank and one-rank logprobs differ by {} on a picked token / {} at the top \
         (line {LOGBROB_LINE})",
        gaps.worst_pick,
        gaps.worst_top
    );
}

/// The serving knobs a gate may force from the environment, so the same gate
/// runs at a reduced envelope as well as at the default point — a 31B needs a
/// smaller one to fit a 48 GiB card at all.
fn envelope_overrides() -> Vec<(&'static str, String)> {
    let mut overrides = Vec::new();
    for (var, knob) in [
        ("PEGAINFER_TP_SLOTS", super::DECODE_SLOTS_ENV),
        ("PEGAINFER_TP_CTX", super::MAX_CONTEXT_ENV),
    ] {
        if let Ok(value) = std::env::var(var) {
            overrides.push((knob, value));
        }
    }
    overrides
}

fn as_refs<'a>(overrides: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
    overrides
        .iter()
        .map(|(knob, value)| (*knob, value.as_str()))
        .collect()
}

/// A request that asks for its prompt's logprobs is scored through a path the
/// sampled gate never touches — rank 0 reads the head's rows back to the host
/// inside its own step — so it gets its own two-rank gate.
#[test]
#[ignore = "needs two GPUs and --test-threads=1"]
fn the_two_rank_engine_scores_prompt_logprobs() {
    const TOP_K: usize = 8;
    // A mid-length prompt, so the scored path runs a real prefill per rank rather
    // than a single-token one. `PEGAINFER_TP_PROMPT_TOKENS` probes another point.
    let len: usize = std::env::var("PEGAINFER_TP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let mut harness = launch_with(&tp2_options(device, peer), &as_refs(&envelope_overrides()));
    // Both submitted before either is drained, so the second scored admission
    // arrives while the first still holds KV — the path a single submit misses.
    let prompts: Vec<Vec<u32>> = [11u32, 26].into_iter().map(|seed| ids(len, seed)).collect();
    let controls: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            // The first request asks for more than one token, so it is still
            // decoding — a live batch — when the second is admitted.
            let max_tokens = if index == 0 { 4 } else { 1 };
            harness.submit_scored(prompt.clone(), max_tokens, Some(TOP_K), Some(TOP_K))
        })
        .collect();
    for (control, prompt) in controls.iter().zip(&prompts) {
        let drained = harness.steps.drain(control.id(), "scored prompt");
        let echo = drained.prompt_echo.expect("the engine echoes the prompt");
        assert_eq!(
            echo.logprobs.len(),
            prompt.len(),
            "one scored row per prompt token"
        );
    }
    let refs: Vec<&_> = controls.iter().collect();
    harness.shutdown(&refs);
}

/// A causal model's row `r` depends only on tokens `..= r`, so the same prompt's
/// early rows must score the same whether the prompt is run whole or cut short.
/// They will not be bit-identical — the prefill GEMM's tiling, and so its
/// accumulation order, depends on the row count — so the gate bounds the shared
/// rows by the same chaotic-shape line `serve_oracle` calibrates (`PREFIX_LINE`),
/// not by the one-rank comparison's tighter one. A **large** gap here is the
/// engine's own long-context bug (a page, position or rope error), with no
/// reference implementation involved.
///
/// `PEGAINFER_TP_PROMPT_TOKENS` sets the long length (default 1024, the window
/// width); the short run is its first half.
/// What one prefix comparison read: how many rows lined up, how many of those
/// shared a token at all (the gap is only read where they do), the worst
/// shared-token gap and where it was, and how the argmaxes moved.
struct PrefixComparison {
    compared: usize,
    overlapped: usize,
    /// Rows whose top-`TOP_K` shared no token at all. They contribute nothing to
    /// `worst`, so they are counted rather than left to pass the gate on one
    /// shared row elsewhere: a widespread error — a page, position or rope fault
    /// — is mostly rows like this.
    disjoint: usize,
    worst: f32,
    worst_at: String,
    flipped: usize,
    first_flip: String,
}

impl PrefixComparison {
    fn report(&self, what: &str, long: usize) {
        eprintln!(
            "{what}: {} vs {} tokens: {} rows, {} with a shared token, {} sharing none; worst \
             shared-token gap {:.4} at {}; {} argmax flips (first {})",
            long / 2,
            long,
            self.compared,
            self.overlapped,
            self.disjoint,
            self.worst,
            self.worst_at,
            self.flipped,
            self.first_flip
        );
    }

    /// The properties every arm needs before its gap means anything: rows were
    /// compared, at least one of them shared a token with the other run, and the
    /// rows that shared *nothing* are not the bulk of them — without that the gap
    /// is a zero nobody measured.
    fn assert_reads_something(&self, what: &str, long: usize) {
        assert!(
            self.compared > 0,
            "{what}: no shared rows were scored at {} vs {long} tokens",
            long / 2
        );
        assert!(
            self.overlapped > 0,
            "{what}: no row's top-{TOP_K} was shared between the {}- and {long}-token runs, so \
             the gap measured nothing",
            long / 2
        );
        // A widespread error makes most rows share nothing, and those rows are
        // invisible to `worst`: this is the bound that sees them. Read against
        // the reading, which is 0 of 511 rows at 1024 tokens — a quarter is far
        // above any run this gate has taken and far below a fault that moves
        // every page.
        assert!(
            self.disjoint * 4 <= self.compared,
            "{what}: {} of {} rows share no token between the {}- and {long}-token runs, more \
             than a quarter of them — a disagreement that widespread is not the row-count \
             dependence this line is calibrated for, and those rows never reach `worst`",
            self.disjoint,
            self.compared,
            long / 2
        );
    }
}

/// Score `long` and its first half on one engine, and hold the rows they share
/// against each other. Split out of the gate below so the two-rank arm can be
/// held against a single-rank control on the same prompt.
fn prefix_comparison(options: &EngineLoadOptions, long: usize) -> PrefixComparison {
    const SALT: u32 = 11;
    let short = long / 2;
    let mut harness = launch_with(options, &as_refs(&envelope_overrides()));
    let mut controls = Vec::new();
    let mut echoes = Vec::new();
    // `ids` is position-indexed, so the short prompt is a prefix of the long one.
    for len in [short, long] {
        let prompt = ids(len, SALT);
        let control = harness.submit_scored(prompt.clone(), 1, Some(TOP_K), Some(TOP_K));
        let drained = harness.steps.drain(control.id(), "prefix");
        echoes.push(drained.prompt_echo.expect("the engine echoes the prompt"));
        controls.push(control);
    }
    let (short_echo, long_echo) = (&echoes[0], &echoes[1]);
    let mut comparison = PrefixComparison {
        compared: 0,
        overlapped: 0,
        disjoint: 0,
        worst: 0.0,
        worst_at: String::new(),
        flipped: 0,
        first_flip: String::new(),
    };
    for row in 0..short - 1 {
        let (Some(a), Some(b)) = (
            short_echo
                .logprobs
                .get(row + 1)
                .and_then(|entry| entry.as_ref()),
            long_echo
                .logprobs
                .get(row + 1)
                .and_then(|entry| entry.as_ref()),
        ) else {
            continue;
        };
        comparison.compared += 1;
        assert_finite(a, &format!("{short}-token run row {row} (engine)"));
        assert_finite(b, &format!("{long}-token run row {row} (engine)"));
        let (left, right) = (top_of(a), top_of(b));
        let mut shared = 0usize;
        for (token, value) in &left {
            if let Some(other) = right.get(token) {
                shared += 1;
                let gap = (value - other).abs();
                if gap > comparison.worst {
                    comparison.worst = gap;
                    comparison.worst_at = format!("row {row} token {token}");
                }
            }
        }
        if shared > 0 {
            comparison.overlapped += 1;
        } else {
            comparison.disjoint += 1;
        }
        let (pa, pb) = (a.top_logprobs[0].0, b.top_logprobs[0].0);
        if pa != pb {
            comparison.flipped += 1;
            if comparison.first_flip.is_empty() {
                comparison.first_flip =
                    format!("row {row}: {short}-token picks {pa}, {long}-token picks {pb}");
            }
        }
    }
    let refs: Vec<&_> = controls.iter().collect();
    harness.shutdown(&refs);
    comparison
}

#[test]
#[ignore = "needs two GPUs and --test-threads=1; checkpoint from PEGAINFER_TEST_MODEL_PATH"]
fn the_two_rank_engine_is_prefix_consistent() {
    let long: usize = std::env::var("PEGAINFER_TP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    // `long / 2 == 0` would make `0..short - 1` wrap in release and spin instead
    // of reaching the assert below.
    assert!(
        long >= 4,
        "the prefix gate needs a prompt of at least 4 tokens"
    );
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let two = prefix_comparison(&tp2_options(device, peer), long);
    two.report("prefix tp2", long);
    two.assert_reads_something("prefix tp2", long);
    // The line is `serve_oracle`'s calibration, not a tight one: scoring the same
    // context two ways (there a greedy walk against a single prefill, here a
    // 512-row prefill against a 1024-row one) moves the logits by a chaotically
    // amplifying amount — measured in-repo at 0.31..5.75 raw logits — because the
    // prefill GEMM's tiling depends on the row count. Measured here: 1.69 on the
    // four-layer synthetic, 4.05 on the 60-layer 31B. A grosser gap (a page,
    // position or rope error) still fails. A checkpoint that fits one card gets
    // the tighter, measured form of this line and the absolute check below
    // together — see `...within_a_single_rank_control`.
    assert!(
        two.worst < PREFIX_LINE,
        "the same prefix scored in a longer run differs by {} at {} (line {PREFIX_LINE}), so the \
         engine is not prefix-consistent",
        two.worst,
        two.worst_at
    );
}

/// The prefix gate's two-rank line, with a control arm. `serve_oracle`'s 12.0
/// bounds raw logits there, held beside an argmax check on every row; here it
/// bounds logprob gaps with the flips only counted, so the line needs a
/// measurement of its own rather than a borrowed one — the same discipline
/// #1132's cache gate applies to its cold leg.
///
/// The control is the *same comparison at world size 1*, so the checkpoint has
/// to fit one card (the 12B; a gate cannot load a 31B twice). It reads what the
/// shape difference is worth with no tensor parallelism in the picture, and the
/// two-rank arm may not multiply it.
#[test]
#[ignore = "needs two GPUs, a one-card checkpoint, and --test-threads=1"]
fn the_two_rank_engine_is_prefix_consistent_within_a_single_rank_control() {
    let long: usize = std::env::var("PEGAINFER_TP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    assert!(
        long >= 4,
        "the prefix gate needs a prompt of at least 4 tokens"
    );
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let control = prefix_comparison(&single_options(device), long);
    control.report("prefix control (one rank)", long);
    control.assert_reads_something("prefix control", long);
    let two = prefix_comparison(&tp2_options(device, peer), long);
    two.report("prefix tp2", long);
    two.assert_reads_something("prefix tp2", long);
    let ceiling = (PREFIX_CONTROL_RATIO * control.worst).max(PREFIX_CONTROL_FLOOR);
    assert!(
        two.worst <= ceiling,
        "at two ranks the same prefix differs by {} at {}, against a one-rank control's {} \
         (ceiling {ceiling} = {PREFIX_CONTROL_RATIO}x control, floor {PREFIX_CONTROL_FLOOR})",
        two.worst,
        two.worst_at,
        control.worst
    );
}

/// The prefix gate's line, on the `serve_oracle` scale (`DRIFT_LINE = 12.0`
/// there) rather than the one-rank comparison's `LOGBROB_LINE`: the drift it
/// looks for is the same chaotic shape-dependence, bounded the same way.
const PREFIX_LINE: f32 = 12.0;

/// The one-card form of the same line: what the two-rank arm may read against
/// the single-rank control's own reading. `serve_oracle` derives its lines the
/// same way when it can afford the control arm — a ratio with a floor, so a
/// control that happened to read exactly zero cannot tighten the bound to
/// nothing.
///
/// Measured on the 12B at the window width (512 vs 1024 tokens): the control
/// reads **6.30** nat and the two-rank arm **9.23**, a ratio of **1.46** — the
/// two-rank drift at that depth is the same phenomenon, not a multiple of it.
/// The ratio is 2.0 rather than 1.5 so a re-run's scatter does not flake the
/// gate; the absolute `PREFIX_LINE` cannot see either reading.
const PREFIX_CONTROL_RATIO: f32 = 2.0;
const PREFIX_CONTROL_FLOOR: f32 = 0.10;

/// The abort machinery's own contract, with no engine in the picture: a
/// reduction still works through the wrapper, an aborted communicator refuses
/// the next one instead of running comm-less, and aborting twice does not reach
/// cudarc's drop path twice — a second `ncclCommAbort` makes cudarc's `Comm`
/// `Drop` panic, which is why the wrapper takes the communicator out rather
/// than leaving the abort to the drop. No checkpoint, so this is the cheap half
/// of the pair; the engine-level half is
/// `a_failed_rank_zero_segment_stops_the_engine_at_two_ranks`, and the release
/// the engine depends on is its own gate beside it,
/// `an_aborted_communicator_releases_a_waiting_reduction`.
#[test]
#[ignore = "needs two GPUs and --test-threads=1"]
fn an_aborted_communicator_refuses_the_next_reduction() {
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let contexts: Vec<DeviceContext> = [device, peer]
        .into_iter()
        .map(|ordinal| DeviceContext::new_with_device(ordinal).expect("a device context"))
        .collect();
    let comms = cudarc::nccl::safe::Comm::from_devices(
        contexts
            .iter()
            .map(|ctx| ctx.stream.clone())
            .collect::<Vec<_>>(),
    )
    .expect("a two-rank communicator");
    let ranks: Vec<crate::layer::TpComm> =
        comms.into_iter().map(crate::layer::TpComm::new).collect();
    let mut buffers: Vec<_> = contexts
        .iter()
        .map(|ctx| ctx.stream.alloc_zeros::<f32>(8).expect("a buffer"))
        .collect();
    // The pair reduces together once, before any abort, each rank's enqueue on
    // its own device — the shape the engine runs and the one the wrapper must
    // not have broken.
    for (index, (rank, buffer)) in ranks.iter().zip(buffers.iter_mut()).enumerate() {
        super::select_device(&contexts[index]).expect("the rank's device is current");
        rank.all_reduce_in_place(buffer, &cudarc::nccl::safe::ReduceOp::Sum)
            .expect("a reduction over a live communicator");
    }
    for ctx in &contexts {
        ctx.stream
            .synchronize()
            .expect("the pair's reduction completes");
    }
    ranks[1].abort();
    ranks[1].abort();
    assert!(
        ranks[1]
            .all_reduce_in_place(&mut buffers[1], &cudarc::nccl::safe::ReduceOp::Sum)
            .is_err(),
        "a reduction after the abort must refuse instead of reducing comm-less"
    );
}

/// The half of the same contract the gate above cannot see. There the pair has
/// already reduced and synchronized before anything is aborted, so the abort
/// only has to refuse what comes *next*; the engine aborts a peer's communicator
/// while that peer is still **inside** a collective (`EngineState::drive_ranks`)
/// and needs the abort to release it, because nothing else can — the peer's own
/// thread is the one that is blocked.
///
/// So this enqueues a reduction on rank 1 alone, with no matching call from rank
/// 0, and aborts rank 1 from another thread: the shape a peer is left in when
/// the other rank's segment stops early. Both halves are bounded, because either
/// can be the failure — the wait ending is what the engine's `ctx.sync()`
/// returns on, and an `ncclCommAbort` that never returns would hang the failing
/// rank's thread before it could report anything. No checkpoint, no engine and
/// no injection hook.
#[test]
#[ignore = "needs two GPUs and --test-threads=1"]
fn an_aborted_communicator_releases_a_waiting_reduction() {
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let contexts: Vec<DeviceContext> = [device, peer]
        .into_iter()
        .map(|ordinal| DeviceContext::new_with_device(ordinal).expect("a device context"))
        .collect();
    let comms = cudarc::nccl::safe::Comm::from_devices(
        contexts
            .iter()
            .map(|ctx| ctx.stream.clone())
            .collect::<Vec<_>>(),
    )
    .expect("a two-rank communicator");
    let ranks: Vec<crate::layer::TpComm> =
        comms.into_iter().map(crate::layer::TpComm::new).collect();
    let mut buffers: Vec<_> = contexts
        .iter()
        .map(|ctx| ctx.stream.alloc_zeros::<f32>(8).expect("a buffer"))
        .collect();
    // The abort runs on a thread of its own, so this thread stays free to watch
    // the peer's stream; the context is re-bound there by the same call a step's
    // thread uses.
    let peer_ctx = DeviceContext {
        ctx: contexts[1].ctx.clone(),
        stream: contexts[1].stream.clone(),
        device_ordinal: contexts[1].device_ordinal,
    };

    super::select_device(&contexts[1]).expect("the peer's device is current");
    ranks[1]
        .all_reduce_in_place(&mut buffers[1], &cudarc::nccl::safe::ReduceOp::Sum)
        .expect("the enqueue itself succeeds");
    // The control arm: with no peer call to pair with, the reduction cannot
    // complete, so what follows measures the abort rather than a reduction that
    // finished by itself.
    assert!(
        stream_waiting(&peer_ctx),
        "rank 1's unpaired reduction left its stream idle, so this run could not \
         measure what an abort releases"
    );

    let (tx, rx) = std::sync::mpsc::channel();
    let (device_ctx, stream_ctx) = (contexts[1].ctx.clone(), contexts[1].stream.clone());
    let ordinal = contexts[1].device_ordinal;
    let aborting_rank = ranks[1].clone();
    std::thread::spawn(move || {
        let ctx = DeviceContext {
            ctx: device_ctx,
            stream: stream_ctx,
            device_ordinal: ordinal,
        };
        let bound = super::bind_engine_thread(&ctx);
        if bound.is_ok() {
            aborting_rank.abort();
        }
        let _ = tx.send(bound.map(|_guard| ()));
    });
    let released = wait_until(Duration::from_secs(30), || !stream_waiting(&peer_ctx));
    let aborting = rx.recv_timeout(Duration::from_secs(30));
    assert!(
        released,
        "rank 1's stream was still waiting on a reduction with no peer 30 s after its \
         communicator was aborted; the engine's failure path relies on that abort releasing a \
         peer already inside a collective"
    );
    assert!(
        matches!(aborting, Ok(Ok(()))),
        "aborting rank 1's communicator did not return within 30 s while a collective was in \
         flight on it ({aborting:?}); the engine calls it from the failing rank's thread, which \
         would hang there instead of stopping the engine"
    );
}

/// Whether `ctx`'s stream still holds a reduction queued — the state a peer waits
/// in until its communicator is aborted. Any other answer, success or the error a
/// torn-down collective leaves behind, means the wait is over.
fn stream_waiting(ctx: &DeviceContext) -> bool {
    matches!(
        unsafe { cudarc::driver::sys::cuStreamQuery(ctx.stream.cu_stream()) },
        cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY
    )
}

/// A rank-0 step failure at two ranks must stop the engine promptly, not wedge
/// it — the pairing the single-rank gate cannot see. The engine aborts every
/// rank's communicator before joining the ranks' threads
/// (`EngineState::drive_ranks`), and this is the end-to-end half of that: the
/// step fails, the scheduler exits within the deadline, and the request is
/// failed rather than left pending.
///
/// The fault needs no injection hook — `u32::MAX` is outside every vocabulary
/// this line ships, and `prepare_single`'s `validate_tokens` refuses it before
/// the tower allocates anything — but it is deliberately *symmetric*: every rank
/// runs the same segment and so refuses the same prompt, which means a peer that
/// never launches its collective is not what this run creates. The abort handle
/// itself is held to its own contract, without an engine, by
/// `an_aborted_communicator_refuses_the_next_reduction`; an asymmetric injection
/// would need a hook in the engine, which the single-rank half
/// (`lane_gates_logprobs::a_failed_prefill_costs_that_request_not_the_engine`)
/// also went out of its way not to add.
#[test]
#[ignore = "needs two GPUs and --test-threads=1; checkpoint from PEGAINFER_TEST_MODEL_PATH"]
fn a_failed_rank_zero_segment_stops_the_engine_at_two_ranks() {
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let mut harness = launch_with(&tp2_options(device, peer), &as_refs(&envelope_overrides()));
    let bad = harness.submit(vec![9, u32::MAX, 11], 4);
    // Bounded on purpose: the property is that the engine *stops*, and a peer
    // left spinning on a collective would keep this waiting instead — which is
    // the wedge this gate exists to name.
    let stopped = wait_until(Duration::from_secs(60), || harness.scheduler_finished());
    let terminal = harness.steps.try_terminal(bad.id());
    assert!(
        stopped,
        "a rank-0 failure at TP2 left the engine running (request terminal {terminal:?}); the \
         peer's communicator should have been aborted before the join"
    );
    assert!(
        matches!(terminal, Some(Terminal::Failed { .. })),
        "the failing request should be failed, not {terminal:?}"
    );
    harness.shutdown(&[]);
}

/// Two ranks against the Hugging Face reference, for the size the gate above
/// cannot afford a single-rank control on: a 31B's whole tower is 57 GiB, so its
/// baseline is impossible on one card and the *reference* is the baseline here.
/// The fixture is that checkpoint's own HF dump
/// (`tools/accuracy/dump_gemma4_hf_golden.py`, selected with
/// `PEGAINFER_GEMMA4_GOLDEN`), compared over the teacher-forced prompt rows'
/// top-64 logprobs.
///
/// What it does not cover: the fixture's layer-boundary probes (the serving path
/// exposes logits, not activations) and its `single` case — row `r` predicts
/// token `r + 1`, so a one-token prompt has no row to score.
#[test]
#[ignore = "needs two GPUs, a 31B fixture in PEGAINFER_GEMMA4_GOLDEN, and --test-threads=1"]
fn the_two_rank_engine_matches_the_hf_reference() {
    /// The fixture's rows are the reference's `log_softmax` over softcapped
    /// logits. A shared token may sit this far apart: bf16 reduction order and a
    /// different attention backend, nothing structural.
    const DRIFT_LINE: f32 = 1.0;
    // Only the nine-token case is asserted, so **long prompts at TP2 are
    // unverified**: running the fixture's 1024-token `edge` fails a strict top-64
    // containment on ~5-8 rows, and those rows fail at one rank as well as at
    // two — which says nothing about the second rank either way, since a failure
    // both arms share is explained by neither. Pinning that depth needs the
    // fixtures' per-case tolerance discipline, not a containment test.
    // (`probed_cases` lists the cases carrying hidden-state probes; `edge` has
    // none because at window width they would dwarf the file, so its absence
    // says nothing about whether the case is asserted.) See
    // docs/models/gemma4/tp.md "Known bounds".
    const CASES: [&str; 1] = ["short"];

    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let dir = crate::testkit::model_path();
    // `golden_bytes` holds the fixture against the checkpoint's own file digests
    // and fails loudly on a mismatch, so the wrong pair cannot be compared.
    let (bytes, manifest) = golden_bytes(&dir);
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("fixture");
    eprintln!(
        "hf: {} at revision {}, {}",
        manifest["model_class"], manifest["revision"], manifest["dtypes"]
    );

    let mut harness = launch_with(&tp2_options(device, peer), &as_refs(&envelope_overrides()));
    let mut controls = Vec::new();
    let mut compared = 0usize;
    let mut same_pick = 0usize;
    let mut outside = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for case in CASES {
        let (_, tokens) = u32_tensor(&fixture, &format!("{case}_tokens"));
        let (shape, ids) = i32_tensor(&fixture, &format!("{case}_topk_ids"));
        let (_, lps) = f32_tensor(&fixture, &format!("{case}_topk_logprobs"));
        let top_k = shape[1];
        assert_eq!(
            shape[0],
            tokens.len(),
            "{case}: one reference row per token"
        );
        let control = harness.submit_scored(tokens.clone(), 1, Some(top_k), Some(top_k));
        let drained = harness.steps.drain(control.id(), case);
        let echo = drained.prompt_echo.expect("the engine echoes the prompt");
        for row in 0..tokens.len() - 1 {
            // Row 0 of the echo is the head's own placeholder, so the row that
            // predicts token `r + 1` sits at `r + 1`.
            let ours = echo
                .logprobs
                .get(row + 1)
                .and_then(|entry| entry.as_ref())
                .unwrap_or_else(|| panic!("{case}: row {row} was not scored"));
            let theirs: Vec<(u32, f32)> = (0..top_k)
                .map(|k| (ids[row * top_k + k] as u32, lps[row * top_k + k]))
                .collect();
            assert!(
                theirs.iter().all(|(_, value)| value.is_finite()),
                "{case} row {row}: the reference fixture carries a non-finite logprob"
            );
            assert_finite(ours, &format!("{case} row {row} (engine)"));
            let ours_top = top_of(ours);
            let (pa, pb) = (ours.top_logprobs[0].0, theirs[0].0);
            compared += 1;
            if pa == pb {
                same_pick += 1;
            } else {
                // A differing pick has to be a near-tie: each pick inside the
                // other run's top-k. Every offender is collected rather than
                // panicking on the first, so one run names them all.
                if !theirs.iter().any(|(token, _)| *token == pa) {
                    outside += 1;
                    if offenders.len() < 10 {
                        offenders.push(format!(
                            "{case} row {row}: engine pick {pa} at {:.4}, reference top-1 {pb} at \
                             {:.4}, reference top-{top_k} floor {:.4}",
                            ours.top_logprobs[0].1,
                            theirs[0].1,
                            theirs.last().expect("non-empty top-k").1
                        ));
                    }
                }
                if !ours_top.contains_key(&pb) {
                    outside += 1;
                    if offenders.len() < 10 {
                        offenders.push(format!(
                            "{case} row {row}: reference pick {pb} outside the engine's top-{top_k}"
                        ));
                    }
                }
            }
            for (token, value) in &theirs {
                if let Some(other) = ours_top.get(token) {
                    let gap = (value - other).abs();
                    if gap > worst {
                        worst = gap;
                        worst_at = format!("{case} row {row} token {token}");
                    }
                }
            }
        }
        controls.push(control);
    }
    eprintln!(
        "hf: {same_pick}/{compared} rows keep the reference's pick; {outside} out-of-top-k; worst \
         shared-token logprob gap {worst:.4} at {worst_at}"
    );
    for offender in &offenders {
        eprintln!("hf: outside: {offender}");
    }
    let controls: Vec<&_> = controls.iter().collect();
    harness.shutdown(&controls);
    assert!(
        outside == 0,
        "{outside} rows have a pick outside the other's top-k; first up to 10: {offenders:#?}"
    );
    assert!(
        worst < DRIFT_LINE,
        "the two-rank engine and the reference differ on {worst_at} by {worst} (line {DRIFT_LINE})"
    );
}
