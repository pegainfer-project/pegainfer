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
use crate::testkit::f32_tensor;
use crate::testkit::golden_bytes;
use crate::testkit::i32_tensor;
use crate::testkit::u32_tensor;

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
    /// Largest gap between the two runs' *top* logprobs, whichever token holds
    /// the top. This is what catches a step whose pick flipped: the picks then
    /// differ, `worst_pick` stays zero, but the mass under the pick still has
    /// to have moved little.
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
    // The control: the same one-rank run again. If this is not bit-identical
    // the comparison below measures harness noise, not tensor parallelism.
    let mut one_again = launch_with(&single_options(device), &overrides);
    let repeat = serve_batch(&mut one_again, &prompts, 16, TOP_K);
    drop(one_again);
    let control = distribution_gap(&single, &repeat, "tp1-repeat");
    assert!(
        control.worst_pick.to_bits() == 0.0f32.to_bits()
            && control.worst_top.to_bits() == 0.0f32.to_bits(),
        "two one-rank runs disagree ({:.4} on a picked token, {:.4} at the top), so the \
         two-rank comparison is measuring nondeterminism",
        control.worst_pick,
        control.worst_top
    );

    let mut two = launch_with(&tp2_options(device, peer), &overrides);
    let tp2 = serve_batch(&mut two, &prompts, 16, TOP_K);
    drop(two);

    // The control pins `single == repeat` bit for bit, so comparing tp2 against
    // the repeat run would repeat this comparison; one of them is enough.
    let gaps = distribution_gap(&single, &tp2, "tp2");
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
        ("PEGAINFER_TP_MAX_PROMPT", super::TP_MAX_PROMPT_ENV),
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
    // The gate's prompt is the TP ceiling itself, so the length the guard
    // refuses past is the length this passes. `PEGAINFER_TP_PROMPT_TOKENS`
    // overrides it to probe another point.
    let len: usize = std::env::var("PEGAINFER_TP_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(super::TP_MAX_PROMPT);
    let (device, peer) = devices();
    assert_ne!(device, peer, "TP2 needs two distinct device ordinals");
    let mut harness = launch_with(&tp2_options(device, peer), &as_refs(&envelope_overrides()));
    // Both submitted before either is drained, so the second scored admission
    // arrives while the first still holds KV — the path a single submit misses.
    let prompts: Vec<Vec<u32>> = [11u32, 26].into_iter().map(|seed| ids(len, seed)).collect();
    let controls: Vec<_> = prompts
        .iter()
        .map(|prompt| harness.submit_scored(prompt.clone(), 1, Some(TOP_K), Some(TOP_K)))
        .collect();
    for (control, prompt) in controls.iter().zip(&prompts) {
        let drained = harness.steps.drain(control.id(), "scored prompt");
        let echo = drained.prompt_echo.expect("the engine echoes the prompt");
        assert_eq!(
            echo.logprobs.len(),
            prompt.len(),
            "one scored prompt row per prompt token"
        );
    }
    let refs: Vec<&_> = controls.iter().collect();
    harness.shutdown(&refs);
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
    // The fixture also carries "edge" (1024 tokens), but that is past the TP
    // prompt ceiling: a prefill that long stalls the single-threaded multi-rank
    // driver before the peer ranks launch (docs/models/gemma4/tp.md "Known
    // bounds"), so admission refuses it (`TP_MAX_PROMPT`). The case joins the
    // gate once the ranks are driven concurrently.
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
        eprintln!(
            "hf: {case}: {} tokens, top_k {top_k}: submitting",
            tokens.len()
        );
        let control = harness.submit_scored(tokens.clone(), 1, Some(top_k), Some(top_k));
        let drained = harness.steps.drain(control.id(), case);
        eprintln!("hf: {case}: drained");
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
            let ours_top = top_of(ours);
            let (pa, pb) = (ours.top_logprobs[0].0, theirs[0].0);
            compared += 1;
            if pa == pb {
                same_pick += 1;
            } else {
                // A differing pick has to be a near-tie, the same rule the
                // one-rank comparison holds itself to.
                assert!(
                    theirs.iter().any(|(token, _)| *token == pa),
                    "{case} row {row}: the engine's pick {pa} is outside the reference's top-{top_k}"
                );
                assert!(
                    ours_top.contains_key(&pb),
                    "{case} row {row}: the reference's pick {pb} is outside the engine's top-{top_k}"
                );
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
        "hf: {same_pick}/{compared} rows keep the reference's pick; worst shared-token logprob gap \
         {worst:.4} at {worst_at}"
    );
    let controls: Vec<&_> = controls.iter().collect();
    harness.shutdown(&controls);
    assert!(
        worst < DRIFT_LINE,
        "the two-rank engine and the reference differ on {worst_at} by {worst} (line {DRIFT_LINE})"
    );
}
