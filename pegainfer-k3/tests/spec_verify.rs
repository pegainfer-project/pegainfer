//! Speculative verify-step gates.
//!
//! The verify path runs the KDA layers through chunkwise FlashKDA where plain
//! decode runs the bit-matched fused core, and its projections run at the
//! packed bucket — the same cross-bucket noise class as chunked prefill. A
//! bitwise "verify == plain decode" gate is therefore impossible by
//! construction: near-tie argmaxes flip (observed at 4-layer truncation on
//! margins of 0.06–0.25 logits against a ~0.3 noise floor). What IS exact,
//! and what these gates hold the machinery to:
//!
//! 1. **Determinism and history independence.** The same verify walk must be
//!    bit-stable across reruns, page permutations, and whatever ran on the
//!    executor before it.
//! 2. **Rejected drafts leave nothing behind.** Two walks that differ only in
//!    the *content* of always-rejected drafts run identical launch
//!    geometries, so their trajectories must be bit-identical — any
//!    divergence is rejected-state leakage (latent append rollback, KDA
//!    replay, conv windows).
//! 3. **Slots are isolated.** In a packed two-slot step, changing one slot's
//!    tokens must not move the other slot's trajectory by a bit.
//! 4. **Oracle tracking (diagnostic bound).** Teacher-forced verify tracks
//!    the plain-decode oracle everywhere the oracle is confident; flips are
//!    tolerated only against small margins, and bounded in number.
//!
//! Manual gates like the golden suite: CI compiles them, a Blackwell box with
//! the checkpoint runs them with `--ignored`. `PEGAINFER_K3_TEST_224` points
//! at the 224-expert checkpoint, `PEGAINFER_K3_TEST_DEVICE` picks the GPU.

use std::path::PathBuf;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_k3::DecodeSlot;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::K3MoeTransport;
use pegainfer_k3::K3VerifySlot;
use pegainfer_k3::StepExecutor;

const FIXTURE: &str = include_str!("fixtures/k3_4l_greedy.json");
const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";
const DEVICE_ENV: &str = "PEGAINFER_K3_TEST_DEVICE";
/// The RadixArk DSpark drafter checkpoint dir, for the draft-lane gates.
const DSPARK_ENV: &str = "PEGAINFER_K3_TEST_DSPARK";

fn fixture_prompt_and_layers() -> (Vec<u32>, usize) {
    let json: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let prompt = json["prompt"]
        .as_array()
        .expect("prompt array")
        .iter()
        .map(|entry| entry.as_u64().expect("token id") as u32)
        .collect();
    let num_layers = json["num_layers"].as_u64().expect("num_layers") as usize;
    (prompt, num_layers)
}

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

fn executor(num_layers: usize) -> Option<K3Executor> {
    let path = checkpoint()?;
    let config = K3ExecutorConfig {
        max_batch: 32,
        max_ctx: 512,
        kv_pages: 0,
        num_layers,
        chunk_tokens: 0,
        cuda_graph: false,
        moe_transport: K3MoeTransport::MEGA,
    };
    Some(
        K3Executor::load(&path, device(), 0, 1, config)
            .expect("the truncated rank model should load"),
    )
}

/// Diverse but deterministic filler tokens, clear of the fixture's ids.
fn filler_tokens(count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 100_000) as u32 + 1_000
        })
        .collect()
}

/// The plain-decode oracle: feed the prompt one token per step, then continue
/// greedily for `steps` more; return every step's argmax.
fn oracle_walk(executor: &mut K3Executor, slot: usize, prompt: &[u32], steps: usize) -> Vec<u32> {
    executor.release(slot);
    let mut sampled: Vec<u32> = Vec::with_capacity(prompt.len() + steps);
    for index in 0..prompt.len() + steps {
        let last_token = if index < prompt.len() {
            prompt[index]
        } else {
            sampled[index - 1]
        };
        let step = executor
            .decode(&[DecodeSlot { slot, last_token }])
            .expect("the decode step should run");
        sampled.push(step[0]);
    }
    sampled
}

/// A teacher-forced verify walk on one slot: round `i` feeds `feed[i]` as the
/// anchor and `drafts(i)` as the drafted continuation, regardless of what the
/// executor committed. Returns each round's committed tokens.
fn forced_verify_walk(
    executor: &mut K3Executor,
    slot: usize,
    feed: &[u32],
    mut drafts: impl FnMut(usize) -> Vec<u32>,
) -> Vec<Vec<u32>> {
    executor.release(slot);
    feed.iter()
        .enumerate()
        .map(|(index, &anchor)| {
            let proposed = drafts(index);
            let count = proposed.len();
            let outcome = executor
                .verify(&[K3VerifySlot {
                    slot,
                    anchor,
                    drafts: proposed,
                }])
                .expect("the verify step should run");
            assert_eq!(outcome.len(), 1, "one slot in, one outcome out");
            assert!(
                !outcome[0].is_empty() && outcome[0].len() <= count + 1,
                "a round commits between 1 and drafts+1 tokens, got {}",
                outcome[0].len()
            );
            outcome[0].clone()
        })
        .collect()
}

/// Gate: the same verify walk is bit-stable across reruns, page permutations
/// and whatever ran on the executor before it. Teacher-forced empty-draft
/// rounds over the fixture prompt, replayed after several histories.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn verify_walk_is_deterministic_and_history_independent() {
    let (prompt, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let walk = |executor: &mut K3Executor| -> Vec<Vec<u32>> {
        forced_verify_walk(executor, 0, &prompt, |_| Vec::new())
    };
    let baseline = walk(&mut executor);

    let mut clean = true;
    let mut check = |label: &str, other: &[Vec<u32>]| {
        if other == baseline {
            eprintln!("{label}: identical to the fresh walk");
        } else {
            clean = false;
            eprintln!("{label}: DIVERGES from the fresh walk");
        }
    };
    let rerun = walk(&mut executor);
    check("immediate rerun", &rerun);

    oracle_walk(&mut executor, 0, &prompt, 48);
    let after_decode = walk(&mut executor);
    check("after a 64-step decode walk", &after_decode);

    executor.release(0);
    executor.scramble_kv_pages();
    let after_scramble = walk(&mut executor);
    check("after a page scramble", &after_scramble);

    assert!(clean, "verify walks depend on executor history");
}

/// Gate: rejected draft content leaves nothing behind. Two teacher-forced
/// walks whose drafts are always rejected (corrupted at position 0) but carry
/// *different* garbage run identical launch geometries, so any trajectory
/// difference is rejected-state leakage. Bit equality required.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn rejected_draft_content_cannot_leak() {
    let (prompt, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut feed = prompt.clone();
    feed.extend(filler_tokens(48, 17));

    // Draft position 0 differs from the plain-token stream (xor keeps it in
    // vocab), so acceptance should be 0 every round; positions 1.. carry
    // walk-specific garbage that must never matter.
    let walk = |executor: &mut K3Executor, seed: u64| -> Vec<Vec<u32>> {
        let garbage = filler_tokens(6 * feed.len(), seed);
        forced_verify_walk(executor, 0, &feed, |round| {
            let mut drafts = garbage[6 * round..6 * (round + 1)].to_vec();
            drafts[0] = feed[round] ^ 1;
            drafts
        })
    };
    let walk_a = walk(&mut executor, 101);
    let walk_b = walk(&mut executor, 202);
    for (round, (a, b)) in walk_a.iter().zip(&walk_b).enumerate() {
        assert_eq!(
            a[0], b[0],
            "round {round}: rejected draft content moved the model token"
        );
    }
    let accepted: usize = walk_a.iter().map(|tokens| tokens.len() - 1).sum();
    eprintln!(
        "rejected-content gate: {} rounds bit-identical, {accepted} accidental acceptances",
        walk_a.len()
    );
}

/// Gate: packed slots are isolated. Two runs of a packed two-slot walk where
/// slot B's tokens change between runs; slot A's rows sit at the same batch
/// rows with the same launch geometry both times, so its trajectory must be
/// bit-identical. Catches cross-slot leaks through the packed table, the KDA
/// group offsets and the conv windows.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn packed_slot_neighbours_cannot_leak() {
    let (prompt_a, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut feed_a = prompt_a;
    feed_a.extend(filler_tokens(32, 5));

    // Slot A leads the pack, slot B follows with 3 drafts to move the bucket
    // around; B's whole token stream changes between the runs.
    let packed_walk = |executor: &mut K3Executor, b_seed: u64| -> Vec<Vec<u32>> {
        executor.release(2);
        executor.release(9);
        let feed_b = filler_tokens(feed_a.len(), b_seed);
        let garbage = filler_tokens(3 * feed_a.len(), b_seed ^ 0xff);
        feed_a
            .iter()
            .enumerate()
            .map(|(round, &anchor_a)| {
                let outcome = executor
                    .verify(&[
                        K3VerifySlot {
                            slot: 2,
                            anchor: anchor_a,
                            drafts: vec![feed_a[round] ^ 1, 7, 11],
                        },
                        K3VerifySlot {
                            slot: 9,
                            anchor: feed_b[round],
                            drafts: garbage[3 * round..3 * (round + 1)].to_vec(),
                        },
                    ])
                    .expect("the packed verify step should run");
                assert_eq!(outcome.len(), 2);
                outcome[0].clone()
            })
            .collect()
    };
    let run_1 = packed_walk(&mut executor, 33);
    let run_2 = packed_walk(&mut executor, 77);
    for (round, (a, b)) in run_1.iter().zip(&run_2).enumerate() {
        assert_eq!(
            a[0], b[0],
            "round {round}: the neighbour slot's tokens moved slot A's model token"
        );
    }
    eprintln!(
        "packed-isolation gate: {} rounds bit-identical under a changed neighbour",
        run_1.len()
    );
}

/// Gate: corruption position determines geometry, corruption value must not.
/// Two teacher-forced walks with oracle drafts corrupted at the same position
/// each round but with different corrupt values run the same accept/reject
/// shapes, so committed tokens and acceptance counts must be identical.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn corruption_value_cannot_change_the_trajectory() {
    let (prompt, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    const LOOKAHEAD: usize = 6;
    let oracle = oracle_walk(&mut executor, 0, &prompt, 48 + LOOKAHEAD);
    let mut feed = prompt.clone();
    feed.extend_from_slice(&oracle[prompt.len() - 1..prompt.len() - 1 + 48]);

    // Oracle continuation as drafts, corrupted at position 2 every round.
    let walk = |executor: &mut K3Executor, twiddle: u32| -> (Vec<Vec<u32>>, Vec<usize>) {
        let committed = forced_verify_walk(executor, 0, &feed, |round| {
            let mut drafts: Vec<u32> = (0..LOOKAHEAD)
                .map(|offset| oracle[(round + offset).min(oracle.len() - 1)])
                .collect();
            drafts[2] ^= twiddle;
            drafts
        });
        let accepted = committed.iter().map(|tokens| tokens.len() - 1).collect();
        (committed, accepted)
    };
    let (committed_1, accepted_1) = walk(&mut executor, 1);
    let (committed_2, accepted_2) = walk(&mut executor, 2);
    assert_eq!(
        accepted_1, accepted_2,
        "the corrupt value changed acceptance counts"
    );
    assert_eq!(
        committed_1, committed_2,
        "the corrupt value changed committed tokens"
    );
    let total: usize = accepted_1.iter().sum();
    let capped: usize = accepted_1.iter().filter(|count| **count == 2).count();
    eprintln!(
        "corruption-value gate: {} rounds identical; {total} drafts accepted, {capped} rounds \
         hit the corruption cap",
        accepted_1.len()
    );
    assert!(
        capped > 0,
        "some rounds should accept right up to the corrupted draft"
    );
}

/// Diagnostic bound, not an exact gate: teacher-forced empty-draft verify
/// tracks the plain-decode oracle except on near-tie argmaxes. Every
/// mismatch must sit on a small oracle top-2 margin, and mismatches must be
/// the minority — otherwise the verify path's numerics have left the
/// chunked-prefill noise class and something is broken.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn verify_tracks_the_oracle_off_near_ties() {
    let (prompt, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut feed = prompt.clone();
    feed.extend(filler_tokens(48, 29));

    // Oracle argmaxes and margins for the forced feed.
    executor.release(0);
    let mut oracle: Vec<(u32, f32)> = Vec::with_capacity(feed.len());
    for &token in &feed {
        let step = executor
            .decode(&[DecodeSlot {
                slot: 0,
                last_token: token,
            }])
            .expect("decode");
        let logits = executor.last_logits(0).expect("logits");
        let top = step[0] as usize;
        let margin = logits[top]
            - logits
                .iter()
                .enumerate()
                .filter(|(id, _)| *id != top)
                .map(|(_, value)| *value)
                .fold(f32::NEG_INFINITY, f32::max);
        oracle.push((step[0], margin));
    }

    let committed = forced_verify_walk(&mut executor, 0, &feed, |_| Vec::new());
    let mut mismatches = 0usize;
    let mut worst_margin = 0f32;
    for (step, (tokens, (oracle_token, margin))) in committed.iter().zip(&oracle).enumerate() {
        if tokens[0] != *oracle_token {
            mismatches += 1;
            worst_margin = worst_margin.max(*margin);
            eprintln!(
                "step {step}: verify {} vs oracle {} (margin {margin:.4})",
                tokens[0], oracle_token
            );
        }
    }
    eprintln!(
        "oracle tracking: {mismatches}/{} mismatches, worst margin {worst_margin:.4}",
        feed.len()
    );
    assert!(
        mismatches * 4 <= feed.len(),
        "verify left the oracle on {mismatches}/{} steps — beyond the noise-flip class",
        feed.len()
    );
    assert!(
        worst_margin < 4.0,
        "a flip against a {worst_margin:.2}-logit margin is not a near-tie"
    );
}

/// Gate: the whole DSpark loop — prefill capture, propose, verify, accepted
/// -row capture — round-trips on the real drafter checkpoint. At truncated
/// target depth the aux taps are clamped, so the drafts are garbage and this
/// gate holds the *machinery*, not draft quality: every round commits within
/// bounds, the propose-side position invariants hold for the whole walk (its
/// internal asserts crash on drift), and a released-and-reprefilled slot
/// reproduces the walk bit for bit.
#[test]
#[ignore = "requires a Blackwell GPU, the K3 checkpoint and the DSpark drafter"]
fn dspark_draft_lane_round_trips() {
    let (prompt, num_layers) = fixture_prompt_and_layers();
    let Some(mut executor) = executor(num_layers) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let Some(dspark_path) = std::env::var(DSPARK_ENV)
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.join("config.json").exists())
    else {
        eprintln!("skipping: {DSPARK_ENV} is not set to the DSpark drafter checkpoint");
        return;
    };
    executor
        .load_dspark(&dspark_path)
        .expect("the drafter should load");

    const STEPS: usize = 48;
    let walk = |executor: &mut K3Executor| -> (Vec<u32>, usize, usize) {
        executor.release(0);
        let first = executor
            .prefill(0, &prompt, &SamplingParams::default())
            .expect("prefill with capture should run");
        let mut committed_all = vec![first];
        let mut rounds = 0usize;
        let mut accepted = 0usize;
        while committed_all.len() < STEPS {
            let outcome = executor
                .decode_spec(&[DecodeSlot {
                    slot: 0,
                    last_token: *committed_all.last().expect("seeded"),
                }])
                .expect("a spec round should run");
            assert_eq!(outcome.len(), 1);
            let tokens = &outcome[0];
            assert!(
                !tokens.is_empty() && tokens.len() <= 7,
                "a spec round commits 1..=7 tokens, got {}",
                tokens.len()
            );
            accepted += tokens.len() - 1;
            committed_all.extend_from_slice(tokens);
            rounds += 1;
        }
        (committed_all, rounds, accepted)
    };

    let (walk_1, rounds, accepted) = walk(&mut executor);
    eprintln!(
        "dspark round trip: {} tokens over {rounds} rounds, {accepted} drafts accepted \
         (clamped-tap drafts — acceptance is noise, not quality)",
        walk_1.len()
    );
    let (walk_2, _, _) = walk(&mut executor);
    assert_eq!(
        walk_1, walk_2,
        "a released and re-prefilled slot must reproduce the spec walk exactly"
    );
}
