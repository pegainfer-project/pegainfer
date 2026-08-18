//! Paged MLA KV cache gates.
//!
//! The MLA cache is a pool of 64-token latent pages behind a per-slot block
//! table, and the absorbed decode kernel walks that table by logical position
//! — physical page ids never enter the arithmetic. That claim is checkable,
//! so these gates check it:
//!
//! 1. **Page permutation is bitwise invisible.** The same replay runs twice,
//!    the second time after scrambling the free list so every page lands at a
//!    different physical id in a different order. Every logit of every step
//!    must be bit-identical.
//! 2. **Long context is self-consistent.** A context far past the old 128-slot
//!    cache (and past one page, and past one *table row* of pages) decodes to
//!    finite logits, reproduces itself bitwise on a fresh executor, and
//!    produces the same trajectory through CUDA graphs.
//! 3. **A/B logit dump.** Not a gate by itself but the instrument for one:
//!    `dump_forced_replay_logits` writes every step's logit row to
//!    `PEGAINFER_K3_LOGIT_DUMP`, so the absorbed kernel can be held to the
//!    expanded kernel it replaced by running the dump on both builds and
//!    comparing the files offline (the M1 revision carries the expanded
//!    attention with the paged write already in place, which makes the diff a
//!    verdict on the attention kernel alone).
//!
//! Manual gates like the golden suite: CI compiles them, a Blackwell box with
//! the checkpoint runs them with `--ignored`. `PEGAINFER_K3_TEST_224` points
//! at the 224-expert checkpoint, `PEGAINFER_K3_TEST_DEVICE` picks the GPU.

use std::io::Write;
use std::path::PathBuf;

use pegainfer_k3::DecodeSlot;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::K3MoeTransport;
use pegainfer_k3::StepExecutor;

const FIXTURE: &str = include_str!("fixtures/k3_4l_greedy.json");
const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";
const DEVICE_ENV: &str = "PEGAINFER_K3_TEST_DEVICE";
const DUMP_ENV: &str = "PEGAINFER_K3_LOGIT_DUMP";

/// The fixture's prompt-then-argmax feed and layer truncation — the same
/// forced-replay inputs the golden gate uses, so the A/B dump below compares
/// the two attention kernels on certified ground.
struct Fixture {
    feed: Vec<u32>,
    num_layers: usize,
}

fn fixture() -> Fixture {
    let json: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let prompt: Vec<u32> = json["prompt"]
        .as_array()
        .expect("prompt array")
        .iter()
        .map(|entry| entry.as_u64().expect("token id") as u32)
        .collect();
    let argmax: Vec<u32> = json["steps"]
        .as_array()
        .expect("steps array")
        .iter()
        .map(|step| step["argmax"].as_u64().expect("argmax") as u32)
        .collect();
    let mut feed = prompt.clone();
    feed.extend(argmax[prompt.len() - 1..argmax.len() - 1].iter().copied());
    Fixture {
        feed,
        num_layers: json["num_layers"].as_u64().expect("num_layers") as usize,
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

fn executor(config: K3ExecutorConfig) -> Option<K3Executor> {
    let path = checkpoint()?;
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

/// Forced replay of `feed` on slot 0, one decode step per token, returning
/// each step's sampled token and its logit row as raw f32 bit patterns.
/// The logits land in bf16, so bit equality here *is* bf16 bit equality.
fn replay_with_logits(executor: &mut K3Executor, feed: &[u32]) -> (Vec<u32>, Vec<Vec<u32>>) {
    executor.release(0);
    let mut sampled = Vec::with_capacity(feed.len());
    let mut logits = Vec::with_capacity(feed.len());
    for &last_token in feed {
        let step = executor
            .decode(&[DecodeSlot {
                slot: 0,
                last_token,
            }])
            .expect("the decode step should run");
        sampled.push(step[0]);
        logits.push(
            executor
                .last_logits(0)
                .expect("logit readback")
                .into_iter()
                .map(f32::to_bits)
                .collect(),
        );
    }
    (sampled, logits)
}

fn assert_bitwise(baseline: &[Vec<u32>], other: &[Vec<u32>], what: &str) {
    assert_eq!(baseline.len(), other.len(), "{what}: step counts differ");
    for (step, (a, b)) in baseline.iter().zip(other).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: step {step} logit widths differ");
        if let Some(id) = (0..a.len()).find(|&id| a[id] != b[id]) {
            panic!(
                "{what}: step {step} logit {id} differs — {:e} vs {:e} \
                 ({:#010x} vs {:#010x})",
                f32::from_bits(a[id]),
                f32::from_bits(b[id]),
                a[id],
                b[id]
            );
        }
    }
    eprintln!("{what}: {} steps bit-identical", baseline.len());
}

/// Gate: scrambling the page pool must not move a single logit bit. The feed
/// spans four pages, so the walk crosses page boundaries in both runs; the
/// scrambled run claims disjoint physical pages in the opposite order.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn page_permutation_leaves_every_logit_bit_identical() {
    let fixture = fixture();
    let config = K3ExecutorConfig {
        max_batch: 1,
        max_ctx: 256,
        kv_pages: 16,
        num_layers: fixture.num_layers,
        chunk_tokens: 0,
        cuda_graph: false,
        moe_transport: K3MoeTransport::MEGA,
    };
    let Some(mut executor) = executor(config) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let mut feed = fixture.feed;
    feed.extend(filler_tokens(200 - feed.len(), 7));

    // Run 1 claims pages 0,1,2,3 (a fresh pool hands them out ascending).
    let (tokens_a, logits_a) = replay_with_logits(&mut executor, &feed);
    // Run 2: the release inside the replay returns run 1's pages, and the
    // scramble reverses the whole list, so the claims are 15,14,13,12 —
    // disjoint pages, opposite order.
    executor.release(0);
    executor.scramble_kv_pages();
    let (tokens_b, logits_b) = replay_with_logits(&mut executor, &feed);

    assert_eq!(tokens_a, tokens_b, "sampled trajectories differ");
    assert_bitwise(&logits_a, &logits_b, "page permutation");
}

/// Gate: a context past the old 128-token cap (and past 1024) decodes to
/// finite logits, reproduces itself bitwise on a fresh executor, and takes
/// the same trajectory through CUDA graphs. Logits are compared at every
/// 128th step and the last one; tokens at every step.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn long_context_decode_is_self_consistent() {
    let fixture = fixture();
    let config = K3ExecutorConfig {
        max_batch: 1,
        max_ctx: 2048,
        kv_pages: 0,
        num_layers: fixture.num_layers,
        chunk_tokens: 0,
        cuda_graph: false,
        moe_transport: K3MoeTransport::MEGA,
    };
    let mut feed = fixture.feed;
    feed.extend(filler_tokens(1100 - feed.len(), 23));

    let sample_step = |step: usize| -> bool { step == feed.len() - 1 || step.is_multiple_of(128) };
    let run = |executor: &mut K3Executor| -> (Vec<u32>, Vec<Vec<u32>>) {
        executor.release(0);
        let mut sampled = Vec::with_capacity(feed.len());
        let mut logits = Vec::new();
        for (step, &last_token) in feed.iter().enumerate() {
            let tokens = executor
                .decode(&[DecodeSlot {
                    slot: 0,
                    last_token,
                }])
                .expect("the decode step should run");
            sampled.push(tokens[0]);
            if sample_step(step) {
                let row = executor.last_logits(0).expect("logit readback");
                assert!(
                    row.iter().all(|logit| logit.is_finite()),
                    "step {step} produced a non-finite logit"
                );
                logits.push(row.into_iter().map(f32::to_bits).collect());
            }
        }
        (sampled, logits)
    };

    let Some(mut eager) = executor(config) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let (tokens_a, logits_a) = run(&mut eager);
    let (tokens_b, logits_b) = run(&mut eager);
    assert_eq!(
        tokens_a, tokens_b,
        "eager rerun took a different trajectory"
    );
    assert_bitwise(&logits_a, &logits_b, "long context, eager rerun");
    drop(eager);

    let mut graphs = executor(K3ExecutorConfig {
        cuda_graph: true,
        ..config
    })
    .expect("the checkpoint was there a moment ago");
    let (tokens_c, logits_c) = run(&mut graphs);
    assert_eq!(tokens_a, tokens_c, "graphs took a different trajectory");
    assert_bitwise(&logits_a, &logits_c, "long context, graphs vs eager");
    eprintln!(
        "long context: {} steps, final context {} tokens",
        feed.len(),
        feed.len()
    );
}

/// Instrument for the absorbed-vs-expanded certification: replay the golden
/// fixture's forced feed and write every step's logit row (widened from bf16)
/// to `PEGAINFER_K3_LOGIT_DUMP` as flat little-endian f32, `[steps, vocab]`
/// row-major. Run once on the expanded-attention revision and once on this
/// one, then compare the files per step in bf16 ULP.
#[test]
#[ignore = "requires a Blackwell GPU and the K3 checkpoint"]
fn dump_forced_replay_logits() {
    let Ok(dump_path) = std::env::var(DUMP_ENV) else {
        eprintln!("skipping: {DUMP_ENV} is not set to an output path");
        return;
    };
    let fixture = fixture();
    let config = K3ExecutorConfig {
        max_batch: 1,
        max_ctx: 128,
        kv_pages: 0,
        num_layers: fixture.num_layers,
        chunk_tokens: 0,
        cuda_graph: false,
        moe_transport: K3MoeTransport::MEGA,
    };
    let Some(mut executor) = executor(config) else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let (sampled, logits) = replay_with_logits(&mut executor, &fixture.feed);
    let mut file = std::fs::File::create(&dump_path).expect("create the dump file");
    for row in &logits {
        let bytes: Vec<u8> = row.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        file.write_all(&bytes).expect("write the dump file");
    }
    eprintln!(
        "dumped {} steps x {} logits to {dump_path}; sampled {:?}",
        logits.len(),
        logits.first().map_or(0, Vec::len),
        sampled
    );
}
