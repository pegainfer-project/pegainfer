//! Kimi-K3 expert-parallel oracle: EP4 must equal EP1 **bit for bit**.
//!
//! The EP scheme is exact by construction, not approximately equal, so the
//! acceptance criterion is bitwise identity rather than a tolerance:
//!
//! * a token's latent row reaches its expert as the same bf16 bytes, so the
//!   activation quant emits the same fp8;
//! * an entry lands in the same masked row (the compaction walks entry order
//!   over the expert's entry set with or without the window) and the whole
//!   global batch still fits one `masked_cap` tile, so the masked GEMM's
//!   per-row accumulation is untouched by how many neighbours share the tile;
//! * the merge is a sum over **disjoint support** — every global entry is
//!   owned by exactly one rank and every other rank staged an exact zero — so
//!   the bf16 all-reduce only ever adds a value to zeros, which is exact in any
//!   reduction order;
//! * the combine's arithmetic is the single-rank combine's, slot by slot.
//!
//! Nothing in that list has slack in it. If any of it were false the right
//! answer would be to fix it, not to widen a tolerance, which is why these
//! gates compare u32 logit bit patterns and refuse to spend a ULP.
//!
//! Manual gates: CI compiles them, a Blackwell box with the checkpoint runs
//! them with `--ignored`. Point `PEGAINFER_K3_TEST_224` at the 224-expert
//! checkpoint directory.
//!
//! **Run one gate per process.** A communicator's lifetime is the process's:
//! a second NCCL bring-up after one has been torn down wedges. So each gate
//! below is its own `#[test]` and they are invoked separately, e.g.
//!
//! ```text
//! PEGAINFER_K3_LAYERS=4 PEGAINFER_K3_MAX_BATCH=16 \
//!   cargo test --release -p pegainfer-k3 --test ep_oracle \
//!   ep4_matches_ep1_bitwise -- --ignored --nocapture
//! ```
//!
//! (never `cargo test --test ep_oracle -- --ignored`, which would run both in
//! one process on the same four GPUs).

use std::path::PathBuf;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_k3::DecodeSlot;
use pegainfer_k3::K3EpRendezvous;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::StepExecutor;

const FIXTURE: &str = include_str!("fixtures/k3_4l_greedy.json");
const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";

/// Ranks under test, one per GPU.
const EP_SIZE: usize = 4;
/// Slots per rank. `EP_SIZE * MAX_BATCH` must fit the masked layout's 128 rows
/// per expert, which is the worst case of one expert claiming every token.
const MAX_BATCH: usize = 16;
/// Greedy steps after the prompt. `prompt.len() + DECODE_STEPS` is the chain
/// step count every rank executes, and the peers pad to exactly that.
const DECODE_STEPS: usize = 24;

struct Fixture {
    prompt: Vec<u32>,
    num_layers: usize,
    max_ctx: usize,
}

fn fixture() -> Fixture {
    let json: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    Fixture {
        prompt: json["prompt"]
            .as_array()
            .expect("prompt array")
            .iter()
            .map(|entry| entry.as_u64().expect("token id") as u32)
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

/// Every rank — and the single-rank reference — runs the same geometry, so the
/// only thing under test is the expert parallelism.
fn config(fixture: &Fixture) -> K3ExecutorConfig {
    K3ExecutorConfig {
        max_batch: MAX_BATCH,
        max_ctx: fixture.max_ctx,
        num_layers: fixture.num_layers,
        // Eager either way: EP forces it off, and the reference has to match.
        cuda_graph: false,
        // The EP oracle is a bitwise chain-vs-chain comparison; MegaMoE is a
        // different arithmetic and is rejected under EP anyway.
        mega: false,
    }
}

/// What a rank-0 run is judged on: every sampled token, and the raw bits of the
/// logit row the last step left. bf16 -> f32 is injective, so comparing f32 bit
/// patterns is comparing the bf16 logits the model actually produced.
struct Trace {
    tokens: Vec<u32>,
    logit_bits: Vec<u32>,
}

impl Trace {
    fn of(executor: &mut K3Executor, prompt: &[u32]) -> Self {
        let tokens = executor
            .greedy_replay(0, prompt, DECODE_STEPS)
            .expect("the rank-0 greedy replay should run");
        let logit_bits = executor
            .last_logits(0)
            .expect("the final logit readback should run")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        Self { tokens, logit_bits }
    }
}

/// Compare two traces and report where they first part company, in the style
/// the free-running oracles use: the first differing index, and how many
/// entries differ in total, so a one-off is distinguishable from a rout.
fn assert_bit_identical(reference: &Trace, candidate: &Trace, what: &str) {
    assert_eq!(
        candidate.tokens.len(),
        reference.tokens.len(),
        "{what}: produced {} tokens, the reference produced {}",
        candidate.tokens.len(),
        reference.tokens.len()
    );
    if let Some(step) =
        (0..reference.tokens.len()).find(|step| candidate.tokens[*step] != reference.tokens[*step])
    {
        let differing = (0..reference.tokens.len())
            .filter(|step| candidate.tokens[*step] != reference.tokens[*step])
            .count();
        panic!(
            "{what}: token stream diverged at step {step} (got {}, reference {}); {differing} of \
             {} steps differ",
            candidate.tokens[step],
            reference.tokens[step],
            reference.tokens.len()
        );
    }
    assert_eq!(
        candidate.logit_bits.len(),
        reference.logit_bits.len(),
        "{what}: logit row is {} wide, the reference row is {}",
        candidate.logit_bits.len(),
        reference.logit_bits.len()
    );
    if let Some(index) = (0..reference.logit_bits.len())
        .find(|index| candidate.logit_bits[*index] != reference.logit_bits[*index])
    {
        let differing = (0..reference.logit_bits.len())
            .filter(|index| candidate.logit_bits[*index] != reference.logit_bits[*index])
            .count();
        panic!(
            "{what}: final logits are not bit-identical — first difference at id {index} \
             (got {:#010x}, reference {:#010x}); {differing} of {} logits differ. The EP scheme \
             is exact by construction, so any difference is a real defect, not noise.",
            candidate.logit_bits[index],
            reference.logit_bits[index],
            reference.logit_bits.len()
        );
    }
    eprintln!(
        "{what}: all {} tokens and all {} final logits are bit-identical",
        reference.tokens.len(),
        reference.logit_bits.len()
    );
}

/// The single-rank reference, on device 0, with no communicator anywhere. Run
/// and dropped before the EP phase takes the device back.
fn single_rank_trace(path: &std::path::Path, fixture: &Fixture) -> Trace {
    let mut executor = K3Executor::load(path, 0, 0, 1, config(fixture))
        .expect("the single-rank truncated model should load");
    let trace = Trace::of(&mut executor, &fixture.prompt);
    eprintln!(
        "EP1 reference: {} tokens, first {:?}",
        trace.tokens.len(),
        &trace.tokens[..trace.tokens.len().min(8)]
    );
    trace
}

/// What the peer ranks do while rank 0 replays the fixture. Either way they
/// execute exactly `prompt.len() + DECODE_STEPS` chain steps, which is what
/// pairs them with rank 0 entry for entry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Peers {
    /// Nothing to serve: every step is a padding step through `decode(&[])`.
    Idle,
    /// A prompt and a continuation of their own, so rank 0's rows share the
    /// fleet's batch, its experts and its collectives with real traffic.
    Busy,
}

/// Run the EP4 phase and return rank 0's trace.
///
/// Weights for **every** rank are resident before any rank is stepped: comm
/// init is lazy on the first step, so a rank that failed to load could
/// otherwise leave its peers blocked in `ncclCommInitRank`, which never times
/// out. Each rank then steps, and drops, its own executor on its own thread —
/// comms belong to the thread that uses them.
fn ep_trace(path: &std::path::Path, fixture: &Fixture, peers: Peers) -> Trace {
    let rendezvous = K3EpRendezvous::new(EP_SIZE);
    let executors: Vec<K3Executor> = (0..EP_SIZE)
        .map(|rank| {
            K3Executor::load_ep(path, rank, rank, config(fixture), rendezvous.clone())
                .unwrap_or_else(|error| panic!("EP rank {rank} should load: {error:#}"))
        })
        .collect();
    let chain_steps = fixture.prompt.len() + DECODE_STEPS;

    std::thread::scope(|scope| {
        let mut rank0 = None;
        let mut handles = Vec::new();
        for (rank, mut executor) in executors.into_iter().enumerate() {
            let prompt = fixture.prompt.clone();
            let handle = scope.spawn(move || {
                let trace = if rank == 0 {
                    Some(Trace::of(&mut executor, &prompt))
                } else {
                    peer_run(&mut executor, rank, &prompt, peers, chain_steps);
                    None
                };
                // Every rank has finished its fixed step count, and a step only
                // returns once its stream has drained, so no peer is inside a
                // collective when this comm goes away.
                drop(executor);
                trace
            });
            handles.push(handle);
        }
        for handle in handles {
            if let Some(trace) = handle.join().expect("an EP rank thread panicked") {
                rank0 = Some(trace);
            }
        }
        rank0.expect("rank 0 produced a trace")
    })
}

fn peer_run(
    executor: &mut K3Executor,
    rank: usize,
    prompt: &[u32],
    peers: Peers,
    chain_steps: usize,
) {
    match peers {
        Peers::Idle => {
            for _ in 0..chain_steps {
                let tokens = executor.decode(&[]).expect("a padding step should run");
                assert!(
                    tokens.is_empty(),
                    "rank {rank}: a padding step must answer nobody"
                );
            }
        }
        Peers::Busy => {
            // A rotation of the fixture prompt: real, in-vocab, diverse tokens,
            // distinct per rank, and the same length, so the peer's prefill
            // spends exactly as many chain steps as rank 0's prompt does.
            let mut own = prompt.to_vec();
            own.rotate_left(rank);
            let params = SamplingParams::default();
            let mut last = executor
                .prefill(0, &own, &params)
                .expect("a peer prefill should run");
            for _ in 0..DECODE_STEPS {
                last = executor
                    .decode(&[DecodeSlot {
                        slot: 0,
                        last_token: last,
                    }])
                    .expect("a peer decode should run")[0];
            }
            assert_eq!(
                own.len() + DECODE_STEPS,
                chain_steps,
                "rank {rank}: a peer must walk the same number of chain steps as rank 0"
            );
        }
    }
}

/// Gate 1 — EP4 equals EP1, bit for bit, with the peers idle.
///
/// Rank 0 replays the fixture's prompt and continuation; ranks 1..4 take the
/// same number of padding steps, which is the steady state of a free-running
/// fleet. Rank 0's answer must be the answer the single-rank executor gave,
/// down to the bits of the last logit row.
#[test]
#[ignore = "requires four Blackwell GPUs and the K3 checkpoint; run alone in its own process"]
fn ep4_matches_ep1_bitwise() {
    let fixture = fixture();
    let Some(path) = checkpoint() else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let reference = single_rank_trace(&path, &fixture);
    let candidate = ep_trace(&path, &fixture, Peers::Idle);
    assert_bit_identical(&reference, &candidate, "EP4 rank 0 vs EP1, idle peers");
}

/// Gate 2 — traffic invariance, bit for bit.
///
/// The same rank-0 replay, but now every peer is prefilling and decoding a
/// prompt of its own: different tokens, different experts claimed, different
/// masked rows filled, different `masked_m`. A rank's own rows must not be able
/// to tell. This is what certifies the expert window and the disjoint-support
/// merge together — that a row's answer depends on its own routing and nothing
/// else in the fleet's batch.
#[test]
#[ignore = "requires four Blackwell GPUs and the K3 checkpoint; run alone in its own process"]
fn ep4_is_invariant_to_peer_traffic() {
    let fixture = fixture();
    let Some(path) = checkpoint() else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let reference = single_rank_trace(&path, &fixture);
    let candidate = ep_trace(&path, &fixture, Peers::Busy);
    assert_bit_identical(&reference, &candidate, "EP4 rank 0 vs EP1, busy peers");
}
