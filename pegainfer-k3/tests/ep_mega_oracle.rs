//! Kimi-K3 expert-parallel MegaMoE oracle: the fused kernel at four ranks.
//!
//! This is THE expert-parallel gate. A step issues no collective of any kind:
//! each rank writes its own rows into its own symmetric slab and launches, and
//! the kernel dispatches across the world over NVLink, computes every expert it
//! owns for whoever sent it work, and combines each token back into the rank
//! that owns it.
//!
//! Two things therefore need certifying, and they are not the same thing:
//!
//! * **Gate 1 — the world size does not change the answer.** A rank's rows must
//!   come back the same whether its experts are all local or spread over four
//!   devices. The two runs are not guaranteed bit-identical: at `ep_size == 1`
//!   the launch picks its block config from the live token count (which is what
//!   keeps it bit-identical to upstream's Python path), while at four ranks
//!   every launch uses one config fixed at the protocol maximum. A different
//!   BLOCK_K is a different MMA K-accumulation order. So the bar is the
//!   fixture's measured noise floor — the same tolerance `golden_decode`
//!   spends — and the gate additionally *reports* whether the run happened to
//!   be bitwise, with the first divergence either way.
//!
//! * **Gate 2 — a rank's rows do not depend on its peers' traffic.** Same rank,
//!   same world, same step count; only the peers' batches move. Every tile
//!   shape is a constant of the instantiation rather than a function of the
//!   live traffic, and the per-row arithmetic (the mid-quant amax, the combine's
//!   reduction order) is row-local, so this one has no slack in it at all:
//!   bitwise is the acceptance criterion.
//!
//! Manual gates: CI compiles them, a Blackwell box with the checkpoint runs
//! them with `--ignored`. Point `PEGAINFER_K3_TEST_224` at the 224-expert
//! checkpoint directory. **Run one gate per process**, on four otherwise-free
//! GPUs:
//!
//! ```text
//! PEGAINFER_K3_LAYERS=4 PEGAINFER_K3_MAX_BATCH=16 \
//!   cargo test --release -p pegainfer-k3 --test ep_mega_oracle \
//!   ep4_mega_matches_ep1_mega -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_k3::DecodeSlot;
use pegainfer_k3::K3EpRendezvous;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::K3MoeTransport;
use pegainfer_k3::StepExecutor;

const FIXTURE: &str = include_str!("fixtures/k3_4l_greedy.json");
const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";

/// Ranks under test, one per GPU. The fused kernel is AOT-instantiated for this
/// width; there is no runtime rank count.
const EP_SIZE: usize = 4;
/// Slots per rank.
const MAX_BATCH: usize = 16;
/// Steps after the prompt, matching the fixture's own continuation length.
const DECODE_STEPS: usize = 24;
/// The noise floor `golden_decode` measured, in bf16 ULP at the logit's
/// magnitude: a step the reference itself decided by less is a coin flip for
/// anything that is not bit-identical.
const NOISE_FLOOR_ULP: f32 = 2.0;

struct Fixture {
    prompt: Vec<u32>,
    /// Argmax after every step, prompt steps included.
    argmax: Vec<u32>,
    top5_ids: Vec<Vec<u32>>,
    top5_logits: Vec<Vec<f32>>,
    num_layers: usize,
    max_ctx: usize,
}

fn fixture() -> Fixture {
    let json: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
    let ids = |value: &serde_json::Value| -> Vec<u32> {
        value
            .as_array()
            .expect("array")
            .iter()
            .map(|entry| entry.as_u64().expect("token id") as u32)
            .collect()
    };
    let rows = |key: &str| -> Vec<serde_json::Value> {
        json["steps"]
            .as_array()
            .expect("steps array")
            .iter()
            .map(|step| step[key].clone())
            .collect()
    };
    Fixture {
        prompt: ids(&json["prompt"]),
        argmax: rows("argmax")
            .iter()
            .map(|value| value.as_u64().expect("argmax") as u32)
            .collect(),
        top5_ids: rows("top5_ids").iter().map(ids).collect(),
        top5_logits: rows("top5_logits")
            .iter()
            .map(|value| {
                value
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

impl Fixture {
    /// The token fed at every step: the prompt one token at a time, then the
    /// fixture's own argmax. Feeding this rather than our own sample keeps a
    /// step's inputs independent of the previous step's outcome, so a
    /// disagreement localizes to the step that produced it instead of to the
    /// first near-tie.
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

    fn is_coin_flip(&self, step: usize) -> bool {
        self.margin_ulp(step) <= NOISE_FLOOR_ULP
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
        kv_pages: 0,
        num_layers: fixture.num_layers,
        chunk_tokens: 0,
        // Eager either way: EP forces capture off and the reference has to
        // match. (The single-rank mega path does capture; this pins it off so
        // the two differ only in world size.)
        cuda_graph: false,
        weight_staging: false,
        moe_transport: K3MoeTransport::MEGA,
    }
}

/// What a rank-0 run is judged on: the forced-replay argmax of every step, and
/// the raw bits of the logit row the last step left. bf16 -> f32 is injective,
/// so comparing f32 bit patterns is comparing the bf16 logits the model
/// actually produced.
struct Trace {
    tokens: Vec<u32>,
    logit_bits: Vec<u32>,
}

impl Trace {
    fn of(executor: &mut K3Executor, feed: &[u32]) -> Self {
        let tokens = executor
            .forced_replay(0, feed)
            .expect("the rank-0 forced replay should run");
        let logit_bits = executor
            .last_logits(0)
            .expect("the final logit readback should run")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        Self { tokens, logit_bits }
    }
}

/// Report, without asserting, exactly how far two traces agree. Returns whether
/// they are bit-identical, so a gate can either require it or merely record it.
fn report_divergence(reference: &Trace, candidate: &Trace, what: &str) -> bool {
    assert_eq!(
        candidate.tokens.len(),
        reference.tokens.len(),
        "{what}: produced {} tokens, the reference produced {}",
        candidate.tokens.len(),
        reference.tokens.len()
    );
    assert_eq!(
        candidate.logit_bits.len(),
        reference.logit_bits.len(),
        "{what}: logit row is {} wide, the reference row is {}",
        candidate.logit_bits.len(),
        reference.logit_bits.len()
    );
    let token_diffs: Vec<usize> = (0..reference.tokens.len())
        .filter(|step| candidate.tokens[*step] != reference.tokens[*step])
        .collect();
    let logit_diffs: Vec<usize> = (0..reference.logit_bits.len())
        .filter(|index| candidate.logit_bits[*index] != reference.logit_bits[*index])
        .collect();
    match (token_diffs.first(), logit_diffs.first()) {
        (None, None) => {
            eprintln!(
                "{what}: BITWISE — all {} tokens and all {} final logits are identical",
                reference.tokens.len(),
                reference.logit_bits.len()
            );
            true
        }
        (first_token, first_logit) => {
            if let Some(step) = first_token {
                eprintln!(
                    "{what}: tokens differ at step {step} (got {}, reference {}); {} of {} steps \
                     differ",
                    candidate.tokens[*step],
                    reference.tokens[*step],
                    token_diffs.len(),
                    reference.tokens.len()
                );
            } else {
                eprintln!("{what}: all {} tokens match", reference.tokens.len());
            }
            if let Some(index) = first_logit {
                eprintln!(
                    "{what}: final logits differ at id {index} (got {:#010x}, reference \
                     {:#010x}); {} of {} logits differ",
                    candidate.logit_bits[*index],
                    reference.logit_bits[*index],
                    logit_diffs.len(),
                    reference.logit_bits.len()
                );
            } else {
                eprintln!(
                    "{what}: all {} final logits are bit-identical",
                    reference.logit_bits.len()
                );
            }
            false
        }
    }
}

/// The bar Gate 1 actually has to clear: a step may only disagree where the
/// reference itself decided by less than the measured noise floor, and even
/// then the sampled token has to be one the reference ranked in its top 5.
fn assert_within_noise_floor(fixture: &Fixture, reference: &Trace, candidate: &Trace, what: &str) {
    let mut spent = 0usize;
    for step in 0..reference.tokens.len() {
        let (got, want) = (candidate.tokens[step], reference.tokens[step]);
        if got == want {
            continue;
        }
        let excused = fixture.is_coin_flip(step)
            && fixture.top5_ids[step].contains(&got)
            && fixture.top5_ids[step].contains(&want);
        assert!(
            excused,
            "{what}: step {step} sampled {got}, the single-rank mega run sampled {want}; the \
             reference margin there is {:.2} bf16 ULP (noise floor {NOISE_FLOOR_ULP}), top-5 ids \
             {:?} logits {:?}. Above the noise floor the two world sizes must agree.",
            fixture.margin_ulp(step),
            fixture.top5_ids[step],
            fixture.top5_logits[step]
        );
        spent += 1;
        eprintln!(
            "{what}: step {step} is a coin flip (reference margin {:.2} bf16 ULP): got {got}, \
             single-rank {want}, reference top-5 {:?}",
            fixture.margin_ulp(step),
            fixture.top5_ids[step]
        );
    }
    eprintln!(
        "{what}: {}/{} steps agree exactly, {spent} inside the noise floor",
        reference.tokens.len() - spent,
        reference.tokens.len()
    );
}

/// The single-rank mega reference, on device 0. Run and dropped before the EP
/// phase takes the device back.
fn single_rank_trace(path: &std::path::Path, fixture: &Fixture) -> Trace {
    let mut executor = K3Executor::load(path, 0, 0, 1, config(fixture))
        .expect("the single-rank truncated mega model should load");
    let trace = Trace::of(&mut executor, &fixture.feed());
    eprintln!(
        "EP1 mega reference: {} tokens, first {:?}",
        trace.tokens.len(),
        &trace.tokens[..trace.tokens.len().min(8)]
    );
    trace
}

/// What the peer ranks do while rank 0 replays the fixture. Either way they
/// execute exactly `prompt.len() + DECODE_STEPS` steps, so all four ranks
/// launch the same number of mega layers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Peers {
    /// Nothing to serve: every step is a padding step through `decode(&[])`,
    /// which under MegaMoE means a launch with zero local tokens — the rank
    /// still serves its own experts for its peers and still meets every
    /// barrier.
    Idle,
    /// A prompt and a continuation of their own, so rank 0's rows share the
    /// fleet's dispatch, the experts and the rings with real traffic.
    Busy,
}

/// Run the EP4 mega phase and return rank 0's trace.
///
/// Every rank's weights and symmetric slab exist before any rank is stepped:
/// the slab base pointers are published at construction and the first step
/// blocks until the whole table is in, which is the group's startup barrier.
fn ep_trace(path: &std::path::Path, fixture: &Fixture, peers: Peers) -> Trace {
    let rendezvous = K3EpRendezvous::new(EP_SIZE);
    let executors: Vec<K3Executor> = (0..EP_SIZE)
        .map(|rank| {
            K3Executor::load_ep(path, rank, rank, config(fixture), rendezvous.clone())
                .unwrap_or_else(|error| panic!("EP mega rank {rank} should load: {error:#}"))
        })
        .collect();
    let steps = fixture.prompt.len() + DECODE_STEPS;
    let feed = fixture.feed();
    assert_eq!(feed.len(), steps, "the forced feed is one token per step");

    std::thread::scope(|scope| {
        let mut rank0 = None;
        let mut handles = Vec::new();
        for (rank, mut executor) in executors.into_iter().enumerate() {
            let feed = feed.clone();
            let prompt = fixture.prompt.clone();
            let handle = scope.spawn(move || {
                let trace = if rank == 0 {
                    Some(Trace::of(&mut executor, &feed))
                } else {
                    peer_run(&mut executor, rank, &prompt, peers, steps);
                    None
                };
                // Every rank has finished its fixed step count and a step only
                // returns once its stream has drained, so no peer is inside a
                // device barrier when this slab goes away.
                drop(executor);
                trace
            });
            handles.push(handle);
        }
        for handle in handles {
            if let Some(trace) = handle.join().expect("an EP mega rank thread panicked") {
                rank0 = Some(trace);
            }
        }
        rank0.expect("rank 0 produced a trace")
    })
}

fn peer_run(executor: &mut K3Executor, rank: usize, prompt: &[u32], peers: Peers, steps: usize) {
    match peers {
        Peers::Idle => {
            for _ in 0..steps {
                let tokens = executor.decode(&[]).expect("a padding step should run");
                assert!(
                    tokens.is_empty(),
                    "rank {rank}: a padding step must answer nobody"
                );
            }
        }
        Peers::Busy => {
            // A rotation of the fixture prompt: real, in-vocab, diverse tokens,
            // distinct per rank, and the same length as rank 0's.
            let mut own = prompt.to_vec();
            own.rotate_left(rank);
            let params = SamplingParams::default();
            let mut last = executor
                .prefill(0, &own, &params)
                .expect("a peer prefill should run");
            // Chunked prefill spends `ceil(len / cap)` steps where rank 0's
            // forced replay spends one step per token, so the peer pads the
            // difference — which is exactly what a free-running rank does
            // whenever its work ends before its peers'.
            let walked = own.len().div_ceil(executor.chunk_tokens()) + DECODE_STEPS;
            assert!(
                walked <= steps,
                "rank {rank}: walked {walked} steps, rank 0 only takes {steps}"
            );
            for _ in 0..DECODE_STEPS {
                last = executor
                    .decode(&[DecodeSlot {
                        slot: 0,
                        last_token: last,
                    }])
                    .expect("a peer decode should run")[0];
            }
            for _ in walked..steps {
                executor
                    .decode(&[])
                    .expect("a peer padding step should run");
            }
        }
    }
}

/// Gate 1 — four ranks answer what one rank answered.
///
/// Rank 0 forced-replays the fixture feed; ranks 1..4 take the same number of
/// padding steps, which is the steady state of a free-running fleet. Compared
/// against the same replay on a single-rank mega executor: exact wherever the
/// reference had a real margin, and the run reports whether it was bitwise.
#[test]
#[ignore = "requires four Blackwell GPUs and the K3 checkpoint; run alone in its own process"]
fn ep4_mega_matches_ep1_mega() {
    let fixture = fixture();
    let Some(path) = checkpoint() else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let reference = single_rank_trace(&path, &fixture);
    let candidate = ep_trace(&path, &fixture, Peers::Idle);
    let what = "EP4 mega rank 0 vs EP1 mega, idle peers";
    let bitwise = report_divergence(&reference, &candidate, what);
    eprintln!("{what}: bitwise = {bitwise}");
    assert_within_noise_floor(&fixture, &reference, &candidate, what);
}

/// Gate 2 — a rank's rows are invariant to its peers' traffic, bit for bit.
///
/// The same rank-0 replay at the same world size, run twice: once with the
/// peers idle and once with every peer prefilling and decoding a prompt of its
/// own. Different experts claimed, different pool blocks, different ring
/// pressure. The block config is fixed at the protocol maximum precisely so
/// that none of that can reach a row's tile shape, and every per-row quantity
/// is row-local, so this has no slack: any difference is a defect.
#[test]
#[ignore = "requires four Blackwell GPUs and the K3 checkpoint; run alone in its own process"]
fn ep4_mega_is_invariant_to_peer_traffic() {
    let fixture = fixture();
    let Some(path) = checkpoint() else {
        eprintln!("skipping: {CHECKPOINT_ENV} is not set to a mounted checkpoint");
        return;
    };
    let idle = ep_trace(&path, &fixture, Peers::Idle);
    let busy = ep_trace(&path, &fixture, Peers::Busy);
    let what = "EP4 mega rank 0, busy peers vs idle peers";
    assert!(
        report_divergence(&idle, &busy, what),
        "{what}: rank 0's own rows moved when its peers' traffic did. With one block config \
         fixed for every rank and every step, nothing about a row's arithmetic may depend on \
         what its neighbours are sending — see the first divergence printed above."
    );
}
