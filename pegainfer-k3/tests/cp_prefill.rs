//! Kimi-K3 context-parallel prefill gates: CP4 against CP1 on the pruned
//! 224-expert checkpoint at EP4 (`docs/models/k3/cp-lane-design.md`, M0).
//!
//! One prompt, two runs on the SAME four-rank expert-parallel world:
//!
//! * **CP1 baseline** — rank 0 walks the whole prompt through the existing
//!   chunked prefill while ranks 1..3 cover its chunk steps with padding
//!   decode steps (the free-running EP contract: only launch counts pair).
//! * **CP4** — the prompt splits into four contiguous segments, one chunk
//!   step per rank, KCP affine state merge + conv halo + MLA latent exchange
//!   stitching them back together.
//!
//! The gate compares the boundary logits and the sampled boundary token. The
//! two paths are NOT bitwise (per-rank FlashKDA chunk boundaries move, the
//! KCP merge accumulates in a different order, the FMHA runs at a different
//! `t_q`), so the split shows up as bf16 accumulation noise that diffuses
//! with depth. Measured on the pruned checkpoint @16k: 1 layer (pure MLA)
//! 1.6e-2, 2 layers 2.0e-2, 4 layers 4.2e-2, full 93 layers 2.5e-1 — a
//! `sqrt(layers)` walk off a ~2e-2 per-split floor. The rel-L2 bar scales
//! accordingly; the hard gates are the argmax and the sampled token. For
//! scale: the M·S-vs-S·M merge-orientation bug measured 2.3e-1 at 4 layers
//! (5x over the depth bar) AND flipped the argmax.
//!
//! Manual gates: CI compiles them, a 4-GPU Blackwell box with the checkpoint
//! runs them with `--ignored`. Point `PEGAINFER_K3_TEST_224` at the
//! 224-expert checkpoint. Full-depth 16k needs ~30 GiB headroom per device
//! beyond the weights; `PEGAINFER_K3_LAYERS` shrinks the build for plumbing
//! runs (numerics then gate a truncated model — still a real CP1-vs-CP4
//! comparison).
//!
//! ```text
//! cargo test --release -p pegainfer-k3 --test cp_prefill \
//!   cp4_prefill_matches_cp1 -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_k3::K3CpGroup;
use pegainfer_k3::K3EpRendezvous;
use pegainfer_k3::K3Executor;
use pegainfer_k3::K3ExecutorConfig;
use pegainfer_k3::StepExecutor;

const CHECKPOINT_ENV: &str = "PEGAINFER_K3_TEST_224";
const EP_SIZE: usize = 4;
/// Serving ceiling for the gate; the one-shot 16k prompt is the yardstick
/// the external vLLM TTFT baseline was measured at (see
/// `docs/models/k3/cp-lane-design.md`).
const MAX_CTX: usize = 16384;
/// Logits gate: relative L2 across the vocab row, CP4 vs CP1. The noise
/// floor is a `sqrt(depth)` diffusion off ~2e-2 per split (module docs);
/// `4e-2·sqrt(layers)` carries ~1.5x headroom over every measured depth
/// while still rejecting the merge-orientation bug 5x over.
fn rel_l2_bar() -> f64 {
    let layers = std::env::var("PEGAINFER_K3_LAYERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(93);
    4e-2 * (layers as f64).sqrt()
}

/// The prompt length the logits gate runs at (`PEGAINFER_K3_CP_PROMPT`,
/// default [`MAX_CTX`]).
fn gate_prompt_ceiling() -> usize {
    std::env::var("PEGAINFER_K3_CP_PROMPT")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(MAX_CTX)
}

fn checkpoint() -> PathBuf {
    let path = std::env::var(CHECKPOINT_ENV)
        .unwrap_or_else(|_| panic!("set {CHECKPOINT_ENV} to the 224-expert checkpoint directory"));
    PathBuf::from(path)
}

fn config() -> K3ExecutorConfig {
    let mut config = K3ExecutorConfig::default().from_env();
    // `PEGAINFER_K3_CP_PROMPT` raises the serving ceiling with the prompt:
    // the 16896-token MegaMoE protocol maximum makes CP4 span 64k prompts in
    // one superstep, and the gate must be able to follow it up.
    config.max_ctx = MAX_CTX.max(gate_prompt_ceiling());
    // One decode slot: the gate never decodes, and a slot's KDA state slab is
    // ~1 GiB across the full depth.
    config.max_batch = 1;
    config
}

/// Deterministic varied token ids. Both paths see the identical sequence, so
/// this only needs to exercise varied embeddings, not read as prose.
fn synth_prompt(len: usize) -> Vec<u32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            1_000 + (state >> 33) as u32 % 49_000
        })
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut diff = 0f64;
    let mut norm = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        diff += f64::from(x - y) * f64::from(x - y);
        norm += f64::from(y) * f64::from(y);
    }
    (diff / norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

/// What one rank thread reports back.
#[derive(Default)]
struct RankReport {
    /// Rank 0: (sampled token, boundary logits, wall seconds) per CP1 run.
    cp1: Vec<(u32, Vec<f32>, f64)>,
    /// Last rank: the same per CP4 run.
    cp4: Vec<(u32, Vec<f32>, f64)>,
}

/// Run `lengths x runs`: for each, one CP1 baseline pass and one CP4 pass,
/// on every rank in lockstep. Returns rank 0's and the last rank's reports.
fn run_gang(lengths: &[usize], runs: usize) -> (RankReport, RankReport) {
    let path = checkpoint();
    let rendezvous = K3EpRendezvous::new(EP_SIZE);
    let group = K3CpGroup::new(EP_SIZE).expect("CP group");
    let mut reports: Vec<Option<RankReport>> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for rank in 0..EP_SIZE {
            let path = path.clone();
            let rendezvous = rendezvous.clone();
            let group = group.clone();
            let handle = scope.spawn(move || -> RankReport {
                let mut executor = K3Executor::load_ep(&path, rank, rank, config(), rendezvous)
                    .unwrap_or_else(|error| panic!("rank {rank} load: {error:#}"));
                let chunk = executor.chunk_tokens();
                let mut report = RankReport::default();
                for &len in lengths {
                    let prompt = synth_prompt(len);
                    for _ in 0..runs {
                        // CP1 baseline: rank 0 prefills, peers pad its chunks.
                        if rank == 0 {
                            let start = Instant::now();
                            let token = executor
                                .prefill(0, &prompt, &SamplingParams::default())
                                .expect("CP1 prefill");
                            let seconds = start.elapsed().as_secs_f64();
                            let logits = executor.last_logits(0).expect("CP1 logits");
                            report.cp1.push((token, logits, seconds));
                        } else {
                            for _ in 0..len.div_ceil(chunk) {
                                executor.decode(&[]).expect("padding step");
                            }
                        }
                        // CP4: everyone owns a segment of the same prompt.
                        let start = Instant::now();
                        let token = executor
                            .prefill_cp(0, &prompt, &group, rank)
                            .expect("CP4 prefill");
                        let seconds = start.elapsed().as_secs_f64();
                        if rank + 1 == EP_SIZE {
                            let token = token.expect("last CP rank samples the boundary");
                            let logits = executor.last_logits(0).expect("CP4 logits");
                            report.cp4.push((token, logits, seconds));
                        }
                    }
                }
                report
            });
            handles.push(handle);
        }
        reports = handles
            .into_iter()
            .map(|handle| Some(handle.join().expect("a CP rank thread panicked")))
            .collect();
    });
    let last = reports[EP_SIZE - 1].take().expect("last rank report");
    let first = reports[0].take().expect("rank 0 report");
    (first, last)
}

/// THE M0 gate: CP4 boundary logits match CP1 at the 16k ceiling and one
/// token under it — the odd length splits into unequal segments, which is
/// what arbitrary serving prompts do (equal splits are the special case).
#[test]
#[ignore = "needs 4 GPUs and the 224-expert checkpoint"]
fn cp4_prefill_matches_cp1() {
    let ceiling = gate_prompt_ceiling();
    let lengths = [ceiling, ceiling - 1];
    let (first, last) = run_gang(&lengths, 1);
    for (index, &len) in lengths.iter().enumerate() {
        let (cp1_token, cp1_logits, cp1_seconds) = &first.cp1[index];
        let (cp4_token, cp4_logits, cp4_seconds) = &last.cp4[index];
        let rel = rel_l2(cp4_logits, cp1_logits);
        let cp1_argmax = argmax(cp1_logits);
        let cp4_argmax = argmax(cp4_logits);
        println!(
            "CP1 vs CP4 @ {len} tokens: rel_l2={rel:.3e}, argmax {cp1_argmax} vs {cp4_argmax}, \
             sampled {cp1_token} vs {cp4_token}, wall {:.1} ms vs {:.1} ms",
            cp1_seconds * 1e3,
            cp4_seconds * 1e3,
        );
        let mut top: Vec<(usize, f32)> = cp1_logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (id, value) in top.iter().take(5) {
            println!(
                "  cp1 top: id {id} logit {value:.4} (cp4 {:.4})",
                cp4_logits[*id]
            );
        }
        assert_eq!(cp1_token, cp4_token, "boundary samples diverged @ {len}");
        assert_eq!(cp1_argmax, cp4_argmax, "boundary argmax diverged @ {len}");
        let bar = rel_l2_bar();
        assert!(
            rel < bar,
            "CP4 boundary logits drifted {rel:.3e} from CP1 @ {len} (bar {bar:.3e})"
        );
    }
}

/// The crossover table: T_cp(4) vs T_cp(1) at the external vLLM baseline's
/// sweep lengths (the full 128–16k ladder), min of 4, one-shot per length.
/// Comparison caliper: the vLLM numbers are server e2e TTFT minus the fixed
/// intercept (its 128-token min); ours here are the forward wall clock
/// directly — the honest cross-engine comparison is HTTP e2e on both sides
/// (see `docs/models/k3/cp-lane-design.md` for the numbers).
#[test]
#[ignore = "needs 4 GPUs and the 224-expert checkpoint"]
fn cp4_prefill_ttft_sweep() {
    let lengths = [128usize, 256, 512, 1024, 2048, 4096, 8192, 16384];
    let runs = 4;
    let (first, last) = run_gang(&lengths, runs);
    println!("| tokens | CP1 min ms | CP4 min ms | speedup |");
    println!("|-------:|-----------:|-----------:|--------:|");
    for (index, &len) in lengths.iter().enumerate() {
        let window = |report: &[(u32, Vec<f32>, f64)]| {
            report[index * runs..(index + 1) * runs]
                .iter()
                .map(|(_, _, seconds)| seconds * 1e3)
                .fold(f64::INFINITY, f64::min)
        };
        let cp1 = window(&first.cp1);
        let cp4 = window(&last.cp4);
        println!("| {len} | {cp1:.1} | {cp4:.1} | {:.2}x |", cp1 / cp4);
    }
}
