//! Sampled speculative decoding must be *distribution-lossless* (#512).
//!
//! Sampled-verify (the #589 mechanism) commits only tokens the verify pass
//! sampled from the target model's own distribution — spec-on and spec-off
//! differ in *how* tokens are produced, never in *what law* they follow. Token-by-token
//! equality is meaningless for sampled decoding (each run is a fresh draw), so
//! the gate is statistical: for a fixed prompt set and representative sampling
//! configs, generate many independent runs per arm and compare the per-position
//! marginal token distributions.
//!
//! Test: at every (config, prompt, position) cell, a two-sample permutation
//! test on the max studentized per-token frequency difference (max-z; TV is
//! reported as the effect size) between the arms' token histograms
//! (1,999 permutations, seeded — deterministic), then Benjamini–Hochberg FDR
//! at q = 0.05 across all cells. Gate: **zero rejected cells**. The max
//! per-cell TV is reported as the effect-size readout. `ignore_eos` keeps every
//! run exactly `POSITIONS` long so the position marginals are uncensored (EOS
//! is then an ordinary token the histograms still compare).
//!
//! The machinery has teeth: `permutation_gate_detects_bias` (pure CPU, always
//! runs) rejects a deliberately biased synthetic sampler, and the
//! `..._null_check` GPU test runs spec-on vs spec-on, which must pass — so a
//! failure of the main gate means a real distribution shift, not harness noise.
//!
//! Runs the two engines sequentially (spec-off arm dropped before the
//! speculative engine loads) so only one Qwen3-4B is resident at a time.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the DFlash drafter. Set
//! `PEGAINFER_TEST_MODEL_PATH` (target) and `PEGAINFER_DFLASH_TEST_MODEL_PATH`
//! (drafter); skips cleanly when either is absent. Knobs:
//! `PEGAINFER_EQ_RUNS` (default 256 per arm), `PEGAINFER_EQ_BATCH` (default
//! 16 concurrent identical requests per wave — the arms see the same batch
//! shapes), `PEGAINFER_EQ_OUT` (optional JSON dump of every token matrix, for
//! offline scrutiny).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod common;

use common::harness::EngineHarness;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B-DFlash-b16");

/// Positions compared per run; also the exact run length (`ignore_eos`).
const POSITIONS: usize = 32;
/// Permutations per cell; p-values resolve down to 1/2000.
const PERMUTATIONS: usize = 1999;
/// BH false-discovery rate across all (config, prompt, position) cells.
const FDR_Q: f64 = 0.05;

/// Checked-in prompt set (reproducibility): sharegpt-flavored variety — chat,
/// instruction, code, math, structured output, free-form continuation.
const PROMPTS: [&str; 8] = [
    "Tell me about the history of the Roman Empire.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "What are the main differences between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "If a train travels 120 km in 1.5 hours, what is its average speed? Explain step by step.",
    "List five common causes of memory leaks in C++ programs, one per line.",
    "The old lighthouse keeper looked out at the storm and",
    "Explain how photosynthesis works to a ten-year-old.",
];

/// Representative sampling configs: pure top-p, pure top-k, combined, and a
/// min_p config — sampled-verify (#512) admits the full sampling surface, so
/// min_p rides the speculative path and belongs in the equivalence matrix.
/// `(temperature, top_k, top_p, min_p)`. Seeded requests get the *stronger*
/// determinism gate (`dflash_sampled_seeded_determinism`) instead.
const CONFIGS: [(f32, i32, f32, f32); 4] = [
    (0.8, -1, 0.95, 0.0),
    (1.0, 50, 1.0, 0.0),
    (0.7, 20, 0.9, 0.0),
    (0.8, -1, 1.0, 0.05),
];

/// One config's collected tokens: `[prompt][run][position]`.
type ConfigTokens = Vec<Vec<Vec<u32>>>;

/// Serialize engine-holding test bodies — one engine resident at a time.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn target_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping sampled-equivalence gate: {MODEL_PATH}/config.json missing; set PEGAINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping sampled-equivalence gate: {DRAFT_PATH}/config.json missing; set PEGAINFER_DFLASH_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn env_knob(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        // The speculative engine forces the prefix cache off; match it on the
        // baseline so both arms take the same cold prefill path.
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(
            0.85,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        )
        .validate()
        .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: draft,
    }
}

fn sampling(cfg: (f32, i32, f32, f32)) -> SamplingParams {
    SamplingParams {
        temperature: cfg.0,
        top_k: cfg.1,
        top_p: cfg.2,
        min_p: cfg.3,
        // Fixed-length runs: position marginals stay uncensored, and EOS is
        // compared as an ordinary token.
        ignore_eos: true,
        ..SamplingParams::default()
    }
}

/// One wave of `n` identical concurrent requests; returns each run's tokens.
/// Concurrency makes the engine form real batches — the serving shape both
/// arms are supposed to be equivalent under — and the two arms see the same
/// wave sizes so batch-composition numerics cancel out of the comparison.
fn sample_wave(
    engine: &EngineHarness,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    n: usize,
) -> Vec<Vec<u32>> {
    // Submit all up front so they coexist in the engine and form real batches.
    let streams: Vec<_> = (0..n)
        .map(|_| engine.submit(request(prompt_tokens.to_vec(), *params, POSITIONS)))
        .collect();

    // Fold each stream to its terminal (updates are buffered per request, so
    // the fold order doesn't matter — they all ran concurrently).
    streams
        .into_iter()
        .map(|stream| {
            let tokens = stream.expect_finished().tokens;
            assert_eq!(
                tokens.len(),
                POSITIONS,
                "ignore_eos run must be exactly {POSITIONS} tokens"
            );
            tokens
        })
        .collect()
}

/// `runs` independent generations for every (config, prompt), in waves of
/// `batch`. Indexed `[config][prompt][run][position]`.
fn collect_arm(
    engine: &EngineHarness,
    encoded: &[Vec<u32>],
    runs: usize,
    batch: usize,
) -> Vec<ConfigTokens> {
    CONFIGS
        .iter()
        .map(|&cfg| {
            let params = sampling(cfg);
            encoded
                .iter()
                .map(|prompt_tokens| {
                    let mut rows = Vec::with_capacity(runs);
                    while rows.len() < runs {
                        let n = batch.min(runs - rows.len());
                        rows.extend(sample_wave(engine, prompt_tokens, &params, n));
                    }
                    rows
                })
                .collect()
        })
        .collect()
}

/// Total-variation distance between the empirical distributions of two token
/// samples: `0.5 * Σ_tok |p_a(tok) - p_b(tok)|`.
fn tv_distance(a: &[u32], b: &[u32]) -> f64 {
    let mut counts: HashMap<u32, (usize, usize)> = HashMap::new();
    for &t in a {
        counts.entry(t).or_default().0 += 1;
    }
    for &t in b {
        counts.entry(t).or_default().1 += 1;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    0.5 * counts
        .values()
        .map(|&(ca, cb)| (ca as f64 / na - cb as f64 / nb).abs())
        .sum::<f64>()
}

/// Max studentized per-token frequency difference between the two samples.
/// This is the permutation test's statistic — chosen over total variation
/// because the realistic sampled-verify bug shifts mass toward ONE token per
/// position (accepting the draft's argmax when it shouldn't), and a max-z is
/// an order of magnitude more sensitive to a single-token shift than TV
/// (standalone power check: 10% mass onto a 5%-base-rate token at n=256/arm —
/// TV permutation detects 11/64 synthetic cells after BH, max-z detects 47).
fn max_z(a: &[u32], b: &[u32]) -> f64 {
    let mut counts: HashMap<u32, (usize, usize)> = HashMap::new();
    for &t in a {
        counts.entry(t).or_default().0 += 1;
    }
    for &t in b {
        counts.entry(t).or_default().1 += 1;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    counts
        .values()
        .map(|&(ca, cb)| {
            let (pa, pb) = (ca as f64 / na, cb as f64 / nb);
            let pooled = (ca + cb) as f64 / (na + nb);
            let se = (pooled * (1.0 - pooled) * (1.0 / na + 1.0 / nb)).sqrt();
            (pa - pb).abs() / se.max(1e-12)
        })
        .fold(0.0f64, f64::max)
}

/// Two-sample permutation test (max-z statistic; TV reported as effect size).
/// Returns `(observed_tv, p)` with the add-one estimate
/// `p = (1 + #{null ≥ obs}) / (1 + PERMUTATIONS)`, unbiased under
/// exchangeability (exactly the H0 "same sampling law").
fn permutation_test(a: &[u32], b: &[u32], rng: &mut StdRng) -> (f64, f64) {
    let observed = max_z(a, b);
    let mut pool: Vec<u32> = a.iter().chain(b.iter()).copied().collect();
    let mut at_least = 0usize;
    for _ in 0..PERMUTATIONS {
        // Partial Fisher–Yates: shuffle the first |a| slots, split, re-measure.
        for i in 0..a.len() {
            let j = rng.random_range(i..pool.len());
            pool.swap(i, j);
        }
        if max_z(&pool[..a.len()], &pool[a.len()..]) >= observed {
            at_least += 1;
        }
    }
    let p = (1 + at_least) as f64 / (1 + PERMUTATIONS) as f64;
    (tv_distance(a, b), p)
}

/// Benjamini–Hochberg: which of `pvals` are rejected at FDR `q`.
fn bh_rejections(pvals: &[f64], q: f64) -> Vec<bool> {
    let m = pvals.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| pvals[i].total_cmp(&pvals[j]));
    let mut cutoff = None;
    for (rank, &idx) in order.iter().enumerate() {
        if pvals[idx] <= q * (rank + 1) as f64 / m as f64 {
            cutoff = Some(rank);
        }
    }
    let mut rejected = vec![false; m];
    if let Some(k) = cutoff {
        for &idx in &order[..=k] {
            rejected[idx] = true;
        }
    }
    rejected
}

struct Cell {
    config: usize,
    prompt: usize,
    position: usize,
    tv: f64,
    p: f64,
}

/// Compare two collected arms cell by cell; returns every cell's TV and p.
fn compare_arms(arm_a: &[ConfigTokens], arm_b: &[ConfigTokens]) -> Vec<Cell> {
    // Seeded: the verdict is a pure function of the collected tokens.
    let mut rng = StdRng::seed_from_u64(0x512_512);
    let mut cells = Vec::new();
    for (c, (pa, pb)) in arm_a.iter().zip(arm_b).enumerate() {
        for (i, (rows_a, rows_b)) in pa.iter().zip(pb).enumerate() {
            for t in 0..POSITIONS {
                let col_a: Vec<u32> = rows_a.iter().map(|r| r[t]).collect();
                let col_b: Vec<u32> = rows_b.iter().map(|r| r[t]).collect();
                let (tv, p) = permutation_test(&col_a, &col_b, &mut rng);
                cells.push(Cell {
                    config: c,
                    prompt: i,
                    position: t,
                    tv,
                    p,
                });
            }
        }
    }
    cells
}

/// Gate on the compared cells: BH-FDR, zero rejections. Prints the readout.
fn assert_equivalent(cells: &[Cell], label: &str) {
    let pvals: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let rejected = bh_rejections(&pvals, FDR_Q);
    let max_tv = cells.iter().map(|c| c.tv).fold(0.0f64, f64::max);
    let min_p = pvals.iter().copied().fold(1.0f64, f64::min);
    eprintln!(
        "{label}: {} cells, max TV {max_tv:.4}, min p {min_p:.4}, BH rejections {}",
        cells.len(),
        rejected.iter().filter(|&&r| r).count()
    );
    let failures: Vec<String> = cells
        .iter()
        .zip(&rejected)
        .filter(|&(_, &r)| r)
        .map(|(c, _)| {
            format!(
                "config {} prompt {} position {}: TV {:.4}, p {:.4}",
                c.config, c.prompt, c.position, c.tv, c.p
            )
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{label}: {} cells rejected at FDR {FDR_Q}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Optional raw dump for offline scrutiny (`PEGAINFER_EQ_OUT`).
fn dump_json(path: &str, arms: [(&str, &[ConfigTokens]); 2]) {
    let mut out = String::from("{");
    for (a, (name, arm)) in arms.iter().enumerate() {
        if a > 0 {
            out.push(',');
        }
        let _ = write!(out, "\"{name}\":[");
        for (c, per_prompt) in arm.iter().enumerate() {
            if c > 0 {
                out.push(',');
            }
            out.push('[');
            for (i, rows) in per_prompt.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                for (r, row) in rows.iter().enumerate() {
                    if r > 0 {
                        out.push(',');
                    }
                    let _ = write!(
                        out,
                        "[{}]",
                        row.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
                    );
                }
                out.push(']');
            }
            out.push(']');
        }
        out.push(']');
    }
    out.push('}');
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(out.as_bytes()))
        .unwrap_or_else(|err| panic!("failed to write {path}: {err}"));
    eprintln!("raw token matrices dumped to {path}");
}

/// Main gate: sampled spec-on vs spec-off statistical equivalence.
#[test]
#[ignore = "GPU equivalence gate — needs Qwen3-4B + DFlash weights and ~20 GiB VRAM headroom"]
fn dflash_sampled_equivalence_gate() {
    let _gpu = GPU.lock().unwrap();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let runs = env_knob("PEGAINFER_EQ_RUNS", 256);
    let batch = env_knob("PEGAINFER_EQ_BATCH", 16);

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // Arm A: plain decode (no draft model loaded — the spec path cannot run).
    let spec_off = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("failed to start spec-off engine"),
        );
        let arm = collect_arm(&engine, &encoded, runs, batch);
        drop(engine);
        std::thread::sleep(Duration::from_secs(2));
        arm
    };

    // Arm B: DFlash speculative engine; sampled requests ride chain rejection.
    let spec_on = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(
                Path::new(&model_path),
                launch_options(Some(PathBuf::from(&draft_path))),
            )
            .expect("failed to start speculative engine"),
        );
        let arm = collect_arm(&engine, &encoded, runs, batch);
        drop(engine);
        arm
    };

    if let Ok(path) = std::env::var("PEGAINFER_EQ_OUT") {
        dump_json(&path, [("spec_off", &spec_off), ("spec_on", &spec_on)]);
    }

    let cells = compare_arms(&spec_off, &spec_on);
    assert_equivalent(&cells, "sampled equivalence spec-off vs spec-on");
}

/// Null check: two independent spec-on collections must pass the same gate.
/// If this fails, the harness (not the engine) is flagging noise as signal.
#[test]
#[ignore = "GPU null check — needs Qwen3-4B + DFlash weights"]
fn dflash_sampled_equivalence_null_check() {
    let _gpu = GPU.lock().unwrap();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let runs = env_knob("PEGAINFER_EQ_RUNS", 256);
    let batch = env_knob("PEGAINFER_EQ_BATCH", 16);

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );
    let first = collect_arm(&engine, &encoded, runs, batch);
    let second = collect_arm(&engine, &encoded, runs, batch);
    drop(engine);

    let cells = compare_arms(&first, &second);
    assert_equivalent(&cells, "null check spec-on vs spec-on");
}

/// Seeded requests get a byte-exact gate rather than a statistical one
/// (glm52's M4 doc: byte gates are for determinism): the same seeded request
/// submitted twice to the SAME spec-on engine must reproduce its tokens
/// exactly — seeded sampling is a pure function of (seed, step, distribution)
/// and c1 keeps the batch shape (hence the logits) deterministic.
#[test]
#[ignore = "GPU seeded-determinism gate — needs Qwen3-4B + DFlash weights"]
fn dflash_sampled_seeded_determinism() {
    let _gpu = GPU.lock().unwrap();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let tokens = tokenizer.encode(prompt, false).expect("encode failed");
        let params = SamplingParams {
            seed: Some(0x00C0_FFEE + i as u64),
            ..sampling(CONFIGS[0])
        };
        let first = sample_wave(&engine, &tokens, &params, 1);
        let second = sample_wave(&engine, &tokens, &params, 1);
        assert_eq!(
            first, second,
            "prompt {i}: a seeded spec-on request must replay byte-identically"
        );
    }
    drop(engine);
}

/// The statistical machinery must *detect* a real shift: a synthetic sampler
/// with a 10% probability mass moved to one token must be rejected, and an
/// identical-law pair must pass. Pure CPU — always runs.
#[test]
fn permutation_gate_detects_bias() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 256;
    // Unbiased law: uniform over 20 tokens. Biased law: token 0 absorbs an
    // extra 10% of the mass (TV = 0.095 from uniform).
    let draw_unbiased = |rng: &mut StdRng| -> u32 { rng.random_range(0..20u32) };
    let draw_biased = |rng: &mut StdRng| -> u32 {
        if rng.random_range(0.0..1.0f64) < 0.10 {
            0
        } else {
            rng.random_range(0..20u32)
        }
    };

    // 64 same-law cells must produce zero BH rejections…
    let mut same_cells = Vec::new();
    let mut test_rng = StdRng::seed_from_u64(0x512_512);
    for _ in 0..64 {
        let a: Vec<u32> = (0..n).map(|_| draw_unbiased(&mut rng)).collect();
        let b: Vec<u32> = (0..n).map(|_| draw_unbiased(&mut rng)).collect();
        let (_, p) = permutation_test(&a, &b, &mut test_rng);
        same_cells.push(p);
    }
    let same_rejected = bh_rejections(&same_cells, FDR_Q);
    assert_eq!(
        same_rejected.iter().filter(|&&r| r).count(),
        0,
        "same-law cells must not be rejected"
    );

    // …while 64 biased cells must be overwhelmingly rejected.
    let mut biased_cells = Vec::new();
    for _ in 0..64 {
        let a: Vec<u32> = (0..n).map(|_| draw_unbiased(&mut rng)).collect();
        let b: Vec<u32> = (0..n).map(|_| draw_biased(&mut rng)).collect();
        let (_, p) = permutation_test(&a, &b, &mut test_rng);
        biased_cells.push(p);
    }
    let biased_rejected = bh_rejections(&biased_cells, FDR_Q)
        .iter()
        .filter(|&&r| r)
        .count();
    // Measured 47/64 with the max-z statistic at these exact sizes; 36 leaves
    // RNG-stream margin while still failing hard if the statistic regresses
    // (the TV statistic this replaced detected only 11/64 here).
    assert!(
        biased_rejected >= 36,
        "a 10%-mass shift must be detected in most cells (got {biased_rejected}/64)"
    );
}
