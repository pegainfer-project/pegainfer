//! DFlash speculative-decoding single-stream latency A/B.
//!
//! Speculative decoding's direct win is single-stream (batch=1) decode latency:
//! plain decode is memory-bound (one target forward per token), while spec
//! amortizes that forward over the accepted run. This measures end-to-end
//! wall-clock to generate a fixed token budget, speculative OFF vs ON, on the
//! same prompts and hardware, and reports the speedup.
//!
//! This is a measurement harness, not a pass/fail gate — it asserts only that
//! spec is not catastrophically slower (a guard against the draft mispredicting
//! everything). Read the printed numbers for the real signal. `--nocapture`.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the DFlash drafter. Set
//! `PEGAINFER_TEST_MODEL_PATH` + `PEGAINFER_DFLASH_TEST_MODEL_PATH`; skips when
//! either is absent. Single-stream only — throughput under load is a separate
//! `vllm bench serve` A/B.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

mod common;

use common::harness::EngineHarness;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B-DFlash-b16");
const GENERATED_TOKENS: usize = 256;

fn target_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => None,
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
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

/// Generate `GENERATED_TOKENS` greedily and return (token_count, elapsed).
/// Counts tokens per stream update — speculative decoding commits multi-token
/// spans, so one update may carry several.
fn timed_generate(engine: &EngineHarness, prompt_tokens: Vec<u32>) -> (usize, Duration) {
    let start = Instant::now();
    let mut stream = engine.submit(request(
        prompt_tokens,
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        GENERATED_TOKENS,
    ));

    let mut count = 0usize;
    loop {
        let update = stream
            .recv()
            .expect("engine closed the stream without a terminal");
        count += update.tokens.len();
        match update.terminal {
            Some(Terminal::Finished { .. }) => return (count, start.elapsed()),
            Some(Terminal::Failed { message, .. }) => panic!("generation failed: {message}"),
            Some(Terminal::Rejected { reason, .. }) => panic!("generation rejected: {reason}"),
            None => {}
        }
    }
}

/// Decode tok/s averaged over the prompts (one warm-up run discarded).
fn measure(engine: &EngineHarness, prompts: &[Vec<u32>]) -> f64 {
    // Warm up CUDA-graph capture / allocator on the first prompt.
    let _ = timed_generate(engine, prompts[0].clone());
    let mut tokens = 0usize;
    let mut elapsed = Duration::ZERO;
    for p in prompts {
        let (n, dt) = timed_generate(engine, p.clone());
        tokens += n;
        elapsed += dt;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

#[test]
fn dflash_speculative_single_stream_speedup() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        eprintln!(
            "skipping dflash perf A/B: set PEGAINFER_TEST_MODEL_PATH + PEGAINFER_DFLASH_TEST_MODEL_PATH"
        );
        return;
    };

    let tokenizer = common::load_tokenizer(&model_path);
    let prompts: Vec<Vec<u32>> = [
        "Write a short essay about the history of the Roman Empire.",
        "Explain how a transformer neural network works, step by step.",
        "List ten facts about the planet Mars and describe each one.",
    ]
    .iter()
    .map(|p| tokenizer.encode(p, false).expect("encode failed"))
    .collect();

    let baseline_tps = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("baseline engine"),
        );
        let tps = measure(&engine, &prompts);
        drop(engine);
        std::thread::sleep(Duration::from_secs(2));
        tps
    };

    let spec_tps = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(
                Path::new(&model_path),
                launch_options(Some(PathBuf::from(&draft_path))),
            )
            .expect("speculative engine"),
        );
        measure(&engine, &prompts)
    };

    let speedup = spec_tps / baseline_tps;
    eprintln!("───────────── DFlash single-stream decode A/B (bs=1) ─────────────");
    eprintln!("  spec OFF (plain decode): {baseline_tps:7.1} tok/s");
    eprintln!("  spec ON  (DFlash):       {spec_tps:7.1} tok/s");
    eprintln!("  speedup:                 {speedup:7.2}×");
    eprintln!("───────────────────────────────────────────────────────────────────────────");

    assert!(
        speedup > 0.8,
        "speculative decode is catastrophically slower ({speedup:.2}×) — draft likely mispredicting"
    );
}
