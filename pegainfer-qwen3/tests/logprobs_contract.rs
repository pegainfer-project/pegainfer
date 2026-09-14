//! Real scheduler/executor gate; skips when default model weights are absent.
#[path = "common/harness.rs"]
mod harness;

use std::path::Path;

use harness::EngineHarness;
use harness::Outcome;
use harness::request;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

const GPU_FRACTION: f64 = 0.32;
const MEMORY_MARGIN: usize = 256 * 1024 * 1024;
const PAGE_TOKENS: usize = 16;
const PREFILL_BUDGET: usize = 64;
const PROMPT_TOKENS: usize = 32;
const PROMPT_TOKEN_BASE: u32 = 1000;
const PROMPT_TOKEN_STRIDE: u32 = 17;
const GENERATED_TOKENS: usize = 6;
// Separate decode and whole-prompt BF16 paths differ numerically. Use 0.10 nat
// as a regression bound with headroom over the measured SM89/Qwen3-4B deltas;
// each run prints every difference so a new model/policy can be assessed.
const LOGPROB_TOLERANCE: f32 = 0.10;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => Some(MODEL_PATH.into()),
        Err(_) => {
            eprintln!(
                "skipping qwen3 logprobs_contract: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn engine(path: &str) -> EngineHarness {
    let options = Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: false,
        max_prefill_tokens: PREFILL_BUDGET,
        memory: Qwen3MemoryOptions::new(GPU_FRACTION, MEMORY_MARGIN, PAGE_TOKENS),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: None,
    };
    EngineHarness::new(
        pegainfer_qwen3::launch(Path::new(path), options).expect("launch real engine"),
    )
}

fn scored_request(
    prompt: &[u32],
    counts: (Option<usize>, Option<usize>),
) -> pegainfer_frontend::engine::Request {
    let params = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let mut req = request(prompt.to_vec(), params, GENERATED_TOKENS);
    req.logprobs = counts.0;
    req.prompt_logprobs = counts.1;
    req
}

fn assert_teacher_forced(generated: &Outcome, replayed: &Outcome, offset: usize) {
    let prompt_scores = replayed.prompt_echo.as_ref().unwrap();
    let mut max_delta = 0.0_f32;
    for i in 0..GENERATED_TOKENS {
        let expected = generated.logprobs[i].as_ref().unwrap().logprob;
        let actual = prompt_scores.logprobs[offset + i].as_ref().unwrap().logprob;
        let delta = (expected - actual).abs();
        max_delta = max_delta.max(delta);
        eprintln!(
            "teacher-forced position={i} decode={expected:.8} prompt={actual:.8} delta={delta:.8} nat"
        );
        assert!(
            delta <= LOGPROB_TOLERANCE,
            "position {i}: decode={expected}, whole-prompt={actual}"
        );
    }
    eprintln!("teacher-forced max_delta={max_delta:.8} nat");
}

#[test]
fn prompt_scores_bypass_cached_prefix_and_match_teacher_forced_decode() {
    let Some(path) = model_path_or_skip() else {
        return;
    };
    let engine = engine(&path);
    let prompt: Vec<_> = (0..PROMPT_TOKENS)
        .map(|i| PROMPT_TOKEN_BASE + i as u32 * PROMPT_TOKEN_STRIDE)
        .collect();
    engine
        .submit(scored_request(&prompt, (None, None)))
        .expect_finished();
    let sampled = engine
        .submit(scored_request(&prompt, (Some(0), None)))
        .expect_finished();
    assert!(sampled.cached_tokens.unwrap() > 0, "seeded prefix must hit");
    let extended = [prompt.clone(), sampled.tokens.clone()].concat();
    let replay = engine
        .submit(scored_request(&extended, (None, Some(0))))
        .expect_finished();
    assert_eq!(
        replay.cached_tokens,
        Some(0),
        "prompt scores bypass warm KV"
    );
    assert_teacher_forced(&sampled, &replay, prompt.len());
}
