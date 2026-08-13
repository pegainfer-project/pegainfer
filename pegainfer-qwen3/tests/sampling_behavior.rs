//! Behavioral coverage for the honored sampling params (#243 Problem B).
//!
//! The frontend guard pins "unsupported params are rejected"; these tests pin
//! the other half — `temperature` / `top_k` / `top_p` actually steer the
//! sampler. The #237 failure class (request params silently falling back to
//! greedy) is caught by the diversity assertion; the collapse identities
//! (`top_k=1`, tiny `top_p`) catch the inverse (sampling ignoring the masks).
//!
//! Per-request seeds are rejected at the frontend and the legacy per-row
//! sampler is not run-to-run deterministic (#284), so seed determinism is
//! intentionally not asserted here.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::sampler::SamplingParams;

mod common;

use common::harness::EngineHarness;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const GENERATED_TOKENS: usize = 32;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 sampling_behavior: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Submit one request and collect the generated token ids until it finishes.
fn generate(engine: &EngineHarness, prompt_tokens: Vec<u32>, params: SamplingParams) -> Vec<u32> {
    engine.generate(prompt_tokens, params, GENERATED_TOKENS)
}

#[test]
fn sampling_params_steer_the_sampler() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = EngineHarness::new(
        pegainfer_qwen3::start_engine(
            Path::new(&model_path),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                seed: 42,
                ..EngineLoadOptions::default()
            },
        )
        .expect("failed to start engine"),
    );
    let tokenizer = common::load_tokenizer(&model_path);

    // Branchy continuation so unrestricted sampling has real entropy to show.
    let prompt = "Here is a short story about a dragon. Once upon a time";
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");

    // Warm the prefix cache first so every measured run takes the same
    // (cache-hit) prefill path; a cold-vs-warm comparison could flip a
    // near-tie token through bf16 wobble and fail spuriously.
    let _ = generate(&handle, prompt_tokens.clone(), SamplingParams::default());

    // Greedy is repeatable token-for-token.
    let greedy = generate(&handle, prompt_tokens.clone(), SamplingParams::default());
    assert_eq!(
        greedy.len(),
        GENERATED_TOKENS,
        "EOS inside a 32-token story tail is unexpected"
    );
    let greedy_again = generate(&handle, prompt_tokens.clone(), SamplingParams::default());
    assert_eq!(greedy, greedy_again, "greedy decode must be deterministic");

    // top_k = 1 leaves a single survivor: identical to greedy at any
    // temperature.
    let top_k_one = generate(
        &handle,
        prompt_tokens.clone(),
        SamplingParams {
            temperature: 0.8,
            top_k: 1,
            ..SamplingParams::default()
        },
    );
    assert_eq!(top_k_one, greedy, "top_k=1 must collapse to greedy");

    // The nucleus is the smallest descending-probability prefix whose mass
    // reaches top_p. The largest probability is always >= 1/vocab (~6.6e-6
    // for Qwen3), so with top_p below that bound the nucleus is exactly the
    // argmax for ANY distribution — the identity holds even on high-entropy
    // steps where the top token carries little mass.
    let top_p_tiny = generate(
        &handle,
        prompt_tokens.clone(),
        SamplingParams {
            temperature: 1.0,
            top_p: 1e-6,
            ..SamplingParams::default()
        },
    );
    assert_eq!(top_p_tiny, greedy, "top_p=1e-6 must collapse to greedy");

    // Unrestricted high-temperature sampling must actually sample: four runs
    // of 32 tokens collapsing to one sequence means the params were dropped
    // on the way to the sampler (#237's silent greedy fallback).
    let hot = SamplingParams {
        temperature: 1.5,
        top_k: -1,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let runs: Vec<Vec<u32>> = (0..4)
        .map(|_| generate(&handle, prompt_tokens.clone(), hot))
        .collect();
    assert!(
        runs.iter().any(|run| *run != runs[0]),
        "4 high-temperature runs were token-identical — sampling params are not reaching the sampler"
    );
}
