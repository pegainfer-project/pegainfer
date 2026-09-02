//! Behavioral coverage for Qwen3.5 sampling params on the scheduler path.
//!
//! This mirrors the Qwen3 test: `temperature` / `top_k` / `top_p` must steer
//! real generation, while `top_k=1` and tiny `top_p` collapse to greedy. It
//! guards #284 against silently falling back to greedy or dropping masks after
//! the single-row sampler was removed.
//!
//! Requires a CUDA GPU and Qwen3.5-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::sampler::SamplingParams;

mod common;

use common::harness::EngineHarness;

const GENERATED_TOKENS: usize = 32;

fn params(mut params: SamplingParams) -> SamplingParams {
    params.ignore_eos = true;
    params
}

/// Submit one request and collect the generated token ids until `Finished`.
fn generate(harness: &EngineHarness, prompt_tokens: Vec<u32>, params: SamplingParams) -> Vec<u32> {
    harness
        .submit(common::harness::request(
            prompt_tokens,
            params,
            GENERATED_TOKENS,
        ))
        .expect_finished()
        .tokens
}

#[test]
fn sampling_params_steer_the_qwen35_sampler() {
    let Some(model_path) = common::model_path_or_skip("sampling_params_steer_the_qwen35_sampler")
    else {
        return;
    };

    let harness = EngineHarness::new(
        pegainfer_qwen35::start_engine(
            Path::new(&model_path),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                seed: 42,
                ..EngineLoadOptions::default()
            },
            4,
            pegainfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS,
        )
        .expect("failed to start Qwen3.5 engine"),
    );
    let tokenizer = common::load_tokenizer(&model_path);

    let prompt = "Here is a short story about a dragon. Once upon a time";
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");

    let greedy_params = params(SamplingParams::default());

    let greedy = generate(&harness, prompt_tokens.clone(), greedy_params);
    assert_eq!(
        greedy.len(),
        GENERATED_TOKENS,
        "ignore_eos should force a full 32-token generation"
    );
    let greedy_again = generate(&harness, prompt_tokens.clone(), greedy_params);
    assert_eq!(greedy, greedy_again, "greedy decode must be deterministic");

    let top_k_one = generate(
        &harness,
        prompt_tokens.clone(),
        params(SamplingParams {
            temperature: 0.8,
            top_k: 1,
            ..SamplingParams::default()
        }),
    );
    assert_eq!(top_k_one, greedy, "top_k=1 must collapse to greedy");

    let top_p_tiny = generate(
        &harness,
        prompt_tokens.clone(),
        params(SamplingParams {
            temperature: 1.0,
            top_p: 1e-6,
            ..SamplingParams::default()
        }),
    );
    assert_eq!(top_p_tiny, greedy, "top_p=1e-6 must collapse to greedy");

    let hot = params(SamplingParams {
        temperature: 1.5,
        top_k: -1,
        top_p: 1.0,
        ..SamplingParams::default()
    });
    let runs: Vec<Vec<u32>> = (0..4)
        .map(|_| generate(&harness, prompt_tokens.clone(), hot))
        .collect();
    assert!(
        runs.iter().any(|run| *run != runs[0]),
        "4 high-temperature runs were token-identical; sampling params are not reaching the sampler"
    );
}
