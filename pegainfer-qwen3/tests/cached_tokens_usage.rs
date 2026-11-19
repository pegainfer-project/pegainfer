//! Prefix-cache observability IT for Qwen3-4B (#246).
//!
//! The frontend reports `usage.prompt_tokens_details.cached_tokens` from the
//! per-request `RequestUpdate.cached_tokens` field. This test pins the engine
//! half of that contract: a cold prompt reports zero cached tokens, a warm
//! repeat of the same prompt reports a nonzero full-block count, and the count
//! never claims the whole prompt (the last token is always recomputed).
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::sampler::SamplingParams;

mod common;

use common::harness::EngineHarness;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const KV_BLOCK_SIZE: usize = 16;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 cached_tokens_usage: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Submit `prompt_tokens`, fold the stream to `Finished`, and return the
/// reported prefix-cache hit count. The first prefill chunk always reports,
/// so a cold run yields `Some(0)` rather than an absent count; the harness
/// fold asserts the count arrives at most once per request.
fn run_and_capture_cached(engine: &EngineHarness, prompt_tokens: Vec<u32>) -> usize {
    engine
        .submit(common::harness::request(
            prompt_tokens,
            SamplingParams::default(),
            4,
        ))
        .expect_finished()
        .cached_tokens
        .expect("cached_tokens must be reported before Finished")
}

#[test]
fn warm_repeat_reports_cached_tokens() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let engine = EngineHarness::new(
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

    let prompt = "The kv cache stores attention keys and values for every \
        generated token so the model never recomputes earlier positions. "
        .repeat(8);
    let prompt_tokens = tokenizer.encode(&prompt, false).expect("encode failed");
    let prompt_len = prompt_tokens.len();
    assert!(
        prompt_len > 2 * KV_BLOCK_SIZE,
        "prompt must span multiple KV blocks for a meaningful hit"
    );

    let cold = run_and_capture_cached(&engine, prompt_tokens.clone());
    assert_eq!(cold, 0, "cold run must report zero cached tokens");

    let warm = run_and_capture_cached(&engine, prompt_tokens);
    assert!(warm > 0, "warm repeat must report a prefix-cache hit");
    assert!(
        warm < prompt_len,
        "at least the last prompt token is always recomputed (warm={warm}, prompt={prompt_len})"
    );
    assert_eq!(
        warm % KV_BLOCK_SIZE,
        0,
        "hits are matched in full blocks (warm={warm})"
    );
    assert_eq!(
        warm,
        (prompt_len - 1) / KV_BLOCK_SIZE * KV_BLOCK_SIZE,
        "warm hit must cover every cacheable full block (prompt={prompt_len})"
    );
}
