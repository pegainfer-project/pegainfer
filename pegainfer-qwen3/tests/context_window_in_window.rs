//! In-window context regression IT for Qwen3-4B.
//!
//! Companion to `context_window.rs` (which proves *oversized* prompts are
//! rejected). This proves the *positive* side of the same fix: a prompt that is
//! larger than the old hardcoded 4096-entry RoPE table but still inside the
//! position-encoding window (`max_position_embeddings` = 40960) must be served
//! end-to-end.
//!
//! A 4097-token prompt spans positions 0..=4096. Position 4096 is exactly the
//! first index past the old table's bounds (valid 0..=4095) — before the fix it
//! read garbage out of bounds (and after the kernel trap was added, would
//! `__trap` the CUDA context). Generating even one token therefore requires the
//! resized RoPE cache to actually exist and be indexed at 4096; if the table
//! ever silently reverts to 4096 entries this test crashes instead of passing.
//!
//! Lives in its own test binary for the same reason as `context_window.rs`:
//! `cargo test` serializes test binaries but parallelizes `#[test]`s within one
//! binary, so two engines on one GPU would contend. One engine per file.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;

// Only the harness slice of the shared module is used here; this test never
// touches the tokenizer helpers.
#[allow(dead_code)]
mod common;

use common::harness::EngineHarness;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 context_window_in_window: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

#[test]
fn in_window_prompt_past_old_rope_table_is_served() {
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

    // 4097 tokens → positions 0..=4096. Position 4096 is the first index past the
    // old 4096-entry RoPE table; serving this prompt requires the resized cache.
    // Token id 1 is a valid vocab id — a forward pass actually runs here (unlike
    // the rejection test, which never reaches prefill).
    let prompt_tokens = vec![1u32; 4097];
    let outcome = engine
        .submit(common::harness::request(
            prompt_tokens,
            SamplingParams::default(),
            1,
        ))
        .outcome();

    match &outcome.terminal {
        Terminal::Finished { .. } => {}
        Terminal::Failed { message, .. } => {
            panic!("in-window prompt errored (resized RoPE cache not exercised?): {message}")
        }
        Terminal::Rejected { reason, .. } => {
            panic!("in-window prompt was wrongly rejected: {reason}")
        }
    }

    assert_eq!(
        outcome.tokens.len(),
        1,
        "expected exactly one generated token for a 4097-token prompt with max_tokens=1"
    );
}
