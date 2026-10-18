//! Context-window admission IT for Qwen3-4B.
//!
//! A prompt longer than the model's position-encoding window must be rejected at
//! admission with a context-length error — and crucially *before* any prefill, so
//! the oversized sequence never reaches the RoPE kernel (whose bounds trap would
//! otherwise take down the CUDA context). Admission rejects on prompt length
//! alone, so this stays cheap despite the 60k-token prompt: no forward pass runs.
//! After the rejection the engine must keep serving normal requests.
//!
//! Lives in its own test binary (not `scheduler_robustness.rs`) because `cargo
//! test` runs test binaries sequentially but parallelizes `#[test]`s within one
//! binary — two engines on one GPU would contend. One engine-starting test per
//! file keeps them serialized.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

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
                "skipping qwen3 context_window: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Submit `prompt` and block until the request finishes; returns the decoded text.
fn generate_text(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let tokens = engine.generate(prompt_tokens, SamplingParams::default(), max_tokens);
    tokenizer.decode(&tokens, true).expect("decode failed")
}

#[test]
fn oversized_prompt_is_rejected_with_context_length_error() {
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

    // Qwen3-4B's max_position_embeddings is 40960; 60k tokens overflows it outright.
    // Token id is irrelevant — the request is rejected before any embedding lookup.
    let prompt_tokens = vec![1u32; 60_000];
    let outcome = engine
        .submit(common::harness::request(
            prompt_tokens,
            SamplingParams::default(),
            8,
        ))
        .outcome();

    match outcome.terminal {
        Terminal::Rejected { reason, .. } => {
            assert!(
                matches!(reason, RejectReason::ContextLength { .. }),
                "expected a context-length rejection, got: {reason}"
            );
        }
        Terminal::Failed { message, .. } => {
            panic!("oversized prompt errored instead of clean rejection: {message}")
        }
        Terminal::Finished { .. } => panic!("oversized prompt should be rejected at admission"),
    }

    // The engine must keep serving normal requests after the rejection.
    let tokenizer = common::load_tokenizer(&model_path);
    let text = generate_text(&engine, &tokenizer, "Hello", 5);
    assert!(
        !text.is_empty(),
        "scheduler dead after context-length rejection"
    );
}
