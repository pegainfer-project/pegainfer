//! Scheduler robustness IT for Qwen3-4B.
//!
//! Numerical regression lives in `hf_golden_gate.rs` (tolerance vs an HF golden);
//! this test owns the one thing that gate does not — that the scheduler keeps
//! running when a client hangs up mid-flight. We submit a request, flip its
//! abort flag immediately, and assert the engine retires that request silently
//! (no terminal update) and still serves the next one. It drives the real
//! engine + `submit` rather than a mocked scheduler, so it exercises the actual
//! abort-flag retirement path.
//!
//! Started via the `--batch-invariant` builder, it also checks that the flag sets
//! the pin policy before serving. CUDA-graph pin behavior is covered by
//! `batch_invariance_decode_gemm_graph`.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when the model is
//! absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;
use std::time::Duration;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kernels::ops::pin_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
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
                "skipping qwen3 scheduler_robustness: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
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

/// A client that aborts right after submitting must not wedge the engine: the
/// scheduler observes the abort flag, retires the orphaned request without a
/// terminal, and later requests are served.
#[test]
fn scheduler_survives_client_abort() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let engine = EngineHarness::new(
        pegainfer_qwen3::start_engine_with_offload(
            Path::new(&model_path),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                seed: 42,
                ..EngineLoadOptions::default()
            },
            pegainfer_qwen3::Qwen3OffloadOptions::disabled(),
            true,
            pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
            pegainfer_qwen3::Qwen3MemoryOptions::default(),
            pegainfer_qwen3::DecodeOverlap::Off,
            true,
            None,
        )
        .expect("failed to start engine"),
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Pin,
        "--batch-invariant did not reach the pin policy before serving"
    );
    let tokenizer = common::load_tokenizer(&model_path);

    // Submit, then abort immediately — the scheduler should observe the abort
    // flag and retire the request silently rather than spinning on it. No
    // terminal arrives for an aborted request, so nothing is awaited here.
    let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
    let stream = engine.submit(common::harness::request(
        prompt_tokens,
        SamplingParams::default(),
        10,
    ));
    stream.control.abort();
    drop(stream);
    std::thread::sleep(Duration::from_millis(500));

    // Barrier: drain the aborted orphan before the counted runs (else its prefill leaks into prefill_served).
    let _ = generate_text(&engine, &tokenizer, "Hello", 1);

    reset_numeric_policy_counters();
    let _ = generate_text(&engine, &tokenizer, "Hello", 1);
    let prefill_served = pin_served();

    reset_numeric_policy_counters();
    let text = generate_text(&engine, &tokenizer, "Hello", 5);
    let full_served = pin_served();
    eprintln!("[scheduler-robustness] pin served: prefill={prefill_served} full={full_served}");

    assert!(!text.is_empty(), "scheduler dead after client abort");
    // The pin ran decode GEMMs beyond prefill, not just prefill's.
    assert!(
        full_served > prefill_served,
        "pin served no decode GEMM beyond prefill (prefill={prefill_served} full={full_served}) — flag→builder→graph-capture may be broken"
    );
}
