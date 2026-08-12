//! Qwen3.5 scheduler-level chunked prefill regression tests.
//!
//! These tests exercise resumed prefill (`base_pos > 0`) through the real
//! scheduler path. A small `max_prefill_tokens` budget forces one request's
//! prompt to be prefilling across multiple scheduler steps; the same prompt is
//! also run with an effectively unchunked budget and the generated greedy token
//! ids must match.

use std::path::Path;

use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen35::runtime::GdnPrefillRuntimeEvidenceHandle;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const CHUNK_BUDGET: usize = 16;
const BASELINE_PREFILL_BUDGET: usize = 1 << 20;
const MAX_BATCH: usize = 2;
const GENERATED_TOKENS: usize = 8;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen35 chunked_prefill: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn start_engine(model_path: &str, max_prefill_tokens: usize) -> EngineHandle {
    pegainfer_qwen35::start_engine(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        MAX_BATCH,
        max_prefill_tokens,
    )
    .expect("failed to start Qwen3.5 engine")
}

fn start_flashinfer_engine(
    model_path: &str,
    max_prefill_tokens: usize,
) -> (EngineHandle, GdnPrefillRuntimeEvidenceHandle) {
    pegainfer_qwen35::runtime::start_engine_with_flashinfer_gdn_for_accuracy(
        Path::new(model_path),
        0,
        MAX_BATCH,
        max_prefill_tokens,
    )
    .expect("start FlashInfer Qwen3.5 scheduler")
}

fn generate(handle: &EngineHandle, prompt_tokens: Vec<u32>) -> (Vec<u32>, FinishReason) {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            trace_parent: None,
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            max_tokens: GENERATED_TOKENS,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(
                TokenEvent::Scheduled { .. }
                | TokenEvent::PromptTokens { .. }
                | TokenEvent::KvTransfer { .. },
            ) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => return (tokens, finish_reason),
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

fn prompt_tokens(model_path: &str) -> Vec<u32> {
    let tokenizer = common::load_tokenizer(model_path);
    let prompt = concat!(
        "Write a concise technical explanation of paged KV cache updates, ",
        "chunked prefill scheduling, and deterministic greedy decoding. ",
        "Mention request state ownership, recurrent state, and why resumed ",
        "prefill must append K/V instead of overwriting earlier pages. ",
        "Then summarize the behavior in three short sentences. ",
        "Repeat the explanation with different wording so the prompt is long ",
        "enough to cross several small prefill chunks."
    );
    tokenizer.encode(prompt, false).expect("encode failed")
}

#[test]
fn chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let prompt_tokens = prompt_tokens(&model_path);
    assert!(
        prompt_tokens.len() > CHUNK_BUDGET * 2,
        "test prompt must force resumed prefill: prompt_len={} chunk_budget={CHUNK_BUDGET}",
        prompt_tokens.len()
    );

    let (baseline_tokens, baseline_finish) = {
        let handle = start_engine(&model_path, BASELINE_PREFILL_BUDGET);
        generate(&handle, prompt_tokens.clone())
    };
    assert_eq!(
        baseline_finish,
        FinishReason::Length,
        "ignore_eos should force baseline generation to the requested length"
    );

    let (chunked_tokens, chunked_finish) = {
        let handle = start_engine(&model_path, CHUNK_BUDGET);
        generate(&handle, prompt_tokens)
    };
    assert_eq!(
        chunked_finish,
        FinishReason::Length,
        "ignore_eos should force chunked generation to the requested length"
    );

    assert_eq!(
        chunked_tokens, baseline_tokens,
        "chunked prefill must match effectively unchunked prefill; a mismatch suggests resumed direct-paged K/V writes used the wrong base_pos and corrupted earlier cache positions"
    );
}

#[test]
#[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and the validated Hv32 FlashInfer artifact"]
fn flashinfer_gdn_chunked_prefill_matches_unchunked_prefill() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let prompt_tokens = prompt_tokens(&model_path);
    assert!(prompt_tokens.len() > CHUNK_BUDGET * 2);

    let (baseline_tokens, baseline_finish) = {
        let (handle, evidence) = start_flashinfer_engine(&model_path, BASELINE_PREFILL_BUDGET);
        assert_eq!(evidence.snapshot().successful_launches, 0);
        let result = generate(&handle, prompt_tokens.clone());
        assert!(
            evidence.snapshot().successful_launches > 0,
            "unchunked candidate replay did not launch FlashInfer"
        );
        result
    };
    assert_eq!(baseline_finish, FinishReason::Length);

    let (chunked_tokens, chunked_finish) = {
        let (handle, evidence) = start_flashinfer_engine(&model_path, CHUNK_BUDGET);
        assert_eq!(evidence.snapshot().successful_launches, 0);
        let result = generate(&handle, prompt_tokens);
        assert!(
            evidence.snapshot().successful_launches > 0,
            "resumed candidate replay did not launch FlashInfer"
        );
        result
    };
    assert_eq!(chunked_finish, FinishReason::Length);
    assert_eq!(
        chunked_tokens, baseline_tokens,
        "FlashInfer resumed prefill must match its effectively unchunked replay"
    );
}
