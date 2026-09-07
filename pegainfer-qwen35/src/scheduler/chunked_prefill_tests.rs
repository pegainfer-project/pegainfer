//! Qwen3.5 scheduler-level chunked prefill regression tests.
//!
//! These tests exercise resumed prefill (`base_pos > 0`) through the real
//! scheduler path. A small `max_prefill_tokens` budget forces one request's
//! prompt to be prefilling across multiple scheduler steps; the same prompt is
//! also run with an effectively unchunked budget and the generated greedy token
//! ids must match.

use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::sampler::SamplingParams;

use crate::test_fixture as common;
use crate::test_fixture::GdnAcceptance;

const CHUNK_BUDGET: usize = 16;
const BASELINE_PREFILL_BUDGET: usize = 1 << 20;
const MAX_BATCH: usize = 2;
const GENERATED_TOKENS: usize = 8;

fn start_engine(
    acceptance: &GdnAcceptance,
    model_path: &str,
    max_prefill_tokens: usize,
) -> EngineHandle {
    acceptance
        .launch_engine(
            model_path,
            MAX_BATCH,
            max_prefill_tokens,
            pegainfer_qwen35::Qwen35DecodeOverlap::Off,
        )
        .expect("failed to start Qwen3.5 engine")
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

#[test]
fn chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv() {
    run_chunked_prefill(&GdnAcceptance::Triton);
}

#[test]
#[ignore = "requires SM120, a validated candidate identity, and Qwen3.5 weights"]
fn candidate_chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv() {
    run_chunked_prefill(&GdnAcceptance::candidate());
}

fn run_chunked_prefill(acceptance: &GdnAcceptance) {
    let Some(model_path) =
        acceptance.model_path("chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv")
    else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let prompt = concat!(
        "Write a concise technical explanation of paged KV cache updates, ",
        "chunked prefill scheduling, and deterministic greedy decoding. ",
        "Mention request state ownership, recurrent state, and why resumed ",
        "prefill must append K/V instead of overwriting earlier pages. ",
        "Then summarize the behavior in three short sentences. ",
        "Repeat the explanation with different wording so the prompt is long ",
        "enough to cross several small prefill chunks."
    );
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    assert!(
        prompt_tokens.len() > CHUNK_BUDGET * 2,
        "test prompt must force resumed prefill: prompt_len={} chunk_budget={CHUNK_BUDGET}",
        prompt_tokens.len()
    );

    let (baseline_tokens, baseline_finish) = {
        let handle = start_engine(acceptance, &model_path, BASELINE_PREFILL_BUDGET);
        generate(&handle, prompt_tokens.clone())
    };
    assert_eq!(
        baseline_finish,
        FinishReason::Length,
        "ignore_eos should force baseline generation to the requested length"
    );
    assert_eq!(baseline_tokens.len(), GENERATED_TOKENS);

    let (chunked_tokens, chunked_finish) = {
        let handle = start_engine(acceptance, &model_path, CHUNK_BUDGET);
        generate(&handle, prompt_tokens)
    };
    assert_eq!(
        chunked_finish,
        FinishReason::Length,
        "ignore_eos should force chunked generation to the requested length"
    );
    assert_eq!(chunked_tokens.len(), GENERATED_TOKENS);

    assert_eq!(
        chunked_tokens, baseline_tokens,
        "chunked prefill must match effectively unchunked prefill; a mismatch suggests resumed direct-paged K/V writes used the wrong base_pos and corrupted earlier cache positions"
    );
}
