//! Qwen3.5 scheduler-level chunked prefill regression tests.
//!
//! These tests exercise resumed prefill (`base_pos > 0`) through the real
//! scheduler path. A small `max_prefill_tokens` budget forces one request's
//! prompt to be prefilling across multiple scheduler steps; the same prompt is
//! also run with an effectively unchunked budget and the generated greedy token
//! ids must match.

use std::path::Path;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;

mod common;

use common::harness::EngineHarness;

const CHUNK_BUDGET: usize = 16;
const BASELINE_PREFILL_BUDGET: usize = 1 << 20;
const MAX_BATCH: usize = 2;
const GENERATED_TOKENS: usize = 8;

fn start_engine(model_path: &str, max_prefill_tokens: usize) -> EngineHarness {
    EngineHarness::new(
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
        .expect("failed to start Qwen3.5 engine"),
    )
}

fn generate(harness: &EngineHarness, prompt_tokens: Vec<u32>) -> (Vec<u32>, FinishReason) {
    let outcome = harness
        .submit(common::harness::request(
            prompt_tokens,
            SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            GENERATED_TOKENS,
        ))
        .expect_finished();
    let Terminal::Finished { reason, .. } = outcome.terminal else {
        unreachable!("expect_finished returned a non-Finished terminal");
    };
    (outcome.tokens, reason)
}

#[test]
fn chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv() {
    let Some(model_path) = common::model_path_or_skip(
        "chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv",
    ) else {
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
        let harness = start_engine(&model_path, BASELINE_PREFILL_BUDGET);
        generate(&harness, prompt_tokens.clone())
    };
    assert_eq!(
        baseline_finish,
        FinishReason::Length,
        "ignore_eos should force baseline generation to the requested length"
    );

    let (chunked_tokens, chunked_finish) = {
        let harness = start_engine(&model_path, CHUNK_BUDGET);
        generate(&harness, prompt_tokens)
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
