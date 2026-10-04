//! Qwen3.5 scheduler-level chunked prefill regression tests.
//!
//! These tests exercise resumed prefill (`base_pos > 0`) through the real
//! scheduler path. A small `max_prefill_tokens` budget forces one request's
//! prompt to be prefilling across multiple scheduler steps; the same prompt is
//! also run with an effectively unchunked budget and the generated greedy token
//! ids must match.

use common::EngineHarness;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::Terminal;
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
) -> EngineHarness {
    acceptance
        .launch_engine(
            model_path,
            MAX_BATCH,
            max_prefill_tokens,
            crate::Qwen35SchedulerPolicy::Off,
            crate::Qwen35DecodeOverlap::Off,
        )
        .map(EngineHarness::new)
        .expect("failed to start Qwen3.5 engine")
}

fn generate(handle: &mut EngineHarness, prompt_tokens: Vec<u32>) -> (Vec<u32>, FinishReason) {
    let control = handle.submit(common::request(
        prompt_tokens,
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        GENERATED_TOKENS,
    ));
    let mut tokens = Vec::new();
    loop {
        let update = handle.next(control.id());
        tokens.extend(update.tokens);
        if let Some(terminal) = update.terminal {
            match terminal {
                Terminal::Finished {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(completion_tokens, GENERATED_TOKENS);
                    return (tokens, reason);
                }
                terminal => panic!("generation did not finish: {terminal:?}"),
            }
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
    run_chunked_prefill(&GdnAcceptance::candidate().expect("candidate prerequisites"));
}

fn run_chunked_prefill(acceptance: &GdnAcceptance) {
    let Some(model_path) = acceptance
        .model_path("chunked_prefill_matches_unchunked_prefill_for_resumed_paged_kv")
        .expect("model prerequisite")
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
        let mut handle = start_engine(acceptance, &model_path, BASELINE_PREFILL_BUDGET);
        generate(&mut handle, prompt_tokens.clone())
    };
    assert_eq!(
        baseline_finish,
        FinishReason::Length,
        "ignore_eos should force baseline generation to the requested length"
    );
    assert_eq!(baseline_tokens.len(), GENERATED_TOKENS);

    let (chunked_tokens, chunked_finish) = {
        let mut handle = start_engine(acceptance, &model_path, CHUNK_BUDGET);
        generate(&mut handle, prompt_tokens)
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
