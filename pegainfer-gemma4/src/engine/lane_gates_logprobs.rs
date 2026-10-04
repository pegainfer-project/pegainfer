use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::Terminal;

use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;
use super::lane_tests::warm_prompt;

/// Decode steps one row at a time and the whole-prompt pass many; both are
/// bf16 forwards, so a token's two scores differ by rounding, not by more.
const LOGPROB_TOLERANCE: f32 = 0.10;

fn assert_echo_covers(echo: &PromptEcho, prompt: &[u32], top_k: usize) {
    assert_eq!(echo.ids, prompt, "the echo names the prompt");
    assert_eq!(echo.logprobs.len(), prompt.len(), "one score a position");
    assert!(
        echo.logprobs[0].is_none(),
        "nothing predicts the first token"
    );
    for (position, score) in echo.logprobs.iter().enumerate().skip(1) {
        let score = score
            .as_ref()
            .unwrap_or_else(|| panic!("position {position} is unscored"));
        assert_eq!(score.top_logprobs.len(), top_k, "position {position} top-k");
    }
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn prompt_scores_bypass_the_prefix_cache_and_match_teacher_forced_decode() {
    let mut harness = launch(&[(super::PREFIX_CACHE_ENV, "4")]);
    let long_prompt = ids(1500, 7);
    let seed = harness.submit(long_prompt.clone(), 4);
    harness.steps.drain(seed.id(), "cache seed");

    let head = warm_prompt(&long_prompt);
    let sampled = harness.submit_scored(head.clone(), 6, Some(0), None);
    let sampled = harness.steps.drain(sampled.id(), "sampled");
    assert!(sampled.cached > 0, "the seeded prefix resumes");

    let replayed = [head.clone(), sampled.ids.clone()].concat();
    let replay = harness.submit_scored(replayed.clone(), 1, None, Some(0));
    let replay = harness.steps.drain(replay.id(), "replayed");
    assert_eq!(replay.cached, 0, "a scored prompt never resumes");
    let echo = replay.prompt_echo.expect("the replay echoes its prompt");
    assert_echo_covers(&echo, &replayed, 0);

    let mut max_delta = 0.0_f32;
    for (i, decoded) in sampled.logprobs.iter().enumerate() {
        let decoded = decoded.as_ref().expect("sampled token scored").logprob;
        let whole = echo.logprobs[head.len() + i]
            .as_ref()
            .expect("prompt position scored")
            .logprob;
        let delta = (decoded - whole).abs();
        max_delta = max_delta.max(delta);
        assert!(
            delta <= LOGPROB_TOLERANCE,
            "position {i}: decode {decoded}, whole prompt {whole}"
        );
    }
    eprintln!("teacher-forced max_delta={max_delta:.6} nat");
    harness.shutdown(&[]);
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn a_scored_prompt_beside_a_live_batch_is_prefilled_whole() {
    // The chunk knob would walk a solo prompt in segments; a scored one
    // still takes one whole-prompt pass, beside the live stream.
    let mut harness = launch(&[(super::MIX_CHUNK_TOKENS_ENV, "256")]);
    let streamer = pin_live_stream(&mut harness);
    let prompt = ids(700, 5);
    let scored = harness.submit_scored(prompt.clone(), 4, None, Some(5));
    let done = harness
        .steps
        .drain(scored.id(), "scored beside a live batch");
    assert_eq!((done.tokens, done.finish), (4, FinishReason::Length));
    assert_echo_covers(&done.prompt_echo.expect("echo"), &prompt, 5);
    let seen = harness.steps.buffered_tokens(streamer.id());
    harness.steps.wait_tokens(streamer.id(), seen + 2);
    harness.shutdown(&[&streamer]);
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn a_low_slot_chunked_pool_scores_up_to_what_it_holds() {
    // Two slots and a 256-row segment budget the local pool for window plus
    // segment; a scored prompt past that is still served whole, and one past
    // the pool is refused at the pool's size.
    let mut harness = launch(&[
        (super::MAX_CONTEXT_ENV, "32768"),
        (super::MIX_CHUNK_TOKENS_ENV, "256"),
        (super::DECODE_SLOTS_ENV, "2"),
    ]);
    let prompt = ids(2000, 3);
    let scored = harness.submit_scored(prompt.clone(), 4, None, Some(0));
    let done = harness
        .steps
        .drain(scored.id(), "scored past window plus segment");
    assert_eq!((done.tokens, done.finish), (4, FinishReason::Length));
    assert_echo_covers(&done.prompt_echo.expect("echo"), &prompt, 0);

    let over = harness.submit_scored(ids(4096, 4), 4, None, Some(0));
    match harness.steps.terminal(over.id()) {
        Terminal::Rejected {
            reason: RejectReason::EchoPrefillTokens { limit, .. },
            ..
        } => assert!(
            (2000..4096).contains(&limit),
            "the pool-sized limit is {limit}"
        ),
        other => panic!("a scored prompt past the pool must be refused, got {other:?}"),
    }
    harness.shutdown(&[]);
}

/// A probe whose prompt is exactly `prompt_len` tokens, so a ceiling can be
/// tested against a length known without reading the prompt back.
fn ceiling_probe(prompt_len: usize, prompt_logprobs: Option<usize>) -> Request {
    Request {
        prompt_tokens: ids(prompt_len, 9),
        params: pegainfer_frontend::sampler::SamplingParams::default(),
        stop_policy: pegainfer_frontend::engine::StopPolicy::default(),
        max_tokens: 4,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: None,
        prompt_logprobs,
        trace_parent: None,
        client_label: None,
    }
}

#[test]
fn only_a_scored_prompt_is_bound_to_the_whole_prompt_ceiling() {
    let raised = 4 * super::MAX_CONTEXT;
    let prompt_len = super::MAX_CONTEXT + 1;
    assert!(
        super::validate_request(
            &ceiling_probe(prompt_len, None),
            raised,
            super::MAX_CONTEXT,
            None
        )
        .is_ok()
    );
    assert!(matches!(
        super::validate_request(
            &ceiling_probe(prompt_len, Some(0)),
            raised,
            super::MAX_CONTEXT,
            None
        ),
        Err(RejectReason::EchoPrefillTokens { .. })
    ));
}

#[test]
fn the_tp_prompt_ceiling_refuses_one_token_past_it() {
    let raised = 4 * super::MAX_CONTEXT;
    let prompt_len = super::MAX_CONTEXT + 1;
    // The ceiling is the boundary: a prompt equal to it passes, one token past
    // it is refused.
    assert!(
        super::validate_request(
            &ceiling_probe(prompt_len, None),
            raised,
            super::MAX_CONTEXT,
            Some(prompt_len)
        )
        .is_ok()
    );
    assert!(matches!(
        super::validate_request(
            &ceiling_probe(prompt_len, None),
            raised,
            super::MAX_CONTEXT,
            Some(prompt_len - 1)
        ),
        Err(RejectReason::Unsupported { .. })
    ));
}

/// A token id outside the embedding fails in `prepare_single`'s
/// `validate_tokens`, before the tower allocates anything or — under tensor
/// parallelism — before rank 0 issues a single collective. So a prompt carrying
/// one is a deterministic prefill fault that needs no injection hook, and the
/// property under test is its **scope**: the driver contract makes `Err` from
/// `Scheduler::step` mean "the engine is beyond use", so a request-local fault
/// that escaped as `Err` would close the step stream and write off every open
/// account. Both prefill paths have to answer it by failing that one request.
#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn a_failed_prefill_costs_that_request_not_the_engine() {
    let mut harness = launch(&[]);
    for (name, salt, prompt_logprobs) in [("scored", 1u32, Some(8)), ("plain", 2, None)] {
        // `u32::MAX` is outside any vocabulary this line ships.
        let bad = harness.submit_scored(vec![9, u32::MAX, 11 + salt], 4, None, prompt_logprobs);
        match harness.steps.terminal(bad.id()) {
            Terminal::Failed { message, .. } => assert!(
                message.contains("prefill failed"),
                "{name}: the request should fail as a prefill failure, got: {message}"
            ),
            other => panic!("{name}: an unusable prompt must fail that request, not {other:?}"),
        }
        // The engine is still serving, and the failed request's pages came back
        // with its KV: a good request is admitted and completes afterwards.
        let good = harness.submit_scored(ids(12, 7 + salt), 4, None, prompt_logprobs);
        let drained = harness.steps.drain(good.id(), name);
        assert_eq!(
            drained.tokens, 4,
            "{name}: the request after a failed prefill still decodes to its budget"
        );
        assert_eq!(
            drained.finish,
            FinishReason::Length,
            "{name}: and finishes on its budget"
        );
    }
    harness.shutdown(&[]);
}
