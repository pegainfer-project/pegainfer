use std::collections::HashMap;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::Terminal;

use super::lane_gates_tp::assert_finite;
use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;

const TOP_K: usize = 8;

/// Four times the sliding window, so the resume runs through a truncated local family.
const RESUME_TOKENS: usize = 4096;
/// Short of the resume, so the resume prefills a suffix through the cache.
const SEED_TOKENS: usize = 4000;
const DECODE_TOKENS: usize = 6;

/// The resumed leg against the cold leg on the same prompt. Measured on the
/// pinned 12B on sm_89 at 4096 tokens: cold 0.278 nat, resumed 0.323.
const LOGPROB_RATIO: f32 = 2.0;
const LOGPROB_FLOOR: f32 = 0.10;
/// The ratio alone passes a fault both legs share, so the cold leg has its own
/// line. At this prompt's first scored position the window fixture's sdpa and
/// eager rows differ by 0.315 nat on the same token; the cold leg measured 0.278.
const COLD_LOGPROB_LINE: f32 = 0.5;

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

/// Decodes from `head`, scores the replayed prompt whole, and returns the worst
/// logprob gap and the decode's resume frontier.
fn teacher_forced_worst(harness: &mut Harness, head: &[u32]) -> (f32, usize) {
    let sampled = harness.steps.drain(
        harness
            .submit_scored(head.to_vec(), DECODE_TOKENS, Some(TOP_K), None)
            .id(),
        "sampled",
    );
    let replayed = [head.to_vec(), sampled.ids.clone()].concat();
    let replay = harness.steps.drain(
        harness
            .submit_scored(replayed.clone(), 1, None, Some(TOP_K))
            .id(),
        "replayed",
    );
    assert_eq!(replay.cached, 0, "a scored prompt never resumes");
    let echo = replay.prompt_echo.expect("the replay echoes its prompt");
    assert_echo_covers(&echo, &replayed, TOP_K);

    // A flipped argmax must stay a near tie: each pick inside the other side's top-k.
    let mut worst = 0.0_f32;
    for (i, decoded) in sampled.logprobs.iter().enumerate() {
        let decoded = decoded.as_ref().expect("sampled token scored");
        let whole = echo.logprobs[head.len() + i]
            .as_ref()
            .expect("prompt position scored");
        assert_finite(decoded, &format!("position {i} (decode)"));
        assert_finite(whole, &format!("position {i} (readback)"));
        worst = worst.max((decoded.logprob - whole.logprob).abs());
        let ours = sampled.ids[i];
        let theirs = whole.top_logprobs[0].0;
        if ours == theirs {
            continue;
        }
        let ours_top: HashMap<u32, f32> = decoded.top_logprobs.iter().copied().collect();
        let theirs_top: HashMap<u32, f32> = whole.top_logprobs.iter().copied().collect();
        assert!(
            theirs_top.contains_key(&ours),
            "position {i}: the decode sampled {ours}, which the readback's top-{TOP_K} does \
             not carry (decode top-{TOP_K} {:?}, readback top-{TOP_K} {:?})",
            decoded.top_logprobs,
            whole.top_logprobs
        );
        assert!(
            ours_top.contains_key(&theirs),
            "position {i}: the readback picks {theirs}, which the decode's top-{TOP_K} does \
             not carry (readback top-{TOP_K} {:?}, decode top-{TOP_K} {:?})",
            whole.top_logprobs,
            decoded.top_logprobs
        );
        // `ours` is the replayed token, already compared above.
        worst = worst.max((ours_top[&theirs] - theirs_top[&theirs]).abs());
    }
    (worst, sampled.cached)
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, fixtures, and --test-threads=1"]
fn prompt_scores_bypass_the_prefix_cache_and_match_teacher_forced_decode() {
    let dir = crate::testkit::model_path();
    let (_, golden) = crate::testkit::golden_bytes(&dir);
    let window_path = crate::testkit::fixture_path(
        "PEGAINFER_GEMMA4_WINDOW_GOLDEN",
        "gemma4-12b-hf-window-golden.safetensors",
    );
    let window_bytes = std::fs::read(&window_path).expect("read the window fixture");
    let window = safetensors::SafeTensors::deserialize(&window_bytes).expect("window fixture");
    let window_manifest = crate::testkit::fixture_manifest(&window_bytes, "gemma4_window_golden");
    assert_eq!(
        window_manifest["revision"], golden["revision"],
        "the window fixture was dumped from a different revision than the checkpoint under test"
    );
    let (_, prose) = crate::testkit::u32_tensor(&window, "w4096_prompt");
    assert!(
        prose.len() >= RESUME_TOKENS,
        "the window fixture's w4096 prompt is {} tokens, not {RESUME_TOKENS}",
        prose.len()
    );
    let head = prose[..RESUME_TOKENS].to_vec();
    let seed_prompt = prose[..SEED_TOKENS].to_vec();

    let mut cold = launch(&[]);
    let (cold_worst, _) = teacher_forced_worst(&mut cold, &head);
    cold.shutdown(&[]);
    assert!(
        cold_worst <= COLD_LOGPROB_LINE,
        "the cold readback and decode differ by {cold_worst} nat (line {COLD_LOGPROB_LINE})"
    );

    let mut harness = launch(&[(super::PREFIX_CACHE_ENV, "4")]);
    let seed = harness.submit(seed_prompt, 4);
    harness.steps.drain(seed.id(), "cache seed");
    let (resumed_worst, resumed_at) = teacher_forced_worst(&mut harness, &head);
    harness.shutdown(&[]);

    assert!(
        resumed_at >= SEED_TOKENS - 1,
        "the seeded prefix resumes only {resumed_at} of its {SEED_TOKENS} tokens, so this \
         run did not prefill a suffix through the cache"
    );
    let ceiling = (LOGPROB_RATIO * cold_worst).max(LOGPROB_FLOOR);
    eprintln!(
        "teacher-forced at a {resumed_at}-token frontier of {RESUME_TOKENS}: cold \
         {cold_worst:.6} nat, resumed {resumed_worst:.6} nat (ceiling {ceiling:.6})"
    );
    assert!(
        resumed_worst <= ceiling,
        "the cache-resumed comparison is {resumed_worst} nat against a cold run's \
         {cold_worst} on the same prompt (ceiling {ceiling} = {LOGPROB_RATIO}x cold, floor \
         {LOGPROB_FLOOR})"
    );
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
        history_tokens: None,
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
        super::validate_request(&ceiling_probe(prompt_len, None), raised, super::MAX_CONTEXT)
            .is_ok()
    );
    assert!(matches!(
        super::validate_request(
            &ceiling_probe(prompt_len, Some(0)),
            raised,
            super::MAX_CONTEXT
        ),
        Err(RejectReason::EchoPrefillTokens { .. })
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
