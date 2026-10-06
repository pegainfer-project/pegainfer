use std::collections::HashMap;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::Terminal;

use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;

/// How many top logprobs each side keeps. A failed pick carries both lists into
/// its message, so a disagreement reads without a re-run.
const TOP_K: usize = 8;

/// The window fixture's real-text prompt, four times the sliding window, so a
/// resume runs through a cached *local* family that the window has already
/// truncated. `serve::oracle`'s waypoint gate reads the same case.
const RESUME_TOKENS: usize = 4096;
/// Where the cache is seeded, short of the resume, so the resume has to prefill
/// a suffix through the cache rather than only decode from it.
const SEED_TOKENS: usize = 4000;
/// How many tokens the decode runs; each is scored on both sides.
const DECODE_TOKENS: usize = 6;

/// The cache-resumed comparison may not come out worse than this many times the
/// same comparison run cold. Both legs are bf16 forwards of one arithmetic in two
/// shapes — six decode steps against a whole-prompt readback — so the cold leg
/// measures what that shape difference is worth on this prompt, and the cache is
/// what this gate is about: it may not multiply it. A fixed line cannot express
/// that; measured here on the pinned 12B at 4096 tokens, the cold leg is 0.278
/// nat and the resumed one 0.323, so a line anywhere near the old 0.10 would have
/// failed the *control*. The ratio is `serve::oracle`'s own shape for a line it
/// has to derive rather than choose — `(2.0 * floor).max(1.0)`.
const LOGPROB_RATIO: f32 = 2.0;
/// The same line's floor, for a prompt whose cold legs happen to agree exactly:
/// without it the ratio would tighten to nothing.
const LOGPROB_FLOOR: f32 = 0.10;

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

/// Decode [`DECODE_TOKENS`] from `head`, read the same positions back through one
/// whole-prompt scored pass, and hold the two against each other: the worst
/// logprob gap on a token both sides kept, with every disagreement required to
/// be a near-tie. Returns that worst and the resume frontier the decode reported,
/// so a caller can tell whether a cache was involved.
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

    // Every position conditions on the same tokens on both sides, so a pick that
    // moved is worth failing on — but only past the near-tie rule that
    // `lane_gates_tp`'s `distribution_gap` applies to the same situation: two
    // independently sampling shapes of one arithmetic disagree about the argmax
    // exactly where its two candidates sit inside reduction-order noise, and
    // refusing that fails a correct build (it does, on this prompt, at position
    // 4, where 236761 and 236770 trade places across a ~0.25 nat gap). A pick
    // neither side kept is a real failure, and the flipped picks' own gaps are
    // folded into the bound so a flip cannot hide one.
    let mut worst = 0.0_f32;
    for (i, decoded) in sampled.logprobs.iter().enumerate() {
        let decoded = decoded.as_ref().expect("sampled token scored");
        let whole = echo.logprobs[head.len() + i]
            .as_ref()
            .expect("prompt position scored");
        // A NaN row is invisible to the comparisons below: `f32::max` steps over
        // a NaN without moving the bound, so a NaN logsumexp — every logprob on
        // the row NaN while the ids stay right — would pass this gate.
        for row in [decoded, whole] {
            assert!(
                row.logprob.is_finite()
                    && row.top_logprobs.iter().all(|(_, value)| value.is_finite()),
                "position {i}: a non-finite logprob reached the comparison"
            );
        }
        let ours = sampled.ids[i];
        let theirs = whole.top_logprobs[0].0;
        if ours == theirs {
            worst = worst.max((decoded.logprob - whole.logprob).abs());
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
        for token in [ours, theirs] {
            if let (Some(a), Some(b)) = (ours_top.get(&token), theirs_top.get(&token)) {
                worst = worst.max((a - b).abs());
            }
        }
    }
    (worst, sampled.cached)
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, fixtures, and --test-threads=1"]
fn prompt_scores_bypass_the_prefix_cache_and_match_teacher_forced_decode() {
    // Real text, and past the sliding window, both on purpose.
    //
    // Real text because this gate compares two runs that sample independently.
    // Synthetic id arithmetic drives the tower into a degenerate distribution
    // whose top two candidates sit inside reduction-order noise — `testkit.rs`
    // records exactly that about it — and with such a prompt the two sides picked
    // different tokens from the second position on, so the comparison below was
    // subtracting two different tokens' logprobs, by up to 0.76 nat. This prompt
    // is tokenized text from `dump_gemma4_window_golden.py`'s corpus.
    //
    // Past the window because four times the window is what makes the resume run
    // through a cached local family the window has already truncated — the case
    // `kv-cache.md`'s frozen checkpoint query is about, and one a prompt that
    // stops at the window cannot reach.
    let dir = crate::testkit::model_path();
    let (_, golden) = crate::testkit::golden_bytes(&dir);
    let window_path = crate::testkit::fixture_path(
        "PEGAINFER_GEMMA4_WINDOW_GOLDEN",
        "gemma4-12b-hf-window-golden.safetensors",
    );
    let window_bytes = std::fs::read(&window_path).expect("read the window fixture");
    let window = safetensors::SafeTensors::deserialize(&window_bytes).expect("window fixture");
    // The window fixture is a separate artifact from the checkpoint, so hold it
    // to the revision the golden was already validated against — the way
    // `serve::oracle`'s waypoint gate does, and not by its file name.
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

    // The control: the same comparison with no cache anywhere near it. It is what
    // makes the assertion below a property rather than a constant.
    let mut cold = launch(&[]);
    let (cold_worst, _) = teacher_forced_worst(&mut cold, &head);
    cold.shutdown(&[]);

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
