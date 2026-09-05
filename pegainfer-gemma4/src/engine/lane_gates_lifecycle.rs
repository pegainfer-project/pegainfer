use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::Terminal;

use super::lane_test_env::scoped_engine_env;
use super::lane_tests::assert_warm_result;
use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;
use super::lane_tests::warm_prompt;

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_raise_reaches_the_frontend() {
    let mut harness = launch(&[
        (super::MAX_CONTEXT_ENV, "32768"),
        (super::MIX_CHUNK_TOKENS_ENV, "2048"),
        (super::DECODE_SLOTS_ENV, "2"),
    ]);
    assert_eq!(harness.servable_len, Some(32768));
    let request = harness.submit(ids(40, 5), 4);
    let served = harness.steps.drain(request.id(), "raised ceiling");
    assert_eq!((served.tokens, served.finish), (4, FinishReason::Length));
    harness.shutdown(&[]);
}

#[test]
#[ignore = "requires the pinned 12B checkpoint and --test-threads=1"]
fn the_raise_refuses_without_its_prerequisites() {
    let dir = crate::testkit::model_path();
    let load = |overrides: &[(&str, &str)]| {
        let policy = super::generation_policy(&dir).expect("policy");
        let _env = scoped_engine_env(overrides);
        super::EngineState::load(&dir, 0, policy, 0x5EED, true)
    };
    let error = load(&[(super::MAX_CONTEXT_ENV, "32768")])
        .err()
        .expect("a raise without chunking must refuse");
    assert!(
        format!("{error:#}").contains("needs PEGAINFER_MIX_CHUNK_TOKENS"),
        "unexpected refusal: {error:#}"
    );
    let error = load(&[
        (super::MAX_CONTEXT_ENV, "32768"),
        (super::MIX_CHUNK_TOKENS_ENV, "2048"),
        (super::ASYNC_PREFILL_ENV, "green:35"),
    ])
    .err()
    .expect("the lane over the default ceiling must refuse");
    assert!(format!("{error:#}").contains("unsupported over"));
    let error = load(&[
        (super::ADMIT_COALESCE_ENV, "300"),
        (super::ASYNC_PREFILL_ENV, "green:35"),
    ])
    .err()
    .expect("the coalesce door and async lane must refuse");
    assert!(format!("{error:#}").contains("the door could only delay it"));
}

fn lane_lifecycle_script(mode: &str) {
    let mut harness = launch(&[
        (super::ASYNC_PREFILL_ENV, mode),
        (super::PREFIX_CACHE_ENV, "4"),
    ]);
    let streamer = pin_live_stream(&mut harness);

    let long_prompt = ids(1500, 7);
    let lane = harness.submit(long_prompt.clone(), 4);
    let queued = harness.submit(ids(60, 3), 4);
    let lane_done = harness.steps.drain(lane.id(), "lane prefill");
    assert_eq!(
        (lane_done.tokens, lane_done.finish),
        (4, FinishReason::Length)
    );
    let queued_done = harness.steps.drain(queued.id(), "queued behind lane");
    assert_eq!(
        (queued_done.tokens, queued_done.finish),
        (4, FinishReason::Length)
    );

    let warm = harness.submit(warm_prompt(&long_prompt), 4);
    assert_warm_result(&mut harness, warm.id(), 1500, "warm suffix on lane");

    let cancelled = harness.submit(ids(1500, 23), 8);
    harness.steps.wait_scheduled(cancelled.id());
    cancelled.abort();

    let mut invalid_prompt = ids(200, 31);
    invalid_prompt[100] = 300_000;
    let invalid = harness.submit(invalid_prompt, 4);
    match harness.steps.terminal(invalid.id()) {
        Terminal::Failed { message, .. } => assert!(
            message.contains("prefill"),
            "a lane launch failure must name prefill: {message}"
        ),
        other => panic!("a lane launch failure must fail that request, got {other:?}"),
    }
    // The failure was the request's alone: the lane drained and the engine
    // still serves.
    let after = harness.submit(ids(40, 37), 4);
    let after_done = harness
        .steps
        .drain(after.id(), "served after a lane launch failure");
    assert_eq!(
        (after_done.tokens, after_done.finish),
        (4, FinishReason::Length)
    );

    harness.shutdown(&[&streamer]);
    let cancelled_terminals = harness.steps.terminals_after_close(cancelled.id());
    assert!(
        cancelled_terminals.is_empty(),
        "frontend abort must retire silently: {cancelled_terminals:?}"
    );
}

fn gather_lifecycle_script() {
    let mut harness = launch(&[(super::PREFIX_CACHE_ENV, "4")]);
    let streamer = pin_live_stream(&mut harness);

    let dead_a = harness.submit(ids(50, 1), 4);
    let dead_b = harness.submit(ids(50, 2), 4);
    dead_a.abort();
    dead_b.abort();
    let invalid = harness.submit(Vec::new(), 4);
    let valid = harness.submit(ids(60, 3), 4);
    match harness.steps.terminal(invalid.id()) {
        Terminal::Rejected { reason, .. } => assert!(
            reason.to_string().contains("empty prompts"),
            "unexpected refusal: {reason}"
        ),
        other => panic!("an empty prompt must be rejected, got {other:?}"),
    }
    let valid_done = harness.steps.drain(valid.id(), "valid after aborts");
    assert_eq!(
        (valid_done.tokens, valid_done.finish),
        (4, FinishReason::Length)
    );

    let long_prompt = ids(1600, 7);
    let long = harness.submit(long_prompt.clone(), 4);
    let long_done = harness.steps.drain(long.id(), "long prefill");
    assert_eq!(
        (long_done.tokens, long_done.finish),
        (4, FinishReason::Length)
    );
    let head = harness.submit(ids(60, 13), 4);
    let warm = harness.submit(warm_prompt(&long_prompt), 4);
    let head_done = harness.steps.drain(head.id(), "gather head");
    assert_eq!(
        (head_done.tokens, head_done.finish),
        (4, FinishReason::Length)
    );
    assert_warm_result(&mut harness, warm.id(), 1600, "warm gathered suffix");

    harness.shutdown(&[&streamer, &dead_a, &dead_b]);
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn the_shared_lane_lifecycle_completes() {
    lane_lifecycle_script("shared");
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn the_green_lane_lifecycle_completes() {
    lane_lifecycle_script("green:35");
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_gathered_lifecycle_completes() {
    gather_lifecycle_script();
}

#[test]
fn pool_pages_follow_the_knobs() {
    assert_eq!(
        super::pool_pages(512, 65, 512, 16, 0, 256),
        Some((1488, 8193))
    );
    assert_eq!(
        super::pool_pages(196, 65, 2048, 2, 0, 1024),
        Some((262, 4097))
    );
    assert_eq!(super::pool_pages(usize::MAX, 65, 512, 16, 0, 256), None);
}
