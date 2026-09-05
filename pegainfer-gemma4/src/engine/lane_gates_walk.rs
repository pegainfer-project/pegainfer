use std::time::Duration;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::RequestControl;

use super::lane_test_env::scoped_engine_env;
use super::lane_tests::Harness;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;
use super::lane_tests::wait_until;

fn load_chunk_state(chunk: &str) -> super::EngineState {
    let dir = crate::testkit::model_path();
    let policy = super::generation_policy(&dir).expect("policy");
    let _env = scoped_engine_env(&[(super::MIX_CHUNK_TOKENS_ENV, chunk)]);
    super::EngineState::load(&dir, 0, policy, 0x5EED, true).expect("engine state")
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_gathered_transient_leaves_headroom() {
    let dir = crate::testkit::model_path();
    let state = load_chunk_state("2048");
    assert_eq!(state.mix_chunk, Some(2048));
    assert_eq!(state.max_context, 8192);
    assert_eq!(state.slots, 16);
    let window = crate::config::Gemma4Config::from_file(&dir)
        .expect("config")
        .sliding_window;
    let window_pages = window.div_ceil(crate::kv::PAGE_SIZE) + 1;
    let provisioned = window_pages
        + 2048usize.div_ceil(crate::kv::PAGE_SIZE)
        + (super::MIX_MAX_PROMPTS - 1)
        + (super::MAX_CONCURRENCY - 1) * window_pages;
    assert_eq!(
        state.serve.local_pool.available_pages(),
        provisioned,
        "the reduced pool is sized by one shared walk segment"
    );

    let mut harness = Harness::from_state(state);
    let prompts = crate::testkit::generate_fixture_prompts();
    let stream_prompt: Vec<u32> = prompts[0].iter().cycle().copied().take(1500).collect();
    let long_prompt: Vec<u32> = prompts[0].iter().cycle().copied().take(5900).collect();
    let streams: Vec<RequestControl> = (0..12)
        .map(|_| harness.submit(stream_prompt.clone(), 60))
        .collect();
    assert!(
        wait_until(Duration::from_secs(30), || harness
            .metrics()
            .num_running_reqs
            >= 12),
        "every stream must hold a decode slot"
    );
    let long: Vec<RequestControl> = (0..3)
        .map(|_| harness.submit(long_prompt.clone(), 2))
        .collect();
    let long_ids: Vec<_> = long.iter().map(RequestControl::id).collect();
    harness.steps.wait_scheduled_together(&long_ids);
    for (index, request) in long.iter().enumerate() {
        assert_eq!(
            harness
                .steps
                .drain(request.id(), &format!("long prompt {index}"))
                .tokens,
            2
        );
    }
    for (index, request) in streams.iter().enumerate() {
        assert_eq!(
            harness
                .steps
                .drain(request.id(), &format!("stream {index}"))
                .tokens,
            60
        );
    }
    harness.shutdown(&[]);
}

fn serial_sequences(prompts: &[Vec<u32>], budgets: &[usize]) -> Vec<Vec<u32>> {
    let mut harness = launch(&[(super::MIX_CHUNK_TOKENS_ENV, "64")]);
    let sequences = prompts
        .iter()
        .zip(budgets)
        .map(|(prompt, &budget)| {
            let request = harness.submit(prompt.clone(), budget);
            harness.steps.drain(request.id(), "serial episode").ids
        })
        .collect();
    harness.shutdown(&[]);
    sequences
}

/// Differential against the same engine: the gathered walk must emit the
/// tokens the serial walk emits. It catches a batching-dependent divergence
/// and nothing about whether those tokens are right; that is the oracle
/// group's `greedy_matches_hf_generate`.
#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_gathered_walk_does_not_depend_on_its_batching() {
    let prompts = crate::testkit::generate_fixture_prompts();
    let budgets = [24usize, 17, 21];
    let serial = serial_sequences(&prompts, &budgets);
    for (index, sequence) in serial.iter().enumerate() {
        assert_eq!(sequence.len(), budgets[index]);
    }

    let mut gathered = launch(&[(super::MIX_CHUNK_TOKENS_ENV, "64")]);
    let requests: Vec<RequestControl> = prompts
        .iter()
        .zip(budgets)
        .map(|(prompt, budget)| gathered.submit(prompt.clone(), budget))
        .collect();
    let request_ids: Vec<_> = requests.iter().map(RequestControl::id).collect();
    gathered.steps.wait_scheduled_together(&request_ids);
    for (index, request) in requests.iter().enumerate() {
        let produced = gathered.steps.drain(request.id(), "gathered walk").ids;
        assert_eq!(
            produced, serial[index],
            "case {index}: gathered walk diverged from serial"
        );
    }
    gathered.shutdown(&[]);

    // The roster drains mid-walk: a rider holds the decode batch while two
    // eight-chunk walkers are scheduled, then leaves. Its row changes the
    // composition of the first mixed step and the numerics are not
    // batch-invariant, so a near-tie prompt need not reproduce its serial
    // tokens here (it does not); the gate holds the drain path's structure.
    let tail_prompts: Vec<Vec<u32>> = prompts[..2]
        .iter()
        .map(|prompt| prompt.iter().cycle().copied().take(512).collect())
        .collect();
    let tail_budgets = [7usize, 9];
    let mut drained = launch(&[(super::MIX_CHUNK_TOKENS_ENV, "64")]);
    let tail_rider = pin_live_stream(&mut drained);
    let tail_a = drained.submit(tail_prompts[0].clone(), tail_budgets[0]);
    let tail_b = drained.submit(tail_prompts[1].clone(), tail_budgets[1]);
    drained
        .steps
        .wait_scheduled_together(&[tail_a.id(), tail_b.id()]);
    tail_rider.abort();
    for (request, budget, label) in [
        (&tail_a, tail_budgets[0], "drained-tail a"),
        (&tail_b, tail_budgets[1], "drained-tail b"),
    ] {
        let done = drained.steps.drain(request.id(), label);
        assert_eq!(
            (done.tokens, done.finish),
            (budget, FinishReason::Length),
            "{label} walks to its budget after the roster drains"
        );
    }
    drained.shutdown(&[]);
    assert!(
        drained
            .steps
            .terminals_after_close(tail_rider.id())
            .is_empty(),
        "the rider abort that drains the roster retires silently"
    );

    let mut cancelled = launch(&[(super::MIX_CHUNK_TOKENS_ENV, "64")]);
    let rider = cancelled.submit(prompts[2].clone(), budgets[2]);
    cancelled.steps.wait_tokens(rider.id(), 1);
    let aborted = cancelled.submit(prompts[0].clone(), budgets[0]);
    aborted.abort();
    let survivor = cancelled.submit(prompts[1].clone(), budgets[1]);
    let survivor_ids = cancelled.steps.drain(survivor.id(), "surviving walker").ids;
    let rider_ids = cancelled.steps.drain(rider.id(), "rider").ids;
    assert_eq!(survivor_ids, serial[1]);
    assert_eq!(rider_ids.len(), budgets[2]);
    cancelled.shutdown(&[]);
    assert!(
        cancelled
            .steps
            .terminals_after_close(aborted.id())
            .is_empty(),
        "aborted walker must retire without a terminal"
    );
}
