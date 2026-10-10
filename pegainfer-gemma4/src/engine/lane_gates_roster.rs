use std::time::Duration;

use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::StopCause;
use pegainfer_frontend::engine::StopPolicy;

use super::lane_tests::Drained;
use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;
use super::lane_tests::wait_until;

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, fixtures, and --test-threads=1"]
fn stop_policy_drains_decode_before_reusing_its_slot() {
    let prompt = crate::testkit::generate_fixture_prompts().remove(0);
    let mut harness = launch(&[(super::DECODE_SLOTS_ENV, "1")]);
    assert!(wait_until(Duration::from_secs(10), || {
        harness.metrics().kv_total_blocks > 0
    }));
    // Both pools retain their graph-padding page even when no request is live.
    let idle = harness.metrics();
    for logprobs in [Some(8), None] {
        let reference = harness.submit_scored(prompt.clone(), 16, logprobs, None);
        let reference = harness.steps.drain(reference.id(), "without a stop");
        assert_eq!(
            (reference.tokens, reference.finish),
            (16, FinishReason::Length)
        );
        assert_eq!(reference.stop_cause, None);
        let position = (1..reference.ids.len() - super::DECODE_PIPELINE_DEPTH)
            .find(|&i| !reference.ids[..i].contains(&reference.ids[i]))
            .expect("the prompt generates a later distinct token with pipeline headroom");
        let trigger = reference.ids[position];
        // The unscored leg has at least two tokens left when it stops, so
        // it settles a staged readback with a successor already in flight.
        let stopped = harness.submit_with_policy(
            prompt.clone(),
            16,
            logprobs,
            None,
            StopPolicy::new(EosPolicy::Ignore, vec![trigger]),
        );
        let next = harness.submit(prompt.clone(), 6);
        let stopped = harness.steps.drain(stopped.id(), "explicit decode stop");
        assert_eq!(stopped.finish, FinishReason::Stop);
        assert_eq!(stopped.stop_cause, Some(StopCause::Token(trigger)));
        assert_eq!(stopped.ids, reference.ids[..=position]);
        assert_eq!(stopped.logprobs, reference.logprobs[..=position]);
        if logprobs.is_some() {
            super::lane_gates_tp::assert_finite(
                stopped.logprobs[position]
                    .as_ref()
                    .expect("the trigger is scored"),
                "decode stop",
            );
        }
        let next = harness.steps.drain(next.id(), "slot reused after a stop");
        assert_eq!((next.tokens, next.finish), (6, FinishReason::Length));
        assert_eq!(next.stop_cause, None);
        assert!(
            wait_until(Duration::from_secs(10), || harness.metrics() == idle),
            "the stopped row and its staged successor release their pages: idle {idle:?}, now {:?}",
            harness.metrics()
        );
    }
    harness.shutdown(&[]);
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_coalesce_door_releases_one_admission_burst() {
    let mut harness = launch(&[
        (super::ADMIT_COALESCE_ENV, "2000"),
        (super::DECODE_SLOTS_ENV, "4"),
    ]);
    let incumbent = pin_live_stream(&mut harness);
    let incumbent_before = harness.steps.buffered_tokens(incumbent.id());
    let second = harness.submit(ids(40, 2), 4);
    let third = harness.submit(ids(40, 3), 4);
    assert!(
        wait_until(Duration::from_millis(500), || {
            harness.steps.buffered_tokens(incumbent.id()) > incumbent_before
        }),
        "the incumbent advances while the admission door is closed"
    );
    assert!(
        !harness.steps.saw_scheduled(second.id()),
        "two arrivals stay behind the unexpired door"
    );
    assert!(
        !harness.steps.saw_scheduled(third.id()),
        "the cohort is still incomplete"
    );

    let fourth = harness.submit(ids(40, 4), 4);
    harness
        .steps
        .wait_scheduled_together(&[second.id(), third.id(), fourth.id()]);
    harness.steps.drain(second.id(), "second");
    harness.steps.drain(third.id(), "third");
    harness.steps.drain(fourth.id(), "fourth");
    harness.shutdown(&[&incumbent]);
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_raised_ceiling_and_slots_hold_at_the_roster_edge() {
    let mut harness = launch(&[
        (super::MAX_CONTEXT_ENV, "32768"),
        (super::MIX_CHUNK_TOKENS_ENV, "2048"),
        (super::DECODE_SLOTS_ENV, "2"),
    ]);
    let first = harness.submit(ids(5000, 1), 16);
    let second = harness.submit(ids(5000, 2), 16);
    let third = harness.submit(ids(5000, 3), 8);
    assert!(
        wait_until(Duration::from_secs(20), || {
            let metrics = harness.metrics();
            metrics.num_running_reqs == 2 && metrics.num_waiting_reqs == 1
        }),
        "the raised-ceiling roster must expose two running and one waiting"
    );
    assert!(
        !harness.steps.saw_scheduled(third.id()),
        "the third request has no Scheduled update while both slots are held"
    );
    let first_done = harness.steps.drain(first.id(), "raised first");
    let second_done = harness.steps.drain(second.id(), "raised second");
    let third_done = harness.steps.drain(third.id(), "raised queued");
    assert_eq!(first_done.tokens, 16);
    assert_eq!(second_done.tokens, 16);
    assert_eq!(third_done.tokens, 8);
    harness.shutdown(&[]);
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn the_full_roster_keeps_its_pipeline_under_a_queue() {
    let mut harness = launch(&[(super::DECODE_SLOTS_ENV, "2")]);
    let first = harness.submit(ids(64, 1), 24);
    let second = harness.submit(ids(64, 2), 40);
    harness.steps.wait_tokens(first.id(), 4);
    harness.steps.wait_tokens(second.id(), 4);
    let queued = harness.submit(ids(64, 3), 6);
    assert!(
        wait_until(Duration::from_secs(10), || {
            let metrics = harness.metrics();
            metrics.num_running_reqs == 2 && metrics.num_waiting_reqs == 1
        }),
        "a full roster must keep the third request queued"
    );
    harness.steps.wait_tokens(first.id(), 8);
    harness.steps.wait_tokens(second.id(), 8);
    assert!(
        !harness.steps.saw_scheduled(queued.id()),
        "the staged incumbents keep advancing without admitting the queued request"
    );
    assert_eq!(harness.steps.drain(first.id(), "incumbent a").tokens, 24);
    assert_eq!(harness.steps.drain(second.id(), "incumbent b").tokens, 40);
    assert_eq!(harness.steps.drain(queued.id(), "queued third").tokens, 6);
    harness.shutdown(&[]);
}

/// The generated states at the slot ceiling. The kernels those states serve
/// through declare how many requests one plan may name; a step with every
/// slot held, half of them decoding while the other half's prompts across
/// the window are admitted, is the step that reaches it.
#[test]
#[ignore = "requires the pinned 12B checkpoint, a build that carries the generated kernels, a GPU, and --test-threads=1"]
fn the_full_roster_serves_through_the_generated_kernels() {
    for knob in ["tilelang", "tilelang640"] {
        let mut harness = launch(&[
            (super::DECODE_SLOTS_ENV, "16"),
            (super::GLOBAL_ATTN_ENV, knob),
        ]);
        let first: Vec<_> = (0..8u32)
            .map(|i| harness.submit(ids(1100 + 3 * i as usize, i + 1), 12))
            .collect();
        for request in &first {
            harness.steps.wait_tokens(request.id(), 2);
        }
        let second: Vec<_> = (8..16u32)
            .map(|i| harness.submit(ids(1100 + 3 * i as usize, i + 1), 12))
            .collect();
        for (slot, request) in first.iter().chain(&second).enumerate() {
            let drained = harness
                .steps
                .drain(request.id(), &format!("{knob} slot {slot}"));
            assert_eq!(drained.tokens, 12, "{knob}: slot {slot} finished short");
        }
        harness.shutdown(&[]);
    }
}

fn run_refill_episode(harness: &mut Harness, prompt: Vec<u32>, budget: usize) -> Drained {
    let request = harness.submit(prompt, budget);
    harness.steps.drain(request.id(), "refill episode")
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
fn an_idle_refill_matches_a_fresh_engine() {
    let prompts = crate::testkit::generate_fixture_prompts();
    let first_len = 64usize;
    let budget = 8usize;
    let first: Vec<u32> = prompts[0].iter().cycle().copied().take(first_len).collect();
    let second: Vec<u32> = prompts[1]
        .iter()
        .cycle()
        .copied()
        .take(first_len + budget - 1)
        .collect();

    let mut refilled_harness = launch(&[(super::DECODE_SLOTS_ENV, "2")]);
    assert_eq!(
        run_refill_episode(&mut refilled_harness, first, budget).tokens,
        budget
    );
    let refilled = run_refill_episode(&mut refilled_harness, second.clone(), budget);
    refilled_harness.shutdown(&[]);

    let mut fresh_harness = launch(&[(super::DECODE_SLOTS_ENV, "2")]);
    let fresh = run_refill_episode(&mut fresh_harness, second, budget);
    fresh_harness.shutdown(&[]);
    assert_eq!(refilled.tokens, budget);
    assert_eq!(
        refilled.ids, fresh.ids,
        "an idle refill answers exactly as a fresh engine does"
    );
}
