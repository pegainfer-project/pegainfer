use std::time::Duration;

use super::lane_tests::Drained;
use super::lane_tests::Harness;
use super::lane_tests::ids;
use super::lane_tests::launch;
use super::lane_tests::pin_live_stream;
use super::lane_tests::wait_until;

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
    let second_done = harness.steps.drain(second.id(), "second");
    let third_done = harness.steps.drain(third.id(), "third");
    let fourth_done = harness.steps.drain(fourth.id(), "fourth");
    assert_eq!(
        (
            second_done.scheduled,
            third_done.scheduled,
            fourth_done.scheduled
        ),
        (1, 1, 1),
        "a full cohort releases as one admission burst"
    );
    harness.shutdown(&[&incumbent]);

    let mut timeout_harness = launch(&[
        (super::ADMIT_COALESCE_ENV, "20"),
        (super::DECODE_SLOTS_ENV, "4"),
    ]);
    let timeout_incumbent = pin_live_stream(&mut timeout_harness);
    let timeout_a = timeout_harness.submit(ids(40, 7), 4);
    let timeout_b = timeout_harness.submit(ids(40, 8), 4);
    std::thread::sleep(Duration::from_millis(30));
    timeout_harness
        .steps
        .wait_scheduled_together(&[timeout_a.id(), timeout_b.id()]);
    let timeout_a_done = timeout_harness.steps.drain(timeout_a.id(), "timeout a");
    let timeout_b_done = timeout_harness.steps.drain(timeout_b.id(), "timeout b");
    assert_eq!(
        (timeout_a_done.scheduled, timeout_b_done.scheduled),
        (1, 1),
        "the elapsed window releases an incomplete cohort"
    );
    timeout_harness.shutdown(&[&timeout_incumbent]);
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
