//! Mechanics tests: admission arithmetic, chunk selection, prefetch policy,
//! and speculative-span resolution — all pure functions of executor state.
//! End-to-end request flow through the engine contract is tested in
//! `crate::frontend_adapter`.

use std::sync::Arc;
use std::sync::Mutex;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_kv_cache::BlockPool;

use super::test_support::FakeExecutor;
use super::test_support::request;
use super::*;
use crate::speculative::VerifyRequestResult;

fn active_state(request_id: u64, generated_count: usize, max_tokens: usize) -> ActiveRequestState {
    ActiveRequestState {
        request_id: RequestId::new(request_id),
        lora_adapter: None,
        last_token: 1,
        generated_count,
        max_tokens,
        prompt_len: 16,
        params: SamplingParams::default(),
        logprobs: 0,
    }
}

#[test]
fn kv_budget_distinguishes_written_tokens_from_lifetime_blocks() {
    let pending = PendingRequest::from_request(RequestId::new(7), request(16, 1));
    assert_eq!(max_request_tokens(&pending), 16);
    assert_eq!(blocks_needed(max_request_tokens(&pending), 16), 1);
    assert_eq!(pending_lifetime_blocks(&pending, 16), 1);

    let pending = PendingRequest::from_request(RequestId::new(8), request(16, 17));
    assert_eq!(max_request_tokens(&pending), 32);
    assert_eq!(blocks_needed(max_request_tokens(&pending), 16), 2);
    assert_eq!(pending_lifetime_blocks(&pending, 16), 3);

    let after_prefill = active_state(8, 1, 17);
    assert_eq!(current_active_tokens(&after_prefill), 16);
    assert_eq!(max_active_tokens(&after_prefill), 32);
    assert_eq!(active_lifetime_blocks(&after_prefill, 16), 3);

    let after_one_decode = active_state(9, 2, 17);
    assert_eq!(current_active_tokens(&after_one_decode), 17);
    assert_eq!(max_active_tokens(&after_one_decode), 32);
    assert_eq!(active_lifetime_blocks(&after_one_decode, 16), 3);
}

#[test]
fn admission_splits_deferred_into_pending_deferred_and_rejected() {
    // block_size 16, per-request cap 4 blocks (max 64 tokens). One active
    // request is mid-flight and will grow into 2 more blocks, so it
    // pre-reserves them out of the budget.
    let active = [ActiveRequestState {
        generated_count: 1, // current tokens = prompt_len (16) -> 1 block
        max_tokens: 18,     // lifetime tokens = 16 + 18 = 34 -> 3 blocks; future growth = 2
        ..active_state(0, 1, 18)
    }];

    let mk = |id: u64, prompt_len, max_tokens| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, max_tokens))
    };
    let deferred = vec![
        mk(1, 16, 1), // one-token completion on a page boundary: admitted
        mk(2, 16, 1), // 1 block: admitted, budget now 0
        mk(3, 16, 1), // 1 block: no budget left -> stays deferred
        mk(4, 80, 1), // 80 prompt tokens -> 5 blocks > cap of 4 -> rejected outright
    ];

    // available 4 blocks - 2 reserved for active growth = budget of 2.
    let outcome =
        admit_deferred_requests(deferred, &active, &[], 16, 4, 4, usize::MAX, 64, 32, |_| 0);

    let ids = |reqs: &[PendingRequest]| reqs.iter().map(|r| r.request_id.raw()).collect::<Vec<_>>();
    assert_eq!(
        ids(&outcome.pending),
        vec![1, 2],
        "admit in order until the budget is spent"
    );
    assert_eq!(
        ids(&outcome.deferred),
        vec![3],
        "budget-starved requests stay deferred, not dropped"
    );
    let rejected_ids = outcome
        .rejected
        .iter()
        .map(|(r, _)| r.request_id.raw())
        .collect::<Vec<_>>();
    assert_eq!(
        rejected_ids,
        vec![4],
        "requests larger than the per-request cap are rejected outright"
    );
}

#[test]
fn requests_exceeding_context_window_are_rejected() {
    let active: [ActiveRequestState; 0] = [];
    let mk = |id: u64, prompt_len, max_tokens| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, max_tokens))
    };

    let deferred = vec![
        mk(1, 16, 16), // request 1: 16 prompt + 16 max = 32 total: admitted
        mk(2, 16, 17), // request 2: 16 prompt + 17 max = 33 total: overflows by 1 token → rejected
        mk(3, 40, 1),  // request 3: 40 prompt + 1 max = 41 total: overflows by 9 tokens → rejected
    ];

    let outcome =
        admit_deferred_requests(deferred, &active, &[], 16, 1000, 1000, 32, 64, 64, |_| 0);

    let pending_ids = outcome
        .pending
        .iter()
        .map(|r| r.request_id.raw())
        .collect::<Vec<_>>();
    assert_eq!(
        pending_ids,
        vec![1],
        "only the request that fits the window is admitted; overflows are rejected, not clamped"
    );

    let rejected_ids = outcome
        .rejected
        .iter()
        .map(|(r, _)| r.request_id.raw())
        .collect::<Vec<_>>();
    assert_eq!(rejected_ids, vec![2, 3]);
    for (_, reason) in &outcome.rejected {
        assert!(
            matches!(reason, RejectReason::ContextLength { limit: 32 }),
            "rejected on the context window, not the KV budget"
        );
    }
}

#[test]
fn admission_respects_decode_batch_capacity() {
    let active: Vec<ActiveRequestState> = (0..64).map(|id| active_state(id, 1, 2)).collect();
    let pending = PendingRequest::from_request(RequestId::new(64), request(16, 1));

    let outcome = admit_deferred_requests(
        vec![pending],
        &active,
        &[],
        16,
        1024,
        1024,
        usize::MAX,
        64,
        32,
        |_| 0,
    );

    assert!(
        outcome.pending.is_empty(),
        "new request must not be admitted past decode scratch capacity"
    );
    assert_eq!(
        outcome.deferred[0].request_id,
        RequestId::new(64),
        "capacity-starved request should stay deferred"
    );
    assert!(outcome.rejected.is_empty());
}

#[test]
fn prefill_chunking_caps_step_tokens_and_keeps_fifo_progress() {
    let mk = |id: u64, prompt_len, max_tokens| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, max_tokens))
    };

    // A prompt larger than the budget is split: the head request gets a
    // budget-sized chunk and everyone behind it waits.
    let mut prefilling = vec![mk(1, 64, 1), mk(2, 16, 1)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(taken.len(), 1);
    assert_eq!(taken[0].request_id, RequestId::new(1));
    assert_eq!(taken[0].step_chunk, 32, "chunk is capped at the budget");
    assert_eq!(
        prefilling[0].request_id,
        RequestId::new(2),
        "follow-up waits for the next step once the budget is spent"
    );

    // Requests pack until the budget is filled exactly; the overflow stays
    // queued in arrival order.
    let mut prefilling = vec![mk(3, 16, 1), mk(4, 16, 1), mk(5, 16, 1)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(
        taken.iter().map(|r| r.step_chunk).collect::<Vec<_>>(),
        vec![16, 16],
        "16 + 16 fills the 32-token budget"
    );
    assert_eq!(prefilling[0].request_id, RequestId::new(5));

    // A partially-prefilled head request only consumes its remainder.
    let mut head = mk(6, 64, 1);
    head.prefill_pos = 48;
    let mut prefilling = vec![head, mk(7, 16, 1)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(
        taken.iter().map(|r| r.step_chunk).collect::<Vec<_>>(),
        vec![16, 16],
        "remainder of the chunked head + the next request share the step"
    );
    assert!(prefilling.is_empty());
}

// The wire protocol does not expose per-step chunk assignments, so gate them
// here.
#[test]
fn request_local_chunks_are_independent_of_earlier_requests() {
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, 1))
    };
    let simulate = |mut prefilling: Vec<PendingRequest>, request_local: bool| {
        let mut target_chunks = Vec::new();
        while prefilling
            .iter()
            .any(|req| req.request_id == RequestId::new(3))
        {
            let taken = take_prefill_chunks(&mut prefilling, 32, request_local);
            let mut continued = Vec::new();
            for mut req in taken {
                if req.request_id == RequestId::new(3) {
                    target_chunks.push(req.step_chunk);
                }
                req.prefill_pos += req.step_chunk;
                if req.remaining_prompt_tokens() > 0 {
                    continued.push(req);
                }
            }
            prefilling.splice(0..0, continued);
        }
        target_chunks
    };
    let alone = || vec![mk(3, 80)];
    let behind_others = || vec![mk(1, 24), mk(2, 16), mk(3, 80)];

    assert_eq!(simulate(alone(), true), vec![32, 32, 16]);
    assert_eq!(simulate(behind_others(), true), vec![32, 32, 16]);

    // Without request-local chunking the same request is cut differently once it queues behind
    // others — the drift this pins. Its absence would make the two asserts above vacuous.
    assert_eq!(simulate(alone(), false), vec![32, 32, 16]);
    assert_ne!(
        simulate(behind_others(), false),
        vec![32, 32, 16],
        "shared-budget chunking must still be batch-dependent, or the request-local asserts prove nothing"
    );
}

#[test]
fn echo_requests_run_only_when_their_prompt_fits_the_prefill_bound() {
    let mk_echo = |id: u64, prompt_len| {
        let mut pending = PendingRequest::from_request(RequestId::new(id), request(prompt_len, 1));
        pending.echo = true;
        pending
    };
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, 1))
    };

    // Oversized echo is rejected by admission. If a caller bypasses
    // admission, the chunk picker must still keep it out of the profiled
    // prefill shape instead of running it whole.
    let mut prefilling = vec![mk_echo(1, 64), mk(2, 16)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(taken.len(), 1);
    assert_eq!(taken[0].request_id, RequestId::new(2));
    assert_eq!(taken[0].step_chunk, 16);
    assert_eq!(
        prefilling[0].request_id,
        RequestId::new(1),
        "oversized echo stays queued if admission was bypassed"
    );

    // An echo that doesn't fit behind earlier work is skipped, not split;
    // later requests may still fill the leftover budget, and the step set
    // stays sorted by request id.
    let mut prefilling = vec![mk(3, 24), mk_echo(4, 16), mk(5, 8)];
    let taken = take_prefill_chunks(&mut prefilling, 32, false);
    assert_eq!(
        taken
            .iter()
            .map(|r| (r.request_id.raw(), r.step_chunk))
            .collect::<Vec<_>>(),
        vec![(3, 24), (5, 8)],
        "echo skipped, leftover budget goes to the next non-echo request"
    );
    assert_eq!(prefilling[0].request_id, RequestId::new(4));
}

#[test]
fn oversized_echo_request_is_rejected_at_admission() {
    let active: [ActiveRequestState; 0] = [];
    let mk_echo = |id: u64, prompt_len| {
        let mut req = request(prompt_len, 1);
        req.echo = true;
        PendingRequest::from_request(RequestId::new(id), req)
    };
    let mk = |id: u64, prompt_len| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, 1))
    };

    let outcome = admit_deferred_requests(
        vec![mk_echo(1, 33), mk(2, 64)],
        &active,
        &[],
        16,
        1024,
        1024,
        usize::MAX,
        64,
        32,
        |_| 0,
    );

    assert_eq!(
        outcome
            .pending
            .iter()
            .map(|r| r.request_id.raw())
            .collect::<Vec<_>>(),
        vec![2],
        "non-echo oversized prompts can still be admitted and chunked"
    );
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.rejected[0].0.request_id, RequestId::new(1));
    assert!(
        matches!(
            outcome.rejected[0].1,
            RejectReason::EchoPrefillTokens { limit: 32 }
        ),
        "oversized echo should be rejected against the profiled prefill bound"
    );
}

#[test]
fn page_boundary_lifetime_blocks_gate_admission() {
    let active: [ActiveRequestState; 0] = [];
    let mk = |id: u64, prompt_len, max_tokens| {
        PendingRequest::from_request(RequestId::new(id), request(prompt_len, max_tokens))
    };

    let under_reserved = admit_deferred_requests(
        vec![mk(1, 16, 17)],
        &active,
        &[],
        16,
        2,
        2,
        usize::MAX,
        64,
        32,
        |_| 0,
    );
    assert!(
        under_reserved.pending.is_empty(),
        "old prompt + max_tokens - 1 arithmetic would admit this request with 2 blocks"
    );
    assert_eq!(under_reserved.rejected.len(), 1);
    assert!(
        matches!(under_reserved.rejected[0].1, RejectReason::KvBudget),
        "request needs 3 lifetime blocks: ceil((16 + 17) / 16)"
    );

    let exactly_reserved = admit_deferred_requests(
        vec![mk(2, 16, 17)],
        &active,
        &[],
        16,
        3,
        3,
        usize::MAX,
        64,
        32,
        |_| 0,
    );
    assert_eq!(
        exactly_reserved.pending[0].request_id,
        RequestId::new(2),
        "ceil((prompt + max_tokens) / block_size) admits the request"
    );
    assert!(exactly_reserved.rejected.is_empty());
}

fn kvbm_peak_draw(prompt_len: usize, max_tokens: usize, block_size: usize) -> usize {
    let pool = BlockPool::new(block_size, 512).expect("test block pool");
    let base = pool.available_blocks();
    let mut peak = 0usize;
    let mut kv = pool.new_request(vec![1; prompt_len], max_tokens, None);

    kv.schedule_prefill(prompt_len, &pool)
        .expect("schedule prefill");
    peak = peak.max(base - pool.available_blocks());
    kv.apply_prefill(100, &pool).expect("apply prefill");
    peak = peak.max(base - pool.available_blocks());

    for step in 1..max_tokens {
        kv.schedule_decode(&pool).expect("schedule decode");
        peak = peak.max(base - pool.available_blocks());
        kv.apply_decode(100 + step as u32, &pool)
            .expect("apply decode");
        peak = peak.max(base - pool.available_blocks());
    }

    kv.release().expect("release request kv");
    assert_eq!(
        pool.available_blocks(),
        base,
        "probe must release every block it draws"
    );
    peak
}

#[test]
fn lifetime_blocks_match_kvbm_peak_draw_at_issue_boundaries() {
    let block_size = 16;
    for (prompt_len, max_tokens) in [(16usize, 17usize), (1, 16), (17, 16)] {
        let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
        let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
        let old = blocks_needed(
            prompt_len.saturating_add(max_tokens.saturating_sub(1)),
            block_size,
        );
        assert_eq!(
            peak, reserved,
            "prompt={prompt_len} max_tokens={max_tokens}"
        );
        assert_eq!(
            old + 1,
            peak,
            "old prompt + max_tokens - 1 arithmetic under-reserved by one block"
        );
    }

    let prompt_len = 33usize;
    let max_tokens = 100usize;
    let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
    let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
    let old = blocks_needed(
        prompt_len.saturating_add(max_tokens.saturating_sub(1)),
        block_size,
    );
    assert_eq!(peak, reserved);
    assert_eq!(
        old, reserved,
        "non-boundary case should not reserve more than the old arithmetic"
    );
}

#[test]
fn lifetime_blocks_never_under_reserve_kvbm_peak_draw() {
    let block_size = 16;
    for prompt_len in 1usize..=64 {
        for max_tokens in 1usize..=64 {
            let reserved = request_lifetime_blocks(prompt_len, max_tokens, block_size);
            let peak = kvbm_peak_draw(prompt_len, max_tokens, block_size);
            assert!(
                peak <= reserved,
                "prompt={prompt_len} max_tokens={max_tokens}: peak={peak}, reserved={reserved}"
            );
        }
    }
}

#[test]
fn echo_requests_are_never_offered_to_prefetch() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let mut executor = FakeExecutor::new(64, dropped);
    let offers = Arc::clone(&executor.prefetch_offers);

    let mk = |id: u64, echo: bool| {
        let mut req = request(32, 1);
        req.echo = echo;
        PendingRequest::from_request(RequestId::new(id), req)
    };
    let mut deferred = vec![mk(1, true), mk(2, false)];
    let mut loading = Vec::new();
    offer_prefetch(&mut executor, &mut deferred, &mut loading, 0);

    // The plain request is probed; the echo request is skipped entirely, so
    // its prefill forwards the whole prompt without parking unspendable KV.
    assert_eq!(*offers.lock().unwrap(), vec![2]);
    let echo = deferred.iter().find(|r| r.request_id.raw() == 1).unwrap();
    assert!(!echo.prefetch_offered, "echo request must stay un-probed");
    let plain = deferred.iter().find(|r| r.request_id.raw() == 2).unwrap();
    assert!(
        plain.prefetch_offered,
        "plain request must be marked probed"
    );
}

// ── Speculative span resolution (multi-token emission) ──────────────────────
// resolve_speculative_outputs walks an accepted span [t_0, .., t_m] and decides,
// per request, whether to emit-and-continue or truncate at a stop token / the
// max-output budget. These were GPU-test-only; the truth table below pins the
// branch behaviour with the FakeExecutor (only is_stop_token matters).

fn spec_active(
    id: u64,
    generated_count: usize,
    max_tokens: usize,
    ignore_eos: bool,
) -> ActiveRequestState {
    ActiveRequestState {
        params: SamplingParams {
            ignore_eos,
            ..SamplingParams::default()
        },
        ..active_state(id, generated_count, max_tokens)
    }
}

fn spec_result(id: u64, accepted: Vec<u32>) -> VerifyRequestResult {
    VerifyRequestResult {
        request_id: RequestId::new(id),
        matched_draft_tokens: accepted.len().saturating_sub(1),
        accepted_tokens: accepted,
    }
}

const SPEC_EOS: u32 = 99;

#[test]
fn speculative_full_span_accept_continues() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 3, 100, false)];
    let results = [spec_result(1, vec![10, 11, 12, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndContinue {
                request_id,
                tokens,
                completion_tokens,
            },
        ] => {
            assert_eq!(*request_id, RequestId::new(1));
            assert_eq!(tokens, &vec![10, 11, 12, 13]);
            assert_eq!(
                *completion_tokens,
                3 + 4,
                "completion = prior generated + span len"
            );
        }
        _ => panic!("expected EmitManyAndContinue"),
    }
}

#[test]
fn speculative_stop_token_midspan_finishes_and_suppresses_eos() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 5, 100, false)];
    // EOS lands at span position 2; tokens before it are emitted, EOS is not.
    let results = [spec_result(1, vec![10, 11, SPEC_EOS, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                ..
            },
        ] => {
            assert_eq!(
                tokens,
                &vec![10, 11],
                "EOS itself is suppressed from emission"
            );
            assert!(matches!(finish_reason, FinishReason::Stop));
        }
        _ => panic!("expected EmitManyAndFinish(Stop)"),
    }
}

#[test]
fn speculative_stop_token_at_span_start_emits_nothing() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 5, 100, false)];
    let results = [spec_result(1, vec![SPEC_EOS, 11, 12])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                ..
            },
        ] => {
            assert!(tokens.is_empty(), "stop at position 0 emits no tokens");
            assert!(matches!(finish_reason, FinishReason::Stop));
        }
        _ => panic!("expected EmitManyAndFinish(Stop)"),
    }
}

#[test]
fn speculative_max_tokens_truncates_midspan() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    // generated 8, budget 10 -> only 2 more tokens fit; the span offers 4.
    let active = [spec_active(1, 8, 10, false)];
    let results = [spec_result(1, vec![10, 11, 12, 13])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [
            effects::DecodeEffect::EmitManyAndFinish {
                tokens,
                finish_reason,
                ..
            },
        ] => {
            assert_eq!(
                tokens,
                &vec![10, 11],
                "the budget-hitting token is emitted, the rest dropped"
            );
            assert!(matches!(finish_reason, FinishReason::Length));
        }
        _ => panic!("expected EmitManyAndFinish(Length)"),
    }
}

#[test]
fn speculative_ignore_eos_does_not_stop() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 0, 100, true)];
    let results = [spec_result(1, vec![SPEC_EOS, SPEC_EOS])];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    match &effects[..] {
        [effects::DecodeEffect::EmitManyAndContinue { tokens, .. }] => {
            assert_eq!(
                tokens,
                &vec![SPEC_EOS, SPEC_EOS],
                "ignore_eos passes stop tokens through"
            );
        }
        _ => panic!("expected EmitManyAndContinue"),
    }
}

#[test]
fn speculative_resolves_each_request_independently() {
    let exec = FakeExecutor::new(64, Arc::new(Mutex::new(Vec::new()))).with_stop_token(SPEC_EOS);
    let active = [spec_active(1, 0, 100, false), spec_active(2, 0, 100, false)];
    let results = [
        spec_result(1, vec![10, 11]),       // continues
        spec_result(2, vec![20, SPEC_EOS]), // finishes on EOS
    ];
    let effects = resolve::resolve_speculative_outputs(&exec, &active, &results);
    assert!(matches!(
        &effects[0],
        effects::DecodeEffect::EmitManyAndContinue { request_id, .. } if *request_id == RequestId::new(1)
    ));
    assert!(matches!(
        &effects[1],
        effects::DecodeEffect::EmitManyAndFinish { request_id, finish_reason: FinishReason::Stop, .. }
            if *request_id == RequestId::new(2)
    ));
}
