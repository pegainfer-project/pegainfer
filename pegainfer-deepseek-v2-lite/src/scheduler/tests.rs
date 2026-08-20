use std::sync::Arc;
use std::sync::atomic::AtomicU8;

use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::RequestAbortReason;
use pegainfer_frontend::engine::RequestTag;
use pegainfer_frontend::engine::StopCause;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::sampler::SamplingParams;
use tokio::sync::mpsc;

use super::grouping::DecodePositionGroupPlan;
use super::grouping::common_decode_position;
use super::grouping::decode_position_groups_for_positions;
use super::*;
use crate::config::test_lite_config;

fn request(
    id: &str,
    prompt_len: usize,
    max_tokens: usize,
) -> (
    PendingRequest,
    pegainfer_frontend::engine::TokenStreamReceiver,
) {
    let (token_tx, token_rx) = TokenSink::standalone();
    (
        PendingRequest {
            request_id: Some(id.to_string()),
            queued_at_unix_s: None,
            prompt_tokens: vec![1; prompt_len],
            params: SamplingParams::default(),
            stop_policy: StopPolicy::default(),
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        },
        token_rx,
    )
}

fn recv_event(rx: &mut pegainfer_frontend::engine::TokenStreamReceiver) -> TokenEvent {
    rx.try_recv().expect("expected event").1
}

fn trace() -> RequestTrace {
    RequestTrace::new(1.0, 2.0, 3.0, 4.0)
}

fn active_state(
    id: &str,
    token_tx: TokenSink,
    prompt_len: usize,
    generated: usize,
    last_token: u32,
    config: &Config,
) -> ActiveRequestState {
    ActiveRequestState {
        request_id: Some(id.to_string()),
        token_tx,
        prompt_len,
        max_tokens: 8,
        generated,
        last_token,
        stop_policy: StopPolicy::default(),
        model_eos_token_id: config.eos_token_id,
        cache: DecodeCache::new(config),
        stats: GenerationStats::default(),
        trace: trace(),
    }
}

#[test]
fn admission_rejects_unsupported_shapes() {
    let context = 16;

    let (mut sampling, _rx) = request("sampling", 1, 1);
    sampling.params.temperature = 0.8;
    assert!(matches!(
        admission_decision(&sampling, context),
        AdmissionDecision::Reject(message) if message.contains("greedy")
    ));

    let (mut logprobs, _rx) = request("logprobs", 1, 1);
    logprobs.logprobs = 1;
    assert!(matches!(
        admission_decision(&logprobs, context),
        AdmissionDecision::Reject(message) if message.contains("logprobs")
    ));

    let (mut lora, _rx) = request("lora", 1, 1);
    lora.lora_adapter = Some("adapter-a".to_string());
    assert!(matches!(
        admission_decision(&lora, context),
        AdmissionDecision::Reject(message) if message.contains("LoRA")
    ));

    let (empty, _rx) = request("empty", 0, 1);
    assert!(matches!(
        admission_decision(&empty, context),
        AdmissionDecision::Reject(message) if message.contains("non-empty prompt")
    ));

    let (zero, _rx) = request("zero", 1, 0);
    assert_eq!(
        admission_decision(&zero, context),
        AdmissionDecision::Finish(FinishReason::Length)
    );
}

#[test]
fn context_overflow_is_rejected() {
    let (req, _rx) = request("too-long", 12, 5);

    assert!(matches!(
        admission_decision(&req, 16),
        AdmissionDecision::Reject(message)
            if message.contains("context") && message.contains("total=17")
    ));
}

#[test]
fn active_cap_defers_in_fcfs_order() {
    let mut pending = VecDeque::new();
    pending.push_back(request("first", 2, 1).0);
    pending.push_back(request("second", 2, 1).0);
    pending.push_back(request("third", 2, 1).0);

    let batch = take_admission_batch(&mut pending, 1, 3, 16);

    assert_eq!(
        batch
            .admitted
            .iter()
            .map(|req| req.request_id.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("first"), Some("second")]
    );
    assert!(batch.rejected.is_empty());
    assert!(batch.finished.is_empty());
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_id.as_deref(), Some("third"));
}

#[test]
fn terminal_requests_do_not_wait_for_active_capacity() {
    let mut pending = VecDeque::new();
    pending.push_back(request("zero", 2, 0).0);
    let (mut invalid, _rx) = request("invalid", 2, 1);
    invalid.logprobs = 1;
    pending.push_back(invalid);
    pending.push_back(request("valid", 2, 1).0);

    let batch = take_admission_batch(&mut pending, 8, 8, 16);

    assert!(batch.admitted.is_empty());
    assert_eq!(batch.finished.len(), 1);
    assert_eq!(batch.finished[0].request_id.as_deref(), Some("zero"));
    assert_eq!(batch.rejected.len(), 1);
    assert_eq!(batch.rejected[0].0.request_id.as_deref(), Some("invalid"));
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_id.as_deref(), Some("valid"));
}

#[test]
fn invalid_request_does_not_block_later_admission_when_cap_has_room() {
    let mut pending = VecDeque::new();
    let (mut invalid, _rx) = request("invalid", 2, 1);
    invalid.logprobs = 1;
    pending.push_back(invalid);
    pending.push_back(request("valid", 2, 1).0);

    let batch = take_admission_batch(&mut pending, 0, 2, 16);

    assert_eq!(batch.rejected.len(), 1);
    assert_eq!(batch.rejected[0].0.request_id.as_deref(), Some("invalid"));
    assert_eq!(batch.admitted.len(), 1);
    assert_eq!(batch.admitted[0].request_id.as_deref(), Some("valid"));
    assert!(pending.is_empty());
}

#[test]
fn terminal_admission_events_keep_scheduler_contract() {
    let (mut zero, mut zero_rx) = request("zero", 2, 0);
    zero.echo = true;
    assert!(send_scheduled(&zero).is_ok());
    assert!(send_prompt_echo(&zero));
    let _ = zero.token_tx.send(TokenEvent::Finished {
        finish_reason: FinishReason::Length,
        stop_cause: None,
        prompt_tokens: zero.prompt_tokens.len(),
        completion_tokens: 0,
    });

    assert!(matches!(
        recv_event(&mut zero_rx),
        TokenEvent::Scheduled { .. }
    ));
    assert!(matches!(
        recv_event(&mut zero_rx),
        TokenEvent::PromptTokens { ids, .. } if ids == vec![1, 1]
    ));
    assert!(matches!(
        recv_event(&mut zero_rx),
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            stop_cause: None,
            completion_tokens: 0,
            ..
        }
    ));

    let (rejected, mut rejected_rx) = request("rejected", 2, 1);
    assert!(send_scheduled(&rejected).is_ok());
    let _ = rejected.token_tx.send(TokenEvent::Rejected {
        message: "nope".to_string(),
        prompt_tokens: rejected.prompt_tokens.len(),
        completion_tokens: 0,
    });

    assert!(matches!(
        recv_event(&mut rejected_rx),
        TokenEvent::Scheduled { .. }
    ));
    assert!(matches!(
        recv_event(&mut rejected_rx),
        TokenEvent::Rejected {
            completion_tokens: 0,
            ..
        }
    ));
}

#[test]
fn immediate_prefill_finish_reports_existing_active_count() {
    let config = test_lite_config();
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let mut state = active_state("one-token", token_tx, 3, 0, 11, &config);
    state.max_tokens = 1;

    assert!(state.emit_token_or_finish(12, 2, 1));

    match recv_event(&mut token_rx) {
        TokenEvent::Token { id, .. } => assert_eq!(id, 12),
        _ => panic!("expected first generated token"),
    }
    match recv_event(&mut token_rx) {
        TokenEvent::Finished {
            finish_reason,
            completion_tokens,
            ..
        } => {
            assert_eq!(finish_reason, FinishReason::Length);
            assert_eq!(completion_tokens, 1);
        }
        _ => panic!("expected terminal length event"),
    }

    let payload = http_trace_payload(
        "one-token",
        &state.trace,
        state.prompt_len,
        state.generated,
        FinishReason::Length,
        None,
    );
    assert_eq!(payload["active_set_size_at_terminal"], 2);
    assert_eq!(payload["pending_queue_size_at_terminal"], 1);
    assert_eq!(payload["healthy_baseline_after_terminal"], false);
}

#[test]
fn send_scheduled_returns_trace_when_client_is_closed() {
    let (pending, rx) = request("closed-before-schedule", 2, 1);
    drop(rx);

    let scheduled = send_scheduled(&pending).expect_err("send should fail");

    assert_eq!(
        scheduled.queued_at_unix_s.to_bits(),
        pending
            .queued_at_unix_s
            .unwrap_or(scheduled.scheduled_at_unix_s)
            .to_bits()
    );
    assert!(scheduled.scheduled_at_unix_s > 0.0);
}

#[test]
fn trace_request_id_falls_back_to_sink_tag() {
    let (sink, _rx) = TokenSink::standalone();

    assert_eq!(trace_request_id(None, &sink), sink.tag().as_ref());
    assert_eq!(trace_request_id(Some("explicit"), &sink), "explicit");
}

#[test]
fn eos_retirement_is_independent_per_request() {
    let config = test_lite_config();
    let (tx_stop, mut rx_stop) = TokenSink::standalone();
    let (tx_live, mut rx_live) = TokenSink::standalone();
    let mut stop_state = active_state("stop", tx_stop, 3, 1, 10, &config);
    let mut live_state = active_state("live", tx_live, 2, 1, 11, &config);

    assert!(stop_state.emit_token_or_finish(config.eos_token_id, 2, 0));
    assert!(!live_state.emit_token_or_finish(12, 2, 0));

    match recv_event(&mut rx_stop) {
        TokenEvent::Token { id, .. } => assert_eq!(id, config.eos_token_id),
        _ => panic!("EOS request should emit its triggering token"),
    }
    match recv_event(&mut rx_stop) {
        TokenEvent::Finished {
            finish_reason,
            stop_cause,
            completion_tokens,
            ..
        } => {
            assert_eq!(finish_reason, FinishReason::Stop);
            assert_eq!(stop_cause, Some(StopCause::Eos(config.eos_token_id)));
            assert_eq!(completion_tokens, 2);
        }
        _ => panic!("EOS request should finish after emitting EOS"),
    }
    match recv_event(&mut rx_live) {
        TokenEvent::Token { id, .. } => assert_eq!(id, 12),
        _ => panic!("live request should receive its own token"),
    }
    assert!(rx_live.try_recv().is_err());
}

#[test]
fn batch_decoded_tokens_retire_eos_independently() {
    let config = test_lite_config();
    let (tx_stop, mut rx_stop) = TokenSink::standalone();
    let (tx_live, mut rx_live) = TokenSink::standalone();
    let stop_state = active_state("stop", tx_stop, 3, 1, 10, &config);
    let live_state = active_state("live", tx_live, 3, 1, 11, &config);

    let mut active_remaining = 2;
    let survivors = apply_decoded_tokens_to_rows(
        vec![0, 1],
        vec![stop_state, live_state],
        vec![config.eos_token_id, 12],
        &mut active_remaining,
        0,
    );

    assert_eq!(active_remaining, 1);
    assert_eq!(survivors.len(), 1);
    assert_eq!(survivors[0].0, 1);
    assert_eq!(survivors[0].1.request_id.as_deref(), Some("live"));
    assert_eq!(survivors[0].1.generated, 2);
    match recv_event(&mut rx_stop) {
        TokenEvent::Token { id, .. } => assert_eq!(id, config.eos_token_id),
        _ => panic!("EOS row should emit its triggering token"),
    }
    match recv_event(&mut rx_stop) {
        TokenEvent::Finished {
            finish_reason,
            stop_cause,
            completion_tokens,
            ..
        } => {
            assert_eq!(finish_reason, FinishReason::Stop);
            assert_eq!(stop_cause, Some(StopCause::Eos(config.eos_token_id)));
            assert_eq!(completion_tokens, 2);
        }
        _ => panic!("EOS row should finish after emitting EOS"),
    }
    match recv_event(&mut rx_live) {
        TokenEvent::Token { id, .. } => assert_eq!(id, 12),
        _ => panic!("live row should receive its decoded token"),
    }
}

#[test]
fn ignored_eos_does_not_disable_explicit_stop_tokens() {
    let config = test_lite_config();
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let mut state = active_state("explicit-stop", token_tx, 3, 1, 10, &config);
    state.stop_policy = StopPolicy {
        eos: EosPolicy::Ignore,
        token_ids: vec![42],
    };
    state.max_tokens = 2;

    assert!(state.emit_token_or_finish(42, 0, 0));

    assert!(matches!(
        recv_event(&mut token_rx),
        TokenEvent::Token { id: 42, .. }
    ));
    assert!(matches!(
        recv_event(&mut token_rx),
        TokenEvent::Finished {
            finish_reason: FinishReason::Stop,
            stop_cause: Some(StopCause::Token(42)),
            completion_tokens: 2,
            ..
        }
    ));
}

#[test]
fn ignored_model_eos_remains_a_normal_generated_token() {
    let config = test_lite_config();
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let mut state = active_state("ignore-eos", token_tx, 3, 1, 10, &config);
    state.stop_policy = StopPolicy {
        eos: EosPolicy::Ignore,
        token_ids: vec![],
    };

    assert!(!state.emit_token_or_finish(config.eos_token_id, 1, 0));

    assert!(matches!(
        recv_event(&mut token_rx),
        TokenEvent::Token { id, .. } if id == config.eos_token_id
    ));
    assert_eq!(state.generated, 2);
    assert!(token_rx.try_recv().is_err());
}

#[test]
fn eos_wins_when_it_is_also_an_explicit_stop_token() {
    let config = test_lite_config();
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let mut state = active_state("overlap", token_tx, 3, 1, 10, &config);
    state.stop_policy = StopPolicy {
        eos: EosPolicy::ModelDefault,
        token_ids: vec![config.eos_token_id],
    };

    assert!(state.emit_token_or_finish(config.eos_token_id, 0, 0));
    assert!(matches!(
        recv_event(&mut token_rx),
        TokenEvent::Token { .. }
    ));
    assert!(matches!(
        recv_event(&mut token_rx),
        TokenEvent::Finished {
            stop_cause: Some(StopCause::Eos(id)),
            ..
        } if id == config.eos_token_id
    ));
}

#[test]
fn cancelled_token_sink_retires_request() {
    let config = test_lite_config();
    let (stream_tx, mut stream_rx) = mpsc::unbounded_channel();
    let cancelled = Arc::new(AtomicU8::new(RequestAbortReason::Cancelled as u8));
    let sink = TokenSink::new(
        RequestTag::from("cancelled"),
        stream_tx,
        Arc::clone(&cancelled),
    );
    let mut state = active_state("cancelled", sink, 2, 1, 11, &config);

    assert!(state.emit_token_or_finish(12, 1, 0));
    assert!(stream_rx.try_recv().is_err());
}

#[test]
fn closed_token_sink_retires_request() {
    let config = test_lite_config();
    let (sink, rx) = TokenSink::standalone();
    drop(rx);
    let mut state = active_state("closed", sink, 2, 1, 11, &config);

    assert!(state.emit_token_or_finish(12, 1, 0));
}

#[test]
fn batch_decode_error_retires_all_active_requests() {
    let config = test_lite_config();
    let (first_tx, mut first_rx) = TokenSink::standalone();
    let (second_tx, mut second_rx) = TokenSink::standalone();
    let active = vec![
        active_state("first", first_tx, 3, 2, 11, &config),
        active_state("second", second_tx, 4, 1, 12, &config),
    ];

    let mut active_remaining = 2;
    retire_rows_with_error(active, "batch failed", &mut active_remaining, 0);
    assert_eq!(active_remaining, 0);

    match recv_event(&mut first_rx) {
        TokenEvent::Error {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert_eq!(message, "batch failed");
            assert_eq!(prompt_tokens, 3);
            assert_eq!(completion_tokens, 2);
        }
        _ => panic!("first active request should receive batch error"),
    }
    match recv_event(&mut second_rx) {
        TokenEvent::Error {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert_eq!(message, "batch failed");
            assert_eq!(prompt_tokens, 4);
            assert_eq!(completion_tokens, 1);
        }
        _ => panic!("second active request should receive batch error"),
    }
}

#[test]
fn subgroup_decode_error_retires_only_group_rows() {
    let config = test_lite_config();
    let (first_tx, mut first_rx) = TokenSink::standalone();
    let (second_tx, mut second_rx) = TokenSink::standalone();
    let (third_tx, mut third_rx) = TokenSink::standalone();
    let first = active_state("first", first_tx, 3, 2, 11, &config);
    let _untouched = active_state("second", second_tx, 4, 1, 12, &config);
    let third = active_state("third", third_tx, 3, 2, 13, &config);

    let mut active_remaining = 3;
    retire_rows_with_error(vec![first, third], "group failed", &mut active_remaining, 0);
    assert_eq!(active_remaining, 1);

    match recv_event(&mut first_rx) {
        TokenEvent::Error { message, .. } => assert_eq!(message, "group failed"),
        _ => panic!("first subgroup row should receive decode error"),
    }
    assert!(second_rx.try_recv().is_err());
    match recv_event(&mut third_rx) {
        TokenEvent::Error { message, .. } => assert_eq!(message, "group failed"),
        _ => panic!("third subgroup row should receive decode error"),
    }
}

#[test]
fn cross_group_terminal_accounting_uses_shared_active_remaining() {
    let config = test_lite_config();
    let (first_tx, mut first_rx) = TokenSink::standalone();
    let (second_tx, mut second_rx) = TokenSink::standalone();
    let (third_tx, mut third_rx) = TokenSink::standalone();
    let first = active_state("first", first_tx, 3, 2, 11, &config);
    let second = active_state("second", second_tx, 3, 2, 12, &config);
    let mut third = active_state("third", third_tx, 3, 1, 13, &config);

    let mut active_remaining = 3;
    retire_rows_with_error(
        vec![first, second],
        "group failed",
        &mut active_remaining,
        0,
    );
    assert_eq!(active_remaining, 1);

    assert!(third.emit_token_or_finish(config.eos_token_id, active_remaining.saturating_sub(1), 0));
    active_remaining = active_remaining.saturating_sub(1);
    assert_eq!(active_remaining, 0);

    let first_payload = match recv_event(&mut first_rx) {
        TokenEvent::Error { .. } => http_trace_payload(
            "first",
            &trace(),
            3,
            2,
            FinishReason::Error,
            Some("group failed"),
        ),
        _ => panic!("first subgroup row should receive decode error"),
    };
    assert_eq!(first_payload["terminal_reason"], "error");
    match recv_event(&mut second_rx) {
        TokenEvent::Error { message, .. } => assert_eq!(message, "group failed"),
        _ => panic!("second subgroup row should receive decode error"),
    }
    match recv_event(&mut third_rx) {
        TokenEvent::Finished { finish_reason, .. } => {
            assert_eq!(finish_reason, FinishReason::Stop);
        }
        _ => panic!("third row should finish independently"),
    }
}

#[test]
fn http_trace_payload_includes_error_and_batch_fields() {
    let mut trace = trace();
    trace.note_scheduler_state(4, 2);
    trace.note_decode_step(2, 7.5);
    trace.note_terminal_state(1, 0);

    let payload = http_trace_payload("req-a", &trace, 3, 2, FinishReason::Error, Some("boom"));

    assert_eq!(payload["request_id"], "req-a");
    assert_eq!(payload["finish_reason"], "error");
    assert_eq!(payload["terminal_reason"], "error");
    assert_eq!(payload["prompt_tokens"], 3);
    assert_eq!(payload["completion_tokens"], 2);
    assert_eq!(payload["active_set_size"], 4);
    assert_eq!(payload["active_set_size_max"], 4);
    assert_eq!(payload["pending_queue_size_max"], 2);
    assert_eq!(payload["active_set_size_at_terminal"], 1);
    assert_eq!(payload["pending_queue_size_at_terminal"], 0);
    assert_eq!(payload["healthy_baseline_after_terminal"], false);
    assert_eq!(payload["decode_batch_size_max"], 2);
    assert_eq!(payload["batch_decode_steps"], 1);
    assert_eq!(payload["singleton_decode_steps"], 0);
    assert_eq!(payload["decode_step_count"], 1);
    assert_eq!(payload["first_decode_ms"], 7.5);
    assert_eq!(payload["decode_total_ms"], 7.5);
    assert_eq!(payload["decode_mean_ms"], 7.5);
    assert_eq!(payload["queue_wait_ms"], 1000.0);
    assert_eq!(
        payload["scheduled_to_first_token_ms"],
        serde_json::Value::Null
    );
    assert!(payload["terminal_unix_s"].as_f64().is_some());
    assert!(payload["scheduled_to_terminal_ms"].as_f64().is_some());
    assert_eq!(payload["error"], "boom");
}

#[test]
fn http_trace_payload_counts_total_and_batched_decode_steps() {
    let mut trace = trace();
    trace.first_token_emit_unix_s = Some(5.0);
    trace.note_decode_step(1, 4.0);
    trace.note_decode_step(4, 8.0);

    let payload = http_trace_payload("req-b", &trace, 3, 2, FinishReason::Length, None);

    assert_eq!(payload["decode_step_count"], 2);
    assert_eq!(payload["batch_decode_steps"], 1);
    assert_eq!(payload["singleton_decode_steps"], 1);
    assert_eq!(payload["decode_batch_size_max"], 4);
    assert_eq!(payload["first_decode_ms"], 4.0);
    assert_eq!(payload["decode_total_ms"], 12.0);
    assert_eq!(payload["decode_mean_ms"], 6.0);
    assert_eq!(payload["scheduled_to_first_token_ms"], 3000.0);
}

#[test]
fn terminal_reason_labels_are_machine_readable() {
    let trace = trace();

    let cancelled = http_trace_payload(
        "cancelled",
        &trace,
        2,
        1,
        FinishReason::Error,
        Some("client cancelled before token emit"),
    );
    assert_eq!(cancelled["terminal_reason"], "cancelled");

    let disconnected = http_trace_payload(
        "disconnected",
        &trace,
        2,
        1,
        FinishReason::Error,
        Some("client disconnected before token emit"),
    );
    assert_eq!(disconnected["terminal_reason"], "disconnected");

    let rejected = http_trace_payload(
        "rejected",
        &trace,
        2,
        0,
        FinishReason::Error,
        Some("DeepSeek-V2-Lite EP=2 mixed serving gate does not return logprobs yet"),
    );
    assert_eq!(rejected["terminal_reason"], "rejected");

    let length = http_trace_payload("length", &trace, 2, 1, FinishReason::Length, None);
    assert_eq!(length["terminal_reason"], "completed_length");
}

#[test]
fn terminal_send_failure_message_distinguishes_cancel_from_disconnect() {
    let (stream_tx, _stream_rx) = mpsc::unbounded_channel();
    let cancelled_flag = Arc::new(AtomicU8::new(RequestAbortReason::Cancelled as u8));
    let cancelled = TokenSink::new(
        RequestTag::from("cancelled"),
        stream_tx,
        Arc::clone(&cancelled_flag),
    );
    assert_eq!(
        terminal_send_failure_message(&cancelled, "token emit"),
        "client cancelled before token emit"
    );

    let (closed, closed_rx) = TokenSink::standalone();
    drop(closed_rx);
    assert_eq!(
        terminal_send_failure_message(&closed, "token emit"),
        "client disconnected before token emit"
    );
}

#[test]
fn decode_grouping_batches_same_position_subgroups() {
    assert!(decode_position_groups_for_positions(&[]).is_empty());
    assert_eq!(
        decode_position_groups_for_positions(&[5]),
        vec![DecodePositionGroupPlan {
            position: 5,
            indices: vec![0],
        }]
    );
    assert_eq!(
        decode_position_groups_for_positions(&[7, 7, 7]),
        vec![DecodePositionGroupPlan {
            position: 7,
            indices: vec![0, 1, 2],
        }]
    );
    assert_eq!(
        decode_position_groups_for_positions(&[7, 8, 7, 8, 9]),
        vec![
            DecodePositionGroupPlan {
                position: 7,
                indices: vec![0, 2],
            },
            DecodePositionGroupPlan {
                position: 8,
                indices: vec![1, 3],
            },
            DecodePositionGroupPlan {
                position: 9,
                indices: vec![4],
            },
        ]
    );
}

#[test]
fn common_decode_position_detects_only_uniform_positions() {
    let config = test_lite_config();
    let (a_tx, _a_rx) = TokenSink::standalone();
    let (b_tx, _b_rx) = TokenSink::standalone();
    let (c_tx, _c_rx) = TokenSink::standalone();

    let uniform = vec![
        active_state("a", a_tx, 4, 1, 10, &config),
        active_state("b", b_tx, 3, 2, 11, &config),
    ];
    assert_eq!(common_decode_position(&uniform), Some(4));

    let mixed = vec![
        active_state("a", c_tx, 4, 1, 10, &config),
        active_state("b", TokenSink::standalone().0, 5, 1, 11, &config),
    ];
    assert_eq!(common_decode_position(&mixed), None);
}

#[test]
fn take_decode_position_groups_preserves_original_indices() {
    let config = test_lite_config();
    let (a_tx, _a_rx) = TokenSink::standalone();
    let (b_tx, _b_rx) = TokenSink::standalone();
    let (c_tx, _c_rx) = TokenSink::standalone();
    let mut active = vec![
        active_state("a", a_tx, 4, 1, 10, &config), // position 4
        active_state("b", b_tx, 4, 2, 11, &config), // position 5
        active_state("c", c_tx, 2, 3, 12, &config), // position 4
    ];

    let groups = take_decode_position_groups(&mut active);

    assert!(active.is_empty());
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].position, 4);
    assert_eq!(
        groups[0]
            .rows
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(groups[1].position, 5);
    assert_eq!(
        groups[1]
            .rows
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn restore_surviving_rows_keeps_original_active_order() {
    let config = test_lite_config();
    let (a_tx, _a_rx) = TokenSink::standalone();
    let (b_tx, _b_rx) = TokenSink::standalone();
    let (d_tx, _d_rx) = TokenSink::standalone();
    let survivors = vec![
        (3, active_state("d", d_tx, 5, 1, 13, &config)),
        (1, active_state("b", b_tx, 4, 1, 11, &config)),
        (0, active_state("a", a_tx, 3, 1, 10, &config)),
    ];

    let restored = restore_surviving_rows(survivors);

    assert_eq!(
        restored
            .iter()
            .map(|state| state.request_id.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("a"), Some("b"), Some("d")]
    );
}
