use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EpBackend;

use super::*;

fn test_request(request_id: &str, token_tx: TokenSink) -> SchedulerRequest {
    test_request_with_shape(request_id, token_tx, vec![1], 1)
}

fn test_request_with_shape(
    request_id: &str,
    token_tx: TokenSink,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
) -> SchedulerRequest {
    SchedulerRequest {
        trace_parent: None,
        request_id: Some(request_id.to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens,
        params: SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx,
        logprobs: 0,
        echo: false,
    }
}

fn active_request(request_id: u64, label: &str, token_tx: TokenSink) -> ActiveRequest35 {
    ActiveRequest35 {
        request_id: Some(label.to_string()),
        token_tx,
        backend_state: ActiveBackendState::Tp {
            request_id: RequestId::new(request_id),
        },
        last_token: 1,
        generated_count: 1,
        max_tokens: 8,
        prompt_len: 1,
        params: SamplingParams::default(),
        logprobs: 0,
    }
}

fn prefilling_request(request_id: u64, label: &str, token_tx: TokenSink) -> PrefillingRequest35 {
    PrefillingRequest35 {
        req: test_request(label, token_tx),
        backend_state: PrefillBackendState::Tp {
            request_id: RequestId::new(request_id),
        },
        cursor: 0,
        step_chunk: 0,
    }
}

#[derive(Default)]
struct PruneTestBackend {
    retired_active: Vec<RequestId>,
    dropped_prefilling: Vec<RequestId>,
}

impl DecodeDispatchBackend for PruneTestBackend {
    fn is_stop_token(&self, _token: u32) -> bool {
        false
    }

    fn retire_request(&mut self, active: &mut Vec<ActiveRequest35>, idx: usize) {
        let removed = active.swap_remove(idx);
        let ActiveBackendState::Tp { request_id } = removed.backend_state else {
            panic!("prune test expected TP active state");
        };
        self.retired_active.push(request_id);
    }
}

impl PrefillPromoteBackend for PruneTestBackend {
    fn is_stop_token(&self, _token: u32) -> bool {
        false
    }

    fn promote_prefill_state(
        &mut self,
        _active_len: usize,
        _state: PrefillBackendState,
    ) -> ActiveBackendState {
        panic!("prune test must not promote prefill state")
    }

    fn drop_prefill_state(&mut self, state: PrefillBackendState) {
        let PrefillBackendState::Tp { request_id } = state else {
            panic!("prune test expected TP prefill state");
        };
        self.dropped_prefilling.push(request_id);
    }
}

#[test]
fn closed_pending_work_is_pruned_before_admission() {
    let (closed_sink, closed_rx) = TokenSink::standalone();
    drop(closed_rx);
    let (open_sink, _open_rx) = TokenSink::standalone();
    let mut pending = vec![
        test_request("closed", closed_sink),
        test_request("open", open_sink),
    ];
    let mut active = Vec::new();
    let mut prefilling = Vec::new();
    let mut backend = PruneTestBackend::default();

    prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending);

    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_id.as_deref(), Some("open"));
    let admission = admit_pending_requests(
        pending,
        &[],
        1,
        16,
        8,
        8,
        128,
        |req| req.prompt_tokens.len(),
        |req| req.max_tokens,
    );
    assert_eq!(admission.pending.len(), 1);
    assert!(admission.deferred.is_empty());
    assert!(admission.rejected.is_empty());
}

#[test]
fn closed_resident_work_is_absent_from_post_prune_load() {
    let (closed_active_sink, closed_active_rx) = TokenSink::standalone();
    drop(closed_active_rx);
    let (open_active_sink, _open_active_rx) = TokenSink::standalone();
    let (closed_prefill_sink, closed_prefill_rx) = TokenSink::standalone();
    drop(closed_prefill_rx);
    let (pending_sink, _pending_rx) = TokenSink::standalone();
    let mut active = vec![
        active_request(10, "active-closed", closed_active_sink),
        active_request(11, "active-open", open_active_sink),
    ];
    let mut prefilling = vec![prefilling_request(
        12,
        "prefill-closed",
        closed_prefill_sink,
    )];
    let mut pending = vec![test_request("pending-open", pending_sink)];
    let mut backend = PruneTestBackend::default();

    prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending);

    assert_eq!(active.len(), 1);
    assert_eq!(active[0].request_id.as_deref(), Some("active-open"));
    assert!(prefilling.is_empty());
    assert_eq!(
        logical_load_counts(&active, &prefilling, 0, pending.len()),
        (1, 1)
    );
    assert_eq!(backend.retired_active, vec![RequestId::new(10)]);
    assert_eq!(backend.dropped_prefilling, vec![RequestId::new(12)]);
}

#[test]
fn closed_resident_frees_capacity_for_same_tick_admission() {
    let (closed_sink, closed_rx) = TokenSink::standalone();
    drop(closed_rx);
    let (pending_sink, _pending_rx) = TokenSink::standalone();
    let mut active = vec![active_request(20, "resident-closed", closed_sink)];
    let mut prefilling = Vec::new();
    let mut pending = vec![test_request("replacement", pending_sink)];
    let mut backend = PruneTestBackend::default();

    prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending);

    let active_budget: Vec<ActiveKvBudget> = active
        .iter()
        .map(|req| ActiveKvBudget {
            prompt_len: req.prompt_len,
            generated_count: req.generated_count,
            max_tokens: req.max_tokens,
        })
        .collect();
    let admission = admit_pending_requests(
        pending,
        &active_budget,
        1usize.saturating_sub(prefilling.len()),
        16,
        8,
        8,
        128,
        |req| req.prompt_tokens.len(),
        |req| req.max_tokens,
    );

    assert!(active.is_empty());
    assert_eq!(backend.retired_active, vec![RequestId::new(20)]);
    assert_eq!(admission.pending.len(), 1);
    assert_eq!(
        admission.pending[0].request_id.as_deref(),
        Some("replacement")
    );
    assert!(admission.deferred.is_empty());
}

fn wait_for_load(
    load_rx: &watch::Receiver<LoadSnapshot>,
    description: &str,
    predicate: impl Fn(LoadSnapshot) -> bool,
) -> LoadSnapshot {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let snapshot = *load_rx.borrow();
        if predicate(snapshot) {
            return snapshot;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {description}; latest snapshot: {snapshot:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn collect_finished_with_timeout(
    token_rx: &mut openinfer_core::engine::TokenStreamReceiver,
    description: &str,
) -> (usize, FinishReason) {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut token_count = 0;
    loop {
        match token_rx.try_recv() {
            Ok((_, TokenEvent::Token { .. })) => token_count += 1,
            Ok((_, TokenEvent::Finished { finish_reason, .. })) => {
                return (token_count, finish_reason);
            }
            Ok((_, TokenEvent::Error { message, .. })) => {
                panic!("{description} failed: {message}")
            }
            Ok((_, TokenEvent::Rejected { message, .. })) => {
                panic!("{description} was rejected: {message}")
            }
            Ok((_, _)) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {description}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                panic!("{description} channel disconnected before Finished")
            }
        }
    }
}

#[test]
fn send_rejection_reports_kv_lifetime_request_tokens() {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let req = SchedulerRequest {
        trace_parent: None,
        request_id: Some("too-large".to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 65,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req, RejectReason::KvBudget);

    match token_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("max_request_tokens=80"),
                "rejection should report the full lifetime KV request"
            );
        }
        _ => panic!("expected rejection event"),
    }
}

#[test]
fn tp_scheduler_uses_eager_only_plan() {
    let pending = vec!["prefill"];
    assert!(
        matches!(
            build_eager_only_plan(true, pending),
            Some(ExecutionPlan::Prefill { pending }) if pending == vec!["prefill"]
        ),
        "TP Phase 1 should prefill first instead of choosing unified"
    );
    assert!(
        matches!(
            build_eager_only_plan::<&str>(true, vec![]),
            Some(ExecutionPlan::Decode)
        ),
        "TP Phase 1 should decode only when no prefill chunk is scheduled"
    );
}

#[test]
fn inflight_prefill_waits_instead_of_parking_after_last_decode_retires() {
    assert!(
        !should_block_on_submit(true, true, true, true),
        "an in-flight prefill must keep the scheduler off submit_rx.blocking_recv()"
    );
}

#[test]
fn tp_engine_rejects_cuda_graph_before_model_load() {
    let err = match crate::start_engine_with_capacity(
        Path::new("unused"),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0, 1],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
        1,
        1,
    ) {
        Ok(_) => panic!("TP CUDA Graph startup should fail"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("eager execution only"));
}

#[test]
#[ignore = "requires one CUDA device and Qwen3.5 weights"]
fn tp1_cancelled_resident_frees_capacity_before_admission() {
    let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
    let model =
        Qwen35Model::from_safetensors_with_options(&model_path, true).expect("load Qwen3.5 model");
    let handle = start_with_capacity(model, 42, 1, 1_024).expect("start TP1 scheduler");
    let load_rx = handle.load_watch().expect("Qwen3.5 load watch");

    let (resident_tx, resident_rx) = TokenSink::standalone();
    handle
        .submit(test_request_with_shape(
            "resident",
            resident_tx,
            vec![9707; 8_192],
            8,
        ))
        .expect("submit resident request");
    wait_for_load(&load_rx, "resident prefill", |snapshot| {
        snapshot.num_running_reqs == 1
    });

    drop(resident_rx);
    let (replacement_tx, mut replacement_rx) = TokenSink::standalone();
    handle
        .submit(test_request_with_shape(
            "replacement",
            replacement_tx,
            vec![9707; 1_024],
            2,
        ))
        .expect("submit replacement request");

    let post_prune = wait_for_load(&load_rx, "post-prune admission boundary", |snapshot| {
        snapshot.num_running_reqs == 0 && snapshot.num_waiting_reqs == 1
    });
    assert_eq!(post_prune.num_running_reqs, 0);
    assert_eq!(post_prune.num_waiting_reqs, 1);

    let (tokens, finish_reason) =
        collect_finished_with_timeout(&mut replacement_rx, "replacement request");
    assert_eq!(tokens, 2);
    assert_eq!(finish_reason, FinishReason::Length);
    wait_for_load(&load_rx, "idle cleanup", |snapshot| {
        snapshot.num_running_reqs == 0
            && snapshot.num_waiting_reqs == 0
            && snapshot.kv_used_blocks == 0
    });
}

#[test]
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_scheduler_chunked_prefill_then_decode_smoke() {
    let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| "/home/data/mgj/qwen35weights".to_string());
    let handle =
        start_tp_with_capacity(&model_path, 42, &[0, 1], 1, 1).expect("start Qwen3.5 TP scheduler");
    let (token_tx, mut token_rx) = TokenSink::standalone();

    handle
        .submit(SchedulerRequest {
            trace_parent: None,
            request_id: Some("tp2-scheduler-smoke".to_string()),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens: vec![151_646, 9707],
            params: SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            max_tokens: 3,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 1,
            echo: false,
        })
        .expect("submit TP scheduler request");

    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => {
                let logprob = logprob.expect("TP scheduler smoke should return token logprob");
                assert!(logprob.logprob.is_finite());
                assert_eq!(logprob.top_logprobs.len(), 1);
                tokens.push(id);
            }
            Some(TokenEvent::Finished {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            }) => {
                assert_eq!(finish_reason, FinishReason::Length);
                assert_eq!(prompt_tokens, 2);
                assert_eq!(completion_tokens, 3);
                assert_eq!(tokens.len(), 3);
                break;
            }
            Some(
                TokenEvent::Scheduled { .. }
                | TokenEvent::PromptTokens { .. }
                | TokenEvent::KvTransfer { .. },
            ) => {}
            Some(TokenEvent::Error { message, .. }) => {
                panic!("TP scheduler smoke failed: {message}")
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("TP scheduler smoke rejected: {message}")
            }
            None => panic!("TP scheduler channel closed before Finished"),
        }
    }
}

#[test]
fn send_rejection_reports_context_window_limit() {
    let (token_tx, mut token_rx) = TokenSink::standalone();
    let req = SchedulerRequest {
        trace_parent: None,
        request_id: Some("too-long".to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: vec![1; 16],
        params: SamplingParams::default(),
        max_tokens: 17,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx,
        logprobs: 0,
        echo: false,
    };

    send_rejection(&req, RejectReason::ContextLength { limit: 32 });

    match token_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("maximum context length of 32 tokens"),
                "rejection should report the context-window limit"
            );
            assert!(
                message.contains("requested 33"),
                "rejection should report prompt + max_tokens"
            );
        }
        _ => panic!("expected rejection event"),
    }
}
