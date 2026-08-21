use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
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
    dropped_prefilling: Vec<(RequestId, DropExpectation)>,
}

impl DecodeDispatchBackend for PruneTestBackend {
    fn is_stop_token(&self, _token: u32) -> bool {
        false
    }

    fn completion_requires_drop_ack(&self) -> bool {
        true
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        active.swap_remove(idx)
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        let ActiveBackendState::Tp { request_id } = state else {
            panic!("prune test expected TP active state");
        };
        self.retired_active.push(*request_id);
        Ok(())
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

    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()> {
        let PrefillBackendState::Tp { request_id } = state else {
            panic!("prune test expected TP prefill state");
        };
        self.dropped_prefilling.push((*request_id, expectation));
        Ok(())
    }
}

struct LifecycleTestBackend {
    stop_token: Option<u32>,
    active_completion_requires_drop_ack: bool,
    fail_active_drop: bool,
    fail_prefill_drop: bool,
    active_drops: Vec<RequestId>,
    active_events_before_drop: Vec<TokenEvent>,
    prefill_drops: Vec<(RequestId, DropExpectation)>,
    observer: Option<pegainfer_frontend::engine::TokenStreamReceiver>,
}

impl LifecycleTestBackend {
    fn new(
        stop_token: Option<u32>,
        observer: pegainfer_frontend::engine::TokenStreamReceiver,
    ) -> Self {
        Self {
            stop_token,
            active_completion_requires_drop_ack: true,
            fail_active_drop: false,
            fail_prefill_drop: false,
            active_drops: Vec::new(),
            active_events_before_drop: Vec::new(),
            prefill_drops: Vec::new(),
            observer: Some(observer),
        }
    }

    fn assert_no_completion_published(&mut self) {
        let Some(observer) = &mut self.observer else {
            return;
        };
        assert!(matches!(
            observer.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }
}

impl DecodeDispatchBackend for LifecycleTestBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_token == Some(token)
    }

    fn completion_requires_drop_ack(&self) -> bool {
        self.active_completion_requires_drop_ack
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        active.swap_remove(idx)
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        let ActiveBackendState::Tp { request_id } = state else {
            panic!("lifecycle test expected TP active state");
        };
        if self.active_completion_requires_drop_ack {
            self.assert_no_completion_published();
        } else if let Some(observer) = &mut self.observer {
            while let Ok((_, event)) = observer.try_recv() {
                self.active_events_before_drop.push(event);
            }
        }
        self.active_drops.push(*request_id);
        anyhow::ensure!(!self.fail_active_drop, "injected active drop failure");
        Ok(())
    }
}

impl PrefillPromoteBackend for LifecycleTestBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_token == Some(token)
    }

    fn promote_prefill_state(
        &mut self,
        _active_len: usize,
        _state: PrefillBackendState,
    ) -> ActiveBackendState {
        panic!("completion lifecycle test must not promote prefill state")
    }

    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()> {
        let PrefillBackendState::Tp { request_id } = state else {
            panic!("lifecycle test expected TP prefill state");
        };
        self.assert_no_completion_published();
        self.prefill_drops.push((*request_id, expectation));
        anyhow::ensure!(!self.fail_prefill_drop, "injected prefill drop failure");
        Ok(())
    }
}

fn next_event(
    rx: &mut pegainfer_frontend::engine::TokenStreamReceiver,
    description: &str,
) -> TokenEvent {
    rx.blocking_recv()
        .unwrap_or_else(|| panic!("{description} channel closed before event"))
        .1
}

fn assert_no_more_events(rx: &mut pegainfer_frontend::engine::TokenStreamReceiver) {
    assert!(
        rx.try_recv().is_err(),
        "request received more than one terminal event"
    );
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

    assert!(
        prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending).is_ok()
    );

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

    assert!(
        prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending).is_ok()
    );

    assert_eq!(active.len(), 1);
    assert_eq!(active[0].request_id.as_deref(), Some("active-open"));
    assert!(prefilling.is_empty());
    assert_eq!(
        logical_load_counts(&active, &prefilling, 0, pending.len()),
        (1, 1)
    );
    assert_eq!(backend.retired_active, vec![RequestId::new(10)]);
    assert_eq!(
        backend.dropped_prefilling,
        vec![(RequestId::new(12), DropExpectation::MustBeAbsent)]
    );
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

    assert!(
        prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending).is_ok()
    );

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

#[test]
fn closed_materialized_prefill_requires_existing_worker_state() {
    let (closed_sink, closed_rx) = TokenSink::standalone();
    drop(closed_rx);
    let mut prefilling = vec![PrefillingRequest35 {
        cursor: 1,
        ..prefilling_request(21, "prefill-materialized", closed_sink)
    }];
    let mut active = Vec::new();
    let mut pending = Vec::new();
    let mut backend = PruneTestBackend::default();

    assert!(
        prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending).is_ok()
    );

    assert!(prefilling.is_empty());
    assert_eq!(
        backend.dropped_prefilling,
        vec![(RequestId::new(21), DropExpectation::MustExist)]
    );
}

#[test]
fn prune_drop_failure_preserves_pending_for_terminal_fanout() {
    let (closed_tx, closed_rx) = TokenSink::standalone();
    drop(closed_rx);
    let (pending_tx, mut pending_rx) = TokenSink::standalone();
    let (_observer_tx, observer_rx) = TokenSink::standalone();
    let mut active = vec![active_request(22, "closed-active", closed_tx)];
    let mut prefilling = Vec::new();
    let mut pending = vec![test_request("live-pending", pending_tx)];
    let mut backend = LifecycleTestBackend::new(None, observer_rx);
    backend.observer = None;
    backend.fail_active_drop = true;

    let failure =
        match prune_closed_requests(&mut backend, &mut active, &mut prefilling, &mut pending) {
            Ok(()) => panic!("injected prune drop should fail"),
            Err(failure) => failure,
        };
    assert!(active.is_empty());
    assert_eq!(failure.transient.len(), 1);
    assert_eq!(pending.len(), 1);

    let (_submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    let (load_tx, _load_rx) = watch::channel(SchedulerMetrics::default());
    terminal_scheduler_shutdown(
        &mut submit_rx,
        &load_tx,
        64,
        active,
        prefilling,
        pending,
        Vec::new(),
        None,
        failure,
    );

    assert!(matches!(
        next_event(&mut pending_rx, "pending after prune failure"),
        TokenEvent::Error { .. }
    ));
    assert_no_more_events(&mut pending_rx);
}

#[test]
fn decode_eos_waits_for_drop_before_finished() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let mut request = active_request(30, "decode-eos", token_tx);
    request.params.ignore_eos = false;
    let mut active = vec![request];
    let mut backend = LifecycleTestBackend::new(Some(9), token_rx);

    assert!(dispatch_decode_tokens(&mut backend, &mut active, &[9], &[None]).is_ok());

    assert!(active.is_empty());
    assert_eq!(backend.active_drops, vec![RequestId::new(30)]);
    let mut token_rx = backend.observer.take().unwrap();
    assert!(matches!(
        next_event(&mut token_rx, "decode EOS"),
        TokenEvent::Finished {
            finish_reason: FinishReason::Stop,
            ..
        }
    ));
    assert_no_more_events(&mut token_rx);
}

#[test]
fn decode_length_waits_for_drop_before_token_and_finished() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let mut request = active_request(31, "decode-length", token_tx);
    request.params.ignore_eos = true;
    request.max_tokens = 2;
    let mut active = vec![request];
    let mut backend = LifecycleTestBackend::new(None, token_rx);

    assert!(dispatch_decode_tokens(&mut backend, &mut active, &[7], &[None]).is_ok());

    assert!(active.is_empty());
    assert_eq!(backend.active_drops, vec![RequestId::new(31)]);
    let mut token_rx = backend.observer.take().unwrap();
    assert!(matches!(
        next_event(&mut token_rx, "decode length token"),
        TokenEvent::Token { id: 7, .. }
    ));
    assert!(matches!(
        next_event(&mut token_rx, "decode length finish"),
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            ..
        }
    ));
    assert_no_more_events(&mut token_rx);
}

#[test]
fn non_tp_decode_preserves_publish_before_retire_order() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let mut request = active_request(36, "single-order", token_tx);
    request.params.ignore_eos = true;
    request.max_tokens = 2;
    let mut active = vec![request];
    let mut backend = LifecycleTestBackend::new(None, token_rx);
    backend.active_completion_requires_drop_ack = false;

    assert!(dispatch_decode_tokens(&mut backend, &mut active, &[8], &[None]).is_ok());

    assert!(active.is_empty());
    assert_eq!(backend.active_drops, vec![RequestId::new(36)]);
    assert_eq!(backend.active_events_before_drop.len(), 2);
    assert!(matches!(
        &backend.active_events_before_drop[0],
        TokenEvent::Token { id: 8, .. }
    ));
    assert!(matches!(
        &backend.active_events_before_drop[1],
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            ..
        }
    ));
}

#[test]
fn decode_completion_drop_failure_publishes_only_terminal_error() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let mut request = active_request(32, "decode-drop-failure", token_tx);
    request.params.ignore_eos = true;
    request.max_tokens = 2;
    let mut active = vec![request];
    let mut backend = LifecycleTestBackend::new(None, token_rx);
    backend.fail_active_drop = true;

    let failure = match dispatch_decode_tokens(&mut backend, &mut active, &[7], &[None]) {
        Ok(()) => panic!("injected active drop should fail"),
        Err(failure) => failure,
    };
    assert!(active.is_empty());
    assert_eq!(failure.transient.len(), 1);
    let mut token_rx = backend.observer.take().unwrap();
    assert!(matches!(
        token_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));

    let (_submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    let (load_tx, _load_rx) = watch::channel(SchedulerMetrics::default());
    terminal_scheduler_shutdown(
        &mut submit_rx,
        &load_tx,
        64,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        failure,
    );

    assert!(matches!(
        next_event(&mut token_rx, "failed decode completion"),
        TokenEvent::Error { .. }
    ));
    assert_no_more_events(&mut token_rx);
}

#[test]
fn immediate_prefill_completion_waits_for_drop() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let mut request = prefilling_request(33, "prefill-length", token_tx);
    request.req.max_tokens = 1;
    request.req.params.ignore_eos = true;
    request.step_chunk = 1;
    let chunk = ScheduledChunk::from(vec![request]);
    let mut active = Vec::new();
    let mut prefilling = Vec::new();
    let mut backend = LifecycleTestBackend::new(None, token_rx);

    assert!(
        promote_or_requeue(
            &mut backend,
            &mut active,
            &mut prefilling,
            chunk,
            &PrefillStepArtifacts::Single {
                tokens: vec![11],
                logprobs: vec![None],
            },
        )
        .is_ok()
    );

    assert!(active.is_empty());
    assert!(prefilling.is_empty());
    assert_eq!(
        backend.prefill_drops,
        vec![(RequestId::new(33), DropExpectation::MustExist)]
    );
    let mut token_rx = backend.observer.take().unwrap();
    assert!(matches!(
        next_event(&mut token_rx, "prefill length token"),
        TokenEvent::Token { id: 11, .. }
    ));
    assert!(matches!(
        next_event(&mut token_rx, "prefill length finish"),
        TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            ..
        }
    ));
    assert_no_more_events(&mut token_rx);
}

#[test]
fn immediate_prefill_drop_failure_publishes_only_terminal_error() {
    let (token_tx, token_rx) = TokenSink::standalone();
    let (remaining_tx, mut remaining_rx) = TokenSink::standalone();
    let mut request = prefilling_request(34, "prefill-drop-failure", token_tx);
    request.req.max_tokens = 1;
    request.req.params.ignore_eos = true;
    request.step_chunk = 1;
    let mut remaining = prefilling_request(35, "remaining-scheduled", remaining_tx);
    remaining.req.max_tokens = 1;
    remaining.req.params.ignore_eos = true;
    remaining.step_chunk = 1;
    let chunk = ScheduledChunk::from(vec![request, remaining]);
    let mut active = Vec::new();
    let mut prefilling = Vec::new();
    let mut backend = LifecycleTestBackend::new(None, token_rx);
    backend.fail_prefill_drop = true;

    let failure = match promote_or_requeue(
        &mut backend,
        &mut active,
        &mut prefilling,
        chunk,
        &PrefillStepArtifacts::Single {
            tokens: vec![12, 13],
            logprobs: vec![None, None],
        },
    ) {
        Ok(()) => panic!("injected prefill drop should fail"),
        Err(failure) => failure,
    };
    assert_eq!(failure.transient.len(), 2);
    let mut token_rx = backend.observer.take().unwrap();
    assert!(matches!(
        token_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        remaining_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));

    let (_submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    let (load_tx, _load_rx) = watch::channel(SchedulerMetrics::default());
    terminal_scheduler_shutdown(
        &mut submit_rx,
        &load_tx,
        64,
        active,
        prefilling,
        Vec::new(),
        Vec::new(),
        None,
        failure,
    );

    assert!(matches!(
        next_event(&mut token_rx, "failed prefill completion"),
        TokenEvent::Error { .. }
    ));
    assert_no_more_events(&mut token_rx);
    assert!(matches!(
        next_event(&mut remaining_rx, "remaining scheduled prefill"),
        TokenEvent::Error { .. }
    ));
    assert_no_more_events(&mut remaining_rx);
}

#[test]
fn terminal_shutdown_closes_drains_and_errors_every_owner_once() {
    let (active_tx, active_rx) = TokenSink::standalone();
    let (prefill_tx, prefill_rx) = TokenSink::standalone();
    let (pending_tx, pending_rx) = TokenSink::standalone();
    let (deferred_tx, deferred_rx) = TokenSink::standalone();
    let (candidate_tx, candidate_rx) = TokenSink::standalone();
    let (scheduled_tx, scheduled_rx) = TokenSink::standalone();
    let (queued_tx, queued_rx) = TokenSink::standalone();
    let (after_close_tx, mut after_close_rx) = TokenSink::standalone();
    let (closed_tx, closed_rx) = TokenSink::standalone();
    drop(closed_rx);

    let active = vec![active_request(40, "active", active_tx)];
    let prefilling = vec![prefilling_request(41, "prefilling", prefill_tx)];
    let pending = vec![
        test_request("duplicate-external-id", pending_tx),
        test_request("closed-sink", closed_tx),
    ];
    let deferred = vec![test_request("duplicate-external-id", deferred_tx)];
    let candidate = CompletionCandidate {
        request: active_request(42, "candidate", candidate_tx),
        final_events: vec![TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: 1,
            completion_tokens: 2,
        }],
    };
    let failure = FatalSchedulerError::new("injected TP replica failure")
        .with_request(candidate.into_terminal())
        .with_request(test_request("scheduled", scheduled_tx));

    let queued = test_request("queued-before-close", queued_tx);
    let after_close = test_request("queued-after-close", after_close_tx);
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel();
    let before_send = Arc::new(Barrier::new(2));
    let sent_before_close = Arc::new(Barrier::new(2));
    let after_receiver_close = Arc::new(Barrier::new(2));
    let sender = {
        let before_send = Arc::clone(&before_send);
        let sent_before_close = Arc::clone(&sent_before_close);
        let after_receiver_close = Arc::clone(&after_receiver_close);
        std::thread::spawn(move || {
            before_send.wait();
            submit_tx
                .send((queued, pegainfer_frontend::engine::KvPrefix::none()))
                .expect("close-before request should be accepted");
            sent_before_close.wait();
            after_receiver_close.wait();
            submit_tx
                .send((after_close, pegainfer_frontend::engine::KvPrefix::none()))
                .is_err()
        })
    };

    before_send.wait();
    sent_before_close.wait();
    let (load_tx, load_rx) = watch::channel(SchedulerMetrics {
        kv_used_blocks: 9,
        kv_total_blocks: 64,
        num_running_reqs: 9,
        num_waiting_reqs: 9,
        spec_decode: None,
    });
    terminal_scheduler_shutdown(
        &mut submit_rx,
        &load_tx,
        64,
        active,
        prefilling,
        pending,
        deferred,
        None,
        failure,
    );
    after_receiver_close.wait();
    assert!(sender.join().expect("submit race thread panicked"));

    let mut receivers = vec![
        ("active", active_rx),
        ("prefilling", prefill_rx),
        ("pending", pending_rx),
        ("deferred", deferred_rx),
        ("candidate", candidate_rx),
        ("scheduled", scheduled_rx),
        ("queued", queued_rx),
    ];
    for (owner, rx) in &mut receivers {
        match next_event(rx, owner) {
            TokenEvent::Error { message, .. } => {
                assert_eq!(message, "injected TP replica failure");
            }
            other => panic!("{owner} received non-error terminal event: {other:?}"),
        }
        assert_no_more_events(rx);
    }
    assert!(after_close_rx.try_recv().is_err());

    let snapshot = *load_rx.borrow();
    assert_eq!(snapshot.kv_used_blocks, 0);
    assert_eq!(snapshot.kv_total_blocks, 64);
    assert_eq!(snapshot.num_running_reqs, 0);
    assert_eq!(snapshot.num_waiting_reqs, 0);
}

fn collect_finished_with_timeout(
    token_rx: &mut pegainfer_frontend::engine::TokenStreamReceiver,
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
fn echo_request_is_rejected_before_backend_admission() {
    let (echo_tx, mut echo_rx) = TokenSink::standalone();
    let (regular_tx, mut regular_rx) = TokenSink::standalone();
    let mut echo = test_request_with_shape("unsupported-echo", echo_tx, vec![1, 2, 3], 4);
    echo.echo = true;
    let regular = test_request("regular", regular_tx);
    let mut pending = vec![echo, regular];

    reject_unsupported_echo(&mut pending);

    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_id.as_deref(), Some("regular"));
    assert!(
        !pending[0].echo,
        "only requests eligible for backend admission may remain"
    );
    match echo_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(message, UNSUPPORTED_ECHO_MESSAGE);
            assert_eq!(prompt_tokens, 3);
            assert_eq!(completion_tokens, 0);
        }
        event => panic!("expected unsupported echo rejection, got {event:?}"),
    }
    assert!(matches!(
        regular_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn inflight_prefill_waits_instead_of_parking_after_last_decode_retires() {
    assert!(
        !should_block_on_submit(true, true),
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
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_scheduler_runs_forced_mixed_steps() {
    let Some(model_path) =
        crate::test_fixture::model_path_or_skip("tp2_scheduler_runs_forced_mixed_steps")
    else {
        return;
    };
    let handle =
        start_tp_with_capacity(&model_path, 42, &[0, 1], 2, 1).expect("start TP2 scheduler");
    let (decode_tx, mut decode_rx) = TokenSink::standalone();
    let (prefill_tx, mut prefill_rx) = TokenSink::standalone();

    handle
        .submit(test_request_with_shape(
            "mixed-active",
            decode_tx,
            vec![151_646],
            8,
        ))
        .expect("submit request that becomes active first");
    handle
        .submit(test_request_with_shape(
            "mixed-prefill",
            prefill_tx,
            vec![151_646, 9707],
            2,
        ))
        .expect("submit request that remains chunk-prefilling");

    let (decode_tokens, decode_finish) =
        collect_finished_with_timeout(&mut decode_rx, "mixed active request");
    let (prefill_tokens, prefill_finish) =
        collect_finished_with_timeout(&mut prefill_rx, "mixed prefill request");
    assert_eq!(decode_tokens, 8);
    assert_eq!(decode_finish, FinishReason::Length);
    assert_eq!(prefill_tokens, 2);
    assert_eq!(prefill_finish, FinishReason::Length);
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
