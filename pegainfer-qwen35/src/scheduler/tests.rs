use std::time::Duration;

use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::Terminal;

use super::*;

fn test_request(label: &str, prompt_tokens: Vec<u32>, max_tokens: usize) -> Request {
    Request {
        prompt_tokens,
        params: SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        stop_policy: StopPolicy::default(),
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: None,
        prompt_logprobs: None,
        trace_parent: None,
        client_label: Some(Arc::from(label)),
    }
}

// The production driver owns registration and commit even in these tests.
fn run_step<F>(requests: Vec<Request>, aborted: &[usize], step: F) -> Vec<RequestUpdate>
where
    F: FnOnce(Vec<QueuedRequest>, &mut RequestLedger) -> Result<()> + Send,
{
    struct TestStep<F> {
        pending: Vec<QueuedRequest>,
        step: Option<F>,
    }
    impl<F> Scheduler for TestStep<F>
    where
        F: FnOnce(Vec<QueuedRequest>, &mut RequestLedger) -> Result<()> + Send,
    {
        fn submit(&mut self, request: QueuedRequest) {
            self.pending.push(request);
        }
        fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
            self.step.take().expect("one test step")(std::mem::take(&mut self.pending), ledger)
        }
        fn metrics(&self) -> SchedulerMetrics {
            SchedulerMetrics::default()
        }
    }
    let (mut handle, wiring) = scheduler_pair();
    let mut steps = handle.take_steps().unwrap();
    let controls: Vec<_> = requests
        .into_iter()
        .map(|request| handle.submit(request))
        .collect();
    for &index in aborted {
        controls[index].abort();
    }
    drop(handle);
    drive(
        TestStep {
            pending: Vec::new(),
            step: Some(step),
        },
        wiring,
    );
    let mut updates = Vec::new();
    while let Ok(step) = steps.try_recv() {
        updates.extend(step.updates);
    }
    updates
}

fn active_request(
    req: QueuedRequest,
    worker_id: u64,
    ledger: &mut RequestLedger,
) -> ActiveRequest35 {
    ledger.admit(req.id);
    ledger.push_tokens(req.id, &[1], &[None]);
    ActiveRequest35 {
        id: req.id,
        client_label: req.request.client_label,
        backend_state: ActiveBackendState::Tp {
            request_id: RequestId::new(worker_id),
            slot_idx: 0,
        },
        last_token: 1,
        max_tokens: req.request.max_tokens,
        prompt_len: req.request.prompt_tokens.len(),
        params: req.request.params,
        logprobs: req.request.logprobs,
    }
}

fn prefilling_request(
    req: QueuedRequest,
    worker_id: u64,
    ledger: &mut RequestLedger,
) -> PrefillingRequest35 {
    ledger.admit(req.id);
    PrefillingRequest35 {
        req,
        backend_state: PrefillBackendState::Tp {
            request_id: RequestId::new(worker_id),
        },
        cursor: 0,
        step_chunk: 0,
    }
}

#[derive(Default)]
struct LifecycleTestBackend {
    stop_token: Option<u32>,
    fail_active_drop: bool,
    fail_prefill_drop: bool,
    active_drops: Vec<RequestId>,
    prefill_drops: Vec<(RequestId, DropExpectation)>,
}

impl DecodeDispatchBackend for LifecycleTestBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_token == Some(token)
    }
    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        active.swap_remove(idx)
    }
    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        let ActiveBackendState::Tp { request_id, .. } = state else {
            panic!("expected TP active state");
        };
        self.active_drops.push(*request_id);
        anyhow::ensure!(!self.fail_active_drop, "injected active drop failure");
        Ok(())
    }
}

impl PrefillPromoteBackend for LifecycleTestBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_token == Some(token)
    }
    fn promote_prefill_state(&mut self, _: usize, _: PrefillBackendState) -> ActiveBackendState {
        panic!("completion lifecycle test must not promote prefill state")
    }
    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()> {
        let PrefillBackendState::Tp { request_id } = state else {
            panic!("expected TP prefill state");
        };
        self.prefill_drops.push((*request_id, expectation));
        anyhow::ensure!(!self.fail_prefill_drop, "injected prefill drop failure");
        Ok(())
    }
}

#[test]
fn closed_pending_work_is_pruned() {
    let updates = run_step(
        vec![
            test_request("closed", vec![1], 1),
            test_request("open", vec![1], 1),
        ],
        &[0],
        |mut pending, ledger| {
            prune_closed_requests(
                &mut LifecycleTestBackend::default(),
                &mut Vec::new(),
                &mut Vec::new(),
                &mut pending,
                ledger,
            )?;
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].request.client_label.as_deref(), Some("open"));
            ledger.retire(pending[0].id);
            Ok(())
        },
    );
    assert!(updates.is_empty());
}

#[test]
fn closed_resident_work_is_pruned() {
    let updates = run_step(
        vec![
            test_request("active-closed", vec![1], 8),
            test_request("active-open", vec![1], 8),
            test_request("prefill-closed", vec![1], 1),
            test_request("pending-open", vec![1], 1),
        ],
        &[0, 2],
        |requests, ledger| {
            let mut requests = requests.into_iter();
            let mut active = vec![
                active_request(requests.next().unwrap(), 10, ledger),
                active_request(requests.next().unwrap(), 11, ledger),
            ];
            let mut prefilling = vec![prefilling_request(requests.next().unwrap(), 12, ledger)];
            let mut pending: Vec<_> = requests.collect();
            let mut backend = LifecycleTestBackend::default();
            prune_closed_requests(
                &mut backend,
                &mut active,
                &mut prefilling,
                &mut pending,
                ledger,
            )?;
            assert_eq!(active.len(), 1);
            assert_eq!(active[0].client_label.as_deref(), Some("active-open"));
            assert!(prefilling.is_empty());
            assert_eq!(backend.active_drops, vec![RequestId::new(10)]);
            assert_eq!(
                backend.prefill_drops,
                vec![(RequestId::new(12), DropExpectation::MustExist)]
            );
            ledger.retire(active[0].id);
            ledger.retire(pending[0].id);
            Ok(())
        },
    );
    assert!(updates.is_empty());
}

#[test]
fn decode_eos_waits_for_drop_before_finished() {
    let mut request = test_request("decode-eos", vec![1], 8);
    request.params.ignore_eos = false;
    let updates = run_step(vec![request], &[], |mut requests, ledger| {
        let mut active = vec![active_request(requests.remove(0), 30, ledger)];
        let mut backend = LifecycleTestBackend {
            stop_token: Some(9),
            ..Default::default()
        };
        dispatch_decode_tokens(&mut backend, &mut active, &[9], &[None], ledger)?;
        assert!(active.is_empty());
        assert_eq!(backend.active_drops, vec![RequestId::new(30)]);
        Ok(())
    });
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].tokens, vec![1]);
    assert!(matches!(
        updates[0].terminal,
        Some(Terminal::Finished {
            reason: FinishReason::Stop,
            completion_tokens: 1,
            stop_cause: None,
            ..
        })
    ));
}

#[test]
fn decode_length_waits_for_drop_before_token_and_finished() {
    let updates = run_step(
        vec![test_request("decode-length", vec![1], 2)],
        &[],
        |mut requests, ledger| {
            let mut active = vec![active_request(requests.remove(0), 31, ledger)];
            let mut backend = LifecycleTestBackend::default();
            dispatch_decode_tokens(&mut backend, &mut active, &[7], &[None], ledger)?;
            assert!(active.is_empty());
            assert_eq!(backend.active_drops, vec![RequestId::new(31)]);
            Ok(())
        },
    );
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].tokens, vec![1, 7]);
    assert!(matches!(
        updates[0].terminal,
        Some(Terminal::Finished {
            reason: FinishReason::Length,
            completion_tokens: 2,
            stop_cause: None,
            ..
        })
    ));
}

#[test]
fn decode_completion_drop_failure_publishes_only_terminal_error() {
    let updates = run_step(
        vec![test_request("decode-drop-failure", vec![1], 2)],
        &[],
        |mut requests, ledger| {
            let mut active = vec![active_request(requests.remove(0), 32, ledger)];
            let id = active[0].id;
            let mut backend = LifecycleTestBackend {
                fail_active_drop: true,
                ..Default::default()
            };
            let result = dispatch_decode_tokens(&mut backend, &mut active, &[7], &[None], ledger);
            assert!(result.is_err());
            assert!(active.is_empty());
            assert_eq!(backend.active_drops, vec![RequestId::new(32)]);
            assert_eq!(
                ledger.completion_tokens(id),
                1,
                "failed retirement must not commit the final token"
            );
            result
        },
    );
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0].tokens, vec![1]);
    assert!(updates[0].terminal.is_none());
    assert!(updates[1].tokens.is_empty());
    assert!(matches!(&updates[1].terminal,
        Some(Terminal::Failed { message, completion_tokens: 1, .. })
        if message.contains("injected active drop failure")
    ));
}

#[test]
fn immediate_prefill_completion_waits_for_drop() {
    let updates = run_step(
        vec![test_request("prefill-length", vec![1], 1)],
        &[],
        |mut requests, ledger| {
            let mut request = prefilling_request(requests.remove(0), 33, ledger);
            request.step_chunk = 1;
            let chunk = ScheduledChunk::from(vec![request]);
            let mut active = Vec::new();
            let mut prefilling = Vec::new();
            let mut backend = LifecycleTestBackend::default();
            promote_or_requeue(
                &mut backend,
                &mut active,
                &mut prefilling,
                chunk,
                &PrefillStepArtifacts::Single {
                    tokens: vec![11],
                    logprobs: vec![None],
                },
                ledger,
            )?;
            assert!(active.is_empty());
            assert!(prefilling.is_empty());
            assert_eq!(
                backend.prefill_drops,
                vec![(RequestId::new(33), DropExpectation::MustExist)]
            );
            Ok(())
        },
    );
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].tokens, vec![11]);
    assert!(matches!(
        updates[0].terminal,
        Some(Terminal::Finished {
            reason: FinishReason::Length,
            stop_cause: None,
            completion_tokens: 1,
            ..
        })
    ));
}

#[test]
fn immediate_prefill_drop_failure_publishes_only_terminal_error() {
    let updates = run_step(
        vec![
            test_request("prefill-drop-failure", vec![1], 1),
            test_request("remaining-scheduled", vec![1], 1),
        ],
        &[],
        |requests, ledger| {
            let mut requests = requests.into_iter();
            let mut first = prefilling_request(requests.next().unwrap(), 34, ledger);
            let id = first.req.id;
            first.step_chunk = 1;
            let mut second = prefilling_request(requests.next().unwrap(), 35, ledger);
            second.step_chunk = 1;
            let mut active = Vec::new();
            let mut prefilling = Vec::new();
            let mut backend = LifecycleTestBackend {
                fail_prefill_drop: true,
                ..Default::default()
            };
            let result = promote_or_requeue(
                &mut backend,
                &mut active,
                &mut prefilling,
                ScheduledChunk::from(vec![first, second]),
                &PrefillStepArtifacts::Single {
                    tokens: vec![12, 13],
                    logprobs: vec![None, None],
                },
                ledger,
            );
            assert!(result.is_err());
            assert_eq!(ledger.completion_tokens(id), 0);
            assert_eq!(
                backend.prefill_drops,
                vec![(RequestId::new(34), DropExpectation::MustExist)]
            );
            result
        },
    );
    for raw_id in 0..2 {
        let request_updates: Vec<_> = updates
            .iter()
            .filter(|update| update.id == FrontendRequestId::new(raw_id))
            .collect();
        let tokens: Vec<_> = request_updates
            .iter()
            .flat_map(|update| &update.tokens)
            .copied()
            .collect();
        assert!(tokens.is_empty());
        let terminals: Vec<_> = request_updates
            .iter()
            .filter_map(|update| update.terminal.as_ref())
            .collect();
        assert_eq!(terminals.len(), 1);
        assert!(matches!(terminals[0], Terminal::Failed { message, .. }
            if message.contains("injected prefill drop failure")));
    }
}

#[test]
fn prefix_cache_chunking_stops_at_snapshot_boundaries() {
    let stride = Some(crate::prefix_cache::SNAPSHOT_STRIDE_TOKENS);
    assert_eq!(clamp_prefill_chunk(0, 900, stride), 256);
    assert_eq!(clamp_prefill_chunk(256, 644, stride), 256);
    assert_eq!(clamp_prefill_chunk(512, 388, stride), 256);
    assert_eq!(clamp_prefill_chunk(768, 132, stride), 132);
    assert_eq!(clamp_prefill_chunk(0, 900, None), 900);
}

#[test]
fn send_rejection_reports_lifetime_kv_and_context_limits() {
    let rejection_reason = |reason, max_tokens| {
        let updates = run_step(
            vec![test_request("rejected", vec![1; 16], max_tokens)],
            &[],
            move |requests, ledger| {
                send_rejection(&requests[0], reason, ledger);
                Ok(())
            },
        );
        assert_eq!(updates.len(), 1);
        match updates.into_iter().next().unwrap().terminal {
            Some(Terminal::Rejected {
                reason,
                prompt_tokens: 16,
            }) => reason,
            other => panic!("expected rejection, got {other:?}"),
        }
    };
    assert_eq!(
        rejection_reason(RejectReason::KvBudget, 49),
        pegainfer_frontend::engine::RejectReason::KvBudget {
            prompt_tokens: 16,
            worst_case_tokens: 65,
        }
    );
    assert_eq!(
        rejection_reason(RejectReason::ContextLength { limit: 32 }, 17),
        pegainfer_frontend::engine::RejectReason::ContextLength {
            prompt_tokens: 16,
            max_tokens: 17,
            limit: 32,
        }
    );
}

#[test]
fn prompt_logprobs_filter_rejects_unsupported_requests() {
    let mut unsupported = test_request("unsupported-prompt-logprobs", vec![1, 2, 3], 4);
    unsupported.prompt_logprobs = Some(0);
    let updates = run_step(
        vec![unsupported, test_request("regular", vec![1], 1)],
        &[],
        |mut pending, ledger| {
            reject_unsupported_prompt_logprobs(&mut pending, ledger);
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].request.client_label.as_deref(), Some("regular"));
            ledger.retire(pending[0].id);
            Ok(())
        },
    );
    assert_eq!(updates.len(), 1);
    assert!(matches!(&updates[0].terminal, Some(Terminal::Rejected {
        reason: pegainfer_frontend::engine::RejectReason::Unsupported { feature }, prompt_tokens: 3,
    }) if feature == "prompt_logprobs"));
}

#[test]
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_scheduler_runs_forced_mixed_steps() {
    let Some(model_path) =
        crate::test_fixture::model_path_or_skip("tp2_scheduler_runs_forced_mixed_steps")
    else {
        return;
    };
    let mut engine = start_tp_with_capacity(&model_path, 42, &[0, 1], 2, 1, false, 0)
        .expect("start TP2 scheduler");
    let LiveScheduler { mut handle, join } = engine.schedulers.remove(0);
    let mut steps = handle.take_steps().unwrap();
    let decode = handle.submit(test_request("mixed-active", vec![151_646], 8));
    let prefill = handle.submit(test_request("mixed-prefill", vec![151_646, 9707], 2));
    let ids = [decode.id(), prefill.id()];
    let expected = [8, 2];
    let mut finished = [false, false];
    let deadline = Instant::now() + Duration::from_secs(30);
    while !finished.iter().all(|done| *done) {
        match steps.try_recv() {
            Ok(step) => {
                for update in step.updates {
                    let index = ids
                        .iter()
                        .position(|id| *id == update.id)
                        .expect("unknown request id");
                    if let Some(terminal) = update.terminal {
                        assert!(
                            matches!(terminal, Terminal::Finished { reason: FinishReason::Length, completion_tokens, .. }
                        if completion_tokens == expected[index])
                        );
                        finished[index] = true;
                    }
                }
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for mixed requests"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                panic!("scheduler exited before mixed requests finished")
            }
        }
    }
    drop(handle);
    join.join().expect("scheduler thread panicked");
}
