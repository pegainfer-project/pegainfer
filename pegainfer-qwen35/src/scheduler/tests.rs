//! Contract-facing scheduler unit tests. No TokenSink, no GenerateRequest,
//! no EngineHandle, no watch load_tx, no tokio submit_rx.
//!
//! Protocol tests that need a live ledger go through `spawn_scheduler` once a
//! fake backend exists (K3 `scheduler/tests.rs`). `RequestLedger::new` is
//! `pub(crate)` in the frontend crate, so this model crate cannot mint one.
//! Abort prune therefore cannot be driven here: see K3
//! `aborted_request_retires_silently_and_frees_its_slot`.
//!
//! Abolished (structure makes the old failure mode impossible — migration-defense):
//! - `terminal_scheduler_shutdown` fan-out: heir is frontend `drive()` `fail_all`
//!   (`pegainfer-frontend/src/engine/driver.rs`).
//! - `completion_requires_drop_ack` / `publish_before_retire` TokenEvent-vs-drop
//!   ordering: one `RequestUpdate` is committed after the whole step, so drop
//!   always happens before the client sees the terminal.
//! - `FatalSchedulerError.transient` + `TokenEvent::Error` fan-out: `step`
//!   returns `Err`, driver `fail_all`s.

use std::path::Path;

use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EpBackend;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LiveScheduler;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;

use super::DropExpectation;
use super::contract_reject_reason;
use super::echo_refusal;
use super::logical_load_counts;
use super::plan::RejectReason as AdmissionReject;
use super::prefill_drop_expectation;
use super::start_tp_with_capacity;

fn request(prompt_len: usize, max_tokens: usize) -> Request {
    request_with_prompt(vec![7; prompt_len], max_tokens)
}

fn request_with_prompt(prompt_tokens: Vec<u32>, max_tokens: usize) -> Request {
    Request {
        prompt_tokens,
        params: SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: 0,
        echo: false,
        trace_parent: None,
        client_label: None,
    }
}

/// Demultiplex the step stream per request. Same shape as K3 / Qwen3 protocol
/// tests; used by the ignored TP2 GPU case.
struct StepCollector {
    steps: StepReceiver,
    buffered: std::collections::HashMap<RequestId, std::collections::VecDeque<RequestUpdate>>,
}

impl StepCollector {
    fn new(steps: StepReceiver) -> Self {
        Self {
            steps,
            buffered: std::collections::HashMap::new(),
        }
    }

    fn next_for(&mut self, id: RequestId) -> RequestUpdate {
        loop {
            if let Some(update) = self
                .buffered
                .get_mut(&id)
                .and_then(std::collections::VecDeque::pop_front)
            {
                return update;
            }
            let step = self
                .steps
                .blocking_recv()
                .expect("step stream closed while awaiting an update");
            for update in step.updates {
                self.buffered
                    .entry(update.id)
                    .or_default()
                    .push_back(update);
            }
        }
    }

    fn collect_terminal(&mut self, id: RequestId) -> (Vec<u32>, Terminal) {
        let mut tokens = Vec::new();
        loop {
            let update = self.next_for(id);
            tokens.extend_from_slice(&update.tokens);
            if let Some(terminal) = update.terminal {
                return (tokens, terminal);
            }
        }
    }
}

fn partition(mut engine: Engine) -> (LiveScheduler, StepCollector) {
    let mut scheduler = engine.schedulers.remove(0);
    let steps = scheduler
        .handle
        .take_steps()
        .expect("a fresh scheduler yields its step stream once");
    (scheduler, StepCollector::new(steps))
}

/// Heir of the TokenSink mixed-step collection.
fn assert_forced_mixed_steps(engine: Engine) {
    let (partition, mut steps) = partition(engine);

    let decode = partition
        .handle
        .submit(request_with_prompt(vec![151_646], 8));
    let prefill = partition
        .handle
        .submit(request_with_prompt(vec![151_646, 9707], 2));

    let (decode_tokens, decode_finish) = steps.collect_terminal(decode.id());
    let (prefill_tokens, prefill_finish) = steps.collect_terminal(prefill.id());
    assert_eq!(decode_tokens.len(), 8);
    assert!(
        matches!(
            decode_finish,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 8,
                ..
            }
        ),
        "{decode_finish:?}"
    );
    assert_eq!(prefill_tokens.len(), 2);
    assert!(
        matches!(
            prefill_finish,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ),
        "{prefill_finish:?}"
    );
}

// ── Rejection Display (same failure mode, contract heir) ────────────────

#[test]
fn send_rejection_reports_kv_lifetime_request_tokens() {
    let reason = contract_reject_reason(16, 65, AdmissionReject::KvBudget);
    match reason {
        RejectReason::KvBudget {
            prompt_tokens,
            worst_case_tokens,
        } => {
            assert_eq!(prompt_tokens, 16);
            assert_eq!(worst_case_tokens, 80);
        }
        other => panic!("expected KvBudget, got {other:?}"),
    }
    let message = reason.to_string();
    assert!(
        message.contains("max_request_tokens=80"),
        "rejection should report the full lifetime KV request: {message}"
    );
}

#[test]
fn send_rejection_reports_context_window_limit() {
    let reason = contract_reject_reason(16, 17, AdmissionReject::ContextLength { limit: 32 });
    match reason {
        RejectReason::ContextLength {
            prompt_tokens,
            max_tokens,
            limit,
        } => {
            assert_eq!((prompt_tokens, max_tokens, limit), (16, 17, 32));
        }
        other => panic!("expected ContextLength, got {other:?}"),
    }
    let message = reason.to_string();
    assert!(
        message.contains("maximum context length of 32 tokens"),
        "rejection should report the context-window limit: {message}"
    );
    assert!(
        message.contains("requested 33"),
        "rejection should report prompt + max_tokens: {message}"
    );
}

#[test]
fn echo_request_is_rejected_before_backend_admission() {
    let mut echo = request(3, 4);
    echo.echo = true;
    let regular = request(1, 1);

    let reason = echo_refusal(&echo).expect("echo must be refused before KV admission");
    match reason {
        RejectReason::EchoPrefillTokens {
            prompt_tokens,
            limit,
        } => {
            assert_eq!(prompt_tokens, 3);
            assert_eq!(
                limit, 0,
                "echo is unsupported, not merely over a prefill cap"
            );
        }
        other => panic!("expected EchoPrefillTokens, got {other:?}"),
    }
    assert_eq!(
        reason.to_string(),
        RejectReason::EchoPrefillTokens {
            prompt_tokens: 3,
            limit: 0,
        }
        .to_string()
    );
    assert!(
        echo_refusal(&regular).is_none(),
        "only echo requests are ineligible for backend admission"
    );
}

// ── Drop expectation (pure data; abort prune needs a ledger) ────────────
//
// `prune_aborted` will take `&mut RequestLedger`, which this crate cannot
// mint. Queued/active/prefilling abort-retire is covered by K3
// `aborted_request_retires_silently_and_frees_its_slot`. What remains
// testable without a ledger is the TP drop expectation on cursor.

#[test]
fn unmaterialized_prefill_drop_expects_absent_worker_state() {
    assert_eq!(prefill_drop_expectation(0), DropExpectation::MustBeAbsent);
}

#[test]
fn materialized_prefill_drop_requires_existing_worker_state() {
    assert_eq!(prefill_drop_expectation(1), DropExpectation::MustExist);
    assert_eq!(prefill_drop_expectation(16), DropExpectation::MustExist);
}

// ── Overlap wait lives inside step, not as driver idle ──────────────────

#[test]
fn overlap_wait_policy_is_inside_step() {
    // When inflight prefill is set and active is empty, `step` waits on the
    // CUDA event (overlap_wait). It must not return idle to the driver until
    // that prefill is finished — the old `should_block_on_submit` gate
    // (`owned_work_empty && !inflight`) is deleted; the wait is inside step.
    // GPU-untestable here; the reachable heir is that inflight still counts
    // as running so metrics cannot look drained mid-wait.
    assert_eq!(
        logical_load_counts(0, 0, 1, 0),
        (1, 0),
        "inflight prefill is running work, not an idle scheduler"
    );
    assert_eq!(logical_load_counts(1, 0, 1, 2), (2, 2));
}

// ── Launch validation ───────────────────────────────────────────────────

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
    let engine =
        start_tp_with_capacity(&model_path, 42, &[0, 1], 2, 1).expect("start TP2 scheduler");
    assert_forced_mixed_steps(engine);
}
