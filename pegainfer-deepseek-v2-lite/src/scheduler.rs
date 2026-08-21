//! Mixed-request greedy serving for the DeepSeek-V2-Lite EP2 gate.
//!
//! This is the first serving-semantics gate for the model. It keeps one
//! `DecodeCache` per active request, admits only shapes the current runtime can
//! honor exactly, and retires each request independently when validation,
//! disconnect, EOS, length, or request-local decode errors occur.

use std::collections::VecDeque;
use std::mem;
use std::time::Instant;

mod grouping;
mod trace;

use anyhow::Result;
use anyhow::ensure;
use grouping::common_decode_position;
use grouping::restore_surviving_rows;
use grouping::take_decode_position_groups;
use log::info;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::engine::unix_now_s;
use pegainfer_frontend::sampler::SamplingParams;
use tokio::sync::mpsc;
use trace::RequestTrace;
use trace::ScheduledTrace;
use trace::http_trace_payload;

use crate::Config;
use crate::attribution::DecodeAttributionProfile;
use crate::host_ops::DecodeCache;
use crate::runtime::DeepSeekV2LiteEp2Generator;
use crate::runtime::GenerationStats;

const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 8;

pub(crate) struct MixedRequestScheduler {
    generator: DeepSeekV2LiteEp2Generator,
    submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    pending: VecDeque<PendingRequest>,
    active: Vec<ActiveRequestState>,
    max_active_requests: usize,
}

struct PendingRequest {
    request_id: Option<String>,
    queued_at_unix_s: Option<f64>,
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    stop_policy: StopPolicy,
    max_tokens: usize,
    lora_adapter: Option<String>,
    token_tx: TokenSink,
    logprobs: usize,
    echo: bool,
}

struct ActiveRequestState {
    request_id: Option<String>,
    token_tx: TokenSink,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    last_token: u32,
    stop_policy: StopPolicy,
    model_eos_token_id: u32,
    cache: DecodeCache,
    stats: GenerationStats,
    trace: RequestTrace,
}

struct AdmissionBatch {
    admitted: Vec<PendingRequest>,
    rejected: Vec<(PendingRequest, String)>,
    finished: Vec<PendingRequest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AdmissionDecision {
    Admit,
    Reject(String),
    Finish(FinishReason),
}

impl MixedRequestScheduler {
    pub(crate) fn new(
        generator: DeepSeekV2LiteEp2Generator,
        submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    ) -> Self {
        Self {
            generator,
            submit_rx,
            pending: VecDeque::new(),
            active: Vec::new(),
            max_active_requests: DEFAULT_MAX_ACTIVE_REQUESTS,
        }
    }

    pub(crate) fn run(mut self) {
        while self.block_until_work() {
            self.drain_pending_submissions();
            self.admit_ready_requests();
            if !self.active.is_empty() {
                self.decode_round();
            }
        }
    }

    fn block_until_work(&mut self) -> bool {
        if !self.pending.is_empty() || !self.active.is_empty() {
            return true;
        }

        match self.submit_rx.blocking_recv() {
            Some((req, _kv_prefix)) => {
                self.pending.push_back(PendingRequest::from(req));
                true
            }
            None => false,
        }
    }

    fn drain_pending_submissions(&mut self) {
        while let Ok((req, _kv_prefix)) = self.submit_rx.try_recv() {
            self.pending.push_back(PendingRequest::from(req));
        }
    }

    fn admit_ready_requests(&mut self) {
        let supported_context = self.generator.config().supported_plain_rope_context();
        let batch = take_admission_batch(
            &mut self.pending,
            self.active.len(),
            self.max_active_requests,
            supported_context,
        );

        for (pending, message) in batch.rejected {
            match send_scheduled(&pending) {
                Ok(scheduled) => {
                    let _ = send_prompt_echo(&pending);
                    log_pending_terminal_trace(
                        &pending,
                        &scheduled,
                        FinishReason::Error,
                        0,
                        Some(&message),
                        self.active.len(),
                        self.pending.len(),
                    );
                    let _ = pending.token_tx.send(TokenEvent::Rejected {
                        message,
                        prompt_tokens: pending.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
                Err(scheduled) => log_schedule_disconnect_trace(
                    &pending,
                    &scheduled,
                    self.active.len(),
                    self.pending.len(),
                ),
            }
        }

        for pending in batch.finished {
            match send_scheduled(&pending) {
                Ok(scheduled) => {
                    let _ = send_prompt_echo(&pending);
                    log_pending_terminal_trace(
                        &pending,
                        &scheduled,
                        FinishReason::Length,
                        0,
                        None,
                        self.active.len(),
                        self.pending.len(),
                    );
                    let _ = pending.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Length,
                        stop_cause: None,
                        prompt_tokens: pending.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
                Err(scheduled) => log_schedule_disconnect_trace(
                    &pending,
                    &scheduled,
                    self.active.len(),
                    self.pending.len(),
                ),
            }
        }

        for pending in batch.admitted {
            if self.active.len() >= self.max_active_requests {
                self.pending.push_front(pending);
                break;
            }
            if let Some(active) = self.prefill_request(pending) {
                self.active.push(active);
            }
        }
    }

    fn prefill_request(&mut self, pending: PendingRequest) -> Option<ActiveRequestState> {
        let prompt_len = pending.prompt_tokens.len();
        let scheduled = match send_scheduled(&pending) {
            Ok(scheduled) => scheduled,
            Err(scheduled) => {
                log_schedule_disconnect_trace(
                    &pending,
                    &scheduled,
                    self.active.len(),
                    self.pending.len(),
                );
                return None;
            }
        };

        if !send_prompt_echo(&pending) {
            let terminal_message = terminal_send_failure_message(&pending.token_tx, "prompt echo");
            log_pending_terminal_trace(
                &pending,
                &scheduled,
                FinishReason::Error,
                0,
                Some(&terminal_message),
                self.active.len(),
                self.pending.len(),
            );
            return None;
        }

        let prefill_start = Instant::now();
        let mut cache = DecodeCache::new(self.generator.config());
        let mut stats = self.generator.new_generation_stats(prompt_len);
        let mut attribution = DecodeAttributionProfile::disabled();
        let next = match self.generator.prefill_next_token(
            &pending.prompt_tokens,
            &mut cache,
            &mut stats,
            &mut attribution,
        ) {
            Ok(token) => token,
            Err(err) => {
                let message = err.to_string();
                log_prefill_error_trace(
                    &pending,
                    &scheduled,
                    prompt_len,
                    prefill_start.elapsed().as_secs_f64() * 1000.0,
                    &message,
                    self.active.len(),
                    self.pending.len(),
                );
                let _ = pending.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: prompt_len,
                    completion_tokens: 0,
                });
                return None;
            }
        };
        let prefill_done_unix_s = unix_now_s();
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut active = ActiveRequestState {
            request_id: pending.request_id,
            token_tx: pending.token_tx,
            prompt_len,
            max_tokens: pending.max_tokens,
            generated: 0,
            last_token: next,
            stop_policy: pending.stop_policy,
            model_eos_token_id: self.generator.config().eos_token_id,
            cache,
            stats,
            trace: RequestTrace::new(
                scheduled.queued_at_unix_s,
                scheduled.scheduled_at_unix_s,
                prefill_done_unix_s,
                prefill_ms,
            ),
        };
        active
            .trace
            .note_scheduler_state(self.active.len() + 1, self.pending.len());

        if active.emit_token_or_finish(next, self.active.len(), self.pending.len()) {
            return None;
        }
        Some(active)
    }

    fn decode_round(&mut self) {
        self.retire_bad_cache_positions();
        let active_set_size = self.active.len();
        if active_set_size == 0 {
            return;
        }
        for state in &mut self.active {
            state
                .trace
                .note_scheduler_state(active_set_size, self.pending.len());
        }
        if active_set_size == 1 {
            let row = self.active.pop().expect("single active row present");
            let mut active_remaining = active_set_size;
            if let Some(survivor) = self.decode_single_row((0, row), &mut active_remaining) {
                self.active.push(survivor.1);
            }
            return;
        }

        if let Some(position) = common_decode_position(&self.active) {
            let rows: Vec<_> = self.active.drain(..).enumerate().collect();
            let mut active_remaining = active_set_size;
            self.active = restore_surviving_rows(self.decode_batch_group(
                position,
                rows,
                &mut active_remaining,
            ));
            return;
        }

        let groups = take_decode_position_groups(&mut self.active);
        let mut survivors = Vec::new();
        let mut active_remaining = active_set_size;
        for group in groups {
            if group.rows.len() > 1 {
                survivors.extend(self.decode_batch_group(
                    group.position,
                    group.rows,
                    &mut active_remaining,
                ));
            } else {
                let mut rows = group.rows;
                if let Some(row) = rows.pop() {
                    if let Some(survivor) = self.decode_single_row(row, &mut active_remaining) {
                        survivors.push(survivor);
                    }
                }
            }
        }
        self.active = restore_surviving_rows(survivors);
    }

    fn retire_bad_cache_positions(&mut self) {
        let config = self.generator.config();
        let active_set_size = self.active.len();
        let pending_queue_size = self.pending.len();
        let mut active_set_size_after_terminal = active_set_size;
        let mut survivors = Vec::with_capacity(self.active.len());
        for state in self.active.drain(..) {
            match state.cache_position(config) {
                Ok(()) => survivors.push(state),
                Err(message) => {
                    active_set_size_after_terminal =
                        active_set_size_after_terminal.saturating_sub(1);
                    state.emit_error(
                        message.to_string(),
                        active_set_size_after_terminal,
                        pending_queue_size,
                    );
                }
            }
        }
        self.active = survivors;
    }

    fn decode_batch_group(
        &mut self,
        position: usize,
        rows: Vec<(usize, ActiveRequestState)>,
        active_remaining: &mut usize,
    ) -> Vec<(usize, ActiveRequestState)> {
        let group_size = rows.len();
        let (indices, mut states): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        let tokens: Vec<_> = states.iter().map(|state| state.last_token).collect();
        let token_index = states
            .iter()
            .map(|state| state.generated)
            .min()
            .unwrap_or(0);
        let prompt_tokens = states.iter().map(|state| state.prompt_len).sum();
        let mut stats = self.generator.new_generation_stats(prompt_tokens);
        let mut attribution = DecodeAttributionProfile::disabled();
        let mut caches: Vec<_> = states
            .iter_mut()
            .map(|state| mem::take(&mut state.cache))
            .collect();
        let decode_start = Instant::now();
        let result = self.generator.decode_next_tokens_batch(
            &tokens,
            position,
            &mut caches,
            &mut stats,
            &mut attribution,
            token_index,
        );
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let pending_queue_size = self.pending.len();
        match result {
            Ok(next_tokens) if next_tokens.len() == group_size => {
                for (state, cache) in states.iter_mut().zip(caches) {
                    state.cache = cache;
                    state.trace.note_decode_step(group_size, decode_ms);
                }
                apply_decoded_tokens_to_rows(
                    indices,
                    states,
                    next_tokens,
                    active_remaining,
                    pending_queue_size,
                )
            }
            // The batched path mutates per-row caches as it advances through the
            // model. This gate avoids full-cache rollback clones; a batch decode
            // failure is therefore a shared runtime error for the active rows.
            Ok(next_tokens) => {
                retire_rows_with_error(
                    states,
                    &format!(
                        "DeepSeek-V2-Lite batched decode returned {} rows for {} active requests",
                        next_tokens.len(),
                        group_size
                    ),
                    active_remaining,
                    pending_queue_size,
                );
                Vec::new()
            }
            Err(err) => {
                retire_rows_with_error(
                    states,
                    &format!(
                        "DeepSeek-V2-Lite batched decode failed for {} active requests: {err}",
                        group_size
                    ),
                    active_remaining,
                    pending_queue_size,
                );
                Vec::new()
            }
        }
    }

    fn decode_single_row(
        &mut self,
        (idx, mut state): (usize, ActiveRequestState),
        active_remaining: &mut usize,
    ) -> Option<(usize, ActiveRequestState)> {
        let token = state.last_token;
        let position = state.next_decode_position();
        let token_index = state.generated;
        let decode_start = Instant::now();
        let result = self.generator.decode_next_token(
            token,
            position,
            &mut state.cache,
            &mut state.stats,
            &mut DecodeAttributionProfile::disabled(),
            token_index,
        );
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(next) => {
                state.trace.note_decode_step(1, decode_ms);
                let terminal_active_set_size = active_remaining.saturating_sub(1);
                if state.emit_token_or_finish(next, terminal_active_set_size, self.pending.len()) {
                    *active_remaining = terminal_active_set_size;
                    None
                } else {
                    Some((idx, state))
                }
            }
            Err(err) => {
                *active_remaining = active_remaining.saturating_sub(1);
                state.emit_error(err.to_string(), *active_remaining, self.pending.len());
                None
            }
        }
    }
}

fn apply_decoded_tokens_to_rows(
    indices: Vec<usize>,
    states: Vec<ActiveRequestState>,
    next_tokens: Vec<u32>,
    active_remaining: &mut usize,
    pending_queue_size: usize,
) -> Vec<(usize, ActiveRequestState)> {
    indices
        .into_iter()
        .zip(states.into_iter().zip(next_tokens))
        .filter_map(|(idx, (mut state, token))| {
            let terminal_active_set_size = active_remaining.saturating_sub(1);
            if state.emit_token_or_finish(token, terminal_active_set_size, pending_queue_size) {
                *active_remaining = terminal_active_set_size;
                None
            } else {
                Some((idx, state))
            }
        })
        .collect()
}

fn retire_rows_with_error(
    states: Vec<ActiveRequestState>,
    message: &str,
    active_remaining: &mut usize,
    pending_queue_size: usize,
) {
    for state in states {
        *active_remaining = active_remaining.saturating_sub(1);
        state.emit_error(message.to_string(), *active_remaining, pending_queue_size);
    }
}

impl From<GenerateRequest> for PendingRequest {
    fn from(req: GenerateRequest) -> Self {
        Self {
            request_id: req.request_id,
            queued_at_unix_s: req.queued_at_unix_s,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            stop_policy: req.stop_policy,
            max_tokens: req.max_tokens,
            lora_adapter: req.lora_adapter,
            token_tx: req.token_tx,
            logprobs: req.logprobs,
            echo: req.echo,
        }
    }
}

impl ActiveRequestState {
    fn next_decode_position(&self) -> usize {
        self.prompt_len + self.generated - 1
    }

    fn cache_position(&self, config: &Config) -> Result<()> {
        let expected = self.next_decode_position();
        let actual = self.cache.position(config)?;
        ensure!(
            actual == expected,
            "DeepSeek-V2-Lite request {:?} cache position mismatch: cache_len={}, expected={expected}",
            self.request_id,
            actual
        );
        Ok(())
    }

    fn emit_token_or_finish(
        &mut self,
        token: u32,
        active_set_size_at_terminal: usize,
        pending_queue_size_at_terminal: usize,
    ) -> bool {
        self.last_token = token;
        let stop_cause = self
            .stop_policy
            .classify(token, |token_id| token_id == self.model_eos_token_id);

        let first_emit_at = self
            .trace
            .first_token_emit_unix_s
            .is_none()
            .then(unix_now_s);
        if self
            .token_tx
            .send(TokenEvent::Token {
                id: token,
                logprob: None,
            })
            .is_err()
        {
            let terminal_message = terminal_send_failure_message(&self.token_tx, "token emit");
            self.log_http_trace(
                FinishReason::Error,
                Some(&terminal_message),
                active_set_size_at_terminal,
                pending_queue_size_at_terminal,
            );
            return true;
        }
        if let Some(first_emit_at) = first_emit_at {
            self.trace.first_token_emit_unix_s = Some(first_emit_at);
        }
        self.generated += 1;

        if let Some(stop_cause) = stop_cause {
            self.log_http_trace(
                FinishReason::Stop,
                None,
                active_set_size_at_terminal,
                pending_queue_size_at_terminal,
            );
            let _ = self.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                stop_cause: Some(stop_cause),
                prompt_tokens: self.prompt_len,
                completion_tokens: self.generated,
            });
            return true;
        }

        if self.generated == self.max_tokens {
            self.log_http_trace(
                FinishReason::Length,
                None,
                active_set_size_at_terminal,
                pending_queue_size_at_terminal,
            );
            let _ = self.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                stop_cause: None,
                prompt_tokens: self.prompt_len,
                completion_tokens: self.generated,
            });
            return true;
        }
        false
    }

    fn emit_error(
        mut self,
        message: String,
        active_set_size_at_terminal: usize,
        pending_queue_size_at_terminal: usize,
    ) {
        self.log_http_trace(
            FinishReason::Error,
            Some(&message),
            active_set_size_at_terminal,
            pending_queue_size_at_terminal,
        );
        let _ = self.token_tx.send(TokenEvent::Error {
            message,
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated,
        });
    }

    fn log_http_trace(
        &mut self,
        finish_reason: FinishReason,
        error: Option<&str>,
        active_set_size_at_terminal: usize,
        pending_queue_size_at_terminal: usize,
    ) {
        self.trace
            .note_terminal_state(active_set_size_at_terminal, pending_queue_size_at_terminal);
        log_http_trace(
            trace_request_id(self.request_id.as_deref(), &self.token_tx),
            &self.trace,
            self.prompt_len,
            self.generated,
            finish_reason,
            error,
        );
    }
}

fn send_scheduled(pending: &PendingRequest) -> std::result::Result<ScheduledTrace, ScheduledTrace> {
    let now = unix_now_s();
    let queued_at_unix_s = pending.queued_at_unix_s.unwrap_or(now);
    let scheduled = ScheduledTrace {
        queued_at_unix_s,
        scheduled_at_unix_s: now,
    };
    if pending
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: now,
            prompt_tokens: pending.prompt_tokens.len(),
            cached_tokens: 0,
        })
        .is_ok()
    {
        Ok(scheduled)
    } else {
        Err(scheduled)
    }
}

fn send_prompt_echo(pending: &PendingRequest) -> bool {
    if !pending.echo {
        return true;
    }
    pending
        .token_tx
        .send(TokenEvent::PromptTokens {
            ids: pending.prompt_tokens.clone(),
            logprobs: vec![None; pending.prompt_tokens.len()],
        })
        .is_ok()
}

fn terminal_send_failure_message(token_tx: &TokenSink, stage: &str) -> String {
    if token_tx.is_disconnected() {
        format!("client disconnected before {stage}")
    } else if token_tx.is_cancelled() {
        format!("client cancelled before {stage}")
    } else {
        format!("client disconnected before {stage}")
    }
}

fn trace_request_id<'a>(request_id: Option<&'a str>, token_tx: &'a TokenSink) -> &'a str {
    request_id.unwrap_or_else(|| token_tx.tag().as_ref())
}

fn log_pending_terminal_trace(
    pending: &PendingRequest,
    scheduled: &ScheduledTrace,
    finish_reason: FinishReason,
    completion_tokens: usize,
    error: Option<&str>,
    active_set_size_at_terminal: usize,
    pending_queue_size_at_terminal: usize,
) {
    let mut trace =
        RequestTrace::terminal(scheduled.queued_at_unix_s, scheduled.scheduled_at_unix_s);
    trace.note_terminal_state(active_set_size_at_terminal, pending_queue_size_at_terminal);
    log_http_trace(
        trace_request_id(pending.request_id.as_deref(), &pending.token_tx),
        &trace,
        pending.prompt_tokens.len(),
        completion_tokens,
        finish_reason,
        error,
    );
}

fn log_schedule_disconnect_trace(
    pending: &PendingRequest,
    scheduled: &ScheduledTrace,
    active_set_size_at_terminal: usize,
    pending_queue_size_at_terminal: usize,
) {
    log_pending_terminal_trace(
        pending,
        scheduled,
        FinishReason::Error,
        0,
        Some(&terminal_send_failure_message(
            &pending.token_tx,
            "scheduled event",
        )),
        active_set_size_at_terminal,
        pending_queue_size_at_terminal,
    );
}

fn log_prefill_error_trace(
    pending: &PendingRequest,
    scheduled: &ScheduledTrace,
    prompt_tokens: usize,
    prefill_ms: f64,
    message: &str,
    active_set_size_at_terminal: usize,
    pending_queue_size_at_terminal: usize,
) {
    let mut trace =
        RequestTrace::terminal(scheduled.queued_at_unix_s, scheduled.scheduled_at_unix_s);
    trace.prefill_done_unix_s = Some(unix_now_s());
    trace.prefill_ms = Some(prefill_ms);
    trace.note_terminal_state(active_set_size_at_terminal, pending_queue_size_at_terminal);
    log_http_trace(
        trace_request_id(pending.request_id.as_deref(), &pending.token_tx),
        &trace,
        prompt_tokens,
        0,
        FinishReason::Error,
        Some(message),
    );
}

fn log_http_trace(
    request_id: &str,
    trace: &RequestTrace,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: FinishReason,
    error: Option<&str>,
) {
    let payload = http_trace_payload(
        request_id,
        trace,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        error,
    );
    info!("pegainfer_http_trace {payload}");
}

fn take_admission_batch(
    pending: &mut VecDeque<PendingRequest>,
    active_len: usize,
    max_active_requests: usize,
    supported_context: usize,
) -> AdmissionBatch {
    let mut batch = AdmissionBatch {
        admitted: Vec::new(),
        rejected: Vec::new(),
        finished: Vec::new(),
    };

    while let Some(pending_req) = pending.pop_front() {
        let can_admit = active_len + batch.admitted.len() < max_active_requests;
        match admission_decision(&pending_req, supported_context) {
            AdmissionDecision::Admit if can_admit => batch.admitted.push(pending_req),
            AdmissionDecision::Admit => {
                pending.push_front(pending_req);
                break;
            }
            AdmissionDecision::Reject(message) => batch.rejected.push((pending_req, message)),
            AdmissionDecision::Finish(FinishReason::Length) => batch.finished.push(pending_req),
            AdmissionDecision::Finish(reason) => {
                batch.rejected.push((
                    pending_req,
                    format!("DeepSeek-V2-Lite unsupported admission finish reason: {reason:?}"),
                ));
            }
        }
    }

    batch
}

fn admission_decision(req: &PendingRequest, supported_context: usize) -> AdmissionDecision {
    let prompt_tokens = req.prompt_tokens.len();
    if !req.params.is_greedy() {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate supports greedy decoding only; requested temperature={}, top_k={}, top_p={}",
            req.params.temperature, req.params.top_k, req.params.top_p
        ));
    }
    if req.logprobs > 0 {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate does not return logprobs yet".to_string(),
        );
    }
    if req.lora_adapter.is_some() {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate does not support LoRA adapters".to_string(),
        );
    }
    if req.prompt_tokens.is_empty() {
        return AdmissionDecision::Reject(
            "DeepSeek-V2-Lite EP=2 mixed serving gate requires a non-empty prompt".to_string(),
        );
    }
    if req.max_tokens == 0 {
        return AdmissionDecision::Finish(FinishReason::Length);
    }

    let Some(requested_context) = prompt_tokens.checked_add(req.max_tokens) else {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate context length overflow: prompt_tokens={prompt_tokens} max_new_tokens={}",
            req.max_tokens
        ));
    };
    if requested_context > supported_context {
        return AdmissionDecision::Reject(format!(
            "DeepSeek-V2-Lite EP=2 mixed serving gate supports plain RoPE context <= {supported_context} tokens; requested prompt_tokens={prompt_tokens} max_new_tokens={} total={requested_context}. YaRN rope_scaling long context is not implemented yet.",
            req.max_tokens
        ));
    }

    AdmissionDecision::Admit
}

#[cfg(test)]
mod tests;
