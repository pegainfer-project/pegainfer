use pegainfer_frontend::engine::FinishReason;

#[derive(Clone, Debug)]
pub(super) struct RequestTrace {
    queued_at_unix_s: f64,
    scheduled_at_unix_s: f64,
    pub(super) prefill_done_unix_s: Option<f64>,
    pub(super) first_token_emit_unix_s: Option<f64>,
    terminal_unix_s: Option<f64>,
    pub(super) prefill_ms: Option<f64>,
    first_decode_ms: Option<f64>,
    decode_total_ms: f64,
    decode_step_count: usize,
    active_set_size_max: usize,
    pending_queue_size_max: usize,
    active_set_size_at_terminal: usize,
    pending_queue_size_at_terminal: usize,
    decode_batch_size_max: usize,
    batch_decode_steps: usize,
    singleton_decode_steps: usize,
}

#[derive(Debug)]
pub(super) struct ScheduledTrace {
    pub(super) queued_at_unix_s: f64,
    pub(super) scheduled_at_unix_s: f64,
}

impl RequestTrace {
    pub(super) fn new(
        queued_at_unix_s: f64,
        scheduled_at_unix_s: f64,
        prefill_done_unix_s: f64,
        prefill_ms: f64,
    ) -> Self {
        Self {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prefill_done_unix_s: Some(prefill_done_unix_s),
            first_token_emit_unix_s: None,
            terminal_unix_s: None,
            prefill_ms: Some(prefill_ms),
            first_decode_ms: None,
            decode_total_ms: 0.0,
            decode_step_count: 0,
            active_set_size_max: 1,
            pending_queue_size_max: 0,
            active_set_size_at_terminal: 0,
            pending_queue_size_at_terminal: 0,
            decode_batch_size_max: 0,
            batch_decode_steps: 0,
            singleton_decode_steps: 0,
        }
    }

    pub(super) fn terminal(queued_at_unix_s: f64, scheduled_at_unix_s: f64) -> Self {
        Self {
            queued_at_unix_s,
            scheduled_at_unix_s,
            prefill_done_unix_s: None,
            first_token_emit_unix_s: None,
            terminal_unix_s: None,
            prefill_ms: None,
            first_decode_ms: None,
            decode_total_ms: 0.0,
            decode_step_count: 0,
            active_set_size_max: 0,
            pending_queue_size_max: 0,
            active_set_size_at_terminal: 0,
            pending_queue_size_at_terminal: 0,
            decode_batch_size_max: 0,
            batch_decode_steps: 0,
            singleton_decode_steps: 0,
        }
    }

    fn note_active_set(&mut self, active_set_size: usize) {
        self.active_set_size_max = self.active_set_size_max.max(active_set_size);
    }

    pub(super) fn note_scheduler_state(
        &mut self,
        active_set_size: usize,
        pending_queue_size: usize,
    ) {
        self.note_active_set(active_set_size);
        self.pending_queue_size_max = self.pending_queue_size_max.max(pending_queue_size);
    }

    pub(super) fn note_decode_step(&mut self, batch_size: usize, decode_ms: f64) {
        self.first_decode_ms.get_or_insert(decode_ms);
        self.decode_total_ms += decode_ms;
        self.decode_step_count += 1;
        self.decode_batch_size_max = self.decode_batch_size_max.max(batch_size);
        if batch_size > 1 {
            self.batch_decode_steps += 1;
        } else {
            self.singleton_decode_steps += 1;
        }
    }

    pub(super) fn note_terminal_state(
        &mut self,
        active_set_size_at_terminal: usize,
        pending_queue_size_at_terminal: usize,
    ) {
        self.terminal_unix_s = Some(pegainfer_frontend::engine::unix_now_s());
        self.active_set_size_at_terminal = active_set_size_at_terminal;
        self.pending_queue_size_at_terminal = pending_queue_size_at_terminal;
        self.note_scheduler_state(active_set_size_at_terminal, pending_queue_size_at_terminal);
    }

    fn queue_wait_ms(&self) -> f64 {
        (self.scheduled_at_unix_s - self.queued_at_unix_s).max(0.0) * 1000.0
    }

    fn scheduled_to_first_token_ms(&self) -> Option<f64> {
        self.first_token_emit_unix_s
            .map(|first| (first - self.scheduled_at_unix_s).max(0.0) * 1000.0)
    }

    fn scheduled_to_terminal_ms(&self) -> Option<f64> {
        self.terminal_unix_s
            .map(|terminal| (terminal - self.scheduled_at_unix_s).max(0.0) * 1000.0)
    }

    fn decode_mean_ms(&self) -> Option<f64> {
        (self.decode_step_count > 0).then(|| self.decode_total_ms / self.decode_step_count as f64)
    }
}

pub(super) fn http_trace_payload(
    request_id: &str,
    trace: &RequestTrace,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: FinishReason,
    error: Option<&str>,
) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "request_id": request_id,
        "queued_at_unix_s": trace.queued_at_unix_s,
        "scheduled_at_unix_s": trace.scheduled_at_unix_s,
        "prefill_done_unix_s": trace.prefill_done_unix_s,
        "first_token_emit_unix_s": trace.first_token_emit_unix_s,
        "terminal_unix_s": trace.terminal_unix_s,
        "queue_wait_ms": trace.queue_wait_ms(),
        "scheduled_to_first_token_ms": trace.scheduled_to_first_token_ms(),
        "scheduled_to_terminal_ms": trace.scheduled_to_terminal_ms(),
        "prefill_ms": trace.prefill_ms,
        "first_decode_ms": trace.first_decode_ms,
        "decode_total_ms": trace.decode_total_ms,
        "decode_mean_ms": trace.decode_mean_ms(),
        "decode_step_count": trace.decode_step_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason_label(finish_reason),
        "terminal_reason": terminal_reason_label(finish_reason, error),
        "active_set_size": trace.active_set_size_max,
        "active_set_size_max": trace.active_set_size_max,
        "pending_queue_size_max": trace.pending_queue_size_max,
        "active_set_size_at_terminal": trace.active_set_size_at_terminal,
        "pending_queue_size_at_terminal": trace.pending_queue_size_at_terminal,
        "healthy_baseline_after_terminal": trace.active_set_size_at_terminal == 0
            && trace.pending_queue_size_at_terminal == 0,
        "decode_batch_size_max": trace.decode_batch_size_max,
        "batch_decode_steps": trace.batch_decode_steps,
        "singleton_decode_steps": trace.singleton_decode_steps,
    });
    if let Some(error) = error
        && let Some(object) = payload.as_object_mut()
    {
        object.insert(
            "error".to_string(),
            serde_json::Value::String(error.to_string()),
        );
    }
    payload
}

fn finish_reason_label(finish_reason: FinishReason) -> &'static str {
    match finish_reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Error => "error",
    }
}

fn terminal_reason_label(finish_reason: FinishReason, error: Option<&str>) -> &'static str {
    match (finish_reason, error) {
        (FinishReason::Length, _) => "completed_length",
        (FinishReason::Stop, _) => "completed_stop",
        (FinishReason::Error, Some(message)) if message.contains("disconnected") => "disconnected",
        (FinishReason::Error, Some(message)) if message.contains("cancelled") => "cancelled",
        (FinishReason::Error, Some(message))
            if message.contains("supports greedy")
                || message.contains("logprobs")
                || message.contains("LoRA")
                || message.contains("non-empty prompt")
                || message.contains("context")
                || message.contains("unsupported") =>
        {
            "rejected"
        }
        (FinishReason::Error, _) => "error",
    }
}
