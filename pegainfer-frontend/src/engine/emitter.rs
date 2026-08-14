//! The scheduler-side adapter that turns model activity into the step-batched
//! wire protocol.
//!
//! The emitter is the single writer of the per-step buffer: handle types carry
//! no buffer access of their own (keeping them `Send` without interior
//! mutability), and every state transition of a request routes through here.
//! Timestamps, terminal token counts, and the one-entry-per-request batching
//! are all stamped in this file — model code cannot get them wrong, or skip
//! them, because it never constructs wire types directly.

use std::collections::HashMap;
use std::time::Instant;

use super::event::FinishReason;
use super::event::TokenLogprob;
use super::request_lifecycle::ActiveRequest;
use super::request_lifecycle::DeferredFinish;
use super::request_lifecycle::QueuedRequest;
use super::request_lifecycle::StepReceiver;
use super::request_lifecycle::StepSender;
use super::step::PromptEcho;
use super::step::RejectReason;
use super::step::RequestId;
use super::step::RequestUpdate;
use super::step::ScheduledInfo;
use super::step::StepOutputs;
use super::step::Terminal;

pub struct StepEmitter {
    tx: StepSender,
    /// Current step's per-request records, in first-touch order. `index` maps
    /// a request to its slot; both reset on [`Self::commit_step`].
    buffer: Vec<Option<RequestUpdate>>,
    index: HashMap<RequestId, usize>,
}

impl StepEmitter {
    pub(crate) fn new(tx: StepSender) -> Self {
        Self {
            tx,
            buffer: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// An emitter over a private channel, for unit tests and direct drivers
    /// that consume the step stream without a frontend.
    #[must_use]
    pub fn standalone() -> (Self, StepReceiver) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    fn entry(&mut self, id: RequestId) -> &mut RequestUpdate {
        let slot = *self.index.entry(id).or_insert_with(|| {
            self.buffer.push(Some(RequestUpdate::empty(id)));
            self.buffer.len() - 1
        });
        self.buffer[slot]
            .as_mut()
            .expect("indexed step entry was extracted without index cleanup")
    }

    /// Remove a request's buffered record (deferred finish, retire).
    fn extract(&mut self, id: RequestId) -> Option<RequestUpdate> {
        let slot = self.index.remove(&id)?;
        self.buffer[slot].take()
    }

    // ── QueuedRequest exits ─────────────────────────────────────────

    /// Admit the request: stamp `scheduled_at`, buffer the admission facts,
    /// hand back the streaming-state handle.
    pub fn admit(&mut self, req: QueuedRequest) -> ActiveRequest {
        let inner = req.consume();
        let prompt_tokens = inner.prompt_len;
        self.entry(inner.core.id).scheduled = Some(ScheduledInfo {
            queued_at: inner.queued_at,
            scheduled_at: Instant::now(),
            prompt_tokens,
        });
        ActiveRequest::new(inner.core, prompt_tokens)
    }

    /// Refuse the request at admission.
    pub fn reject(&mut self, req: QueuedRequest, reason: RejectReason) {
        let inner = req.consume();
        self.entry(inner.core.id).terminal = Some(Terminal::Rejected {
            reason,
            prompt_tokens: inner.prompt_len,
        });
    }

    /// Retire a queued request whose frontend already gave up on it (see
    /// [`QueuedRequest::is_aborted`]). Silent: the frontend dropped its state for
    /// this id, so there is no one to address.
    pub fn retire_queued(&mut self, req: QueuedRequest) {
        let inner = req.consume();
        log::debug!("request retired before admission: {}", inner.core.id);
        self.extract(inner.core.id);
    }

    // ── ActiveRequest streaming ─────────────────────────────────────────

    /// Append committed tokens. `logprobs` must be empty (none requested) or
    /// parallel to `ids`; the buffer keeps the two aligned across pushes.
    pub fn push_tokens(
        &mut self,
        request: &mut ActiveRequest,
        ids: &[u32],
        logprobs: &[Option<TokenLogprob>],
    ) {
        assert!(
            logprobs.is_empty() || logprobs.len() == ids.len(),
            "logprobs must be absent or parallel to tokens"
        );
        let inner = request.inner_mut();
        inner.completion_tokens += ids.len();
        let id = inner.core.id;
        let entry = self.entry(id);
        entry.tokens.extend_from_slice(ids);
        if logprobs.is_empty() {
            entry.logprobs.extend(ids.iter().map(|_| None));
        } else {
            entry.logprobs.extend_from_slice(logprobs);
        }
    }

    /// Report the prefix-cache hit count, in the step the scheduler learns it.
    pub fn set_cached_tokens(&mut self, request: &mut ActiveRequest, cached_tokens: usize) {
        let id = request.inner().core.id;
        self.entry(id).cached_tokens = Some(cached_tokens);
    }

    /// Echo the prompt back (echo mode), once, when prefill completes.
    pub fn echo_prompt(&mut self, request: &mut ActiveRequest, echo: PromptEcho) {
        let id = request.inner().core.id;
        self.entry(id).prompt_echo = Some(echo);
    }

    /// Attach P/D handoff metadata to this step's record.
    pub fn kv_transfer(&mut self, request: &mut ActiveRequest, params: serde_json::Value) {
        let id = request.inner().core.id;
        self.entry(id).kv_transfer = Some(params);
    }

    // ── ActiveRequest exits ─────────────────────────────────────────────

    /// Finish the request. Token counts come from the emitter's tally.
    pub fn finish(&mut self, request: ActiveRequest, reason: FinishReason) {
        let inner = request.consume();
        self.entry(inner.core.id).terminal = Some(Terminal::Finished {
            reason,
            prompt_tokens: inner.prompt_tokens,
            completion_tokens: inner.completion_tokens,
        });
    }

    /// Fail the request with an engine-side error.
    pub fn fail(&mut self, request: ActiveRequest, message: impl Into<String>) {
        let inner = request.consume();
        self.entry(inner.core.id).terminal = Some(Terminal::Failed {
            message: message.into(),
            prompt_tokens: inner.prompt_tokens,
            completion_tokens: inner.completion_tokens,
        });
    }

    /// Retire an aborted request (see [`ActiveRequest::aborted`]). Silent,
    /// and discards anything buffered for it this step: the frontend already
    /// dropped its state for this id.
    pub fn retire(&mut self, request: ActiveRequest) {
        let inner = request.consume();
        log::debug!(
            "request retired: frontend aborted: {} tokens_shipped={}",
            inner.core.id,
            inner.completion_tokens
        );
        self.extract(inner.core.id);
    }

    /// Turn the request's finish into a token the scheduler can deliver later
    /// from any thread (P/D prefill roles gate `Finished` on KV-save
    /// visibility). The request's buffered update for this step — tokens
    /// included — folds into the returned message, so late delivery cannot
    /// reorder against the step stream.
    pub fn defer_finish(&mut self, request: ActiveRequest, reason: FinishReason) -> DeferredFinish {
        let inner = request.consume();
        let mut update = self
            .extract(inner.core.id)
            .unwrap_or_else(|| RequestUpdate::empty(inner.core.id));
        update.terminal = Some(Terminal::Finished {
            reason,
            prompt_tokens: inner.prompt_tokens,
            completion_tokens: inner.completion_tokens,
        });
        DeferredFinish::new(update, self.tx.clone())
    }

    /// Fail every remaining request with one message — the teardown sweep a
    /// scheduler runs before reporting a fatal error. (Requests it forgets are
    /// still covered by the handle drop bomb, with a less precise message.)
    pub fn fail_all(&mut self, requests: impl IntoIterator<Item = ActiveRequest>, message: &str) {
        for request in requests {
            self.fail(request, message);
        }
    }

    // ── Step boundary ───────────────────────────────────────────────────

    /// Ship the step's buffer as one message; a step that touched nothing
    /// ships nothing. Called once per driver iteration — model code never
    /// calls this (the driver owns the cadence).
    pub fn commit_step(&mut self) {
        self.index.clear();
        if self.buffer.is_empty() {
            return;
        }
        let updates: Vec<RequestUpdate> = self
            .buffer
            .drain(..)
            .flatten()
            .filter(|update| !update.is_vacant())
            .collect();
        if updates.is_empty() {
            return;
        }
        // A closed receiver means the frontend is gone; the driver notices
        // through the submission channel and winds down.
        let _ = self.tx.send(StepOutputs { updates });
    }
}

#[cfg(test)]
mod tests {
    use super::super::step::Request;
    use super::super::step::Terminal;
    use super::super::wiring::scheduler_pair;
    use super::*;

    fn request(prompt: Vec<u32>) -> Request {
        Request {
            prompt_tokens: prompt,
            params: crate::sampler::SamplingParams::default(),
            max_tokens: 8,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs: 0,
            echo: false,
            trace_parent: None,
            client_label: None,
        }
    }

    #[test]
    fn admission_tokens_and_finish_fold_into_one_entry() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1, 2, 3]));
        let req = backend.submissions.try_recv().expect("req");

        let mut active = backend.emitter.admit(req);
        backend.emitter.push_tokens(&mut active, &[10, 11], &[]);
        backend.emitter.set_cached_tokens(&mut active, 2);
        backend.emitter.finish(active, FinishReason::Stop);
        backend.emitter.commit_step();

        let mut steps = handle_steps(handle);
        let step = steps.try_recv().expect("one step message");
        assert_eq!(step.updates.len(), 1);
        let update = &step.updates[0];
        let scheduled = update.scheduled.as_ref().expect("admission facts");
        assert_eq!(scheduled.prompt_tokens, 3);
        assert!(scheduled.scheduled_at >= scheduled.queued_at);
        assert_eq!(update.tokens, vec![10, 11]);
        assert_eq!(update.logprobs.len(), 2);
        assert_eq!(update.cached_tokens, Some(2));
        assert!(matches!(
            update.terminal,
            Some(Terminal::Finished {
                reason: FinishReason::Stop,
                prompt_tokens: 3,
                completion_tokens: 2,
            })
        ));
        assert!(steps.try_recv().is_err());
    }

    #[test]
    fn reject_carries_prompt_len_and_no_tokens() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1; 5]));
        let req = backend.submissions.try_recv().expect("req");
        backend.emitter.reject(
            req,
            RejectReason::ContextLength {
                prompt_tokens: 5,
                max_tokens: 8,
                limit: 4,
            },
        );
        backend.emitter.commit_step();

        let step = handle_steps(handle).try_recv().expect("step");
        let update = &step.updates[0];
        assert!(update.scheduled.is_none());
        assert!(update.tokens.is_empty());
        assert!(matches!(
            &update.terminal,
            Some(Terminal::Rejected {
                reason: RejectReason::ContextLength { limit: 4, .. },
                prompt_tokens: 5,
            })
        ));
    }

    #[test]
    fn retire_discards_buffered_output() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1]));
        let req = backend.submissions.try_recv().expect("req");
        let mut active = backend.emitter.admit(req);
        backend.emitter.push_tokens(&mut active, &[9], &[]);
        backend.emitter.retire(active);
        backend.emitter.commit_step();

        // Scheduled was buffered before the retire extracted the entry, so
        // nothing observable remains this step.
        assert!(handle_steps(handle).try_recv().is_err());
    }

    #[test]
    fn defer_finish_folds_step_output_and_delivers_late() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1, 2]));
        let req = backend.submissions.try_recv().expect("req");
        let mut active = backend.emitter.admit(req);
        backend.emitter.push_tokens(&mut active, &[7], &[]);
        let deferred = backend.emitter.defer_finish(active, FinishReason::Length);
        backend.emitter.commit_step();

        let mut steps = handle_steps(handle);
        // The request's whole record rode into the deferred finish; the step
        // itself shipped nothing.
        assert!(steps.try_recv().is_err());

        std::thread::spawn(move || deferred.send())
            .join()
            .expect("deferred sender thread");
        let step = steps.try_recv().expect("late finish message");
        let update = &step.updates[0];
        assert!(update.scheduled.is_some());
        assert_eq!(update.tokens, vec![7]);
        assert!(matches!(
            update.terminal,
            Some(Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 1,
            })
        ));
    }

    #[test]
    fn dropped_handles_answer_with_failed_terminals() {
        let (handle, mut backend) = scheduler_pair();
        let _c0 = handle.submit(request(vec![1]));
        let _c1 = handle.submit(request(vec![2, 3]));
        let dropped_req = backend.submissions.try_recv().expect("req 0");
        let admitted = backend
            .emitter
            .admit(backend.submissions.try_recv().expect("req 1"));

        drop(dropped_req);
        drop(admitted);

        let mut steps = handle_steps(handle);
        for expected_prompt in [1usize, 2] {
            let step = steps.try_recv().expect("drop-bomb message");
            assert!(matches!(
                step.updates[0].terminal,
                Some(Terminal::Failed { prompt_tokens, .. }) if prompt_tokens == expected_prompt
            ));
        }
    }

    #[test]
    fn abort_flag_is_visible_through_both_states() {
        let (handle, mut backend) = scheduler_pair();
        let control = handle.submit(request(vec![1]));
        let req = backend.submissions.try_recv().expect("req");
        assert!(!req.is_aborted());

        control.abort();
        assert!(req.is_aborted());
        let active = backend.emitter.admit(req);
        assert!(active.is_aborted());
        backend.emitter.retire(active);
    }

    #[test]
    fn submit_to_a_dead_scheduler_fails_the_request() {
        let (handle, backend) = scheduler_pair();
        drop(backend);
        let _control = handle.submit(request(vec![1, 2, 3, 4]));
        let step = handle_steps(handle).try_recv().expect("drop-bomb message");
        assert!(matches!(
            step.updates[0].terminal,
            Some(Terminal::Failed {
                prompt_tokens: 4,
                ..
            })
        ));
    }

    fn handle_steps(mut handle: super::super::wiring::SchedulerHandle) -> StepReceiver {
        handle.take_steps().expect("step stream not yet taken")
    }
}
