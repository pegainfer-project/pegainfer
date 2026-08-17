//! [`RequestLedger`] — the account book of live requests, and the per-step
//! statement it ships on the step stream.
//!
//! One account per unanswered request. The driver opens it at
//! [`RequestLedger::register`]; exactly one terminal transition closes it
//! ([`RequestLedger::finish`], [`RequestLedger::fail`],
//! [`RequestLedger::reject`], [`RequestLedger::retire`], or
//! [`RequestLedger::defer_finish`]); accounts still open when the ledger
//! drops are written off as `Failed` terminals (engine teardown). Because
//! every open is a submit and every close is a terminal, "terminal exactly
//! once" and "nothing after the terminal" are enforced here: touching a
//! closed or unknown account panics at the offending scheduler call site.
//!
//! Token counts on terminals derive from the ledger's own tally
//! ([`RequestLedger::push_tokens`]), never from model-side arithmetic.
//!
//! Scheduler-facing methods are `pub`. Opening accounts and shipping the
//! statement ([`RequestLedger::register`], [`RequestLedger::commit_step`],
//! [`RequestLedger::fail_all`]) are `pub(crate)`: the driver owns the
//! cadence, and model crates cannot reach them.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::event::FinishReason;
use super::event::TokenLogprob;
use super::request_lifecycle::DeferredFinish;
use super::request_lifecycle::StepSender;
use super::request_lifecycle::RequestEnvelope;
use super::step::PromptEcho;
use super::step::RejectReason;
use super::step::RequestId;
use super::step::RequestUpdate;
use super::step::ScheduledInfo;
use super::step::StepOutputs;
use super::step::QueuedRequest;
use super::step::Terminal;

/// One open account: the request's admission facts and running tally. The
/// payload is not here — it went to the scheduler at `submit`; the account is
/// what the engine still owes the client.
struct Account {
    abort: Arc<AtomicBool>,
    prompt_len: usize,
    queued_at: Instant,
    state: AccountState,
}

enum AccountState {
    /// Submitted, not yet admitted.
    Queued,
    /// Admitted; tokens tallied as they are pushed.
    Active { completion_tokens: usize },
}

impl Account {
    fn completion_tokens(&self) -> usize {
        match self.state {
            AccountState::Queued => 0,
            AccountState::Active { completion_tokens } => completion_tokens,
        }
    }
}

/// The current step's outgoing updates, merged per request so one step ships
/// at most one record per id.
#[derive(Default)]
struct StepStatement {
    buffer: Vec<Option<RequestUpdate>>,
    index: HashMap<RequestId, usize>,
}

impl StepStatement {
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

    fn take_updates(&mut self) -> Vec<RequestUpdate> {
        self.index.clear();
        self.buffer
            .drain(..)
            .flatten()
            .filter(|update| !update.is_vacant())
            .collect()
    }
}

pub struct RequestLedger {
    accounts: HashMap<RequestId, Account>,
    statement: StepStatement,
    tx: StepSender,
}

impl RequestLedger {
    pub(crate) fn new(tx: StepSender) -> Self {
        Self {
            accounts: HashMap::new(),
            statement: StepStatement::default(),
            tx,
        }
    }

    fn account(&self, id: RequestId) -> &Account {
        self.accounts
            .get(&id)
            .unwrap_or_else(|| panic!("no open account for {id}: already answered, or never registered"))
    }

    fn close(&mut self, id: RequestId) -> Account {
        self.accounts
            .remove(&id)
            .unwrap_or_else(|| panic!("no open account for {id}: already answered, or never registered"))
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// The frontend stopped wanting this request. The scheduler answers by
    /// [`Self::retire`] on its next touch; abort can land at any moment, so
    /// the probe belongs on every finish path too.
    pub fn is_aborted(&self, id: RequestId) -> bool {
        self.account(id).abort.load(Ordering::Acquire)
    }

    /// Whether the account is open and admitted. For defensive paths that
    /// resolve effects for requests which may have been answered earlier in
    /// the same step (batch failure, duplicate effects).
    pub fn is_active(&self, id: RequestId) -> bool {
        matches!(
            self.accounts.get(&id),
            Some(Account {
                state: AccountState::Active { .. },
                ..
            })
        )
    }

    /// Tokens shipped so far, from the ledger's tally.
    pub fn completion_tokens(&self, id: RequestId) -> usize {
        self.account(id).completion_tokens()
    }

    // ── Queued-state exits ──────────────────────────────────────────────

    /// Admit the request: stamp `scheduled_at` and buffer the admission facts.
    pub fn admit(&mut self, id: RequestId) {
        let account = self
            .accounts
            .get_mut(&id)
            .unwrap_or_else(|| panic!("no open account for {id}: already answered, or never registered"));
        assert!(
            matches!(account.state, AccountState::Queued),
            "admit on already-admitted request {id}"
        );
        account.state = AccountState::Active {
            completion_tokens: 0,
        };
        let scheduled = ScheduledInfo {
            queued_at: account.queued_at,
            scheduled_at: Instant::now(),
            prompt_tokens: account.prompt_len,
        };
        self.statement.entry(id).scheduled = Some(scheduled);
    }

    /// Refuse the request at admission.
    pub fn reject(&mut self, id: RequestId, reason: RejectReason) {
        let account = self.close(id);
        assert!(
            matches!(account.state, AccountState::Queued),
            "reject after admission for {id}: an admitted request finishes or fails"
        );
        self.statement.entry(id).terminal = Some(Terminal::Rejected {
            reason,
            prompt_tokens: account.prompt_len,
        });
    }

    // ── Streaming ───────────────────────────────────────────────────────

    /// Append committed tokens. `logprobs` must be empty (none requested) or
    /// parallel to `ids`; the statement keeps the two aligned across pushes.
    pub fn push_tokens(&mut self, id: RequestId, ids: &[u32], logprobs: &[Option<TokenLogprob>]) {
        assert!(
            logprobs.is_empty() || logprobs.len() == ids.len(),
            "logprobs must be absent or parallel to tokens"
        );
        let account = self
            .accounts
            .get_mut(&id)
            .unwrap_or_else(|| panic!("no open account for {id}: already answered, or never registered"));
        let AccountState::Active { completion_tokens } = &mut account.state else {
            panic!("push_tokens on {id} before admission");
        };
        *completion_tokens += ids.len();
        let entry = self.statement.entry(id);
        entry.tokens.extend_from_slice(ids);
        if logprobs.is_empty() {
            entry.logprobs.extend(ids.iter().map(|_| None));
        } else {
            entry.logprobs.extend_from_slice(logprobs);
        }
    }

    /// Report the prefix-cache hit count, in the step the scheduler learns it.
    pub fn set_cached_tokens(&mut self, id: RequestId, cached_tokens: usize) {
        assert!(self.is_active(id), "set_cached_tokens on {id} before admission");
        self.statement.entry(id).cached_tokens = Some(cached_tokens);
    }

    /// Echo the prompt back (echo mode), once, when prefill completes.
    pub fn echo_prompt(&mut self, id: RequestId, echo: PromptEcho) {
        assert!(self.is_active(id), "echo_prompt on {id} before admission");
        self.statement.entry(id).prompt_echo = Some(echo);
    }

    /// Attach P/D handoff metadata to this step's record.
    pub fn kv_transfer(&mut self, id: RequestId, params: serde_json::Value) {
        assert!(self.is_active(id), "kv_transfer on {id} before admission");
        self.statement.entry(id).kv_transfer = Some(params);
    }

    // ── Terminal transitions ────────────────────────────────────────────

    /// Finish the request. Token counts come from the ledger's tally.
    pub fn finish(&mut self, id: RequestId, reason: FinishReason) {
        let account = self.close(id);
        let AccountState::Active { completion_tokens } = account.state else {
            panic!("finish on {id} before admission");
        };
        self.statement.entry(id).terminal = Some(Terminal::Finished {
            reason,
            prompt_tokens: account.prompt_len,
            completion_tokens,
        });
    }

    /// Fail the request with an engine-side error. Valid in both states — a
    /// queued request can die to an engine error before admission.
    pub fn fail(&mut self, id: RequestId, message: impl Into<String>) {
        let account = self.close(id);
        self.statement.entry(id).terminal = Some(Terminal::Failed {
            message: message.into(),
            prompt_tokens: account.prompt_len,
            completion_tokens: account.completion_tokens(),
        });
    }

    /// Close an aborted request's account (see [`Self::is_aborted`]). Silent,
    /// and discards anything buffered for it this step: the frontend already
    /// dropped its state for this id, so there is no one to address.
    pub fn retire(&mut self, id: RequestId) {
        let account = self.close(id);
        log::debug!(
            "request retired: frontend aborted: {id} tokens_shipped={}",
            account.completion_tokens()
        );
        self.statement.extract(id);
    }

    /// Turn the request's finish into a token the scheduler can deliver later
    /// from any thread (P/D prefill roles gate `Finished` on KV-save
    /// visibility). Closes the account; the request's buffered update for
    /// this step — tokens included — folds into the returned message, so late
    /// delivery cannot reorder against the step stream.
    pub fn defer_finish(&mut self, id: RequestId, reason: FinishReason) -> DeferredFinish {
        let account = self.close(id);
        let AccountState::Active { completion_tokens } = account.state else {
            panic!("defer_finish on {id} before admission");
        };
        let mut update = self
            .statement
            .extract(id)
            .unwrap_or_else(|| RequestUpdate::empty(id));
        update.terminal = Some(Terminal::Finished {
            reason,
            prompt_tokens: account.prompt_len,
            completion_tokens,
        });
        DeferredFinish::new(update, self.tx.clone())
    }

    // ── Driver face ─────────────────────────────────────────────────────

    /// Open the account and hand back the [`QueuedRequest`] for
    /// [`super::Scheduler::submit`]. Consumes (and disarms) the envelope: from
    /// here on, the account carries the answer-on-drop duty.
    pub(crate) fn register(&mut self, envelope: RequestEnvelope) -> QueuedRequest {
        let inner = envelope.consume();
        let account = Account {
            abort: inner.abort,
            prompt_len: inner.request.prompt_tokens.len(),
            queued_at: inner.queued_at,
            state: AccountState::Queued,
        };
        let previous = self.accounts.insert(inner.id, account);
        assert!(previous.is_none(), "duplicate request id {}", inner.id);
        QueuedRequest {
            id: inner.id,
            request: inner.request,
        }
    }

    /// Ship the step's statement as one message; a step that touched nothing
    /// ships nothing. Called once per driver iteration — model code never
    /// calls this (the driver owns the cadence).
    pub(crate) fn commit_step(&mut self) {
        let updates = self.statement.take_updates();
        if updates.is_empty() {
            return;
        }
        // A closed receiver means the frontend is gone; the driver notices
        // through the submission channel and winds down.
        let _ = self.tx.send(StepOutputs { updates });
    }

    /// Write off every open account with one message — the driver's teardown
    /// sweep after a fatal `step` error, carrying the real error instead of
    /// the drop sweep's generic one.
    pub(crate) fn fail_all(&mut self, message: &str) {
        let ids: Vec<RequestId> = self.accounts.keys().copied().collect();
        for id in ids {
            self.fail(id, message);
        }
    }
}

/// The teardown sweep: accounts still open when the ledger falls are engine
/// bugs or teardown races, and each must surface as a finished stream instead
/// of a client hang.
impl Drop for RequestLedger {
    fn drop(&mut self) {
        if self.accounts.is_empty() {
            return;
        }
        let updates = self
            .accounts
            .drain()
            .map(|(id, account)| {
                let message = match account.state {
                    AccountState::Queued => "request dropped by the engine before it was answered",
                    AccountState::Active { .. } => "request dropped by the engine mid-stream",
                };
                let mut update = RequestUpdate::empty(id);
                update.terminal = Some(Terminal::Failed {
                    message: message.to_string(),
                    prompt_tokens: account.prompt_len,
                    completion_tokens: account.completion_tokens(),
                });
                update
            })
            .collect();
        let _ = self.tx.send(StepOutputs { updates });
    }
}

#[cfg(test)]
mod tests {
    use super::super::request_lifecycle::StepReceiver;
    use super::super::step::Request;
    use super::super::step::Terminal;
    use super::super::wiring::SchedulerHandle;
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
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;

        backend.ledger.admit(id);
        backend.ledger.push_tokens(id, &[10, 11], &[]);
        backend.ledger.set_cached_tokens(id, 2);
        backend.ledger.finish(id, FinishReason::Stop);
        backend.ledger.commit_step();

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
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;
        backend.ledger.reject(
            id,
            RejectReason::ContextLength {
                prompt_tokens: 5,
                max_tokens: 8,
                limit: 4,
            },
        );
        backend.ledger.commit_step();

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
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;
        backend.ledger.admit(id);
        backend.ledger.push_tokens(id, &[9], &[]);
        backend.ledger.retire(id);
        backend.ledger.commit_step();

        // Scheduled was buffered before the retire extracted the entry, so
        // nothing observable remains this step.
        assert!(handle_steps(handle).try_recv().is_err());
    }

    #[test]
    fn defer_finish_folds_step_output_and_delivers_late() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1, 2]));
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;
        backend.ledger.admit(id);
        backend.ledger.push_tokens(id, &[7], &[]);
        let deferred = backend.ledger.defer_finish(id, FinishReason::Length);
        backend.ledger.commit_step();

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
    fn dropped_ledger_writes_off_open_accounts() {
        let (handle, mut backend) = scheduler_pair();
        let _c0 = handle.submit(request(vec![1]));
        let _c1 = handle.submit(request(vec![2, 3]));
        let envelope = backend.submissions.try_recv().expect("envelope 0");
        let _queued = backend.ledger.register(envelope).id;
        let envelope = backend.submissions.try_recv().expect("envelope 1");
        let admitted = backend.ledger.register(envelope).id;
        backend.ledger.admit(admitted);

        // Ledger falls with both accounts open: one queued, one active.
        drop(backend);

        let mut steps = handle_steps(handle);
        let step = steps.try_recv().expect("write-off message");
        let mut prompts: Vec<usize> = step
            .updates
            .iter()
            .map(|update| match update.terminal {
                Some(Terminal::Failed { prompt_tokens, .. }) => prompt_tokens,
                ref other => panic!("write-off must be Failed, got {other:?}"),
            })
            .collect();
        prompts.sort_unstable();
        assert_eq!(prompts, vec![1, 2]);
    }

    #[test]
    fn abort_flag_is_visible_through_both_states() {
        let (handle, mut backend) = scheduler_pair();
        let control = handle.submit(request(vec![1]));
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;
        assert!(!backend.ledger.is_aborted(id));

        control.abort();
        assert!(backend.ledger.is_aborted(id));
        backend.ledger.admit(id);
        assert!(backend.ledger.is_aborted(id));
        backend.ledger.retire(id);
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

    #[test]
    #[should_panic(expected = "no open account")]
    fn touching_a_closed_account_panics() {
        let (handle, mut backend) = scheduler_pair();
        let _control = handle.submit(request(vec![1]));
        let envelope = backend.submissions.try_recv().expect("envelope");
        let id = backend.ledger.register(envelope).id;
        backend.ledger.admit(id);
        backend.ledger.finish(id, FinishReason::Stop);
        backend.ledger.push_tokens(id, &[1], &[]);
    }

    fn handle_steps(mut handle: SchedulerHandle) -> StepReceiver {
        handle.take_steps().expect("step stream not yet taken")
    }
}
