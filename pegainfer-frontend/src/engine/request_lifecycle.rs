//! Submission envelope and late-delivery handles for one request's lifetime.
//!
//! The scheduler side of the contract deals in plain [`RequestId`]s against
//! the [`super::RequestLedger`] — no per-request typestate handles cross the
//! `Scheduler` trait. What lives here are the pieces that carry a request
//! *outside* the ledger's reach and therefore need their own answer-on-drop
//! guarantee:
//!
//! - [`RequestEnvelope`] — the channel envelope between
//!   [`super::SchedulerHandle::submit`] and the driver's
//!   [`super::RequestLedger::register`]. It covers the window where the
//!   request exists but no ledger account does yet: sitting in the submit
//!   channel, or sent to a scheduler thread that already exited.
//! - [`DeferredFinish`] — a finish extracted from the ledger for delivery
//!   from another thread at a later time (P/D prefill roles withhold
//!   `Finished` until the step's KV saves are peer-visible).
//! - [`RequestControl`] — the frontend's per-request abort flag.
//!
//! The first two hold `Option<inner>` for one reason only: Rust's `Drop`
//! cannot move fields out, so a consuming transition `take`s the inner and
//! the drop bomb fires on `Some`. An unanswered request must surface as a
//! `Failed` terminal, never a client hang.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::step::Request;
use super::step::RequestId;
use super::step::RequestUpdate;
use super::step::StepOutputs;
use super::step::Terminal;

/// The scheduler-to-frontend stream: one message per step. The ledger holds
/// the sender; drop bombs and deferred finishes hold clones for solo sends.
pub(crate) type StepSender = tokio::sync::mpsc::UnboundedSender<StepOutputs>;
pub type StepReceiver = tokio::sync::mpsc::UnboundedReceiver<StepOutputs>;

/// One submitted request in flight between the frontend and the driver.
/// Minted by [`super::SchedulerHandle::submit`]; consumed (and thereby
/// disarmed) by [`super::RequestLedger::register`], which opens the ledger
/// account that takes over the answer-on-drop duty.
pub(crate) struct RequestEnvelope {
    /// `Some` until [`Self::consume`]; `Drop` fires on `Some`.
    inner: Option<EnvelopeInner>,
}

pub(crate) struct EnvelopeInner {
    pub(crate) id: RequestId,
    pub(crate) abort: Arc<AtomicBool>,
    pub(crate) tx: StepSender,
    pub(crate) request: Request,
    pub(crate) queued_at: Instant,
}

impl RequestEnvelope {
    pub(crate) fn new(
        id: RequestId,
        abort: Arc<AtomicBool>,
        tx: StepSender,
        request: Request,
        queued_at: Instant,
    ) -> Self {
        Self {
            inner: Some(EnvelopeInner {
                id,
                abort,
                tx,
                request,
                queued_at,
            }),
        }
    }

    pub(crate) fn consume(mut self) -> EnvelopeInner {
        self.inner.take().expect("submission consumed twice")
    }
}

/// A submission dropped unconsumed never reached a ledger account: the
/// scheduler is gone (send failed, or the channel fell with the driver).
/// Answer the request so the client observes a terminal instead of a hang.
impl Drop for RequestEnvelope {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let mut update = RequestUpdate::empty(inner.id);
        update.terminal = Some(Terminal::Failed {
            message: "request dropped by the engine before it was answered".to_string(),
            prompt_tokens: inner.request.prompt_tokens.len(),
            completion_tokens: 0,
        });
        let _ = inner.tx.send(StepOutputs {
            updates: vec![update],
        });
    }
}

/// A finish whose delivery the scheduler handed off (e.g. a P/D prefill role
/// withholding `Finished` until this step's KV saves are peer-visible). Made
/// by [`super::RequestLedger::defer_finish`], which closes the request's
/// ledger account and folds its already-buffered step record — tokens
/// included — into this message, so sending it from any thread at any later
/// time cannot reorder against the step stream: the whole per-request record
/// travels as one entry.
pub struct DeferredFinish {
    /// `Some` until [`Self::send`] delivers; `Drop` fires on `Some`.
    inner: Option<DeferredInner>,
}

pub(crate) struct DeferredInner {
    pub(crate) update: RequestUpdate,
    pub(crate) tx: StepSender,
}

impl DeferredFinish {
    pub(crate) fn new(update: RequestUpdate, tx: StepSender) -> Self {
        Self {
            inner: Some(DeferredInner { update, tx }),
        }
    }

    fn inner(&self) -> &DeferredInner {
        self.inner
            .as_ref()
            .expect("deferred finish accessed after consumption")
    }

    pub fn id(&self) -> RequestId {
        self.inner().update.id
    }

    /// Deliver the terminal. Consumes the finish.
    pub fn send(mut self) {
        let inner = self.inner.take().expect("deferred finish sent twice");
        let _ = inner.tx.send(StepOutputs {
            updates: vec![inner.update],
        });
    }
}

/// The same drop bomb as the submission envelope: an unsent deferred finish
/// is a holder bug, and it must surface as a finished stream, not a client
/// hang. The buffered tokens still ship, but the terminal is rewritten to
/// `Failed` — the withheld `Finished` doubles as a barrier signal (P/D
/// KV-ready), so a dropped handle must not fake its success.
impl Drop for DeferredFinish {
    fn drop(&mut self) {
        let Some(mut inner) = self.inner.take() else {
            return;
        };
        let (prompt_tokens, completion_tokens) = match inner.update.terminal {
            Some(Terminal::Finished {
                prompt_tokens,
                completion_tokens,
                ..
            }) => (prompt_tokens, completion_tokens),
            _ => (0, inner.update.tokens.len()),
        };
        inner.update.terminal = Some(Terminal::Failed {
            message: "deferred finish dropped by the engine before delivery".to_string(),
            prompt_tokens,
            completion_tokens,
        });
        let _ = inner.tx.send(StepOutputs {
            updates: vec![inner.update],
        });
    }
}

/// The frontend's per-request cancel handle, returned by
/// [`super::SchedulerHandle::submit`]. Flipping the flag is the only abort
/// mechanism: the scheduler observes it through
/// [`super::RequestLedger::is_aborted`] and retires the request reactively;
/// no channel is closed.
pub struct RequestControl {
    id: RequestId,
    abort: Arc<AtomicBool>,
}

impl RequestControl {
    pub(crate) fn new(id: RequestId, abort: Arc<AtomicBool>) -> Self {
        Self { id, abort }
    }

    pub fn id(&self) -> RequestId {
        self.id
    }

    /// Mark the request aborted. `Release` orders the store after the
    /// caller's own teardown of per-request state, mirroring the scheduler's
    /// `Acquire` load.
    pub fn abort(&self) {
        self.abort.store(true, Ordering::Release);
    }
}
