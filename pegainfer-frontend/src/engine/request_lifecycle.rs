//! Typestate handles for one request's lifetime on the scheduler side.
//!
//! A request is a two-state machine encoded as two owned types, so the event
//! protocol is enforced by move semantics instead of convention:
//!
//! ```text
//! QueuedRequest ──admit──▶ ActiveRequest ──finish/fail/defer──▶ consumed
//!       │                        │
//!       └──reject/retire─────────┴──retire──▶ consumed (silent)
//! ```
//!
//! Every transition consumes the handle, so "terminal exactly once" and
//! "nothing after the terminal" cannot be miscoded. A handle dropped without a
//! transition is a scheduler bug; its `Drop` emits a `Failed` terminal so the
//! bug surfaces as a finished stream instead of a client hang. That drop bomb
//! is also what answers every in-flight request when a scheduler aborts: the
//! driver drops the scheduler, the handles fall, the terminals ship.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::step::Request;
use super::step::RequestId;
use super::step::RequestUpdate;
use super::step::StepOutputs;
use super::step::Terminal;

/// The scheduler-to-frontend stream: one message per step. Handles hold a
/// clone only for the drop bomb and for deferred finishes; ordinary emission
/// goes through the emitter's per-step buffer.
pub(crate) type StepSender = tokio::sync::mpsc::UnboundedSender<StepOutputs>;
pub type StepReceiver = tokio::sync::mpsc::UnboundedReceiver<StepOutputs>;

pub(crate) struct HandleCore {
    pub(crate) id: RequestId,
    pub(crate) abort: Arc<AtomicBool>,
    pub(crate) tx: StepSender,
}

impl HandleCore {
    fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::Acquire)
    }

    /// Ship a lone update outside any step batch (drop bombs, deferred
    /// finishes). A closed receiver means the whole frontend is gone; nothing
    /// to do then.
    fn send_solo(&self, update: RequestUpdate) {
        let _ = self.tx.send(StepOutputs {
            updates: vec![update],
        });
    }
}

/// A submitted request the scheduler has not yet answered. Minted only by
/// [`super::SchedulerHandle::submit`]; consumed by exactly one of
/// [`super::StepEmitter::admit`], [`super::StepEmitter::reject`], or
/// [`super::StepEmitter::retire_queued`].
pub struct QueuedRequest {
    /// `Some` until a transition consumes the handle; `Drop` fires on `Some`.
    inner: Option<QueuedInner>,
}

pub(crate) struct QueuedInner {
    pub(crate) core: HandleCore,
    /// Boxed so a queued request stays pointer-small in scheduler registries;
    /// `None` after [`QueuedRequest::take_request`] moved the payload out.
    request: Option<Box<Request>>,
    /// Cached so admission facts and the drop bomb survive the payload move.
    pub(crate) prompt_len: usize,
    pub(crate) queued_at: Instant,
}

impl QueuedRequest {
    pub(crate) fn new(core: HandleCore, request: Request, queued_at: Instant) -> Self {
        Self {
            inner: Some(QueuedInner {
                core,
                prompt_len: request.prompt_tokens.len(),
                request: Some(Box::new(request)),
                queued_at,
            }),
        }
    }

    fn inner(&self) -> &QueuedInner {
        self.inner
            .as_ref()
            .expect("queued request accessed after consumption")
    }

    pub(crate) fn consume(mut self) -> QueuedInner {
        self.inner.take().expect("queued request consumed twice")
    }

    pub fn id(&self) -> RequestId {
        self.inner().core.id
    }

    /// Borrow the payload. Panics after [`Self::take_request`].
    pub fn request(&self) -> &Request {
        self.inner()
            .request
            .as_ref()
            .expect("request payload already taken")
    }

    /// Move the payload out, leaving a handle that is still the request's
    /// lifecycle handle (admit/reject/retire and the drop bomb keep working
    /// off the cached prompt length). `None` on a second call — schedulers
    /// that ingest the payload on submit take it exactly once.
    pub fn take_request(&mut self) -> Option<Request> {
        self.inner
            .as_mut()
            .expect("queued request accessed after consumption")
            .request
            .take()
            .map(|request| *request)
    }

    pub fn queued_at(&self) -> Instant {
        self.inner().queued_at
    }

    /// The frontend stopped wanting this request while it queued. The
    /// scheduler answers by [`super::StepEmitter::retire_queued`].
    pub fn is_aborted(&self) -> bool {
        self.inner().core.is_aborted()
    }
}

impl Drop for QueuedRequest {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let mut update = RequestUpdate::empty(inner.core.id);
        update.terminal = Some(Terminal::Failed {
            message: "request dropped by the engine before it was answered".to_string(),
            prompt_tokens: inner.prompt_len,
            completion_tokens: 0,
        });
        inner.core.send_solo(update);
    }
}

/// An admitted request occupying scheduler state. Pure capability plus
/// counters — no channel writes happen through it directly; the emitter is
/// the single writer of the step buffer, taking `&mut ActiveRequest` to keep
/// the counters honest. `Send`, so a scheduler may park it across threads
/// (deferred finishes hand it to [`super::StepEmitter::defer_finish`]).
pub struct ActiveRequest {
    inner: Option<ActiveInner>,
}

pub(crate) struct ActiveInner {
    pub(crate) core: HandleCore,
    pub(crate) prompt_tokens: usize,
    /// Tokens shipped so far, tallied by the emitter on every push. Terminal
    /// counts derive from this, never from model-side arithmetic.
    pub(crate) completion_tokens: usize,
}

impl ActiveRequest {
    pub(crate) fn new(core: HandleCore, prompt_tokens: usize) -> Self {
        Self {
            inner: Some(ActiveInner {
                core,
                prompt_tokens,
                completion_tokens: 0,
            }),
        }
    }

    pub(crate) fn inner(&self) -> &ActiveInner {
        self.inner
            .as_ref()
            .expect("active request accessed after consumption")
    }

    pub(crate) fn inner_mut(&mut self) -> &mut ActiveInner {
        self.inner
            .as_mut()
            .expect("active request accessed after consumption")
    }

    pub(crate) fn consume(mut self) -> ActiveInner {
        self.inner.take().expect("active request consumed twice")
    }

    pub fn id(&self) -> RequestId {
        self.inner().core.id
    }

    pub fn completion_tokens(&self) -> usize {
        self.inner().completion_tokens
    }

    /// The frontend stopped wanting this request's output. The scheduler
    /// answers by [`super::StepEmitter::retire`] on its next touch.
    pub fn is_aborted(&self) -> bool {
        self.inner().core.is_aborted()
    }
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let mut update = RequestUpdate::empty(inner.core.id);
        update.terminal = Some(Terminal::Failed {
            message: "request dropped by the engine mid-stream".to_string(),
            prompt_tokens: inner.prompt_tokens,
            completion_tokens: inner.completion_tokens,
        });
        inner.core.send_solo(update);
    }
}

/// A finish whose delivery the scheduler handed off (e.g. a P/D prefill role
/// withholding `Finished` until this step's KV saves are peer-visible). Made
/// by [`super::StepEmitter::defer_finish`], which folds the request's
/// already-buffered update — tokens included — into this message, so sending
/// it from any thread at any later time cannot reorder against the step
/// stream: the whole per-request record travels as one entry.
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

/// The same drop bomb as the other handles: an unsent deferred finish is a
/// holder bug, and it must surface as a finished stream, not a client hang.
/// The buffered tokens still ship, but the terminal is rewritten to `Failed`
/// — the withheld `Finished` doubles as a barrier signal (P/D KV-ready), so a
/// dropped handle must not fake its success.
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
/// mechanism: the scheduler observes it and retires the request reactively;
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
