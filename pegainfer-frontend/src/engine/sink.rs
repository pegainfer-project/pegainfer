use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::event::RequestTag;
use super::event::TokenEvent;

/// The single output channel an engine dispatches *all* requests' token events
/// into, each tagged with its [`RequestTag`]. One receiver (the frontend demux
/// loop) drains it, replacing the former per-request fan-out of N channels and
/// N consumer tasks — N distinct sleeping consumers cost N wakeups per step,
/// one shared consumer costs ~1.
pub type TokenStreamSender = mpsc::UnboundedSender<(RequestTag, TokenEvent)>;
pub type TokenStreamReceiver = mpsc::UnboundedReceiver<(RequestTag, TokenEvent)>;

/// Per-request handle the scheduler holds to emit [`TokenEvent`]s.
///
/// Drop-in for the former `UnboundedSender<TokenEvent>`: it keeps the same
/// `send` / `is_closed` / `Clone` surface, so scheduler call sites are
/// unchanged. Internally each event is tagged with the request's
/// [`RequestTag`] and pushed onto one shared [`TokenStreamSender`].
///
/// Cancellation moved from "drop the per-request receiver" to a shared abort
/// reason: the frontend aborts a *single* request by setting its reason without
/// closing the channel the other requests still use. `send` and `is_closed`
/// then report that request as gone, so the scheduler retires it on its next
/// emit — the same *reactive* retirement the old consumer-drop gave, reached
/// through the reason rather than channel closure. `tx.is_closed()` is the
/// engine-wide signal (the whole demux is gone); the per-request signal is the
/// abort reason. The reason is set with `Release` and read with `Acquire` so
/// the abort is ordered against the frontend dropping the request's stream
/// state.
#[derive(Clone)]
pub struct TokenSink {
    tag: RequestTag,
    tx: TokenStreamSender,
    abort_reason: Arc<AtomicU8>,
}

impl TokenSink {
    pub fn new(tag: RequestTag, tx: TokenStreamSender, abort_reason: Arc<AtomicU8>) -> Self {
        Self {
            tag,
            tx,
            abort_reason,
        }
    }

    /// Emit one event for this request. Returns `Err` (handing the event back)
    /// when the request was aborted or the shared receiver is gone — both of
    /// which the scheduler reads as "consumer dropped, retire the request",
    /// the same contract as the old per-request channel.
    #[allow(clippy::result_large_err)]
    pub fn send(&self, event: TokenEvent) -> Result<(), mpsc::error::SendError<TokenEvent>> {
        if self.abort_reason() != RequestAbortReason::None {
            return Err(mpsc::error::SendError(event));
        }
        self.tx.send((self.tag.clone(), event)).map_err(|err| {
            let (_, event) = err.0;
            mpsc::error::SendError(event)
        })
    }

    /// `true` once the request is aborted or the shared receiver is gone.
    pub fn is_closed(&self) -> bool {
        self.abort_reason() != RequestAbortReason::None || self.tx.is_closed()
    }

    /// `true` once the frontend explicitly cancelled this request after the
    /// stream had already started.
    pub fn is_cancelled(&self) -> bool {
        self.abort_reason() == RequestAbortReason::Cancelled
    }

    /// `true` once the frontend observed a client disconnect before the first
    /// response chunk for this request reached the client.
    pub fn is_disconnected(&self) -> bool {
        self.abort_reason() == RequestAbortReason::Disconnected
    }

    /// Current per-request abort reason.
    fn abort_reason(&self) -> RequestAbortReason {
        RequestAbortReason::from_raw(self.abort_reason.load(Ordering::Acquire))
    }

    /// The request id this sink tags its events with.
    pub fn tag(&self) -> &RequestTag {
        &self.tag
    }

    /// A sink backed by its own private channel, for direct drivers
    /// (benchmarks, integration tests, the simulator) that consume one
    /// request's events without the shared frontend demux. The returned
    /// receiver yields the tagged events; the cancel flag is never tripped.
    pub fn standalone() -> (Self, TokenStreamReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = Self::new(
            Arc::from("local"),
            tx,
            Arc::new(AtomicU8::new(RequestAbortReason::None as u8)),
        );
        (sink, rx)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RequestAbortReason {
    None = 0,
    Cancelled = 1,
    Disconnected = 2,
}

impl RequestAbortReason {
    pub(crate) fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::Cancelled,
            2 => Self::Disconnected,
            _ => Self::None,
        }
    }

    pub(crate) fn store(self, abort_reason: &AtomicU8) {
        abort_reason.store(self as u8, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_sink_distinguishes_cancelled_from_closed_receiver() {
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sink = TokenSink::new(Arc::from("request-a"), tx, Arc::clone(&abort_reason));

        assert!(!sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(!sink.is_closed());
        sink.send(TokenEvent::Token {
            id: 7,
            logprob: None,
        })
        .expect("uncancelled sink should send");
        assert_eq!(rx.try_recv().expect("tagged event").0.as_ref(), "request-a");

        RequestAbortReason::Cancelled.store(&abort_reason);
        assert!(sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 8,
                logprob: None,
            })
            .is_err()
        );
    }

    #[test]
    fn token_sink_closed_receiver_is_not_explicit_cancel() {
        let (sink, rx) = TokenSink::standalone();

        drop(rx);

        assert!(!sink.is_cancelled());
        assert!(!sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 7,
                logprob: None,
            })
            .is_err()
        );
    }

    #[test]
    fn token_sink_distinguishes_disconnected_from_cancelled() {
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let (tx, _rx) = mpsc::unbounded_channel();
        let sink = TokenSink::new(Arc::from("request-a"), tx, Arc::clone(&abort_reason));

        RequestAbortReason::Disconnected.store(&abort_reason);

        assert!(!sink.is_cancelled());
        assert!(sink.is_disconnected());
        assert!(sink.is_closed());
        assert!(
            sink.send(TokenEvent::Token {
                id: 7,
                logprob: None,
            })
            .is_err()
        );
    }
}
