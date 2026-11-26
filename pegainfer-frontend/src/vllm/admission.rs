//! Bounded admission: caps in-flight generation requests and returns 429/503 when the queue times out.
//!
//! Permits stay held until the response body ends, so slow clients cannot turn the queue into unbounded memory.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use axum::Json;
use axum::body::Body;
use axum::body::BodyDataStream;
use axum::body::Bytes;
use axum::extract::Request;
use axum::extract::State;
use axum::http::Method;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum::response::Response;
use futures::Stream;
use serde_json::json;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tokio::sync::TryAcquireError;

const MAX_ACTIVE_REQUESTS: usize = 256;
const MAX_WAITING_REQUESTS: usize = 256;
const ADMISSION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Reject {
    Full,
    Deadline,
    Draining,
}

pub(crate) struct Admission {
    slots: Arc<Semaphore>,
    waiting: AtomicUsize,
    accepting: AtomicBool,
    rejected_full: AtomicU64,
    rejected_deadline: AtomicU64,
    rejected_engine: AtomicU64,
}

impl Admission {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(MAX_ACTIVE_REQUESTS)),
            waiting: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
            rejected_full: AtomicU64::new(0),
            rejected_deadline: AtomicU64::new(0),
            rejected_engine: AtomicU64::new(0),
        })
    }

    pub(crate) fn active(&self) -> usize {
        MAX_ACTIVE_REQUESTS.saturating_sub(self.slots.available_permits())
    }

    pub(crate) fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    pub(crate) fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::Release);
        // Wake queued waiters now instead of letting them sit until the 30s
        // deadline; the drain must not keep clients queued behind it.
        self.slots.close();
    }

    pub(crate) fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    pub(crate) fn rejected_full(&self) -> u64 {
        self.rejected_full.load(Ordering::Acquire)
    }

    pub(crate) fn rejected_deadline(&self) -> u64 {
        self.rejected_deadline.load(Ordering::Acquire)
    }

    pub(crate) fn rejected_engine(&self) -> u64 {
        self.rejected_engine.load(Ordering::Acquire)
    }

    /// Reserve capacity for one engine request when the HTTP request expands
    /// into several of them (`prompt[]`, `n`, scoring items). The bridge calls
    /// this immediately before submitting; a full gate refuses only that
    /// engine request, so the scheduler never receives unbounded work.
    pub(crate) fn try_acquire(&self) -> Result<AdmissionPermit, Reject> {
        if !self.accepting.load(Ordering::Acquire) {
            return Err(Reject::Draining);
        }
        match self.slots.clone().try_acquire_owned() {
            Ok(permit) => Ok(AdmissionPermit { _permit: permit }),
            Err(TryAcquireError::NoPermits) => {
                self.rejected_engine.fetch_add(1, Ordering::AcqRel);
                Err(Reject::Full)
            }
            Err(TryAcquireError::Closed) => Err(Reject::Draining),
        }
    }

    async fn acquire(self: &Arc<Self>) -> Result<AdmissionPermit, Reject> {
        if !self.accepting.load(Ordering::Acquire) {
            return Err(Reject::Draining);
        }
        if self.waiting.load(Ordering::Acquire) == 0 {
            match self.slots.clone().try_acquire_owned() {
                Ok(permit) => return Ok(AdmissionPermit { _permit: permit }),
                Err(TryAcquireError::NoPermits) => {}
                Err(TryAcquireError::Closed) => return Err(Reject::Draining),
            }
        }

        if self.waiting.fetch_add(1, Ordering::AcqRel) >= MAX_WAITING_REQUESTS {
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            self.rejected_full.fetch_add(1, Ordering::AcqRel);
            return Err(Reject::Full);
        }
        // The client can disconnect while waiting; the guard releases the queue slot if the future is dropped.
        let waiting = WaitingGuard(&self.waiting);
        let acquired =
            tokio::time::timeout(ADMISSION_TIMEOUT, self.slots.clone().acquire_owned()).await;
        drop(waiting);
        match acquired {
            Ok(Ok(permit)) => Ok(AdmissionPermit { _permit: permit }),
            Ok(Err(_)) => Err(Reject::Draining),
            Err(_) => {
                self.rejected_deadline.fetch_add(1, Ordering::AcqRel);
                Err(Reject::Deadline)
            }
        }
    }
}

struct WaitingGuard<'a>(&'a AtomicUsize);

impl Drop for WaitingGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct AdmissionPermit {
    _permit: OwnedSemaphorePermit,
}

pub(crate) async fn admit(
    State(admission): State<Arc<Admission>>,
    request: Request,
    next: Next,
) -> Response {
    if !is_admitted(request.method(), request.uri().path()) {
        return next.run(request).await;
    }
    let permit = match admission.acquire().await {
        Ok(permit) => permit,
        Err(Reject::Full) => {
            return rejection(
                StatusCode::SERVICE_UNAVAILABLE,
                "admission queue is full; retry later",
                "ServiceUnavailable",
            );
        }
        Err(Reject::Deadline) => {
            return rejection(
                StatusCode::TOO_MANY_REQUESTS,
                "admission queue wait exceeded 30s; retry later",
                "TooManyRequests",
            );
        }
        Err(Reject::Draining) => {
            return rejection(
                StatusCode::SERVICE_UNAVAILABLE,
                "server is draining; retry later",
                "ServiceUnavailable",
            );
        }
    };

    let response = next.run(request).await;
    let (parts, body) = response.into_parts();
    let body = Body::from_stream(PermitStream {
        inner: body.into_data_stream(),
        _permit: permit,
    });
    Response::from_parts(parts, body)
}

fn is_admitted(method: &Method, path: &str) -> bool {
    *method == Method::POST && (path.starts_with("/v1/") || path == "/generative_scoring")
}

fn rejection(status: StatusCode, message: &str, kind: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": kind,
            }
        })),
    )
        .into_response()
}

struct PermitStream {
    inner: BodyDataStream,
    _permit: AdmissionPermit,
}

impl Stream for PermitStream {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::Admission;
    use super::MAX_ACTIVE_REQUESTS;
    use super::Reject;

    #[tokio::test]
    async fn cancelled_waiter_releases_queue_slot() {
        let admission = Admission::new();
        let mut held = Vec::new();
        for _ in 0..MAX_ACTIVE_REQUESTS {
            held.push(admission.acquire().await.expect("fast path acquire"));
        }
        assert_eq!(admission.active(), MAX_ACTIVE_REQUESTS);

        let cancelled = tokio::time::timeout(Duration::from_millis(20), admission.acquire()).await;
        assert!(cancelled.is_err());
        assert_eq!(admission.waiting(), 0);
    }

    #[tokio::test]
    async fn stop_accepting_wakes_waiters_with_draining() {
        let admission = Admission::new();
        let mut held = Vec::new();
        for _ in 0..MAX_ACTIVE_REQUESTS {
            held.push(admission.acquire().await.expect("fast path acquire"));
        }

        let waiter = {
            let admission = admission.clone();
            tokio::spawn(async move { admission.acquire().await })
        };
        tokio::task::yield_now().await;
        assert_eq!(admission.waiting(), 1);

        admission.stop_accepting();
        let result = waiter.await.expect("waiter task");
        assert!(matches!(result, Err(Reject::Draining)));
        assert_eq!(admission.waiting(), 0);
    }

    #[tokio::test]
    async fn engine_gate_refuses_past_capacity() {
        let admission = Admission::new();
        let mut held = Vec::new();
        for _ in 0..MAX_ACTIVE_REQUESTS {
            held.push(admission.try_acquire().expect("engine permit"));
        }
        assert!(matches!(admission.try_acquire(), Err(Reject::Full)));
        assert_eq!(admission.active(), MAX_ACTIVE_REQUESTS);
        assert_eq!(admission.rejected_engine(), 1);
    }

    #[tokio::test]
    async fn draining_gate_refuses_engine_work() {
        let admission = Admission::new();
        admission.stop_accepting();
        assert!(matches!(admission.try_acquire(), Err(Reject::Draining)));
    }

    #[test]
    fn scoring_path_is_gated() {
        use axum::http::Method;
        assert!(super::is_admitted(&Method::POST, "/generative_scoring"));
        assert!(super::is_admitted(&Method::POST, "/v1/completions"));
        assert!(!super::is_admitted(&Method::GET, "/v1/models"));
    }
}
