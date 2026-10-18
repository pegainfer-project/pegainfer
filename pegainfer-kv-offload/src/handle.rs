//! Pollable handles for I/O submitted to pegaflow's runtime.
//!
//! Every offload operation the scheduler cares about resolves through one
//! [`OffloadHandle`]: submit on the engine, then poll at step boundaries —
//! never block the serving loop (#799/#802). Dropping a handle detaches the
//! observer, not the I/O: the underlying task keeps running (and keeps its
//! keep-alive payload) until it settles on its own.

use pegaflow_core::EngineError;
use pegaflow_core::QueryLeaseId;
use tokio::sync::oneshot;

/// In-flight handle for one operation submitted to pegaflow's worker.
///
/// [`Self::poll`] keeps scheduler admission non-blocking; [`Self::wait`]
/// blocks for tests and non-pipelined callers.
pub struct OffloadHandle<T> {
    rx: oneshot::Receiver<Result<T, EngineError>>,
}

/// CPU→GPU load; resolves when the DMA completes.
pub type LoadHandle = OffloadHandle<()>;

/// GPU→CPU save; resolves when the host tier has captured the data (the
/// source GPU blocks are reusable from that point).
pub(crate) type SaveHandle = OffloadHandle<()>;

/// Prefix query; resolves to a [`QueryOutcome`] once pegaflow has counted the
/// local hit and — for `Loading` outcomes — leaves the background fetch
/// running (re-submit with the same `req_id` to poll it).
pub(crate) type QueryHandle = OffloadHandle<QueryOutcome>;

impl<T> OffloadHandle<T> {
    pub(crate) fn from_rx(rx: oneshot::Receiver<Result<T, EngineError>>) -> Self {
        Self { rx }
    }

    /// Non-blocking check for a scheduler tick. `None` while still in flight.
    pub fn poll(&mut self) -> Option<Result<T, EngineError>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err(EngineError::Storage(
                "offload worker dropped reply".into(),
            ))),
        }
    }

    /// Block the current thread until the operation settles.
    pub fn wait(self) -> Result<T, EngineError> {
        self.rx
            .blocking_recv()
            .unwrap_or_else(|_| Err(EngineError::Storage("offload worker dropped reply".into())))
    }

    /// Await settlement from an async context ([`Self::poll`]/[`Self::wait`]
    /// are the duals for the synchronous scheduler thread).
    pub async fn settle(self) -> Result<T, EngineError> {
        self.rx
            .await
            .unwrap_or_else(|_| Err(EngineError::Storage("offload worker dropped reply".into())))
    }

    /// A handle that is already settled with `result` — lets scheduler tests
    /// drive their admission poll path without a pegaflow worker.
    pub(crate) fn settled(result: Result<T, EngineError>) -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(result);
        Self { rx }
    }

    /// A handle plus the sender that settles it — scheduler tests exercise
    /// the parked/in-flight path, then resolve it on their own schedule.
    pub fn in_flight() -> (Self, oneshot::Sender<Result<T, EngineError>>) {
        let (tx, rx) = oneshot::channel();
        (Self { rx }, tx)
    }
}

/// A query hit: how many prefix blocks pegaflow can return from its CPU tier,
/// and the lease that owns those blocks until `OffloadEngine::load` consumes
/// it. `num_blocks == 0` means a full miss and `lease` is `None`.
pub struct QueryHit {
    pub lease: Option<QueryLeaseId>,
    pub num_blocks: usize,
}

/// Outcome of an `OffloadEngine` query.
pub enum QueryOutcome {
    /// Terminal: `hit.num_blocks` prefix blocks are host-resident and leased.
    Ready(QueryHit),
    /// pegaflow kicked off an async fetch of the missing prefix from a remote
    /// peer (P2P) or SSD. Not terminal: re-query with the same `req_id` next
    /// tick to poll; the fetch resolves to `Ready` (with the pulled blocks) or
    /// falls back to a plain local hit count. Only occurs with a deeper tier
    /// configured — never in the host-memory-only setup.
    Loading,
}
