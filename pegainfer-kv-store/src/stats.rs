//! Terminal-state accounting for every async KV operation.
//!
//! The design contract (`docs/subsystems/kv-cache/design.md`): every resolve
//! and save reports exactly one terminal — success, degraded, or failed —
//! never a silent drop. Counters are the aggregation sink; the log lines are
//! the print sink; tracing spans hang off the same call sites later.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// Why a resolve returned less than the tier could theoretically serve.
/// Observability-only: the caller sees just a smaller `hit_tokens`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DegradeReason {
    /// The request was cancelled mid-resolve; remaining I/O was skipped.
    Cancelled,
    /// The tier stayed `Loading` past the resolve deadline.
    DeadlineExceeded,
    /// The tier query or load failed.
    TierError,
    /// The host hit's destination pages never fit the pool within the whole
    /// resolve deadline (the resolve waits for the pool before degrading —
    /// pressure that clears in time is not a degrade).
    PoolPressure,
}

#[derive(Debug, Default)]
pub struct KvStoreStats {
    pub(crate) resolves: AtomicU64,
    /// Resolves that returned with a hold (any hit at all, GPU or host).
    pub(crate) resolve_hits: AtomicU64,
    /// Host-tier blocks actually loaded onto the GPU.
    pub resolve_loaded_blocks: AtomicU64,
    pub resolve_degraded: AtomicU64,
    /// Load waits abandoned at the deadline: the reservation stays with the
    /// detached task, and if the DMA never settles those blocks never return
    /// — this counter is how pool-drain from hung DMAs stays visible.
    pub loads_abandoned: AtomicU64,
    pub saves_submitted: AtomicU64,
    pub saves_failed: AtomicU64,
    /// Handoff-class saves that settled with an error: the checkpoint the
    /// consuming peer expects is missing, observable there as a short hit.
    pub handoff_failed: AtomicU64,
    pub retires_parked: AtomicU64,
}

impl KvStoreStats {
    pub(crate) fn record_degrade(&self, req_id: &str, reason: DegradeReason) {
        self.resolve_degraded.fetch_add(1, Ordering::Relaxed);
        match reason {
            // Cancellation is a normal ending, not a fault.
            DegradeReason::Cancelled => {
                log::debug!("kv-store resolve {req_id}: cancelled, resources released");
            }
            reason => {
                log::warn!("kv-store resolve {req_id}: degraded ({reason:?})");
            }
        }
    }
}
