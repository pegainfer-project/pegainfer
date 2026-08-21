//! Rank-local scheduler load snapshot exported to the frontend.

use std::collections::VecDeque;

use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_kv_store::BlockPool;
use tokio::sync::watch;

use super::RankSlots;

/// Publish this rank's truthful scheduler snapshot. Cached pages that kvbm
/// can evict count as available; the reserved padding page is excluded from
/// both used and total. The frontend scores this rank's engine by it, and an
/// unbound request's least-load placement (in `EngineHandle::submit`) reads
/// the same numbers.
pub(super) fn publish_load(
    load_tx: &watch::Sender<SchedulerMetrics>,
    pool: &BlockPool,
    slots: &RankSlots,
    pending: &VecDeque<super::offload::Resolved>,
    resolving: usize,
) {
    let kv_total_blocks = pool.total_blocks() - 1;
    load_tx.send_replace(SchedulerMetrics {
        kv_used_blocks: kv_total_blocks.saturating_sub(pool.available_blocks()) as u64,
        kv_total_blocks: kv_total_blocks as u64,
        num_running_reqs: slots.iter().flatten().count() as u64,
        // Every admitted-but-not-yet-slotted request counts as waiting for
        // the whole intake-to-slot window, whether it is still resolving
        // off-thread or already queued: the engine's drain decrements the
        // resolver count in the same breath it pushes into the deque, so
        // the frontend's placement signal never undercounts a rank
        // mid-resolve and never counts a request twice.
        num_waiting_reqs: (pending.len() + resolving) as u64,
        spec_decode: None,
    });
}
