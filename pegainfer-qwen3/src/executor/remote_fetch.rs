//! The per-tick decision for a CPU-tier KV prefetch query.
//!
//! `begin_kv_prefetch` (first query) and `poll_remote_fetch` (re-query while
//! parked in [`RemoteFetch`](super::PrefetchPhase::RemoteFetch)) share one
//! cascade: still loading → wait; timeout / zero-hit / query error → prefill
//! from scratch; leased hit past the reservation budget → release the lease
//! and prefill from scratch; leased hit within budget → start the H2D load.
//! This module owns that cascade so the two call sites cannot drift, and so
//! the timeout / zero-hit / budget branches are testable without a GPU, an
//! RDMA fabric, or a MetaServer (only *injecting* a real `Loading` needs
//! those; deciding on one does not).
//!
//! The lease type is generic: every branch that receives a lease must route
//! it into the returned action, so a silently dropped (never-released) lease
//! is unrepresentable. Production monomorphizes `L` to pegaflow's opaque
//! `QueryLeaseId`; tests use plain integers.

/// A lease-carrying view of [`QueryOutcome`](pegainfer_kv_offload::QueryOutcome),
/// decoupled from pegaflow types so the decision stays constructible in tests.
///
/// `lease: None` only occurs together with `num_blocks == 0` (the engine
/// builds a lease for every non-empty hit); the pair is kept as-is rather
/// than re-encoded because the decision treats every `None` as a miss.
pub(super) enum QueryView<L> {
    Loading,
    Ready { lease: Option<L>, num_blocks: usize },
}

impl From<pegainfer_kv_offload::QueryOutcome> for QueryView<pegainfer_kv_offload::QueryLeaseId> {
    fn from(outcome: pegainfer_kv_offload::QueryOutcome) -> Self {
        match outcome {
            pegainfer_kv_offload::QueryOutcome::Loading => QueryView::Loading,
            pegainfer_kv_offload::QueryOutcome::Ready(hit) => QueryView::Ready {
                lease: hit.lease,
                num_blocks: hit.num_blocks,
            },
        }
    }
}

/// What to do with the request after one prefetch query tick.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum RemoteFetchAction<L> {
    /// Remote fetch still in flight: park (first tick) or stay parked.
    Wait,
    /// Give up the prefetch — deadline passed, zero-hit, or the query
    /// errored — and prefill from scratch. No lease was taken.
    Scratch,
    /// The hit is leased but reserving `num_blocks` would eat into blocks
    /// promised to admitted requests: release the lease, then prefill from
    /// scratch.
    Release(L),
    /// Leased hit within budget: reserve blocks and submit the H2D load.
    Load(L, usize),
}

/// Decide one prefetch query tick.
///
/// `query` runs only when the deadline has not passed — a parked request
/// whose deadline expired must not issue another query RPC — which is why
/// this takes a closure rather than a pre-computed outcome. A query `Err` is
/// already logged at the call site's level of context, so it folds into
/// [`RemoteFetchAction::Scratch`] here.
///
/// `wait_on_miss` is the P/D handoff race guard: when the prefill peer's KV
/// is *expected* (vLLM-prefill mode), a zero-hit means "the producer's save
/// or MetaServer registration hasn't landed yet", not "nobody has it" — so
/// the request stays parked and re-queries instead of prefilling from
/// scratch. The caller bounds it with a miss deadline (passing `false` once
/// that window closes); a query error still folds to `Scratch` — a broken
/// local engine won't heal by waiting.
///
/// `park_on_loading` is the same guard for the `Loading` answer: a query
/// only STARTS the async fetch, so in vLLM-compat mode the first shot is
/// always `Loading` — with the miss breaker open, parking on it would stall
/// every cold request for the full fetch deadline despite the breaker's
/// promise to prefill immediately. The abandoned fetch still lands in the
/// local host tier, so a later request's first-shot query can hit `Ready`
/// and re-arm waiting. Plain offload mode (no prefill peer) always parks.
pub(super) fn remote_fetch_action<L, E>(
    timed_out: bool,
    wait_on_miss: bool,
    park_on_loading: bool,
    query: impl FnOnce() -> Result<QueryView<L>, E>,
    available_blocks: usize,
    reserve_floor: usize,
) -> RemoteFetchAction<L> {
    if timed_out {
        return RemoteFetchAction::Scratch;
    }
    let (lease, num_blocks) = match query() {
        Ok(QueryView::Loading) => {
            return if park_on_loading {
                RemoteFetchAction::Wait
            } else {
                RemoteFetchAction::Scratch
            };
        }
        Ok(QueryView::Ready { lease, num_blocks }) => (lease, num_blocks),
        Err(_) => return RemoteFetchAction::Scratch,
    };
    let Some(lease) = lease else {
        return if wait_on_miss {
            RemoteFetchAction::Wait // producer's registration not visible yet
        } else {
            RemoteFetchAction::Scratch // miss
        };
    };
    // Blocks promised to admitted requests are off-limits: reserving into
    // them makes a later prefill chunk or decode growth fail allocation.
    if available_blocks.saturating_sub(reserve_floor) < num_blocks {
        return RemoteFetchAction::Release(lease);
    }
    RemoteFetchAction::Load(lease, num_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Action = RemoteFetchAction<u32>;

    fn ready(lease: Option<u32>, num_blocks: usize) -> QueryView<u32> {
        QueryView::Ready { lease, num_blocks }
    }

    /// Past the deadline the request prefills from scratch and, critically,
    /// no further query RPC is issued.
    #[test]
    fn timeout_gives_up_without_querying() {
        let action = remote_fetch_action::<u32, ()>(
            true,
            false,
            true,
            || unreachable!("post-deadline tick must not query"),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// A remote fetch still in flight keeps the request parked.
    #[test]
    fn loading_waits() {
        let action =
            remote_fetch_action::<u32, ()>(false, false, true, || Ok(QueryView::Loading), 0, 0);
        assert_eq!(action, Action::Wait);
    }

    /// Breaker open in vLLM-compat mode: `Loading` must not park — the
    /// first-shot query always answers `Loading` there, so parking would
    /// stall every cold request for the full fetch deadline.
    #[test]
    fn loading_scratches_when_parking_disabled() {
        let action =
            remote_fetch_action::<u32, ()>(false, false, false, || Ok(QueryView::Loading), 0, 0);
        assert_eq!(action, Action::Scratch);
    }

    /// Zero-hit (peer evicted the blocks, or no owner): no lease exists, so
    /// the request just prefills from scratch.
    #[test]
    fn zero_hit_prefills_from_scratch() {
        let action = remote_fetch_action(
            false,
            false,
            true,
            || Ok::<_, ()>(ready(None, 0)),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// A query error folds into prefill-from-scratch.
    #[test]
    fn query_error_prefills_from_scratch() {
        let action = remote_fetch_action::<u32, &str>(
            false,
            false,
            true,
            || Err("rpc failed"),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// A leased hit that would eat into promised blocks routes the lease out
    /// for release — the type makes dropping it silently unrepresentable.
    #[test]
    fn budget_guard_releases_the_lease() {
        let action = remote_fetch_action(
            false,
            false,
            true,
            || Ok::<_, ()>(ready(Some(7), 10)),
            12,
            3,
        );
        assert_eq!(action, Action::Release(7));
    }

    /// `reserve_floor > available_blocks` must saturate, not underflow.
    #[test]
    fn budget_guard_saturates_when_floor_exceeds_available() {
        let action =
            remote_fetch_action(false, false, true, || Ok::<_, ()>(ready(Some(7), 1)), 2, 5);
        assert_eq!(action, Action::Release(7));
    }

    /// Exactly-at-budget reserves and loads: the guard is strict-less-than.
    #[test]
    fn exact_budget_boundary_loads() {
        let action =
            remote_fetch_action(false, false, true, || Ok::<_, ()>(ready(Some(7), 9)), 12, 3);
        assert_eq!(action, Action::Load(7, 9));
    }

    /// The normal path: leased hit within budget starts the H2D load.
    #[test]
    fn leased_hit_within_budget_loads() {
        let action = remote_fetch_action(
            false,
            false,
            true,
            || Ok::<_, ()>(ready(Some(42), 4)),
            64,
            8,
        );
        assert_eq!(action, Action::Load(42, 4));
    }

    /// P/D handoff race: a zero-hit with remote KV expected keeps the request
    /// parked — the producer's registration hasn't landed yet.
    #[test]
    fn expected_remote_miss_waits() {
        let action = remote_fetch_action(
            false,
            true,
            true,
            || Ok::<_, ()>(ready(None, 0)),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Wait);
    }

    /// Once the miss window closes the caller passes `wait_on_miss = false`
    /// and a still-missing prefix degrades to prefill-from-scratch.
    #[test]
    fn expected_remote_miss_window_closed_prefills_from_scratch() {
        let action = remote_fetch_action(
            false,
            false,
            true,
            || Ok::<_, ()>(ready(None, 0)),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// `wait_on_miss` guards only the miss branch: a query error is a broken
    /// local engine, not a registration race, and never waits.
    #[test]
    fn query_error_never_waits_even_when_remote_expected() {
        let action = remote_fetch_action::<u32, &str>(
            false,
            true,
            true,
            || Err("rpc failed"),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }

    /// `wait_on_miss` does not bypass the hard deadline.
    #[test]
    fn timeout_overrides_wait_on_miss() {
        let action = remote_fetch_action::<u32, ()>(
            true,
            true,
            true,
            || unreachable!("post-deadline tick must not query"),
            usize::MAX,
            0,
        );
        assert_eq!(action, Action::Scratch);
    }
}
