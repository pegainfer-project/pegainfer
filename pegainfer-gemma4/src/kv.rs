//! Per-request KV state across the two families, and the admission that
//! keeps their pools consistent.

use std::collections::VecDeque;

use anyhow::Result;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::kv_pool::KvReservation;
use pegainfer_core::kv_pool::KvState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReleaseState {
    frontier: usize,
    origin_pages: usize,
    resident_pages: usize,
}

#[derive(Clone, Copy)]
struct ReleaseStep {
    tokens: usize,
    window: usize,
    page_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReleasePlan {
    frontier: usize,
    origin_pages: usize,
    release_pages: usize,
}

fn plan_release(state: ReleaseState, step: ReleaseStep) -> Result<ReleasePlan> {
    let frontier = state
        .frontier
        .checked_add(step.tokens)
        .ok_or_else(|| anyhow::anyhow!("the KV frontier overflows while advancing"))?;
    let origin_pages = frontier.saturating_sub(step.window) / step.page_size;
    let release_pages = origin_pages.checked_sub(state.origin_pages).ok_or_else(|| {
        anyhow::anyhow!(
            "the origin is already {} pages in where a frontier of {frontier} allows {origin_pages}",
            state.origin_pages
        )
    })?;
    anyhow::ensure!(
        release_pages <= state.resident_pages,
        "releasing {release_pages} pages needs more than the {} resident: the row and \
         a frontier of {frontier} have drifted apart",
        state.resident_pages
    );
    Ok(ReleasePlan {
        frontier,
        origin_pages,
        release_pages,
    })
}

/// The local family's state once the window can move: pages are held as
/// one reservation each so the front can be released page by page, which a
/// single request-owned permit cannot do. `origin_pages` counts what has
/// been released, which is exactly what separates the two coordinate
/// systems: `absolute = cache_relative + origin_pages * page_size`.
pub(crate) struct SlidingLocalKv {
    pool: KvPool,
    resident: VecDeque<KvReservation>,
    origin_pages: usize,
    frontier: usize,
}

impl SlidingLocalKv {
    pub(crate) fn new(pool: KvPool) -> Self {
        Self {
            pool,
            resident: VecDeque::new(),
            origin_pages: 0,
            frontier: 0,
        }
    }

    /// Rebuild a state from a prefix-cache restore: `resident` covers the
    /// window pages `[origin_pages, ceil(frontier/page))`, exactly the
    /// shape a request that prefilled to `frontier` and released its
    /// out-of-window front would hold.
    pub(crate) fn restore(
        pool: KvPool,
        resident: Vec<KvReservation>,
        origin_pages: usize,
        frontier: usize,
    ) -> Self {
        Self {
            pool,
            resident: resident.into(),
            origin_pages,
            frontier,
        }
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.frontier
    }

    pub(crate) fn held_pages(&self) -> usize {
        self.resident.len()
    }

    pub(crate) fn origin_pages(&self) -> usize {
        self.origin_pages
    }

    pub(crate) fn page_row(&self) -> Vec<i32> {
        let mut row = Vec::with_capacity(self.resident.len());
        self.extend_page_row(&mut row);
        row
    }

    pub(crate) fn extend_page_row(&self, row: &mut Vec<i32>) {
        for reservation in &self.resident {
            reservation.extend_page_indices_i32(row);
        }
    }

    pub(crate) fn layout(&self) -> &KvLayout {
        self.pool.layout()
    }

    pub(crate) fn belongs_to(&self, pool: &KvPool) -> bool {
        std::ptr::eq(self.pool.buffer(), pool.buffer())
    }

    pub(crate) fn extend_resident(&mut self, pages: Vec<KvReservation>) {
        self.resident.extend(pages);
    }

    /// Move the frontier without releasing anything: the overlapped
    /// prefill defers its release to join time, and the eviction A/B pins
    /// the footprint. Everything else goes through
    /// [`Self::advance_and_release`].
    pub(crate) fn advance(&mut self, count: usize) {
        self.frontier += count;
    }

    /// Move the frontier and drop whatever that makes releasable, as one
    /// settled move: a page goes once `(p + 1) * page_size + window <= frontier`,
    /// since the next query at the frontier reads keys from
    /// `frontier - (window - 1)` on. Everything is checked against the
    /// prospective frontier before any field changes, so a refused move
    /// leaves the request exactly where it was.
    pub(crate) fn advance_and_release(&mut self, count: usize, window: usize) -> Result<()> {
        let plan = plan_release(
            ReleaseState {
                frontier: self.frontier,
                origin_pages: self.origin_pages,
                resident_pages: self.resident.len(),
            },
            ReleaseStep {
                tokens: count,
                window,
                page_size: self.pool.layout().page_size,
            },
        )?;
        self.resident.drain(..plan.release_pages);
        self.origin_pages = plan.origin_pages;
        self.frontier = plan.frontier;
        Ok(())
    }
}

pub(crate) struct GemmaKv {
    pub(crate) local: SlidingLocalKv,
    pub(crate) global: KvState,
}

/// A refused side drops the other reservation, leaving both pools at their
/// pre-request occupancy. The page count is the exact frontier account: a
/// ceiling over the post-step kv_len, per family.
/// Pages a family still has to reserve to cover `kv_len` tokens, given what
/// its account already holds. `None` means the account is already past the
/// frontier — a bookkeeping error, not a surplus to spend.
pub(crate) fn pages_to_reserve(kv_len: usize, accounted: usize, page_size: usize) -> Option<usize> {
    kv_len.div_ceil(page_size).checked_sub(accounted)
}

pub(crate) fn admit_tokens(
    local_pool: &KvPool,
    global_pool: &KvPool,
    kv: &mut GemmaKv,
    new_tokens: usize,
) -> Result<()> {
    anyhow::ensure!(
        kv.local.belongs_to(local_pool) && kv.global.belongs_to(global_pool),
        "this KV state was allocated from different pools; admitting against \
         these would hand out page ids the executor cannot address"
    );
    let kv_len = kv.local.seq_len() + new_tokens;
    anyhow::ensure!(
        kv.local.seq_len() == kv.global.seq_len(),
        "the two families' frontiers diverged: local {} global {}",
        kv.local.seq_len(),
        kv.global.seq_len()
    );
    // Not saturating: a state past its frontier's account is a bookkeeping
    // error, and swallowing it defers the failure to the step that reads the
    // surplus page.
    let mut need = [0usize; 2];
    for (slot, (family, accounted, page_size)) in [
        (
            "local",
            kv.local.origin_pages() + kv.local.held_pages(),
            local_pool.layout().page_size,
        ),
        (
            "global",
            kv.global.held_pages(),
            global_pool.layout().page_size,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        need[slot] = pages_to_reserve(kv_len, accounted, page_size).ok_or_else(|| {
            let want = kv_len.div_ceil(page_size);
            anyhow::anyhow!(
                "{family} family already accounts for {accounted} pages where \
                 {kv_len} tokens need {want}"
            )
        })?;
    }
    let (local_need, global_need) = (need[0], need[1]);
    // One reservation per local page: the front is released page by page, so
    // the pages cannot share a permit.
    let mut local_rs = Vec::with_capacity(local_need);
    while local_rs.len() < local_need {
        match local_pool.try_reserve(1) {
            Some(r) => local_rs.push(r),
            None => break,
        }
    }
    let local_granted = local_rs.len() == local_need;
    let global_r = if local_granted {
        global_pool.try_reserve(global_need)
    } else {
        None
    };
    match (local_granted, global_r) {
        (true, Some(global_r)) => {
            kv.local.extend_resident(local_rs);
            kv.global.commit_reservation(global_r);
            Ok(())
        }
        (local_granted, global_r) => {
            let global_granted = global_r.is_some();
            // Report availability after rollback.
            drop((local_rs, global_r));
            anyhow::bail!(
                "admission refused for {new_tokens} tokens (kv_len {kv_len}): \
                 local need {local_need} avail {} ({}), global need {global_need} avail {} ({})",
                local_pool.available_pages(),
                if local_granted {
                    "granted, rolled back"
                } else {
                    "refused"
                },
                global_pool.available_pages(),
                if global_granted {
                    "granted, rolled back"
                } else {
                    "refused"
                },
            )
        }
    }
}

pub(crate) const PAGE_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use pegainfer_core::tensor::DeviceContext;

    use super::*;

    const WINDOW: usize = 1024;
    const SEGMENT: usize = 128;
    const TOTAL: usize = 1500;

    fn tiny_pools(ctx: &DeviceContext) -> (KvPool, KvPool) {
        let local = KvPool::new(ctx, 1, 1, 1, PAGE_SIZE, 4).expect("local pool");
        let global = KvPool::new(ctx, 1, 1, 1, PAGE_SIZE, 2).expect("global pool");
        (local, global)
    }

    fn kv_from(local: &KvPool, global: &KvPool) -> GemmaKv {
        GemmaKv {
            local: SlidingLocalKv::new(local.clone()),
            global: global.alloc(),
        }
    }

    fn apply_release(state: ReleaseState, tokens: usize) -> ReleaseState {
        let plan = plan_release(
            state,
            ReleaseStep {
                tokens,
                window: WINDOW,
                page_size: PAGE_SIZE,
            },
        )
        .expect("release plan");
        ReleaseState {
            frontier: plan.frontier,
            origin_pages: plan.origin_pages,
            resident_pages: state.resident_pages - plan.release_pages,
        }
    }

    fn segmented(mut state: ReleaseState) -> ReleaseState {
        let mut left = TOTAL;
        while left > 0 {
            let tokens = left.min(SEGMENT);
            state = apply_release(state, tokens);
            left -= tokens;
        }
        state
    }

    #[test]
    fn whole_and_segmented_release_share_one_ledger() {
        let initial = ReleaseState {
            frontier: 0,
            origin_pages: 0,
            resident_pages: TOTAL.div_ceil(PAGE_SIZE),
        };
        assert_eq!(segmented(initial), apply_release(initial, TOTAL));
    }

    #[test]
    fn segment_admission_keeps_the_same_ledger_and_bounds_residency() {
        let mut state = ReleaseState {
            frontier: 0,
            origin_pages: 0,
            resident_pages: 0,
        };
        let mut peak = 0;
        let mut left = TOTAL;
        while left > 0 {
            let tokens = left.min(SEGMENT);
            state.resident_pages += pages_to_reserve(
                state.frontier + tokens,
                state.origin_pages + state.resident_pages,
                PAGE_SIZE,
            )
            .expect("the account cannot sit past its own frontier");
            peak = peak.max(state.resident_pages);
            state = apply_release(state, tokens);
            left -= tokens;
        }
        let whole = apply_release(
            ReleaseState {
                frontier: 0,
                origin_pages: 0,
                resident_pages: TOTAL.div_ceil(PAGE_SIZE),
            },
            TOTAL,
        );
        assert_eq!(state, whole);
        let cap = WINDOW.div_ceil(PAGE_SIZE) + SEGMENT.div_ceil(PAGE_SIZE) + 1;
        assert!(peak <= cap, "resident peak {peak} exceeds cap {cap}");
    }

    #[test]
    fn admission_is_atomic_across_pools() {
        let ctx = DeviceContext::new().expect("GPU required");
        let (local, global) = tiny_pools(&ctx);
        let mut kv = kv_from(&local, &global);
        let refused = admit_tokens(&local, &global, &mut kv, 17);
        assert!(refused.is_err(), "partial admission must refuse");
        assert_eq!(local.available_pages(), 3, "local occupancy must roll back");
        assert_eq!(global.available_pages(), 1, "global occupancy untouched");
        assert_eq!((kv.local.held_pages(), kv.global.held_pages()), (0, 0));

        admit_tokens(&local, &global, &mut kv, PAGE_SIZE).expect("one page each");
        assert_eq!((local.available_pages(), global.available_pages()), (2, 0));
        assert_eq!((kv.local.held_pages(), kv.global.held_pages()), (1, 1));
    }
}
