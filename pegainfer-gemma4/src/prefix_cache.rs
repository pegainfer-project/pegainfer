//! Conversation-tail prefix cache: an engine-side, copy-on-restore cache of
//! completed prompt-state KV. A checkpoint is a request's post-prefill KV,
//! copied wholesale into cache-owned pages — the append-only global family
//! up to the prompt frontier plus the local family's resident window there.
//! A hit resumes at the longest common prefix with the new prompt, clamped
//! into the tail range the captured window can serve: no window, no resume,
//! both families restore together or not at all. Real chat turns diverge
//! from the cached sequence only near the tail (the previous completion's
//! re-rendering), which is exactly where the window lives.
//!
//! Fail-closed gate: without `PEGAINFER_PREFIX_CACHE=K` in the environment
//! the cache holds nothing and resolves nothing, and the engine behaves as
//! before.

use pegainfer_core::kv_pool::KvReservation;

use crate::kv::PAGE_SIZE;

/// A resume below this many tokens is not worth its page copies.
const MIN_RESUME_TOKENS: usize = 64;

/// One captured conversation tail. The page reservations are cache-owned;
/// dropping the entry returns every page to its pool.
pub(crate) struct CachedKv {
    /// The captured rendered prompt — prompt-only by design: generated
    /// tokens never re-render to the same ids, so only the prompt region
    /// can ever be hit again.
    pub(crate) token_ids: Vec<u32>,
    /// Cache-owned copy of the global family's pages `[0, token_ids.len())`.
    pub(crate) global_pages: KvReservation,
    /// Cache-owned copies of the local family's resident window pages, in
    /// front-to-back order, one page per reservation (the shape
    /// [`crate::kv::SlidingLocalKv`] releases from).
    pub(crate) local_pages: Vec<KvReservation>,
    /// Front-released page count at capture — the window's origin.
    pub(crate) local_origin: usize,
    /// Stable identity: the restore that resumed from this entry names it
    /// as the successor's ancestor at insert time.
    pub(crate) id: u64,
    stamp: u64,
}

impl CachedKv {
    pub(crate) fn new(
        token_ids: Vec<u32>,
        global_pages: KvReservation,
        local_pages: Vec<KvReservation>,
        local_origin: usize,
    ) -> Self {
        Self {
            token_ids,
            global_pages,
            local_pages,
            local_origin,
            id: 0,
            stamp: 0,
        }
    }
}

/// LRU set of conversation tails, capacity-capped.
pub(crate) struct PrefixCache {
    entries: Vec<CachedKv>,
    cap: usize,
    window: usize,
    clock: u64,
}

/// Global-family pages budgeted per cache entry — half the serving
/// context. Capture refuses a longer prompt, so the cache can never hold
/// more than the share of the pool its entries paid for at startup.
pub(crate) fn entry_global_pages(max_context: usize) -> usize {
    max_context.div_ceil(PAGE_SIZE) / 2
}

/// The fail-closed gate: `PEGAINFER_PREFIX_CACHE=K` enables a K-entry
/// cache; unset or unparsable-as-positive disables it entirely.
pub(crate) fn prefix_cache_cap() -> Option<usize> {
    std::env::var("PEGAINFER_PREFIX_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k > 0)
}

impl PrefixCache {
    pub(crate) fn new(cap: usize, window: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
            cap,
            window,
            clock: 0,
        }
    }

    /// Best resumable point across the cached tails: the longest common
    /// prefix with `prompt`, clamped into the entry's tail-window range —
    /// the local window only exists at the captured tail, so a resume at
    /// `t` needs `origin(t) >= entry.origin` (any `t` for an entry shorter
    /// than the window). `t < prompt.len()` keeps at least one suffix
    /// token, so the resumed prefill produces the next logits.
    pub(crate) fn resolve(&mut self, prompt: &[u32]) -> Option<(&CachedKv, usize)> {
        self.clock += 1;
        let clock = self.clock;
        let picked = self
            .entries
            .iter_mut()
            .filter_map(|e| {
                let lcp = e
                    .token_ids
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count();
                let t = lcp.min(prompt.len() - 1);
                let floor = if e.local_origin == 0 {
                    MIN_RESUME_TOKENS
                } else {
                    (e.local_origin * PAGE_SIZE + self.window).max(MIN_RESUME_TOKENS)
                };
                if t < floor {
                    return None;
                }
                Some((e, t))
            })
            .max_by_key(|&(_, t)| t);
        let (best, t) = picked?;
        best.stamp = clock;
        log::debug!(
            "gemma4 prefix-cache hit: resume at {t} of {} prompt tokens (entry {})",
            prompt.len(),
            best.token_ids.len()
        );
        Some((best, t))
    }

    /// Insert a captured tail. When the captured request resumed from a
    /// cached entry, that entry — and only that entry — is its stale
    /// ancestor and is replaced: lineage is the restore itself, never
    /// inferred from token content, so two conversations sharing all but a
    /// few trailing tokens both stay. Beyond that, a full cache evicts LRU.
    pub(crate) fn insert(&mut self, mut entry: CachedKv, ancestor: Option<u64>) {
        self.clock += 1;
        entry.stamp = self.clock;
        entry.id = self.clock;
        if let Some(id) = ancestor {
            self.entries.retain(|e| e.id != id);
        }
        if self.entries.len() >= self.cap {
            self.evict_lru();
        }
        self.entries.push(entry);
    }

    /// Drop the least-recently-used entry, returning whether one existed —
    /// the pool-pressure valve: an admission that cannot reserve pages
    /// evicts and retries before requeueing.
    pub(crate) fn evict_lru(&mut self) -> bool {
        let Some((idx, _)) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.stamp) else {
            return false;
        };
        self.entries.swap_remove(idx);
        true
    }
}

#[cfg(test)]
mod tests {
    // Host-side index logic only; the GPU capture/restore halves live in
    // `serve` and are covered by the on-box restore gate. Reservations
    // require a pool, so the resolve arithmetic is exercised through a
    // pool-free twin.

    /// The resolve arithmetic, extracted: lcp clamped to the tail-window
    /// floor; returns the resume point.
    fn resume_at(entry: &[u32], origin: usize, window: usize, prompt: &[u32]) -> Option<usize> {
        let lcp = entry.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let t = lcp.min(prompt.len() - 1);
        let floor = if origin == 0 { 1 } else { origin * 16 + window };
        (t >= floor).then_some(t)
    }

    #[test]
    fn lcp_clamps_and_respects_window_floor() {
        // Short entry (origin 0): any common prefix resumes, clamped to
        // leave one suffix token.
        assert_eq!(resume_at(&[1, 2, 3, 4], 0, 16, &[1, 2, 3, 4, 5]), Some(4));
        assert_eq!(resume_at(&[1, 2, 3, 4], 0, 16, &[1, 2, 9, 9, 9]), Some(2));
        assert_eq!(resume_at(&[1, 2, 3, 4], 0, 16, &[1, 2, 3, 4]), Some(3));
        assert_eq!(resume_at(&[9, 9], 0, 16, &[1, 2]), None);
        // Released-front entry: resume only inside the surviving window
        // (origin 2, window 16 -> floor 48).
        let entry: Vec<u32> = (0..64).collect();
        let mut prompt = entry.clone();
        prompt.push(999);
        assert_eq!(resume_at(&entry, 2, 16, &prompt), Some(64));
        // Divergence before the floor: the window cannot serve it.
        let mut early = entry.clone();
        early[40] = 777;
        assert_eq!(resume_at(&entry, 2, 16, &early), None);
    }
}
