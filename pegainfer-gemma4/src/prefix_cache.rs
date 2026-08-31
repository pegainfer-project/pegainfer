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

#[derive(Clone, Copy)]
struct ResumeCandidate<'a> {
    token_ids: &'a [u32],
    local_origin: usize,
}

fn resume_point(candidate: ResumeCandidate<'_>, window: usize, prompt: &[u32]) -> Option<usize> {
    let lcp = candidate
        .token_ids
        .iter()
        .zip(prompt)
        .take_while(|(cached, requested)| cached == requested)
        .count();
    let resume = lcp.min(prompt.len().checked_sub(1)?);
    let floor = if candidate.local_origin == 0 {
        MIN_RESUME_TOKENS
    } else {
        (candidate.local_origin * PAGE_SIZE + window).max(MIN_RESUME_TOKENS)
    };
    (resume >= floor).then_some(resume)
}

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
                let resume = resume_point(
                    ResumeCandidate {
                        token_ids: &e.token_ids,
                        local_origin: e.local_origin,
                    },
                    self.window,
                    prompt,
                )?;
                Some((e, resume))
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
            self.entries.retain(|entry| entry.id != id);
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
        let Some((index, _)) = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.stamp)
        else {
            return false;
        };
        self.entries.swap_remove(index);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resume(entry: &[u32], local_origin: usize, window: usize, prompt: &[u32]) -> Option<usize> {
        resume_point(
            ResumeCandidate {
                token_ids: entry,
                local_origin,
            },
            window,
            prompt,
        )
    }

    #[test]
    fn resume_calls_the_production_helper_at_the_63_64_boundary() {
        let entry: Vec<u32> = (0..80).collect();
        let mut at_63 = entry.clone();
        at_63[63] = 999;
        let mut at_64 = entry.clone();
        at_64[64] = 999;
        assert_eq!(resume(&entry, 0, PAGE_SIZE, &at_63), None);
        assert_eq!(resume(&entry, 0, PAGE_SIZE, &at_64), Some(64));
    }

    #[test]
    fn resume_clamps_to_suffix_and_respects_the_window_floor() {
        let entry: Vec<u32> = (0..96).collect();
        let mut extended = entry.clone();
        extended.push(999);
        assert_eq!(resume(&entry, 0, PAGE_SIZE, &extended), Some(96));
        assert_eq!(resume(&entry, 0, PAGE_SIZE, &entry), Some(95));
        assert_eq!(resume(&entry, 0, PAGE_SIZE, &[]), None);

        let mut before_window = entry.clone();
        before_window[47] = 777;
        assert_eq!(resume(&entry, 2, PAGE_SIZE, &before_window), None);
        let mut at_window = entry.clone();
        at_window[48] = 777;
        assert_eq!(resume(&entry, 2, PAGE_SIZE, &at_window), None);
        let mut after_minimum = entry.clone();
        after_minimum[64] = 777;
        assert_eq!(resume(&entry, 2, PAGE_SIZE, &after_minimum), Some(64));
    }

    #[test]
    fn global_page_budget_rounds_before_halving() {
        assert_eq!(entry_global_pages(8192), 256);
        assert_eq!(entry_global_pages(8193), 256);
        assert_eq!(entry_global_pages(8224), 257);
    }
}
