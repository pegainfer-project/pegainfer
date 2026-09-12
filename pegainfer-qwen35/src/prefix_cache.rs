//! Joint full-attention KV and recurrent/conv prefix cache for Qwen3.5.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::KvView;
use pegainfer_kv_cache::RequestKv;

use crate::config::Config35;
use crate::config::LocalGeometry;
use crate::recurrent_state::RecurrentState;

/// Token interval at which a complete recurrent/conv snapshot may be cached.
pub(crate) const SNAPSHOT_STRIDE_TOKENS: usize = 256;

/// Content-addressed identity of one complete hybrid-model prefix boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PrefixBoundaryKey {
    /// Canonical full-attention KV lineage hash for this prefix.
    pub(crate) sequence_hash: [u8; 16],
    /// Exclusive token position represented by both KV and recurrent state.
    pub(crate) boundary_tokens: usize,
}

/// Cumulative counters for joint KV/recurrent prefix-cache activity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefixCacheStats {
    /// Requests that restored both KV and a recurrent snapshot.
    pub(crate) joint_hits: u64,
    /// Prompt tokens reused across all joint hits.
    pub(crate) joint_hit_tokens: u64,
    /// Requests with eligible resident KV but no matching snapshot.
    pub(crate) kv_only_fallbacks: u64,
    /// Individual eligible boundaries without a matching snapshot.
    pub(crate) snapshot_misses: u64,
    /// New recurrent snapshots published.
    pub(crate) inserts: u64,
    /// Published snapshots replaced by LRU insertion.
    pub(crate) evictions: u64,
}

/// One recurrent snapshot indexed by a reusable prefix boundary.
struct SnapshotEntry {
    /// Identically numbered physical recurrent snapshot on every rank.
    recurrent_slot: usize,
    /// Logical timestamp used to select an unpinned LRU victim.
    last_used: AtomicU64,
}

/// RAII pin on a recurrent snapshot during restore.
pub(crate) struct SnapshotGuard {
    /// Token boundary represented by the pinned entry.
    boundary: usize,
    /// The directory's extra strong references are active restore pins.
    entry: Arc<SnapshotEntry>,
}

impl SnapshotGuard {
    pub(crate) fn boundary(&self) -> usize {
        self.boundary
    }

    pub(crate) fn recurrent_slot(&self) -> usize {
        self.entry.recurrent_slot
    }
}

/// Mutable prefix-cache state shared by every execution rank.
struct SnapshotCache {
    /// Published boundary key to its recurrent snapshot entry.
    entries: HashMap<PrefixBoundaryKey, Arc<SnapshotEntry>>,
    /// Unpublished slots available without eviction.
    free_slots: Vec<usize>,
    /// Total number of preallocated physical snapshot slots.
    slot_count: usize,
    /// Monotonic logical time for LRU ordering.
    clock: u64,
}

impl SnapshotCache {
    fn new(slot_count: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(slot_count),
            free_slots: (0..slot_count).rev().collect(),
            slot_count,
            clock: 0,
        }
    }

    /// Number of currently published joint entries.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of preallocated recurrent-state slots.
    fn capacity(&self) -> usize {
        self.slot_count
    }

    /// Advance the non-zero logical clock used by the LRU policy.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    /// Look up `key`, refresh its LRU timestamp, and pin its entry.
    fn lookup(&mut self, key: PrefixBoundaryKey) -> Option<SnapshotGuard> {
        let last_used = self.tick();
        let entry = self.entries.get(&key)?;
        entry.last_used.store(last_used, Ordering::Relaxed);
        Some(SnapshotGuard {
            boundary: key.boundary_tokens,
            entry: Arc::clone(entry),
        })
    }

    fn contains(&self, key: PrefixBoundaryKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Reserve a free or unpinned LRU slot without publishing the new entry.
    fn reserve(&mut self, key: PrefixBoundaryKey) -> Option<SnapshotReservation> {
        let last_used = self.tick();
        if let Some(entry) = self.entries.get(&key) {
            entry.last_used.store(last_used, Ordering::Relaxed);
            return None;
        }

        let (slot, evicted) = if let Some(slot) = self.free_slots.pop() {
            (slot, false)
        } else {
            let (&victim_key, slot) = self
                .entries
                .iter()
                .filter(|(_, entry)| Arc::strong_count(entry) == 1)
                .min_by_key(|(_, entry)| entry.last_used.load(Ordering::Relaxed))
                .map(|(key, entry)| (key, entry.recurrent_slot))?;
            let evicted = self
                .entries
                .remove(&victim_key)
                .expect("LRU victim must still be present");
            debug_assert_eq!(evicted.recurrent_slot, slot);
            (slot, true)
        };

        Some(SnapshotReservation {
            recurrent_slot: slot,
            key,
            replaced: evicted,
        })
    }

    /// Publish after every rank has enqueued its recurrent-state copy.
    fn publish(&mut self, reservation: SnapshotReservation) {
        let last_used = self.tick();
        let previous = self.entries.insert(
            reservation.key,
            Arc::new(SnapshotEntry {
                recurrent_slot: reservation.recurrent_slot,
                last_used: AtomicU64::new(last_used),
            }),
        );
        debug_assert!(previous.is_none());
    }

    /// Return an unpublished slot after a physical copy failure.
    fn abort(&mut self, reservation: SnapshotReservation) {
        debug_assert!(!self.entries.contains_key(&reservation.key));
        self.free_slots.push(reservation.recurrent_slot);
    }
}

/// One pending central-directory insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotReservation {
    recurrent_slot: usize,
    key: PrefixBoundaryKey,
    /// Whether this insertion replaced an unpinned snapshot entry.
    replaced: bool,
}

impl SnapshotReservation {
    pub(crate) fn recurrent_slot(self) -> usize {
        self.recurrent_slot
    }

    fn was_replacement(self) -> bool {
        self.replaced
    }
}

/// Rank-local physical recurrent/conv snapshot allocations.
pub(crate) struct RecurrentStateStore {
    slots: Vec<RecurrentState>,
}

impl RecurrentStateStore {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        slot_count: usize,
    ) -> Result<Self> {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(RecurrentState::new(ctx, config, geometry)?);
        }
        Ok(Self { slots })
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn save(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        src: &RecurrentState,
    ) -> Result<()> {
        let dst = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }

    pub(crate) fn restore(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        let src = self
            .slots
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?;
        dst.copy_from(ctx, src)
    }
}

/// The only Qwen3.5 scheduler interface allowed to reconcile paged KV with
/// recurrent/conv state.
pub(crate) struct Qwen35PrefixCache {
    /// Logical block pool paired with the full-attention GPU KV buffer.
    kv: KvCacheManager,
    /// Prefix key, slot, pin, and LRU state for reusable entries.
    snapshots: SnapshotCache,
    /// Scheduler-thread-owned cumulative metrics.
    stats: PrefixCacheStats,
}

impl Qwen35PrefixCache {
    /// Build the joint coordinator for `snapshot_slots` per-rank allocations.
    pub(crate) fn new(kv: KvCacheManager, snapshot_slots: usize) -> Result<Self> {
        anyhow::ensure!(
            SNAPSHOT_STRIDE_TOKENS.is_multiple_of(kv.pool().block_size()),
            "Qwen3.5 snapshot stride {SNAPSHOT_STRIDE_TOKENS} must be a multiple of KV block size {}",
            kv.pool().block_size()
        );
        Ok(Self {
            kv,
            snapshots: SnapshotCache::new(snapshot_slots),
            stats: PrefixCacheStats::default(),
        })
    }

    /// Logical KV block pool used for request allocation and admission.
    pub(crate) fn pool(&self) -> &pegainfer_kv_cache::BlockPool {
        self.kv.pool()
    }

    /// Rank-0 physical full-attention KV storage indexed by [`Self::pool`].
    pub(crate) fn buffer(&self) -> &KvBuffer {
        self.kv.buffer()
    }

    /// Whether joint prefix reuse and snapshot publication are enabled.
    pub(crate) fn enabled(&self) -> bool {
        self.snapshots.capacity() > 0
    }

    /// Total number of preallocated recurrent snapshot slots.
    pub(crate) fn snapshot_slots(&self) -> usize {
        self.snapshots.capacity()
    }

    /// Number of snapshot slots that currently have a published key.
    pub(crate) fn snapshot_occupancy(&self) -> usize {
        self.snapshots.len()
    }

    /// Return a point-in-time copy of cumulative cache metrics.
    pub(crate) fn stats(&self) -> PrefixCacheStats {
        self.stats
    }

    /// Return the longest prefix with both active KV and a recurrent snapshot.
    /// Its KV pages are outside the available pool and may be shared.
    pub(crate) fn active_joint_prefix_pages(
        &self,
        prompt_tokens: &[u32],
        lora_name: Option<&str>,
    ) -> usize {
        if !self.enabled() {
            return 0;
        }
        let hashes = self
            .kv
            .pool()
            .active_prefix_hashes(prompt_tokens.to_vec(), lora_name);
        let block_size = self.kv.pool().block_size();
        let resident_tokens = hashes.len() * block_size;
        eligible_boundaries(resident_tokens, SNAPSHOT_STRIDE_TOKENS)
            .find(|&boundary| {
                let key = PrefixBoundaryKey {
                    sequence_hash: hashes[boundary / block_size - 1],
                    boundary_tokens: boundary,
                };
                self.snapshots.contains(key)
            })
            .map_or(0, |boundary| boundary / block_size)
    }

    /// Create request-local KV state and select the longest joint prefix.
    ///
    /// If a joint prefix is found, the KV blocks are attached to the request
    /// and a guard is returned to prevent eviction until the recurrent restore
    /// is enqueued (see `Qwen35PrefixCache::finish_restore`).
    /// If no joint prefix is found, the request is still created and returned.
    pub(crate) fn begin_request(
        &mut self,
        prompt_tokens: &[u32],
        max_output_tokens: usize,
        lora_name: Option<&str>,
        allow_match: bool,
    ) -> Result<(RequestKv, Option<SnapshotGuard>)> {
        let mut request =
            self.kv
                .pool()
                .new_request(prompt_tokens.to_vec(), max_output_tokens, lora_name);
        if !self.enabled() || !allow_match {
            return Ok((request, None));
        }

        let probe = self
            .kv
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_name);
        let resident_tokens = probe.reusable_blocks() * self.kv.pool().block_size();
        let mut saw_eligible_kv = false;
        for boundary in eligible_boundaries(resident_tokens, SNAPSHOT_STRIDE_TOKENS) {
            saw_eligible_kv = true;
            let Some(sequence_hash) = probe.boundary_hash(boundary) else {
                continue;
            };
            let key = PrefixBoundaryKey {
                sequence_hash,
                boundary_tokens: boundary,
            };
            let Some(guard) = self.snapshots.lookup(key) else {
                self.stats.snapshot_misses += 1;
                continue;
            };

            let max_blocks = boundary / self.kv.pool().block_size();
            let attached = match request.match_and_add_prefix_up_to(self.kv.pool(), max_blocks) {
                Ok(attached) => attached,
                Err(error) => {
                    let _ = request.release();
                    return Err(error);
                }
            };
            if attached != boundary {
                let _ = request.release();
                anyhow::bail!(
                    "Qwen3.5 joint prefix attach selected {boundary} tokens but attached {attached}"
                );
            }
            return Ok((request, Some(guard)));
        }

        if saw_eligible_kv {
            self.stats.kv_only_fallbacks += 1;
        }
        Ok((request, None))
    }

    /// Finish a restore after the caller has enqueued the recurrent-state copy.
    ///
    /// `begin_request` performs lookup and KV attachment. The physical
    /// recurrent-state copy remains executor-specific: single-GPU execution
    /// copies from the local store, while TP coordinates all workers. This
    /// validates the positions and releases the guard once every copy is
    /// enqueued in stream order.
    pub(crate) fn finish_restore(
        &mut self,
        request: &RequestKv,
        guard: SnapshotGuard,
        recurrent_positions: &[usize],
    ) -> Result<usize> {
        let boundary = guard.boundary();
        anyhow::ensure!(
            request.kv_position() == boundary
                && !recurrent_positions.is_empty()
                && recurrent_positions
                    .iter()
                    .all(|&position| position == boundary),
            "Qwen3.5 joint prefix restore position mismatch: kv={}, recurrent={recurrent_positions:?}, boundary={}",
            request.kv_position(),
            boundary,
        );
        self.stats.joint_hits += 1;
        self.stats.joint_hit_tokens += boundary as u64;
        // End the cache pin after restore enqueue and position validation.
        drop(guard);
        Ok(boundary)
    }

    /// Reserve the KV pages required by the next prefill forward.
    pub(crate) fn schedule_prefill(&self, request: &mut RequestKv, tokens: usize) -> Result<()> {
        request
            .schedule_prefill(tokens, self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 prefill KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for prefill kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn prefill_view(&self, request: &RequestKv, tokens: usize) -> KvView {
        request.prefill_view(tokens)
    }

    /// Apply one successful whole-model prefill window.
    ///
    /// KV is applied for every window. The returned boundary can be passed to
    /// [`Self::reserve_snapshot`], which accepts only non-zero multiples of
    /// [`SNAPSHOT_STRIDE_TOKENS`].
    pub(crate) fn apply_prefill(
        &self,
        request: &mut RequestKv,
        first_token: Option<u32>,
    ) -> Result<usize> {
        let applied = if let Some(first_token) = first_token {
            request.apply_prefill(first_token, self.kv.pool())
        } else {
            request.apply_prefill_chunk(self.kv.pool())
        };
        // Retain registered KV only when prefix caching is enabled.
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        applied?;
        let boundary = request.kv_position();
        Ok(boundary)
    }

    /// Reserve a recurrent-state slot for an eligible applied boundary.
    ///
    /// Returns a reservation when the boundary is eligible and a free or
    /// unpinned slot is available. The caller copies recurrent state into that
    /// slot on every rank, then publishes or aborts the reservation. Duplicate
    /// boundaries, disabled caching, ineligible boundaries, and fully pinned
    /// capacity return `None`.
    pub(crate) fn reserve_snapshot(
        &mut self,
        request: &RequestKv,
        boundary: usize,
    ) -> Result<Option<SnapshotReservation>> {
        if !self.enabled() {
            return Ok(None);
        }
        if boundary == 0 || !boundary.is_multiple_of(SNAPSHOT_STRIDE_TOKENS) {
            return Ok(None);
        }
        let sequence_hash = request
            .registered_boundary_hash(boundary)
            .ok_or_else(|| anyhow::anyhow!("no registered KV hash at boundary {boundary}"))?;
        let key = PrefixBoundaryKey {
            sequence_hash,
            boundary_tokens: boundary,
        };
        Ok(self.snapshots.reserve(key))
    }

    /// Publish after rank-local snapshot copies succeed. `reserve_snapshot`
    /// already validated registered KV at the boundary.
    pub(crate) fn publish_snapshot(&mut self, reservation: SnapshotReservation) {
        self.stats.inserts += 1;
        if reservation.was_replacement() {
            self.stats.evictions += 1;
        }
        self.snapshots.publish(reservation);
    }

    /// Abort a prefix reservation after any rank-local copy failure.
    pub(crate) fn abort_snapshot(&mut self, reservation: SnapshotReservation) {
        self.snapshots.abort(reservation);
    }

    /// Reserve KV capacity for the next one-token decode forward.
    pub(crate) fn schedule_decode(&self, request: &mut RequestKv) -> Result<()> {
        request
            .schedule_decode(self.kv.pool())
            .map_err(|e| anyhow::anyhow!("Qwen3.5 decode KV schedule failed: {e}"))
    }

    /// Build the exact, immutable KV page-table view for decode kernels.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn decode_view(&self, request: &RequestKv) -> KvView {
        request.decode_view()
    }

    /// Apply the KV written by decode and record the newly sampled token.
    pub(crate) fn apply_decode(&self, request: &mut RequestKv, token: u32) -> Result<()> {
        let applied = request.apply_decode(token, self.kv.pool());
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        applied?;
        Ok(())
    }

    /// Roll back pages reserved by a scheduled step that did not apply.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn revert_schedule(&self, request: &mut RequestKv) -> Result<()> {
        request.revert_schedule()
    }

    /// Release all request KV.
    pub(crate) fn release_request(&self, request: &mut RequestKv) -> Result<()> {
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        request.release()
    }
}

/// Yield reusable snapshot boundaries from longest to shortest.
fn eligible_boundaries(resident_tokens: usize, stride: usize) -> impl Iterator<Item = usize> {
    let highest = resident_tokens / stride * stride;
    (1..=highest / stride).rev().map(move |n| n * stride)
}

#[cfg(test)]
mod tests {
    use pegainfer_kv_cache::BlockPool;

    use super::PrefixBoundaryKey;
    use super::SNAPSHOT_STRIDE_TOKENS;
    use super::SnapshotCache;
    use super::eligible_boundaries;

    fn key(tag: u8) -> PrefixBoundaryKey {
        PrefixBoundaryKey {
            sequence_hash: [tag; 16],
            boundary_tokens: 256,
        }
    }

    #[test]
    fn joint_boundaries_descend_on_snapshot_stride() {
        assert_eq!(
            eligible_boundaries(255, 256).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(eligible_boundaries(256, 256).collect::<Vec<_>>(), [256]);
        assert_eq!(
            eligible_boundaries(900, 256).collect::<Vec<_>>(),
            [768, 512, 256]
        );
    }

    #[test]
    fn state_publishes_only_after_explicit_commit() {
        let mut state = SnapshotCache::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        assert!(state.lookup(key(1)).is_none());
        state.publish(reservation);
        assert!(state.lookup(key(1)).is_some());
    }

    #[test]
    fn aborted_write_does_not_expose_partial_snapshot() {
        let mut state = SnapshotCache::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        state.abort(reservation);
        assert!(state.lookup(key(1)).is_none());
        assert!(state.reserve(key(2)).is_some());
    }

    #[test]
    fn duplicate_refreshes_lru_without_allocating_a_slot() {
        let mut state = SnapshotCache::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty directory must reserve a write");
        state.publish(reservation);
        assert!(state.reserve(key(1)).is_none());
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn lru_evicts_untouched_snapshot_but_preserves_touched_snapshot() {
        let mut state = SnapshotCache::new(2);
        for tag in [1, 2] {
            let reservation = state
                .reserve(key(tag))
                .expect("state should have a free slot");
            state.publish(reservation);
        }
        drop(state.lookup(key(1)).expect("key 1 should be present"));
        let reservation = state
            .reserve(key(3))
            .expect("an unpinned LRU victim should be available");
        state.publish(reservation);
        assert!(state.lookup(key(1)).is_some());
        assert!(state.lookup(key(2)).is_none());
        assert!(state.lookup(key(3)).is_some());
    }

    #[test]
    fn all_pinned_snapshots_make_insertion_a_soft_miss() {
        let mut state = SnapshotCache::new(1);
        let reservation = state
            .reserve(key(1))
            .expect("empty state must reserve a write");
        state.publish(reservation);
        let guard = state.lookup(key(1)).expect("key 1 should be present");
        assert!(state.reserve(key(2)).is_none());
        drop(guard);
        assert!(state.reserve(key(2)).is_some());
    }

    #[test]
    fn released_prefix_kv_is_reclaimable_without_snapshot_eviction() {
        let pool = BlockPool::new(16, 32).expect("block pool");
        let baseline = pool.available_blocks();
        let prompt = vec![7; SNAPSHOT_STRIDE_TOKENS + 16];
        let mut request = pool.new_request(prompt.clone(), 0, None);
        request
            .schedule_prefill(prompt.len(), &pool)
            .expect("schedule prefill");
        request.apply_prefill_chunk(&pool).expect("apply prefill");
        let cached_boundary = request
            .registered_boundary_hash(SNAPSHOT_STRIDE_TOKENS)
            .expect("full snapshot boundary is registered");

        let mut state = SnapshotCache::new(1);
        let key = PrefixBoundaryKey {
            sequence_hash: cached_boundary,
            boundary_tokens: SNAPSHOT_STRIDE_TOKENS,
        };
        let reservation = state
            .reserve(key)
            .expect("empty state must reserve a write");
        state.publish(reservation);
        request.release().expect("release request");

        // Released KV remains reusable and counts as available capacity.
        assert_eq!(pool.available_blocks(), baseline);

        let mut warm = pool.new_request(prompt, 0, None);
        assert_eq!(
            warm.match_and_add_prefix(&pool).expect("match inactive KV"),
            SNAPSHOT_STRIDE_TOKENS
        );
        warm.release().expect("release warm request");

        // A full-pool cold reservation reclaims KV without evicting the snapshot.
        let cold_prompt = vec![9; baseline * pool.block_size()];
        let mut cold = pool.new_request(cold_prompt.clone(), 0, None);
        cold.schedule_prefill(cold_prompt.len(), &pool)
            .expect("inactive KV must satisfy cold allocation");
        cold.revert_schedule().expect("revert cold reservation");
        cold.release().expect("release cold request");

        assert!(state.lookup(key).is_some());
    }
}
