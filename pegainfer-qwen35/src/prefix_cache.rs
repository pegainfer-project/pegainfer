//! Joint full-attention KV and recurrent/conv prefix cache for Qwen3.5.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::sys::CUevent_flags;
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

/// Last stride boundary at or below `history_tokens` (the prompt length without one).
pub(crate) fn history_stop(
    prompt_len: usize,
    history_tokens: Option<usize>,
    stride: usize,
) -> usize {
    history_tokens.map_or(prompt_len, |h| h.min(prompt_len)) / stride * stride
}

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
    /// Prompt tokens that reached an enabled prefix-cache lookup.
    pub(crate) queries: u64,
    /// Of `joint_hit_tokens`, tokens restored from the pinned host tier.
    pub(crate) host_hits: u64,
    /// GPU -> host copies completed for a reused snapshot slot.
    pub(crate) spills: u64,
    /// Published host snapshots replaced by a new spill.
    pub(crate) host_evictions: u64,
}

/// Which backing tier holds one published snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SnapshotTier {
    Gpu,
    Host,
}

/// One recurrent snapshot indexed by a reusable prefix boundary.
struct SnapshotEntry {
    /// Physical slot in the GPU store or the pinned host slab.
    slot: usize,
    tier: SnapshotTier,
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

    /// Slot number in the GPU store or the host slab.
    pub(crate) fn recurrent_slot(&self) -> usize {
        self.entry.slot
    }

    pub(crate) fn tier(&self) -> SnapshotTier {
        self.entry.tier
    }
}

/// A pending GPU -> host copy. The caller performs it before
/// [`SnapshotCache::commit_plan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotSpill {
    pub(crate) victim_key: PrefixBoundaryKey,
    pub(crate) gpu_slot: usize,
    pub(crate) host_slot: usize,
}

/// A reservation decision plus the optional spill the caller must perform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotReservePlan {
    reservation: SnapshotReservation,
    spill: Option<SnapshotSpill>,
    evicted_host: bool,
}

impl SnapshotReservePlan {
    pub(crate) fn reservation(self) -> SnapshotReservation {
        self.reservation
    }

    pub(crate) fn spill(self) -> Option<SnapshotSpill> {
        self.spill
    }

    pub(crate) fn evicted_host(self) -> bool {
        self.evicted_host
    }
}

/// Mutable prefix-cache directory shared by every execution rank.
struct SnapshotCache {
    entries: HashMap<PrefixBoundaryKey, Arc<SnapshotEntry>>,
    free_gpu_slots: Vec<usize>,
    free_host_slots: Vec<usize>,
    gpu_slot_count: usize,
    host_slot_count: usize,
    clock: u64,
}

impl SnapshotCache {
    fn new(gpu_slot_count: usize, host_slot_count: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(gpu_slot_count + host_slot_count),
            free_gpu_slots: (0..gpu_slot_count).rev().collect(),
            free_host_slots: (0..host_slot_count).rev().collect(),
            gpu_slot_count,
            host_slot_count,
            clock: 0,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn capacity(&self) -> usize {
        self.gpu_slot_count
    }

    fn host_capacity(&self) -> usize {
        self.host_slot_count
    }

    fn host_len(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.tier == SnapshotTier::Host)
            .count()
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

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

    fn touch(&mut self, key: PrefixBoundaryKey) {
        let last_used = self.tick();
        if let Some(entry) = self.entries.get(&key) {
            entry.last_used.store(last_used, Ordering::Relaxed);
        }
    }

    /// Drop unpinned entries whose KV is already gone. Their snapshots cannot
    /// hit, so they are not worth spilling.
    fn drop_dead_unpinned(
        &mut self,
        tier: SnapshotTier,
        is_live: &dyn Fn(PrefixBoundaryKey) -> bool,
    ) {
        let dead: Vec<(PrefixBoundaryKey, usize)> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.tier == tier && Arc::strong_count(entry) == 1)
            .filter(|(key, _)| !is_live(**key))
            .map(|(key, entry)| (*key, entry.slot))
            .collect();
        for (key, slot) in dead {
            let removed = self
                .entries
                .remove(&key)
                .expect("dead snapshot must still be present");
            debug_assert_eq!(removed.slot, slot);
            match tier {
                SnapshotTier::Gpu => self.free_gpu_slots.push(slot),
                SnapshotTier::Host => self.free_host_slots.push(slot),
            }
        }
    }

    fn take_unpinned_victim(&mut self, tier: SnapshotTier) -> Option<(PrefixBoundaryKey, usize)> {
        let (&victim_key, slot) = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.tier == tier && Arc::strong_count(entry) == 1)
            .min_by_key(|(_, entry)| entry.last_used.load(Ordering::Relaxed))
            .map(|(key, entry)| (key, entry.slot))?;
        let victim = self
            .entries
            .remove(&victim_key)
            .expect("LRU victim must still be present");
        debug_assert_eq!(victim.slot, slot);
        debug_assert_eq!(victim.tier, tier);
        Some((victim_key, slot))
    }

    /// Decide which slot the next snapshot uses. With a host slab, a reused GPU
    /// slot returns a spill plan the caller must perform before `commit_plan`.
    fn reserve(
        &mut self,
        key: PrefixBoundaryKey,
        is_live: &dyn Fn(PrefixBoundaryKey) -> bool,
    ) -> Option<SnapshotReservePlan> {
        if self.contains(key) {
            self.touch(key);
            return None;
        }
        self.drop_dead_unpinned(SnapshotTier::Gpu, is_live);
        if let Some(slot) = self.free_gpu_slots.pop() {
            return Some(SnapshotReservePlan {
                reservation: SnapshotReservation {
                    key,
                    recurrent_slot: slot,
                    replaced: false,
                },
                spill: None,
                evicted_host: false,
            });
        }
        let (victim_key, victim_slot) = self.take_unpinned_victim(SnapshotTier::Gpu)?;
        if self.host_slot_count > 0 {
            self.drop_dead_unpinned(SnapshotTier::Host, is_live);
            let (host_slot, evicted_host) = if let Some(slot) = self.free_host_slots.pop() {
                (slot, false)
            } else if let Some((_, slot)) = self.take_unpinned_victim(SnapshotTier::Host) {
                (slot, true)
            } else {
                self.insert(victim_key, victim_slot, SnapshotTier::Gpu);
                return None;
            };
            return Some(SnapshotReservePlan {
                reservation: SnapshotReservation {
                    key,
                    recurrent_slot: victim_slot,
                    replaced: evicted_host,
                },
                spill: Some(SnapshotSpill {
                    victim_key,
                    gpu_slot: victim_slot,
                    host_slot,
                }),
                evicted_host,
            });
        }
        Some(SnapshotReservePlan {
            reservation: SnapshotReservation {
                key,
                recurrent_slot: victim_slot,
                replaced: true,
            },
            spill: None,
            evicted_host: false,
        })
    }

    fn insert(&mut self, key: PrefixBoundaryKey, slot: usize, tier: SnapshotTier) {
        let last_used = self.tick();
        let previous = self.entries.insert(
            key,
            Arc::new(SnapshotEntry {
                slot,
                tier,
                last_used: AtomicU64::new(last_used),
            }),
        );
        debug_assert!(previous.is_none());
    }

    /// Commit a plan after the caller performed its spill.
    fn commit_plan(&mut self, plan: SnapshotReservePlan) {
        if let Some(spill) = plan.spill {
            self.insert(spill.victim_key, spill.host_slot, SnapshotTier::Host);
        }
    }

    /// Abort a plan: restore a spilled victim or release the free slot.
    fn abort_plan(&mut self, plan: SnapshotReservePlan) {
        if let Some(spill) = plan.spill {
            self.free_host_slots.push(spill.host_slot);
            self.insert(spill.victim_key, spill.gpu_slot, SnapshotTier::Gpu);
        } else {
            self.free_gpu_slots.push(plan.reservation.recurrent_slot);
        }
    }

    fn publish(&mut self, reservation: SnapshotReservation) {
        self.insert(
            reservation.key,
            reservation.recurrent_slot,
            SnapshotTier::Gpu,
        );
    }

    fn abort(&mut self, reservation: SnapshotReservation) {
        debug_assert!(!self.entries.contains_key(&reservation.key));
        self.free_gpu_slots.push(reservation.recurrent_slot);
    }
}

/// One pending central-directory insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SnapshotReservation {
    recurrent_slot: usize,
    key: PrefixBoundaryKey,
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

/// Pinned-host backing for recurrent/conv snapshots.
pub(crate) struct HostSnapshotSlab {
    state: PinnedHostSlice<f32>,
    conv: PinnedHostSlice<half::bf16>,
    copy_stream: Arc<CudaStream>,
    source_ready: CudaEvent,
    spill_events: Vec<CudaEvent>,
    slots: usize,
    layers: usize,
    state_per_layer: usize,
    conv_per_layer: usize,
    state_per_slot: usize,
    conv_per_slot: usize,
}

impl HostSnapshotSlab {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        slots: usize,
    ) -> Result<Self> {
        anyhow::ensure!(slots > 0, "host snapshot slab needs at least one slot");
        let layers = config.num_hidden_layers - config.num_full_attention_layers();
        let state_per_layer = geometry.local_linear_num_value_heads()
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_per_layer = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);
        let state_per_slot = layers * state_per_layer;
        let conv_per_slot = layers * conv_per_layer;
        let state = unsafe { ctx.ctx.alloc_pinned::<f32>(slots * state_per_slot) }
            .map_err(|e| anyhow::anyhow!("allocate pinned recurrent slab: {e}"))?;
        let conv = unsafe { ctx.ctx.alloc_pinned::<half::bf16>(slots * conv_per_slot) }
            .map_err(|e| anyhow::anyhow!("allocate pinned conv slab: {e}"))?;
        let copy_stream = ctx
            .ctx
            .new_stream()
            .map_err(|e| anyhow::anyhow!("create host snapshot copy stream: {e}"))?;
        let source_ready = ctx
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map_err(|e| anyhow::anyhow!("create host snapshot source event: {e}"))?;
        let spill_events = (0..slots)
            .map(|_| {
                ctx.ctx
                    .new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))
                    .map_err(|e| anyhow::anyhow!("create host snapshot spill event: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            state,
            conv,
            copy_stream,
            source_ready,
            spill_events,
            slots,
            layers,
            state_per_layer,
            conv_per_layer,
            state_per_slot,
            conv_per_slot,
        })
    }

    pub(crate) fn spill_from(
        &mut self,
        ctx: &DeviceContext,
        src: &RecurrentState,
        slot: usize,
    ) -> Result<()> {
        anyhow::ensure!(slot < self.slots, "host snapshot slot {slot} out of range");
        anyhow::ensure!(
            src.layers.len() == self.layers,
            "host snapshot layer mismatch: src={}, expected={}",
            src.layers.len(),
            self.layers
        );
        let state = self.state.as_mut_slice()?;
        let conv = self.conv.as_mut_slice()?;
        let state_base = slot * self.state_per_slot;
        let conv_base = slot * self.conv_per_slot;
        self.source_ready.record(&ctx.stream)?;
        self.copy_stream.wait(&self.source_ready)?;
        for layer in 0..self.layers {
            let state_at = state_base + layer * self.state_per_layer;
            self.copy_stream.memcpy_dtoh(
                &src.layers[layer].state,
                &mut state[state_at..state_at + self.state_per_layer],
            )?;
            let conv_at = conv_base + layer * self.conv_per_layer;
            self.copy_stream.memcpy_dtoh(
                &src.layers[layer].conv_state.data,
                &mut conv[conv_at..conv_at + self.conv_per_layer],
            )?;
        }
        self.spill_events[slot].record(&self.copy_stream)?;
        Ok(())
    }

    pub(crate) fn restore_into(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        anyhow::ensure!(slot < self.slots, "host snapshot slot {slot} out of range");
        anyhow::ensure!(
            dst.layers.len() == self.layers,
            "host snapshot layer mismatch: dst={}, expected={}",
            dst.layers.len(),
            self.layers
        );
        let state = self.state.as_slice()?;
        let conv = self.conv.as_slice()?;
        ctx.stream.wait(&self.spill_events[slot])?;
        let state_base = slot * self.state_per_slot;
        let conv_base = slot * self.conv_per_slot;
        for layer in 0..self.layers {
            let state_at = state_base + layer * self.state_per_layer;
            ctx.stream.memcpy_htod(
                &state[state_at..state_at + self.state_per_layer],
                &mut dst.layers[layer].state,
            )?;
            let conv_at = conv_base + layer * self.conv_per_layer;
            ctx.stream.memcpy_htod(
                &conv[conv_at..conv_at + self.conv_per_layer],
                &mut dst.layers[layer].conv_state.data,
            )?;
        }
        Ok(())
    }

    fn wait_spill(&self, ctx: &DeviceContext, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.slots, "host snapshot slot {slot} out of range");
        ctx.stream.wait(&self.spill_events[slot])?;
        Ok(())
    }
}

/// Rank-local physical recurrent/conv snapshot allocations.
pub(crate) struct RecurrentStateStore {
    slots: Vec<RecurrentState>,
    host: Option<HostSnapshotSlab>,
    pending_spill: Vec<Option<usize>>,
}

impl RecurrentStateStore {
    pub(crate) fn new_with_host(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        gpu_slot_count: usize,
        host_slot_count: usize,
    ) -> Result<Self> {
        let mut slots = Vec::with_capacity(gpu_slot_count);
        for _ in 0..gpu_slot_count {
            slots.push(RecurrentState::new(ctx, config, geometry)?);
        }
        let host = if host_slot_count == 0 {
            None
        } else {
            Some(HostSnapshotSlab::new(
                ctx,
                config,
                geometry,
                host_slot_count,
            )?)
        };
        Ok(Self {
            pending_spill: vec![None; gpu_slot_count],
            slots,
            host,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn state(&self, slot: usize) -> Result<&RecurrentState> {
        self.slots
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))
    }

    pub(crate) fn save(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        src: &RecurrentState,
    ) -> Result<()> {
        let host_slot = self
            .pending_spill
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {slot} out of range"))?
            .take();
        if let Some(host_slot) = host_slot {
            self.host
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("snapshot store has no host slab"))?
                .wait_spill(ctx, host_slot)?;
        }
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
        let src = self.state(slot)?;
        dst.copy_from(ctx, src)
    }

    pub(crate) fn spill_to_host(
        &mut self,
        ctx: &DeviceContext,
        gpu_slot: usize,
        host_slot: usize,
    ) -> Result<()> {
        let src = self
            .slots
            .get(gpu_slot)
            .ok_or_else(|| anyhow::anyhow!("snapshot slot {gpu_slot} out of range"))?;
        self.host
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("snapshot store has no host slab"))?
            .spill_from(ctx, src, host_slot)?;
        self.pending_spill[gpu_slot] = Some(host_slot);
        Ok(())
    }

    pub(crate) fn restore_from_host(
        &self,
        ctx: &DeviceContext,
        host_slot: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        self.host
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("snapshot store has no host slab"))?
            .restore_into(ctx, host_slot, dst)
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
        Self::new_with_host(kv, snapshot_slots, 0)
    }

    /// Build the joint coordinator with a pinned-host snapshot budget.
    pub(crate) fn new_with_host(
        kv: KvCacheManager,
        snapshot_slots: usize,
        host_slots: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            SNAPSHOT_STRIDE_TOKENS.is_multiple_of(kv.pool().block_size()),
            "Qwen3.5 snapshot stride {SNAPSHOT_STRIDE_TOKENS} must be a multiple of KV block size {}",
            kv.pool().block_size()
        );
        Ok(Self {
            kv,
            snapshots: SnapshotCache::new(snapshot_slots, host_slots),
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

    pub(crate) fn snapshot_host_slots(&self) -> usize {
        self.snapshots.host_capacity()
    }

    pub(crate) fn snapshot_host_occupancy(&self) -> usize {
        self.snapshots.host_len()
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
        history_tokens: Option<usize>,
    ) -> usize {
        if !self.enabled() {
            return 0;
        }
        let hashes = self
            .kv
            .pool()
            .active_prefix_hashes(prompt_tokens.to_vec(), lora_name);
        let block_size = self.kv.pool().block_size();
        let resident_tokens = (hashes.len() * block_size).min(history_stop(
            prompt_tokens.len(),
            history_tokens,
            SNAPSHOT_STRIDE_TOKENS,
        ));
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
        history_tokens: Option<usize>,
        allow_match: bool,
    ) -> Result<(RequestKv, Option<SnapshotGuard>)> {
        let mut request =
            self.kv
                .pool()
                .new_request(prompt_tokens.to_vec(), max_output_tokens, lora_name);
        if !self.enabled() || !allow_match {
            return Ok((request, None));
        }
        self.stats.queries += prompt_tokens.len() as u64;

        let probe = self
            .kv
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_name);
        // Resuming past the history stop would skip it, so the next turn could not reuse it.
        let resident_tokens = (probe.reusable_blocks() * self.kv.pool().block_size()).min(
            history_stop(prompt_tokens.len(), history_tokens, SNAPSHOT_STRIDE_TOKENS),
        );
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
        let tier = guard.tier();
        self.stats.joint_hits += 1;
        self.stats.joint_hit_tokens += boundary as u64;
        if tier == SnapshotTier::Host {
            self.stats.host_hits += boundary as u64;
        }
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
        if let Some(first_token) = first_token {
            request.apply_prefill(first_token, self.kv.pool())
        } else {
            request.apply_prefill_chunk(self.kv.pool())
        }?;
        // Retain registered KV only when prefix caching is enabled.
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
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
    fn snapshot_key(
        &self,
        request: &RequestKv,
        boundary: usize,
    ) -> Result<Option<PrefixBoundaryKey>> {
        if !self.enabled() {
            return Ok(None);
        }
        if boundary == 0 || !boundary.is_multiple_of(SNAPSHOT_STRIDE_TOKENS) {
            return Ok(None);
        }
        let sequence_hash = request
            .registered_boundary_hash(boundary)
            .ok_or_else(|| anyhow::anyhow!("no registered KV hash at boundary {boundary}"))?;
        Ok(Some(PrefixBoundaryKey {
            sequence_hash,
            boundary_tokens: boundary,
        }))
    }

    /// Decide the next snapshot's slot/tier and its optional spill plan.
    pub(crate) fn plan_reserve_snapshot(
        &mut self,
        request: &RequestKv,
        boundary: usize,
    ) -> Result<Option<SnapshotReservePlan>> {
        let Some(key) = self.snapshot_key(request, boundary)? else {
            return Ok(None);
        };
        let kv = &self.kv;
        Ok(self
            .snapshots
            .reserve(key, &|key| kv.pool().has_resident_hash(key.sequence_hash)))
    }

    pub(crate) fn commit_reserve_snapshot(&mut self, plan: SnapshotReservePlan) {
        if plan.spill().is_some() {
            self.stats.spills += 1;
        }
        if plan.evicted_host() {
            self.stats.host_evictions += 1;
        }
        self.snapshots.commit_plan(plan);
    }

    pub(crate) fn abort_reserve_snapshot(&mut self, plan: SnapshotReservePlan) {
        self.snapshots.abort_plan(plan);
    }

    /// Publish after rank-local snapshot copies succeed.
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
        request.apply_decode(token, self.kv.pool())?;
        if !self.enabled() {
            request.mark_blocks_reset_on_release();
        }
        Ok(())
    }

    /// Roll back pages reserved by a scheduled step that did not apply.
    #[allow(clippy::unused_self)] // keep KV state transitions behind this facade
    pub(crate) fn revert_schedule(&self, request: &mut RequestKv) -> Result<()> {
        request.revert_schedule()
    }

    /// Best-effort rollback for every request scheduled by one failed step.
    pub(crate) fn revert_scheduled_requests<'a>(
        &self,
        requests: impl IntoIterator<Item = &'a mut RequestKv>,
    ) {
        for request in requests {
            if let Err(error) = self.revert_schedule(request) {
                log::warn!("failed to revert KV schedule: {error}");
            }
        }
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
    use super::Config35;
    use super::DeviceContext;
    use super::LocalGeometry;
    use super::PrefixBoundaryKey;
    use super::RecurrentState;
    use super::RecurrentStateStore;
    use super::SnapshotCache;
    use super::SnapshotTier;
    use super::eligible_boundaries;
    use crate::config::TensorParallelConfig;

    fn live(_: PrefixBoundaryKey) -> bool {
        true
    }

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
        let mut state = SnapshotCache::new(1, 0);
        let plan = state
            .reserve(key(1), &live)
            .expect("empty state must reserve");
        assert!(state.lookup(key(1)).is_none());
        state.commit_plan(plan);
        state.publish(plan.reservation());
        assert!(state.lookup(key(1)).is_some());
    }

    #[test]
    fn aborted_write_does_not_expose_partial_snapshot() {
        let mut state = SnapshotCache::new(1, 0);
        let plan = state
            .reserve(key(1), &live)
            .expect("empty state must reserve");
        state.abort_plan(plan);
        assert!(state.lookup(key(1)).is_none());
        assert!(state.reserve(key(2), &live).is_some());
    }

    #[test]
    fn duplicate_reservation_refreshes_lru() {
        let mut state = SnapshotCache::new(2, 0);
        for tag in [1, 2] {
            let plan = state.reserve(key(tag), &live).expect("free slot");
            state.commit_plan(plan);
            state.publish(plan.reservation());
        }
        assert!(state.reserve(key(1), &live).is_none());
        let plan = state.reserve(key(3), &live).expect("unpinned LRU victim");
        state.commit_plan(plan);
        state.publish(plan.reservation());
        assert!(state.lookup(key(1)).is_some());
        assert!(state.lookup(key(2)).is_none());
        assert!(state.lookup(key(3)).is_some());
    }

    #[test]
    fn lru_evicts_untouched_snapshot_but_preserves_touched_snapshot() {
        let mut state = SnapshotCache::new(2, 0);
        for tag in [1, 2] {
            let plan = state.reserve(key(tag), &live).expect("free slot");
            state.commit_plan(plan);
            state.publish(plan.reservation());
        }
        drop(state.lookup(key(1)).expect("key 1 should be present"));
        let plan = state.reserve(key(3), &live).expect("unpinned LRU victim");
        state.commit_plan(plan);
        state.publish(plan.reservation());
        assert!(state.lookup(key(1)).is_some());
        assert!(state.lookup(key(2)).is_none());
        assert!(state.lookup(key(3)).is_some());
    }

    #[test]
    fn all_pinned_snapshots_make_insertion_a_soft_miss() {
        let mut state = SnapshotCache::new(1, 0);
        let plan = state
            .reserve(key(1), &live)
            .expect("empty state must reserve");
        state.commit_plan(plan);
        state.publish(plan.reservation());
        let guard = state.lookup(key(1)).expect("key 1 should be present");
        assert!(state.reserve(key(2), &live).is_none());
        drop(guard);
        assert!(state.reserve(key(2), &live).is_some());
    }

    #[test]
    fn host_plan_commit_moves_victim_to_host() {
        let mut state = SnapshotCache::new(1, 1);
        let plan = state.reserve(key(1), &live).expect("free GPU slot");
        state.commit_plan(plan);
        state.publish(plan.reservation());
        let plan = state.reserve(key(2), &live).expect("spill plan");
        let spill = plan.spill().expect("host spill");
        assert_eq!(spill.gpu_slot, 0);
        assert_eq!(spill.host_slot, 0);
        state.commit_plan(plan);
        state.publish(plan.reservation());
        assert_eq!(
            state.lookup(key(1)).expect("victim").tier(),
            SnapshotTier::Host
        );
        assert_eq!(
            state.lookup(key(2)).expect("new entry").tier(),
            SnapshotTier::Gpu
        );
    }

    #[test]
    fn host_victim_is_counted_as_an_eviction() {
        let mut state = SnapshotCache::new(1, 1);
        for tag in [1, 2] {
            let plan = state.reserve(key(tag), &live).expect("reserve");
            state.commit_plan(plan);
            state.publish(plan.reservation());
        }
        let plan = state.reserve(key(3), &live).expect("host spill");
        assert!(plan.evicted_host());
        assert!(plan.reservation().was_replacement());
    }

    #[test]
    fn dead_victim_is_dropped_before_spill() {
        let mut state = SnapshotCache::new(1, 1);
        let plan = state.reserve(key(1), &live).expect("free GPU slot");
        state.commit_plan(plan);
        state.publish(plan.reservation());

        let dead = key(1);
        let plan = state
            .reserve(key(2), &|candidate| candidate != dead)
            .expect("dead victim frees the slot");
        assert!(plan.spill().is_none());
        assert!(state.lookup(dead).is_none());
        assert!(
            state
                .entries
                .values()
                .all(|entry| entry.tier != SnapshotTier::Host)
        );
        state.commit_plan(plan);
        state.publish(plan.reservation());
        assert!(state.lookup(key(2)).is_some());
    }

    #[test]
    fn host_plan_abort_restores_victim() {
        let mut state = SnapshotCache::new(1, 1);
        let plan = state.reserve(key(1), &live).expect("free GPU slot");
        state.commit_plan(plan);
        state.publish(plan.reservation());
        let plan = state.reserve(key(2), &live).expect("spill plan");
        state.abort_plan(plan);
        assert_eq!(
            state.lookup(key(1)).expect("victim").tier(),
            SnapshotTier::Gpu
        );
        assert!(state.lookup(key(2)).is_none());
    }

    #[test]
    #[ignore = "requires a CUDA device and a Qwen3.5 model directory (PEGAINFER_TEST_MODEL_PATH)"]
    fn host_slab_round_trips_every_layer() {
        let Some(model_path) =
            crate::test_fixture::model_path_or_skip("host_slab_round_trips_every_layer")
        else {
            return;
        };
        let ctx = DeviceContext::new_with_device(0).expect("create CUDA context");
        let config = Config35::from_file(&model_path).expect("load Qwen3.5 config");
        let geometry =
            LocalGeometry::try_new(&config, TensorParallelConfig::default()).expect("geometry");
        let layers = config.num_hidden_layers - config.num_full_attention_layers();
        let state_len = geometry.local_linear_num_value_heads()
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let conv_len = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);

        let mut store =
            RecurrentStateStore::new_with_host(&ctx, &config, geometry, 1, 1).expect("store");
        for layer in 0..layers {
            let host_state: Vec<f32> = (0..state_len)
                .map(|i| (layer * 1000 + i) as f32 * 0.125 - 7.0)
                .collect();
            let host_conv: Vec<half::bf16> = (0..conv_len)
                .map(|i| half::bf16::from_f32((layer * 100 + i) as f32 * 0.25 - 3.0))
                .collect();
            ctx.stream
                .memcpy_htod(&host_state, &mut store.slots[0].layers[layer].state)
                .expect("fill recurrent");
            ctx.stream
                .memcpy_htod(
                    &host_conv,
                    &mut store.slots[0].layers[layer].conv_state.data,
                )
                .expect("fill conv");
        }

        store.spill_to_host(&ctx, 0, 0).expect("spill");
        let mut restored = RecurrentState::new(&ctx, &config, geometry).expect("restore state");
        store
            .restore_from_host(&ctx, 0, &mut restored)
            .expect("restore from host");

        for layer in 0..layers {
            let expected_state: Vec<f32> = (0..state_len)
                .map(|i| (layer * 1000 + i) as f32 * 0.125 - 7.0)
                .collect();
            let expected_conv: Vec<half::bf16> = (0..conv_len)
                .map(|i| half::bf16::from_f32((layer * 100 + i) as f32 * 0.25 - 3.0))
                .collect();
            assert_eq!(
                ctx.stream
                    .clone_dtoh(&restored.layers[layer].state)
                    .expect("read recurrent"),
                expected_state,
                "recurrent layer {layer}"
            );
            assert_eq!(
                ctx.stream
                    .clone_dtoh(&restored.layers[layer].conv_state.data)
                    .expect("read conv"),
                expected_conv,
                "conv layer {layer}"
            );
        }
    }
}
