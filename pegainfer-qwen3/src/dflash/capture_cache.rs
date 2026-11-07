//! Prefix-block capture cache for the DFlash/DSpark drafter.
//!
//! The prefix KV cache reuses the target's KV, but the drafter also consumes
//! target hidden rows captured during prefill/verify. Those rows are not
//! derivable from K/V, so a cache hit has to carry them alongside the block.
//! Entries are keyed by the KV block's `SequenceHash` (the same lineage hash
//! the prefix cache matches on) and hold one KV block's worth of capture rows.
//!
//! The cache is lane-owned and single-threaded: restore/publish run on the
//! worker thread that owns the drafter, so no pinning is needed.

use std::collections::HashMap;

use anyhow::Result;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

struct Entry {
    slot: usize,
    last_used: u64,
}

pub(crate) struct DFlashCaptureCache {
    /// `[capture_dim, slots * block_size]`; slot `s` owns rows
    /// `[s * block_size, (s + 1) * block_size)`.
    slots: HiddenStates,
    entries: HashMap<[u8; 16], Entry>,
    free_slots: Vec<usize>,
    clock: u64,
    slot_count: usize,
    block_size: usize,
}

impl DFlashCaptureCache {
    /// Allocate the cache from a fixed byte budget. `None` when the budget is
    /// zero or smaller than one block.
    pub(crate) fn new(
        ctx: &DeviceContext,
        capture_dim: usize,
        block_size: usize,
        budget_bytes: usize,
    ) -> Result<Option<Self>> {
        if budget_bytes == 0 || capture_dim == 0 || block_size == 0 {
            return Ok(None);
        }
        let slot_bytes = capture_dim
            .checked_mul(block_size)
            .and_then(|v| v.checked_mul(std::mem::size_of::<half::bf16>()))
            .ok_or_else(|| anyhow::anyhow!("DFlash capture cache slot size overflow"))?;
        let slot_count = budget_bytes / slot_bytes;
        if slot_count == 0 {
            log::warn!(
                "Qwen3 DFlash capture cache budget {} bytes is smaller than one {}-token block \
                 ({} bytes); prefix caching stays off with the drafter",
                budget_bytes,
                block_size,
                slot_bytes
            );
            return Ok(None);
        }
        let slots = HiddenStates::zeros(ctx, capture_dim, slot_count * block_size)?;
        Ok(Some(Self {
            slots,
            entries: HashMap::with_capacity(slot_count),
            free_slots: (0..slot_count).rev().collect(),
            clock: 0,
            slot_count,
            block_size,
        }))
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slot_count
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    /// Look up every hash and return the matching slots, newest-use first.
    /// All-or-nothing: a partial run is a miss, so a restore never seeds half
    /// a prefix.
    pub(crate) fn lookup(&mut self, hashes: &[[u8; 16]]) -> Option<Vec<usize>> {
        let last_used = self.tick();
        let mut slots = Vec::with_capacity(hashes.len());
        for hash in hashes {
            let entry = self.entries.get_mut(hash)?;
            entry.last_used = last_used;
            slots.push(entry.slot);
        }
        Some(slots)
    }

    /// Copy one block of `src` into the cache under `hash`. Returns the evicted
    /// hash when the LRU had to make room.
    pub(crate) fn insert(
        &mut self,
        ctx: &DeviceContext,
        hash: [u8; 16],
        src: &HiddenStates,
        src_token_offset: usize,
    ) -> Result<Option<[u8; 16]>> {
        let last_used = self.tick();
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.last_used = last_used;
            let slot = entry.slot;
            self.copy_into(ctx, slot, src, src_token_offset)?;
            return Ok(None);
        }

        let (slot, evicted) = if let Some(slot) = self.free_slots.pop() {
            (slot, None)
        } else {
            let (&victim, slot) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(hash, entry)| (hash, entry.slot))
                .ok_or_else(|| anyhow::anyhow!("DFlash capture cache has no slots"))?;
            self.entries.remove(&victim);
            (slot, Some(victim))
        };
        self.copy_into(ctx, slot, src, src_token_offset)?;
        self.entries.insert(hash, Entry { slot, last_used });
        Ok(evicted)
    }

    /// Copy a cached block into `dst` at `dst_token_offset`.
    pub(crate) fn copy_block(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        dst: &mut HiddenStates,
        dst_token_offset: usize,
    ) -> Result<()> {
        let src_offset = slot * self.block_size;
        anyhow::ensure!(
            slot < self.slot_count,
            "DFlash capture slot {slot} out of range {}",
            self.slot_count
        );
        ops::copy_hidden_token_range_into(
            ctx,
            &self.slots,
            src_offset,
            dst,
            dst_token_offset,
            self.block_size,
        )
    }

    fn copy_into(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        src: &HiddenStates,
        src_token_offset: usize,
    ) -> Result<()> {
        ops::copy_hidden_token_range_into(
            ctx,
            src,
            src_token_offset,
            &mut self.slots,
            slot * self.block_size,
            self.block_size,
        )
    }
}
