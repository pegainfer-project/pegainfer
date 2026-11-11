//! Paged draft KV, addressed by the target KV's block IDs.
//!
//! The target radix cache decides which blocks survive; the draft pool uses the
//! same block IDs, so a target prefix hit also carries the draft K/V for that
//! prefix. A block is materialized before it is registered, so every matchable
//! block has all of its rows written; the executor keeps a per-block content tag
//! to reject blocks whose rows were never materialized (a request that ran
//! without a drafter state).

use anyhow::Result;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

pub(crate) struct DFlashDraftKvPool {
    /// One `[kv_dim, total_blocks * block_size]` tensor per draft layer.
    pub(crate) k: Vec<HiddenStates>,
    pub(crate) v: Vec<HiddenStates>,
    pub(crate) block_size: usize,
    pub(crate) total_blocks: usize,
}

impl DFlashDraftKvPool {
    pub(crate) fn new(
        ctx: &DeviceContext,
        layers: usize,
        kv_dim: usize,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            layers > 0 && kv_dim > 0 && total_blocks > 0 && block_size > 0,
            "DFlash draft KV pool needs positive geometry"
        );
        let slots = total_blocks
            .checked_mul(block_size)
            .ok_or_else(|| anyhow::anyhow!("DFlash draft KV pool slot count overflow"))?;
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        for _ in 0..layers {
            k.push(HiddenStates::zeros(ctx, kv_dim, slots)?);
            v.push(HiddenStates::zeros(ctx, kv_dim, slots)?);
        }
        Ok(Self {
            k,
            v,
            block_size,
            total_blocks,
        })
    }

    pub(crate) fn slots(&self) -> usize {
        self.total_blocks * self.block_size
    }
}
