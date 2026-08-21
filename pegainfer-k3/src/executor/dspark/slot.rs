//! Per-slot DSpark draft state: the slot's draft KV, its pending captured
//! context rows, and the projected context pair. A child module of `dspark`
//! (fields are `pub(super)`) split out for the module size budget.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use super::DSPARK_KV_DIM;
use super::DSPARK_LAYERS;
use super::K3_DSPARK_BLOCK;
use super::K3_DSPARK_CONTEXT_DIM;
use super::K3_HIDDEN;

pub(super) struct DsparkLayerKv {
    pub(super) k: HiddenStates,
    pub(super) v: HiddenStates,
}

/// Per-slot draft state: the draft KV over committed tokens, the pending
/// captured-context rows not yet projected, and the per-round projected
/// context (persists across the layer loop, so it lives here, not in the
/// shared scratch). Everything is preallocated to `cache_len` at load — a
/// mid-serving draft round must never hit the allocator.
pub(crate) struct K3DsparkSlotState {
    pub(super) layers: Vec<DsparkLayerKv>,
    /// Captured target hidden `[pending_len, 35840]` awaiting projection.
    pub(super) pending: HiddenStates,
    pub(super) pending_len: usize,
    pub(super) committed_len: usize,
    pub(super) context_projected: HiddenStates,
    pub(super) context_hidden: HiddenStates,
    /// The drafter's KV capacity ([`super::K3DsparkModel::cache_len`]) — the
    /// pending-context growth cap and the overflow guard bound.
    cache_len: usize,
}

impl K3DsparkSlotState {
    /// Device bytes one slot pins at `cache_len` (the `[cache_len, 35840]`
    /// pending slab dominates) — what arming actually costs per batch slot.
    pub(crate) fn device_bytes(cache_len: usize) -> usize {
        let per_token = DSPARK_LAYERS * 2 * DSPARK_KV_DIM + K3_DSPARK_CONTEXT_DIM + 2 * K3_HIDDEN;
        cache_len * per_token * size_of::<bf16>()
    }

    pub(crate) fn new(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            layers.push(DsparkLayerKv {
                k: HiddenStates::zeros(ctx, DSPARK_KV_DIM, cache_len)?,
                v: HiddenStates::zeros(ctx, DSPARK_KV_DIM, cache_len)?,
            });
        }
        let mut pending = HiddenStates::zeros(ctx, K3_DSPARK_CONTEXT_DIM, cache_len)?;
        pending.seq_len = 0;
        Ok(Self {
            layers,
            pending,
            pending_len: 0,
            committed_len: 0,
            context_projected: HiddenStates::zeros(ctx, K3_HIDDEN, cache_len)?,
            context_hidden: HiddenStates::zeros(ctx, K3_HIDDEN, cache_len)?,
            cache_len,
        })
    }

    /// Clear the slot for a new request. The KV/pending contents need no
    /// scrubbing: `committed_len`/`pending_len` gate every read, and new
    /// rows overwrite in place.
    pub(crate) fn reset(&mut self) {
        self.committed_len = 0;
        self.pending_len = 0;
        self.pending.seq_len = 0;
    }

    /// Append `count` consecutive captured rows (each `[35840]`, starting at
    /// `first_row` of the step's capture slab) to the pending context. The
    /// buffer holds `cache_len` rows from birth — allocation-free by
    /// construction.
    pub(crate) fn append_captured_rows(
        &mut self,
        ctx: &DeviceContext,
        captured: &CudaSlice<bf16>,
        first_row: usize,
        count: usize,
    ) -> Result<()> {
        let required = self.pending_len + count;
        ensure!(
            self.committed_len + required + K3_DSPARK_BLOCK <= self.cache_len,
            "dspark pending context would exceed the draft cache: committed={}, pending={required}",
            self.committed_len
        );
        let src = captured
            .slice(first_row * K3_DSPARK_CONTEXT_DIM..(first_row + count) * K3_DSPARK_CONTEXT_DIM);
        let mut dst = self
            .pending
            .data
            .slice_mut(self.pending_len * K3_DSPARK_CONTEXT_DIM..required * K3_DSPARK_CONTEXT_DIM);
        ctx.stream.memcpy_dtod(&src, &mut dst)?;
        self.pending_len = required;
        self.pending.seq_len = required;
        Ok(())
    }

    /// Point the projected-context pair at this round's rows. Preallocated to
    /// `cache_len` — the bound is already enforced by the caller's overflow
    /// guard, so exceeding it here is a bug, not a growth request.
    pub(super) fn set_context_len(&mut self, context_len: usize) -> Result<()> {
        ensure!(
            context_len <= self.context_projected.data.len() / K3_HIDDEN,
            "dspark context length {context_len} exceeds the preallocated cap"
        );
        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        Ok(())
    }
}
