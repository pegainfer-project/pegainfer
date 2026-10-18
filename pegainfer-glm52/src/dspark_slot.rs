//! Per-slot DSpark draft state: the slot's draft KV, its pending captured
//! context rows, and the projected context pair. A child module of `dspark`
//! (fields are `pub(super)`) split out for the module size budget.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use super::DSPARK_LAYERS;
use super::DSPARK_QKV_DIM;
use super::GLM52_DSPARK_BLOCK;
use super::GLM52_DSPARK_CONTEXT_DIM;
use super::GLM52_HIDDEN;

pub(super) struct DsparkLayerKv {
    pub(super) k: HiddenStates,
    pub(super) v: HiddenStates,
}

/// Per-slot draft state: the draft KV over committed tokens, the pending
/// captured-context rows not yet projected, and the per-round projected
/// context (persists across the layer loop, so it lives here, not in the
/// shared scratch). Everything is preallocated to `cache_len` at load — a
/// mid-serving draft round must never hit the allocator (a transient OOM
/// there would tear the whole engine down), and the launch-time VRAM probe
/// already charged the full-cap footprint ([`glm52_dspark_arena_bytes`]).
pub(crate) struct Glm52DsparkSlotState {
    pub(super) layers: Vec<DsparkLayerKv>,
    /// Captured target hidden `[pending_len, 30720]` awaiting projection.
    pub(super) pending: HiddenStates,
    pub(super) pending_len: usize,
    pub(super) committed_len: usize,
    pub(super) context_projected: HiddenStates,
    pub(super) context_hidden: HiddenStates,
    /// The drafter's KV capacity ([`Glm52DsparkModel::cache_len`]) — the
    /// pending-context growth cap and the overflow guard bound.
    cache_len: usize,
}

impl Glm52DsparkSlotState {
    pub(crate) fn new(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            layers.push(DsparkLayerKv {
                k: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, cache_len)?,
                v: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, cache_len)?,
            });
        }
        let mut pending = HiddenStates::zeros(ctx, GLM52_DSPARK_CONTEXT_DIM, cache_len)?;
        pending.seq_len = 0;
        Ok(Self {
            layers,
            pending,
            pending_len: 0,
            committed_len: 0,
            context_projected: HiddenStates::zeros(ctx, GLM52_HIDDEN, cache_len)?,
            context_hidden: HiddenStates::zeros(ctx, GLM52_HIDDEN, cache_len)?,
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

    /// Append one step row's captured hidden (a `[30720]` row of the step
    /// capture buffer) to the pending context. The buffer holds `cache_len`
    /// rows from birth — allocation-free by construction.
    pub(crate) fn append_captured_row(
        &mut self,
        ctx: &DeviceContext,
        captured: &CudaSlice<bf16>,
        row: usize,
    ) -> Result<()> {
        let required = self.pending_len + 1;
        ensure!(
            self.committed_len + required + GLM52_DSPARK_BLOCK <= self.cache_len,
            "dspark pending context would exceed the draft cache: committed={}, pending={required}",
            self.committed_len
        );
        let src =
            captured.slice(row * GLM52_DSPARK_CONTEXT_DIM..(row + 1) * GLM52_DSPARK_CONTEXT_DIM);
        let mut dst = self.pending.data.slice_mut(
            self.pending_len * GLM52_DSPARK_CONTEXT_DIM..required * GLM52_DSPARK_CONTEXT_DIM,
        );
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
            context_len <= self.context_projected.data.len() / GLM52_HIDDEN,
            "dspark context length {context_len} exceeds the preallocated cap"
        );
        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        Ok(())
    }
}
