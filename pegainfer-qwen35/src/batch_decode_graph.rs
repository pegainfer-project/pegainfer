//! CUDA Graph state for Qwen3.5 batched decode with bucket padding.

use std::sync::Arc;

use anyhow::Result;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::tensor::DeviceContext;

use super::config::Config35;
use super::config::LocalGeometry;
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::LinearStatePointerTables;
use super::recurrent_state::RecurrentState;

/// Bucket sizes for CUDA Graph capture. Actual batch is padded to nearest bucket.
pub(crate) const BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Maximum supported batch size (= largest bucket).
pub(crate) const MAX_BATCH: usize = 64;

/// Find the smallest bucket >= `bs`. Panics if `bs` > MAX_BATCH.
pub(crate) fn bucket_for(bs: usize) -> usize {
    for &b in BATCH_BUCKETS {
        if b >= bs {
            return b;
        }
    }
    panic!(
        "batch size {bs} exceeds largest bucket {}",
        BATCH_BUCKETS.last().unwrap()
    );
}

/// Per-cohort Qwen3.5 CUDA Graph decode state.
///
/// - In single-GPU mode (`with_capacity_and_padding`): pre-allocates `slot_capacity`
///   request `RecurrentState` slots plus a shared `padding_state`; decode refreshes the
///   fixed-address pointer table by active slot index and points padding rows to
///   `padding_state`, avoiding D2D compaction on retirement and preventing padding rows
///   from clobbering slots currently in prefill.
/// - In TP mode (`with_capacity`): keeps the existing single-cohort dense `0..bs` layout
///   and `move_slot_within`.
pub(crate) struct BatchDecodeGraphState {
    pub(crate) buffers: BatchDecodeBuffers35,
    pub(crate) slot_states: Vec<RecurrentState>,
    slot_state_ptrs: Vec<Vec<u64>>,
    slot_conv_ptrs: Vec<Vec<u64>>,
    /// Shared single-GPU padding state (prevents padding rows from writing active/prefilling slots); `None` on TP.
    padding_state: Option<Arc<RecurrentState>>,
    padding_state_ptrs: Vec<u64>,
    padding_conv_ptrs: Vec<u64>,
    pub(crate) linear_pointer_tables: LinearStatePointerTables,
    /// One `CudaGraphState` per BATCH_BUCKETS entry (indexed by position).
    pub(crate) graphs: Vec<CudaGraphState>,
}

impl BatchDecodeGraphState {
    /// TP-compatible constructor: allocates `max_batch` fixed slots and initializes pointer tables to `0..max_batch`.
    pub(crate) fn with_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        page_size: usize,
        padding_page_id: i32,
        max_batch: usize,
    ) -> Result<Self> {
        let buffers = BatchDecodeBuffers35::new(
            ctx,
            config,
            geometry,
            max_batch,
            page_size,
            padding_page_id,
        )?;

        let mut slot_states = Vec::with_capacity(max_batch);
        let mut slot_state_ptrs = Vec::with_capacity(max_batch);
        let mut slot_conv_ptrs = Vec::with_capacity(max_batch);
        for _ in 0..max_batch {
            let state = RecurrentState::new(ctx, config, geometry)?;
            let (s_ptrs, c_ptrs) = state.layer_device_ptrs(ctx);
            slot_state_ptrs.push(s_ptrs);
            slot_conv_ptrs.push(c_ptrs);
            slot_states.push(state);
        }
        let linear_pointer_tables = {
            let mut slot_refs: Vec<&mut RecurrentState> = slot_states.iter_mut().collect();
            LinearStatePointerTables::from_recurrent_refs(
                ctx,
                config,
                &mut slot_refs,
                max_batch,
                "Qwen3.5 graph",
            )?
        };

        let graphs = BATCH_BUCKETS
            .iter()
            .map(|_| CudaGraphState::new())
            .collect();

        Ok(Self {
            buffers,
            slot_states,
            slot_state_ptrs,
            slot_conv_ptrs,
            padding_state: None,
            padding_state_ptrs: Vec::new(),
            padding_conv_ptrs: Vec::new(),
            linear_pointer_tables,
            graphs,
        })
    }

    /// Single-GPU cohort constructor: allocates `slot_capacity` request slots, sizes decode buffers
    /// and pointer tables to `bucket_for(slot_capacity)`, and isolates padding writes via `padding_state`.
    pub(crate) fn with_capacity_and_padding(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        page_size: usize,
        padding_page_id: i32,
        slot_capacity: usize,
        padding_state: Arc<RecurrentState>,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=MAX_BATCH).contains(&slot_capacity),
            "Qwen3.5 cohort slot_capacity must be in 1..={MAX_BATCH}, got {slot_capacity}"
        );
        let graph_bucket = bucket_for(slot_capacity);
        let buffers = BatchDecodeBuffers35::new(
            ctx,
            config,
            geometry,
            graph_bucket,
            page_size,
            padding_page_id,
        )?;

        let mut slot_states = Vec::with_capacity(slot_capacity);
        let mut slot_state_ptrs = Vec::with_capacity(slot_capacity);
        let mut slot_conv_ptrs = Vec::with_capacity(slot_capacity);
        for _ in 0..slot_capacity {
            let state = RecurrentState::new(ctx, config, geometry)?;
            let (s_ptrs, c_ptrs) = state.layer_device_ptrs(ctx);
            slot_state_ptrs.push(s_ptrs);
            slot_conv_ptrs.push(c_ptrs);
            slot_states.push(state);
        }

        let (padding_state_ptrs, padding_conv_ptrs) = padding_state.layer_device_ptrs(ctx);
        let linear_pointer_tables =
            LinearStatePointerTables::with_capacity(ctx, config, graph_bucket, "Qwen3.5 graph")?;

        let graphs = BATCH_BUCKETS
            .iter()
            .map(|_| CudaGraphState::new())
            .collect();

        Ok(Self {
            buffers,
            slot_states,
            slot_state_ptrs,
            slot_conv_ptrs,
            padding_state: Some(padding_state),
            padding_state_ptrs,
            padding_conv_ptrs,
            linear_pointer_tables,
            graphs,
        })
    }

    /// Syncs this cohort's active slots to the fixed-address linear pointer table (skips H2D when unchanged).
    /// Active in single-GPU pooled mode only; TP mode keeps the static `0..max_batch` mapping.
    pub(crate) fn sync_active_slots(
        &mut self,
        ctx: &DeviceContext,
        active_slots: &[usize],
        padded_bs: usize,
    ) -> Result<()> {
        if self.padding_state.is_none() {
            debug_assert!(
                active_slots
                    .iter()
                    .enumerate()
                    .all(|(row, &slot)| slot == row),
                "TP fixed-layout graph expects dense 0..bs slots"
            );
            return Ok(());
        }
        self.linear_pointer_tables.sync_slots(
            ctx,
            &self.slot_state_ptrs,
            &self.slot_conv_ptrs,
            active_slots,
            &self.padding_state_ptrs,
            &self.padding_conv_ptrs,
            padded_bs,
            "Qwen3.5 graph",
        )
    }

    /// Zeros the given slot in place so the next admission can write directly into it.
    pub(crate) fn reset_slot(&mut self, ctx: &DeviceContext, slot_idx: usize) -> Result<()> {
        self.slot_states[slot_idx].reset(ctx)
    }

    /// D2D copy `src` recurrent state into `slot_idx` (TP graph mode).
    pub(crate) fn copy_state_to_slot(
        &mut self,
        ctx: &DeviceContext,
        src: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        self.slot_states[slot_idx]
            .copy_from(ctx, src)
            .map_err(|e| anyhow::anyhow!("copy recurrent state to slot {slot_idx}: {e}"))
    }

    /// D2D move slot `from` into `to` for TP mid-batch retirement slot compaction.
    pub(crate) fn move_slot_within(
        &mut self,
        ctx: &DeviceContext,
        from: usize,
        to: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            from != to,
            "TP slot compaction move {from} -> {to} is a no-op"
        );
        anyhow::ensure!(
            from < self.slot_states.len() && to < self.slot_states.len(),
            "TP slot compaction move {from} -> {to} exceeds {} slots",
            self.slot_states.len()
        );
        let (lo, hi) = (from.min(to), from.max(to));
        let (left, right) = self.slot_states.split_at_mut(hi);
        let (src, dst) = if from < to {
            (&left[from], &mut right[0])
        } else {
            (&right[0], &mut left[lo])
        };
        anyhow::ensure!(
            src.layers.len() == dst.layers.len(),
            "TP slot compaction layer count mismatch: {} vs {}",
            src.layers.len(),
            dst.layers.len()
        );
        for (src_layer, dst_layer) in src.layers.iter().zip(dst.layers.iter_mut()) {
            ctx.stream
                .memcpy_dtod(&src_layer.state, &mut dst_layer.state)
                .map_err(|e| anyhow::anyhow!("move recurrent state slot {from} -> {to}: {e}"))?;
            ctx.stream
                .memcpy_dtod(&src_layer.conv_state.data, &mut dst_layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("move conv state slot {from} -> {to}: {e}"))?;
        }
        let seq_len = src.seq_len;
        let dst = &mut self.slot_states[to];
        dst.seq_len = seq_len;
        Ok(())
    }
}
