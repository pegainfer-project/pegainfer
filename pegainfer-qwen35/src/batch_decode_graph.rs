//! CUDA Graph state for Qwen3.5 batched decode with bucket padding.

use anyhow::Result;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvPool;
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

/// CUDA Graph state for Qwen3.5 batch decode.
///
/// Owns one pre-allocated `RecurrentState` slot per decode-batch capacity
/// and shared decode buffers. Slot `i` always maps to position `i` in the batch — when a request
/// occupies slot `i`, its recurrent state lives at `slot_states[i]` for the
/// entire lifetime of that request in the batch. The CUDA Graph captured for
/// a given bucket size always accesses `slot_states[0..bucket_size]`, so GPU
/// pointer addresses are identical on every replay.
///
/// # Slot management
///
/// Callers must ensure that positions 0..batch_size are active requests and
/// positions batch_size..padded_batch_size are padding. When a request
/// finishes mid-batch, move the last slot's data to fill the gap:
/// ```text
/// copy_state_to_slot(ctx, last_slot_src, vacated_slot_idx)
/// ```
/// and update the caller's slot-to-request mapping accordingly.
pub(crate) struct BatchDecodeGraphState {
    pub(crate) buffers: BatchDecodeBuffers35,
    pub(crate) slot_states: Vec<RecurrentState>,
    pub(crate) linear_pointer_tables: LinearStatePointerTables,
    /// One `CudaGraphState` per BATCH_BUCKETS entry (indexed by position).
    pub(crate) graphs: Vec<CudaGraphState>,
}

impl BatchDecodeGraphState {
    /// Create a graph state at `max_batch` slots (loader-validated bucket).
    pub(crate) fn with_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        kv_pool: &KvPool,
        max_batch: usize,
    ) -> Result<Self> {
        let padding_page_id = kv_pool.padding_page_id();
        let max_total_pages = kv_pool.capacity_pages();

        let buffers = BatchDecodeBuffers35::new(
            ctx,
            config,
            geometry,
            max_batch,
            max_total_pages,
            padding_page_id,
        )?;

        let mut slot_states = Vec::with_capacity(max_batch);
        for _ in 0..max_batch {
            slot_states.push(RecurrentState::new(ctx, config, geometry)?);
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
            linear_pointer_tables,
            graphs,
        })
    }

    /// D2D copy `src` recurrent state into slot `slot_idx`.
    ///
    /// Call once when a request joins the batch (after prefill finishes).
    /// After this call, `slot_states[slot_idx]` IS the canonical state; the
    /// original `src` is no longer used.
    pub(crate) fn copy_state_to_slot(
        &mut self,
        ctx: &DeviceContext,
        src: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        let dst = &mut self.slot_states[slot_idx];
        for (dst_layer, src_layer) in dst.layers.iter_mut().zip(src.layers.iter()) {
            ctx.stream
                .memcpy_dtod(&src_layer.state, &mut dst_layer.state)
                .map_err(|e| anyhow::anyhow!("copy recurrent state to slot {slot_idx}: {e}"))?;
            ctx.stream
                .memcpy_dtod(&src_layer.conv_state.data, &mut dst_layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("copy conv state to slot {slot_idx}: {e}"))?;
        }
        dst.seq_len = src.seq_len;
        Ok(())
    }

    /// D2D move slot `from` into slot `to`, leaving `to` as the canonical
    /// state. Used by TP slot compaction when a mid-batch request retires:
    /// the last occupied slot moves into the vacated slot so decode rows stay
    /// dense (`0..bs`). Slot `from` keeps stale bytes afterwards; admission
    /// overwrites it (`copy_state_to_slot`), and padding rows may clobber it —
    /// both benign by design.
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
