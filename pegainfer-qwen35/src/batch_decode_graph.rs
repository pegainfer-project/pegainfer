//! CUDA Graph state for Qwen3.5 batched decode with bucket padding.

use anyhow::Result;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::tensor::DeviceContext;

use super::config::Config35;
use super::config::TensorParallelConfig;
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
        tensor_parallel: TensorParallelConfig,
        kv_pool: &KvPool,
        max_batch: usize,
    ) -> Result<Self> {
        let padding_page_id = kv_pool.padding_page_id();
        let max_total_pages = kv_pool.capacity_pages();

        let buffers = BatchDecodeBuffers35::new(
            ctx,
            config,
            tensor_parallel,
            max_batch,
            max_total_pages,
            padding_page_id,
        )?;

        let mut slot_states = Vec::with_capacity(max_batch);
        for _ in 0..max_batch {
            slot_states.push(RecurrentState::new(ctx, config)?);
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
        anyhow::ensure!(
            slot_idx < self.slot_states.len(),
            "Qwen3.5 graph slot {slot_idx} exceeds capacity {}",
            self.slot_states.len()
        );
        let dst = &mut self.slot_states[slot_idx];
        dst.copy_from(ctx, src)
            .map_err(|e| anyhow::anyhow!("copy recurrent state to slot {slot_idx}: {e}"))?;
        Ok(())
    }

    /// D2D copy slot `slot_idx` recurrent state into a standalone state.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-private verifier substrate uses this before serving wiring"
        )
    )]
    pub(crate) fn copy_slot_to_state(
        &self,
        ctx: &DeviceContext,
        slot_idx: usize,
        dst: &mut RecurrentState,
    ) -> Result<()> {
        anyhow::ensure!(
            slot_idx < self.slot_states.len(),
            "Qwen3.5 graph slot {slot_idx} exceeds capacity {}",
            self.slot_states.len()
        );
        dst.copy_from(ctx, &self.slot_states[slot_idx])
            .map_err(|e| anyhow::anyhow!("copy recurrent slot {slot_idx} to state: {e}"))?;
        Ok(())
    }

    /// D2D copy one graph slot's recurrent/conv state into another slot.
    pub(crate) fn copy_slot_to_slot(
        &mut self,
        ctx: &DeviceContext,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            src_slot_idx < self.slot_states.len(),
            "Qwen3.5 recurrent source slot {src_slot_idx} out of range {}",
            self.slot_states.len()
        );
        anyhow::ensure!(
            dst_slot_idx < self.slot_states.len(),
            "Qwen3.5 recurrent destination slot {dst_slot_idx} out of range {}",
            self.slot_states.len()
        );
        if src_slot_idx == dst_slot_idx {
            return Ok(());
        }
        if src_slot_idx < dst_slot_idx {
            let (left, right) = self.slot_states.split_at_mut(dst_slot_idx);
            let src = &left[src_slot_idx];
            let dst = &mut right[0];
            dst.copy_from(ctx, src)
        } else {
            let (left, right) = self.slot_states.split_at_mut(src_slot_idx);
            let dst = &mut left[dst_slot_idx];
            let src = &right[0];
            dst.copy_from(ctx, src)
        }
        .map_err(|e| {
            anyhow::anyhow!("copy Qwen3.5 recurrent slot {src_slot_idx} to {dst_slot_idx}: {e}")
        })
    }
}
