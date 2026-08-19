//! CUDA Graph state for Qwen3.5 batched decode with bucket padding.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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

#[derive(Default)]
struct DecodeGraphEvidenceCounters {
    captures: AtomicU64,
    replays: AtomicU64,
    eager_fallbacks: AtomicU64,
    state_slot_copies: AtomicU64,
    state_slot_reuses: AtomicU64,
    slot_compactions: AtomicU64,
}

#[derive(Clone, Default)]
pub(crate) struct DecodeGraphEvidenceHandle {
    counters: Arc<DecodeGraphEvidenceCounters>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodeGraphEvidenceSnapshot {
    pub(crate) captures: u64,
    pub(crate) replays: u64,
    pub(crate) eager_fallbacks: u64,
    pub(crate) state_slot_copies: u64,
    pub(crate) state_slot_reuses: u64,
    pub(crate) slot_compactions: u64,
}

impl DecodeGraphEvidenceHandle {
    pub(crate) fn snapshot(&self) -> DecodeGraphEvidenceSnapshot {
        DecodeGraphEvidenceSnapshot {
            captures: self.counters.captures.load(Ordering::Relaxed),
            replays: self.counters.replays.load(Ordering::Relaxed),
            eager_fallbacks: self.counters.eager_fallbacks.load(Ordering::Relaxed),
            state_slot_copies: self.counters.state_slot_copies.load(Ordering::Relaxed),
            state_slot_reuses: self.counters.state_slot_reuses.load(Ordering::Relaxed),
            slot_compactions: self.counters.slot_compactions.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record_capture(&self) {
        self.counters.captures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_replay(&self) {
        self.counters.replays.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_eager_fallback(&self) {
        self.counters
            .eager_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_state_slot_copy(&self, reused: bool) {
        self.counters
            .state_slot_copies
            .fetch_add(1, Ordering::Relaxed);
        if reused {
            self.counters
                .state_slot_reuses
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_slot_compaction(&self) {
        self.counters
            .slot_compactions
            .fetch_add(1, Ordering::Relaxed);
    }
}

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
    pub(crate) evidence: DecodeGraphEvidenceHandle,
}

impl BatchDecodeGraphState {
    /// Create a graph state at `max_batch` slots (loader-validated bucket).
    pub(crate) fn with_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        tensor_parallel: TensorParallelConfig,
        kv_pool: &KvPool,
        max_batch: usize,
        evidence: DecodeGraphEvidenceHandle,
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
            evidence,
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
        let reused = dst.seq_len != 0;
        for (dst_layer, src_layer) in dst.layers.iter_mut().zip(src.layers.iter()) {
            ctx.stream
                .memcpy_dtod(&src_layer.state, &mut dst_layer.state)
                .map_err(|e| anyhow::anyhow!("copy recurrent state to slot {slot_idx}: {e}"))?;
            ctx.stream
                .memcpy_dtod(&src_layer.conv_state.data, &mut dst_layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("copy conv state to slot {slot_idx}: {e}"))?;
        }
        dst.seq_len = src.seq_len;
        self.evidence.record_state_slot_copy(reused);
        Ok(())
    }

    pub(crate) fn record_slot_compaction(&self) {
        self.evidence.record_slot_compaction();
    }
}
