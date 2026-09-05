use std::sync::Arc;

use cudarc::driver::CudaStream;
use kvbm_logical::events::KvCacheEvent;
use tokio::sync::broadcast;

use crate::buffer::KvBuffer;
use crate::pool::BlockPool;

/// Engine-level KV cache for full-attention models: one logical
/// [`BlockPool`] + one GPU [`KvBuffer`] with matching geometry.
///
/// MLA models (kimi-k2) consume [`BlockPool`] directly and own their
/// physical buffers — this facade is the full-attention pairing only.
pub struct KvCacheManager {
    pool: BlockPool,
    buffer: KvBuffer,
}

impl KvCacheManager {
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_scratch_pages(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            0,
        )
    }

    /// Like [`new`](Self::new), plus `scratch_pages` extra pages in the GPU
    /// buffer that the pool never hands out. Their page ids are
    /// `num_blocks..num_blocks + scratch_pages`; ownership and content are the
    /// caller's business (e.g. the speculative hedge's alternative-chain verify
    /// KV). The transaction layer is unaware of them by construction.
    pub fn new_with_scratch_pages(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        scratch_pages: usize,
    ) -> anyhow::Result<Self> {
        let buffer = KvBuffer::new(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks.checked_add(scratch_pages).ok_or_else(|| {
                anyhow::anyhow!("KV pages {num_blocks} + scratch {scratch_pages} overflow")
            })?,
        )?;
        let pool = BlockPool::new(block_size, num_blocks)?;
        Ok(Self { pool, buffer })
    }

    /// Like [`new`](Self::new) but the pool emits KV block events; returns the
    /// receiver to drain. See [`BlockPool::with_events`].
    pub fn new_with_events(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<(Self, broadcast::Receiver<KvCacheEvent>)> {
        let buffer = KvBuffer::new(
            stream,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
        )?;
        let (pool, events) = BlockPool::with_events(block_size, num_blocks)?;
        Ok((Self { pool, buffer }, events))
    }

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    pub fn buffer(&self) -> &KvBuffer {
        &self.buffer
    }
}
