use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use half::bf16;

use crate::KvLayout;

struct Inner {
    buffer: CudaSlice<bf16>,
    layout: KvLayout,
    num_blocks: usize,
}

/// GPU KV cache buffer without an allocator.
///
/// Owns the device memory and layout geometry but delegates block
/// allocation to an external `BlockManager` (kvbm-logical).
#[derive(Clone)]
pub struct KvBuffer {
    inner: Arc<Inner>,
}

impl KvBuffer {
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_blocks * layout.page_stride;
        let buffer: CudaSlice<bf16> = stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("KvBuffer alloc failed: {e}"))?;
        Ok(Self {
            inner: Arc::new(Inner {
                buffer,
                layout,
                num_blocks,
            }),
        })
    }

    pub fn layout(&self) -> &KvLayout {
        &self.inner.layout
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        &self.inner.buffer
    }

    /// Base device address of the fused KV buffer.
    ///
    /// Stable for the buffer's lifetime — cudarc allocations don't move — so
    /// the KV-offload connector registers this once with pegaflow and the
    /// page-first [`KvLayout`] strides reach every (layer, block, K/V) segment
    /// from it. The returned address outlives the transient stream-ordering
    /// guard precisely because the `Arc<Inner>` keeps the slice alive.
    pub fn device_ptr(&self, stream: &CudaStream) -> u64 {
        let (ptr, _guard) = self.inner.buffer.device_ptr(stream);
        ptr
    }

    pub fn num_blocks(&self) -> usize {
        self.inner.num_blocks
    }

    /// Stream-ordered device-to-device copy of one whole page (every layer's
    /// K/V for `page_size` tokens — one contiguous `page_stride` range in the
    /// page-first layout). Both pages must lie inside the buffer; scratch
    /// pages past the pool's block count are valid targets.
    pub fn copy_page(
        &self,
        stream: &CudaStream,
        src_page: usize,
        dst_page: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            src_page < self.inner.num_blocks && dst_page < self.inner.num_blocks,
            "KV page copy out of range: {src_page} -> {dst_page} of {}",
            self.inner.num_blocks
        );
        if src_page == dst_page {
            return Ok(());
        }
        let stride = self.inner.layout.page_stride;
        let elem = std::mem::size_of::<bf16>();
        let (base, _guard) = self.inner.buffer.device_ptr(stream);
        let src = base + (src_page * stride * elem) as u64;
        let dst = base + (dst_page * stride * elem) as u64;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, stride * elem, stream.cu_stream())
        }
        .map_err(|e| anyhow::anyhow!("KV page copy failed: {e}"))
    }
}
