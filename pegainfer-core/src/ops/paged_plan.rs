use std::ops::Deref;

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::kv_pool::KvDesc;
use crate::tensor::DeviceContext;

/// Host-side CSR for the split-KV decode kernel: padded request/chunk indices,
/// the per-slot validity mask, and the per-request chunk offsets.
pub struct SplitKvCsr {
    pub request_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub block_valid_mask: Vec<u8>,
    pub o_indptr: Vec<i32>,
}

/// Build the split-KV CSR for `kv_lens` at a fixed `chunk_size`, padded to
/// `padded_bs * cap` slots. Errors if a request needs more than `cap` chunks.
pub fn build_split_kv_csr(
    chunk_size: usize,
    cap: usize,
    kv_lens: &[usize],
    padded_bs: usize,
) -> Result<SplitKvCsr> {
    anyhow::ensure!(chunk_size > 0, "split-KV chunk_size must be > 0");
    anyhow::ensure!(cap > 0, "split-KV cap must be > 0");
    anyhow::ensure!(
        kv_lens.len() <= padded_bs,
        "kv_lens length {} exceeds padded batch {padded_bs}",
        kv_lens.len()
    );
    let padded_slots = padded_bs * cap;
    let mut request_indices = Vec::with_capacity(padded_slots);
    let mut kv_tile_indices = Vec::with_capacity(padded_slots);
    let mut block_valid_mask = Vec::with_capacity(padded_slots);
    let mut o_indptr = Vec::with_capacity(padded_bs + 1);
    o_indptr.push(0);

    for (request_idx, &kv_len) in kv_lens.iter().enumerate() {
        let chunks = kv_len.div_ceil(chunk_size).max(1);
        anyhow::ensure!(
            chunks <= cap,
            "split-KV chunk count {chunks} exceeds bound {cap} \
             (kv_len={kv_len}, chunk_size={chunk_size}); context limit misconfigured"
        );
        for chunk_idx in 0..chunks {
            request_indices.push(request_idx as i32);
            kv_tile_indices.push(chunk_idx as i32);
            block_valid_mask.push(1);
        }
        o_indptr.push(request_indices.len() as i32);
    }
    for _ in kv_lens.len()..padded_bs {
        o_indptr.push(request_indices.len() as i32);
    }
    while request_indices.len() < padded_slots {
        request_indices.push(0);
        kv_tile_indices.push(0);
        block_valid_mask.push(0);
    }

    Ok(SplitKvCsr {
        request_indices,
        kv_tile_indices,
        block_valid_mask,
        o_indptr,
    })
}

pub struct PrefillPagedPlan {
    inner: pegainfer_kernels::ops::PrefillPagedPlan,
}

impl PrefillPagedPlan {
    pub fn new(
        ctx: &DeviceContext,
        desc: &KvDesc<'_>,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_with_cta_tile_q(
            ctx,
            desc,
            start_pos,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_cta_tile_q(
        ctx: &DeviceContext,
        desc: &KvDesc<'_>,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let page_indices: Vec<i32> = desc
            .page_indices()
            .iter()
            .map(|p| p.index() as i32)
            .collect();
        Ok(Self {
            inner: pegainfer_kernels::ops::PrefillPagedPlan::new_with_cta_tile_q(
                ctx,
                &page_indices,
                desc.last_page_len(),
                start_pos,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cta_tile_q_override,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_batch_with_cta_tile_q(
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        Ok(Self {
            inner: pegainfer_kernels::ops::PrefillPagedPlan::new_batch_with_cta_tile_q(
                ctx,
                page_indices,
                last_page_lens,
                start_positions,
                seq_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                cta_tile_q_override,
            )?,
        })
    }

    /// Pre-allocate a worst-case-sized plan to be refilled in place by
    /// [`Self::update_batch_with_cta_tile_q`] (graph-stable buffer reuse).
    pub fn new_preallocated(
        ctx: &DeviceContext,
        max_total_tokens: usize,
        max_total_pages: usize,
        max_batch: usize,
        max_tiles: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: pegainfer_kernels::ops::PrefillPagedPlan::new_preallocated(
                ctx,
                max_total_tokens,
                max_total_pages,
                max_batch,
                max_tiles,
            )?,
        })
    }

    /// Refill a pre-allocated plan in place (no allocation, pointers unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn update_batch_with_cta_tile_q(
        &mut self,
        ctx: &DeviceContext,
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<()> {
        self.inner.update_batch_with_cta_tile_q(
            ctx,
            page_indices,
            last_page_lens,
            start_positions,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cta_tile_q_override,
        )
    }

    pub fn page_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.page_indices_d()
    }
    pub fn page_indptr_d(&self) -> &CudaSlice<i32> {
        self.inner.page_indptr_d()
    }
    pub fn last_page_len_d(&self) -> &CudaSlice<i32> {
        self.inner.last_page_len_d()
    }
    pub fn q_indptr_d(&self) -> &CudaSlice<i32> {
        self.inner.q_indptr_d()
    }
    pub fn request_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.request_indices_d()
    }
    pub fn qo_tile_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.qo_tile_indices_d()
    }
    pub fn kv_tile_indices_d(&self) -> &CudaSlice<i32> {
        self.inner.kv_tile_indices_d()
    }
    pub fn kv_chunk_size_d(&self) -> &CudaSlice<i32> {
        self.inner.kv_chunk_size_d()
    }
    pub fn decode_metadata_d_mut(&mut self) -> (&mut CudaSlice<i32>, &mut CudaSlice<i32>) {
        self.inner.decode_metadata_d_mut()
    }
    pub fn total_num_rows_d(&self) -> &CudaSlice<u32> {
        self.inner.total_num_rows_d()
    }
    pub fn batch_size(&self) -> i32 {
        self.inner.batch_size()
    }
    pub fn num_tiles(&self) -> i32 {
        self.inner.num_tiles()
    }
}

impl Deref for PrefillPagedPlan {
    type Target = pegainfer_kernels::ops::PrefillPagedPlan;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
