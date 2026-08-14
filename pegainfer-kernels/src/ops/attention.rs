use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::paged_kv::PagedKvLayout;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

/// GQA group sizes (query heads / KV heads) instantiated by FlashInfer's
/// `DISPATCH_GQA_GROUP_SIZE`; any other ratio throws at dispatch time.
pub const SUPPORTED_GQA_GROUP_SIZES: &[usize] = &[1, 2, 3, 4, 8];

// ============================================================================
// Paged prefill (FlashInfer BatchPrefillWithPagedKVCache)
// ============================================================================

/// Pre-computed GPU metadata for paged prefill attention.
///
/// Built once per prefill call, shared across all layers.
/// Supports both single-request (`new`) and multi-request (`new_batch`) prefill.
pub struct PrefillPagedPlan {
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    batch_indices_d: CudaSlice<i32>,
    positions_d: CudaSlice<i32>,
    q_indptr_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    qo_tile_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    total_num_rows_d: CudaSlice<u32>,
    num_tiles: i32,
    batch_size: i32,
    total_tokens: usize,
    cta_tile_q: i32,
    head_dim: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
}

impl PrefillPagedPlan {
    pub fn page_indices_d(&self) -> &CudaSlice<i32> {
        &self.page_indices_d
    }
    pub fn page_indptr_d(&self) -> &CudaSlice<i32> {
        &self.page_indptr_d
    }
    pub fn last_page_len_d(&self) -> &CudaSlice<i32> {
        &self.last_page_len_d
    }
    pub fn batch_indices_d(&self) -> &CudaSlice<i32> {
        &self.batch_indices_d
    }
    pub fn positions_d(&self) -> &CudaSlice<i32> {
        &self.positions_d
    }
    pub fn q_indptr_d(&self) -> &CudaSlice<i32> {
        &self.q_indptr_d
    }
    pub fn request_indices_d(&self) -> &CudaSlice<i32> {
        &self.request_indices_d
    }
    pub fn qo_tile_indices_d(&self) -> &CudaSlice<i32> {
        &self.qo_tile_indices_d
    }
    pub fn kv_tile_indices_d(&self) -> &CudaSlice<i32> {
        &self.kv_tile_indices_d
    }
    pub fn kv_chunk_size_d(&self) -> &CudaSlice<i32> {
        &self.kv_chunk_size_d
    }
    pub fn total_num_rows_d(&self) -> &CudaSlice<u32> {
        &self.total_num_rows_d
    }
    pub fn batch_size(&self) -> i32 {
        self.batch_size
    }
    pub fn num_tiles(&self) -> i32 {
        self.num_tiles
    }
    fn cta_tile_q(&self) -> i32 {
        self.cta_tile_q
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_cta_tile_q(
        ctx: &DeviceContext,
        page_indices_i32: &[i32],
        last_page_len: usize,
        start_pos: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let kv_len = start_pos + seq_len;

        let page_indices_d = ctx.stream.clone_htod(page_indices_i32)?;
        let page_indptr_d = ctx
            .stream
            .clone_htod(&[0i32, page_indices_i32.len() as i32])?;
        let last_page_len_d = ctx.stream.clone_htod(&[last_page_len as i32])?;

        let batch_indices_d = ctx.stream.clone_htod(&vec![0i32; seq_len])?;
        let positions: Vec<i32> = (start_pos as i32..(start_pos + seq_len) as i32).collect();
        let positions_d = ctx.stream.clone_htod(&positions)?;

        let num_tiles = unsafe {
            ffi::batch_prefill_paged_num_tiles_with_cta_tile_q(
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        };
        anyhow::ensure!(
            num_tiles > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );
        let cta_tile_q = unsafe {
            ffi::batch_prefill_cta_tile_q_with_override(
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        };
        anyhow::ensure!(
            cta_tile_q > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );

        let q_indptr_d = ctx.stream.clone_htod(&[0i32, seq_len as i32])?;
        let request_indices_d = ctx.stream.clone_htod(&vec![0i32; num_tiles as usize])?;
        let qo_tile_indices: Vec<i32> = (0..num_tiles).collect();
        let qo_tile_indices_d = ctx.stream.clone_htod(&qo_tile_indices)?;
        let kv_tile_indices_d = ctx.stream.clone_htod(&vec![0i32; num_tiles as usize])?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&[kv_len as i32])?;
        let total_num_rows_d = ctx.stream.clone_htod(&[seq_len as u32])?;

        Ok(Self {
            page_indices_d,
            page_indptr_d,
            last_page_len_d,
            batch_indices_d,
            positions_d,
            q_indptr_d,
            request_indices_d,
            qo_tile_indices_d,
            kv_tile_indices_d,
            kv_chunk_size_d,
            total_num_rows_d,
            num_tiles,
            batch_size: 1,
            total_tokens: seq_len,
            cta_tile_q,
            head_dim,
            num_qo_heads: num_q_heads,
            num_kv_heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_batch_with_cta_tile_q(
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
        let host = BatchPlanHost::compute(
            page_indices,
            last_page_lens,
            start_positions,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cta_tile_q_override,
        )?;

        // Upload all to GPU
        Ok(Self {
            page_indices_d: ctx.stream.clone_htod(&host.all_page_indices)?,
            page_indptr_d: ctx.stream.clone_htod(&host.page_indptr)?,
            last_page_len_d: ctx.stream.clone_htod(&host.last_page_lens_i32)?,
            batch_indices_d: ctx.stream.clone_htod(&host.batch_indices)?,
            positions_d: ctx.stream.clone_htod(&host.positions)?,
            q_indptr_d: ctx.stream.clone_htod(&host.q_indptr)?,
            request_indices_d: ctx.stream.clone_htod(&host.request_indices_v)?,
            qo_tile_indices_d: ctx.stream.clone_htod(&host.qo_tile_indices_v)?,
            kv_tile_indices_d: ctx.stream.clone_htod(&host.kv_tile_indices_v)?,
            kv_chunk_size_d: ctx.stream.clone_htod(&host.kv_chunk_sizes)?,
            total_num_rows_d: ctx.stream.clone_htod(&[host.total_tokens as u32])?,
            num_tiles: host.num_tiles,
            batch_size: host.batch_size as i32,
            total_tokens: host.total_tokens,
            cta_tile_q: host.cta_tile_q as i32,
            head_dim,
            num_qo_heads: num_q_heads,
            num_kv_heads,
        })
    }

    /// Allocate a worst-case-sized plan once, to be refilled in place by
    /// [`Self::update_batch_with_cta_tile_q`]. Buffer pointers stay fixed across
    /// updates so a CUDA Graph captured against them remains valid on replay.
    /// Geometry starts unset; update before forward.
    pub fn new_preallocated(
        ctx: &DeviceContext,
        max_total_tokens: usize,
        max_total_pages: usize,
        max_batch: usize,
        max_tiles: usize,
    ) -> Result<Self> {
        Ok(Self {
            page_indices_d: ctx.stream.alloc_zeros(max_total_pages)?,
            page_indptr_d: ctx.stream.alloc_zeros(max_batch + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(max_batch)?,
            batch_indices_d: ctx.stream.alloc_zeros(max_total_tokens)?,
            positions_d: ctx.stream.alloc_zeros(max_total_tokens)?,
            q_indptr_d: ctx.stream.alloc_zeros(max_batch + 1)?,
            request_indices_d: ctx.stream.alloc_zeros(max_tiles)?,
            qo_tile_indices_d: ctx.stream.alloc_zeros(max_tiles)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(max_tiles)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(max_batch)?,
            total_num_rows_d: ctx.stream.alloc_zeros(1)?,
            num_tiles: 0,
            batch_size: 0,
            total_tokens: 0,
            cta_tile_q: 0,
            head_dim: 0,
            num_qo_heads: 0,
            num_kv_heads: 0,
        })
    }

    /// Recompute the host-side batch metadata and `memcpy_htod` it into the
    /// pre-allocated device buffers (no allocation, no pointer change). The host
    /// computation is identical to [`Self::new_batch_with_cta_tile_q`]; only the
    /// upload differs (overwrite in place vs. fresh `clone_htod`).
    ///
    /// `memcpy_htod` copies `src.len()` elements and tolerates a larger
    /// destination, so the worst-case allocation may exceed the actual fill.
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
        let host = BatchPlanHost::compute(
            page_indices,
            last_page_lens,
            start_positions,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cta_tile_q_override,
        )?;

        anyhow::ensure!(
            host.all_page_indices.len() <= self.page_indices_d.len(),
            "verify plan page_indices ({}) exceeds preallocated capacity ({})",
            host.all_page_indices.len(),
            self.page_indices_d.len(),
        );
        anyhow::ensure!(
            host.page_indptr.len() <= self.page_indptr_d.len(),
            "verify plan page_indptr ({}) exceeds preallocated capacity ({})",
            host.page_indptr.len(),
            self.page_indptr_d.len(),
        );
        anyhow::ensure!(
            host.last_page_lens_i32.len() <= self.last_page_len_d.len(),
            "verify plan last_page_lens ({}) exceeds preallocated capacity ({})",
            host.last_page_lens_i32.len(),
            self.last_page_len_d.len(),
        );
        anyhow::ensure!(
            host.batch_indices.len() <= self.batch_indices_d.len(),
            "verify plan batch_indices ({}) exceeds preallocated capacity ({})",
            host.batch_indices.len(),
            self.batch_indices_d.len(),
        );
        anyhow::ensure!(
            host.positions.len() <= self.positions_d.len(),
            "verify plan positions ({}) exceeds preallocated capacity ({})",
            host.positions.len(),
            self.positions_d.len(),
        );
        anyhow::ensure!(
            host.q_indptr.len() <= self.q_indptr_d.len(),
            "verify plan q_indptr ({}) exceeds preallocated capacity ({})",
            host.q_indptr.len(),
            self.q_indptr_d.len(),
        );
        anyhow::ensure!(
            host.request_indices_v.len() <= self.request_indices_d.len(),
            "verify plan tiles ({}) exceeds preallocated capacity ({})",
            host.request_indices_v.len(),
            self.request_indices_d.len(),
        );
        anyhow::ensure!(
            host.kv_chunk_sizes.len() <= self.kv_chunk_size_d.len(),
            "verify plan kv_chunk_sizes ({}) exceeds preallocated capacity ({})",
            host.kv_chunk_sizes.len(),
            self.kv_chunk_size_d.len(),
        );

        ctx.stream
            .memcpy_htod(&host.all_page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&host.page_indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&host.last_page_lens_i32, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&host.batch_indices, &mut self.batch_indices_d)?;
        ctx.stream
            .memcpy_htod(&host.positions, &mut self.positions_d)?;
        ctx.stream
            .memcpy_htod(&host.q_indptr, &mut self.q_indptr_d)?;
        ctx.stream
            .memcpy_htod(&host.request_indices_v, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&host.qo_tile_indices_v, &mut self.qo_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&host.kv_tile_indices_v, &mut self.kv_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&host.kv_chunk_sizes, &mut self.kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&[host.total_tokens as u32], &mut self.total_num_rows_d)?;

        self.num_tiles = host.num_tiles;
        self.batch_size = host.batch_size as i32;
        self.total_tokens = host.total_tokens;
        self.cta_tile_q = host.cta_tile_q as i32;
        self.head_dim = head_dim;
        self.num_qo_heads = num_q_heads;
        self.num_kv_heads = num_kv_heads;
        Ok(())
    }
}

/// Host-side batch-prefill metadata, computed identically for the fresh
/// (`new_batch_with_cta_tile_q`) and in-place (`update_batch_with_cta_tile_q`)
/// paths so the two never diverge.
struct BatchPlanHost {
    all_page_indices: Vec<i32>,
    page_indptr: Vec<i32>,
    last_page_lens_i32: Vec<i32>,
    batch_indices: Vec<i32>,
    positions: Vec<i32>,
    q_indptr: Vec<i32>,
    request_indices_v: Vec<i32>,
    qo_tile_indices_v: Vec<i32>,
    kv_tile_indices_v: Vec<i32>,
    kv_chunk_sizes: Vec<i32>,
    num_tiles: i32,
    batch_size: usize,
    total_tokens: usize,
    cta_tile_q: usize,
}

impl BatchPlanHost {
    #[allow(clippy::too_many_arguments)]
    fn compute(
        page_indices: &[Vec<i32>],
        last_page_lens: &[usize],
        start_positions: &[usize],
        seq_lens: &[usize],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cta_tile_q_override: i32,
    ) -> Result<Self> {
        let batch_size = page_indices.len();
        assert_eq!(batch_size, last_page_lens.len());
        assert_eq!(batch_size, start_positions.len());
        assert_eq!(batch_size, seq_lens.len());
        let total_tokens: usize = seq_lens.iter().sum();
        let group_size = num_q_heads / num_kv_heads;

        // Page metadata (concatenated across requests, CSR format)
        let mut all_page_indices = Vec::new();
        let mut page_indptr = vec![0i32];
        let mut last_page_lens_i32 = Vec::with_capacity(batch_size);
        let mut kv_chunk_sizes = Vec::with_capacity(batch_size);

        for (i, pages) in page_indices.iter().enumerate() {
            all_page_indices.extend_from_slice(pages);
            page_indptr.push(all_page_indices.len() as i32);
            last_page_lens_i32.push(last_page_lens[i] as i32);
            kv_chunk_sizes.push((start_positions[i] + seq_lens[i]) as i32);
        }

        // Per-token metadata
        let mut batch_indices = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        for (i, &seq_len) in seq_lens.iter().enumerate() {
            let start = start_positions[i];
            batch_indices.extend(std::iter::repeat_n(i as i32, seq_len));
            positions.extend((start..start + seq_len).map(|p| p as i32));
        }

        // Q token boundaries (CSR)
        let mut q_indptr = vec![0i32];
        for &seq_len in seq_lens {
            let prev = *q_indptr.last().unwrap();
            q_indptr.push(prev + seq_len as i32);
        }

        // Tile plan: use global cta_tile_q for consistent tiling
        let cta_tile_q = unsafe {
            ffi::batch_prefill_cta_tile_q_with_override(
                total_tokens as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                cta_tile_q_override,
            )
        } as usize;
        anyhow::ensure!(
            cta_tile_q > 0,
            "invalid prefill CTA tile override {cta_tile_q_override}"
        );

        let mut request_indices_v = Vec::new();
        let mut qo_tile_indices_v = Vec::new();
        let mut kv_tile_indices_v = Vec::new();
        for (req_idx, &seq_len) in seq_lens.iter().enumerate() {
            let packed_qo_len = seq_len * group_size;
            let num_tiles_req = packed_qo_len.div_ceil(cta_tile_q);
            for tile in 0..num_tiles_req {
                request_indices_v.push(req_idx as i32);
                qo_tile_indices_v.push(tile as i32);
                kv_tile_indices_v.push(0i32);
            }
        }
        let num_tiles = request_indices_v.len() as i32;

        Ok(Self {
            all_page_indices,
            page_indptr,
            last_page_lens_i32,
            batch_indices,
            positions,
            q_indptr,
            request_indices_v,
            qo_tile_indices_v,
            kv_tile_indices_v,
            kv_chunk_sizes,
            num_tiles,
            batch_size,
            total_tokens,
            cta_tile_q,
        })
    }
}

/// Per-layer paged prefill: QK norm + RoPE, append K/V to paged, batch prefill attention.
///
/// Token positions (RoPE, scatter, attention) come from the plan's per-token
/// position array, so chunked or prefix-cached prefill (start > 0) works for
/// any batch size.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_into(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let total_tokens = plan.total_tokens;
    let kv_dim = num_kv_heads * head_dim;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (bi_ptr, _) = plan.batch_indices_d.device_ptr(&ctx.stream);
    let (pos_ptr, _) = plan.positions_d.device_ptr(&ctx.stream);
    let (qi_ptr, _) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    unsafe {
        // RoPE positions always come from the plan's per-token array — it is
        // the single source of truth for each token's absolute position. A
        // scalar-start_pos fast path for batch_size == 1 used to live here;
        // it silently rotated prefix-cache-hit suffixes from position 0 and
        // both entry points launch the same kernel anyway.
        ffi::qk_norm_rope_batched_decode_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            total_tokens as i32,
            rms_eps,
            (cos_cache.data.len() / head_dim) as i32,
            stream,
        );

        let src_stride_n = kv_dim as i64;
        let src_stride_h = head_dim as i64;

        let result = ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            bi_ptr as *const i32,
            pos_ptr as *const i32,
            total_tokens as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        );
        if result != 0 {
            anyhow::bail!(
                "paged_kv_scatter_cuda failed for layer {layer} with error {result}{}",
                crate::ops::ffi_exception_message(result)
            );
        }

        let result = ffi::batch_prefill_paged_cuda_with_cta_tile_q(
            q_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            total_tokens as i32,
            plan.batch_size,
            plan.num_tiles,
            stride_page,
            sm_scale,
            plan.cta_tile_q(),
            stream,
        );
        if result != 0 {
            anyhow::bail!(
                "batch_prefill_paged_cuda failed for layer {layer} with error {result}{}",
                crate::ops::ffi_exception_message(result)
            );
        }
    }

    Ok(())
}

// ============================================================================
// Paged attention decode (FlashInfer)
// ============================================================================

/// Batched QK RMSNorm + RoPE for decode: per-request positions from GPU array.
///
/// Q/K are modified in place over the row window `[row_offset, row_offset + num_rows)` — the decode
/// rows of a unified step sit behind its prefill rows. `positions_d` is indexed from local row 0.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_batch_decode_into(
    ctx: &DeviceContext,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    row_offset: usize,
    num_rows: usize,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let q_byte_offset = checked_row_offset(q, row_offset, num_rows, "qk_norm_rope_batch_decode q")?;
    let k_byte_offset = checked_row_offset(k, row_offset, num_rows, "qk_norm_rope_batch_decode k")?;

    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let q_ptr = q_ptr + q_byte_offset;
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let k_ptr = k_ptr + k_byte_offset;
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions_d.device_ptr(&ctx.stream);

    unsafe {
        ffi::qk_norm_rope_batched_decode_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            num_rows as i32,
            rms_eps,
            (cos_cache.data.len() / head_dim) as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// QK RMSNorm + RoPE for one DFlash request's draft block.
///
/// `q` is a row sub-range of a batched buffer: `q_row_offset` rows precede this
/// request's `q_seq_len` query rows. The kernel still sees a single-request Q
/// shape — we just advance the device pointer to the request's slice. `k` is the
/// request's own varlen tail scratch (whole buffer), so it needs no offset.
#[allow(clippy::too_many_arguments)]
pub fn dflash_qk_norm_rope_into(
    ctx: &DeviceContext,
    q: &mut HiddenStates,
    q_row_offset: usize,
    q_seq_len: usize,
    k: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_start_pos: usize,
    k_start_pos: usize,
    rms_eps: f32,
) -> Result<()> {
    assert_eq!(q.hidden_dim, num_q_heads * head_dim);
    assert_eq!(k.hidden_dim, num_kv_heads * head_dim);
    assert_eq!(q_norm_weight.len, head_dim);
    assert_eq!(k_norm_weight.len, head_dim);
    assert!(
        q_row_offset + q_seq_len <= q.seq_len,
        "dflash_qk_norm_rope q row range [{}..{}) exceeds seq_len {}",
        q_row_offset,
        q_row_offset + q_seq_len,
        q.seq_len
    );

    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let q_ptr = q_ptr + (q_row_offset * q.hidden_dim * std::mem::size_of::<bf16>()) as u64;
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::dflash_qk_norm_rope_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q_seq_len as i32,
            k.seq_len as i32,
            q_start_pos as i32,
            k_start_pos as i32,
            rms_eps,
            (cos_cache.data.len() / head_dim) as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!("dflash_qk_norm_rope_cuda failed with error {result}");
    }
    Ok(())
}

/// Validate a `[offset, offset + span)` row window into `t` and return the byte
/// offset of `offset`, all with checked arithmetic. Release builds leave
/// `overflow-checks` off, so an adversarial `offset` (e.g. `usize::MAX`) would
/// otherwise wrap the range check and the pointer past its allocation, faulting
/// only at the next sync as `CUDA_ERROR_ILLEGAL_ADDRESS`. Returns `Err`, never panics.
fn checked_row_offset(t: &HiddenStates, offset: usize, span: usize, name: &str) -> Result<u64> {
    let end = offset
        .checked_add(span)
        .ok_or_else(|| anyhow::anyhow!("{name} row range overflow"))?;
    anyhow::ensure!(
        end <= t.seq_len,
        "{name} row range [{offset}..{end}) exceeds seq_len {}",
        t.seq_len
    );
    t.checked_extent(name)?;
    let bytes = offset
        .checked_mul(t.hidden_dim)
        .and_then(|elems| elems.checked_mul(std::mem::size_of::<bf16>()))
        .ok_or_else(|| anyhow::anyhow!("{name} byte offset overflow"))?;
    Ok(bytes as u64)
}

/// `DeviceVec` fields are public too: reject a logical `len` past the backing
/// allocation before a kernel indexes through it.
fn ensure_vec_backed(v: &DeviceVec, name: &str) -> Result<()> {
    anyhow::ensure!(
        v.data.len() >= v.len,
        "{name} backing len {} < len {}",
        v.data.len(),
        v.len
    );
    Ok(())
}

/// Plain RoPE (no QK-norm) for one EAGLE-3 draft step, because EAGLE-3 has no per-head q/k norm
/// we implement a new kernel for EAGLE-3
#[allow(clippy::too_many_arguments)]
pub fn eagle3_rope_into(
    ctx: &DeviceContext,
    q: &mut HiddenStates,
    q_row_offset: usize,
    q_seq_len: usize,
    k: &mut HiddenStates,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_start_pos: usize,
    k_start_pos: usize,
) -> Result<()> {
    assert_eq!(q.hidden_dim, num_q_heads * head_dim);
    assert_eq!(k.hidden_dim, num_kv_heads * head_dim);
    // `q`/`k` are capacity-backed and reachable from safe re-exported code;
    // validate the row window and backing length before deriving raw pointers.
    let q_byte_offset = checked_row_offset(q, q_row_offset, q_seq_len, "eagle3_rope q")?;
    k.checked_extent("eagle3_rope k")?;
    // The kernel indexes both caches with the same `pos * head_dim + d`, and
    // `cos_max_pos` is derived from `cos_cache` alone; a shorter `sin_cache`
    // would let the kernel read out of bounds.
    assert_eq!(
        sin_cache.data.len(),
        cos_cache.data.len(),
        "eagle3_rope sin_cache len {} != cos_cache len {}",
        sin_cache.data.len(),
        cos_cache.data.len()
    );

    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let q_ptr = q_ptr + q_byte_offset;
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::eagle3_rope_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q_seq_len as i32,
            k.seq_len as i32,
            q_start_pos as i32,
            k_start_pos as i32,
            (cos_cache.data.len() / head_dim) as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!("eagle3_rope_cuda failed with error {result}");
    }
    Ok(())
}

/// Non-causal prefill attention for one DFlash request's draft block.
///
/// `q` and `output` share the SAME row sub-range of batched buffers: request
/// `i` owns rows `[row_offset, row_offset + q_seq_len)` in both, because the
/// draft writes each request's attention output back into the row slot its
/// queries came from. The k/v caches are the request's own whole buffers. The
/// kernel sees a single-request shape — we advance the Q/output device pointers
/// to the request's slice.
#[allow(clippy::too_many_arguments)]
pub fn single_prefill_nhd_noncausal_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    q_seq_len: usize,
    k_cache: &HiddenStates,
    v_cache: &HiddenStates,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<()> {
    assert_eq!(q.hidden_dim, num_q_heads * head_dim);
    assert_eq!(output.hidden_dim, q.hidden_dim);
    assert_eq!(output.seq_len, q.seq_len);
    assert_eq!(k_cache.hidden_dim, num_kv_heads * head_dim);
    assert_eq!(v_cache.hidden_dim, k_cache.hidden_dim);
    assert_eq!(v_cache.seq_len, k_cache.seq_len);
    assert!(kv_len <= k_cache.seq_len);
    assert!(
        row_offset + q_seq_len <= q.seq_len,
        "single_prefill row range [{}..{}) exceeds seq_len {}",
        row_offset,
        row_offset + q_seq_len,
        q.seq_len
    );

    // q and output share row_offset (asserted same seq_len/hidden_dim above).
    let byte_offset = (row_offset * q.hidden_dim * std::mem::size_of::<bf16>()) as u64;
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + byte_offset;
    let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + byte_offset;
    // FlashInfer's prefill kernel is a compile-time HEAD_DIM template: 128 is
    // the Qwen3 DFlash drafter, 64 the GLM5.2 DSpark drafter.
    let kernel = match head_dim {
        128 => ffi::single_prefill_nhd_noncausal_cuda,
        64 => ffi::single_prefill_nhd_noncausal_cuda_hd64,
        other => anyhow::bail!(
            "single_prefill_nhd_noncausal has no head_dim {other} instantiation (64/128 only)"
        ),
    };
    let result = unsafe {
        kernel(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q_seq_len as i32,
            kv_len as i32,
            k_cache.seq_len as i32,
            1.0f32 / (head_dim as f32).sqrt(),
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "single_prefill_nhd_noncausal_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Single-query **decode** over a contiguous NHD KV cache — the draft chain's
/// per-step attention. One query (`q.seq_len == 1`) attends the whole `[0, kv_len)`
/// prefix.
///
/// `q`/`output` are `[q_dim, 1]` and the k/v caches are the request's own whole
/// buffers `[kv_dim, max_seq_len]` (NHD token-major). No RoPE inside — the caller
/// applies [`eagle3_rope_into`] first.
#[allow(clippy::too_many_arguments)]
pub fn single_decode_nhd_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k_cache: &HiddenStates,
    v_cache: &HiddenStates,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<()> {
    assert_eq!(
        q.seq_len, 1,
        "single_decode_nhd is single-query; got q.seq_len {}",
        q.seq_len
    );
    assert_eq!(q.hidden_dim, num_q_heads * head_dim);
    assert_eq!(output.hidden_dim, q.hidden_dim);
    assert_eq!(output.seq_len, 1);
    assert_eq!(k_cache.hidden_dim, num_kv_heads * head_dim);
    assert_eq!(v_cache.hidden_dim, k_cache.hidden_dim);
    assert_eq!(v_cache.seq_len, k_cache.seq_len);
    assert!(kv_len <= k_cache.seq_len);
    // Shape asserts only relate the public metadata; validate it against the
    // backing allocations too, since safe callers can inflate `.seq_len` past a
    // small buffer (all `HiddenStates` fields are public).
    q.checked_extent("single_decode q")?;
    k_cache.checked_extent("single_decode k_cache")?;
    v_cache.checked_extent("single_decode v_cache")?;
    output.checked_extent("single_decode output")?;

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::single_decode_nhd_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            kv_len as i32,
            k_cache.seq_len as i32,
            1.0f32 / (head_dim as f32).sqrt(),
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "single_decode_nhd_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Causal NHD single-sequence prefill. Used for EAGLE-3's
/// teacher-forced prefill in a single batched forward.
#[allow(clippy::too_many_arguments)]
pub fn single_prefill_nhd_causal_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    q_seq_len: usize,
    k_cache: &HiddenStates,
    v_cache: &HiddenStates,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<()> {
    assert_eq!(q.hidden_dim, num_q_heads * head_dim);
    assert_eq!(output.hidden_dim, q.hidden_dim);
    assert_eq!(output.seq_len, q.seq_len);
    assert_eq!(k_cache.hidden_dim, num_kv_heads * head_dim);
    assert_eq!(v_cache.hidden_dim, k_cache.hidden_dim);
    assert_eq!(v_cache.seq_len, k_cache.seq_len);
    assert!(kv_len <= k_cache.seq_len);
    assert!(
        q_seq_len <= kv_len,
        "causal prefill q_seq_len {q_seq_len} exceeds kv_len {kv_len}"
    );
    // Validate the row window with checked arithmetic and every backing
    // allocation before deriving pointers; `q`/`output` share the row sub-range.
    let byte_offset = checked_row_offset(q, row_offset, q_seq_len, "single_prefill_causal q")?;
    output.checked_extent("single_prefill_causal output")?;
    k_cache.checked_extent("single_prefill_causal k_cache")?;
    v_cache.checked_extent("single_prefill_causal v_cache")?;
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + byte_offset;
    let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + byte_offset;
    let result = unsafe {
        ffi::single_prefill_nhd_causal_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            q_seq_len as i32,
            kv_len as i32,
            k_cache.seq_len as i32,
            1.0f32 / (head_dim as f32).sqrt(),
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "single_prefill_nhd_causal_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Batched QK RMSNorm + partial RoPE for Qwen3.5 HD256 decode.
///
/// Reads Q from interleaved `q_full` ([q, gate] per head), writes prepared Q into `q`,
/// and normalizes/applies partial RoPE to `k` in-place using per-request positions.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_batched_decode_hd256_into(
    ctx: &DeviceContext,
    q_full: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) {
    let batch_size = q.seq_len;
    debug_assert_eq!(q_full.seq_len, batch_size);
    debug_assert_eq!(k.seq_len, batch_size);

    let (qf_ptr, _gqf) = q_full.data.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions_d.device_ptr(&ctx.stream);

    unsafe {
        ffi::qk_norm_partial_rope_batched_decode_hd256_cuda(
            qf_ptr as *const ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            q_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            batch_size as i32,
            rotary_dim as i32,
            rms_eps,
            crate::tensor::active_cu_stream(ctx),
        );
    }
}

/// Batched paged attention decode: append K/V + FlashInfer BatchDecode for batch_size >= 1.
///
/// Q: HiddenStates [q_dim, batch_size], output: HiddenStates [q_dim, batch_size].
/// Metadata arrays are concatenated across requests (CSR format).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    // Step 1: Append K/V to paged cache (batched) using the same generic
    // scatter path as prefill, with explicit request indices and positions.
    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_kv_scatter_cuda (batch decode) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    // Step 2: Paged attention decode (batched)
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_cuda (batch) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

/// Batched paged attention decode using FlashInfer partition-KV/split-K.
///
/// This is intended for low-batch, long-context decode where the non-partition
/// grid `(batch, kv_heads)` does not expose enough CTAs.
///
/// Q/K/V/output are read and written over `[row_offset, row_offset + batch_size)`; every paged and
/// split-KV metadata array is indexed from local row 0.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_split_kv_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    row_offset: usize,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    split_request_indices_d: &CudaSlice<i32>,
    split_kv_tile_indices_d: &CudaSlice<i32>,
    split_kv_chunk_size_d: &CudaSlice<i32>,
    split_o_indptr_d: &CudaSlice<i32>,
    split_block_valid_mask_d: &CudaSlice<u8>,
    split_tmp_v: &mut CudaSlice<bf16>,
    split_tmp_s: &mut CudaSlice<f32>,
    split_padded_slots: usize,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let q_byte_offset = checked_row_offset(
        q,
        row_offset,
        batch_size,
        "paged_attention_batch_decode_split_kv q",
    )?;
    let k_byte_offset = checked_row_offset(
        k,
        row_offset,
        batch_size,
        "paged_attention_batch_decode_split_kv k",
    )?;
    let v_byte_offset = checked_row_offset(
        v,
        row_offset,
        batch_size,
        "paged_attention_batch_decode_split_kv v",
    )?;
    let out_byte_offset = checked_row_offset(
        output,
        row_offset,
        batch_size,
        "paged_attention_batch_decode_split_kv output",
    )?;

    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_byte_offset;
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let k_ptr = k_ptr + k_byte_offset;
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let v_ptr = v_ptr + v_byte_offset;
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + out_byte_offset;
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (split_ri_ptr, _gsri) = split_request_indices_d.device_ptr(&ctx.stream);
    let (split_kti_ptr, _gskti) = split_kv_tile_indices_d.device_ptr(&ctx.stream);
    let (split_kcs_ptr, _gskcs) = split_kv_chunk_size_d.device_ptr(&ctx.stream);
    let (split_o_indptr_ptr, _gsoi) = split_o_indptr_d.device_ptr(&ctx.stream);
    let (split_valid_ptr, _gsv) = split_block_valid_mask_d.device_ptr(&ctx.stream);
    let (split_tmp_v_ptr, _gstmpv) = split_tmp_v.device_ptr_mut(&ctx.stream);
    let (split_tmp_s_ptr, _gstmps) = split_tmp_s.device_ptr_mut(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_kv_scatter_cuda (batch split-K decode) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_split_kv_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            split_ri_ptr as *const i32,
            split_kti_ptr as *const i32,
            split_kcs_ptr as *const i32,
            split_o_indptr_ptr as *const i32,
            split_valid_ptr as *const u8,
            split_tmp_v_ptr as *mut ffi::Half,
            split_tmp_s_ptr as *mut f32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            split_padded_slots as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_split_kv_cuda (batch) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scatter_decode_kv_into_paged(
    ctx: &DeviceContext,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    batch_size: usize,
    op_name: &str,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (pos_ptr, _gpos) = positions_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);

    let src_stride_n = (num_kv_heads * head_dim) as i64;
    let src_stride_h = head_dim as i64;
    let result = unsafe {
        ffi::paged_kv_scatter_cuda(
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            ri_ptr as *const i32,
            pos_ptr as *const i32,
            batch_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            stride_page,
            src_stride_n,
            src_stride_h,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_kv_scatter_cuda ({op_name}) failed for layer {layer}, bs={batch_size}, \
             kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}: {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    debug_assert_eq!(head_dim, 256);
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = last_page_len_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    scatter_decode_kv_into_paged(
        ctx,
        k,
        v,
        kv_buffer,
        layout,
        layer,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        positions_d,
        request_indices_d,
        batch_size,
        "batch hd256 decode",
    )?;

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda_hd256(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_cuda_hd256 (batch) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_via_prefill_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    positions_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    debug_assert_eq!(head_dim, 256);
    anyhow::ensure!(
        batch_size == plan.total_tokens && batch_size == plan.batch_size as usize,
        "decode-via-prefill plan shape mismatch: bs={batch_size}, total_tokens={}, plan_batch={}",
        plan.total_tokens,
        plan.batch_size
    );

    scatter_decode_kv_into_paged(
        ctx,
        k,
        v,
        kv_buffer,
        layout,
        layer,
        &plan.page_indices_d,
        &plan.page_indptr_d,
        &plan.last_page_len_d,
        positions_d,
        &plan.batch_indices_d,
        batch_size,
        "batch hd256 decode via prefill",
    )?;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _gqti) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _gtnr) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::batch_prefill_paged_cuda_hd256(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            batch_size as i32,
            plan.batch_size,
            plan.num_tiles,
            stride_page,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "batch_prefill_paged_cuda_hd256 (decode via prefill) failed for layer {layer}, \
             bs={batch_size}, tiles={}, qo_heads={num_qo_heads}, kv_heads={num_kv_heads}: {result}{}",
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

// ============================================================================
// hd512 (Gemma 4 global layers)
// ============================================================================

/// Decode-metadata aggregate for the hd512 direct-decode path, validated at
/// use: the wrapper calls [`Self::validate`] with the batch size it derives.
///
/// `page_indptr` needs at least `batch_size + 1` elements, the other per-request
/// slices at least `batch_size`; `page_indices` must be non-empty (its
/// per-request reach is device-side data and cannot be verified here).
///
/// The pages referenced by `page_indices` must lie within `kv_buffer`; the
/// wrapper cannot verify device-side contents — caller contract.
pub struct Hd512DecodeMetadata<'a> {
    page_indices: &'a CudaSlice<i32>,
    page_indptr: &'a CudaSlice<i32>,
    last_page_len: &'a CudaSlice<i32>,
    positions: &'a CudaSlice<i32>,
    request_indices: &'a CudaSlice<i32>,
    kv_tile_indices: &'a CudaSlice<i32>,
    kv_chunk_size: &'a CudaSlice<i32>,
}

impl<'a> Hd512DecodeMetadata<'a> {
    pub fn new(
        page_indices: &'a CudaSlice<i32>,
        page_indptr: &'a CudaSlice<i32>,
        last_page_len: &'a CudaSlice<i32>,
        positions: &'a CudaSlice<i32>,
        request_indices: &'a CudaSlice<i32>,
        kv_tile_indices: &'a CudaSlice<i32>,
        kv_chunk_size: &'a CudaSlice<i32>,
    ) -> Self {
        Self {
            page_indices,
            page_indptr,
            last_page_len,
            positions,
            request_indices,
            kv_tile_indices,
            kv_chunk_size,
        }
    }

    pub fn validate(&self, batch_size: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.page_indices.is_empty(),
            "Hd512DecodeMetadata: page_indices must be non-empty"
        );
        anyhow::ensure!(
            self.page_indptr.len() > batch_size,
            "Hd512DecodeMetadata: page_indptr length {} must be >= batch_size + 1 = {}",
            self.page_indptr.len(),
            batch_size + 1
        );
        for (name, len) in [
            ("last_page_len", self.last_page_len.len()),
            ("positions", self.positions.len()),
            ("request_indices", self.request_indices.len()),
            ("kv_tile_indices", self.kv_tile_indices.len()),
            ("kv_chunk_size", self.kv_chunk_size.len()),
        ] {
            anyhow::ensure!(
                len >= batch_size,
                "Hd512DecodeMetadata: {name} length {len} must be >= batch_size {batch_size}",
            );
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    meta: &Hd512DecodeMetadata,
    output: &mut HiddenStates,
    num_qo_heads: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        head_dim == 512,
        "hd512 decode expects head_dim 512, got {}",
        head_dim
    );
    let batch_size = q.seq_len;
    meta.validate(batch_size)?;
    anyhow::ensure!(
        output.seq_len == batch_size,
        "hd512 decode output.seq_len {} != q.seq_len {batch_size}",
        output.seq_len
    );
    anyhow::ensure!(
        k.seq_len == batch_size,
        "hd512 decode k.seq_len {} != q.seq_len {batch_size}",
        k.seq_len
    );
    anyhow::ensure!(
        v.seq_len == batch_size,
        "hd512 decode v.seq_len {} != q.seq_len {batch_size}",
        v.seq_len
    );
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * 512,
        "hd512 decode q.hidden_dim {} != num_qo_heads {} * 512",
        q.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        output.hidden_dim == num_qo_heads * 512,
        "hd512 decode output.hidden_dim {} != num_qo_heads {} * 512",
        output.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        k.hidden_dim == num_kv_heads * 512,
        "hd512 decode k.hidden_dim {} != num_kv_heads {} * 512",
        k.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        v.hidden_dim == num_kv_heads * 512,
        "hd512 decode v.hidden_dim {} != num_kv_heads {} * 512",
        v.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        layer < layout.num_layers,
        "hd512 decode layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    q.checked_extent("batch_decode_hd512 q")?;
    output.checked_extent("batch_decode_hd512 output")?;
    k.checked_extent("batch_decode_hd512 k")?;
    v.checked_extent("batch_decode_hd512 v")?;
    let page_size = layout.page_size;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = meta.page_indices.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = meta.page_indptr.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = meta.last_page_len.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = meta.request_indices.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = meta.kv_tile_indices.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = meta.kv_chunk_size.device_ptr(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    scatter_decode_kv_into_paged(
        ctx,
        k,
        v,
        kv_buffer,
        layout,
        layer,
        meta.page_indices,
        meta.page_indptr,
        meta.last_page_len,
        meta.positions,
        meta.request_indices,
        batch_size,
        "batch hd512 decode",
    )?;

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_cuda_hd512 (batch) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_via_prefill_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    positions_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        head_dim == 512,
        "hd512 decode expects head_dim 512, got {}",
        head_dim
    );
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * 512,
        "hd512 decode-via-prefill q.hidden_dim {} != num_qo_heads {} * 512",
        q.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        output.hidden_dim == num_qo_heads * 512,
        "hd512 decode-via-prefill output.hidden_dim {} != num_qo_heads {} * 512",
        output.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        q.seq_len >= plan.total_tokens,
        "hd512 decode-via-prefill q.seq_len {} < plan.total_tokens {}",
        q.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        output.seq_len >= plan.total_tokens,
        "hd512 decode-via-prefill output.seq_len {} < plan.total_tokens {}",
        output.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        layer < layout.num_layers,
        "hd512 decode-via-prefill layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    q.checked_extent("decode_via_prefill_hd512 q")?;
    output.checked_extent("decode_via_prefill_hd512 output")?;
    anyhow::ensure!(
        positions_d.len() >= plan.batch_size() as usize,
        "hd512 decode-via-prefill positions_d length {} must be >= plan.batch_size {}",
        positions_d.len(),
        plan.batch_size(),
    );
    let batch_size = plan.batch_size() as usize;
    anyhow::ensure!(
        k.hidden_dim == num_kv_heads * 512,
        "hd512 decode-via-prefill k.hidden_dim {} != num_kv_heads {} * 512",
        k.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        v.hidden_dim == num_kv_heads * 512,
        "hd512 decode-via-prefill v.hidden_dim {} != num_kv_heads {} * 512",
        v.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        k.seq_len >= batch_size,
        "hd512 decode-via-prefill k.seq_len {} < batch_size {}",
        k.seq_len,
        batch_size
    );
    anyhow::ensure!(
        v.seq_len >= batch_size,
        "hd512 decode-via-prefill v.seq_len {} < batch_size {}",
        v.seq_len,
        batch_size
    );
    k.checked_extent("decode_via_prefill_hd512 k")?;
    v.checked_extent("decode_via_prefill_hd512 v")?;
    anyhow::ensure!(
        plan.head_dim == 512,
        "plan built for head_dim {}, expected 512",
        plan.head_dim
    );
    anyhow::ensure!(
        plan.num_kv_heads == num_kv_heads,
        "plan built for num_kv_heads {}, expected {}",
        plan.num_kv_heads,
        num_kv_heads
    );
    anyhow::ensure!(
        plan.num_qo_heads == num_qo_heads,
        "plan built for num_qo_heads {}, expected {}",
        plan.num_qo_heads,
        num_qo_heads
    );
    anyhow::ensure!(
        plan.total_tokens == plan.batch_size() as usize,
        "decode-via-prefill plan shape mismatch: total_tokens={}, plan_batch={}",
        plan.total_tokens,
        plan.batch_size()
    );

    // Plan tiles laid out for a cta_tile_q override would be misread by the
    // kernel's own 16|32 derivation.
    let kernel_cta_tile_q = unsafe {
        ffi::batch_prefill_cta_tile_q_with_override(
            plan.total_tokens as i32,
            plan.num_qo_heads as i32,
            plan.num_kv_heads as i32,
            plan.head_dim as i32,
            0, // auto-detect, matching the hd512 kernel
        )
    };
    anyhow::ensure!(
        kernel_cta_tile_q == plan.cta_tile_q(),
        "hd512 decode-via-prefill cta_tile_q: plan recorded {}, kernel derives {}; \
         overridden plans are not supported at hd512",
        plan.cta_tile_q(),
        kernel_cta_tile_q
    );

    scatter_decode_kv_into_paged(
        ctx,
        k,
        v,
        kv_buffer,
        layout,
        layer,
        &plan.page_indices_d,
        &plan.page_indptr_d,
        &plan.last_page_len_d,
        positions_d,
        &plan.batch_indices_d,
        plan.batch_size() as usize,
        "batch hd512 decode via prefill",
    )?;

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _gqti) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _gtnr) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::batch_prefill_paged_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            plan.total_tokens as i32,
            plan.batch_size,
            plan.num_tiles,
            stride_page,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "batch_prefill_paged_cuda_hd512 (decode via prefill) failed for layer {layer}, \
             bs={}, tiles={}, qo_heads={num_qo_heads}, kv_heads={num_kv_heads}: {result}{}",
            plan.batch_size,
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// The pages referenced by `plan.page_indices_d()` must lie within
/// `kv_buffer`; the wrapper cannot verify device-side contents — caller
/// contract.
pub fn batch_prefill_paged_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_qo_heads: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        head_dim == 512,
        "hd512 prefill expects head_dim 512, got {}",
        head_dim
    );
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * 512,
        "hd512 prefill q.hidden_dim {} != num_qo_heads {} * 512",
        q.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        output.hidden_dim == num_qo_heads * 512,
        "hd512 prefill output.hidden_dim {} != num_qo_heads {} * 512",
        output.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        q.seq_len >= plan.total_tokens,
        "hd512 prefill q.seq_len {} < plan.total_tokens {}",
        q.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        output.seq_len >= plan.total_tokens,
        "hd512 prefill output.seq_len {} < plan.total_tokens {}",
        output.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        layer < layout.num_layers,
        "hd512 prefill layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    q.checked_extent("batch_prefill_paged_hd512 q")?;
    output.checked_extent("batch_prefill_paged_hd512 output")?;
    anyhow::ensure!(
        plan.head_dim == 512,
        "plan built for head_dim {}, expected 512",
        plan.head_dim
    );
    anyhow::ensure!(
        plan.num_kv_heads == num_kv_heads,
        "plan built for num_kv_heads {}, expected {}",
        plan.num_kv_heads,
        num_kv_heads
    );
    anyhow::ensure!(
        plan.num_qo_heads == num_qo_heads,
        "plan built for num_qo_heads {}, expected {}",
        plan.num_qo_heads,
        num_qo_heads
    );

    // Plan tiles laid out for a cta_tile_q override would be misread by the
    // kernel's own 16|32 derivation.
    let kernel_cta_tile_q = unsafe {
        ffi::batch_prefill_cta_tile_q_with_override(
            plan.total_tokens as i32,
            plan.num_qo_heads as i32,
            plan.num_kv_heads as i32,
            plan.head_dim as i32,
            0, // auto-detect, matching the hd512 kernel
        )
    };
    anyhow::ensure!(
        kernel_cta_tile_q == plan.cta_tile_q(),
        "hd512 prefill cta_tile_q: plan recorded {}, kernel derives {}; \
         overridden plans are not supported at hd512",
        plan.cta_tile_q(),
        kernel_cta_tile_q
    );

    let k_offset = (layer * layout.layer_stride) as i64;
    let v_offset = (layer * layout.layer_stride + layout.kv_block_len) as i64;
    let stride_page = layout.page_stride as i64;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _gqti) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _gtnr) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::batch_prefill_paged_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            plan.total_tokens as i32,
            plan.batch_size(),
            plan.num_tiles,
            stride_page,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "batch_prefill_paged_cuda_hd512 (prefill) failed for layer {layer}, \
             bs={}, tiles={}, qo_heads={num_qo_heads}, kv_heads={num_kv_heads}: {result}{}",
            plan.batch_size(),
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

/// `q` rows are NHD `[seq, heads, 512]`; `k_cache`/`v_cache` are HND
/// `[heads, max_seq_len, 512]` — the head axis is strided by `max_seq_len`.
pub fn single_prefill_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    q_seq_len: usize,
    k_cache: &HiddenStates,
    v_cache: &HiddenStates,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
) -> Result<()> {
    assert_eq!(q.hidden_dim, num_q_heads * 512);
    assert_eq!(output.hidden_dim, q.hidden_dim);
    assert_eq!(output.seq_len, q.seq_len);
    assert_eq!(k_cache.hidden_dim, num_kv_heads * 512);
    assert_eq!(v_cache.hidden_dim, k_cache.hidden_dim);
    assert_eq!(v_cache.seq_len, k_cache.seq_len);
    assert!(kv_len <= k_cache.seq_len);
    assert!(
        q_seq_len <= kv_len,
        "causal prefill q_seq_len {q_seq_len} exceeds kv_len {kv_len}"
    );
    let byte_offset = checked_row_offset(q, row_offset, q_seq_len, "single_prefill_hd512 q")?;
    output.checked_extent("single_prefill_hd512 output")?;
    k_cache.checked_extent("single_prefill_hd512 k_cache")?;
    v_cache.checked_extent("single_prefill_hd512 v_cache")?;
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + byte_offset;
    let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + byte_offset;
    let result = unsafe {
        ffi::single_prefill_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            q_seq_len as i32,
            kv_len as i32,
            k_cache.seq_len as i32,
            1.0f32 / (512.0f32).sqrt(),
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "single_prefill_cuda_hd512 failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Plain-w QK RMSNorm + RoPE prep at head_dim 256 (Gemma 4 local layers):
/// Q and K each land in a contiguous token-major output shaped like its
/// input. No gate, no V, and the plain weight multiply — not the (1+w)
/// offset of the qwen35 hd256 sibling. `single_prefill` consumes a
/// contiguous **HND** cache, so these outputs are reassembled per head
/// before attention, not fed directly. `rotary_dim` must be positive, even
/// and <= 256 (launcher-enforced, surfaced here as `Err`); Gemma 4 local
/// layers pass 256, and a smaller even width keeps the hd512 sibling's
/// partial-RoPE tail semantics.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_prefill_hd256_plain_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    q_out: &mut HiddenStates,
    k_out: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    start_pos: usize,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q.seq_len;
    let q_dim = num_q_heads
        .checked_mul(256)
        .ok_or_else(|| anyhow::anyhow!("hd256 plain prep num_q_heads {num_q_heads} overflows"))?;
    let kv_dim = num_kv_heads
        .checked_mul(256)
        .ok_or_else(|| anyhow::anyhow!("hd256 plain prep num_kv_heads {num_kv_heads} overflows"))?;
    anyhow::ensure!(
        q.hidden_dim == q_dim,
        "hd256 plain prep q.hidden_dim {} != num_q_heads {} * 256",
        q.hidden_dim,
        num_q_heads
    );
    anyhow::ensure!(
        k.hidden_dim == kv_dim,
        "hd256 plain prep k.hidden_dim {} != num_kv_heads {} * 256",
        k.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.hidden_dim,
        "hd256 plain prep q_out.hidden_dim {} != q.hidden_dim {}",
        q_out.hidden_dim,
        q.hidden_dim
    );
    anyhow::ensure!(
        k_out.hidden_dim == k.hidden_dim,
        "hd256 plain prep k_out.hidden_dim {} != k.hidden_dim {}",
        k_out.hidden_dim,
        k.hidden_dim
    );
    anyhow::ensure!(
        k.seq_len == seq_len,
        "hd256 plain prep k.seq_len {} != q.seq_len {seq_len}",
        k.seq_len
    );
    anyhow::ensure!(
        q_out.seq_len == seq_len,
        "hd256 plain prep q_out.seq_len {} != q.seq_len {seq_len}",
        q_out.seq_len
    );
    anyhow::ensure!(
        k_out.seq_len == seq_len,
        "hd256 plain prep k_out.seq_len {} != q.seq_len {seq_len}",
        k_out.seq_len
    );
    anyhow::ensure!(
        q_norm_weight.len == 256,
        "hd256 plain prep q_norm_weight len {} != 256",
        q_norm_weight.len
    );
    anyhow::ensure!(
        k_norm_weight.len == 256,
        "hd256 plain prep k_norm_weight len {} != 256",
        k_norm_weight.len
    );
    let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
        anyhow::anyhow!("hd256 plain prep start_pos {start_pos} + seq_len {seq_len} overflows")
    })?;
    anyhow::ensure!(
        end_pos <= cos_max_pos,
        "hd256 plain prep start_pos {start_pos} + seq_len {seq_len} > \
         cos_max_pos {cos_max_pos}"
    );
    let table_len = cos_max_pos.checked_mul(rotary_dim).ok_or_else(|| {
        anyhow::anyhow!(
            "hd256 plain prep cos_max_pos {cos_max_pos} * rotary_dim {rotary_dim} overflows"
        )
    })?;
    anyhow::ensure!(
        cos_cache.len >= table_len,
        "hd256 plain prep cos_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        cos_cache.len
    );
    anyhow::ensure!(
        sin_cache.len >= table_len,
        "hd256 plain prep sin_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        sin_cache.len
    );
    ensure_vec_backed(q_norm_weight, "hd256 plain prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd256 plain prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd256 plain prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd256 plain prep sin_cache")?;
    // The kernel indexes token-major offsets and `pos * rotary_dim + d` in
    // i32, so the extents and the rope table extent must fit.
    let q_extent = q.checked_extent("hd256 plain prep q")?;
    let k_extent = k.checked_extent("hd256 plain prep k")?;
    q_out.checked_extent("hd256 plain prep q_out")?;
    k_out.checked_extent("hd256 plain prep k_out")?;
    super::checked_i32(q_extent, "hd256 plain prep q extent")?;
    super::checked_i32(k_extent, "hd256 plain prep k extent")?;
    super::checked_i32(table_len, "hd256 plain prep rope table extent")?;
    let num_q_heads = super::checked_i32(num_q_heads, "hd256 plain prep num_q_heads")?;
    let num_kv_heads = super::checked_i32(num_kv_heads, "hd256 plain prep num_kv_heads")?;
    let seq_len = super::checked_i32(seq_len, "hd256 plain prep seq_len")?;
    let start_pos = super::checked_i32(start_pos, "hd256 plain prep start_pos")?;
    let cos_max_pos = super::checked_i32(cos_max_pos, "hd256 plain prep cos_max_pos")?;
    let rotary_dim = super::checked_i32(rotary_dim, "hd256 plain prep rotary_dim")?;

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (ko_ptr, _gko) = k_out.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::qk_norm_rope_prefill_hd256_plain_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            ko_ptr as *mut ffi::Half,
            num_q_heads,
            num_kv_heads,
            seq_len,
            start_pos,
            cos_max_pos,
            rotary_dim,
            rms_eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "qk_norm_rope_prefill_hd256_plain_cuda failed with error \
             {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Causal single-sequence prefill at head_dim 256 over a contiguous **HND**
/// K/V cache — `k[head, pos, dim]` with `k_cache.seq_len` allocated rows per
/// head — matching `single_prefill_cuda_hd256`'s stride contract. Q and the
/// output stay token-major (`[num_q_heads * 256, seq_len]`).
///
/// `sm_scale` is explicit because it is model policy, not geometry: qwen35
/// wants the conventional `rsqrt(head_dim)`, but Gemma 4 runs unscaled
/// attention (`scaling = 1.0` in the HF reference), and a hardcoded rsqrt
/// would silently mis-scale its logits by 16x.
#[allow(clippy::too_many_arguments)]
pub fn single_prefill_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    q_seq_len: usize,
    k_cache: &HiddenStates,
    v_cache: &HiddenStates,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
    sm_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "single_prefill_hd256 sm_scale {sm_scale} must be finite"
    );
    anyhow::ensure!(
        num_q_heads > 0 && num_kv_heads > 0 && q_seq_len > 0,
        "single_prefill_hd256 num_q_heads {num_q_heads}, num_kv_heads \
         {num_kv_heads} and q_seq_len {q_seq_len} must be positive"
    );
    anyhow::ensure!(
        num_q_heads.is_multiple_of(num_kv_heads)
            && SUPPORTED_GQA_GROUP_SIZES.contains(&(num_q_heads / num_kv_heads)),
        "single_prefill_hd256 GQA group {num_q_heads} q / {num_kv_heads} kv \
         heads not in the instantiated sizes {SUPPORTED_GQA_GROUP_SIZES:?}"
    );
    let q_dim = num_q_heads.checked_mul(256).ok_or_else(|| {
        anyhow::anyhow!("single_prefill_hd256 num_q_heads {num_q_heads} overflows")
    })?;
    let kv_dim = num_kv_heads.checked_mul(256).ok_or_else(|| {
        anyhow::anyhow!("single_prefill_hd256 num_kv_heads {num_kv_heads} overflows")
    })?;
    anyhow::ensure!(
        q.hidden_dim == q_dim,
        "single_prefill_hd256 q.hidden_dim {} != num_q_heads {} * 256",
        q.hidden_dim,
        num_q_heads
    );
    anyhow::ensure!(
        output.hidden_dim == q.hidden_dim && output.seq_len == q.seq_len,
        "single_prefill_hd256 output {}x{} != q {}x{}",
        output.hidden_dim,
        output.seq_len,
        q.hidden_dim,
        q.seq_len
    );
    anyhow::ensure!(
        k_cache.hidden_dim == kv_dim,
        "single_prefill_hd256 k_cache.hidden_dim {} != num_kv_heads {} * 256",
        k_cache.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        v_cache.hidden_dim == k_cache.hidden_dim && v_cache.seq_len == k_cache.seq_len,
        "single_prefill_hd256 v_cache {}x{} != k_cache {}x{}",
        v_cache.hidden_dim,
        v_cache.seq_len,
        k_cache.hidden_dim,
        k_cache.seq_len
    );
    anyhow::ensure!(
        kv_len <= k_cache.seq_len,
        "single_prefill_hd256 kv_len {kv_len} exceeds cache rows {}",
        k_cache.seq_len
    );
    anyhow::ensure!(
        q_seq_len <= kv_len,
        "causal prefill q_seq_len {q_seq_len} exceeds kv_len {kv_len}"
    );
    let byte_offset = checked_row_offset(q, row_offset, q_seq_len, "single_prefill_hd256 q")?;
    // The C entry builds its 32-bit strides q_stride_n = num_q_heads * 256
    // and kv_stride_h = cache rows * 256; with positive head counts and
    // rows, each stride is bounded by its buffer's extent, so extents
    // fitting i32 cover the stride products too.
    let q_extent = q.checked_extent("single_prefill_hd256 q")?;
    let out_extent = output.checked_extent("single_prefill_hd256 output")?;
    let k_extent = k_cache.checked_extent("single_prefill_hd256 k_cache")?;
    let v_extent = v_cache.checked_extent("single_prefill_hd256 v_cache")?;
    super::checked_i32(q_extent, "single_prefill_hd256 q extent")?;
    super::checked_i32(out_extent, "single_prefill_hd256 output extent")?;
    super::checked_i32(k_extent, "single_prefill_hd256 k_cache extent")?;
    super::checked_i32(v_extent, "single_prefill_hd256 v_cache extent")?;
    let num_q_heads = super::checked_i32(num_q_heads, "single_prefill_hd256 num_q_heads")?;
    let num_kv_heads = super::checked_i32(num_kv_heads, "single_prefill_hd256 num_kv_heads")?;
    let q_seq_len = super::checked_i32(q_seq_len, "single_prefill_hd256 q_seq_len")?;
    let kv_len = super::checked_i32(kv_len, "single_prefill_hd256 kv_len")?;
    let kv_stride = super::checked_i32(k_cache.seq_len, "single_prefill_hd256 cache rows")?;
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + byte_offset;
    let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + byte_offset;
    let result = unsafe {
        ffi::single_prefill_cuda_hd256(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            num_q_heads,
            num_kv_heads,
            q_seq_len,
            kv_len,
            kv_stride,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "single_prefill_cuda_hd256 failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// QK RMSNorm + partial RoPE for the hd512 paged-prefill prep (Gemma 4
/// global layers). Q is normalised + partially rotated into a contiguous
/// `q_out`; K is normalised + partially rotated straight into the paged KV
/// pool at layer `layer`'s K block (feeds `batch_prefill_paged`, not
/// `single_prefill`). No gate, no V; plain-w RMSNorm (not the (1+w) offset
/// of hd256).
///
/// Validated here: layout geometry consistency (the derived fields are pub
/// and hand-constructible), `layer < layout.num_layers`, `kv_pool` holding
/// whole pages (its capacity in pages becomes the kernel's page bound),
/// `page_indices` covering `[start_pos, start_pos + seq_len)`,
/// `start_pos + seq_len <= cos_max_pos` (host-side layer of the position
/// defence; the kernel `__trap()`s on any out-of-range pos or page id as
/// the second), table lengths, and 512-element norm weights. `rotary_dim`
/// must be positive, even and <= 512 — enforced by the launcher, which
/// returns -1; propagated here as `Err`.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_paged_prefill_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    q_out: &mut HiddenStates,
    kv_pool: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    layer: usize,
    page_indices: &CudaSlice<i32>,
    start_pos: usize,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q.seq_len;
    anyhow::ensure!(
        q.hidden_dim == num_q_heads * 512,
        "hd512 prefill prep q.hidden_dim {} != num_q_heads {} * 512",
        q.hidden_dim,
        num_q_heads
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.hidden_dim,
        "hd512 prefill prep q_out.hidden_dim {} != q.hidden_dim {}",
        q_out.hidden_dim,
        q.hidden_dim
    );
    anyhow::ensure!(
        q_out.seq_len == seq_len,
        "hd512 prefill prep q_out.seq_len {} != q.seq_len {seq_len}",
        q_out.seq_len
    );
    anyhow::ensure!(
        k.hidden_dim == num_kv_heads * 512,
        "hd512 prefill prep k.hidden_dim {} != num_kv_heads {} * 512",
        k.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        k.seq_len == seq_len,
        "hd512 prefill prep k.seq_len {} != q.seq_len {seq_len}",
        k.seq_len
    );
    anyhow::ensure!(
        layout.head_dim == 512,
        "hd512 prefill prep layout.head_dim {} != 512",
        layout.head_dim
    );
    anyhow::ensure!(
        layout.num_kv_heads == num_kv_heads,
        "hd512 prefill prep layout.num_kv_heads {} != num_kv_heads {num_kv_heads}",
        layout.num_kv_heads
    );
    // Keep the pool-capacity division below defined for public layouts.
    anyhow::ensure!(
        layout.page_size > 0 && layout.num_kv_heads > 0,
        "hd512 prefill prep layout.page_size {} / num_kv_heads {} must be positive",
        layout.page_size,
        layout.num_kv_heads
    );
    let kv_block_len = layout.page_size * layout.num_kv_heads * layout.head_dim;
    anyhow::ensure!(
        layout.kv_block_len == kv_block_len,
        "hd512 prefill prep layout.kv_block_len {} != page_size {} * num_kv_heads \
         {} * head_dim {}",
        layout.kv_block_len,
        layout.page_size,
        layout.num_kv_heads,
        layout.head_dim
    );
    anyhow::ensure!(
        layout.layer_stride == 2 * kv_block_len,
        "hd512 prefill prep layout.layer_stride {} != 2 * kv_block_len {kv_block_len}",
        layout.layer_stride
    );
    anyhow::ensure!(
        layout.page_stride == layout.num_layers * layout.layer_stride,
        "hd512 prefill prep layout.page_stride {} != num_layers {} * layer_stride {}",
        layout.page_stride,
        layout.num_layers,
        layout.layer_stride
    );
    anyhow::ensure!(
        layer < layout.num_layers,
        "hd512 prefill prep layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    let num_pages = kv_pool.len() / layout.page_stride;
    anyhow::ensure!(
        kv_pool.len().is_multiple_of(layout.page_stride),
        "hd512 prefill prep kv_pool.len {} is not a multiple of layout.page_stride {}",
        kv_pool.len(),
        layout.page_stride
    );
    anyhow::ensure!(
        num_pages >= 1,
        "hd512 prefill prep kv_pool.len {} holds no whole page (page_stride {})",
        kv_pool.len(),
        layout.page_stride
    );
    anyhow::ensure!(
        q_norm_weight.len == 512,
        "hd512 prefill prep q_norm_weight len {} != 512",
        q_norm_weight.len
    );
    anyhow::ensure!(
        k_norm_weight.len == 512,
        "hd512 prefill prep k_norm_weight len {} != 512",
        k_norm_weight.len
    );
    anyhow::ensure!(
        page_indices.len() * layout.page_size >= start_pos + seq_len,
        "hd512 prefill prep pages cover {} tokens (len {} * page_size {}), need \
         start_pos {start_pos} + seq_len {seq_len}",
        page_indices.len() * layout.page_size,
        page_indices.len(),
        layout.page_size
    );
    anyhow::ensure!(
        start_pos + seq_len <= cos_max_pos,
        "hd512 prefill prep start_pos {start_pos} + seq_len {seq_len} > \
         cos_max_pos {cos_max_pos}"
    );
    anyhow::ensure!(
        cos_cache.len >= cos_max_pos * rotary_dim,
        "hd512 prefill prep cos_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        cos_cache.len
    );
    anyhow::ensure!(
        sin_cache.len >= cos_max_pos * rotary_dim,
        "hd512 prefill prep sin_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        sin_cache.len
    );
    q.checked_extent("hd512 prefill prep q")?;
    q_out.checked_extent("hd512 prefill prep q_out")?;
    k.checked_extent("hd512 prefill prep k")?;
    ensure_vec_backed(q_norm_weight, "hd512 prefill prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd512 prefill prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd512 prefill prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd512 prefill prep sin_cache")?;

    // Derive raw offsets from the validated layout.
    let k_offset_elems = (layer * layout.layer_stride) as i64;
    let page_size = layout.page_size;
    let stride_page = layout.page_stride as i64;

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    // The KV state owns mutation; this call borrows its shared pool handle.
    let (pool_ptr, _gp) = kv_pool.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::qk_norm_partial_rope_paged_prefill_hd512_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            pool_ptr as *mut ffi::Half,
            k_offset_elems,
            pi_ptr as *const i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            start_pos as i32,
            cos_max_pos as i32,
            rotary_dim as i32,
            rms_eps,
            page_size as i32,
            num_pages as i32,
            stride_page,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda failed with error \
             {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Batched QK RMSNorm + partial RoPE for the hd512 decode prep (Gemma 4
/// global layers). Q is normalised + partially rotated into a contiguous
/// `q_out`; K is normalised + partially rotated in place (the caller
/// scatters it into the paged pool afterwards). No gate, no V; plain-w
/// RMSNorm.
///
/// Reading `positions_d` on the host would require a D2H synchronization,
/// so the kernel `__trap()`s on positions outside [0, cos_max_pos).
/// Validated here: table lengths, `positions_d.len() == batch_size`, and
/// 512-element norm weights. `rotary_dim` must be positive, even and <= 512
/// — enforced by the launcher; propagated here as `Err`.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_batched_decode_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    q_out: &mut HiddenStates,
    k: &mut HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    positions_d: &CudaSlice<i32>,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let batch_size = q.seq_len;
    anyhow::ensure!(
        q.hidden_dim == num_q_heads * 512,
        "hd512 decode prep q.hidden_dim {} != num_q_heads {} * 512",
        q.hidden_dim,
        num_q_heads
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.hidden_dim,
        "hd512 decode prep q_out.hidden_dim {} != q.hidden_dim {}",
        q_out.hidden_dim,
        q.hidden_dim
    );
    anyhow::ensure!(
        q_out.seq_len == batch_size,
        "hd512 decode prep q_out.seq_len {} != q.seq_len {batch_size}",
        q_out.seq_len
    );
    anyhow::ensure!(
        k.hidden_dim == num_kv_heads * 512,
        "hd512 decode prep k.hidden_dim {} != num_kv_heads {} * 512",
        k.hidden_dim,
        num_kv_heads
    );
    anyhow::ensure!(
        k.seq_len == batch_size,
        "hd512 decode prep k.seq_len {} != q.seq_len {batch_size}",
        k.seq_len
    );
    anyhow::ensure!(
        q_norm_weight.len == 512,
        "hd512 decode prep q_norm_weight len {} != 512",
        q_norm_weight.len
    );
    anyhow::ensure!(
        k_norm_weight.len == 512,
        "hd512 decode prep k_norm_weight len {} != 512",
        k_norm_weight.len
    );
    anyhow::ensure!(
        positions_d.len() == batch_size,
        "hd512 decode prep positions len {} != batch_size {batch_size}",
        positions_d.len()
    );
    anyhow::ensure!(
        cos_cache.len >= cos_max_pos * rotary_dim,
        "hd512 decode prep cos_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        cos_cache.len
    );
    anyhow::ensure!(
        sin_cache.len >= cos_max_pos * rotary_dim,
        "hd512 decode prep sin_cache len {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        sin_cache.len
    );
    q.checked_extent("hd512 decode prep q")?;
    q_out.checked_extent("hd512 decode prep q_out")?;
    k.checked_extent("hd512 decode prep k")?;
    ensure_vec_backed(q_norm_weight, "hd512 decode prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd512 decode prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd512 decode prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd512 decode prep sin_cache")?;

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pos_ptr, _gp) = positions_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::qk_norm_partial_rope_batched_decode_hd512_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *mut ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            cos_max_pos as i32,
            qo_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            batch_size as i32,
            rotary_dim as i32,
            rms_eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "qk_norm_partial_rope_batched_decode_hd512_cuda failed with error \
             {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}
