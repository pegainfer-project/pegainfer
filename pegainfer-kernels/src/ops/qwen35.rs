//! Qwen3.5 HD256 attention, GDN decode and explicit FlashInfer GDN prefill.

mod decode_gemm;
mod gdn;

use anyhow::Context as _;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
pub use decode_gemm::Qwen35DecodeGemm;
pub use gdn::Qwen35GdnAot;
pub use gdn::Qwen35GdnGeometry;
pub use gdn::Qwen35GdnWorkspace;
use half::bf16;

use super::PrefillPagedPlan;
use super::attention::PagedGeometry;
use super::attention::checked_paged_geometry;
use crate::ffi;
use crate::paged_kv::PagedKvLayout;
use crate::tensor::Columns;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

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

    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        stride_page,
        ..
    } = checked_paged_geometry(
        op_name,
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;

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

    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        stride_page,
        ..
    } = checked_paged_geometry(
        "batch hd256 decode",
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;

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

/// Split-KV variant of [`paged_attention_batch_decode_hd256_into`]: the KV range
/// is divided across `split_kv` CTAs per (kv head, request) and the partial
/// softmax states are merged by a second kernel. The caller owns the scratch
/// buffers and must size them for `[split_kv, batch_size, num_qo_heads, 256]`
/// (f32) and `[split_kv, batch_size, num_qo_heads]` (f32, twice).
///
/// It wins where the FlashInfer path starves the device: the split grid turns a
/// single-request decode from 4 CTAs into `4 * split_kv`. The kernel is shaped
/// for Qwen3.5-4B's GQA group (16 query heads over 4 kv heads at head_dim 256)
/// and refuses other geometries rather than serving a wrong answer.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_split_hd256_into(
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
    kv_chunk_size_d: &CudaSlice<i32>,
    partial_o: &mut CudaSlice<f32>,
    partial_m: &mut CudaSlice<f32>,
    partial_l: &mut CudaSlice<f32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
    split_kv: usize,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let page_size = layout.page_size;

    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        stride_page,
        ..
    } = checked_paged_geometry(
        "batch hd256 split decode",
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;

    let partial_elems = split_kv
        .checked_mul(batch_size)
        .and_then(|n| n.checked_mul(num_qo_heads))
        .context("split decode partial buffer shape overflow")?;
    anyhow::ensure!(
        split_kv >= 1
            && partial_o.len() >= partial_elems * head_dim
            && partial_m.len() >= partial_elems
            && partial_l.len() >= partial_elems,
        "split decode scratch too small: splits={split_kv}, bs={batch_size}, heads={num_qo_heads}, \
         dim={head_dim}, o={}, m={}, l={}",
        partial_o.len(),
        partial_m.len(),
        partial_l.len()
    );

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = page_indptr_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = kv_chunk_size_d.device_ptr(&ctx.stream);
    let (po_ptr, _gpo) = partial_o.device_ptr_mut(&ctx.stream);
    let (pm_ptr, _gpm) = partial_m.device_ptr_mut(&ctx.stream);
    let (pl_ptr, _gpl) = partial_l.device_ptr_mut(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    // The current token's K/V has to be in the pool before the read, exactly as
    // the non-split path orders it.
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
        "batch hd256 split decode",
    )?;

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    let result = unsafe {
        ffi::paged_attention_decode_split_hd256_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            kcs_ptr as *const i32,
            po_ptr as *mut f32,
            pm_ptr as *mut f32,
            pl_ptr as *mut f32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            batch_size as i32,
            split_kv as i32,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_split_hd256_cuda failed with error {result}{}",
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
        batch_size == plan.total_tokens && batch_size == plan.batch_size() as usize,
        "decode-via-prefill plan shape mismatch: bs={batch_size}, total_tokens={}, plan_batch={}",
        plan.total_tokens,
        plan.batch_size()
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

    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        stride_page,
        ..
    } = checked_paged_geometry(
        "batch hd256 decode via prefill",
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;
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
            plan.batch_size(),
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

/// Gated full attention over one prompt chunk at head dim 256, the Qwen3.5
/// shape: `q_full` holds `[q | gate]` per head, `k` and `v` the projections.
/// `(1 + w)` q/k norm and partial RoPE from `start_pos`, K/V written into the
/// pool at `layer` over the plan's pages, causal FlashInfer prefill, then the
/// sigmoid of the gate half on `output`. `q_prepped` keeps the normed and
/// rotated Q.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_hd256_into(
    ctx: &DeviceContext,
    q_full: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    start_pos: usize,
    rotary_dim: usize,
    rms_eps: f32,
    q_prepped: &mut HiddenStates,
    output: &mut HiddenStates,
) -> Result<()> {
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        head_dim == 256,
        "hd256 prefill: the layout has head_dim {head_dim}"
    );
    let num_q_heads = output.hidden_dim / head_dim;
    anyhow::ensure!(
        num_q_heads * head_dim == output.hidden_dim
            && q_prepped.hidden_dim == output.hidden_dim
            && q_full.hidden_dim == 2 * output.hidden_dim
            && k.hidden_dim == num_kv_heads * head_dim
            && v.hidden_dim == num_kv_heads * head_dim,
        "hd256 prefill widths: q_full {} q {} k {} v {} output {} for {num_kv_heads} kv heads",
        q_full.hidden_dim,
        q_prepped.hidden_dim,
        k.hidden_dim,
        v.hidden_dim,
        output.hidden_dim
    );
    let seq_len = q_full.seq_len;
    anyhow::ensure!(
        [
            k.seq_len,
            v.seq_len,
            q_prepped.seq_len,
            output.seq_len,
            plan.total_tokens
        ]
        .iter()
        .all(|&rows| rows == seq_len),
        "hd256 prefill rows: q_full {seq_len} k {} v {} q {} output {} plan {}",
        k.seq_len,
        v.seq_len,
        q_prepped.seq_len,
        output.seq_len,
        plan.total_tokens
    );
    q_full.checked_extent("hd256 prefill q_full")?;
    k.checked_extent("hd256 prefill k")?;
    v.checked_extent("hd256 prefill v")?;
    q_prepped.checked_extent("hd256 prefill q_prepped")?;
    output.checked_extent("hd256 prefill output")?;
    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        page_size,
        num_pages,
        stride_page,
    } = checked_paged_geometry(
        "hd256 prefill",
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;
    anyhow::ensure!(
        plan.max_page_index < num_pages,
        "hd256 prefill plan references page {} but the pool holds {num_pages} pages; the kernels \
         compute addresses from page ids with no device-side bounds check",
        plan.max_page_index
    );
    anyhow::ensure!(
        plan.max_last_page_len <= page_size,
        "hd256 prefill plan last page length {} exceeds page_size {page_size}",
        plan.max_last_page_len
    );
    let start_pos_d: CudaSlice<i32> = ctx.stream.clone_htod(&[crate::ops::checked_i32(
        start_pos,
        "hd256 prefill start_pos",
    )?])?;
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    let (qf_ptr, _gqf) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (qp_ptr, _gqp) = q_prepped.data.device_ptr_mut(&ctx.stream);
    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (sp_ptr, _gsp) = start_pos_d.device_ptr(&ctx.stream);
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
    let stream = crate::tensor::active_cu_stream(ctx);

    unsafe {
        ffi::prefill_attention_hd256_prep_paged_cuda(
            qf_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qp_ptr as *mut ffi::Half,
            buf_ptr as *mut ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            rotary_dim as i32,
            rms_eps,
            layout.page_size as i32,
            stride_page,
            stream,
        );
    }
    let result = unsafe {
        ffi::batch_prefill_paged_cuda_hd256(
            qp_ptr as *const ffi::Half,
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
            num_q_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            seq_len as i32,
            plan.batch_size(),
            plan.num_tiles,
            stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "batch_prefill_paged_cuda_hd256 failed for layer {layer}, seq_len={seq_len}, tiles={}, \
             qo_heads={num_q_heads}, kv_heads={num_kv_heads}: {result}{}",
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }
    unsafe {
        ffi::attention_gate_batch_hd256_cuda(
            qf_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            num_q_heads as i32,
            seq_len as i32,
            stream,
        );
    }
    Ok(())
}

/// GDN key head dim baked into the kernels: the decode kernel's tiling and shared
/// memory layout (`GDR_KEY_DIM` in `gated_delta_rule.cu`) and the chunk-wise
/// prefill's Triton AOT specialization keys both hardcode it.
pub const GDN_AOT_KEY_HEAD_DIM: usize = 128;

/// GDN value head dim, baked in for the same reasons (`GDR_VAL_DIM`). It is also
/// the width of the per-head gated norm's weight.
pub const GDN_AOT_VALUE_HEAD_DIM: usize = 128;

/// Largest depthwise conv1d kernel the in-tree conv carries state for:
/// `conv1d.cu` sizes its state as `kernel_size - 1 <= 3` taps.
pub const LINEAR_CONV_MAX_KERNEL_DIM: usize = 4;

/// One Gated DeltaNet decode step for a batch of requests.
///
/// `qkv` is the **post-conv1d, post-SiLU** projection, `[q | k | v]` per token:
/// `num_key_heads * key_dim` query columns, then the same number of key columns,
/// then `num_value_heads * val_dim` value columns. `state_ptrs` is a device array
/// of per-request states, each `[num_value_heads, key_dim, val_dim]` f32 with
/// `val_dim` contiguous (the FLA `[H, K, V]` convention); a request's state is
/// read and updated in place. `output` is `num_value_heads * val_dim` wide.
///
/// Head dims must be the baked-in ones ([`GDN_AOT_KEY_HEAD_DIM`],
/// [`GDN_AOT_VALUE_HEAD_DIM`]); head counts are free, so a different value-head
/// count needs no rebuild.
///
/// `b_proj` and `a_proj` may each be one band of a fused beta/alpha projection;
/// their slot stride is then the tensor they are a band of.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_decode_batch_into<'a>(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: impl Into<Columns<'a>>,
    a_proj: impl Into<Columns<'a>>,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state_ptrs: &CudaSlice<u64>,
    output: &mut HiddenStates,
    batch_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) {
    let b_proj = b_proj.into();
    let a_proj = a_proj.into();
    b_proj.assert_in_row("gated_delta_rule_decode b_proj");
    a_proj.assert_in_row("gated_delta_rule_decode a_proj");
    assert_eq!(qkv.seq_len, batch_size);
    assert_eq!(b_proj.states.seq_len, batch_size);
    assert_eq!(a_proj.states.seq_len, batch_size);
    assert_eq!(output.seq_len, batch_size);
    assert!(
        num_key_heads > 0 && num_value_heads > 0 && num_value_heads.is_multiple_of(num_key_heads),
        "value heads {num_value_heads} must be a positive whole number of \
         key-head groups of {num_key_heads}"
    );
    assert_eq!(
        qkv.hidden_dim,
        2 * num_key_heads * key_dim + num_value_heads * val_dim
    );
    assert_eq!(b_proj.width, num_value_heads);
    assert_eq!(a_proj.width, num_value_heads);
    assert_eq!(output.hidden_dim, num_value_heads * val_dim);
    assert_eq!(dt_bias.len, num_value_heads);
    assert_eq!(a_log.len(), num_value_heads);
    assert_eq!(key_dim, GDN_AOT_KEY_HEAD_DIM);
    assert_eq!(val_dim, GDN_AOT_VALUE_HEAD_DIM);
    assert!(state_ptrs.len() >= batch_size);

    let (b_base, _gb) = b_proj.states.data.device_ptr(&ctx.stream);
    let b_ptr = b_base + (b_proj.col * std::mem::size_of::<bf16>()) as u64;
    let (a_base, _ga) = a_proj.states.data.device_ptr(&ctx.stream);
    let a_ptr = a_base + (a_proj.col * std::mem::size_of::<bf16>()) as u64;
    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (state_ptrs, _gsp) = state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_decode_batch_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            state_ptrs as *const u64,
            o_ptr as *mut ffi::Half,
            batch_size as i32,
            num_key_heads as i32,
            num_value_heads as i32,
            key_dim as i32,
            val_dim as i32,
            b_proj.states.hidden_dim as i32,
            a_proj.states.hidden_dim as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// One causal depthwise conv1d decode step for a batch of requests, fused with
/// the SiLU the GDN block consumes.
///
/// `x` is the **pre-conv** projection row, `num_channels` wide. `conv_weight` is
/// the flattened `[num_channels, 1, kernel_size]` depthwise weight. `conv_state_ptrs`
/// is a device array of per-request states, each `num_channels * (kernel_size - 1)`
/// bf16 taps holding the previous inputs oldest-first; it is updated in place from
/// `x`, so `kernel_size` must not exceed [`LINEAR_CONV_MAX_KERNEL_DIM`]. `out` is
/// `silu(bf16(conv))`, rounded exactly like the reference (`conv1d` then `silu` in
/// bf16), not a fused `silu(sum)`.
///
/// Slots of the conv input stride by the tensor's own row width, so `x` may be
/// the qkv band of a fused qkv+z projection. The conv output is this operator's
/// own buffer and keeps the band's width as its slot stride.
pub fn conv1d_decode_batch_into<'a>(
    ctx: &DeviceContext,
    x: impl Into<Columns<'a>>,
    conv_weight: &DeviceVec,
    conv_state_ptrs: &CudaSlice<u64>,
    out: &mut HiddenStates,
    kernel_size: usize,
) {
    let x = x.into();
    x.assert_in_row("conv1d_decode x");
    let batch_size = x.states.seq_len;
    let num_channels = x.width;
    assert_eq!(out.hidden_dim, num_channels);
    assert_eq!(out.seq_len, batch_size);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert!(kernel_size <= LINEAR_CONV_MAX_KERNEL_DIM);
    assert!(conv_state_ptrs.len() >= batch_size);

    let (x_ptr, _gx) = x.states.data.device_ptr(&ctx.stream);
    let x_ptr = x_ptr + (x.col * std::mem::size_of::<bf16>()) as u64;
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptrs, _gs) = conv_state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_decode_batch_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptrs as *const u64,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            x.states.hidden_dim as i32,
            batch_size as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// Causal depthwise conv1d prefill over a whole `[T, C]` tensor, fused with the
/// SiLU the GDN block consumes.
///
/// `x_seq` must be the whole tensor, because the kernel strides by its
/// `hidden_dim`. `conv_state` holds the previous `kernel_size - 1` inputs per
/// channel, oldest first, and is updated in place to the last taps of `x_seq`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_prefill_batch_into(
    ctx: &DeviceContext,
    x_seq: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out_seq: &mut HiddenStates,
    kernel_size: usize,
) {
    let num_channels = x_seq.hidden_dim;
    assert_eq!(out_seq.hidden_dim, num_channels);
    assert_eq!(out_seq.seq_len, x_seq.seq_len);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert_eq!(conv_state.len, num_channels * (kernel_size - 1));

    let (x_ptr, _gx) = x_seq.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = conv_state.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out_seq.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_prefill_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            x_seq.seq_len as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// Scratch buffers for one chunk-wise GDR prefill call.
///
/// The chunk-wise path is a multi-stage pipeline rather than one opaque kernel
/// launch, so its intermediates are explicit and pre-allocated: one scratch per
/// pass length, batch size 1, forward only, chunk size 64.
pub struct GdrChunkwiseScratch35 {
    /// Chunk-local cumulative gate, fp32: [seq_len, num_value_heads]
    g_cumsum: CudaSlice<f32>,
    /// Beta values, fp32: [seq_len, num_value_heads]
    beta: CudaSlice<f32>,

    /// Expanded + normalized q in token-major layout: [seq_len, num_value_heads * key_dim]
    q_expanded: HiddenStates,
    /// Expanded + normalized k in token-major layout: [seq_len, num_value_heads * key_dim]
    k_expanded: HiddenStates,
    /// Raw v in token-major layout: [seq_len, num_value_heads * value_dim]
    v_raw: HiddenStates,

    /// Chunk attention matrix storage, fp32: [seq_len, num_value_heads, chunk_size]
    a_tril: CudaSlice<f32>,
    /// Inverse (I + A)^-1 in bf16: [seq_len, num_value_heads, chunk_size]
    a_inv: CudaSlice<bf16>,

    /// Prepared W tensor in token-major layout: [seq_len, num_value_heads * key_dim]
    w: HiddenStates,
    /// Prepared U tensor in token-major layout: [seq_len, num_value_heads * value_dim]
    u: HiddenStates,
    /// New value tensor consumed by chunk output stage: [seq_len, num_value_heads * value_dim]
    v_new: HiddenStates,

    /// Per-chunk recurrent state snapshots, fp32: [num_chunks, num_value_heads, key_dim, value_dim]
    chunk_state: CudaSlice<f32>,
}

impl GdrChunkwiseScratch35 {
    pub const CHUNK_SIZE: usize = 64;

    pub fn from_dims(
        ctx: &DeviceContext,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let kv_hidden_dim = num_value_heads * key_dim;
        let vv_hidden_dim = num_value_heads * value_dim;
        let num_chunks = seq_len.div_ceil(Self::CHUNK_SIZE);

        let g_cumsum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc g_cumsum failed: {}", e))?;
        let beta: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc beta failed: {}", e))?;
        let a_tril: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_tril failed: {}", e))?;
        let a_inv: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_inv failed: {}", e))?;
        let chunk_state: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(num_chunks * num_value_heads * value_dim * key_dim)
            .map_err(|e| anyhow::anyhow!("Alloc chunk_state failed: {}", e))?;

        Ok(Self {
            g_cumsum,
            beta,
            q_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            k_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            v_raw: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            a_tril,
            a_inv,
            w: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            u: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            v_new: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            chunk_state,
        })
    }

    pub fn num_chunks(seq_len: usize) -> usize {
        seq_len.div_ceil(Self::CHUNK_SIZE)
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_prepare_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    q_out: &mut HiddenStates,
    k_out: &mut HiddenStates,
    v_out: &mut HiddenStates,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    num_key_heads: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (q_out_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (k_out_ptr, _gko) = k_out.data.device_ptr_mut(&ctx.stream);
    let (v_out_ptr, _gvo) = v_out.data.device_ptr_mut(&ctx.stream);
    let (g_out_ptr, _ggo) = g_out.device_ptr_mut(&ctx.stream);
    let (beta_out_ptr, _gbetao) = beta_out.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            num_key_heads as i32,
            num_value_heads as i32,
            qkv.hidden_dim as i32,
            qkv.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_cumsum_inplace(
    ctx: &DeviceContext,
    g_cumsum: &mut CudaSlice<f32>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (g_ptr, _gg) = g_cumsum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_cumsum_cuda(
            g_ptr as *const f32,
            g_ptr as *mut f32,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_a_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    a_tril: &mut CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_tril.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_a_cuda(
            k_ptr as *const ffi::Half,
            g_ptr as *const f32,
            beta_ptr as *const f32,
            a_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_solve_into(
    ctx: &DeviceContext,
    a_tril: &CudaSlice<f32>,
    a_inv: &mut CudaSlice<bf16>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (a_ptr, _ga) = a_tril.device_ptr(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_solve_cuda(
            a_ptr as *const f32,
            ai_ptr as *mut ffi::Half,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_recompute_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    v: &HiddenStates,
    beta: &CudaSlice<f32>,
    w: &mut HiddenStates,
    u: &mut HiddenStates,
    a_inv: &CudaSlice<bf16>,
    g_cumsum: &CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr_mut(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr_mut(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_recompute_cuda(
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            beta_ptr as *const f32,
            w_ptr as *mut ffi::Half,
            u_ptr as *mut ffi::Half,
            ai_ptr as *const ffi::Half,
            g_ptr as *const f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_state_stage_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    w: &HiddenStates,
    u: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    chunk_state: &mut CudaSlice<f32>,
    v_new: &mut HiddenStates,
    num_value_heads: usize,
) -> Result<()> {
    assert_eq!(k.hidden_dim, w.hidden_dim);
    assert_eq!(u.hidden_dim, v_new.hidden_dim);
    assert_eq!(k.seq_len, w.seq_len);
    assert_eq!(k.seq_len, u.seq_len);
    assert_eq!(k.seq_len, v_new.seq_len);

    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr_mut(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_state_cuda(
            k_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            g_ptr as *const f32,
            s_ptr as *const f32,
            cs_ptr as *mut f32,
            vn_ptr as *mut ffi::Half,
            s_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_o_stage_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v_new: &HiddenStates,
    chunk_state: &CudaSlice<f32>,
    g_cumsum: &CudaSlice<f32>,
    output: &mut HiddenStates,
    num_value_heads: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(q.hidden_dim, k.hidden_dim);
    assert_eq!(v_new.hidden_dim, output.hidden_dim);
    assert_eq!(q.seq_len, k.seq_len);
    assert_eq!(q.seq_len, v_new.seq_len);
    assert_eq!(q.seq_len, output.seq_len);

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_o_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            vn_ptr as *const ffi::Half,
            cs_ptr as *const f32,
            g_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            q.seq_len as i32,
            num_value_heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Chunk-wise gated delta rule prefill: the Triton AOT path, seven stages over
/// one pre-allocated scratch.
///
/// `qkv` is the **post-conv1d, post-SiLU** projection, `[q | k | v]` per token
/// (`num_key_heads * key_dim` query columns, the same number of key columns, then
/// `num_value_heads * val_dim` value columns); `b_proj` and `a_proj` are the
/// contiguous `[T, num_value_heads]` beta and alpha projections. `state` is the
/// `[num_value_heads, key_dim, val_dim]` f32 recurrent state, read as the initial
/// state and updated in place. The head dims must be the baked-in ones
/// ([`GDN_AOT_KEY_HEAD_DIM`], [`GDN_AOT_VALUE_HEAD_DIM`]); head counts are runtime
/// arguments, and a key head serves the contiguous group of value heads
/// `num_value_heads / num_key_heads` wide. The scratch must have been sized for
/// exactly `qkv.seq_len` tokens.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_prefill_chunkwise_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    scratch: &mut GdrChunkwiseScratch35,
    output: &mut HiddenStates,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    assert_eq!(scratch.q_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.k_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_raw.seq_len, qkv.seq_len);
    assert_eq!(scratch.w.seq_len, qkv.seq_len);
    assert_eq!(scratch.u.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_new.seq_len, qkv.seq_len);
    assert_eq!(scratch.q_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.k_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.v_raw.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.w.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.u.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.v_new.hidden_dim, num_value_heads * val_dim);

    let expected_gate_len = qkv.seq_len * num_value_heads;
    let expected_chunk_a_len = qkv.seq_len * num_value_heads * GdrChunkwiseScratch35::CHUNK_SIZE;
    let expected_chunk_ai_len = expected_chunk_a_len;
    let expected_chunk_state_len =
        GdrChunkwiseScratch35::num_chunks(qkv.seq_len) * num_value_heads * val_dim * key_dim;
    assert_eq!(scratch.g_cumsum.len(), expected_gate_len);
    assert_eq!(scratch.beta.len(), expected_gate_len);
    assert_eq!(scratch.a_tril.len(), expected_chunk_a_len);
    assert_eq!(scratch.a_inv.len(), expected_chunk_ai_len);
    assert_eq!(scratch.chunk_state.len(), expected_chunk_state_len);

    gated_delta_rule_prefill_chunk_prepare_into(
        ctx,
        qkv,
        b_proj,
        a_proj,
        dt_bias,
        a_log,
        &mut scratch.q_expanded,
        &mut scratch.k_expanded,
        &mut scratch.v_raw,
        &mut scratch.g_cumsum,
        &mut scratch.beta,
        num_key_heads,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill prepare failed: {e}"))?;
    gated_delta_rule_prefill_chunk_cumsum_inplace(
        ctx,
        &mut scratch.g_cumsum,
        qkv.seq_len,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill cumsum failed: {e}"))?;
    gated_delta_rule_prefill_chunk_a_into(
        ctx,
        &scratch.k_expanded,
        &scratch.g_cumsum,
        &scratch.beta,
        &mut scratch.a_tril,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill A stage failed: {e}"))?;
    gated_delta_rule_prefill_chunk_solve_into(
        ctx,
        &scratch.a_tril,
        &mut scratch.a_inv,
        qkv.seq_len,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill solve failed: {e}"))?;
    gated_delta_rule_prefill_chunk_recompute_into(
        ctx,
        &scratch.k_expanded,
        &scratch.v_raw,
        &scratch.beta,
        &mut scratch.w,
        &mut scratch.u,
        &scratch.a_inv,
        &scratch.g_cumsum,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill recompute failed: {e}"))?;
    gated_delta_rule_prefill_chunk_state_stage_into(
        ctx,
        &scratch.k_expanded,
        &scratch.w,
        &scratch.u,
        &scratch.g_cumsum,
        state,
        &mut scratch.chunk_state,
        &mut scratch.v_new,
        num_value_heads,
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill state stage failed: {e}"))?;
    gated_delta_rule_prefill_chunk_o_stage_into(
        ctx,
        &scratch.q_expanded,
        &scratch.k_expanded,
        &scratch.v_new,
        &scratch.chunk_state,
        &scratch.g_cumsum,
        output,
        num_value_heads,
        1.0 / (key_dim as f32).sqrt(),
    )
    .map_err(|e| anyhow::anyhow!("GDR prefill output stage failed: {e}"))
}

#[cfg(test)]
mod tests;
