//! Qwen3.5 HD256 attention and explicit FlashInfer GDN candidate operations.

mod decode_gemm;
mod gdn;
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

#[cfg(test)]
mod tests;
