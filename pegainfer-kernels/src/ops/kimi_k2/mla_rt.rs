use anyhow::Result;
use anyhow::bail;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use super::mla::KIMI_K2_MLA_KV_LORA_RANK;
use super::mla::KIMI_K2_MLA_ROPE_DIM;
use super::mla::KimiMlaPagedKvLayout;
use super::mla::validate_paged_layout;
use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::GpuTensor;
use crate::tensor::HiddenStates;

#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_rope_split_decode_rt(
    ctx: &DeviceContext,
    q_proj: &HiddenStates,
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    q_nope: &mut HiddenStates,
    q_pe: &mut HiddenStates,
    append_kpe: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    local_heads: usize,
) -> Result<()> {
    let batch_size = q_proj.seq_len;
    let (q_proj_ptr, _g0) = q_proj.data.device_ptr(&ctx.stream);
    let (k_rope_ptr, _g1) = k_rope.data.device_ptr(&ctx.stream);
    let (cos_ptr, _g2) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _g3) = sin.device_ptr(&ctx.stream);
    let (positions_ptr, _g4) = positions_d.device_ptr(&ctx.stream);
    let (q_nope_ptr, _g5) = q_nope.data.device_ptr_mut(&ctx.stream);
    let (q_pe_ptr, _g6) = q_pe.data.device_ptr_mut(&ctx.stream);
    let (append_kpe_ptr, _g7) = append_kpe.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_rope_split_decode_cuda(
            q_proj_ptr as *const ffi::Half,
            k_rope_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            positions_ptr as *const i32,
            q_nope_ptr as *mut ffi::Half,
            q_pe_ptr as *mut ffi::Half,
            append_kpe_ptr as *mut ffi::Half,
            batch_size as i32,
            local_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_mla_absorb_q_nope_rt(
    ctx: &DeviceContext,
    kv_b_proj: &DeviceMatrix,
    q_nope: &HiddenStates,
    q_abs_nope: &mut HiddenStates,
    local_heads: usize,
) -> Result<()> {
    let (weight_ptr, _g0) = kv_b_proj.data.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q_nope.data.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = q_abs_nope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_absorb_q_nope_cuda(
            weight_ptr as *const ffi::Half,
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            q_nope.seq_len as i32,
            local_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_mla_absorb_q_nope_cuda failed with cudaError={result}");
    }
    Ok(())
}

pub fn kimi_mla_v_up_rt(
    ctx: &DeviceContext,
    kv_b_proj: &DeviceMatrix,
    latent: &HiddenStates,
    output: &mut HiddenStates,
    local_heads: usize,
) -> Result<()> {
    let (weight_ptr, _g0) = kv_b_proj.data.device_ptr(&ctx.stream);
    let (latent_ptr, _g1) = latent.data.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_v_up_cuda(
            weight_ptr as *const ffi::Half,
            latent_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            latent.seq_len as i32,
            local_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!("kimi_mla_v_up_cuda failed with cudaError={result}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kimi_flashinfer_batch_decode_mla_rt(
    ctx: &DeviceContext,
    q_abs_nope: &HiddenStates,
    q_pe: &HiddenStates,
    output: &mut HiddenStates,
    ckv_cache: &CudaSlice<half::bf16>,
    kpe_cache: &CudaSlice<half::bf16>,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    sm_scale: f32,
    local_heads: usize,
) -> Result<()> {
    validate_paged_layout(layout, page_indices_d, page_indptr_d, last_page_len_d)?;
    let (q_abs_ptr, _g0) = q_abs_nope.data.device_ptr(&ctx.stream);
    let (q_pe_ptr, _g1) = q_pe.data.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = output.data.device_ptr_mut(&ctx.stream);
    let (ckv_ptr, _g3) = ckv_cache.device_ptr(&ctx.stream);
    let (kpe_ptr, _g4) = kpe_cache.device_ptr(&ctx.stream);
    let (pi_ptr, _g5) = page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _g6) = page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _g7) = last_page_len_d.device_ptr(&ctx.stream);
    let (ri_ptr, _g8) = request_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _g9) = kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _g10) = kv_chunk_size_d.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::kimi_flashinfer_batch_decode_mla_cuda(
            q_abs_ptr as *const ffi::Half,
            q_pe_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            ckv_ptr as *const ffi::Half,
            kpe_ptr as *const ffi::Half,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            local_heads as i32,
            layout.ckv_stride_page as i64,
            layout.ckv_stride_n as i64,
            layout.kpe_stride_page as i64,
            layout.kpe_stride_n as i64,
            layout.page_size as i32,
            layout.batch_size as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!(
            "kimi_flashinfer_batch_decode_mla_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Assemble the suffix rows of prefill attention inputs. `start_pos` is the
/// cached-prefix length (0 for a cold prefill): q covers the suffix tokens,
/// k/v rows land at `start_pos..start_pos + seq_len` inside caches sized for
/// `start_pos + seq_len` tokens, and RoPE positions are absolute.
#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_rope_assemble_prefill_rt(
    ctx: &DeviceContext,
    q_proj: &HiddenStates,
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    kv_b: &HiddenStates,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    q_attn: &mut HiddenStates,
    k_cache: &mut CudaSlice<half::bf16>,
    v_cache: &mut CudaSlice<half::bf16>,
    start_pos: usize,
    local_heads: usize,
) -> Result<()> {
    let seq_len = q_proj.seq_len;
    let kv_len = start_pos + seq_len;
    let (q_ptr, _g0) = q_proj.data.device_ptr(&ctx.stream);
    let (kr_ptr, _g1) = k_rope.data.device_ptr(&ctx.stream);
    let (kv_ptr, _g2) = kv_b.data.device_ptr(&ctx.stream);
    let (cos_ptr, _g3) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _g4) = sin.device_ptr(&ctx.stream);
    let (qa_ptr, _g5) = q_attn.data.device_ptr_mut(&ctx.stream);
    let (kc_ptr, _g6) = k_cache.device_ptr_mut(&ctx.stream);
    let (vc_ptr, _g7) = v_cache.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_rope_assemble_prefill_cuda(
            q_ptr as *const ffi::Half,
            kr_ptr as *const ffi::Half,
            kv_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qa_ptr as *mut ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            seq_len as i32,
            start_pos as i32,
            kv_len as i32,
            local_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Scatter pooled latent ckv pages into a contiguous token-major buffer so
/// the kv_b decompression GEMM can run over a cached prefix. `pages_offset`
/// is the slot's start inside the uploaded page-index array (FlashInfer CSR).
pub fn kimi_mla_gather_cached_ckv_rt(
    ctx: &DeviceContext,
    ckv_cache: &CudaSlice<half::bf16>,
    page_indices_d: &CudaSlice<i32>,
    pages_offset: usize,
    layout: &KimiMlaPagedKvLayout,
    ckv_out: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
) -> Result<()> {
    let cached_len = ckv_out.seq_len;
    let (cache_ptr, _g0) = ckv_cache.device_ptr(&ctx.stream);
    let (pages_ptr, _g1) = page_indices_d.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = ckv_out.data.device_ptr_mut(&ctx.stream);
    let slot_pages_ptr = (pages_ptr as *const i32).wrapping_add(pages_offset);
    let result = unsafe {
        ffi::kimi_mla_gather_cached_ckv_cuda(
            cache_ptr as *const ffi::Half,
            slot_pages_ptr,
            out_ptr as *mut ffi::Half,
            cached_len as i32,
            layout.page_size as i32,
            layout.ckv_stride_page as i64,
            layout.ckv_stride_n as i64,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Build k/v cache rows `0..cached_len` from the decompressed latent prefix
/// (`kv_b`, the GEMM output over the gathered ckv) plus the pooled kpe pages.
/// Pool kpe is post-RoPE, so it is broadcast per head without re-rotation.
#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_assemble_cached_kv_rt(
    ctx: &DeviceContext,
    kv_b: &HiddenStates,
    kpe_cache: &CudaSlice<half::bf16>,
    page_indices_d: &CudaSlice<i32>,
    pages_offset: usize,
    layout: &KimiMlaPagedKvLayout,
    k_cache: &mut CudaSlice<half::bf16>,
    v_cache: &mut CudaSlice<half::bf16>,
    kv_len: usize,
    local_heads: usize,
) -> Result<()> {
    let cached_len = kv_b.seq_len;
    let (kv_ptr, _g0) = kv_b.data.device_ptr(&ctx.stream);
    let (kpe_ptr, _g1) = kpe_cache.device_ptr(&ctx.stream);
    let (pages_ptr, _g2) = page_indices_d.device_ptr(&ctx.stream);
    let (kc_ptr, _g3) = k_cache.device_ptr_mut(&ctx.stream);
    let (vc_ptr, _g4) = v_cache.device_ptr_mut(&ctx.stream);
    let slot_pages_ptr = (pages_ptr as *const i32).wrapping_add(pages_offset);
    let result = unsafe {
        ffi::kimi_mla_assemble_cached_kv_cuda(
            kv_ptr as *const ffi::Half,
            kpe_ptr as *const ffi::Half,
            slot_pages_ptr,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            cached_len as i32,
            kv_len as i32,
            local_heads as i32,
            layout.page_size as i32,
            layout.kpe_stride_page as i64,
            layout.kpe_stride_n as i64,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Suffix-aware single prefill: q covers `seq_len` suffix tokens, k/v cover
/// `kv_len = cached + seq_len` tokens. FlashInfer's causal mask aligns
/// bottom-right, which is exactly the absolute-position causal mask for a
/// suffix starting at `kv_len - seq_len`.
pub fn kimi_flashinfer_single_prefill_mla_rt(
    ctx: &DeviceContext,
    q_attn: &HiddenStates,
    k_cache: &CudaSlice<half::bf16>,
    v_cache: &CudaSlice<half::bf16>,
    output: &mut HiddenStates,
    sm_scale: f32,
    kv_len: usize,
    local_heads: usize,
) -> Result<()> {
    let seq_len = q_attn.seq_len;
    let (q_ptr, _g0) = q_attn.data.device_ptr(&ctx.stream);
    let (k_ptr, _g1) = k_cache.device_ptr(&ctx.stream);
    let (v_ptr, _g2) = v_cache.device_ptr(&ctx.stream);
    let (out_ptr, _g3) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_flashinfer_single_prefill_mla_cuda(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            local_heads as i32,
            seq_len as i32,
            kv_len as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!(
            "kimi_flashinfer_single_prefill_mla_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}
