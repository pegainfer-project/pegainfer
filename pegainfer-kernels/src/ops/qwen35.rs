//! Qwen3.5 wrappers: the HD256 full-attention decode prep and paged attention
//! reads, and the Gated DeltaNet decode step over
//! `csrc/qwen35/{gated_delta_rule,conv1d}.cu`.

use anyhow::Context as _;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
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
