//! Gated DeltaNet recurrent decode: the causal depthwise conv1d and the rank-1
//! state update it feeds.
//!
//! Sources: `csrc/qwen35/{gated_delta_rule,conv1d}.cu`, compiled under the
//! kernels crate's `qwen35` feature — the same gate this module rides. The
//! Qwen3.5 line calls them from its batched decode, and the Qwen3.8-Flash-Next
//! (`pegainfer-qwen4`) line calls them from its reuse test today and its executor
//! next, because the two lines' GDN geometry is identical: 16 key heads and 48
//! value heads at head dim 128, and a 4-tap convolution. That is why there is one
//! wrapper here rather than one per line.
//!
//! Head counts and the conv kernel size are runtime arguments. The two head dims
//! are not: `gated_delta_rule.cu` tiles on the `GDR_KEY_DIM`/`GDR_VAL_DIM`
//! constants, and the chunk-wise prefill's Triton AOT keys are specialized on the
//! same numbers, so the wrappers assert them instead of passing them through.

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

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
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_decode_batch_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
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
    assert_eq!(qkv.seq_len, batch_size);
    assert_eq!(b_proj.seq_len, batch_size);
    assert_eq!(a_proj.seq_len, batch_size);
    assert_eq!(output.seq_len, batch_size);
    assert_eq!(b_proj.hidden_dim, num_value_heads);
    assert_eq!(a_proj.hidden_dim, num_value_heads);
    assert_eq!(output.hidden_dim, num_value_heads * val_dim);
    assert_eq!(key_dim, GDN_AOT_KEY_HEAD_DIM);
    assert_eq!(val_dim, GDN_AOT_VALUE_HEAD_DIM);
    assert!(state_ptrs.len() >= batch_size);

    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (state_ptrs, _gsp) = state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        pegainfer_kernels::ffi::gated_delta_rule_decode_batch_cuda(
            qkv_ptr as *const pegainfer_kernels::ffi::Half,
            b_ptr as *const pegainfer_kernels::ffi::Half,
            a_ptr as *const pegainfer_kernels::ffi::Half,
            dt_ptr as *const pegainfer_kernels::ffi::Half,
            alog_ptr as *const f32,
            state_ptrs as *const u64,
            o_ptr as *mut pegainfer_kernels::ffi::Half,
            batch_size as i32,
            num_key_heads as i32,
            num_value_heads as i32,
            key_dim as i32,
            val_dim as i32,
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
pub fn conv1d_decode_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state_ptrs: &CudaSlice<u64>,
    out: &mut HiddenStates,
    kernel_size: usize,
) {
    let batch_size = x.seq_len;
    let num_channels = x.hidden_dim;
    assert_eq!(out.hidden_dim, num_channels);
    assert_eq!(out.seq_len, batch_size);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert!(kernel_size <= LINEAR_CONV_MAX_KERNEL_DIM);
    assert!(conv_state_ptrs.len() >= batch_size);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptrs, _gs) = conv_state_ptrs.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        pegainfer_kernels::ffi::conv1d_decode_batch_cuda(
            x_ptr as *const pegainfer_kernels::ffi::Half,
            w_ptr as *const pegainfer_kernels::ffi::Half,
            s_ptrs as *const u64,
            o_ptr as *mut pegainfer_kernels::ffi::Half,
            num_channels as i32,
            batch_size as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}
