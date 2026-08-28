//! Capsule-vendored vLLM fused KDA decode step
//! (`cubin/k3/kda_decode_fusion_h96_sm103.cubin`).
//!
//! One launch replaces the native conv_silu x3 + kda_core chain: short
//! convolution (window update in place), silu, joint q/k L2 norm, lower-bound
//! decay, delta rule (recurrent state update in place), gated output RMS
//! norm. The formulas are the native kernels' exact spellings — same
//! `GATE_LOWER_BOUND=-5`, `scale=head_dim^-0.5`, `RMS_EPS=1e-5`, tap order
//! and `[head, v_dim, k_dim]` state layout — differing only in rounding
//! chains (the native TileLang path lands several intermediates in bf16, the
//! vLLM kernel keeps them f32).
//!
//! Layout contract (compiled into the cubin, tier `heads=96, head_dim=128`):
//!
//! * `x` — packed pre-conv rows `[b, 3 * 12288]` bf16, `q|k|v` bands.
//! * `conv` — packed window slab `[rows, 3 taps, q|k|v, 12288]` bf16; the
//!   tap stride `3 * 12288` is compile-time. Updated in place (shift + newest
//!   row), unlike the native parity-pair windows.
//! * `state` — `[rows, 96, 128, 128]` f32, updated in place.

use core::ffi::c_void;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::K3_CONV_WIDTH;
use crate::ops::K3_KDA_DIM;
use crate::ops::K3_KDA_HEAD_DIM;
use crate::ops::K3_KDA_HEADS;
use crate::tensor::DeviceContext;

/// Row stride of the packed pre-conv `q|k|v` input, elements.
pub const K3_CAPSULE_X_ROW: usize = 3 * K3_KDA_DIM;
/// Per-slot elements of the packed conv window slab: `taps x (q|k|v) x dim`,
/// with the tap stride `3 * K3_KDA_DIM` compiled into the cubin.
pub const K3_CAPSULE_CONV_SLOT: usize = (K3_CONV_WIDTH - 1) * 3 * K3_KDA_DIM;
/// Per-slot elements of the recurrent state.
pub const K3_CAPSULE_STATE_SLOT: usize = K3_KDA_HEADS * K3_KDA_HEAD_DIM * K3_KDA_HEAD_DIM;

/// The native kernels' compiled-in constants, restated for the capsule launch
/// (`generate.py`: `GATE_LOWER_BOUND`, `RMS_EPS`).
const LOWER_BOUND: f32 = -5.0;
const ONORM_EPS: f32 = 1e-5;

/// One fused KDA decode step over `b` rows, states updated in place.
#[allow(clippy::too_many_arguments)]
pub fn k3_capsule_kda_decode_launch(
    ctx: &DeviceContext,
    b: usize,
    x: &CudaSlice<bf16>,
    cw_q: &CudaSlice<f32>,
    cw_k: &CudaSlice<f32>,
    cw_v: &CudaSlice<f32>,
    conv: &mut CudaSlice<bf16>,
    a_log: &CudaSlice<f32>,
    g: &CudaSlice<bf16>,
    dt_bias: &CudaSlice<f32>,
    beta: &CudaSlice<bf16>,
    onorm_g: &CudaSlice<bf16>,
    onorm_weight: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(b > 0, "K3 capsule KDA needs rows");
    let taps = K3_CONV_WIDTH * K3_KDA_DIM;
    ensure!(
        x.len() >= b * K3_CAPSULE_X_ROW
            && cw_q.len() >= taps
            && cw_k.len() >= taps
            && cw_v.len() >= taps
            && conv.len() >= b * K3_CAPSULE_CONV_SLOT
            && a_log.len() >= K3_KDA_HEADS
            && g.len() >= b * K3_KDA_DIM
            && dt_bias.len() >= K3_KDA_DIM
            && beta.len() >= b * K3_KDA_HEADS
            && onorm_g.len() >= b * K3_KDA_DIM
            && onorm_weight.len() >= K3_KDA_HEAD_DIM
            && state.len() >= b * K3_CAPSULE_STATE_SLOT
            && out.len() >= b * K3_KDA_DIM,
        "K3 capsule KDA buffers too small for b={b}: x {}, conv {}, state {}, out {}",
        x.len(),
        conv.len(),
        state.len(),
        out.len()
    );
    let (x_ptr, _x_guard) = x.device_ptr(&ctx.stream);
    let (cw_q_ptr, _cwq_guard) = cw_q.device_ptr(&ctx.stream);
    let (cw_k_ptr, _cwk_guard) = cw_k.device_ptr(&ctx.stream);
    let (cw_v_ptr, _cwv_guard) = cw_v.device_ptr(&ctx.stream);
    let (conv_ptr, _conv_guard) = conv.device_ptr_mut(&ctx.stream);
    let (a_log_ptr, _a_guard) = a_log.device_ptr(&ctx.stream);
    let (g_ptr, _g_guard) = g.device_ptr(&ctx.stream);
    let (dt_ptr, _dt_guard) = dt_bias.device_ptr(&ctx.stream);
    let (beta_ptr, _bt_guard) = beta.device_ptr(&ctx.stream);
    let (og_ptr, _og_guard) = onorm_g.device_ptr(&ctx.stream);
    let (ow_ptr, _ow_guard) = onorm_weight.device_ptr(&ctx.stream);
    let (state_ptr, _st_guard) = state.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);

    let band = (K3_KDA_DIM * size_of::<bf16>()) as u64;
    let scale = (K3_KDA_HEAD_DIM as f32).powf(-0.5);
    unsafe {
        ffi::k3_capsule_kda_decode_cuda(
            x_ptr as *const c_void,
            (x_ptr + band) as *const c_void,
            (x_ptr + 2 * band) as *const c_void,
            cw_q_ptr as *const f32,
            cw_k_ptr as *const f32,
            cw_v_ptr as *const f32,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            conv_ptr as *mut c_void,
            (conv_ptr + band) as *mut c_void,
            (conv_ptr + 2 * band) as *mut c_void,
            a_log_ptr as *const f32,
            g_ptr as *const c_void,
            dt_ptr as *const f32,
            beta_ptr as *const c_void,
            og_ptr as *const c_void,
            ow_ptr as *const f32,
            core::ptr::null(),
            core::ptr::null(),
            state_ptr as *mut f32,
            out_ptr as *mut c_void,
            i32::try_from(b)?,
            i32::try_from(K3_KDA_HEADS)?,
            i32::try_from(K3_KDA_HEADS)?,
            LOWER_BOUND,
            scale,
            ONORM_EPS,
            i64::try_from(K3_CAPSULE_X_ROW)?,
            i64::try_from(K3_KDA_HEADS)?,
            i64::try_from(K3_KDA_DIM)?,
            i64::try_from(K3_CAPSULE_CONV_SLOT)?,
            i64::try_from(K3_CAPSULE_STATE_SLOT)?,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("K3 capsule KDA decode (B={b}) launch failed: {err}"))
}
