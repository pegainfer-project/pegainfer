//! Kimi-K3 chunked-prefill KDA core over the vendored FlashKDA kernel.
//!
//! FlashKDA (`third_party/flash-kda`, MIT, MoonshotAI) is the upstream
//! chunkwise Kimi-Delta-Attention forward: 16-token intra-chunk tiles
//! (kernel 1) and the inter-chunk state recurrence plus output (kernel 2),
//! CUTLASS/CuTe on SM90 TMA. One call advances a whole prefill chunk through
//! one KDA layer — the b=1-per-token walk collapses into two launches.
//!
//! The C ABI shim (`csrc/k3/k3_flash_kda.cu`) pins the one configuration this
//! engine uses: `D = 128`, f32 recurrent state carried in and out, one
//! sequence per call. The gate math is applied in-kernel from the
//! pre-activation projection — `decay = exp(lower_bound * sigmoid(exp(A_log)
//! * (g + dt_bias)))`, the same formula the sequential TileLang core spells —
//! and q/k are L2-normalized in-kernel (f32 chain, where the TileLang core
//! deliberately mirrors the reference's bf16 chain: chunked prefill is a
//! noise-floor path, not a bitwise one).

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ffi::Half;
use crate::tensor::DeviceContext;

/// FlashKDA's head geometry: K = V = 128 is the only compiled width.
pub const K3_FLASH_KDA_HEAD_DIM: usize = 128;

/// Workspace bytes one forward of `t_total` tokens at `heads` heads needs.
pub fn k3_flash_kda_workspace_bytes(t_total: usize, heads: usize) -> usize {
    unsafe { ffi::k3_flash_kda_workspace_bytes(t_total as i32, heads as i32) as usize }
}

/// Row offsets addressing one segment of a larger step through the forward.
///
/// `row` shifts q/k/v/g/beta/out by that many token rows; `state_in_row` /
/// `state_out_row` pick a `[heads, 128, 128]` recurrent row out of a pooled
/// state slab. A verify step runs one call per (slot, segment) over the
/// step-wide projection buffers, so every operand needs its own base.
#[derive(Clone, Copy, Default)]
pub struct K3FlashKdaSpan {
    pub row: usize,
    pub state_in_row: usize,
    pub state_out_row: usize,
}

/// One token segment through one KDA layer's chunkwise forward.
///
/// `q`/`k`/`v`/`g` and `out` are `[t_total, heads * 128]` bf16 rows starting
/// at `span.row` (`g` is the pre-activation gate projection); `beta` is
/// `[t_total, heads]` bf16 logits from the same row, transposed into
/// `beta_scratch` (`[heads, t_total]`) for the kernel's 1D TMA. `a_log
/// [heads]` and `dt_bias [heads * 128]` are f32 weights. `state_in` /
/// `state_out` are read at their span rows — one `[heads, 128, 128]` f32
/// recurrent row each — and the addressed rows may not alias.
#[allow(clippy::too_many_arguments)]
pub fn k3_flash_kda_fwd_launch(
    ctx: &DeviceContext,
    t_total: usize,
    heads: usize,
    span: K3FlashKdaSpan,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    g: &CudaSlice<bf16>,
    beta: &CudaSlice<bf16>,
    beta_scratch: &mut CudaSlice<bf16>,
    a_log: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    state_in: &CudaSlice<f32>,
    state_out: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
    scale: f32,
    lower_bound: f32,
) -> Result<()> {
    let d = K3_FLASH_KDA_HEAD_DIM;
    let width = heads * d;
    let rows = span.row + t_total;
    ensure!(t_total > 0, "K3 FlashKDA got an empty chunk");
    ensure!(
        q.len() >= rows * width
            && k.len() >= rows * width
            && v.len() >= rows * width
            && g.len() >= rows * width
            && out.len() >= rows * width,
        "K3 FlashKDA q/k/v/g/out buffers too small for t={t_total}+{}, heads={heads}",
        span.row
    );
    ensure!(
        beta.len() >= rows * heads && beta_scratch.len() >= t_total * heads,
        "K3 FlashKDA beta buffers too small for t={t_total}+{}, heads={heads}: beta {}, scratch {}",
        span.row,
        beta.len(),
        beta_scratch.len()
    );
    ensure!(
        a_log.len() >= heads && dt_bias.len() >= width,
        "K3 FlashKDA a_log/dt_bias too small for heads={heads}"
    );
    let state = heads * d * d;
    ensure!(
        state_in.len() >= (span.state_in_row + 1) * state
            && state_out.len() >= (span.state_out_row + 1) * state,
        "K3 FlashKDA state rows too small for heads={heads}: in {} @row {}, out {} @row {}",
        state_in.len(),
        span.state_in_row,
        state_out.len(),
        span.state_out_row
    );
    let needed = k3_flash_kda_workspace_bytes(t_total, heads);
    ensure!(
        workspace.len() >= needed,
        "K3 FlashKDA workspace of {} bytes is under the {needed} the chunk needs",
        workspace.len()
    );

    let row_shift = |ptr: u64, stride: usize| ptr + (span.row * stride * size_of::<bf16>()) as u64;
    let (beta_ptr, _beta_guard) = beta.device_ptr(&ctx.stream);
    let beta_ptr = row_shift(beta_ptr, heads);
    let (beta_t_ptr, _beta_t_guard) = beta_scratch.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_flash_kda_beta_transpose(
            beta_ptr as *const Half,
            beta_t_ptr as *mut Half,
            t_total as i32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 FlashKDA beta transpose (T={t_total}, H={heads}): {error}"))?;

    let state_shift = |ptr: u64, row: usize| ptr + (row * state * size_of::<f32>()) as u64;
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let q_ptr = row_shift(q_ptr, width);
    let (k_ptr, _k_guard) = k.device_ptr(&ctx.stream);
    let k_ptr = row_shift(k_ptr, width);
    let (v_ptr, _v_guard) = v.device_ptr(&ctx.stream);
    let v_ptr = row_shift(v_ptr, width);
    let (g_ptr, _g_guard) = g.device_ptr(&ctx.stream);
    let g_ptr = row_shift(g_ptr, width);
    let (a_log_ptr, _a_log_guard) = a_log.device_ptr(&ctx.stream);
    let (dt_bias_ptr, _dt_bias_guard) = dt_bias.device_ptr(&ctx.stream);
    let (state_in_ptr, _state_in_guard) = state_in.device_ptr(&ctx.stream);
    let state_in_ptr = state_shift(state_in_ptr, span.state_in_row);
    let (state_out_ptr, _state_out_guard) = state_out.device_ptr_mut(&ctx.stream);
    let state_out_ptr = state_shift(state_out_ptr, span.state_out_row);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let out_ptr = row_shift(out_ptr, width);
    let (ws_ptr, _ws_guard) = workspace.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_flash_kda_fwd(
            q_ptr as *const Half,
            k_ptr as *const Half,
            v_ptr as *const Half,
            g_ptr as *const Half,
            beta_t_ptr as *const Half,
            a_log_ptr as *const f32,
            dt_bias_ptr as *const f32,
            state_in_ptr as *const f32,
            state_out_ptr as *mut f32,
            out_ptr as *mut Half,
            ws_ptr as *mut core::ffi::c_void,
            t_total as i32,
            heads as i32,
            scale,
            lower_bound,
            ctx.stream.cu_stream(),
        )
    };
    rc.result().map_err(|error| {
        anyhow!("K3 FlashKDA forward (T={t_total}, H={heads}) failed: {error} (NOT_SUPPORTED = built without an accelerated SM90+ target)")
    })
}
