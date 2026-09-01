//! Kimi-K3 bf16 elementwise family: the residual add, the MLA sigmoid
//! output gate, the situ activation, and the KDA o_norm x output gate.
//!
//! Hand-written replacements for the retired TileLang batched kernels (one
//! element per thread, 2-byte accesses, 1–2 TB/s on the chunked-prefill
//! rows). Eight columns per thread with 16-byte loads and stores and the
//! retired kernels' exact arithmetic — cutlass's bf16 `+`/`*` are `__hadd` /
//! `__hmul`, the casts `cvt.rn`, and `o_norm_gate`'s 128-wide xor butterfly
//! is reproduced pair for pair — so every landing is bit-identical; see
//! `csrc/k3/k3_elementwise.cu`. Batch is a plain launch dimension.

use core::ffi::c_void;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

type BinaryLauncher = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    cudarc::driver::sys::CUstream,
) -> cudarc::driver::sys::CUresult;

fn binary_launch(
    ctx: &DeviceContext,
    launcher: BinaryLauncher,
    what: &str,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(b > 0 && n > 0, "K3 {what} needs rows and columns");
    ensure!(
        a.len() >= b * n && bt.len() >= b * n && o.len() >= b * n,
        "K3 {what} buffers too small for b={b}, n={n}: a {}, bt {}, o {}",
        a.len(),
        bt.len(),
        o.len()
    );
    let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
    let (bt_ptr, _bt_guard) = bt.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    unsafe {
        launcher(
            a_ptr as *const c_void,
            bt_ptr as *const c_void,
            o_ptr as *mut c_void,
            i32::try_from(b)?,
            i32::try_from(n)?,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("K3 {what} (B={b}, N={n}) launch failed: {err}"))
}

/// `o = a + bt`, added in bf16 — the residual adds, and routed + shared.
pub fn k3_add2_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    binary_launch(ctx, ffi::k3_add2_cuda, "add2", b, n, a, bt, o)
}

/// `o = a * bf16(sigmoid(bt))`, the MLA sigmoid output gate. The sigmoid is
/// taken in f32 and lands in bf16 before the product.
pub fn k3_mul_sigmoid_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    binary_launch(ctx, ffi::k3_mul_sigmoid_cuda, "mul_sigmoid", b, n, a, bt, o)
}

/// K3's situ activation: `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)` in f32,
/// landing bf16 once. The two betas are compiled in.
pub fn k3_situ_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    g: &CudaSlice<bf16>,
    u: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    binary_launch(ctx, ffi::k3_situ_cuda, "situ", b, n, g, u, o)
}

/// `kda_core`'s tail on its own: per (row, head) the f32 rms_norm of the bf16
/// attention landing `x` times the o_norm gamma `go [head_dim]`, landed once,
/// times the bf16 sigmoid of the output-gate projection `g2`. `head_dim` must
/// be 128.
#[allow(clippy::too_many_arguments)]
pub fn k3_o_norm_gate_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    x: &CudaSlice<bf16>,
    g2: &CudaSlice<bf16>,
    go: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        b > 0 && num_heads > 0,
        "K3 o_norm_gate needs rows and heads"
    );
    ensure!(
        head_dim == 128,
        "K3 o_norm_gate serves head_dim 128, got {head_dim}"
    );
    let kp = num_heads * head_dim;
    ensure!(
        x.len() >= b * kp && g2.len() >= b * kp && out.len() >= b * kp && go.len() >= head_dim,
        "K3 o_norm_gate buffers too small for b={b}, kp={kp}: x {}, g2 {}, go {}, out {}",
        x.len(),
        g2.len(),
        go.len(),
        out.len()
    );
    let (x_ptr, _x_guard) = x.device_ptr(&ctx.stream);
    let (g2_ptr, _g2_guard) = g2.device_ptr(&ctx.stream);
    let (go_ptr, _go_guard) = go.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::k3_o_norm_gate_cuda(
            x_ptr as *const c_void,
            g2_ptr as *const c_void,
            go_ptr as *const f32,
            out_ptr as *mut c_void,
            i32::try_from(b)?,
            i32::try_from(num_heads)?,
            i32::try_from(head_dim)?,
            eps,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| {
        anyhow!("K3 o_norm_gate (B={b}, KH={num_heads}, KD={head_dim}) launch failed: {err}")
    })
}
