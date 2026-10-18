//! Kimi-K3 MoE router: sigmoid scores plus biased top-k selection.
//!
//! Hand-written replacement for the retired TileLang `router_topk_batched`
//! kernel, whose serial thread-0 scan cost ~65us per launch at 896 experts.
//! The selection is a block-parallel argmax per round with the serial
//! kernel's exact arithmetic and lowest-index tie-break, so the outputs are
//! bit-identical — see `csrc/k3/k3_router_topk.cu` for the argument. Batch is
//! a plain launch dimension (no per-bucket instantiation), but callers still
//! run the compiled buckets: every other kernel in the step is bucket-shaped.

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

/// Sigmoid router plus biased top-k over already-merged f32 score rows.
///
/// The weights come from the *un-biased* scores, are normalized with a
/// `+1e-20` guard and scaled by the bf16 routed scale `rs`. Ties break to the
/// lowest expert index.
#[allow(clippy::too_many_arguments)]
pub fn k3_router_topk_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    num_experts: usize,
    topk: usize,
    s: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    rs: &CudaSlice<bf16>,
    idx: &mut CudaSlice<i32>,
    wts: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(b > 0, "K3 router needs rows");
    ensure!(
        topk <= num_experts,
        "K3 router topk={topk} exceeds the expert count {num_experts}"
    );
    ensure!(
        s.len() >= b * num_experts
            && bias.len() >= num_experts
            && !rs.is_empty()
            && idx.len() >= b * topk
            && wts.len() >= b * topk,
        "K3 router buffers too small for b={b}, experts={num_experts}, topk={topk}: \
         s {}, bias {}, rs {}, idx {}, wts {}",
        s.len(),
        bias.len(),
        rs.len(),
        idx.len(),
        wts.len()
    );
    let (s_ptr, _s_guard) = s.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (rs_ptr, _rs_guard) = rs.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = idx.device_ptr_mut(&ctx.stream);
    let (wts_ptr, _wts_guard) = wts.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::k3_router_topk_cuda(
            s_ptr as *const f32,
            bias_ptr as *const f32,
            rs_ptr as *const c_void,
            idx_ptr as *mut i32,
            wts_ptr as *mut f32,
            i32::try_from(b)?,
            i32::try_from(num_experts)?,
            i32::try_from(topk)?,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| {
        anyhow!("K3 router_topk (B={b}, E={num_experts}, TOPK={topk}) launch failed: {err}")
    })
}
