//! Kimi-K3 matmul landing: merge one column span of a `[split_k, nt]` f32
//! partial and land bf16 once.
//!
//! Hand-written replacement for the retired TileLang `land_batched` kernel
//! (one element per thread, 2-byte stores, 2.3 TB/s on the chunked-prefill
//! landings). Eight columns per thread with 16-byte loads and stores, the
//! retired kernel's exact arithmetic — ascending-segment f32 sum, one
//! round-to-nearest-even cast — so the landing is bit-identical; see
//! `csrc/k3/k3_land.cu`. Batch is a plain launch dimension.

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

/// Merge the column span `[off, off + n)` of each row's `[split_k, nt]`
/// partials and land bf16 once — the landing of every matmul in the certified
/// spelling.
#[allow(clippy::too_many_arguments)]
pub fn k3_land_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    nt: usize,
    n: usize,
    off: usize,
    split_k: usize,
    p: &CudaSlice<f32>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        b > 0 && split_k > 0,
        "K3 land needs rows and at least one segment"
    );
    ensure!(
        off + n <= nt,
        "K3 land span [{off}, {off}+{n}) does not fit the partial width {nt}"
    );
    ensure!(
        p.len() >= b * split_k * nt && o.len() >= b * n,
        "K3 land buffers too small for b={b}, nt={nt}, n={n}, split_k={split_k}: p {}, o {}",
        p.len(),
        o.len()
    );
    let (p_ptr, _p_guard) = p.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::k3_land_cuda(
            p_ptr as *const f32,
            o_ptr as *mut c_void,
            i32::try_from(b)?,
            i32::try_from(nt)?,
            i32::try_from(n)?,
            i32::try_from(off)?,
            i32::try_from(split_k)?,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| {
        anyhow!("K3 land (B={b}, NT={nt}, N={n}, OFF={off}, SK={split_k}) launch failed: {err}")
    })
}
