//! Kimi-K3 chunked-prefill causal conv + silu, walking a segment's rows in
//! place.
//!
//! Replaces the chunk path's landing + strided window copies + batched decode
//! conv + carry copy with one launch per segment that reads the f32 partial
//! rows `t-3..t` directly and takes the first rows' missing inputs from the
//! carried window. Term-for-term the batched conv kernel's arithmetic, so
//! bit-identical — see `csrc/k3/k3_conv_silu_chunk.cu`.

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

/// Window slots carried between segments (`K3_CONV_WIDTH - 1`).
const STATE: usize = 3;

/// Conv + silu over `tokens` consecutive rows of one q/k/v stream.
///
/// `partial` rows `partial_row..partial_row + tokens` (each `inner` f32) are
/// the segment's projection; `taps` is `[4, inner]` f32; `carry` row
/// `carry_row` (each `STATE * inner` bf16) is the window preceding the
/// segment. `out` rows `out_row..` receive the bf16 conv output. `next`, when
/// given, names the `[STATE, inner]` row that receives the window carrying
/// into the segment after row `commit_rows - 1`; it must be given exactly
/// when `commit_rows > 0`.
#[allow(clippy::too_many_arguments)]
pub fn k3_conv_silu_chunk_launch(
    ctx: &DeviceContext,
    inner: usize,
    tokens: usize,
    commit_rows: usize,
    partial: &CudaSlice<f32>,
    partial_row: usize,
    taps: &CudaSlice<f32>,
    carry: &CudaSlice<bf16>,
    carry_row: usize,
    out: &mut CudaSlice<bf16>,
    out_row: usize,
    next: Option<(&mut CudaSlice<bf16>, usize)>,
) -> Result<()> {
    ensure!(tokens > 0, "K3 conv chunk needs rows");
    ensure!(
        commit_rows <= tokens,
        "K3 conv chunk commits {commit_rows} of {tokens} rows"
    );
    ensure!(
        inner % 8 == 0,
        "K3 conv chunk needs an 8-column-aligned width, got {inner}"
    );
    ensure!(
        (commit_rows > 0) == next.is_some(),
        "K3 conv chunk carries a window exactly when it commits rows"
    );
    ensure!(
        partial.len() >= (partial_row + tokens) * inner
            && taps.len() >= 4 * inner
            && carry.len() >= (carry_row + 1) * STATE * inner
            && out.len() >= (out_row + tokens) * inner,
        "K3 conv chunk buffers too small for tokens={tokens}, inner={inner}: \
         partial {} (row {partial_row}), taps {}, carry {} (row {carry_row}), out {} (row {out_row})",
        partial.len(),
        taps.len(),
        carry.len(),
        out.len()
    );
    let (p_ptr, _p_guard) = partial.device_ptr(&ctx.stream);
    let (cw_ptr, _cw_guard) = taps.device_ptr(&ctx.stream);
    let (carry_ptr, _carry_guard) = carry.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let mut next_guard = None;
    let next_ptr = match next {
        Some((slab, row)) => {
            ensure!(
                slab.len() >= (row + 1) * STATE * inner,
                "K3 conv chunk carry-out slab too small for row {row}: {}",
                slab.len()
            );
            let (ptr, guard) = slab.device_ptr_mut(&ctx.stream);
            next_guard = Some(guard);
            ptr as usize + row * STATE * inner * size_of::<bf16>()
        }
        None => 0,
    };
    let rc = unsafe {
        ffi::k3_conv_silu_chunk_cuda(
            (p_ptr as usize + partial_row * inner * size_of::<f32>()) as *const f32,
            cw_ptr as *const f32,
            (carry_ptr as usize + carry_row * STATE * inner * size_of::<bf16>()) as *const c_void,
            (out_ptr as usize + out_row * inner * size_of::<bf16>()) as *mut c_void,
            next_ptr as *mut c_void,
            i32::try_from(tokens)?,
            i32::try_from(commit_rows)?,
            i32::try_from(inner)?,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    drop(next_guard);
    rc.result().map_err(|err| {
        anyhow!("K3 conv_silu_chunk (tokens={tokens}, commit={commit_rows}, inner={inner}) launch failed: {err}")
    })
}
