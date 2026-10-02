//! Block-local dynamic grouped convolution for the native DFlash2 backbone.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

/// Apply one side of the conv pair, using coefficients produced by the shared
/// GEMM. `dynamic` is `[rows, 2 * taps * groups]`; `base` is `[2, taps, hidden]`.
/// Pre/post sides reuse the same coefficients computed before the sublayer.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_grouped_conv_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    dynamic: &HiddenStates,
    base: &CudaSlice<bf16>,
    block_size: usize,
    group_size: usize,
    side: usize,
    output: &mut HiddenStates,
) -> Result<()> {
    input.checked_extent("DFlash2 conv input")?;
    dynamic.checked_extent("DFlash2 conv coefficients")?;
    output.checked_extent("DFlash2 conv output")?;
    ensure!(
        block_size > 0 && input.seq_len > 0 && input.seq_len.is_multiple_of(block_size),
        "DFlash2 conv input must contain complete nonempty draft blocks"
    );
    ensure!(
        group_size > 0 && input.hidden_dim > 0 && input.hidden_dim.is_multiple_of(group_size),
        "DFlash2 conv group size must divide hidden width"
    );
    let groups = input.hidden_dim / group_size;
    ensure!(
        dynamic.seq_len == input.seq_len
            && dynamic.hidden_dim > 0
            && dynamic.hidden_dim.is_multiple_of(2 * groups),
        "DFlash2 conv coefficient shape must be [rows, 2 * taps * groups]"
    );
    let taps = dynamic.hidden_dim / (2 * groups);
    let base_elements = dynamic
        .hidden_dim
        .checked_mul(group_size)
        .context("DFlash2 conv base extent overflow")?;
    ensure!(
        base.len() == base_elements && side < 2,
        "DFlash2 conv base must have shape [2, taps, hidden], with side 0 or 1"
    );
    ensure!(
        output.seq_len == input.seq_len && output.hidden_dim == input.hidden_dim,
        "DFlash2 conv output shape must match input"
    );

    let (input_ptr, _input) = input.data.device_ptr(&ctx.stream);
    let (dynamic_ptr, _dynamic) = dynamic.data.device_ptr(&ctx.stream);
    let (base_ptr, _base) = base.device_ptr(&ctx.stream);
    let (output_ptr, _output) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dflash2_grouped_conv_cuda(
            input_ptr as *const ffi::Half,
            dynamic_ptr as *const ffi::Half,
            base_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            i32::try_from(input.seq_len)?,
            i32::try_from(input.hidden_dim)?,
            i32::try_from(block_size)?,
            i32::try_from(group_size)?,
            i32::try_from(taps)?,
            side as i32,
            active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}
