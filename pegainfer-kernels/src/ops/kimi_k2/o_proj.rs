use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use super::experts::KIMI_K2_HIDDEN;
use super::mla::KIMI_K2_MLA_V_HEAD_DIM;
use crate::ffi;
use crate::ops::ffi_status_detail;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::GpuTensor;
use crate::tensor::HiddenStates;

pub const KIMI_O_PROJ_CUBLASLT_INPUT: usize = 64 * KIMI_K2_MLA_V_HEAD_DIM;
const KIMI_O_PROJ_CUBLASLT_MAX_BATCH: usize = 64;

pub fn kimi_o_proj_cublaslt_supports_batch_size(batch_size: usize) -> bool {
    (1..=KIMI_O_PROJ_CUBLASLT_MAX_BATCH).contains(&batch_size)
}

pub fn kimi_o_proj_cublaslt_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let batch_size = x.seq_len;
    ensure!(
        weight.rows == KIMI_K2_HIDDEN && weight.cols == KIMI_O_PROJ_CUBLASLT_INPUT,
        "Kimi o_proj cuBLASLt weight shape mismatch: got [{},{}], expected [{},{}]",
        weight.rows,
        weight.cols,
        KIMI_K2_HIDDEN,
        KIMI_O_PROJ_CUBLASLT_INPUT
    );
    ensure!(
        x.hidden_dim == KIMI_O_PROJ_CUBLASLT_INPUT,
        "Kimi o_proj cuBLASLt input hidden mismatch: got {}, expected {}",
        x.hidden_dim,
        KIMI_O_PROJ_CUBLASLT_INPUT
    );
    ensure!(
        out.seq_len == batch_size,
        "Kimi o_proj cuBLASLt output batch mismatch: out={}, input={}",
        out.seq_len,
        batch_size
    );
    ensure!(
        kimi_o_proj_cublaslt_supports_batch_size(batch_size),
        "Kimi o_proj cuBLASLt supports batch_size 1..={}; got {}",
        KIMI_O_PROJ_CUBLASLT_MAX_BATCH,
        batch_size
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        let status = ffi::kimi_o_proj_cublaslt_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            KIMI_K2_HIDDEN as i32,
            batch_size as i32,
            KIMI_O_PROJ_CUBLASLT_INPUT as i32,
            ctx.stream.cu_stream(),
        );
        if status != 0 {
            bail!(
                "Kimi o_proj cuBLASLt failed: {}, batch_size={batch_size}",
                ffi_status_detail(status)
            );
        }
    }
    Ok(())
}
