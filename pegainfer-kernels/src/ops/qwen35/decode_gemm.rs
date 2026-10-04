use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;

/// A model-owned decode recipe with FP32 split-K partials/reduction.
/// Reuses the owning decode thread's descriptors and workspace; no GPU buffer
/// or change to the shared algorithm cache is associated with this object.
pub struct Qwen35DecodeGemm {
    algorithm: [u64; 8],
    shape: [i32; 3],
    device_ordinal: usize,
}

impl Qwen35DecodeGemm {
    /// Prepare on the model's decode thread after ordinary GEMM tuning, before
    /// CUDA Graph capture. An unsupported higher-precision recipe is an error.
    pub fn prepare(ctx: &DeviceContext, weights: &DeviceMatrix, batch: usize) -> Result<Self> {
        ensure!(
            ctx.stream.context() == &ctx.ctx && weights.data.context() == &ctx.ctx,
            "Qwen3.5 decode GEMM weight/stream context mismatch"
        );
        let shape = [
            i32::try_from(weights.rows)?,
            i32::try_from(batch)?,
            i32::try_from(weights.cols)?,
        ];
        crate::ops::linear::ensure_tuned_policy(weights.rows, batch, weights.cols)?;
        ensure!(
            !crate::tensor::has_stream_override(),
            "Qwen3.5 decode GEMM requires the base decode stream"
        );
        ensure!(
            shape.iter().all(|&dim| dim > 0),
            "empty Qwen3.5 decode GEMM shape"
        );
        ensure!(
            weights.data.len() >= weights.rows * weights.cols,
            "Qwen3.5 decode GEMM weight allocation is too small"
        );
        let mut algorithm = [0; 8];
        let (weight_ptr, _weights) = weights.data.device_ptr(&ctx.stream);
        let status = unsafe {
            ffi::pegainfer_qwen35_decode_gemm_prepare(
                weight_ptr as *const ffi::Half,
                shape[0],
                shape[1],
                shape[2],
                algorithm.as_mut_ptr(),
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            status == 0,
            "Qwen3.5 decode GEMM FP32 reduction preparation failed: status={status}"
        );
        Ok(Self {
            algorithm,
            shape,
            device_ordinal: ctx.ctx.ordinal(),
        })
    }

    pub fn launch(
        &self,
        ctx: &DeviceContext,
        weights: &DeviceMatrix,
        input: &HiddenStates,
        output: &mut HiddenStates,
    ) -> Result<()> {
        let [rows, batch, cols] = self.shape.map(|dim| dim as usize);
        crate::ops::linear::ensure_tuned_policy(rows, batch, cols)?;
        ensure!(
            !crate::tensor::has_stream_override(),
            "Qwen3.5 decode GEMM requires the base decode stream"
        );
        ensure!(
            ctx.ctx.ordinal() == self.device_ordinal
                && [
                    &ctx.ctx,
                    weights.data.context(),
                    input.data.context(),
                    output.data.context()
                ]
                .into_iter()
                .all(|context| context == ctx.stream.context()),
            "Qwen3.5 decode GEMM buffer/stream context mismatch"
        );
        ensure!(
            weights.rows == rows
                && weights.cols == cols
                && input.seq_len == batch
                && input.hidden_dim == cols
                && output.seq_len == batch
                && output.hidden_dim == rows
                && weights.data.len() >= rows * cols
                && input.data.len() >= batch * cols
                && output.data.len() >= batch * rows,
            "Qwen3.5 decode GEMM tensors do not match the prepared shape"
        );
        let (weight_ptr, _weights) = weights.data.device_ptr(&ctx.stream);
        let (input_ptr, _input) = input.data.device_ptr(&ctx.stream);
        let (output_ptr, _output) = output.data.device_ptr_mut(&ctx.stream);
        let status = unsafe {
            ffi::pegainfer_qwen35_decode_gemm_launch(
                self.algorithm.as_ptr(),
                weight_ptr as *const ffi::Half,
                input_ptr as *const ffi::Half,
                output_ptr as *mut ffi::Half,
                self.shape[0],
                self.shape[1],
                self.shape[2],
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            status == 0,
            "Qwen3.5 decode GEMM launch failed: status={status}"
        );
        Ok(())
    }
}
