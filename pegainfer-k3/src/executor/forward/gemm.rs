//! The dense-projection GEMM the decode step lands from.
//!
//! Every dense projection in the certified step is a matmul followed by a
//! separate bf16 landing (`k3_land_batched` with `split_k = 1`), so the GEMM
//! must deliver an **f32 partial** in the landing kernel's `[rows, nt]`
//! row-major layout. That is exactly `ops::gemm_bf16_f32` — bf16 operands,
//! f32 output, the workspace-free cuBLAS handle, CUDA-graph safe.
//!
//! This module adds the two offsets that wrapper does not expose and the step
//! genuinely needs, using the same FFI entry point and pointer arithmetic
//! `ops::gemm_rows_into_checked` already applies to its bf16 twin:
//!
//! * a **weight row band**, so one fused checkpoint tensor can feed several
//!   independent partials. The KDA input projection is the reason: the output
//!   gate is landed out of the fused `[4 * inner, hidden]` product with the
//!   certified `(4*inner, inner, 3*inner)` span, while the q/k/v streams must
//!   reach `k3_conv_silu_batched` as *contiguous* `[rows, inner]` partials.
//!   Four banded GEMMs serve both layouts and multiply nothing twice.
//! * a **column offset and stride in the destination**, so a banded product
//!   can be placed inside a wider partial when the landing span demands it.
//!
//! Nothing else about the call changes: `alpha = 1`, `beta = 0`, `OP_T` on the
//! checkpoint's `[out, in]` weight, f32 accumulation.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;
use pegainfer_kernels::ffi;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::active_cu_stream;

/// Where a banded product lands inside its f32 partial.
#[derive(Clone, Copy, Debug)]
pub(crate) struct K3PartialSpan {
    /// Column of the partial the band's first output feature lands on.
    pub(crate) offset: usize,
    /// Row stride of the partial, i.e. the landing kernel's `nt`.
    pub(crate) stride: usize,
}

impl K3PartialSpan {
    /// The partial is exactly this product and nothing else.
    pub(crate) fn whole(width: usize) -> Self {
        Self {
            offset: 0,
            stride: width,
        }
    }
}

/// `partial[row, span.offset + j] = sum_k x[row, k] * weight[weight_row + j, k]`
/// in f32, for `rows` rows and `out_features` output features.
///
/// `weight` keeps the checkpoint's `[out, in]` row-major orientation; the GEMM
/// takes it with `OP_T`. `x` is `[rows, weight.cols]` row-major and dense.
pub(crate) fn k3_gemm_partial(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    weight_row: usize,
    out_features: usize,
    x: &CudaSlice<bf16>,
    rows: usize,
    partial: &mut CudaSlice<f32>,
    span: K3PartialSpan,
) -> Result<()> {
    let k = weight.cols;
    ensure!(
        weight_row + out_features <= weight.rows,
        "K3 GEMM band [{weight_row}, {}) exceeds the weight's {} rows",
        weight_row + out_features,
        weight.rows
    );
    ensure!(
        span.offset + out_features <= span.stride,
        "K3 GEMM span [{}, {}) does not fit the partial width {}",
        span.offset,
        span.offset + out_features,
        span.stride
    );
    ensure!(
        x.len() >= rows * k && partial.len() >= rows * span.stride,
        "K3 GEMM buffers too small for rows={rows}, k={k}, nt={}: x {}, partial {}",
        span.stride,
        x.len(),
        partial.len()
    );

    let (weight_ptr, _weight_guard) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _x_guard) = x.device_ptr(&ctx.stream);
    let (partial_ptr, _partial_guard) = partial.device_ptr_mut(&ctx.stream);
    let a = weight_ptr + (weight_row * k * size_of::<bf16>()) as u64;
    let c = partial_ptr + (span.offset * size_of::<f32>()) as u64;

    // Column-major BLAS on row-major buffers: A is the weight seen as
    // `[k, out_features]` with OP_T, B is `x` seen as `[k, rows]`, and the
    // `[out_features, rows]` column-major result with leading dimension
    // `span.stride` *is* the row-major `[rows, span.stride]` partial.
    let status = unsafe {
        ffi::gemm_bf16_f32_cuda(
            1,
            0,
            i32::try_from(out_features)?,
            i32::try_from(rows)?,
            i32::try_from(k)?,
            a as *const ffi::Half,
            i32::try_from(k)?,
            x_ptr as *const ffi::Half,
            i32::try_from(k)?,
            c as *mut f32,
            i32::try_from(span.stride)?,
            active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "K3 dense GEMM ({out_features}x{k}, rows={rows}) failed: {}",
        if status >= 100_000 {
            format!("cublasStatus={}", status - 100_000)
        } else {
            format!("cudaError={status}")
        }
    );
    Ok(())
}

/// The common case: the whole weight, into a partial that holds only it.
pub(crate) fn k3_gemm_full(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &CudaSlice<bf16>,
    rows: usize,
    partial: &mut CudaSlice<f32>,
) -> Result<()> {
    let span = K3PartialSpan::whole(weight.rows);
    k3_gemm_partial(ctx, weight, 0, weight.rows, x, rows, partial, span)
}
