//! GLM5.2 weight-only FP8 GEMV and bf16 SwiGLU helpers used by dense,
//! attention, indexer, and shared-expert projections.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const FP8_BLOCK: usize = 128;

/// Plain `SiLU(gate) * up` over `[rows, 2*inter]` into `[rows, inter]`.
pub fn glm52_silu_and_mul_bf16_launch(
    ctx: &DeviceContext,
    rows: usize,
    inter: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && inter > 0,
        "GLM5.2 SiLU needs positive rows/inter, got {rows}/{inter}"
    );
    ensure!(
        input.len() >= rows * 2 * inter && output.len() >= rows * inter,
        "GLM5.2 SiLU buffers too small: in {}, out {}",
        input.len(),
        output.len()
    );
    let (in_ptr, _i) = input.device_ptr(&ctx.stream);
    let (out_ptr, _o) = output.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_silu_and_mul_bf16_cuda(
            in_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows as i32,
            inter as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 SiLU launch failed: {err}"))
}

/// f32 partial-scratch floats PER ROW the tensor-core batched GEMV path can
/// need: max ksplit (16) × max whitelisted n (dense gate|up 24576). Callers
/// size their buffer as `rows * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW` — one
/// constant instead of mirroring the CUDA-side per-shape config table; the
/// launcher still guards the exact requirement and rejects a short buffer.
/// (The Blackwell table's larger splits only apply to small n — 48 × 2048
/// stays well inside this budget; the CUDA-side guard is the invariant.)
pub const GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW: usize = 16 * 24576;

/// Fused fixed-order k-slice reduce + `SiLU(gate) * up`: consumes the f32
/// partials a [`glm52_fp8_weight_only_gemv_partials_launch`] left in scratch
/// for a packed gate|up projection. Bit-identical to the standalone
/// reduce -> silu pair (both sums round to bf16 before the SiLU math).
pub fn glm52_gemv_reduce_silu_mul_launch(
    ctx: &DeviceContext,
    rows: usize,
    inter: usize,
    ksplit: usize,
    partial: &CudaSlice<f32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && inter > 0 && ksplit >= 1,
        "GLM5.2 reduce-SiLU needs positive rows/inter/ksplit, got {rows}/{inter}/{ksplit}"
    );
    ensure!(
        partial.len() >= ksplit * rows * 2 * inter && output.len() >= rows * inter,
        "GLM5.2 reduce-SiLU buffers too small: partial {} (need {}), out {} (need {})",
        partial.len(),
        ksplit * rows * 2 * inter,
        output.len(),
        rows * inter
    );
    let (p_ptr, _p) = partial.device_ptr(&ctx.stream);
    let (out_ptr, _o) = output.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_gemv_reduce_silu_mul_cuda(
            p_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            rows as i32,
            inter as i32,
            ksplit as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 reduce-SiLU launch failed: {err}"))
}

/// Fixed-order k-slice reduce for a horizontally packed projection pair:
/// de-interleaves the packed `[ksplit, rows, n_a + n_b]` partials into the two
/// compact per-projection bf16 outputs. Same summation order and rounding as
/// the plain reduce.
#[allow(clippy::too_many_arguments)]
pub fn glm52_gemv_split_reduce_launch(
    ctx: &DeviceContext,
    rows: usize,
    n_a: usize,
    n_b: usize,
    ksplit: usize,
    partial: &CudaSlice<f32>,
    out_a: &mut CudaSlice<bf16>,
    out_b: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0 && n_a > 0 && n_b > 0 && ksplit >= 1,
        "GLM5.2 split-reduce needs positive rows/n_a/n_b/ksplit, got {rows}/{n_a}/{n_b}/{ksplit}"
    );
    ensure!(
        partial.len() >= ksplit * rows * (n_a + n_b)
            && out_a.len() >= rows * n_a
            && out_b.len() >= rows * n_b,
        "GLM5.2 split-reduce buffers too small: partial {} (need {}), out_a {} (need {}), out_b {} (need {})",
        partial.len(),
        ksplit * rows * (n_a + n_b),
        out_a.len(),
        rows * n_a,
        out_b.len(),
        rows * n_b
    );
    let (p_ptr, _p) = partial.device_ptr(&ctx.stream);
    let (a_ptr, _a) = out_a.device_ptr_mut(&ctx.stream);
    let (b_ptr, _b) = out_b.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_gemv_split_reduce_cuda(
            p_ptr as *const f32,
            a_ptr as *mut ffi::Half,
            b_ptr as *mut ffi::Half,
            rows as i32,
            n_a as i32,
            n_b as i32,
            ksplit as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 split-reduce launch failed: {err}"))
}

/// Whether the batched GEMV mma table has an entry for (batch, n, k) — the
/// gate for routing a horizontally packed projection through one batched
/// launch. Without an entry the packed launch would take the register tile
/// and write an interleaved layout no consumer understands, so callers fall
/// back to separate per-projection launches.
pub fn glm52_gemv_mma_routes(batch: usize, n: usize, k: usize) -> Result<bool> {
    let mut ksplit: i32 = 0;
    unsafe { ffi::glm52_gemv_mma_ksplit_cuda(batch as i32, n as i32, k as i32, &raw mut ksplit) }
        .result()
        .map_err(|err| anyhow!("GLM5.2 mma ksplit query failed: {err}"))?;
    Ok(ksplit > 0)
}

/// Plain weight-only fp8 GEMV (bs=1): `out[n] = deq(weight[n,k]) @ activation[k]`
/// with the bf16 activation read directly (no activation quant, no scale
/// relayout) and the e4m3 block-scale weight dequanted on the fly. Replaces
/// the TRTLLM CUTLASS block-scale GEMM at m=1, where the M-tile pads 1->64 and
/// runs compute-bound. `scale_bytes` is the checkpoint `weight_scale_inv`
/// (f32 `[ceil(n/128), k/128]`) kept as raw bytes.
///
/// `scratch` is the caller-owned f32 partial buffer for the rows-4/8
/// tensor-core path. Ownership contract: the layer forward overlaps the ctx
/// and aux streams, so a scratch buffer must belong to exactly one launch
/// stream — every scratch struct owns its own. Pass `None` only on rows ≤ 2
/// paths; an mma-routed launch without a buffer fails loudly.
pub fn glm52_fp8_weight_only_gemv_launch(
    ctx: &DeviceContext,
    rows: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    scale_bytes: &CudaSlice<u8>,
    scratch: Option<&mut CudaSlice<f32>>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    gemv_batched_launch(
        ctx,
        rows,
        n,
        k,
        activation,
        weight,
        scale_bytes,
        scratch,
        out,
        None,
    )
}

/// Partials-producing twin of [`glm52_fp8_weight_only_gemv_launch`]: when the
/// (batch, shape) routes to the tensor-core mma path, the launch stops at the
/// f32 k-slice partials in `scratch` and returns the split factor (`out` is
/// untouched); otherwise it behaves exactly like the plain launch and returns
/// 0. Callers pair a non-zero return with a fused reduce consumer.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_weight_only_gemv_partials_launch(
    ctx: &DeviceContext,
    rows: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    scale_bytes: &CudaSlice<u8>,
    scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<usize> {
    let mut ksplit: i32 = 0;
    gemv_batched_launch(
        ctx,
        rows,
        n,
        k,
        activation,
        weight,
        scale_bytes,
        Some(scratch),
        out,
        Some(&mut ksplit),
    )?;
    Ok(ksplit as usize)
}

/// Shared body of the two entry points above: validation, pointer extraction,
/// and the FFI call (plain when `ksplit_out` is `None`, partials otherwise).
#[allow(clippy::too_many_arguments)]
fn gemv_batched_launch(
    ctx: &DeviceContext,
    rows: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    scale_bytes: &CudaSlice<u8>,
    scratch: Option<&mut CudaSlice<f32>>,
    out: &mut CudaSlice<bf16>,
    ksplit_out: Option<&mut i32>,
) -> Result<()> {
    ensure!(
        rows > 0 && n > 0 && k > 0,
        "GLM5.2 linear GEMV needs positive rows/n/k, got {rows}/{n}/{k}"
    );
    let scale_len = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4;
    ensure!(
        weight.len() >= n * k
            && scale_bytes.len() >= scale_len
            && activation.len() >= rows * k
            && out.len() >= rows * n,
        "GLM5.2 linear GEMV buffers too small: w {} (need {}), scale {} (need {scale_len}), act {} (need {}), out {} (need {})",
        weight.len(),
        n * k,
        scale_bytes.len(),
        activation.len(),
        rows * k,
        out.len(),
        rows * n
    );
    ensure!(
        ksplit_out.is_none() || rows <= 48,
        "GLM5.2 linear GEMV partials path is not token-split past 48 rows (got {rows})"
    );
    let (act_ptr, _a) = activation.device_ptr(&ctx.stream);
    let (w_ptr, _w) = weight.device_ptr(&ctx.stream);
    let (s_ptr, _s) = scale_bytes.device_ptr(&ctx.stream);
    let (out_ptr, _o) = out.device_ptr_mut(&ctx.stream);
    let (scr_ptr, scr_floats, _scr) = match scratch {
        Some(buf) => {
            let len = buf.len();
            let (ptr, guard) = buf.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, len, Some(guard))
        }
        None => (std::ptr::null_mut(), 0, None),
    };
    // rows == 1 runs the m=1 kernel; rows 2 the bit-parity register tile;
    // rows 4/8 the tensor-core mma path on winning shapes (deterministic per
    // bucket, not bit-identical to m=1). The CUDA side whitelists the
    // supported batches — a drifted GLM52_DECODE_BUCKETS crashes here.
    match ksplit_out {
        None => {
            // 64/96-row buckets (#817) split on the token axis into two
            // whitelisted sub-batches — the register tile at those widths
            // would spill (the #813 lesson), and the activation/out layouts
            // are token-major so the split is a pointer offset.
            let halves: &[(usize, usize)] = match rows {
                64 => &[(32, 0), (32, 32)],
                96 => &[(48, 0), (48, 48)],
                _ => &[(rows, 0)],
            };
            let mut result = cudarc::driver::sys::CUresult::CUDA_SUCCESS;
            for &(sub, off) in halves {
                result = unsafe {
                    ffi::glm52_fp8_weight_only_gemv_batched_cuda(
                        (act_ptr as *const ffi::Half).add(off * k),
                        w_ptr as *const u8,
                        s_ptr as *const f32,
                        (out_ptr as *mut ffi::Half).add(off * n),
                        scr_ptr,
                        scr_floats,
                        sub as i32,
                        n as i32,
                        k as i32,
                        ctx.stream.cu_stream(),
                    )
                };
                if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    break;
                }
            }
            result
        }
        Some(ksplit) => unsafe {
            ffi::glm52_fp8_weight_only_gemv_partials_cuda(
                act_ptr as *const ffi::Half,
                w_ptr as *const u8,
                s_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                scr_ptr,
                scr_floats,
                rows as i32,
                n as i32,
                k as i32,
                ctx.stream.cu_stream(),
                ksplit,
            )
        },
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 linear GEMV launch failed for rows={rows}, n={n}, k={k}: {err}"))
}

/// MLA's bs=1 q_a and kv_a projections in one graph node. The two weights and
/// outputs remain separate; CUDA concatenates only their block grids.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_weight_only_gemv_pair_launch(
    ctx: &DeviceContext,
    k: usize,
    activation: &CudaSlice<bf16>,
    n_a: usize,
    weight_a: &CudaSlice<u8>,
    scale_a: &CudaSlice<u8>,
    out_a: &mut CudaSlice<bf16>,
    n_b: usize,
    weight_b: &CudaSlice<u8>,
    scale_b: &CudaSlice<u8>,
    out_b: &mut CudaSlice<bf16>,
) -> Result<()> {
    let scale_len = |n: usize| n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4;
    ensure!(
        k > 0
            && activation.len() >= k
            && weight_a.len() >= n_a * k
            && scale_a.len() >= scale_len(n_a)
            && out_a.len() >= n_a
            && weight_b.len() >= n_b * k
            && scale_b.len() >= scale_len(n_b)
            && out_b.len() >= n_b,
        "GLM5.2 paired GEMV buffers do not cover [{n_a},{k}] + [{n_b},{k}]"
    );
    let (act_ptr, _act) = activation.device_ptr(&ctx.stream);
    let (wa_ptr, _wa) = weight_a.device_ptr(&ctx.stream);
    let (sa_ptr, _sa) = scale_a.device_ptr(&ctx.stream);
    let (oa_ptr, _oa) = out_a.device_ptr_mut(&ctx.stream);
    let (wb_ptr, _wb) = weight_b.device_ptr(&ctx.stream);
    let (sb_ptr, _sb) = scale_b.device_ptr(&ctx.stream);
    let (ob_ptr, _ob) = out_b.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_fp8_weight_only_gemv_pair_cuda(
            act_ptr as *const ffi::Half,
            wa_ptr as *const u8,
            sa_ptr as *const f32,
            oa_ptr as *mut ffi::Half,
            n_a as i32,
            wb_ptr as *const u8,
            sb_ptr as *const f32,
            ob_ptr as *mut ffi::Half,
            n_b as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 paired GEMV launch failed: {err}"))
}
