use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;
use crate::tensor::HiddenStatesRef;

/// Generic strided-batched bf16 GEMM — one `cublasGemmStridedBatchedEx` on the
/// graph-safe (workspace-free, decode/capture) cuBLAS handle. `lda`/`ldb`/`ldc`
/// are cuBLAS column-major leading dims and `stride_*` are per-batch element
/// strides; each buffer must hold at least `stride * batch` elements. This is
/// the shared primitive for per-head batched GEMMs (MLA absorption
/// `q_nope @ W_UK` and `latent @ W_UV`, batch = head count) so model crates do
/// not hand-roll bespoke batched-GEMM kernels. `transpose_*` map to CUBLAS_OP_T.
///
/// Args use BLAS notation (a/b/c, m/n/k) to match the cuBLAS call it wraps.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gemm_strided_batched_bf16(
    ctx: &DeviceContext,
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &impl cudarc::driver::DevicePtr<bf16>,
    lda: usize,
    stride_a: usize,
    b: &impl cudarc::driver::DevicePtr<bf16>,
    ldb: usize,
    stride_b: usize,
    c: &mut cudarc::driver::CudaSlice<bf16>,
    ldc: usize,
    stride_c: usize,
    batch: usize,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0 && batch > 0,
        "gemm_strided_batched_bf16 empty dims: m={m} n={n} k={k} batch={batch}"
    );
    ensure!(
        a.len() >= stride_a * batch,
        "gemm_strided_batched_bf16 A too small: have {}, need {}",
        a.len(),
        stride_a * batch
    );
    ensure!(
        b.len() >= stride_b * batch,
        "gemm_strided_batched_bf16 B too small: have {}, need {}",
        b.len(),
        stride_b * batch
    );
    ensure!(
        c.len() >= stride_c * batch,
        "gemm_strided_batched_bf16 C too small: have {}, need {}",
        c.len(),
        stride_c * batch
    );
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (c_ptr, _gc) = c.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::gemm_strided_batched_bf16_cuda(
            i32::from(transpose_a),
            i32::from(transpose_b),
            m as i32,
            n as i32,
            k as i32,
            a_ptr as *const ffi::Half,
            lda as i32,
            stride_a as i64,
            b_ptr as *const ffi::Half,
            ldb as i32,
            stride_b as i64,
            c_ptr as *mut ffi::Half,
            ldc as i32,
            stride_c as i64,
            batch as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "gemm_strided_batched_bf16 failed: status={status} (m={m} n={n} k={k} batch={batch})"
    );
    Ok(())
}

/// f32 strided-batched GEMM, same cuBLAS boundary as
/// [`gemm_strided_batched_bf16`] but with f32 operands and an `accumulate`
/// switch (`C += A·B` instead of `C = A·B`). The consumer is the K3 KCP state
/// merge (per-head fp32 `S' = M·S + D`), which pre-fills `C` with `D` and
/// accumulates the transition product in one call.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gemm_strided_batched_f32(
    ctx: &DeviceContext,
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &impl cudarc::driver::DevicePtr<f32>,
    lda: usize,
    stride_a: usize,
    b: &impl cudarc::driver::DevicePtr<f32>,
    ldb: usize,
    stride_b: usize,
    accumulate: bool,
    c: &mut cudarc::driver::CudaSlice<f32>,
    ldc: usize,
    stride_c: usize,
    batch: usize,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0 && batch > 0,
        "gemm_strided_batched_f32 empty dims: m={m} n={n} k={k} batch={batch}"
    );
    ensure!(
        a.len() >= stride_a * batch,
        "gemm_strided_batched_f32 A too small: have {}, need {}",
        a.len(),
        stride_a * batch
    );
    ensure!(
        b.len() >= stride_b * batch,
        "gemm_strided_batched_f32 B too small: have {}, need {}",
        b.len(),
        stride_b * batch
    );
    ensure!(
        c.len() >= stride_c * batch,
        "gemm_strided_batched_f32 C too small: have {}, need {}",
        c.len(),
        stride_c * batch
    );
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (c_ptr, _gc) = c.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::gemm_strided_batched_f32_cuda(
            i32::from(transpose_a),
            i32::from(transpose_b),
            m as i32,
            n as i32,
            k as i32,
            a_ptr as *const f32,
            lda as i32,
            stride_a as i64,
            b_ptr as *const f32,
            ldb as i32,
            stride_b as i64,
            if accumulate { 1.0 } else { 0.0 },
            c_ptr as *mut f32,
            ldc as i32,
            stride_c as i64,
            batch as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "gemm_strided_batched_f32 failed: status={status} (m={m} n={n} k={k} batch={batch})"
    );
    Ok(())
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
pub fn gemm_bf16_f32(
    ctx: &DeviceContext,
    transpose_a: bool,
    transpose_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &cudarc::driver::CudaSlice<bf16>,
    lda: usize,
    b: &cudarc::driver::CudaSlice<bf16>,
    ldb: usize,
    c: &mut cudarc::driver::CudaSlice<f32>,
    ldc: usize,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0 && a.len() >= m * k && b.len() >= n * k && c.len() >= m * n,
        "gemm_bf16_f32 buffers do not fit [{m},{n},{k}]"
    );
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (c_ptr, _gc) = c.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::gemm_bf16_f32_cuda(
            i32::from(transpose_a),
            i32::from(transpose_b),
            m as i32,
            n as i32,
            k as i32,
            a_ptr as *const ffi::Half,
            lda as i32,
            b_ptr as *const ffi::Half,
            ldb as i32,
            c_ptr as *mut f32,
            ldc as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(status == 0, "gemm_bf16_f32 failed: status={status}");
    Ok(())
}

/// GEMMs at or below this N consult the cublasLt plan cache. 32 covers the
/// decode buckets where cuBLAS's GemmEx heuristic picks badly (worst at
/// N in [8, 16], where it skips split-K entirely); above that the GEMMs are
/// wide enough that the default selection is fine.
pub const GEMM_LT_MAX_N: usize = 32;

/// Mirrors GEMM_LT_UNTUNED in csrc/shared/linear.cu.
const GEMM_LT_UNTUNED: i32 = -1;

/// Select the fastest cublasLt algo for a (num_rows, n, cols) GEMM and cache
/// it for this thread's subsequent decode GEMMs of that shape.
///
/// `samples` are (weight, row_offset) pairs of identically shaped slices.
/// Passing several layers' weights keeps the timing loop out of L2, so the
/// ranking matches steady-state decode where weights stream from DRAM. Must
/// run on the executor thread before CUDA Graph capture.
pub fn gemm_lt_tune(
    ctx: &DeviceContext,
    samples: &[(&DeviceMatrix, usize)],
    num_rows: usize,
    n: usize,
) -> Result<()> {
    let (first, _) = samples
        .first()
        .ok_or_else(|| anyhow::anyhow!("gemm_lt_tune requires at least one weight sample"))?;
    let cols = first.cols;
    let mut guards = Vec::with_capacity(samples.len());
    let mut ptrs: Vec<*const ffi::Half> = Vec::with_capacity(samples.len());
    for (weight, row_offset) in samples {
        assert_eq!(weight.cols, cols);
        assert!(row_offset + num_rows <= weight.rows);
        let (w_ptr, guard) = weight.data.device_ptr(&ctx.stream);
        ptrs.push(
            (w_ptr + (row_offset * cols * std::mem::size_of::<half::bf16>()) as u64)
                as *const ffi::Half,
        );
        guards.push(guard);
    }
    let status = unsafe {
        ffi::gemm_lt_tune_cuda(
            ptrs.as_ptr(),
            ptrs.len() as i32,
            num_rows as i32,
            n as i32,
            cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if status != 0 {
        bail!(
            "cublasLt GEMM tuning failed: status={}, m={}, n={}, k={}",
            status,
            num_rows,
            n,
            cols
        );
    }
    Ok(())
}

/// Mirrors the GEMM_LT_PIN_* sentinels in csrc/shared/linear.cu.
const GEMM_LT_PIN_UNTUNED: i32 = -1;
const GEMM_LT_PIN_UNSUPPORTED: i32 = -2;
const GEMM_LT_PIN_NONDET: i32 = -3;

#[derive(Clone, Copy, Debug)]
pub struct PinAlgoConfig {
    pub splitk: i32,
    reduction_scheme: i32,
}

impl PinAlgoConfig {
    pub fn reduction_scheme_name(self) -> &'static str {
        match self.reduction_scheme {
            0 => "none",
            1 => "inplace",
            2 => "compute_type",
            4 => "output_type",
            _ => "unknown",
        }
    }
}

/// Pin one cublasLt algo for `(num_rows, cols)`, selected by the heuristic at `rep_n` (shape-only),
/// reused for every N. Run on the GEMM-issuing thread (the plan cache is thread-local).
pub fn gemm_lt_pin_tune(num_rows: usize, rep_n: usize, cols: usize) -> Result<PinAlgoConfig> {
    let mut splitk = 0;
    let mut reduction_scheme = 0;
    let status = unsafe {
        ffi::gemm_lt_pin_tune_cuda(
            num_rows as i32,
            rep_n as i32,
            cols as i32,
            &raw mut splitk,
            &raw mut reduction_scheme,
        )
    };
    match status {
        0 => Ok(PinAlgoConfig {
            splitk,
            reduction_scheme,
        }),
        GEMM_LT_PIN_NONDET => bail!(
            "cublasLt pin tuning selected an unknown reduction scheme: m={}, k={}, reduction_scheme={}",
            num_rows,
            cols,
            reduction_scheme
        ),
        s => bail!(
            "cublasLt pin tuning failed: status={}, m={}, rep_n={}, k={}",
            s,
            num_rows,
            rep_n,
            cols
        ),
    }
}

/// Run the pinned `(rows, cols)` algo at this N. `Ok(false)` = algo can't serve this N; bails if
/// never pinned. Used by the kernel serviceability test only — the production Pin path is
/// `launch_gemm_pin`, which fails loud instead of returning `Ok(false)`.
pub fn gemm_lt_pin_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    let status = unsafe {
        ffi::gemm_lt_pin_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    match status {
        0 => Ok(true),
        GEMM_LT_PIN_UNSUPPORTED => Ok(false),
        GEMM_LT_PIN_UNTUNED => bail!(
            "gemm_lt_pin_into_checked: (m={}, k={}) was never pinned — call gemm_lt_pin_tune first",
            weight.rows,
            weight.cols
        ),
        s if s >= 100_000 => bail!(
            "cublasLt pin GEMM failed: cublas_status={}, m={}, n={}, k={}",
            s - 100_000,
            weight.rows,
            x.seq_len,
            weight.cols
        ),
        s => bail!(
            "cublasLt pin GEMM launch failed: cuda_status={}, m={}, n={}, k={}",
            s,
            weight.rows,
            x.seq_len,
            weight.cols
        ),
    }
}

/// Boot self-check: does the pinned `(rows, cols)` algo serve `n`? Host-side
/// `cublasLtMatmulAlgoCheck` only — no GEMM launch, no buffers. `Ok(false)` = won't serve this N
/// (including workspace-over-budget, exactly as production rejects it); bails if `(rows, cols)` was
/// never pinned.
pub fn gemm_lt_pin_check(rows: usize, n: usize, cols: usize) -> Result<bool> {
    let status = unsafe { ffi::gemm_lt_pin_check_cuda(rows as i32, n as i32, cols as i32) };
    match status {
        0 => Ok(true),
        GEMM_LT_PIN_UNSUPPORTED => Ok(false),
        GEMM_LT_PIN_UNTUNED => bail!(
            "gemm_lt_pin_check: (m={rows}, k={cols}) was never pinned — call gemm_lt_pin_warmup first"
        ),
        // No matmul is launched, so the only non-sentinel status is a cublas layout-create failure.
        s => {
            bail!("gemm_lt_pin_check: cublas layout error (status {s}), m={rows}, n={n}, k={cols}")
        }
    }
}

/// GEMM on a row sub-range of a weight matrix: Y = W[row_offset..row_offset+M, :] @ X
pub fn gemm_rows_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_rows_into_checked(ctx, weight, row_offset, num_rows, x, out)
        .expect("GEMM row-range launch failed");
}

/// Checked row-range GEMM. New hot paths should use this form so cuBLAS launch
/// failures surface at the operator boundary instead of at a later collective.
pub fn gemm_rows_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert!(row_offset + num_rows <= weight.rows);
    assert_eq!(weight.cols, x.hidden_dim);
    assert_eq!(out.hidden_dim, num_rows);
    assert_eq!(out.seq_len, x.seq_len);

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let w_sub = w_ptr + (row_offset * weight.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_sub as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        num_rows,
        x.seq_len,
        weight.cols,
        x.seq_len == 1,
        ctx,
    )
}

/// Row-range GEMM over a token span of `x`:
/// `Y = W[row_offset..row_offset+num_rows, :] @ X[x_row0..x_row0+out.seq_len]`.
/// The span length is `out.seq_len`; `x.seq_len` bounds the readable rows.
pub fn gemm_rows_span_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    x_row0: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    assert!(row_offset + num_rows <= weight.rows);
    assert_eq!(weight.cols, x.hidden_dim);
    assert_eq!(out.hidden_dim, num_rows);
    assert!(
        x_row0 + out.seq_len <= x.seq_len,
        "x span [{x_row0}, {x_row0}+{}) exceeds x.seq_len {}",
        out.seq_len,
        x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let w_sub = w_ptr + (row_offset * weight.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let x_sub = x_ptr + (x_row0 * x.hidden_dim * std::mem::size_of::<half::bf16>()) as u64;
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_sub as *const ffi::Half,
        x_sub as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        num_rows,
        out.seq_len,
        weight.cols,
        out.seq_len == 1,
        ctx,
    )
}

/// Matrix-vector multiplication: y = A @ x (via cuBLAS GEMM with N=1)
/// A: (M, K) row-major, x: (K,), y: (M,)
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        a_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        a.rows,
        1,
        a.cols,
        true,
        ctx,
    )
}
/// Linear layer: y = weight @ x
pub fn linear(ctx: &DeviceContext, x: &DeviceVec, weight: &DeviceMatrix) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
}

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_into_checked(ctx, weight, x, &mut out)?;
    Ok(out)
}

/// Per-token GEMM: each row is computed through the same N=1 cuBLAS boundary
/// used by decode. This is useful for narrow batch-shaped accuracy gates where
/// row-wise parity with the serial decode oracle matters more than throughput.
pub fn gemm_per_token(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_per_token_into_checked(ctx, weight, x, &mut out)?;
    Ok(out)
}

/// GEMM into pre-allocated output buffer (zero allocation).
/// For seq_len=1, uses the graph-safe cuBLAS handle (no workspace) for lower
/// latency while preserving numerical parity with the prefill path.
pub fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    gemm_into_checked(ctx, weight, x, out).expect("GEMM launch failed");
}

/// Checked GEMM using the default policy: graph-safe handle for single-token
/// decode, workspace-backed handle for prefill/batched projections.
pub fn gemm_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_into_with_policy(ctx, weight, x, out, x.seq_len == 1)
}

/// GEMM for a contiguous token range in `x` into a compact output buffer.
pub fn gemm_token_range_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    token_offset: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert!(
        token_offset + out.seq_len <= x.seq_len,
        "token range [{}..{}) exceeds input seq_len {}",
        token_offset,
        token_offset + out.seq_len,
        x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_base, _gx) = x.data.device_ptr(&ctx.stream);
    let x_ptr = x_base + (token_offset * x.hidden_dim * std::mem::size_of::<half::bf16>()) as u64;
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        weight.rows,
        out.seq_len,
        weight.cols,
        out.seq_len == 1,
        ctx,
    )
}

/// Checked GEMM that always uses the workspace-free cuBLAS handle. Kimi decode
/// uses this for active-batch sizes 1..=4 so graph-readiness is not tied to a
/// bs1-only condition.
pub fn gemm_graphsafe_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x.as_ref(), out, true)
}

pub fn gemm_graphsafe_ref_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: HiddenStatesRef<'_>,
    out: &mut HiddenStates,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x, out, true)
}

fn gemm_per_token_into_checked(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        let status = ffi::gemm_per_token_cuda(
            w_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            weight.rows as i32,
            x.seq_len as i32,
            weight.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS per-token GEMM failed: cublas_status={}, m={}, batch={}, k={}",
                    status - 100_000,
                    weight.rows,
                    x.seq_len,
                    weight.cols
                );
            }
            bail!(
                "CUDA per-token GEMM launch failed: cuda_status={}, m={}, batch={}, k={}",
                status,
                weight.rows,
                x.seq_len,
                weight.cols
            );
        }
    }
    Ok(())
}

fn gemm_into_with_policy(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    gemm_ref_into_with_policy(ctx, weight, x.as_ref(), out, graphsafe)
}

fn gemm_ref_into_with_policy(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: HiddenStatesRef<'_>,
    out: &mut HiddenStates,
    graphsafe: bool,
) -> Result<()> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    launch_gemm(
        w_ptr as *const ffi::Half,
        x_ptr as *const ffi::Half,
        y_ptr as *mut ffi::Half,
        weight.rows,
        x.seq_len,
        weight.cols,
        graphsafe,
        ctx,
    )
}

/// Process-global numeric policy for projection GEMMs (atomics, not thread-local: set on the main
/// thread before worker construction, read on the worker). `Tuned` (default) = production path;
/// `Pin` = batch-invariant pinned algo (one cuBLASLt algo per `(M,K)`; `launch_gemm_pin` bails — no
/// per-token fallback — if it can't serve an N or under a stream override); `PerToken` = N=1 oracle.
/// Set via `set_numeric_policy` before executor
/// construction / graph capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum NumericPolicy {
    Tuned = 0,
    Pin = 1,
    PerToken = 2,
}

static NUMERIC_POLICY: AtomicU8 = AtomicU8::new(NumericPolicy::Tuned as u8);
static PIN_SERVED: AtomicU64 = AtomicU64::new(0);
static PER_TOKEN_SERVED: AtomicU64 = AtomicU64::new(0);

/// Set the process-global policy. Must run before executor construction / graph capture (the
/// captured algo is the one live at capture). Tests drive baseline/pin/per-token through it.
pub fn set_numeric_policy(p: NumericPolicy) {
    NUMERIC_POLICY.store(p as u8, Ordering::Release);
}

pub fn numeric_policy() -> NumericPolicy {
    match NUMERIC_POLICY.load(Ordering::Acquire) {
        1 => NumericPolicy::Pin,
        2 => NumericPolicy::PerToken,
        _ => NumericPolicy::Tuned,
    }
}

/// NUMERIC_POLICY defaults to Tuned; only qwen3 sets Pin/PerToken, so typed GEMM (kimi) never fires this.
pub(crate) fn ensure_tuned_policy(m: usize, n: usize, k: usize) -> Result<()> {
    let policy = numeric_policy();
    ensure!(
        policy == NumericPolicy::Tuned,
        "typed GEMM is not covered by the batch-invariant pin: policy={policy:?}, m={m}, n={n}, k={k}"
    );
    Ok(())
}

/// Count of projection GEMMs served by the pinned algo: process-global, mixed across prefill +
/// decode + unified and across TP ranks. Under `Pin` a GEMM either serves (counted here) or
/// `launch_gemm_pin` bails — there is no silent per-token fallback. Decode counts the graph
/// capture, not replays; read quiesced.
pub fn pin_served() -> u64 {
    PIN_SERVED.load(Ordering::Relaxed)
}

/// Count of projection GEMMs served by the PerToken policy path (`launch_gemm_pertoken`). Symmetric
/// to `pin_served`: nonzero proves PerToken hit the oracle, not a silent fallback.
pub fn per_token_served() -> u64 {
    PER_TOKEN_SERVED.load(Ordering::Relaxed)
}

/// Resets the pin and per-token served counters.
pub fn reset_numeric_policy_counters() {
    PIN_SERVED.store(0, Ordering::Relaxed);
    PER_TOKEN_SERVED.store(0, Ordering::Relaxed);
}

/// Fixed representative N at which every (M,K) pin is resolved — one algo for ALL
/// N (NOT first-call-N, which would make the pinned algo call-order-dependent).
const PIN_REP_N: i32 = 32;

/// Pin one `(num_rows, cols)` algo at the canonical `PIN_REP_N`, reused for all N.
pub fn gemm_lt_pin_warmup(num_rows: usize, cols: usize) -> Result<PinAlgoConfig> {
    gemm_lt_pin_tune(num_rows, PIN_REP_N as usize, cols)
}

/// `Pin` policy: run the boot-warmed pinned (m,k) algo at live N; bails if (m,k) isn't pinned or the
/// algo can't serve N (no lazy tune, no per-token fallback, which would break invariance).
fn launch_gemm_pin(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
) -> Result<()> {
    // decode-overlap (the only stream-override source) is rejected before capture, so a stream
    // override here is misuse — bail rather than fall back to per-token.
    if crate::tensor::has_stream_override() {
        bail!(
            "batch-invariant Pin GEMM cannot run under a stream override (m={m}, n={n}, k={k}): \
             --batch-invariant is incompatible with --decode-overlap; run without one of them"
        );
    }
    // Only the boot-warmed base projections are pinned; any other (m,k) is unverified —
    // gemm_lt_pin_cuda returns GEMM_LT_PIN_UNTUNED and the match below bails. No lazy tune: it would
    // run an unchecked shape and allocate illegally mid-capture.
    unsafe {
        let status = ffi::gemm_lt_pin_cuda(
            w_ptr,
            x_ptr,
            y_ptr,
            m as i32,
            n as i32,
            k as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        match status {
            0 => {
                PIN_SERVED.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            GEMM_LT_PIN_UNSUPPORTED | GEMM_LT_PIN_UNTUNED => bail!(
                "batch-invariant Pin GEMM cannot serve N={n} at (m={m}, k={k}): either this N is \
                 outside the pinned envelope, or (M,K) was never warmed (e.g. a DFlash drafter or \
                 LoRA prefill-delta shape), or the numeric policy was set after executor \
                 construction. Run without --batch-invariant or report this shape"
            ),
            s if s >= 100_000 => {
                bail!(
                    "cuBLAS pin GEMM failed: cublas_status={}, m={m}, n={n}, k={k}",
                    s - 100_000
                )
            }
            s => bail!("CUDA pin GEMM launch failed: cuda_status={s}, m={m}, n={n}, k={k}"),
        }
    }
}

/// `PerToken` policy: the N=1-per-column oracle (invariant by construction).
fn launch_gemm_pertoken(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
) -> Result<()> {
    unsafe {
        let status = ffi::gemm_per_token_cuda(
            w_ptr,
            x_ptr,
            y_ptr,
            m as i32,
            n as i32,
            k as i32,
            crate::tensor::active_cu_stream(ctx),
        );
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS per-token GEMM failed: cublas_status={}, m={m}, n={n}, k={k}",
                    status - 100_000
                );
            }
            bail!("CUDA per-token GEMM launch failed: cuda_status={status}, m={m}, n={n}, k={k}");
        }
    }
    PER_TOKEN_SERVED.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn launch_gemm(
    w_ptr: *const ffi::Half,
    x_ptr: *const ffi::Half,
    y_ptr: *mut ffi::Half,
    m: usize,
    n: usize,
    k: usize,
    graphsafe: bool,
    ctx: &DeviceContext,
) -> Result<()> {
    // Non-Tuned policies route every N through the pinned/per-token path; Tuned falls through.
    match numeric_policy() {
        NumericPolicy::Pin => return launch_gemm_pin(w_ptr, x_ptr, y_ptr, m, n, k, ctx),
        NumericPolicy::PerToken => return launch_gemm_pertoken(w_ptr, x_ptr, y_ptr, m, n, k, ctx),
        NumericPolicy::Tuned => {}
    }
    // Split-concurrent prefill must keep even N=1 projections on its dedicated
    // handle so it cannot rebind the graph-safe decode handle.
    let graphsafe = graphsafe && !crate::tensor::has_prefill_stream_override();
    unsafe {
        // Small-N projections run the cublasLt algo selected by gemm_lt_tune.
        // Shapes this thread never tuned report GEMM_LT_UNTUNED and keep their
        // existing cublasGemmEx path (and its capture behavior) unchanged.
        //
        // NOTE: gemm_lt is disabled when a stream override is active (SM-partition
        // concurrent mode). cuBLASLt has device-global state that conflicts when
        // two green-ctx streams run cublasLtMatmul concurrently, causing Xid 31.
        let mut status = if n <= GEMM_LT_MAX_N && !crate::tensor::has_stream_override() {
            ffi::gemm_lt_cuda(
                w_ptr,
                x_ptr,
                y_ptr,
                m as i32,
                n as i32,
                k as i32,
                crate::tensor::active_cu_stream(ctx),
            )
        } else {
            GEMM_LT_UNTUNED
        };
        if status == GEMM_LT_UNTUNED {
            status = if graphsafe {
                ffi::gemm_graphsafe_cuda(
                    w_ptr,
                    x_ptr,
                    y_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    crate::tensor::active_cu_stream(ctx),
                )
            } else {
                ffi::gemm_cuda(
                    w_ptr,
                    x_ptr,
                    y_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    crate::tensor::active_cu_stream(ctx),
                )
            };
        }
        if status != 0 {
            if status >= 100_000 {
                bail!(
                    "cuBLAS GEMM failed: cublas_status={}, m={}, n={}, k={}, graphsafe={}",
                    status - 100_000,
                    m,
                    n,
                    k,
                    graphsafe
                );
            }
            bail!(
                "CUDA GEMM launch failed: cuda_status={}, m={}, n={}, k={}, graphsafe={}",
                status,
                m,
                n,
                k,
                graphsafe
            );
        }
    }
    Ok(())
}
