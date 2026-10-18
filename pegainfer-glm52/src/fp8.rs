//! Shared GLM5.2 fp8 block-scaled projection primitives (decode, row-batched).
//!
//! Every dense/attention/expert projection in GLM5.2 is fp8 e4m3 with a per-128
//! block `weight_scale_inv`. At m=1 the projection is a weight-only GEMV: the
//! bf16 activation is read directly and the fp8 weight is block-scale dequanted
//! on the fly (`csrc/glm52/glm52_moe_gemv.cu`) — no activation quant, no TMA
//! scale relayout, one kernel instead of three, and weight-memory-bound instead
//! of the TRTLLM CUTLASS M-tile padding 1→64. MLA, the dense MLP, and the MoE
//! shared expert all share these helpers; only the EP8 routed-expert chain
//! (multi-row) stays on the grouped CUTLASS path.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::glm52_fp8_groupwise_gemm_sm100_bank_launch;
use pegainfer_kernels::ops::glm52_fp8_groupwise_gemm_sm100_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_launch;
use pegainfer_kernels::ops::glm52_fp8_weight_only_gemv_launch;
use pegainfer_kernels::ops::glm52_fp8_weight_only_gemv_pair_launch;
use pegainfer_kernels::ops::glm52_fp8_weight_only_gemv_partials_launch;
use pegainfer_kernels::ops::glm52_gemv_reduce_silu_mul_launch;
use pegainfer_kernels::ops::glm52_silu_and_mul_bf16_launch;
use pegainfer_kernels::tensor::DeviceContext;

pub(crate) const FP8_BLOCK: usize = 128;
const FP8_GEMM_WORKSPACE_BYTES: usize = 32 << 20;

/// The batched GEMV's row ceiling: past this the register tile spills and
/// the fp8 GEMM path (activation quant + FlashInfer SM100 groupwise GEMM)
/// wins — measured on the #812 verify-span buckets, where GEMV-at-48 tripled
/// the step time.
pub(crate) const FP8_GEMV_MAX_ROWS: usize = 8;

pub(crate) struct Glm52Fp8GemmScratch {
    rows: usize,
    width: usize,
    activation: CudaSlice<u8>,
    activation_scale: CudaSlice<f32>,
    workspace: CudaSlice<u8>,
}

impl Glm52Fp8GemmScratch {
    pub(crate) fn new(ctx: &DeviceContext, rows: usize, width: usize) -> Result<Self> {
        ensure!(
            rows > 0 && rows.is_multiple_of(4) && width.is_multiple_of(FP8_BLOCK),
            "GLM5.2 FP8 GEMM scratch requires rows%4=0 and width%128=0"
        );
        // Scratch construction runs before the bucket's CUDA-graph capture —
        // the only legal window for the DSL AOT module load.
        pegainfer_kernels::ops::glm52_fp8_dsl_preload();
        Ok(Self {
            rows,
            width,
            activation: ctx.stream.alloc_zeros::<u8>(rows * width)?,
            activation_scale: ctx
                .stream
                .alloc_zeros::<f32>(rows * width.div_ceil(FP8_BLOCK))?,
            workspace: ctx.stream.alloc_zeros::<u8>(FP8_GEMM_WORKSPACE_BYTES)?,
        })
    }
}

pub(crate) fn fp8_linear_large_m_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    rows: usize,
    input: &impl cudarc::driver::DevicePtr<bf16>,
    scratch: &mut Glm52Fp8GemmScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0
            && rows.is_multiple_of(4)
            && rows <= scratch.rows
            && w.k <= scratch.width
            && input.len() >= rows * w.k
            && out.len() >= rows * w.n,
        "GLM5.2 large-M FP8 projection buffers do not fit [{rows}, {}, {}]",
        w.n,
        w.k
    );
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows,
            width: w.k,
            group_size: FP8_BLOCK,
        },
        input,
        &mut scratch.activation,
        &mut scratch.activation_scale,
    )?;
    glm52_fp8_groupwise_gemm_sm100_launch(
        ctx,
        rows,
        w.n,
        w.k,
        &scratch.activation,
        &scratch.activation_scale,
        &w.weight,
        &w.scale,
        out,
        &mut scratch.workspace,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_linear_large_m_bank_into(
    ctx: &DeviceContext,
    rows: usize,
    n: usize,
    k: usize,
    input: &impl cudarc::driver::DevicePtr<bf16>,
    weight: &CudaSlice<u8>,
    weight_offset: usize,
    scale: &CudaSlice<f32>,
    scale_offset: usize,
    scratch: &mut Glm52Fp8GemmScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0
            && rows.is_multiple_of(4)
            && rows <= scratch.rows
            && k <= scratch.width
            && input.len() >= rows * k
            && out.len() >= rows * n,
        "GLM5.2 bank FP8 projection buffers do not fit [{rows}, {n}, {k}]"
    );
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows,
            width: k,
            group_size: FP8_BLOCK,
        },
        input,
        &mut scratch.activation,
        &mut scratch.activation_scale,
    )?;
    glm52_fp8_groupwise_gemm_sm100_bank_launch(
        ctx,
        rows,
        n,
        k,
        &scratch.activation,
        &scratch.activation_scale,
        weight,
        weight_offset,
        scale,
        scale_offset,
        out,
        &mut scratch.workspace,
    )
}

/// OCP `float8_e4m3fn` decode (bias 7, no inf; subnormals supported). Used by the
/// host-side dequant paths (kv_b absorb factors), not the GPU kernels.
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b >> 7) & 1 == 1 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let mag = if e == 0 {
        2f32.powi(-6) * (m / 8.0)
    } else {
        2f32.powi(e - 7) * (1.0 + m / 8.0)
    };
    sign * mag
}

pub(crate) fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Raw fp8 block-scaled projection bytes (row-major weight `[n,k]` + per-128-block
/// `weight_scale_inv` `[n/128, k/128]` f32).
pub(crate) struct Glm52ProjBytes<'a> {
    pub(crate) weight: &'a [u8],
    pub(crate) scale: &'a [u8],
    pub(crate) n: usize,
    pub(crate) k: usize,
}

/// One fp8 projection resident on device.
pub(crate) struct ProjWeight {
    pub(crate) weight: CudaSlice<u8>,
    pub(crate) scale: CudaSlice<u8>,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl ProjWeight {
    #[cfg(test)]
    pub(crate) fn upload(ctx: &DeviceContext, b: &Glm52ProjBytes) -> Result<Self> {
        ensure!(
            b.weight.len() == b.n * b.k,
            "GLM5.2 proj weight bytes {} != n*k {}",
            b.weight.len(),
            b.n * b.k
        );
        ensure!(
            b.scale.len() == b.n.div_ceil(FP8_BLOCK) * b.k.div_ceil(FP8_BLOCK) * 4,
            "GLM5.2 proj scale bytes {} unexpected for [{},{}]",
            b.scale.len(),
            b.n,
            b.k
        );
        let mut weight = ctx.stream.alloc_zeros::<u8>(b.weight.len())?;
        let mut scale = ctx.stream.alloc_zeros::<u8>(b.scale.len())?;
        ctx.stream.memcpy_htod(b.weight, &mut weight)?;
        ctx.stream.memcpy_htod(b.scale, &mut scale)?;
        Ok(Self {
            weight,
            scale,
            n: b.n,
            k: b.k,
        })
    }

    /// Wrap already-resident GPU buffers (the production loader path), moving them
    /// in with no copy. `weight` is the fp8 `[n,k]` e4m3 bytes, `scale` the f32
    /// `weight_scale_inv` (`[n/128, k/128]`) kept as raw `u8`. Same validation as
    /// `upload`, so a packaging drift crashes here, not in the kernel.
    pub(crate) fn from_device(
        weight: CudaSlice<u8>,
        scale: CudaSlice<u8>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        ensure!(
            weight.len() == n * k,
            "GLM5.2 proj weight bytes {} != n*k {}",
            weight.len(),
            n * k
        );
        ensure!(
            scale.len() == n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK) * 4,
            "GLM5.2 proj scale bytes {} unexpected for [{n},{k}]",
            scale.len()
        );
        Ok(Self {
            weight,
            scale,
            n,
            k,
        })
    }
}

impl ProjWeight {
    /// n-side (output-row) shard: rows `[row_start, row_start + rows)` as a
    /// fresh projection. Contiguous device copies. Both bounds must sit on
    /// the 128-row scale-block grid — the per-block `weight_scale_inv` rows
    /// travel with their weight rows (attention-TP head shards: q_b 8 heads
    /// x 256 = 2048 rows, indexer wq_b 4 x 128 = 512, both aligned).
    pub(crate) fn slice_rows(
        &self,
        ctx: &DeviceContext,
        row_start: usize,
        rows: usize,
    ) -> Result<ProjWeight> {
        ensure!(
            row_start.is_multiple_of(FP8_BLOCK) && rows.is_multiple_of(FP8_BLOCK),
            "GLM5.2 proj row slice [{row_start}, +{rows}) off the {FP8_BLOCK}-row scale grid"
        );
        ensure!(
            row_start + rows <= self.n,
            "GLM5.2 proj row slice [{row_start}, +{rows}) exceeds n {}",
            self.n
        );
        let mut weight = ctx.stream.alloc_zeros::<u8>(rows * self.k)?;
        ctx.stream.memcpy_dtod(
            &self
                .weight
                .slice(row_start * self.k..(row_start + rows) * self.k),
            &mut weight,
        )?;
        let scale_cols = self.k.div_ceil(FP8_BLOCK);
        let mut scale = ctx
            .stream
            .alloc_zeros::<u8>((rows / FP8_BLOCK) * scale_cols * 4)?;
        ctx.stream.memcpy_dtod(
            &self.scale.slice(
                (row_start / FP8_BLOCK) * scale_cols * 4
                    ..((row_start + rows) / FP8_BLOCK) * scale_cols * 4,
            ),
            &mut scale,
        )?;
        ProjWeight::from_device(weight, scale, rows, self.k)
    }

    /// k-side (input-column) shard: columns `[col_start, col_start + cols)`
    /// of every row (attention-TP o_proj, whose INPUT is head-major). The
    /// gather is strided, so it bounces through the host once — load-time
    /// only. Bounds must sit on the 128-column scale-block grid.
    pub(crate) fn slice_cols(
        &self,
        ctx: &DeviceContext,
        col_start: usize,
        cols: usize,
    ) -> Result<ProjWeight> {
        ensure!(
            col_start.is_multiple_of(FP8_BLOCK) && cols.is_multiple_of(FP8_BLOCK),
            "GLM5.2 proj col slice [{col_start}, +{cols}) off the {FP8_BLOCK}-col scale grid"
        );
        ensure!(
            col_start + cols <= self.k,
            "GLM5.2 proj col slice [{col_start}, +{cols}) exceeds k {}",
            self.k
        );
        let full_w = ctx.stream.clone_dtoh(&self.weight)?;
        let mut w_host = vec![0u8; self.n * cols];
        for row in 0..self.n {
            w_host[row * cols..(row + 1) * cols].copy_from_slice(
                &full_w[row * self.k + col_start..row * self.k + col_start + cols],
            );
        }
        let full_s = ctx.stream.clone_dtoh(&self.scale)?;
        let scale_rows = self.n.div_ceil(FP8_BLOCK);
        let scale_cols = self.k.div_ceil(FP8_BLOCK);
        let sliced_cols = cols / FP8_BLOCK;
        let col_block = col_start / FP8_BLOCK;
        let mut s_host = vec![0u8; scale_rows * sliced_cols * 4];
        for row in 0..scale_rows {
            let src = (row * scale_cols + col_block) * 4;
            s_host[row * sliced_cols * 4..(row + 1) * sliced_cols * 4]
                .copy_from_slice(&full_s[src..src + sliced_cols * 4]);
        }
        let mut weight = ctx.stream.alloc_zeros::<u8>(w_host.len())?;
        ctx.stream.memcpy_htod(&w_host, &mut weight)?;
        let mut scale = ctx.stream.alloc_zeros::<u8>(s_host.len())?;
        ctx.stream.memcpy_htod(&s_host, &mut scale)?;
        ProjWeight::from_device(weight, scale, self.n, cols)
    }
}

/// Pack two fp8 projections that share the same input into one `[a.n + b.n, k]`
/// projection (weight bytes and per-128-block scale rows concatenated along n).
/// Requires `a.n` to be a multiple of 128 so `b`'s scale rows stay aligned to
/// the packed row/128 grid. Clients: gate|up (a single GEMV writes the
/// `[gate | up]` layout the SwiGLU consumes) and the MLA q_a|kv_a horizontal
/// pack (`mla_front::pack_qa_kva`).
pub(crate) fn pack_proj_pair(
    ctx: &DeviceContext,
    a: &ProjWeight,
    b: &ProjWeight,
) -> Result<ProjWeight> {
    ensure!(
        a.k == b.k,
        "GLM5.2 proj pack k mismatch: {} vs {}",
        a.k,
        b.k
    );
    ensure!(
        a.n.is_multiple_of(FP8_BLOCK),
        "GLM5.2 proj pack first n {} not a multiple of {FP8_BLOCK}",
        a.n
    );
    let n = a.n + b.n;
    let mut weight = ctx.stream.alloc_zeros::<u8>(n * a.k)?;
    ctx.stream
        .memcpy_dtod(&a.weight, &mut weight.slice_mut(0..a.n * a.k))?;
    ctx.stream
        .memcpy_dtod(&b.weight, &mut weight.slice_mut(a.n * a.k..n * a.k))?;
    let scale_cols = a.k.div_ceil(FP8_BLOCK);
    let a_scale_len = (a.n / FP8_BLOCK) * scale_cols * 4;
    let b_scale_len = b.n.div_ceil(FP8_BLOCK) * scale_cols * 4;
    let mut scale = ctx.stream.alloc_zeros::<u8>(a_scale_len + b_scale_len)?;
    ctx.stream
        .memcpy_dtod(&a.scale, &mut scale.slice_mut(0..a_scale_len))?;
    ctx.stream.memcpy_dtod(
        &b.scale,
        &mut scale.slice_mut(a_scale_len..a_scale_len + b_scale_len),
    )?;
    ProjWeight::from_device(weight, scale, n, a.k)
}

/// One fp8 projection into a pre-allocated output: weight-only GEMV of `rows`
/// bf16 activation rows against the fp8 `[n,k]` weight, block scale dequanted
/// on the fly. rows > 1 runs a weight-stationary batched kernel — the weight
/// is still read once. rows 1/2 keep bit-stable CUDA-core paths; rows 4/8
/// dispatch to a tensor-core mma path on winning shapes (deterministic per
/// bucket, not bit-identical to rows=1 — cross-bucket FP divergence is the
/// accepted whole-step contract; numerics notes in glm52_moe_gemv.cu).
pub(crate) fn fp8_linear_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    rows: usize,
    input: &CudaSlice<bf16>,
    scratch: Option<&mut CudaSlice<f32>>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        input.len() >= rows * w.k,
        "GLM5.2 fp8_linear input {} < rows {rows} * k {}",
        input.len(),
        w.k
    );
    glm52_fp8_weight_only_gemv_launch(
        ctx, rows, w.n, w.k, input, &w.weight, &w.scale, scratch, out,
    )
}

/// [`fp8_linear_into`] that stops at the f32 k-slice partials when the (rows,
/// shape) routes to the mma path, so a fused epilogue can absorb the
/// fixed-order reduce. Returns the ksplit the partials were written with;
/// 0 means the register-tile path already wrote bf16 into `out`.
pub(crate) fn fp8_linear_partials_into(
    ctx: &DeviceContext,
    w: &ProjWeight,
    rows: usize,
    input: &CudaSlice<bf16>,
    scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<usize> {
    ensure!(
        input.len() >= rows * w.k,
        "GLM5.2 fp8_linear input {} < rows {rows} * k {}",
        input.len(),
        w.k
    );
    glm52_fp8_weight_only_gemv_partials_launch(
        ctx, rows, w.n, w.k, input, &w.weight, &w.scale, scratch, out,
    )
}

/// Two bs=1 projections sharing one activation and one CUDA graph node.
/// Restricted by the CUDA ABI to MLA q_a + kv_a; this helper owns the common
/// shape and buffer checks without exposing raw projection storage elsewhere.
pub(crate) fn fp8_linear_pair_into(
    ctx: &DeviceContext,
    a: &ProjWeight,
    b: &ProjWeight,
    input: &CudaSlice<bf16>,
    out_a: &mut CudaSlice<bf16>,
    out_b: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(a.k == b.k, "GLM5.2 paired projection k mismatch");
    glm52_fp8_weight_only_gemv_pair_launch(
        ctx, a.k, input, a.n, &a.weight, &a.scale, out_a, b.n, &b.weight, &b.scale, out_b,
    )
}

/// Allocating convenience over [`fp8_linear_into`] for the oracle-gate/test
/// paths. Returns `[n]` bf16.
#[cfg(test)]
pub(crate) fn fp8_linear(
    ctx: &DeviceContext,
    w: &ProjWeight,
    input: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let mut out = ctx.stream.alloc_zeros::<bf16>(w.n)?;
    fp8_linear_into(ctx, w, 1, input, None, &mut out)?;
    Ok(out)
}

/// Persistent scratch for one fp8 SwiGLU MLP shape. Sized to an exact
/// `intermediate` (the dense MLP and the MoE shared expert differ — 12288 vs
/// 2048 — and each gets its own instance so a cross-wiring crashes here).
pub(crate) struct Glm52MlpScratch {
    intermediate: usize,
    rows: usize,
    gate_up: CudaSlice<bf16>,
    silu_out: CudaSlice<bf16>,
    // Owned mma partial buffer: one per scratch so the ctx/aux stream overlap
    // can never see a shared pointer (see glm52_fp8_weight_only_gemv_launch).
    gemv_partial: CudaSlice<f32>,
    // Wide-bucket route (#812): rows past the GEMV ceiling run the fp8 GEMM
    // instead. Same per-scratch ownership discipline as `gemv_partial`.
    wide: Option<Glm52Fp8GemmScratch>,
}

impl Glm52MlpScratch {
    pub(crate) fn new(ctx: &DeviceContext, intermediate: usize, rows: usize) -> Result<Self> {
        ensure!(
            intermediate.is_multiple_of(FP8_BLOCK),
            "GLM5.2 fp8_mlp intermediate {intermediate} not a multiple of {FP8_BLOCK}"
        );
        ensure!(rows > 0, "GLM5.2 fp8_mlp scratch needs positive rows");
        Ok(Self {
            intermediate,
            rows,
            gate_up: ctx.stream.alloc_zeros::<bf16>(rows * 2 * intermediate)?,
            silu_out: ctx.stream.alloc_zeros::<bf16>(rows * intermediate)?,
            gemv_partial: ctx
                .stream
                .alloc_zeros::<f32>(rows * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?,
            wide: (rows > FP8_GEMV_MAX_ROWS)
                .then(|| {
                    // gate|up consumes hidden (k = 6144); down consumes the
                    // intermediate — one scratch covers both legs.
                    Glm52Fp8GemmScratch::new(
                        ctx,
                        rows.next_multiple_of(4),
                        crate::config::GLM52_HIDDEN.max(intermediate),
                    )
                })
                .transpose()?,
        })
    }
}

/// A plain fp8 SwiGLU MLP over the scratch's `rows` tokens into a
/// pre-allocated output: `down(silu(gate(h)) * up(h))` with the gate|up
/// projections PACKED into one `[2*intermediate, k]` weight (see
/// [`pack_proj_pair`]) — a single GEMV writes the `[gate | up]` layout the
/// SwiGLU consumes, no concat copies. `out` is `[rows, down.n]` bf16.
pub(crate) fn fp8_mlp_into(
    ctx: &DeviceContext,
    gate_up: &ProjWeight,
    down: &ProjWeight,
    input: &CudaSlice<bf16>,
    s: &mut Glm52MlpScratch,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    // gate_up/down internal consistency is pinned where the weight bundles
    // are built (`Glm52DenseMlpWeights` / `Glm52MoeSharedExpert` check exact
    // shapes); the scratch pairing is the one cross-object join this call
    // introduces, so it is the one thing validated here.
    let intermediate = gate_up.n / 2;
    ensure!(
        s.intermediate == intermediate,
        "GLM5.2 fp8_mlp scratch sized for intermediate {} but weights have {intermediate}",
        s.intermediate
    );
    // Wide buckets (#812): both projections run the fp8 GEMM; the SwiGLU is
    // the standalone bf16 kernel (no partials to absorb).
    if let Some(wide) = s.wide.as_mut() {
        fp8_linear_large_m_into(ctx, gate_up, s.rows, input, wide, &mut s.gate_up)?;
        glm52_silu_and_mul_bf16_launch(ctx, s.rows, intermediate, &s.gate_up, &mut s.silu_out)?;
        return fp8_linear_large_m_into(ctx, down, s.rows, &s.silu_out, wide, out);
    }
    // The gate|up projection stops at its f32 k-slice partials when the
    // (rows, shape) routes to the mma path, and the SwiGLU absorbs the
    // fixed-order reduce (one launch instead of two, bit-identical); the
    // register-tile rows keep the bf16 GEMV -> standalone SwiGLU pair.
    let ksplit = glm52_fp8_weight_only_gemv_partials_launch(
        ctx,
        s.rows,
        gate_up.n,
        gate_up.k,
        input,
        &gate_up.weight,
        &gate_up.scale,
        &mut s.gemv_partial,
        &mut s.gate_up,
    )?;
    if ksplit == 0 {
        // bf16 SwiGLU (no route weight, no activation quant) -> bf16 down input.
        glm52_silu_and_mul_bf16_launch(ctx, s.rows, intermediate, &s.gate_up, &mut s.silu_out)?;
    } else {
        glm52_gemv_reduce_silu_mul_launch(
            ctx,
            s.rows,
            intermediate,
            ksplit,
            &s.gemv_partial,
            &mut s.silu_out,
        )?;
    }
    fp8_linear_into(
        ctx,
        down,
        s.rows,
        &s.silu_out,
        Some(&mut s.gemv_partial),
        out,
    )
}

#[cfg(test)]
mod tests {
    use pegainfer_kernels::ops::glm52_gemv_mma_routes;
    use pegainfer_kernels::ops::glm52_gemv_split_reduce_launch;

    use super::*;

    /// e4m3fn bytes avoiding the NaN encodings (0x7F / 0xFF).
    fn synth_fp8(len: usize, salt: usize) -> Vec<u8> {
        (0..len)
            .map(|i| {
                let b = ((i * 31 + salt * 17) % 256) as u8;
                if b & 0x7F == 0x7F { b & 0x7E } else { b }
            })
            .collect()
    }

    fn synth_scale(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
        (0..rows * cols)
            .flat_map(|i| (0.001f32 + 0.0005 * ((i + salt) % 7) as f32).to_le_bytes())
            .collect()
    }

    /// Batch-8 parity of the horizontal q_a|kv_a pack against the separate
    /// launches at the production shapes. kv_a must be BIT-exact (the packed
    /// {16,1} config k-slices its rows exactly like the separate launch);
    /// q_a's split factor changes (48 -> 16 slices), so it gets a tolerance.
    /// Skips (trivially green) where the mma table has no packed route.
    #[test]
    #[ignore = "requires a GPU"]
    fn qa_kva_pack_batch8_parity() -> Result<()> {
        let ctx = DeviceContext::new_with_device(0)?;
        let (n_qa, n_kva, k, t) = (2048usize, 576usize, 6144usize, 8usize);
        if !glm52_gemv_mma_routes(t, n_qa + n_kva, k)? {
            eprintln!("no packed mma route on this arch; parity is vacuous");
            return Ok(());
        }
        let q_a = ProjWeight::upload(
            &ctx,
            &Glm52ProjBytes {
                weight: &synth_fp8(n_qa * k, 1),
                scale: &synth_scale(n_qa / FP8_BLOCK, k / FP8_BLOCK, 1),
                n: n_qa,
                k,
            },
        )?;
        let kv_a = ProjWeight::upload(
            &ctx,
            &Glm52ProjBytes {
                weight: &synth_fp8(n_kva * k, 2),
                scale: &synth_scale(n_kva.div_ceil(FP8_BLOCK), k / FP8_BLOCK, 2),
                n: n_kva,
                k,
            },
        )?;
        let act_host: Vec<bf16> = (0..t * k)
            .map(|i| bf16::from_f32(((i % 61) as f32 - 30.0) * 0.03))
            .collect();
        let activation = ctx.stream.clone_htod(&act_host)?;
        let mut scratch = ctx
            .stream
            .alloc_zeros::<f32>(t * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?;

        let mut qa_sep = ctx.stream.alloc_zeros::<bf16>(t * n_qa)?;
        let mut kva_sep = ctx.stream.alloc_zeros::<bf16>(t * n_kva)?;
        fp8_linear_into(&ctx, &q_a, t, &activation, Some(&mut scratch), &mut qa_sep)?;
        fp8_linear_into(
            &ctx,
            &kv_a,
            t,
            &activation,
            Some(&mut scratch),
            &mut kva_sep,
        )?;

        let packed = pack_proj_pair(&ctx, &q_a, &kv_a)?;
        let mut packed_sink = ctx.stream.alloc_zeros::<bf16>(t * packed.n)?;
        let ksplit = fp8_linear_partials_into(
            &ctx,
            &packed,
            t,
            &activation,
            &mut scratch,
            &mut packed_sink,
        )?;
        assert!(ksplit > 0, "packed launch took the register tile");
        let mut qa_fused = ctx.stream.alloc_zeros::<bf16>(t * n_qa)?;
        let mut kva_fused = ctx.stream.alloc_zeros::<bf16>(t * n_kva)?;
        glm52_gemv_split_reduce_launch(
            &ctx,
            t,
            n_qa,
            n_kva,
            ksplit,
            &scratch,
            &mut qa_fused,
            &mut kva_fused,
        )?;

        let kva_sep_h = ctx.stream.clone_dtoh(&kva_sep)?;
        let kva_fused_h = ctx.stream.clone_dtoh(&kva_fused)?;
        assert_eq!(kva_sep_h, kva_fused_h, "kv_a must be bit-exact");

        let qa_sep_h = ctx.stream.clone_dtoh(&qa_sep)?;
        let qa_fused_h = ctx.stream.clone_dtoh(&qa_fused)?;
        let mut worst = 0f32;
        for (a, b) in qa_sep_h.iter().zip(&qa_fused_h) {
            let (a, b) = (a.to_f32(), b.to_f32());
            let rel = (a - b).abs() / a.abs().max(1e-3);
            worst = worst.max(rel);
        }
        assert!(worst < 1e-2, "q_a fused/separate rel diff {worst}");
        Ok(())
    }

    /// Byte-level geometry check for the attention-TP shard helpers: build a
    /// projection whose every weight/scale byte encodes its own (row, col)
    /// coordinate, slice, and compare against the host-computed reference.
    #[test]
    #[ignore = "requires a GPU"]
    fn proj_slice_geometry() -> Result<()> {
        let ctx = DeviceContext::new_with_device(0)?;
        let (n, k) = (384, 256);
        let weight: Vec<u8> = (0..n * k).map(|i| (i * 7 % 251) as u8).collect();
        let scale_rows = n / FP8_BLOCK;
        let scale_cols = k / FP8_BLOCK;
        let scale: Vec<u8> = (0..scale_rows * scale_cols * 4)
            .map(|i| (i * 13 % 241) as u8)
            .collect();
        let proj = ProjWeight::upload(
            &ctx,
            &Glm52ProjBytes {
                weight: &weight,
                scale: &scale,
                n,
                k,
            },
        )?;

        // Rows [128, 384): weight rows + scale row-blocks 1..3 travel along.
        let rows = proj.slice_rows(&ctx, FP8_BLOCK, 2 * FP8_BLOCK)?;
        assert_eq!((rows.n, rows.k), (2 * FP8_BLOCK, k));
        assert_eq!(
            ctx.stream.clone_dtoh(&rows.weight)?,
            weight[FP8_BLOCK * k..384 * k]
        );
        assert_eq!(
            ctx.stream.clone_dtoh(&rows.scale)?,
            scale[scale_cols * 4..3 * scale_cols * 4]
        );

        // Columns [128, 256): strided gather of every row's tail half.
        let cols = proj.slice_cols(&ctx, FP8_BLOCK, FP8_BLOCK)?;
        assert_eq!((cols.n, cols.k), (n, FP8_BLOCK));
        let mut w_ref = Vec::with_capacity(n * FP8_BLOCK);
        let mut s_ref = Vec::with_capacity(scale_rows * 4);
        for row in 0..n {
            w_ref.extend_from_slice(&weight[row * k + FP8_BLOCK..(row + 1) * k]);
        }
        for row in 0..scale_rows {
            let src = (row * scale_cols + 1) * 4;
            s_ref.extend_from_slice(&scale[src..src + 4]);
        }
        assert_eq!(ctx.stream.clone_dtoh(&cols.weight)?, w_ref);
        assert_eq!(ctx.stream.clone_dtoh(&cols.scale)?, s_ref);

        // Misaligned or out-of-range slices must refuse.
        assert!(proj.slice_rows(&ctx, 64, FP8_BLOCK).is_err());
        assert!(proj.slice_rows(&ctx, 0, n + FP8_BLOCK).is_err());
        assert!(proj.slice_cols(&ctx, 64, FP8_BLOCK).is_err());
        Ok(())
    }
}
