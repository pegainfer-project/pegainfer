//! GLM5.2 MLA front half: the q_a/kv_a/q_b projections and norms from
//! `hidden[T, 6144]` into the persistent front scratch, plus the layer's
//! device-resident attention weights. The attend half (absorb, FlashMLA
//! sparse decode, o_proj) lives in `mla_decode` and consumes the front's
//! buffers.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use pegainfer_kernels::ops::extract_hidden_rows_raw_into;
use pegainfer_kernels::ops::glm52_gemv_mma_routes;
use pegainfer_kernels::ops::glm52_gemv_split_reduce_launch;
use pegainfer_kernels::ops::glm52_mla_ckv_split_launch;
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;

use crate::config::GLM52_HEADS;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_KV_A_OUT;
use crate::config::GLM52_KV_LORA_RANK;
use crate::config::GLM52_Q_LORA_RANK;
use crate::config::GLM52_QK_HEAD_DIM;
use crate::config::GLM52_QK_NOPE_HEAD_DIM;
use crate::config::GLM52_QK_ROPE_HEAD_DIM;
use crate::config::GLM52_RMS_EPS as RMS_EPS;
use crate::config::GLM52_V_HEAD_DIM;
use crate::fp8::FP8_BLOCK;
use crate::fp8::Glm52Fp8GemmScratch;
use crate::fp8::Glm52ProjBytes;
use crate::fp8::ProjWeight;
use crate::fp8::bytes_to_f32;
use crate::fp8::e4m3_to_f32;
use crate::fp8::fp8_linear_into;
use crate::fp8::fp8_linear_large_m_into;
use crate::fp8::fp8_linear_pair_into;
use crate::fp8::fp8_linear_partials_into;
use crate::fp8::pack_proj_pair;
use crate::rows::Rows;

// Local short names for the config-owned architecture constants (the module
// is dense with shape math; the values live in one place).
const HEADS: usize = GLM52_HEADS;
const HIDDEN: usize = GLM52_HIDDEN;
const Q_LORA: usize = GLM52_Q_LORA_RANK;
const QK_NOPE: usize = GLM52_QK_NOPE_HEAD_DIM; // absorbed q nope width per head
const Q_HEAD: usize = GLM52_QK_HEAD_DIM; // qk_nope(192) + qk_rope(64)
const ROPE_DIM: usize = GLM52_QK_ROPE_HEAD_DIM;
const KV_LORA: usize = GLM52_KV_LORA_RANK;
const KV_A_OUT: usize = GLM52_KV_A_OUT; // compressed_kv(512) + k_pe(64)
const V_HEAD: usize = GLM52_V_HEAD_DIM;
const KV_B_ROWS_PER_HEAD: usize = QK_NOPE + V_HEAD; // 448

/// One MLA layer's attention weights, device-resident. `heads` is the number
/// of q/v heads THIS instance carries: the full 64, or an attention-TP head
/// shard (8 of 64 per rank) — q_b/kv_b/o_proj arrive pre-sliced and every
/// head-indexed shape below follows `heads`. The FlashMLA query/latent stay
/// full-width regardless (see `Glm52MlaAttendScratch`).
pub(crate) struct Glm52MlaLayerWeights {
    q_a: ProjWeight,
    q_a_ln: DeviceVec,
    q_b: ProjWeight,
    kv_a: ProjWeight,
    pub(crate) kv_a_ln: DeviceVec,
    pub(crate) o_proj: ProjWeight,
    pub(crate) w_uk: CudaSlice<bf16>, // [heads, 192, 512]
    pub(crate) w_uv: CudaSlice<bf16>, // [heads, 256, 512]
    // Horizontal q_a|kv_a pack ([2624, 6144]): both projections read `hidden`,
    // and one batched mma launch over the packed weight costs the same as q_a
    // alone (7.3 vs 7.1 us at batch 8 on GB300 — the kv_a bytes ride free).
    // Built only where the mma table routes the packed shape (Blackwell batch
    // 8); everywhere else the separate weights above stay the launch path, so
    // the twin costs ~15.4 MiB/layer only where it earns its keep. The
    // wide-bucket GEMM route (#812) reuses the same pack: one GEMM + one
    // activation quant replace the separate q_a/kv_a pair there too.
    qa_kva: Option<ProjWeight>,
    pub(crate) heads: usize,
}

impl Glm52MlaLayerWeights {
    /// Build from raw checkpoint bytes: upload the fp8 projections + bf16
    /// layernorm gammas, and host-dequant kv_b into the bf16 absorb factors
    /// W_UK = kv_b[:, :192, :], W_UV = kv_b[:, 192:, :].
    #[cfg(test)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        q_a: &Glm52ProjBytes,
        q_a_ln: &[u8],
        q_b: &Glm52ProjBytes,
        kv_a: &Glm52ProjBytes,
        kv_a_ln: &[u8],
        kv_b: &Glm52ProjBytes,
        o_proj: &Glm52ProjBytes,
    ) -> Result<Self> {
        // Pin every projection to the MLA architecture at load time: a checkpoint
        // with the wrong shape would otherwise sail through the self-consistent
        // `upload` len check and only die deep in the forward (a GPU slice panic).
        let check = |label: &str, p: &Glm52ProjBytes, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        let heads = mla_heads(q_b.n, kv_b.n, o_proj.k)?;
        check("q_a_proj", q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", q_b, heads * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", kv_b, heads * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", o_proj, HIDDEN, heads * V_HEAD)?;
        let (w_uk, w_uv) = dequant_kv_b(ctx, kv_b)?;
        let q_a = ProjWeight::upload(ctx, q_a)?;
        let kv_a = ProjWeight::upload(ctx, kv_a)?;
        let qa_kva = pack_qa_kva(ctx, &q_a, &kv_a)?;
        Ok(Self {
            q_a,
            q_a_ln: DeviceVec::from_safetensors(ctx, q_a_ln)?,
            q_b: ProjWeight::upload(ctx, q_b)?,
            kv_a,
            kv_a_ln: DeviceVec::from_safetensors(ctx, kv_a_ln)?,
            o_proj: ProjWeight::upload(ctx, o_proj)?,
            w_uk,
            w_uv,
            qa_kva,
            heads,
        })
    }

    /// Build from already-resident weights (the production loader path). The fp8
    /// projections + layernorm gammas are moved in; `kv_b` is consumed to derive
    /// the bf16 absorb factors (its fp8 bytes are pulled back to host once for the
    /// block-scaled dequant, then dropped — it is not stored). Same architecture
    /// shape checks as `from_host`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_device(
        ctx: &DeviceContext,
        q_a: ProjWeight,
        q_a_ln: DeviceVec,
        q_b: ProjWeight,
        kv_a: ProjWeight,
        kv_a_ln: DeviceVec,
        kv_b: &ProjWeight,
        o_proj: ProjWeight,
    ) -> Result<Self> {
        let check = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        let heads = mla_heads(q_b.n, kv_b.n, o_proj.k)?;
        check("q_a_proj", &q_a, Q_LORA, HIDDEN)?;
        check("q_b_proj", &q_b, heads * Q_HEAD, Q_LORA)?;
        check("kv_a_proj_with_mqa", &kv_a, KV_A_OUT, HIDDEN)?;
        check("kv_b_proj", kv_b, heads * KV_B_ROWS_PER_HEAD, KV_LORA)?;
        check("o_proj", &o_proj, HIDDEN, heads * V_HEAD)?;
        ensure!(
            q_a_ln.len == Q_LORA && kv_a_ln.len == KV_LORA,
            "GLM5.2 MLA layernorm lengths q_a_ln {} / kv_a_ln {} != {Q_LORA}/{KV_LORA}",
            q_a_ln.len,
            kv_a_ln.len
        );
        let kv_b_weight = ctx.stream.clone_dtoh(&kv_b.weight)?;
        let kv_b_scale = ctx.stream.clone_dtoh(&kv_b.scale)?;
        let (w_uk, w_uv) = dequant_kv_b(
            ctx,
            &Glm52ProjBytes {
                weight: &kv_b_weight,
                scale: &kv_b_scale,
                n: kv_b.n,
                k: kv_b.k,
            },
        )?;
        let qa_kva = pack_qa_kva(ctx, &q_a, &kv_a)?;
        Ok(Self {
            q_a,
            q_a_ln,
            q_b,
            kv_a,
            kv_a_ln,
            o_proj,
            w_uk,
            w_uv,
            qa_kva,
            heads,
        })
    }
}

/// The horizontal q_a|kv_a pack, built only where the batched mma table
/// routes the packed shape (see the `qa_kva` field comment). The batch-8
/// query covers every fused bucket: smaller batches never have a packed
/// entry, and the runtime re-checks its own row count before taking the
/// fused path.
fn pack_qa_kva(
    ctx: &DeviceContext,
    q_a: &ProjWeight,
    kv_a: &ProjWeight,
) -> Result<Option<ProjWeight>> {
    if !glm52_gemv_mma_routes(crate::fp8::FP8_GEMV_MAX_ROWS, q_a.n + kv_a.n, q_a.k)? {
        return Ok(None);
    }
    Ok(Some(pack_proj_pair(ctx, q_a, kv_a)?))
}

/// Per-rank bytes the q_a|kv_a packed twins will allocate during rank-model
/// build (0 where the mma table has no packed route). The context-cap budget
/// subtracts this up front — the twins land after the free-VRAM probe, and an
/// unledgered ~1.2 GiB eats the post-build headroom floor. The route query
/// reads the LAUNCHING process's device arch; a mixed-arch fleet (`--glm52-ranks`
/// spans nodes) could drift from another node's packing, where the post-build
/// headroom re-probe is the backstop.
pub(crate) fn glm52_qa_kva_twin_bytes() -> Result<usize> {
    let n = Q_LORA + KV_A_OUT;
    if !glm52_gemv_mma_routes(crate::fp8::FP8_GEMV_MAX_ROWS, n, HIDDEN)? {
        return Ok(0);
    }
    let scale = n.div_ceil(FP8_BLOCK) * HIDDEN.div_ceil(FP8_BLOCK) * 4;
    Ok(crate::config::GLM52_LAYERS * (n * HIDDEN + scale))
}

/// Derive the head count carried by a (possibly attention-TP-sharded) weight
/// set from the q_b/kv_b/o_proj shapes; the three must agree and the FlashMLA
/// full width (64) is the ceiling.
fn mla_heads(q_b_n: usize, kv_b_n: usize, o_proj_k: usize) -> Result<usize> {
    ensure!(
        q_b_n.is_multiple_of(Q_HEAD) && q_b_n > 0,
        "GLM5.2 q_b_proj n {q_b_n} is not a positive multiple of {Q_HEAD}"
    );
    let heads = q_b_n / Q_HEAD;
    ensure!(
        heads <= HEADS,
        "GLM5.2 MLA heads {heads} exceeds the architecture's {HEADS}"
    );
    ensure!(
        kv_b_n == heads * KV_B_ROWS_PER_HEAD && o_proj_k == heads * V_HEAD,
        "GLM5.2 MLA head-count mismatch: q_b says {heads} heads, kv_b n {kv_b_n} (want {}), o_proj k {o_proj_k} (want {})",
        heads * KV_B_ROWS_PER_HEAD,
        heads * V_HEAD
    );
    Ok(heads)
}

/// Host-dequant kv_b (fp8 e4m3 block-scaled) into bf16 W_UK [heads,192,512]
/// (nope) and W_UV [heads,256,512] (v) absorb factors, head-major, uploaded
/// to device. The head count follows kv_b.n (full 64 or an attention-TP
/// shard whose rows were sliced upstream).
fn dequant_kv_b(
    ctx: &DeviceContext,
    kv_b: &Glm52ProjBytes,
) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
    let heads = kv_b.n / KV_B_ROWS_PER_HEAD;
    // kv_b is indexed raw below (it does not pass through ProjWeight::upload), so
    // self-defend its byte lengths here — a truncated blob must error, not panic.
    ensure!(
        kv_b.weight.len() == kv_b.n * kv_b.k,
        "GLM5.2 kv_b weight bytes {} != n*k {}",
        kv_b.weight.len(),
        kv_b.n * kv_b.k
    );
    ensure!(
        kv_b.scale.len() == kv_b.n.div_ceil(FP8_BLOCK) * kv_b.k.div_ceil(FP8_BLOCK) * 4,
        "GLM5.2 kv_b scale bytes {} unexpected for [{},{}]",
        kv_b.scale.len(),
        kv_b.n,
        kv_b.k
    );
    let scale_cols = KV_LORA / FP8_BLOCK;
    let scale = bytes_to_f32(kv_b.scale);
    let mut w_uk = vec![bf16::from_f32(0.0); heads * QK_NOPE * KV_LORA];
    let mut w_uv = vec![bf16::from_f32(0.0); heads * V_HEAD * KV_LORA];
    for h in 0..heads {
        for r in 0..KV_B_ROWS_PER_HEAD {
            let row = h * KV_B_ROWS_PER_HEAD + r;
            for j in 0..KV_LORA {
                let s = scale[(row / FP8_BLOCK) * scale_cols + j / FP8_BLOCK];
                let val = bf16::from_f32(e4m3_to_f32(kv_b.weight[row * KV_LORA + j]) * s);
                if r < QK_NOPE {
                    w_uk[(h * QK_NOPE + r) * KV_LORA + j] = val;
                } else {
                    w_uv[(h * V_HEAD + (r - QK_NOPE)) * KV_LORA + j] = val;
                }
            }
        }
    }
    let mut uk = ctx.stream.alloc_zeros::<bf16>(w_uk.len())?;
    let mut uv = ctx.stream.alloc_zeros::<bf16>(w_uv.len())?;
    ctx.stream.memcpy_htod(&w_uk, &mut uk)?;
    ctx.stream.memcpy_htod(&w_uv, &mut uv)?;
    Ok((uk, uv))
}

/// Persistent scratch for the MLA front-half projections of one step's
/// `tokens` rows: everything derivable from `hidden` alone, before the sparse
/// top-k is known. Written in place by [`glm52_mla_front_into`] every step
/// (pointer-stable for graph capture); one instance is shared across all 78
/// layers — layer N's values are dead once layer N's attend consumed them.
/// `q_resid` is exposed because the DSA indexer consumes it (`wq_b(q_resid)`)
/// to *produce* the top-k that the attend half then attends over. The row
/// count is baked here so every front kernel derives it from one place.
pub(crate) struct Glm52MlaFront {
    q_a: CudaSlice<bf16>,             // [T, 2048] pre q_a_layernorm
    pub(crate) q_resid: Rows<Q_LORA>, // [T, 2048] post q_a_layernorm
    // [T, heads, 256] (compact head shard); read by the attend half.
    pub(crate) q_full: CudaSlice<bf16>,
    pub(crate) ckv: CudaSlice<bf16>, // [T, 576] = compressed_kv | k_pe
    kv_c_raw: CudaSlice<bf16>,       // [T, 512] pre kv_a_layernorm
    pub(crate) kv_c: CudaSlice<bf16>, // [T, 512] post kv_a_layernorm
    pub(crate) k_pe: CudaSlice<bf16>, // [T, 64] pre-rope
    pub(crate) heads: usize,
    // Owned mma partial buffer for the front projections (q_a/q_b/kv_a). One
    // per scratch struct: the ctx/aux stream overlap must never share one.
    gemv_partial: CudaSlice<f32>,
    // Validation-only bf16 sink for the packed q_a|kv_a partials launch (the
    // route gate guarantees the mma path, which leaves it untouched).
    qa_kva_out: CudaSlice<bf16>, // [T, 2624]
    // Wide-bucket route (#812): q_a/q_b/kv_a run the fp8 GEMM past the GEMV
    // row ceiling. Owned per scratch, like `gemv_partial`.
    wide: Option<Glm52Fp8GemmScratch>,
}

impl Glm52MlaFront {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize, heads: usize) -> Result<Self> {
        Self::with_decode_scratch(ctx, tokens, heads, true)
    }

    pub(crate) fn new_prefill(ctx: &DeviceContext, tokens: usize, heads: usize) -> Result<Self> {
        Self::with_decode_scratch(ctx, tokens, heads, false)
    }

    fn with_decode_scratch(
        ctx: &DeviceContext,
        tokens: usize,
        heads: usize,
        decode_scratch: bool,
    ) -> Result<Self> {
        ensure!(
            (1..=HEADS).contains(&heads),
            "GLM5.2 MLA front heads {heads} out of 1..={HEADS}"
        );
        Ok(Self {
            q_a: ctx.stream.alloc_zeros::<bf16>(tokens * Q_LORA)?,
            q_resid: Rows::zeros(ctx, tokens)?,
            q_full: ctx.stream.alloc_zeros::<bf16>(tokens * heads * Q_HEAD)?,
            ckv: ctx.stream.alloc_zeros::<bf16>(tokens * KV_A_OUT)?,
            kv_c_raw: ctx.stream.alloc_zeros::<bf16>(tokens * KV_LORA)?,
            kv_c: ctx.stream.alloc_zeros::<bf16>(tokens * KV_LORA)?,
            k_pe: ctx.stream.alloc_zeros::<bf16>(tokens * ROPE_DIM)?,
            heads,
            gemv_partial: ctx.stream.alloc_zeros::<f32>(if decode_scratch {
                tokens * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW
            } else {
                1
            })?,
            qa_kva_out: ctx.stream.alloc_zeros::<bf16>(if decode_scratch {
                tokens * (Q_LORA + KV_A_OUT)
            } else {
                1
            })?,
            wide: (decode_scratch && tokens > crate::fp8::FP8_GEMV_MAX_ROWS)
                .then(|| {
                    // q_a/kv_a consume hidden (k = 6144), q_b consumes Q_LORA.
                    Glm52Fp8GemmScratch::new(ctx, tokens.next_multiple_of(4), HIDDEN)
                })
                .transpose()?,
        })
    }

    /// The step row count every front buffer was sized for.
    fn tokens(&self) -> usize {
        self.q_resid.tokens()
    }
}

pub(crate) fn glm52_mla_prefill_front_into(
    ctx: &DeviceContext,
    weights: &Glm52MlaLayerWeights,
    rows: usize,
    hidden: &CudaSlice<bf16>,
    scratch: &mut Glm52Fp8GemmScratch,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    ensure!(
        rows > 0
            && rows.is_multiple_of(4)
            && rows <= front.tokens()
            && weights.heads == front.heads,
        "GLM5.2 prefill MLA front shape mismatch"
    );
    fp8_linear_large_m_into(ctx, &weights.q_a, rows, hidden, scratch, &mut front.q_a)?;
    rms_norm_rows_into(
        ctx,
        &front.q_a,
        &weights.q_a_ln,
        RMS_EPS,
        Q_LORA,
        rows,
        front.q_resid.data_mut(),
    )?;
    fp8_linear_large_m_into(
        ctx,
        &weights.q_b,
        rows,
        front.q_resid.data(),
        scratch,
        &mut front.q_full,
    )?;
    fp8_linear_large_m_into(ctx, &weights.kv_a, rows, hidden, scratch, &mut front.ckv)?;
    glm52_mla_ckv_split_launch(ctx, rows, &front.ckv, &mut front.kv_c_raw, &mut front.k_pe)?;
    rms_norm_rows_into(
        ctx,
        &front.kv_c_raw,
        &weights.kv_a_ln,
        RMS_EPS,
        KV_LORA,
        rows,
        &mut front.kv_c,
    )
}

/// The q-phase of the MLA front: `q_a` projection + q_a_layernorm over the
/// front's `tokens` rows. Split out because `q_resid` is everything the DSA
/// indexer needs — the caller can fork the indexer onto an aux stream right
/// after this and run [`glm52_mla_front_rest_into`] concurrently.
pub(crate) fn glm52_mla_front_q_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
) -> Result<()> {
    let t = front.tokens();
    if t == 1 {
        // q_a and kv_a share `hidden`. On bs=1 one concatenated grid removes
        // a graph node while keeping both checkpoint weights/output buffers.
        fp8_linear_pair_into(
            ctx,
            &w.q_a,
            &w.kv_a,
            hidden.data(),
            &mut front.q_a,
            &mut front.ckv,
        )?;
    } else if let Some(qa_kva) = qa_kva_fused(w, t)? {
        // One batched launch over the packed weight computes q_a AND kv_a
        // (see the `qa_kva` field comment); the split-reduce de-interleaves
        // the partials into the compact q_a/ckv buffers every downstream
        // consumer expects. `front_rest` skips its kv_a launch (same
        // `qa_kva_fused` predicate).
        let ksplit = fp8_linear_partials_into(
            ctx,
            qa_kva,
            t,
            hidden.data(),
            &mut front.gemv_partial,
            &mut front.qa_kva_out,
        )?;
        ensure!(
            ksplit > 0,
            "GLM5.2 packed q_a|kv_a launch took the register tile despite the route gate"
        );
        glm52_gemv_split_reduce_launch(
            ctx,
            t,
            Q_LORA,
            KV_A_OUT,
            ksplit,
            &front.gemv_partial,
            &mut front.q_a,
            &mut front.ckv,
        )?;
    } else if let Some(wide) = front.wide.as_mut() {
        if let Some(qa_kva) = w.qa_kva.as_ref() {
            // One packed GEMM computes q_a AND kv_a from ONE quantization of
            // `hidden` — the #812 profile showed the wide step dominated by
            // skinny per-projection GEMMs plus their activation quants; the
            // pack halves both for this pair. `front_rest` skips its kv_a
            // launch (same `qa_kva_packed_wide` predicate).
            fp8_linear_large_m_into(ctx, qa_kva, t, hidden.data(), wide, &mut front.qa_kva_out)?;
            extract_hidden_rows_raw_into(
                ctx,
                &front.qa_kva_out,
                Q_LORA + KV_A_OUT,
                &mut front.q_a,
                Q_LORA,
                0,
                t,
            )?;
            extract_hidden_rows_raw_into(
                ctx,
                &front.qa_kva_out,
                Q_LORA + KV_A_OUT,
                &mut front.ckv,
                KV_A_OUT,
                Q_LORA,
                t,
            )?;
        } else {
            fp8_linear_large_m_into(ctx, &w.q_a, t, hidden.data(), wide, &mut front.q_a)?;
        }
    } else {
        fp8_linear_into(
            ctx,
            &w.q_a,
            t,
            hidden.data(),
            Some(&mut front.gemv_partial),
            &mut front.q_a,
        )?;
    }
    rms_norm_rows_into(
        ctx,
        &front.q_a,
        &w.q_a_ln,
        RMS_EPS,
        Q_LORA,
        t,
        front.q_resid.data_mut(),
    )
}

/// The packed q_a|kv_a weight, iff this row count routes to the batched mma
/// path — the ONE predicate both front halves consult (front_q takes the
/// fused launch, front_rest skips its kv_a).
fn qa_kva_fused(w: &Glm52MlaLayerWeights, t: usize) -> Result<Option<&ProjWeight>> {
    let Some(qa_kva) = w.qa_kva.as_ref() else {
        return Ok(None);
    };
    Ok(glm52_gemv_mma_routes(t, qa_kva.n, qa_kva.k)?.then_some(qa_kva))
}

/// Whether the wide-bucket route already computed kv_a through the packed
/// q_a|kv_a GEMM (#812) — front_rest's twin of [`qa_kva_fused`].
fn qa_kva_packed_wide(w: &Glm52MlaLayerWeights, front: &Glm52MlaFront) -> bool {
    w.qa_kva.is_some() && front.wide.is_some()
}

/// The remainder of the MLA front: q_b + kv_a projections and the kv_c/k_pe
/// unpacking, over the front's `tokens` rows. Independent of the indexer.
///
/// `fold_kv_pack` (the FlashInfer backend): the ckv split and kv_a RMSNorm
/// are fused into the attend-side `glm52_mla_front_pack_fp8_launch`, so this
/// half leaves `front.ckv` raw and never touches kv_c/k_pe.
pub(crate) fn glm52_mla_front_rest_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
    fold_kv_pack: bool,
) -> Result<()> {
    let t = front.tokens();
    ensure!(
        w.heads == front.heads,
        "GLM5.2 MLA front sized for {} heads but weights carry {}",
        front.heads,
        w.heads
    );
    if let Some(wide) = front.wide.as_mut() {
        fp8_linear_large_m_into(
            ctx,
            &w.q_b,
            t,
            front.q_resid.data(),
            wide,
            &mut front.q_full,
        )?;
    } else {
        fp8_linear_into(
            ctx,
            &w.q_b,
            t,
            front.q_resid.data(),
            Some(&mut front.gemv_partial),
            &mut front.q_full,
        )?; // [T, heads, 256]
    }
    if t != 1 && qa_kva_fused(w, t)?.is_none() && !qa_kva_packed_wide(w, front) {
        if let Some(wide) = front.wide.as_mut() {
            fp8_linear_large_m_into(ctx, &w.kv_a, t, hidden.data(), wide, &mut front.ckv)?;
        } else {
            fp8_linear_into(
                ctx,
                &w.kv_a,
                t,
                hidden.data(),
                Some(&mut front.gemv_partial),
                &mut front.ckv,
            )?;
        }
    }
    if fold_kv_pack {
        return Ok(());
    }
    glm52_mla_ckv_split_launch(ctx, t, &front.ckv, &mut front.kv_c_raw, &mut front.k_pe)?;
    rms_norm_rows_into(
        ctx,
        &front.kv_c_raw,
        &w.kv_a_ln,
        RMS_EPS,
        KV_LORA,
        t,
        &mut front.kv_c,
    )
}

/// MLA front half: fp8 projections + norms from `hidden[T, 6144]` into the
/// persistent front scratch.
#[cfg(test)]
pub(crate) fn glm52_mla_front_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    front: &mut Glm52MlaFront,
    fold_kv_pack: bool,
) -> Result<()> {
    glm52_mla_front_q_into(ctx, w, hidden, front)?;
    glm52_mla_front_rest_into(ctx, w, hidden, front, fold_kv_pack)
}
