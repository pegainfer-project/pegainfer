//! GLM5.2 DSA indexer decode forward, row-batched: produces per-row
//! `topk_indices[T, topk]`.
//!
//! Aligned to vllm `DeepseekV32Indexer` (the production reference). The
//! indexer computes per-token similarity against an index-K cache, selects
//! sparse top-k=2048 slots, and returns global KV cache slot indices for the
//! FlashMLA sparse decode to attend over.
//!
//! Data flow (see `docs/models/glm52/indexer-forward.md` for the vllm
//! cross-reference):
//!
//! ```text
//! q_resid[2048]  (from q_a_layernorm(q_a_proj(hidden)) — produced by the MLA layer)
//!   |
//!   +-- wq_b (fp8 linear) -> q[32, 128]
//!   |     +-- layer_norm (FlashInfer, eps=1e-6, has bias) -> k[128]
//!   |     +-- RoPE (non-interleaved/half-split, q[:64], k[:64], cos/sin[32])
//!   |     +-- q per-token-group fp8 quant -> q_fp8[32*128], q_scale[32]
//!   |     +-- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5
//!   |
//! hidden[6144]
//!   +-- wk (fp8 linear) -> k_raw[128]
//!   +-- weights_proj (bf16 min-latency GEMV) -> weights[32]
//!   +-- k quant + cache write (glm52_indexer_k_quant_and_cache)
//!   |
//!   +-- DeepGEMM paged MQA logits (fuses per-head ReLU + weighting)
//!   +-- bf16→f32 cast
//!   +-- FlashInfer deterministic top-k K=2048
//!   +-- local top-k offsets -> global KV slots
//! ```

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use pegainfer_kernels::ops::GLM52_INDEXER_HEAD_DIM;
use pegainfer_kernels::ops::GLM52_INDEXER_TOPK;
use pegainfer_kernels::ops::Glm52DeepGemmMqaLogitsShape;
use pegainfer_kernels::ops::Glm52IndexerCacheInsert;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::Glm52IndexerLocalTopKToSlots;
use pegainfer_kernels::ops::Glm52IndexerTopK;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::bf16_bytes_to_f32_into;
use pegainfer_kernels::ops::gemm_strided_batched_bf16;
use pegainfer_kernels::ops::glm52_deepgemm_mqa_logits_unpaged_launch;
use pegainfer_kernels::ops::glm52_deepgemm_paged_mqa_logits_launch;
use pegainfer_kernels::ops::glm52_deepgemm_paged_mqa_metadata_launch;
use pegainfer_kernels::ops::glm52_flashinfer_topk_2048_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_launch;
use pegainfer_kernels::ops::glm52_indexer_k_gather_launch;
use pegainfer_kernels::ops::glm52_indexer_k_quant_and_cache_launch;
use pegainfer_kernels::ops::glm52_indexer_local_topk_to_slots_launch;
use pegainfer_kernels::ops::glm52_indexer_rope_launch;
use pegainfer_kernels::ops::glm52_indexer_topk_to_slots_lut_launch;
use pegainfer_kernels::ops::glm52_indexer_weights_fold_launch;
use pegainfer_kernels::ops::glm52_indexer_weights_proj_launch;
use pegainfer_kernels::ops::layer_norm_into;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_INDEX_HEADS;
use crate::config::GLM52_Q_LORA_RANK;
use crate::fp8::FP8_BLOCK;
use crate::fp8::Glm52Fp8GemmScratch;
#[cfg(test)]
use crate::fp8::Glm52ProjBytes;
use crate::fp8::ProjWeight;
#[cfg(test)]
use crate::fp8::fp8_linear;
use crate::fp8::fp8_linear_into;
use crate::fp8::fp8_linear_large_m_into;
use crate::rows::Rows;

const HIDDEN: usize = GLM52_HIDDEN;
const Q_LORA: usize = GLM52_Q_LORA_RANK;
const INDEX_HEADS: usize = GLM52_INDEX_HEADS;
const INDEX_HEAD_DIM: usize = GLM52_INDEX_HEAD_DIM;
// vllm: softmax_scale = head_dim ** -0.5 = 128 ** -0.5
const SOFTMAX_SCALE: f32 = 0.088_388_35; // 1.0 / 128.0f32.sqrt()
// vllm: n_heads ** -0.5 = 32 ** -0.5
const N_HEADS_SCALE: f32 = 0.176_776_7; // 1.0 / 32.0f32.sqrt()
const K_NORM_EPS: f32 = 1.0e-6;

/// One DSA indexer layer's weights, device-resident.
pub(crate) struct Glm52IndexerLayerWeights {
    wq_b: ProjWeight,              // [32*128, 2048]
    wk: ProjWeight,                // [128, 6144]
    weights_proj: CudaSlice<bf16>, // [32, 6144] — bf16 GEMV (transformers _keep_in_fp32_modules)
    k_norm_w: CudaSlice<f32>,      // [128] — LayerNorm gamma (f32 for FlashInfer)
    k_norm_b: CudaSlice<f32>,      // [128] — LayerNorm beta  (f32 for FlashInfer)
}

impl Glm52IndexerLayerWeights {
    /// Build from raw checkpoint bytes (the test path). Same pattern as
    /// `Glm52MlaLayerWeights::from_host`. `weights_proj` is a bf16 `[32, 6144]`
    /// tensor (transformers keeps it in fp32 via `_keep_in_fp32_modules`, but
    /// the checkpoint stores bf16).
    #[cfg(test)]
    pub(crate) fn from_host(
        ctx: &DeviceContext,
        wq_b: &Glm52ProjBytes,
        wk: &Glm52ProjBytes,
        weights_proj_bf16: &[u8],
        k_norm_w: &[u8],
        k_norm_b: &[u8],
    ) -> Result<Self> {
        let check = |label: &str, p: &Glm52ProjBytes, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 indexer {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("wq_b", wq_b, INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
        check("wk", wk, INDEX_HEAD_DIM, HIDDEN)?;
        ensure!(
            weights_proj_bf16.len() == INDEX_HEADS * HIDDEN * 2,
            "GLM5.2 indexer weights_proj bytes {} != {} (bf16 [32, 6144])",
            weights_proj_bf16.len(),
            INDEX_HEADS * HIDDEN * 2
        );
        ensure!(
            k_norm_w.len() == INDEX_HEAD_DIM * 2,
            "GLM5.2 indexer k_norm_w bytes {} != {}",
            k_norm_w.len(),
            INDEX_HEAD_DIM * 2
        );
        ensure!(
            k_norm_b.len() == INDEX_HEAD_DIM * 2,
            "GLM5.2 indexer k_norm_b bytes {} != {}",
            k_norm_b.len(),
            INDEX_HEAD_DIM * 2
        );

        let w = ProjWeight::upload(ctx, wq_b)?;
        let k = ProjWeight::upload(ctx, wk)?;
        let proj_bf16: &[bf16] = unsafe {
            std::slice::from_raw_parts(
                weights_proj_bf16.as_ptr().cast::<bf16>(),
                INDEX_HEADS * HIDDEN,
            )
        };
        let mut weights_proj = ctx.stream.alloc_zeros::<bf16>(INDEX_HEADS * HIDDEN)?;
        ctx.stream.memcpy_htod(proj_bf16, &mut weights_proj)?;
        let norm_w = upcast_bf16_to_f32(ctx, k_norm_w)?;
        let norm_b = upcast_bf16_to_f32(ctx, k_norm_b)?;
        Ok(Self {
            wq_b: w,
            wk: k,
            weights_proj,
            k_norm_w: norm_w,
            k_norm_b: norm_b,
        })
    }

    /// Build from already-resident weights (the production loader path). The
    /// fp8 projections and the bf16 `weights_proj` are moved in; the two
    /// 128-element k_norm tensors come as host bytes because the checkpoint
    /// stores bf16 and FlashInfer LayerNorm needs f32 gamma/beta.
    pub(crate) fn from_device(
        ctx: &DeviceContext,
        wq_b: ProjWeight,
        wk: ProjWeight,
        weights_proj: CudaSlice<bf16>,
        k_norm_w: &[u8],
        k_norm_b: &[u8],
    ) -> Result<Self> {
        let check = |label: &str, p: &ProjWeight, n: usize, k: usize| -> Result<()> {
            ensure!(
                p.n == n && p.k == k,
                "GLM5.2 indexer {label} shape [{},{}] != [{n},{k}]",
                p.n,
                p.k
            );
            Ok(())
        };
        check("wq_b", &wq_b, INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
        check("wk", &wk, INDEX_HEAD_DIM, HIDDEN)?;
        ensure!(
            weights_proj.len() == INDEX_HEADS * HIDDEN,
            "GLM5.2 indexer weights_proj len {} != {} (bf16 [32, 6144])",
            weights_proj.len(),
            INDEX_HEADS * HIDDEN
        );
        Ok(Self {
            wq_b,
            wk,
            weights_proj,
            k_norm_w: upcast_bf16_to_f32(ctx, k_norm_w)?,
            k_norm_b: upcast_bf16_to_f32(ctx, k_norm_b)?,
        })
    }
}

/// Copy bf16 bytes from a checkpoint tensor and upcast to f32 on host, then
/// upload to device. Used for k_norm weight/bias (FlashInfer LayerNorm
/// requires f32 gamma/beta).
#[allow(clippy::cast_ptr_alignment)]
fn upcast_bf16_to_f32(ctx: &DeviceContext, src: &[u8]) -> Result<CudaSlice<f32>> {
    ensure!(
        src.len() == INDEX_HEAD_DIM * 2,
        "GLM5.2 indexer k_norm bytes {} != {}",
        src.len(),
        INDEX_HEAD_DIM * 2
    );
    let bf16_vals: &[bf16] =
        unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<bf16>(), INDEX_HEAD_DIM) };
    let f32_vals: Vec<f32> = bf16_vals.iter().map(|v| v.to_f32()).collect();
    let mut dst = ctx.stream.alloc_zeros::<f32>(INDEX_HEAD_DIM)?;
    ctx.stream.memcpy_htod(&f32_vals, &mut dst)?;
    Ok(dst)
}

/// Cache-fill phase: compute k for one token and write it into the index_k_cache.
/// Used during prefill to populate the cache for all positions before the
/// topk query. Does NOT compute logits or topk — only wk + LayerNorm + RoPE(k)
/// + quant + cache-write.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_indexer_cache_fill(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &CudaSlice<bf16>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    cache_layout: Glm52IndexerCacheLayout,
    slot_mapping: &CudaSlice<i64>,
) -> Result<()> {
    ensure!(
        hidden.len() >= HIDDEN,
        "GLM5.2 indexer cache_fill hidden too small"
    );

    let k_raw = fp8_linear(ctx, &w.wk, hidden)?; // [128]
    let mut k = ctx.stream.alloc_zeros::<bf16>(INDEX_HEAD_DIM)?;
    layer_norm_into(
        ctx,
        &k_raw,
        &w.k_norm_w,
        &w.k_norm_b,
        K_NORM_EPS,
        INDEX_HEAD_DIM,
        1,
        &mut k,
    )?;

    // RoPE: the kernel applies to both q and k; use a dummy q buffer.
    let mut q_dummy = ctx
        .stream
        .alloc_zeros::<bf16>(INDEX_HEADS * INDEX_HEAD_DIM)?;
    glm52_indexer_rope_launch(ctx, &mut q_dummy, &mut k, INDEX_HEADS, 1, cos, sin)?;

    glm52_indexer_k_quant_and_cache_launch(
        ctx,
        Glm52IndexerCacheInsert {
            tokens: 1,
            layout: cache_layout,
        },
        &k,
        index_k_cache,
        slot_mapping,
    )?;
    Ok(())
}

/// Persistent scratch for the DSA indexer forward: every intermediate plus
/// the DeepGEMM MQA logits shape (fixed at build — the decode batch, fixed
/// paged layout and logits stride; `shape.batch_size` is the row capacity
/// every buffer is sized for). `global_slots` doubles as the cross-layer
/// top-k carry: a full-indexer layer writes it, the following shared layers
/// read it until the next full layer overwrites it.
pub(crate) struct Glm52IndexerScratch {
    shape: Glm52DeepGemmMqaLogitsShape,
    q: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    weights_bf16: CudaSlice<bf16>,
    q_fp8: CudaSlice<u8>,
    q_scale: CudaSlice<f32>,
    weights_folded: CudaSlice<f32>,
    schedule_meta: CudaSlice<i32>,
    context_lens: CudaSlice<i32>,
    logits: CudaSlice<u8>,
    logits_f32: CudaSlice<f32>,
    topk_offsets: CudaSlice<i32>,
    topk_values: CudaSlice<f32>,
    pub(crate) global_slots: CudaSlice<i32>,
    topk_lens: CudaSlice<i32>,
    // Owned mma partial buffer (wq_b/wk). The indexer chain runs on the AUX
    // stream concurrently with the ctx-side MLA front — this buffer being
    // owned here (not shared per device) is what makes that overlap safe.
    gemv_partial: CudaSlice<f32>,
    large_gemm: Option<Glm52Fp8GemmScratch>,
}

impl Glm52IndexerScratch {
    pub(crate) fn new(ctx: &DeviceContext, shape: Glm52DeepGemmMqaLogitsShape) -> Result<Self> {
        // Wide buckets (#812) route wq_b/wk through the fp8 GEMM — the same
        // path the prefill-side indexer already takes.
        Self::with_large_gemm(ctx, shape, shape.batch_size > crate::fp8::FP8_GEMV_MAX_ROWS)
    }

    fn with_large_gemm(
        ctx: &DeviceContext,
        shape: Glm52DeepGemmMqaLogitsShape,
        large_gemm: bool,
    ) -> Result<Self> {
        let t = shape.batch_size;
        let logits_elems = t * shape.next_n * shape.logits_stride;
        Ok(Self {
            q: ctx
                .stream
                .alloc_zeros::<bf16>(t * INDEX_HEADS * INDEX_HEAD_DIM)?,
            k_raw: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEAD_DIM)?,
            k: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEAD_DIM)?,
            weights_bf16: ctx.stream.alloc_zeros::<bf16>(t * INDEX_HEADS)?,
            q_fp8: ctx
                .stream
                .alloc_zeros::<u8>(t * INDEX_HEADS * INDEX_HEAD_DIM)?,
            q_scale: ctx.stream.alloc_zeros::<f32>(t * INDEX_HEADS)?,
            weights_folded: ctx.stream.alloc_zeros::<f32>(t * INDEX_HEADS)?,
            schedule_meta: ctx
                .stream
                .alloc_zeros::<i32>(shape.schedule_metadata_len())?,
            context_lens: ctx.stream.alloc_zeros::<i32>(t)?,
            logits: ctx.stream.alloc_zeros::<u8>(logits_elems * 2)?, // bf16
            logits_f32: ctx.stream.alloc_zeros::<f32>(logits_elems)?,
            topk_offsets: ctx.stream.alloc_zeros::<i32>(t * GLM52_INDEXER_TOPK)?,
            topk_values: ctx.stream.alloc_zeros::<f32>(t * GLM52_INDEXER_TOPK)?,
            global_slots: ctx.stream.alloc_zeros::<i32>(t * GLM52_INDEXER_TOPK)?,
            topk_lens: ctx.stream.alloc_zeros::<i32>(t)?,
            gemv_partial: ctx.stream.alloc_zeros::<f32>(if large_gemm {
                1
            } else {
                t * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW
            })?,
            large_gemm: large_gemm
                .then(|| Glm52Fp8GemmScratch::new(ctx, t, HIDDEN))
                .transpose()?,
            shape,
        })
    }

    /// Rebind the MQA shape to the launch-decided pool block count. Every
    /// buffer here is sized by batch/logits stride, never by the block count
    /// — `num_kv_blocks` only feeds the forward-time cache layout — so a
    /// scratch built before the measured KV fill needs exactly this rebind.
    pub(crate) fn set_num_kv_blocks(&mut self, num_kv_blocks: usize) {
        self.shape.num_kv_blocks = num_kv_blocks;
    }

    /// Build the paged MQA shape for one query token per batch row.
    pub(crate) fn paged_mqa_shape(
        batch: usize,
        cache_layout: Glm52IndexerCacheLayout,
        block_table_stride: usize,
        num_sms: usize,
        max_model_len: usize,
    ) -> Glm52DeepGemmMqaLogitsShape {
        Glm52DeepGemmMqaLogitsShape {
            batch_size: batch,
            next_n: 1,
            num_heads: INDEX_HEADS,
            head_dim: GLM52_INDEXER_HEAD_DIM,
            num_kv_blocks: cache_layout.cache_blocks,
            block_kv: cache_layout.cache_block_size,
            kv_cache_layer_offset_bytes: cache_layout.cache_layer_offset_bytes,
            kv_cache_stride_bytes: cache_layout.cache_block_stride_bytes,
            is_context_lens_2d: false,
            is_varlen: false,
            logits_stride: max_model_len.next_multiple_of(256),
            block_table_stride,
            num_sms,
        }
    }
}

/// DSA indexer decode forward over the scratch's `shape.batch_size` rows:
/// computes each row's sparse top-k slot indices for the FlashMLA sparse
/// decode into `s.global_slots` (`[T, topk]`).
///
/// - `q_resid` is the MLA layer's q_a_layernorm output (`[T, 2048]`).
/// - `hidden` is the step's hidden states (`[T, 6144]`).
/// - `cos`/`sin` carry one indexer RoPE `[32]` row per token.
/// - `index_k_cache` is the buffer holding the paged fp8 indexer key cache
///   (mutable — each row's new k is quantized and written into it at
///   `slot_mapping[row]`), with this layer's slice starting at
///   `index_k_offset` (0 for a dedicated arena; the page-first slab passes
///   the layer's page offset). Block count and stride come from the
///   scratch's build-time shape (one source of truth — a shape/layout
///   mismatch is unrepresentable); the offset rides each call because the
///   scratch is shared by every layer.
/// - `block_table` (`[T, block_table_stride]`) / `seq_lens` (`[T]`) describe
///   each row's paged KV region for logits + slot conversion.
///
/// `s.global_slots` ends as `topk_indices[T, topk]` (i32, `-1`-padded for
/// short context). `topk` is the attend plan's index-list length (≤ 2048): a
/// short-context step selects top-`topk` instead of top-2048 — identical
/// selection whenever `seq_len <= topk` (both are "all tokens"), which is
/// exactly the regime the caller's graph tiering guarantees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_indexer_forward_into(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &Rows<HIDDEN>,
    q_resid: &Rows<Q_LORA>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    index_k_offset: usize,
    slot_mapping: &CudaSlice<i64>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    topk: usize,
    s: &mut Glm52IndexerScratch,
) -> Result<()> {
    let mut shape = s.shape;
    shape.kv_cache_layer_offset_bytes = index_k_offset;
    let t = shape.batch_size;
    // The paged cache layout the scratch shape was built from
    // (`Glm52IndexerScratch::paged_mqa_shape` copies block count/size/stride
    // in), with this call's layer offset applied.
    let cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: shape.num_kv_blocks,
        cache_block_size: shape.block_kv,
        cache_layer_offset_bytes: index_k_offset,
        cache_block_stride_bytes: shape.kv_cache_stride_bytes,
    };
    // `topk` comes from the attend plan; its 1..=GLM52_INDEXER_TOPK range is
    // pinned at compile time against the attention topk (model.rs const
    // asserts).

    // ---- projections ----
    if let Some(gemm) = &mut s.large_gemm {
        fp8_linear_large_m_into(ctx, &w.wq_b, t, q_resid.data(), gemm, &mut s.q)?;
        fp8_linear_large_m_into(ctx, &w.wk, t, hidden.data(), gemm, &mut s.k_raw)?;
    } else {
        fp8_linear_into(
            ctx,
            &w.wq_b,
            t,
            q_resid.data(),
            Some(&mut s.gemv_partial),
            &mut s.q,
        )?;
        fp8_linear_into(
            ctx,
            &w.wk,
            t,
            hidden.data(),
            Some(&mut s.gemv_partial),
            &mut s.k_raw,
        )?;
    }
    // weights_proj: bf16 in/out (transformers keeps it fp32 via
    // _keep_in_fp32_modules; the checkpoint stores bf16). Min-latency GEMV,
    // [T, 32] row-major out — the layout the fold consumes.
    if s.large_gemm.is_some() {
        gemm_strided_batched_bf16(
            ctx,
            true,
            false,
            INDEX_HEADS,
            t,
            HIDDEN,
            &w.weights_proj,
            HIDDEN,
            0,
            hidden.data(),
            HIDDEN,
            0,
            &mut s.weights_bf16,
            INDEX_HEADS,
            0,
            1,
        )?;
    } else {
        glm52_indexer_weights_proj_launch(
            ctx,
            hidden.data(),
            &w.weights_proj,
            t,
            INDEX_HEADS,
            HIDDEN,
            &mut s.weights_bf16,
        )?;
    }
    // ---- k LayerNorm (eps=1e-6, with bias), one CTA per row ----
    layer_norm_into(
        ctx,
        &s.k_raw,
        &w.k_norm_w,
        &w.k_norm_b,
        K_NORM_EPS,
        INDEX_HEAD_DIM,
        t,
        &mut s.k,
    )?;

    // ---- interleave RoPE (q[:64] per head, k[:64]; per-row position) ----
    glm52_indexer_rope_launch(ctx, &mut s.q, &mut s.k, INDEX_HEADS, t, cos, sin)?;

    // ---- q per-token-group fp8 quant ----
    // q is [T, 32, 128] flattened; quant per 128-group (one group per head).
    glm52_fp8_per_token_group_quant_bf16_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: t * INDEX_HEADS,
            width: INDEX_HEAD_DIM,
            group_size: FP8_BLOCK,
        },
        &s.q,
        &mut s.q_fp8,
        &mut s.q_scale,
    )?;

    // ---- weights fold: weights * q_scale * softmax_scale * n_heads^-0.5 ----
    // On-device (bit-identical multiply order to the retired host fold): the
    // two D2H readbacks + H2D here were the only mid-step stream syncs, and a
    // captured graph cannot contain them. Pure elementwise over [T, 32].
    glm52_indexer_weights_fold_launch(
        ctx,
        &s.weights_bf16,
        &s.q_scale,
        SOFTMAX_SCALE,
        N_HEADS_SCALE,
        &mut s.weights_folded,
    )?;

    // ---- k quant + cache write ----
    glm52_indexer_k_quant_and_cache_launch(
        ctx,
        Glm52IndexerCacheInsert {
            tokens: t,
            layout: cache_layout,
        },
        &s.k,
        index_k_cache,
        slot_mapping,
    )?;

    // ---- DeepGEMM paged MQA logits ----
    // The indexer cache layout interleaves fp8 keys and f32 scales per block:
    //   [block_size * 128 fp8][block_size * 4 f32 scale] per block.
    // DeepGEMM reads both from this single buffer — the TMA descriptors
    // use kv_cache_stride_bytes to jump over the scale region between blocks,
    // and the scales pointer is computed as kv_cache + block_kv * head_dim.
    // (Matches vllm's decode-path API — no separate scales buffer needed.)
    ctx.stream
        .memcpy_dtod(&seq_lens.slice(0..t), &mut s.context_lens)?;
    glm52_deepgemm_paged_mqa_metadata_launch(
        ctx,
        shape,
        &mut s.context_lens,
        &mut s.schedule_meta,
        None,
    )?;

    // kv_cache_scales are embedded in the interleaved cache buffer — the CUDA
    // wrapper computes the scales pointer internally from kv_cache + offset.
    // No separate scales allocation needed.
    glm52_deepgemm_paged_mqa_logits_launch(
        ctx,
        shape,
        &s.q_fp8,
        index_k_cache,
        &s.weights_folded,
        &s.context_lens,
        &mut s.logits,
        block_table,
        None,
        &mut s.schedule_meta,
    )?;

    // DeepGEMM outputs bf16 logits; FlashInfer top-k expects f32.
    // The sm90 kernel already fuses per-head ReLU (fmaxf(score, 0) * weight)
    // matching transformers' F.relu(scores) — no extra ReLU needed here.
    bf16_bytes_to_f32_into(ctx, &s.logits, &mut s.logits_f32)?;

    glm52_flashinfer_topk_2048_launch(
        ctx,
        Glm52IndexerTopK {
            num_rows: t,
            top_k: topk,
            max_len: shape.logits_stride,
        },
        &s.logits_f32,
        &s.context_lens,
        &mut s.topk_offsets,
        &mut s.topk_values,
    )?;

    // ---- local top-k offsets -> global KV slots (per row) ----
    glm52_indexer_local_topk_to_slots_launch(
        ctx,
        Glm52IndexerLocalTopKToSlots {
            num_tokens: t,
            topk,
            block_size: cache_layout.cache_block_size,
            block_table_cols: shape.block_table_stride,
        },
        &s.topk_offsets,
        &s.context_lens,
        block_table,
        &mut s.global_slots,
        &mut s.topk_lens,
    )?;

    Ok(())
}

/// Allocating convenience over [`glm52_indexer_forward_into`] for the
/// oracle-gate/test paths. Returns `topk_indices[2048]` (i32, `-1`-padded).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_indexer_forward(
    ctx: &DeviceContext,
    w: &Glm52IndexerLayerWeights,
    hidden: &Rows<HIDDEN>,
    q_resid: &Rows<Q_LORA>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    index_k_cache: &mut CudaSlice<u8>,
    cache_layout: Glm52IndexerCacheLayout,
    slot_mapping: &CudaSlice<i64>,
    block_table: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    num_sms: usize,
    max_model_len: usize,
) -> Result<CudaSlice<i32>> {
    let shape = Glm52IndexerScratch::paged_mqa_shape(
        1,
        cache_layout,
        block_table.len(),
        num_sms,
        max_model_len,
    );
    let mut s = Glm52IndexerScratch::new(ctx, shape)?;
    glm52_indexer_forward_into(
        ctx,
        w,
        hidden,
        q_resid,
        cos,
        sin,
        index_k_cache,
        cache_layout.cache_layer_offset_bytes,
        slot_mapping,
        block_table,
        seq_lens,
        GLM52_INDEXER_TOPK,
        &mut s,
    )?;
    Ok(s.global_slots)
}

/// Upper bound on requests sharing one prefill chunk (sizes the
/// per-request gather tables; exceeding it fails the chunk explicitly).
const GLM52_INDEXER_PREFILL_MAX_REQUESTS: usize = 128;

/// One request's segment inside the current prefill chunk.
#[derive(Clone, Copy, Debug)]
struct Glm52IndexerPrefillSegment {
    q_start: usize,
    q_end: usize,
    kv_len: usize,
}

/// Chunk-scale DSA indexer scratch for the TP prefill path: projections run
/// once per layer at chunk M, the paged index-K cache is gathered into the
/// compact unpaged layout, and the DeepGEMM SM100 unpaged MQA logits kernel
/// (vLLM's DSv3.2 indexer prefill kernel) runs per request segment in
/// attention-tile slices. Replaces the 32-row paged-MQA sub-tiling that cost
/// ~1.76 s per 16K chunk.
pub(crate) struct Glm52IndexerPrefillScratch {
    chunk_rows: usize,
    attn_tile: usize,
    kv_cap: usize,
    logits_stride: usize,
    table_width: usize,
    // chunk-scale projection intermediates
    q: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    weights_bf16: CudaSlice<bf16>,
    q_fp8: CudaSlice<u8>,
    q_scale: CudaSlice<f32>,
    weights_folded: CudaSlice<f32>,
    // gathered compact K + slot LUT (per layer)
    k_compact: CudaSlice<u8>,
    k_scale_compact: CudaSlice<f32>,
    slot_lut: CudaSlice<i32>,
    // per-subtile logits/top-k
    logits: CudaSlice<f32>,
    topk_offsets: CudaSlice<i32>,
    topk_values: CudaSlice<f32>,
    // per-chunk plan (staged once per forward)
    ks_zero: CudaSlice<i32>,
    ke_dev: CudaSlice<i32>,
    gather_table: CudaSlice<i32>,
    gather_lens: CudaSlice<i32>,
    segments: Vec<Glm52IndexerPrefillSegment>,
    host_ke: Vec<i32>,
    host_table: Vec<i32>,
    host_lens: Vec<i32>,
}

impl Glm52IndexerPrefillScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        chunk_rows: usize,
        attn_tile: usize,
        table_width: usize,
    ) -> Result<Self> {
        ensure!(
            chunk_rows > 0 && attn_tile > 0 && table_width > 0,
            "GLM5.2 indexer prefill scratch shapes must be positive"
        );
        let chunk = chunk_rows.next_multiple_of(4);
        // The compact K buffer holds ONE request's logical context at a time
        // (the gather runs per segment), so it is sized by the per-request
        // context cap — NOT the physical pool. Requests sharing cached
        // prefix pages therefore cannot overflow it: the physical pool
        // admits shared pages once, while each segment re-gathers its own
        // logical view into the same buffer.
        let kv_cap = (table_width * 64).next_multiple_of(4);
        let logits_stride = kv_cap.next_multiple_of(256) + 256;
        let tile = attn_tile.next_multiple_of(4);
        Ok(Self {
            chunk_rows: chunk,
            attn_tile,
            kv_cap,
            logits_stride,
            table_width,
            q: ctx
                .stream
                .alloc_zeros::<bf16>(chunk * INDEX_HEADS * INDEX_HEAD_DIM)?,
            k_raw: ctx.stream.alloc_zeros::<bf16>(chunk * INDEX_HEAD_DIM)?,
            k: ctx.stream.alloc_zeros::<bf16>(chunk * INDEX_HEAD_DIM)?,
            weights_bf16: ctx.stream.alloc_zeros::<bf16>(chunk * INDEX_HEADS)?,
            q_fp8: ctx
                .stream
                .alloc_zeros::<u8>(chunk * INDEX_HEADS * INDEX_HEAD_DIM)?,
            q_scale: ctx.stream.alloc_zeros::<f32>(chunk * INDEX_HEADS)?,
            weights_folded: ctx.stream.alloc_zeros::<f32>(chunk * INDEX_HEADS)?,
            k_compact: ctx.stream.alloc_zeros::<u8>(kv_cap * INDEX_HEAD_DIM)?,
            k_scale_compact: ctx.stream.alloc_zeros::<f32>(kv_cap.next_multiple_of(4))?,
            slot_lut: ctx.stream.alloc_zeros::<i32>(kv_cap)?,
            logits: ctx.stream.alloc_zeros::<f32>(tile * logits_stride)?,
            topk_offsets: ctx.stream.alloc_zeros::<i32>(tile * GLM52_INDEXER_TOPK)?,
            topk_values: ctx.stream.alloc_zeros::<f32>(tile * GLM52_INDEXER_TOPK)?,
            ks_zero: ctx.stream.alloc_zeros::<i32>(chunk)?,
            ke_dev: ctx.stream.alloc_zeros::<i32>(chunk)?,
            gather_table: ctx
                .stream
                .alloc_zeros::<i32>(GLM52_INDEXER_PREFILL_MAX_REQUESTS * table_width)?,
            gather_lens: ctx
                .stream
                .alloc_zeros::<i32>(GLM52_INDEXER_PREFILL_MAX_REQUESTS)?,
            segments: Vec::new(),
            host_ke: vec![0; chunk],
            host_table: vec![0; GLM52_INDEXER_PREFILL_MAX_REQUESTS * table_width],
            host_lens: vec![0; GLM52_INDEXER_PREFILL_MAX_REQUESTS],
        })
    }

    /// Stage the per-chunk request plan: segment ranges, per-query kv ends,
    /// per-query LUT bases, and the per-request gather tables. Called once
    /// per prefill batch, before the layer loop.
    pub(crate) fn stage_chunk(
        &mut self,
        ctx: &DeviceContext,
        batch: &crate::runner::Glm52PrefillBatch,
    ) -> Result<()> {
        let rows = batch.token_ids.len();
        ensure!(
            rows > 0 && rows <= self.chunk_rows,
            "GLM5.2 indexer prefill chunk of {rows} rows exceeds capacity {}",
            self.chunk_rows
        );
        self.segments.clear();
        let width = self.table_width;
        let mut request = 0usize;
        let mut row = 0usize;
        while row < rows {
            while row >= batch.request_indptr[request + 1] as usize {
                request += 1;
            }
            let q_start = row;
            let q_end = (batch.request_indptr[request + 1] as usize).min(rows);
            let kv_len = batch.positions[q_end - 1] as usize + 1;
            let seg_index = self.segments.len();
            ensure!(
                seg_index < GLM52_INDEXER_PREFILL_MAX_REQUESTS,
                "GLM5.2 prefill chunk spans more than {GLM52_INDEXER_PREFILL_MAX_REQUESTS} requests"
            );
            ensure!(
                kv_len <= self.kv_cap,
                "GLM5.2 prefill request context {kv_len} exceeds the compact index-K cap {}",
                self.kv_cap
            );
            let block_start = batch.block_indptr[request] as usize;
            let block_end = batch.block_indptr[request + 1] as usize;
            let blocks = &batch.block_ids[block_start..block_end];
            ensure!(
                blocks.len() <= width && blocks.len() * 64 >= kv_len,
                "GLM5.2 prefill request block table does not cover its context"
            );
            self.host_table[seg_index * width..seg_index * width + blocks.len()]
                .copy_from_slice(blocks);
            self.host_lens[seg_index] = kv_len as i32;
            for r in q_start..q_end {
                self.host_ke[r] = batch.positions[r] as i32 + 1;
            }
            self.segments.push(Glm52IndexerPrefillSegment {
                q_start,
                q_end,
                kv_len,
            });
            row = q_end;
        }
        let segs = self.segments.len();
        ctx.stream
            .memcpy_htod(&self.host_ke[..rows], &mut self.ke_dev.slice_mut(..rows))?;
        ctx.stream.memcpy_htod(
            &self.host_table[..segs * width],
            &mut self.gather_table.slice_mut(..segs * width),
        )?;
        ctx.stream.memcpy_htod(
            &self.host_lens[..segs],
            &mut self.gather_lens.slice_mut(..segs),
        )?;
        Ok(())
    }

    /// One full-indexer layer over the whole staged chunk: chunk-M
    /// projections, K quant + cache write, paged->compact K gather, then per
    /// request segment the unpaged MQA logits + top-k + LUT slot conversion,
    /// writing straight into the executor's chunk-scale carry.
    /// `cache_layout` rides each call because one scratch serves caches with
    /// different geometry: the main layers' slab slices (per-layer offset,
    /// page stride) and the TP4 MTP proposal cache (dense index-K region).
    #[allow(clippy::too_many_arguments)]
    /// Re-commit the chunk's already-quantizable K rows (the `self.k` buffer
    /// `run_layer` just produced) into a second cache under its own layout.
    /// Writes exactly `rows` slots — a restored page whose rows are not in
    /// this chunk is never touched, which is what makes this safe to aim at
    /// a cache holding host-restored content.
    pub(crate) fn commit_k_rows(
        &self,
        ctx: &DeviceContext,
        index_k_cache: &mut CudaSlice<u8>,
        cache_layout: Glm52IndexerCacheLayout,
        slot_mapping: &CudaSlice<i64>,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.chunk_rows,
            "GLM5.2 indexer K mirror commit before stage_chunk"
        );
        glm52_indexer_k_quant_and_cache_launch(
            ctx,
            Glm52IndexerCacheInsert {
                tokens: rows,
                layout: cache_layout,
            },
            &self.k,
            index_k_cache,
            slot_mapping,
        )
    }

    pub(crate) fn run_layer(
        &mut self,
        ctx: &DeviceContext,
        w: &Glm52IndexerLayerWeights,
        hidden: &CudaSlice<bf16>,
        q_resid: &CudaSlice<bf16>,
        cos: &CudaSlice<bf16>,
        sin: &CudaSlice<bf16>,
        index_k_cache: &mut CudaSlice<u8>,
        cache_layout: Glm52IndexerCacheLayout,
        slot_mapping: &CudaSlice<i64>,
        rows: usize,
        gemm: &mut Glm52Fp8GemmScratch,
        carry_slots: &mut CudaSlice<i32>,
        carry_lens: &mut CudaSlice<i32>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.chunk_rows && !self.segments.is_empty(),
            "GLM5.2 indexer prefill layer before stage_chunk"
        );
        let rows4 = rows.next_multiple_of(4);
        fp8_linear_large_m_into(ctx, &w.wq_b, rows4, q_resid, gemm, &mut self.q)?;
        fp8_linear_large_m_into(ctx, &w.wk, rows4, hidden, gemm, &mut self.k_raw)?;
        gemm_strided_batched_bf16(
            ctx,
            true,
            false,
            INDEX_HEADS,
            rows,
            HIDDEN,
            &w.weights_proj,
            HIDDEN,
            0,
            hidden,
            HIDDEN,
            0,
            &mut self.weights_bf16,
            INDEX_HEADS,
            0,
            1,
        )?;
        layer_norm_into(
            ctx,
            &self.k_raw,
            &w.k_norm_w,
            &w.k_norm_b,
            K_NORM_EPS,
            INDEX_HEAD_DIM,
            rows,
            &mut self.k,
        )?;
        glm52_indexer_rope_launch(ctx, &mut self.q, &mut self.k, INDEX_HEADS, rows, cos, sin)?;
        glm52_fp8_per_token_group_quant_bf16_launch(
            ctx,
            Glm52MoeQuantShape {
                rows: rows * INDEX_HEADS,
                width: INDEX_HEAD_DIM,
                group_size: FP8_BLOCK,
            },
            &self.q,
            &mut self.q_fp8,
            &mut self.q_scale,
        )?;
        glm52_indexer_weights_fold_launch(
            ctx,
            &self.weights_bf16,
            &self.q_scale,
            SOFTMAX_SCALE,
            N_HEADS_SCALE,
            &mut self.weights_folded,
        )?;
        glm52_indexer_k_quant_and_cache_launch(
            ctx,
            Glm52IndexerCacheInsert {
                tokens: rows,
                layout: cache_layout,
            },
            &self.k,
            index_k_cache,
            slot_mapping,
        )?;
        let segments = self.segments.clone();
        for (seg_index, seg) in segments.iter().enumerate() {
            // Per-segment gather into the shared compact buffer (base 0):
            // requests sharing cached prefix pages each re-gather their own
            // logical context, so the buffer is bounded by one request's cap
            // regardless of how the physical pool deduplicates pages.
            glm52_indexer_k_gather_launch(
                ctx,
                1,
                self.table_width,
                cache_layout.cache_block_size,
                cache_layout.cache_layer_offset_bytes,
                cache_layout.cache_block_stride_bytes,
                index_k_cache,
                &self.gather_table.slice(seg_index * self.table_width..),
                &self.gather_lens.slice(seg_index..),
                &self.ks_zero.slice(..1),
                &mut self.k_compact,
                &mut self.k_scale_compact,
                &mut self.slot_lut,
            )?;
            let mut sub = seg.q_start;
            while sub < seg.q_end {
                let t = (seg.q_end - sub).min(self.attn_tile);
                glm52_deepgemm_mqa_logits_unpaged_launch(
                    ctx,
                    t,
                    seg.kv_len,
                    self.logits_stride,
                    &self.q_fp8.slice(sub * INDEX_HEADS * INDEX_HEAD_DIM..),
                    &self.k_compact,
                    &self.k_scale_compact,
                    &self.weights_folded.slice(sub * INDEX_HEADS..),
                    &self.ks_zero.slice(..t),
                    &self.ke_dev.slice(sub..sub + t),
                    &mut self.logits,
                )?;
                glm52_flashinfer_topk_2048_launch(
                    ctx,
                    Glm52IndexerTopK {
                        num_rows: t,
                        top_k: GLM52_INDEXER_TOPK,
                        max_len: self.logits_stride,
                    },
                    &self.logits,
                    &self.ke_dev.slice(sub..sub + t),
                    &mut self.topk_offsets,
                    &mut self.topk_values,
                )?;
                let mut slots_out = carry_slots.slice_mut(sub * GLM52_INDEXER_TOPK..);
                let mut lens_out = carry_lens.slice_mut(sub..);
                glm52_indexer_topk_to_slots_lut_launch(
                    ctx,
                    t,
                    GLM52_INDEXER_TOPK,
                    &self.topk_offsets,
                    &self.ke_dev.slice(sub..sub + t),
                    &self.ks_zero.slice(..t),
                    &self.slot_lut,
                    &mut slots_out,
                    &mut lens_out,
                )?;
                sub += t;
            }
        }
        Ok(())
    }
}
