//! GLM5.2 decoder-layer composition for row-batched decode: two-norm residual
//! layout around the MLA/DSA attention and the dense-or-MoE MLP. Every buffer
//! carries the step's `tokens` independent rows.
//!
//! Per-layer math (vllm `DeepseekV2DecoderLayer`, verified for `glm_moe_dsa`):
//!
//! ```text
//! residual = hidden
//! x = rms_norm(hidden, input_layernorm)
//! attn = MLA(x)                       # + DSA indexer or shared top-k
//! hidden = residual + attn
//! residual = hidden
//! x = rms_norm(hidden, post_attention_layernorm)
//! mlp = dense_mlp(x) | moe(x)
//! hidden = residual + mlp
//! ```
//!
//! Cross-layer top-k sharing (the GLM5.2 divergence from DSv3.2): only `full`
//! layers own indexer weights and compute a fresh top-k; `shared` layers reuse
//! the previous full layer's `topk_indices` verbatim. That reuse is sound
//! because the indices are global KV slots and every layer shares one block
//! table / slot mapping. The carry is threaded through `topk_carry`: a full
//! layer overwrites it, a shared layer requires it.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::add_into;
use pegainfer_kernels::ops::fused_add_rms_norm_round_into;
#[cfg(test)]
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_RMS_EPS as RMS_EPS;
use crate::dense::Glm52DenseMlpWeights;
#[cfg(test)]
use crate::dense::glm52_dense_mlp_forward_into;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::indexer::glm52_indexer_forward_into;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_mla_attend_into;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::mla_front::glm52_mla_front_q_into;
use crate::mla_front::glm52_mla_front_rest_into;
use crate::moe_ep8::Glm52MoeEp8LayerWeights;
use crate::rows::Rows;
use crate::scratch::Glm52DecodeScratch;

const HIDDEN: usize = GLM52_HIDDEN;

/// The MLP half of a decoder layer: dense (layers 0..first_k_dense_replace),
/// the EP8 rank-0 MoE (router + shared + this rank's 32 experts; the expert
/// compute itself runs through the collective driver in `moe_ep8`, so the
/// single-layer forward below rejects it). Boxed: layer weight structs are
/// built once and held in a 78-entry vec — the indirection is free, the enum
/// stays small.
pub(crate) enum Glm52LayerMlp {
    Dense(Box<Glm52DenseMlpWeights>),
    MoeEp8(Box<Glm52MoeEp8LayerWeights>),
    /// TP topology: the router is the only per-layer MLP weight here — the
    /// routed experts AND the shared expert live in the rank's slice bank
    /// (`Glm52MoeTpRank.slices`, shared folded at bank index 256).
    MoeTp(Box<crate::moe_decode::Glm52MoeRouterWeights>),
}

/// The DSA indexer role of a decoder layer (`config.indexer_types[layer]`):
/// `Full` owns indexer weights and computes a fresh top-k; `Shared` reuses the
/// previous full layer's top-k and has no indexer weights in the checkpoint.
pub(crate) enum Glm52LayerIndexer {
    Full(Box<Glm52IndexerLayerWeights>),
    Shared,
}

/// One decoder layer's weights, device-resident.
pub(crate) struct Glm52DecoderLayerWeights {
    pub(crate) input_ln: DeviceVec,     // bf16 [HIDDEN]
    pub(crate) post_attn_ln: DeviceVec, // bf16 [HIDDEN]
    pub(crate) mla: Glm52MlaLayerWeights,
    pub(crate) indexer: Glm52LayerIndexer,
    pub(crate) mlp: Glm52LayerMlp,
}

/// One decoder layer's slice offsets inside a KV slab page: the fp8_ds_mla
/// MLA slice (64 x 656 B) and, on full-indexer layers, the DeepGEMM-layout
/// index-K slice (64 x 132 B). Byte offsets address `block * page_stride`
/// within the owning [`Glm52KvSlab`]; the production offset table comes from
/// `crate::model::glm52_page_layout`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Glm52LayerCaches {
    pub(crate) mla_offset: usize,
    pub(crate) index_k_offset: Option<usize>,
}

/// A rank's KV slab: one device allocation addressed in `num_blocks` 64-token
/// pages sitting `page_stride` bytes apart, each page holding every layer's
/// cache slices for one pool block ([`Glm52LayerCaches`] carries the
/// offsets). The stride is a multiple of the 656-byte cache token row (the
/// FlashMLA TMA contract). The allocation carries one page's content of tail
/// slack past `num_blocks * page_stride`: the kernels-layer extent checks
/// are conservative (`layer_offset + num_blocks * stride`), and the slack
/// keeps the highest-offset slices addressable without a per-layer tight
/// bound.
pub(crate) struct Glm52KvSlab {
    pub(crate) slab: CudaSlice<u8>,
    pub(crate) page_stride: usize,
    pub(crate) num_blocks: usize,
}

/// Everything one decode step shares across layers: the token position, the two
/// rotary tables (MLA interleaved; indexer half-split — different conventions,
/// same `[32]` cos/sin extent), and the paging plumbing common to every layer's
/// caches.
pub(crate) struct Glm52DecodeStep<'a> {
    pub(crate) mla_cos: &'a CudaSlice<bf16>,
    pub(crate) mla_sin: &'a CudaSlice<bf16>,
    pub(crate) idx_cos: &'a CudaSlice<bf16>,
    pub(crate) idx_sin: &'a CudaSlice<bf16>,
    /// Sparse-MLA contract plus the selected backend's scheduler state,
    /// shared by every layer.
    pub(crate) mla_sched: &'a Glm52MlaSchedMetadata,
    pub(crate) slot_mapping: &'a CudaSlice<i64>,
    pub(crate) block_table: &'a CudaSlice<i32>,
    pub(crate) seq_lens: &'a CudaSlice<i32>,
}

/// Persistent per-layer composition scratch: the residual-stream boundary
/// buffers shared by all 78 layers (layer N's values are dead once layer N's
/// closing add consumed them), sized for the step's `tokens` rows.
pub(crate) struct Glm52LayerScratch {
    /// input_layernorm output — the MLA/indexer input. Written by the
    /// PREVIOUS layer's closing fused add+norm (layer 0's comes from a
    /// standalone norm of the embedding).
    pub(crate) normed: Rows<GLM52_HIDDEN>,
    /// attention outputs, ping-ponged by layer parity: after layer L's
    /// closing fused add, `attn[L % 2]` carries the residual stream INTO
    /// layer L+1 while layer L+1's attention writes `attn[(L + 1) % 2]`.
    pub(crate) attn: [Rows<GLM52_HIDDEN>; 2],
    /// post_attention_layernorm output — the MLP input.
    pub(crate) normed2: Rows<GLM52_HIDDEN>,
    /// the MLP half's final contribution (dense out, or routed+shared sum).
    pub(crate) mlp_out: Rows<GLM52_HIDDEN>,
    /// MoE shared-expert contribution.
    pub(crate) shared_out: Rows<GLM52_HIDDEN>,
}

impl Glm52LayerScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        Ok(Self {
            normed: Rows::zeros(ctx, tokens)?,
            attn: [Rows::zeros(ctx, tokens)?, Rows::zeros(ctx, tokens)?],
            normed2: Rows::zeros(ctx, tokens)?,
            mlp_out: Rows::zeros(ctx, tokens)?,
            shared_out: Rows::zeros(ctx, tokens)?,
        })
    }
}

/// The attention half of one decoder layer for one token: input norm → MLA
/// front → DSA indexer (or shared top-k carry) → MLA attend → fused
/// add+post-attention-norm. The residual-stream input is `s.hidden`; the
/// results land in `s.layer.attn` (the carried residual) and `s.layer.normed2`
/// (the MLP input).
///
/// The cross-layer top-k carry lives in `s.idx.global_slots`: a `Full` layer
/// overwrites it, a `Shared` layer reuses it. `carry_ready` guards the read —
/// callers must pass a fresh `false` per step (layer 0 is always `Full`, so an
/// in-order full-stack walk refreshes the carry before any read, but a stale
/// buffer from a previous step must not be silently accepted if a walk started
/// at a `Shared` layer).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_layer_attention_half(
    ctx: &DeviceContext,
    aux: Option<&DeviceContext>,
    w: &Glm52DecoderLayerWeights,
    slab: &mut Glm52KvSlab,
    caches: Glm52LayerCaches,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    carry_ready: &mut bool,
    parity: usize,
    first_layer: bool,
    tp_ar: Option<&mut crate::moe_tp::Glm52MoeTpState>,
) -> Result<()> {
    // Attention-TP: a head-sharded layer (8 of 64 heads) produces an o_proj
    // PARTIAL that must cross the AR brick before the residual add; holding
    // full heads with AR wiring (or a shard without it) is a build bug —
    // crash here, not on silently-wrong hidden states.
    let sharded = w.mla.heads != crate::config::GLM52_HEADS;
    ensure!(
        sharded == tp_ar.is_some(),
        "GLM5.2 attention-TP wiring mismatch: layer holds {} heads but AR is {}",
        w.mla.heads,
        if tp_ar.is_some() { "wired" } else { "absent" }
    );
    // `s.layer.normed` (this layer's input_layernorm output) is already
    // populated: by the previous layer's closing fused add+norm, or — for
    // the first layer — by the caller's standalone norm of the embedding.
    //
    // On full-indexer layers with an aux stream, the DSA indexer chain runs
    // concurrently with the rest of the MLA front: the indexer only needs
    // `normed` + `q_resid` (the q-phase), while q_b/kv_a are independent of
    // it. Same kernels either way — byte-identical; the fork/join events
    // become graph edges at capture.
    // The scratch buffers and the attend plan were all built from one row
    // count (`Glm52DecodeScratch::new` / `Glm52BucketState`), so the batch is
    // read from the plan without re-validating the buffers against it.
    let tokens = step.mla_sched.batch();
    glm52_mla_front_q_into(ctx, &w.mla, &s.layer.normed, &mut s.mla_front)?;
    let mut topk_ready = None;
    match &w.indexer {
        Glm52LayerIndexer::Full(indexer) => {
            let index_k_offset = caches
                .index_k_offset
                .context("GLM5.2 full-indexer layer has no index-K slice in the KV page")?;
            let idx_ctx = if let Some(aux) = aux {
                let q_ready = ctx.stream.record_event(None)?;
                aux.stream.wait(&q_ready)?;
                aux
            } else {
                ctx
            };
            glm52_indexer_forward_into(
                idx_ctx,
                indexer,
                &s.layer.normed,
                &s.mla_front.q_resid,
                step.idx_cos,
                step.idx_sin,
                &mut slab.slab,
                index_k_offset,
                step.slot_mapping,
                step.block_table,
                step.seq_lens,
                step.mla_sched.topk(),
                &mut s.idx,
            )?;
            if let Some(aux) = aux {
                topk_ready = Some(aux.stream.record_event(None)?);
            }
            *carry_ready = true;
        }
        Glm52LayerIndexer::Shared => {
            ensure!(
                caches.index_k_offset.is_none(),
                "GLM5.2 shared-indexer layer unexpectedly owns an index-K slice"
            );
        }
    }
    ensure!(
        *carry_ready,
        "GLM5.2 shared-indexer layer reached before any full indexer ran"
    );
    glm52_mla_front_rest_into(
        ctx,
        &w.mla,
        &s.layer.normed,
        &mut s.mla_front,
        step.mla_sched.folds_kv_pack(),
    )?;
    if let Some(topk_ready) = &topk_ready {
        // Join before the attend consumes `s.idx.global_slots`.
        ctx.stream.wait(topk_ready)?;
    }
    let (attn_lo, attn_hi) = s.layer.attn.split_at_mut(1);
    let (attn_out, attn_other) = if parity == 0 {
        (&mut attn_lo[0], &attn_hi[0])
    } else {
        (&mut attn_hi[0], &attn_lo[0])
    };
    glm52_mla_attend_into(
        ctx,
        &w.mla,
        &s.mla_front,
        step.mla_cos,
        step.mla_sin,
        &mut slab.slab,
        caches.mla_offset,
        step.slot_mapping,
        &s.idx.global_slots,
        step.seq_lens,
        step.mla_sched,
        &mut s.mla_attend,
        attn_out,
    )?;
    if let Some(state) = tp_ar {
        // Sharded: this rank's head shard produced an o_proj PARTIAL; the
        // NCCL all-reduce sums the ranks' partials in place (identical bytes
        // on every rank) before the residual add. Callers run this path
        // eagerly — the collective stays out of CUDA graph capture (#805).
        state.prefill_allreduce_in_place(ctx, tokens, attn_out.data_mut())?;
    }

    // Fused add+norm at the post-attention boundary (bit-identical to separate
    // add + rms_norm — the `_round` variant rounds the sum to bf16 before the
    // variance, exactly like the plain add would). The residual stream enters
    // in the OTHER parity's attn buffer (written by the previous layer's
    // closing fused add), or in `s.hidden` for the first layer (the embedding).
    let residual: &CudaSlice<bf16> = if first_layer {
        s.hidden.data()
    } else {
        attn_other.data()
    };
    fused_add_rms_norm_round_into(
        ctx,
        attn_out.data_mut(),
        residual,
        &w.post_attn_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed2.data_mut(),
    )?;
    Ok(())
}

/// The layer's closing residual add, FUSED with the next layer's
/// input_layernorm (bit-identical to separate add + rms_norm, same `_round`
/// kernel as the mid-layer boundary): `attn[parity] += mlp_out` becomes the
/// residual stream into layer L+1, and `s.layer.normed` becomes L+1's
/// attention input.
pub(crate) fn glm52_layer_finish_fused(
    ctx: &DeviceContext,
    s: &mut Glm52DecodeScratch,
    parity: usize,
    next_input_ln: &DeviceVec,
    tp_padded_mlp: bool,
) -> Result<()> {
    let tokens = s.layer.normed.tokens();
    let mlp_out = if tp_padded_mlp {
        &s.tp_mlp_out
    } else {
        s.layer.mlp_out.data()
    };
    fused_add_rms_norm_round_into(
        ctx,
        s.layer.attn[parity].data_mut(),
        mlp_out,
        next_input_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed.data_mut(),
    )
}

/// The LAST layer's closing residual add: `s.hidden = attn[parity] + mlp_out`
/// (the final norm consumes `s.hidden`).
pub(crate) fn glm52_layer_finish(
    ctx: &DeviceContext,
    s: &mut Glm52DecodeScratch,
    parity: usize,
    tp_padded_mlp: bool,
) -> Result<()> {
    let mlp_out = if tp_padded_mlp {
        &s.tp_mlp_out
    } else {
        s.layer.mlp_out.data()
    };
    add_into(
        ctx,
        s.layer.attn[parity].data(),
        mlp_out,
        s.hidden.tokens() * HIDDEN,
        s.hidden.data_mut(),
    )
}

/// Dense-layer oracle helper. Production drives the two halves directly;
/// collective MoE layers fail closed here.
#[cfg(test)]
pub(crate) fn glm52_decoder_layer_forward(
    ctx: &DeviceContext,
    w: &Glm52DecoderLayerWeights,
    slab: &mut Glm52KvSlab,
    caches: Glm52LayerCaches,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    carry_ready: &mut bool,
) -> Result<()> {
    // Oracle-gate walk: one layer per call, stream in `s.hidden` — standalone
    // input norm + fixed parity 0 (no cross-layer fusion in this unit).
    let tokens = s.hidden.tokens();
    rms_norm_rows_into(
        ctx,
        s.hidden.data(),
        &w.input_ln,
        RMS_EPS,
        HIDDEN,
        tokens,
        s.layer.normed.data_mut(),
    )?;
    glm52_layer_attention_half(
        ctx,
        None,
        w,
        slab,
        caches,
        step,
        s,
        carry_ready,
        0,
        true,
        None,
    )?;
    match &w.mlp {
        Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
            ctx,
            dense,
            s.layer.normed2.data(),
            &mut s.dense_mlp,
            s.layer.mlp_out.data_mut(),
        )?,
        Glm52LayerMlp::MoeEp8(_) | Glm52LayerMlp::MoeTp(_) => anyhow::bail!(
            "GLM5.2 EP/TP MoE layers require their collective drivers, not the single-layer forward"
        ),
    }
    glm52_layer_finish(ctx, s, 0, false)
}
