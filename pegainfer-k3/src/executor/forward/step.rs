//! One step of the forward pass: the certified engine's launch sequence,
//! batched. Decode and a prefill chunk run the SAME trunk below; the arms
//! diverge only at the leaves ([`decode`](super::decode) walks tokens through
//! the KDA recurrence and absorbed paged MLA, [`prefill`](super::prefill)
//! runs chunkwise FlashKDA and dense-FMHA MLA) and at the epilogue, which a
//! chunk skips.
//!
//! The order below is the reference `decode_step` line for line. Two families
//! of call replace a reference spelling rather than reproducing it, and both
//! are deliberate:
//!
//! * **dense projections.** The reference merges an eight-way split-K partial
//!   with `land`; here cuBLASLt produces the whole product as a single f32
//!   partial and the same `k3_land_batched` merges it with `split_k = 1`, which
//!   is the identical bf16 landing over one segment. The landing spans are the
//!   reference's, unchanged.
//! * **routed experts.** The reference's per-row `packed_expert_gemv` becomes
//!   one fused MegaMoE launch ([`routed_experts_mega`]), which dispatches,
//!   runs both FP8xFP4 GEMMs, applies situ, re-quantizes and combines — across
//!   the whole expert-parallel world when there is one. [`routed_experts`] is
//!   the masked grouped-GEMM chain it replaced, kept as the numerics anchor and
//!   reachable only from test configurations.
//!
//! Everything else — norms, landings, the convolution, the delta rule, MLA
//! attention, the router, situ, the attention-residual mix — is the certified
//! batched kernel, called with the reference's operands.
//!
//! No call here reads device memory back to the host or varies its launch
//! geometry with device state, so the whole sequence is capturable.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3_MLA_HEADS;
use pegainfer_kernels::ops::K3_MOE_QUANT_GROUP;
use pegainfer_kernels::ops::K3_ROUTER_TOPK;
use pegainfer_kernels::ops::K3DeepGemmFp8Fp4Kind;
use pegainfer_kernels::ops::K3MegaActivation;
use pegainfer_kernels::ops::K3MegaShape;
use pegainfer_kernels::ops::K3MoeRouteShape;
use pegainfer_kernels::ops::argmax_bf16_split_into;
use pegainfer_kernels::ops::copy_hidden_rows_raw_into;
use pegainfer_kernels::ops::embedding_rows_into;
use pegainfer_kernels::ops::extract_hidden_rows_raw_into;
use pegainfer_kernels::ops::k3_add2_batched_launch;
use pegainfer_kernels::ops::k3_attnres_mix_batched_launch;
use pegainfer_kernels::ops::k3_attnres_scores_batched_launch;
use pegainfer_kernels::ops::k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch;
use pegainfer_kernels::ops::k3_fp8_scale_pack_ue8m0_launch;
use pegainfer_kernels::ops::k3_land_batched_launch;
use pegainfer_kernels::ops::k3_land_rms_norm_rbs_batched_launch;
use pegainfer_kernels::ops::k3_mega_moe_launch;
use pegainfer_kernels::ops::k3_mega_write_inputs_launch;
use pegainfer_kernels::ops::k3_mla_paged_attn_launch;
use pegainfer_kernels::ops::k3_moe_gather_fp8_quant_masked_launch;
use pegainfer_kernels::ops::k3_moe_local_route_metadata_launch;
use pegainfer_kernels::ops::k3_moe_weighted_combine_launch;
use pegainfer_kernels::ops::k3_mul_sigmoid_batched_launch;
use pegainfer_kernels::ops::k3_rms_norm_rbs_batched_launch;
use pegainfer_kernels::ops::k3_router_topk_batched_launch;
use pegainfer_kernels::ops::k3_situ_and_mul_fp8_quant_masked_launch;
use pegainfer_kernels::ops::k3_situ_batched_launch;
use pegainfer_kernels::tensor::DeviceContext;

use super::super::buffers::K3_MLA_FUSED;
use super::super::buffers::K3LayerState;
use super::super::buffers::K3Scratch;
use super::super::buffers::K3StatePool;
use super::super::buffers::parity_pair;
use super::super::cp::K3CpScratch;
use super::super::paged_kv::K3_KV_PAGE_TOKENS;
use super::super::paged_kv::K3_MLA_LATENT_ROW;
use super::super::paged_kv::K3PagedKv;
use super::decode::kda_attention;
use super::gemm::k3_gemm_full;
use super::prefill::kda_attention_chunk;
use super::prefill::mla_attention_chunk_cp;
use super::prefill::mla_attention_chunk_fmha;
use crate::config::K3_ATTN_INNER;
use crate::config::K3_DENSE_INTERMEDIATE;
use crate::config::K3_EXPERT_INTERMEDIATE;
use crate::config::K3_HIDDEN;
use crate::config::K3_KV_A_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_Q_B_OUT;
use crate::config::K3_Q_LORA_RANK;
use crate::config::K3_QK_ROPE_HEAD_DIM;
use crate::config::K3_ROUTED_EXPERT_HIDDEN;
use crate::config::K3_SHARED_INTERMEDIATE;
use crate::config::K3_VOCAB;
use crate::model::K3ExpertBankForm;
use crate::model::K3LayerAttention;
use crate::model::K3LayerMlp;
use crate::model::K3LayerWeights;
use crate::model::K3MlaWeights;
use crate::model::K3MoeWeights;
use crate::model::K3RankModel;

/// Everything a step needs that is neither weights, state nor scratch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct K3StepShape {
    /// Compiled batch bucket every kernel runs at.
    pub(crate) bucket: usize,
    /// Leading rows of the bucket this rank actually owns this step; the rest
    /// are padding. The expert-parallel MegaMoE path reads it (see
    /// [`routed_experts_mega`]), and a prefill chunk reads it as its live
    /// token count.
    pub(crate) live_rows: usize,
    /// Which half of each ping-pong state slab this step reads.
    pub(crate) parity: usize,
    /// Prefill chunks only: tokens of the sequence already cached before this
    /// chunk (the MLA context span). Zero for decode steps.
    pub(crate) chunk_start: usize,
    /// Rank-local expert groups (the masked GEMM's instantiation).
    pub(crate) groups: usize,
    /// Rows reserved per expert in the masked layout.
    pub(crate) masked_cap: usize,
    /// Multiprocessor count the masked GEMM was instantiated for.
    pub(crate) num_sms: usize,
    /// Run the routed experts through the fused MegaMoE kernel instead of the
    /// masked chain.
    pub(crate) mega: bool,
}

impl K3StepShape {
    fn route(self) -> K3MoeRouteShape {
        K3MoeRouteShape {
            tokens: self.bucket,
            topk: K3_ROUTER_TOPK,
            groups: self.groups,
            masked_cap: self.masked_cap,
        }
    }
}

/// One slot's contiguous row segment of a verify step, and how its KDA state
/// advances across it.
///
/// A verify step packs several slots into one bucket: each contributes its
/// deferred-commit replay (the tokens the last round accepted, whose KDA
/// state advance was deferred) followed by its speculative span (the anchor
/// and the drafts under verification). The commit rows move the slot's
/// recurrent/conv state from its parity slab into the other one; the spec
/// rows continue from that committed state but their successor state is
/// discarded — acceptance is not known until the host reads the argmaxes
/// back, so the accepted prefix replays as the NEXT round's commit rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct K3KdaGroup {
    /// First batch row of the segment.
    pub(crate) row: usize,
    /// Leading replay rows whose state advance commits (parity flips).
    pub(crate) commit_rows: usize,
    /// Trailing speculative rows whose successor state is discarded.
    pub(crate) spec_rows: usize,
    /// The slot's row in the state pool's KDA/conv slabs.
    pub(crate) state_row: usize,
    /// Which parity slab holds the slot's committed state.
    pub(crate) parity: usize,
}

/// Which of the three step families this launch sequence runs. The trunk is
/// identical; the arms diverge at the KDA/MLA leaves and the epilogue.
#[derive(Clone, Copy)]
pub(crate) enum K3StepMode<'a> {
    /// One independent sequence per row: fused per-token KDA core, absorbed
    /// paged MLA, full epilogue.
    Decode,
    /// Rows are consecutive tokens of ONE sequence: chunkwise KDA, dense-FMHA
    /// MLA, no epilogue (the boundary sample runs separately).
    PrefillChunk,
    /// Speculative verify: rows are packed per-slot segments. Chunkwise KDA
    /// per group with deferred commit, absorbed paged MLA over the packed
    /// verify table, full epilogue (the argmaxes decide acceptance).
    Verify(&'a [K3KdaGroup]),
}

/// Where a step deposits the aux hidden states the DSpark draft lane feeds
/// on: for each tap layer `t`, rows `0..rows` of the pre-norm snapshot
/// mixture read at the top of layer `t + 1` (what SGLang's
/// `_dspark_capture_stream` distilled — with the attn-res bank, K3's "stream
/// value after layer t" is the mixture its next consumer computes, not the
/// raw residual) are copied into their column segment of `slab`
/// (`[capacity, taps.len() * hidden]`, tap order = column order). Pure extra
/// dtod traffic — nothing in the step reads it back.
pub(crate) struct K3AuxSink<'a> {
    pub(crate) slab: &'a mut CudaSlice<bf16>,
    /// Leading step rows to capture (a chunk's live tokens, a verify step's
    /// packed rows).
    pub(crate) rows: usize,
    /// 0-based tap layer indices; tap `t` is captured at the top of `t + 1`.
    pub(crate) taps: &'a [usize],
}

#[allow(clippy::too_many_arguments)]
pub(super) fn k3_step(
    ctx: &DeviceContext,
    model: &K3RankModel,
    shape: K3StepShape,
    mode: K3StepMode<'_>,
    state: &mut K3StatePool,
    scratch: &mut K3Scratch,
    mut aux: Option<K3AuxSink<'_>>,
    mut cp: Option<&mut K3CpScratch>,
) -> Result<()> {
    let b = shape.bucket;
    let K3StatePool {
        layers: layer_state,
        kv,
        blocks: snapshots,
        block_count,
        ..
    } = state;
    let block_count = *block_count;
    // Index of the current MLA layer within the paged pool's layer slices.
    let mut mla_index = 0usize;

    embedding_rows_into(
        ctx,
        &model.embed,
        &scratch.token_ids,
        b,
        &mut scratch.hidden,
    )?;

    for (index, (layer, layer_state)) in model.layers.iter().zip(layer_state.iter_mut()).enumerate()
    {
        let geometry = layer.geometry;
        copy_rows(ctx, &scratch.hidden, &mut scratch.prefix, b * K3_HIDDEN)?;
        if geometry.nb_in > 0 {
            attn_res(
                ctx,
                b,
                geometry.nb_in,
                &scratch.prefix,
                snapshots,
                &layer.sw_attn,
                &mut scratch.scores,
                &mut scratch.mixed,
            )?;
        } else {
            copy_rows(ctx, &scratch.prefix, &mut scratch.mixed, b * K3_HIDDEN)?;
        }
        // DSpark aux capture. The checkpoint was distilled from SGLang's
        // capture points, and with K3's attn-res bank those are NOT the raw
        // residual stream: tap layer `t` captures the pre-norm snapshot
        // mixture layer `t+1`'s attention consumes (`_dspark_capture_stream`
        // / `aggregate_stream` upstream) — exactly `scratch.mixed` here.
        if index > 0
            && let Some(sink) = aux.as_mut()
            && let Some(tap) = sink.taps.iter().position(|&t| t + 1 == index)
        {
            copy_hidden_rows_raw_into(
                ctx,
                &scratch.mixed,
                K3_HIDDEN,
                sink.slab,
                sink.taps.len() * K3_HIDDEN,
                tap * K3_HIDDEN,
                sink.rows,
            )?;
        }
        if geometry.snapshot {
            // `blk[:, nb_in, :] = ps` — the snapshot the later mixes see.
            copy_hidden_rows_raw_into(
                ctx,
                &scratch.prefix,
                K3_HIDDEN,
                snapshots,
                block_count * K3_HIDDEN,
                geometry.nb_in * K3_HIDDEN,
                b,
            )?;
        }

        match (&layer.attn, layer_state) {
            (K3LayerAttention::Kda(kda), K3LayerState::Kda(kda_state)) => match mode {
                K3StepMode::Decode => {
                    let (recurrent_read, recurrent_write) =
                        parity_pair(&mut kda_state.recurrent, shape.parity);
                    let (conv_read, conv_write) = parity_pair(&mut kda_state.conv, shape.parity);
                    kda_attention(
                        ctx,
                        b,
                        layer,
                        kda,
                        recurrent_read,
                        recurrent_write,
                        conv_read,
                        conv_write,
                        scratch,
                    )?;
                }
                K3StepMode::PrefillChunk => {
                    // A prefill chunk is the one-group case: the whole chunk
                    // commits, from the pool's single state row.
                    let chunk = [K3KdaGroup {
                        row: 0,
                        commit_rows: shape.live_rows,
                        spec_rows: 0,
                        state_row: 0,
                        parity: shape.parity,
                    }];
                    kda_attention_chunk(
                        ctx,
                        shape,
                        layer,
                        kda,
                        kda_state,
                        &chunk,
                        scratch,
                        cp.as_deref_mut(),
                    )?;
                }
                K3StepMode::Verify(groups) => {
                    kda_attention_chunk(ctx, shape, layer, kda, kda_state, groups, scratch, None)?;
                }
            },
            (K3LayerAttention::Mla(mla), K3LayerState::Mla) => {
                mla_attention(
                    ctx,
                    shape,
                    mode,
                    layer,
                    mla,
                    kv,
                    mla_index,
                    scratch,
                    cp.as_deref_mut(),
                )?;
                mla_index += 1;
            }
            _ => anyhow::bail!("K3 layer state does not match the layer's attention kind"),
        }

        if geometry.snapshot {
            copy_rows(ctx, &scratch.attn_out, &mut scratch.prefix2, b * K3_HIDDEN)?;
        } else {
            k3_add2_batched_launch(
                ctx,
                b,
                K3_HIDDEN,
                &scratch.prefix,
                &scratch.attn_out,
                &mut scratch.prefix2,
            )?;
        }

        attn_res(
            ctx,
            b,
            geometry.nb_mlp,
            &scratch.prefix2,
            snapshots,
            &layer.sw_mlp,
            &mut scratch.scores,
            &mut scratch.mixed2,
        )?;

        match &layer.mlp {
            K3LayerMlp::Moe(moe) => moe_mlp(ctx, b, shape, layer, moe, scratch)?,
            K3LayerMlp::Dense(dense) => {
                k3_rms_norm_rbs_batched_launch(
                    ctx,
                    b,
                    K3_HIDDEN,
                    &scratch.mixed2,
                    &layer.gamma_post.data,
                    &mut scratch.normed,
                )?;
                k3_gemm_full(
                    ctx,
                    &dense.wgu,
                    &scratch.normed,
                    b,
                    &mut scratch.dense_partial,
                )?;
                let width = 2 * K3_DENSE_INTERMEDIATE;
                k3_land_batched_launch(
                    ctx,
                    b,
                    width,
                    K3_DENSE_INTERMEDIATE,
                    0,
                    1,
                    &scratch.dense_partial,
                    &mut scratch.dense_gate,
                )?;
                k3_land_batched_launch(
                    ctx,
                    b,
                    width,
                    K3_DENSE_INTERMEDIATE,
                    K3_DENSE_INTERMEDIATE,
                    1,
                    &scratch.dense_partial,
                    &mut scratch.dense_up,
                )?;
                k3_situ_batched_launch(
                    ctx,
                    b,
                    K3_DENSE_INTERMEDIATE,
                    &scratch.dense_gate,
                    &scratch.dense_up,
                    &mut scratch.dense_act,
                )?;
                k3_gemm_full(
                    ctx,
                    &dense.w_dn,
                    &scratch.dense_act,
                    b,
                    &mut scratch.hidden_partial,
                )?;
                k3_land_batched_launch(
                    ctx,
                    b,
                    K3_HIDDEN,
                    K3_HIDDEN,
                    0,
                    1,
                    &scratch.hidden_partial,
                    &mut scratch.mlp_out,
                )?;
            }
        }

        k3_add2_batched_launch(
            ctx,
            b,
            K3_HIDDEN,
            &scratch.prefix2,
            &scratch.mlp_out,
            &mut scratch.hidden,
        )?;
    }

    // A prefill chunk stops here: only the boundary token's sample is ever
    // read, so the caller runs [`super::k3_prefill_boundary_sample`] once
    // after the final chunk instead of paying a chunk-wide lm_head per step —
    // which also keeps the vocab-wide buffers sized by the decode rows, not
    // the chunk bucket.
    if matches!(mode, K3StepMode::PrefillChunk) {
        return Ok(());
    }
    attn_res(
        ctx,
        b,
        model.blocks,
        &scratch.hidden,
        snapshots,
        &model.sw_out,
        &mut scratch.scores,
        &mut scratch.mixed,
    )?;
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        &scratch.mixed,
        &model.gamma_final.data,
        &mut scratch.normed,
    )?;
    k3_gemm_full(
        ctx,
        &model.w_lm,
        &scratch.normed,
        b,
        &mut scratch.logit_partial,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_VOCAB,
        K3_VOCAB,
        0,
        1,
        &scratch.logit_partial,
        &mut scratch.logits,
    )?;
    argmax_bf16_split_into(
        ctx,
        &scratch.logits,
        b,
        K3_VOCAB,
        &mut scratch.argmax_partial_values,
        &mut scratch.argmax_partial_indices,
        &mut scratch.argmax_values,
        &mut scratch.argmax_indices,
    )
}

/// Score the `blocks + 1` attention-residual candidates and mix them.
#[allow(clippy::too_many_arguments)]
pub(super) fn attn_res(
    ctx: &DeviceContext,
    b: usize,
    blocks: usize,
    prefix: &CudaSlice<bf16>,
    snapshots: &CudaSlice<bf16>,
    scoring: &CudaSlice<f32>,
    scores: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    k3_attnres_scores_batched_launch(
        ctx, b, blocks, K3_HIDDEN, prefix, snapshots, scoring, scores,
    )?;
    k3_attnres_mix_batched_launch(ctx, b, blocks, K3_HIDDEN, prefix, snapshots, scores, out)
}

fn copy_rows(
    ctx: &DeviceContext,
    source: &CudaSlice<bf16>,
    target: &mut CudaSlice<bf16>,
    len: usize,
) -> Result<()> {
    let origin = source.slice(..len);
    let mut destination = target.slice_mut(..len);
    ctx.stream
        .memcpy_dtod(&origin, &mut destination)
        .map_err(|error| anyhow::anyhow!("K3 residual copy failed: {error}"))
}

// ── MLA ─────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn mla_attention(
    ctx: &DeviceContext,
    shape: K3StepShape,
    mode: K3StepMode<'_>,
    layer: &K3LayerWeights,
    w: &K3MlaWeights,
    kv: &mut K3PagedKv,
    mla_index: usize,
    s: &mut K3Scratch,
    cp: Option<&mut K3CpScratch>,
) -> Result<()> {
    let b = shape.bucket;
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        &s.mixed,
        &layer.gamma_in.data,
        &mut s.normed,
    )?;
    k3_gemm_full(ctx, &w.wfu, &s.normed, b, &mut s.mla_fused_partial)?;
    k3_land_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_MLA_FUSED,
        K3_Q_LORA_RANK,
        0,
        1,
        &s.mla_fused_partial,
        &w.gamma_q_a.data,
        &mut s.q_norm,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_MLA_FUSED,
        K3_KV_A_OUT,
        K3_Q_LORA_RANK,
        1,
        &s.mla_fused_partial,
        &mut s.kv_a,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_MLA_FUSED,
        K3_ATTN_INNER,
        K3_Q_LORA_RANK + K3_KV_A_OUT,
        1,
        &s.mla_fused_partial,
        &mut s.mla_gate,
    )?;
    // The kv latent and the shared rope half are the two column windows of the
    // landed `kv_a` row; a batched norm needs the latent dense on its own.
    extract_hidden_rows_raw_into(
        ctx,
        &s.kv_a,
        K3_KV_A_OUT,
        &mut s.kv_latent,
        K3_KV_LORA_RANK,
        0,
        b,
    )?;
    extract_hidden_rows_raw_into(
        ctx,
        &s.kv_a,
        K3_KV_A_OUT,
        &mut s.rope,
        K3_QK_ROPE_HEAD_DIM,
        K3_KV_LORA_RANK,
        b,
    )?;
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_KV_LORA_RANK,
        &s.kv_latent,
        &w.gamma_kv_a.data,
        &mut s.kv_norm,
    )?;
    // Paged latent append: the post-norm kv latent and the shared rope half
    // are the whole cached quantity (NoPE — nothing here is
    // position-dependent), written into this layer's slice of the row's
    // current page. The expanded K/V the reference builds from it is folded
    // into the absorbed attention below.
    kv.append_latent(ctx, mla_index, b, &s.kv_row, &s.kv_norm, &s.rope)?;
    k3_gemm_full(ctx, &w.w_q_b, &s.q_norm, b, &mut s.q_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_Q_B_OUT,
        K3_Q_B_OUT,
        0,
        1,
        &s.q_partial,
        &mut s.query,
    )?;

    if matches!(mode, K3StepMode::PrefillChunk) {
        // Chunked prefill takes the non-absorbed dense route (vLLM's recipe):
        // gather the cached latent — the chunk's own rows were appended just
        // above, so the cache holds the whole [context | chunk] span — expand
        // it through kv_b into per-head K/V scratch, and one bottom-right-
        // aligned causal FMHA serves every chunk query. Only the paged latent
        // persists. Padding rows of `attn` keep stale (finite) data; their
        // results are discarded like everywhere else in the chunk path.
        // Under CP the gather is replaced by assembly from the gang's
        // published latents; the FMHA is the same call.
        if let Some(cp) = cp {
            mla_attention_chunk_cp(ctx, shape, w, kv, mla_index, s, cp)?;
        } else {
            mla_attention_chunk_fmha(ctx, shape, w, kv, mla_index, s)?;
        }
    } else {
        // Absorbed MLA over the paged latent: the kernel folds `w_kv_b`'s
        // per-head W_UK into the query and expands the attended latent with
        // W_UV, so the per-step kv_b expansion and the expanded K/V cache no
        // longer exist. A verify step's rows are packed, not slot-indexed, so
        // it walks the packed verify table; causality is the per-row context
        // length either way.
        let table = match mode {
            K3StepMode::Verify(_) => &kv.verify_table_dev,
            _ => &kv.table_dev,
        };
        k3_mla_paged_attn_launch(
            ctx,
            b,
            K3_MLA_HEADS,
            &s.query,
            &w.w_kv_b.data,
            &kv.slab,
            mla_index * K3_KV_PAGE_TOKENS * K3_MLA_LATENT_ROW,
            kv.page_stride(),
            table,
            kv.max_pages_per_slot,
            &s.context_len,
            &w.scale.data,
            &mut s.attn,
        )?;
    }
    k3_mul_sigmoid_batched_launch(ctx, b, K3_ATTN_INNER, &s.attn, &s.mla_gate, &mut s.gated)?;
    k3_gemm_full(ctx, &w.w_o, &s.gated, b, &mut s.hidden_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        K3_HIDDEN,
        0,
        1,
        &s.hidden_partial,
        &mut s.attn_out,
    )
}

// ── Latent MoE ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn moe_mlp(
    ctx: &DeviceContext,
    b: usize,
    shape: K3StepShape,
    layer: &K3LayerWeights,
    w: &K3MoeWeights,
    s: &mut K3Scratch,
) -> Result<()> {
    let experts = w.w_router.rows;
    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        &s.mixed2,
        &layer.gamma_post.data,
        &mut s.normed,
    )?;
    // Routing reads the pre-down hidden, as in the reference.
    k3_gemm_full(ctx, &w.w_router, &s.normed, b, &mut s.router_partial)?;
    k3_router_topk_batched_launch(
        ctx,
        b,
        experts,
        K3_ROUTER_TOPK,
        &s.router_partial,
        &w.bias,
        &w.rs.data,
        &mut s.topk_idx,
        &mut s.topk_weight,
    )?;
    k3_gemm_full(ctx, &w.w_lat_down, &s.normed, b, &mut s.latent_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_ROUTED_EXPERT_HIDDEN,
        K3_ROUTED_EXPERT_HIDDEN,
        0,
        1,
        &s.latent_partial,
        &mut s.latent,
    )?;

    // MegaMoE covers every world size, single rank included; the masked chain
    // is only reachable from a test configuration.
    if shape.mega {
        routed_experts_mega(ctx, shape, w, s)?;
    } else {
        routed_experts(ctx, shape, w, s)?;
    }

    k3_rms_norm_rbs_batched_launch(
        ctx,
        b,
        K3_ROUTED_EXPERT_HIDDEN,
        &s.routed_latent,
        &w.gamma_lat.data,
        &mut s.routed_latent_norm,
    )?;
    k3_gemm_full(
        ctx,
        &w.w_lat_up,
        &s.routed_latent_norm,
        b,
        &mut s.hidden_partial,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        K3_HIDDEN,
        0,
        1,
        &s.hidden_partial,
        &mut s.routed,
    )?;

    k3_gemm_full(ctx, &w.wsh, &s.normed, b, &mut s.shared_partial)?;
    let shared_width = 2 * K3_SHARED_INTERMEDIATE;
    k3_land_batched_launch(
        ctx,
        b,
        shared_width,
        K3_SHARED_INTERMEDIATE,
        0,
        1,
        &s.shared_partial,
        &mut s.shared_gate,
    )?;
    k3_land_batched_launch(
        ctx,
        b,
        shared_width,
        K3_SHARED_INTERMEDIATE,
        K3_SHARED_INTERMEDIATE,
        1,
        &s.shared_partial,
        &mut s.shared_up,
    )?;
    k3_situ_batched_launch(
        ctx,
        b,
        K3_SHARED_INTERMEDIATE,
        &s.shared_gate,
        &s.shared_up,
        &mut s.shared_act,
    )?;
    k3_gemm_full(ctx, &w.sh_down, &s.shared_act, b, &mut s.hidden_partial)?;
    k3_land_batched_launch(
        ctx,
        b,
        K3_HIDDEN,
        K3_HIDDEN,
        0,
        1,
        &s.hidden_partial,
        &mut s.shared,
    )?;
    k3_add2_batched_launch(ctx, b, K3_HIDDEN, &s.routed, &s.shared, &mut s.mlp_out)
}

/// The whole routed-expert stage as one DeepGEMM MegaMoE launch: dispatch,
/// both FP8xFP4 GEMMs, the situ activation, the mid-quantization and the
/// weighted combine.
///
/// Not bit-equivalent to [`routed_experts`], and not meant to be: the fused
/// kernel multiplies the routing weight into the activation *before* the down
/// projection (the chain applies it at combine time) and mid-quantizes per 32
/// elements rather than per 128. The chain therefore stays the oracle; this
/// path is gated on the golden fixture's noise floor, not on bitwise equality.
fn routed_experts_mega(
    ctx: &DeviceContext,
    shape: K3StepShape,
    w: &K3MoeWeights,
    s: &mut K3Scratch,
) -> Result<()> {
    ensure!(
        w.experts.form == K3ExpertBankForm::Mega,
        "K3 MegaMoE step reached a bank built for {:?}",
        w.experts.form
    );
    let mega = s
        .mega
        .as_mut()
        .context("K3 MegaMoE step ran without its symmetric buffer")?;
    // Single-rank steps run the whole bucket: the count has to be a function of
    // the bucket alone, because the step is captured into a CUDA graph. Above
    // one rank the rows leave this device — a padding row would put junk
    // routing on the wire and make a peer serve experts for a token nobody
    // wants — so only the live prefix is sent, and an idle rank sends nothing
    // at all while still launching (and so still serving its own experts for
    // its peers, and still meeting every barrier).
    let num_tokens = if mega.num_ranks > 1 {
        shape.live_rows
    } else {
        shape.bucket
    };
    mega.count_launch();
    let symm_ptrs = mega
        .peers()
        .context("K3 MegaMoE EP step ran before its peers published their symmetric slabs")?
        .to_vec();
    k3_mega_write_inputs_launch(
        ctx,
        &mega.layout,
        &mut mega.symm,
        num_tokens,
        K3_ROUTED_EXPERT_HIDDEN,
        K3_ROUTER_TOPK,
        &s.latent,
        &s.topk_idx,
        &s.topk_weight,
    )?;
    k3_mega_moe_launch(
        ctx,
        &mega.layout,
        &mut mega.symm,
        &symm_ptrs,
        K3MegaShape {
            num_tokens,
            num_max_tokens_per_rank: mega.max_tokens,
            // GLOBAL: the kernel turns an expert id into a destination rank by
            // dividing by the per-rank block, and the router already emits
            // global ids, so nothing is rebased on either side.
            num_experts: mega.routed_experts,
            num_topk: K3_ROUTER_TOPK,
            hidden: K3_ROUTED_EXPERT_HIDDEN,
            intermediate_hidden: K3_EXPERT_INTERMEDIATE,
            num_sms: shape.num_sms,
            num_ranks: mega.num_ranks,
            rank_idx: mega.rank_idx,
        },
        K3MegaActivation::Situ,
        &w.experts.w13_weight,
        &w.experts.w13_scale,
        &w.experts.w2_weight,
        &w.experts.w2_scale,
        &mut s.routed_latent,
    )
}

/// The masked FP8xFP4 grouped-GEMM chain, standing in for the reference's
/// per-row expert GEMVs and its weighted combine.
fn routed_experts(
    ctx: &DeviceContext,
    shape: K3StepShape,
    w: &K3MoeWeights,
    s: &mut K3Scratch,
) -> Result<()> {
    let route = shape.route();
    let latent = K3_ROUTED_EXPERT_HIDDEN;
    let inter = K3_EXPERT_INTERMEDIATE;
    let quant = K3_MOE_QUANT_GROUP;
    let moe = s
        .moe
        .as_mut()
        .context("K3 masked-chain step ran without its working set")?;

    k3_moe_local_route_metadata_launch(
        ctx,
        route,
        &s.topk_idx,
        &mut moe.masked_m,
        &mut moe.slot_map,
    )?;
    k3_moe_gather_fp8_quant_masked_launch(
        ctx,
        route,
        latent,
        &s.latent,
        &s.topk_idx,
        &moe.slot_map,
        &mut moe.w13_activation,
        &mut moe.w13_scale,
    )?;
    k3_fp8_scale_pack_ue8m0_launch(
        ctx,
        route.groups,
        latent / quant,
        route.masked_cap,
        &moe.w13_scale,
        &mut moe.w13_scale_packed,
    )?;
    k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch(
        ctx,
        K3DeepGemmFp8Fp4Kind::W13,
        route.groups,
        route.masked_cap,
        shape.num_sms,
        &moe.w13_activation,
        &moe.w13_scale_packed,
        &w.experts.w13_weight,
        &w.experts.w13_scale,
        &moe.masked_m,
        &mut moe.w13_out,
    )?;
    k3_situ_and_mul_fp8_quant_masked_launch(
        ctx,
        route,
        inter,
        &moe.w13_out,
        &s.topk_idx,
        &moe.slot_map,
        &mut moe.w2_activation,
        &mut moe.w2_scale,
    )?;
    k3_fp8_scale_pack_ue8m0_launch(
        ctx,
        route.groups,
        inter / quant,
        route.masked_cap,
        &moe.w2_scale,
        &mut moe.w2_scale_packed,
    )?;
    k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch(
        ctx,
        K3DeepGemmFp8Fp4Kind::W2,
        route.groups,
        route.masked_cap,
        shape.num_sms,
        &moe.w2_activation,
        &moe.w2_scale_packed,
        &w.experts.w2_weight,
        &w.experts.w2_scale,
        &moe.masked_m,
        &mut moe.w2_out,
    )?;
    k3_moe_weighted_combine_launch(
        ctx,
        route,
        latent,
        &moe.w2_out,
        &s.topk_idx,
        &moe.slot_map,
        &s.topk_weight,
        &mut s.routed_latent,
    )
}
