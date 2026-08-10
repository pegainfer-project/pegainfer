//! Captured GLM5.2 whole-step forward body.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::add_scaled_bf16_into;
use pegainfer_kernels::ops::argmax_bf16_split_into;
use pegainfer_kernels::ops::copy_hidden_rows_raw_into;
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;

use crate::bookend::glm52_embed_into;
use crate::bookend::glm52_final_norm_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_VOCAB;
use crate::dense::glm52_dense_mlp_forward_into;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52KvSlab;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_layer_attention_half;
use crate::layer::glm52_layer_finish;
use crate::layer::glm52_layer_finish_fused;
use crate::moe_decode::run_ep_router_into;
use crate::moe_ep::Glm52MoeEpState;
use crate::moe_ep8::Glm52MoeEp8LayerWeights;
use crate::moe_tp::Glm52MoeTpRank;
use crate::scratch::Glm52DecodeScratch;

/// The captured region of one decode step: embed → 78 layers → lm_head →
/// device argmax over the step's `batch` rows (read from the attend plan —
/// the single source of truth for the step's row count). Shared verbatim by
/// every batch bucket; only the plan, scratch, block table, and
/// `global_tokens` differ per shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_step_body(
    ctx: &DeviceContext,
    aux: &DeviceContext,
    mut ep8: Option<&mut Glm52MoeEpState>,
    mut tp: Option<&mut Glm52MoeTpRank>,
    layers: &[Glm52DecoderLayerWeights],
    slab: &mut Glm52KvSlab,
    caches: &[Glm52LayerCaches],
    embed: &DeviceMatrix,
    final_norm: &DeviceVec,
    lm_head: &DeviceMatrix,
    vocab_start: usize,
    token_ids: &CudaSlice<u32>,
    step: &Glm52DecodeStep<'_>,
    s: &mut Glm52DecodeScratch,
    global_tokens: usize,
) -> Result<()> {
    let batch = step.mla_sched.batch();
    ensure!(
        layers.len() == caches.len(),
        "GLM5.2 step body layer/cache-offset tables disagree: {} vs {}",
        layers.len(),
        caches.len()
    );
    if let Some(rank) = tp.as_ref()
        && !rank.slices.is_empty()
    {
        anyhow::bail!("GLM5.2 TP4 is prefill-only; decode whole-step path (LL MoE/AR) was removed");
    }
    glm52_embed_into(ctx, embed, token_ids, &mut s.hidden)?;
    // Layer 0's input norm is standalone (the embedding is the residual);
    // every later layer's input norm is fused into the previous layer's
    // closing add (`glm52_layer_finish_fused`).
    rms_norm_rows_into(
        ctx,
        s.hidden.data(),
        &layers[0].input_ln,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        batch,
        s.layer.normed.data_mut(),
    )?;
    let mut carry_ready = false;
    for (layer, weights) in layers.iter().enumerate() {
        let parity = layer % 2;
        // Attention-TP: a head-sharded layer's o_proj partial crosses the
        // NCCL all-reduce inside the attention half.
        let tp_ar = if weights.mla.heads == crate::config::GLM52_HEADS {
            None
        } else {
            let rank = tp
                .as_deref_mut()
                .context("GLM5.2 sharded attention without TP state")?;
            Some(&mut rank.state)
        };
        glm52_layer_attention_half(
            ctx,
            Some(aux),
            weights,
            slab,
            caches[layer],
            step,
            s,
            &mut carry_ready,
            parity,
            layer == 0,
            tp_ar,
        )
        .with_context(|| format!("GLM5.2 layer {layer} attention half"))?;
        let tp_padded_mlp = false;
        match &weights.mlp {
            Glm52LayerMlp::Dense(dense) => glm52_dense_mlp_forward_into(
                ctx,
                dense,
                s.layer.normed2.data(),
                &mut s.dense_mlp,
                s.layer.mlp_out.data_mut(),
            )?,
            Glm52LayerMlp::MoeEp8(moe) => {
                let ep8 = ep8
                    .as_deref_mut()
                    .context("GLM5.2 EP MoE layer reached without DeepEP state")?;
                glm52_moe_ep_layer(ctx, aux, ep8, moe, s, batch, global_tokens)
                    .with_context(|| format!("GLM5.2 layer {layer} EP MoE"))?;
            }
            Glm52LayerMlp::MoeTp(_) => {
                anyhow::bail!("GLM5.2 TP4 is prefill-only; decode MoE TP path was removed");
            }
        }
        if layer + 1 < layers.len() {
            glm52_layer_finish_fused(ctx, s, parity, &layers[layer + 1].input_ln, tp_padded_mlp)?;
        } else {
            glm52_layer_finish(ctx, s, parity, tp_padded_mlp)?;
        }
        // DSpark aux-hidden capture: after layer L's closing add the residual
        // stream lives in `attn[parity]` (updated in place by the fused
        // add+norm; none of the capture layers is the last layer, which lands
        // in `s.hidden` instead). Recorded into the step graph — pointer-
        // stable, ~60 KB/row per step — only when the drafter was requested
        // at launch (otherwise neither the buffer nor the copy nodes exist).
        if let (Some(feature), Some(captured)) = (
            crate::dspark::GLM52_DSPARK_AUX_LAYERS
                .iter()
                .position(|&aux| aux == layer),
            s.captured.as_mut(),
        ) {
            copy_hidden_rows_raw_into(
                ctx,
                s.layer.attn[parity].data(),
                GLM52_HIDDEN,
                captured.data_mut(),
                crate::dspark::GLM52_DSPARK_CONTEXT_DIM,
                feature * GLM52_HIDDEN,
                batch,
            )?;
        }
    }

    glm52_final_norm_into(ctx, &s.hidden, final_norm, &mut s.final_normed)?;
    glm52_lm_head_into(ctx, &s.final_normed, lm_head, &mut s.logits)?;
    // Device greedy argmax per row (same semantics as a host scan: lowest
    // index wins ties, NaN never wins) — the step's egress shrinks from the
    // full vocab rows to 6 bytes per row, and the kernel chain ends on-device
    // (the graph boundary). Two-stage: per-4096-tile partials in parallel,
    // then one finalize block per row — bit-identical to the single-block
    // scan (the partials carry global indices, same total order), and each
    // row's result is independent of its slot-mates.
    argmax_bf16_split_into(
        ctx,
        s.logits.data(),
        batch,
        lm_head.rows,
        &mut s.argmax_partial_values,
        &mut s.argmax_partial_indices,
        &mut s.argmax_values,
        &mut s.argmax_indices,
    )?;

    if tp.is_some() {
        anyhow::bail!("GLM5.2 TP4 is prefill-only; decode vocab-parallel LL allreduce was removed");
    }
    ensure!(
        lm_head.rows == GLM52_VOCAB && vocab_start == 0,
        "GLM5.2 non-TP decode received a sharded vocabulary head"
    );
    Ok(())
}

/// One layer's EP MoE half (EP8 or EP4 — the state carries the chain):
/// shared expert forked to the aux stream, routed path through router +
/// DeepEP dispatch/expert-GEMM/combine, joined by the closing add into
/// `mlp_out`. The events recorded here during capture become graph edges;
/// replay keeps the parallel branches.
pub(super) fn glm52_moe_ep_layer(
    ctx: &DeviceContext,
    aux: &DeviceContext,
    ep8: &mut Glm52MoeEpState,
    moe: &Glm52MoeEp8LayerWeights,
    s: &mut Glm52DecodeScratch,
    batch: usize,
    global_tokens: usize,
) -> Result<()> {
    // Fork: the shared expert only needs `normed2`, so it runs on the aux
    // stream concurrently with the routed path's dispatch/grouped-GEMM/
    // combine — the cooperative collectives occupy a fixed SM slice and
    // mostly wait on peers, leaving the rest of the GPU free.
    let normed_ready = ctx.stream.record_event(None)?;
    aux.stream.wait(&normed_ready)?;
    moe.shared.forward_into(
        aux,
        s.layer.normed2.data(),
        &mut s.shared_mlp,
        s.layer.shared_out.data_mut(),
    )?;
    let shared_done = aux.stream.record_event(None)?;

    run_ep_router_into(ctx, &moe.router, s.layer.normed2.data(), &mut s.router)?;
    let dispatched = ep8.routed_forward(
        ctx,
        &moe.bank,
        Some((s.layer.normed2.data(), &s.router.route, batch)),
        global_tokens,
    )?;
    ensure!(
        dispatched,
        "EP MoE returned no combined output for the dispatched rows"
    );
    // Join: the closing add consumes both branches.
    ctx.stream.wait(&shared_done)?;
    add_scaled_bf16_into(
        ctx,
        ep8.combined(),
        crate::config::GLM52_ROUTED_SCALING_FACTOR as f32,
        s.layer.shared_out.data(),
        batch * GLM52_HIDDEN,
        s.layer.mlp_out.data_mut(),
    )
}
