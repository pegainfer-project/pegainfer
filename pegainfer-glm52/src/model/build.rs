//! Weight-extraction helpers: build one decoder layer's GPU-resident
//! weights (MLA + indexer + dense/MoE MLP + layernorms) from the resident
//! tensor map.

use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;

use crate::config::GLM52_DENSE_LAYERS;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::glm52_layer_has_full_indexer;
use crate::dense::Glm52DenseMlpWeights;
use crate::fp8::ProjWeight;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerIndexer;
use crate::layer::Glm52LayerMlp;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::moe_decode::Glm52MoeExpertBank;
use crate::moe_decode::Glm52MoeRouterWeights;
use crate::moe_decode::Glm52MoeSharedExpert;
use crate::moe_ep8::Glm52MoeEp8LayerWeights;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::retype_owned;

/// Take one fp8 projection (weight + scale) out of the resident tensor map.
fn take_proj(w: &mut Glm52RankGpuWeights, stem: &str, n: usize, k: usize) -> Result<ProjWeight> {
    ProjWeight::from_device(
        w.take_tensor(&format!("{stem}.weight"))?,
        w.take_tensor(&format!("{stem}.weight_scale_inv"))?,
        n,
        k,
    )
}

/// Take a bf16 vector (e.g. a layernorm gamma) out of the resident map.
pub(super) fn take_bf16_vec(
    ctx: &DeviceContext,
    w: &mut Glm52RankGpuWeights,
    name: &str,
    len: usize,
) -> Result<DeviceVec> {
    let raw = w.take_tensor(name)?;
    ensure!(
        raw.len() == len * 2,
        "GLM5.2 tensor {name} bytes {} != bf16 [{len}]",
        raw.len()
    );
    Ok(DeviceVec {
        data: retype_owned::<bf16>(&ctx.stream, raw)?,
        len,
    })
}

pub(super) fn build_decoder_layer(
    ctx: &DeviceContext,
    w: &mut Glm52RankGpuWeights,
    layer: usize,
    moe_topo: crate::Glm52MoeTopo,
    attn_shard: Option<usize>,
) -> Result<Glm52DecoderLayerWeights> {
    let p = format!("model.layers.{layer}");

    let kv_b_full = take_proj(w, &format!("{p}.self_attn.kv_b_proj"), 28_672, 512)?;
    let q_b_full = take_proj(w, &format!("{p}.self_attn.q_b_proj"), 16_384, 2048)?;
    let o_proj_full = take_proj(w, &format!("{p}.self_attn.o_proj"), GLM52_HIDDEN, 16_384)?;
    // Attention-TP head shard: rank r keeps its 1/TP slice of q/v heads
    // (TP4 = 16 heads) — q_b/kv_b output rows, o_proj input
    // columns; `Glm52MlaLayerWeights` derives its head count from these
    // shapes. The indexer is NOT sharded (its logits sum over all 32 index
    // heads BEFORE the top-2048, so a shard would need a cross-rank logits
    // reduction — the whole 21-layer indexer weight set is ~0.18 GB,
    // replication is the right trade).
    let (q_b, kv_b, o_proj) = match attn_shard {
        Some(rank) => {
            let heads = crate::config::GLM52_HEADS / moe_topo.device_count();
            let q_rows = heads * crate::config::GLM52_QK_HEAD_DIM;
            let kv_rows =
                heads * (crate::config::GLM52_QK_NOPE_HEAD_DIM + crate::config::GLM52_V_HEAD_DIM);
            let o_cols = heads * crate::config::GLM52_V_HEAD_DIM;
            (
                q_b_full.slice_rows(ctx, rank * q_rows, q_rows)?,
                kv_b_full.slice_rows(ctx, rank * kv_rows, kv_rows)?,
                o_proj_full.slice_cols(ctx, rank * o_cols, o_cols)?,
            )
        }
        None => (q_b_full, kv_b_full, o_proj_full),
    };
    let mla = Glm52MlaLayerWeights::from_device(
        ctx,
        take_proj(w, &format!("{p}.self_attn.q_a_proj"), 2048, GLM52_HIDDEN)?,
        take_bf16_vec(ctx, w, &format!("{p}.self_attn.q_a_layernorm.weight"), 2048)?,
        q_b,
        take_proj(
            w,
            &format!("{p}.self_attn.kv_a_proj_with_mqa"),
            576,
            GLM52_HIDDEN,
        )?,
        take_bf16_vec(ctx, w, &format!("{p}.self_attn.kv_a_layernorm.weight"), 512)?,
        &kv_b,
        o_proj,
    )?;
    // kv_b is only dequanted into the absorb factors, not stored — free the
    // fp8 blob before the indexer/MLP uploads below.
    drop(kv_b);

    let indexer = if glm52_layer_has_full_indexer(layer) {
        let ip = format!("{p}.self_attn.indexer");
        let k_norm_w = ctx
            .stream
            .clone_dtoh(&w.take_tensor(&format!("{ip}.k_norm.weight"))?)?;
        let k_norm_b = ctx
            .stream
            .clone_dtoh(&w.take_tensor(&format!("{ip}.k_norm.bias"))?)?;
        let weights_proj = retype_owned::<bf16>(
            &ctx.stream,
            w.take_tensor(&format!("{ip}.weights_proj.weight"))?,
        )?;
        Glm52LayerIndexer::Full(Box::new(Glm52IndexerLayerWeights::from_device(
            ctx,
            take_proj(w, &format!("{ip}.wq_b"), 32 * GLM52_INDEX_HEAD_DIM, 2048)?,
            take_proj(w, &format!("{ip}.wk"), GLM52_INDEX_HEAD_DIM, GLM52_HIDDEN)?,
            weights_proj,
            &k_norm_w,
            &k_norm_b,
        )?))
    } else {
        Glm52LayerIndexer::Shared
    };

    let mp = format!("{p}.mlp");
    let mlp = if layer < GLM52_DENSE_LAYERS {
        Glm52LayerMlp::Dense(Box::new(Glm52DenseMlpWeights::from_device(
            ctx,
            &take_proj(w, &format!("{mp}.gate_proj"), 12_288, GLM52_HIDDEN)?,
            &take_proj(w, &format!("{mp}.up_proj"), 12_288, GLM52_HIDDEN)?,
            take_proj(w, &format!("{mp}.down_proj"), GLM52_HIDDEN, 12_288)?,
        )?))
    } else if moe_topo.uses_tensor_replicated_moe() {
        // Tensor-replicated topology keeps routed and shared experts in the
        // rank slice bank; the resident first pass carries only the router.
        Glm52LayerMlp::MoeTp(Box::new(Glm52MoeRouterWeights::new(
            ctx,
            w.take_tensor(&format!("{mp}.gate.weight"))?,
            w.take_tensor(&format!("{mp}.gate.e_score_correction_bias"))?,
        )?))
    } else {
        Glm52LayerMlp::MoeEp8(Box::new(Glm52MoeEp8LayerWeights {
            router: Glm52MoeRouterWeights::new(
                ctx,
                w.take_tensor(&format!("{mp}.gate.weight"))?,
                w.take_tensor(&format!("{mp}.gate.e_score_correction_bias"))?,
            )?,
            shared: Glm52MoeSharedExpert::new(
                ctx,
                &take_proj(
                    w,
                    &format!("{mp}.shared_experts.gate_proj"),
                    2048,
                    GLM52_HIDDEN,
                )?,
                &take_proj(
                    w,
                    &format!("{mp}.shared_experts.up_proj"),
                    2048,
                    GLM52_HIDDEN,
                )?,
                take_proj(
                    w,
                    &format!("{mp}.shared_experts.down_proj"),
                    GLM52_HIDDEN,
                    2048,
                )?,
            )?,
            bank: Glm52MoeExpertBank::from_regions(ctx, w.take_expert_layer(layer)?)?,
        }))
    };

    Ok(Glm52DecoderLayerWeights {
        input_ln: take_bf16_vec(ctx, w, &format!("{p}.input_layernorm.weight"), GLM52_HIDDEN)?,
        post_attn_ln: take_bf16_vec(
            ctx,
            w,
            &format!("{p}.post_attention_layernorm.weight"),
            GLM52_HIDDEN,
        )?,
        mla,
        indexer,
        mlp,
    })
}
