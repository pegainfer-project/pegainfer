use super::*;

pub(super) fn load_layer_forward_cache(
    ctx: &KimiRankGpuContext,
    weights: &KimiRankGpuWeights,
    layer: &KimiLayerWeightNames,
) -> Result<KimiLayerForwardCache> {
    let q_a_proj = raw_tensor(weights, &layer.attention.q_a_proj)?
        .copy_bf16_matrix_from_shape(ctx, "attention_q_a_proj")?;
    ensure!(
        q_a_proj.rows == KIMI_K2_Q_LORA_RANK && q_a_proj.cols == KIMI_K2_HIDDEN,
        "layer {} q_a_proj shape must be [{}, {}], got [{}, {}]",
        layer.layer_idx,
        KIMI_K2_Q_LORA_RANK,
        KIMI_K2_HIDDEN,
        q_a_proj.rows,
        q_a_proj.cols
    );
    let kv_a_proj_with_mqa = raw_tensor(weights, &layer.attention.kv_a_proj_with_mqa)?
        .copy_bf16_matrix_from_shape(ctx, "attention_kv_a_proj_with_mqa")?;
    ensure!(
        kv_a_proj_with_mqa.rows == KIMI_K2_MLA_KV_A_OUT
            && kv_a_proj_with_mqa.cols == KIMI_K2_HIDDEN,
        "layer {} kv_a_proj_with_mqa shape must be [{}, {}], got [{}, {}]",
        layer.layer_idx,
        KIMI_K2_MLA_KV_A_OUT,
        KIMI_K2_HIDDEN,
        kv_a_proj_with_mqa.rows,
        kv_a_proj_with_mqa.cols
    );
    let device_ctx = ctx.as_device_context();
    let fused_qkv_a_proj = GpuWeight::from_device_matrix(DeviceMatrix::vstack(
        &device_ctx,
        &[&q_a_proj, &kv_a_proj_with_mqa],
    )?)?;
    let attention = KimiAttentionForwardCache {
        input_norm: NormWeight::from_device_vec(
            raw_tensor(weights, &layer.attention.input_layernorm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_HIDDEN,
                "attention_input_norm",
            )?,
        )?,
        fused_qkv_a_proj,
        q_a_norm: NormWeight::from_device_vec(
            raw_tensor(weights, &layer.attention.q_a_layernorm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_Q_LORA_RANK,
                "attention_q_a_norm",
            )?,
        )?,
        q_b_proj: raw_tensor(weights, &layer.attention.q_b_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_q_b_proj")?,
        kv_a_norm: NormWeight::from_device_vec(
            raw_tensor(weights, &layer.attention.kv_a_layernorm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_MLA_KV_LORA_RANK,
                "attention_kv_a_norm",
            )?,
        )?,
        kv_b_proj: raw_tensor(weights, &layer.attention.kv_b_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_kv_b_proj")?,
        o_proj: raw_tensor(weights, &layer.attention.o_proj)?
            .copy_bf16_matrix_from_shape(ctx, "attention_o_proj")?,
        post_attention_norm: NormWeight::from_device_vec(
            raw_tensor(weights, &layer.attention.post_attention_layernorm)?.copy_bf16_vec(
                ctx,
                KIMI_K2_HIDDEN,
                "post_attention_norm",
            )?,
        )?,
    };

    let kind = match &layer.kind {
        KimiLayerWeightKindNames::Dense(mlp) => {
            let gate_proj = raw_tensor(weights, &mlp.gate_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_gate_proj")?;
            let up_proj = raw_tensor(weights, &mlp.up_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_up_proj")?;
            let down_proj = raw_tensor(weights, &mlp.down_proj)?
                .copy_bf16_matrix_from_shape(ctx, "dense_down_proj")?;
            ensure_dense_mlp_shapes("dense_mlp", &gate_proj, &up_proj, &down_proj)?;
            let gate_up_proj = DeviceMatrix::vstack(&device_ctx, &[&gate_proj, &up_proj])?;
            KimiLayerForwardKindCache::Dense(KimiDenseForwardCache {
                gate_up_proj,
                down_proj,
            })
        }
        KimiLayerWeightKindNames::Moe(moe) => {
            let router = KimiRouterGpuWeights {
                gate_weight: raw_tensor(weights, &moe.router.gate_weight)?,
                e_score_correction_bias: raw_tensor(weights, &moe.router.e_score_correction_bias)?,
            }
            .copy_to_device_weights(ctx)?;
            let shared_gate_proj = raw_tensor(weights, &moe.shared_experts.gate_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_gate_proj")?;
            let shared_up_proj = raw_tensor(weights, &moe.shared_experts.up_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_up_proj")?;
            let shared_down_proj = raw_tensor(weights, &moe.shared_experts.down_proj)?
                .copy_bf16_matrix_from_shape(ctx, "shared_down_proj")?;
            ensure_dense_mlp_shapes(
                "shared_expert",
                &shared_gate_proj,
                &shared_up_proj,
                &shared_down_proj,
            )?;
            let shared_gate_up_proj =
                DeviceMatrix::vstack(&device_ctx, &[&shared_gate_proj, &shared_up_proj])?;
            KimiLayerForwardKindCache::Moe(KimiMoeForwardCache {
                router,
                shared_gate_up_proj,
                shared_down_proj,
            })
        }
    };

    Ok(KimiLayerForwardCache {
        layer_idx: layer.layer_idx,
        attention,
        kind,
    })
}

pub(super) fn raw_tensor<'a>(
    weights: &'a KimiRankGpuWeights,
    name: &str,
) -> Result<&'a KimiGpuRawTensor> {
    weights
        .tensors
        .get(name)
        .with_context(|| format!("missing Kimi forward tensor {name}"))
}

fn ensure_dense_mlp_shapes(
    label: &str,
    gate: &DeviceMatrix,
    up: &DeviceMatrix,
    down: &DeviceMatrix,
) -> Result<()> {
    ensure!(
        gate.cols == KIMI_K2_HIDDEN,
        "{label} gate input dim must be {}, got {}",
        KIMI_K2_HIDDEN,
        gate.cols
    );
    ensure!(
        up.rows == gate.rows && up.cols == gate.cols,
        "{label} up shape [{}, {}] must match gate [{}, {}]",
        up.rows,
        up.cols,
        gate.rows,
        gate.cols
    );
    ensure!(
        down.rows == KIMI_K2_HIDDEN && down.cols == gate.rows,
        "{label} down shape [{}, {}] must be [{}, {}]",
        down.rows,
        down.cols,
        KIMI_K2_HIDDEN,
        gate.rows
    );
    Ok(())
}
