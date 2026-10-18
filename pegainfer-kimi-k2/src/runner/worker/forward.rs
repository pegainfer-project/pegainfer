use super::runtime::*;
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_decode_batch_next_token_kernels(
    device_ctx: &DeviceContext,
    decode_aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    cache: &KimiOneTokenForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    kv_pool: &mut KimiWorkerKvPool,
    decode_arena: &mut KimiWorkerDecodeArena,
    active_len: usize,
    local_heads: usize,
    mut deepep: Option<&mut crate::runner::moe_deepep::KimiMoeDeepEpState>,
) -> Result<()> {
    typed_ops::embedding_vocab_shard_into(
        device_ctx,
        &cache.token_embedding,
        &decode_arena.token_ids_d,
        &mut decode_arena.scratch.mla.hidden,
        cache.vocab_start as u32,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
        device_ctx,
        &mut decode_arena.scratch.mla.hidden,
        &mut decode_arena.scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    for layer in &cache.layers {
        forward_mla_decode_layer_into(
            device_ctx,
            &layer.attention,
            kv_pool,
            decode_arena,
            layer.layer_idx,
            local_heads,
        )
        .with_context(|| format!("Kimi MLA batch decode layer {}", layer.layer_idx))?;
        maybe_all_reduce_hidden_via_f32_in_place(
            device_ctx,
            &mut decode_arena.scratch.mla.projected,
            &mut decode_arena.scratch.comm.hidden_allreduce_f32,
            comm,
        )?;
        typed_ops::fused_add_rms_norm_round_into(
            device_ctx,
            &mut decode_arena.scratch.mla.hidden,
            &decode_arena.scratch.mla.projected,
            &layer.attention.post_attention_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut decode_arena.scratch.mla.normed,
        )?;
        match &layer.kind {
            KimiLayerForwardKindCache::Dense(dense) => {
                forward_dense_mlp_decode_normed_into(
                    device_ctx,
                    comm,
                    dense,
                    &mut decode_arena.scratch,
                )
                .with_context(|| {
                    format!("Kimi dense batch decode MLP layer {}", layer.layer_idx)
                })?;
            }
            KimiLayerForwardKindCache::Moe(moe) => {
                if let Some(dp) = deepep.as_mut() {
                    let arena_seq_len = decode_arena.scratch.mla.hidden.seq_len;
                    decode_arena.scratch.set_moe_seq_len(active_len)?;
                    let deepep_result =
                        crate::runner::moe_deepep::forward_moe_layer_decode_deepep_normed(
                            device_ctx,
                            decode_aux_ctx,
                            layer.layer_idx,
                            moe,
                            expert_kernels,
                            &mut decode_arena.scratch,
                            dp,
                        );
                    let restore_result = decode_arena.scratch.set_moe_seq_len(arena_seq_len);
                    restore_result?;
                    deepep_result.with_context(|| {
                        format!("Kimi MoE DeepEP batch decode layer {}", layer.layer_idx)
                    })?;
                } else {
                    crate::runner::moe_nccl::forward_moe_layer_decode_normed_into(
                        device_ctx,
                        decode_aux_ctx,
                        comm,
                        layer.layer_idx,
                        moe,
                        expert_kernels,
                        &mut decode_arena.scratch,
                    )
                    .with_context(|| format!("Kimi MoE batch decode layer {}", layer.layer_idx))?;
                }
            }
        }
    }

    let active_len = decode_arena.scratch.mla.hidden.seq_len;
    typed_ops::rms_norm_into(
        device_ctx,
        &decode_arena.scratch.mla.hidden,
        &cache.final_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut decode_arena.scratch.mla.normed,
    )?;
    typed_ops::gemm_runtime_out_graphsafe_into(
        device_ctx,
        &cache.lm_head,
        &decode_arena.scratch.mla.normed,
        &mut decode_arena.logits,
    )?;
    launch_local_top1_batch(
        device_ctx,
        &decode_arena.logits,
        active_len,
        &mut decode_arena.scratch.sampling.top1_value_scratch,
        &mut decode_arena.scratch.sampling.top1_out,
        &mut decode_arena.scratch.sampling.top1_partial_values,
        &mut decode_arena.scratch.sampling.top1_partial_indices,
    )
}

fn forward_mla_decode_layer_into(
    ctx: &DeviceContext,
    attention: &KimiAttentionForwardCache,
    kv_pool: &mut KimiWorkerKvPool,
    arena: &mut KimiWorkerDecodeArena,
    layer_idx: usize,
    local_heads: usize,
) -> Result<()> {
    let KimiWorkerDecodeArena {
        layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        batch_indices_d,
        positions_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        cos_d,
        sin_d,
        scratch,
        ..
    } = arena;
    let layer_cache = kv_pool.layer_mut(layer_idx)?;

    typed_ops::rms_norm_into(
        ctx,
        &scratch.mla.hidden,
        &attention.input_norm,
        KIMI_K2_RMS_NORM_EPS,
        &mut scratch.mla.normed,
    )?;
    typed_ops::gemm_graphsafe_into(
        ctx,
        &attention.fused_qkv_a_proj,
        &scratch.mla.normed,
        &mut scratch.mla.qkv_a,
    )?;
    kimi_mla_split_qkv_a_norm(
        ctx,
        &scratch.mla.qkv_a,
        &attention.q_a_norm,
        &attention.kv_a_norm,
        &mut scratch.mla.q_a_normed,
        &mut scratch.mla.compressed_normed,
        &mut scratch.mla.k_rope,
        KIMI_K2_RMS_NORM_EPS,
    )?;
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &attention.q_b_proj,
        &scratch.mla.q_a_normed,
        &mut scratch.mla.q_proj,
    )?;
    kimi_mla_rope_split_decode_rt(
        ctx,
        &scratch.mla.q_proj,
        &scratch.mla.k_rope,
        cos_d,
        sin_d,
        positions_d,
        &mut scratch.mla.q_nope,
        &mut scratch.mla.q_pe,
        &mut scratch.mla.append_kpe,
        local_heads,
    )?;
    kimi_mla_absorb_q_nope_rt(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.q_nope,
        &mut scratch.mla.q_abs_nope,
        local_heads,
    )?;
    kimi_mla_paged_kv_append(
        ctx,
        &mut layer_cache.ckv_cache,
        &mut layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        &scratch.mla.compressed_normed,
        &scratch.mla.append_kpe,
        batch_indices_d,
        positions_d,
    )?;
    kimi_flashinfer_batch_decode_mla_rt(
        ctx,
        &scratch.mla.q_abs_nope,
        &scratch.mla.q_pe,
        &mut scratch.mla.latent,
        &layer_cache.ckv_cache,
        &layer_cache.kpe_cache,
        *layout,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        kimi_mla_softmax_scale(),
        local_heads,
    )?;
    kimi_mla_v_up_rt(
        ctx,
        &attention.kv_b_proj,
        &scratch.mla.latent,
        &mut scratch.mla.attn_out,
        local_heads,
    )?;
    if attention.o_proj.rows == KIMI_K2_HIDDEN
        && attention.o_proj.cols == KIMI_O_PROJ_CUBLASLT_INPUT
        && kimi_o_proj_cublaslt_supports_batch_size(scratch.mla.attn_out.seq_len)
    {
        kimi_o_proj_cublaslt_into(
            ctx,
            &attention.o_proj,
            &scratch.mla.attn_out,
            &mut scratch.mla.projected,
        )?;
    } else {
        typed_ops::gemm_dm_hs_to_typed_graphsafe(
            ctx,
            &attention.o_proj,
            &scratch.mla.attn_out,
            &mut scratch.mla.projected,
        )?;
    }
    Ok(())
}

pub(super) fn forward_dense_mlp_batch_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    dense: &KimiDenseForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
) -> Result<()> {
    let seq_len = hidden.seq_len;
    typed_ops::rms_norm_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    )?;
    let mut gate_up = HiddenStates::zeros(ctx, dense.gate_up_proj.rows, seq_len)?;
    typed_ops::gemm_dm_typed_to_hs(ctx, &dense.gate_up_proj, normed, &mut gate_up)?;
    let mut activated = HiddenStates::zeros(ctx, dense.down_proj.cols, seq_len)?;
    typed_ops::silu_mul_hs_fused_into(ctx, &gate_up, &mut activated)?;
    let mut mlp_out = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
    typed_ops::gemm_dm_hs_to_typed(ctx, &dense.down_proj, &activated, &mut mlp_out)?;
    if let Some(comm) = comm {
        comm.all_reduce_in_place(&mut mlp_out.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;
    }
    typed_ops::add_into(ctx, hidden, &mlp_out, next_hidden)?;
    std::mem::swap(hidden, next_hidden);
    Ok(())
}

fn forward_dense_mlp_decode_normed_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    dense: &KimiDenseForwardCache,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &dense.gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.dense_mlp.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.dense_mlp.gate_up,
        &mut scratch.dense_mlp.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &dense.down_proj,
        &scratch.dense_mlp.activated,
        &mut scratch.mla.projected,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;
    typed_ops::add_into(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}
