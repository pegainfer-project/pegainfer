//! EP4 layer-6 MoE oracle gate for the SM100 DeepGEMM expert chain.
//!
//! Rank 0 walks the full decoder layer (attention + indexer + EP4 MoE +
//! shared expert) over the same seeded input as the EP1 gate; ranks 1..3
//! hold their 64 local experts and replay one collective per position. The
//! The probes use the `deepgemm` oracle regime: non-expert projections retain
//! their GEMV precision while routed experts use UE8M0-scaled FP8 operands.
//! Passing proves dispatch → metadata → DeepGEMM W13/W2 → combine lands on
//! the modeled layer output.

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::add_scaled_bf16_into;
use pegainfer_kernels::ops::glm52_ep_deepep_unique_id;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use pegainfer_kernels::tensor::DeviceContext;

use super::layer::GateLayerMlp;
use super::layer::LayerTensors;
use super::layer::MOE_ORACLE_CTX;
use super::layer::MOE_ORACLE_DEEPGEMM_LAYER_PROBES;
use super::layer::MOE_ORACLE_DEEPGEMM_LAYER_TOL;
use super::layer::MOE_ORACLE_HIDDEN_DIGEST;
use super::layer::MOE_ORACLE_INPUT_SCALE;
use super::layer::MOE_ORACLE_LAYER;
use super::layer::MOE_ORACLE_SEED;
use super::layer::assert_layer_probes;
use super::layer::checked_hidden;
use super::layer::load_decoder_layer;
use super::layer::load_rank_expert_bank;
use super::layer::model_path;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_layer_attention_half;
use crate::layer::glm52_layer_finish;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::INDEX_CACHE_BLOCK;
use crate::model::NUM_SMS;
use crate::model::rope_tables;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::run_ep_router;
use crate::moe_ep::Glm52MoeEpRankState;
use crate::moe_ep::glm52_moe_ep_routed_forward;
use crate::scratch::Glm52DecodeScratch;

const EP_RANKS: usize = 4;
/// Every global-token protocol value a production engine can pass
/// — one per decode bucket, largest first (the worst-case row bound leads).
const GLOBAL_TOKEN_BUCKETS: [usize; GLM52_DECODE_BUCKETS.len()] = {
    let mut buckets = [0usize; GLM52_DECODE_BUCKETS.len()];
    let mut i = 0;
    while i < GLM52_DECODE_BUCKETS.len() {
        buckets[i] = EP_RANKS * GLM52_DECODE_BUCKETS[GLM52_DECODE_BUCKETS.len() - 1 - i];
        i += 1;
    }
    buckets
};

#[test]
#[ignore = "requires 4 GPUs (GB300/H200) + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn layer_moe_ep4_oracle_gate() -> Result<()> {
    let hidden_host = checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?;
    let unique_id = glm52_ep_deepep_unique_id(4)?;
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);

    // Expert ranks: pack the 64 local experts, then replay one collective per
    // position. Context creation inside is collective with rank 0's below.
    let handles: Vec<_> = (1..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("ep4-gate-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, MOE_ORACLE_LAYER, rank, EP_RANKS)?;
                    let mut ep4 =
                        Glm52MoeEpRankState::<pegainfer_kernels::ops::Glm52Ep4DeepEpAbi>::new(
                            &ctx, &unique_id, EP_RANKS, rank,
                        )?;
                    for global_tokens in GLOBAL_TOKEN_BUCKETS {
                        for _position in 0..MOE_ORACLE_CTX {
                            let dispatched = glm52_moe_ep_routed_forward(
                                &ctx,
                                &mut ep4,
                                &bank,
                                None,
                                global_tokens,
                            )?;
                            ensure!(!dispatched, "expert rank produced a combined output");
                        }
                    }
                    Ok(())
                })
                .expect("spawn ep4 gate rank thread")
        })
        .collect();

    // Rank 0: full decoder layer with the EP4 MoE half, prefill-via-decode.
    let ctx = DeviceContext::new_with_device(0)?;
    let w = load_decoder_layer(
        &ctx,
        &model_path(),
        MOE_ORACLE_LAYER,
        GateLayerMlp::MoeEp4Rank0,
    )?;
    let mut ep4 = Glm52MoeEpRankState::<pegainfer_kernels::ops::Glm52Ep4DeepEpAbi>::new(
        &ctx, &unique_id, EP_RANKS, 0,
    )?;
    let outputs: Result<Vec<Vec<f32>>> = GLOBAL_TOKEN_BUCKETS
        .into_iter()
        .map(|global_tokens| {
            run_layer_prefill_ep4(
                &ctx,
                &w,
                &mut ep4,
                &hidden_host,
                MOE_ORACLE_CTX,
                global_tokens,
            )
        })
        .collect();

    // The DeepEP context drop is collective: rank 0 must drop BEFORE joining
    // the expert threads (see the EP8 gate).
    drop(ep4);
    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("ep4 gate rank thread panicked")
            .with_context(|| format!("ep4 gate rank {}", rank + 1))?;
    }
    for (outputs, global_tokens) in outputs?.iter().zip(GLOBAL_TOKEN_BUCKETS) {
        assert_layer_probes(
            &format!("layer6/moe/ep4/g{global_tokens}"),
            outputs,
            MOE_ORACLE_DEEPGEMM_LAYER_PROBES,
            MOE_ORACLE_DEEPGEMM_LAYER_TOL,
            4,
        );
    }
    Ok(())
}

/// The EP4 variant of the gate's prefill-via-decode walk: same decode
/// environment as `oracle::layer::run_layer_prefill`, with the MLP half
/// driven through the DeepGEMM collective chain. Shared with the
/// free-running DP gates (`freerun_ep4`), which re-run it under
/// heterogeneous cross-rank traffic.
pub(super) fn run_layer_prefill_ep4(
    ctx: &DeviceContext,
    w: &crate::layer::Glm52DecoderLayerWeights,
    ep4: &mut Glm52MoeEpRankState<pegainfer_kernels::ops::Glm52Ep4DeepEpAbi>,
    hidden_host: &[bf16],
    oracle_ctx: usize,
    global_tokens: usize,
) -> Result<Vec<f32>> {
    let Glm52LayerMlp::MoeEp8(moe) = &w.mlp else {
        anyhow::bail!("ep4 gate requires the MoeEp8 layer weights");
    };
    // Single-layer slab: one `[MLA | index-K]` page per 64-token block, so
    // the contract/layout strides are the test slab's page stride.
    let num_blocks = oracle_ctx.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE);
    let (mut slab, caches) = crate::model::glm52_test_layer_slab(ctx, num_blocks)?;
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks,
        kv_layer_offset_bytes: 0,
        kv_block_stride_bytes: slab.page_stride,
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
        sm_scale: GLM52_SM_SCALE,
    };
    let index_blocks = oracle_ctx.div_ceil(INDEX_CACHE_BLOCK);
    let index_cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: index_blocks,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_layer_offset_bytes: 0,
        cache_block_stride_bytes: slab.page_stride,
    };

    let block_table_host: Vec<i32> = (0..index_blocks as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(index_blocks)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    let mut cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mut sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mla_sched = Glm52MlaSchedMetadata::new(ctx, contract, w.mla.heads)?;

    let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
        1,
        index_cache_layout,
        index_blocks,
        NUM_SMS,
        oracle_ctx,
    );
    let mut scratch =
        Glm52DecodeScratch::new(ctx, &contract, mqa_shape, crate::config::GLM52_HEADS, false)?;

    let mut outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
    for position in 0..oracle_ctx {
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            scratch.hidden.data_mut(),
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;
        ctx.stream
            .memcpy_htod(&[position as i64], &mut slot_mapping)?;
        ctx.stream
            .memcpy_htod(&[(position + 1) as i32], &mut seq_lens)?;

        let step = Glm52DecodeStep {
            mla_cos: &cos,
            mla_sin: &sin,
            idx_cos: &cos,
            idx_sin: &sin,
            mla_sched: &mla_sched,
            slot_mapping: &slot_mapping,
            block_table: &block_table,
            seq_lens: &seq_lens,
        };
        let mut carry_ready = false;
        pegainfer_kernels::ops::rms_norm_rows_into(
            ctx,
            scratch.hidden.data(),
            &w.input_ln,
            crate::config::GLM52_RMS_EPS,
            HIDDEN,
            1,
            scratch.layer.normed.data_mut(),
        )?;
        glm52_layer_attention_half(
            ctx,
            None,
            w,
            &mut slab,
            caches,
            &step,
            &mut scratch,
            &mut carry_ready,
            0,
            true,
            None,
        )?;
        let route = run_ep_router(ctx, &moe.router, scratch.layer.normed2.data())?;
        let dispatched = glm52_moe_ep_routed_forward(
            ctx,
            ep4,
            &moe.bank,
            Some((scratch.layer.normed2.data(), &route, 1)),
            global_tokens,
        )?;
        ensure!(dispatched, "rank-0 EP4 MoE returned no combined output");
        moe.shared.forward_into(
            ctx,
            scratch.layer.normed2.data(),
            &mut scratch.shared_mlp,
            scratch.layer.shared_out.data_mut(),
        )?;
        add_scaled_bf16_into(
            ctx,
            ep4.combined(),
            crate::config::GLM52_ROUTED_SCALING_FACTOR as f32,
            scratch.layer.shared_out.data(),
            HIDDEN,
            scratch.layer.mlp_out.data_mut(),
        )?;
        glm52_layer_finish(ctx, &mut scratch, 0, false)?;
        let out_host = ctx.stream.clone_dtoh(scratch.hidden.data())?;
        outputs.extend(out_host.iter().map(|v| v.to_f32()));
    }
    Ok(outputs)
}
