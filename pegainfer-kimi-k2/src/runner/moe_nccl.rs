//! NCCL all-gather / reduce-scatter MoE backend (the TP-replicated path).
//!
//! Sibling of [`super::moe_deepep`]: both implement the same Kimi-K2 MoE layer
//! (shared expert + router + routed Marlin experts + scaled residual add). They
//! differ in how routed tokens cross ranks — which in turn dictates the routing
//! layout and on-rank buffer format:
//!
//! | Stage              | NCCL backend (this file)                     | DeepEP backend ([`super::moe_deepep`]) |
//! |--------------------|----------------------------------------------|------------------------------------|
//! | Routing layout     | every rank routes the *replicated* hidden    | `dispatch` ships tokens expert-major |
//! | Local experts      | Marlin GEMM over all routed slots on-rank    | Marlin GEMM over received slots      |
//! | Cross-rank combine | NCCL `reduce_scatter` / `all_reduce` of F32  | `dispatch` / `combine` collectives   |
//!
//! Because the hidden state is TP-replicated, every rank computes the *same*
//! routed contribution and the collective sums duplicates away. This requires a
//! TP `Comm` (`comm.is_some()`); the TP1 / expert-parallel case must use the
//! DeepEP path instead — the routed entry points here `ensure!` a `Comm`.
//!
//! Two entry points, mirroring the call paths in [`super::worker`]:
//! - [`forward_moe_layer_decode_normed_into`] — CUDA-graph-safe batch decode,
//!   router overlapped on the aux stream, combine via `reduce_scatter`.
//! - [`forward_moe_layer_batch_into`] — single-prompt prefill, fresh
//!   allocations, combine via bulk `all_reduce`.

use anyhow::Context;
use anyhow::Result;
use cudarc::nccl::ReduceOp;
use cudarc::nccl::safe::Comm;
use pegainfer_kernels::ops::KIMI_K2_EP_WORLD;
use pegainfer_kernels::ops::KIMI_K2_ROUTER_SCALE;
use pegainfer_kernels::ops::KimiMarlinRouteWorkspace;
use pegainfer_kernels::ops::KimiMarlinWna16Workspace;
use pegainfer_kernels::ops::KimiRouterBatch;
use pegainfer_kernels::ops::KimiRouterConfig;
use pegainfer_kernels::ops::KimiRouterOutput;
use pegainfer_kernels::ops::kimi_add_f32_bf16_to_bf16;
use pegainfer_kernels::ops::kimi_marlin_sum_topk_rows_f32;
use pegainfer_kernels::ops::kimi_marlin_w13_swiglu;
use pegainfer_kernels::ops::kimi_marlin_wna16_w2_gemm;
use pegainfer_kernels::ops::kimi_marlin_wna16_w13_gemm;
use pegainfer_kernels::ops::kimi_moe_marlin_align_block_size;
use pegainfer_kernels::ops::kimi_residual_add_scaled_f32;
use pegainfer_kernels::ops::kimi_router_noaux_tc_launch;
use pegainfer_kernels::ops::repeat_f32_for_reduce_scatter_into;
use pegainfer_kernels::ops::scale_f32_in_place;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::GpuTensor;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::NormWeight;
use pegainfer_kernels::typed_ops;

use super::worker::KimiMoeForwardCache;
use super::worker::KimiWorkerDecodeScratch;
use super::worker::MARLIN_W13_OUT_DIM;
use super::worker::all_reduce_f32_in_place;
use super::worker::kimi_marlin_block_size;
use super::worker::maybe_all_reduce_hidden_via_f32_in_place;
use super::worker::reduce_scatter_f32_hidden_into;
use crate::config::KIMI_K2_EXPERT_INTERMEDIATE;
use crate::config::KIMI_K2_HIDDEN;
use crate::config::KIMI_K2_RMS_NORM_EPS;
use crate::config::KIMI_K2_ROUTED_EXPERTS;
use crate::config::KIMI_K2_TOPK;
use crate::weights::KimiRankExpertMarlinWeights;

/// CUDA-graph-safe batch decode MoE. The post-attention norm has already been
/// applied into `scratch.mla.normed`; this records the norm-ready event before
/// fanning the router out onto the aux stream.
pub(super) fn forward_moe_layer_decode_normed_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record fused norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE layer {layer_idx} aux wait fused norm_ready"))?;
    forward_moe_layer_decode_normed_after_event_into(
        ctx,
        aux_ctx,
        comm,
        layer_idx,
        moe,
        expert_kernels,
        scratch,
    )
}

fn forward_moe_layer_decode_normed_after_event_into(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
) -> Result<()> {
    let seq_len = scratch.mla.hidden.seq_len;
    typed_ops::gemm_dm_typed_to_hs_graphsafe(
        ctx,
        &moe.shared_gate_up_proj,
        &scratch.mla.normed,
        &mut scratch.shared_expert.gate_up,
    )?;
    typed_ops::silu_mul_hs_fused_into(
        ctx,
        &scratch.shared_expert.gate_up,
        &mut scratch.shared_expert.activated,
    )?;
    typed_ops::gemm_dm_hs_to_typed_graphsafe(
        ctx,
        &moe.shared_down_proj,
        &scratch.shared_expert.activated,
        &mut scratch.mla.projected,
    )?;
    maybe_all_reduce_hidden_via_f32_in_place(
        ctx,
        &mut scratch.mla.projected,
        &mut scratch.comm.hidden_allreduce_f32,
        comm,
    )?;

    // Router + routed experts (aux stream)
    {
        let mut router_output = KimiRouterOutput {
            topk_weight: &mut scratch.router.router_topk_weight.data,
            topk_idx: &mut scratch.router.router_topk_idx.data,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            &scratch.mla.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut scratch.router.router_logits.data,
            &mut router_output,
        )?;
    }
    let routing = kimi_moe_marlin_align_block_size(
        aux_ctx,
        &mut scratch.marlin_route_workspace,
        &scratch.router.router_topk_idx.data,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin.w13_out.data)?;
    kimi_marlin_wna16_w13_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.mla.normed,
        &layer_weights.w13,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.w13_out,
    )?;
    kimi_marlin_w13_swiglu(
        aux_ctx,
        &scratch.marlin.w13_out,
        &mut scratch.marlin.activated,
    )?;
    aux_ctx
        .stream
        .memset_zeros(&mut scratch.marlin.expert_output.data)?;
    kimi_marlin_wna16_w2_gemm(
        aux_ctx,
        &mut scratch.marlin_workspace,
        &routing,
        &scratch.marlin.activated,
        &layer_weights.w2_down,
        &scratch.router.router_topk_weight.data,
        &mut scratch.marlin.expert_output,
    )?;
    kimi_marlin_sum_topk_rows_f32(
        aux_ctx,
        &scratch.marlin.expert_output,
        seq_len,
        &mut scratch.comm.routed_out_f32,
    )?;
    repeat_f32_for_reduce_scatter_into(
        aux_ctx,
        &scratch.comm.routed_out_f32,
        &mut scratch.comm.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_EP_WORLD,
    )?;

    let routed_local_done = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE layer {layer_idx} record routed_local_done"))?;
    ctx.stream
        .wait(&routed_local_done)
        .with_context(|| format!("Kimi MoE layer {layer_idx} main wait routed_local_done"))?;
    let nccl_comm = comm.ok_or_else(|| {
        anyhow::anyhow!("NCCL MoE routed path requires TP comm (use DeepEP for TP1)")
    })?;
    reduce_scatter_f32_hidden_into(
        &scratch.comm.routed_reduce_scatter_send_f32,
        seq_len * KIMI_K2_EP_WORLD,
        KIMI_K2_HIDDEN,
        &mut scratch.comm.routed_out_f32,
        seq_len,
        KIMI_K2_EP_WORLD,
        nccl_comm,
    )?;

    kimi_residual_add_scaled_f32(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &scratch.comm.routed_out_f32,
        KIMI_K2_ROUTER_SCALE,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

/// Single-prompt prefill MoE: fresh per-call allocations, combine via a single
/// bulk `all_reduce` over the `seq_len * hidden` F32 routed buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_batch_into(
    ctx: &DeviceContext,
    comm: Option<&Comm>,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
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
    let mut shared_gate_up = HiddenStates::zeros(ctx, moe.shared_gate_up_proj.rows, seq_len)?;
    typed_ops::gemm_dm_typed_to_hs(ctx, &moe.shared_gate_up_proj, normed, &mut shared_gate_up)?;
    let mut shared_activated = HiddenStates::zeros(ctx, moe.shared_down_proj.cols, seq_len)?;
    typed_ops::silu_mul_hs_fused_into(ctx, &shared_gate_up, &mut shared_activated)?;
    let mut shared_out = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
    typed_ops::gemm_dm_hs_to_typed(
        ctx,
        &moe.shared_down_proj,
        &shared_activated,
        &mut shared_out,
    )?;
    if let Some(comm) = comm {
        comm.all_reduce_in_place(&mut shared_out.data, &ReduceOp::Sum)
            .map_err(|err| {
                anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
            })?;
    }

    let mut router_logits = ctx.stream.alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_topk_weight = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    let mut router_topk_idx = ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    {
        let mut output = KimiRouterOutput {
            topk_weight: &mut router_topk_weight,
            topk_idx: &mut router_topk_idx,
        };
        kimi_router_noaux_tc_launch(
            ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: seq_len,
                active_tokens: seq_len,
                padded_tokens: seq_len,
            },
            normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut router_logits,
            &mut output,
        )?;
    }

    let marlin_block_size = kimi_marlin_block_size(seq_len);
    let mut route_workspace = KimiMarlinRouteWorkspace::new(ctx, seq_len, marlin_block_size)?;
    let routing = kimi_moe_marlin_align_block_size(
        ctx,
        &mut route_workspace,
        &router_topk_idx,
        seq_len,
        seq_len,
        expert_kernels.local_expert_range.start,
    )?;
    let layer_weights = expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights();

    let mut marlin_workspace = KimiMarlinWna16Workspace::new(
        ctx,
        routing.max_m_blocks,
        KIMI_K2_HIDDEN,
        marlin_block_size,
    )?;
    let mut w13_out = GpuTensor::<MARLIN_W13_OUT_DIM>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_wna16_w13_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        normed,
        &layer_weights.w13,
        &router_topk_weight,
        &mut w13_out,
    )?;
    let mut activated = GpuTensor::<KIMI_K2_EXPERT_INTERMEDIATE>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_w13_swiglu(ctx, &w13_out, &mut activated)?;
    let mut expert_output = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, routing.route_elems)?;
    kimi_marlin_wna16_w2_gemm(
        ctx,
        &mut marlin_workspace,
        &routing,
        &activated,
        &layer_weights.w2_down,
        &router_topk_weight,
        &mut expert_output,
    )?;

    let mut routed_out_f32 = ctx.stream.alloc_zeros(seq_len * KIMI_K2_HIDDEN)?;
    kimi_marlin_sum_topk_rows_f32(ctx, &expert_output, seq_len, &mut routed_out_f32)?;
    let nccl_comm = comm.ok_or_else(|| {
        anyhow::anyhow!("NCCL MoE batch routed path requires TP comm (use DeepEP for TP1)")
    })?;
    all_reduce_f32_in_place(&mut routed_out_f32, nccl_comm)?;
    scale_f32_in_place(
        ctx,
        &mut routed_out_f32,
        seq_len * KIMI_K2_HIDDEN,
        KIMI_K2_ROUTER_SCALE,
    )?;
    pegainfer_kernels::typed_pipeline! {
        ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS;
        add(hidden, &shared_out => next_hidden);
    }
    kimi_add_f32_bf16_to_bf16(ctx, &routed_out_f32, next_hidden, hidden)?;
    Ok(())
}
