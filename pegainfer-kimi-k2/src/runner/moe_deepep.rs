//! DeepEP elastic all-to-all MoE path (TP1/DP8/EP8).
//!
//! Cross-rank token movement uses DeepEP's expanded dispatch: the recv buffer
//! is expert-major with each local expert's segment aligned to the Marlin
//! block size (8), so the Marlin GEMMs read it directly through identity
//! `sorted_token_ids` built on-stream from the dispatch's expert prefix sum
//! ([`kimi_deepep_build_marlin_routing_on_stream`]) — zero gather/scatter
//! copies, zero D2H.
//!
//! # Decode vs prefill
//!
//! Decode (`do_cpu_sync = false`) runs against fixed worst-case buffers
//! allocated once at enable time: the whole layer is host-quiet, which is
//! what makes future CUDA-graph capture possible (#227). Prefill
//! (`do_cpu_sync = true`) spins the CPU on the dispatch counts so the recv
//! buffers can be bounded by the prompt length instead of the 8-rank worst
//! case (~1.9 GB).
//!
//! # Router scale
//!
//! W2 applies the per-slot router weight; combine reduces the weighted
//! expert outputs back to source tokens in bf16. `KIMI_K2_ROUTER_SCALE` is
//! applied at the residual add ([`kimi_residual_add_scaled_bf16`]), matching
//! the NCCL backend's convention.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::DeepEp;
use pegainfer_kernels::ops::DeepEpDispatchScratch;
use pegainfer_kernels::ops::KIMI_K2_EP_WORLD;
use pegainfer_kernels::ops::KIMI_K2_LOCAL_EXPERTS;
use pegainfer_kernels::ops::KIMI_K2_ROUTER_SCALE;
use pegainfer_kernels::ops::KIMI_K2_SHARED_GATE_UP;
use pegainfer_kernels::ops::KimiMarlinInt4ExpertWeights;
use pegainfer_kernels::ops::KimiMarlinRouteWorkspace;
use pegainfer_kernels::ops::KimiMarlinWna16Workspace;
use pegainfer_kernels::ops::KimiRouterBatch;
use pegainfer_kernels::ops::KimiRouterConfig;
use pegainfer_kernels::ops::KimiRouterOutput;
use pegainfer_kernels::ops::deepep_info;
use pegainfer_kernels::ops::kimi_deepep_build_marlin_routing_on_stream;
use pegainfer_kernels::ops::kimi_marlin_w13_swiglu_expanded;
use pegainfer_kernels::ops::kimi_marlin_wna16_expanded_w2_gemm;
use pegainfer_kernels::ops::kimi_marlin_wna16_expanded_w13_gemm;
use pegainfer_kernels::ops::kimi_residual_add_scaled_bf16;
use pegainfer_kernels::ops::kimi_router_noaux_tc_launch;
use pegainfer_kernels::ops::kimi_shared_gate_up_cublaslt_into;
use pegainfer_kernels::ops::kimi_shared_gate_up_cublaslt_supports_batch_size;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::GpuTensor;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::NormWeight;
use pegainfer_kernels::typed_ops;

use super::worker::KimiMoeForwardCache;
use super::worker::KimiWorkerDecodeScratch;
use super::worker::MARLIN_W13_OUT_DIM;
use crate::config::KIMI_K2_EXPERT_INTERMEDIATE;
use crate::config::KIMI_K2_HIDDEN;
use crate::config::KIMI_K2_RMS_NORM_EPS;
use crate::config::KIMI_K2_ROUTED_EXPERTS;
use crate::config::KIMI_K2_TOPK;
use crate::weights::KimiRankExpertMarlinWeights;

/// DeepEP expert alignment == Marlin block size: the property that lets
/// Marlin consume the expanded recv buffer in place.
const DEEPEP_MARLIN_BLOCK: usize = 8;

/// Per-rank prompt-token cap of one prefill dispatch, baked into the shim's
/// prefill kernel specialization. On the DP path every prefill dispatches
/// its whole (uncached) prompt suffix through DeepEP, so this is the DP
/// per-request prompt cap; decode dispatches only batch-many tokens.
pub(super) const DEEPEP_MAX_DISPATCH_TOKENS: usize = 2048;

/// Per-rank DeepEP context plus the decode-path buffers, allocated once at
/// enable time (fixed worst case → crash early on OOM, pointer-stable for
/// future graph capture).
pub(super) struct KimiMoeDeepEpState {
    ep: DeepEp,
    scratch: DeepEpDispatchScratch,
    /// Expanded recv slots `[decode_worst_expanded_tokens, hidden]`;
    /// `seq_len` is pinned at the worst case — the routing metadata's
    /// device-side `num_tokens_post_padded` bounds the actual work.
    recv_x: GpuTensor<KIMI_K2_HIDDEN>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    w13_out: GpuTensor<MARLIN_W13_OUT_DIM>,
    activated: GpuTensor<KIMI_K2_EXPERT_INTERMEDIATE>,
    expert_output: GpuTensor<KIMI_K2_HIDDEN>,
    combined: GpuTensor<KIMI_K2_HIDDEN>,
    route_workspace: KimiMarlinRouteWorkspace,
    marlin_workspace: KimiMarlinWna16Workspace,
}

impl KimiMoeDeepEpState {
    /// Collective: all ranks' worker threads must call concurrently with the
    /// same unique id, device set.
    pub(super) fn new(
        ctx: &DeviceContext,
        unique_id: &[u8; 128],
        num_ranks: usize,
        rank_idx: usize,
    ) -> Result<Self> {
        let info = deepep_info();
        ensure!(
            info.num_ranks as usize == KIMI_K2_EP_WORLD
                && info.num_experts as usize == KIMI_K2_ROUTED_EXPERTS
                && info.num_local_experts as usize == KIMI_K2_LOCAL_EXPERTS
                && info.num_topk as usize == KIMI_K2_TOPK
                && info.hidden as usize == KIMI_K2_HIDDEN,
            "DeepEP shim config does not match Kimi-K2: {info:?}"
        );
        ensure!(
            info.expert_alignment as usize == DEEPEP_MARLIN_BLOCK,
            "DeepEP expert_alignment {} must equal the Marlin block size {}",
            info.expert_alignment,
            DEEPEP_MARLIN_BLOCK
        );
        ensure!(
            info.prefill_max_tokens_per_rank as usize == DEEPEP_MAX_DISPATCH_TOKENS,
            "DeepEP prefill cap {} does not match DEEPEP_MAX_DISPATCH_TOKENS {}",
            info.prefill_max_tokens_per_rank,
            DEEPEP_MAX_DISPATCH_TOKENS
        );
        ensure!(
            num_ranks == KIMI_K2_EP_WORLD,
            "Kimi DeepEP requires {KIMI_K2_EP_WORLD} ranks, got {num_ranks}"
        );

        let ep = DeepEp::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("Kimi rank {rank_idx} DeepEP context create"))?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        let recv_tokens = info.decode_worst_recv_tokens as usize;
        let route_workspace = KimiMarlinRouteWorkspace::new(ctx, expanded, DEEPEP_MARLIN_BLOCK)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            DEEPEP_MARLIN_BLOCK,
        )?;
        Ok(Self {
            ep,
            scratch: DeepEpDispatchScratch::new_decode(ctx)?,
            recv_x: GpuTensor::zeros(ctx, expanded)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_tokens * (KIMI_K2_TOPK + 2))?,
            w13_out: GpuTensor::zeros(ctx, expanded)?,
            activated: GpuTensor::zeros(ctx, expanded)?,
            expert_output: GpuTensor::zeros(ctx, expanded)?,
            combined: GpuTensor::zeros(ctx, info.decode_max_tokens_per_rank as usize)?,
            route_workspace,
            marlin_workspace,
        })
    }

    pub(super) fn ep(&self) -> &DeepEp {
        &self.ep
    }
}

/// Per-prompt prefill buffers, bounded by the prompt suffix length: only the
/// owning rank dispatches real tokens, the other DP ranks send one dummy
/// token each, so this rank can receive at most
/// `ep_max_seq_len + (KIMI_K2_EP_WORLD - 1)` tokens per layer.
pub(super) struct KimiMoeDeepEpPrefill {
    scratch: DeepEpDispatchScratch,
    recv_capacity: usize,
    expanded_capacity: usize,
    recv_x: GpuTensor<KIMI_K2_HIDDEN>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    w13_out: GpuTensor<MARLIN_W13_OUT_DIM>,
    activated: GpuTensor<KIMI_K2_EXPERT_INTERMEDIATE>,
    expert_output: GpuTensor<KIMI_K2_HIDDEN>,
    combined: GpuTensor<KIMI_K2_HIDDEN>,
    route_workspace: KimiMarlinRouteWorkspace,
    marlin_workspace: KimiMarlinWna16Workspace,
}

impl KimiMoeDeepEpPrefill {
    pub(super) fn new(ctx: &DeviceContext, ep_max_seq_len: usize) -> Result<Self> {
        ensure!(
            ep_max_seq_len > 0 && ep_max_seq_len <= DEEPEP_MAX_DISPATCH_TOKENS,
            "Kimi DeepEP prefill seq_len {ep_max_seq_len} out of 1..={DEEPEP_MAX_DISPATCH_TOKENS}"
        );
        let recv_capacity = ep_max_seq_len + (KIMI_K2_EP_WORLD - 1);
        // Every recv'd token expands to at most `topk` slots here; per-expert
        // segment alignment adds at most `alignment - 1` rows per expert.
        let expanded_capacity = (recv_capacity * KIMI_K2_TOPK
            + KIMI_K2_LOCAL_EXPERTS * (DEEPEP_MARLIN_BLOCK - 1))
            .next_multiple_of(DEEPEP_MARLIN_BLOCK);

        let route_workspace =
            KimiMarlinRouteWorkspace::new(ctx, expanded_capacity, DEEPEP_MARLIN_BLOCK)?;
        let marlin_workspace = KimiMarlinWna16Workspace::new(
            ctx,
            route_workspace.max_m_blocks,
            KIMI_K2_HIDDEN,
            DEEPEP_MARLIN_BLOCK,
        )?;
        Ok(Self {
            scratch: DeepEpDispatchScratch::new_prefill(ctx)?,
            recv_capacity,
            expanded_capacity,
            recv_x: GpuTensor::zeros(ctx, expanded_capacity)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded_capacity)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_capacity * (KIMI_K2_TOPK + 2))?,
            w13_out: GpuTensor::zeros(ctx, expanded_capacity)?,
            activated: GpuTensor::zeros(ctx, expanded_capacity)?,
            expert_output: GpuTensor::zeros(ctx, expanded_capacity)?,
            combined: GpuTensor::zeros(ctx, ep_max_seq_len)?,
            route_workspace,
            marlin_workspace,
        })
    }
}

fn layer_marlin_weights(
    expert_kernels: &KimiRankExpertMarlinWeights,
    layer_idx: usize,
) -> Result<KimiMarlinInt4ExpertWeights<'_>> {
    Ok(expert_kernels
        .layers
        .iter()
        .find(|layer| layer.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!("Kimi rank expert Marlin package missing layer {layer_idx}")
        })?
        .as_marlin_weights())
}

/// One MoE decode layer over the post-attention normed hidden state. Pure
/// TP1: every DP rank calls this simultaneously per layer (the dispatch and
/// combine are collective).
pub(super) fn forward_moe_layer_decode_deepep_normed(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    expert_kernels: &KimiRankExpertMarlinWeights,
    scratch: &mut KimiWorkerDecodeScratch,
    dp: &mut KimiMoeDeepEpState,
) -> Result<()> {
    let batch_size = scratch.mla.hidden.seq_len;

    // Shared expert (main stream) + router (aux stream) both consume the
    // post-attention normed hidden state, so start the router as soon as the
    // norm is ready instead of waiting for the shared expert to finish.
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE DeepEP layer {layer_idx} record norm_ready"))?;
    aux_ctx
        .stream
        .wait(&norm_ready)
        .with_context(|| format!("Kimi MoE DeepEP layer {layer_idx} aux wait norm_ready"))?;
    {
        let mut router_output = KimiRouterOutput {
            topk_weight: &mut scratch.router.router_topk_weight.data,
            topk_idx: &mut scratch.router.router_topk_idx.data,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size,
                active_tokens: batch_size,
                padded_tokens: batch_size,
            },
            &scratch.mla.normed,
            &moe.router.gate_weight,
            &moe.router.e_score_correction_bias,
            &mut scratch.router.router_logits.data,
            &mut router_output,
        )?;
    }
    let route_ready = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE DeepEP layer {layer_idx} record route_ready"))?;

    if moe.shared_gate_up_proj.rows == KIMI_K2_SHARED_GATE_UP
        && kimi_shared_gate_up_cublaslt_supports_batch_size(batch_size)
    {
        kimi_shared_gate_up_cublaslt_into(
            ctx,
            &moe.shared_gate_up_proj,
            &scratch.mla.normed,
            &mut scratch.shared_expert.gate_up,
        )?;
    } else {
        typed_ops::gemm_dm_typed_to_hs_graphsafe(
            ctx,
            &moe.shared_gate_up_proj,
            &scratch.mla.normed,
            &mut scratch.shared_expert.gate_up,
        )?;
    }
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

    ctx.stream
        .wait(&route_ready)
        .with_context(|| format!("Kimi MoE DeepEP layer {layer_idx} main wait route_ready"))?;

    dp.ep
        .decode_dispatch(
            ctx,
            &scratch.mla.normed.data,
            &scratch.router.router_topk_idx.data,
            &scratch.router.router_topk_weight.data,
            batch_size,
            &mut dp.scratch,
            &mut dp.recv_x.data,
            &mut dp.recv_topk_weight,
            &mut dp.recv_src_metadata,
        )
        .with_context(|| format!("deepep decode dispatch layer {layer_idx}"))?;

    let layer_weights = layer_marlin_weights(expert_kernels, layer_idx)?;
    let routing = kimi_deepep_build_marlin_routing_on_stream(
        ctx,
        &mut dp.route_workspace,
        &dp.scratch.psum_expert,
        DEEPEP_MARLIN_BLOCK,
        dp.recv_x.seq_len,
    )
    .with_context(|| format!("deepep build Marlin routing layer {layer_idx}"))?;
    kimi_marlin_wna16_expanded_w13_gemm(
        ctx,
        &mut dp.marlin_workspace,
        &routing,
        &dp.recv_x,
        &layer_weights.w13,
        &dp.recv_topk_weight,
        &mut dp.w13_out,
    )?;
    kimi_marlin_w13_swiglu_expanded(
        ctx,
        &dp.w13_out,
        routing.num_tokens_post_padded,
        &mut dp.activated,
    )?;
    kimi_marlin_wna16_expanded_w2_gemm(
        ctx,
        &mut dp.marlin_workspace,
        &routing,
        &dp.activated,
        &layer_weights.w2_down,
        &dp.recv_topk_weight,
        &mut dp.expert_output,
    )?;

    dp.ep
        .decode_combine(
            ctx,
            &dp.expert_output.data,
            &dp.scratch,
            &dp.recv_src_metadata,
            &scratch.router.router_topk_idx.data,
            batch_size,
            &mut dp.combined.data,
        )
        .with_context(|| format!("deepep decode combine layer {layer_idx}"))?;

    kimi_residual_add_scaled_bf16(
        ctx,
        &scratch.mla.hidden,
        &scratch.mla.projected,
        &dp.combined,
        KIMI_K2_ROUTER_SCALE,
        &mut scratch.mla.normed,
    )?;
    std::mem::swap(&mut scratch.mla.hidden, &mut scratch.mla.normed);
    Ok(())
}

/// One MoE prefill layer: the whole prompt suffix dispatched in a single
/// DeepEP collective. All EP ranks must call this simultaneously — padding
/// ranks pass their one dummy token.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_layer_prefill_deepep(
    ctx: &DeviceContext,
    aux_ctx: &DeviceContext,
    ep: &DeepEp,
    layer_idx: usize,
    moe: &KimiMoeForwardCache,
    post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
    expert_kernels: &KimiRankExpertMarlinWeights,
    hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
    next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    pf: &mut KimiMoeDeepEpPrefill,
) -> Result<()> {
    let seq_len = hidden.seq_len;

    typed_ops::rms_norm_into(
        ctx,
        hidden,
        post_attention_norm,
        KIMI_K2_RMS_NORM_EPS,
        normed,
    )?;

    // Shared expert on the main stream (TP1, no all-reduce).
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

    // Router on the aux stream, overlapping the shared expert.
    let norm_ready = ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE DeepEP prefill layer {layer_idx} record norm_ready"))?;
    aux_ctx.stream.wait(&norm_ready).with_context(|| {
        format!("Kimi MoE DeepEP prefill layer {layer_idx} aux wait norm_ready")
    })?;

    let mut router_logits: CudaSlice<f32> = aux_ctx
        .stream
        .alloc_zeros(seq_len * KIMI_K2_ROUTED_EXPERTS)?;
    let mut router_topk_weight: CudaSlice<f32> =
        aux_ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    let mut router_topk_idx: CudaSlice<i32> = aux_ctx.stream.alloc_zeros(seq_len * KIMI_K2_TOPK)?;
    {
        let mut output = KimiRouterOutput {
            topk_weight: &mut router_topk_weight,
            topk_idx: &mut router_topk_idx,
        };
        kimi_router_noaux_tc_launch(
            aux_ctx,
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
    let route_ready = aux_ctx
        .stream
        .record_event(None)
        .with_context(|| format!("Kimi MoE DeepEP prefill layer {layer_idx} record route_ready"))?;
    ctx.stream.wait(&route_ready).with_context(|| {
        format!("Kimi MoE DeepEP prefill layer {layer_idx} main wait route_ready")
    })?;

    ep.prefill_dispatch_send(
        ctx,
        &normed.data,
        &router_topk_idx,
        &router_topk_weight,
        seq_len,
        &mut pf.scratch,
    )
    .with_context(|| format!("deepep prefill dispatch_send layer {layer_idx}"))?;

    // CPU spin on the dispatch counts (the GPU keeps draining the stream);
    // crash early if a skewed routing exceeds the prompt-sized buffers.
    let counts = ep
        .prefill_wait_counts()
        .with_context(|| format!("deepep prefill wait_counts layer {layer_idx}"))?;
    ensure!(
        counts.num_recv_tokens <= pf.recv_capacity
            && counts.num_expanded_tokens <= pf.expanded_capacity,
        "deepep prefill layer {layer_idx} counts exceed capacity: recv {}/{}, expanded {}/{}",
        counts.num_recv_tokens,
        pf.recv_capacity,
        counts.num_expanded_tokens,
        pf.expanded_capacity
    );
    ep.prefill_dispatch_recv(
        ctx,
        counts,
        &pf.scratch,
        &mut pf.recv_x.data,
        &mut pf.recv_topk_weight,
        &mut pf.recv_src_metadata,
    )
    .with_context(|| format!("deepep prefill dispatch_recv layer {layer_idx}"))?;

    let layer_weights = layer_marlin_weights(expert_kernels, layer_idx)?;
    let routing = kimi_deepep_build_marlin_routing_on_stream(
        ctx,
        &mut pf.route_workspace,
        &pf.scratch.psum_expert,
        DEEPEP_MARLIN_BLOCK,
        pf.expanded_capacity,
    )
    .with_context(|| format!("deepep prefill build Marlin routing layer {layer_idx}"))?;
    kimi_marlin_wna16_expanded_w13_gemm(
        ctx,
        &mut pf.marlin_workspace,
        &routing,
        &pf.recv_x,
        &layer_weights.w13,
        &pf.recv_topk_weight,
        &mut pf.w13_out,
    )?;
    kimi_marlin_w13_swiglu_expanded(
        ctx,
        &pf.w13_out,
        routing.num_tokens_post_padded,
        &mut pf.activated,
    )?;
    kimi_marlin_wna16_expanded_w2_gemm(
        ctx,
        &mut pf.marlin_workspace,
        &routing,
        &pf.activated,
        &layer_weights.w2_down,
        &pf.recv_topk_weight,
        &mut pf.expert_output,
    )?;

    ep.prefill_combine(
        ctx,
        &pf.expert_output.data,
        &pf.scratch,
        &pf.recv_src_metadata,
        counts,
        &router_topk_idx,
        seq_len,
        &mut pf.combined.data,
    )
    .with_context(|| format!("deepep prefill combine layer {layer_idx}"))?;

    kimi_residual_add_scaled_bf16(
        ctx,
        hidden,
        &shared_out,
        &pf.combined,
        KIMI_K2_ROUTER_SCALE,
        next_hidden,
    )?;
    std::mem::swap(hidden, next_hidden);
    Ok(())
}
