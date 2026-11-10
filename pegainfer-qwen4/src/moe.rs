//! The sparse MoE block in prefill: the router, the routed experts as one
//! cuBLAS GEMM pair per expert over its token slots, the shared expert and the
//! block sum.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::Qwen4SharedBranchScratch;
use pegainfer_kernels::ops::gather_hidden_tokens_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::gemm_rows_span_at_into_checked;
use pegainfer_kernels::ops::qwen4_moe_combine_into;
use pegainfer_kernels::ops::qwen4_moe_route_into;
use pegainfer_kernels::ops::qwen4_moe_route_meta_into;
use pegainfer_kernels::ops::qwen4_moe_shared_branch_into;
use pegainfer_kernels::ops::qwen4_moe_weighted_combine_into;
use pegainfer_kernels::ops::silu_mul_fused_batch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use crate::weights::Geometry;
use crate::weights::MoeBlock;

/// Every stage of one block forward, kept for the reference gate.
pub struct MoeTrace {
    pub router_logits: HiddenStates,
    pub routing_weights: CudaSlice<bf16>,
    pub selected_experts: CudaSlice<i32>,
    pub expert_out: HiddenStates,
    pub shared_out: HiddenStates,
    pub shared_gate_logit: HiddenStates,
    pub out: HiddenStates,
}

pub fn moe_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &MoeBlock,
    hidden: &HiddenStates,
) -> Result<MoeTrace> {
    let tokens = hidden.seq_len;
    let pairs = tokens * g.top_k;
    let geometry = g.moe();

    let mut router_logits = HiddenStates::zeros(ctx, g.experts, tokens)?;
    gemm_into_checked(ctx, &w.router, hidden, &mut router_logits)?;
    let mut routing_weights: CudaSlice<bf16> = ctx.stream.alloc_zeros(pairs)?;
    let mut selected_experts: CudaSlice<i32> = ctx.stream.alloc_zeros(pairs)?;
    qwen4_moe_route_into(
        ctx,
        &geometry,
        g.norm_topk_prob,
        &router_logits,
        &mut routing_weights,
        &mut selected_experts,
    )?;

    let mut indptr: CudaSlice<i32> = ctx.stream.alloc_zeros(g.experts + 1)?;
    let mut gather_rows: CudaSlice<i32> = ctx.stream.alloc_zeros(pairs)?;
    let mut route_slot: CudaSlice<i32> = ctx.stream.alloc_zeros(pairs)?;
    qwen4_moe_route_meta_into(
        ctx,
        &geometry,
        &selected_experts,
        tokens,
        &mut indptr,
        &mut gather_rows,
        &mut route_slot,
    )?;
    let indptr_host: Vec<i32> = ctx.stream.clone_dtoh(&indptr)?;
    ctx.sync()?;

    let mut x_sorted = HiddenStates::zeros(ctx, g.hidden, pairs)?;
    gather_hidden_tokens_into(ctx, hidden, &gather_rows, pairs, &mut x_sorted)?;
    let gate_up_width = 2 * g.moe_intermediate;
    let mut gate_up = HiddenStates::zeros(ctx, gate_up_width, pairs)?;
    for expert in 0..g.experts {
        let (first, end) = (
            indptr_host[expert] as usize,
            indptr_host[expert + 1] as usize,
        );
        gemm_rows_span_at_into_checked(
            ctx,
            &w.gate_up,
            expert * gate_up_width,
            gate_up_width,
            &x_sorted,
            first,
            &mut gate_up,
            first,
            end - first,
        )?;
    }
    let mut activated = HiddenStates::zeros(ctx, g.moe_intermediate, pairs)?;
    silu_mul_fused_batch_into(ctx, &gate_up, &mut activated)?;
    let mut down_out = HiddenStates::zeros(ctx, g.hidden, pairs)?;
    for expert in 0..g.experts {
        let (first, end) = (
            indptr_host[expert] as usize,
            indptr_host[expert + 1] as usize,
        );
        gemm_rows_span_at_into_checked(
            ctx,
            &w.down,
            expert * g.hidden,
            g.hidden,
            &activated,
            first,
            &mut down_out,
            first,
            end - first,
        )?;
    }
    let mut expert_out = HiddenStates::zeros(ctx, g.hidden, tokens)?;
    qwen4_moe_weighted_combine_into(
        ctx,
        &geometry,
        &down_out,
        &routing_weights,
        &route_slot,
        &mut expert_out,
    )?;

    let mut shared_scratch = Qwen4SharedBranchScratch::new(ctx, &geometry, tokens)?;
    let mut shared_out = HiddenStates::zeros(ctx, g.hidden, tokens)?;
    qwen4_moe_shared_branch_into(
        ctx,
        &geometry,
        hidden,
        &w.shared_gate_proj,
        &w.shared_up_proj,
        &w.shared_down_proj,
        &mut shared_scratch,
        &mut shared_out,
    )?;
    let mut shared_gate_logit = HiddenStates::zeros(ctx, 1, tokens)?;
    gemm_into_checked(ctx, &w.shared_gate, hidden, &mut shared_gate_logit)?;
    let mut out = HiddenStates::zeros(ctx, g.hidden, tokens)?;
    qwen4_moe_combine_into(
        ctx,
        &geometry,
        &shared_gate_logit.data,
        &expert_out,
        &shared_out,
        &mut out,
    )?;

    Ok(MoeTrace {
        router_logits,
        routing_weights,
        selected_experts,
        expert_out,
        shared_out,
        shared_gate_logit,
        out,
    })
}
