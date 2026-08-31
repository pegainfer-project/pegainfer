//! The routed-expert half of a Gemma 4 MoE layer.
//!
//! The experts stay packed as NVFP4 on the card and the GEMM reads them that
//! way: widening a layer's experts into bf16 would cost four times the
//! checkpoint in traffic every step, which is what the Marlin kernel exists to
//! avoid. What the loader rewrites once — the weight order, the block-scale
//! encoding — is described in `weights::StackedProjection`.
//!
//! Routing is planned on the device. Grouping the picks into the kernel's
//! fixed-width blocks is a scan the host could do, but it would have to read
//! the picks back first, and a stream synchronize per layer costs more than
//! the scan is worth.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::MarlinDispatch;
use pegainfer_kernels::ops::MoeAlignScratch;
use pegainfer_kernels::ops::gemma4_marlin_nvfp4_moe;
use pegainfer_kernels::ops::gemma4_moe_router_topk_into;
use pegainfer_kernels::ops::gemma4_moe_sum_topk_into;
use pegainfer_kernels::ops::marlin_moe_align_block_size;

use crate::config::MoeConfig;
use crate::layer::LayerGeometry;
use crate::weights::Gemma4Moe;

// Decode's M tile; a thin step pads the least.
const DECODE_BLOCK: usize = 16;
// The `thread_m_blocks = 4` tile: one weight stripe serves four times the rows.
const PREFILL_BLOCK: usize = 64;
// A 1024-row step's routes: below it the coarse block loses first-token
// time (measured down to 256 slots), at it the two blocks tie on a lone
// prompt, and the mixed steps a busy server prefills through gain.
const PREFILL_MIN_SLOTS: usize = 1024 * 8;

fn marlin_block(slots: usize) -> usize {
    if slots >= PREFILL_MIN_SLOTS {
        PREFILL_BLOCK
    } else {
        DECODE_BLOCK
    }
}

// The shared launcher wants at most four lock words per SM and a non-null
// fp32 staging buffer; Gemma's whole-column schedule reduces each output
// element inside one CTA, so neither is ever written. The lock count covers
// any device up to 1024 SMs.
const MARLIN_LOCKS: usize = 4 * 1024;
const MARLIN_STAGING_ELEMS: usize = 1024;

struct MoeScratchSizes {
    slots: usize,
    blocks: usize,
    padded: usize,
}

fn checked_mul(left: usize, right: usize, quantity: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: MoE scratch {quantity} overflows usize"))
}

fn checked_add(left: usize, right: usize, quantity: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: MoE scratch {quantity} overflows usize"))
}

fn scratch_sizes(max_rows: usize, moe: &MoeConfig) -> Result<MoeScratchSizes> {
    let slots = checked_mul(max_rows, moe.top_k, "routed slots")?;
    let blocks = checked_add(
        slots.div_ceil(DECODE_BLOCK),
        moe.num_experts,
        "expert blocks",
    )?;
    let narrow_padded = checked_mul(blocks, DECODE_BLOCK, "decode padded rows")?;
    let padded = if slots >= PREFILL_MIN_SLOTS {
        let expert_padding = checked_mul(moe.num_experts, PREFILL_BLOCK, "prefill expert padding")?;
        let coarse_padded = checked_add(slots, expert_padding, "prefill padded rows")?;
        narrow_padded.max(coarse_padded)
    } else {
        narrow_padded
    };
    Ok(MoeScratchSizes {
        slots,
        blocks,
        padded,
    })
}

/// Buffers one [`moe_into`] call needs, sized for the widest step the server
/// admits.
pub(crate) struct MoeScratch {
    max_rows: usize,
    router_in: HiddenStates,
    logits: HiddenStates,
    index: CudaSlice<i32>,
    weight: CudaSlice<f32>,
    sorted_token_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    padded_total: CudaSlice<i32>,
    locks: CudaSlice<i32>,
    c_tmp: CudaSlice<f32>,
    moe_in: HiddenStates,
    routed_gate: HiddenStates,
    routed_up: HiddenStates,
    routed_act: HiddenStates,
    routed_down: HiddenStates,
    expert_out: HiddenStates,
    expert_offsets: CudaSlice<u32>,
    expert_cursor: CudaSlice<u32>,
}

impl MoeScratch {
    pub(crate) fn new(ctx: &DeviceContext, geom: &LayerGeometry, max_rows: usize) -> Result<Self> {
        let moe = geom
            .moe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: no MoE scratch without a routed config"))?;
        let hidden = |rows| HiddenStates::zeros(ctx, geom.hidden_size, rows);
        let narrow = |rows| HiddenStates::zeros(ctx, moe.intermediate_size, rows);
        let sizes = scratch_sizes(max_rows, moe)?;
        Ok(Self {
            max_rows,
            router_in: hidden(max_rows)?,
            logits: HiddenStates::zeros(ctx, moe.num_experts, max_rows)?,
            index: ctx.stream.alloc_zeros::<i32>(sizes.slots)?,
            weight: ctx.stream.alloc_zeros::<f32>(sizes.slots)?,
            sorted_token_ids: ctx.stream.alloc_zeros::<i32>(sizes.padded)?,
            expert_ids: ctx.stream.alloc_zeros::<i32>(sizes.blocks)?,
            padded_total: ctx.stream.alloc_zeros::<i32>(1)?,
            locks: ctx.stream.alloc_zeros::<i32>(MARLIN_LOCKS)?,
            c_tmp: ctx.stream.alloc_zeros::<f32>(MARLIN_STAGING_ELEMS)?,
            moe_in: hidden(max_rows)?,
            routed_gate: narrow(sizes.slots)?,
            routed_up: narrow(sizes.slots)?,
            routed_act: narrow(sizes.slots)?,
            routed_down: hidden(sizes.slots)?,
            expert_out: hidden(max_rows)?,
            expert_offsets: ctx.stream.alloc_zeros::<u32>(moe.num_experts + 1)?,
            expert_cursor: ctx.stream.alloc_zeros::<u32>(1)?,
        })
    }

    fn set_rows(&mut self, rows: usize, top_k: usize) -> Result<()> {
        ensure!(
            rows <= self.max_rows,
            "MoE scratch holds {} rows, not {rows}",
            self.max_rows
        );
        for buf in [
            &mut self.router_in,
            &mut self.logits,
            &mut self.moe_in,
            &mut self.expert_out,
        ] {
            buf.seq_len = rows;
        }
        for buf in [
            &mut self.routed_gate,
            &mut self.routed_up,
            &mut self.routed_act,
            &mut self.routed_down,
        ] {
            buf.seq_len = rows * top_k;
        }
        Ok(())
    }
}

/// The routed branch of the feed forward block: `out` receives the sum of the
/// dense output and the expert output, each through its own norm, which the
/// caller still has to put through the shared post-feedforward norm.
///
/// `residual` is the block input — both the router and the experts read it,
/// not the dense branch's output.
pub(crate) fn moe_into(
    ctx: &DeviceContext,
    moe: &Gemma4Moe,
    geom: &LayerGeometry,
    residual: &HiddenStates,
    dense: &HiddenStates,
    scratch: &mut MoeScratch,
    out: &mut HiddenStates,
) -> Result<()> {
    let config = geom
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: a routed layer needs a routed config"))?;
    let rows = residual.seq_len;
    scratch.set_rows(rows, config.top_k)?;
    // The GEMM reads the projection's extent from the weights that were
    // loaded, not from the config that described them, so a checkpoint whose
    // experts disagree with its own config stops here rather than reading
    // past a buffer.
    let width = moe.gate.rows;
    let depth = moe.gate.values;
    ensure!(
        width == config.intermediate_size
            && depth == geom.hidden_size
            && moe.up.rows == width
            && moe.up.values == depth
            && moe.down.rows == depth
            && moe.down.values == width,
        "Gemma 4: the routed experts are {width} x {depth}, but the config says {} x {}",
        config.intermediate_size,
        geom.hidden_size
    );

    // Both residual norms share one reduction. The router branch still
    // rounds before its `hidden ** -0.5` scalar multiply.
    ops::rms_norm_batch_dual_into(
        ctx,
        residual,
        &moe.router_scale,
        &moe.pre_feedforward_layernorm_2,
        geom.rms_norm_eps,
        (geom.hidden_size as f32).powf(-0.5),
        &mut scratch.router_in,
        &mut scratch.moe_in,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &moe.router_proj,
        0,
        config.num_experts,
        &scratch.router_in,
        &mut scratch.logits,
    )?;
    gemma4_moe_router_topk_into(
        ctx,
        &scratch.logits,
        &moe.router_per_expert_scale,
        config.top_k,
        &mut scratch.index,
        &mut scratch.weight,
    )?;

    let slots = rows * config.top_k;
    let block = marlin_block(slots);
    marlin_moe_align_block_size(
        ctx,
        &scratch.index,
        rows,
        config.top_k,
        config.num_experts,
        block,
        &mut MoeAlignScratch {
            sorted_token_ids: &mut scratch.sorted_token_ids,
            expert_ids: &mut scratch.expert_ids,
            num_tokens_post_padded: &mut scratch.padded_total,
            expert_offsets: &mut scratch.expert_offsets,
            expert_cursor: &mut scratch.expert_cursor,
        },
    )?;

    let gather = MarlinDispatch {
        sorted_token_ids: &scratch.sorted_token_ids,
        expert_ids: &scratch.expert_ids,
        num_tokens_post_padded: &scratch.padded_total,
        topk_weights: &scratch.weight,
        block_size: block,
        top_k: config.top_k,
        mul_topk_weights: false,
    };
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.moe_in,
        &moe.gate.qweight,
        &moe.gate.scales,
        &moe.gate.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &gather,
        rows,
        width,
        depth,
        &mut scratch.routed_gate,
    )?;
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.moe_in,
        &moe.up.qweight,
        &moe.up.scales,
        &moe.up.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &gather,
        rows,
        width,
        depth,
        &mut scratch.routed_up,
    )?;
    ops::gelu_tanh_mul_batch_into(
        ctx,
        &scratch.routed_gate,
        &scratch.routed_up,
        &mut scratch.routed_act,
    )?;

    // The rows are already one per pick, so the second projection reads them
    // straight through and is the one that applies the router's weights.
    let combine = MarlinDispatch {
        top_k: 1,
        mul_topk_weights: true,
        ..gather
    };
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.routed_act,
        &moe.down.qweight,
        &moe.down.scales,
        &moe.down.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &combine,
        slots,
        depth,
        width,
        &mut scratch.routed_down,
    )?;
    gemma4_moe_sum_topk_into(
        ctx,
        &scratch.routed_down,
        config.top_k,
        &mut scratch.expert_out,
    )?;

    ops::dual_rms_norm_add_batch_into(
        ctx,
        dense,
        &moe.post_feedforward_layernorm_1,
        &scratch.expert_out,
        &moe.post_feedforward_layernorm_2,
        geom.rms_norm_eps,
        out,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
