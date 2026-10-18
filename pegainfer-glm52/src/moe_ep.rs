//! GLM5.2 routed-MoE decode: the DeepEP dispatch/combine collectives around
//! SM100 DeepGEMM masked grouped GEMMs, generic over
//! the shim instantiation (EP4 = 4 ranks × 64 local experts, EP8 = 8 × 32 —
//! every shape below comes from the shim's `DeepEpInfo`, so a new EP width
//! is a new instantiation, not new code).
//!
//! Every rank enters the collective per MoE layer with the protocol-max
//! `global_tokens`; the expert GEMMs consume the DeepEP aligned receive
//! layout through a compact masked-layout bridge:
//!
//! ```text
//! dispatch(x bf16, global topk)        # collective; recv = expert-major
//!   → psum → masked row metadata
//!   → UE8M0 fp8 quant → DeepGEMM W13
//!   → silu(gate)·up + UE8M0 fp8 quant → DeepGEMM W2
//!   → route_weight + remap to aligned slots
//!   → combine                          # collective; sums slots per token
//! ```
//!
//! All buffers are fixed at startup and every launch is host-quiet, preserving
//! whole-step CUDA Graph capture. Metadata traps on a cross-rank token-count
//! disagreement or a per-expert row count beyond the protocol maximum.

use std::mem::size_of;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ffi::DeepEpInfo;
use pegainfer_kernels::ops::DeepEpAbi;
use pegainfer_kernels::ops::DeepEpBase;
use pegainfer_kernels::ops::DeepEpDispatchScratch;
use pegainfer_kernels::ops::GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT;
use pegainfer_kernels::ops::GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT;
use pegainfer_kernels::ops::Glm52DeepEpAbi;
use pegainfer_kernels::ops::Glm52DeepGemmGroupedFp8Kind;
use pegainfer_kernels::ops::Glm52Ep4DeepEpAbi;
use pegainfer_kernels::ops::Glm52Ep16DeepEpAbi;
use pegainfer_kernels::ops::Glm52Ep32DeepEpAbi;
use pegainfer_kernels::ops::Glm52Ep64DeepEpAbi;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::glm52_deepgemm_sm100_grouped_fp8_metadata_launch;
use pegainfer_kernels::ops::glm52_deepgemm_sm100_masked_grouped_fp8_launch;
use pegainfer_kernels::ops::glm52_deepgemm_sm100_masked_out_to_aligned_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_masked_launch;
use pegainfer_kernels::ops::glm52_fp8_scale_pack_ue8m0_launch;
use pegainfer_kernels::ops::glm52_silu_and_mul_per_token_group_quant_bf16_masked_launch;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_DENSE_LAYERS;
use crate::config::GLM52_LAYERS;
use crate::model::GLM52_MAX_STEP_ROWS;
use crate::moe_decode::EXPERTS;
use crate::moe_decode::Glm52MoeExpertBank;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::QUANT_GROUP;
use crate::moe_decode::RoutedTopk;
use crate::moe_decode::TOPK;
use crate::moe_decode::W2_K;
use crate::moe_decode::W2_N;
use crate::moe_decode::W13_N;

/// Extra post-weight-probe bytes introduced by the SM100 DeepGEMM path over
/// the retired weight-only chain already covered by the empirical 5 GiB
/// reserve. A negative delta is not credited back: the reserve is a safety
/// floor for unmodelled allocator/graph costs, not allocatable KV capacity.
pub(crate) fn glm52_deepgemm_vram_charge_bytes(
    topology: crate::Glm52MoeTopo,
    native_mtp: bool,
) -> Result<usize> {
    let expert_bank_layers = GLM52_LAYERS - GLM52_DENSE_LAYERS + usize::from(native_mtp);
    match topology {
        crate::Glm52MoeTopo::Ep4 => deepgemm_vram_charge_for::<Glm52Ep4DeepEpAbi>(
            topology.expected_ep_size(),
            expert_bank_layers,
        ),
        crate::Glm52MoeTopo::Ep8 => deepgemm_vram_charge_for::<Glm52DeepEpAbi>(
            topology.expected_ep_size(),
            expert_bank_layers,
        ),
        crate::Glm52MoeTopo::Ep16 => deepgemm_vram_charge_for::<Glm52Ep16DeepEpAbi>(
            topology.expected_ep_size(),
            expert_bank_layers,
        ),
        crate::Glm52MoeTopo::Ep32 => deepgemm_vram_charge_for::<Glm52Ep32DeepEpAbi>(
            topology.expected_ep_size(),
            expert_bank_layers,
        ),
        crate::Glm52MoeTopo::Ep64 => deepgemm_vram_charge_for::<Glm52Ep64DeepEpAbi>(
            topology.expected_ep_size(),
            expert_bank_layers,
        ),
        crate::Glm52MoeTopo::Tp4 => Ok(0),
    }
}

fn deepgemm_vram_charge_for<A: DeepEpAbi>(
    num_ranks: usize,
    expert_bank_layers: usize,
) -> Result<usize> {
    let info = A::info();
    ensure!(
        info.num_ranks as usize == num_ranks
            && info.num_experts as usize == EXPERTS
            && info.num_topk as usize == TOPK
            && info.hidden as usize == HIDDEN,
        "GLM5.2 DeepGEMM VRAM ledger does not match the DeepEP ABI: {info:?}"
    );
    let n_local = info.num_local_experts as usize;
    ensure!(
        n_local * num_ranks == EXPERTS,
        "GLM5.2 DeepGEMM VRAM ledger cannot partition {EXPERTS} experts across {num_ranks} ranks"
    );

    // The weight probe sees checkpoint f32 block scales. Model build replaces
    // them with DeepGEMM's output-row-expanded UE8M0 layout, one local bank
    // per sparse target layer plus the optional native-MTP layer.
    let checkpoint_scale_bytes_per_layer = n_local
        * ((W13_N / QUANT_GROUP) * (HIDDEN / QUANT_GROUP)
            + (W2_N / QUANT_GROUP) * (W2_K / QUANT_GROUP))
        * size_of::<f32>();
    let deepgemm_scale_bytes_per_layer =
        n_local * ((HIDDEN / 512) * W13_N + (W2_K / 512) * W2_N) * size_of::<i32>();

    let expanded = info.decode_worst_expanded_tokens as usize;
    let masked_rows = n_local * deepgemm_masked_cap(num_ranks);

    // Path-specific buffers only; recv_x, route metadata, zeros, expert_out,
    // combined output, and DeepEP's own context/scratch are unchanged and
    // cancel out. The old tile budget formula is retained here solely as the
    // baseline represented by the measured reserve.
    let old_max_tiles = n_local + (num_ranks * GLM52_MAX_STEP_ROWS * TOPK).div_ceil(8);
    let weight_only_scratch_bytes = 2 * old_max_tiles * size_of::<i32>()
        + size_of::<i32>()
        + expanded * W13_N * size_of::<bf16>()
        + expanded * W2_K * size_of::<bf16>();
    let deepgemm_scratch_bytes = (n_local + 1) * size_of::<i64>()
        + n_local * size_of::<i32>()
        + expanded * size_of::<i32>()
        + masked_rows * HIDDEN * size_of::<u8>()
        + masked_rows * (HIDDEN / QUANT_GROUP) * size_of::<f32>()
        + masked_rows * (HIDDEN / 512) * size_of::<i32>()
        + masked_rows * W13_N * size_of::<bf16>()
        + masked_rows * W2_K * size_of::<u8>()
        + masked_rows * (W2_K / QUANT_GROUP) * size_of::<f32>()
        + masked_rows * (W2_K / 512) * size_of::<i32>()
        + masked_rows * W2_N * size_of::<bf16>();

    let weight_only_bytes =
        expert_bank_layers * checkpoint_scale_bytes_per_layer + weight_only_scratch_bytes;
    let deepgemm_bytes =
        expert_bank_layers * deepgemm_scale_bytes_per_layer + deepgemm_scratch_bytes;
    Ok(deepgemm_bytes.saturating_sub(weight_only_bytes))
}

/// Per-rank DeepEP context plus every buffer the routed-expert chain touches,
/// allocated once at startup at worst-case capacity (pointer-stable for
/// graph capture, no per-layer allocator traffic — the EP8 discipline).
pub(crate) struct Glm52MoeEpRankState<A: DeepEpAbi> {
    ep: DeepEpBase<A>,
    scratch: DeepEpDispatchScratch,
    info: DeepEpInfo,
    masked_cap: usize,
    num_sms: usize,
    recv_x: CudaSlice<bf16>,
    recv_topk_weight: CudaSlice<f32>,
    recv_src_metadata: CudaSlice<i32>,
    /// Dispatch inputs for token-less expert ranks (num_tokens = 0 still
    /// requires valid pointers).
    zero_x: CudaSlice<bf16>,
    zero_topk_idx: CudaSlice<i32>,
    zero_topk_weight: CudaSlice<f32>,
    expert_offsets: CudaSlice<i64>,
    masked_m: CudaSlice<i32>,
    row_map: CudaSlice<i32>,
    w13_act: CudaSlice<u8>,
    w13_scale_f32: CudaSlice<f32>,
    w13_scale: CudaSlice<i32>,
    /// W13 gate|up output in `[local_experts, masked_cap, W13_N]`.
    w13_out: CudaSlice<bf16>,
    w2_act: CudaSlice<u8>,
    w2_scale_f32: CudaSlice<f32>,
    w2_scale: CudaSlice<i32>,
    w2_out: CudaSlice<bf16>,
    /// W2 output remapped into the aligned recv slots `decode_combine` reads.
    expert_out: CudaSlice<bf16>,
    /// The combined routed output for this rank's source tokens (row-major
    /// `[tokens, HIDDEN]`), sized at the shim's per-rank decode cap.
    combined: CudaSlice<bf16>,
}

impl<A: DeepEpAbi> Glm52MoeEpRankState<A> {
    /// The routed output rows written by the last dispatching
    /// [`glm52_moe_ep_routed_forward`] call (valid only when that call
    /// returned `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        &self.combined
    }

    /// Collective: all ranks' worker threads must call concurrently with the
    /// same unique id, device set.
    pub(crate) fn new(
        ctx: &DeviceContext,
        unique_id: &[u8; 128],
        num_ranks: usize,
        rank_idx: usize,
    ) -> Result<Self> {
        let info = A::info();
        ensure!(
            info.num_experts as usize == EXPERTS
                && info.num_topk as usize == TOPK
                && info.hidden as usize == HIDDEN
                && info.expert_alignment as usize == GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT,
            "GLM5.2 DeepEP shim config does not match the model: {info:?}"
        );
        ensure!(
            num_ranks == info.num_ranks as usize,
            "GLM5.2 DeepEP requires {} ranks, got {num_ranks}",
            info.num_ranks
        );
        ensure!(
            info.num_local_experts as usize * info.num_ranks as usize == EXPERTS,
            "GLM5.2 DeepEP shim local experts do not partition the routed set: {info:?}"
        );
        let ep = DeepEpBase::<A>::new(unique_id, num_ranks, rank_idx)
            .with_context(|| format!("GLM5.2 rank {rank_idx} DeepEP context create"))?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        let recv_tokens = info.decode_worst_recv_tokens as usize;
        let n_local = info.num_local_experts as usize;
        let masked_cap = deepgemm_masked_cap(num_ranks);
        let masked_rows = n_local * masked_cap;
        let num_sms = ctx.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )? as usize;
        ensure!(
            matches!(num_sms, 148 | 152),
            "GLM5.2 SM100 DeepGEMM supports B200/GB300 SM counts {{148,152}}, got {num_sms}"
        );
        Ok(Self {
            ep,
            scratch: DeepEpDispatchScratch::new_decode_with(ctx, &info)?,
            info,
            masked_cap,
            num_sms,
            recv_x: ctx.stream.alloc_zeros(expanded * HIDDEN)?,
            recv_topk_weight: ctx.stream.alloc_zeros(expanded)?,
            recv_src_metadata: ctx.stream.alloc_zeros(recv_tokens * (TOPK + 2))?,
            zero_x: ctx.stream.alloc_zeros(HIDDEN)?,
            zero_topk_idx: ctx.stream.alloc_zeros(TOPK)?,
            zero_topk_weight: ctx.stream.alloc_zeros(TOPK)?,
            expert_offsets: ctx.stream.alloc_zeros(n_local + 1)?,
            masked_m: ctx.stream.alloc_zeros(n_local)?,
            row_map: ctx.stream.alloc_zeros(expanded)?,
            w13_act: ctx.stream.alloc_zeros(masked_rows * HIDDEN)?,
            w13_scale_f32: ctx
                .stream
                .alloc_zeros(masked_rows * (HIDDEN / QUANT_GROUP))?,
            w13_scale: ctx.stream.alloc_zeros(masked_rows * (HIDDEN / 512))?,
            w13_out: ctx.stream.alloc_zeros(masked_rows * W13_N)?,
            w2_act: ctx.stream.alloc_zeros(masked_rows * W2_K)?,
            w2_scale_f32: ctx.stream.alloc_zeros(masked_rows * (W2_K / QUANT_GROUP))?,
            w2_scale: ctx.stream.alloc_zeros(masked_rows * (W2_K / 512))?,
            w2_out: ctx.stream.alloc_zeros(masked_rows * W2_N)?,
            expert_out: ctx.stream.alloc_zeros(expanded * W2_N)?,
            combined: ctx
                .stream
                .alloc_zeros(info.decode_max_tokens_per_rank as usize * HIDDEN)?,
        })
    }
}

fn deepgemm_masked_cap(num_ranks: usize) -> usize {
    (num_ranks * GLM52_MAX_STEP_ROWS).next_multiple_of(GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT)
}

/// One MoE layer's routed contribution. Every rank must enter the collective
/// simultaneously per layer with the same `global_tokens` bound.
pub(crate) fn glm52_moe_ep_routed_forward<A: DeepEpAbi>(
    ctx: &DeviceContext,
    state: &mut Glm52MoeEpRankState<A>,
    bank: &Glm52MoeExpertBank,
    token: Option<(&CudaSlice<bf16>, &RoutedTopk, usize)>,
    global_tokens: usize,
) -> Result<bool> {
    let n_local = state.info.num_local_experts as usize;
    ensure!(
        bank.n_experts() == n_local,
        "GLM5.2 EP MoE needs the {n_local}-expert rank-local bank, got {}",
        bank.n_experts()
    );
    let expanded = state.info.decode_worst_expanded_tokens as usize;
    let num_tokens = token.map_or(0, |(_, _, t)| t);
    ensure!(
        token.is_none() || num_tokens > 0,
        "GLM5.2 EP MoE dispatching rank must pass a positive token count"
    );
    ensure!(
        global_tokens >= num_tokens && global_tokens > 0,
        "GLM5.2 EP MoE global_tokens {global_tokens} must be positive and >= local tokens {num_tokens}"
    );
    // Startup scratch covers the protocol's max global token count.
    let max_global_tokens = state.info.num_ranks as usize * GLM52_MAX_STEP_ROWS;
    ensure!(
        global_tokens <= max_global_tokens,
        "GLM5.2 EP MoE global_tokens {global_tokens} exceeds the protocol cap {max_global_tokens}"
    );
    let expanded_rows = global_tokens * TOPK;
    let bound_rows = expanded.min(
        expanded_rows + (GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT - 1) * expanded_rows.min(n_local),
    );

    // Collective dispatch: bf16 token rows → expert-major aligned recv slots.
    {
        let (x, topk_idx, topk_weight) = match token {
            Some((normed, route, _)) => (normed, &route.topk_idx, &route.topk_weight),
            None => (&state.zero_x, &state.zero_topk_idx, &state.zero_topk_weight),
        };
        state.ep.decode_dispatch(
            ctx,
            x,
            topk_idx,
            topk_weight,
            num_tokens,
            &mut state.scratch,
            &mut state.recv_x,
            &mut state.recv_topk_weight,
            &mut state.recv_src_metadata,
        )?;
    }

    glm52_deepgemm_sm100_grouped_fp8_metadata_launch(
        ctx,
        n_local,
        bound_rows,
        state.masked_cap,
        &state.scratch.psum_expert,
        &mut state.expert_offsets,
        &mut state.masked_m,
        &mut state.row_map,
    )?;

    glm52_fp8_per_token_group_quant_bf16_masked_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: bound_rows,
            width: HIDDEN,
            group_size: QUANT_GROUP,
        },
        n_local,
        state.masked_cap,
        &state.recv_x,
        &mut state.w13_act,
        &mut state.w13_scale_f32,
        &state.expert_offsets,
        n_local,
        &state.row_map,
    )?;
    glm52_fp8_scale_pack_ue8m0_launch(
        ctx,
        n_local,
        HIDDEN / QUANT_GROUP,
        state.masked_cap,
        &state.w13_scale_f32,
        &mut state.w13_scale,
    )?;
    glm52_deepgemm_sm100_masked_grouped_fp8_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W13,
        n_local,
        state.masked_cap,
        state.num_sms,
        &state.w13_act,
        &state.w13_scale,
        &bank.w13_weight,
        &bank.w13_scale,
        &state.masked_m,
        &mut state.w13_out,
    )?;

    glm52_silu_and_mul_per_token_group_quant_bf16_masked_launch(
        ctx,
        Glm52MoeQuantShape {
            rows: bound_rows,
            width: W2_K,
            group_size: QUANT_GROUP,
        },
        n_local,
        state.masked_cap,
        &state.w13_out,
        &mut state.w2_act,
        &mut state.w2_scale_f32,
        &state.expert_offsets,
        n_local,
        &state.row_map,
    )?;
    glm52_fp8_scale_pack_ue8m0_launch(
        ctx,
        n_local,
        W2_K / QUANT_GROUP,
        state.masked_cap,
        &state.w2_scale_f32,
        &mut state.w2_scale,
    )?;
    glm52_deepgemm_sm100_masked_grouped_fp8_launch(
        ctx,
        Glm52DeepGemmGroupedFp8Kind::W2,
        n_local,
        state.masked_cap,
        state.num_sms,
        &state.w2_act,
        &state.w2_scale,
        &bank.w2_weight,
        &bank.w2_scale,
        &state.masked_m,
        &mut state.w2_out,
    )?;
    glm52_deepgemm_sm100_masked_out_to_aligned_launch(
        ctx,
        n_local,
        state.masked_cap,
        W2_N,
        &state.w2_out,
        &state.masked_m,
        &state.expert_offsets,
        &state.recv_topk_weight,
        &mut state.expert_out,
    )?;

    // Collective combine: weighted expert outputs → per-source-token sums.
    let topk_idx = match token {
        Some((_, route, _)) => &route.topk_idx,
        None => &state.zero_topk_idx,
    };
    state.ep.decode_combine(
        ctx,
        &state.expert_out,
        &state.scratch,
        &state.recv_src_metadata,
        topk_idx,
        num_tokens,
        &mut state.combined,
    )?;

    Ok(token.is_some())
}

/// One rank's EP MoE state: one DeepEP shim per EP width around the shared
/// SM100 DeepGEMM expert chain.
pub(crate) enum Glm52MoeEpState {
    Ep4(Box<Glm52MoeEpRankState<Glm52Ep4DeepEpAbi>>),
    Ep8(Box<Glm52MoeEpRankState<Glm52DeepEpAbi>>),
    Ep16(Box<Glm52MoeEpRankState<Glm52Ep16DeepEpAbi>>),
    Ep32(Box<Glm52MoeEpRankState<Glm52Ep32DeepEpAbi>>),
    Ep64(Box<Glm52MoeEpRankState<Glm52Ep64DeepEpAbi>>),
}

impl Glm52MoeEpState {
    pub(crate) fn routed_forward(
        &mut self,
        ctx: &DeviceContext,
        bank: &Glm52MoeExpertBank,
        token: Option<(&CudaSlice<bf16>, &RoutedTopk, usize)>,
        global_tokens: usize,
    ) -> Result<bool> {
        match self {
            Self::Ep4(state) => glm52_moe_ep_routed_forward(ctx, state, bank, token, global_tokens),
            Self::Ep8(state) => glm52_moe_ep_routed_forward(ctx, state, bank, token, global_tokens),
            Self::Ep16(state) => {
                glm52_moe_ep_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::Ep32(state) => {
                glm52_moe_ep_routed_forward(ctx, state, bank, token, global_tokens)
            }
            Self::Ep64(state) => {
                glm52_moe_ep_routed_forward(ctx, state, bank, token, global_tokens)
            }
        }
    }

    /// The routed output rows written by the last dispatching
    /// `routed_forward` call (valid only when that call returned `true`).
    pub(crate) fn combined(&self) -> &CudaSlice<bf16> {
        match self {
            Self::Ep4(state) => state.combined(),
            Self::Ep8(state) => state.combined(),
            Self::Ep16(state) => state.combined(),
            Self::Ep32(state) => state.combined(),
            Self::Ep64(state) => state.combined(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::deepgemm_masked_cap;
    use super::glm52_deepgemm_vram_charge_bytes;
    use crate::Glm52MoeTopo;

    #[test]
    fn masked_cap_covers_every_supported_ep_width() {
        // ranks x GLM52_MAX_STEP_ROWS (96 since #817), aligned up to 128.
        assert_eq!(deepgemm_masked_cap(4), 384);
        assert_eq!(deepgemm_masked_cap(8), 768);
        assert_eq!(deepgemm_masked_cap(16), 1536);
        assert_eq!(deepgemm_masked_cap(32), 3072);
        assert_eq!(deepgemm_masked_cap(64), 6144);
    }

    #[test]
    fn post_weight_vram_charge_tracks_topology_and_native_mtp() {
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep4, false).expect("EP4 charge"),
            1_984_001_028
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep8, false).expect("EP8 charge"),
            1_272_383_620
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep16, false).expect("EP16 charge"),
            841_490_500
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep32, false).expect("EP32 charge"),
            475_088_932
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep64, false).expect("EP64 charge"),
            392_500_244
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Tp4, false).expect("TP4 charge"),
            0
        );
        assert_eq!(
            glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep4, true).expect("EP4 MTP charge")
                - glm52_deepgemm_vram_charge_bytes(Glm52MoeTopo::Ep4, false).expect("EP4 charge"),
            18_284_544,
            "native MTP adds one converted expert bank but reuses the EP scratch"
        );
    }
}
