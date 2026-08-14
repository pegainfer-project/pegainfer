//! Kimi-K3 batched-decode MoE expert chain: the step-time kernels around the
//! FP8xFP4 masked grouped GEMM (see [`super::deepgemm`]).
//!
//! Production routing goes through the fused MegaMoE kernel (see
//! [`super::mega_moe`]). This chain is the numerics anchor behind it — the path
//! the certified f32 reference is checked against, and the one the fused
//! kernel's golden gate is A/B'd against — and it runs single-rank only.
//!
//! Per MoE layer (all routed experts local):
//!
//! 1. [`k3_moe_local_route_metadata_launch`] — per-expert row counts plus the
//!    expanded-entry -> masked-slot map.
//! 2. [`k3_moe_gather_fp8_quant_masked_launch`] — local gather fused with the
//!    W13 A-operand quant, then [`k3_fp8_scale_pack_ue8m0_launch`] for the SFA.
//! 3. masked grouped GEMM W13 -> `[groups, masked_cap, 2 * inter]` bf16.
//! 4. [`k3_situ_and_mul_fp8_quant_masked_launch`] — the K3 situ activation and
//!    the W2 A-operand quant, then [`k3_fp8_scale_pack_ue8m0_launch`] again.
//! 5. masked grouped GEMM W2 -> `[groups, masked_cap, hidden]` bf16.
//! 6. [`k3_moe_weighted_combine_launch`] — weighted scatter back to token-major
//!    hidden states.
//!
//! An *entry* is one expanded `(token, topk-slot)` pair at index
//! `token * topk + slot`, and that order is the deterministic order the whole
//! chain uses: it fixes which masked row an entry lands in, and it fixes the
//! combine's accumulation order (f32, no atomics, one bf16 round), so replays
//! are bit-reproducible. `topk_idx` carries GLOBAL expert ids; an entry is
//! active exactly when `topk_idx - local_expert_base` falls in `[0, groups)` —
//! that is how padded batch rows are excluded. Every consumer re-reads
//! `topk_idx` and skips inactive entries, so a stale slot map is never silently
//! consumed.
//!
//! Nothing here allocates, reads back to the host, or varies its launch
//! geometry with device state, so the chain is CUDA-graph capturable.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// K elements per activation quant group (and per UE8M0 scale).
pub const K3_MOE_QUANT_GROUP: usize = 128;

/// Routing geometry shared by every kernel in the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K3MoeRouteShape {
    /// Token rows in this step (padded batch rows included).
    pub tokens: usize,
    /// Router slots per token (16 for K3).
    pub topk: usize,
    /// Rank-local expert groups; must match the masked GEMM instantiation.
    pub groups: usize,
    /// Rows reserved per expert in the masked layout; multiple of 128.
    pub masked_cap: usize,
}

impl K3MoeRouteShape {
    /// Expanded `(token, topk-slot)` pairs.
    #[must_use]
    pub const fn entries(self) -> usize {
        self.tokens * self.topk
    }

    /// Rows in the masked expert-major layout.
    #[must_use]
    pub const fn masked_rows(self) -> usize {
        self.groups * self.masked_cap
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.tokens > 0
                && self.topk > 0
                && self.groups > 0
                && self.masked_cap > 0
                && self.masked_cap.is_multiple_of(K3_MOE_QUANT_GROUP),
            "K3 MoE route shape needs tokens/topk/groups > 0 and masked_cap a positive multiple of {K3_MOE_QUANT_GROUP}, got {self:?}"
        );
        Ok(())
    }
}

/// Local routing metadata: `masked_m[groups]` real rows per expert and
/// `slot_map[tokens * topk]` (`expert * masked_cap + rank`, or `-1` when the
/// entry is inactive).
///
/// An expert claiming more than `masked_cap` entries traps device-side rather
/// than aliasing the next expert's rows.
pub fn k3_moe_local_route_metadata_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    topk_idx: &CudaSlice<i32>,
    masked_m: &mut CudaSlice<i32>,
    slot_map: &mut CudaSlice<i32>,
) -> Result<()> {
    windowed_route_metadata_launch(ctx, shape, 0, topk_idx, masked_m, slot_map)
}

/// The windowed form the kernel actually takes: `topk_idx` carries global
/// expert ids and an entry is active exactly when `topk_idx -
/// local_expert_base` lands in `[0, shape.groups)`. Private, because the only
/// window the chain uses is the whole expert set — the parameter is kept
/// because it is one subtract and it keeps the active-entry test in one place,
/// which every consumer re-derives.
fn windowed_route_metadata_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    local_expert_base: usize,
    topk_idx: &CudaSlice<i32>,
    masked_m: &mut CudaSlice<i32>,
    slot_map: &mut CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        topk_idx.len() >= shape.entries()
            && masked_m.len() >= shape.groups
            && slot_map.len() >= shape.entries(),
        "K3 MoE route metadata buffers too small for {shape:?}: topk_idx {}, masked_m {}, slot_map {}",
        topk_idx.len(),
        masked_m.len(),
        slot_map.len()
    );
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_map.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_moe_local_route_metadata_cuda(
            idx_ptr as *const i32,
            masked_ptr as *mut i32,
            slot_ptr as *mut i32,
            shape.tokens as i32,
            shape.topk as i32,
            shape.groups as i32,
            shape.masked_cap as i32,
            i32::try_from(local_expert_base)?,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MoE route metadata launch failed: {err}"))
}

/// Local gather fused with the W13 A-operand quant: token-major bf16 latents
/// `[tokens, hidden]` -> fp8 e4m3 `[groups * masked_cap, hidden]` plus MN-major
/// UE8M0 f32 group scales `[groups, hidden / 128, masked_cap]`.
///
/// Pass the scales through [`k3_fp8_scale_pack_ue8m0_launch`] before the GEMM.
#[allow(clippy::too_many_arguments)]
pub fn k3_moe_gather_fp8_quant_masked_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    hidden: usize,
    latent: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    slot_map: &CudaSlice<i32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    windowed_gather_fp8_quant_masked_launch(
        ctx, shape, hidden, 0, latent, topk_idx, slot_map, output, scales,
    )
}

/// The windowed form (see `windowed_route_metadata_launch`).
#[allow(clippy::too_many_arguments)]
fn windowed_gather_fp8_quant_masked_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    hidden: usize,
    local_expert_base: usize,
    latent: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    slot_map: &CudaSlice<i32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        hidden > 0 && hidden.is_multiple_of(K3_MOE_QUANT_GROUP),
        "K3 MoE gather quant needs hidden a positive multiple of {K3_MOE_QUANT_GROUP}, got {hidden}"
    );
    ensure!(
        latent.len() >= shape.tokens * hidden
            && topk_idx.len() >= shape.entries()
            && slot_map.len() >= shape.entries()
            && output.len() >= shape.masked_rows() * hidden
            && scales.len() >= shape.groups * (hidden / K3_MOE_QUANT_GROUP) * shape.masked_cap,
        "K3 MoE gather quant buffers too small for {shape:?} (hidden={hidden}): latent {}, topk_idx {}, slot_map {}, out {}, scales {}",
        latent.len(),
        topk_idx.len(),
        slot_map.len(),
        output.len(),
        scales.len()
    );
    let (latent_ptr, _latent_guard) = latent.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_map.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_moe_gather_fp8_quant_masked_cuda(
            latent_ptr as *const ffi::Half,
            idx_ptr as *const i32,
            slot_ptr as *const i32,
            out_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.tokens as i32,
            shape.topk as i32,
            hidden as i32,
            shape.groups as i32,
            shape.masked_cap as i32,
            i32::try_from(local_expert_base)?,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MoE gather FP8 quant launch failed: {err}"))
}

/// K3 situ activation over the masked gate|up rows, then the W2 A-operand
/// quant.
///
/// `gate_up` is the W13 masked output `[groups * masked_cap, 2 * inter]` bf16
/// with gate in the first `inter` columns and up in the second. The activation
/// is computed in f32 over those bf16 values (the GEMM's bf16 store *is* the
/// rounding step):
/// `act = 4 * tanh(g / 4) * sigmoid(g) * 25 * tanh(u / 25)`. Router weights are
/// applied later, in [`k3_moe_weighted_combine_launch`].
#[allow(clippy::too_many_arguments)]
pub fn k3_situ_and_mul_fp8_quant_masked_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    inter: usize,
    gate_up: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    slot_map: &CudaSlice<i32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    situ_and_mul_fp8_quant_windowed_launch(
        ctx, shape, inter, 0, gate_up, topk_idx, slot_map, output, scales,
    )
}

/// The windowed form (see `windowed_route_metadata_launch`).
#[allow(clippy::too_many_arguments)]
fn situ_and_mul_fp8_quant_windowed_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    inter: usize,
    local_expert_base: usize,
    gate_up: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    slot_map: &CudaSlice<i32>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        inter > 0 && inter.is_multiple_of(K3_MOE_QUANT_GROUP),
        "K3 MoE situ quant needs inter a positive multiple of {K3_MOE_QUANT_GROUP}, got {inter}"
    );
    ensure!(
        gate_up.len() >= shape.masked_rows() * 2 * inter
            && topk_idx.len() >= shape.entries()
            && slot_map.len() >= shape.entries()
            && output.len() >= shape.masked_rows() * inter
            && scales.len() >= shape.groups * (inter / K3_MOE_QUANT_GROUP) * shape.masked_cap,
        "K3 MoE situ quant buffers too small for {shape:?} (inter={inter}): gate_up {}, topk_idx {}, slot_map {}, out {}, scales {}",
        gate_up.len(),
        topk_idx.len(),
        slot_map.len(),
        output.len(),
        scales.len()
    );
    let (gate_up_ptr, _gate_up_guard) = gate_up.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_map.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_situ_and_mul_fp8_quant_masked_cuda(
            gate_up_ptr as *const ffi::Half,
            idx_ptr as *const i32,
            slot_ptr as *const i32,
            out_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.tokens as i32,
            shape.topk as i32,
            inter as i32,
            shape.groups as i32,
            shape.masked_cap as i32,
            i32::try_from(local_expert_base)?,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MoE situ FP8 quant launch failed: {err}"))
}

/// f32 MN-major group scales `[groups, scale_cols, cap]` -> the packed UE8M0
/// i32 SFA tensor `[groups, scale_cols / 4, cap]` the masked GEMM reads.
///
/// `scale_cols` is `k / 128` and must be a multiple of 4. Dense full-cover
/// pass: every packed word is rewritten, so a stale exponent byte from an
/// earlier step never reaches the GEMM.
pub fn k3_fp8_scale_pack_ue8m0_launch(
    ctx: &DeviceContext,
    groups: usize,
    scale_cols: usize,
    cap: usize,
    scales: &CudaSlice<f32>,
    packed: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && cap > 0 && scale_cols > 0 && scale_cols.is_multiple_of(4),
        "K3 FP8 scale pack needs groups/cap > 0 and scale_cols a positive multiple of 4, got groups={groups}, scale_cols={scale_cols}, cap={cap}"
    );
    ensure!(
        scales.len() >= groups * scale_cols * cap
            && packed.len() >= groups * (scale_cols / 4) * cap,
        "K3 FP8 scale pack buffers too small for {groups} groups (scale_cols={scale_cols}, cap={cap}): scales {}, packed {}",
        scales.len(),
        packed.len()
    );
    let (scale_ptr, _scale_guard) = scales.device_ptr(&ctx.stream);
    let (packed_ptr, _packed_guard) = packed.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_fp8_scale_pack_ue8m0_cuda(
            scale_ptr as *const f32,
            packed_ptr as *mut i32,
            groups as i32,
            scale_cols as i32,
            cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 FP8 scale pack launch failed: {err}"))
}

/// Weighted combine: masked W2 rows `[groups * masked_cap, hidden]` bf16 ->
/// token-major `[tokens, hidden]` bf16.
///
/// Each output element accumulates its token's active entries in topk-slot
/// order in f32 with no atomics and rounds to bf16 once, so the result is
/// deterministic across runs and graph replays. Tokens with no active entry
/// land an exact zero, so the output is fully covered every step.
#[allow(clippy::too_many_arguments)]
pub fn k3_moe_weighted_combine_launch(
    ctx: &DeviceContext,
    shape: K3MoeRouteShape,
    hidden: usize,
    expert_out: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    slot_map: &CudaSlice<i32>,
    topk_weight: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        hidden > 0,
        "K3 MoE weighted combine needs a positive hidden, got {hidden}"
    );
    ensure!(
        expert_out.len() >= shape.masked_rows() * hidden
            && topk_idx.len() >= shape.entries()
            && slot_map.len() >= shape.entries()
            && topk_weight.len() >= shape.entries()
            && out.len() >= shape.tokens * hidden,
        "K3 MoE weighted combine buffers too small for {shape:?} (hidden={hidden}): expert_out {}, topk_idx {}, slot_map {}, topk_weight {}, out {}",
        expert_out.len(),
        topk_idx.len(),
        slot_map.len(),
        topk_weight.len(),
        out.len()
    );
    let (expert_ptr, _expert_guard) = expert_out.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_map.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_moe_weighted_combine_cuda(
            expert_ptr as *const ffi::Half,
            idx_ptr as *const i32,
            slot_ptr as *const i32,
            weight_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            shape.tokens as i32,
            shape.topk as i32,
            hidden as i32,
            shape.groups as i32,
            shape.masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MoE weighted combine launch failed: {err}"))
}
