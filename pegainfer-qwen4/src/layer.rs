//! One `linear_attention` decoder layer in prefill, stage by stage: the
//! attention hyper-connection mix, the Gated DeltaNet block on the qwen35
//! kernels, the scatter back into the stream, then the same around the MoE block.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::GatedNormActivation;
use pegainfer_kernels::ops::GdrChunkwiseScratch35;
use pegainfer_kernels::ops::HcMixScratch;
use pegainfer_kernels::ops::conv1d_prefill_batch_into;
use pegainfer_kernels::ops::gated_delta_rule_prefill_chunkwise_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::hc_mix_into;
use pegainfer_kernels::ops::hc_scatter_add_into;
use pegainfer_kernels::ops::rms_norm_gated_batch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use crate::moe::MoeTrace;
use crate::moe::moe_prefill;
use crate::weights::Geometry;
use crate::weights::HyperConnection;
use crate::weights::LinearLayer;

#[cfg(test)]
mod golden_gate;

/// Every stage of one layer forward, kept for the reference gate.
pub struct LayerTrace {
    pub attn_mixed: HiddenStates,
    pub attn_inject: HiddenStates,
    pub gdn_qkv: HiddenStates,
    pub gdn_z: HiddenStates,
    pub gdn_b: HiddenStates,
    pub gdn_a: HiddenStates,
    pub gdn_core: HiddenStates,
    pub gdn_normed: HiddenStates,
    pub attn_out: HiddenStates,
    pub stream_after_attn: HiddenStates,
    pub mlp_mixed: HiddenStates,
    pub mlp_inject: HiddenStates,
    pub moe: MoeTrace,
    pub stream_out: HiddenStates,
}

fn mix(
    ctx: &DeviceContext,
    g: &Geometry,
    hc: &HyperConnection,
    stream: &HiddenStates,
) -> Result<(HiddenStates, HiddenStates)> {
    let rows = stream.seq_len;
    let mut scratch = HcMixScratch::new(ctx, rows, g.hidden, g.hc_count, g.hc_lowrank, true)?;
    let mut mixed = HiddenStates::zeros(ctx, g.hidden, rows)?;
    let mut inject = HiddenStates::zeros(ctx, g.hc_count, rows)?;
    hc_mix_into(
        ctx,
        &hc.norm,
        &hc.down,
        &hc.up,
        hc.inject.as_ref(),
        stream,
        g.hc_count,
        g.rms_eps,
        &mut scratch,
        &mut mixed,
        Some(&mut inject),
    )?;
    Ok((mixed, inject))
}

/// One whole-sequence pass over a fresh layer state (zero conv and recurrent state).
pub fn linear_layer_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &LinearLayer,
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    let rows = stream_in.seq_len;
    ensure!(
        stream_in.hidden_dim == g.stream_width(),
        "stream is {} wide, the geometry has {}",
        stream_in.hidden_dim,
        g.stream_width()
    );
    let (attn_mixed, attn_inject) = mix(ctx, g, &w.attn_hc, stream_in)?;

    let mut gdn_qkv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    gemm_into_checked(ctx, &w.gdn.in_proj_qkv, &attn_mixed, &mut gdn_qkv)?;
    let mut gdn_z = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gemm_into_checked(ctx, &w.gdn.in_proj_z, &attn_mixed, &mut gdn_z)?;
    let mut gdn_b = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.gdn.in_proj_b, &attn_mixed, &mut gdn_b)?;
    let mut gdn_a = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.gdn.in_proj_a, &attn_mixed, &mut gdn_a)?;

    let mut conv_state = DeviceVec::zeros(ctx, g.qkv_width() * (g.conv_kernel - 1))?;
    let mut qkv_conv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    conv1d_prefill_batch_into(
        ctx,
        &gdn_qkv,
        &w.gdn.conv_weight,
        &mut conv_state,
        &mut qkv_conv,
        g.conv_kernel,
    );
    let mut state: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(g.value_heads * g.key_dim * g.value_dim)?;
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(ctx, g.value_heads, g.key_dim, g.value_dim, rows)?;
    let mut gdn_core = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gated_delta_rule_prefill_chunkwise_into(
        ctx,
        &qkv_conv,
        &gdn_b,
        &gdn_a,
        &w.gdn.dt_bias,
        &w.gdn.a_log,
        &mut state,
        &mut scratch,
        &mut gdn_core,
        g.key_heads,
        g.value_heads,
        g.key_dim,
        g.value_dim,
    )?;
    let mut gdn_normed = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    rms_norm_gated_batch_into(
        ctx,
        &gdn_core,
        &w.gdn.norm_weight,
        &gdn_z,
        &mut gdn_normed,
        g.value_heads,
        g.value_dim,
        g.rms_eps,
        GatedNormActivation::Sigmoid,
    );
    let mut attn_out = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &w.gdn.out_proj, &gdn_normed, &mut attn_out)?;
    let mut stream_after_attn = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    hc_scatter_add_into(
        ctx,
        &attn_out,
        &attn_inject,
        stream_in,
        &mut stream_after_attn,
    )?;

    let (mlp_mixed, mlp_inject) = mix(ctx, g, &w.mlp_hc, &stream_after_attn)?;
    let moe = moe_prefill(ctx, g, &w.moe, &mlp_mixed)?;
    let mut stream_out = HiddenStates::zeros(ctx, g.stream_width(), rows)?;
    hc_scatter_add_into(
        ctx,
        &moe.out,
        &mlp_inject,
        &stream_after_attn,
        &mut stream_out,
    )?;

    Ok(LayerTrace {
        attn_mixed,
        attn_inject,
        gdn_qkv,
        gdn_z,
        gdn_b,
        gdn_a,
        gdn_core,
        gdn_normed,
        attn_out,
        stream_after_attn,
        mlp_mixed,
        mlp_inject,
        moe,
        stream_out,
    })
}
