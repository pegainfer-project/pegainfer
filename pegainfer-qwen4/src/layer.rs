//! One decoder layer in prefill, stage by stage: the attention hyper-connection
//! mix, the token mixer (Gated DeltaNet on the qwen35 kernels, or gated full
//! attention on the qwen35 hd256 prefill), the scatter back into the stream,
//! then the same around the MoE block.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::ops::GatedNormActivation;
use pegainfer_kernels::ops::GdrChunkwiseScratch35;
use pegainfer_kernels::ops::HcMixScratch;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::attention_prefill_hd256_into;
use pegainfer_kernels::ops::conv1d_prefill_batch_into;
use pegainfer_kernels::ops::gated_delta_rule_prefill_chunkwise_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::hc_mix_into;
use pegainfer_kernels::ops::hc_scatter_add_into;
use pegainfer_kernels::ops::rms_norm_gated_batch_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use crate::moe::MoeTrace;
use crate::moe::moe_prefill;
use crate::weights::DecoderLayer;
use crate::weights::FullAttention;
use crate::weights::GatedDeltaNet;
use crate::weights::Geometry;
use crate::weights::HyperConnection;
use crate::weights::Mixer;

#[cfg(test)]
mod golden_gate;

/// Tokens per page of the attention layers' K/V, the size every FlashInfer
/// caller in the tree uses.
const KV_PAGE: usize = 16;

/// The attention layers' cos/sin tables, `[positions * rotary_dim]` half split,
/// frequencies spaced over the rotated dims as the reference spaces them.
pub struct Rope {
    pub cos: DeviceVec,
    pub sin: DeviceVec,
}

impl Rope {
    pub fn new(ctx: &DeviceContext, g: &Geometry, positions: usize) -> Result<Self> {
        let (cos, sin) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: g.rotary_dim,
                frequency_dim: g.rotary_dim,
                max_seq_len: positions,
                theta: g.rope_theta,
            },
        )?;
        Ok(Self { cos, sin })
    }
}

pub struct GdnTrace {
    pub qkv: HiddenStates,
    pub z: HiddenStates,
    pub b: HiddenStates,
    pub a: HiddenStates,
    pub core: HiddenStates,
    pub normed: HiddenStates,
}

pub struct AttentionTrace {
    pub q_proj: HiddenStates,
    pub k_proj: HiddenStates,
    pub v_proj: HiddenStates,
    /// Q after the `(1 + w)` norm and the partial RoPE, `[rows, heads * head_dim]`.
    pub q_prepped: HiddenStates,
    /// This prompt's K/V in the one layer paged format the kernel reads.
    pub kv: CudaSlice<bf16>,
    pub layout: PagedKvLayout,
    /// The attention output times the sigmoid gate, `o_proj`'s input.
    pub gated: HiddenStates,
}

pub enum MixerTrace {
    Linear(GdnTrace),
    Full(AttentionTrace),
}

/// Every stage of one layer forward, kept for the reference gate.
pub struct LayerTrace {
    pub attn_mixed: HiddenStates,
    pub attn_inject: HiddenStates,
    pub mixer: MixerTrace,
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

/// The Gated DeltaNet block from a zero conv and recurrent state; returns its
/// stages and the `out_proj` output.
fn gdn_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &GatedDeltaNet,
    mixed: &HiddenStates,
) -> Result<(GdnTrace, HiddenStates)> {
    let rows = mixed.seq_len;
    let mut qkv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    gemm_into_checked(ctx, &w.in_proj_qkv, mixed, &mut qkv)?;
    let mut z_proj = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gemm_into_checked(ctx, &w.in_proj_z, mixed, &mut z_proj)?;
    let mut b_proj = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.in_proj_b, mixed, &mut b_proj)?;
    let mut a_proj = HiddenStates::zeros(ctx, g.value_heads, rows)?;
    gemm_into_checked(ctx, &w.in_proj_a, mixed, &mut a_proj)?;

    let mut conv_state = DeviceVec::zeros(ctx, g.qkv_width() * (g.conv_kernel - 1))?;
    let mut qkv_conv = HiddenStates::zeros(ctx, g.qkv_width(), rows)?;
    conv1d_prefill_batch_into(
        ctx,
        &qkv,
        &w.conv_weight,
        &mut conv_state,
        &mut qkv_conv,
        g.conv_kernel,
    );
    let mut state: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(g.value_heads * g.key_dim * g.value_dim)?;
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(ctx, g.value_heads, g.key_dim, g.value_dim, rows)?;
    let mut core = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    gated_delta_rule_prefill_chunkwise_into(
        ctx,
        &qkv_conv,
        &b_proj,
        &a_proj,
        &w.dt_bias,
        &w.a_log,
        &mut state,
        &mut scratch,
        &mut core,
        g.key_heads,
        g.value_heads,
        g.key_dim,
        g.value_dim,
    )?;
    let mut normed = HiddenStates::zeros(ctx, g.value_width(), rows)?;
    rms_norm_gated_batch_into(
        ctx,
        &core,
        &w.norm_weight,
        &z_proj,
        &mut normed,
        g.value_heads,
        g.value_dim,
        g.rms_eps,
        GatedNormActivation::Sigmoid,
    );
    let mut out = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &w.out_proj, &normed, &mut out)?;
    Ok((
        GdnTrace {
            qkv,
            z: z_proj,
            b: b_proj,
            a: a_proj,
            core,
            normed,
        },
        out,
    ))
}

/// Gated full attention over the whole prompt, dense and causal: the reference
/// is QSA, whose selection keeps every causal key while the visible prefix fits
/// its budget, so this equals it up to that length and no further. K/V live in
/// a one layer paged scratch for the prompt alone; returns the stages and the
/// `o_proj` output.
fn attention_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &FullAttention,
    rope: &Rope,
    mixed: &HiddenStates,
) -> Result<(AttentionTrace, HiddenStates)> {
    let rows = mixed.seq_len;
    let covered = rope.cos.len / g.rotary_dim;
    ensure!(
        rows <= covered,
        "the RoPE table covers {covered} positions, the prompt has {rows}"
    );
    let mut q_proj = HiddenStates::zeros(ctx, g.q_width(), rows)?;
    gemm_into_checked(ctx, &w.q_proj, mixed, &mut q_proj)?;
    let mut k_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.k_proj, mixed, &mut k_proj)?;
    let mut v_proj = HiddenStates::zeros(ctx, g.kv_width(), rows)?;
    gemm_into_checked(ctx, &w.v_proj, mixed, &mut v_proj)?;

    let layout = PagedKvLayout::new(1, g.kv_heads, g.head_dim, KV_PAGE);
    let pages = rows.div_ceil(KV_PAGE);
    let kv: CudaSlice<bf16> = ctx.stream.alloc_zeros(pages * layout.page_stride)?;
    let page_ids: Vec<i32> = (0..pages).map(i32::try_from).collect::<Result<_, _>>()?;
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &page_ids,
        rows - KV_PAGE * (pages - 1),
        0,
        rows,
        g.heads,
        g.kv_heads,
        g.head_dim,
        0,
    )?;
    let mut q_prepped = HiddenStates::zeros(ctx, g.attn_width(), rows)?;
    let mut gated = HiddenStates::zeros(ctx, g.attn_width(), rows)?;
    attention_prefill_hd256_into(
        ctx,
        &q_proj,
        &k_proj,
        &v_proj,
        &w.q_norm,
        &w.k_norm,
        &rope.cos,
        &rope.sin,
        &kv,
        &layout,
        0,
        &plan,
        0,
        g.rotary_dim,
        g.rms_eps,
        &mut q_prepped,
        &mut gated,
    )?;
    let mut out = HiddenStates::zeros(ctx, g.hidden, rows)?;
    gemm_into_checked(ctx, &w.o_proj, &gated, &mut out)?;
    Ok((
        AttentionTrace {
            q_proj,
            k_proj,
            v_proj,
            q_prepped,
            kv,
            layout,
            gated,
        },
        out,
    ))
}

/// One whole-sequence pass over a fresh layer state.
pub fn layer_prefill(
    ctx: &DeviceContext,
    g: &Geometry,
    w: &DecoderLayer,
    rope: &Rope,
    stream_in: &HiddenStates,
) -> Result<LayerTrace> {
    let rows = stream_in.seq_len;
    ensure!(rows > 0, "the prompt is empty");
    ensure!(
        stream_in.hidden_dim == g.stream_width(),
        "stream is {} wide, the geometry has {}",
        stream_in.hidden_dim,
        g.stream_width()
    );
    let (attn_mixed, attn_inject) = mix(ctx, g, &w.attn_hc, stream_in)?;
    let (mixer, attn_out) = match &w.mixer {
        Mixer::Linear(gdn) => {
            let (trace, out) = gdn_prefill(ctx, g, gdn, &attn_mixed)?;
            (MixerTrace::Linear(trace), out)
        }
        Mixer::Full(attn) => {
            let (trace, out) = attention_prefill(ctx, g, attn, rope, &attn_mixed)?;
            (MixerTrace::Full(trace), out)
        }
    };
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
        mixer,
        attn_out,
        stream_after_attn,
        mlp_mixed,
        mlp_inject,
        moe,
        stream_out,
    })
}
