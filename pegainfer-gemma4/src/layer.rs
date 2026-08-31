//! The per-layer pieces the serving path assembles: each family's geometry,
//! the global family's proportional RoPE tables, and the attention epilogue
//! both kinds share.
//!
//! The graph these serve is the HF reference's, with the constants a
//! from-the-paper implementation gets wrong: the four norm sites are all
//! norm-then-add (sandwich, not the fused add-then-norm shape), attention is
//! unscaled (`scaling = 1.0` in the reference — not `head_dim**-0.5`), V takes
//! a weightless RMS norm and no RoPE (on global layers V is `k_proj`'s raw
//! output, forked before `k_norm`), RoPE rotates the full head width (the
//! global family's partiality lives in its tables), and `layer_scalar`
//! multiplies the layer output after both residual adds.

use anyhow::Context as _;
use anyhow::Result;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Gemma4Config;
use crate::config::MoeConfig;
use crate::moe::MoeScratch;
use crate::weights::Gemma4Layer;

/// The geometry a layer runs at, read off the validated config — the local
/// and global kinds differ only in head width and KV head count.
pub(crate) struct LayerGeometry {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) moe: Option<MoeConfig>,
}

impl LayerGeometry {
    pub(crate) fn local_of(config: &Gemma4Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
            moe: config.moe,
        }
    }

    pub(crate) fn global_of(config: &Gemma4Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_global_key_value_heads,
            head_dim: config.global_head_dim,
            rms_norm_eps: config.rms_norm_eps,
            moe: config.moe,
        }
    }
}

/// Cos/sin tables for the global layers' proportional RoPE, in the same
/// `[pos * head_dim + d]` layout. The HF reference
/// (`_compute_proportional_rope_parameters`) is not the qwen35-style
/// leading-block partial rotation: the first `rotary_dim / 2` inverse
/// frequencies use the FULL head_dim as the exponent denominator
/// (`theta^(2i/head_dim)`, not `/rotary_dim`), the remaining band is
/// zero-padded, and `rotate_half` then pairs `(d, d + head_dim/2)` across
/// the whole head — zero frequency makes the un-rotated band an exact
/// identity (cos 1, sin 0). The prep kernel therefore runs at
/// `rotary_dim = head_dim`; the partiality lives in these tables.
pub(crate) fn build_proportional_rope_tables(
    ctx: &DeviceContext,
    rope_theta: f32,
    head_dim: usize,
    rotary_dim: usize,
    max_pos: usize,
) -> Result<(DeviceVec, DeviceVec)> {
    anyhow::ensure!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "proportional rope theta {rope_theta} must be positive and finite"
    );
    anyhow::ensure!(
        head_dim > 0
            && head_dim.is_multiple_of(2)
            && rotary_dim > 0
            && rotary_dim.is_multiple_of(2),
        "proportional rope dims must be positive and even: head_dim {head_dim}, rotary_dim \
         {rotary_dim}"
    );
    anyhow::ensure!(
        rotary_dim <= head_dim,
        "proportional rope rotary_dim {rotary_dim} exceeds head_dim {head_dim}"
    );
    anyhow::ensure!(max_pos > 0, "proportional rope max_pos must be positive");
    let table_len = max_pos.checked_mul(head_dim).ok_or_else(|| {
        anyhow::anyhow!("proportional rope table size {max_pos} x {head_dim} overflows")
    })?;
    let rope_angles = rotary_dim / 2;
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..rope_angles)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let mut cos = vec![bf16::from_f32(1.0); table_len];
    let mut sin = vec![bf16::from_f32(0.0); table_len];
    for pos in 0..max_pos {
        let row = pos * head_dim;
        for (i, &frequency) in inv_freq.iter().enumerate() {
            let angle = pos as f32 * frequency;
            let (c, s_) = (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()));
            cos[row + i] = c;
            cos[row + i + half_dim] = c;
            sin[row + i] = s_;
            sin[row + i + half_dim] = s_;
        }
    }
    Ok((
        DeviceVec::from_host(ctx, &cos)?,
        DeviceVec::from_host(ctx, &sin)?,
    ))
}

/// Buffers one [`attention_epilogue_into`] call needs, held across every
/// layer of every step.
pub(crate) struct EpilogueScratch {
    max_rows: usize,
    attn_proj: HiddenStates,
    residual: HiddenStates,
    mlp_in: HiddenStates,
    gate: HiddenStates,
    up: HiddenStates,
    act: HiddenStates,
    down: HiddenStates,
    moe: Option<MoeScratch>,
}

impl EpilogueScratch {
    pub(crate) fn new(ctx: &DeviceContext, geom: &LayerGeometry, max_rows: usize) -> Result<Self> {
        let hidden = |rows| HiddenStates::zeros(ctx, geom.hidden_size, rows);
        let wide = |rows| HiddenStates::zeros(ctx, geom.intermediate_size, rows);
        Ok(Self {
            max_rows,
            attn_proj: hidden(max_rows)?,
            residual: hidden(max_rows)?,
            mlp_in: hidden(max_rows)?,
            gate: wide(max_rows)?,
            up: wide(max_rows)?,
            act: wide(max_rows)?,
            down: hidden(max_rows)?,
            moe: match geom.moe {
                Some(_) => Some(MoeScratch::new(ctx, geom, max_rows)?),
                None => None,
            },
        })
    }

    /// Reshape every buffer to this step's row count, refusing one past the
    /// allocation: the ops only assert that tensors agree with each other,
    /// so an oversize count would reach a kernel as an out-of-bounds write.
    pub(crate) fn set_rows(&mut self, seq_len: usize) -> Result<()> {
        anyhow::ensure!(
            seq_len <= self.max_rows,
            "epilogue scratch holds {} rows, not {seq_len}",
            self.max_rows
        );
        for buf in [
            &mut self.attn_proj,
            &mut self.residual,
            &mut self.mlp_in,
            &mut self.gate,
            &mut self.up,
            &mut self.act,
            &mut self.down,
        ] {
            buf.seq_len = seq_len;
        }
        Ok(())
    }
}

/// Everything downstream of attention — o_proj through the `layer_scalar`
/// multiply (applied after both residual adds, not either branch) — is
/// identical for both layer kinds; one implementation keeps the two
/// forwards' numerics from drifting apart.
pub(crate) fn attention_epilogue_into(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LayerGeometry,
    x: &HiddenStates,
    attn: &HiddenStates,
    scratch: &mut EpilogueScratch,
    out: &mut HiddenStates,
) -> Result<()> {
    let seq_len = x.seq_len;
    let scratch_rows = scratch.attn_proj.seq_len;
    anyhow::ensure!(
        scratch_rows == seq_len,
        "epilogue scratch is shaped for {scratch_rows} rows, not {seq_len}"
    );
    let out_elems = geom
        .hidden_size
        .checked_mul(seq_len)
        .context("epilogue output extent overflows")?;
    anyhow::ensure!(
        out.data.len() >= out_elems,
        "epilogue output holds {} elements, not {out_elems}",
        out.data.len()
    );
    out.hidden_dim = geom.hidden_size;
    out.seq_len = seq_len;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.o_proj,
        0,
        geom.hidden_size,
        attn,
        &mut scratch.attn_proj,
    )?;
    // The first normalized value and its residual sum still round to bf16
    // before the second reduction reads them.
    ops::rms_norm_add_rms_norm_round_batch_into(
        ctx,
        &scratch.attn_proj,
        &layer.post_attention_layernorm,
        x,
        &layer.pre_feedforward_layernorm,
        geom.rms_norm_eps,
        &mut scratch.residual,
        &mut scratch.mlp_in,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.mlp.gate,
        0,
        geom.intermediate_size,
        &scratch.mlp_in,
        &mut scratch.gate,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.mlp.up,
        0,
        geom.intermediate_size,
        &scratch.mlp_in,
        &mut scratch.up,
    )?;
    ops::gelu_tanh_mul_batch_into(ctx, &scratch.gate, &scratch.up, &mut scratch.act)?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.mlp.down,
        0,
        geom.hidden_size,
        &scratch.act,
        &mut scratch.down,
    )?;
    let feed_forward = match (&layer.moe, &mut scratch.moe) {
        (Some(moe), Some(moe_scratch)) => {
            crate::moe::moe_into(
                ctx,
                moe,
                geom,
                &scratch.residual,
                &scratch.down,
                moe_scratch,
                // The attention projection is dead after `residual` is formed
                // and has the same shape, so the routed-only result reuses it.
                &mut scratch.attn_proj,
            )?;
            &scratch.attn_proj
        }
        (None, _) => &scratch.down,
        (Some(_), None) => anyhow::bail!("Gemma 4: a routed layer met a dense epilogue scratch"),
    };
    ops::rms_norm_add_scale_batch_into(
        ctx,
        feed_forward,
        &layer.post_feedforward_layernorm,
        &scratch.residual,
        layer.layer_scalar,
        geom.rms_norm_eps,
        out,
    )?;
    Ok(())
}
