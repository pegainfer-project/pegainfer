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
use pegainfer_core::tensor::Columns;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Gemma4Config;
use crate::config::MoeConfig;
use crate::config::TensorParallelConfig;
use crate::moe::MoeScratch;
use crate::weights::Gemma4Layer;
use crate::weights::Linear;
use crate::weights::LinearScratch;

/// The NCCL communicator of this rank's tensor-parallel group, or `None` at
/// world size 1, where every collective is a no-op.
pub(crate) type TpComm = cudarc::nccl::safe::Comm;

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
    /// This rank's geometry for the sliding family. The residual stream and
    /// the head width stay whole — every rank holds whole heads — while the
    /// head counts and the MLP width shard.
    pub(crate) fn local_of(config: &Gemma4Config, tp: TensorParallelConfig) -> Result<Self> {
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: tp.local_intermediate(config)?,
            num_q_heads: tp.local_q_heads(config)?,
            num_kv_heads: tp.local_sliding_kv_heads(config)?,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
            moe: config.moe,
        })
    }

    /// This rank's geometry for the global family. Its KV heads take the
    /// two-branch range — shard when the world size divides them, replicate
    /// when it does not — which the config resolves.
    pub(crate) fn global_of(config: &Gemma4Config, tp: TensorParallelConfig) -> Result<Self> {
        Ok(Self {
            hidden_size: config.hidden_size,
            intermediate_size: tp.local_intermediate(config)?,
            num_q_heads: tp.local_q_heads(config)?,
            num_kv_heads: tp.global_kv_head_range(config)?.1,
            head_dim: config.global_head_dim,
            rms_norm_eps: config.rms_norm_eps,
            moe: config.moe,
        })
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
    mlp: Activations,
    act: HiddenStates,
    down: HiddenStates,
    moe: Option<MoeScratch>,
    /// Serves the attention projections too.
    pub(crate) linear: LinearScratch,
}

/// Where gate and up land: buffers of their own, or one row per token
/// holding both so a single GEMM fills it. Split for the same reason the
/// attention projections are: the fused shape is a different cuBLAS shape.
enum Activations {
    Separate {
        gate: HiddenStates,
        up: HiddenStates,
    },
    Fused(HiddenStates),
}

impl Activations {
    fn set_rows(&mut self, seq_len: usize) {
        match self {
            Self::Separate { gate, up } => {
                gate.seq_len = seq_len;
                up.seq_len = seq_len;
            }
            Self::Fused(both) => both.seq_len = seq_len,
        }
    }

    fn project(
        &mut self,
        ctx: &DeviceContext,
        gate_up: &Linear,
        x: &HiddenStates,
        width: usize,
        linear: &LinearScratch,
    ) -> Result<(Columns<'_>, Columns<'_>)> {
        anyhow::ensure!(
            gate_up.rows() == 2 * width,
            "gate|up holds {} rows, not 2 x {width}",
            gate_up.rows()
        );
        match self {
            Self::Separate { gate, up } => {
                let gate_up = gate_up.bf16()?;
                ops::gemm_rows_into_checked(ctx, gate_up, 0, width, x, gate)?;
                ops::gemm_rows_into_checked(ctx, gate_up, width, width, x, up)?;
                Ok(((&*gate).into(), (&*up).into()))
            }
            Self::Fused(both) => {
                gate_up.project_into(ctx, x, linear, both)?;
                Ok((both.columns(0, width), both.columns(width, width)))
            }
        }
    }
}

impl EpilogueScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        geom: &LayerGeometry,
        max_rows: usize,
        fused: bool,
        linear: LinearScratch,
    ) -> Result<Self> {
        let hidden = |rows| HiddenStates::zeros(ctx, geom.hidden_size, rows);
        let wide = |rows| HiddenStates::zeros(ctx, geom.intermediate_size, rows);
        Ok(Self {
            max_rows,
            attn_proj: hidden(max_rows)?,
            residual: hidden(max_rows)?,
            mlp_in: hidden(max_rows)?,
            mlp: if fused {
                Activations::Fused(HiddenStates::zeros(
                    ctx,
                    2 * geom.intermediate_size,
                    max_rows,
                )?)
            } else {
                Activations::Separate {
                    gate: wide(max_rows)?,
                    up: wide(max_rows)?,
                }
            },
            act: wide(max_rows)?,
            down: hidden(max_rows)?,
            moe: match geom.moe {
                Some(_) => Some(MoeScratch::new(ctx, geom, max_rows)?),
                None => None,
            },
            linear,
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
            &mut self.act,
            &mut self.down,
        ] {
            buf.seq_len = seq_len;
        }
        self.mlp.set_rows(seq_len);
        Ok(())
    }
}

/// Sum this rank's partial projection across the tensor-parallel group, in
/// place. Only the live rows are handed to NCCL: a decode arena is padded to
/// its power-of-two bucket and the rows past `seq_len` are zero padding no
/// rank reads back, so reducing them would move bytes for nothing. The extent
/// is `hidden_size * seq_len`, the same on every rank of the step; inside a
/// captured graph `seq_len` is the bucket, so the recorded shape stays
/// constant across replays. At world size 1 `comm` is `None` and this is free.
fn all_reduce_rows(
    comm: Option<&TpComm>,
    geom: &LayerGeometry,
    buf: &mut HiddenStates,
    seq_len: usize,
) -> Result<()> {
    let Some(comm) = comm else {
        return Ok(());
    };
    let elems = geom
        .hidden_size
        .checked_mul(seq_len)
        .context("tensor-parallel reduction extent overflows")?;
    anyhow::ensure!(
        buf.data.len() >= elems,
        "a tensor-parallel reduction needs {elems} elements, the buffer holds {}",
        buf.data.len()
    );
    comm.all_reduce_in_place(
        &mut buf.data.slice_mut(..elems),
        &cudarc::nccl::safe::ReduceOp::Sum,
    )
    .map_err(|e| anyhow::anyhow!("gemma4 tensor-parallel all-reduce failed: {e:?}"))?;
    Ok(())
}

/// Everything downstream of attention — o_proj through the `layer_scalar`
/// multiply (applied after both residual adds, not either branch) — is
/// identical for both layer kinds; one implementation keeps the two
/// forwards' numerics from drifting apart.
///
/// Under tensor parallelism both projections here are row-parallel sums, so
/// each is reduced across the group before the residual it feeds is formed.
pub(crate) fn attention_epilogue_into(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LayerGeometry,
    comm: Option<&TpComm>,
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
    layer
        .attention
        .o_proj
        .project_into(ctx, attn, &scratch.linear, &mut scratch.attn_proj)?;
    // o_proj is row-parallel: its output is this rank's partial sum over
    // the whole hidden width, so it is summed across the group before the
    // residual add reads it.
    all_reduce_rows(comm, geom, &mut scratch.attn_proj, seq_len)?;
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
    if !layer
        .mlp
        .gate_up
        .gelu_mul_into(ctx, &scratch.mlp_in, &scratch.linear, &mut scratch.act)?
    {
        let (gate, up) = scratch.mlp.project(
            ctx,
            &layer.mlp.gate_up,
            &scratch.mlp_in,
            geom.intermediate_size,
            &scratch.linear,
        )?;
        ops::gelu_tanh_mul_batch_into(ctx, gate, up, &mut scratch.act)?;
    }
    layer
        .mlp
        .down
        .project_into(ctx, &scratch.act, &scratch.linear, &mut scratch.down)?;
    let feed_forward: &mut HiddenStates = match (&layer.moe, &mut scratch.moe) {
        (Some(moe), Some(moe_scratch)) => {
            crate::moe::moe_into(
                ctx,
                moe,
                geom,
                &scratch.residual,
                &scratch.down,
                moe_scratch,
                // The attention projection is dead after `residual` is formed
                // and has the same shape, so the routed block's result reuses it.
                &mut scratch.attn_proj,
            )?;
            &mut scratch.attn_proj
        }
        (None, _) => &mut scratch.down,
        (Some(_), None) => anyhow::bail!("Gemma 4: a routed layer met a dense epilogue scratch"),
    };
    // `down` is row-parallel as well, and a routed block's output is summed
    // the same way.
    all_reduce_rows(comm, geom, feed_forward, seq_len)?;
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
