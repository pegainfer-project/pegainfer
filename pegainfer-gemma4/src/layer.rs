//! Gemma 4 decoder layers, prefill form: the local (sliding-attention) and
//! global (full-attention) kinds, sharing one attention epilogue.
//!
//! The graph is the HF reference's, with the constants a from-the-paper
//! implementation gets wrong: the four norm sites are all norm-then-add
//! (sandwich, not the fused add-then-norm shape), attention is unscaled
//! (`scaling = 1.0` in the reference — not `head_dim**-0.5`), V takes a
//! weightless RMS norm and no RoPE (on global layers V is `k_proj`'s raw
//! output, forked before `k_norm`), RoPE rotates the full head width (the
//! global family's partiality lives in its tables), and `layer_scalar`
//! multiplies the layer output after both residual adds.
//!
//! There is no KV cache: K and V stay contiguous and feed `single_prefill`,
//! exact from position zero — for the local kind only while the prompt fits
//! the sliding window. Both forwards reject anything outside that domain.

use anyhow::Context as _;
use anyhow::Result;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Gemma4Config;
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

/// Token-major `[num_heads * head_dim, seq_len]` rows into a contiguous HND
/// cache of the same total size: `[head][pos][head_dim]`, `seq_len` rows per
/// head. Each head's column window is extracted to a `[head_dim, seq_len]`
/// block, then placed at its head-major offset.
#[cfg(test)]
fn nhd_to_hnd(
    ctx: &DeviceContext,
    src: &HiddenStates,
    num_heads: usize,
    head_dim: usize,
) -> Result<HiddenStates> {
    let seq_len = src.seq_len;
    anyhow::ensure!(
        src.hidden_dim == num_heads * head_dim,
        "nhd_to_hnd src.hidden_dim {} != num_heads {} * head_dim {}",
        src.hidden_dim,
        num_heads,
        head_dim
    );
    let mut hnd = HiddenStates::zeros(ctx, src.hidden_dim, seq_len)?;
    let mut head_block = HiddenStates::zeros(ctx, head_dim, seq_len)?;
    let block_len = head_dim * seq_len;
    for head in 0..num_heads {
        ops::extract_hidden_rows_raw_into(
            ctx,
            &src.data,
            src.hidden_dim,
            &mut head_block.data,
            head_dim,
            head * head_dim,
            seq_len,
        )?;
        let src_view = head_block.data.slice(0..block_len);
        let mut dst_view = hnd.data.slice_mut(head * block_len..(head + 1) * block_len);
        ctx.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .map_err(|e| anyhow::anyhow!("nhd_to_hnd head {head} copy failed: {e}"))?;
    }
    Ok(hnd)
}

/// v_norm is weightless (`with_scale=False`): a plain-w RMS norm with a ones
/// weight is the same arithmetic. The `[kv_dim, seq_len]` buffer is
/// reinterpreted as `[head_dim, seq_len * num_kv_heads]` — heads are
/// contiguous within each token row, so the narrower row width makes the
/// reduction per (token, head) — then the shape is restored.
#[cfg(test)]
fn weightless_value_norm(
    ctx: &DeviceContext,
    mut v_states: HiddenStates,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
) -> Result<HiddenStates> {
    let seq_len = v_states.seq_len;
    let kv_dim = v_states.hidden_dim;
    anyhow::ensure!(
        kv_dim == num_kv_heads * head_dim,
        "weightless_value_norm v.hidden_dim {kv_dim} != num_kv_heads {num_kv_heads} * head_dim \
         {head_dim}"
    );
    let ones = DeviceVec::from_host(ctx, &vec![bf16::from_f32(1.0); head_dim])?;
    v_states.hidden_dim = head_dim;
    v_states.seq_len = seq_len * num_kv_heads;
    let mut v_normed = HiddenStates::zeros(ctx, head_dim, seq_len * num_kv_heads)?;
    ops::rms_norm_batch_into(ctx, &v_states, &ones, rms_norm_eps, &mut v_normed);
    v_normed.hidden_dim = kv_dim;
    v_normed.seq_len = seq_len;
    Ok(v_normed)
}

/// Buffers one [`attention_epilogue_into`] call needs. The serving path holds
/// one across every layer of every step; the oracle builds a fresh one per
/// call, which is what the allocating [`attention_epilogue`] does.
pub(crate) struct EpilogueScratch {
    max_rows: usize,
    attn_proj: HiddenStates,
    o_normed: HiddenStates,
    h2: HiddenStates,
    mlp_in: HiddenStates,
    gate: HiddenStates,
    up: HiddenStates,
    act: HiddenStates,
    down: HiddenStates,
    down_normed: HiddenStates,
}

impl EpilogueScratch {
    pub(crate) fn new(ctx: &DeviceContext, geom: &LayerGeometry, max_rows: usize) -> Result<Self> {
        let hidden = |rows| HiddenStates::zeros(ctx, geom.hidden_size, rows);
        let wide = |rows| HiddenStates::zeros(ctx, geom.intermediate_size, rows);
        Ok(Self {
            max_rows,
            attn_proj: hidden(max_rows)?,
            o_normed: hidden(max_rows)?,
            h2: hidden(max_rows)?,
            mlp_in: hidden(max_rows)?,
            gate: wide(max_rows)?,
            up: wide(max_rows)?,
            act: wide(max_rows)?,
            down: hidden(max_rows)?,
            down_normed: hidden(max_rows)?,
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
            &mut self.o_normed,
            &mut self.h2,
            &mut self.mlp_in,
            &mut self.gate,
            &mut self.up,
            &mut self.act,
            &mut self.down,
            &mut self.down_normed,
        ] {
            buf.seq_len = seq_len;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn attention_epilogue(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LayerGeometry,
    x: &HiddenStates,
    attn: &HiddenStates,
) -> Result<HiddenStates> {
    let mut scratch = EpilogueScratch::new(ctx, geom, x.seq_len)?;
    let mut out = HiddenStates::zeros(ctx, geom.hidden_size, x.seq_len)?;
    attention_epilogue_into(ctx, layer, geom, x, attn, &mut scratch, &mut out)?;
    Ok(out)
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
    ops::rms_norm_batch_into(
        ctx,
        &scratch.attn_proj,
        &layer.post_attention_layernorm,
        geom.rms_norm_eps,
        &mut scratch.o_normed,
    );
    ops::add_batch_into(ctx, x, &scratch.o_normed, &mut scratch.h2)?;

    ops::rms_norm_batch_into(
        ctx,
        &scratch.h2,
        &layer.pre_feedforward_layernorm,
        geom.rms_norm_eps,
        &mut scratch.mlp_in,
    );
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
    ops::rms_norm_batch_into(
        ctx,
        &scratch.down,
        &layer.post_feedforward_layernorm,
        geom.rms_norm_eps,
        &mut scratch.down_normed,
    );

    ops::add_batch_into(ctx, &scratch.h2, &scratch.down_normed, out)?;
    ops::scale_bf16_in_place(ctx, out, layer.layer_scalar)?;
    Ok(())
}

/// Runs one local layer on `x` (`[hidden_size, seq_len]`), tokens at
/// positions `start_pos..start_pos + seq_len`. Buffers are allocated per
/// call: this is the correctness building block, and the executor that would
/// own persistent buffers does not exist yet.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn local_layer_forward(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LayerGeometry,
    x: &HiddenStates,
    start_pos: usize,
    sliding_window: usize,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    cos_max_pos: usize,
) -> Result<HiddenStates> {
    let seq_len = x.seq_len;
    // Probe form: no KV cache exists, so history before start_pos would be
    // silently missing, and past the window the full causal mask no longer
    // equals sliding attention. Reject both instead of mis-computing.
    anyhow::ensure!(
        start_pos == 0,
        "local_layer_forward is prefill-from-zero only; start_pos {start_pos} needs a KV cache"
    );
    anyhow::ensure!(
        seq_len <= sliding_window,
        "local_layer_forward seq_len {seq_len} exceeds sliding_window {sliding_window}; the window \
         would truncate"
    );
    let q_dim = geom.num_q_heads * geom.head_dim;
    let kv_dim = geom.num_kv_heads * geom.head_dim;
    anyhow::ensure!(
        x.hidden_dim == geom.hidden_size,
        "local layer x.hidden_dim {} != hidden_size {}",
        x.hidden_dim,
        geom.hidden_size
    );
    anyhow::ensure!(
        geom.head_dim == 256,
        "local layer head_dim {} != 256, which the prep kernel is instantiated at",
        geom.head_dim
    );
    let v_proj = layer
        .attention
        .v_proj
        .as_ref()
        .context("local layer requires v_proj; only global layers ship without one")?;

    let mut normed_x = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::rms_norm_batch_into(
        ctx,
        x,
        &layer.input_layernorm,
        geom.rms_norm_eps,
        &mut normed_x,
    );

    let mut q_states = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    let mut k_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
    let mut v_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.q_proj,
        0,
        q_dim,
        &normed_x,
        &mut q_states,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.k_proj,
        0,
        kv_dim,
        &normed_x,
        &mut k_states,
    )?;
    ops::gemm_rows_into_checked(ctx, v_proj, 0, kv_dim, &normed_x, &mut v_states)?;

    // Full-head rotation: Gemma 4 local layers have no partial factor, so
    // rotary_dim is the head width.
    let mut q_prep = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    let mut k_prep = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
    ops::qk_norm_rope_prefill_hd256_plain_into(
        ctx,
        &q_states,
        &k_states,
        &mut q_prep,
        &mut k_prep,
        &layer.attention.q_norm,
        &layer.attention.k_norm,
        cos_cache,
        sin_cache,
        start_pos,
        cos_max_pos,
        geom.num_q_heads,
        geom.num_kv_heads,
        geom.head_dim,
        geom.rms_norm_eps,
    )?;

    let v_normed = weightless_value_norm(
        ctx,
        v_states,
        geom.num_kv_heads,
        geom.head_dim,
        geom.rms_norm_eps,
    )?;

    // single_prefill's contiguous cache is HND — k[head, pos, dim] — while
    // the prep and the GEMMs emit token-major rows, so K and V are
    // reassembled per head. Per-call copies; the executor owns avoiding
    // this once it owns layouts.
    let k_hnd = nhd_to_hnd(ctx, &k_prep, geom.num_kv_heads, geom.head_dim)?;
    let v_hnd = nhd_to_hnd(ctx, &v_normed, geom.num_kv_heads, geom.head_dim)?;

    // Unscaled attention: the reference sets scaling = 1.0 for both layer
    // kinds; sm_scale = rsqrt(head_dim) here would shrink logits to 1/16.
    let mut attn = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    ops::single_prefill_hd256_into(
        ctx,
        &q_prep,
        0,
        seq_len,
        &k_hnd,
        &v_hnd,
        &mut attn,
        geom.num_q_heads,
        geom.num_kv_heads,
        seq_len,
        1.0,
    )?;

    attention_epilogue(ctx, layer, geom, x, &attn)
}

/// Runs one global (full-attention) layer on `x`; same per-call-buffer probe
/// form as [`local_layer_forward`].
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn global_layer_forward(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LayerGeometry,
    x: &HiddenStates,
    start_pos: usize,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    cos_max_pos: usize,
) -> Result<HiddenStates> {
    let seq_len = x.seq_len;
    anyhow::ensure!(
        start_pos == 0,
        "global_layer_forward is prefill-from-zero only; start_pos {start_pos} needs a KV cache"
    );
    anyhow::ensure!(
        start_pos
            .checked_add(seq_len)
            .is_some_and(|end| end <= cos_max_pos),
        "global layer positions {start_pos}..{start_pos}+{seq_len} exceed the rope table's \
         cos_max_pos {cos_max_pos}; the prep kernel traps on out-of-range positions"
    );
    let q_dim = geom.num_q_heads * geom.head_dim;
    let kv_dim = geom.num_kv_heads * geom.head_dim;
    anyhow::ensure!(
        x.hidden_dim == geom.hidden_size,
        "global layer x.hidden_dim {} != hidden_size {}",
        x.hidden_dim,
        geom.hidden_size
    );
    anyhow::ensure!(
        geom.head_dim == 512,
        "global layer head_dim {} != 512, which the prep kernel is instantiated at",
        geom.head_dim
    );
    anyhow::ensure!(
        layer.attention.v_proj.is_none(),
        "global layer must not carry a v_proj; the checkpoint ships its \
         full_attention layers without one"
    );

    let mut normed_x = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::rms_norm_batch_into(
        ctx,
        x,
        &layer.input_layernorm,
        geom.rms_norm_eps,
        &mut normed_x,
    );

    let mut q_states = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    let mut k_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.q_proj,
        0,
        q_dim,
        &normed_x,
        &mut q_states,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.k_proj,
        0,
        kv_dim,
        &normed_x,
        &mut k_states,
    )?;

    // The K=V fork: value is k_proj's raw output, copied BEFORE k_norm and
    // RoPE touch K — a runtime fork of the projection, not shared storage;
    // after the norms the two tensors differ bitwise.
    let mut v_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
    ctx.stream
        .memcpy_dtod(&k_states.data, &mut v_states.data)
        .map_err(|e| anyhow::anyhow!("global layer V fork copy failed: {e}"))?;

    // Norm + rotate Q and K. The decode-shaped prep entry is the contiguous
    // one (its prefill sibling writes a paged pool); positions are the
    // explicit per-token array it expects. rotary_dim is the FULL head — the
    // proportional tables carry the partiality (see the table builder).
    let positions: Vec<i32> = (0..seq_len)
        .map(|t| {
            i32::try_from(start_pos + t)
                .map_err(|_| anyhow::anyhow!("position {} does not fit i32", start_pos + t))
        })
        .collect::<Result<_>>()?;
    let positions_d = ctx
        .stream
        .clone_htod(&positions)
        .map_err(|e| anyhow::anyhow!("positions H2D failed: {e}"))?;
    let mut q_prep = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    ops::qk_norm_partial_rope_batched_decode_hd512_into(
        ctx,
        &q_states,
        &mut q_prep,
        &mut k_states,
        &layer.attention.q_norm,
        &layer.attention.k_norm,
        cos_cache,
        sin_cache,
        &positions_d,
        cos_max_pos,
        geom.num_q_heads,
        geom.num_kv_heads,
        geom.head_dim,
        geom.rms_norm_eps,
    )?;

    let v_normed = weightless_value_norm(
        ctx,
        v_states,
        geom.num_kv_heads,
        geom.head_dim,
        geom.rms_norm_eps,
    )?;

    // single_prefill_hd512 reads HND; reassemble per head (at one KV head
    // this is a plain copy, but the layout contract stays explicit).
    let k_hnd = nhd_to_hnd(ctx, &k_states, geom.num_kv_heads, geom.head_dim)?;
    let v_hnd = nhd_to_hnd(ctx, &v_normed, geom.num_kv_heads, geom.head_dim)?;

    let mut attn = HiddenStates::zeros(ctx, q_dim, seq_len)?;
    ops::single_prefill_hd512_into(
        ctx,
        &q_prep,
        0,
        seq_len,
        &k_hnd,
        &v_hnd,
        &mut attn,
        geom.num_q_heads,
        geom.num_kv_heads,
        seq_len,
        1.0,
    )?;

    attention_epilogue(ctx, layer, geom, x, &attn)
}

/// The oracle: replay the HF golden fixture's layer probes through this
/// implementation on the real checkpoint. See
/// `docs/models/gemma4/hf-golden.md` for what the fixture pins.
#[cfg(test)]
#[path = "layer_oracle.rs"]
mod oracle;
