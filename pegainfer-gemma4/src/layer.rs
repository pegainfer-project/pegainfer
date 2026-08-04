//! One Gemma 4 local (sliding-attention) decoder layer, prefill form.
//!
//! The graph is the HF reference's, with the constants a from-the-paper
//! implementation gets wrong: the four norm sites are all norm-then-add
//! (sandwich, not the fused add-then-norm shape), attention is unscaled
//! (`scaling = 1.0` in the reference — not `head_dim**-0.5`), V takes a
//! weightless RMS norm and no RoPE, RoPE rotates the full 256-wide head,
//! and `layer_scalar` multiplies the layer output after both residual adds.
//!
//! There is no KV cache: K and V stay contiguous and feed `single_prefill`,
//! which computes exact sliding attention for any prompt short enough that
//! the window never truncates (`seq_len <= sliding_window`; the window
//! first evicts at `sliding_window + 1` tokens). The window boundary
//! belongs to the SWA kernels and the prefill ladder, not to this layer.

use anyhow::Context as _;
use anyhow::Result;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::weights::Gemma4Layer;

/// The geometry a local layer runs at, read off the validated config.
pub(crate) struct LocalLayerGeometry {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_norm_eps: f32,
}

/// Token-major `[num_heads * head_dim, seq_len]` rows into a contiguous HND
/// cache of the same total size: `[head][pos][head_dim]`, `seq_len` rows per
/// head. Each head's column window is extracted to a `[head_dim, seq_len]`
/// block, then placed at its head-major offset.
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

/// Runs one local layer on `x` (`[hidden_size, seq_len]`), tokens at
/// positions `start_pos..start_pos + seq_len`. Buffers are allocated per
/// call: this is the correctness building block, and the executor that would
/// own persistent buffers does not exist yet.
#[allow(clippy::too_many_arguments)]
pub(crate) fn local_layer_forward(
    ctx: &DeviceContext,
    layer: &Gemma4Layer,
    geom: &LocalLayerGeometry,
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

    // v_norm is weightless (`with_scale=False`): a plain-w RMS norm with a
    // ones weight is the same arithmetic. The `[kv_dim, seq_len]` buffer is
    // reinterpreted as `[head_dim, seq_len * num_kv_heads]` — heads are
    // contiguous within each token row, so the narrower row width makes the
    // reduction per (token, head) — then the shape is restored.
    let ones = DeviceVec::from_host(ctx, &vec![bf16::from_f32(1.0); geom.head_dim])?;
    v_states.hidden_dim = geom.head_dim;
    v_states.seq_len = seq_len * geom.num_kv_heads;
    let mut v_normed = HiddenStates::zeros(ctx, geom.head_dim, seq_len * geom.num_kv_heads)?;
    ops::rms_norm_batch_into(ctx, &v_states, &ones, geom.rms_norm_eps, &mut v_normed);
    v_normed.hidden_dim = kv_dim;
    v_normed.seq_len = seq_len;

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

    let mut attn_proj = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.attention.o_proj,
        0,
        geom.hidden_size,
        &attn,
        &mut attn_proj,
    )?;
    let mut o_normed = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::rms_norm_batch_into(
        ctx,
        &attn_proj,
        &layer.post_attention_layernorm,
        geom.rms_norm_eps,
        &mut o_normed,
    );
    let mut h2 = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::add_batch_into(ctx, x, &o_normed, &mut h2)?;

    let mut mlp_in = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::rms_norm_batch_into(
        ctx,
        &h2,
        &layer.pre_feedforward_layernorm,
        geom.rms_norm_eps,
        &mut mlp_in,
    );
    let mut gate = HiddenStates::zeros(ctx, geom.intermediate_size, seq_len)?;
    let mut up = HiddenStates::zeros(ctx, geom.intermediate_size, seq_len)?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.mlp.gate,
        0,
        geom.intermediate_size,
        &mlp_in,
        &mut gate,
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &layer.mlp.up,
        0,
        geom.intermediate_size,
        &mlp_in,
        &mut up,
    )?;
    let mut act = HiddenStates::zeros(ctx, geom.intermediate_size, seq_len)?;
    ops::gelu_tanh_mul_batch_into(ctx, &gate, &up, &mut act)?;
    let mut down = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::gemm_rows_into_checked(ctx, &layer.mlp.down, 0, geom.hidden_size, &act, &mut down)?;
    let mut down_normed = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::rms_norm_batch_into(
        ctx,
        &down,
        &layer.post_feedforward_layernorm,
        geom.rms_norm_eps,
        &mut down_normed,
    );

    let mut out = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
    ops::add_batch_into(ctx, &h2, &down_normed, &mut out)?;

    // layer_scalar multiplies the layer output after both residual adds —
    // not either branch.
    ops::scale_bf16_in_place(ctx, &mut out, layer.layer_scalar)?;

    Ok(out)
}

/// The oracle: replay the HF golden fixture's local-layer probes through
/// this layer implementation on the real checkpoint. See
/// `docs/models/gemma4/hf-golden.md` for what the fixture pins.
#[cfg(test)]
#[path = "layer_oracle.rs"]
mod oracle;
