//! K3 DSpark draft lane (rank-local): the RadixArk community
//! `kimi-k3-dspark` drafter — a 5-layer qwen3-architecture GQA backbone at
//! K3's hidden 7168 plus a rank-256 Markov head — proposing 6 greedy draft
//! tokens per round from the target's captured aux hidden states. No
//! collectives anywhere: the draft is replicated on every rank and runs
//! between that rank's steps, DP over its slots.
//!
//! Layout facts are pinned against the checkpoint's own `dflash.py`
//! (`extract_context_feature`: `hidden_states[layer_id + 1]` over the HF
//! `output_hidden_states` list, whose entry `k + 1` is the residual stream
//! after 0-based layer `k`): the fc context rows are the target's residual
//! captured AFTER 0-based layers `K3_DSPARK_AUX_LAYERS` — the ids as the
//! config spells them, no off-by-one (unlike the vLLM-trained GLM5.2
//! checkpoint). The block input is `[anchor, mask x 6]` and draft `k` is
//! sampled from block position `k`'s logits — position 0 (the anchor row)
//! predicts the token right after the anchor, exactly like the reference
//! `run_markov_block`; the Markov loop starts there with `prev(0) = anchor`.
//! (Reading drafts from positions 1..=6 instead proposes every draft one
//! position ahead of where verify compares it and collapses acceptance to
//! ~0 — the bring-up bug.) The pending context always ends one row before
//! the anchor: the anchor's own hidden is only captured when the anchor is
//! fed to the target.
//!
//! The checkpoint carries NO `embed_tokens` or `lm_head` of its own — the
//! block embedding and the draft logits reuse the target's matrices. The
//! confidence head ships in the checkpoint but is not loaded (fixed
//! 6-draft rounds for now); the whole lane runs eagerly (no CUDA graphs
//! yet — bring-up first).

use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_kernels::ops::add_batch_into;
use pegainfer_kernels::ops::copy_hidden_token_range_into;
use pegainfer_kernels::ops::dflash_qk_norm_rope_into;
use pegainfer_kernels::ops::embedding_batch;
use pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::gemm_rows_into_checked;
use pegainfer_kernels::ops::markov_step_argmax_into;
use pegainfer_kernels::ops::markov_step_argmax_partials_len;
use pegainfer_kernels::ops::rms_norm_batch_into;
use pegainfer_kernels::ops::silu_mul_batch_into;
use pegainfer_kernels::ops::single_prefill_nhd_noncausal_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use crate::config::K3_HIDDEN;
use crate::config::K3_VOCAB;

/// Draft block width: anchor + 6 mask positions.
pub(crate) const K3_DSPARK_BLOCK: usize = 7;

/// Drafts proposed per round (anchor-drop: block position 0 is the anchor).
pub(crate) const K3_DSPARK_DRAFTS: usize = K3_DSPARK_BLOCK - 1;

/// Target layers whose post-layer residual stream feeds the fc context
/// projection, in fc column order — 0-based "residual after layer L". The
/// checkpoint's `dflash_config.target_layer_ids` verbatim (its own
/// `extract_context_feature` indexes `hidden_states[id + 1]`, the HF list's
/// after-layer-`id` entry). All five are MLA layers.
pub(crate) const K3_DSPARK_AUX_LAYERS: [usize; 5] = [7, 23, 51, 67, 83];

/// fc input width: the 5 captured layers' hidden states concatenated per
/// token, in `K3_DSPARK_AUX_LAYERS` order.
pub(crate) const K3_DSPARK_CONTEXT_DIM: usize = K3_DSPARK_AUX_LAYERS.len() * K3_HIDDEN;

/// The draft writes `block` transient rows past the committed+context length,
/// so its KV/rope tables need `block` positions of headroom past the target's
/// context cap.
pub(crate) fn dspark_cache_len(max_model_len: usize) -> usize {
    max_model_len + K3_DSPARK_BLOCK
}

const DSPARK_LAYERS: usize = 5;
const DSPARK_HEADS: usize = 64;
const DSPARK_KV_HEADS: usize = 16;
const DSPARK_HEAD_DIM: usize = 64;
const DSPARK_Q_DIM: usize = DSPARK_HEADS * DSPARK_HEAD_DIM;
const DSPARK_KV_DIM: usize = DSPARK_KV_HEADS * DSPARK_HEAD_DIM;
const DSPARK_INTER: usize = 14_336;
const DSPARK_MARKOV_RANK: usize = 256;
const DSPARK_MASK_TOKEN: u32 = 163_824;
const DSPARK_RMS_EPS: f32 = 1.0e-5;
// YaRN rope (the checkpoint's `rope_parameters`): qwen3-style full-dim
// rotary with transformers' default correction betas and the default
// attention factor `0.1 * ln(factor) + 1`, baked into the cos/sin tables.
const DSPARK_ROPE_THETA: f32 = 10_000.0;
const DSPARK_YARN_FACTOR: f32 = 16.0;
const DSPARK_YARN_ORIGINAL_MAX_POS: usize = 65_536;
const DSPARK_YARN_BETA_FAST: f32 = 32.0;
const DSPARK_YARN_BETA_SLOW: f32 = 1.0;

struct DsparkLayer {
    input_ln: DeviceVec,
    /// vstacked `[q; k; v]` `[4096 + 1024 + 1024, 7168]`.
    qkv: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    post_ln: DeviceVec,
    /// vstacked `[gate; up]` `[2 * 14336, 7168]`.
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

pub(crate) struct K3DsparkModel {
    layers: Vec<DsparkLayer>,
    /// Draft final norm (before the reused target lm_head).
    norm: DeviceVec,
    /// Norm applied to the fc-projected context rows.
    hidden_norm: DeviceVec,
    /// Context projection `[7168, 35840]`.
    fc: DeviceMatrix,
    /// Markov head: `bias(prev) = w2 @ w1[prev]`, both `[163840, 256]`.
    markov_w1: DeviceMatrix,
    markov_w2: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Draft KV/rope capacity: the target's context cap plus one block of
    /// transient headroom (see [`dspark_cache_len`]).
    cache_len: usize,
}

/// Crash-early config pin: this module hardcodes the checkpoint's geometry,
/// so a different checkpoint dir must fail at load, not produce garbage.
fn validate_config(path: &Path) -> Result<()> {
    let config_path = path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", config_path.display()))?;
    let expect = |field: &str, want: serde_json::Value| -> Result<()> {
        let got = json
            .get(field)
            .with_context(|| format!("dspark config missing `{field}`"))?;
        ensure!(
            *got == want,
            "dspark config `{field}` = {got}, this build expects {want}"
        );
        Ok(())
    };
    expect("model_type", "qwen3".into())?;
    expect("block_size", (K3_DSPARK_BLOCK as u64).into())?;
    expect("markov_rank", (DSPARK_MARKOV_RANK as u64).into())?;
    expect("markov_head_type", "vanilla".into())?;
    expect("num_hidden_layers", (DSPARK_LAYERS as u64).into())?;
    expect("hidden_size", (K3_HIDDEN as u64).into())?;
    expect("num_attention_heads", (DSPARK_HEADS as u64).into())?;
    expect("num_key_value_heads", (DSPARK_KV_HEADS as u64).into())?;
    expect("head_dim", (DSPARK_HEAD_DIM as u64).into())?;
    expect("intermediate_size", (DSPARK_INTER as u64).into())?;
    expect("vocab_size", (K3_VOCAB as u64).into())?;
    expect("num_target_layers", 93u64.into())?;
    expect("use_sliding_window", false.into())?;
    let dflash = json
        .get("dflash_config")
        .context("dspark config missing dflash_config")?;
    ensure!(
        dflash.get("mask_token_id") == Some(&(DSPARK_MASK_TOKEN as u64).into()),
        "dspark mask_token_id {:?} is not {DSPARK_MASK_TOKEN}",
        dflash.get("mask_token_id")
    );
    let aux: Vec<u64> = K3_DSPARK_AUX_LAYERS.iter().map(|id| *id as u64).collect();
    ensure!(
        dflash.get("target_layer_ids") == Some(&serde_json::json!(aux)),
        "dspark target_layer_ids {:?} is not {aux:?}",
        dflash.get("target_layer_ids")
    );
    let rope = json
        .get("rope_parameters")
        .context("dspark config missing rope_parameters")?;
    let rope_type = rope
        .get("rope_type")
        .context("dspark rope_parameters missing `rope_type`")?;
    ensure!(
        rope_type == "yarn",
        "dspark rope_parameters `rope_type` = {rope_type}, this build expects \"yarn\""
    );
    // Numeric fields compare as f64: `16` and `16.0` are the same config,
    // but distinct `serde_json::Number`s.
    for (field, want) in [
        ("factor", DSPARK_YARN_FACTOR as f64),
        (
            "original_max_position_embeddings",
            DSPARK_YARN_ORIGINAL_MAX_POS as f64,
        ),
        ("rope_theta", DSPARK_ROPE_THETA as f64),
    ] {
        let got = rope
            .get(field)
            .with_context(|| format!("dspark rope_parameters missing `{field}`"))?;
        ensure!(
            got.as_f64() == Some(want),
            "dspark rope_parameters `{field}` = {got}, this build expects {want}"
        );
    }
    Ok(())
}

/// YaRN cos/sin tables for the draft's full-dim rotary, laid out like
/// [`pegainfer_core::rope::precompute_rope`]'s (`[seq, dim]`, half-split
/// duplicated), with transformers' qwen3-YaRN semantics: NTK-by-parts
/// interpolation between the correction dims, and the default attention
/// factor multiplied into both tables.
fn build_yarn_rope_tables(seq_len: usize) -> (Vec<bf16>, Vec<bf16>) {
    let dim = DSPARK_HEAD_DIM;
    let half_dim = dim / 2;
    let correction_dim = |num_rotations: f32| -> f32 {
        (dim as f32
            * (DSPARK_YARN_ORIGINAL_MAX_POS as f32 / (num_rotations * 2.0 * std::f32::consts::PI))
                .ln())
            / (2.0 * DSPARK_ROPE_THETA.ln())
    };
    let low = correction_dim(DSPARK_YARN_BETA_FAST).floor().max(0.0);
    let high = correction_dim(DSPARK_YARN_BETA_SLOW)
        .ceil()
        .min((dim - 1) as f32);
    let denom = if (high - low).abs() < f32::EPSILON {
        0.001
    } else {
        high - low
    };
    let attention_factor = 0.1 * DSPARK_YARN_FACTOR.ln() + 1.0;

    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / dim as f32;
        let extrapolation = 1.0 / DSPARK_ROPE_THETA.powf(exponent);
        let interpolation = extrapolation / DSPARK_YARN_FACTOR;
        let ramp = ((i as f32 - low) / denom).clamp(0.0, 1.0);
        inv_freq.push(interpolation * ramp + extrapolation * (1.0 - ramp));
    }

    let mut cos = vec![bf16::ZERO; seq_len * dim];
    let mut sin = vec![bf16::ZERO; seq_len * dim];
    for token in 0..seq_len {
        for i in 0..half_dim {
            let freq = token as f32 * inv_freq[i];
            let c = bf16::from_f32(freq.cos() * attention_factor);
            let s = bf16::from_f32(freq.sin() * attention_factor);
            cos[token * dim + i] = c;
            cos[token * dim + half_dim + i] = c;
            sin[token * dim + i] = s;
            sin[token * dim + half_dim + i] = s;
        }
    }
    (cos, sin)
}

fn ensure_matrix(m: &DeviceMatrix, name: &str, rows: usize, cols: usize) -> Result<()> {
    ensure!(
        m.rows == rows && m.cols == cols,
        "dspark tensor {name} is [{}, {}], expected [{rows}, {cols}]",
        m.rows,
        m.cols
    );
    Ok(())
}

impl K3DsparkModel {
    pub(crate) fn load(ctx: &DeviceContext, path: &Path, max_model_len: usize) -> Result<Self> {
        validate_config(path)?;
        let cache_len = dspark_cache_len(max_model_len);
        let path_str = path
            .to_str()
            .context("dspark model path is not valid UTF-8")?;
        let (shard_paths, weight_map) = load_shard_info(path_str)?;
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for layer in 0..DSPARK_LAYERS {
            let p = format!("layers.{layer}");
            let q = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.q_proj.weight"),
            )?;
            let k = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.k_proj.weight"),
            )?;
            let v = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.v_proj.weight"),
            )?;
            ensure_matrix(&q, "q_proj", DSPARK_Q_DIM, K3_HIDDEN)?;
            ensure_matrix(&k, "k_proj", DSPARK_KV_DIM, K3_HIDDEN)?;
            ensure_matrix(&v, "v_proj", DSPARK_KV_DIM, K3_HIDDEN)?;
            let qkv = DeviceMatrix::vstack(ctx, &[&q, &k, &v])?;
            drop((q, k, v));
            let gate = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.gate_proj.weight"),
            )?;
            let up = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.up_proj.weight"),
            )?;
            ensure_matrix(&gate, "gate_proj", DSPARK_INTER, K3_HIDDEN)?;
            ensure_matrix(&up, "up_proj", DSPARK_INTER, K3_HIDDEN)?;
            let gate_up = DeviceMatrix::vstack(ctx, &[&gate, &up])?;
            drop((gate, up));
            let o_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.o_proj.weight"),
            )?;
            ensure_matrix(&o_proj, "o_proj", K3_HIDDEN, DSPARK_Q_DIM)?;
            let down = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.down_proj.weight"),
            )?;
            ensure_matrix(&down, "down_proj", K3_HIDDEN, DSPARK_INTER)?;
            layers.push(DsparkLayer {
                input_ln: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.input_layernorm.weight"),
                )?,
                qkv,
                o_proj,
                q_norm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.self_attn.q_norm.weight"),
                )?,
                k_norm: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.self_attn.k_norm.weight"),
                )?,
                post_ln: load_tensor_1d(
                    ctx,
                    &shards,
                    &weight_map,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                gate_up,
                down,
            });
        }

        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;
        ensure_matrix(&fc, "fc", K3_HIDDEN, K3_DSPARK_CONTEXT_DIM)?;
        let markov_w1 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w1.weight")?;
        let markov_w2 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w2.weight")?;
        ensure_matrix(&markov_w1, "markov_w1", K3_VOCAB, DSPARK_MARKOV_RANK)?;
        ensure_matrix(&markov_w2, "markov_w2", K3_VOCAB, DSPARK_MARKOV_RANK)?;

        let (cos_host, sin_host) = build_yarn_rope_tables(cache_len);
        let cos_cache = DeviceVec::from_host(ctx, &cos_host)?;
        let sin_cache = DeviceVec::from_host(ctx, &sin_host)?;
        ctx.sync()?;

        Ok(Self {
            layers,
            norm: load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?,
            hidden_norm: load_tensor_1d(ctx, &shards, &weight_map, "hidden_norm.weight")?,
            fc,
            markov_w1,
            markov_w2,
            cos_cache,
            sin_cache,
            cache_len,
        })
    }

    /// The draft KV/rope capacity slot states must be allocated with.
    pub(crate) fn cache_len(&self) -> usize {
        self.cache_len
    }

    /// Propose [`K3_DSPARK_DRAFTS`] draft tokens for each state, batched.
    ///
    /// Dense ops (embedding, norms, q/o/mlp GEMMs, logits) run once over the
    /// `active * 7` batched rows; the varlen ops (context projection, tail
    /// concat, k/v GEMMs, rope, KV copy, attention) loop per request. Each
    /// state's pending context is drained into its draft KV here
    /// (`committed_len` advances by the context length). Eager, always.
    ///
    /// `anchors[i] = (token, position)`: the verified token each block extends
    /// and its sequence position — asserted against the state's own
    /// `committed + pending` walk, so scheduler/draft position drift crashes
    /// instead of silently proposing from the wrong rope phase.
    pub(crate) fn propose(
        &self,
        ctx: &DeviceContext,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        states: &mut [&mut K3DsparkSlotState],
        anchors: &[(u32, usize)],
        scratch: &mut K3DsparkScratch,
    ) -> Result<Vec<[u32; K3_DSPARK_DRAFTS]>> {
        let active = states.len();
        ensure!(active > 0, "dspark propose needs at least one request");
        ensure!(
            anchors.len() == active,
            "dspark propose: {} states vs {} anchors",
            active,
            anchors.len()
        );
        let block = K3_DSPARK_BLOCK;
        let block_rows = active * block;

        let mut context_lens = Vec::with_capacity(active);
        for (i, state) in states.iter().enumerate() {
            let context_len = state.pending_len;
            ensure!(
                context_len > 0,
                "dspark propose before any captured context (slot index {i})"
            );
            let (_, anchor_pos) = anchors[i];
            ensure!(
                anchor_pos == state.committed_len + context_len,
                "dspark anchor position {anchor_pos} != committed {} + pending {} (slot index {i})",
                state.committed_len,
                context_len
            );
            ensure!(
                state.committed_len + context_len + block <= self.cache_len,
                "dspark draft cache overflow: committed={}, tail={}, cap={}",
                state.committed_len,
                context_len + block,
                self.cache_len
            );
            context_lens.push(context_len);
        }

        scratch.activate(block_rows)?;

        // Block token ids: [anchor, mask x 6] per request.
        scratch.block_token_ids_h[..block_rows].fill(DSPARK_MASK_TOKEN);
        for (i, &(anchor, _)) in anchors.iter().enumerate() {
            scratch.block_token_ids_h[i * block] = anchor;
        }
        {
            let mut dst = scratch.token_ids_d.slice_mut(..block_rows);
            ctx.stream
                .memcpy_htod(&scratch.block_token_ids_h[..block_rows], &mut dst)?;
        }

        for (i, state) in states.iter_mut().enumerate() {
            state.set_context_len(context_lens[i])?;
            state.pending.seq_len = context_lens[i];
        }

        {
            let K3DsparkScratch {
                hidden,
                hidden_out,
                normed,
                q_batch,
                attn_output,
                o_buf,
                gate_out,
                up_out,
                act_out,
                logits_normed,
                logits,
                tail_input,
                k_tail,
                v_tail,
                token_ids_d,
                ..
            } = &mut *scratch;

            embedding_batch(ctx, embed, token_ids_d, hidden)?;
            // Project each state's pending context once, up front.
            for state in states.iter_mut() {
                gemm_into_checked(ctx, &self.fc, &state.pending, &mut state.context_projected)?;
                rms_norm_batch_into(
                    ctx,
                    &state.context_projected,
                    &self.hidden_norm,
                    DSPARK_RMS_EPS,
                    &mut state.context_hidden,
                );
            }
            for (l, layer) in self.layers.iter().enumerate() {
                // Batch-wide dense prolog: input norm + q GEMM.
                rms_norm_batch_into(ctx, hidden, &layer.input_ln, DSPARK_RMS_EPS, normed);
                gemm_rows_into_checked(ctx, &layer.qkv, 0, DSPARK_Q_DIM, normed, q_batch)?;
                // Per-slot varlen middle: the tail scratch is shared, so each
                // slot's prep must be consumed before the next overwrites it.
                for (i, state) in states.iter_mut().enumerate().take(active) {
                    let context_len = context_lens[i];
                    let tail_len = context_len + block;
                    let row_offset = i * block;
                    tail_input.seq_len = tail_len;
                    k_tail.seq_len = tail_len;
                    v_tail.seq_len = tail_len;
                    copy_hidden_token_range_into(
                        ctx,
                        &state.context_hidden,
                        0,
                        tail_input,
                        0,
                        context_len,
                    )?;
                    copy_hidden_token_range_into(
                        ctx,
                        normed,
                        row_offset,
                        tail_input,
                        context_len,
                        block,
                    )?;
                    gemm_rows_into_checked(
                        ctx,
                        &layer.qkv,
                        DSPARK_Q_DIM,
                        DSPARK_KV_DIM,
                        tail_input,
                        k_tail,
                    )?;
                    gemm_rows_into_checked(
                        ctx,
                        &layer.qkv,
                        DSPARK_Q_DIM + DSPARK_KV_DIM,
                        DSPARK_KV_DIM,
                        tail_input,
                        v_tail,
                    )?;
                    dflash_qk_norm_rope_into(
                        ctx,
                        q_batch,
                        row_offset,
                        block,
                        k_tail,
                        &layer.q_norm,
                        &layer.k_norm,
                        &self.cos_cache,
                        &self.sin_cache,
                        DSPARK_HEADS,
                        DSPARK_KV_HEADS,
                        DSPARK_HEAD_DIM,
                        state.committed_len + context_len,
                        state.committed_len,
                        DSPARK_RMS_EPS,
                    )?;
                    let cache = &mut state.layers[l];
                    copy_hidden_token_range_into(
                        ctx,
                        k_tail,
                        0,
                        &mut cache.k,
                        state.committed_len,
                        tail_len,
                    )?;
                    copy_hidden_token_range_into(
                        ctx,
                        v_tail,
                        0,
                        &mut cache.v,
                        state.committed_len,
                        tail_len,
                    )?;
                    single_prefill_nhd_noncausal_into(
                        ctx,
                        q_batch,
                        row_offset,
                        block,
                        &cache.k,
                        &cache.v,
                        attn_output,
                        DSPARK_HEADS,
                        DSPARK_KV_HEADS,
                        DSPARK_HEAD_DIM,
                        state.committed_len + tail_len,
                    )?;
                }
                // Dense tail: o_proj + post-norm + MLP + residual.
                gemm_into_checked(ctx, &layer.o_proj, attn_output, o_buf)?;
                fused_add_rms_norm_round_batch_into(
                    ctx,
                    hidden,
                    o_buf,
                    &layer.post_ln,
                    DSPARK_RMS_EPS,
                    normed,
                )?;
                gemm_rows_into_checked(ctx, &layer.gate_up, 0, DSPARK_INTER, normed, gate_out)?;
                gemm_rows_into_checked(
                    ctx,
                    &layer.gate_up,
                    DSPARK_INTER,
                    DSPARK_INTER,
                    normed,
                    up_out,
                )?;
                silu_mul_batch_into(ctx, gate_out, up_out, act_out)?;
                gemm_into_checked(ctx, &layer.down, act_out, o_buf)?;
                add_batch_into(ctx, hidden, o_buf, hidden_out)?;
                std::mem::swap(&mut *hidden, &mut *hidden_out);
            }
            rms_norm_batch_into(ctx, hidden, &self.norm, DSPARK_RMS_EPS, logits_normed);
            gemm_into_checked(ctx, lm_head, logits_normed, logits)?;
        }
        // Host bookkeeping: the pending context is drained into the draft KV.
        for (i, state) in states.iter_mut().enumerate() {
            state.pending_len = 0;
            state.pending.seq_len = 0;
            state.committed_len += context_lens[i];
        }

        self.markov_propose(ctx, anchors, scratch)
    }

    /// Anchor-drop Markov sampling: 6 sequential steps reading block rows
    /// `0..=5` (row 0 is the anchor row), `prev(0) = anchor`, `prev(k) =
    /// draft k-1`; each step is one embedding gather + one rank-256 GEMM +
    /// one strided argmax-with-bias.
    fn markov_propose(
        &self,
        ctx: &DeviceContext,
        anchors: &[(u32, usize)],
        scratch: &mut K3DsparkScratch,
    ) -> Result<Vec<[u32; K3_DSPARK_DRAFTS]>> {
        let rows = anchors.len();
        let block = K3_DSPARK_BLOCK;
        scratch.w1emb.seq_len = rows;
        scratch.bias.seq_len = rows;

        let anchor_tokens: Vec<u32> = anchors.iter().map(|&(token, _)| token).collect();
        {
            let mut prev = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(&anchor_tokens, &mut prev)?;
        }
        // Fixed-orientation ping-pong between the two token buffers.
        //
        // Draft `k` reads block-row `k`'s logits: row 0 is the anchor row and
        // its logits predict the token right after the anchor — the reference
        // `run_markov_block` starts at row 0, and starting at row 1 instead
        // proposes every draft one position ahead of where verify compares it
        // (the acceptance-collapse bug this line used to be).
        for step in 0..block - 1 {
            let (prev, next): (&CudaSlice<u32>, &mut CudaSlice<u32>) = if step % 2 == 0 {
                (&scratch.prev_tokens, &mut scratch.next_tokens)
            } else {
                (&scratch.next_tokens, &mut scratch.prev_tokens)
            };
            embedding_batch(ctx, &self.markov_w1, prev, &mut scratch.w1emb)?;
            gemm_into_checked(ctx, &self.markov_w2, &scratch.w1emb, &mut scratch.bias)?;
            markov_step_argmax_into(
                ctx,
                &scratch.logits,
                &scratch.bias,
                block,
                step,
                rows,
                &mut scratch.partial_values,
                &mut scratch.partial_indices,
                next,
                &mut scratch.sampled_tokens,
            )?;
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block);
        let sampled = ctx.stream.clone_dtoh(&sampled_view)?;
        ctx.sync()?;
        Ok((0..rows)
            .map(|i| std::array::from_fn(|k| sampled[i * block + k]))
            .collect())
    }
}

/// Rank-level draft scratch, allocated once for the whole slot batch. Dense
/// buffers hold `max_slots * block` rows; the varlen tail buffers hold one
/// request's `context + block` rows, preallocated to the draft cache cap.
pub(crate) struct K3DsparkScratch {
    max_rows: usize,
    block_token_ids_h: Vec<u32>,
    token_ids_d: CudaSlice<u32>,
    hidden: HiddenStates,
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,
    logits_normed: HiddenStates,
    logits: HiddenStates,
    tail_input: HiddenStates,
    k_tail: HiddenStates,
    v_tail: HiddenStates,
    // Markov sample-loop scratch.
    w1emb: HiddenStates,
    bias: HiddenStates,
    partial_values: CudaSlice<f32>,
    partial_indices: CudaSlice<i32>,
    prev_tokens: CudaSlice<u32>,
    next_tokens: CudaSlice<u32>,
    sampled_tokens: CudaSlice<u32>,
}

impl K3DsparkScratch {
    pub(crate) fn new(ctx: &DeviceContext, max_slots: usize, cache_len: usize) -> Result<Self> {
        let max_rows = max_slots * K3_DSPARK_BLOCK;
        let tail_capacity = cache_len;
        let partials = markov_step_argmax_partials_len(max_slots, K3_VOCAB);
        Ok(Self {
            max_rows,
            block_token_ids_h: vec![DSPARK_MASK_TOKEN; max_rows],
            token_ids_d: ctx.stream.alloc_zeros(max_rows)?,
            hidden: HiddenStates::zeros(ctx, K3_HIDDEN, max_rows)?,
            hidden_out: HiddenStates::zeros(ctx, K3_HIDDEN, max_rows)?,
            normed: HiddenStates::zeros(ctx, K3_HIDDEN, max_rows)?,
            q_batch: HiddenStates::zeros(ctx, DSPARK_Q_DIM, max_rows)?,
            attn_output: HiddenStates::zeros(ctx, DSPARK_Q_DIM, max_rows)?,
            o_buf: HiddenStates::zeros(ctx, K3_HIDDEN, max_rows)?,
            gate_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            up_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            act_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            logits_normed: HiddenStates::zeros(ctx, K3_HIDDEN, max_rows)?,
            logits: HiddenStates::zeros(ctx, K3_VOCAB, max_rows)?,
            tail_input: HiddenStates::zeros(ctx, K3_HIDDEN, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, DSPARK_KV_DIM, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, DSPARK_KV_DIM, tail_capacity)?,
            w1emb: HiddenStates::zeros(ctx, DSPARK_MARKOV_RANK, max_slots)?,
            bias: HiddenStates::zeros(ctx, K3_VOCAB, max_slots)?,
            partial_values: ctx.stream.alloc_zeros(partials)?,
            partial_indices: ctx.stream.alloc_zeros(partials)?,
            prev_tokens: ctx.stream.alloc_zeros(max_slots)?,
            next_tokens: ctx.stream.alloc_zeros(max_slots)?,
            sampled_tokens: ctx.stream.alloc_zeros(max_rows)?,
        })
    }

    /// Point the dense buffers at the active prefix (never reallocates).
    fn activate(&mut self, block_rows: usize) -> Result<()> {
        ensure!(
            block_rows <= self.max_rows,
            "dspark batch {block_rows} rows exceeds scratch capacity {}",
            self.max_rows
        );
        self.hidden.seq_len = block_rows;
        self.hidden_out.seq_len = block_rows;
        self.normed.seq_len = block_rows;
        self.q_batch.seq_len = block_rows;
        self.attn_output.seq_len = block_rows;
        self.o_buf.seq_len = block_rows;
        self.gate_out.seq_len = block_rows;
        self.up_out.seq_len = block_rows;
        self.act_out.seq_len = block_rows;
        self.logits_normed.seq_len = block_rows;
        self.logits.seq_len = block_rows;
        Ok(())
    }
}

mod slot;
pub(crate) use slot::K3DsparkSlotState;

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic bf16 value, bit-identical to the torch
    /// reference's `synth` (splitmix-style hash, byte lane 32..39, exact
    /// bf16 grid `[-128, 127] / 512`).
    fn synth_val(idx: u64, seed: u64) -> bf16 {
        let x = idx.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let b = ((x >> 32) & 0xFF) as i64 - 128;
        bf16::from_f32(b as f32 / 512.0)
    }

    fn synth_matrix(
        ctx: &DeviceContext,
        seed: u64,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceMatrix> {
        let mut host = vec![bf16::ZERO; rows * cols];
        for (i, v) in host.iter_mut().enumerate() {
            *v = synth_val(i as u64, seed);
        }
        DeviceMatrix::from_host(ctx, &host, rows, cols)
    }

    /// Numeric cross-check against the checkpoint's own `dflash.py` /
    /// `dspark.py` math: real drafter weights, synthetic embed/lm_head and
    /// context rows shared bit-exactly with the torch reference
    /// (`k3_dspark_reference.py`), compare drafts and per-row top-8 logits.
    /// Ignored: needs a GPU and the checkpoint
    /// (`PEGAINFER_K3_TEST_DSPARK`, default `/mnt/shared/weights/kimi-k3-dspark`).
    fn append_synth_rows(
        ctx: &DeviceContext,
        state: &mut K3DsparkSlotState,
        seed: u64,
        rows: usize,
    ) -> Result<()> {
        let mut host = vec![bf16::ZERO; rows * K3_DSPARK_CONTEXT_DIM];
        for (i, v) in host.iter_mut().enumerate() {
            *v = synth_val(i as u64, seed);
        }
        let start = state.pending_len;
        let mut dst = state
            .pending
            .data
            .slice_mut(start * K3_DSPARK_CONTEXT_DIM..(start + rows) * K3_DSPARK_CONTEXT_DIM);
        ctx.stream.memcpy_htod(&host, &mut dst)?;
        state.pending_len = start + rows;
        state.pending.seq_len = start + rows;
        Ok(())
    }

    fn print_round(
        ctx: &DeviceContext,
        scratch: &K3DsparkScratch,
        round: usize,
        drafts: &[u32; K3_DSPARK_DRAFTS],
    ) -> Result<()> {
        let logits_view = scratch.logits.data.slice(..K3_DSPARK_BLOCK * K3_VOCAB);
        let logits_h: Vec<bf16> = ctx.stream.clone_dtoh(&logits_view)?;
        ctx.sync()?;
        println!("round {round} drafts: {drafts:?}");
        for row in 0..K3_DSPARK_BLOCK {
            let base = row * K3_VOCAB;
            let mut idx: Vec<usize> = (0..K3_VOCAB).collect();
            idx.sort_by(|&a, &b| {
                logits_h[base + b]
                    .to_f32()
                    .total_cmp(&logits_h[base + a].to_f32())
            });
            let top: Vec<String> = idx[..8]
                .iter()
                .map(|&i| format!("{i}:{:.4}", logits_h[base + i].to_f32()))
                .collect();
            println!("round {round} row {row} top8: {}", top.join(" "));
        }
        Ok(())
    }

    /// Numeric cross-check against the checkpoint's own `dflash.py` /
    /// `dspark.py` math: real drafter weights, synthetic embed/lm_head and
    /// context rows shared bit-exactly with the torch reference
    /// (`k3_dspark_reference.py`), compare drafts and per-row top-8 logits.
    /// Round 1 runs at serving-scale context (197 rows, cold cache); round 2
    /// exercises the cached-KV + rope-offset path (committed 197, 5 fresh
    /// rows). Ignored: needs a GPU and the checkpoint
    /// (`PEGAINFER_K3_TEST_DSPARK`, default `/mnt/shared/weights/kimi-k3-dspark`).
    #[test]
    #[ignore]
    fn dspark_reference_cross_check() -> Result<()> {
        const T1: usize = 197;
        const T2: usize = 5;
        const ANCHOR1: (u32, usize) = (777, T1);
        const ANCHOR2: (u32, usize) = (888, T1 + T2);

        let path = std::env::var("PEGAINFER_K3_TEST_DSPARK")
            .unwrap_or_else(|_| "/mnt/shared/weights/kimi-k3-dspark".into());
        let ctx = DeviceContext::new()?;
        let model = K3DsparkModel::load(&ctx, Path::new(&path), 4096)?;
        let embed = synth_matrix(&ctx, 0x0101, K3_VOCAB, K3_HIDDEN)?;
        let lm_head = synth_matrix(&ctx, 0x0202, K3_VOCAB, K3_HIDDEN)?;

        let mut state = K3DsparkSlotState::new(&ctx, model.cache_len())?;
        let mut scratch = K3DsparkScratch::new(&ctx, 1, model.cache_len())?;

        append_synth_rows(&ctx, &mut state, 0x0303, T1)?;
        let drafts = model.propose(
            &ctx,
            &embed,
            &lm_head,
            &mut [&mut state],
            &[ANCHOR1],
            &mut scratch,
        )?;
        print_round(&ctx, &scratch, 1, &drafts[0])?;

        append_synth_rows(&ctx, &mut state, 0x0404, T2)?;
        let drafts = model.propose(
            &ctx,
            &embed,
            &lm_head,
            &mut [&mut state],
            &[ANCHOR2],
            &mut scratch,
        )?;
        print_round(&ctx, &scratch, 2, &drafts[0])?;
        Ok(())
    }
}
