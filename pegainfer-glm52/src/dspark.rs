//! GLM5.2 DSpark draft lane (rank-local): the community
//! `RedHatAI/GLM-5.2-speculator.dspark` drafter — a 5-layer qwen3-architecture
//! dense backbone at GLM's hidden 6144 plus a rank-256 Markov head — proposing
//! 7 greedy draft tokens per round from the target's captured aux hidden
//! states. No collectives anywhere: the draft is replicated on every rank and
//! runs between that rank's steps, DP over its slots.
//!
//! Layout facts are pinned against the `vllm-project/speculators` source
//! (docs/models/glm52/dspark-mtp.md "Layout pinned against speculators
//! source"): the block input is `[anchor, mask x 7]` and block position 0 is
//! the anchor, NOT a draft (anchor-drop) — drafts are read from block
//! positions 1..=7, and the Markov loop starts at position 1 with
//! `prev(1) = anchor`. The context rows the fc projection consumes are the
//! target's residual stream captured AFTER `GLM52_DSPARK_AUX_LAYERS` (see the
//! constant's comment for the off-by-one against the checkpoint's ids), and
//! the pending context always ends one row before the anchor: the anchor's
//! own hidden is only captured when the anchor is fed to the target.
//!
//! The draft's `embed_tokens`/`lm_head` are byte-identical to the target's
//! (sha256-compared at checkpoint inspection) and are NOT loaded — the block
//! embedding and the draft logits reuse the target's matrices.

use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
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

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_VOCAB;
use crate::model::GLM52_MAX_BATCH_PER_RANK;

/// Draft block width: anchor + 7 mask positions. Equals the top decode bucket
/// — a full verify span is one bucket-8 step.
pub(crate) const GLM52_DSPARK_BLOCK: usize = 8;

/// Drafts proposed per round (anchor-drop: block position 0 is the anchor).
pub(crate) const GLM52_DSPARK_DRAFTS: usize = GLM52_DSPARK_BLOCK - 1;

/// Target layers whose post-layer residual stream feeds the fc context
/// projection, in fc column order — OUR 0-based "residual after layer L"
/// indexing. The checkpoint config says `aux_hidden_state_layer_ids = [8, 23,
/// 39, 55, 70]`, but vLLM (which produced the training data) captures
/// `hidden + residual` BEFORE running layer `idx` — id k is the residual
/// stream after k layers, i.e. after layer k-1. Capturing after layers
/// {8,23,...} (the qwen3 DFlash-checkpoint convention) would be off by one
/// here.
pub(crate) const GLM52_DSPARK_AUX_LAYERS: [usize; 5] = [7, 22, 38, 54, 69];

/// fc input width: the 5 captured layers' hidden states concatenated per
/// token, in `GLM52_DSPARK_AUX_LAYERS` order.
pub(crate) const GLM52_DSPARK_CONTEXT_DIM: usize = GLM52_DSPARK_AUX_LAYERS.len() * GLM52_HIDDEN;

/// The draft writes `block` transient rows past the committed+context length,
/// and the last verifiable anchor sits at position `max_model_len - 1` — so
/// the draft KV/rope tables need `block` positions of headroom past the
/// target's context cap.
pub(crate) fn dspark_cache_len(max_model_len: usize) -> usize {
    max_model_len + GLM52_DSPARK_BLOCK
}

/// Exact GPU bytes the DSpark lane allocates on a rank for a given context
/// cap — the same terms `load`/`Glm52DsparkScratch::new`/
/// `Glm52DsparkSlotState::new` allocate (everything is preallocated to the
/// cap at load: a mid-serving draft round never touches the allocator, so
/// this ledger IS the allocation, not a shadow of it). Per slot: the
/// per-layer draft KV, the pending captured-context rows, and the projected
/// context pair; rank-wide: the varlen tail scratch and the rope tables.
/// The launch-time VRAM probe charges this before deciding `max_model_len`.
pub(crate) fn glm52_dspark_arena_bytes(max_model_len: usize) -> usize {
    let cache_len = dspark_cache_len(max_model_len);
    let bf16 = size_of::<half::bf16>();
    let kv = DSPARK_LAYERS * 2 * DSPARK_QKV_DIM * cache_len * bf16;
    let pending = GLM52_DSPARK_CONTEXT_DIM * cache_len * bf16;
    let context = 2 * GLM52_HIDDEN * cache_len * bf16;
    let tail = (GLM52_HIDDEN + 2 * DSPARK_QKV_DIM) * cache_len * bf16;
    let rope = 2 * cache_len * DSPARK_HEAD_DIM * bf16;
    // Cap-scaled term: like the native-MTP arena, charge the RUNTIME slot
    // count — the aux-hidden lanes are per admitted slot, not per ceiling.
    crate::model::glm52_decode_slots() * (kv + pending + context) + tail + rope
}

const DSPARK_LAYERS: usize = 5;
const DSPARK_HEADS: usize = 64;
const DSPARK_HEAD_DIM: usize = 64;
/// 64 q-heads == 64 kv-heads (MHA): q and kv projections are both 4096 wide.
const DSPARK_QKV_DIM: usize = DSPARK_HEADS * DSPARK_HEAD_DIM;
const DSPARK_INTER: usize = 12_288;
const DSPARK_MARKOV_RANK: usize = 256;
const DSPARK_MASK_TOKEN: u32 = 154_856;
const DSPARK_ROPE_THETA: f32 = 8_000_000.0;
const DSPARK_RMS_EPS: f32 = 1.0e-5;

struct DsparkLayer {
    input_ln: DeviceVec,
    /// vstacked `[q; k; v]` `[3 * 4096, 6144]`.
    qkv: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    post_ln: DeviceVec,
    /// vstacked `[gate; up]` `[2 * 12288, 6144]`.
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

pub(crate) struct Glm52DsparkModel {
    layers: Vec<DsparkLayer>,
    /// Draft final norm (before the reused target lm_head).
    norm: DeviceVec,
    /// Norm applied to the fc-projected context rows.
    hidden_norm: DeviceVec,
    /// Context projection `[6144, 30720]`.
    fc: DeviceMatrix,
    /// Markov head: `bias(prev) = w2 @ w1[prev]`, both `[154880, 256]`.
    markov_w1: DeviceMatrix,
    markov_w2: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// Draft KV/rope capacity: the target's `max_model_len` plus one block of
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
    expect("speculators_model_type", "dspark".into())?;
    expect("block_size", 8.into())?;
    expect("mask_token_id", (DSPARK_MASK_TOKEN as u64).into())?;
    expect("markov_rank", (DSPARK_MARKOV_RANK as u64).into())?;
    expect("markov_head_type", "vanilla".into())?;
    expect(
        "aux_hidden_state_layer_ids",
        serde_json::json!([8, 23, 39, 55, 70]),
    )?;
    // DeepSpec-style anchor-first checkpoints carry a `num_anchors` marker;
    // this module implements anchor-drop only.
    ensure!(
        json.get("num_anchors").is_none(),
        "dspark config has `num_anchors` (a DeepSpec anchor-first marker); \
         this module implements the speculators anchor-drop layout only"
    );
    let tl = json
        .get("transformer_layer_config")
        .context("dspark config missing transformer_layer_config")?;
    for (field, want) in [
        ("model_type", serde_json::json!("qwen3")),
        ("num_hidden_layers", (DSPARK_LAYERS as u64).into()),
        ("hidden_size", (GLM52_HIDDEN as u64).into()),
        ("num_attention_heads", (DSPARK_HEADS as u64).into()),
        ("num_key_value_heads", (DSPARK_HEADS as u64).into()),
        ("head_dim", (DSPARK_HEAD_DIM as u64).into()),
        ("intermediate_size", (DSPARK_INTER as u64).into()),
        ("vocab_size", (GLM52_VOCAB as u64).into()),
        ("use_sliding_window", false.into()),
    ] {
        let got = tl
            .get(field)
            .with_context(|| format!("dspark transformer_layer_config missing `{field}`"))?;
        ensure!(
            *got == want,
            "dspark transformer_layer_config `{field}` = {got}, this build expects {want}"
        );
    }
    Ok(())
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

impl Glm52DsparkModel {
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
            ensure_matrix(&q, "q_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
            ensure_matrix(&k, "k_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
            ensure_matrix(&v, "v_proj", DSPARK_QKV_DIM, GLM52_HIDDEN)?;
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
            ensure_matrix(&gate, "gate_proj", DSPARK_INTER, GLM52_HIDDEN)?;
            ensure_matrix(&up, "up_proj", DSPARK_INTER, GLM52_HIDDEN)?;
            let gate_up = DeviceMatrix::vstack(ctx, &[&gate, &up])?;
            drop((gate, up));
            let o_proj = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.self_attn.o_proj.weight"),
            )?;
            ensure_matrix(&o_proj, "o_proj", GLM52_HIDDEN, DSPARK_QKV_DIM)?;
            let down = load_tensor_2d(
                ctx,
                &shards,
                &weight_map,
                &format!("{p}.mlp.down_proj.weight"),
            )?;
            ensure_matrix(&down, "down_proj", GLM52_HIDDEN, DSPARK_INTER)?;
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
        ensure_matrix(&fc, "fc", GLM52_HIDDEN, GLM52_DSPARK_CONTEXT_DIM)?;
        let markov_w1 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w1.weight")?;
        let markov_w2 = load_tensor_2d(ctx, &shards, &weight_map, "markov_head.markov_w2.weight")?;
        ensure_matrix(&markov_w1, "markov_w1", GLM52_VOCAB, DSPARK_MARKOV_RANK)?;
        ensure_matrix(&markov_w2, "markov_w2", GLM52_VOCAB, DSPARK_MARKOV_RANK)?;

        // embed_tokens / lm_head / confidence_head are intentionally not
        // loaded: the first two are byte-identical to the target's, the
        // confidence head is Phase 2.
        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: DSPARK_HEAD_DIM,
                frequency_dim: DSPARK_HEAD_DIM,
                max_seq_len: cache_len,
                theta: DSPARK_ROPE_THETA,
            },
        )?;
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

    /// Propose `GLM52_DSPARK_DRAFTS` draft tokens for each state, batched.
    ///
    /// Dense ops (embedding, norms, q/o/mlp GEMMs, logits) run once over the
    /// `active * 8` batched buffers; the varlen ops (context projection, tail
    /// concat, k/v GEMMs, rope, KV copy, attention) loop per request — the
    /// same split as the qwen3 DFlash lane. Each state's pending context is
    /// drained into its draft KV here (`committed_len` advances by the
    /// context length).
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
        states: &mut [&mut Glm52DsparkSlotState],
        anchors: &[(u32, usize)],
        scratch: &mut Glm52DsparkScratch,
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let active = states.len();
        ensure!(active > 0, "dspark propose needs at least one request");
        ensure!(
            anchors.len() == active,
            "dspark propose: {} states vs {} anchors",
            active,
            anchors.len()
        );
        let block = GLM52_DSPARK_BLOCK;
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
            let tail_len = context_len + block;
            ensure!(
                state.committed_len + tail_len <= self.cache_len,
                "dspark draft cache overflow: committed={}, tail={tail_len}, cap={}",
                state.committed_len,
                self.cache_len
            );
            context_lens.push(context_len);
        }

        scratch.activate(block_rows);

        // Block token ids: [anchor, mask x 7] per request.
        scratch.block_token_ids_h[..block_rows].fill(DSPARK_MASK_TOKEN);
        for (i, &(anchor, _)) in anchors.iter().enumerate() {
            scratch.block_token_ids_h[i * block] = anchor;
        }
        {
            let mut dst = scratch.token_ids_d.slice_mut(..block_rows);
            ctx.stream
                .memcpy_htod(&scratch.block_token_ids_h[..block_rows], &mut dst)?;
        }

        // Host metadata that shapes the GEMMs below — hoisted ahead of the
        // graphable region (during replay the closures do not run, and the
        // shapes are baked at capture).
        for (i, state) in states.iter_mut().enumerate() {
            state.set_context_len(context_lens[i])?;
            state.pending.seq_len = context_lens[i];
        }
        let max_tail = context_lens.iter().max().copied().unwrap_or(0) + block;
        ensure!(
            max_tail <= scratch.tail_input.data.len() / GLM52_HIDDEN,
            "dspark tail length {max_tail} exceeds the preallocated cap"
        );
        // The tail seq_lens must be set OUTSIDE the graphable region: a replay
        // does not run the captured closure, but the always-eager dynamic
        // middle consumes these bounds every round — a stale value from a
        // previous accept length either trips the copy range check (engine
        // teardown) or ropes garbage rows. At bs=1 this single assignment is
        // the round's truth; at bs>1 state_prep re-sets them per slot (eager).
        let tail_len0 = context_lens[0] + block;
        scratch.tail_input.seq_len = tail_len0;
        scratch.k_tail.seq_len = tail_len0;
        scratch.v_tail.seq_len = tail_len0;

        // Piecewise forward graph (bs=1): only the `committed_len`-derived
        // arguments vary per round (rope positions, KV-append offsets,
        // attention kv_len — 4 launches/layer, kept EAGER since a captured
        // FlashInfer prefill bakes its KV iteration count). The rest is
        // shape-static per `context_len` → `DSPARK_LAYERS + 1` dense segments,
        // replayed. rows > 1 falls back to eager (shared tail scratch, see
        // state_prep). First round per key stays eager (cuBLAS lazy workspace
        // would abort capture). The hidden/hidden_out swap is host-side and
        // baked by capture; no eager op touches either buffer.
        // DSPARK_NO_FORWARD_GRAPH=1 disables.
        let slot_ident = {
            let (ptr, _guard) = states[0].pending.data.device_ptr(&ctx.stream);
            ptr
        };
        let fwd_key = (slot_ident, context_lens[0]);
        let fwd_on = active == 1
            && context_lens[0] <= GLM52_DSPARK_BLOCK
            && std::env::var_os("DSPARK_NO_FORWARD_GRAPH").is_none();
        let use_graph = fwd_on && scratch.forward_warm.contains(&fwd_key);
        if use_graph {
            scratch
                .forward_graphs
                .entry(fwd_key)
                .or_insert_with(|| (0..=DSPARK_LAYERS).map(|_| CudaGraphState::new()).collect());
        }

        {
            let Glm52DsparkScratch {
                forward_graphs,
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

            // The batch-wide dense prolog of layer `l`: input norm + q GEMM.
            macro_rules! layer_prolog {
                ($l:expr) => {{
                    let layer = &self.layers[$l];
                    rms_norm_batch_into(ctx, hidden, &layer.input_ln, DSPARK_RMS_EPS, normed);
                    gemm_rows_into_checked(ctx, &layer.qkv, 0, DSPARK_QKV_DIM, normed, q_batch)?;
                }};
            }
            // One slot's tail assembly + k/v GEMMs into the SHARED tail
            // scratch. The tail must be consumed (state_dynamic!) before the
            // next slot's prep overwrites it — at `active == 1` the graphed
            // composition satisfies this trivially; at `active > 1` the layer
            // interleaves prep/dynamic per slot exactly like the pre-graph
            // code did.
            macro_rules! state_prep {
                ($l:expr, $i:expr, $state:expr) => {{
                    let layer = &self.layers[$l];
                    let state = $state;
                    let i: usize = $i;
                    {
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
                            DSPARK_QKV_DIM,
                            DSPARK_QKV_DIM,
                            tail_input,
                            k_tail,
                        )?;
                        gemm_rows_into_checked(
                            ctx,
                            &layer.qkv,
                            2 * DSPARK_QKV_DIM,
                            DSPARK_QKV_DIM,
                            tail_input,
                            v_tail,
                        )?;
                    }
                }};
            }
            // The dense tail of layer `l`: o_proj + post-norm + MLP + residual.
            macro_rules! layer_tail {
                ($l:expr) => {{
                    let layer = &self.layers[$l];
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
                }};
            }
            // One slot's round-varying middle — always eager.
            macro_rules! state_dynamic {
                ($l:expr, $i:expr, $state:expr) => {{
                    let layer = &self.layers[$l];
                    let state = $state;
                    let i: usize = $i;
                    {
                        let context_len = context_lens[i];
                        let tail_len = context_len + block;
                        let row_offset = i * block;
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
                            DSPARK_HEADS,
                            DSPARK_HEAD_DIM,
                            state.committed_len + context_len,
                            state.committed_len,
                            DSPARK_RMS_EPS,
                        )?;
                        let cache = &mut state.layers[$l];
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
                            DSPARK_HEADS,
                            DSPARK_HEAD_DIM,
                            state.committed_len + tail_len,
                        )?;
                    }
                }};
            }
            macro_rules! run_seg {
                ($idx:expr, $body:block) => {{
                    if use_graph {
                        forward_graphs
                            .get_mut(&fwd_key)
                            .expect("forward graph entry created before the segments")[$idx]
                            .run_or_capture(ctx, || -> Result<()> { $body Ok(()) })?;
                    } else {
                        (|| -> Result<()> { $body Ok(()) })()?;
                    }
                }};
            }

            macro_rules! context_projection {
                () => {{
                    for state in states.iter_mut() {
                        gemm_into_checked(
                            ctx,
                            &self.fc,
                            &state.pending,
                            &mut state.context_projected,
                        )?;
                        rms_norm_batch_into(
                            ctx,
                            &state.context_projected,
                            &self.hidden_norm,
                            DSPARK_RMS_EPS,
                            &mut state.context_hidden,
                        );
                    }
                }};
            }

            if active == 1 {
                // bs=1: the single slot's prep can sit in the dense segments
                // (nothing else touches the shared tail scratch before the
                // dynamic middle consumes it), which is what makes the
                // piecewise graph composition legal.
                run_seg!(0, {
                    embedding_batch(ctx, embed, token_ids_d, hidden)?;
                    context_projection!();
                    layer_prolog!(0);
                    state_prep!(0, 0, &mut *states[0]);
                });
                for l in 0..DSPARK_LAYERS {
                    state_dynamic!(l, 0, &mut *states[0]);
                    if l + 1 < DSPARK_LAYERS {
                        run_seg!(l + 1, {
                            layer_tail!(l);
                            layer_prolog!(l + 1);
                            state_prep!(l + 1, 0, &mut *states[0]);
                        });
                    } else {
                        // Final segment: last layer tail + draft head logits.
                        run_seg!(l + 1, {
                            layer_tail!(l);
                            rms_norm_batch_into(
                                ctx,
                                hidden,
                                &self.norm,
                                DSPARK_RMS_EPS,
                                logits_normed,
                            );
                            gemm_into_checked(ctx, lm_head, logits_normed, logits)?;
                        });
                    }
                }
            } else {
                // bs>1 (eager, ungraphed): the tail scratch is shared across
                // slots, so each slot's prep must be consumed by its dynamic
                // middle before the next slot's prep overwrites it — the
                // original per-slot interleave.
                embedding_batch(ctx, embed, token_ids_d, hidden)?;
                context_projection!();
                for l in 0..DSPARK_LAYERS {
                    layer_prolog!(l);
                    for (i, state) in states.iter_mut().enumerate().take(active) {
                        state_prep!(l, i, &mut **state);
                        state_dynamic!(l, i, &mut **state);
                    }
                    layer_tail!(l);
                }
                rms_norm_batch_into(ctx, hidden, &self.norm, DSPARK_RMS_EPS, logits_normed);
                gemm_into_checked(ctx, lm_head, logits_normed, logits)?;
            }
        }
        if fwd_on {
            scratch.forward_warm.insert(fwd_key);
        }
        // Host bookkeeping the eager path used to do inline.
        for (i, state) in states.iter_mut().enumerate() {
            state.pending_len = 0;
            state.pending.seq_len = 0;
            state.committed_len += context_lens[i];
        }

        self.markov_propose(ctx, anchors, scratch)
    }

    /// Anchor-drop Markov sampling: 7 sequential steps at block positions
    /// 1..=7, `prev(1) = anchor`, `prev(k) = draft k-1`; each step is one
    /// embedding gather + one rank-256 GEMM + one strided argmax-with-bias.
    fn markov_propose(
        &self,
        ctx: &DeviceContext,
        anchors: &[(u32, usize)],
        scratch: &mut Glm52DsparkScratch,
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let rows = anchors.len();
        let block = GLM52_DSPARK_BLOCK;
        scratch.w1emb.seq_len = rows;
        scratch.bias.seq_len = rows;

        let anchor_tokens: Vec<u32> = anchors.iter().map(|&(token, _)| token).collect();
        {
            let mut prev = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(&anchor_tokens, &mut prev)?;
        }
        // Graph the 7-step chain: everything in the loop is pointer- and
        // shape-static per `rows` (`step` bakes per node; anchor h2d and the
        // d2h stay outside), so after a one-round warm-up (cuBLAS lazy
        // workspace) it replays as ONE launch instead of ~21.
        // DSPARK_NO_MARKOV_GRAPH=1 restores the plain loop.
        let use_graph = std::env::var_os("DSPARK_NO_MARKOV_GRAPH").is_none();
        if use_graph && scratch.markov_warm[rows - 1] {
            let Glm52DsparkScratch {
                markov_graphs,
                w1emb,
                bias,
                logits,
                prev_tokens,
                next_tokens,
                sampled_tokens,
                partial_values,
                partial_indices,
                ..
            } = scratch;
            markov_graphs[rows - 1].run_or_capture(ctx, || {
                for step in 1..block {
                    let (prev, next): (&CudaSlice<u32>, &mut CudaSlice<u32>) = if step % 2 == 1 {
                        (&*prev_tokens, &mut *next_tokens)
                    } else {
                        (&*next_tokens, &mut *prev_tokens)
                    };
                    embedding_batch(ctx, &self.markov_w1, prev, w1emb)?;
                    gemm_into_checked(ctx, &self.markov_w2, w1emb, bias)?;
                    markov_step_argmax_into(
                        ctx,
                        logits,
                        bias,
                        block,
                        step,
                        rows,
                        partial_values,
                        partial_indices,
                        next,
                        sampled_tokens,
                    )?;
                }
                Ok(())
            })?;
            let sampled_view = scratch.sampled_tokens.slice(..rows * block);
            let sampled = ctx.stream.clone_dtoh(&sampled_view)?;
            return Ok((0..rows)
                .map(|i| std::array::from_fn(|k| sampled[i * block + 1 + k]))
                .collect());
        }
        scratch.markov_warm[rows - 1] = true;
        // Fixed-orientation ping-pong, NOT a field swap: captured graphs bake
        // the buffer addresses, and an odd number of swaps on an eager round
        // for another row count would desync the anchor h2d from every
        // already-captured chain.
        for step in 1..block {
            let (prev, next): (&CudaSlice<u32>, &mut CudaSlice<u32>) = if step % 2 == 1 {
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
        Ok((0..rows)
            .map(|i| std::array::from_fn(|k| sampled[i * block + 1 + k]))
            .collect())
    }
}

/// Prefix-match speculative acceptance — ported from
/// `pegainfer-qwen3/src/speculative.rs::accept_prefix_match`.
///
/// * `proposed` — the `K` draft tokens fed after the anchor.
/// * `target_tokens` — the verify step's committed token after each of the
///   `K + 1` span rows (`target_tokens[0]` follows the anchor,
///   `target_tokens[K]` is the model's continuation after the whole run):
///   the fused argmax for a greedy request, the per-row sampled token for a
///   non-greedy one.
///
/// Returns the longest accepted prefix of `proposed` followed by exactly one
/// model token (the correction at the first divergence, or the bonus
/// continuation when every draft is accepted) — always `1..=K + 1` tokens, so
/// a verify step always makes at least one token of progress.
///
/// With sampled `target_tokens` this rule is LOSSLESS speculative sampling
/// for a deterministic (greedy) draft: row `k`'s token is a true sample from
/// the target distribution given the accepted prefix, and it is committed
/// whether or not it matches `proposed[k]` — the match only decides whether
/// the round keeps riding. Every committed token is therefore distributed
/// exactly as plain sampled decode (measured A/B on jz-38 2026-07-06: the
/// full rejection-sampling variant buys ≤ 1.5% throughput over this rule on
/// code at temperature 1.0 — not worth its complexity).
#[must_use]
pub(crate) fn accept_prefix_match(proposed: &[u32], target_tokens: &[u32]) -> Vec<u32> {
    debug_assert_eq!(
        target_tokens.len(),
        proposed.len() + 1,
        "verify must produce one committed token per draft plus a bonus"
    );
    let n = proposed
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut committed = Vec::with_capacity(n + 1);
    committed.extend_from_slice(&proposed[..n]);
    // `n <= proposed.len() < target_tokens.len()`, so this index is valid.
    committed.push(target_tokens[n]);
    committed
}

/// Rank-level draft scratch, allocated once for the whole slot batch. Dense
/// buffers hold `GLM52_MAX_BATCH_PER_RANK * block` rows; the varlen tail
/// buffers hold one request's `context + block` rows and grow on demand.
pub(crate) struct Glm52DsparkScratch {
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
    /// One captured Markov-chain graph per active row count (index `rows-1`),
    /// plus a per-count warm flag: the first round for a count runs the plain
    /// loop so cuBLAS's lazy workspace allocation happens outside capture.
    markov_graphs: Vec<CudaGraphState>,
    markov_warm: Vec<bool>,
    /// Piecewise forward graphs (bs=1 only), keyed by **(slot identity,
    /// context_len)** — the captured segments bake per-slot buffer pointers
    /// (`state.pending`, `state.context_projected`, `state.context_hidden`),
    /// so a graph captured for one slot must never replay for another. Slot
    /// identity is the device address of the slot's `pending` buffer (stable
    /// for the slot's lifetime; the `&mut` reference itself is not). Bounded
    /// by `GLM52_MAX_BATCH_PER_RANK x GLM52_DSPARK_BLOCK` entries.
    forward_graphs: std::collections::HashMap<(u64, usize), Vec<CudaGraphState>>,
    forward_warm: std::collections::HashSet<(u64, usize)>,
}

impl Glm52DsparkScratch {
    pub(crate) fn new(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let max_rows = GLM52_MAX_BATCH_PER_RANK * GLM52_DSPARK_BLOCK;
        // The varlen tail holds one request's context + block rows; context
        // is bounded by the draft cache, so `cache_len` covers every round —
        // preallocated so a draft round never touches the allocator (and the
        // VRAM probe's ledger charged exactly this).
        let tail_capacity = cache_len;
        let partials = markov_step_argmax_partials_len(GLM52_MAX_BATCH_PER_RANK, GLM52_VOCAB);
        Ok(Self {
            block_token_ids_h: vec![DSPARK_MASK_TOKEN; max_rows],
            token_ids_d: ctx.stream.alloc_zeros(max_rows)?,
            hidden: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            hidden_out: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            normed: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            q_batch: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, max_rows)?,
            attn_output: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, max_rows)?,
            o_buf: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            gate_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            up_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            act_out: HiddenStates::zeros(ctx, DSPARK_INTER, max_rows)?,
            logits_normed: HiddenStates::zeros(ctx, GLM52_HIDDEN, max_rows)?,
            logits: HiddenStates::zeros(ctx, GLM52_VOCAB, max_rows)?,
            tail_input: HiddenStates::zeros(ctx, GLM52_HIDDEN, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, DSPARK_QKV_DIM, tail_capacity)?,
            w1emb: HiddenStates::zeros(ctx, DSPARK_MARKOV_RANK, GLM52_MAX_BATCH_PER_RANK)?,
            bias: HiddenStates::zeros(ctx, GLM52_VOCAB, GLM52_MAX_BATCH_PER_RANK)?,
            partial_values: ctx.stream.alloc_zeros(partials)?,
            partial_indices: ctx.stream.alloc_zeros(partials)?,
            prev_tokens: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            next_tokens: ctx.stream.alloc_zeros(GLM52_MAX_BATCH_PER_RANK)?,
            sampled_tokens: ctx.stream.alloc_zeros(max_rows)?,
            markov_graphs: (0..GLM52_MAX_BATCH_PER_RANK)
                .map(|_| CudaGraphState::new())
                .collect(),
            markov_warm: vec![false; GLM52_MAX_BATCH_PER_RANK],
            forward_graphs: std::collections::HashMap::new(),
            forward_warm: std::collections::HashSet::new(),
        })
    }

    /// Point the dense buffers at the active prefix (never reallocates).
    fn activate(&mut self, block_rows: usize) {
        assert!(
            block_rows <= GLM52_MAX_BATCH_PER_RANK * GLM52_DSPARK_BLOCK,
            "dspark batch {block_rows} rows exceeds scratch capacity"
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
    }
}

#[path = "dspark_slot.rs"]
mod slot;
pub(crate) use slot::Glm52DsparkSlotState;

/// Test-only constructors (`synthetic`, `randomize_for_test`) live in a
/// child module file so this one stays under the module size budget; a child
/// module keeps access to the private fields.
#[cfg(test)]
#[path = "dspark_test_support.rs"]
mod test_support;
