//! Qwen3.8-Flash-Next QSA: token-indexer block scoring and sparse attention.
//!
//! Turns the fused `index_qk_proj` output into one score per (query row,
//! visible micro-block); selection — top-k plus the unconditional tail — is a
//! later stage, deliberately not here. Frozen semantics live in
//! `csrc/qwen4/qwen4_qsa_indexer_score.cu` (transformers 5.16.1 `qwen4_exp`
//! `Qwen4ExpTextQSAIndexer.forward`); the kernel is compiled for one geometry
//! and this wrapper sizes its buffers from it. The checkpoint's own indexer
//! fields are checked where the model crate reads its config, once that caller
//! lands — this stage takes only `rms_eps` from the caller.
//!
//! The sparse-attention forward below is the middle of the three units the QSA
//! bring-up plan splits the layer into: the indexer scores and selects, this
//! consumes the selection as a visibility mask and produces the gated attention
//! output that `o_proj` reads. The mask is therefore an input, never derived
//! here. Frozen semantics live in `csrc/qwen4/qwen4_qsa_attn_forward.cu`.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::active_cu_stream;

/// The geometry the kernel is compiled for, used to size this wrapper's buffer
/// checks. The kernel reads with the same numbers: four query heads and one
/// shared key head at 128 (a 640-wide fused row), four tokens per micro-block,
/// and 64 rotated dims — also the shared attention table's row stride.
const FUSED_WIDTH: usize = 640;
const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 64;
const COMPRESS_RATIO: usize = 4;

/// Score every complete micro-block of `fused_qk` for every query row.
///
/// `fused_qk` is the whole fused `index_qk_proj` output, one row per token:
/// `FUSED_WIDTH` columns, the query heads first and the raw key head last.
/// The key array must start at token 0 — the micro-block grid starts at the
/// first visible token and this kernel reads it as token 0 (see the kernel's
/// grid contract).
///
/// `rows[j]` is each query row's position (the RoPE position and the
/// `fused_qk` row its query heads take) and `visible_len[j]` how many tokens
/// it can see, counted from token 0 and at most `rows[j] + 1`, which is what
/// causal text passes. Row `j` gets `visible_len[j] / COMPRESS_RATIO` scores at
/// stride `score_cols`;
/// entries past that are left untouched, as the reference leaves them NaN — a
/// caller that wants to tell an unwritten entry from a real one has to poison
/// the buffer itself (the parity gate pre-fills a sentinel). Both arrays are
/// checked on the device: a row outside the key array, or a visible length past
/// the row's own token, traps, so a broken caller fails visibly instead of
/// getting a plausible score for blocks it should not see.
///
/// `pooled_keys` is the workspace: `tokens / COMPRESS_RATIO` rows of
/// `HEAD_DIM`, reused across calls.
///
/// `cos`/`sin` are the shared attention RoPE tables (`pegainfer_core::rope`,
/// `[max_pos, ROTARY_DIM]`, position-major): the indexer rotates with the same
/// position embeddings the attention uses, so this stage computes no trig. They
/// must be built with `rotary_dim` and `frequency_dim` both `ROTARY_DIM`, the
/// frequencies the reference's partial RoPE uses, and cover every position this
/// call can read, `0..tokens`.
pub fn qsa_indexer_block_scores_into(
    ctx: &DeviceContext,
    fused_qk: &CudaSlice<bf16>,
    q_norm_weight: &CudaSlice<bf16>,
    k_norm_weight: &CudaSlice<bf16>,
    cos: &DeviceVec,
    sin: &DeviceVec,
    rows: &CudaSlice<i32>,
    visible_len: &CudaSlice<i32>,
    tokens: usize,
    scores: &mut CudaSlice<f32>,
    score_cols: usize,
    pooled_keys: &mut CudaSlice<bf16>,
    rms_eps: f32,
) -> Result<()> {
    // NaN fails the comparison, as it does in the kernel's own guard.
    ensure!(
        rms_eps >= 0.0,
        "QSA indexer rms_norm_eps {rms_eps} must be non-negative"
    );
    let blocks = tokens / COMPRESS_RATIO;
    ensure!(
        score_cols >= blocks,
        "QSA indexer score_cols {score_cols} cannot hold {blocks} blocks"
    );
    let query_rows = rows.len();
    ensure!(
        visible_len.len() >= query_rows,
        "QSA indexer visible_len {} shorter than the {query_rows} query rows",
        visible_len.len()
    );
    ensure!(
        fused_qk.len() >= tokens.saturating_mul(FUSED_WIDTH),
        "QSA indexer fused_qk holds {} bf16, needs {} for {tokens} tokens",
        fused_qk.len(),
        tokens.saturating_mul(FUSED_WIDTH)
    );
    ensure!(
        q_norm_weight.len() >= HEAD_DIM && k_norm_weight.len() >= HEAD_DIM,
        "QSA indexer norm weights must hold {HEAD_DIM} bf16"
    );
    let table_len = tokens.saturating_mul(ROTARY_DIM);
    ensure!(
        cos.len >= table_len && sin.len >= table_len,
        "QSA indexer cos/sin tables hold {}/{} bf16, need {table_len} for {tokens} positions",
        cos.len,
        sin.len
    );
    ensure!(
        scores.len() >= query_rows.saturating_mul(score_cols),
        "QSA indexer scores holds {} f32, needs {} for {query_rows} x {score_cols}",
        scores.len(),
        query_rows.saturating_mul(score_cols)
    );
    ensure!(
        pooled_keys.len() >= blocks.saturating_mul(HEAD_DIM),
        "QSA indexer pooled_keys workspace holds {} bf16, needs {}",
        pooled_keys.len(),
        blocks.saturating_mul(HEAD_DIM)
    );
    if query_rows == 0 || tokens == 0 {
        return Ok(());
    }
    // The kernel takes extents as i32; checked before any pointer is formed.
    let query_rows = checked_i32(query_rows, "QSA indexer query rows")?;
    let tokens = checked_i32(tokens, "QSA indexer tokens")?;
    let score_cols = checked_i32(score_cols, "QSA indexer score columns")?;

    let (fused_ptr, _fused_guard) = fused_qk.device_ptr(&ctx.stream);
    let (q_norm_ptr, _q_norm_guard) = q_norm_weight.device_ptr(&ctx.stream);
    let (k_norm_ptr, _k_norm_guard) = k_norm_weight.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.data.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.data.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = rows.device_ptr(&ctx.stream);
    let (visible_ptr, _visible_guard) = visible_len.device_ptr(&ctx.stream);
    let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
    let (pooled_ptr, _pooled_guard) = pooled_keys.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_qsa_indexer_block_scores_cuda(
            fused_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            rows_ptr as *const i32,
            visible_ptr as *const i32,
            scores_ptr as *mut f32,
            pooled_ptr as *mut ffi::Half,
            query_rows,
            tokens,
            score_cols,
            rms_eps,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer block score launch failed: {err}"))
}

/// The attention geometry the QSA kernels are compiled for. The kernel body
/// carries these as constants and refuses a call whose rotary geometry disagrees
/// with them, so this struct exists to hold the same numbers on the Rust side:
/// callers size their buffers with it, and the parity gate holds it against the
/// frozen config the goldens were dumped from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaAttnGeometry {
    /// `num_attention_heads`.
    pub heads: usize,
    /// `num_key_value_heads`; head `h` reads KV head `h / group()`.
    pub kv_heads: usize,
    /// `head_dim`, and the width the norms and the RoPE split work on.
    pub head_dim: usize,
    /// `head_dim * partial_rotary_factor`: the rotated prefix of a head.
    pub rotary_dim: usize,
    /// `rope_parameters.rope_theta`.
    pub rope_theta: f32,
    /// `rms_norm_eps`.
    pub rms_eps: f32,
}

impl QsaAttnGeometry {
    /// The longest key array one call accepts: a row of float32 scores lives in
    /// dynamic shared memory, and a longer context needs the streaming softmax
    /// rather than a longer row.
    ///
    /// The kernel carries the same budget as a compiled constant and refuses past
    /// it, so a drift between the two fails closed either way — and the parity
    /// gate holds them together, probing the kernel's copy with this constant and
    /// running one call at exactly it.
    pub const MAX_TOKENS: usize = 8192;

    /// The frozen Qwen3.8-Flash-Next text geometry.
    pub const fn qwen38_flash_next() -> Self {
        Self {
            heads: 24,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 1.0e7,
            rms_eps: 1.0e-6,
        }
    }

    /// Query heads per KV head (`num_key_value_groups`).
    pub const fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    /// One fused `q_proj` row: every head's query dims followed by that head's
    /// output-gate dims.
    pub const fn query_width(&self) -> usize {
        self.heads * self.head_dim * 2
    }

    /// One `k_proj` or `v_proj` row.
    pub const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// One `o_proj` input row, head-major.
    pub const fn out_width(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// Replay the QSA sparse-attention forward over one layer's captured
/// activations and write both of the reference's observable outputs.
///
/// `queries` holds `query_rows` fused `q_proj` rows ([`QsaAttnGeometry::query_width`]
/// wide); `rows` gives the token each of those rows belongs to and `positions`
/// the position of every one of the `tokens` tokens, so the evaluated rows may be
/// any subset in any order. `mask` is the caller's visibility, one byte per
/// (query row, token), non-zero where visible — the indexer's selection AND
/// causality, which this call never re-derives. `normed_keys` is the key pass's
/// workspace and must hold `tokens * geometry.kv_width()` bf16.
///
/// Every element of `attn_weights` (`heads * query_rows * tokens`, head-major as
/// the reference returns them) and of `attn_out` (`query_rows * out_width`, the
/// input to `o_proj`) is written.
pub fn qsa_attn_forward_into(
    ctx: &DeviceContext,
    geometry: &QsaAttnGeometry,
    queries: &CudaSlice<bf16>,
    keys: &CudaSlice<bf16>,
    values: &CudaSlice<bf16>,
    mask: &CudaSlice<u8>,
    rows: &CudaSlice<i32>,
    positions: &CudaSlice<i32>,
    q_norm_weight: &CudaSlice<bf16>,
    k_norm_weight: &CudaSlice<bf16>,
    attn_weights: &mut CudaSlice<bf16>,
    attn_out: &mut CudaSlice<bf16>,
    normed_keys: &mut CudaSlice<bf16>,
) -> Result<()> {
    let query_rows = rows.len();
    let tokens = positions.len();
    let query_width = geometry.query_width();
    let kv_width = geometry.kv_width();
    ensure!(
        queries.len() >= query_rows * query_width,
        "qsa_attn_forward_into queries hold {} bf16, need {query_rows} rows of {query_width}",
        queries.len()
    );
    ensure!(
        keys.len() >= tokens * kv_width && values.len() >= tokens * kv_width,
        "qsa_attn_forward_into keys {} / values {} bf16, need {tokens} rows of {kv_width}",
        keys.len(),
        values.len()
    );
    ensure!(
        mask.len() >= query_rows * tokens,
        "qsa_attn_forward_into mask holds {} bytes, need {query_rows}x{tokens}",
        mask.len()
    );
    ensure!(
        q_norm_weight.len() >= geometry.head_dim && k_norm_weight.len() >= geometry.head_dim,
        "qsa_attn_forward_into norm weights {} / {} bf16, need {}",
        q_norm_weight.len(),
        k_norm_weight.len(),
        geometry.head_dim
    );
    ensure!(
        attn_weights.len() >= geometry.heads * query_rows * tokens,
        "qsa_attn_forward_into attn_weights holds {} bf16, need {}x{query_rows}x{tokens}",
        attn_weights.len(),
        geometry.heads
    );
    ensure!(
        attn_out.len() >= query_rows * geometry.out_width(),
        "qsa_attn_forward_into attn_out holds {} bf16, need {query_rows} rows of {}",
        attn_out.len(),
        geometry.out_width()
    );
    ensure!(
        normed_keys.len() >= tokens * kv_width,
        "qsa_attn_forward_into normed_keys workspace holds {} bf16, need {tokens} rows of {kv_width}",
        normed_keys.len()
    );
    ensure!(
        tokens <= QsaAttnGeometry::MAX_TOKENS,
        "qsa_attn_forward_into: {tokens} tokens exceed the {} the score row's \
         shared-memory budget allows; a longer context needs the streaming softmax",
        QsaAttnGeometry::MAX_TOKENS
    );

    let (q_ptr, _gq) = queries.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = keys.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = values.device_ptr(&ctx.stream);
    let (m_ptr, _gm) = mask.device_ptr(&ctx.stream);
    let (r_ptr, _gr) = rows.device_ptr(&ctx.stream);
    let (p_ptr, _gp) = positions.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = attn_weights.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = attn_out.device_ptr_mut(&ctx.stream);
    let (nk_ptr, _gnk) = normed_keys.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen4_qsa_attn_forward_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            m_ptr as *const u8,
            r_ptr as *const i32,
            p_ptr as *const i32,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            w_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            nk_ptr as *mut ffi::Half,
            checked_i32(query_rows, "qsa_attn_forward_into query_rows")?,
            checked_i32(tokens, "qsa_attn_forward_into tokens")?,
            geometry.rotary_dim as i32,
            geometry.rope_theta,
            geometry.rms_eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}
