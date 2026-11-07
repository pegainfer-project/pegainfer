//! Qwen3.8-Flash-Next (Qwen4 line) kernels.
//!
//! [`qsa_indexer_block_scores_into`] turns the fused `index_qk_proj` output
//! into one score per (query row, visible micro-block);
//! [`qsa_indexer_select_into`] turns those scores into the row's token-level
//! visibility mask (top-k plus the unconditional tail). Frozen semantics live
//! in `csrc/qwen4/qwen4_qsa_indexer_score.cu` and
//! `csrc/qwen4/qwen4_qsa_indexer_select.cu` (transformers 5.16.1 `qwen4_exp`
//! `Qwen4ExpTextQSAIndexer.forward`); the kernels are compiled for one geometry
//! and these wrappers size their buffers from it. The checkpoint's own indexer
//! fields are checked where the model crate reads its config, once that caller
//! lands — the score stage takes only `rms_eps` from the caller.
//!
//! The grouped `(1+weight)` RMSNorm serves the hyper-connection and PLE streams
//! (`csrc/qwen4/qwen4_grouped_rmsnorm.cu`); its contract is on the wrapper.
//!
//! The sparse-attention forward below is the attention half of the three units
//! QSA splits the layer into: the indexer scores and selects, this consumes the
//! selection as a visibility mask and produces the gated attention output that
//! `o_proj` reads. The mask is therefore an input, never derived here. It reads
//! the same shared cos/sin table as the indexer. Frozen semantics live in
//! `csrc/qwen4/qwen4_qsa_attn_forward.cu`.

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
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

/// The geometry the kernel is compiled for, used to size this wrapper's buffer
/// checks. The kernel reads with the same numbers: four query heads and one
/// shared key head at 128 (a 640-wide fused row), four tokens per micro-block,
/// and 64 rotated dims — also the shared attention table's row stride.
const FUSED_WIDTH: usize = 640;
const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 64;
const COMPRESS_RATIO: usize = 4;
/// `kWarps` in the score kernel; one CTA covers that many blocks per pass.
const SCORE_WARPS: usize = 4;

/// Target resident CTAs per SM for the score pass.
const SCORE_CTAS_PER_SM: usize = 4;

fn qsa_score_splits(query_rows: usize, blocks: usize, num_sms: usize) -> usize {
    let max_splits = blocks.div_ceil(SCORE_WARPS).max(1);
    num_sms
        .saturating_mul(SCORE_CTAS_PER_SM)
        .div_ceil(query_rows)
        .clamp(1, max_splits)
}

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
/// `num_sms` is the SM count of `ctx`'s device, queried once by the caller.
/// The score pass splits each row's blocks across `grid.x` CTAs to target
/// `SCORE_CTAS_PER_SM` resident CTAs per SM; taking the count as an argument
/// keeps the driver query off the launch path, where it cost more than the
/// split saves on an already-large grid.
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
    num_sms: usize,
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
    ensure!(num_sms > 0, "QSA indexer num_sms must be positive");
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
    let score_splits = qsa_score_splits(query_rows, blocks, num_sms);
    let query_rows = checked_i32(query_rows, "QSA indexer query rows")?;
    let tokens = checked_i32(tokens, "QSA indexer tokens")?;
    let score_cols = checked_i32(score_cols, "QSA indexer score columns")?;
    let score_splits = checked_i32(score_splits, "QSA indexer score splits")?;

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
            score_splits,
            rms_eps,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer block score launch failed: {err}"))
}

/// Batched grouped `(1+weight)` RMSNorm over `HiddenStates`: `hidden_dim` is
/// `hidden_dim / group_size` adjacent groups, each reduced on its own under one
/// shared `eps`. The `(1 + w)` scale is indexed by the row's *global* column, so
/// group `k` reads `weight[k * group_size .. (k + 1) * group_size]`.
///
/// Qwen3.8-Flash-Next's `Qwen4ExpTextRMSNorm(hidden, eps, group_size)` uses this
/// for the hyper-connection `hc_norm` and the PLE norms, both a 10240-wide
/// stream in four 2560-wide groups. `group_size` must fill one whole thread
/// round of aligned vectors; other sizes are refused. A call whose weight
/// slices are all equal reproduces
/// [`crate::ops::rms_norm_batch_offset_into`]'s bits; see
/// `tests/grouped_rmsnorm_equivalence.rs`.
pub fn rms_norm_grouped_offset_batch_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    group_size: usize,
    eps: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    const OP: &str = "rms_norm_grouped_offset_batch_into";
    ensure!(
        x.hidden_dim > 0 && x.seq_len > 0 && group_size > 0,
        "{OP} dimensions must be non-zero"
    );
    ensure!(
        x.hidden_dim.is_multiple_of(group_size),
        "{OP} hidden_dim {} is not a whole number of {group_size}-wide groups",
        x.hidden_dim
    );
    ensure!(
        weight.len == x.hidden_dim,
        "{OP} weight len {} != hidden_dim {}",
        weight.len,
        x.hidden_dim
    );
    ensure!(
        weight.data.len() >= x.hidden_dim,
        "{OP} weight backing too small"
    );
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "{OP} out shape mismatch"
    );
    x.checked_extent("rms_norm_grouped_offset_batch_into x")?;
    out.checked_extent("rms_norm_grouped_offset_batch_into out")?;
    let hidden_dim = checked_i32(
        x.hidden_dim,
        "rms_norm_grouped_offset_batch_into hidden_dim",
    )?;
    let group_size = checked_i32(group_size, "rms_norm_grouped_offset_batch_into group_size")?;
    let seq_len = checked_i32(x.seq_len, "rms_norm_grouped_offset_batch_into seq_len")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::rms_norm_grouped_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            hidden_dim,
            group_size,
            seq_len,
            eps,
            active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// The budget's block ceiling: the checkpoint's 2048-token budget at four
/// tokens per block.
const MAX_KEPT_BLOCKS: usize = 512;

/// Turn one score per (query row, complete block) into the row's token-level
/// visibility mask (semantics: `csrc/qwen4/qwen4_qsa_indexer_select.cu`).
///
/// `scores` is `[rows.len(), score_cols]` f32 with row stride `score_cols`; the
/// buffer may hold spare capacity (`scores.len() >= rows.len() * score_cols`).
/// Only `(rows[j] + 1) / 4` entries of row `j` are read. `mask` is one byte per
/// (row, token), row stride `tokens`. `budget` is the token budget, a positive
/// multiple of four keeping at most 512 blocks, and the score row must hold
/// `tokens / 4` blocks.
///
/// An empty launch (`rows.is_empty()` or `tokens == 0`) is a no-op. A malformed
/// budget and a short buffer are refused before any pointer is formed.
pub fn qsa_indexer_select_into(
    ctx: &DeviceContext,
    scores: &CudaSlice<f32>,
    score_cols: usize,
    rows: &CudaSlice<i32>,
    budget: usize,
    tokens: usize,
    mask: &mut CudaSlice<u8>,
) -> Result<()> {
    if rows.is_empty() || tokens == 0 {
        return Ok(());
    }
    ensure!(
        budget > 0
            && budget.is_multiple_of(COMPRESS_RATIO)
            && budget / COMPRESS_RATIO <= MAX_KEPT_BLOCKS,
        "QSA indexer budget {budget} must be a positive multiple of {COMPRESS_RATIO} \
         keeping at most {MAX_KEPT_BLOCKS} blocks"
    );
    ensure!(
        scores.len() >= rows.len().saturating_mul(score_cols),
        "QSA indexer score buffer {} f32 cannot hold {} rows of {score_cols}",
        scores.len(),
        rows.len()
    );
    ensure!(
        score_cols >= tokens / COMPRESS_RATIO,
        "QSA indexer score row holds {score_cols} blocks, but a mask over \
         {tokens} tokens needs {}",
        tokens / COMPRESS_RATIO
    );
    ensure!(
        mask.len() >= rows.len().saturating_mul(tokens),
        "QSA indexer mask holds {} bytes, needs {} for {} x {tokens}",
        mask.len(),
        rows.len().saturating_mul(tokens),
        rows.len()
    );
    // The kernel takes extents as i32; checked before any pointer is formed.
    let query_rows = checked_i32(rows.len(), "QSA indexer query rows")?;
    let tokens = checked_i32(tokens, "QSA indexer tokens")?;
    let score_cols = checked_i32(score_cols, "QSA indexer score columns")?;
    let budget = checked_i32(budget, "QSA indexer budget")?;
    let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = rows.device_ptr(&ctx.stream);
    let (mask_ptr, _mask_guard) = mask.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_qsa_indexer_select_cuda(
            scores_ptr as *const f32,
            rows_ptr as *const i32,
            mask_ptr as *mut u8,
            query_rows,
            tokens,
            score_cols,
            budget,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer select launch failed: {err}"))
}

/// The attention geometry the QSA kernels are compiled for. The kernel body
/// carries these as constants and refuses a call whose geometry is not the
/// compiled one, so this struct exists to hold the same numbers on the Rust
/// side: callers size their buffers with it, and the parity gate holds it
/// against the frozen config the goldens were dumped from.
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

    /// Refuse any geometry other than the one the CUDA translation unit was
    /// compiled for. Every width this wrapper sizes a buffer with must equal
    /// the kernel's own constant, or a caller that allocates by this struct
    /// would be indexed with the compiled widths.
    fn ensure_compiled(&self) -> Result<()> {
        let compiled = QsaAttnGeometry::qwen38_flash_next();
        ensure!(
            *self == compiled,
            "qsa_attn_forward_into geometry {self:?} is not the compiled \
             Qwen3.8-Flash-Next geometry {compiled:?}"
        );
        Ok(())
    }
}

/// Replay the QSA sparse-attention forward over one layer's captured
/// activations and write both of the reference's observable outputs.
///
/// `queries` holds `query_rows` fused `q_proj` rows ([`QsaAttnGeometry::query_width`]
/// wide); `rows` gives the token each of those rows belongs to and `positions`
/// the position of every one of the `tokens` tokens, so the evaluated rows may be
/// any subset in any order. Every `rows` and `positions` entry must be in
/// `0..tokens`; the kernel traps otherwise. `mask` is the caller's visibility, one
/// byte per (query row, token), non-zero where visible — the indexer's selection
/// AND causality, which this call never re-derives. A fully hidden row is outside
/// the contract: the reference's two masks sum to `-inf` and its softmax is NaN,
/// while this kernel's single minimum produces a uniform row. `cos`/`sin` are the
/// shared attention RoPE tables (`[tokens, geometry.rotary_dim]`, position-major,
/// half-split duplicated); `normed_keys` is the key pass's workspace and must hold
/// `tokens * geometry.kv_width()` bf16.
///
/// An empty launch (`rows` or `positions` empty) is a no-op. Every element of
/// `attn_weights` (`heads * query_rows * tokens`, head-major as the reference
/// returns them) and of `attn_out` (`query_rows * out_width`, the input to
/// `o_proj`) is written.
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
    cos: &DeviceVec,
    sin: &DeviceVec,
    attn_weights: &mut CudaSlice<bf16>,
    attn_out: &mut CudaSlice<bf16>,
    normed_keys: &mut CudaSlice<bf16>,
) -> Result<()> {
    geometry.ensure_compiled()?;
    let query_rows = rows.len();
    let tokens = positions.len();
    if query_rows == 0 || tokens == 0 {
        return Ok(());
    }
    let query_width = geometry.query_width();
    let kv_width = geometry.kv_width();
    let table_len = tokens * geometry.rotary_dim;
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
        cos.len >= table_len && sin.len >= table_len,
        "qsa_attn_forward_into cos/sin tables hold {}/{} bf16, need {table_len} \
         for {tokens} positions",
        cos.len,
        sin.len
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
    let (cos_ptr, _gcos) = cos.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gsin) = sin.data.device_ptr(&ctx.stream);
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
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
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

#[cfg(test)]
mod tests {
    use super::QsaAttnGeometry;
    use super::qsa_score_splits;

    #[test]
    fn attention_geometry_must_be_the_compiled_one() {
        let good = QsaAttnGeometry::qwen38_flash_next();
        assert!(good.ensure_compiled().is_ok());

        let mut mismatched = good;
        mismatched.heads = 12;
        assert!(mismatched.ensure_compiled().is_err());
        mismatched = good;
        mismatched.kv_heads = 1;
        assert!(mismatched.ensure_compiled().is_err());
        mismatched = good;
        mismatched.head_dim = 128;
        assert!(mismatched.ensure_compiled().is_err());
        mismatched = good;
        mismatched.rotary_dim = 32;
        assert!(mismatched.ensure_compiled().is_err());
        mismatched = good;
        mismatched.rope_theta = 1.0e6;
        assert!(mismatched.ensure_compiled().is_err());
        mismatched = good;
        mismatched.rms_eps = 0.0;
        assert!(mismatched.ensure_compiled().is_err());
    }

    /// Four CTAs per SM, capped by the row's block count. These include the
    /// mid-size bands a row-count cutoff would otherwise run unsplit.
    #[test]
    fn score_splits_follow_the_sm_budget() {
        assert_eq!(qsa_score_splits(300, 128, 92), 2);
        assert_eq!(qsa_score_splits(512, 128, 92), 1);
        assert_eq!(qsa_score_splits(300, 128, 132), 2);
        assert_eq!(qsa_score_splits(512, 128, 132), 2);
        assert_eq!(qsa_score_splits(1, 128, 92), 32);
        assert_eq!(qsa_score_splits(6, 513, 92), 62);
    }
}
