//! Qwen3.8-Flash-Next (Qwen4 line) kernels.
//!
//! QSA token-indexer block scoring turns the fused `index_qk_proj` output into
//! one score per (query row, visible micro-block); selection — top-k plus the
//! unconditional tail — is a later stage, deliberately not here. Frozen
//! semantics live in `csrc/qwen4/qwen4_qsa_indexer_score.cu` (transformers
//! 5.16.1 `qwen4_exp` `Qwen4ExpTextQSAIndexer.forward`); the kernel is compiled
//! for one geometry and this wrapper sizes its buffers from it. The checkpoint's
//! own indexer fields are checked where the model crate reads its config, once
//! that caller lands — this stage takes only `rms_eps` from the caller.
//!
//! The grouped `(1+weight)` RMSNorm serves the hyper-connection and PLE streams
//! (`csrc/qwen4/qwen4_grouped_rmsnorm.cu`); its contract is on the wrapper.

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

/// Batched grouped `(1+weight)` RMSNorm over `HiddenStates`: `hidden_dim` is
/// `hidden_dim / group_size` adjacent groups, each reduced on its own under one
/// shared `eps`. The `(1 + w)` scale is indexed by the row's *global* column, so
/// group `k` reads `weight[k * group_size .. (k + 1) * group_size]`.
///
/// Qwen3.8-Flash-Next's `Qwen4ExpTextRMSNorm(hidden, eps, group_size)` uses this
/// for the hyper-connection `hc_norm` and the PLE norms, both a 10240-wide
/// stream in four 2560-wide groups. A call whose weight slices are all equal
/// reproduces [`crate::ops::rms_norm_batch_offset_into`]'s bits; see
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
