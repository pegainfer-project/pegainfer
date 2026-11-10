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
//! The hyper-connection tail below is `Qwen4ExpTextGatedResidual`'s elementwise half:
//! a scaled SiLU, an affine sigmoid and the branch combine. The norm and the
//! projections it sits between already have entry points above and in the shared
//! surface; only these three were missing. Frozen semantics live in
//! `csrc/qwen4/qwen4_hc_block.cu`.
//!
//! The MoE block below owns only what has no shared entry point: the routing
//! kernel and the shared expert's per-token gate combine; the router GEMM, the
//! shared branch's projections and its `silu(gate)·up` go through the same ops
//! every other line uses. Frozen semantics: `Qwen4ExpTextSparseMoeBlock` in
//! `csrc/qwen4/qwen4_moe_route.cu`.
//!
//! The PLE block below is the per-layer-embedding block one decoder layer
//! injects, from its two projections onward. Its three grouped `(1 + w)`
//! RMSNorms are calls to the grouped entry point above, on the stream in
//! place. Frozen semantics: `csrc/qwen4/qwen4_ple_block.cu`.

use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::ops::ffi_status_detail;
use crate::ops::gemm_into_checked;
use crate::ops::silu_mul_batch_into;
use crate::ops::silu_mul_fused_batch_into;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
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

fn ensure_finite(value: f32, what: &str) -> Result<()> {
    ensure!(
        value.is_finite(),
        "hyper-connection {what} {value} is not finite"
    );
    Ok(())
}

fn launch(result: CUresult, what: &str) -> Result<()> {
    ensure!(
        result == CUresult::CUDA_SUCCESS,
        "hyper-connection {what} launch failed: {result:?}"
    );
    Ok(())
}

/// `out = bf16(silu(f32(x) * scale))`, elementwise.
///
/// The mix path's `silu(input_mix_weight_down(n) / hc_count)`: `hc_count` is a
/// power of two, so dividing the bf16 projection output by it is exact and the
/// caller folds it into `scale`, leaving the SiLU's own cast as the only
/// rounding — which is where the reference's lands.
pub fn hc_silu_scaled_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(scale, "silu scale")?;
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "hyper-connection silu output [{}, {}] does not match input [{}, {}]",
        out.hidden_dim,
        out.seq_len,
        x.hidden_dim,
        x.seq_len
    );
    let n = x.checked_extent("hyper-connection silu input")?;
    out.checked_extent("hyper-connection silu output")?;
    let n = checked_i32(n, "hyper-connection silu element count")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen4_hc_silu_scaled_cuda(
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n,
            scale,
            active_cu_stream(ctx),
        )
    };
    launch(result, "silu_scaled")
}

/// `out = bf16(gain * f32(bf16(sigmoid(f32(x) * scale))))`, elementwise.
///
/// Two reference steps share it: `scale = 1, gain = 1` is the mix path's
/// `sigmoid(input_mix_weight_up(g))`, and `scale = 1/hc_count, gain = 2` is the
/// injection weights `2 * sigmoid(block_inject_weight(n) / hc_count)`. The
/// inner cast is the reference's own — `torch.sigmoid` on a bf16 tensor
/// materialises bf16 before anything reads it.
pub fn hc_sigmoid_affine_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    scale: f32,
    gain: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(scale, "sigmoid scale")?;
    ensure_finite(gain, "sigmoid gain")?;
    ensure!(
        out.hidden_dim == x.hidden_dim && out.seq_len == x.seq_len,
        "hyper-connection sigmoid output [{}, {}] does not match input [{}, {}]",
        out.hidden_dim,
        out.seq_len,
        x.hidden_dim,
        x.seq_len
    );
    let n = x.checked_extent("hyper-connection sigmoid input")?;
    out.checked_extent("hyper-connection sigmoid output")?;
    let n = checked_i32(n, "hyper-connection sigmoid element count")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen4_hc_sigmoid_affine_cuda(
            x_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n,
            scale,
            gain,
            active_cu_stream(ctx),
        )
    };
    launch(result, "sigmoid_affine")
}

/// Combine `branches` gated branches of a hyper-connection stream:
///
/// ```text
/// out[r, c] = bf16(sum_j f32(bf16(a[r, j*hidden+c] * b[r, j*hidden+c])) / branches)
/// ```
///
/// `a` and `b` are row-major `[rows, branches * hidden]`, `out` is
/// `[rows, hidden]`. The inner cast is the reference's bf16 `gate * normed`
/// product; the f32 accumulate, the division by the branch count and the
/// single cast out are torch's `.mean(dim=-2)`.
pub fn hc_mul_branch_combine_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    branches: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        branches > 0,
        "hyper-connection branch count must be positive"
    );
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "hyper-connection combine reads mismatched streams: {} against {}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.hidden_dim == branches * out.hidden_dim,
        "hyper-connection combine: {branches} branches of {} do not span a {}-wide stream",
        out.hidden_dim,
        a.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len && a.seq_len == out.seq_len,
        "hyper-connection combine row counts disagree: {} / {} / {}",
        a.seq_len,
        b.seq_len,
        out.seq_len
    );
    // The shape checks above make `a`'s logical extent `rows * branches *
    // hidden` and `out`'s `rows * hidden` — exactly the two element counts the
    // launcher bounds, so this is what keeps the kernel's source index inside
    // the backing allocations.
    a.checked_extent("hyper-connection combine gate stream")?;
    b.checked_extent("hyper-connection combine normed stream")?;
    out.checked_extent("hyper-connection combine output")?;
    let branches_i32 = checked_i32(branches, "hyper-connection branch count")?;
    let rows = checked_i32(a.seq_len, "hyper-connection combine row count")?;
    let hidden = checked_i32(out.hidden_dim, "hyper-connection combine branch width")?;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen4_hc_mul_branch_combine_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows,
            hidden,
            branches_i32,
            active_cu_stream(ctx),
        )
    };
    launch(result, "mul_branch_combine")
}

/// Scatter a sublayer output back into the hyper-connection stream:
///
/// ```text
/// out[s, b*H + h] = bf16(hyper[s, b*H + h] + bf16(f32(x[s, h]) * f32(inject[s, b])))
/// ```
///
/// `x` is `[rows, HIDDEN]` (the sublayer's 2560-wide output), `inject`
/// `[rows, branches]` (the gated injection weights), and `hyper`/`out`
/// `[rows, branches * HIDDEN]`. The two bf16 roundings are the reference's two
/// elementwise ops (`out * inject`, then `hyper_input + ...`). `out` may alias
/// `hyper`: every element reads its own base before writing it.
pub fn hc_scatter_add_into(
    ctx: &DeviceContext,
    x: &HiddenStates,
    inject: &HiddenStates,
    hyper: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        x.seq_len == inject.seq_len && x.seq_len == hyper.seq_len && x.seq_len == out.seq_len,
        "hyper-connection scatter rows differ: x {}, inject {}, hyper {}, out {}",
        x.seq_len,
        inject.seq_len,
        hyper.seq_len,
        out.seq_len
    );
    let branches = inject.hidden_dim;
    ensure!(
        branches > 0 && hyper.hidden_dim == x.hidden_dim * branches,
        "hyper-connection scatter: sublayer {} x {branches} does not span the {}-wide stream",
        x.hidden_dim,
        hyper.hidden_dim
    );
    ensure!(
        out.hidden_dim == hyper.hidden_dim,
        "hyper-connection scatter: output is {}-wide, stream is {}-wide",
        out.hidden_dim,
        hyper.hidden_dim
    );
    x.checked_extent("hyper-connection scatter sublayer output")?;
    inject.checked_extent("hyper-connection scatter injection weights")?;
    hyper.checked_extent("hyper-connection scatter stream")?;
    out.checked_extent("hyper-connection scatter output")?;
    let rows = checked_i32(x.seq_len, "hyper-connection scatter row count")?;
    let hidden = checked_i32(x.hidden_dim, "hyper-connection scatter branch width")?;
    let branches = checked_i32(branches, "hyper-connection scatter branch count")?;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (inject_ptr, _gi) = inject.data.device_ptr(&ctx.stream);
    let (hyper_ptr, _gh) = hyper.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen4_hc_scatter_cuda(
            x_ptr as *const ffi::Half,
            inject_ptr as *const ffi::Half,
            hyper_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows,
            hidden,
            branches,
            active_cu_stream(ctx),
        )
    };
    launch(result, "scatter")
}

// ── the mix path, composed ────────────────────────────────────────────────

/// Scratch for one hyper-connection mix pass: one buffer per reference step,
/// so a decode step allocates nothing.
pub struct HcMixScratch {
    /// The grouped norm's output, `[rows, branches * hidden]`.
    pub normed: HiddenStates,
    /// The rank-`lowrank` bottleneck before and after the SiLU.
    pub down: HiddenStates,
    pub bottleneck: HiddenStates,
    /// The up projection and its sigmoid, `[rows, branches * hidden]`.
    pub up: HiddenStates,
    pub gate: HiddenStates,
    /// The injection head's logits, `[rows, branches]`, when the module has one.
    pub inject_logits: Option<HiddenStates>,
}

impl HcMixScratch {
    /// Allocate the scratch for `rows` tokens of `branches` `hidden`-wide
    /// branches and a `lowrank` bottleneck.
    pub fn new(
        ctx: &DeviceContext,
        rows: usize,
        hidden: usize,
        branches: usize,
        lowrank: usize,
        with_inject: bool,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && hidden > 0 && branches > 0 && lowrank > 0,
            "hyper-connection mix scratch dimensions must be non-zero"
        );
        let stream = hidden
            .checked_mul(branches)
            .ok_or_else(|| anyhow::anyhow!("hyper-connection stream width overflows usize"))?;
        Ok(Self {
            normed: HiddenStates::zeros(ctx, stream, rows)?,
            down: HiddenStates::zeros(ctx, lowrank, rows)?,
            bottleneck: HiddenStates::zeros(ctx, lowrank, rows)?,
            up: HiddenStates::zeros(ctx, stream, rows)?,
            gate: HiddenStates::zeros(ctx, stream, rows)?,
            inject_logits: with_inject
                .then(|| HiddenStates::zeros(ctx, branches, rows))
                .transpose()?,
        })
    }
}

/// One `Qwen4ExpTextGatedResidual` mix pass, composed from the shared surface:
///
/// ```text
/// n     = grouped (1 + w) RMSNorm(hyper)              // branches groups
/// g     = silu(down(n) / branches)
/// gate  = sigmoid(up(g))
/// mixed = (gate * n).mean(branches)                   // [rows, hidden]
/// inj   = 2 * sigmoid(inject_weight(n) / branches)    // [rows, branches]
/// ```
///
/// `hyper` is the `[rows, branches * hidden]` stream; `mixed` is `[rows,
/// hidden]`; `inject_out` is `[rows, branches]` and must be present exactly
/// when `inject_weight` is. The reference rounds to bf16 at each of these
/// steps, and each composed op lands its rounding where the reference does.
/// The per-layer modules pass their injection head; the top-level readout
/// (`use_combine=False`) passes neither. `branches` is `hc_count`, which must
/// be a power of two for the folded `1 / branches` scales to be the
/// reference's exact division.
#[allow(clippy::too_many_arguments)]
pub fn hc_mix_into(
    ctx: &DeviceContext,
    norm: &DeviceVec,
    down: &DeviceMatrix,
    up: &DeviceMatrix,
    inject_weight: Option<&DeviceMatrix>,
    hyper: &HiddenStates,
    branches: usize,
    rms_eps: f32,
    scratch: &mut HcMixScratch,
    mixed: &mut HiddenStates,
    inject_out: Option<&mut HiddenStates>,
) -> Result<()> {
    ensure!(
        branches > 0 && hyper.hidden_dim.is_multiple_of(branches),
        "hyper-connection mix: {} branches do not divide a {}-wide stream",
        branches,
        hyper.hidden_dim
    );
    // The reference divides by `hc_count` in bf16 on both the mix path and the
    // injection head. Both folds here multiply by `1 / branches` instead; that
    // is the same value only where the division is exact, i.e. for a
    // power-of-two count. Anything else would round somewhere the reference
    // does not, so it is refused rather than served approximately.
    ensure!(
        branches.is_power_of_two(),
        "hyper-connection mix: {branches} branches is not a power of two, so the folded \
         1/branches scale is not the reference's exact bf16 division"
    );
    ensure!(
        rms_eps.is_finite() && rms_eps > 0.0,
        "hyper-connection mix: eps {rms_eps} is not positive"
    );
    let hidden = hyper.hidden_dim / branches;
    let rows = hyper.seq_len;
    let stream = hyper.hidden_dim;
    ensure!(
        norm.len == stream,
        "hyper-connection mix: norm weight is {} wide, stream is {stream}",
        norm.len
    );
    for (what, buf, want) in [
        ("normed", &scratch.normed, stream),
        ("up", &scratch.up, stream),
        ("gate", &scratch.gate, stream),
        ("mixed", &*mixed, hidden),
    ] {
        ensure!(
            buf.hidden_dim == want && buf.seq_len == rows,
            "hyper-connection mix: {what} is [{}, {}], need [{rows}, {want}]",
            buf.seq_len,
            buf.hidden_dim
        );
    }
    ensure!(
        scratch.down.hidden_dim == scratch.bottleneck.hidden_dim
            && scratch.down.seq_len == rows
            && scratch.bottleneck.seq_len == rows,
        "hyper-connection mix: bottleneck buffers disagree with {rows} rows"
    );
    let lowrank = scratch.down.hidden_dim;
    let with_inject = inject_weight.is_some();
    ensure!(
        with_inject == inject_out.is_some() && with_inject == scratch.inject_logits.is_some(),
        "hyper-connection mix: the injection head, its output and its logits must agree"
    );
    if let Some(out) = inject_out.as_ref() {
        ensure!(
            out.hidden_dim == branches && out.seq_len == rows,
            "hyper-connection mix: injection output is [{}, {}], need [{rows}, {branches}]",
            out.seq_len,
            out.hidden_dim
        );
    }
    if let Some(logits) = scratch.inject_logits.as_ref() {
        ensure!(
            logits.hidden_dim == branches && logits.seq_len == rows,
            "hyper-connection mix: injection logits are [{}, {}], need [{rows}, {branches}]",
            logits.seq_len,
            logits.hidden_dim
        );
    }

    let divisor = 1.0 / branches as f32;
    rms_norm_grouped_offset_batch_into(ctx, hyper, norm, hidden, rms_eps, &mut scratch.normed)?;
    gemm_into_checked(ctx, down, &scratch.normed, &mut scratch.down)?;
    ensure!(
        scratch.down.hidden_dim == lowrank,
        "hyper-connection mix: down projection produced {} columns, scratch holds {lowrank}",
        scratch.down.hidden_dim
    );
    hc_silu_scaled_into(ctx, &scratch.down, divisor, &mut scratch.bottleneck)?;
    gemm_into_checked(ctx, up, &scratch.bottleneck, &mut scratch.up)?;
    hc_sigmoid_affine_into(ctx, &scratch.up, 1.0, 1.0, &mut scratch.gate)?;
    hc_mul_branch_combine_into(ctx, &scratch.gate, &scratch.normed, branches, mixed)?;
    if let (Some(weight), Some(logits), Some(out)) =
        (inject_weight, scratch.inject_logits.as_mut(), inject_out)
    {
        gemm_into_checked(ctx, weight, &scratch.normed, logits)?;
        hc_sigmoid_affine_into(ctx, logits, divisor, 2.0, out)?;
    }
    Ok(())
}

/// The checkpoint's MoE block geometry. Frozen with the line: the routing
/// kernel is compiled for the expert count and top-k, and the shared branch's
/// shapes are asserted against these widths before any launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4MoeGeometry {
    /// `text_config.hidden_size`.
    pub hidden: usize,
    /// `text_config.num_experts`. The routing kernel is compiled for this.
    pub experts: usize,
    /// `text_config.num_experts_per_tok`. The routing kernel is compiled for
    /// this.
    pub top_k: usize,
    /// `text_config.shared_expert_intermediate_size`.
    pub shared_intermediate: usize,
}

impl Qwen4MoeGeometry {
    /// The frozen Qwen3.8-Flash-Next text config's MoE block.
    pub fn qwen38_flash_next() -> Self {
        Self {
            hidden: 2560,
            experts: 512,
            top_k: 10,
            shared_intermediate: 640,
        }
    }

    /// Refuse a geometry the kernels were not compiled for. Called by every
    /// wrapper here before it forms a pointer, and public so a caller deriving
    /// geometry from a checkpoint's config can check it without a launch.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.experts == 512 && self.top_k == 10,
            "MoE geometry {self:?} is not the geometry \
             csrc/qwen4/qwen4_moe_route.cu was compiled for (512 experts, top-10)"
        );
        ensure!(
            self.hidden > 0 && self.shared_intermediate > 0,
            "MoE geometry {self:?} carries a zero width"
        );
        Ok(())
    }
}

/// Route one batch of bf16 router logits.
///
/// `logits` is the router GEMM's stored bf16 output, `[seq_len, experts]` —
/// the reference's softmax reads exactly those bits, so the GEMM's rounding is
/// part of the routing contract. `weights` receives `[seq_len, top_k]` bf16
/// weights and `ids` the matching i32 expert ids, in the canonical order:
/// descending probability, ascending expert index inside a tie. `renorm` is
/// the checkpoint's `norm_topk_prob`, renormalizing the selected weights to
/// sum to 1; it is absent from `config.json` and defaults to `true` upstream.
pub fn qwen4_moe_route_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    renorm: bool,
    logits: &HiddenStates,
    weights: &mut CudaSlice<bf16>,
    ids: &mut CudaSlice<i32>,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        logits.hidden_dim == geometry.experts,
        "MoE route: logits carry {} columns, the geometry has {} experts",
        logits.hidden_dim,
        geometry.experts
    );
    // `HiddenStates`'s dims are labels; the kernel reads the allocation with
    // them as strides, so the backing is what gets checked.
    logits.checked_extent("MoE route logits")?;
    let rows = logits.seq_len;
    let picks = rows.checked_mul(geometry.top_k).ok_or_else(|| {
        anyhow!(
            "MoE route: {rows} tokens x top-{} overflows usize",
            geometry.top_k
        )
    })?;
    ensure!(
        weights.len() >= picks,
        "MoE route: weights holds {} bf16, needs {picks} for {rows} tokens x top-{}",
        weights.len(),
        geometry.top_k
    );
    ensure!(
        ids.len() >= picks,
        "MoE route: ids holds {} i32, needs {picks}",
        ids.len()
    );
    if rows == 0 {
        return Ok(());
    }
    let rows = checked_i32(rows, "MoE route rows")?;
    let experts = checked_i32(geometry.experts, "MoE route experts")?;
    let top_k = checked_i32(geometry.top_k, "MoE route top_k")?;

    let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
    let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
    let (ids_ptr, _ids_guard) = ids.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_moe_route_cuda(
            logits_ptr as *const ffi::Half,
            weights_ptr as *mut ffi::Half,
            ids_ptr as *mut i32,
            rows,
            experts,
            top_k,
            i32::from(renorm),
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("Qwen3.8-FN MoE route launch failed: {err}"))
}

/// Caller-owned scratch for [`qwen4_moe_shared_branch_into`]: the two
/// projections and the activated intermediate, one `[tokens,
/// shared_intermediate]` buffer each.
pub struct Qwen4SharedBranchScratch {
    gate: HiddenStates,
    up: HiddenStates,
    intermediate: HiddenStates,
}

impl Qwen4SharedBranchScratch {
    pub fn new(ctx: &DeviceContext, geometry: &Qwen4MoeGeometry, tokens: usize) -> Result<Self> {
        geometry.validate()?;
        let alloc = |name: &str| {
            HiddenStates::zeros(ctx, geometry.shared_intermediate, tokens)
                .map_err(|err| anyhow!("{name} scratch: {err}"))
        };
        Ok(Self {
            gate: alloc("shared gate")?,
            up: alloc("shared up")?,
            intermediate: alloc("shared intermediate")?,
        })
    }

    /// The token count this scratch was allocated for.
    pub fn tokens(&self) -> usize {
        self.gate.seq_len
    }
}

/// The shared expert: `down(silu(gate(x)) · up(x))`, composed of the
/// production GEMM and SiLU-mul ops with the reference's bf16 rounding at each
/// boundary (both projections, the activation, the product, the down
/// projection). The learned scalar gate is *not* applied here — the reference
/// captures `shared_out` before it, and the combine below folds it in.
pub fn qwen4_moe_shared_branch_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    hidden: &HiddenStates,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    scratch: &mut Qwen4SharedBranchScratch,
    shared_out: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        hidden.hidden_dim == geometry.hidden,
        "MoE shared branch: hidden carries {} columns, the geometry has {}",
        hidden.hidden_dim,
        geometry.hidden
    );
    for (name, proj, rows, cols) in [
        (
            "gate_proj",
            gate_proj,
            geometry.shared_intermediate,
            geometry.hidden,
        ),
        (
            "up_proj",
            up_proj,
            geometry.shared_intermediate,
            geometry.hidden,
        ),
        (
            "down_proj",
            down_proj,
            geometry.hidden,
            geometry.shared_intermediate,
        ),
    ] {
        ensure!(
            proj.rows == rows && proj.cols == cols,
            "MoE shared branch: {name} is {}x{}, the geometry needs {rows}x{cols}",
            proj.rows,
            proj.cols
        );
    }
    let tokens = hidden.seq_len;
    ensure!(
        scratch.tokens() == tokens,
        "MoE shared branch: scratch holds {} tokens, hidden carries {tokens}",
        scratch.tokens()
    );
    ensure!(
        shared_out.hidden_dim == geometry.hidden && shared_out.seq_len == tokens,
        "MoE shared branch: shared_out is {}x{}, needs {tokens}x{}",
        shared_out.seq_len,
        shared_out.hidden_dim,
        geometry.hidden
    );
    // The GEMM and SiLU ops below assert the labels only, so the two buffers
    // whose shape this wrapper owns are held against their allocations here —
    // a caller-shaped logical size past the backing would otherwise reach the
    // kernels as an out-of-range stride.
    hidden.checked_extent("MoE shared branch hidden")?;
    shared_out.checked_extent("MoE shared branch shared_out")?;
    if tokens == 0 {
        return Ok(());
    }

    gemm_into_checked(ctx, gate_proj, hidden, &mut scratch.gate)?;
    gemm_into_checked(ctx, up_proj, hidden, &mut scratch.up)?;
    silu_mul_batch_into(ctx, &scratch.gate, &scratch.up, &mut scratch.intermediate)?;
    gemm_into_checked(ctx, down_proj, &scratch.intermediate, shared_out)
}

/// The block sum: `out = expert_out + bf16(sigmoid(gate_logit)) · shared_out`,
/// every operator bf16-rounded where the reference rounds.
///
/// `gate_logit` is the `[1, hidden]` gate linear's stored bf16 output, one
/// scalar per token. The routed-expert half (`expert_out`) is an input, the
/// expert path's combined output.
pub fn qwen4_moe_combine_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    gate_logit: &CudaSlice<bf16>,
    expert_out: &HiddenStates,
    shared_out: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    let tokens = expert_out.seq_len;
    for (name, mat) in [("expert_out", expert_out), ("shared_out", shared_out)] {
        ensure!(
            mat.hidden_dim == geometry.hidden,
            "MoE combine: {name} carries {} columns, the geometry has {}",
            mat.hidden_dim,
            geometry.hidden
        );
    }
    ensure!(
        shared_out.seq_len == tokens && out.seq_len == tokens && out.hidden_dim == geometry.hidden,
        "MoE combine: shape mismatch — expert_out {tokens}x{}, shared_out {}x{}, out {}x{}",
        expert_out.hidden_dim,
        shared_out.seq_len,
        shared_out.hidden_dim,
        out.seq_len,
        out.hidden_dim
    );
    // The kernel strides all three matrices by `hidden`, so what is checked is
    // the allocation behind the labels, not the labels.
    expert_out.checked_extent("MoE combine expert_out")?;
    shared_out.checked_extent("MoE combine shared_out")?;
    out.checked_extent("MoE combine out")?;
    ensure!(
        gate_logit.len() >= tokens,
        "MoE combine: gate_logit holds {} bf16, needs one per token ({tokens})",
        gate_logit.len()
    );
    if tokens == 0 {
        return Ok(());
    }
    let tokens = checked_i32(tokens, "MoE combine tokens")?;
    let hidden = checked_i32(geometry.hidden, "MoE combine hidden")?;

    let (gate_ptr, _gate_guard) = gate_logit.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
    let (shared_ptr, _shared_guard) = shared_out.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_moe_shared_combine_cuda(
            gate_ptr as *const ffi::Half,
            expert_ptr as *const ffi::Half,
            shared_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            tokens,
            hidden,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("Qwen3.8-FN MoE shared combine launch failed: {err}"))
}

// ── the routed experts ────────────────────────────────────────────────────

/// One expert's row range inside a fused stack, as [`Qwen4ExpertLayout`] reports
/// it: `(row_offset, rows, cols)` in the flattened stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4ExpertView {
    /// First row of the expert's range in the flattened stack.
    pub row_offset: usize,
    /// Rows the range spans: `2 * intermediate` in the fused gate|up stack,
    /// `hidden` in the down stack.
    pub rows: usize,
    /// Columns: `hidden` in the fused gate|up stack, `intermediate` in down.
    pub cols: usize,
}

/// How one layer's two fused expert stacks flatten into rows.
///
/// The checkpoint stores `mlp.experts.gate_up_proj` `[experts, 2*intermediate,
/// hidden]` and `mlp.experts.down_proj` `[experts, hidden, intermediate]`, both
/// contiguous, with the expert as the **outermost** axis and **no `.weight`
/// suffix** — two tensors per layer, not 1536. Flattened row-major that is
/// `experts * 2*intermediate` rows of `hidden` and `experts * hidden` rows of
/// `intermediate`, so an expert's weights are a contiguous row range and every
/// per-expert entry point on the shared GEMM surface (`gemm_rows_into_checked`)
/// takes them without a copy. `pegainfer-qwen4`'s manifest derives the same two
/// headers (`ExpectedShape::ExpertStack`); this type is the device-side half of
/// that contract, and it refuses anything the flattened view cannot express.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4ExpertLayout {
    experts: usize,
    hidden: usize,
    intermediate: usize,
    gate_up_len: usize,
    down_len: usize,
}

impl Qwen4ExpertLayout {
    /// Validate a MoE geometry and the routed-expert width, and derive both
    /// stacks' flattened shapes. Device-free: a caller holding only a config can
    /// check the layout without a device, and the lengths this returns are the
    /// two allocations a loader has to find.
    pub fn new(geometry: &Qwen4MoeGeometry, intermediate: usize) -> Result<Self> {
        geometry.validate()?;
        ensure!(
            intermediate > 0,
            "routed expert intermediate width must be non-zero"
        );
        // cuBLAS takes each expert's own extents, and the kernels take the
        // expert and token strides, as i32. Both are checked here so no launch
        // can see a truncated extent.
        let gate_up_rows = intermediate.checked_mul(2).ok_or_else(|| {
            anyhow!("routed expert fused gate|up rows overflow at intermediate {intermediate}")
        })?;
        for (what, value) in [
            ("routed expert intermediate", intermediate),
            ("routed expert hidden", geometry.hidden),
            ("routed expert fused gate|up rows", gate_up_rows),
        ] {
            ensure!(
                i32::try_from(value).is_ok(),
                "{what} {value} does not fit i32"
            );
        }
        let gate_up_len = geometry
            .experts
            .checked_mul(gate_up_rows)
            .and_then(|rows| rows.checked_mul(geometry.hidden))
            .ok_or_else(|| anyhow!("routed expert fused gate|up stack overflows usize"))?;
        let down_len = geometry
            .experts
            .checked_mul(geometry.hidden)
            .and_then(|rows| rows.checked_mul(intermediate))
            .ok_or_else(|| anyhow!("routed expert down stack overflows usize"))?;
        // The kernels compute offsets in 64-bit arithmetic; this is the bound
        // the host-side allocation has to satisfy anyway.
        for (what, value) in [
            ("fused gate|up stack", gate_up_len),
            ("down stack", down_len),
        ] {
            ensure!(
                isize::try_from(value).is_ok(),
                "routed expert {what} of {value} bf16 elements is not addressable"
            );
        }
        Ok(Self {
            experts: geometry.experts,
            hidden: geometry.hidden,
            intermediate,
            gate_up_len,
            down_len,
        })
    }

    /// `num_experts`: the stack's outermost axis.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// `hidden_size`: the fused stack's column count, the down stack's row count
    /// per expert.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `moe_intermediate_size`: one expert's activation width.
    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    /// Rows one expert occupies in the fused gate|up stack: `2 * intermediate`,
    /// gate first.
    pub fn gate_up_rows(&self) -> usize {
        2 * self.intermediate
    }

    /// Elements in the flattened fused gate|up stack.
    pub fn gate_up_len(&self) -> usize {
        self.gate_up_len
    }

    /// Elements in the flattened down stack.
    pub fn down_len(&self) -> usize {
        self.down_len
    }

    /// The fused gate|up rows of one expert: `[2 * intermediate, hidden]`,
    /// the first `intermediate` rows gate and the rest up.
    pub fn gate_up_of(&self, expert: usize) -> Result<Qwen4ExpertView> {
        ensure!(
            expert < self.experts,
            "routed expert {expert} is outside the stack's {} experts",
            self.experts
        );
        Ok(Qwen4ExpertView {
            row_offset: expert * self.gate_up_rows(),
            rows: self.gate_up_rows(),
            cols: self.hidden,
        })
    }

    /// The down rows of one expert: `[hidden, intermediate]`.
    pub fn down_of(&self, expert: usize) -> Result<Qwen4ExpertView> {
        ensure!(
            expert < self.experts,
            "routed expert {expert} is outside the stack's {} experts",
            self.experts
        );
        Ok(Qwen4ExpertView {
            row_offset: expert * self.hidden,
            rows: self.hidden,
            cols: self.intermediate,
        })
    }
}

/// One layer's two fused routed-expert stacks, resident on the device.
///
/// The stacks are stored as the checkpoint's bytes: the expert axis is outer and
/// each expert's rows are contiguous, so nothing is copied or transposed to
/// serve a per-expert GEMM. [`Self::from_host`] is the test path; the shard
/// reader is slice B's loader, which can hand its own buffers to
/// [`Self::from_buffers`] instead of materialising a host copy.
pub struct Qwen4ExpertStacks {
    layout: Qwen4ExpertLayout,
    gate_up: DeviceMatrix,
    down: DeviceMatrix,
}

impl Qwen4ExpertStacks {
    /// Upload both stacks from host bytes, in the checkpoint's own layout.
    ///
    /// `gate_up` is `[experts, 2*intermediate, hidden]` and `down` is
    /// `[experts, hidden, intermediate]`, both flattened row-major. The lengths
    /// are part of the contract: a stack one expert short, or with the fused
    /// halves swapped in shape, is refused rather than uploaded.
    pub fn from_host(
        ctx: &DeviceContext,
        geometry: &Qwen4MoeGeometry,
        intermediate: usize,
        gate_up: &[bf16],
        down: &[bf16],
    ) -> Result<Self> {
        let layout = Qwen4ExpertLayout::new(geometry, intermediate)?;
        ensure!(
            gate_up.len() == layout.gate_up_len(),
            "fused gate|up stack holds {} bf16, the geometry needs {} \
             ({} experts x {} rows x {})",
            gate_up.len(),
            layout.gate_up_len(),
            layout.experts(),
            layout.gate_up_rows(),
            layout.hidden()
        );
        ensure!(
            down.len() == layout.down_len(),
            "down stack holds {} bf16, the geometry needs {} ({} experts x {} x {})",
            down.len(),
            layout.down_len(),
            layout.experts(),
            layout.hidden(),
            layout.intermediate()
        );
        let gate_up = DeviceMatrix::from_host(
            ctx,
            gate_up,
            layout.experts() * layout.gate_up_rows(),
            layout.hidden(),
        )?;
        let down = DeviceMatrix::from_host(
            ctx,
            down,
            layout.experts() * layout.hidden(),
            layout.intermediate(),
        )?;
        Ok(Self {
            layout,
            gate_up,
            down,
        })
    }

    /// Adopt two device-resident stacks, checking their shapes against the
    /// geometry. This is the loader's entry: no host copy, no re-upload.
    pub fn from_buffers(
        ctx: &DeviceContext,
        geometry: &Qwen4MoeGeometry,
        intermediate: usize,
        gate_up: CudaSlice<bf16>,
        down: CudaSlice<bf16>,
    ) -> Result<Self> {
        let layout = Qwen4ExpertLayout::new(geometry, intermediate)?;
        ensure!(
            gate_up.len() == layout.gate_up_len(),
            "fused gate|up stack allocates {} bf16, the geometry needs {}",
            gate_up.len(),
            layout.gate_up_len()
        );
        ensure!(
            down.len() == layout.down_len(),
            "down stack allocates {} bf16, the geometry needs {}",
            down.len(),
            layout.down_len()
        );
        for (what, buffer) in [("fused gate|up", &gate_up), ("down", &down)] {
            ensure!(
                Arc::ptr_eq(buffer.stream(), &ctx.stream),
                "routed expert {what} stack belongs to another device context"
            );
        }
        Ok(Self {
            layout,
            gate_up: DeviceMatrix {
                data: gate_up,
                rows: layout.experts() * layout.gate_up_rows(),
                cols: layout.hidden(),
            },
            down: DeviceMatrix {
                data: down,
                rows: layout.experts() * layout.hidden(),
                cols: layout.intermediate(),
            },
        })
    }

    /// The layout the two buffers were checked against.
    pub fn layout(&self) -> Qwen4ExpertLayout {
        self.layout
    }

    /// The fused gate|up stack, flattened: `[experts * 2*intermediate, hidden]`.
    /// A per-expert GEMM can take a row range of it through
    /// [`crate::ops::gemm_rows_into_checked`].
    pub fn gate_up_stack(&self) -> &DeviceMatrix {
        &self.gate_up
    }

    /// The down stack, flattened: `[experts * hidden, intermediate]`.
    pub fn down_stack(&self) -> &DeviceMatrix {
        &self.down
    }

    /// One expert's gate|up rows.
    pub fn gate_up_view(&self, expert: usize) -> Result<Qwen4ExpertView> {
        self.layout.gate_up_of(expert)
    }

    /// One expert's down rows.
    pub fn down_view(&self, expert: usize) -> Result<Qwen4ExpertView> {
        self.layout.down_of(expert)
    }
}

/// Caller-owned scratch for [`qwen4_moe_routed_experts_into`]: one row per
/// `(token, slot)` route for each step of the reference's forward, plus the
/// device pointer arrays the batched GEMMs read.
///
/// The capacity is `tokens * top_k` routes, fixed at construction, so a serving
/// loop allocates nothing per step — the property a CUDA Graph capture needs and
/// the reason the backend keeps no per-expert capacity at all.
pub struct Qwen4MoeExpertScratch {
    tokens: usize,
    layout: Qwen4ExpertLayout,
    gate_up_pointers: Qwen4RoutePointers,
    down_pointers: Qwen4RoutePointers,
    /// The fused gate|up projection's output, `[routes, 2 * intermediate]`.
    gate_up: HiddenStates,
    /// `silu(gate) * up`, `[routes, intermediate]`.
    activation: HiddenStates,
    /// The down projection's output, `[routes, hidden]`.
    rows: HiddenStates,
}

/// The three pointer arrays one batched GEMM reads: `(weights, inputs,
/// outputs)`, one entry per route, all device-resident.
struct Qwen4RoutePointers {
    weights: CudaSlice<u64>,
    inputs: CudaSlice<u64>,
    outputs: CudaSlice<u64>,
}

impl Qwen4RoutePointers {
    fn new(ctx: &DeviceContext, routes: usize) -> Result<Self> {
        let alloc = |name: &str| {
            ctx.stream
                .alloc_zeros::<u64>(routes)
                .map_err(|err| anyhow!("routed expert {name} pointer array: {err}"))
        };
        Ok(Self {
            weights: alloc("weight")?,
            inputs: alloc("input")?,
            outputs: alloc("output")?,
        })
    }
}

impl Qwen4MoeExpertScratch {
    /// Allocate the scratch for `tokens` tokens' worth of routes. `tokens` must
    /// be positive: an empty step issues no expert work, so it needs no scratch
    /// and [`qwen4_moe_routed_experts_into`] returns before reading any.
    pub fn new(
        ctx: &DeviceContext,
        geometry: &Qwen4MoeGeometry,
        intermediate: usize,
        tokens: usize,
    ) -> Result<Self> {
        let layout = Qwen4ExpertLayout::new(geometry, intermediate)?;
        ensure!(
            tokens > 0,
            "routed expert scratch needs at least one token; an empty step has no expert work"
        );
        let routes = tokens
            .checked_mul(geometry.top_k)
            .ok_or_else(|| anyhow!("routed expert routes overflow at {tokens} tokens"))?;
        // The kernels take the route count and per-route extents as i32 and cap
        // the batch at what cuBLAS accepts; refuse here rather than at a launch.
        ensure!(
            i32::try_from(routes).is_ok(),
            "routed expert route count {routes} does not fit i32"
        );
        let alloc = |name: &str, hidden_dim: usize, rows: usize| {
            HiddenStates::zeros(ctx, hidden_dim, rows)
                .map_err(|err| anyhow!("routed expert {name}: {err}"))
        };
        Ok(Self {
            tokens,
            layout,
            gate_up_pointers: Qwen4RoutePointers::new(ctx, routes)?,
            down_pointers: Qwen4RoutePointers::new(ctx, routes)?,
            gate_up: alloc("gate|up output", layout.gate_up_rows(), routes)?,
            activation: alloc("activation", layout.intermediate(), routes)?,
            rows: alloc("down output", layout.hidden(), routes)?,
        })
    }

    /// The token count this scratch was allocated for.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The per-route rows this scratch holds: `tokens * top_k`.
    pub fn routes(&self) -> usize {
        self.rows.seq_len
    }
}

/// Compute one layer's routed-expert output in bf16:
/// `routed[token] = Σ_{experts ascending} bf16(y_e · w)`, the reference's fold.
///
/// `hidden` is the block's `[tokens, hidden]` input, `weights`/`ids` the routing
/// kernel's `[tokens * top_k]` bf16 weights and i32 expert ids (route
/// `token * top_k + slot`), `stacks` the layer's two fused expert stacks, and
/// `routed` the `[tokens, hidden]` output. The four rounding points and the
/// accumulation order are the reference's — see `csrc/qwen4/qwen4_moe_experts.cu`
/// for the sequence and `tests/qwen4_moe_block_parity.rs` for the gate.
///
/// Backend: one N=1 cuBLAS GEMM per route against the route's own expert rows,
/// driven by device-resident pointer arrays, with the grouping state the
/// alternative needs absent by construction — an empty expert issues no call, a
/// skewed one issues a call per row, and no host ever learns the counts. The
/// cost is weight traffic (each expert's rows are read once per route), recorded
/// in `docs/models/qwen4/bring-up.md`; a grouped backend is measured against
/// this one rather than assumed better.
///
/// Nothing here allocates or synchronises, so a decode step is capturable.
#[allow(clippy::too_many_arguments)]
pub fn qwen4_moe_routed_experts_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    stacks: &Qwen4ExpertStacks,
    hidden: &HiddenStates,
    weights: &CudaSlice<bf16>,
    ids: &CudaSlice<i32>,
    scratch: &mut Qwen4MoeExpertScratch,
    routed: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    let layout = stacks.layout();
    ensure!(
        layout.experts() == geometry.experts && layout.hidden() == geometry.hidden,
        "routed expert stacks are laid out for {} experts of {} columns, the geometry has {} of {}",
        layout.experts(),
        layout.hidden(),
        geometry.experts,
        geometry.hidden
    );
    ensure!(
        hidden.hidden_dim == geometry.hidden,
        "routed expert input carries {} columns, the geometry has {}",
        hidden.hidden_dim,
        geometry.hidden
    );
    let tokens = hidden.seq_len;
    ensure!(
        routed.hidden_dim == geometry.hidden && routed.seq_len == tokens,
        "routed expert output is {}x{}, needs {tokens}x{}",
        routed.seq_len,
        routed.hidden_dim,
        geometry.hidden
    );
    hidden.checked_extent("routed expert input")?;
    routed.checked_extent("routed expert output")?;
    let routes = tokens
        .checked_mul(geometry.top_k)
        .ok_or_else(|| anyhow!("routed expert routes overflow at {tokens} tokens"))?;
    ensure!(
        weights.len() >= routes,
        "routed expert weights hold {} bf16, need {routes} for {tokens} tokens x top-{}",
        weights.len(),
        geometry.top_k
    );
    ensure!(
        ids.len() >= routes,
        "routed expert ids hold {} i32, need {routes}",
        ids.len()
    );
    if tokens == 0 {
        return Ok(());
    }
    ensure!(
        scratch.tokens == tokens && scratch.layout == layout,
        "routed expert scratch holds {} tokens of a {:?} layout, the call has {tokens} of {layout:?}",
        scratch.tokens,
        scratch.layout
    );
    for (what, stream) in [
        ("input", hidden.data.stream()),
        ("weights", weights.stream()),
        ("ids", ids.stream()),
        ("routed output", routed.data.stream()),
    ] {
        ensure!(
            Arc::ptr_eq(stream, &ctx.stream),
            "routed expert {what} belongs to another device context"
        );
    }

    let routes_i32 = checked_i32(routes, "routed expert routes")?;
    let top_k = checked_i32(geometry.top_k, "routed expert top_k")?;
    let experts = checked_i32(geometry.experts, "routed expert experts")?;
    let hidden_dim = checked_i32(geometry.hidden, "routed expert hidden")?;
    let intermediate = checked_i32(layout.intermediate(), "routed expert intermediate")?;
    let gate_up_rows = checked_i32(layout.gate_up_rows(), "routed expert gate|up rows")?;
    let stream = active_cu_stream(ctx);

    // 1. Resolve every route's expert and token into the two pointer arrays and
    //    range-check the ids on the device.
    {
        let (ids_ptr, _ids_guard) = ids.device_ptr(&ctx.stream);
        let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
        let (gate_up_ptr, _gu_guard) = stacks.gate_up.data.device_ptr(&ctx.stream);
        let (down_ptr, _down_guard) = stacks.down.data.device_ptr(&ctx.stream);
        let (proj_ptr, _proj_guard) = scratch.gate_up.data.device_ptr_mut(&ctx.stream);
        let (act_ptr, _act_guard) = scratch.activation.data.device_ptr_mut(&ctx.stream);
        let (rows_ptr, _rows_guard) = scratch.rows.data.device_ptr_mut(&ctx.stream);
        let (a_gu, _a_gu) = scratch.gate_up_pointers.weights.device_ptr_mut(&ctx.stream);
        let (x_gu, _x_gu) = scratch.gate_up_pointers.inputs.device_ptr_mut(&ctx.stream);
        let (y_gu, _y_gu) = scratch.gate_up_pointers.outputs.device_ptr_mut(&ctx.stream);
        let (a_d, _a_d) = scratch.down_pointers.weights.device_ptr_mut(&ctx.stream);
        let (x_d, _x_d) = scratch.down_pointers.inputs.device_ptr_mut(&ctx.stream);
        let (y_d, _y_d) = scratch.down_pointers.outputs.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::qwen4_moe_route_pointers_cuda(
                ids_ptr as *const i32,
                hidden_ptr as *const ffi::Half,
                gate_up_ptr as *const ffi::Half,
                down_ptr as *const ffi::Half,
                proj_ptr as *mut ffi::Half,
                act_ptr as *mut ffi::Half,
                rows_ptr as *mut ffi::Half,
                a_gu as *mut *const ffi::Half,
                x_gu as *mut *const ffi::Half,
                y_gu as *mut *mut ffi::Half,
                a_d as *mut *const ffi::Half,
                x_d as *mut *const ffi::Half,
                y_d as *mut *mut ffi::Half,
                routes_i32,
                top_k,
                experts,
                hidden_dim,
                intermediate,
                stream,
            )
        }
        .result()
        .map_err(|err| anyhow!("routed expert pointer assembly launch failed: {err}"))?;
    }

    // 2. `gate, up = F.linear(x, gate_up_proj[e])`: one batched GEMM over the
    //    routes, each against its own expert's rows, output bf16.
    pointer_gemm_into(
        ctx,
        &scratch.gate_up_pointers,
        &scratch.gate_up.data,
        gate_up_rows,
        hidden_dim,
        routes_i32,
    )?;

    // 3. `h = silu(gate) * up`: the shared entry point, which rounds the
    //    activation to bf16 before the product exactly as the reference does.
    silu_mul_fused_batch_into(ctx, &scratch.gate_up, &mut scratch.activation)?;

    // 4. `y = F.linear(h, down_proj[e])`, same batched shape.
    pointer_gemm_into(
        ctx,
        &scratch.down_pointers,
        &scratch.rows.data,
        hidden_dim,
        intermediate,
        routes_i32,
    )?;

    // 5. Scale by the routing weight and fold in ascending expert index.
    qwen4_moe_route_scale_fold_into(ctx, geometry, &scratch.rows, weights, ids, routed)
}

/// Fold one route batch into per-token outputs:
/// `out[token] = Σ_{experts ascending} bf16(bf16(rows[route] · w[route]))`.
///
/// This is the reference's `final_hidden_states.index_add_` half — the routing
/// scale rounds to bf16 first, and the accumulator rounds once per contribution
/// because the reference's accumulator is a bf16 tensor, not because this
/// implementation chose to. Contributions are visited in **ascending expert
/// index** (the reference's expert loop is outer), which is not slot order:
/// `tests/qwen4_moe_block_parity.rs` shows a case where the two differ bit for
/// bit. Ids outside `[0, experts)` fail the token closed with a NaN row.
///
/// `rows` is `[tokens * top_k, hidden]` indexed by route (`token * top_k +
/// slot`), `weights`/`ids` that route array, and `out` `[tokens, hidden]`. Any
/// per-route backend produces `rows`, so this stage is public on its own.
pub fn qwen4_moe_route_scale_fold_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    rows: &HiddenStates,
    weights: &CudaSlice<bf16>,
    ids: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        rows.hidden_dim == geometry.hidden,
        "routed fold rows carry {} columns, the geometry has {}",
        rows.hidden_dim,
        geometry.hidden
    );
    ensure!(
        rows.seq_len.is_multiple_of(geometry.top_k),
        "routed fold holds {} rows, which is not a whole number of top-{} picks",
        rows.seq_len,
        geometry.top_k
    );
    let tokens = rows.seq_len / geometry.top_k;
    ensure!(
        out.hidden_dim == geometry.hidden && out.seq_len == tokens,
        "routed fold output is {}x{}, needs {tokens}x{}",
        out.seq_len,
        out.hidden_dim,
        geometry.hidden
    );
    ensure!(
        weights.len() >= rows.seq_len,
        "routed fold weights hold {} bf16, need {} rows",
        weights.len(),
        rows.seq_len
    );
    ensure!(
        ids.len() >= rows.seq_len,
        "routed fold ids hold {} i32, need {} rows",
        ids.len(),
        rows.seq_len
    );
    rows.checked_extent("routed fold rows")?;
    out.checked_extent("routed fold output")?;
    if tokens == 0 {
        return Ok(());
    }
    let tokens = checked_i32(tokens, "routed fold tokens")?;
    let top_k = checked_i32(geometry.top_k, "routed fold top_k")?;
    let experts = checked_i32(geometry.experts, "routed fold experts")?;
    let hidden = checked_i32(geometry.hidden, "routed fold hidden")?;

    let (rows_ptr, _rows_guard) = rows.data.device_ptr(&ctx.stream);
    let (weights_ptr, _weights_guard) = weights.device_ptr(&ctx.stream);
    let (ids_ptr, _ids_guard) = ids.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_moe_route_scale_fold_cuda(
            rows_ptr as *const ffi::Half,
            weights_ptr as *const ffi::Half,
            ids_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            tokens,
            top_k,
            experts,
            hidden,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("routed expert fold launch failed: {err}"))
}

/// One batched pointer GEMM over `routes` N=1 problems, into `rows`' backing.
fn pointer_gemm_into(
    ctx: &DeviceContext,
    pointers: &Qwen4RoutePointers,
    rows: &CudaSlice<bf16>,
    m: i32,
    k: i32,
    routes: i32,
) -> Result<()> {
    let (weights, _weights_guard) = pointers.weights.device_ptr(&ctx.stream);
    let (inputs, _inputs_guard) = pointers.inputs.device_ptr(&ctx.stream);
    let (outputs, _outputs_guard) = pointers.outputs.device_ptr(&ctx.stream);
    ensure!(
        rows.len() >= rows_span(m, routes),
        "routed expert GEMM output holds {} bf16, needs {}",
        rows.len(),
        rows_span(m, routes)
    );
    let status = unsafe {
        ffi::qwen4_moe_pointer_gemm_cuda(
            weights as *const *const core::ffi::c_void,
            inputs as *const *const core::ffi::c_void,
            outputs as *const *mut core::ffi::c_void,
            m,
            k,
            routes,
            active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "routed expert pointer GEMM failed: {}, m={m}, k={k}, routes={routes}",
        ffi_status_detail(status)
    );
    Ok(())
}

/// Elements the routed expert output rows must hold: `routes * m`.
fn rows_span(m: i32, routes: i32) -> usize {
    (m.max(0) as usize).saturating_mul(routes.max(0) as usize)
}

/// The checkpoint's PLE block geometry. Frozen with the line: `pegainfer-qwen4`
/// serves one `qwen4_exp` text geometry, and the kernels are compiled for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PleBlockGeometry {
    /// `hidden_size`, which the block's `ple_embed_dim` equals: one branch's
    /// width, and the group a norm reduces over.
    pub hidden: usize,
    /// `hc_count`: hyper-connection branches in the stream.
    pub hc_count: usize,
    /// `ple_conv_kernel_size`: taps of the depthwise conv.
    pub conv_kernel: usize,
    /// `ngram_size`: the conv's dilation, so its state is `(conv_kernel - 1) *
    /// dilation` rows and its receptive field one more.
    pub dilation: usize,
    /// `rms_norm_eps`, shared by all three `(1 + w)` norms.
    pub rms_eps: f32,
}

impl PleBlockGeometry {
    /// The frozen Qwen3.8-Flash-Next text config's PLE block.
    pub fn qwen38_flash_next() -> Self {
        Self {
            hidden: 2560,
            hc_count: 4,
            conv_kernel: 4,
            dilation: 3,
            rms_eps: 1.0e-6,
        }
    }

    /// Columns of the stream the block reads and writes.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// `sqrt(hidden_size)` in the f32 the reference's opmath divides by.
    ///
    /// Computed the way the reference does — an f64 square root narrowed to f32 —
    /// and passed to the kernel rather than compiled into it, because the gate
    /// *divides* by this: multiplying by its reciprocal rounds differently, and
    /// the difference lands on every gate.
    pub fn sqrt_hidden(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }

    /// Rows the conv reads before the sequence starts: its zero initial state.
    pub fn conv_state(&self) -> usize {
        self.conv_kernel.saturating_sub(1) * self.dilation
    }

    /// Scratch [`ple_block_into`] needs for `seq_len` rows, in bf16 elements.
    ///
    /// Four regions of `seq_len * width()`, one per intermediate: the two
    /// grouped norms over the two input streams, the gated stream and the
    /// conv's normed input. Every region is a whole stream, because each is
    /// written by one launch and read by the next.
    pub fn workspace_len(&self, seq_len: usize) -> usize {
        4 * seq_len * self.width()
    }

    /// Refuse a geometry the kernels were not compiled for. Called by
    /// [`ple_block_into`] before it forms a pointer, and public so a caller
    /// deriving geometry from a checkpoint's config can check it without a
    /// launch.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden == 2560 && self.hc_count == 4 && self.conv_kernel == 4,
            "PLE block geometry {self:?} is not the geometry \
             csrc/qwen4/qwen4_ple_block.cu was compiled for"
        );
        // The dilation is a runtime argument (the state it implies is a bound
        // inside the kernel, not a buffer), but only `ngram_size` is the frozen
        // semantics: a different one convolves the wrong rows and still produces
        // a plausible stream.
        ensure!(
            self.dilation == 3,
            "PLE block dilation {} is not the compiled ngram_size 3",
            self.dilation
        );
        // Bit equality, not a tolerance: the norms' epsilon enters an rsqrt, so
        // a drifted value changes every element by less than the comparison can
        // see and more than the block should tolerate.
        ensure!(
            self.rms_eps.to_bits() == 1.0e-6f32.to_bits(),
            "PLE block rms_norm_eps {} is not the compiled 1e-6",
            self.rms_eps
        );
        Ok(())
    }
}

/// Run one PLE block over `seq_len` rows.
///
/// `hidden`, `key` and `out` are row-major `[seq_len, width()]` — `hidden` is
/// the residual stream the query norm reads, `key` the projection the key norm
/// reads. `value` is `[seq_len, hidden]` and is broadcast over the branches. The
/// three norm weights are one per channel of the stream, `[width()]`, and
/// `conv_weight` is the depthwise `[width(), 1, conv_kernel]` flattened.
///
/// `workspace` is caller-owned scratch of [`PleBlockGeometry::workspace_len`]
/// elements; see that method for the layout. Nothing is allocated here, so the
/// call is capturable.
pub fn ple_block_into(
    ctx: &DeviceContext,
    geometry: &PleBlockGeometry,
    hidden: &CudaSlice<bf16>,
    key: &CudaSlice<bf16>,
    value: &CudaSlice<bf16>,
    norm_key: &CudaSlice<bf16>,
    norm_query: &CudaSlice<bf16>,
    norm_conv: &CudaSlice<bf16>,
    conv_weight: &CudaSlice<bf16>,
    seq_len: usize,
    out: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<bf16>,
) -> Result<()> {
    geometry.validate()?;
    let width = geometry.width();
    let group = geometry.hidden;
    let stream_elements = seq_len
        .checked_mul(width)
        .ok_or_else(|| anyhow!("PLE block extent overflow at {seq_len} rows"))?;
    ensure!(
        hidden.len() >= stream_elements && key.len() >= stream_elements,
        "PLE block inputs hold {} and {} bf16, need {stream_elements} each for {seq_len} rows of \
         {width}",
        hidden.len(),
        key.len()
    );
    ensure!(
        value.len() >= seq_len.saturating_mul(group),
        "PLE block value holds {} bf16, needs {} for {seq_len} rows of {group}",
        value.len(),
        seq_len.saturating_mul(group)
    );
    ensure!(
        norm_key.len() >= width && norm_query.len() >= width && norm_conv.len() >= width,
        "PLE block norm weights must hold {width} bf16 each, got {}, {} and {}",
        norm_key.len(),
        norm_query.len(),
        norm_conv.len()
    );
    ensure!(
        conv_weight.len() >= width.saturating_mul(geometry.conv_kernel),
        "PLE block conv weight holds {} bf16, needs {} for {width} channels of {} taps",
        conv_weight.len(),
        width.saturating_mul(geometry.conv_kernel),
        geometry.conv_kernel
    );
    ensure!(
        out.len() >= stream_elements,
        "PLE block out holds {} bf16, needs {stream_elements}",
        out.len()
    );
    let needed = geometry.workspace_len(seq_len);
    ensure!(
        workspace.len() >= needed,
        "PLE block workspace holds {} bf16, needs {needed} for {seq_len} rows",
        workspace.len()
    );
    if seq_len == 0 {
        return Ok(());
    }
    // The entry points take extents as i32; checked before any pointer is formed.
    let rows = checked_i32(seq_len, "PLE block rows")?;
    let width_i = checked_i32(width, "PLE block stream width")?;
    let group_i = checked_i32(group, "PLE block group width")?;
    let branches_i = checked_i32(geometry.hc_count, "PLE block branch count")?;
    let conv_kernel_i = checked_i32(geometry.conv_kernel, "PLE block conv kernel")?;
    let dilation_i = checked_i32(geometry.dilation, "PLE block conv dilation")?;

    let (hidden_ptr, _hidden_guard) = hidden.device_ptr(&ctx.stream);
    let (key_ptr, _key_guard) = key.device_ptr(&ctx.stream);
    let (value_ptr, _value_guard) = value.device_ptr(&ctx.stream);
    let (norm_key_ptr, _norm_key_guard) = norm_key.device_ptr(&ctx.stream);
    let (norm_query_ptr, _norm_query_guard) = norm_query.device_ptr(&ctx.stream);
    let (norm_conv_ptr, _norm_conv_guard) = norm_conv.device_ptr(&ctx.stream);
    let (conv_weight_ptr, _conv_weight_guard) = conv_weight.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let stream = active_cu_stream(ctx);

    // Workspace regions, one whole stream each, addressed in bytes.
    let element_bytes = size_of::<bf16>() as u64;
    let region = stream_elements as u64 * element_bytes;
    let query_normed = workspace_ptr;
    let key_normed = workspace_ptr + region;
    let gated = workspace_ptr + 2 * region;
    let conv_in = workspace_ptr + 3 * region;

    // 1. The two input norms, one launch each. A row of the stream is one
    //    branch-major group after another, which is the grouped entry point's
    //    own layout, so the norms read the caller's streams in place.
    grouped_norm(
        hidden_ptr,
        norm_query_ptr,
        query_normed,
        width_i,
        group_i,
        rows,
        geometry.rms_eps,
        stream,
    )?;
    grouped_norm(
        key_ptr,
        norm_key_ptr,
        key_normed,
        width_i,
        group_i,
        rows,
        geometry.rms_eps,
        stream,
    )?;

    // 2. Per-branch gate dot, signed sqrt, sigmoid, and the value broadcast.
    let status = unsafe {
        ffi::qwen4_ple_gate_cuda(
            key_normed as *const ffi::Half,
            query_normed as *const ffi::Half,
            value_ptr as *const ffi::Half,
            gated as *mut ffi::Half,
            rows,
            group_i,
            branches_i,
            geometry.sqrt_hidden(),
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block gate launch failed: {err}"))?;

    // 3. The conv's own grouped norm over the gated stream.
    grouped_norm(
        gated,
        norm_conv_ptr,
        conv_in,
        width_i,
        group_i,
        rows,
        geometry.rms_eps,
        stream,
    )?;

    // 4. Dilated causal depthwise conv, silu, and the residual add back onto the
    //    gated stream, landing in the caller's output.
    let status = unsafe {
        ffi::qwen4_ple_conv_residual_cuda(
            conv_in as *const ffi::Half,
            gated as *const ffi::Half,
            conv_weight_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows,
            group_i,
            branches_i,
            conv_kernel_i,
            dilation_i,
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block conv launch failed: {err}"))?;
    Ok(())
}

/// One `(1 + w)` grouped RMSNorm over a whole row-major stream, through the same
/// entry point [`rms_norm_grouped_offset_batch_into`] wraps.
///
/// Addressed by raw pointer rather than through `HiddenStates` because the
/// block's intermediates are workspace regions, not slices the caller owns. The
/// entry point returns a status — unlike the batched one it replaces here — so a
/// shape it was not compiled for surfaces at this launch instead of as the next
/// one's sticky error.
#[allow(clippy::too_many_arguments)]
fn grouped_norm(
    x: u64,
    weight: u64,
    out: u64,
    hidden_dim: i32,
    group_size: i32,
    seq_len: i32,
    eps: f32,
    stream: CUstream,
) -> Result<()> {
    let status = unsafe {
        ffi::rms_norm_grouped_offset_cuda(
            x as *const ffi::Half,
            weight as *const ffi::Half,
            out as *mut ffi::Half,
            hidden_dim,
            group_size,
            seq_len,
            eps,
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block grouped norm launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::PleBlockGeometry;
    use super::Qwen4MoeGeometry;
    use super::qsa_score_splits;

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

    /// The geometry is the hand-off with the kernel's compiled constants, so a
    /// mismatch has to be refused before a pointer is formed. `hidden` and
    /// `shared_intermediate` are not compiled into the route kernel — the
    /// wrappers hold them against the caller's shapes — but a zero width must
    /// still fail here, device-free.
    #[test]
    fn only_the_compiled_geometry_passes() {
        let compiled = Qwen4MoeGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        for (field, mutated) in [
            (
                "experts",
                Qwen4MoeGeometry {
                    experts: 128,
                    ..compiled
                },
            ),
            (
                "top_k",
                Qwen4MoeGeometry {
                    top_k: 8,
                    ..compiled
                },
            ),
            (
                "hidden",
                Qwen4MoeGeometry {
                    hidden: 0,
                    ..compiled
                },
            ),
            (
                "shared_intermediate",
                Qwen4MoeGeometry {
                    shared_intermediate: 0,
                    ..compiled
                },
            ),
        ] {
            assert!(
                mutated.validate().is_err(),
                "a geometry with another {field} must be refused"
            );
        }
    }

    /// The geometry is the one hand-off this block has with the kernels'
    /// compiled constants, so a mismatch has to be refused before a pointer is
    /// formed. Device-free: no launch is involved.
    #[test]
    fn only_the_compiled_ple_geometry_passes() {
        let compiled = PleBlockGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        assert_eq!(compiled.width(), 10_240);
        assert_eq!(compiled.conv_state(), 9);
        // `sqrt(2560)` narrowed from f64, the divisor the gate uses.
        // Bit equality: the gate divides by this, so a last-place difference is
        // a different divisor and not a rounding of the same one.
        assert_eq!(
            compiled.sqrt_hidden().to_bits(),
            ((2560f64).sqrt() as f32).to_bits()
        );
        assert_eq!(compiled.workspace_len(2), 4 * 2 * 10_240);
        for (field, mutated) in [
            (
                "hidden",
                PleBlockGeometry {
                    hidden: 1280,
                    ..compiled
                },
            ),
            (
                "hc_count",
                PleBlockGeometry {
                    hc_count: 2,
                    ..compiled
                },
            ),
            (
                "conv_kernel",
                PleBlockGeometry {
                    conv_kernel: 2,
                    ..compiled
                },
            ),
            (
                "dilation",
                PleBlockGeometry {
                    dilation: 1,
                    ..compiled
                },
            ),
            (
                "rms_eps",
                PleBlockGeometry {
                    rms_eps: 1.0e-5,
                    ..compiled
                },
            ),
            (
                // One f32 ulp: the geometry's own bit-equality check, and what a
                // config drifting in the last place hits.
                "rms_eps one ulp",
                PleBlockGeometry {
                    rms_eps: f32::from_bits(1.0e-6f32.to_bits() + 1),
                    ..compiled
                },
            ),
        ] {
            assert!(
                mutated.validate().is_err(),
                "a geometry with another {field} must be refused"
            );
        }
    }
}
