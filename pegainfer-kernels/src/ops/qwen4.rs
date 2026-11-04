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

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUresult;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::ops::gemm_into_checked;
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
/// out[r, c] = bf16(combine_scale * sum_j f32(bf16(a[r, j*hidden+c] * b[r, j*hidden+c])))
/// ```
///
/// `a` and `b` are row-major `[rows, branches * hidden]`, `out` is
/// `[rows, hidden]`. The inner cast is the reference's bf16 `gate * normed`
/// product; the f32 accumulate and the single cast out are torch's
/// `.mean(dim=-2)`, so `combine_scale = 1.0 / branches as f32` reproduces it.
/// A different `combine_scale` is a caller asking for a different reduce — the
/// parity gate's negative control passes `1.0` for the sum.
pub fn hc_mul_branch_combine_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    branches: usize,
    combine_scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure_finite(combine_scale, "combine scale")?;
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
            combine_scale,
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
    hc_mul_branch_combine_into(
        ctx,
        &scratch.gate,
        &scratch.normed,
        branches,
        divisor,
        mixed,
    )?;
    if let (Some(weight), Some(logits), Some(out)) =
        (inject_weight, scratch.inject_logits.as_mut(), inject_out)
    {
        gemm_into_checked(ctx, weight, &scratch.normed, logits)?;
        hc_sigmoid_affine_into(ctx, logits, divisor, 2.0, out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
