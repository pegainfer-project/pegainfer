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

/// The largest number of blocks a budget may keep: the checkpoint's 2048-token
/// budget at the compiled four tokens per block. The kernel refuses the same
/// bound on the device; this copy shapes the host-side message and lets the
/// wrapper refuse before a launch (the score wrapper mirrors its compiled
/// geometry the same way).
const MAX_KEPT_BLOCKS: usize = 512;

/// The selection the reference computes; every production caller passes this.
pub const QSA_SELECT_PRODUCTION: i32 = 0;
/// Gate-only: leave the row's incomplete tail unset.
pub const QSA_SELECT_DROP_TAIL: i32 = 1;
/// Gate-only: keep one block fewer than the budget asks for.
pub const QSA_SELECT_KEEP_FEWER: i32 = 2;
/// Gate-only: keep one block more than the budget asks for, capped at the
/// blocks the row can see.
pub const QSA_SELECT_KEEP_MORE: i32 = 3;
/// Gate-only: set every column past the row's own token.
pub const QSA_SELECT_IGNORE_CAUSALITY: i32 = 4;
/// Gate-only: keep the lowest scores instead of the highest.
pub const QSA_SELECT_INVERT_ORDER: i32 = 5;

/// The extents `qsa_indexer_select_into` validates before it forms a pointer,
/// as the kernel's `i32` arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SelectPlan {
    query_rows: i32,
    tokens: i32,
    score_cols: i32,
}

/// The launcher's pre-launch contract. Split from the launch so the guards can
/// be held without a device (see the unit tests below): every check here is
/// host arithmetic over buffer lengths and scalars.
///
/// `score_capacity` is the flat length of the score buffer and `score_cols` its
/// row stride, which is only unambiguous when the length divides evenly by the
/// row count. The stride must hold `tokens / COMPRESS_RATIO` blocks — one row's
/// ceiling — while the kernel reads only each row's own `(rows[j] + 1) / 4`.
fn select_plan(
    query_rows: usize,
    tokens: usize,
    score_capacity: usize,
    mask_capacity: usize,
    budget: i32,
    variant: i32,
) -> Result<SelectPlan> {
    ensure!(
        (QSA_SELECT_PRODUCTION..=QSA_SELECT_INVERT_ORDER).contains(&variant),
        "QSA indexer select variant {variant} is not one of the gate's controls"
    );
    // Unsigned arithmetic: `i32::is_multiple_of` is not stable in the pinned
    // toolchain, and the conversion already rejects a negative budget.
    let budget_blocks = usize::try_from(budget)
        .ok()
        .filter(|blocks| blocks.is_multiple_of(COMPRESS_RATIO))
        .map(|blocks| blocks / COMPRESS_RATIO);
    ensure!(
        budget_blocks.is_some_and(|blocks| blocks > 0 && blocks <= MAX_KEPT_BLOCKS),
        "QSA indexer budget {budget} must be a positive multiple of {COMPRESS_RATIO} \
         keeping at most {MAX_KEPT_BLOCKS} blocks"
    );
    ensure!(query_rows > 0, "QSA indexer select needs query rows");
    ensure!(tokens > 0, "QSA indexer select needs tokens");
    // The 64-bit mask cell count comes first: a product that does not fit would
    // otherwise be divided into a plausible-looking stride below.
    let cells = query_rows
        .checked_mul(tokens)
        .ok_or_else(|| anyhow::anyhow!("QSA indexer mask {query_rows} x {tokens} overflows"))?;
    ensure!(
        score_capacity.is_multiple_of(query_rows),
        "QSA indexer score buffer {score_capacity} f32 does not divide into {query_rows} rows"
    );
    let score_cols = score_capacity / query_rows;
    ensure!(
        score_cols >= tokens / COMPRESS_RATIO,
        "QSA indexer score row holds {score_cols} blocks, one row can see \
         {} of {tokens} tokens",
        tokens / COMPRESS_RATIO
    );
    ensure!(
        mask_capacity >= cells,
        "QSA indexer mask holds {} bytes, needs {cells} for {query_rows} x {tokens}",
        mask_capacity
    );
    Ok(SelectPlan {
        query_rows: checked_i32(query_rows, "QSA indexer query rows")?,
        tokens: checked_i32(tokens, "QSA indexer tokens")?,
        score_cols: checked_i32(score_cols, "QSA indexer score columns")?,
    })
}

/// Turn one score per (query row, complete block) into the row's token-level
/// visibility mask: the tokens of the `min(budget / COMPRESS_RATIO, blocks)`
/// highest block scores plus the row's incomplete tail.
///
/// `scores` is `[rows.len(), score_cols]` f32 with row stride
/// `scores.len() / rows.len()`; only `(rows[j] + 1) / 4` entries of row `j` are
/// read, so the NaN padding the scoring stage leaves past that prefix is never
/// touched. `rows[j]` is row `j`'s position, also its row order in `scores` and
/// `mask`. `budget` is the token budget, a positive multiple of four keeping at
/// most 512 blocks; `tokens` is the mask's column count and the row's causal
/// ceiling (`pos < tokens`).
///
/// The visible list starts at token 0, the scoring stage's grid contract: row
/// `j` sees exactly `[0, rows[j]]`, and its blocks are `[0, (rows[j] + 1) / 4)`.
/// A prefix-evicted visible list is out of contract (it needs a shifted key
/// array on both stages), which no buffer check here can detect.
///
/// `mask` is one byte per (row, token), `0` or `1`, row stride `tokens`: `1`
/// exactly for the tokens of the selected blocks, whose blocks cover tokens
/// below the row's position, and for the tail `4 * blocks ..= pos`. Causality
/// is therefore inherent, not an extra condition. Ties go to the lower block
/// index (descending score, ascending index), the canonical order the parity
/// gate holds the golden to; `torch.topk` leaves a block tied at the cut open,
/// so the fixture's own pick can differ there and still be valid.
///
/// This is [`qsa_indexer_select_variant_into`] with [`QSA_SELECT_PRODUCTION`];
/// the production path has no mutation knob. A non-finite score inside a row's
/// block prefix or an out-of-range `rows` entry traps on the device: the
/// caller's input is broken, and a silently shorter visible set would hand the
/// attention softmax an undefined row. The kernel repeats the budget and score
/// row width checks the host entry already made, so a caller reaching it by
/// another path still fails visibly. The wrapper refuses an empty launch, a
/// malformed budget and a short buffer before any pointer is formed.
pub fn qsa_indexer_select_into(
    ctx: &DeviceContext,
    scores: &CudaSlice<f32>,
    rows: &CudaSlice<i32>,
    budget: i32,
    tokens: usize,
    mask: &mut CudaSlice<u8>,
) -> Result<()> {
    qsa_indexer_select_variant_into(
        ctx,
        scores,
        rows,
        budget,
        tokens,
        mask,
        QSA_SELECT_PRODUCTION,
    )
}

/// [`qsa_indexer_select_into`] with the parity gate's mutation knob.
///
/// `variant` is [`QSA_SELECT_PRODUCTION`] in production. The other
/// `QSA_SELECT_*` values are the gate's negative controls — they run the same
/// launch and mutate one step of the selection so the gate proves its assertions
/// bite — and are refused outside `0..=5`.
pub fn qsa_indexer_select_variant_into(
    ctx: &DeviceContext,
    scores: &CudaSlice<f32>,
    rows: &CudaSlice<i32>,
    budget: i32,
    tokens: usize,
    mask: &mut CudaSlice<u8>,
    variant: i32,
) -> Result<()> {
    let plan = select_plan(
        rows.len(),
        tokens,
        scores.len(),
        mask.len(),
        budget,
        variant,
    )?;
    let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = rows.device_ptr(&ctx.stream);
    let (mask_ptr, _mask_guard) = mask.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_qsa_indexer_select_cuda(
            scores_ptr as *const f32,
            rows_ptr as *const i32,
            mask_ptr as *mut u8,
            plan.query_rows,
            plan.tokens,
            plan.score_cols,
            budget,
            variant,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer select launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows, tokens, score capacity and mask capacity of one valid launch:
    /// two query rows, eight tokens, two blocks per score row.
    const VALID: (usize, usize, usize, usize) = (2, 8, 4, 16);

    fn plan(
        query_rows: usize,
        tokens: usize,
        scores: usize,
        mask: usize,
        budget: i32,
        variant: i32,
    ) -> Result<SelectPlan> {
        select_plan(query_rows, tokens, scores, mask, budget, variant)
    }

    #[test]
    fn a_complete_launch_passes() {
        let (rows, tokens, scores, mask) = VALID;
        assert_eq!(
            plan(rows, tokens, scores, mask, 8, QSA_SELECT_PRODUCTION).expect("valid"),
            SelectPlan {
                query_rows: 2,
                tokens: 8,
                score_cols: 2,
            }
        );
        // The budget ceiling itself, and the gate's last control, both pass.
        assert!(plan(rows, tokens, scores, mask, 2048, QSA_SELECT_INVERT_ORDER).is_ok());
    }

    #[test]
    fn malformed_budgets_are_refused() {
        let (rows, tokens, scores, mask) = VALID;
        for budget in [0, -4, 2, 7, 2052, i32::MIN, i32::MAX] {
            assert!(
                plan(rows, tokens, scores, mask, budget, QSA_SELECT_PRODUCTION).is_err(),
                "budget {budget} must be refused"
            );
        }
        // 2048 tokens is the checkpoint's budget and the block ceiling; one
        // more multiple of four is past it.
        assert!(plan(rows, tokens, scores, mask, 2048, 0).is_ok());
        assert!(plan(rows, tokens, scores, mask, 2052, 0).is_err());
    }

    #[test]
    fn empty_extents_are_refused() {
        let (_, tokens, scores, mask) = VALID;
        assert!(plan(0, tokens, 0, mask, 8, 0).is_err(), "no rows");
        assert!(plan(2, 0, scores, 0, 8, 0).is_err(), "no tokens");
    }

    #[test]
    fn short_buffers_are_refused() {
        let (rows, tokens, scores, mask) = VALID;
        // 3 f32 does not divide into 2 rows: the stride would be ambiguous.
        assert!(plan(rows, tokens, 3, mask, 8, 0).is_err());
        // One column per row cannot hold the two blocks an 8-token row sees.
        assert!(plan(rows, tokens, 2, mask, 8, 0).is_err());
        // One byte short of the [2, 8] mask.
        assert!(plan(rows, tokens, scores, mask - 1, 8, 0).is_err());
        // A score row wider than the block grid is fine: only the prefix is read.
        assert!(plan(rows, tokens, 2 * 16, mask, 8, 0).is_ok());
    }

    #[test]
    fn unknown_variants_are_refused() {
        let (rows, tokens, scores, mask) = VALID;
        for variant in [-1, 6, i32::MIN, i32::MAX] {
            assert!(
                plan(rows, tokens, scores, mask, 8, variant).is_err(),
                "variant {variant} must be refused"
            );
        }
        for variant in 0..=5 {
            assert!(plan(rows, tokens, scores, mask, 8, variant).is_ok());
        }
    }

    #[test]
    fn an_overflowing_mask_product_is_refused() {
        assert!(plan(usize::MAX, 8, 8, usize::MAX, 8, 0).is_err());
    }
}
