//! Shared launch for the QSA indexer trap binaries. Each binary fires one trap,
//! because `__trap()` leaves the context in a sticky error state and anything
//! sharing the process afterwards would fail for the wrong reason.

use half::bf16;
use pegainfer_kernels::ops::qsa_indexer_block_scores_into;
use pegainfer_kernels::tensor::DeviceVec;

const TOKENS: usize = 8;
// The kernel's compiled geometry, used to size the buffers. Only their capacity
// is checked, and the trap fires before any of them is read, so they hold zeros.
const FUSED_WIDTH: usize = 640;
const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 64;
const SCORE_COLS: usize = TOKENS / 4;
const EPS: f32 = 1e-6;

/// Score one row of an 8 token key array at position `row` seeing `visible`
/// tokens, and require the kernel's trap. A trap ends the launch with
/// `CUDA_ERROR_LAUNCH_FAILED`, reported at the launch or at the next sync. Any
/// other outcome fails, including a wrapper check, an invalid argument and an
/// illegal address, so the test cannot pass without reaching the trap.
pub(crate) fn assert_traps(row: i32, visible: i32) {
    let Some(ctx) = crate::common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let zeros = |len: usize| ctx.stream.clone_htod(&vec![bf16::ZERO; len]).expect("H2D");
    let fused = zeros(TOKENS * FUSED_WIDTH);
    let q_norm = zeros(HEAD_DIM);
    let k_norm = zeros(HEAD_DIM);
    let cos = DeviceVec::from_host(ctx, &vec![bf16::ZERO; TOKENS * ROTARY_DIM]).expect("cos");
    let sin = DeviceVec::from_host(ctx, &vec![bf16::ZERO; TOKENS * ROTARY_DIM]).expect("sin");
    let rows = ctx.stream.clone_htod(&[row]).expect("rows H2D");
    let visible_len = ctx.stream.clone_htod(&[visible]).expect("visible H2D");
    let mut scores = ctx.stream.alloc_zeros::<f32>(SCORE_COLS).expect("scores");
    let mut pooled = ctx
        .stream
        .alloc_zeros::<bf16>(SCORE_COLS * HEAD_DIM)
        .expect("pooled");

    let launch = qsa_indexer_block_scores_into(
        ctx,
        &fused,
        &q_norm,
        &k_norm,
        &cos,
        &sin,
        &rows,
        &visible_len,
        TOKENS,
        &mut scores,
        SCORE_COLS,
        &mut pooled,
        EPS,
    );
    let error = match launch {
        Ok(()) => ctx.sync().err().unwrap_or_else(|| {
            panic!("row {row} seeing {visible} of {TOKENS} tokens must fail visibly")
        }),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("CUDA_ERROR_LAUNCH_FAILED"),
        "row {row} seeing {visible} tokens must end in the kernel's trap, got: {message}"
    );
}
