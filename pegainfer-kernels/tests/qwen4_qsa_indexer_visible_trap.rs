//! The QSA indexer's out-of-range `visible_len` trap. One trap per binary:
//! `__trap()` leaves the context in a sticky error state, so anything sharing
//! the process afterwards would fail for the wrong reason.
//!
//! Manual gate — CI compiles this but never runs it.

#![cfg(feature = "qwen4")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::qsa_indexer_block_scores_into;
use pegainfer_kernels::tensor::DeviceVec;

const TOKENS: usize = 8;
// The kernel's compiled geometry, used to size the buffers; only their capacity
// is checked, and the trap fires before any of them is read.
const FUSED_WIDTH: usize = 640;
const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 64;
const EPS: f32 = 1e-6;
const SCORE_COLS: usize = TOKENS / 4;

/// A `visible_len` entry outside `[0, tokens]` must fail visibly rather than
/// being clamped into a plausible score. The kernel traps; the async error
/// surfaces at the launch or the next sync.
#[test]
fn out_of_range_visible_traps() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let score_cols = SCORE_COLS;
    let fused = ctx
        .stream
        .clone_htod(&vec![bf16::ZERO; TOKENS * FUSED_WIDTH])
        .expect("fused");
    let q_norm = ctx
        .stream
        .clone_htod(&vec![bf16::ZERO; HEAD_DIM])
        .expect("q norm");
    let k_norm = ctx
        .stream
        .clone_htod(&vec![bf16::ZERO; HEAD_DIM])
        .expect("k norm");
    // The trap fires before any table read, so the contents are irrelevant;
    // only the tables' capacity is checked by the wrapper.
    let cos = DeviceVec::from_host(ctx, &vec![bf16::ZERO; TOKENS * ROTARY_DIM]).expect("cos table");
    let sin = DeviceVec::from_host(ctx, &vec![bf16::ZERO; TOKENS * ROTARY_DIM]).expect("sin table");
    let rows = ctx.stream.clone_htod(&[0i32]).expect("rows H2D");
    // One token past the key array: the valid range is 0..=8.
    let visible = ctx
        .stream
        .clone_htod(&[(TOKENS + 1) as i32])
        .expect("visible H2D");
    let mut scores = ctx
        .stream
        .alloc_zeros::<f32>(score_cols)
        .expect("scores alloc");
    let mut pooled = ctx
        .stream
        .alloc_zeros::<bf16>(score_cols * HEAD_DIM)
        .expect("pooled alloc");

    let res = qsa_indexer_block_scores_into(
        ctx,
        &fused,
        &q_norm,
        &k_norm,
        &cos,
        &sin,
        &rows,
        &visible,
        TOKENS,
        &mut scores,
        score_cols,
        &mut pooled,
        EPS,
    );
    match res {
        Ok(()) => assert!(
            ctx.sync().is_err(),
            "visible {} in an {TOKENS}-token buffer must fail visibly",
            TOKENS + 1
        ),
        // The trap may surface at the launch instead of the next sync. Any other
        // error means a wrapper check ran instead of the kernel, which would
        // make this gate green without ever exercising the trap.
        Err(err) => assert!(
            format!("{err:#}").contains("launch failed"),
            "the trap must come from the launch, not a wrapper check: {err:#}"
        ),
    }
}
