//! The QSA selector's non-finite score trap, one trap per binary (see
//! `qwen4_qsa_trap`): `__trap()` leaves the context in a sticky error state, so
//! every trap gets its own process.
//!
//! A NaN inside a row's block prefix is an upstream contract violation the
//! kernel fails closed on. Skipping it — which is what a bare comparison does —
//! would drop that block from the visible set without a trace, and the mask's
//! complement drives the attention softmax, where an all-masked row is
//! undefined. The NaN padding past the prefix is a different thing and is
//! ignored (the parity gate feeds it back verbatim).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;

use pegainfer_kernels::ops::qsa_indexer_select_into;

/// One query row at position 7 sees two complete blocks of this key array.
const TOKENS: usize = 8;
const ROW: i32 = 7;
const BUDGET: i32 = 8;

#[test]
fn a_non_finite_score_traps() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // The second block is poisoned inside the row's contract (two blocks for
    // eight visible tokens), not in the padding past them.
    let scores = ctx
        .stream
        .clone_htod(&[1.0f32, f32::NAN])
        .expect("scores H2D");
    let rows = ctx.stream.clone_htod(&[ROW]).expect("rows H2D");
    let mut mask = ctx.stream.alloc_zeros::<u8>(TOKENS).expect("mask");
    let launch = qsa_indexer_select_into(&ctx, &scores, &rows, BUDGET, TOKENS, &mut mask, 0);
    let error = match launch {
        Ok(()) => ctx.sync().err().unwrap_or_else(|| {
            panic!("a NaN block score must fail visibly instead of selecting around it")
        }),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("CUDA_ERROR_LAUNCH_FAILED"),
        "a NaN block score must end in the kernel's trap, got: {message}"
    );
}
