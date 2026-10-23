//! The QSA selector's negative `rows` trap, one trap per binary (see
//! `qwen4_qsa_trap`): `__trap()` leaves the context in a sticky error state, so
//! every trap gets its own process.
//!
//! `rows` is device-resident metadata, so the selector checks it in the kernel
//! rather than copying it back. A negative position cannot be clamped: it would
//! describe a visibility set the caller never asked for, and the mask would look
//! plausible.
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`, last, on a
//! disposable device: a deliberate trap can leave a shared GPU needing a reset
//! (observed on an RTX 5060 / CUDA 12.9.1, where `cuInit` then failed for every
//! later process until the device was reset).

#![cfg(feature = "qwen4")]

mod common;

use pegainfer_kernels::ops::qsa_indexer_select_into;

const TOKENS: usize = 8;
const BUDGET: i32 = 8;

#[test]
fn a_negative_row_traps() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // Position -1 is the first token before the eight-token mask row. The score
    // row holds the two blocks the row's ceiling allows, so only the position
    // can reject it.
    let scores = ctx.stream.clone_htod(&[1.0f32, 0.5]).expect("scores H2D");
    let rows = ctx.stream.clone_htod(&[-1i32]).expect("rows H2D");
    let mut mask = ctx.stream.alloc_zeros::<u8>(TOKENS).expect("mask");
    let launch = qsa_indexer_select_into(&ctx, &scores, &rows, BUDGET, TOKENS, &mut mask);
    let error = match launch {
        Ok(()) => ctx
            .sync()
            .err()
            .unwrap_or_else(|| panic!("row -1 must fail visibly in an {TOKENS}-token mask")),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("CUDA_ERROR_LAUNCH_FAILED"),
        "row -1 must end in the kernel's trap, got: {message}"
    );
}
