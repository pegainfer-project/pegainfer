//! The QSA selector's non-finite score trap, one trap per binary (see
//! `qwen4_qsa_trap`).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;
mod qwen4_qsa_trap;

/// A NaN inside a row's block prefix must fail visibly: skipping it (what a
/// bare comparison does) would drop a block from the visible set without a
/// trace.
#[test]
fn a_non_finite_score_traps() {
    // The second block is poisoned inside the row's contract (two blocks for
    // eight visible tokens), not in the padding past them.
    qwen4_qsa_trap::assert_select_traps("a NaN block", &[1.0, f32::NAN], &[7]);
}
