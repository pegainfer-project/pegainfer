//! The QSA selector's negative `rows` trap, one trap per binary (see
//! `qwen4_qsa_trap`).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;
mod qwen4_qsa_trap;

/// A negative `rows` entry must fail visibly; clamping it would describe a
/// visibility set the caller never asked for.
#[test]
fn a_negative_row_traps() {
    // Position -1 is the first token before the eight-token mask row.
    qwen4_qsa_trap::assert_select_traps("row -1", &[1.0, 0.5], &[-1]);
}
