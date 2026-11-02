//! The QSA sparse-attention negative `rows` trap, one trap per binary (see
//! `qwen4_qsa_trap`).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;
mod qwen4_qsa_trap;

/// A negative `rows` entry must fail visibly instead of being clamped to token
/// zero.
#[test]
fn negative_row_traps() {
    qwen4_qsa_trap::assert_attn_row_traps(-1);
}
