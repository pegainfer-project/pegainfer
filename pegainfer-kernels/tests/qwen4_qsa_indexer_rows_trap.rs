//! The QSA indexer's out-of-range `rows` trap, one trap per binary (see
//! `qwen4_qsa_trap`).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;
mod qwen4_qsa_trap;

/// A `rows` entry past the key array must fail visibly. Checking it on the host
/// would need a D2H copy of device-resident metadata, so the kernel traps.
#[test]
fn out_of_range_row_traps() {
    // Row 8 is the first position past the 8-token key array.
    qwen4_qsa_trap::assert_traps(8, 8);
}
