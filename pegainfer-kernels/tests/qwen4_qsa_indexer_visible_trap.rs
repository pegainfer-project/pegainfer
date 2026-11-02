//! The QSA indexer's out-of-range `visible_len` trap, one trap per binary (see
//! `qwen4_qsa_trap`).
//!
//! Manual gate: no CI job enables `qwen4`, so CI neither builds nor runs it. Run
//! it with `--features qwen4` and `PEGAINFER_REQUIRE_GPU=1`.

#![cfg(feature = "qwen4")]

mod common;
mod qwen4_qsa_trap;

/// A `visible_len` entry outside `[0, rows[j] + 1]` must fail visibly rather
/// than scoring blocks past the row.
#[test]
fn out_of_range_visible_traps() {
    // Row 3 sees at most 4 tokens. Eight stays inside the key array, so only
    // the cap at the row's own token can reject it.
    qwen4_qsa_trap::assert_traps(3, 8);
}
