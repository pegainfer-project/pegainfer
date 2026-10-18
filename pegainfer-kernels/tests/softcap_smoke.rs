//! Device gate for the final-logit softcap in csrc/shared/elementwise.cu.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip.

mod common;

use half::bf16;
use pegainfer_kernels::ops::softcap_bf16_in_place;
use pegainfer_kernels::tensor::HiddenStates;

/// Mirrors the kernel arithmetic: f32 tanh, one rounding back to bf16.
fn host_softcap(x: f32, cap: f32) -> f32 {
    bf16::from_f32(cap * (x / cap).tanh()).to_f32()
}

#[test]
fn softcap_matches_host_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // Values at and past the cap are the sensitive cases: near zero the cap
    // is a near-identity (tanh linear region), so a mutation that drops the
    // division or the multiply only shows at |x| comparable to cap. 30.0 is
    // the value every published Gemma 4 size declares.
    let cap = 30.0f32;
    let vals = [-90.0f32, -30.0, -4.0, -0.5, 0.0, 0.5, 4.0, 30.0, 90.0];
    let host: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut buf = HiddenStates::from_host(ctx, &host, vals.len(), 1).expect("buf H2D");

    softcap_bf16_in_place(ctx, &mut buf, cap).expect("softcap launch");

    let got = buf.to_host(ctx).expect("buf D2H");
    for (i, &v) in vals.iter().enumerate() {
        let e = host_softcap(v, cap);
        // Two bf16 ulp relative, floored near zero: host tanh and device
        // tanhf may differ in the last f32 ulp.
        let tol = (e.abs() * 0.008).max(0.02);
        assert!(
            (got[i] - e).abs() <= tol,
            "softcap[{i}] (x {v}): got {}, expected {e} (tolerance {tol})",
            got[i]
        );
    }
}

#[test]
fn softcap_rejects_bad_cap() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let mut buf = HiddenStates::zeros(ctx, 4, 1).expect("buf alloc");
    for bad in [0.0f32, -30.0, f32::NAN, f32::INFINITY] {
        assert!(
            softcap_bf16_in_place(ctx, &mut buf, bad).is_err(),
            "cap {bad} must be rejected"
        );
    }
}
