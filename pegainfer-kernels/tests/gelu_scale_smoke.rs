//! Device gate for the Gemma 4 elementwise additions in
//! csrc/shared/elementwise.cu: gelu_tanh_mul and scale_bf16.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip.
//!
//! Scope note: on the bf16 grid the tanh approximation and the erf GELU are
//! numerically indistinguishable (their worst-case gap is below half a bf16
//! ulp), and the 2-ulp tolerance that absorbs device-vs-host tanh drift also
//! absorbs the kernel's intermediate bf16 round — this gate pins the formula
//! (a SiLU swap, a dropped cubic term), not the variant or the op-sequence
//! rounding. The variant question rides on the Gemma 4 layer oracle against
//! the HF fixture.

mod common;

use half::bf16;
use pegainfer_kernels::ops::gelu_tanh_mul_batch_into;
use pegainfer_kernels::ops::scale_bf16_in_place;
use pegainfer_kernels::tensor::HiddenStates;

/// Mirrors the kernel arithmetic: activation in f32, cast to bf16, then
/// multiplied in f32 and rounded once more.
fn host_gelu_tanh_mul(g: f32, u: f32) -> f32 {
    let inner = 0.797_884_6_f32 * (g + 0.044_715_f32 * g * g * g);
    let gelu_g = 0.5_f32 * g * (1.0_f32 + inner.tanh());
    let gelu_bf16 = bf16::from_f32(gelu_g).to_f32();
    bf16::from_f32(gelu_bf16 * u).to_f32()
}

#[test]
fn gelu_tanh_mul_matches_host_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // x = 2.0 is the constant-sensitive case: dropping the 0.044715 cubic
    // term shifts gelu(2) by ~1.7%, and at up 2.0 the product moves 0.0625 —
    // twice the 0.031 tolerance there; x = 1.0 separates GELU from a SiLU
    // swap (0.8412 vs 0.7311). Negatives cover the branch where GELU
    // saturates toward zero.
    let gate_vals = [-8.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 8.0];
    let up_vals = [2.0f32, -1.5, 1.0, 3.0, 2.0, -2.0, 1.0, 2.0, 2.0];

    let gate_host: Vec<bf16> = gate_vals.iter().map(|&v| bf16::from_f32(v)).collect();
    let up_host: Vec<bf16> = up_vals.iter().map(|&v| bf16::from_f32(v)).collect();
    let n = gate_vals.len();
    let gate = HiddenStates::from_host(ctx, &gate_host, n, 1).expect("gate H2D");
    let up = HiddenStates::from_host(ctx, &up_host, n, 1).expect("up H2D");
    let mut out = HiddenStates::zeros(ctx, n, 1).expect("out alloc");

    gelu_tanh_mul_batch_into(ctx, &gate, &up, &mut out).expect("gelu launch");

    let got = out.to_host(ctx).expect("out D2H");
    for (i, (&g, &u)) in gate_vals.iter().zip(&up_vals).enumerate() {
        let e = host_gelu_tanh_mul(g, u);
        // Two bf16 ulp relative, floored for near-zero expectations: host
        // tanh and device tanhf may differ in the last f32 ulp, which after
        // the bf16 round can move one grid step on boundary values.
        let tol = (e.abs() * 0.008).max(0.02);
        assert!(
            (got[i] - e).abs() <= tol,
            "gelu_tanh_mul[{i}] (gate {g}, up {u}): got {}, expected {e} \
             (tolerance {tol})",
            got[i]
        );
    }
}

/// Powers of two times small integers stay exactly representable in bf16
/// under a power-of-two scale, so the assertion is bitwise equality — an
/// off-by-one bound that skips the last element cannot pass.
#[test]
#[allow(clippy::float_cmp)]
fn scale_bf16_scales_exactly() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let vals = [-4.0f32, -1.5, -0.5, 0.0, 0.25, 1.0, 3.0, 8.0];
    let host: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut buf = HiddenStates::from_host(ctx, &host, vals.len(), 1).expect("buf H2D");

    scale_bf16_in_place(ctx, &mut buf, 2.0).expect("scale launch");

    let got = buf.to_host(ctx).expect("buf D2H");
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            got[i],
            v * 2.0,
            "scale_bf16[{i}]: got {}, expected {}",
            got[i],
            v * 2.0
        );
    }
}

#[test]
fn scale_bf16_rejects_non_finite_scale() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let mut buf = HiddenStates::zeros(ctx, 4, 1).expect("buf alloc");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            scale_bf16_in_place(ctx, &mut buf, bad).is_err(),
            "scale {bad} must be rejected"
        );
    }
}
