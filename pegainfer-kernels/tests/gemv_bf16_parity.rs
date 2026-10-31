//! Batch-1 matvec gate.
//!
//! A line that opts in routes a single decode token's projections through the
//! warp-per-row matvec instead of the tiled GEMM. This gate holds that kernel to
//! a float64 reference on the shapes Qwen3.5-4B launches per decode step, holds
//! the shapes it cannot serve (unaligned K, too few rows, more than one token)
//! on the GEMM path, and holds a line that never opts in on the GEMM path too —
//! a silent fallback would make the parity numbers meaningless.
//!
//!   cargo test --release -p pegainfer-kernels --test gemv_bf16_parity -- --nocapture

use std::sync::Mutex;

use half::bf16;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::gemv_served;
use pegainfer_kernels::ops::set_n1_matvec;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

/// The routing flag and its counter are process-global, so these gates run one
/// at a time: in parallel, one test's served count would land inside another
/// test's before/after window.
static GATE: Mutex<()> = Mutex::new(());

/// (label, M = out rows, K = in cols): the projections of one Qwen3.5-4B layer
/// plus the LM head, at the hidden sizes the decode step uses.
const SERVED: &[(&str, usize, usize)] = &[
    ("gate_up", 18432, 2560),
    ("down_proj", 2560, 9728),
    ("o_proj", 2560, 2560),
    ("q_proj", 4096, 2560),
    ("k_proj", 1024, 2560),
];

/// Deterministic value in [-1, 1) from (seed, a, b).
fn g(seed: u64, a: usize, b: usize) -> f32 {
    let mut h = seed
        ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn w_data(seed: u64, rows: usize, cols: usize) -> Vec<bf16> {
    (0..rows * cols)
        .map(|i| bf16::from_f32(g(seed, i / cols, i % cols)))
        .collect()
}

/// Column-major activations [cols, tokens].
fn x_data(seed: u64, cols: usize, tokens: usize) -> Vec<bf16> {
    (0..cols * tokens)
        .map(|i| bf16::from_f32(g(seed, i / cols, i % cols)))
        .collect()
}

/// Row `row` of `W` dotted with `x`, accumulated in float64 and rounded once.
fn reference(w: &[bf16], x: &[bf16], cols: usize, row: usize) -> f32 {
    let mut acc = 0.0f64;
    for c in 0..cols {
        acc += f64::from(w[row * cols + c].to_f32()) * f64::from(x[c].to_f32());
    }
    bf16::from_f32(acc as f32).to_f32()
}

/// One bf16 ulp is 2^-8 of the magnitude; the float64 reference differs from the
/// kernel's float32 accumulation by far less than that.
fn tolerance(reference: f32) -> f32 {
    2.0 * reference.abs().max(1.0) * 2f32.powi(-8)
}

fn check_shape(ctx: &DeviceContext, name: &str, rows: usize, cols: usize, seed: u64) {
    let w = w_data(seed, rows, cols);
    let x = x_data(seed ^ 0x5A5A, cols, 1);
    let w_d = DeviceMatrix::from_host(ctx, &w, rows, cols).expect("W h2d");
    let x_d = HiddenStates::from_host(ctx, &x, cols, 1).expect("X h2d");
    let mut out = HiddenStates::zeros(ctx, rows, 1).expect("out");

    let served_before = gemv_served();
    gemm_into_checked(ctx, &w_d, &x_d, &mut out).expect("projection");
    let served_after = gemv_served();
    assert!(
        served_after > served_before,
        "{name}: the batch-1 projection did not take the matvec"
    );

    let got = out.to_host(ctx).expect("d2h");
    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for (row, &served) in got.iter().enumerate() {
        let want = reference(&w, &x, cols, row);
        let diff = (served - want).abs();
        let rel = diff / want.abs().max(1e-6);
        assert!(
            diff <= tolerance(want),
            "{name}: row {row} got {served} want {want} (|Δ|={diff:.3e}, rel={rel:.3e})"
        );
        if diff > worst {
            worst = diff;
            worst_row = row;
        }
    }
    eprintln!("{name:<12} {rows:>6}x{cols:<5} worst|Δ|={worst:.3e} (row {worst_row})");
}

#[test]
fn batch_one_projections_take_the_matvec() {
    let _gate = GATE.lock().expect("gate lock");
    set_n1_matvec(true);
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping gemv_bf16_parity: no CUDA device — gate inconclusive");
        return;
    };
    for (i, &(name, rows, cols)) in SERVED.iter().enumerate() {
        check_shape(&ctx, name, rows, cols, 0x9E37_0000 ^ i as u64);
    }
}

#[test]
fn lines_that_never_opt_in_keep_the_gemm() {
    let _gate = GATE.lock().expect("gate lock");
    set_n1_matvec(false);
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping gemv_bf16_parity: no CUDA device — gate inconclusive");
        return;
    };
    let (name, rows, cols) = SERVED[0];
    let w = w_data(0x0F0F, rows, cols);
    let x = x_data(0x1E1E, cols, 1);
    let w_d = DeviceMatrix::from_host(&ctx, &w, rows, cols).expect("W h2d");
    let x_d = HiddenStates::from_host(&ctx, &x, cols, 1).expect("X h2d");
    let mut out = HiddenStates::zeros(&ctx, rows, 1).expect("out");
    let served_before = gemv_served();
    gemm_into_checked(&ctx, &w_d, &x_d, &mut out).expect("projection");
    assert_eq!(
        gemv_served(),
        served_before,
        "{name}: a batch-1 projection on a line that never opted in must stay on the GEMM"
    );
}

#[test]
fn shapes_the_matvec_cannot_serve_stay_on_the_gemm() {
    let _gate = GATE.lock().expect("gate lock");
    set_n1_matvec(true);
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping gemv_bf16_parity: no CUDA device — gate inconclusive");
        return;
    };
    // K not a multiple of 8: the 16-byte load beats go out of alignment.
    let rows = 2560;
    let cols = 2570;
    let w = w_data(0x1234, rows, cols);
    let x = x_data(0x5678, cols, 1);
    let w_d = DeviceMatrix::from_host(&ctx, &w, rows, cols).expect("W h2d");
    let x_d = HiddenStates::from_host(&ctx, &x, cols, 1).expect("X h2d");
    let mut out = HiddenStates::zeros(&ctx, rows, 1).expect("out");
    let served_before = gemv_served();
    gemm_into_checked(&ctx, &w_d, &x_d, &mut out).expect("projection");
    assert_eq!(
        gemv_served(),
        served_before,
        "K={cols} is not 16-byte aligned and must stay on the GEMM"
    );

    // Two tokens are outside the matvec's contract even when K aligns.
    let cols = 2560;
    let w = w_data(0x1234, rows, cols);
    let x = x_data(0x5678, cols, 2);
    let w_d = DeviceMatrix::from_host(&ctx, &w, rows, cols).expect("W h2d");
    let x_d = HiddenStates::from_host(&ctx, &x, cols, 2).expect("X h2d");
    let mut out = HiddenStates::zeros(&ctx, rows, 2).expect("out");
    gemm_into_checked(&ctx, &w_d, &x_d, &mut out).expect("projection");
    assert_eq!(
        gemv_served(),
        served_before,
        "a two-token batch must stay on the GEMM"
    );

    // Fewer rows than the fill threshold: the grid cannot saturate the device.
    let rows = 128;
    let cols = 2560;
    let w = w_data(0x9ABC, rows, cols);
    let x = x_data(0xDEF0, cols, 1);
    let w_d = DeviceMatrix::from_host(&ctx, &w, rows, cols).expect("W h2d");
    let x_d = HiddenStates::from_host(&ctx, &x, cols, 1).expect("X h2d");
    let mut out = HiddenStates::zeros(&ctx, rows, 1).expect("out");
    gemm_into_checked(&ctx, &w_d, &x_d, &mut out).expect("projection");
    assert_eq!(
        gemv_served(),
        served_before,
        "M={rows} is below the fill threshold and must stay on the GEMM"
    );
}
