//! Batch-invariance gate at the GEMM seam.
//!
//! A projection GEMM Y = W·X runs at N = total batched tokens; cuBLAS picks its tile/split-K by N,
//! so a fixed token's bf16 output changes with the batch it rides in. Column 0 is held identical
//! across an N-sweep, run three ways: baseline (drifts = the bug), pin (one cublasLt algo for all
//! N = the fix), per_token (N=1 oracle). bf16→f32 is lossless, so the comparison is bitwise.
//!
//!   PEGAINFER_BI_REPS=8 cargo test --release -p pegainfer-kernels \
//!     --test batch_invariance_gemm -- --nocapture --test-threads=1

use half::bf16;
use pegainfer_kernels::ops::GEMM_LT_MAX_N;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::gemm_lt_pin_into_checked;
use pegainfer_kernels::ops::gemm_lt_pin_tune;
use pegainfer_kernels::ops::gemm_per_token;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

/// (label, M = out rows, K = in cols). down_proj is the shape whose winning
/// split-K varied across N in the original repro — the strongest drift.
const SHAPES: &[(&str, usize, usize)] = &[
    ("down_proj", 2560, 9728),
    ("o_proj", 2560, 2560),
    ("qkv_wide", 4096, 2560),
];

/// N sweep straddling the cublasLt/GemmEx boundary (GEMM_LT_MAX_N).
const NS: &[usize] = &[1, 2, 4, 8, 16, 24, 32, 48, 64];

const REP_N: usize = 32;

fn reps() -> usize {
    std::env::var("PEGAINFER_BI_REPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// Deterministic value in [-1, 1) from (seed, a, b), independent of N.
fn g(seed: u64, a: usize, b: usize) -> f32 {
    let mut h = seed
        ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Row-major weight [rows, cols]: element (m, k) at m*cols + k.
fn w_data(seed: u64, rows: usize, cols: usize) -> Vec<bf16> {
    (0..rows * cols)
        .map(|i| bf16::from_f32(g(seed, i / cols, i % cols)))
        .collect()
}

/// Column-major activations [k, n]: element (col c, row r) at c*k + r.
fn x_data(seed: u64, k: usize, n: usize) -> Vec<bf16> {
    let mut v = Vec::with_capacity(k * n);
    for c in 0..n {
        for r in 0..k {
            v.push(bf16::from_f32(g(seed, c, r)));
        }
    }
    v
}

#[derive(Clone, Copy)]
enum Arm {
    Baseline,
    Pin,
    PerToken,
}

/// Token-0 output for `arm` at this N, or `None` if the pinned algo cannot serve
/// this N (pin arm only).
fn col0(
    ctx: &DeviceContext,
    arm: Arm,
    w: &DeviceMatrix,
    x: &HiddenStates,
    m: usize,
) -> Option<Vec<f32>> {
    match arm {
        Arm::Baseline => {
            let mut out = HiddenStates::zeros(ctx, m, x.seq_len).expect("out");
            gemm_into_checked(ctx, w, x, &mut out).expect("baseline gemm");
            Some(out.to_host(ctx).expect("d2h")[..m].to_vec())
        }
        Arm::Pin => {
            let mut out = HiddenStates::zeros(ctx, m, x.seq_len).expect("out");
            if gemm_lt_pin_into_checked(ctx, w, x, &mut out).expect("pin gemm") {
                Some(out.to_host(ctx).expect("d2h")[..m].to_vec())
            } else {
                None
            }
        }
        Arm::PerToken => {
            let out = gemm_per_token(ctx, w, x).expect("per-token gemm");
            Some(out.to_host(ctx).expect("d2h")[..m].to_vec())
        }
    }
}

/// Run one arm across the N-sweep; return (maxΔ across N, samples that ran).
#[allow(clippy::many_single_char_names)]
fn run_arm(
    ctx: &DeviceContext,
    arm: Arm,
    w: &DeviceMatrix,
    xseed: u64,
    m: usize,
    k: usize,
) -> (f32, usize) {
    let mut reference: Option<Vec<f32>> = None;
    let mut maxd = 0.0f32;
    let mut ran = 0usize;
    for &n in NS {
        let x = HiddenStates::from_host(ctx, &x_data(xseed, k, n), k, n).expect("X h2d");
        if let Some(c0) = col0(ctx, arm, w, &x, m) {
            ran += 1;
            match &reference {
                None => reference = Some(c0),
                Some(r) => {
                    maxd = c0
                        .iter()
                        .zip(r)
                        .map(|(x, y)| (x - y).abs())
                        .fold(maxd, f32::max);
                }
            }
        }
    }
    (maxd, ran)
}

#[allow(clippy::float_cmp)]
#[test]
fn batch_invariance_gemm_gate() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping batch_invariance_gemm: no CUDA device — gate inconclusive");
        return;
    };
    let reps = reps();
    for _rep in 0..reps {
        let mut any_baseline_drift = false;

        for (i, &(name, m, k)) in SHAPES.iter().enumerate() {
            let wseed = 0x5717_0000 ^ i as u64;
            let xseed = 0x0414_A000 ^ i as u64;
            let w = DeviceMatrix::from_host(&ctx, &w_data(wseed, m, k), m, k).expect("W h2d");

            gemm_lt_pin_tune(m, REP_N, k).expect("pin tune");
            let (base_d, _) = run_arm(&ctx, Arm::Baseline, &w, xseed, m, k);
            let (pin_d, pin_ran) = run_arm(&ctx, Arm::Pin, &w, xseed, m, k);
            let (ora_d, _) = run_arm(&ctx, Arm::PerToken, &w, xseed, m, k);

            if base_d > 0.0 {
                any_baseline_drift = true;
            }
            assert_eq!(
                ora_d, 0.0,
                "shape {name}: per-token ORACLE drifted (maxΔ={ora_d:.2e}) — harness bug, not a fix result"
            );
            assert_eq!(
                pin_d, 0.0,
                "shape {name}: PIN is NOT batch-invariant (maxΔ={pin_d:.2e}) — fix candidate FAILED"
            );
            let decode_ns = NS.iter().filter(|&&n| n <= GEMM_LT_MAX_N).count();
            assert!(
                pin_ran >= decode_ns,
                "shape {name}: pin served only {pin_ran} of the sweep, < {decode_ns} decode buckets (N<={GEMM_LT_MAX_N}) — invariance not exercised"
            );
        }

        assert!(
            any_baseline_drift,
            "no baseline drift on any shape this rep — was not reproduced here, so the pin pass is vacuous"
        );
    }
}
