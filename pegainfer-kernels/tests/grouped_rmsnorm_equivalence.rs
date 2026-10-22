//! Gates for Qwen3.8-Flash-Next's grouped `(1 + w)` RMSNorm.
//!
//! The comparison is against an f64 CPU oracle that walks the group structure
//! itself and calls no kernel, so agreement is not a same-source comparison.
//! The degeneracy gate pins the arithmetic to the vendor batched norm when all
//! weight slices are equal; the geometry gate pins the wrapper's rejections.
//!
//! Manual gate: no CI row enables `qwen4`. Run on a GPU box with
//!
//! ```text
//! cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test grouped_rmsnorm_equivalence
//! ```
//!
//! `PEGAINFER_REQUIRE_GPU=1` turns a missing device into a failure rather than a
//! skip.

mod common;

use half::bf16;
use pegainfer_kernels::ops::rms_norm_batch_offset_into;
use pegainfer_kernels::ops::rms_norm_grouped_offset_batch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

/// Frozen Qwen3.8-Flash-Next text geometry
/// (`pegainfer-qwen4/tests/frozen_config.json`): `hc_count` 4 branches of
/// `hidden_size` 2560 make the 10240-wide hyper-connection stream, and
/// `rms_norm_eps` is 1e-6. The PLE norms use the same width and the same group
/// size, so one geometry covers both call sites.
const BRANCHES: usize = 4;
const BRANCH: usize = 2560;
const STREAM: usize = BRANCHES * BRANCH;
const EPS: f32 = 1e-6;
const TOKENS: usize = 3;
/// Rows in the flattened `[tokens, BRANCHES, BRANCH]` view.
const ROWS: usize = TOKENS * BRANCHES;

/// bf16 stores 7 fraction bits, so one step is `2^-7` of the magnitude at the
/// top of a binade and `2^-8` at the bottom. The device and the oracle both
/// round once from f32; the only differences are f32 vs f64 accumulation and
/// `rsqrt.approx`, which can flip a boundary value by a step or two. `2^-6` is
/// two to four bf16 steps and still far below the 4x-16x gain error a wrong
/// weight slice produces.
const RELATIVE_BOUND: f32 = 1.0 / 64.0;

/// Per-branch input scale (2^3, 2^0, 2^-3, 2^-6). Deliberately spread branch
/// energies are what make the *reduction* grouping observable: one shared
/// 10240-wide rms would be set by the loud branch alone, leaving the quiet ones
/// normalised by two orders of magnitude too much, rather than by a rounding.
const BRANCH_SCALE: [f32; BRANCHES] = [8.0, 1.0, 0.125, 0.015_625];
/// Per-branch weight gain (2^-2 .. 2^4), so a row served the wrong branch's
/// `(1+w)` slice is off by at least 4x.
const BRANCH_GAIN: [f32; BRANCHES] = [0.25, 1.0, 4.0, 16.0];

/// A deterministic input with one junk block per (token, branch): distinct per
/// element (so no two rows can coincide once the norm erases the magnitude) and
/// scaled by that branch's `BRANCH_SCALE` (a power of two, so the bf16 store of
/// the scaled value is exact and the oracle and the kernel see identical bytes).
fn stream_host(tokens: usize) -> Vec<bf16> {
    let mut out = Vec::with_capacity(tokens * STREAM);
    for block in 0..(tokens * BRANCHES) {
        let scale = BRANCH_SCALE[block % BRANCHES];
        out.extend(
            common::fill(0x5EED_0000 ^ block as u64, BRANCH)
                .into_iter()
                .map(|value| bf16::from_f32(value.to_f32() * scale)),
        );
    }
    out
}

/// `weight` is `[STREAM]`, indexed by the row's global column: the four 2560-wide
/// slices differ both in content and in gain, so they are not interchangeable.
fn weight_host() -> Vec<bf16> {
    (0..STREAM)
        .map(|column| {
            let profile = 0.5 + (column % 23) as f32 / 23.0;
            bf16::from_f32(BRANCH_GAIN[column / BRANCH] * profile)
        })
        .collect()
}

/// Independent CPU oracle for the grouped `(1+w)` norm: per (token, branch), a
/// mean of squares over that branch's own `BRANCH` columns, then `(1 + w)`
/// indexed by that branch's *global* column. Everything in f64, so the oracle's
/// own error sits orders of magnitude below the bf16 grid. It calls no kernel —
/// comparing the device against a second implementation of the same formula
/// would only prove the formula is self-consistent.
fn grouped_reference(x: &[bf16], weight: &[bf16], rows: usize, eps: f64) -> Vec<bf16> {
    assert_eq!(x.len(), rows * BRANCH, "input is not {rows}x{BRANCH}");
    assert_eq!(weight.len(), STREAM, "weight is not {STREAM}-wide");
    let mut out = vec![bf16::from_f32(0.0); x.len()];
    for row in 0..rows {
        let base = row * BRANCH;
        let sum_sq: f64 = (0..BRANCH)
            .map(|column| {
                let value = f64::from(x[base + column].to_f32());
                value * value
            })
            .sum();
        let rms_rcp = 1.0 / (sum_sq / BRANCH as f64 + eps).sqrt();
        for column in 0..BRANCH {
            let value = f64::from(x[base + column].to_f32())
                * rms_rcp
                * (1.0 + f64::from(weight[row % BRANCHES * BRANCH + column].to_f32()));
            out[base + column] = bf16::from_f32(value as f32);
        }
    }
    out
}

fn device_states(
    ctx: &DeviceContext,
    host: &[bf16],
    hidden_dim: usize,
    rows: usize,
) -> HiddenStates {
    assert_eq!(
        host.len(),
        hidden_dim * rows,
        "input is not {hidden_dim}x{rows}"
    );
    let mut states = HiddenStates::zeros(ctx, hidden_dim, rows).expect("hidden states");
    ctx.stream
        .memcpy_htod(host, &mut states.data)
        .expect("x up");
    states
}

/// The stream as the grouped entry point takes it: one row per token, `STREAM`
/// wide, split into `BRANCHES` groups.
fn grouped_states(ctx: &DeviceContext, host: &[bf16], tokens: usize) -> HiddenStates {
    device_states(ctx, host, STREAM, tokens)
}

/// The same bytes as the reshape takes them: one row per (token, branch),
/// `BRANCH` wide — the layout `grouped_reference` returns.
fn reshaped_states(ctx: &DeviceContext, host: &[bf16], tokens: usize) -> HiddenStates {
    device_states(ctx, host, BRANCH, tokens * BRANCHES)
}

fn device_weight(ctx: &DeviceContext, host: &[bf16]) -> DeviceVec {
    DeviceVec::from_host(ctx, host).expect("weight up")
}

fn read_back(ctx: &DeviceContext, states: &HiddenStates) -> Vec<bf16> {
    ctx.stream.clone_dtoh(&states.data).expect("D2H")
}

/// The element whose deviation exceeds its bound by the most, with the numbers
/// needed to report it. A negative excess means every element is inside its
/// bound. Non-finite values are rejected rather than ranked: a NaN compares false
/// against everything, so ranking one would let an all-NaN output through.
fn worst_excess(got: &[bf16], want: &[bf16], relative_bound: f32) -> (f32, usize, f32, f32) {
    assert_eq!(got.len(), want.len(), "compared runs differ in length");
    assert!(!got.is_empty(), "nothing to compare");
    let mut worst = (f32::NEG_INFINITY, 0usize, 0.0f32, 0.0f32);
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        let (got, want) = (got.to_f32(), want.to_f32());
        assert!(
            got.is_finite() && want.is_finite(),
            "element {index} is not finite: device {got}, oracle {want}"
        );
        let (deviation, magnitude) = ((got - want).abs(), want.abs());
        let excess = deviation - relative_bound * magnitude;
        if excess > worst.0 {
            worst = (excess, index, deviation, magnitude);
        }
    }
    worst
}

fn assert_within(what: &str, got: &[bf16], want: &[bf16], relative_bound: f32) {
    let (excess, index, deviation, magnitude) = worst_excess(got, want, relative_bound);
    assert!(
        excess <= 0.0,
        "{what}: element {index} deviates by {deviation} (oracle {magnitude}), which is \
         {excess} past the relative bound of 1/64 (2-4 bf16 steps)"
    );
}

fn assert_bitwise(what: &str, got: &[bf16], want: &[bf16]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: compared runs differ in length"
    );
    let mut diverged = 0usize;
    let mut first = String::new();
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        if got.to_bits() != want.to_bits() {
            if diverged == 0 {
                first = format!("element {index}: device {got:?}, oracle {want:?}");
            }
            diverged += 1;
        }
    }
    assert_eq!(
        diverged, 0,
        "{what}: {diverged} elements differ, first is {first}"
    );
}

/// The largest absolute deviation, and where it is — a diagnostic, not a bound:
/// the gate is `worst_excess`.
fn worst_deviation(got: &[bf16], want: &[bf16]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "compared runs differ in length");
    let mut worst = (0.0f32, 0usize);
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        let deviation = (got.to_f32() - want.to_f32()).abs();
        if deviation > worst.0 {
            worst = (deviation, index);
        }
    }
    worst
}

/// Largest `|reference|` in a run — the scale the deviation is reported against.
fn largest_magnitude(values: &[bf16]) -> f32 {
    values
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0f32, f32::max)
}

/// The grouped call against the independent oracle, at the frozen geometry and
/// the frozen `eps`, in both serving shapes: one token (the decode step, 4
/// blocks) and several (prefill, 4 x tokens blocks). `seq_len` reaches the kernel
/// only through the grid, so the two are worth a pass each.
#[test]
fn the_grouped_norm_matches_its_own_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let weight = weight_host();
    let device_weight = device_weight(&ctx, &weight);

    for tokens in [1, TOKENS] {
        let x = stream_host(tokens);
        let reference = grouped_reference(&x, &weight, tokens * BRANCHES, f64::from(EPS));
        let states = grouped_states(&ctx, &x, tokens);
        let mut out = HiddenStates::zeros(&ctx, STREAM, tokens).expect("grouped out");
        rms_norm_grouped_offset_batch_into(&ctx, &states, &device_weight, BRANCH, EPS, &mut out)
            .expect("grouped norm");
        let got = read_back(&ctx, &out);

        assert_within(
            &format!("grouped norm vs oracle ({tokens} token(s))"),
            &got,
            &reference,
            RELATIVE_BOUND,
        );
        let (largest, at) = worst_deviation(&got, &reference);
        eprintln!(
            "grouped norm vs oracle ({tokens} token(s)): {} elements, largest deviation \
             {largest} at element {at} (largest |oracle| {}), relative bound 1/64 (2-4 bf16 \
             steps)",
            got.len(),
            largest_magnitude(&reference),
        );
    }
}

/// The grouped entry point must not change the arithmetic of the call it
/// generalises: with all four weight slices equal, a 4-group call and the
/// vendor 1-group call have to agree bit for bit.
#[test]
fn the_grouped_norm_degenerates_to_the_vendor_batched_norm() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let x = stream_host(TOKENS);
    let weight = weight_host();
    let whole = grouped_states(&ctx, &x, TOKENS);
    let reshaped = reshaped_states(&ctx, &x, TOKENS);

    let pooled: Vec<bf16> = (0..STREAM).map(|column| weight[column % BRANCH]).collect();
    let pooled_weight = device_weight(&ctx, &pooled);
    let slice_zero = device_weight(&ctx, &weight[..BRANCH]);
    let mut pooled_out = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("pooled out");
    rms_norm_grouped_offset_batch_into(&ctx, &whole, &pooled_weight, BRANCH, EPS, &mut pooled_out)
        .expect("pooled grouped norm");
    let mut ungrouped_out = HiddenStates::zeros(&ctx, BRANCH, ROWS).expect("ungrouped out");
    rms_norm_batch_offset_into(&ctx, &reshaped, &slice_zero, EPS, &mut ungrouped_out)
        .expect("ungrouped norm");
    // Both runs read the same bytes row for row: the grouped call's row `r` is
    // the reshape's row `r`, and the pooled weight hands it `slice_zero`.
    assert_bitwise(
        "4 equal groups vs the ungrouped batched norm",
        &read_back(&ctx, &pooled_out),
        &read_back(&ctx, &ungrouped_out),
    );
}

/// Broken geometry must be an error, not a silent wrong-width launch: a group
/// size that does not divide the row, and a weight that is not the row's width.
#[test]
fn the_grouped_norm_rejects_broken_geometry() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let x = stream_host(TOKENS);
    let weight = weight_host();
    let whole = grouped_states(&ctx, &x, TOKENS);
    let full_weight = device_weight(&ctx, &weight);
    let slice_zero = device_weight(&ctx, &weight[..BRANCH]);

    let mut out = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("reject out");
    assert!(
        rms_norm_grouped_offset_batch_into(&ctx, &whole, &full_weight, STREAM - 1, EPS, &mut out)
            .is_err(),
        "a group size that does not divide the row must be rejected"
    );
    assert!(
        rms_norm_grouped_offset_batch_into(&ctx, &whole, &slice_zero, BRANCH, EPS, &mut out)
            .is_err(),
        "a 2560-wide weight with a 10240-wide row must be rejected"
    );
}
