//! Does an existing kernel express Qwen3.8-Flash-Next's grouped `(1+w)` RMSNorm?
//!
//! The hyper-connection block's stream is `hc_count` 4 branches of `hidden_size`
//! 2560 (10240 wide). The reference reduces each branch over its own 2560 columns
//! and *then* scales by `(1 + w)` with `w` indexed by the row's **global**
//! column, so branch `k` reads `w[k*2560 .. (k+1)*2560]`. The bring-up question
//! was whether `rms_norm_batched_offset_cuda` already covers that — "just view the
//! stream as `4 x tokens` rows of 2560" — and this file answers it:
//!
//! * the **reduction** is expressible that way (one CTA per 2560-wide row);
//! * the **weight is not**. `GemmaRMSNorm` dispatches `RMSNormKernel`, which
//!   loads `weight + elem` with no row term
//!   (`third_party/flashinfer/include/flashinfer/norm.cuh:123`), so every row
//!   reads `weight[0..2560)` and branches 1..3 would be scaled by branch 0's
//!   slice. Nothing in the kernel's parameter list can change that: `d` sets the
//!   reduction width *and* the weight span, and the row stride is not a weight
//!   index.
//!
//! [`rms_norm_grouped_offset_batch_into`] is the kernel that does express it.
//! Five gates below pin that claim, two of them negative controls:
//!
//! | gate | what it pins |
//! | --- | --- |
//! | `the_reshaped_batched_norm_scales_every_branch_with_branch_zero` | the reshape is right on branch 0, and serves the wrong 4x-16x weight gain on 1..3 |
//! | `one_full_width_reduction_cannot_express_the_branches` | the other reshape error (one 10240-wide reduction) collapses the quiet branches to ~0 |
//! | `the_grouped_norm_matches_its_own_reference` | the grouped call equals the oracle within a stated tolerance |
//! | `the_grouped_norm_is_bit_exact_on_a_dyadic_input` | bit-for-bit equality where the arithmetic allows it |
//! | `the_grouped_norm_degenerates_to_the_vendor_batched_norm` | the new kernel's arithmetic is the vendor kernel's, bit for bit |
//!
//! The two negative controls exercise the *existing* vendor entry point, never
//! [`rms_norm_grouped_offset_batch_into`]. They are the executable form of the
//! design decision that justifies a copied kernel (no reshape reaches the
//! grouped weight address), the calibration for `TOLERANCE` (the wrong calls
//! miss by 4x-16x, hundreds of ULP, not by a rounding), and the dedup tripwire:
//! if `RMSNormKernel` ever grows a group-aware weight address, the reshape gate
//! fails — the signal to retire `GroupedRMSNormKernel` in favour of the vendor
//! path rather than keep both.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! `PEGAINFER_REQUIRE_GPU=1`, which turns a missing device into a failure rather
//! than a skip.
//!
//! Reference strategy: every numeric comparison is against `grouped_reference`,
//! an f64 CPU oracle that walks the branch structure itself and calls no kernel,
//! so agreeing with it is not a same-source comparison. The negative controls
//! assert the *wrong* calls do not match and quote the deviation, so no gate can
//! pass because "nothing was compared" or because the tolerance swallowed it.

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
/// Rows in the flattened `[tokens, BRANCHES, BRANCH]` view — what the reshape
/// hypothesis would feed the existing batched entry point.
const ROWS: usize = TOKENS * BRANCHES;

/// bf16 carries 8 explicit mantissa bits, so one ULP is `2^-8` of a magnitude.
/// The device keeps the whole `x * rms_rcp * (1 + w)` product in f32 and rounds
/// once, exactly like the reference, so the only expected difference is the
/// reduction order (the same warp tree, but summed in f32 instead of f64) plus
/// `rsqrt.approx`: that moves the product by well under one bf16 ULP, which can
/// flip a value sitting on a rounding boundary into its neighbour. Four ULP
/// leaves room for that and still sits far below what the negative controls have
/// to catch — a wrong `(1+w)` slice is a 4x-16x gain error, hundreds of ULP.
const ULP: f32 = 1.0 / 256.0;
const TOLERANCE: f32 = 4.0 * ULP;

/// Per-branch input scale (2^3, 2^0, 2^-3, 2^-6). Deliberately spread branch
/// energies are what make the *reduction* grouping observable: one shared
/// 10240-wide rms would be set by the loud branch alone, leaving the quiet ones
/// normalised by two orders of magnitude too much, rather than by a rounding.
const BRANCH_SCALE: [f32; BRANCHES] = [8.0, 1.0, 0.125, 0.015625];
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

/// The same bytes as the reshape hypothesis takes them: one row per
/// (token, branch), `BRANCH` wide — the layout `grouped_reference` returns.
fn reshaped_states(ctx: &DeviceContext, host: &[bf16], tokens: usize) -> HiddenStates {
    device_states(ctx, host, BRANCH, tokens * BRANCHES)
}

fn device_weight(ctx: &DeviceContext, host: &[bf16]) -> DeviceVec {
    DeviceVec::from_host(ctx, host).expect("weight up")
}

fn read_back(ctx: &DeviceContext, states: &HiddenStates) -> Vec<bf16> {
    ctx.stream.clone_dtoh(&states.data).expect("D2H")
}

/// The rows that belong to one `BRANCHES`-wide branch, concatenated: element `n`
/// of the result is column `n % BRANCH` of branch row `n / BRANCH`.
fn branch_rows(values: &[bf16], branches: &[usize]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(branches.len() * BRANCH);
    for row in branches {
        out.extend_from_slice(&values[row * BRANCH..(row + 1) * BRANCH]);
    }
    out
}

/// The element whose deviation exceeds its bound by the most, with the numbers
/// needed to report it. A negative excess means every element is inside its
/// bound. Non-finite values are rejected rather than ranked: a NaN compares false
/// against everything, so ranking one would let an all-NaN output through.
fn worst_excess(got: &[bf16], want: &[bf16], tolerance: f32) -> (f32, usize, f32, f32) {
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
        let excess = deviation - tolerance * magnitude;
        if excess > worst.0 {
            worst = (excess, index, deviation, magnitude);
        }
    }
    worst
}

fn assert_within(what: &str, got: &[bf16], want: &[bf16], tolerance: f32) {
    let (excess, index, deviation, magnitude) = worst_excess(got, want, tolerance);
    assert!(
        excess <= 0.0,
        "{what}: element {index} deviates by {deviation} (oracle {magnitude}), which is \
         {excess} past the bound of {} x bf16 ULP",
        tolerance / ULP
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

/// The largest absolute deviation, and where it is — the metric a negative
/// control is judged on, since "not bitwise equal" would be satisfied by a
/// one-ULP wobble that proves nothing.
fn worst_deviation(got: &[bf16], want: &[bf16]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "compared runs differ in length");
    let mut worst = (0.0f32, 0usize);
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        let deviation = (got.to_f32() - want.to_f32()).abs();
        assert!(
            deviation.is_finite(),
            "element {index}: device {got:?} against oracle {want:?}"
        );
        if deviation > worst.0 {
            worst = (deviation, index);
        }
    }
    worst
}

/// Largest `|reference|` in a run — the scale a deviation has to be judged
/// against. A control is only meaningful if it is wrong by a large fraction of
/// this, not by a rounding.
fn largest_magnitude(values: &[bf16]) -> f32 {
    values
        .iter()
        .map(|v| v.to_f32().abs())
        .fold(0.0f32, f32::max)
}

/// A deviation this large a fraction of the run's largest reference value cannot
/// be a rounding artefact, so a control that stays under it is not testing a
/// wrong grouping at all.
const DECISIVE: f32 = 0.1;

/// The bring-up hypothesis was "no new kernel: view the 10240-wide stream as
/// `4 x tokens` rows of 2560". That gets the reduction right and the `(1+w)`
/// weight wrong — the vendor kernel hands every one of those rows
/// `weight[0..2560)`. This gate pins both halves of that claim: branch 0, which
/// is the slice the reshape does serve, matches the oracle; branches 1..3 get the
/// wrong gain (4x-16x) and miss the oracle by the largest magnitude in the run.
/// This drives the vendor entry point only; it is a negative control for the
/// design decision, not a test of the new kernel.
#[test]
fn the_reshaped_batched_norm_scales_every_branch_with_branch_zero() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let x = stream_host(TOKENS);
    let weight = weight_host();
    let reference = grouped_reference(&x, &weight, ROWS, f64::from(EPS));

    let reshaped = reshaped_states(&ctx, &x, TOKENS);
    let slice_zero = device_weight(&ctx, &weight[..BRANCH]);
    let mut reshaped_out = HiddenStates::zeros(&ctx, BRANCH, ROWS).expect("reshaped out");
    rms_norm_batch_offset_into(&ctx, &reshaped, &slice_zero, EPS, &mut reshaped_out)
        .expect("reshaped norm");
    let got = read_back(&ctx, &reshaped_out);

    let first: Vec<usize> = (0..ROWS)
        .filter(|row| row.is_multiple_of(BRANCHES))
        .collect();
    let rest: Vec<usize> = (0..ROWS)
        .filter(|row| !row.is_multiple_of(BRANCHES))
        .collect();
    assert_within(
        "reshaped call, branch 0 (the slice every row is served)",
        &branch_rows(&got, &first),
        &branch_rows(&reference, &first),
        TOLERANCE,
    );

    let (deviation, index) =
        worst_deviation(&branch_rows(&got, &rest), &branch_rows(&reference, &rest));
    let scale = largest_magnitude(&branch_rows(&reference, &rest));
    assert!(
        deviation > DECISIVE * scale,
        "the reshaped call must be WRONG on branches 1..3 — that is the finding this gate \
         exists to prove — but its worst deviation is only {deviation} (element {index}) \
         against a largest reference magnitude of {scale}"
    );
    eprintln!(
        "reshaped call, branches 1..3: worst deviation {deviation} at element {index}, \
         largest |reference| {scale} ({:.1}x), while the tolerance is {} bf16 ULP = {:.4}x",
        deviation / scale,
        TOLERANCE / ULP,
        TOLERANCE
    );
}

/// The other way to "just reshape": one 10240-wide reduction per token with the
/// full weight. The weight is then right and the rms is shared across branches
/// with deliberately different energies, so each branch is scaled by the other
/// branches' energy — the reduction side of the same claim. What that does to the
/// output depends on the data (here the quiet branches collapse towards zero
/// instead of reaching the oracle's magnitude); the gate is the deviation against
/// the run's largest magnitude, not any particular ratio. Like its sibling, it
/// drives the vendor entry point only and is a negative control for the design
/// decision, not a test of the new kernel.
#[test]
fn one_full_width_reduction_cannot_express_the_branches() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let x = stream_host(TOKENS);
    let weight = weight_host();
    let reference = grouped_reference(&x, &weight, ROWS, f64::from(EPS));

    let mut out = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("full-width out");
    let states = device_states(&ctx, &x, STREAM, TOKENS);
    let full_weight = device_weight(&ctx, &weight);
    rms_norm_batch_offset_into(&ctx, &states, &full_weight, EPS, &mut out)
        .expect("full-width norm");
    let got = read_back(&ctx, &out);

    let (deviation, index) = worst_deviation(&got, &reference);
    let scale = largest_magnitude(&reference);
    assert!(
        deviation > DECISIVE * scale,
        "a single 10240-wide reduction must not reproduce four 2560-wide ones, but its worst \
         deviation is only {deviation} (element {index}) against a largest reference \
         magnitude of {scale}"
    );
    eprintln!(
        "full-width reduction: worst deviation {deviation} at element {index}, largest \
         |reference| {scale} ({:.1}x)",
        deviation / scale
    );
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

        let (excess, index, deviation, magnitude) = worst_excess(&got, &reference, TOLERANCE);
        assert!(
            excess <= 0.0,
            "grouped norm vs oracle ({tokens} token(s)): element {index} deviates by \
             {deviation} (oracle {magnitude}), which is {excess} past the bound of {} x \
             bf16 ULP",
            TOLERANCE / ULP
        );
        let (largest, at) = worst_deviation(&got, &reference);
        eprintln!(
            "grouped norm vs oracle ({tokens} token(s)): {} elements, largest deviation \
             {largest} at element {at} (largest |oracle| {}), worst margin excess {excess} \
             at element {index} against {} bf16 ULP",
            got.len(),
            largest_magnitude(&reference),
            TOLERANCE / ULP
        );
    }
}

/// Bit-for-bit equality, on the input family where the arithmetic allows it.
///
/// Per branch, every element is `±k * 2^a` with `k in {1, 3}` and exactly 960
/// threes, so the branch's sum of squares is `2560 * 4 * 2^(2a)`: every partial
/// sum is an exact multiple of `2^(2a)` under any summation order, the f32
/// division by 2560 and the `rsqrt` of the resulting power of four are exact, and
/// `2^a * 2^(-a-1)` is an exact power of two — leaving `k/2 * (1+w)` with `(1+w)`
/// a power of two, which bf16 holds exactly. The final rounding is therefore the
/// identity and the device output can be compared to the oracle bit for bit.
///
/// `eps` has to be zero for this: a nonzero eps takes the rsqrt argument off the
/// power-of-two lattice, so no construction makes it exact. The eps path is
/// covered by the tolerance gate above, which runs the frozen 1e-6.
#[test]
fn the_grouped_norm_is_bit_exact_on_a_dyadic_input() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // >4 octaves of branch scale, and `(1+w)` alternating between 2^m and
    // 2^(m+1) inside each branch, so neither the branch index nor the column
    // index can be permuted unnoticed.
    const BRANCH_EXPONENT: [i32; BRANCHES] = [0, 1, -2, 3];
    const WEIGHT_EXPONENT: [i32; BRANCHES] = [0, 1, -1, 2];

    let mut x = Vec::with_capacity(ROWS * BRANCH);
    let mut weight = vec![bf16::from_f32(0.0); STREAM];
    for row in 0..ROWS {
        let branch = row % BRANCHES;
        let exponent = BRANCH_EXPONENT[branch];
        for column in 0..BRANCH {
            // 3 of every 8 columns carry 3 instead of 1: 960 threes, so the sum
            // of squares is 960*9 + 1600*1 = 10240 = 4 * 2560.
            let magnitude = if column % 8 < 3 { 3.0f32 } else { 1.0f32 };
            let sign = if (column / 8).is_multiple_of(2) {
                1.0f32
            } else {
                -1.0f32
            };
            x.push(bf16::from_f32(sign * magnitude * 2f32.powi(exponent)));
        }
    }
    for column in 0..STREAM {
        let branch = column / BRANCH;
        let exponent = WEIGHT_EXPONENT[branch] + (column % 2) as i32;
        // A power of two minus one is exact in bf16, and `1 + w` is then exactly
        // `2^exponent`.
        weight[column] = bf16::from_f32(2f32.powi(exponent) - 1.0);
    }

    let oracle = grouped_reference(&x, &weight, ROWS, 0.0);
    // The construction's closed form, derived independently of the oracle's
    // iterative reduction: `|x| * rms_rcp = k/2`, so the output is `±k/2 * 2^m`.
    // If this disagrees with the oracle, the *construction* is not the exact one
    // the doc comment claims and the bitwise comparison below would be vacuous.
    let closed_form: Vec<bf16> = (0..ROWS * BRANCH)
        .map(|index| {
            let column = index % BRANCH;
            let branch = (index / BRANCH) % BRANCHES;
            let magnitude = if column % 8 < 3 { 3.0f32 } else { 1.0f32 };
            let sign = if (column / 8).is_multiple_of(2) {
                1.0f32
            } else {
                -1.0f32
            };
            let exponent = WEIGHT_EXPONENT[branch] + (column % 2) as i32 - 1;
            bf16::from_f32(sign * magnitude * 2f32.powi(exponent))
        })
        .collect();
    assert_bitwise(
        "dyadic construction: oracle vs closed form",
        &oracle,
        &closed_form,
    );
    // Guard the construction itself, not just one adjacent pair: an input that
    // collapsed to a single value would make the bitwise comparison below
    // vacuous, so require both a wide spread and many distinct outputs.
    let distinct = {
        let mut bits: Vec<u16> = closed_form.iter().map(|value| value.to_bits()).collect();
        bits.sort_unstable();
        bits.dedup();
        bits.len()
    };
    let spread = largest_magnitude(&oracle);
    assert!(
        spread > 4.0 && distinct >= 8,
        "the dyadic input must produce a spread, non-degenerate output: largest magnitude \
         {spread}, {distinct} distinct values"
    );

    let states = grouped_states(&ctx, &x, TOKENS);
    let device_weight = device_weight(&ctx, &weight);
    let mut out = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("dyadic out");
    rms_norm_grouped_offset_batch_into(&ctx, &states, &device_weight, BRANCH, 0.0, &mut out)
        .expect("grouped norm");
    let got = read_back(&ctx, &out);

    assert_bitwise("grouped norm on a dyadic input", &got, &oracle);
    eprintln!(
        "grouped norm on a dyadic input: {} elements, all bit-identical to the oracle",
        got.len()
    );
}

/// The grouped entry point must not change the arithmetic of the call it
/// generalises, or the two could disagree on any geometry they share. With all
/// four weight slices equal, a 4-group call and the vendor 1-group call have to
/// agree bit for bit; so do `group_size == hidden_dim` (which the entry point
/// hands straight to the vendor template) and the ungrouped entry point.
#[test]
fn the_grouped_norm_degenerates_to_the_vendor_batched_norm() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let x = stream_host(TOKENS);
    let weight = weight_host();
    let whole = grouped_states(&ctx, &x, TOKENS);
    let reshaped = reshaped_states(&ctx, &x, TOKENS);
    let full_weight = device_weight(&ctx, &weight);

    // (a) Four equal slices: the grouping must then be inert.
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

    // (b) `group_size == hidden_dim` is the ungrouped case on the whole stream,
    // and the entry point hands it straight to the vendor template.
    let mut single_group = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("single group out");
    rms_norm_grouped_offset_batch_into(&ctx, &whole, &full_weight, STREAM, EPS, &mut single_group)
        .expect("single-group norm");
    let mut ungrouped_stream = HiddenStates::zeros(&ctx, STREAM, TOKENS).expect("stream out");
    rms_norm_batch_offset_into(&ctx, &whole, &full_weight, EPS, &mut ungrouped_stream)
        .expect("ungrouped stream norm");
    assert_bitwise(
        "group_size == hidden_dim vs the ungrouped batched norm",
        &read_back(&ctx, &single_group),
        &read_back(&ctx, &ungrouped_stream),
    );

    // (c) Broken geometry must be an error, not a silent wrong-width launch: a
    // group size that does not divide the row, and a weight that is not the
    // row's width.
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

/// Optional corroboration against the reference implementation's own fixture
/// (`test_data/qwen38fn-hc-golden.safetensors`, PR #1153, not in `main`). It
/// stores the module's `hc_norm` weight, two `hyper_input` streams and the
/// `normed` outputs the reference produced for them, so this gate replays real
/// reference bytes rather than a formula either side can agree on by accident.
///
/// Point `PEGAINFER_QWEN38FN_HC_GOLDEN` at the fixture to run it; nothing else in
/// this file depends on it, and the gate skips (rather than fails) without it.
#[test]
#[ignore = "optional: needs test_data/qwen38fn-hc-golden.safetensors from PR #1153"]
fn the_grouped_norm_reproduces_the_stored_reference() {
    let Ok(path) = std::env::var("PEGAINFER_QWEN38FN_HC_GOLDEN") else {
        eprintln!("skipping: PEGAINFER_QWEN38FN_HC_GOLDEN is not set");
        return;
    };
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(&path).expect("read the hc golden");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("parse the hc golden");

    let (weight, weight_shape) = bf16_tensor(&tensors, "hc_norm");
    assert_eq!(weight_shape, vec![STREAM], "hc_norm shape");
    let weight_device = device_weight(&ctx, &weight);
    // The first slice is what the reshape hypothesis would serve to every row.
    let slice_zero = device_weight(&ctx, &weight[..BRANCH]);

    for case in 0..2 {
        let (input, input_shape) = bf16_tensor(&tensors, &format!("case{case}_hyper_input"));
        let (expected, expected_shape) = bf16_tensor(&tensors, &format!("case{case}_normed"));
        assert_eq!(input_shape, expected_shape, "case {case} shapes");
        assert_eq!(
            input_shape.len(),
            3,
            "case {case} is not a [1, tokens, {STREAM}] stream"
        );
        let tokens = input_shape[1];
        let rows = tokens * BRANCHES;

        let states = grouped_states(&ctx, &input, tokens);
        let mut grouped = HiddenStates::zeros(&ctx, STREAM, tokens).expect("golden out");
        rms_norm_grouped_offset_batch_into(
            &ctx,
            &states,
            &weight_device,
            BRANCH,
            EPS,
            &mut grouped,
        )
        .expect("grouped norm");
        let got = read_back(&ctx, &grouped);
        assert_within(
            &format!("stored hc golden case {case} ({tokens} tokens), grouped norm"),
            &got,
            &expected,
            TOLERANCE,
        );
        let (stored_deviation, stored_at) = worst_deviation(&got, &expected);

        let reshaped = reshaped_states(&ctx, &input, tokens);
        let mut reshaped_out = HiddenStates::zeros(&ctx, BRANCH, rows).expect("reshaped out");
        rms_norm_batch_offset_into(&ctx, &reshaped, &slice_zero, EPS, &mut reshaped_out)
            .expect("reshaped norm");
        let (deviation, index) = worst_deviation(&read_back(&ctx, &reshaped_out), &expected);
        let scale = largest_magnitude(&expected);
        assert!(
            deviation > DECISIVE * scale,
            "stored hc golden case {case}: the reshaped call was expected to miss the stored \
             reference by a wide margin, but its worst deviation is {deviation} (element \
             {index}) against a largest stored magnitude of {scale}"
        );
        eprintln!(
            "stored hc golden case {case} ({tokens} tokens): grouped norm off by \
             {stored_deviation} at element {stored_at} (largest stored magnitude {scale}), \
             against the {} bf16 ULP bound of {:.4}; reshaped call off by {deviation} ({:.1}x)",
            TOLERANCE / ULP,
            TOLERANCE * scale,
            deviation / scale
        );
    }
}

/// A bf16 tensor from a safetensors fixture, as `(values, shape)`.
fn bf16_tensor(tensors: &safetensors::SafeTensors<'_>, name: &str) -> (Vec<bf16>, Vec<usize>) {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("{name}: {err}"));
    assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name} is not bf16");
    let values = view
        .data()
        .chunks_exact(2)
        .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    (values, view.shape().to_vec())
}
