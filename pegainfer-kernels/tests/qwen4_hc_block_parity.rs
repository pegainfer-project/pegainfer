//! Qwen3.8-Flash-Next hyper-connection gated residual replayed against the
//! frozen reference golden.
//!
//! `test_data/qwen38fn-hc-golden.safetensors` was dumped from transformers
//! 5.16.1 `qwen4_exp` (`Qwen4ExpTextGatedResidual`) by
//! `tools/accuracy/dump_qwen38fn_hc_golden.py`, at the real frozen-checkpoint
//! geometry — `hc_count` 4, `hidden_size` 2560 (a 10240-wide stream) and
//! `hc_lowrank` 320. Nothing is shrunk, because the two low-rank mix
//! projections *are* the component under test.
//!
//! Each case's `hyper_input` goes through the shared GPU surface — the grouped
//! `(1 + w)` RMSNorm as four `rms_norm_batch_offset_into` calls on the real
//! per-branch weight slice, the three projections through the cuBLAS wrapper,
//! and the elementwise tail through the `qwen4` kernels — and every tensor
//! the reference recorded is compared:
//!
//! * `normed` — the grouped norm's output, the block's only norm;
//! * `mixed` — the rank-320 mix gate's branch mean (`use_combine=True`);
//! * `inject` — the per-branch injection weights, `2 * sigmoid(. / hc_count)`;
//! * `mixed_only` — the same module re-read as the top-level readout
//!   (`use_combine=False`), which carries no injection head.
//!
//! Ungated: the scatter half of the residual block, `mixed + inject * sublayer`
//! folded back into the 4-branch stream. It needs a sublayer output, which the
//! golden does not carry — the dump records this gate's own inputs and outputs,
//! not a layer's. `inject` is checked here, so the scatter's only per-branch
//! coefficient is already held; the fold itself waits on a golden that carries
//! a sublayer.
//!
//! LAYOUT. `rms_norm_batched_offset_cuda` pins `stride_input` to `hidden_dim`,
//! so a branch of a row-major 10240-wide stream is not addressable as `[S,
//! 2560]`. The gate therefore feeds the norm one branch-major `[S, 2560]`
//! buffer per branch — four launches, each with that branch's own slice of the
//! norm weight — and assembles the row-major stream the projections read back
//! on device with plain `memcpy_dtod`. If/when PR #1158 lands
//! `rms_norm_grouped_offset_cuda`, production switches to that single-launch
//! grouped entry point and the assembly disappears with it; this gate's
//! four-call form is bit-identical on that degenerate path, so the numbers
//! recorded here do not move when the switch happens.
//!
//! Provenance is asserted, not assumed: the fixture records the reference
//! version and the sha256 of the frozen config it was dumped from, and that
//! config's hyper-connection geometry is held against the geometry this gate
//! replays. A fixture from another checkpoint or another semantics fails here
//! instead of being skipped — and the provenance test needs no device, so a
//! missing GPU cannot skip it either.
//!
//! The replay tests skip without a device; `PEGAINFER_REQUIRE_GPU=1` turns that
//! into a failure.
//!
//! ```text
//! cargo test --release --locked -p pegainfer-kernels --features qwen4 \
//!   --test qwen4_hc_block_parity -- --nocapture
//! ```

#![cfg(feature = "qwen4")]

mod common;

use std::fmt::Write as _;

use anyhow::Result;
use anyhow::anyhow;
use half::bf16;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::hc_mul_branch_combine_into;
use pegainfer_kernels::ops::hc_sigmoid_affine_into;
use pegainfer_kernels::ops::hc_silu_scaled_into;
use pegainfer_kernels::ops::rms_norm_batch_into;
use pegainfer_kernels::ops::rms_norm_batch_offset_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-hc-golden.safetensors";
const METADATA_KEY: &str = "qwen4_hc_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";

/// `hc_count`: branches in the hyper-connection stream.
const HC_COUNT: usize = 4;
/// `hidden_size`: features per branch, and the grouped norm's group size.
const HIDDEN: usize = 2560;
/// `hc_lowrank`: the mix gate's bottleneck.
const LOWRANK: usize = 320;
/// `hc_count * hidden_size`: the stream width.
const STREAM: usize = HC_COUNT * HIDDEN;
/// `rms_norm_eps`, held bit-for-bit against the frozen config.
const RMS_EPS: f32 = 1.0e-6;
/// The two recorded sequence lengths.
const CASES: [usize; 2] = [6, 10];

/// Parity bound, `abs(got - want) <= ATOL + RTOL * abs(want)`.
///
/// Both sides are bf16 at every boundary, so the unit of disagreement is one
/// bf16 ulp of the value compared: 0.4% of it. The reference rounds once per
/// submodule and this replay rounds at the same points, so the freedom left is
/// only *which side* of a bf16 midpoint a value lands on. Two reductions can
/// move one: the grouped norm's f32 sum of squares runs in FlashInfer's vector
/// order where torch runs in its own, and cuBLAS accumulates the three
/// projections in a different order than the torch 2.14.1+cpu bf16 matmul the
/// fixture was dumped with. Either shifts an f32 result by a few ulps, which
/// flips the bf16 cast only for a value already within those ulps of a
/// midpoint.
///
/// Measured worst element per tensor over both cases (RTX 5060 sm_120, CUDA
/// 13.2): `normed` 9.77e-4 — case 0's 61440 elements are bit-identical —
/// `mixed` 9.77e-3, `inject` 7.81e-3, `mixed_only` 9.77e-3. At the magnitudes
/// involved that is one to two bf16 ulps, i.e. one flipped rounding in a chain
/// of seven (norm, down GEMM, SiLU, up GEMM, sigmoid, branch product, branch
/// mean). Re-measured after `silu_f32` moved to the tree's
/// `x / (1 + expf(-x))` form: no number above changed.
///
/// The margin, in the arithmetic those numbers support: the worst element any
/// of the four tensors shows is 9.77e-3. `ATOL` alone is 4.0e-2 — 4.1x it
/// (4.0e-2 / 9.77e-3) — and at `|want| = 1` the whole bound `ATOL + RTOL *
/// |want|` is 8.0e-2, 8.2x it. Below `|want| = 1` the `RTOL` term shrinks
/// towards `ATOL`, so the ratio never falls under 4.1; above it, the ratio only
/// grows. Four-to-eight times the disagreement actually observed leaves room
/// for a machine whose cuBLAS picks a different accumulation order and flips
/// once more at every stage — and, per the controls below, no room for a wrong
/// block.
///
/// The negative controls are what pin that down. The tightest, a sigmoid where
/// the reference puts the SiLU on the down output, moves `mixed` by 4.96e-1 at
/// its worst and puts 5676 of 15360 elements — 37% — outside the bound; the
/// coarsest, plain `w` for `(1 + w)`, moves `normed` by 4.84 and puts 98934 of
/// 102400 outside. Every control fails on a large fraction of its elements
/// rather than on a marginal few, so the bound discriminates rather than
/// absorbs.
const ATOL: f32 = 4.0e-2;
const RTOL: f32 = 4.0e-2;

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn config_path() -> String {
    format!("{}/../{CONFIG}", env!("CARGO_MANIFEST_DIR"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

// ── provenance ────────────────────────────────────────────────────────────

/// The fixture's own record of what it was dumped from, held against the
/// reference version, the frozen config's bytes and the hyper-connection
/// geometry this gate replays. Returns the manifest.
fn assert_provenance(bytes: &[u8]) -> serde_json::Value {
    let (_, header) = SafeTensors::read_metadata(bytes).expect("fixture metadata");
    let map = header
        .metadata()
        .as_ref()
        .expect("fixture carries a metadata map");
    let recorded = map
        .get(METADATA_KEY)
        .unwrap_or_else(|| panic!("fixture metadata has no `{METADATA_KEY}` key"))
        .clone();
    let manifest: serde_json::Value = serde_json::from_str(&recorded)
        .unwrap_or_else(|e| panic!("`{METADATA_KEY}` does not parse as JSON: {e}"));

    let version = manifest["format_version"]
        .as_u64()
        .expect("manifest records format_version");
    assert_eq!(
        version, FORMAT_VERSION,
        "fixture format_version {version} is not the {FORMAT_VERSION} this \
         gate reads; regenerate with tools/accuracy/dump_qwen38fn_hc_golden.py"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG),
        "fixture was dumped from a config other than {CONFIG}"
    );
    assert_eq!(
        manifest["transformers_version"].as_str(),
        Some(REF_TRANSFORMERS),
        "fixture was dumped from a different transformers; the gated-residual \
         semantics are frozen against {REF_TRANSFORMERS}"
    );

    let config_bytes = std::fs::read(config_path()).expect("frozen config readable");
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(sha256_hex(&config_bytes).as_str()),
        "fixture's config_sha256 does not match {CONFIG}; the golden belongs \
         to another checkpoint"
    );

    // The geometry is read from the same bytes the sha256 just pinned, so a
    // config that moved under the fixture cannot pass on the hash alone.
    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).expect("frozen config is JSON");
    let text = &config["text_config"];
    assert_eq!(
        text["hc_count"].as_u64(),
        Some(HC_COUNT as u64),
        "frozen config hc_count is not the {HC_COUNT} this gate replays"
    );
    assert_eq!(
        text["hidden_size"].as_u64(),
        Some(HIDDEN as u64),
        "frozen config hidden_size is not the {HIDDEN} this gate replays"
    );
    assert_eq!(
        text["hc_lowrank"].as_u64(),
        Some(LOWRANK as u64),
        "frozen config hc_lowrank is not the {LOWRANK} this gate replays"
    );
    let eps = text["rms_norm_eps"]
        .as_f64()
        .expect("frozen config rms_norm_eps");
    assert_eq!(
        (eps as f32).to_bits(),
        RMS_EPS.to_bits(),
        "frozen config rms_norm_eps {eps} is not the {RMS_EPS} this gate passes"
    );
    manifest
}

#[test]
fn fixture_provenance_is_the_frozen_config() {
    let bytes = std::fs::read(fixture_path()).expect("golden fixture readable");
    let manifest = assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    for name in [
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "mixer_norm",
        "mixer_down",
        "mixer_up",
    ] {
        assert!(
            fixture.tensor(name).is_ok(),
            "fixture is missing the `{name}` weight"
        );
    }
    println!(
        "provenance ok: transformers {} / torch {} / {} / seed {}",
        manifest["transformers_version"].as_str().unwrap_or("?"),
        manifest["torch_version"].as_str().unwrap_or("?"),
        manifest["platform"].as_str().unwrap_or("?"),
        manifest["seed"].as_u64().unwrap_or(0),
    );
}

// ── fixture access ────────────────────────────────────────────────────────

/// A stored bf16 tensor, held against the shape this gate derives from the
/// frozen config rather than against whatever the file says. Copied out element
/// by element, so the comparison never depends on the fixture buffer's
/// alignment.
fn golden_bf16(fixture: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Vec<bf16> {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture tensor `{name}`: {e}"));
    assert_eq!(view.dtype(), Dtype::BF16, "`{name}` dtype");
    assert_eq!(view.shape(), shape, "`{name}` shape");
    view.data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_le_bytes(*chunk))
        .collect()
}

fn golden_seq_len(fixture: &SafeTensors<'_>, name: &str, want: usize) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture tensor `{name}`: {e}"));
    assert_eq!(view.dtype(), Dtype::I32, "`{name}` dtype");
    assert_eq!(view.shape(), [1], "`{name}` shape");
    let got = i32::from_le_bytes(view.data().as_chunks::<4>().0[0]);
    assert_eq!(
        usize::try_from(got).expect("sequence length is positive"),
        want,
        "`{name}` records a sequence length this gate does not replay"
    );
}

/// Split a row-major `[rows, HC_COUNT * HIDDEN]` stream into one
/// `[rows, HIDDEN]` block per branch: what the grouped norm reads, since it
/// cannot stride a branch out of the stream itself.
fn branch_major(stream: &[bf16], rows: usize) -> [Vec<bf16>; HC_COUNT] {
    std::array::from_fn(|branch| {
        (0..rows)
            .flat_map(|row| {
                let at = row * STREAM + branch * HIDDEN;
                stream[at..at + HIDDEN].iter().copied()
            })
            .collect()
    })
}

fn f32_of(values: &[bf16]) -> Vec<f32> {
    values.iter().copied().map(bf16::to_f32).collect()
}

// ── the block ─────────────────────────────────────────────────────────────

/// One `Qwen4ExpTextGatedResidual` instance: the grouped norm's weight split
/// per branch, the two low-rank mix projections, and — only for
/// `use_combine=True` — the per-branch injection head.
struct GatedResidual {
    norm_branches: [DeviceVec; HC_COUNT],
    down: DeviceMatrix,
    up: DeviceMatrix,
    inject: Option<DeviceMatrix>,
}

impl GatedResidual {
    fn load(
        ctx: &DeviceContext,
        fixture: &SafeTensors<'_>,
        norm: &str,
        down: &str,
        up: &str,
        inject: Option<&str>,
    ) -> Result<Self> {
        let weight = golden_bf16(fixture, norm, &[STREAM]);
        Ok(Self {
            norm_branches: std::array::from_fn(|branch| {
                DeviceVec::from_host(ctx, &weight[branch * HIDDEN..(branch + 1) * HIDDEN])
                    .expect("upload norm weight branch")
            }),
            down: DeviceMatrix::from_host(
                ctx,
                &golden_bf16(fixture, down, &[LOWRANK, STREAM]),
                LOWRANK,
                STREAM,
            )?,
            up: DeviceMatrix::from_host(
                ctx,
                &golden_bf16(fixture, up, &[STREAM, LOWRANK]),
                STREAM,
                LOWRANK,
            )?,
            inject: inject
                .map(|name| {
                    DeviceMatrix::from_host(
                        ctx,
                        &golden_bf16(fixture, name, &[HC_COUNT, STREAM]),
                        HC_COUNT,
                        STREAM,
                    )
                })
                .transpose()?,
        })
    }
}

/// Which activation a mix stage runs. The reference puts SiLU on the down
/// projection's output and sigmoid on the up projection's; every other
/// combination is a negative control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Silu,
    Sigmoid,
}

/// One knob per way the block can be built wrong. [`Variant::REFERENCE`] is the
/// reference; each negative control flips exactly one field and names itself.
#[derive(Clone, Copy, Debug)]
struct Variant {
    label: &'static str,
    /// `(1 + w)` against plain `w` in the grouped norm.
    offset_norm: bool,
    /// The `/ hc_count` both the mix path and the injection weights carry.
    hc_divisor: f32,
    down_act: Act,
    gate_act: Act,
    /// `.mean(-2)` against a sum over branches.
    combine_scale: f32,
}

impl Variant {
    const REFERENCE: Self = Self {
        label: "reference",
        offset_norm: true,
        hc_divisor: 1.0 / HC_COUNT as f32,
        down_act: Act::Silu,
        gate_act: Act::Sigmoid,
        combine_scale: 1.0 / HC_COUNT as f32,
    };
}

/// The grouped `(1 + w)` RMSNorm: one launch per branch over that branch's
/// `[rows, HIDDEN]` buffer, with that branch's own `HIDDEN`-wide slice of the
/// norm weight. Same arithmetic one strided or grouped launch would do — each
/// call is one CTA per row over one group — and the same single f32 → bf16 cast
/// per group.
fn grouped_norm(
    ctx: &DeviceContext,
    branches: &[HiddenStates; HC_COUNT],
    weight: &[DeviceVec; HC_COUNT],
    variant: Variant,
    out: &mut [HiddenStates; HC_COUNT],
) -> Result<()> {
    for ((x, w), dst) in branches.iter().zip(weight).zip(out.iter_mut()) {
        if variant.offset_norm {
            rms_norm_batch_offset_into(ctx, x, w, RMS_EPS, dst)?;
        } else {
            rms_norm_batch_into(ctx, x, w, RMS_EPS, dst);
        }
    }
    Ok(())
}

/// Assemble the four per-branch norm buffers into the row-major
/// `[rows, HC_COUNT * HIDDEN]` stream the three projections read. Byte movement
/// only, on device: it copies what the norm wrote and computes nothing, so it
/// cannot mask a numeric difference. If/when PR #1158's grouped norm entry
/// point lands it writes this layout directly and retires the copy.
fn assemble_stream(
    ctx: &DeviceContext,
    branches: &[HiddenStates; HC_COUNT],
    out: &mut HiddenStates,
) -> Result<()> {
    for row in 0..out.seq_len {
        for (branch, src) in branches.iter().enumerate() {
            let from = src.data.slice(row * HIDDEN..(row + 1) * HIDDEN);
            let at = row * STREAM + branch * HIDDEN;
            let mut into = out.data.slice_mut(at..at + HIDDEN);
            ctx.stream
                .memcpy_dtod(&from, &mut into)
                .map_err(|e| anyhow!("stream assembly copy failed: {e}"))?;
        }
    }
    Ok(())
}

fn activate(
    ctx: &DeviceContext,
    act: Act,
    x: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    match act {
        Act::Silu => hc_silu_scaled_into(ctx, x, scale, out),
        Act::Sigmoid => hc_sigmoid_affine_into(ctx, x, scale, 1.0, out),
    }
}

struct Replay {
    /// The grouped norm's output, row-major `[rows, STREAM]`.
    normed: HiddenStates,
    mixed: HiddenStates,
    inject: Option<HiddenStates>,
}

/// Replay one `Qwen4ExpTextGatedResidual` forward from the branch-major
/// `hyper`, rounding to bf16 at each of the reference's submodule boundaries.
fn replay(
    ctx: &DeviceContext,
    module: &GatedResidual,
    hyper: &[HiddenStates; HC_COUNT],
    rows: usize,
    variant: Variant,
) -> Result<Replay> {
    let mut normed_branches: [HiddenStates; HC_COUNT] = std::array::from_fn(|_| {
        HiddenStates::zeros(ctx, HIDDEN, rows).expect("normed branch buffer")
    });
    grouped_norm(
        ctx,
        hyper,
        &module.norm_branches,
        variant,
        &mut normed_branches,
    )?;
    let mut normed = HiddenStates::zeros(ctx, STREAM, rows)?;
    assemble_stream(ctx, &normed_branches, &mut normed)?;

    let mut down = HiddenStates::zeros(ctx, LOWRANK, rows)?;
    gemm_into_checked(ctx, &module.down, &normed, &mut down)?;
    let mut bottleneck = HiddenStates::zeros(ctx, LOWRANK, rows)?;
    activate(
        ctx,
        variant.down_act,
        &down,
        variant.hc_divisor,
        &mut bottleneck,
    )?;

    let mut up = HiddenStates::zeros(ctx, STREAM, rows)?;
    gemm_into_checked(ctx, &module.up, &bottleneck, &mut up)?;
    let mut gate = HiddenStates::zeros(ctx, STREAM, rows)?;
    activate(ctx, variant.gate_act, &up, 1.0, &mut gate)?;

    let mut mixed = HiddenStates::zeros(ctx, HIDDEN, rows)?;
    hc_mul_branch_combine_into(
        ctx,
        &gate,
        &normed,
        HC_COUNT,
        variant.combine_scale,
        &mut mixed,
    )?;

    // `2 * sigmoid(block_inject_weight(normed) / hc_count)`. The division and
    // the gain are both exact powers of two in bf16, so the one rounding that
    // matters is the sigmoid's, which the kernel places where torch places it.
    let inject = match &module.inject {
        Some(weight) => {
            let mut logits = HiddenStates::zeros(ctx, HC_COUNT, rows)?;
            gemm_into_checked(ctx, weight, &normed, &mut logits)?;
            let mut out = HiddenStates::zeros(ctx, HC_COUNT, rows)?;
            hc_sigmoid_affine_into(ctx, &logits, variant.hc_divisor, 2.0, &mut out)?;
            Some(out)
        }
        None => None,
    };

    Ok(Replay {
        normed,
        mixed,
        inject,
    })
}

// ── comparison ────────────────────────────────────────────────────────────

struct Diff {
    max_abs: f32,
    /// `abs / max(abs(want), 1)`, so a tensor whose values are all under 1
    /// still reports a number that means something next to `ATOL`.
    max_rel: f32,
    /// Elements outside `ATOL + RTOL * abs(want)`.
    outside: usize,
    len: usize,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "compared tensors differ in length");
    let mut out = Diff {
        max_abs: 0.0,
        max_rel: 0.0,
        outside: 0,
        len: got.len(),
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        assert!(
            abs.is_finite(),
            "element {i}: non-finite difference ({g} against {w})"
        );
        if abs > ATOL + RTOL * w.abs() {
            out.outside += 1;
        }
        if abs > out.max_abs {
            out.max_abs = abs;
        }
        let rel = abs / w.abs().max(1.0);
        if rel > out.max_rel {
            out.max_rel = rel;
        }
    }
    out
}

fn report(label: &str, got: &[f32], want: &[f32]) -> Diff {
    let d = diff(got, want);
    println!(
        "  {label:<10} n={:<6} max_abs={:.3e} max_rel={:.3e} outside={:<5} bound@|w|=1 {:.3e}",
        d.len,
        d.max_abs,
        d.max_rel,
        d.outside,
        ATOL + RTOL,
    );
    d
}

// ── the gate ──────────────────────────────────────────────────────────────

struct Case {
    index: usize,
    rows: usize,
    /// `hyper_input`, split per branch for the grouped norm.
    hyper: [Vec<bf16>; HC_COUNT],
    normed: Vec<bf16>,
    mixed: Vec<bf16>,
    inject: Vec<bf16>,
    mixed_only: Vec<bf16>,
}

fn load_cases(fixture: &SafeTensors<'_>) -> Vec<Case> {
    (0..CASES.len())
        .map(|index| {
            let rows = CASES[index];
            let p = format!("case{index}_");
            golden_seq_len(fixture, &format!("{p}seq_len"), rows);
            let hyper = golden_bf16(fixture, &format!("{p}hyper_input"), &[1, rows, STREAM]);
            Case {
                index,
                rows,
                hyper: branch_major(&hyper, rows),
                normed: golden_bf16(fixture, &format!("{p}normed"), &[1, rows, STREAM]),
                mixed: golden_bf16(fixture, &format!("{p}mixed"), &[1, rows, HIDDEN]),
                inject: golden_bf16(fixture, &format!("{p}inject"), &[1, rows, HC_COUNT]),
                mixed_only: golden_bf16(fixture, &format!("{p}mixed_only"), &[1, rows, HIDDEN]),
            }
        })
        .collect()
}

struct Harness {
    ctx: DeviceContext,
    cases: Vec<Case>,
    combine: GatedResidual,
    mixer: GatedResidual,
}

/// The fixture, both modules and a device. Every test goes through here, so a
/// provenance mismatch fails all of them rather than only the positive one.
fn harness() -> Option<Harness> {
    let ctx = common::device_or_skip()?;
    let bytes = std::fs::read(fixture_path()).expect("golden fixture readable");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let combine = GatedResidual::load(
        &ctx,
        &fixture,
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        Some("block_inject_weight"),
    )
    .expect("combine weights");
    let mixer = GatedResidual::load(&ctx, &fixture, "mixer_norm", "mixer_down", "mixer_up", None)
        .expect("mixer weights");
    Some(Harness {
        ctx,
        cases: load_cases(&fixture),
        combine,
        mixer,
    })
}

/// A recorded tensor a negative control can move.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Out {
    Normed,
    Mixed,
    Inject,
    Readout,
}

/// A control that reaches the mix path but neither the norm nor the injection
/// head: only `mixed` can move.
const MIX_ONLY: &[Out] = &[Out::Mixed];
const MIX_AND_INJECT: &[Out] = &[Out::Mixed, Out::Inject];
const MIX_AND_READOUT: &[Out] = &[Out::Mixed, Out::Readout];
/// A control on the norm, which is every other tensor's only input.
const EVERYTHING: &[Out] = &[Out::Normed, Out::Mixed, Out::Inject, Out::Readout];

impl Harness {
    fn upload(&self, case: &Case) -> [HiddenStates; HC_COUNT] {
        std::array::from_fn(|branch| {
            HiddenStates::from_host(&self.ctx, &case.hyper[branch], HIDDEN, case.rows)
                .expect("upload hyper branch")
        })
    }

    /// The positive gate: every recorded tensor of both modules, every case.
    fn assert_reference(&self, case: &Case) {
        let hyper = self.upload(case);
        let combine = replay(
            &self.ctx,
            &self.combine,
            &hyper,
            case.rows,
            Variant::REFERENCE,
        )
        .expect("replay combine block");
        self.parity("normed", &combine.normed, &case.normed);
        self.parity("mixed", &combine.mixed, &case.mixed);
        self.parity(
            "inject",
            combine.inject.as_ref().expect("combine injects"),
            &case.inject,
        );
        let mixer = replay(
            &self.ctx,
            &self.mixer,
            &hyper,
            case.rows,
            Variant::REFERENCE,
        )
        .expect("replay readout");
        assert!(
            mixer.inject.is_none(),
            "the use_combine=False readout has no injection head"
        );
        self.parity("mixed_only", &mixer.mixed, &case.mixed_only);
    }

    /// One negative control over every case, against exactly the recorded
    /// tensors the control moves. Checking only those is the point: a control
    /// that also "failed" on a tensor it never touches would say nothing about
    /// whether the bound is tight, and one that failed on nothing would mean
    /// the control never reached the arithmetic.
    fn assert_diverges(&self, variant: Variant, moved: &[Out]) {
        for case in &self.cases {
            println!("case {} (S = {}):", case.index, case.rows);
            let hyper = self.upload(case);
            let combine = replay(&self.ctx, &self.combine, &hyper, case.rows, variant)
                .expect("replay negative control");
            let readout = moved.contains(&Out::Readout).then(|| {
                replay(&self.ctx, &self.mixer, &hyper, case.rows, variant)
                    .expect("replay negative readout")
            });
            for out in moved {
                match out {
                    Out::Normed => self.reject("normed", &combine.normed, &case.normed, variant),
                    Out::Mixed => self.reject("mixed", &combine.mixed, &case.mixed, variant),
                    Out::Inject => self.reject(
                        "inject",
                        combine.inject.as_ref().expect("combine injects"),
                        &case.inject,
                        variant,
                    ),
                    Out::Readout => {
                        let mixer = readout.as_ref().expect("readout replayed above");
                        self.reject("mixed_only", &mixer.mixed, &case.mixed_only, variant);
                    }
                }
            }
        }
    }

    fn parity(&self, label: &str, got: &HiddenStates, want: &[bf16]) {
        let got = got.to_host(&self.ctx).expect("download replay output");
        let d = report(label, &got, &f32_of(want));
        assert_eq!(
            d.outside, 0,
            "{label}: {} of {} elements outside abs {ATOL} + rel {RTOL}, worst {:.3e}",
            d.outside, d.len, d.max_abs
        );
    }

    fn reject(&self, label: &str, got: &HiddenStates, want: &[bf16], variant: Variant) {
        let got = got.to_host(&self.ctx).expect("download replay output");
        let d = report(label, &got, &f32_of(want));
        assert!(
            d.outside > 0,
            "{label} under `{}` still passes at abs {ATOL} + rel {RTOL} (worst {:.3e}): \
             a bound that absorbs a wrong block proves nothing",
            variant.label,
            d.max_abs
        );
        println!(
            "  -> rejected `{}`: {} of {} elements outside, worst {:.3e}",
            variant.label, d.outside, d.len, d.max_abs
        );
    }
}

#[test]
fn hyper_connection_block_matches_golden() {
    let Some(h) = harness() else {
        return;
    };
    println!("parity bound: abs {ATOL} + rel {RTOL} (see the ATOL docs)");
    for case in &h.cases {
        println!("case {} (S = {}):", case.index, case.rows);
        h.assert_reference(case);
    }
}

#[test]
fn negative_sum_over_branches_fails() {
    let Some(h) = harness() else {
        return;
    };
    // `mixed` comes out `hc_count` times the golden; the norm and the injection
    // head never see the combine, so they are not in play.
    let variant = Variant {
        label: "sum over branches instead of mean",
        combine_scale: 1.0,
        ..Variant::REFERENCE
    };
    h.assert_diverges(variant, MIX_AND_READOUT);
}

#[test]
fn negative_missing_hc_divisor_fails() {
    let Some(h) = harness() else {
        return;
    };
    // The reference divides by `hc_count` on both the mix path and the
    // injection weights; dropping it moves both and leaves the norm alone.
    let variant = Variant {
        label: "no /hc_count on the mix path or the injection weights",
        hc_divisor: 1.0,
        ..Variant::REFERENCE
    };
    h.assert_diverges(variant, MIX_AND_INJECT);
}

#[test]
fn negative_swapped_or_dropped_mix_activations_fail() {
    let Some(h) = harness() else {
        return;
    };
    // The reference is SiLU on the down output then sigmoid on the up output.
    // Swapping the two, and dropping either one, all have to fail.
    for variant in [
        Variant {
            label: "mix activations swapped: sigmoid then silu",
            down_act: Act::Sigmoid,
            gate_act: Act::Silu,
            ..Variant::REFERENCE
        },
        Variant {
            label: "silu instead of sigmoid on the mix gate",
            gate_act: Act::Silu,
            ..Variant::REFERENCE
        },
        Variant {
            label: "sigmoid instead of silu on the mix down output",
            down_act: Act::Sigmoid,
            ..Variant::REFERENCE
        },
    ] {
        println!("--- {}", variant.label);
        h.assert_diverges(variant, MIX_ONLY);
    }
}

#[test]
fn negative_plain_weight_norm_fails() {
    let Some(h) = harness() else {
        return;
    };
    // The norm is the block's only input to everything else, so plain `w`
    // against the reference's `(1 + w)` moves all four recorded tensors.
    let variant = Variant {
        label: "plain-w norm instead of (1+w)",
        offset_norm: false,
        ..Variant::REFERENCE
    };
    h.assert_diverges(variant, EVERYTHING);
}
