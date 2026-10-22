//! Qwen3.8-Flash-Next PLE block against the frozen reference golden.
//!
//! `test_data/qwen38fn-ple-golden.safetensors` was dumped from transformers
//! 5.16.1 `qwen4_exp` by `tools/accuracy/dump_qwen4_ple_golden.py`. Each case
//! carries the block's inputs *after* the two projections the 95 GiB n-gram
//! table feeds — `hidden`, `key` and `value` — plus the reference's own block
//! output `out`, so this gate replays those inputs through the ops layer and
//! compares. The projections and the table gather are not covered here; the
//! golden starts after them.
//!
//! This is the device half of a two-half consumer. The host half,
//! `pegainfer-qwen4/tests/ple_ngram_replay.rs`, runs with no CUDA at all and
//! replays the same stored `out` **bit-identically** in f32 on the CPU
//! (81920/81920 and 122880/122880 elements exact), which is what makes the
//! bound below tight enough to mean something: every rounding point the
//! reference has is known and reproducible, so the only thing left for a device
//! to be wrong about is a knob, not a tolerance. The device path turns out to be
//! nearly as exact — 204799 of the 204800 stored outputs come back bit-for-bit,
//! and the one that does not is a single bf16 ulp — see [`OUT_MAX_ABS`].
//!
//! Provenance is asserted on every load, never skipped: the manifest's
//! `format_version`, `transformers_version` and `config`/`config_sha256` are
//! held against this build's frozen config, and that config's PLE geometry
//! against `PleBlockGeometry::qwen38_flash_next()`. A fixture from another
//! checkpoint or another semantics fails here instead of being skipped.
//!
//! Without a device the comparisons skip; `PEGAINFER_REQUIRE_GPU=1` turns that
//! into a failure. The two device-free halves — provenance and the entry points'
//! refusal contract — never skip.
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_ple_block_parity -- --nocapture
//!
//! **What this fixture cannot reach**, recorded rather than papered over:
//!
//! - The gate's `clamp_min(1e-6)` floor and its `sign(0)` branch are both inert
//!   here: the smallest `|pre_gate|` either case produces is 0.0232, five orders
//!   above the floor, so no gate is clamped and none is exactly zero, and a wrong
//!   floor constant would pass. Both are transcribed from the reference's own
//!   `abs().clamp_min(1e-6).sqrt() * sign()`, and the host oracle in
//!   `pegainfer-qwen4` carries the same two branches with the same non-exercise.
//! - Row counts are 8 and 12, the fixture's own. `seq_len` 0 is pinned only as a
//!   guard (the zero-extent positive controls below, which form no launch), and
//!   `seq_len` 1 — where every conv tap but the last reads the zero state — is
//!   compared against nothing, because the golden records no such case and the
//!   fixture is committed.
//! - The ops layer's own extent guards (`hidden.len()`, `workspace.len()` and the
//!   rest) are not exercised: they run before any pointer is formed but after a
//!   `CudaSlice` exists, so reaching one needs a device and a deliberately short
//!   buffer. The entry points' guards — the ones that stand between a caller and
//!   an out-of-bounds read — are covered below.

#![cfg(feature = "qwen4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PleBlockGeometry;
use pegainfer_kernels::ops::PleBlockVariant;
use pegainfer_kernels::ops::ple_block_into;
use pegainfer_kernels::ops::ple_block_variant_into;
use pegainfer_kernels::tensor::DeviceContext;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-ple-golden.safetensors";
const METADATA_KEY: &str = "qwen4_ple_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";
const CASES: usize = 2;

/// Block-output parity bound, absolute, on a stream whose largest stored element
/// is 0.8438.
///
/// **Measured on one sm_120 card (RTX 5060 Laptop, CUDA 12.9.1): case 0 is
/// bit-identical, 81920/81920, and case 1 differs on exactly one element of
/// 122880, by 1.221e-4 — one bf16 ulp at that element's magnitude (0.024292
/// against 0.024414), 0.0076 of this bound.** Over both cases, 204799 of 204800
/// outputs are bit-for-bit the reference's.
///
/// The host replay of this same block is bit-identical too, so the bound is not
/// covering a semantics gap. It covers the three places the device path cannot
/// be the host path, all of them the same size:
///
/// 1. **`rsqrt.approx.ftz.f32`.** The three grouped norms are the existing
///    batched `(1 + w)` RMSNorm, whose reciprocal comes from the PTX
///    approximation instruction, where the reference and the host replay both use
///    a correctly rounded `1 / sqrt`. Its ~1.4e-7 relative error against a bf16
///    step of ~3.9e-3 flips the bf16 rounding of roughly one normed element in
///    30000; an instrumented run of this same call during bring-up found exactly
///    two such elements across case 0's three norms (245760 outputs), each one
///    bf16 ulp of itself, and the gated stream bit-identical.
/// 2. **A different reduction tree.** The 2560-wide sum of squares and the
///    2560-wide gate dot are f32 butterflies here, an f32 tree in the reference
///    and an f64 exact sum in the host replay. All three land within an f32 ulp
///    or two of each other, four orders below the bf16 step the result is
///    rounded to, so they flip the same ~1-in-10^4..10^5 of elements.
/// 3. **`expf`**, the one op the host gate already budgets for: its allowance is
///    a whole bf16 ulp injected into *every* `sigmoid` and *every* `silu`, which
///    moves `out` by at most 7.8e-3.
///
/// One flipped bf16 anywhere upstream reaches `out` as at most about one bf16 ulp
/// of `out` at that element: the gate is a single scalar per (row, branch), so a
/// flipped `sigmoid` moves that branch's 2560 channels by a common relative
/// step; the conv, its silu (derivative bounded by ~1.1) and the residual add
/// cannot amplify it further. One ulp at the largest stored element is 2^-8 =
/// 3.9e-3, and the bound is four of those — which is also exactly the host
/// gate's bound, so the two halves of this consumer share one number and can be
/// read against each other.
///
/// It cannot be tightened towards the measured 1.2e-4, and the reason is not "it
/// currently passes": `rsqrt.approx`, `expf` and the reduction order are
/// architecture- and libm-defined, so another GPU flips a *different* set of
/// elements, possibly including one near 0.8438 where a single ulp is 3.9e-3.
/// The flip count is a discrete random variable, not a constant. Nor does it need
/// tightening: every structural fault this block can have — a collapsed rounding
/// point, the wrong group, the wrong divisor, the wrong taps — moves elements by
/// orders of this bound, which is what the three controls below measure at 17x,
/// 21x and 49x outside it.
const OUT_MAX_ABS: f32 = 1.6e-2;

/// Written over the output before every launch, so a kernel that wrote part of
/// the stream cannot pass as one that wrote all of it.
const SENTINEL: bf16 = bf16::from_bits(0x7fc0);

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn config_path() -> String {
    format!("{}/../{CONFIG}", env!("CARGO_MANIFEST_DIR"))
}

/// Bit equality for the scalars the kernels pin. They refuse any other value, so
/// a tolerance here would let a config drift through.
fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// The fixture's own record of what it was dumped from, held against the
/// reference version, the frozen config's bytes and the PLE geometry this crate
/// compiles for. Panics rather than skips: provenance is the whole reason a
/// parity number means anything.
fn assert_provenance(bytes: &[u8]) -> serde_json::Value {
    let (_, metadata) = SafeTensors::read_metadata(bytes).expect("fixture metadata");
    let recorded = metadata
        .metadata()
        .as_ref()
        .expect("fixture metadata map")
        .get(METADATA_KEY)
        .expect("fixture manifest key")
        .clone();
    let manifest: serde_json::Value =
        serde_json::from_str(&recorded).expect("fixture manifest is not JSON");

    assert_eq!(
        manifest["format_version"].as_u64(),
        Some(FORMAT_VERSION),
        "fixture format_version; this gate reads version {FORMAT_VERSION}"
    );
    assert_eq!(
        manifest["transformers_version"].as_str(),
        Some(REF_TRANSFORMERS),
        "fixture was dumped from another reference version, and the frozen \
         semantics are that one's"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG),
        "fixture was dumped from another config"
    );

    let config_bytes =
        std::fs::read(config_path()).unwrap_or_else(|e| panic!("read {}: {e}", config_path()));
    let expected = sha256_hex(&config_bytes);
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(expected.as_str()),
        "{CONFIG} is not the config the fixture was dumped from"
    );

    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).expect("frozen config is not JSON");
    let text = &config["text_config"];
    let geometry = PleBlockGeometry::qwen38_flash_next();
    let field = |name: &str| -> usize {
        text[name]
            .as_u64()
            .unwrap_or_else(|| panic!("{name} must be an integer")) as usize
    };
    assert_eq!(field("hidden_size"), geometry.hidden);
    // The block's gate divides by `sqrt(hidden_size)`, and its value stream is
    // `ple_embed_dim` wide; the reference requires the two to be the same width.
    assert_eq!(field("ple_embed_dim"), geometry.hidden);
    assert_eq!(field("hc_count"), geometry.hc_count);
    assert_eq!(field("ple_conv_kernel_size"), geometry.conv_kernel);
    // The conv's dilation is `ngram_size`, which is also the widest n-gram.
    assert_eq!(field("ngram_size"), geometry.dilation);
    assert_eq!(geometry.conv_state(), 9);
    let eps = text["rms_norm_eps"]
        .as_f64()
        .expect("text_config.rms_norm_eps must be a number");
    assert!(
        same_f32(eps as f32, geometry.rms_eps),
        "rms_norm_eps {eps} is not the compiled {}",
        geometry.rms_eps
    );
    let ple_layers = text["ple_layer_ids"].as_array().expect("ple_layer_ids");
    assert_eq!(
        ple_layers.len(),
        1,
        "the reference offsets each PLE layer's prime run by its position, and \
         this crate derives one table"
    );
    geometry.validate().expect("the frozen geometry");
    // Recorded, not asserted: the host the golden was dumped on decides the
    // order its float32 reductions round in, which is where the host replay's
    // bit-identity was measured and not something this gate can reproduce.
    println!(
        "golden provenance: transformers {}, torch {}, {}, seed {}",
        manifest["transformers_version"].as_str().unwrap_or("?"),
        manifest["torch_version"].as_str().unwrap_or("?"),
        manifest["platform"].as_str().unwrap_or("?"),
        manifest["seed"].as_u64().unwrap_or(0),
    );
    manifest
}

fn i32_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<i32>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::I32, "{name} must be i32");
    let host = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), host)
}

fn bf16_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<bf16>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::BF16, "{name} must be bf16");
    let shape = view.shape().to_vec();
    let host = view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
        .collect();
    (shape, host)
}

/// One recorded sequence: the block's inputs and the reference's output.
struct Case {
    index: usize,
    seq_len: usize,
    hidden: Vec<bf16>,
    key: Vec<bf16>,
    value: Vec<bf16>,
    reference: Vec<bf16>,
}

fn load_case(fixture: &SafeTensors<'_>, index: usize, geometry: &PleBlockGeometry) -> Case {
    let width = geometry.width();
    let name = format!("case{index}");

    let (seq_shape, seq_cell) = i32_tensor(fixture, &format!("{name}_seq_len"));
    assert_eq!(seq_shape, vec![1], "{name}_seq_len must be one cell");
    let seq_len = usize::try_from(seq_cell[0])
        .ok()
        .filter(|&len| len > 0)
        .unwrap_or_else(|| panic!("{name}_seq_len must be a positive row count"));

    // The recording is whole-sequence, so the row selection has to be the
    // identity: the conv reads nine rows before the sequence with a zero state,
    // and a dropped row would need that row's conv and window state to stay
    // correct. Asserted rather than assumed, because a gate that quietly
    // compared a prefix would still report a number.
    let (rows_shape, rows) = i32_tensor(fixture, &format!("{name}_rows"));
    assert_eq!(rows_shape, vec![seq_len], "{name}_rows must be one per row");
    assert!(
        rows.iter().enumerate().all(|(i, &row)| row == i as i32),
        "{name}_rows must be the identity 0..{seq_len}: the block is replayed \
         whole-sequence and the fixture records no cache state"
    );

    let (hidden_shape, hidden) = bf16_tensor(fixture, &format!("{name}_hidden"));
    let (key_shape, key) = bf16_tensor(fixture, &format!("{name}_key"));
    let (value_shape, value) = bf16_tensor(fixture, &format!("{name}_value"));
    let (out_shape, reference) = bf16_tensor(fixture, &format!("{name}_out"));
    for (what, shape) in [
        ("hidden", hidden_shape),
        ("key", key_shape),
        ("out", out_shape),
    ] {
        assert_eq!(shape, vec![seq_len, width], "{name}_{what} shape");
    }
    assert_eq!(
        value_shape,
        vec![seq_len, geometry.hidden],
        "{name}_value shape"
    );
    Case {
        index,
        seq_len,
        hidden,
        key,
        value,
        reference,
    }
}

/// One case's parity: how much was compared, the worst element and where, the
/// shape of the residual, and how much of the bound it used.
///
/// The identical count and the two thresholds are reported rather than asserted:
/// a handful of bf16-rounding flips and a structural fault are different animals
/// and these counts say which one this is. A systematic error shows up as a
/// large identical-count shortfall *and* a large mean, where rounding noise
/// shows up as a small mean with a max at one or two ulps.
struct Summary {
    compared: usize,
    max_abs: f32,
    mean_abs: f32,
    identical: usize,
    /// `(row, channel, got, want)`
    worst: (usize, usize, f32, f32),
    over_1e3: usize,
    over_1e4: usize,
    bound_fraction: f32,
}

/// Run one case through the ops layer and compare against the stored output.
fn check_case(
    ctx: &DeviceContext,
    geometry: &PleBlockGeometry,
    variant: PleBlockVariant,
    weights: &Weights,
    case: &Case,
) -> Summary {
    let width = geometry.width();
    let elements = case.seq_len * width;

    let hidden = ctx.stream.clone_htod(&case.hidden).expect("hidden upload");
    let key = ctx.stream.clone_htod(&case.key).expect("key upload");
    let value = ctx.stream.clone_htod(&case.value).expect("value upload");
    // Pre-filled with a NaN sentinel: every element of `out` is written by the
    // conv pass, so anything left holding the sentinel is an unwritten row.
    let mut out = ctx
        .stream
        .clone_htod(&vec![SENTINEL; elements])
        .expect("out buffer");
    let mut workspace = ctx
        .stream
        .alloc_zeros::<bf16>(geometry.workspace_len(case.seq_len))
        .expect("PLE workspace");

    // The reference variant goes through the production entry point, which is
    // the one a serving path calls; the controls go through the variant entry
    // it delegates to.
    let result = match variant {
        PleBlockVariant::Reference => ple_block_into(
            ctx,
            geometry,
            &hidden,
            &key,
            &value,
            &weights.norm_key,
            &weights.norm_query,
            &weights.norm_conv,
            &weights.conv1d,
            case.seq_len,
            &mut out,
            &mut workspace,
        ),
        control => ple_block_variant_into(
            ctx,
            geometry,
            control,
            &hidden,
            &key,
            &value,
            &weights.norm_key,
            &weights.norm_query,
            &weights.norm_conv,
            &weights.conv1d,
            case.seq_len,
            &mut out,
            &mut workspace,
        ),
    };
    result.expect("PLE block");
    ctx.sync().expect("sync");
    let got = ctx.stream.clone_dtoh(&out).expect("out download");

    assert_eq!(
        got.len(),
        case.reference.len(),
        "case {}: length",
        case.index
    );
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut identical = 0usize;
    let mut over_1e3 = 0usize;
    let mut over_1e4 = 0usize;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
    for (i, (x, want)) in got.iter().zip(&case.reference).enumerate() {
        let g = x.to_f32();
        let w = want.to_f32();
        // Finiteness first: the sentinel is a NaN, so a NaN the kernels
        // *produced* would otherwise be reported as an unwritten element.
        assert!(
            g.is_finite(),
            "case {}: element {i} (row {}, channel {}) came back {}",
            case.index,
            i / width,
            i % width,
            if x.to_bits() == SENTINEL.to_bits() {
                "still holding the sentinel, so the conv pass did not write the whole stream"
            } else {
                "non-finite"
            }
        );
        assert!(
            w.is_finite(),
            "case {}: the golden's element {i} is not finite",
            case.index
        );
        let delta = (g - w).abs();
        sum_abs += f64::from(delta);
        identical += usize::from(x.to_bits() == want.to_bits());
        over_1e3 += usize::from(delta > 1.0e-3);
        over_1e4 += usize::from(delta > 1.0e-4);
        if delta > max_abs {
            max_abs = delta;
            worst = (i / width, i % width, g, w);
        }
    }
    Summary {
        compared: got.len(),
        max_abs,
        mean_abs: (sum_abs / got.len() as f64) as f32,
        identical,
        worst,
        over_1e3,
        over_1e4,
        bound_fraction: max_abs / OUT_MAX_ABS,
    }
}

fn print_summary(label: &str, geometry: &PleBlockGeometry, case: &Case, summary: &Summary) {
    println!(
        "{label} case {}: S={} stream={}x{} out max_abs={:.3e} at row {} channel {} \
         ({:.6} against {:.6}), mean_abs={:.3e}, bit_identical={}/{}, above 1e-3 {}/{}, \
         above 1e-4 {}/{}, bound fraction {:.4} of {OUT_MAX_ABS:.1e}",
        case.index,
        case.seq_len,
        geometry.hc_count,
        geometry.hidden,
        summary.max_abs,
        summary.worst.0,
        summary.worst.1,
        summary.worst.2,
        summary.worst.3,
        summary.mean_abs,
        summary.identical,
        summary.compared,
        summary.over_1e3,
        summary.compared,
        summary.over_1e4,
        summary.compared,
        summary.bound_fraction,
    );
}

/// The block's own weights, uploaded once and shared by every case: the three
/// stream-wide `(1 + w)` norm weights and the depthwise conv.
struct Weights {
    norm_key: CudaSlice<bf16>,
    norm_query: CudaSlice<bf16>,
    norm_conv: CudaSlice<bf16>,
    conv1d: CudaSlice<bf16>,
}

fn load_weights(
    ctx: &DeviceContext,
    fixture: &SafeTensors<'_>,
    geometry: &PleBlockGeometry,
) -> Weights {
    let width = geometry.width();
    // The conv weight is stored rank-3 (`[width, 1, conv_kernel]`, the
    // reference's own depthwise shape) and read flat, one tap run per channel.
    let upload = |name: &str, shape: &[usize]| -> CudaSlice<bf16> {
        let (stored, host) = bf16_tensor(fixture, name);
        assert_eq!(&stored, shape, "{name} must be {shape:?}");
        ctx.stream
            .clone_htod(&host)
            .unwrap_or_else(|e| panic!("{name} upload: {e}"))
    };
    Weights {
        norm_key: upload("norm_key", &[width]),
        norm_query: upload("norm_query", &[width]),
        norm_conv: upload("norm_conv", &[width]),
        conv1d: upload("conv1d_weight", &[width, 1, geometry.conv_kernel]),
    }
}

/// The fixture's provenance and its geometry against this crate's compiled
/// constants. Device-free, and never behind `device_or_skip`: a fixture from
/// another reference would make every number below meaningless.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = PleBlockGeometry::qwen38_flash_next();
    let cases: Vec<Case> = (0..CASES)
        .map(|index| load_case(&fixture, index, &geometry))
        .collect();
    let lengths: Vec<usize> = cases.iter().map(|case| case.seq_len).collect();
    // The two cases the dump records, asserted rather than assumed. The block
    // itself is EOS-blind — the hash is not, and the device-free gate covers
    // that — but the row counts are what make both ends of the conv's warm-up
    // exercised: a case no longer than the nine-row state has rows whose taps are
    // all padding, and a case longer than it has rows whose four taps are all
    // real. A fixture that recorded only one of the two would still produce a
    // parity number.
    assert_eq!(lengths, vec![8, 12], "the recorded case lengths");
    assert!(
        lengths.iter().any(|&len| len <= geometry.conv_state()),
        "a case must be short enough that some rows read the whole zero state, \
         or the conv's warm-up is untested"
    );
    assert!(
        lengths.iter().any(|&len| len > geometry.conv_state()),
        "a case must be long enough that some rows read four real taps"
    );
}

/// The kernels' own half of the refusal contract, likewise device-free: every
/// entry point rejects a geometry it was not compiled for *before* it forms a
/// launch, so this can be pinned without a device and must never be skippable.
///
/// The two zero-extent calls per entry point come first as a **positive control**
/// — the compiled geometry with nothing to do, which the guards must let through
/// and which forms no launch. Without them an entry point that refused
/// everything would make every refusal below vacuous.
#[test]
fn the_entry_points_refuse_a_geometry_they_were_not_compiled_for() {
    use cudarc::driver::sys::CUresult;
    use cudarc::driver::sys::CUstream;
    use pegainfer_kernels::ffi::Half;
    use pegainfer_kernels::ffi::qwen4_ple_conv_residual_cuda as conv;
    use pegainfer_kernels::ffi::qwen4_ple_gate_cuda as gate;
    use pegainfer_kernels::ffi::qwen4_ple_split_branches_cuda as split;

    // Dangling and well-aligned, never dereferenced: every call below leaves
    // through a guard or through the zero-extent early return, and a launch is
    // the only thing that would touch a pointer.
    let half: *const Half = std::ptr::NonNull::<Half>::dangling().as_ptr();
    // Any positive finite divisor passes the guard; the compiled geometry's own
    // `sqrt(2560)` keeps the positive controls honest about what they let through.
    let sqrt_hidden = PleBlockGeometry::qwen38_flash_next().sqrt_hidden();
    let invalid = CUresult::CUDA_ERROR_INVALID_VALUE as i32;
    let success = CUresult::CUDA_SUCCESS as i32;
    let null_stream = CUstream::default();

    // Positive controls: the compiled geometry with no rows to do.
    assert_eq!(
        (unsafe { split(half, half.cast_mut(), 0, 10240, 2560, 4, null_stream) }) as i32,
        success,
        "the compiled split geometry must pass the guards"
    );
    assert_eq!(
        (unsafe {
            gate(
                half,
                half,
                half,
                half.cast_mut(),
                0,
                2560,
                4,
                2560,
                sqrt_hidden,
                1,
                null_stream,
            )
        }) as i32,
        success,
        "the compiled gate geometry must pass the guards"
    );
    assert_eq!(
        (unsafe {
            conv(
                half,
                half,
                half,
                half.cast_mut(),
                0,
                2560,
                4,
                4,
                3,
                null_stream,
            )
        }) as i32,
        success,
        "the compiled conv geometry must pass the guards"
    );

    for (what, result) in [
        (
            "a split over a stream that is not the compiled 10240 wide",
            (unsafe { split(half, half.cast_mut(), 0, 10239, 2560, 4, null_stream) }) as i32,
        ),
        (
            "a split over a group that is not the compiled 2560 wide",
            (unsafe { split(half, half.cast_mut(), 0, 10240, 1280, 4, null_stream) }) as i32,
        ),
        (
            "a split over a branch count that is not the compiled 4",
            (unsafe { split(half, half.cast_mut(), 0, 10240, 2560, 2, null_stream) }) as i32,
        ),
        (
            "a negative row count",
            (unsafe { split(half, half.cast_mut(), -1, 10240, 2560, 4, null_stream) }) as i32,
        ),
        (
            "a gate over a group that is not the compiled 2560 wide",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    1280,
                    4,
                    1280,
                    sqrt_hidden,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a gate over a branch count that is not the compiled 4",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    2,
                    2560,
                    sqrt_hidden,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a gate whose span is neither the group nor the stream",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    5120,
                    sqrt_hidden,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a gate whose sign restore is neither 0 nor 1",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    2560,
                    sqrt_hidden,
                    2,
                    null_stream,
                )
            }) as i32,
        ),
        (
            // A zero divisor would put an infinity in every gate.
            "a gate with a zero sqrt_hidden",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    2560,
                    0.0,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            // `!(x > 0)` refuses NaN, which would otherwise poison every gate
            // silently rather than fail loudly.
            "a gate with a NaN sqrt_hidden",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    2560,
                    f32::NAN,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a conv over a group that is not the compiled 2560 wide",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    1280,
                    4,
                    4,
                    3,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a conv over a branch count that is not the compiled 4",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    2,
                    4,
                    3,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a conv over a kernel that is not the compiled 4 taps",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    2,
                    3,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a conv with a zero dilation",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    4,
                    0,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a conv with a negative dilation",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    4,
                    -3,
                    null_stream,
                )
            }) as i32,
        ),
    ] {
        assert_eq!(result, invalid, "{what} must be refused");
    }

    // One null per kind of argument, so the guards are pinned to cover the
    // pointers and not only the extents.
    for (what, result) in [
        (
            "a null split source",
            (unsafe {
                split(
                    std::ptr::null(),
                    half.cast_mut(),
                    0,
                    10240,
                    2560,
                    4,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a null split destination",
            (unsafe { split(half, std::ptr::null_mut(), 0, 10240, 2560, 4, null_stream) }) as i32,
        ),
        (
            "a null gate value",
            (unsafe {
                gate(
                    half,
                    half,
                    std::ptr::null(),
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    2560,
                    sqrt_hidden,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a null gated output",
            (unsafe {
                gate(
                    half,
                    half,
                    half,
                    std::ptr::null_mut(),
                    0,
                    2560,
                    4,
                    2560,
                    sqrt_hidden,
                    1,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a null conv weight",
            (unsafe {
                conv(
                    half,
                    half,
                    std::ptr::null(),
                    half.cast_mut(),
                    0,
                    2560,
                    4,
                    4,
                    3,
                    null_stream,
                )
            }) as i32,
        ),
        (
            "a null conv output",
            (unsafe {
                conv(
                    half,
                    half,
                    half,
                    std::ptr::null_mut(),
                    0,
                    2560,
                    4,
                    4,
                    3,
                    null_stream,
                )
            }) as i32,
        ),
    ] {
        assert_eq!(result, invalid, "{what} must be refused");
    }
}

#[test]
fn ple_block_matches_the_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = PleBlockGeometry::qwen38_flash_next();
    let weights = load_weights(&ctx, &fixture, &geometry);

    let mut total = 0usize;
    let mut worst_abs = 0.0f32;
    let mut worst_fraction = 0.0f32;
    let mut identical = 0usize;
    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let summary = check_case(&ctx, &geometry, PleBlockVariant::Reference, &weights, &case);
        print_summary("reference", &geometry, &case, &summary);
        total += summary.compared;
        identical += summary.identical;
        worst_abs = worst_abs.max(summary.max_abs);
        worst_fraction = worst_fraction.max(summary.bound_fraction);
        assert!(
            summary.max_abs <= OUT_MAX_ABS,
            "case {}: the PLE block deviates by {:.3e} > {OUT_MAX_ABS:.3e} at row {} channel {} \
             ({:.6} against {:.6}); see OUT_MAX_ABS for how the bound is derived",
            index,
            summary.max_abs,
            summary.worst.0,
            summary.worst.1,
            summary.worst.2,
            summary.worst.3,
        );
    }
    println!(
        "parity over {CASES} cases and {total} block outputs: max abs {worst_abs:.3e}, worst \
         bound fraction {worst_fraction:.4} of {OUT_MAX_ABS:.1e}, {identical} of {total} \
         bit-identical to the golden"
    );
    assert!(
        worst_fraction <= 1.0,
        "the worst block output is {worst_fraction} of the bound"
    );
}

/// Control (a): the gate is `sign(g) * sqrt(|g|)`. Dropping the sign restore
/// pushes every negative gate through the sigmoid as its magnitude, which
/// inverts that branch's contribution to the stream.
#[test]
fn an_unsigned_sqrt_breaks_the_block_output() {
    assert_control_fails(
        PleBlockVariant::UnsignedSqrtGate,
        "unsigned sqrt on the gate",
    );
}

/// Control (b): the conv is dilated by `ngram_size` (3), with a `(kernel - 1) *
/// dilation` = 9 row zero state. Dilation 1 reads three of its four taps from
/// rows the reference never looks at.
#[test]
fn dilation_one_breaks_the_block_output() {
    assert_control_fails(PleBlockVariant::Dilation1Conv, "conv dilation 1");
}

/// Control (c): the gate dot is per BRANCH — 2560 channels, one gate per
/// branch, all four branches scaling the same `value` row. One dot over the
/// whole 10240-wide stream instead gives every branch the same, larger gate.
#[test]
fn a_full_stream_gate_breaks_the_block_output() {
    assert_control_fails(PleBlockVariant::FullStreamGate, "a full-stream gate dot");
}

/// A negative control that passes is worse than no control: it means the gate
/// cannot see the fault it claims to pin. Every case must leave the bound, not
/// just one, or a knob that only matters for a long sequence would look pinned
/// by a short one.
fn assert_control_fails(variant: PleBlockVariant, label: &str) {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = PleBlockGeometry::qwen38_flash_next();
    let weights = load_weights(&ctx, &fixture, &geometry);

    let mut worst = 0.0f32;
    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let summary = check_case(&ctx, &geometry, variant, &weights, &case);
        print_summary(label, &geometry, &case, &summary);
        worst = worst.max(summary.max_abs);
        assert!(
            summary.max_abs > OUT_MAX_ABS,
            "case {}: {label} stayed within {:.3e} of the golden ({:.4} of the bound), so the \
             device path does not pin the knob this control turns",
            index,
            summary.max_abs,
            summary.bound_fraction
        );
    }
    println!(
        "{label}: worst deviation {worst:.3e} = {:.0}x the {OUT_MAX_ABS:.1e} bound",
        worst / OUT_MAX_ABS
    );
}
