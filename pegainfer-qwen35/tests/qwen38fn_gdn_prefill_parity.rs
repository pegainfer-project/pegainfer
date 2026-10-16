//! Qwen3.8-Flash-Next Gated DeltaNet prefill replay against the frozen golden.
//!
//! `test_data/qwen38fn-gdn-golden.safetensors` — dumped by
//! `tools/accuracy/dump_qwen38fn_gdn_golden.py` from transformers 5.16.1
//! `qwen4_exp` — records, per case, the GDN sublayer's activations at the
//! boundary a reusing line enters: the pre-conv `mixed_qkv`, the reference's
//! post-conv pre-repeat `query`/`key`/`value`, the derived gate pair `g`/`beta`,
//! the output gate `z`, and the reference's gated-norm output `out`, which is
//! the sublayer's output *before* `out_proj`. This gate replays them through the
//! in-tree GDN stack — Qwen3.5's causal `conv1d` prefill kernel, the Triton AOT
//! chunk-wise gated delta rule, and the shared gated RMSNorm with the sigmoid
//! activation this checkpoint's `output_gate_type` selects — and compares `out`.
//!
//! The two halves of the chunk-wise operator meet at the golden's boundary:
//! `gated_delta_rule_prefill_chunk_prepare_into` does the head expansion and the
//! q/k L2 norm on the conv output, then the recorded `g`/`beta` replace what it
//! derived and `gated_delta_rule_prefill_chunkwise_from_gates_into` runs the
//! chunk stages. The fixture stores the gate pair *after* derivation, and the
//! `a`/`b` projections, `dt_bias` and `A_log` that would reproduce it are not in
//! it, so the derivation itself is not this gate's subject — the chunk math the
//! derivation feeds is.
//!
//! S=16 and S=64 are single-chunk; S=80 and S=128 cross the 64-token chunk
//! boundary, so the reference's carried recurrent state is covered. Negative
//! controls: the silu gate this checkpoint's `output_gate_type` rules out, a
//! skipped conv leg, and S=128 replayed as two independent single-chunk runs
//! with the state carry dropped.
//!
//! Placement: the chunk-wise driver and its scratch live in `pegainfer-qwen35`
//! (`pegainfer-kernels` exposes only the raw per-stage FFI), so the reader sits
//! beside the operator it drives instead of re-sequencing stages over FFI.
//!
//! Provenance is asserted on every load, never skipped: the metadata key must
//! parse, carry `format_version` 2 and the sha256 of
//! `pegainfer-qwen4/tests/frozen_config.json`, name the reference version the
//! semantics are frozen against, and that config's GDN geometry must equal the
//! shapes this gate compiles for. A missing device is likewise a failure, not a
//! skip: this binary only builds under `--features qwen35`, which already needs
//! the GPU toolchain.
//!
//!   ./docker/dev.sh run cargo test --release --locked -p pegainfer-qwen35 \
//!     --features qwen35 --test qwen38fn_gdn_prefill_parity -- --nocapture

#![cfg(feature = "qwen35")]

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::GatedNormActivation;
use pegainfer_kernels::ops::rms_norm_gated_batch_into;
use pegainfer_qwen35::prefill_buffers::GdrChunkwiseScratch35;
use pegainfer_qwen35::runtime_ops as ops;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-gdn-golden.safetensors";
const METADATA_KEY: &str = "qwen4_gdn_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";

/// The frozen config's GDN geometry, which is also what the Triton AOT chunk
/// kernels are compiled for (head dims 128) and what the fixture's shapes are
/// held against in [`assert_provenance`].
const NUM_KEY_HEADS: usize = 16;
const NUM_VALUE_HEADS: usize = 48;
const KEY_DIM: usize = 128;
const VALUE_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
/// `rms_norm_eps`, the eps the reference's gated norm is built with.
const NORM_EPS: f32 = 1e-6;

const KEY_TOTAL: usize = NUM_KEY_HEADS * KEY_DIM;
const QKV_DIM: usize = 2 * KEY_TOTAL + NUM_VALUE_HEADS * VALUE_DIM;
const Z_DIM: usize = NUM_VALUE_HEADS * VALUE_DIM;
const STATE_LEN: usize = NUM_VALUE_HEADS * KEY_DIM * VALUE_DIM;

/// The reference's chunk rule pads to 64 and carries the state across chunks;
/// the first two cases are single-chunk, the last two cross the boundary.
const CASES: [usize; 4] = [16, 64, 80, 128];
const CHUNK: usize = 64;

/// Gated-norm output parity bound, `abs(got - want) <= ATOL + RTOL * abs(want)`.
///
/// Both sides are bf16, but they do not round the same number of times. The
/// reference casts the normalized value to bf16, multiplies it by its bf16
/// weight and only then applies `sigmoid(z)`, while this kernel keeps the whole
/// `x * inv_rms * weight * sigmoid(z)` product in fp32 and casts once; upstream
/// of the norm the chunk stages hold their `a_inv`/`w`/`u`/`v_new` intermediates
/// in bf16 where the reference keeps fp32, over 64-term dot products.
///
/// Measured on this host (RTX 5060, sm_120) over the four cases, against outputs
/// whose absmax is 5.1..7.0. "budget used" is the worst element's
/// `abs(got-want) / (ATOL + RTOL*abs(want))`, the fraction of its allowance the
/// tightest element eats (the report prints it per case):
///
/// | case | chunks | max abs diff | worst budget used | out of bound |
/// | ---- | ------ | ------------ | ----------------- | ------------ |
/// | S=16 | 1 | 4.69e-2 | 27.8% | 0 / 98304 |
/// | S=64 | 1 | 4.69e-2 | 22.3% | 0 / 393216 |
/// | S=80 | 2 | 6.25e-2 | 30.1% | 0 / 491520 |
/// | S=128 | 2 | 6.25e-2 | 25.5% | 0 / 786432 |
///
/// No element leaves the bound in any case, and the worst is two bf16 ulps of the
/// value carrying it (6.25e-2 at 4.47 and at 5.81, where one ulp is 3.125e-2):
/// rounding-order noise, not a systematic bias — the multi-chunk cases show the
/// same shape as the single-chunk ones, so the carried state adds no drift of its
/// own. The tightest element uses 30.1% of its allowance, leaving ~3.3x headroom;
/// at `want = 0` the bound is `ATOL` = 1.5e-1, 2.4x the worst max-abs. The margin
/// is kept wide on purpose: the recurrent state is f32 but the stored `out` is
/// bf16, so the comparison is quantized to bf16 ulps of values up to ~7, and a
/// bound near the noise floor would flake on any toolchain that reorders a dot
/// product without being wrong.
///
/// A real fault costs one to three orders more and (except the decayed carry)
/// spreads instead of touching a handful: the silu gate moves elements by
/// 2.1e1..3.7e1, a skipped conv leg by 6.2..9.1, and a dropped state carry by
/// ~3.7 in the tail chunk that lost it (see the negative controls, which assert
/// the magnitude and, per control, the spread).
const ATOL: f32 = 1.5e-1;
const RTOL: f32 = 2.5e-2;

/// Conv-leg bound, same form. Measured **bit-exact** — max abs diff 0.0 over all
/// 2.9M compared elements in the four cases: the kernel accumulates the four taps
/// in fp32, rounds the sum to bf16 and applies silu to the rounded value, which
/// is the order the reference's `causal_conv1d_fn` fallback uses. The bound is
/// therefore not a noise allowance but a fuse: it only gives a future toolchain
/// room to move the last bit, and any semantic change (a tap order, the causal
/// padding, the head split, a missing silu) is orders past it.
const CONV_ATOL: f32 = 1.0e-3;
const CONV_RTOL: f32 = 1.0e-3;

/// A negative control has to fail the comparison, and it has to fail it for the
/// reason the control names. Two floors per control: how far past the bound the
/// worst element lands (`CONTROL_MIN_WORST_RATIO`, shared), and how much of the
/// output leaves the bound (a per-control fraction). The worst-ratio floor is the
/// primary discriminator — every control drives its worst element far past the
/// bound — while the fraction floor separates a real stage fault from a local
/// perturbation. The fraction is per-control because the faults are not equally
/// wide, and pinning each to its own measured slice is what keeps the assertion
/// honest rather than vacuous:
///
/// | control | worst element past bound | elements out of bound |
/// | ------- | ------------------------ | --------------------- |
/// | silu gate (S=16) | 88x | 24.8% |
/// | conv leg skipped | 29..34x | 52.7..53.3% |
/// | state carry dropped (tail half) | 15.4x | 0.51% |
///
/// The dropped-carry fault is genuinely narrow: the gate decays the carried state
/// over 64 tokens, so only the tail elements the state still moves leave the
/// bound — but they leave it by 15x, and the control separately asserts the
/// *first* half still matches bit-for-bit, which is what pins the miss on the
/// dropped carry rather than on the split. Each fraction floor sits ~2x below its
/// measured slice, so a future toolchain that softens a fault slightly still trips
/// it while rounding noise (a handful of elements, worst ratio < 1) never would.
const CONTROL_MIN_WORST_RATIO: f32 = 10.0;
const CONTROL_SILU_MIN_FRACTION: f32 = 0.10;
const CONTROL_CONV_MIN_FRACTION: f32 = 0.25;
const CONTROL_STATE_MIN_FRACTION: f32 = 0.002;

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn workspace_path(relative: &str) -> String {
    format!("{}/../{relative}", env!("CARGO_MANIFEST_DIR"))
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
/// reference version, the frozen config's bytes and the GDN geometry this gate
/// compiles for. A fixture from another checkpoint, another layout or another
/// reference version fails here instead of being replayed.
fn assert_provenance(bytes: &[u8]) {
    let (_, metadata) = SafeTensors::read_metadata(bytes).expect("fixture metadata");
    let recorded = metadata
        .metadata()
        .as_ref()
        .and_then(|map| map.get(METADATA_KEY))
        .unwrap_or_else(|| {
            panic!(
                "fixture carries no `{METADATA_KEY}` metadata key; regenerate it with \
                 tools/accuracy/dump_qwen38fn_gdn_golden.py"
            )
        });
    let manifest: serde_json::Value = serde_json::from_str(recorded)
        .unwrap_or_else(|e| panic!("`{METADATA_KEY}` metadata is not JSON: {e}"));

    assert_eq!(
        manifest["format_version"].as_u64(),
        Some(FORMAT_VERSION),
        "fixture format_version is not {FORMAT_VERSION}: {manifest}"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG),
        "fixture was not dumped from {CONFIG}: {manifest}"
    );
    assert_eq!(
        manifest["transformers_version"].as_str(),
        Some(REF_TRANSFORMERS),
        "fixture was not dumped from transformers {REF_TRANSFORMERS}: {manifest}"
    );

    let config_bytes =
        std::fs::read(workspace_path(CONFIG)).unwrap_or_else(|e| panic!("read {CONFIG}: {e}"));
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(sha256_hex(&config_bytes).as_str()),
        "fixture's config_sha256 does not match the bytes of {CONFIG} in this tree: {manifest}"
    );

    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"));
    let text = &config["text_config"];
    let geometry = [
        ("linear_num_key_heads", NUM_KEY_HEADS),
        ("linear_num_value_heads", NUM_VALUE_HEADS),
        ("linear_key_head_dim", KEY_DIM),
        ("linear_value_head_dim", VALUE_DIM),
        ("linear_conv_kernel_dim", CONV_KERNEL),
    ];
    for (key, expected) in geometry {
        assert_eq!(
            text[key].as_u64(),
            Some(expected as u64),
            "{CONFIG} moved {key} off the geometry this gate compiles for"
        );
    }
    assert_eq!(
        text["output_gate_type"].as_str(),
        Some("sigmoid"),
        "{CONFIG} moved output_gate_type off sigmoid, which is the gate this fixture records"
    );
    assert_eq!(
        text["rms_norm_eps"].as_f64().map(|eps| eps as f32),
        Some(NORM_EPS),
        "{CONFIG} moved rms_norm_eps off the eps this gate passes the gated norm"
    );
}

fn load_fixture() -> Vec<u8> {
    let path = fixture_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("read {path}: {e} — the golden is committed, so this is not a skip")
    });
    assert_provenance(&bytes);
    bytes
}

fn checked_view<'data>(
    st: &SafeTensors<'data>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> TensorView<'data> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture tensor {name}: {e}"));
    assert_eq!(view.dtype(), dtype, "{name} dtype");
    assert_eq!(view.shape(), shape, "{name} shape");
    view
}

fn bf16_tensor(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Vec<bf16> {
    let view = checked_view(st, name, Dtype::BF16, shape);
    view.data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn f32_tensor(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Vec<f32> {
    let view = checked_view(st, name, Dtype::F32, shape);
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// One recorded case: the activations at the reuse boundary and the reference's
/// gated-norm output.
struct Case {
    seq_len: usize,
    /// Pre-conv `in_proj_qkv` output, [seq_len, QKV_DIM].
    mixed_qkv: Vec<bf16>,
    /// Post-conv, pre-repeat q/k/v — what the conv leg is held against.
    query: Vec<bf16>,
    key: Vec<bf16>,
    value: Vec<bf16>,
    /// Output gate, [seq_len, Z_DIM].
    z: Vec<bf16>,
    /// Derived gate pair, [seq_len, NUM_VALUE_HEADS].
    g: Vec<f32>,
    beta: Vec<bf16>,
    /// The reference's gated-norm output, [seq_len, Z_DIM].
    out: Vec<bf16>,
}

impl Case {
    fn load(st: &SafeTensors<'_>, index: usize) -> Self {
        let name = format!("case{index}_seq_len");
        let seq_len = {
            let view = checked_view(st, &name, Dtype::I32, &[1]);
            i32::from_le_bytes(view.data().try_into().expect("seq_len is 4 bytes")) as usize
        };
        assert_eq!(
            CASES.get(index),
            Some(&seq_len),
            "case{index} records S={seq_len}, the fixture layout expects {CASES:?}"
        );
        let p = format!("case{index}_");
        Self {
            seq_len,
            mixed_qkv: bf16_tensor(st, &format!("{p}mixed_qkv"), &[1, seq_len, QKV_DIM]),
            query: bf16_tensor(
                st,
                &format!("{p}query"),
                &[1, seq_len, NUM_KEY_HEADS, KEY_DIM],
            ),
            key: bf16_tensor(
                st,
                &format!("{p}key"),
                &[1, seq_len, NUM_KEY_HEADS, KEY_DIM],
            ),
            value: bf16_tensor(
                st,
                &format!("{p}value"),
                &[1, seq_len, NUM_VALUE_HEADS, VALUE_DIM],
            ),
            z: bf16_tensor(
                st,
                &format!("{p}z"),
                &[1, seq_len, NUM_VALUE_HEADS, VALUE_DIM],
            ),
            g: f32_tensor(st, &format!("{p}g"), &[1, seq_len, NUM_VALUE_HEADS]),
            beta: bf16_tensor(st, &format!("{p}beta"), &[1, seq_len, NUM_VALUE_HEADS]),
            out: bf16_tensor(
                st,
                &format!("{p}out"),
                &[1, seq_len, NUM_VALUE_HEADS, VALUE_DIM],
            ),
        }
    }

    /// The recorded post-conv activation in the conv output's own layout:
    /// `[q | k | v]` per token, q/k still pre-repeat.
    fn conv_expectation(&self) -> Vec<bf16> {
        let s = self.seq_len;
        let mut rows = Vec::with_capacity(s * QKV_DIM);
        for t in 0..s {
            rows.extend_from_slice(&self.query[t * KEY_TOTAL..(t + 1) * KEY_TOTAL]);
            rows.extend_from_slice(&self.key[t * KEY_TOTAL..(t + 1) * KEY_TOTAL]);
            rows.extend_from_slice(&self.value[t * Z_DIM..(t + 1) * Z_DIM]);
        }
        rows
    }

    /// The rows `[from, to)` as a case of their own, for the split replay.
    fn rows(&self, from: usize, to: usize) -> Self {
        let span = to - from;
        let slice = |whole: &[bf16], width: usize| -> Vec<bf16> {
            whole[from * width..to * width].to_vec()
        };
        Self {
            seq_len: span,
            mixed_qkv: slice(&self.mixed_qkv, QKV_DIM),
            query: slice(&self.query, KEY_TOTAL),
            key: slice(&self.key, KEY_TOTAL),
            value: slice(&self.value, Z_DIM),
            z: slice(&self.z, Z_DIM),
            g: self.g[from * NUM_VALUE_HEADS..to * NUM_VALUE_HEADS].to_vec(),
            beta: slice(&self.beta, NUM_VALUE_HEADS),
            out: slice(&self.out, Z_DIM),
        }
    }
}

/// Whether the conv leg runs, or the pre-conv activations go straight into the
/// chunk stages (negative control).
#[derive(Clone, Copy)]
enum ConvLeg {
    Apply,
    Skip,
}

struct Replay {
    /// Gated-norm output, [seq_len, Z_DIM] as f32.
    out: Vec<f32>,
    /// The conv leg's output, [seq_len, QKV_DIM] as f32 — the pre-conv input
    /// when the leg was skipped.
    conv: Vec<f32>,
}

/// Replay `rows` tokens of a case through the in-tree GDN prefill stack.
///
/// `conv_state` and `state` are the caller's so a split replay can carry the
/// conv history while dropping the recurrent state, which is exactly what the
/// no-carry control has to isolate.
#[allow(clippy::too_many_arguments)]
fn replay(
    ctx: &DeviceContext,
    case: &Case,
    conv_weight: &DeviceVec,
    norm_weight: &CudaSlice<f32>,
    conv_state: &mut DeviceVec,
    state: &mut CudaSlice<f32>,
    conv: ConvLeg,
    activation: GatedNormActivation,
) -> Result<Replay> {
    let rows = case.seq_len;
    let qkv = HiddenStates {
        data: ctx.stream.clone_htod(&case.mixed_qkv)?,
        hidden_dim: QKV_DIM,
        seq_len: rows,
    };
    let mut conv_out = HiddenStates::zeros(ctx, QKV_DIM, rows)?;
    if let ConvLeg::Apply = conv {
        ops::conv1d_prefill_batch_into(
            ctx,
            &qkv,
            conv_weight,
            conv_state,
            &mut conv_out,
            CONV_KERNEL,
        );
    }
    let prepared_qkv = match conv {
        ConvLeg::Apply => &conv_out,
        ConvLeg::Skip => &qkv,
    };
    let conv_host: Vec<f32> = ctx
        .stream
        .clone_dtoh(&prepared_qkv.data)?
        .iter()
        .map(|x| x.to_f32())
        .collect();

    // The prepare stage's own gate derivation has no inputs here: the fixture
    // starts after it. Zeroes keep the launch deterministic, and the recorded
    // pair below overwrites both outputs.
    let zeros = vec![bf16::ZERO; rows * NUM_VALUE_HEADS];
    let b_proj = HiddenStates {
        data: ctx.stream.clone_htod(&zeros)?,
        hidden_dim: NUM_VALUE_HEADS,
        seq_len: rows,
    };
    let a_proj = HiddenStates {
        data: ctx.stream.clone_htod(&zeros)?,
        hidden_dim: NUM_VALUE_HEADS,
        seq_len: rows,
    };
    let dt_bias = DeviceVec::from_host(ctx, &[bf16::ZERO; NUM_VALUE_HEADS])?;
    let a_log: CudaSlice<f32> = ctx.stream.alloc_zeros(NUM_VALUE_HEADS)?;
    let mut scratch =
        GdrChunkwiseScratch35::from_dims(ctx, NUM_VALUE_HEADS, KEY_DIM, VALUE_DIM, rows)?;
    ops::gated_delta_rule_prefill_chunk_prepare_into(
        ctx,
        prepared_qkv,
        &b_proj,
        &a_proj,
        &dt_bias,
        &a_log,
        &mut scratch,
        NUM_KEY_HEADS,
        NUM_VALUE_HEADS,
    )?;

    ctx.stream.memcpy_htod(&case.g, &mut scratch.g_cumsum)?;
    let beta: Vec<f32> = case.beta.iter().map(|b| b.to_f32()).collect();
    ctx.stream.memcpy_htod(&beta, &mut scratch.beta)?;

    let mut core = HiddenStates::zeros(ctx, Z_DIM, rows)?;
    ops::gated_delta_rule_prefill_chunkwise_from_gates_into(
        ctx,
        &mut scratch,
        state,
        &mut core,
        NUM_VALUE_HEADS,
        KEY_DIM,
        VALUE_DIM,
    )?;

    let z = HiddenStates {
        data: ctx.stream.clone_htod(&case.z)?,
        hidden_dim: Z_DIM,
        seq_len: rows,
    };
    let mut normed = HiddenStates::zeros(ctx, Z_DIM, rows)?;
    rms_norm_gated_batch_into(
        ctx,
        &core,
        norm_weight,
        &z,
        &mut normed,
        NUM_VALUE_HEADS,
        VALUE_DIM,
        NORM_EPS,
        activation,
    );
    ctx.sync()?;

    let out = ctx
        .stream
        .clone_dtoh(&normed.data)?
        .iter()
        .map(|x| x.to_f32())
        .collect();
    Ok(Replay {
        out,
        conv: conv_host,
    })
}

/// One replay of a whole case, from a fresh conv state and a fresh recurrent
/// state, the way the reference's `cache_params=None` prefill runs.
fn replay_case(
    ctx: &DeviceContext,
    case: &Case,
    weights: &Weights<'_>,
    conv: ConvLeg,
    activation: GatedNormActivation,
) -> Result<Replay> {
    let mut conv_state = DeviceVec::zeros(ctx, QKV_DIM * (CONV_KERNEL - 1))?;
    let mut state: CudaSlice<f32> = ctx.stream.alloc_zeros(STATE_LEN)?;
    replay(
        ctx,
        case,
        weights.conv1d,
        weights.norm,
        &mut conv_state,
        &mut state,
        conv,
        activation,
    )
}

struct Weights<'a> {
    conv1d: &'a DeviceVec,
    norm: &'a CudaSlice<f32>,
}

/// `abs(got - want)` against the `ATOL + RTOL * abs(want)` bound.
struct Delta {
    max_abs: f32,
    at: usize,
    /// Elements outside the bound.
    violations: usize,
    /// Worst `abs(got - want) / (ATOL + RTOL * abs(want))` — the fraction of the
    /// tolerance budget the closest element eats, so a pass says how much headroom
    /// is left (`< 1.0`) and a failure says how far past the bound it is.
    worst_ratio: f32,
    /// The element that reaches `worst_ratio`, with its own `abs(got - want)` and
    /// its bound, so the report names the budget actually used rather than only the
    /// max-abs element (the two differ because the bound scales with `abs(want)`).
    worst_ratio_at: usize,
    worst_ratio_abs: f32,
    worst_ratio_bound: f32,
    /// The bound parameters, echoed so a passing line still records its tolerance.
    atol: f32,
    rtol: f32,
}

fn compare(got: &[f32], want: &[bf16], atol: f32, rtol: f32) -> Delta {
    assert_eq!(got.len(), want.len(), "compared runs differ in length");
    let mut delta = Delta {
        max_abs: 0.0,
        at: 0,
        violations: 0,
        worst_ratio: 0.0,
        worst_ratio_at: 0,
        worst_ratio_abs: 0.0,
        worst_ratio_bound: 0.0,
        atol,
        rtol,
    };
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = w.to_f32();
        let d = (g - w).abs();
        // A NaN compares false against everything, so it would leave the worst
        // at zero: fail here instead of passing a comparison silently.
        assert!(d.is_finite(), "element {i}: {g} against {w}");
        if d > delta.max_abs {
            delta.max_abs = d;
            delta.at = i;
        }
        let bound = atol + rtol * w.abs();
        if d > bound {
            delta.violations += 1;
        }
        let ratio = d / bound;
        if ratio > delta.worst_ratio {
            delta.worst_ratio = ratio;
            delta.worst_ratio_at = i;
            delta.worst_ratio_abs = d;
            delta.worst_ratio_bound = bound;
        }
    }
    delta
}

/// House report line: per-case max-abs, the bound it was held against, and how
/// much of that bound the closest element used.
fn report(what: &str, delta: &Delta, got: &[f32], want: &[bf16]) {
    let pct = 100.0 * delta.violations as f32 / want.len() as f32;
    println!(
        "{what}: max_abs={:.3e} at {} (got={:.6} want={:.6}) | bound {:.2e}+{:.2e}*|want| \
         -> tightest element {} used {:.3e} of {:.3e} = {:.1}% of budget | out-of-bound {}/{} \
         ({:.2}%)",
        delta.max_abs,
        delta.at,
        got[delta.at],
        want[delta.at].to_f32(),
        delta.atol,
        delta.rtol,
        delta.worst_ratio_at,
        delta.worst_ratio_abs,
        delta.worst_ratio_bound,
        100.0 * delta.worst_ratio,
        delta.violations,
        want.len(),
        pct,
    );
}

/// The controls assert the comparison *fails*, and that it fails the way the
/// named fault fails: the worst element lands far past the bound
/// (`CONTROL_MIN_WORST_RATIO`), and the slice of the output that leaves the bound
/// reaches the control's own floor. The floor is per-control because the faults
/// are not equally wide — a wrong activation or a skipped conv rewrites most of
/// the output, while a dropped state carry only moves the tail elements the decayed
/// state still reaches — but every floor sits well above rounding noise, so a
/// handful of bf16-ulp elements never satisfies a control.
fn assert_control_failed(what: &str, delta: &Delta, got: &[f32], want: &[bf16], min_fraction: f32) {
    report(what, delta, got, want);
    let fraction = delta.violations as f32 / want.len() as f32;
    assert!(
        delta.worst_ratio >= CONTROL_MIN_WORST_RATIO,
        "{what} was supposed to break the comparison, but its worst element is only {:.2}x past \
         the bound (max_abs {:.3e}) — that is rounding noise, not this fault",
        delta.worst_ratio,
        delta.max_abs
    );
    assert!(
        fraction >= min_fraction,
        "{what} was supposed to break the comparison: only {}/{} elements ({:.1}%) left the \
         bound (worst_ratio {:.2}, max_abs {:.3e}), which is a local perturbation, not a fault \
         in the stage this control removes",
        delta.violations,
        want.len(),
        fraction * 100.0,
        delta.worst_ratio,
        delta.max_abs
    );
}

fn load_weights(ctx: &DeviceContext, st: &SafeTensors<'_>) -> Result<(DeviceVec, CudaSlice<f32>)> {
    let conv1d = bf16_tensor(st, "conv1d_weight", &[QKV_DIM, 1, CONV_KERNEL]);
    let conv1d = DeviceVec::from_host(ctx, &conv1d)?;
    // The gated norm takes its weight in f32; the fixture stores the reference's
    // bf16 parameter, which upcasts exactly.
    let norm: Vec<f32> = bf16_tensor(st, "norm_weight", &[VALUE_DIM])
        .iter()
        .map(|w| w.to_f32())
        .collect();
    let norm = ctx.stream.clone_htod(&norm)?;
    Ok((conv1d, norm))
}

#[test]
fn conv1d_prefill_leg_matches_recorded_post_conv_qkv() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let bytes = load_fixture();
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (conv1d, norm) = load_weights(&ctx, &st)?;
    let weights = Weights {
        conv1d: &conv1d,
        norm: &norm,
    };

    for (index, seq_len) in CASES.iter().enumerate() {
        let case = Case::load(&st, index);
        let replay = replay_case(
            &ctx,
            &case,
            &weights,
            ConvLeg::Apply,
            GatedNormActivation::Sigmoid,
        )?;
        let want = case.conv_expectation();
        let delta = compare(&replay.conv, &want, CONV_ATOL, CONV_RTOL);
        report(
            &format!("case{index} S={seq_len} conv leg"),
            &delta,
            &replay.conv,
            &want,
        );
        assert_eq!(
            delta.violations, 0,
            "case{index} S={seq_len}: the causal conv1d prefill kernel does not reproduce the \
             reference's post-conv q/k/v (worst_ratio {:.2}); the head split, the tap order, the \
             causal zero padding or the silu-on-rounded-sum order moved",
            delta.worst_ratio
        );
    }
    Ok(())
}

#[test]
fn gdn_prefill_replay_matches_recorded_gated_norm_output() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let bytes = load_fixture();
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (conv1d, norm) = load_weights(&ctx, &st)?;
    let weights = Weights {
        conv1d: &conv1d,
        norm: &norm,
    };

    for (index, seq_len) in CASES.iter().enumerate() {
        let case = Case::load(&st, index);
        let replay = replay_case(
            &ctx,
            &case,
            &weights,
            ConvLeg::Apply,
            GatedNormActivation::Sigmoid,
        )?;
        let delta = compare(&replay.out, &case.out, ATOL, RTOL);
        let chunks = seq_len.div_ceil(CHUNK);
        report(
            &format!("case{index} S={seq_len} ({chunks} chunk(s)) gated-norm out"),
            &delta,
            &replay.out,
            &case.out,
        );
        assert!(
            replay.out.iter().all(|x| x.is_finite()),
            "case{index} S={seq_len}: replay produced a non-finite output"
        );
        assert_eq!(
            delta.violations, 0,
            "case{index} S={seq_len}: the in-tree GDN prefill stack does not reproduce the \
             reference's gated-norm output across {chunks} chunk(s) (worst_ratio {:.2}). For the \
             multi-chunk cases the carried recurrent state is part of what is being compared.",
            delta.worst_ratio
        );
    }
    Ok(())
}

/// The trap this checkpoint sets: `output_gate_type` is `"sigmoid"`, while the
/// Qwen3.5 Gated DeltaNet the kernels were written for gates with silu. Same
/// kernel, same weights, wrong activation.
#[test]
fn negative_control_silu_gate_breaks_the_comparison() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let bytes = load_fixture();
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (conv1d, norm) = load_weights(&ctx, &st)?;
    let weights = Weights {
        conv1d: &conv1d,
        norm: &norm,
    };

    for (index, seq_len) in CASES.iter().enumerate() {
        let case = Case::load(&st, index);
        let replay = replay_case(
            &ctx,
            &case,
            &weights,
            ConvLeg::Apply,
            GatedNormActivation::Silu,
        )?;
        let delta = compare(&replay.out, &case.out, ATOL, RTOL);
        assert_control_failed(
            &format!("case{index} S={seq_len} silu gate"),
            &delta,
            &replay.out,
            &case.out,
            CONTROL_SILU_MIN_FRACTION,
        );
    }
    Ok(())
}

/// Without the depthwise conv the chunk stages see the raw `in_proj_qkv` output:
/// no four-token history and no silu, so nothing downstream can land.
#[test]
fn negative_control_skipped_conv_leg_breaks_the_comparison() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let bytes = load_fixture();
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (conv1d, norm) = load_weights(&ctx, &st)?;
    let weights = Weights {
        conv1d: &conv1d,
        norm: &norm,
    };

    for (index, seq_len) in CASES.iter().enumerate() {
        let case = Case::load(&st, index);
        let replay = replay_case(
            &ctx,
            &case,
            &weights,
            ConvLeg::Skip,
            GatedNormActivation::Sigmoid,
        )?;
        let delta = compare(&replay.out, &case.out, ATOL, RTOL);
        assert_control_failed(
            &format!("case{index} S={seq_len} conv leg skipped"),
            &delta,
            &replay.out,
            &case.out,
            CONTROL_CONV_MIN_FRACTION,
        );
    }
    Ok(())
}

/// The multi-chunk cases only mean something if the state actually crosses the
/// boundary: S=128 replayed as two independent 64-row runs, each from a zero
/// recurrent state, must miss the recorded output — and it must miss it in the
/// second half only, which is what pins the fault on the dropped carry rather
/// than on the split itself. The conv history *is* carried here, so the conv leg
/// stays bit-identical to the whole-sequence run.
#[test]
fn negative_control_dropped_state_carry_breaks_the_comparison() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let bytes = load_fixture();
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (conv1d, norm) = load_weights(&ctx, &st)?;

    let index = CASES
        .iter()
        .position(|s| *s == 2 * CHUNK)
        .expect("a 2-chunk case");
    let case = Case::load(&st, index);
    let seq_len = case.seq_len;

    let mut conv_state = DeviceVec::zeros(&ctx, QKV_DIM * (CONV_KERNEL - 1))?;
    let mut out: Vec<f32> = Vec::with_capacity(seq_len * Z_DIM);
    for half in 0..2 {
        let rows = case.rows(half * CHUNK, (half + 1) * CHUNK);
        // A fresh zero state per half: this is the carry being dropped.
        let mut state: CudaSlice<f32> = ctx.stream.alloc_zeros(STATE_LEN)?;
        let replay = replay(
            &ctx,
            &rows,
            &conv1d,
            &norm,
            &mut conv_state,
            &mut state,
            ConvLeg::Apply,
            GatedNormActivation::Sigmoid,
        )?;
        out.extend_from_slice(&replay.out);
    }

    let first = out[..CHUNK * Z_DIM].to_vec();
    let second = out[CHUNK * Z_DIM..].to_vec();
    let want_first = case.out[..CHUNK * Z_DIM].to_vec();
    let want_second = case.out[CHUNK * Z_DIM..].to_vec();

    let head = compare(&first, &want_first, ATOL, RTOL);
    report(
        &format!("case{index} S={seq_len} split rows [0,{CHUNK})"),
        &head,
        &first,
        &want_first,
    );
    assert_eq!(
        head.violations, 0,
        "case{index}: the first chunk of the split replay already misses, so this control would \
         not be isolating the state carry (worst_ratio {:.2})",
        head.worst_ratio
    );

    let tail = compare(&second, &want_second, ATOL, RTOL);
    assert_control_failed(
        &format!(
            "case{index} S={seq_len} split rows [{CHUNK},{seq_len}) without the carried state"
        ),
        &tail,
        &second,
        &want_second,
        CONTROL_STATE_MIN_FRACTION,
    );
    Ok(())
}
