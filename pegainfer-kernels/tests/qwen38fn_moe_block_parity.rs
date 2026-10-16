//! Qwen3.8-Flash-Next MoE block against the frozen reference golden, on device.
//!
//! The device half of the consumer for
//! `test_data/qwen38fn-moe-golden.safetensors` (dumped by
//! `tools/accuracy/dump_qwen38fn_moe_golden.py` from transformers 5.16.1
//! `Qwen4ExpTextSparseMoeBlock` at the frozen checkpoint's real geometry:
//! hidden 2560, 512 experts, top-10 routed + 1 shared, shared intermediate
//! 640). The device-free half — `pegainfer-qwen4/tests/moe_block_replay.rs` —
//! pins the same chain in f32 on the host and stays CI-runnable without a GPU;
//! this gate drives the *production* device path through its ops layer:
//!
//! | step | production op | against the golden |
//! |------|---------------|--------------------|
//! | router | `ops::gemm_into_checked` (cuBLAS bf16, f32 accum) | `LOGIT_ULPS` bf16 ULP |
//! | routing | `ops::qwen38fn_moe_route_into` (new kernel) | bitwise weights, canonical tie rule |
//! | router→routing chain | the same two, composed | bitwise on the device's own logits; the golden's tie rows may re-tie |
//! | shared | `ops::qwen38fn_moe_shared_branch_into` (GEMMs + `silu_mul_batch_into`) | 1 ULP + the derived quadrature floor |
//! | combine | `ops::qwen38fn_moe_combine_into` (new kernel) | bitwise-grade; see `OUT_ULPS` |
//!
//! Provenance is asserted, never skipped: the fixture's metadata manifest must
//! parse, carry `format_version` 2, name `pegainfer-qwen4/tests/frozen_config.json`
//! as its config, and carry that file's sha256 — the same contract the
//! device-free gate holds. Without a device the device tests skip;
//! `PEGAINFER_REQUIRE_GPU=1` turns that into a failure.
//!
//! NOT gated here: `expert_out` stays a recorded activation — the fused
//! 512-expert stack (~4.7 GiB bf16) is deliberately not in the fixture, so the
//! routed-expert GEMM waits for the MoE implementation (#1107). The combine
//! reads the golden's stored `expert_out` exactly like the device-free gate.
//!
//! Three negative controls, each asserting the comparison FAILS, run through
//! the same production entry points: skipping the top-k renormalization
//! (`norm_topk_prob` absent from `config.json`, defaulting to True upstream),
//! carrying the softmax probabilities in bf16 instead of the reference's fp32
//! upcast, and dropping the shared expert's learned scalar sigmoid gate.
//!
//! Those are three of the device-free gate's **five** traps — the three that
//! have a device-side knob. Its other two, `control_gelu_instead_of_silu_fails`
//! and `control_unrounded_router_logits_fails`, are not ported and cannot be:
//! on device the SwiGLU is the production `silu_mul_batch_into` (SiLU is
//! hardwired, no activation switch) and the router logits are the production
//! GEMM's bf16 store (there is no f32-logit variant to misuse), so neither
//! mistake is reachable through this crate's entry points. The device-free
//! replay stays their gate.
//!
//! ```text
//! ./docker/dev.sh run cargo test --release --locked -p pegainfer-kernels \
//!   --features qwen4 --test qwen38fn_moe_block_parity -- --nocapture
//! ```

#![cfg(feature = "qwen4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::Qwen38fnMoeGeometry;
use pegainfer_kernels::ops::Qwen38fnRouteSemantics;
use pegainfer_kernels::ops::Qwen38fnSharedBranchScratch;
use pegainfer_kernels::ops::add_into;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::qwen38fn_moe_combine_into;
use pegainfer_kernels::ops::qwen38fn_moe_route_into;
use pegainfer_kernels::ops::qwen38fn_moe_shared_branch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-moe-golden.safetensors";
const METADATA_KEY: &str = "qwen4_moe_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";
/// Two prefill batches and one decode row; token counts come from the shapes.
const CASES: usize = 3;

// ---------------------------------------------------------------------------
// Tolerances — measured on an RTX 5060 (sm_120), CUDA 12.9.1, cuBLAS bf16
// with f32 accumulation. Every bound states what was measured against it.
// ---------------------------------------------------------------------------

/// bf16's worst-case relative ULP: consecutive bf16 values are between 2^-8
/// and 2^-7 apart relatively, so 2^-7 bounds one rounding step (the device-free
/// gate's constant).
const BF16_RELATIVE_ULP: f64 = 1.0 / 128.0;

/// Router logits, cuBLAS against the golden's torch CPU GEMM. The relative
/// term is the golden's own single bf16 output rounding; the absolute term is
/// cuBLAS's accumulation order, which at K=2560 on sm_120 moves near-zero
/// logits by more than their (tiny) own ULP — measured worst 0.03125 = 2^-5
/// absolute over the three cases, so the floor is 2^-4, twice the measurement.
/// The device-free gate, whose GEMM shares torch CPU's order, needed no floor
/// (1 ULP on 8 of 10752); this is the "measure, don't assume 1 ULP" case.
const LOGIT_ULPS: i64 = 1;
const LOGIT_ATOL: f64 = 0.0625;

/// Routing weights, device kernel against the host canonical route over the
/// *same* logits: both compute p/Σ in f32 and round once, so the residual is
/// `expf` and the softmax denominator's summation order — ~2^-20 relative,
/// invisible after the bf16 store unless a weight sits at a rounding midpoint.
/// Measured: bitwise on 210/210; the bound keeps 1 ULP of room for a re-dump.
const ROUTE_WEIGHT_ULPS: i64 = 1;

/// Chain bounds, derived from `LOGIT_ATOL` (call it `a`), not fitted: a
/// renormalized weight is `e^l_j / Σ_sel e^l_i` — the softmax's global
/// denominator cancels — so a logit perturbation |δ| ≤ a moves it by at most
/// e^{2a} − 1 ≈ 0.133 relative (numerator and denominator each shift by ≤ a
/// in opposite directions), and it can move the top-10 boundary only where
/// the golden's 10th and 11th probabilities sit closer than the same factor
/// e^{2a} ≈ 1.133. Measured worst against these: see the printout.
const CHAIN_WEIGHT_REL: f64 = 0.1332; // e^{2·LOGIT_ATOL} − 1, rounded up
const CHAIN_GAP_RATIO: f64 = 1.1332; // e^{2·LOGIT_ATOL}, rounded up

/// Epsilon for *ordering* checks only. The host reference sums 512 exps
/// sequentially and the kernel a fixed tree, so the same probability pair can
/// come out ~2^-20 relative apart in one and exactly equal in the other. That
/// noise (~1e-8 at these magnitudes) is not what sizes this constant — the
/// fixture does. Measured over the golden's 21 tokens, the smallest *nonzero*
/// 10th-vs-11th probability gap is 4.157e-4 (case0 token 1: 0.0135116 vs
/// 0.0130959), so 1e-5 sits 41x under it: wide enough to absorb the summation
/// order, narrow enough that a real boundary crossing still fails. Exact ties
/// are not this constant's business — case1 has three tied rows, and they go
/// through `grade_golden_token`'s weight-matched-set rule instead; the strict
/// canonical tie rule is enforced only on exact equality, where both sides
/// agree by construction.
const ORDER_EPS: f64 = 1.0e-5;

/// Headroom on the shared branch's derived floor. The device-free gate needs
/// 2.0 for one rounding step of difference in its intermediate; the device
/// path's intermediate can carry more: the gate/up projections themselves may
/// land 1 ULP from the host replay's, and `silu(gate)·up` composes those.
/// Three ULPs of intermediate difference times the gate's 2.0 headroom is 6.0.
/// Measured worst fraction of the floored bound: see the printout.
const FLOOR_SAFETY: f64 = 6.0;

/// Block output: the combine's operands are all stored bf16 except the gate
/// sigmoid, whose logit comes off a cuBLAS GEMM (≤ `LOGIT_ULPS` ULP) and whose
/// expf may round apart from the host's. Two ULPs on the sum plus the derived
/// absolute floors in the checks below. Measured: bitwise-grade; see printout.
const OUT_ULPS: i64 = 2;

/// Renormalized top-k weights must sum to 1 within this. Derived, not fitted:
/// each stored weight is `bf16(w_j)` for an f32 `w_j`, one bf16 rounding moves
/// a value by at most half a spacing (2^-9 relative), and the ten `w_j` sum to
/// 1 — so the ten roundings move the sum by at most 2^-9 ≈ 0.00195, the
/// device-free gate's bound. Its measurement over the fixture's 21 tokens is a
/// worst |Σw − 1| of 0.00171, and 0.01 keeps ~5x of room over that for a
/// re-dump. The renorm control is nowhere near that margin: the *tightest* of
/// the 21 un-renormed masses still sits 0.1425 away from 1 (case1 token 8),
/// i.e. 14x this tolerance. The wider 0.34–0.47 range is token 0 of the three
/// cases — the row the controls print, and the widest, not the tightest.
const WEIGHT_SUM_TOL: f64 = 0.01;

// ---------------------------------------------------------------------------
// Host reference — the device-free gate's arithmetic, kept bit-identical to it
// so both consumers of the golden grade with the same yardstick.
// ---------------------------------------------------------------------------

/// A row-major `[rows, cols]` matrix of bf16-representable f32 values.
#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Round to bf16 — one dtype boundary of the reference.
    fn round(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(|&v| f32::from(bf16::from_f32(v)))
                .collect(),
        }
    }

    fn max_abs(&self) -> f32 {
        self.data.iter().map(|v| v.abs()).fold(0f32, f32::max)
    }

    fn to_bf16(&self) -> Vec<bf16> {
        self.data.iter().map(|&v| bf16::from_f32(v)).collect()
    }
}

/// `y = x · wᵀ`, f32 accumulation, unrounded — torch's `F.linear` on bf16
/// operands; the caller rounds where the reference rounds.
fn linear(x: &Mat, w: &Mat) -> Mat {
    let mut out = Mat::zeros(x.rows, w.rows);
    for (xr, orow) in x.data.chunks(x.cols).zip(out.data.chunks_mut(w.rows)) {
        for (slot, wr) in orow.iter_mut().zip(w.data.chunks(w.cols)) {
            let mut acc = 0f32;
            for (a, b) in xr.iter().zip(wr) {
                acc += a * b;
            }
            *slot = acc;
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch's SiLU on a bf16 operand: computed in f32, stored bf16.
fn silu(x: f32) -> f32 {
    f32::from(bf16::from_f32(x * sigmoid(x)))
}

/// The reference's routing softmax: max-subtracted fp32, summed sequentially —
/// bitwise the device-free gate's `softmax_f32`.
fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// The host replay of the shared branch: `down(silu(gate)·up)` with the
/// reference's bf16 rounding at every boundary, plus (a) the per-element
/// quadrature floor that bounds one rounding step of intermediate difference
/// propagated through the down projection, and (b) the per-token scalar gate
/// sigmoid. Bitwise the device-free gate's `shared_branch`.
struct SharedReplay {
    out: Mat,
    floor: Vec<f64>,
    gate_sigmoid: Vec<f32>,
}

fn shared_replay(fixture: &Fixture, case: &Case) -> SharedReplay {
    let gate = linear(&case.hidden, &fixture.shared_gate_proj).round();
    let up = linear(&case.hidden, &fixture.shared_up_proj).round();
    let tokens = case.tokens;
    let inter = fixture.geometry.shared_intermediate;
    let mut intermediate = Mat::zeros(tokens, inter);
    for t in 0..tokens {
        for j in 0..inter {
            let v = silu(gate.at(t, j)) * up.at(t, j);
            intermediate.data[t * inter + j] = f32::from(bf16::from_f32(v));
        }
    }

    let down = &fixture.shared_down_proj;
    let mut shared = Mat::zeros(tokens, fixture.geometry.hidden);
    let mut floor = vec![0f64; tokens * fixture.geometry.hidden];
    for t in 0..tokens {
        let h = intermediate.row(t);
        let frow = &mut floor[t * down.rows..(t + 1) * down.rows];
        for ((slot, w), f) in shared.data[t * down.rows..(t + 1) * down.rows]
            .iter_mut()
            .zip(down.data.chunks(down.cols))
            .zip(frow)
        {
            let mut acc = 0f32;
            let mut sq = 0f64;
            for (&hv, &wv) in h.iter().zip(w) {
                acc += hv * wv;
                let term = f64::from(hv) * f64::from(wv);
                sq += term * term;
            }
            *slot = acc;
            *f = FLOOR_SAFETY * BF16_RELATIVE_ULP * sq.sqrt();
        }
    }

    let gate_logit = linear(&case.hidden, &fixture.shared_expert_gate).round();
    let gate_sigmoid = (0..tokens)
        .map(|t| f32::from(bf16::from_f32(sigmoid(gate_logit.at(t, 0)))))
        .collect();

    SharedReplay {
        out: shared.round(),
        floor,
        gate_sigmoid,
    }
}

fn bf16_ulp(x: f32) -> f64 {
    let mag = bf16::from_f32(x.abs());
    let next = bf16::from_bits(mag.to_bits() + 1);
    if next.is_infinite() {
        return f64::from(mag) * BF16_RELATIVE_ULP;
    }
    f64::from(next) - f64::from(mag)
}

/// bf16 sign-magnitude folded onto a monotone axis (the device-free gate's
/// `ordered_key`), so a bit-pattern distance is a value distance.
fn ordered_key(x: f32) -> i64 {
    let bits = i64::from(bf16::from_f32(x).to_bits());
    if bits & 0x8000 == 0 {
        0x8000 + bits
    } else {
        0x8000 - (bits & 0x7fff)
    }
}

fn ulp_distance(a: f32, b: f32) -> i64 {
    (ordered_key(a) - ordered_key(b)).abs()
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// An element passes when `|got - want| <= ulps · ulp(|want|) + floor[i]`.
struct Tol {
    ulps: i64,
    floor: Vec<f64>,
}

impl Tol {
    const fn ulps(ulps: i64) -> Self {
        Self {
            ulps,
            floor: Vec::new(),
        }
    }

    fn with_floor(ulps: i64, floor: Vec<f64>) -> Self {
        Self { ulps, floor }
    }
}

struct Report {
    label: String,
    total: usize,
    exact: usize,
    worst_ulp: i64,
    worst_abs: f64,
    /// Worst `|diff| / limit`; above 1.0 the element violates the tolerance.
    worst_slack: f64,
    violations: usize,
}

impl Report {
    fn print(&self) {
        println!(
            "  {:<52} n={:<6} bitwise={:<6} worst_ulp={:<4} worst_abs={:.6} worst_bound_frac={:.4} \
             violations={}",
            self.label,
            self.total,
            self.exact,
            self.worst_ulp,
            self.worst_abs,
            self.worst_slack,
            self.violations
        );
    }

    fn assert_clean(&self) {
        self.print();
        assert_eq!(
            self.violations, 0,
            "{}: {} of {} elements outside tolerance (worst abs {:.6}, worst bound fraction {:.4})",
            self.label, self.violations, self.total, self.worst_abs, self.worst_slack
        );
    }
}

#[allow(
    clippy::float_cmp,
    reason = "both sides are bf16-representable, so equality is bitwise equality"
)]
fn compare(label: &str, got: &[f32], want: &[f32], tol: &Tol) -> Report {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {} vs {} elements",
        got.len(),
        want.len()
    );
    assert!(
        tol.floor.is_empty() || tol.floor.len() == want.len(),
        "{label}: floor covers {} of {} elements",
        tol.floor.len(),
        want.len()
    );
    let mut report = Report {
        label: label.to_string(),
        total: want.len(),
        exact: 0,
        worst_ulp: 0,
        worst_abs: 0.0,
        worst_slack: 0.0,
        violations: 0,
    };
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "{label}[{i}]: non-finite device {got} vs golden {want}"
        );
        let abs = (f64::from(got) - f64::from(want)).abs();
        if got == want {
            report.exact += 1;
        }
        report.worst_ulp = report.worst_ulp.max(ulp_distance(got, want));
        report.worst_abs = report.worst_abs.max(abs);
        let limit = tol.ulps as f64 * bf16_ulp(want) + *tol.floor.get(i).unwrap_or(&0.0);
        let slack = if limit > 0.0 {
            abs / limit
        } else if abs == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        report.worst_slack = report.worst_slack.max(slack);
        if abs > limit {
            report.violations += 1;
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct Case {
    index: usize,
    tokens: usize,
    hidden: Mat,
    router_logits: Mat,
    routing_weights: Mat,
    selected_experts: Vec<i32>,
    expert_out: Mat,
    shared_out: Mat,
    out: Mat,
}

struct Fixture {
    geometry: Qwen38fnMoeGeometry,
    gate: Mat,
    shared_gate_proj: Mat,
    shared_up_proj: Mat,
    shared_down_proj: Mat,
    shared_expert_gate: Mat,
    cases: Vec<Case>,
}

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn config_path() -> String {
    format!("{}/../{CONFIG}", env!("CARGO_MANIFEST_DIR"))
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
/// reference version, the frozen config's bytes and the geometry this crate
/// compiles for. Every failure is a hard panic: a golden whose provenance
/// cannot be established must not degrade into a skip. Returns the config bytes.
fn assert_provenance(bytes: &[u8]) -> Vec<u8> {
    let (_, header) = SafeTensors::read_metadata(bytes).expect("parse fixture header");
    let kv = header.metadata().clone().unwrap_or_else(|| {
        panic!(
            "{FIXTURE} carries no __metadata__ map; the dump writes one JSON key \
             ({METADATA_KEY}) — re-dump with tools/accuracy/dump_qwen38fn_moe_golden.py"
        )
    });
    let raw = kv.get(METADATA_KEY).unwrap_or_else(|| {
        panic!(
            "{FIXTURE} metadata has no {METADATA_KEY} key; it carries {:?}",
            kv.keys().collect::<Vec<_>>()
        )
    });
    let manifest: serde_json::Value = serde_json::from_str(raw)
        .unwrap_or_else(|e| panic!("{METADATA_KEY} is not JSON ({e}): {raw}"));

    let version = manifest["format_version"]
        .as_u64()
        .unwrap_or_else(|| panic!("{METADATA_KEY} carries no integer format_version: {manifest}"));
    assert_eq!(
        version, FORMAT_VERSION,
        "fixture format_version {version} is not {FORMAT_VERSION}; this gate does not know that \
         layout (the dump script is the authority)"
    );
    assert_eq!(
        manifest["transformers_version"].as_str(),
        Some(REF_TRANSFORMERS),
        "fixture was dumped from another reference version, and the frozen semantics are that \
         one's"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG),
        "fixture was dumped against config {:?}, not {CONFIG}",
        manifest["config"]
    );
    let config_bytes =
        std::fs::read(config_path()).unwrap_or_else(|e| panic!("read frozen config {CONFIG}: {e}"));
    let sha = sha256_hex(&config_bytes);
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(sha.as_str()),
        "fixture config_sha256 is not sha256({CONFIG}) = {sha}; the golden was dumped from a \
         different checkpoint geometry"
    );

    // The frozen config's own MoE geometry, held against what this crate
    // compiles for — a config that moved must fail here, not re-shape the gate.
    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"));
    let text = &config["text_config"];
    let geometry = Qwen38fnMoeGeometry::qwen38_flash_next();
    let field = |name: &str| {
        text[name].as_u64().unwrap_or_else(|| {
            panic!(
                "{CONFIG} text_config.{name} is not an integer: {:?}",
                text[name]
            )
        }) as usize
    };
    assert_eq!(field("hidden_size"), geometry.hidden, "hidden_size");
    assert_eq!(field("num_experts"), geometry.experts, "num_experts");
    assert_eq!(
        field("num_experts_per_tok"),
        geometry.top_k,
        "num_experts_per_tok"
    );
    assert_eq!(
        field("shared_expert_intermediate_size"),
        geometry.shared_intermediate,
        "shared_expert_intermediate_size"
    );
    assert_eq!(
        text["hidden_act"].as_str(),
        Some("silu"),
        "the shared branch replays SiLU; the frozen config says {:?}",
        text["hidden_act"]
    );
    geometry.validate().expect("the frozen geometry");

    // Recorded, not asserted: the host the golden was dumped on decides the
    // order its f32 reductions round in, which is where the measured bounds'
    // residuals come from.
    println!(
        "golden provenance: format_version {version}, config_sha256 {sha}, seed {}, \
         transformers {}, torch {}, {}",
        manifest["seed"],
        manifest["transformers_version"]
            .as_str()
            .unwrap_or("<missing>"),
        manifest["torch_version"].as_str().unwrap_or("<missing>"),
        manifest["platform"].as_str().unwrap_or("<missing>")
    );
    println!(
        "golden geometry: hidden {}, {} experts, top-{} routed + 1 shared, shared intermediate {}",
        geometry.hidden, geometry.experts, geometry.top_k, geometry.shared_intermediate
    );
    config_bytes
}

fn load_fixture() -> Fixture {
    let bytes = std::fs::read(fixture_path()).unwrap_or_else(|e| {
        panic!(
            "read {FIXTURE}: {e}; the fixture is committed and must never be skipped — re-dump \
             it with tools/accuracy/dump_qwen38fn_moe_golden.py if it is really gone"
        )
    });
    assert_provenance(&bytes);
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture safetensors");
    let geometry = Qwen38fnMoeGeometry::qwen38_flash_next();

    let gate = bf16_mat(&st, "gate_weight", geometry.experts, geometry.hidden);
    let shared_gate_proj = bf16_mat(
        &st,
        "shared_gate_proj",
        geometry.shared_intermediate,
        geometry.hidden,
    );
    let shared_up_proj = bf16_mat(
        &st,
        "shared_up_proj",
        geometry.shared_intermediate,
        geometry.hidden,
    );
    let shared_down_proj = bf16_mat(
        &st,
        "shared_down_proj",
        geometry.hidden,
        geometry.shared_intermediate,
    );
    let shared_expert_gate = bf16_mat(&st, "shared_expert_gate_weight", 1, geometry.hidden);

    let cases = (0..CASES)
        .map(|index| {
            let prefix = format!("case{index}_");
            let hidden = bf16_rows(&st, &format!("{prefix}hidden"), geometry.hidden);
            let tokens = hidden.rows;
            assert!(tokens > 0, "{prefix}hidden has no rows");
            let case = Case {
                index,
                tokens,
                router_logits: bf16_mat(
                    &st,
                    &format!("{prefix}router_logits"),
                    tokens,
                    geometry.experts,
                ),
                routing_weights: bf16_mat(
                    &st,
                    &format!("{prefix}routing_weights"),
                    tokens,
                    geometry.top_k,
                ),
                selected_experts: i32_mat(
                    &st,
                    &format!("{prefix}selected_experts"),
                    tokens,
                    geometry.top_k,
                ),
                expert_out: bf16_rows(&st, &format!("{prefix}expert_out"), geometry.hidden),
                shared_out: bf16_rows(&st, &format!("{prefix}shared_out"), geometry.hidden),
                out: bf16_rows(&st, &format!("{prefix}out"), geometry.hidden),
                hidden,
            };
            for (name, mat) in [
                ("expert_out", &case.expert_out),
                ("shared_out", &case.shared_out),
                ("out", &case.out),
            ] {
                assert_eq!(
                    mat.rows, tokens,
                    "{prefix}{name} has {} rows, {prefix}hidden has {tokens}",
                    mat.rows
                );
            }
            case
        })
        .collect();

    Fixture {
        geometry,
        gate,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        shared_expert_gate,
        cases,
    }
}

fn bf16_mat(st: &SafeTensors<'_>, name: &str, rows: usize, cols: usize) -> Mat {
    let (mat, shape) = bf16_tensor(st, name);
    assert_eq!(shape, [rows, cols], "{name} shape");
    mat
}

fn bf16_rows(st: &SafeTensors<'_>, name: &str, cols: usize) -> Mat {
    let (mat, shape) = bf16_tensor(st, name);
    assert_eq!(shape[1], cols, "{name} column count");
    mat
}

fn bf16_tensor(st: &SafeTensors<'_>, name: &str) -> (Mat, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture is missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::BF16, "{name} must be bf16");
    let shape = t.shape().to_vec();
    assert_eq!(shape.len(), 2, "{name} must be 2-D, got {shape:?}");
    let data: Vec<f32> = t
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(bf16::from_le_bytes(*b)))
        .collect();
    assert_eq!(data.len(), shape[0] * shape[1], "{name} element count");
    (
        Mat {
            rows: shape[0],
            cols: shape[1],
            data,
        },
        shape,
    )
}

fn i32_mat(st: &SafeTensors<'_>, name: &str, rows: usize, cols: usize) -> Vec<i32> {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("fixture is missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name} must be i32");
    assert_eq!(t.shape(), [rows, cols], "{name} shape");
    t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

fn upload_matrix(ctx: &DeviceContext, mat: &Mat) -> DeviceMatrix {
    DeviceMatrix {
        data: ctx
            .stream
            .clone_htod(&mat.to_bf16())
            .expect("weight upload"),
        rows: mat.rows,
        cols: mat.cols,
    }
}

fn upload_hidden(ctx: &DeviceContext, mat: &Mat) -> HiddenStates {
    HiddenStates {
        data: ctx
            .stream
            .clone_htod(&mat.to_bf16())
            .expect("activation upload"),
        hidden_dim: mat.cols,
        seq_len: mat.rows,
    }
}

fn download(ctx: &DeviceContext, slice: &CudaSlice<bf16>, len: usize) -> Vec<f32> {
    assert!(
        slice.len() >= len,
        "download: buffer holds {} elements, needs {len}",
        slice.len()
    );
    let host: Vec<bf16> = ctx.stream.clone_dtoh(slice).expect("download");
    host[..len].iter().map(|&v| f32::from(v)).collect()
}

fn download_i32(ctx: &DeviceContext, slice: &CudaSlice<i32>, len: usize) -> Vec<i32> {
    let host: Vec<i32> = ctx.stream.clone_dtoh(slice).expect("download");
    host[..len].to_vec()
}

/// The production router GEMM on one case's hidden states.
fn device_router_logits(ctx: &DeviceContext, gate: &DeviceMatrix, case: &Case) -> Mat {
    let hidden = upload_hidden(ctx, &case.hidden);
    let mut logits =
        HiddenStates::zeros(ctx, gate.rows, case.tokens).expect("router logits buffer");
    gemm_into_checked(ctx, gate, &hidden, &mut logits).expect("router GEMM");
    ctx.sync().expect("sync");
    let data = download(ctx, &logits.data, case.tokens * gate.rows);
    Mat {
        rows: case.tokens,
        cols: gate.rows,
        data,
    }
}

/// The production routing kernel on logits already on device.
fn device_route(
    ctx: &DeviceContext,
    geometry: &Qwen38fnMoeGeometry,
    semantics: Qwen38fnRouteSemantics,
    logits: &HiddenStates,
) -> (Vec<f32>, Vec<i32>) {
    let picks = logits.seq_len * geometry.top_k;
    let mut weights = ctx.stream.alloc_zeros::<bf16>(picks).expect("weights");
    let mut ids = ctx.stream.alloc_zeros::<i32>(picks).expect("ids");
    qwen38fn_moe_route_into(ctx, geometry, &semantics, logits, &mut weights, &mut ids)
        .expect("MoE route");
    ctx.sync().expect("sync");
    (
        download(ctx, &weights, picks),
        download_i32(ctx, &ids, picks),
    )
}

/// The production shared-expert branch on one case's hidden states.
fn device_shared_branch(
    ctx: &DeviceContext,
    geometry: &Qwen38fnMoeGeometry,
    fixture: &Fixture,
    case: &Case,
    hidden: &HiddenStates,
) -> Mat {
    let gate_proj = upload_matrix(ctx, &fixture.shared_gate_proj);
    let up_proj = upload_matrix(ctx, &fixture.shared_up_proj);
    let down_proj = upload_matrix(ctx, &fixture.shared_down_proj);
    let mut scratch =
        Qwen38fnSharedBranchScratch::new(ctx, geometry, case.tokens).expect("scratch");
    let mut shared_out =
        HiddenStates::zeros(ctx, geometry.hidden, case.tokens).expect("shared_out buffer");
    qwen38fn_moe_shared_branch_into(
        ctx,
        geometry,
        hidden,
        &gate_proj,
        &up_proj,
        &down_proj,
        &mut scratch,
        &mut shared_out,
    )
    .expect("shared branch");
    ctx.sync().expect("sync");
    let data = download(ctx, &shared_out.data, case.tokens * geometry.hidden);
    Mat {
        rows: case.tokens,
        cols: geometry.hidden,
        data,
    }
}

// ---------------------------------------------------------------------------
// Routing grading — the device-free gate's tie contract, applied to the
// kernel's output.
// ---------------------------------------------------------------------------

/// One case's self-consistency census: the device picks graded against the
/// fp32 probabilities of the logits the kernel actually read.
struct RouteCensus {
    /// Weight elements bitwise equal to the canonical renormalized weight of
    /// the expert the device stored beside them.
    weights_exact: usize,
    weights_total: usize,
    worst_weight_ulp: i64,
    worst_weight_abs: f64,
    /// Weights whose sum-mass check (renorm only) failed.
    sum_violations: usize,
}

/// One case's cross-check census against the golden's *stored* picks and
/// weights, graded with the golden's own probabilities.
struct GoldenCensus {
    rows: usize,
    /// Rows whose expert set equals the golden's.
    set_matches: usize,
    /// Rows whose set differs, excused by an exact fp32 tie at the golden's
    /// k-th boundary (torch.topk defines no tie order).
    tie_drifts: usize,
    /// Rows whose set differs with no exact tie, excused only in the chain by
    /// a boundary gap inside `CHAIN_GAP_RATIO` (the GEMM's logit noise can
    /// flip a near-tie the golden left open).
    near_tie_drifts: usize,
    /// Golden (expert, weight) pairs whose weight the device reproduces
    /// bitwise (or, in a tie substitution, matches within the bound).
    weights_exact: usize,
    weights_total: usize,
    worst_weight_ulp: i64,
    worst_weight_rel: f64,
    weight_violations: usize,
    /// The first out-of-bound weight's context, for the caller's assert.
    first_violation: Option<String>,
}

/// Grade one token's device picks against `probs` — the fp32 probabilities of
/// the logits the kernel read. Self-consistency only: the golden cross-check
/// is [`grade_golden_token`].
#[allow(
    clippy::float_cmp,
    reason = "fp32 probabilities out of one softmax, compared for exact ties"
)]
fn grade_token(
    case: &Case,
    token: usize,
    probs: &[f32],
    got_ids: &[i32],
    got_w: &[f32],
    census: &mut RouteCensus,
) {
    let top_k = got_ids.len();
    let experts = probs.len();

    // Unique, in range.
    let mut selected = vec![false; experts];
    for (slot, &e) in got_ids.iter().enumerate() {
        assert!(
            e >= 0 && (e as usize) < experts,
            "case{} token {token} slot {slot}: expert {e} outside 0..{experts}",
            case.index
        );
        assert!(
            !selected[e as usize],
            "case{} token {token}: expert {e} selected twice",
            case.index
        );
        selected[e as usize] = true;
    }

    // A valid top-k set under the host probabilities, to within the softmax's
    // own f32 noise: no unselected expert beats a selected one.
    let min_selected = got_ids
        .iter()
        .map(|&e| f64::from(probs[e as usize]))
        .fold(f64::INFINITY, f64::min);
    let max_unselected = (0..experts)
        .filter(|&e| !selected[e])
        .map(|e| f64::from(probs[e]))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_selected + ORDER_EPS >= max_unselected,
        "case{} token {token}: device selection is not a top-{top_k} set — selected min \
         {min_selected:.9} < unselected max {max_unselected:.9}",
        case.index
    );

    // Canonical order: descending probability (to within f32 softmax noise),
    // ascending index inside an exact tie.
    for slot in 0..top_k.saturating_sub(1) {
        let (a, b) = (got_ids[slot] as usize, got_ids[slot + 1] as usize);
        let (pa, pb) = (f64::from(probs[a]), f64::from(probs[b]));
        assert!(
            pa + ORDER_EPS >= pb,
            "case{} token {token} slot {slot}: device order puts p[{a}]={pa:.9} before \
             p[{b}]={pb:.9}",
            case.index
        );
        if pa == pb {
            assert!(
                a < b,
                "case{} token {token} slot {slot}: exact tie {pa:.9} must break by ascending \
                 index, got {a} then {b}",
                case.index
            );
        }
    }

    // Each weight is the renormalized probability of the expert stored beside
    // it, summed over the device's own picks in the device's own (descending)
    // order — the order the kernel accumulates in.
    let sum: f32 = got_ids.iter().map(|&e| probs[e as usize]).sum();
    for slot in 0..top_k {
        let want = f32::from(bf16::from_f32(probs[got_ids[slot] as usize] / sum));
        let got = got_w[slot];
        census.weights_total += 1;
        if got == want {
            census.weights_exact += 1;
        }
        let ulp = ulp_distance(got, want);
        let abs = (f64::from(got) - f64::from(want)).abs();
        census.worst_weight_ulp = census.worst_weight_ulp.max(ulp);
        census.worst_weight_abs = census.worst_weight_abs.max(abs);
        assert!(
            ulp <= ROUTE_WEIGHT_ULPS,
            "case{} token {token} slot {slot}: device weight {got} is {ulp} bf16 ULP from the \
             canonical renormalized weight {want} of expert {}",
            case.index,
            got_ids[slot]
        );
    }

    // The renormalized weights sum to 1 (the property norm_topk_prob drops).
    let weight_sum: f64 = got_w.iter().map(|&w| f64::from(w)).sum();
    if (weight_sum - 1.0).abs() > WEIGHT_SUM_TOL {
        census.sum_violations += 1;
    }
}

/// Cross-check one token's device picks against the golden's stored pair,
/// graded with the golden's own probabilities (`probs`). `weight_rel` is the
/// relative bound on a golden weight the device must reproduce: 0.0 for the
/// isolated step (the device reads the golden's exact logits, so the bound is
/// `ROUTE_WEIGHT_ULPS` alone), `CHAIN_WEIGHT_REL` for the chain (the GEMM's
/// logit noise moves every renormalized weight by up to e^{2a}−1). A set
/// difference must be excused by the golden's k-th boundary: an exact fp32 tie
/// always (torch.topk's order is undefined there, and the kernel takes the
/// canonical pick), or — chain only — a gap inside `CHAIN_GAP_RATIO`.
#[allow(
    clippy::float_cmp,
    reason = "fp32 probabilities out of one softmax, compared for exact ties"
)]
fn grade_golden_token(
    case: &Case,
    token: usize,
    probs: &[f32],
    got_ids: &[i32],
    got_w: &[f32],
    weight_rel: f64,
    census: &mut GoldenCensus,
) {
    let top_k = got_ids.len();
    census.rows += 1;

    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let boundary_tie = sorted[top_k - 1] == sorted[top_k];
    let near_tie = sorted[top_k] > 0.0
        && f64::from(sorted[top_k - 1]) <= CHAIN_GAP_RATIO * f64::from(sorted[top_k]);

    let got_set: Vec<usize> = got_ids.iter().map(|&e| e as usize).collect();
    let stored: Vec<usize> = case.selected_experts[token * top_k..(token + 1) * top_k]
        .iter()
        .map(|&e| e as usize)
        .collect();
    let mut a = got_set.clone();
    a.sort_unstable();
    let mut b = stored.clone();
    b.sort_unstable();

    if a == b {
        census.set_matches += 1;
        // Weight-matched by expert: every golden slot's weight must be the
        // device weight stored beside the same expert.
        for (slot, &e) in stored.iter().enumerate() {
            let device_slot = got_set
                .iter()
                .position(|&d| d == e)
                .expect("sets are equal, so the expert is present");
            let golden_w = case.routing_weights.at(token, slot);
            grade_golden_weight(
                census,
                case,
                token,
                got_w[device_slot],
                golden_w,
                weight_rel,
                "matched set",
            );
        }
        return;
    }

    // A set difference: excused only at the golden's boundary.
    assert!(
        boundary_tie || (weight_rel > 0.0 && near_tie),
        "case{} token {token}: device experts {a:?} differ from the golden's {b:?} with the \
         golden's k-th boundary untied (p10 {:.9} vs p11 {:.9}) to excuse it",
        case.index,
        sorted[top_k - 1],
        sorted[top_k]
    );
    if boundary_tie {
        census.tie_drifts += 1;
    } else {
        census.near_tie_drifts += 1;
    }
    // Both sets hold `top_k` unique ids, so the device picks the golden missed
    // are exactly as many as the golden's experts the device missed. Pool them
    // and *consume* them: a re-scanning `find` would let a row drifting two
    // experts match both golden-missing slots to the same substitute and pass
    // on a weight it had already spent. The `<= 1` assert makes exhaustion of
    // the pool loud rather than letting a two-expert drift grade as two
    // independent one-expert boundary ties.
    let mut substitutes: Vec<(usize, f32)> = got_set
        .iter()
        .zip(got_w)
        .filter(|(d, _)| !b.contains(d))
        .map(|(d, &w)| (*d, w))
        .collect();
    assert!(
        substitutes.len() <= 1,
        "case{} token {token}: the device set {a:?} drifts from the golden's {b:?} by {} \
         experts; only a single boundary pick is excusable",
        case.index,
        substitutes.len()
    );
    for (slot, &e) in stored.iter().enumerate() {
        let golden_w = case.routing_weights.at(token, slot);
        if let Some(device_slot) = got_set.iter().position(|&d| d == e) {
            grade_golden_weight(
                census,
                case,
                token,
                got_w[device_slot],
                golden_w,
                weight_rel,
                "drifted row",
            );
        } else {
            // The golden's expert lost the tie (or the near-tie) the
            // device broke canonically: the device's substitute must sit
            // at the same boundary probability and carry the same weight
            // within the bound — the weight-matched-set rule.
            let (sub_expert, sub_weight) = substitutes.pop().unwrap_or_else(|| {
                panic!(
                    "case{} token {token}: the golden's expert {e} is missing but no \
                     unconsumed device pick substitutes for it",
                    case.index
                )
            });
            let p_sub = probs[sub_expert];
            let p_golden = probs[e];
            let tied = p_sub == p_golden
                || (weight_rel > 0.0
                    && f64::from(p_sub) <= CHAIN_GAP_RATIO * f64::from(p_golden)
                    && f64::from(p_golden) <= CHAIN_GAP_RATIO * f64::from(p_sub));
            assert!(
                tied,
                "case{} token {token}: the substitute for the golden's expert {e} \
                 (p {p_golden:.9}) is expert {sub_expert} at p {p_sub:.9} — not a boundary tie",
                case.index
            );
            grade_golden_weight(
                census,
                case,
                token,
                sub_weight,
                golden_w,
                weight_rel,
                "tie substitution",
            );
        }
    }
}

/// Grade one golden (expert, weight) pair against the device weight beside it.
/// `weight_rel` is 0.0 for the isolated step (bound is `ROUTE_WEIGHT_ULPS`
/// alone) and `CHAIN_WEIGHT_REL` for the chain (the GEMM's logit noise moves
/// every renormalized weight relatively). Free function, not a closure, so the
/// `census` mutations between calls do not fight a captured borrow.
#[allow(
    clippy::float_cmp,
    reason = "both sides are bf16-representable, so equality is bitwise equality"
)]
fn grade_golden_weight(
    census: &mut GoldenCensus,
    case: &Case,
    token: usize,
    got: f32,
    golden_w: f32,
    weight_rel: f64,
    what: &str,
) {
    census.weights_total += 1;
    if got == golden_w {
        census.weights_exact += 1;
    }
    let ulp = ulp_distance(got, golden_w);
    let diff = (f64::from(got) - f64::from(golden_w)).abs();
    let rel = if golden_w > 0.0 {
        diff / f64::from(golden_w)
    } else {
        ulp as f64
    };
    census.worst_weight_ulp = census.worst_weight_ulp.max(ulp);
    census.worst_weight_rel = census.worst_weight_rel.max(rel);
    let bound = weight_rel * f64::from(golden_w) + ROUTE_WEIGHT_ULPS as f64 * bf16_ulp(golden_w);
    if diff > bound {
        census.weight_violations += 1;
        if census.first_violation.is_none() {
            census.first_violation = Some(format!(
                "case{} token {token} ({what}): device weight {got} misses the golden's \
                 {golden_w} by {diff:.6} (bound {bound:.6}, rel {rel:.6})",
                case.index
            ));
        }
    }
}

fn grade_route(
    case: &Case,
    probs: &[Vec<f32>],
    got_ids: &[i32],
    got_w: &[f32],
    renorm: bool,
) -> RouteCensus {
    let top_k = case.routing_weights.cols;
    let mut census = RouteCensus {
        weights_exact: 0,
        weights_total: 0,
        worst_weight_ulp: 0,
        worst_weight_abs: 0.0,
        sum_violations: 0,
    };
    for t in 0..case.tokens {
        grade_token(
            case,
            t,
            &probs[t],
            &got_ids[t * top_k..(t + 1) * top_k],
            &got_w[t * top_k..(t + 1) * top_k],
            &mut census,
        );
    }
    if renorm {
        assert_eq!(
            census.sum_violations, 0,
            "case{}: {} tokens' renormalized weights do not sum to 1 within {WEIGHT_SUM_TOL}",
            case.index, census.sum_violations
        );
    }
    census
}

/// The golden cross-check for one case: the device picks and weights against
/// the golden's stored pair, graded with the golden's own probabilities and a
/// relative weight bound of `weight_rel` (0.0 isolated, `CHAIN_WEIGHT_REL`
/// chained).
fn grade_golden(
    case: &Case,
    golden_probs: &[Vec<f32>],
    got_ids: &[i32],
    got_w: &[f32],
    weight_rel: f64,
) -> GoldenCensus {
    let top_k = case.routing_weights.cols;
    let mut census = GoldenCensus {
        rows: 0,
        set_matches: 0,
        tie_drifts: 0,
        near_tie_drifts: 0,
        weights_exact: 0,
        weights_total: 0,
        worst_weight_ulp: 0,
        worst_weight_rel: 0.0,
        weight_violations: 0,
        first_violation: None,
    };
    for t in 0..case.tokens {
        grade_golden_token(
            case,
            t,
            &golden_probs[t],
            &got_ids[t * top_k..(t + 1) * top_k],
            &got_w[t * top_k..(t + 1) * top_k],
            weight_rel,
            &mut census,
        );
    }
    census
}

fn print_route(label: &str, census: &RouteCensus, golden: &GoldenCensus) {
    println!(
        "  {label}: self-consistency — {} of {} weights bitwise vs the canonical renorm of the \
         same logits, worst {} bf16 ULP, worst abs {:.6} (bound {} ULP), weight-sum violations {}",
        census.weights_exact,
        census.weights_total,
        census.worst_weight_ulp,
        census.worst_weight_abs,
        ROUTE_WEIGHT_ULPS,
        census.sum_violations,
    );
    println!(
        "  {label}: vs the golden — {} rows, {} expert sets equal, {} tie drifts, {} near-tie \
         drifts; {} of {} stored weights bitwise, worst {} ULP, worst rel {:.6}, violations {}",
        golden.rows,
        golden.set_matches,
        golden.tie_drifts,
        golden.near_tie_drifts,
        golden.weights_exact,
        golden.weights_total,
        golden.worst_weight_ulp,
        golden.worst_weight_rel,
        golden.weight_violations,
    );
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// The fixture's provenance and geometry, independent of any device: this half
/// must never be skippable.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let st = SafeTensors::deserialize(&bytes).expect("parse fixture safetensors");
    let geometry = Qwen38fnMoeGeometry::qwen38_flash_next();
    // The case count is part of the contract: a truncated dump must not pass
    // as a complete one.
    for index in 0..CASES {
        let (_, shape) = bf16_tensor(&st, &format!("case{index}_hidden"));
        assert_eq!(shape[1], geometry.hidden, "case{index}_hidden width");
        assert!(shape[0] > 0, "case{index}_hidden has no rows");
    }
}

/// The entry points' own refusal contract, device-free: both refuse a geometry
/// they were not compiled for and a null pointer before any launch, and let an
/// empty extent through. Every call leaves through a guard — the pointers
/// dangle and are never dereferenced.
#[test]
fn the_entry_points_refuse_what_they_were_not_compiled_for() {
    use cudarc::driver::sys::CUresult;
    use pegainfer_kernels::ffi::Half;
    use pegainfer_kernels::ffi::qwen38fn_moe_route_cuda as route;
    use pegainfer_kernels::ffi::qwen38fn_moe_shared_combine_cuda as combine;

    // Dangling and well-aligned, never dereferenced — every call below leaves
    // through a guard, before any launch. Typed per parameter position, since
    // `null()` is `*const` and the mutable outputs are `*mut`.
    let half_const: *const Half = std::ptr::NonNull::<Half>::dangling().as_ptr();
    let half_mut: *mut Half = std::ptr::NonNull::<Half>::dangling().as_ptr();
    let int_mut: *mut i32 = std::ptr::NonNull::<i32>::dangling().as_ptr();
    let invalid = CUresult::CUDA_ERROR_INVALID_VALUE as i32;
    let success = CUresult::CUDA_SUCCESS as i32;

    // (rows, experts, top_k, renorm, probs_bf16)
    let route_call = |args: (i32, i32, i32, i32, i32)| {
        let (rows, experts, top_k, renorm, probs_bf16) = args;
        (unsafe {
            route(
                half_const,
                half_mut,
                int_mut,
                rows,
                experts,
                top_k,
                renorm,
                probs_bf16,
                std::ptr::null_mut(),
            )
        }) as i32
    };

    // Zero extent with the compiled geometry: the guards are the whole call,
    // the positive control against a refusal of everything.
    assert_eq!(
        route_call((0, 512, 10, 1, 0)),
        success,
        "the compiled geometry must pass the guards"
    );
    for (what, result) in [
        (
            "an expert count that is not the compiled 512",
            route_call((0, 128, 10, 1, 0)),
        ),
        (
            "an expert count one past the compiled 512",
            route_call((0, 513, 10, 1, 0)),
        ),
        (
            "a top-k that is not the compiled 10",
            route_call((0, 512, 8, 1, 0)),
        ),
        ("a negative row count", route_call((-1, 512, 10, 1, 0))),
        (
            "a renorm flag that is not 0/1",
            route_call((0, 512, 10, 2, 0)),
        ),
        (
            "a probs_bf16 flag that is not 0/1",
            route_call((0, 512, 10, 1, -1)),
        ),
    ] {
        assert_eq!(result, invalid, "{what} must be refused");
    }
    for (what, logits, weights, ids) in [
        ("a null logits", std::ptr::null::<Half>(), half_mut, int_mut),
        (
            "a null weights",
            half_const,
            std::ptr::null_mut::<Half>(),
            int_mut,
        ),
        (
            "a null ids",
            half_const,
            half_mut,
            std::ptr::null_mut::<i32>(),
        ),
    ] {
        let result =
            unsafe { route(logits, weights, ids, 0, 512, 10, 1, 0, std::ptr::null_mut()) } as i32;
        assert_eq!(result, invalid, "{what} must be refused");
    }

    // (tokens, hidden)
    let combine_call = |args: (i32, i32)| {
        let (tokens, hidden) = args;
        (unsafe {
            combine(
                half_const,
                half_const,
                half_const,
                half_mut,
                tokens,
                hidden,
                std::ptr::null_mut(),
            )
        }) as i32
    };
    assert_eq!(
        combine_call((0, 2560)),
        success,
        "an empty combine must pass the guards"
    );
    assert_eq!(combine_call((-1, 2560)), invalid, "negative tokens");
    assert_eq!(combine_call((1, 0)), invalid, "a zero hidden width");
    for (what, gate, expert, shared, out) in [
        (
            "a null gate_logit",
            std::ptr::null::<Half>(),
            half_const,
            half_const,
            half_mut,
        ),
        (
            "a null expert_out",
            half_const,
            std::ptr::null::<Half>(),
            half_const,
            half_mut,
        ),
        (
            "a null shared_out",
            half_const,
            half_const,
            std::ptr::null::<Half>(),
            half_mut,
        ),
        (
            "a null out",
            half_const,
            half_const,
            half_const,
            std::ptr::null_mut::<Half>(),
        ),
    ] {
        let result =
            unsafe { combine(gate, expert, shared, out, 0, 2560, std::ptr::null_mut()) } as i32;
        assert_eq!(result, invalid, "{what} must be refused");
    }
}

/// Step 1 — the router GEMM through the production cuBLAS op. The bound is one
/// bf16 ULP *plus* the measured absolute floor `LOGIT_ATOL`: cuBLAS's f32
/// accumulation order over K=2560 differs from torch CPU's enough that
/// near-zero logits land many of their own (tiny) ULPs away — a pure relative
/// bound is the wrong yard for a GEMM whose outputs straddle zero. Measured
/// worst 0.03125 absolute (see `LOGIT_ATOL`); the printout carries each case.
#[test]
fn router_gemm_matches_the_golden() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let gate = upload_matrix(&ctx, &fixture.gate);
    for case in &fixture.cases {
        println!("case{} (T={}): router GEMM", case.index, case.tokens);
        let logits = device_router_logits(&ctx, &gate, case);
        let floor = vec![LOGIT_ATOL; logits.data.len()];
        let report = compare(
            "router_logits = gemm_into_checked(hidden, gateᵀ)",
            &logits.data,
            &case.router_logits.data,
            &Tol::with_floor(LOGIT_ULPS, floor),
        );
        report.assert_clean();
        println!(
            "  |logit| max {:.4}; {} of {} logits bitwise, worst {} ULP, worst abs {:.6} \
             (floor {LOGIT_ATOL}) — the reference keeps the router in bf16 (only its softmax \
             upcasts), so the ULP is the golden's own output rounding and the residual is \
             cuBLAS accumulation order",
            case.router_logits.max_abs(),
            report.exact,
            report.total,
            report.worst_ulp,
            report.worst_abs
        );
    }
}

/// Step 2 — the routing kernel on the golden's *stored* bf16 logits: the exact
/// bits the reference's softmax consumed. Graded twice: self-consistency
/// against the host canonical route over those same logits (bitwise weights,
/// canonical tie order), and against the golden's stored pair as a
/// weight-matched set (the fixture genuinely contains exact boundary ties,
/// where torch.topk's order is undefined, so `weight_rel = 0`).
#[test]
fn routing_matches_the_golden_on_stored_logits() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): routing on the golden's stored bf16 logits",
            case.index, case.tokens
        );
        let logits = upload_hidden(&ctx, &case.router_logits);
        let (got_w, got_ids) = device_route(
            &ctx,
            &geometry,
            Qwen38fnRouteSemantics::norm_topk_prob(true),
            &logits,
        );
        let probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(case.router_logits.row(t)))
            .collect();
        let census = grade_route(case, &probs, &got_ids, &got_w, true);
        let golden = grade_golden(case, &probs, &got_ids, &got_w, 0.0);
        print_route("routing on stored logits", &census, &golden);
        assert_eq!(
            golden.weight_violations,
            0,
            "case{}: {} stored weights outside the bound: {}",
            case.index,
            golden.weight_violations,
            golden.first_violation.as_deref().unwrap_or("<none>")
        );
    }
}

/// The kernel's fail-closed rule on device: one non-finite logit poisons its
/// whole row — NaN weights and placeholder ids (the slot index), so nothing
/// downstream can index a gather by a winner value — and *only* that row: the
/// untouched rows of the same launch still route canonically. The softmax
/// would otherwise mask a -inf/NaN into ordinary-looking weights (the trap
/// gemma4's router rule exists for).
#[test]
#[allow(
    clippy::float_cmp,
    reason = "both sides are bf16-representable, so equality is bitwise equality"
)]
fn a_non_finite_logit_fails_its_row_closed() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let case = &fixture.cases[0];
    assert!(
        case.tokens >= 3,
        "case0 must carry a poisoned row, a +inf row and a clean one"
    );

    // Row 0 gets a NaN logit, row 1 a +inf one; the rest stay golden.
    let mut logits = case.router_logits.clone();
    logits.data[3] = f32::NAN;
    logits.data[geometry.experts + 7] = f32::INFINITY;
    let logits_dev = upload_hidden(&ctx, &logits);
    let (got_w, got_ids) = device_route(
        &ctx,
        &geometry,
        Qwen38fnRouteSemantics::norm_topk_prob(true),
        &logits_dev,
    );
    let top_k = geometry.top_k;
    for row in 0..2 {
        for slot in 0..top_k {
            let w = got_w[row * top_k + slot];
            assert!(
                w.is_nan(),
                "poisoned row {row} slot {slot}: weight {w} is not the fail-closed NaN"
            );
            assert_eq!(
                got_ids[row * top_k + slot] as usize,
                slot,
                "poisoned row {row} slot {slot}: the placeholder id must be the slot index"
            );
        }
    }
    // An untouched row of the same launch still routes canonically, bitwise.
    let clean = 2usize;
    let probs = softmax_f32(case.router_logits.row(clean));
    let order: Vec<usize> = {
        let mut o: Vec<usize> = (0..probs.len()).collect();
        o.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        o.truncate(top_k);
        o
    };
    let sum: f32 = order.iter().map(|&e| probs[e]).sum();
    for slot in 0..top_k {
        let want = f32::from(bf16::from_f32(probs[order[slot]] / sum));
        let got = got_w[clean * top_k + slot];
        assert_eq!(
            got, want,
            "clean row {clean} slot {slot}: the poisoned rows must not disturb it"
        );
        assert_eq!(
            got_ids[clean * top_k + slot] as usize,
            order[slot],
            "clean row {clean} slot {slot}: pick"
        );
    }
    println!(
        "  rows 0 (NaN) and 1 (+inf) failed closed — {} NaN weights, placeholder ids 0..{top_k}; \
         row {clean} of the same launch routed bitwise canonical",
        2 * top_k
    );
}

/// Step 3 — the chain the reference actually runs: production GEMM → routing
/// kernel on the GEMM's own bf16 logits. Graded two ways: self-consistency
/// against the host canonical route over the *device's* logits (isolates the
/// kernel from the GEMM's accumulation order — bitwise), and against the
/// golden's stored pair with `weight_rel = CHAIN_WEIGHT_REL` (the GEMM's
/// ≤`LOGIT_ATOL` logit noise moves every renormalized weight by up to
/// e^{2a}−1 and can flip a near-tie boundary the golden left open).
#[test]
fn chained_router_to_routing_matches_the_golden() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let gate = upload_matrix(&ctx, &fixture.gate);
    for case in &fixture.cases {
        println!(
            "case{} (T={}): router → routing chain",
            case.index, case.tokens
        );
        let logits = device_router_logits(&ctx, &gate, case);
        let logits_dev = upload_hidden(&ctx, &logits);
        let (got_w, got_ids) = device_route(
            &ctx,
            &geometry,
            Qwen38fnRouteSemantics::norm_topk_prob(true),
            &logits_dev,
        );
        // Self-consistency: the kernel against the host canonical over the same
        // (device) logits — no GEMM noise in this comparison.
        let device_probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(logits.row(t)))
            .collect();
        let census = grade_route(case, &device_probs, &got_ids, &got_w, true);
        // Against the golden, with the chain's derived relative bound.
        let golden_probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(case.router_logits.row(t)))
            .collect();
        let golden = grade_golden(case, &golden_probs, &got_ids, &got_w, CHAIN_WEIGHT_REL);
        print_route("chain (GEMM→routing)", &census, &golden);
        assert_eq!(
            golden.weight_violations,
            0,
            "case{}: {} chained weights outside the bound: {}",
            case.index,
            golden.weight_violations,
            golden.first_violation.as_deref().unwrap_or("<none>")
        );
    }
}

/// Step 4 — the shared expert through the production composition: two cuBLAS
/// projections, the production `silu_mul_batch_into`, the down projection.
/// The bound is the device-free gate's: 1 ULP plus the quadrature floor
/// derived from the host replay's intermediate, with the device path's larger
/// `FLOOR_SAFETY` (its intermediate can carry the projections' own 1-ULP
/// difference). Measured per case below.
#[test]
fn shared_branch_matches_the_golden() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    for case in &fixture.cases {
        println!("case{} (T={}): shared expert", case.index, case.tokens);
        let hidden = upload_hidden(&ctx, &case.hidden);
        let device = device_shared_branch(&ctx, &geometry, &fixture, case, &hidden);
        let reference = shared_replay(&fixture, case);
        let report = compare(
            "shared_out = down(silu(gate)·up) via production ops",
            &device.data,
            &case.shared_out.data,
            &Tol::with_floor(1, reference.floor.clone()),
        );
        report.assert_clean();
        // The host replay's own distance to the golden, for scale: the device
        // path must not be materially worse than the host's accumulation-order
        // residual.
        let host = compare(
            "shared_out host replay (scale reference)",
            &reference.out.data,
            &case.shared_out.data,
            &Tol::with_floor(1, reference.floor.clone()),
        );
        println!(
            "  |shared_out| max {:.4}; device worst {:.6} ({:.3} of the bound), host replay \
             worst {:.6}; floor derived as {:.2e}·√Σ(h·w)² over the {}-term down projection",
            case.shared_out.max_abs(),
            report.worst_abs,
            report.worst_slack,
            host.worst_abs,
            FLOOR_SAFETY * BF16_RELATIVE_ULP,
            geometry.shared_intermediate
        );
    }
}

/// Step 5 — the block combine: the production gate GEMM plus the fused
/// sigmoid-gate kernel, on the golden's stored `expert_out` and `shared_out`
/// (the expert stack is not in the fixture — see the header), and then again on
/// the device's own shared branch.
#[test]
fn block_output_matches_the_golden() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let gate_w = upload_matrix(&ctx, &fixture.shared_expert_gate);
    for case in &fixture.cases {
        println!("case{} (T={}): block combine", case.index, case.tokens);
        let hidden = upload_hidden(&ctx, &case.hidden);

        // The learned scalar gate through the production GEMM.
        let mut gate_logit = HiddenStates::zeros(&ctx, 1, case.tokens).expect("gate logit buffer");
        gemm_into_checked(&ctx, &gate_w, &hidden, &mut gate_logit).expect("gate GEMM");

        let expert = upload_hidden(&ctx, &case.expert_out);
        let shared = upload_hidden(&ctx, &case.shared_out);
        let mut out = HiddenStates::zeros(&ctx, geometry.hidden, case.tokens).expect("out buffer");
        qwen38fn_moe_combine_into(
            &ctx,
            &geometry,
            &gate_logit.data,
            &expert,
            &shared,
            &mut out,
        )
        .expect("combine");
        ctx.sync().expect("sync");
        let got = download(&ctx, &out.data, case.tokens * geometry.hidden);

        // Identity on stored operands: every input is a golden bf16 value and
        // the only recomputation is the scalar gate, so the residual is the
        // gate's own rounding — `OUT_ULPS` ULPs of the sum plus the absolute
        // term 2^-7·|shared_out|, which bounds one ULP of gate error on the
        // product (sigmoid <= 1) and survives cancellation in the sum where
        // |out| << |gated|.
        let gate_floor: Vec<f64> = case
            .shared_out
            .data
            .iter()
            .map(|&s| BF16_RELATIVE_ULP * f64::from(s).abs())
            .collect();
        let report = compare(
            "out = expert_out + sigmoid(gate)·shared_out (stored)",
            &got,
            &case.out.data,
            &Tol::with_floor(OUT_ULPS, gate_floor),
        );
        report.assert_clean();
        println!(
            "  stored operands: {} of {} bitwise, worst {} ULP, worst abs {:.6} ({:.3} of the \
             bound)",
            report.exact, report.total, report.worst_ulp, report.worst_abs, report.worst_slack
        );

        // The same sum on the device's own shared branch: inherits the shared
        // branch's floor, scaled by the gate (sigmoid <= 1 cannot amplify),
        // plus the gate's own absolute term 2^-7·|shared_out|.
        let device_shared = device_shared_branch(&ctx, &geometry, &fixture, case, &hidden);
        let device_shared_dev = upload_hidden(&ctx, &device_shared);
        let mut out2 = HiddenStates::zeros(&ctx, geometry.hidden, case.tokens).expect("out buffer");
        qwen38fn_moe_combine_into(
            &ctx,
            &geometry,
            &gate_logit.data,
            &expert,
            &device_shared_dev,
            &mut out2,
        )
        .expect("combine");
        ctx.sync().expect("sync");
        let got2 = download(&ctx, &out2.data, case.tokens * geometry.hidden);

        let reference = shared_replay(&fixture, case);
        let floor: Vec<f64> = reference
            .floor
            .chunks(geometry.hidden)
            .zip(reference.gate_sigmoid.chunks(1))
            .zip(device_shared.data.chunks(geometry.hidden))
            .flat_map(|((row, g), shared_row)| {
                row.iter().zip(shared_row).map(move |(f, s)| {
                    f * f64::from(g[0]) + BF16_RELATIVE_ULP * f64::from(*s).abs()
                })
            })
            .collect();
        let report = compare(
            "out = expert_out + sigmoid(gate)·shared_out (device branch)",
            &got2,
            &case.out.data,
            &Tol::with_floor(OUT_ULPS, floor),
        );
        report.assert_clean();
        println!(
            "  device shared branch: {} of {} bitwise, worst {} ULP, worst abs {:.6} ({:.3} of \
             the bound); expert_out is the fixture's recorded activation, not a replay — the \
             routed-expert GEMM waits for #1107",
            report.exact, report.total, report.worst_ulp, report.worst_abs, report.worst_slack
        );
    }
}

// ---------------------------------------------------------------------------
// Negative controls: each drives the production entry point with the port
// mistake and asserts the comparison against the golden FAILS.
// ---------------------------------------------------------------------------

/// (a) `norm_topk_prob = false` — the trap the fixture exists for. The field is
/// absent from `config.json` and defaults to True upstream; the kernel takes it
/// as a flag, so the control runs the production kernel with renorm off and
/// every stored weight must move.
#[test]
fn control_skipping_topk_renorm_fails() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_abs = 0f64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — no top-k renormalization",
            case.index, case.tokens
        );
        let logits = upload_hidden(&ctx, &case.router_logits);
        let (got_w, _got_ids) = device_route(
            &ctx,
            &geometry,
            Qwen38fnRouteSemantics::norm_topk_prob(false),
            &logits,
        );
        // Positional against the golden's stored weights — order-blind by
        // necessity here: the un-renormalized probabilities are ~0.4x the
        // renormalized weights, so *every* slot must land outside the
        // tolerance no matter how torch ordered the row.
        let report = compare(
            "control: routing_weights without renorm",
            &got_w,
            &case.routing_weights.data,
            &Tol::ulps(ROUTE_WEIGHT_ULPS),
        );
        report.print();
        let mass: f64 = got_w[..geometry.top_k].iter().map(|&w| f64::from(w)).sum();
        println!(
            "  token 0 top-{} probability mass without renorm: {mass:.6} (the golden's weights \
             sum to 1)",
            geometry.top_k
        );
        total += report.total;
        violations += report.violations;
        worst_abs = worst_abs.max(report.worst_abs);
    }
    assert_eq!(
        violations, total,
        "control (a): dropping the renormalization must move every weight; {violations} of \
         {total} did, so the control no longer proves the renorm is load-bearing"
    );
    assert!(
        worst_abs > 0.1,
        "control (a): worst abs deviation {worst_abs:.6} is too small to be a decisive control"
    );
}

/// (b) bf16 softmax — the reference upcasts (`softmax(router_logits,
/// dtype=torch.float)`); the kernel's `softmax_bf16` flag is exactly this port
/// mistake. The control runs the *same* kernel on the *same* stored logits both
/// ways — fp32 (the reference, which `routing_matches_the_golden_on_stored_logits`
/// pins to the golden) and bf16 — so the only difference is the softmax dtype
/// and both sides share the canonical pick order (no tie-order confound). At
/// least a fifth of the weights must move, and the fp32 side must be the one
/// the golden keeps.
#[test]
fn control_bf16_softmax_fails() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_ulp = 0i64;
    let mut id_rows_moved = 0usize;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — bf16 softmax",
            case.index, case.tokens
        );
        let logits = upload_hidden(&ctx, &case.router_logits);
        // The reference path (fp32 softmax), pinned to the golden elsewhere.
        let (fp32_w, fp32_ids) = device_route(
            &ctx,
            &geometry,
            Qwen38fnRouteSemantics::norm_topk_prob(true),
            &logits,
        );
        // The port mistake (bf16 softmax), same kernel, same logits.
        let (bf16_w, bf16_ids) = device_route(
            &ctx,
            &geometry,
            Qwen38fnRouteSemantics {
                renorm: true,
                softmax_bf16: true,
            },
            &logits,
        );
        // The picks are canonically ordered on both sides, so the weight
        // comparison is position-aligned. A bf16 softmax that also re-ordered
        // the top-k would be an even louder failure; count those rows rather
        // than assert on them, so the control stays a demonstration, not a
        // second gate.
        let top_k = geometry.top_k;
        id_rows_moved += (0..case.tokens)
            .filter(|&t| {
                fp32_ids[t * top_k..(t + 1) * top_k] != bf16_ids[t * top_k..(t + 1) * top_k]
            })
            .count();
        let report = compare(
            "control: bf16-softmax weights vs the fp32 reference",
            &bf16_w,
            &fp32_w,
            &Tol::ulps(0),
        );
        report.print();
        total += report.total;
        violations += report.violations;
        worst_ulp = worst_ulp.max(report.worst_ulp);

        // And the fp32 side is the golden's: re-assert the positive match here
        // so the control cannot pass by both sides drifting off the golden.
        let probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(case.router_logits.row(t)))
            .collect();
        let golden = grade_golden(case, &probs, &fp32_ids, &fp32_w, 0.0);
        assert_eq!(
            golden.weight_violations, 0,
            "case{}: the fp32 side of the control drifted off the golden",
            case.index
        );
    }
    assert!(
        violations > 0,
        "control (b): a bf16 softmax reproduced the fp32 reference bitwise, so the gate does not \
         actually pin the reference's fp32 upcast"
    );
    assert!(
        violations * 5 >= total,
        "control (b): only {violations} of {total} weights moved — too weak to call the fp32 \
         softmax load-bearing"
    );
    println!(
        "  {violations} of {total} weights differ by up to {worst_ulp} bf16 ULP between the bf16 \
         and fp32 softmax ({id_rows_moved} rows also re-ordered their picks); the fp32 path is \
         the golden's on all of them"
    );
}

/// (c) Dropping the shared expert's learned scalar sigmoid gate — summing the
/// raw shared output into the routed output through the production elementwise
/// add instead of the gated combine.
#[test]
fn control_dropping_shared_sigmoid_gate_fails() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let fixture = load_fixture();
    let geometry = fixture.geometry;
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_abs = 0f64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — no shared scalar sigmoid gate",
            case.index, case.tokens
        );
        let expert = upload_hidden(&ctx, &case.expert_out);
        let shared = upload_hidden(&ctx, &case.shared_out);
        let mut out = ctx
            .stream
            .alloc_zeros::<bf16>(case.tokens * geometry.hidden)
            .expect("out buffer");
        let n = case.tokens * geometry.hidden;
        add_into(&ctx, &expert.data, &shared.data, n, &mut out).expect("add");
        ctx.sync().expect("sync");
        let got = download(&ctx, &out, n);

        // Held to exactly the tolerance the gated device path passes: the
        // shared branch's floor scaled by the gate, plus the gate's own
        // absolute term on the stored shared output.
        let reference = shared_replay(&fixture, case);
        let floor: Vec<f64> = reference
            .floor
            .chunks(geometry.hidden)
            .zip(reference.gate_sigmoid.chunks(1))
            .zip(case.shared_out.data.chunks(geometry.hidden))
            .flat_map(|((row, g), shared_row)| {
                row.iter().zip(shared_row).map(move |(f, s)| {
                    f * f64::from(g[0]) + BF16_RELATIVE_ULP * f64::from(*s).abs()
                })
            })
            .collect();
        let report = compare(
            "control: out = expert_out + shared_out",
            &got,
            &case.out.data,
            &Tol::with_floor(OUT_ULPS, floor),
        );
        report.print();
        let (lo, hi) = reference
            .gate_sigmoid
            .iter()
            .fold((f32::INFINITY, 0f32), |(lo, hi), &g| (lo.min(g), hi.max(g)));
        println!(
            "  the gate this control drops: sigmoid in [{lo:.6}, {hi:.6}] over {} tokens",
            case.tokens
        );
        total += report.total;
        violations += report.violations;
        worst_abs = worst_abs.max(report.worst_abs);
    }
    assert!(
        violations * 2 >= total,
        "control (c): only {violations} of {total} block outputs moved without the shared gate"
    );
    assert!(
        worst_abs > 1.0,
        "control (c): worst abs deviation {worst_abs:.6} is within noise of the gated path"
    );
}
