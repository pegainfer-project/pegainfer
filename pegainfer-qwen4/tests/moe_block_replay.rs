//! Device-free replay gate for the Qwen3.8-Flash-Next MoE block golden — the
//! reader `test_data/qwen38fn-moe-golden.safetensors` was dumped for (#1103:
//! a golden lands together with the code that checks it).
//!
//! The fixture (`tools/accuracy/dump_qwen38fn_moe_golden.py`, transformers
//! `Qwen4ExpTextSparseMoeBlock` at the frozen checkpoint's real geometry:
//! hidden 2560, 512 experts, top-10 routed + 1 shared, expert and shared
//! intermediate 640) stores, per case, the reference's own activations — router
//! logits, renormalized top-k weights, selected experts, routed-expert output,
//! shared-expert output and block output — plus every weight those steps read:
//! the router matrix, the shared expert's three projections and its scalar gate.
//!
//! This gate replays them on the host, with no device, no Triton and no
//! checkpoint:
//!
//! | step | replay | tolerance | measured worst |
//! |------|--------|-----------|----------------|
//! | router | `bf16(hidden · gateᵀ)`, f32 accumulation | 1 bf16 ULP | 1 ULP on 8 of 10752 logits |
//! | routing | fp32 softmax over 512 → top-10 → renormalize | bitwise | bitwise on 210/210 |
//! | router→routing chain | the same, on the replay's own bf16 logits | bitwise | bitwise on 210/210 |
//! | shared | bf16-rounded SwiGLU + down projection | 1 ULP + a derived floor | 0.0625 abs, 0.41 of the floor |
//! | combine | `expert_out + sigmoid(gate_logit) · shared_out`, stored operands | bitwise | bitwise on 53760/53760 |
//!
//! Three traps are pinned here because all three produce fluent, wrong text
//! rather than a crash:
//!
//! * `norm_topk_prob` is **absent** from the checkpoint's `config.json` and
//!   defaults to `True` upstream, so a port that reads only `config.json` skips
//!   the top-k renormalization. Every stored weight then moves (210/210
//!   elements, up to 0.24 absolute) — see `control_skipping_topk_renorm_fails`.
//!   `pegainfer_qwen4::config` carries the default and its own unit test pins
//!   it; this gate pins the *numbers* it implies.
//! * the shared expert's output is scaled by a **learned per-token scalar
//!   sigmoid gate** (`shared_expert_gate`, a `[1, 2560]` linear) before it is
//!   summed with the routed output. Dropping it moves the block output by up to
//!   22.2 absolute — see `control_dropping_shared_sigmoid_gate_fails`.
//! * the router **stores bf16** logits and its softmax reads those bits, so a
//!   port that keeps the accumulation in f32 routes differently: 171 of 210
//!   stored weights move, by up to 5 bf16 ULP — see
//!   `control_unrounded_router_logits_fails`, which is what the chain row above
//!   exists to catch.
//!
//! Tolerances are the reference's own rounding, not slack for the port. The
//! model is bf16, so every dtype boundary in the replay rounds to bf16 exactly
//! where the reference does ([`Mat::round`]) and the arithmetic between
//! boundaries is f32 like torch's; the only residual is accumulation *order*,
//! which is three orders of magnitude below a bf16 ULP (2^-8..2^-7 relative) at
//! these widths. Measured, not assumed: 8 of 10752 replayed logits land 1 ULP
//! from the golden and the rest are bitwise, so 1 ULP is the router's bound; the
//! routing weights read the golden's own stored bf16 logits and reproduce all
//! 210 of them bitwise, so theirs is bitwise. The shared branch needs one more
//! term: a 1-ULP difference in its
//! `silu(gate)·up` intermediate propagates through the 640-term down
//! projection, where cancellation makes the relative error large on near-zero
//! outputs. That term is bounded from the replay's own intermediate
//! (`FLOOR_SAFETY · BF16_RELATIVE_ULP · √Σ(h·w)²`) instead of a magic constant.
//!
//! NOT gated here: `expert_out` is a *recorded activation*, not a replay — the
//! fused 512-expert stack is 2.52e9 parameters (~4.7 GiB in bf16) and is
//! deliberately not stored. So the routed-expert GEMM (the fused 3D
//! `gate_up_proj` / `down_proj` layout and its scatter/gather) and the GPU
//! router kernel stay ungated until the MoE implementation (#1107) lands and
//! replays this fixture on device. What this gate owns is the part a port gets
//! silently wrong: router, renorm, shared branch, combine.
//!
//! ```text
//! cargo test --release --locked -p pegainfer-qwen4 --test moe_block_replay -- --nocapture
//! ```

use half::bf16;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/qwen38fn-moe-golden.safetensors"
);
const FROZEN_CONFIG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/frozen_config.json");
const METADATA_KEY: &str = "qwen4_moe_golden";
const FORMAT_VERSION: u64 = 2;
/// The fixture's `config` field, asserted as well as hashed: a moved frozen
/// config must fail the gate, not silently re-hash a new file.
const CONFIG_IN_FIXTURE: &str = "pegainfer-qwen4/tests/frozen_config.json";
/// The dump's case count (two prefill batches and one decode row). Token counts
/// come from the tensor shapes; this only pins how many cases the fixture
/// carries, so a truncated dump cannot pass as a complete one.
const CASES: usize = 3;

/// bf16's worst-case relative ULP: 7 stored mantissa bits plus the implicit one
/// put consecutive values between `2^-8` and `2^-7` apart relatively, so `2^-7`
/// bounds one rounding step at any magnitude.
const BF16_RELATIVE_ULP: f64 = 1.0 / 128.0; // 2^-7
/// Headroom on the derived shared-branch floor: quadrature assumes the
/// intermediate's rounding errors are uncorrelated, which the down projection's
/// signs do not guarantee. Measured worst element: 0.41 of the floored bound,
/// while every control exceeds it by more than 5x.
const FLOOR_SAFETY: f64 = 2.0;

// ---------------------------------------------------------------------------
// bf16 arithmetic surface
// ---------------------------------------------------------------------------

/// A row-major `[rows, cols]` matrix of bf16-representable `f32` values. The
/// reference is a bf16 model, so every dtype boundary in the replay rounds
/// explicitly through [`Mat::round`]; the `f32` storage only widens the rounded
/// result, which keeps `F.linear`, `silu`, `sigmoid` and `+` at torch's
/// precision without carrying a second numeric type through the maths.
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

    fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Round to bf16 — one dtype boundary of the reference. `half` rounds to
    /// nearest even, as torch does.
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
}

/// `y = x · wᵀ`, accumulated in f32 and returned unrounded — torch's `F.linear`
/// on bf16 operands accumulates in f32 and rounds the output once, so the caller
/// decides where the dtype boundary falls.
fn linear(x: &Mat, w: &Mat) -> Mat {
    assert_eq!(
        x.cols, w.cols,
        "linear: contraction dim {} vs {}",
        x.cols, w.cols
    );
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

/// The reference's `act_fn` on a bf16 operand: torch computes SiLU in f32
/// (`opmath`) and stores bf16.
fn silu(x: f32) -> f32 {
    f32::from(bf16::from_f32(x * sigmoid(x)))
}

/// `gelu_pytorch_tanh`, the activation the checkpoint's *vision* config uses —
/// copying it into the text MLP is the bug this control stands for. The exact-erf
/// variant deviates from the golden identically (measured 1.438 / 1.562 / 1.906
/// max abs on cases 0 / 1 / 2).
fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    f32::from(bf16::from_f32(
        0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh()),
    ))
}

/// One bf16 ULP at `|x|`, where `x` is itself bf16-representable. Zero maps to
/// the smallest positive subnormal, so near-zero references get no relative
/// slack — the reason the shared branch also carries an absolute floor. At
/// bf16's largest finite magnitude there is no next value, so the step falls
/// back to the relative spacing; without that the limit would saturate to
/// infinity and the element's check would go vacuous.
fn bf16_ulp(x: f32) -> f64 {
    let mag = bf16::from_f32(x.abs());
    let next = bf16::from_bits(mag.to_bits() + 1);
    if next.is_infinite() {
        return f64::from(mag) * BF16_RELATIVE_ULP;
    }
    f64::from(next) - f64::from(mag)
}

/// bf16 sign-magnitude folded onto a monotone axis, so a bit-pattern distance is
/// a value distance (and `+0` / `-0` are 0 apart).
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

/// An element passes when `|mine - reference| <= ulps · ulp(|reference|) +
/// floor[i]`. `ulps = 0` with an empty floor is the bitwise contract.
struct Tol {
    ulps: i64,
    floor: Vec<f64>,
}

impl Tol {
    /// Bitwise: the reference's own rounding reproduced exactly.
    const fn bitwise() -> Self {
        Self {
            ulps: 0,
            floor: Vec::new(),
        }
    }

    /// One bf16 ULP of the stored value, for a step whose only residual is f32
    /// accumulation order.
    const fn one_ulp() -> Self {
        Self {
            ulps: 1,
            floor: Vec::new(),
        }
    }

    /// One ULP plus a per-element absolute floor.
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
            "  {:<46} n={:<6} bitwise={:<6} worst_ulp={:<4} worst_abs={:.6} worst_slack={:.4} \
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
            "{}: {} of {} elements outside tolerance (worst abs {:.6}, worst slack {:.4})",
            self.label, self.violations, self.total, self.worst_abs, self.worst_slack
        );
    }
}

#[allow(
    clippy::float_cmp,
    reason = "both sides are bf16-representable, so equality is bitwise equality"
)]
fn compare(label: &str, mine: &Mat, reference: &Mat, tol: &Tol) -> Report {
    assert_eq!(
        (mine.rows, mine.cols),
        (reference.rows, reference.cols),
        "{label}: shape {}x{} vs {}x{}",
        mine.rows,
        mine.cols,
        reference.rows,
        reference.cols
    );
    assert!(
        tol.floor.is_empty() || tol.floor.len() == reference.data.len(),
        "{label}: floor covers {} of {} elements",
        tol.floor.len(),
        reference.data.len()
    );
    let mut report = Report {
        label: label.to_string(),
        total: reference.data.len(),
        exact: 0,
        worst_ulp: 0,
        worst_abs: 0.0,
        worst_slack: 0.0,
        violations: 0,
    };
    for (i, (&got, &want)) in mine.data.iter().zip(&reference.data).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "{label}[{i}]: non-finite replay {got} vs golden {want}"
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

struct Geometry {
    hidden: usize,
    experts: usize,
    top_k: usize,
    shared_intermediate: usize,
}

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
    geometry: Geometry,
    gate: Mat,
    shared_gate_proj: Mat,
    shared_up_proj: Mat,
    shared_down_proj: Mat,
    shared_expert_gate: Mat,
    cases: Vec<Case>,
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

/// Hold the fixture's metadata manifest against the frozen config it was dumped
/// from, and hand back that config's bytes for the geometry read. Every failure
/// here is a hard panic: a golden whose provenance cannot be established must
/// not degrade into a skip.
fn check_provenance(bytes: &[u8]) -> Vec<u8> {
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
        "fixture format_version {version} is not {FORMAT_VERSION}; this reader does not know that \
         layout (the dump script is the authority)"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG_IN_FIXTURE),
        "fixture was dumped against config {:?}, not {CONFIG_IN_FIXTURE}",
        manifest["config"]
    );
    let config_bytes = std::fs::read(FROZEN_CONFIG)
        .unwrap_or_else(|e| panic!("read frozen config {FROZEN_CONFIG}: {e}"));
    let sha = sha256_hex(&config_bytes);
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(sha.as_str()),
        "fixture config_sha256 is not sha256({FROZEN_CONFIG}) = {sha}; the golden was dumped from \
         a different checkpoint geometry"
    );
    println!(
        "fixture provenance: format_version {version}, config_sha256 {sha}, seed {}, \
         transformers {}, torch {}, {}",
        manifest["seed"],
        manifest["transformers_version"]
            .as_str()
            .unwrap_or("<missing>"),
        manifest["torch_version"].as_str().unwrap_or("<missing>"),
        manifest["platform"].as_str().unwrap_or("<missing>")
    );
    config_bytes
}

fn load_fixture() -> Fixture {
    let bytes = std::fs::read(FIXTURE).unwrap_or_else(|e| {
        panic!(
            "read {FIXTURE}: {e}; the fixture is committed and must never be skipped — re-dump it \
             with tools/accuracy/dump_qwen38fn_moe_golden.py if it is really gone"
        )
    });
    let config_bytes = check_provenance(&bytes);
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .unwrap_or_else(|e| panic!("parse {FROZEN_CONFIG}: {e}"));
    let text = &config["text_config"];
    let geometry = Geometry {
        hidden: int_field(text, "hidden_size"),
        experts: int_field(text, "num_experts"),
        top_k: int_field(text, "num_experts_per_tok"),
        shared_intermediate: int_field(text, "shared_expert_intermediate_size"),
    };
    assert_eq!(
        text["hidden_act"].as_str(),
        Some("silu"),
        "the shared branch replays SiLU; the frozen config says {:?}",
        text["hidden_act"]
    );

    let st = SafeTensors::deserialize(&bytes).expect("parse fixture safetensors");
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
            // The token count is the `hidden` row count: the fixture carries no
            // separate `seq_len` tensor, and the block is row-wise, so a case is
            // nothing but its rows.
            let hidden = bf16_rows(&st, &format!("{prefix}hidden"), geometry.hidden);
            let tokens = hidden.rows;
            assert!(tokens > 0, "{prefix}hidden has no rows");
            let router_logits = bf16_mat(
                &st,
                &format!("{prefix}router_logits"),
                tokens,
                geometry.experts,
            );
            let routing_weights = bf16_mat(
                &st,
                &format!("{prefix}routing_weights"),
                tokens,
                geometry.top_k,
            );
            let selected_experts = i32_mat(
                &st,
                &format!("{prefix}selected_experts"),
                tokens,
                geometry.top_k,
            );
            let expert_out = bf16_rows(&st, &format!("{prefix}expert_out"), geometry.hidden);
            let shared_out = bf16_rows(&st, &format!("{prefix}shared_out"), geometry.hidden);
            let out = bf16_rows(&st, &format!("{prefix}out"), geometry.hidden);
            for (name, mat) in [
                ("expert_out", &expert_out),
                ("shared_out", &shared_out),
                ("out", &out),
            ] {
                assert_eq!(
                    mat.rows, tokens,
                    "{prefix}{name} has {} rows, {prefix}hidden has {tokens}",
                    mat.rows
                );
            }
            Case {
                index,
                tokens,
                hidden,
                router_logits,
                routing_weights,
                selected_experts,
                expert_out,
                shared_out,
                out,
            }
        })
        .collect();

    let fixture = Fixture {
        geometry,
        gate,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        shared_expert_gate,
        cases,
    };
    println!(
        "fixture geometry: hidden {}, {} experts, top-{} routed + 1 shared, shared intermediate \
         {}; case token counts {:?}",
        fixture.geometry.hidden,
        fixture.geometry.experts,
        fixture.geometry.top_k,
        fixture.geometry.shared_intermediate,
        fixture.cases.iter().map(|c| c.tokens).collect::<Vec<_>>()
    );
    // Strictly fewer routed experts than experts, or two of the routing checks
    // go vacuous: with every expert selected there is no unselected one to beat,
    // and no k-th boundary whose tie the selection rule may cross.
    assert!(
        fixture.geometry.top_k < fixture.geometry.experts,
        "top_k {} must be strictly below {} experts for the routing checks to mean anything",
        fixture.geometry.top_k,
        fixture.geometry.experts
    );
    fixture
}

fn int_field(text: &serde_json::Value, key: &str) -> usize {
    text[key].as_u64().unwrap_or_else(|| {
        panic!(
            "{FROZEN_CONFIG} text_config.{key} is not an integer: {:?}",
            text[key]
        )
    }) as usize
}

/// A bf16 tensor whose shape is fully pinned by the frozen geometry.
fn bf16_mat(st: &SafeTensors<'_>, name: &str, rows: usize, cols: usize) -> Mat {
    let (mat, shape) = bf16_tensor(st, name);
    assert_eq!(shape, [rows, cols], "{name} shape");
    mat
}

/// A bf16 tensor whose row count the caller derives from the shape.
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
// Replay steps
// ---------------------------------------------------------------------------

/// Step 1 — the router: `F.linear(hidden, gate_weight)` on bf16 operands, i.e.
/// f32 accumulation rounded once to bf16. The reference does **not** upcast the
/// router (`softmax` is the only step that takes `dtype=torch.float`), so the
/// golden's logits are bf16 and 1 bf16 ULP is the golden's own output rounding.
fn router_logits(fixture: &Fixture, case: &Case) -> Mat {
    linear(&case.hidden, &fixture.gate).round()
}

/// The reference's routing: fp32 softmax over all 512 logits, top-k, then —
/// because `norm_topk_prob` defaults to `True` upstream and is absent from
/// `config.json` — renormalize the k survivors so they sum to 1, and store bf16.
///
/// These three pieces already exist as a production host path for another line:
/// `pegainfer-deepseek-v2-lite/src/host_ops.rs` carries `gate_logits_host`,
/// `softmax` and `topk_softmax_routes`, the last with the identical descending /
/// ascending-index tie rule. They are not reusable here — that crate is
/// CUDA-linked and feature-gated, the helpers are crate-private, and the
/// contract differs (it re-sorts the picks by expert index for HF's unsorted
/// gate path and applies DeepSeek's own renormalization), while this crate
/// deliberately carries no CUDA dependency at all. So the replay keeps its own
/// copy; #1107's qwen4 MoE implementation is where the two converge, and it
/// should replay this fixture through that shared code instead of a third copy.
struct Routing {
    /// fp32 probabilities over every expert, per token.
    probs: Vec<Vec<f32>>,
    /// The replay's own top-k: descending by probability, ascending by expert
    /// index inside a tie (a canonical order; `torch.topk` does not define one).
    selected: Vec<Vec<usize>>,
    /// Renormalized weights, bf16, in the replay's own order.
    weights: Mat,
}

fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    // Max-subtracted, so the largest term is exactly 1.0 and `sum >= 1.0`:
    // unlike the DeepSeek copy this needs no zero-sum guard.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn route(case: &Case, top_k: usize, probs: Vec<Vec<f32>>, renorm: bool) -> Routing {
    let mut selected = Vec::with_capacity(case.tokens);
    let mut weights = Mat::zeros(case.tokens, top_k);
    for (t, p) in probs.iter().enumerate().take(case.tokens) {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        order.truncate(top_k);
        let sum: f32 = order.iter().map(|&e| p[e]).sum();
        for (slot, &expert) in order.iter().enumerate() {
            let w = if renorm { p[expert] / sum } else { p[expert] };
            weights.row_mut(t)[slot] = f32::from(bf16::from_f32(w));
        }
        selected.push(order);
    }
    Routing {
        probs,
        selected,
        weights,
    }
}

/// Step 2 reads the golden's **stored** bf16 logits, not the logits step 1
/// recomputes: those bits are exactly what the reference's softmax consumed, and
/// gemma4's MoE reference check makes the same call for the same reason — "the
/// pick and weight contract is then evaluated exactly on those stored logits,
/// which is what the kernel consumes" (`pegainfer-gemma4/src/moe/tests.rs:286`).
/// Step 1 pins the logits to 1 ULP on its own, so the chain is covered without
/// letting one accumulation-order rounding flip an exact fp32 tie in here.
fn routing(fixture: &Fixture, case: &Case, renorm: bool) -> Routing {
    let probs = (0..case.tokens)
        .map(|t| softmax_f32(case.router_logits.row(t)))
        .collect();
    route(case, fixture.geometry.top_k, probs, renorm)
}

/// Step 3 — the shared expert: `down(act(gate(x)) * up(x))`, with the
/// reference's bf16 rounding after each operator (both projections, the
/// activation, the product, the down projection). Emulating every boundary is
/// the tighter choice and the measured one: one unrounded f32 pass through the
/// same four operators lands 0.125 max abs from the golden with only ~40% of
/// its elements bitwise, while rounding where the reference rounds lands 0.0625
/// max abs with 99.8%+ bitwise — so the residual is accumulation order alone,
/// not a precision choice the port made differently.
struct Shared {
    /// `down(act(gate)·up)`, bf16 — the golden's `caseN_shared_out`, which the
    /// reference captures **before** the scalar gate below.
    out: Mat,
    /// Per-element absolute floor: one bf16 ULP of the `silu(gate)·up`
    /// intermediate propagated through `shared_down_proj` in quadrature, times
    /// [`FLOOR_SAFETY`].
    floor: Vec<f64>,
    /// The learned per-token scalar gate, `sigmoid(shared_expert_gate · x)`,
    /// bf16 — the reference applies it to the shared output before the sum.
    gate_sigmoid: Vec<f32>,
}

fn shared_branch(fixture: &Fixture, case: &Case, act: impl Fn(f32) -> f32) -> Shared {
    let gate = linear(&case.hidden, &fixture.shared_gate_proj).round();
    let up = linear(&case.hidden, &fixture.shared_up_proj).round();
    let mut intermediate = Mat::zeros(case.tokens, fixture.geometry.shared_intermediate);
    for t in 0..case.tokens {
        for (j, slot) in intermediate.row_mut(t).iter_mut().enumerate() {
            *slot = f32::from(bf16::from_f32(act(gate.at(t, j)) * up.at(t, j)));
        }
    }

    // Down projection and its quadrature floor in one pass: the floor is the
    // same contraction over squared terms, so it costs one extra accumulator.
    let down = &fixture.shared_down_proj;
    let mut shared = Mat::zeros(case.tokens, fixture.geometry.hidden);
    let mut floor = vec![0f64; case.tokens * fixture.geometry.hidden];
    for t in 0..case.tokens {
        let h = intermediate.row(t);
        let frow = &mut floor[t * down.rows..(t + 1) * down.rows];
        let wrows = down.data.chunks(down.cols);
        for ((slot, w), f) in shared.row_mut(t).iter_mut().zip(wrows).zip(frow) {
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
    let gate_sigmoid = (0..case.tokens)
        .map(|t| f32::from(bf16::from_f32(sigmoid(gate_logit.at(t, 0)))))
        .collect();

    Shared {
        out: shared.round(),
        floor,
        gate_sigmoid,
    }
}

/// Step 4 — the block sum: `out = expert_out + sigmoid(gate_logit) ·
/// shared_out`, every operator bf16. `expert_out` is the fixture's recorded
/// activation (the 512-expert stack is not stored), so this pins the gate, the
/// scaling and the sum — not the routed GEMM.
fn combine(expert_out: &Mat, shared_out: &Mat, gate_sigmoid: &[f32]) -> Mat {
    assert_eq!(
        gate_sigmoid.len(),
        shared_out.rows,
        "combine: one gate scalar per token, got {} for {} tokens",
        gate_sigmoid.len(),
        shared_out.rows
    );
    let mut out = Mat::zeros(shared_out.rows, shared_out.cols);
    for (t, orow) in out.data.chunks_mut(shared_out.cols).enumerate() {
        let g = gate_sigmoid[t];
        let erow = expert_out.row(t);
        let srow = shared_out.row(t);
        for ((&e, &s), slot) in erow.iter().zip(srow).zip(orow) {
            let gated = f32::from(bf16::from_f32(g * s));
            *slot = f32::from(bf16::from_f32(e + gated));
        }
    }
    out
}

/// The shared floor carried through the block sum: the gate scales the shared
/// branch's error and cannot amplify it (`sigmoid <= 1`).
fn combine_floor(shared: &Shared) -> Vec<f64> {
    shared
        .floor
        .chunks(shared.out.cols)
        .zip(&shared.gate_sigmoid)
        .flat_map(|(row, &g)| row.iter().map(move |&f| f64::from(g) * f))
        .collect()
}

fn gate_sigmoid_range(shared: &Shared) -> (f32, f32) {
    (
        shared
            .gate_sigmoid
            .iter()
            .fold(f32::INFINITY, |a, &v| a.min(v)),
        shared.gate_sigmoid.iter().fold(0f32, |a, &v| a.max(v)),
    )
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

#[test]
fn router_logits_replay() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        println!("case{} (T={}): router", case.index, case.tokens);
        compare(
            "router_logits = bf16(hidden · gateᵀ)",
            &router_logits(&fixture, case),
            &case.router_logits,
            &Tol::one_ulp(),
        )
        .assert_clean();
        println!(
            "  |logit| max {:.4}; the reference keeps the router in bf16 (only its softmax \
             upcasts), so 1 bf16 ULP is the golden's own output rounding, and f32 accumulation \
             order (~1e-5 relative over {} terms) cannot exceed it",
            case.router_logits.max_abs(),
            fixture.geometry.hidden
        );
    }
}

/// The tie census one case's selection contract reports back.
struct SelectionCensus {
    /// Rows where the k-th and (k+1)-th fp32 probabilities are exactly equal, so
    /// any of the tied experts is a legitimate pick.
    boundary_ties: usize,
    /// Rows whose positional order differs from the canonical rule (descending
    /// probability, ascending index inside a tie).
    order_mismatches: usize,
    /// Rows whose expert *set* differs from the canonical top-k set — only ever
    /// a boundary tie, which the contract below asserts.
    set_drifts: usize,
    /// Per-token sum of the stored renormalized weights.
    weight_sums: Vec<f64>,
}

/// What one case's stored `selected_experts` / `routing_weights` pair must
/// satisfy, judged against `probs` — the fp32 probabilities the golden's row is
/// read with — and `canonical`, the replay's own top-k over those same
/// probabilities. Both the stored-logit replay and the chained router→routing
/// replay grade the golden through here, so the contract has one owner.
#[allow(
    clippy::float_cmp,
    reason = "fp32 probabilities out of one softmax, compared for exact ties"
)]
fn assert_selection(case: &Case, probs: &[Vec<f32>], canonical: &[Vec<usize>]) -> SelectionCensus {
    let top_k = case.routing_weights.cols;
    let mut census = SelectionCensus {
        boundary_ties: 0,
        order_mismatches: 0,
        set_drifts: 0,
        weight_sums: Vec::with_capacity(case.tokens),
    };
    for t in 0..case.tokens {
        let probs = &probs[t];
        let stored = &case.selected_experts[t * top_k..(t + 1) * top_k];
        assert_eq!(
            stored.len(),
            top_k,
            "case{} token {t} stores {} experts, expected {top_k}",
            case.index,
            stored.len()
        );

        // A valid top-k set: no selected expert is beaten by an unselected one.
        // Tie-blind by construction, which is the point.
        let mut selected_mask = vec![false; probs.len()];
        for &e in stored {
            assert!(
                (e as usize) < probs.len(),
                "case{} token {t}: expert {e} outside 0..{}",
                case.index,
                probs.len()
            );
            assert!(
                !selected_mask[e as usize],
                "case{} token {t}: expert {e} selected twice",
                case.index
            );
            selected_mask[e as usize] = true;
        }
        let min_selected = stored
            .iter()
            .map(|&e| f64::from(probs[e as usize]))
            .fold(f64::INFINITY, f64::min);
        let max_unselected = (0..probs.len())
            .filter(|&e| !selected_mask[e])
            .map(|e| f64::from(probs[e]))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            min_selected >= max_unselected,
            "case{} token {t}: stored selection is not a top-{top_k} set — selected min \
             {min_selected:.9} < unselected max {max_unselected:.9}",
            case.index
        );

        // Each stored weight must be the renormalized probability of the expert
        // stored beside it — the weight-matched half of the tie rule. It binds
        // slot j to expert j, so a permutation that kept the weight multiset
        // would still fail.
        let sum: f32 = stored.iter().map(|&e| probs[e as usize]).sum();
        for (slot, &e) in stored.iter().enumerate() {
            let want = f32::from(bf16::from_f32(probs[e as usize] / sum));
            let got = case.routing_weights.at(t, slot);
            assert_eq!(
                got, want,
                "case{} token {t} slot {slot}: stored weight {got} is not the renormalized \
                 probability of the stored expert {e} ({want})",
                case.index
            );
        }

        // The order rule, and its teeth. `torch.topk` leaves the order of an
        // exact tie undefined — inside the top-k (5 of case0's 8 rows carry a
        // repeated probability) and at the k-th boundary (3 of case1's 12 rows;
        // row 0 takes expert 393 where index order would take 151, both at
        // p = 0.027970612). So the comparison is a weight-matched set, in three
        // parts:
        //   1. the stored probabilities are the top-k's, in descending order,
        //      bitwise slot by slot (below);
        //   2. each stored weight is the renormalized probability of the expert
        //      stored beside it, bitwise (above);
        //   3. the stored experts are the canonical top-k set unless an exact
        //      tie straddles the k-th boundary, where any of the tied experts is
        //      a legitimate pick (below).
        // gemma4's equivalent check *can* assert its picks positionally
        // (`assert_eq!(picks, expected_picks)`, pegainfer-gemma4/src/moe/
        // tests.rs:359) because the other side there is its own kernel, which
        // implements the same canonical rule. `torch.topk` does not, so
        // positional equality here would grade torch's tie order rather than the
        // routing contract.
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        for (slot, &e) in stored.iter().enumerate() {
            assert_eq!(
                probs[e as usize], sorted[slot],
                "case{} token {t} slot {slot}: stored expert {e} carries probability {}, not the \
                 slot's {} — the stored row is not the top-{top_k} probability sequence in \
                 descending order",
                case.index, probs[e as usize], sorted[slot]
            );
        }
        let mut stored_set: Vec<usize> = stored.iter().map(|&e| e as usize).collect();
        stored_set.sort_unstable();
        let mut canonical_set = canonical[t].clone();
        canonical_set.sort_unstable();
        if stored_set != canonical_set {
            census.set_drifts += 1;
        }
        if sorted[top_k - 1] == sorted[top_k] {
            census.boundary_ties += 1;
        } else {
            assert_eq!(
                stored_set, canonical_set,
                "case{} token {t}: stored experts differ from the canonical top-{top_k} set with \
                 no fp32 tie at the boundary to excuse it",
                case.index
            );
        }
        if canonical[t] != stored.iter().map(|&e| e as usize).collect::<Vec<_>>() {
            census.order_mismatches += 1;
        }

        // The renormalized top-k must sum to 1 — the property
        // `norm_topk_prob = false` silently drops. Derived bound: each stored
        // weight is `bf16(w_j)` for an f32 `w_j` that sums to 1, and one bf16
        // rounding moves a value by at most half a spacing (2^-9 relative), so
        // ten of them move the sum by at most 2^-9 ≈ 0.002 — measured worst
        // 0.0017. The gate sits at 0.01, five times the derivation so a re-dump
        // keeps its room, and the un-renormalized sum lands 0.34-0.47 away, i.e.
        // 34x outside it.
        let weight_sum: f64 = (0..top_k)
            .map(|slot| f64::from(case.routing_weights.at(t, slot)))
            .sum();
        assert!(
            (weight_sum - 1.0).abs() <= 0.01,
            "case{} token {t}: renormalized top-{top_k} weights sum to {weight_sum}, not 1",
            case.index
        );
        census.weight_sums.push(weight_sum);
    }
    census
}

fn print_selection(case: &Case, census: &SelectionCensus) {
    let top_k = case.routing_weights.cols;
    println!(
        "  selection: all {} tokens are valid top-{top_k} sets whose probability sequence and \
         per-slot weights match bitwise, and whose expert sets match the canonical top-k wherever \
         the boundary is untied; boundary-tie rows {}, rows whose positional order differs from \
         the canonical (desc prob, asc index) rule {}, rows whose expert set drifts from it {} — \
         tie permutations and tie substitutions, which is why the rule is a weight-matched set and \
         not a positional one",
        case.tokens, census.boundary_ties, census.order_mismatches, census.set_drifts
    );
    println!(
        "  renormalized weight sum per token: [{:.6}, {:.6}]",
        census
            .weight_sums
            .iter()
            .fold(f64::INFINITY, |a, &v| a.min(v)),
        census.weight_sums.iter().fold(0f64, |a, &v| a.max(v))
    );
}

#[test]
fn routing_replay() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        println!(
            "case{} (T={}): routing on the golden's stored bf16 logits",
            case.index, case.tokens
        );
        let routed = routing(&fixture, case, true);

        // The renormalized weights must reproduce the golden bitwise: they are
        // 10 fp32 probabilities divided by their sum and rounded once, so the
        // only freedom a port has is the softmax precision — which is what
        // `control_bf16_softmax_fails` exercises.
        compare(
            "routing_weights (fp32 softmax + renorm)",
            &routed.weights,
            &case.routing_weights,
            &Tol::bitwise(),
        )
        .assert_clean();
        print_selection(
            case,
            &assert_selection(case, &routed.probs, &routed.selected),
        );
    }
}

/// The chain the reference actually runs: router → softmax → top-k → renorm,
/// with the router's output rounded to bf16 on the way, because that is what the
/// reference stores and what its softmax then reads. Step 1 alone cannot see
/// this: its 1-ULP bound also passes a port that keeps the logits in f32 and
/// never rounds them, and step 2 reads the golden's stored logits by design.
/// Softmaxing *unrounded* logits moves 171 of the golden's 210 weights (up to 5
/// bf16 ULP) — see `control_unrounded_router_logits_fails` — so this is the
/// check that pins the router's bf16 store.
#[test]
fn chained_router_to_routing_replay() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        println!(
            "case{} (T={}): router → routing chain",
            case.index, case.tokens
        );
        let logits = router_logits(&fixture, case);
        let probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(logits.row(t)))
            .collect();
        let routed = route(case, fixture.geometry.top_k, probs, true);
        compare(
            "routing_weights from replayed logits",
            &routed.weights,
            &case.routing_weights,
            &Tol::bitwise(),
        )
        .assert_clean();
        let census = assert_selection(case, &routed.probs, &routed.selected);
        print_selection(case, &census);
    }
}

#[test]
fn shared_branch_replay() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        println!("case{} (T={}): shared expert", case.index, case.tokens);
        let shared = shared_branch(&fixture, case, silu);
        compare(
            "shared_out = bf16 down(silu(gate)·up)",
            &shared.out,
            &case.shared_out,
            &Tol::with_floor(1, shared.floor.clone()),
        )
        .assert_clean();
        let (lo, hi) = gate_sigmoid_range(&shared);
        println!(
            "  floor: derived per element as {:.2e}·√Σ(h·w)² over the {}-term down projection \
             (max {:.6}); |shared_out| max {:.4}; scalar gate sigmoid in [{:.6}, {:.6}]",
            FLOOR_SAFETY * BF16_RELATIVE_ULP,
            fixture.geometry.shared_intermediate,
            shared.floor.iter().fold(0f64, |a, &b| a.max(b)),
            case.shared_out.max_abs(),
            lo,
            hi
        );
    }
}

#[test]
fn block_output_replay() {
    let fixture = load_fixture();
    for case in &fixture.cases {
        println!("case{} (T={}): block combine", case.index, case.tokens);
        let shared = shared_branch(&fixture, case, silu);

        // The combine identity on the golden's own activations: bitwise, since
        // every operand is a stored bf16 value and the only recomputation is
        // the scalar gate.
        compare(
            "out = expert_out + sigmoid(gate)·shared_out",
            &combine(&case.expert_out, &case.shared_out, &shared.gate_sigmoid),
            &case.out,
            &Tol::bitwise(),
        )
        .assert_clean();

        // The same sum with the shared branch replayed rather than read: it
        // inherits the shared branch's floor, scaled by the gate.
        compare(
            "out from the replayed shared branch",
            &combine(&case.expert_out, &shared.out, &shared.gate_sigmoid),
            &case.out,
            &Tol::with_floor(1, combine_floor(&shared)),
        )
        .assert_clean();
        println!(
            "  out == expert_out + gated shared_out bitwise on {} elements; expert_out itself is \
             the fixture's recorded activation, not a replay (the 512-expert stack is not stored)",
            case.out.data.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Negative controls: each asserts that the comparison FAILS
// ---------------------------------------------------------------------------

/// (a) `norm_topk_prob = false` — the trap the fixture exists for. The field is
/// absent from `config.json` and defaults to `True` upstream, so a port that
/// reads only the checkpoint config renormalizes nothing and ships plausible,
/// wrong routing weights.
#[test]
fn control_skipping_topk_renorm_fails() {
    let fixture = load_fixture();
    let top_k = fixture.geometry.top_k;
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_abs = 0f64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — no top-k renormalization",
            case.index, case.tokens
        );
        let unnorm = routing(&fixture, case, false);
        let report = compare(
            "control: routing_weights without renorm",
            &unnorm.weights,
            &case.routing_weights,
            &Tol::bitwise(),
        );
        report.print();
        let mass: f64 = (0..top_k)
            .map(|slot| f64::from(unnorm.weights.at(0, slot)))
            .sum();
        println!(
            "  token 0 top-{top_k} probability mass without renorm: {mass:.6} (the golden's \
             weights sum to 1)"
        );
        total += report.total;
        violations += report.violations;
        worst_abs = worst_abs.max(report.worst_abs);
    }
    assert_eq!(
        violations, total,
        "control (a): dropping the renormalization must move every weight; {violations} of {total} \
         did, so the control no longer proves the renorm is load-bearing"
    );
    assert!(
        worst_abs > 0.1,
        "control (a): worst abs deviation {worst_abs:.6} is too small to be a decisive control"
    );
}

/// (b) bf16 softmax — the reference calls `softmax(router_logits,
/// dtype=torch.float)`; carrying the 512 probabilities in bf16 instead (what
/// `torch.softmax(bf16_logits)` gives without `dtype=`) rounds every probability
/// to 8 mantissa bits before the top-k and the renormalization.
#[test]
fn control_bf16_softmax_fails() {
    let fixture = load_fixture();
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_ulp = 0i64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — bf16 softmax",
            case.index, case.tokens
        );
        let probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| {
                softmax_f32(case.router_logits.row(t))
                    .iter()
                    .map(|&p| f32::from(bf16::from_f32(p)))
                    .collect()
            })
            .collect();
        let routed = route(case, fixture.geometry.top_k, probs, true);
        let report = compare(
            "control: routing_weights from bf16 softmax",
            &routed.weights,
            &case.routing_weights,
            &Tol::bitwise(),
        );
        report.print();
        total += report.total;
        violations += report.violations;
        worst_ulp = worst_ulp.max(report.worst_ulp);
    }
    assert!(
        violations > 0,
        "control (b): a bf16 softmax reproduced the golden bitwise, so the gate's bitwise \
         tolerance does not actually pin the reference's fp32 upcast"
    );
    assert!(
        violations * 5 >= total,
        "control (b): only {violations} of {total} weights moved — too weak to call the fp32 \
         softmax load-bearing"
    );
    println!(
        "  {violations} of {total} weights differ by up to {worst_ulp} bf16 ULP; the fp32 path is \
         bitwise on all of them"
    );
}

/// (c) Dropping the shared expert's learned scalar sigmoid gate — summing the
/// raw shared output into the routed output instead.
#[test]
fn control_dropping_shared_sigmoid_gate_fails() {
    let fixture = load_fixture();
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_abs = 0f64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — no shared scalar sigmoid gate",
            case.index, case.tokens
        );
        let shared = shared_branch(&fixture, case, silu);
        let ones = vec![1.0f32; case.tokens];
        let report = compare(
            "control: out = expert_out + shared_out",
            &combine(&case.expert_out, &case.shared_out, &ones),
            &case.out,
            &Tol::with_floor(1, combine_floor(&shared)),
        );
        report.print();
        let (lo, hi) = gate_sigmoid_range(&shared);
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
        "control (c): worst abs deviation {worst_abs:.6} is within noise of the gated replay"
    );
}

/// (d) `gelu_pytorch_tanh` in place of SiLU on the shared branch — the
/// activation the checkpoint's *vision* config uses, so copying it into the text
/// MLP is a live risk.
#[test]
fn control_gelu_instead_of_silu_fails() {
    let fixture = load_fixture();
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_abs = 0f64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — gelu instead of silu",
            case.index, case.tokens
        );
        let gelu = shared_branch(&fixture, case, gelu_tanh);
        // The bound is the *correct* replay's floor, so the control is held to
        // exactly the tolerance the positive gate passes.
        let reference = shared_branch(&fixture, case, silu);
        let report = compare(
            "control: shared_out with gelu_tanh",
            &gelu.out,
            &case.shared_out,
            &Tol::with_floor(1, reference.floor.clone()),
        );
        report.print();
        total += report.total;
        violations += report.violations;
        worst_abs = worst_abs.max(report.worst_abs);
    }
    assert!(
        violations * 2 >= total,
        "control (d): only {violations} of {total} shared outputs moved under gelu"
    );
    assert!(
        worst_abs > 1.0,
        "control (d): worst abs deviation {worst_abs:.6} is within noise of the SiLU replay"
    );
}

/// (e) A router that never stores bf16 — softmaxing the f32 accumulation
/// directly. Step 1's 1-ULP logit bound cannot see this (an unrounded value sits
/// within half an ULP of the rounded one, so it *passes* that bound), and step 2
/// reads the golden's stored logits by design, so without the chain check a port
/// could keep the router in f32 and clear every other gate while shifting the
/// routing. Measured: 171 of 210 weights move, by up to 5 bf16 ULP.
#[test]
fn control_unrounded_router_logits_fails() {
    let fixture = load_fixture();
    let (mut total, mut violations) = (0usize, 0usize);
    let mut worst_ulp = 0i64;
    for case in &fixture.cases {
        println!(
            "case{} (T={}): control — router logits kept in f32",
            case.index, case.tokens
        );
        // Step 1 without its `.round()`: the same f32 accumulation, straight
        // into the softmax.
        let logits = linear(&case.hidden, &fixture.gate);
        let probs: Vec<Vec<f32>> = (0..case.tokens)
            .map(|t| softmax_f32(logits.row(t)))
            .collect();
        let routed = route(case, fixture.geometry.top_k, probs, true);
        let report = compare(
            "control: routing_weights from f32 logits",
            &routed.weights,
            &case.routing_weights,
            &Tol::bitwise(),
        );
        report.print();
        total += report.total;
        violations += report.violations;
        worst_ulp = worst_ulp.max(report.worst_ulp);
    }
    assert!(
        violations * 2 >= total,
        "control (e): only {violations} of {total} weights moved — too weak to pin the router's \
         bf16 store"
    );
    println!(
        "  {violations} of {total} weights differ by up to {worst_ulp} bf16 ULP; the chained bf16 \
         path is bitwise on all of them"
    );
}
