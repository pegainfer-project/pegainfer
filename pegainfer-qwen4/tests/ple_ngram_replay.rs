//! Host-side reader for `test_data/qwen38fn-ple-golden.safetensors` — the
//! consumer half of the Qwen3.8-Flash-Next per-layer-embedding (PLE) golden.
//!
//! The fixture was recorded once from the HF reference (`Qwen4ExpTextPLELayer` +
//! `Qwen4ExpTextNGramEmbedding`, transformers 5.16.1 `models/qwen4_exp`) by
//! `tools/accuracy/dump_qwen38fn_ple_golden.py`, at the frozen checkpoint's real
//! geometry. This gate replays it with integer and f32/f64 host math only — no
//! CUDA, no weights, no Python — so the two things a PLE implementation can get
//! wrong are pinned before the kernel that will need them exists:
//!
//! 1. **The n-gram hash, exactly.** `ngram_ids` is recomputed from `input_ids`
//!    and compared with i64 equality, zero tolerance. It is pure integer math,
//!    so any deviation is a bug and not noise: the EOS-resetting context window,
//!    the bigram/trigram head layout (16 heads = 2 orders x 8) and the per-head
//!    mul-xor fold all have to be right for a single id to land.
//! 2. **The PLE block, to the bf16 noise floor.** `out` is recomputed from the
//!    stored `hidden`/`key`/`value` (the block's inputs, recorded past the
//!    projections that the 95 GiB n-gram table feeds) and compared against the
//!    stored block output within [`OUT_MAX_ABS`].
//!
//! Provenance is checked on every load, never skipped: the `qwen4_ple_golden`
//! manifest must parse, carry `format_version` 2, and name
//! `pegainfer-qwen4/tests/frozen_config.json` with the sha256 that file actually
//! hashes to. `src/config.rs::the_committed_fixture_is_the_frozen_revision` pins
//! that file to the frozen revision, so the chain is fixture -> config ->
//! checkpoint. The same loader also cross-checks the config's `eos_token_id`
//! against the fixture's stored scalar and every tensor width against the
//! config's geometry. The per-head moduli and offsets it replays with are the
//! fixture's own, checked here to be an increasing run with contiguous offsets —
//! the same table `NgramTable::derive` builds from config, whose endpoints and
//! totals `the_ngram_table_derivation_matches_the_checkpoint` pins.
//!
//! Negative controls: a gate that only ever passes proves nothing, so four
//! single-knob mutations of this same replay are asserted to *fail* — dropping
//! the EOS window reset, dilation 1 instead of `ngram_size`, an unsigned sqrt on
//! the gate, and a swapped bigram/trigram head order.
//!
//! Still ungated here (both land with the device-side PLE work in #1106): the
//! n-gram table lookup/gather that turns `ngram_ids` into the 160-dim head
//! vectors, and the GPU dilated depthwise conv. This gate pins the hash that
//! addresses the table and the arithmetic around the conv, on the host.

use half::bf16;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde_json::Value;
use sha2::Digest;

/// Metadata key carrying the fixture manifest.
const META_KEY: &str = "qwen4_ple_golden";

/// Layout revision this reader implements. The dump script bumps it on any layout
/// change; a mismatch fails loudly rather than being misread.
const FORMAT_VERSION: u64 = 2;

/// Repo-relative frozen-config path the manifest must name.
const CONFIG_PATH: &str = "pegainfer-qwen4/tests/frozen_config.json";

/// Floor of the gate's signed sqrt, matching the reference's `clamp_min(1e-6)`.
const GATE_CLAMP_MIN: f32 = 1e-6;

/// Block-output tolerance, absolute.
///
/// The measured residual on the recording platform is **0.0**: this replay is
/// bit-identical to the stored `out` for both cases (0 of 81920 and 0 of 122880
/// elements differ), because it emulates each bf16 rounding point of the
/// reference chain — including the gate's *double* rounding, where the group sum
/// is rounded to bf16 before the `1/sqrt(hidden)` divide, and again after it.
///
/// The allowance covers the one non-portable op in that chain, `expf` (inside
/// `sigmoid` and `silu`); `sqrt`, division and multiplication are IEEE
/// correctly-rounded and the f64 reductions below are this file's own, so both
/// are machine-independent. A perturbation study — nudging *every* `sigmoid` and
/// *every* `silu` output by a whole bf16 ulp, far more than any libm
/// disagreement — moves `out` by at most 7.8e-3, i.e. 2 bf16 ulps at the largest
/// stored |out| (0.8438). 1.6e-2 is 2x that bound, and still 17x below the
/// weakest negative control (unsigned sqrt, 0.27); dilation 1 deviates by 0.79.
const OUT_MAX_ABS: f32 = 1.6e-2;

/// The frozen checkpoint's geometry, read from the same file the manifest's
/// `config_sha256` is checked against.
struct Geometry {
    hidden: usize,
    hc: usize,
    ngram_size: usize,
    heads_per_ngram: usize,
    conv_kernel: usize,
    eps: f32,
    eos: i64,
}

impl Geometry {
    /// Hyper-connection stream width: `hc_count * hidden_size`.
    fn width(&self) -> usize {
        self.hc * self.hidden
    }

    /// Hashed n-gram heads: `(ngram_size - 1) * heads_per_ngram`.
    fn heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// `sqrt(hidden_size)` in the f32 the reference's opmath divides by. The
    /// gate divides by this rather than multiplying by its reciprocal: the two
    /// round differently.
    fn sqrt_hidden(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }
}

fn geometry() -> Geometry {
    let cfg: Value = serde_json::from_str(include_str!("frozen_config.json"))
        .expect("frozen_config.json must parse");
    let text = &cfg["text_config"];
    let field = |name: &str| -> usize {
        text[name]
            .as_u64()
            .unwrap_or_else(|| panic!("frozen_config.json text_config.{name} must be an integer"))
            as usize
    };
    let geo = Geometry {
        hidden: field("hidden_size"),
        hc: field("hc_count"),
        ngram_size: field("ngram_size"),
        heads_per_ngram: field("heads_per_ngram"),
        conv_kernel: field("ple_conv_kernel_size"),
        eps: text["rms_norm_eps"]
            .as_f64()
            .expect("text_config.rms_norm_eps must be a number") as f32,
        eos: text["eos_token_id"]
            .as_i64()
            .expect("text_config.eos_token_id must be an integer"),
    };
    assert!(
        geo.ngram_size >= 2 && geo.heads_per_ngram > 0 && geo.conv_kernel > 0,
        "degenerate PLE geometry: ngram_size {} heads_per_ngram {} conv_kernel {}",
        geo.ngram_size,
        geo.heads_per_ngram,
        geo.conv_kernel
    );
    geo
}

/// One recorded sequence: the block's inputs and both expected outputs.
struct Case {
    name: String,
    input_ids: Vec<i64>,
    ngram_ids: Vec<i64>,
    hidden: Vec<bf16>,
    key: Vec<bf16>,
    value: Vec<bf16>,
    out: Vec<bf16>,
}

impl Case {
    fn seq_len(&self) -> usize {
        self.input_ids.len()
    }
}

/// The whole fixture, decoded to owned vectors (the safetensors view borrows the
/// file bytes).
struct Fixture {
    geo: Geometry,
    eos: i64,
    /// One odd multiplier per context position, XOR-folded into the hash.
    multipliers: Vec<i64>,
    /// Per-head prime modulus, then per-head base offset into the n-gram table.
    head_vocab: Vec<i64>,
    head_offsets: Vec<i64>,
    norm_key: Vec<bf16>,
    norm_query: Vec<bf16>,
    norm_conv: Vec<bf16>,
    /// Depthwise conv weight, `[width][1][conv_kernel]` flattened.
    conv1d: Vec<bf16>,
    cases: Vec<Case>,
}

fn i64_tensor(st: &SafeTensors<'_>, name: &str) -> (Vec<i64>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I64, "{name} must be i64");
    let values = t
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect();
    (values, t.shape().to_vec())
}

fn bf16_tensor(st: &SafeTensors<'_>, name: &str, width: usize) -> Vec<bf16> {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::BF16, "{name} must be bf16");
    let shape = t.shape().to_vec();
    assert_eq!(
        shape.last().copied(),
        Some(width),
        "{name} must be {} wide, got shape {shape:?}",
        width
    );
    t.data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
        .collect()
}

impl Fixture {
    fn load() -> Fixture {
        let geo = geometry();
        let path = format!(
            "{}/../test_data/qwen38fn-ple-golden.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "read {path}: {e} — the PLE golden is committed; regenerate it with \
                 tools/accuracy/dump_qwen38fn_ple_golden.py only if the layout changes"
            )
        });
        let st = SafeTensors::deserialize(&bytes).expect("parse the PLE golden safetensors");
        check_provenance(&bytes);

        let (eos, eos_shape) = i64_tensor(&st, "eos_token_id");
        assert_eq!(eos_shape, vec![1], "eos_token_id must be one scalar");
        assert_eq!(
            eos[0], geo.eos,
            "the fixture's eos_token_id must be the frozen config's"
        );

        let (multipliers, mult_shape) = i64_tensor(&st, "layer_multipliers");
        assert_eq!(
            mult_shape,
            vec![geo.ngram_size],
            "layer_multipliers must carry one multiplier per context position"
        );
        let (head_vocab, vocab_shape) = i64_tensor(&st, "ngram_heads_vocab_sizes");
        let (head_offsets, offset_shape) = i64_tensor(&st, "ngram_heads_offsets");
        assert_eq!(
            (vocab_shape, offset_shape),
            (vec![geo.heads()], vec![geo.heads()]),
            "both n-gram head buffers must be one entry per head"
        );
        assert!(
            head_vocab.iter().all(|&size| size > 1)
                && head_vocab.windows(2).all(|w| w[1] > w[0])
                && head_offsets[0] == 0
                && head_offsets
                    .windows(2)
                    .zip(&head_vocab)
                    .all(|(w, &size)| w[1] == w[0] + size),
            "the head moduli must be an increasing run with contiguous offsets"
        );

        let width = geo.width();
        let conv1d = bf16_tensor(&st, "conv1d_weight", geo.conv_kernel);
        let cases = load_cases(&st, &geo);
        Fixture {
            geo,
            eos: eos[0],
            multipliers,
            head_vocab,
            head_offsets,
            norm_key: bf16_tensor(&st, "norm_key", width),
            norm_query: bf16_tensor(&st, "norm_query", width),
            norm_conv: bf16_tensor(&st, "norm_conv", width),
            conv1d,
            cases,
        }
    }
}

/// The manifest is the fixture's provenance: it must parse, carry the layout this
/// reader implements, and point at the frozen config by content hash.
fn check_provenance(bytes: &[u8]) {
    let (_, header) = SafeTensors::read_metadata(bytes).expect("parse the PLE golden header");
    let manifest = header
        .metadata()
        .as_ref()
        .and_then(|meta| meta.get(META_KEY))
        .unwrap_or_else(|| panic!("golden has no {META_KEY:?} metadata manifest"))
        .clone();
    let manifest: Value =
        serde_json::from_str(&manifest).expect("the golden manifest must be JSON");
    let version = manifest["format_version"]
        .as_u64()
        .expect("manifest.format_version must be an integer");
    assert_eq!(
        version, FORMAT_VERSION,
        "fixture layout {version} is not the {FORMAT_VERSION} this reader implements"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG_PATH),
        "manifest.config must name the frozen config this reader hashes"
    );
    let digest = sha2::Sha256::digest(include_str!("frozen_config.json").as_bytes());
    let hex = digest
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            use std::fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(hex.as_str()),
        "the fixture was recorded against a different {CONFIG_PATH}"
    );
}

/// Decode every `caseN_*` group, checking the shapes against the config geometry
/// as it goes.
fn load_cases(st: &SafeTensors<'_>, geo: &Geometry) -> Vec<Case> {
    let width = geo.width();
    let heads = geo.heads();
    let mut cases = Vec::new();
    for index in 0.. {
        let name = format!("case{index}");
        if st.tensor(&format!("{name}_seq_len")).is_err() {
            break;
        }
        let (seq_len_cell, seq_shape) = i32_tensor(st, &format!("{name}_seq_len"));
        assert_eq!(seq_shape, vec![1], "{name}_seq_len must be one cell");
        let seq_len = usize::try_from(seq_len_cell[0])
            .ok()
            .filter(|&len| len > 0)
            .unwrap_or_else(|| panic!("{name}_seq_len must be a positive row count"));

        let (rows, rows_shape) = i32_tensor(st, &format!("{name}_rows"));
        assert_eq!(rows_shape, vec![seq_len], "{name}_rows must be one per row");
        assert!(
            rows.iter().enumerate().all(|(i, &row)| row == i as i32),
            "{name}_rows must be the identity 0..{seq_len}: this replay is whole-sequence (the \
             dump records no cache state), so a row selection would need the conv and window \
             state of the dropped rows to stay correct"
        );

        let (input_ids, id_shape) = i64_tensor(st, &format!("{name}_input_ids"));
        assert_eq!(
            id_shape,
            vec![seq_len],
            "{name}_input_ids must be one per row"
        );
        let (ngram_ids, ngram_shape) = i64_tensor(st, &format!("{name}_ngram_ids"));
        assert_eq!(
            ngram_shape,
            vec![seq_len, heads],
            "{name}_ngram_ids must be [rows, heads]"
        );
        cases.push(Case {
            name,
            input_ids,
            ngram_ids,
            hidden: bf16_tensor(st, &format!("case{index}_hidden"), width),
            key: bf16_tensor(st, &format!("case{index}_key"), width),
            value: bf16_tensor(st, &format!("case{index}_value"), geo.hidden),
            out: bf16_tensor(st, &format!("case{index}_out"), width),
        });
    }
    assert!(
        cases.len() >= 2,
        "the dump records two case groups (S=8 without EOS, S=12 with a mid-sequence EOS) but \
         only {} decoded — a group is missing or the naming changed",
        cases.len()
    );
    for case in &cases {
        let rows = case.seq_len();
        assert!(
            case.hidden.len() == rows * width
                && case.key.len() == rows * width
                && case.out.len() == rows * width
                && case.value.len() == rows * geo.hidden
                && case.ngram_ids.len() == rows * heads,
            "{}: an activation does not cover {} rows of the config geometry",
            case.name,
            rows
        );
    }
    cases
}

fn i32_tensor(st: &SafeTensors<'_>, name: &str) -> (Vec<i32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name} must be i32");
    let values = t
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    (values, t.shape().to_vec())
}

/// Knobs the negative controls turn. The defaults are the reference behaviour.
#[derive(Clone, Copy)]
struct Hash {
    /// `_shift_right_ignore_eos`' cummax segment reset: n-grams never cross an
    /// EOS boundary. Off = one window for the whole history.
    eos_reset: bool,
    /// Head layout, bigram block then trigram block. Off = the two swapped.
    bigram_first: bool,
}

impl Hash {
    fn reference() -> Self {
        Self {
            eos_reset: true,
            bigram_first: true,
        }
    }
}

#[derive(Clone, Copy)]
struct Block {
    /// The conv's dilation; the reference uses `ngram_size`.
    dilation: usize,
    /// Restore the gate's sign after the sqrt, as the reference does.
    signed_sqrt: bool,
}

impl Block {
    fn reference(geo: &Geometry) -> Self {
        Self {
            dilation: geo.ngram_size,
            signed_sqrt: true,
        }
    }
}

/// The reference's `_shift_right_ignore_eos` on one sequence: the token `shift`
/// positions back, or `eos` where the segment does not reach that far.
///
/// `eos_reset` reproduces the cummax over EOS positions that makes
/// `segment_start` the position after the last EOS at or before `p`, so a window
/// never pulls a token from across a segment boundary.
fn shift_right_ignore_eos(history: &[i64], shift: usize, eos: i64, eos_reset: bool) -> Vec<i64> {
    if shift == 0 {
        return history.to_vec();
    }
    // `previous_eos_inclusive[p]`: the last index <= p holding EOS, or -1.
    let mut previous_inclusive = vec![-1i64; history.len()];
    let mut running = -1i64;
    for (p, &token) in history.iter().enumerate() {
        if token == eos {
            running = p as i64;
        }
        previous_inclusive[p] = running;
    }
    (0..history.len())
        .map(|p| {
            let source = p as i64 - shift as i64;
            // `previous_eos` excludes p itself: an EOS token starts the *next*
            // segment, it does not belong to the one it closes.
            let previous_eos = if p == 0 {
                -1
            } else {
                previous_inclusive[p - 1]
            };
            let in_segment = if eos_reset {
                p as i64 - (previous_eos + 1)
            } else {
                p as i64
            };
            if in_segment >= shift as i64 && source >= 0 {
                history[source as usize]
            } else {
                eos
            }
        })
        .collect()
}

/// Recompute `ngram_ids`: `[seq_len, heads]` flattened, row-major.
///
/// Port of `Qwen4ExpTextNGramEmbedding.forward`. The history is the `ngram_size -
/// 1` EOS-padded context tokens followed by the sequence (a full-sequence
/// forward has no cache state); each order XOR-folds `token * multiplier` over
/// its window, then every head reduces that fold modulo its own prime and adds
/// its own table offset. The reference drops the warm-up rows and concatenates
/// the per-order head blocks, bigram first.
fn ngram_ids(fx: &Fixture, input_ids: &[i64], knobs: Hash) -> Vec<i64> {
    let geo = &fx.geo;
    let context = geo.ngram_size - 1;
    let heads = geo.heads_per_ngram;
    let total_heads = geo.heads();
    let mut history = vec![fx.eos; context];
    history.extend_from_slice(input_ids);
    let shifted: Vec<Vec<i64>> = (0..geo.ngram_size)
        .map(|shift| shift_right_ignore_eos(&history, shift, fx.eos, knobs.eos_reset))
        .collect();

    let mut ids = vec![0i64; input_ids.len() * total_heads];
    let orders: Vec<usize> = if knobs.bigram_first {
        (2..=geo.ngram_size).collect()
    } else {
        (2..=geo.ngram_size).rev().collect()
    };
    for (slot, &order) in orders.iter().enumerate() {
        // Heads `[slot * heads, (slot + 1) * heads)` carry this order; the head's
        // own prime and offset come from the order-independent stored buffers.
        let base_head = (order - 2) * heads;
        for row in 0..input_ids.len() {
            let p = row + context;
            // Every `token * multiplier` is non-negative — the reference bounds
            // each multiplier by `(2^63 - 1) / vocab_size` so the product cannot
            // overflow — and an XOR of non-negative i64s keeps the sign bit clear,
            // so the fold is never negative (measured: 0 of 40 folds). The
            // wrapping multiply is faithful rather than load-bearing.
            let mut mixed = shifted[0][p].wrapping_mul(fx.multipliers[0]);
            for (position, window) in shifted.iter().enumerate().take(order).skip(1) {
                mixed ^= window[p].wrapping_mul(fx.multipliers[position]);
            }
            for head in 0..heads {
                let table_head = base_head + head;
                // `rem_euclid`, not `%`: `torch.remainder` is a floored remainder,
                // which only differs from Rust's truncated one on a negative fold.
                let bucket = mixed.rem_euclid(fx.head_vocab[table_head]);
                ids[row * total_heads + slot * heads + head] = bucket + fx.head_offsets[table_head];
            }
        }
    }
    ids
}

/// The reference's grouped RMSNorm: `bf16(x_f32 * rsqrt(mean(x^2) + eps) * (1 +
/// w))` over `hc_count` groups of `hidden_size`, with `x^2` rounded to f32 per
/// element exactly as `pow(2)` does.
///
/// The mean accumulates in f64. The reference's f32 tree reduction lands within
/// an f32 ulp of that, and the trailing bf16 rounding absorbs the difference —
/// verified bit-identical end to end on both cases — so the exact sum is the
/// portable choice: it does not depend on the shape of the reference's tree.
fn grouped_norm(x: &[bf16], weight: &[bf16], geo: &Geometry) -> Vec<bf16> {
    let width = geo.width();
    assert!(
        !x.is_empty() && x.len().is_multiple_of(width) && weight.len() == width,
        "a grouped norm needs whole rows of {width} channels and one weight each"
    );
    let mut out = vec![bf16::ZERO; x.len()];
    for (row, chunk) in x.chunks(width).enumerate() {
        for base in (0..width).step_by(geo.hidden) {
            let group = &chunk[base..base + geo.hidden];
            let sum = group
                .iter()
                .map(|v| {
                    let f = v.to_f32();
                    f64::from(f * f)
                })
                .sum::<f64>();
            let mean = (sum as f32) / geo.hidden as f32;
            let scale = 1.0 / (mean + geo.eps).sqrt();
            for (j, &v) in group.iter().enumerate() {
                let normed = v.to_f32() * scale;
                out[row * width + base + j] =
                    bf16::from_f32(normed * (1.0 + weight[base + j].to_f32()));
            }
        }
    }
    out
}

/// `torch.sign`: 0 at zero, unlike Rust's `signum` on `-0.0`.
fn sign(x: f32) -> f32 {
    f32::from(x > 0.0) - f32::from(x < 0.0)
}

fn sigmoid(x: bf16) -> bf16 {
    let v = x.to_f32();
    bf16::from_f32(1.0 / (1.0 + (-v).exp()))
}

fn silu(x: bf16) -> bf16 {
    let v = x.to_f32();
    bf16::from_f32(v / (1.0 + (-v).exp()))
}

/// Replay the PLE block on one case: `[seq_len, hc_count * hidden_size]`.
///
/// Port of `Qwen4ExpTextPLELayer.forward` from its two recorded projections, with
/// a bf16 rounding wherever the reference's submodule boundaries put one.
fn ple_block(fx: &Fixture, case: &Case, knobs: Block) -> Vec<bf16> {
    let geo = &fx.geo;
    let width = geo.width();
    let seq_len = case.seq_len();
    let key_normed = grouped_norm(&case.key, &fx.norm_key, geo);
    let query_normed = grouped_norm(&case.hidden, &fx.norm_query, geo);
    let sqrt_hidden = geo.sqrt_hidden();

    // `sigmoid(signed_sqrt(dot(key, query) / sqrt(hidden))) * value`, broadcast
    // over the `hc_count` streams that share one value vector.
    let mut gated = vec![bf16::ZERO; seq_len * width];
    for row in 0..seq_len {
        for group in 0..geo.hc {
            let mut acc = 0f64;
            for j in 0..geo.hidden {
                let i = row * width + group * geo.hidden + j;
                let product = bf16::from_f32(key_normed[i].to_f32() * query_normed[i].to_f32());
                acc += f64::from(product.to_f32());
            }
            // Two bf16 roundings: the sum, then the divide. Collapsing them into
            // one is worth ~1 ulp on every gate and breaks bit-exactness.
            let sum = bf16::from_f32(acc as f32);
            let pre_gate = bf16::from_f32(sum.to_f32() / sqrt_hidden);
            let magnitude = bf16::from_f32(pre_gate.to_f32().abs().max(GATE_CLAMP_MIN));
            let root = bf16::from_f32(magnitude.to_f32().sqrt());
            let restored = if knobs.signed_sqrt {
                sign(pre_gate.to_f32())
            } else {
                1.0
            };
            let gate = bf16::from_f32(root.to_f32() * restored);
            let weight = sigmoid(gate).to_f32();
            for j in 0..geo.hidden {
                let value = case.value[row * geo.hidden + j].to_f32();
                gated[row * width + group * geo.hidden + j] = bf16::from_f32(weight * value);
            }
        }
    }

    // `gated + silu(causal dilated depthwise conv(norm_conv(gated)))` over a zero
    // initial state of `(kernel - 1) * dilation` positions.
    let conv_in = grouped_norm(&gated, &fx.norm_conv, geo);
    let state = (geo.conv_kernel - 1) * knobs.dilation;
    let mut out = vec![bf16::ZERO; seq_len * width];
    for channel in 0..width {
        for row in 0..seq_len {
            let mut acc = 0f32;
            for tap in 0..geo.conv_kernel {
                let source = row as i64 - (state - knobs.dilation * tap) as i64;
                if source >= 0 {
                    acc += conv_in[source as usize * width + channel].to_f32()
                        * fx.conv1d[channel * geo.conv_kernel + tap].to_f32();
                }
            }
            let convolved = bf16::from_f32(acc);
            out[row * width + channel] =
                bf16::from_f32(gated[row * width + channel].to_f32() + silu(convolved).to_f32());
        }
    }
    out
}

/// Element-wise agreement of a replay against the stored tensor.
struct Delta {
    max_abs: f32,
    mean_abs: f32,
    identical: usize,
    total: usize,
}

fn delta(got: &[bf16], want: &[bf16]) -> Delta {
    assert_eq!(got.len(), want.len(), "replay shape must match the golden");
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut identical = 0;
    for (g, w) in got.iter().zip(want) {
        let d = (g.to_f32() - w.to_f32()).abs();
        max_abs = max_abs.max(d);
        sum_abs += f64::from(d);
        identical += usize::from(g.to_bits() == w.to_bits());
    }
    Delta {
        max_abs,
        mean_abs: (sum_abs / want.len() as f64) as f32,
        identical,
        total: want.len(),
    }
}

fn eos_positions(case: &Case, eos: i64) -> Vec<usize> {
    case.input_ids
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == eos)
        .map(|(p, _)| p)
        .collect()
}

#[test]
fn the_ngram_hash_replays_exactly() {
    let fx = Fixture::load();
    for case in &fx.cases {
        let got = ngram_ids(&fx, &case.input_ids, Hash::reference());
        let mismatched = got
            .iter()
            .zip(&case.ngram_ids)
            .filter(|(g, w)| g != w)
            .count();
        println!(
            "{}: S={} eos_at={:?} heads={} ids={}/{} exact",
            case.name,
            case.seq_len(),
            eos_positions(case, fx.eos),
            fx.geo.heads(),
            case.ngram_ids.len() - mismatched,
            case.ngram_ids.len()
        );
        assert_eq!(
            mismatched,
            0,
            "{}: {} of {} hashed n-gram ids differ from the golden (zero tolerance: this is \
             integer math)",
            case.name,
            mismatched,
            case.ngram_ids.len()
        );
    }
}

#[test]
fn the_ple_block_replays_within_the_bf16_noise_floor() {
    let fx = Fixture::load();
    for case in &fx.cases {
        let got = ple_block(&fx, case, Block::reference(&fx.geo));
        let d = delta(&got, &case.out);
        println!(
            "{}: S={} out max_abs={:.3e} mean_abs={:.3e} bit_identical={}/{} tol={:.1e} \
             dilation={} signed_sqrt=true",
            case.name,
            case.seq_len(),
            d.max_abs,
            d.mean_abs,
            d.identical,
            d.total,
            OUT_MAX_ABS,
            fx.geo.ngram_size
        );
        assert!(
            d.max_abs <= OUT_MAX_ABS,
            "{}: PLE block output deviates by {:.3e} > {:.3e} (mean {:.3e}, {}/{} elements \
             differ); see OUT_MAX_ABS for how the bound is derived",
            case.name,
            d.max_abs,
            OUT_MAX_ABS,
            d.mean_abs,
            d.total - d.identical,
            d.total
        );
    }
}

/// Control (a): without the cummax EOS reset a window reaches across a segment
/// boundary, so the case with a mid-sequence EOS must change — and the EOS-free
/// case must not, which is what makes this a targeted control rather than a
/// blanket break. The change is narrow by construction: only the row right after
/// the EOS pulls a token from across it, and only for the orders whose window
/// reaches that far (the trigram heads, 8 of the 16).
#[test]
fn dropping_the_eos_window_reset_breaks_the_eos_case() {
    let fx = Fixture::load();
    let knobs = Hash {
        eos_reset: false,
        ..Hash::reference()
    };
    let mut broke = 0;
    for case in &fx.cases {
        let positions = eos_positions(case, fx.eos);
        let got = ngram_ids(&fx, &case.input_ids, knobs);
        let mismatched = got
            .iter()
            .zip(&case.ngram_ids)
            .filter(|(g, w)| g != w)
            .count();
        println!(
            "{}: S={} eos_at={:?} -> {}/{} ids differ without the EOS reset",
            case.name,
            case.seq_len(),
            positions,
            mismatched,
            case.ngram_ids.len()
        );
        if positions.is_empty() {
            assert_eq!(
                mismatched, 0,
                "{} has no EOS, so dropping the reset must leave its ids alone",
                case.name
            );
        } else {
            assert!(
                mismatched > 0,
                "{} carries an EOS at {positions:?} but dropping the reset changed nothing — the \
                 control no longer exercises the segment boundary",
                case.name
            );
            broke += 1;
        }
    }
    assert!(
        broke > 0,
        "no case in the fixture carries a mid-sequence EOS, so the reset is untested"
    );
}

/// Control (d): the head layout is (order, head) with the bigram block first.
/// Swapping the two blocks keeps every id in range, so only a layout assertion
/// catches it.
#[test]
fn swapping_the_bigram_and_trigram_heads_breaks_the_hash() {
    let fx = Fixture::load();
    let knobs = Hash {
        bigram_first: false,
        ..Hash::reference()
    };
    for case in &fx.cases {
        let got = ngram_ids(&fx, &case.input_ids, knobs);
        let mismatched = got
            .iter()
            .zip(&case.ngram_ids)
            .filter(|(g, w)| g != w)
            .count();
        println!(
            "{}: {}/{} ids differ with the order blocks swapped",
            case.name,
            mismatched,
            case.ngram_ids.len()
        );
        assert_eq!(
            mismatched,
            case.ngram_ids.len(),
            "{}: swapping bigram/trigram must move every id — the two orders occupy disjoint \
             offset ranges and each remainder is smaller than the gap between them, so a swapped \
             id can never land on the stored one",
            case.name
        );
    }
}

/// Control (b): the conv is dilated by `ngram_size`, with a `(kernel - 1) *
/// dilation` zero state. Dilation 1 reads the wrong taps.
#[test]
fn dilation_one_breaks_the_block_output() {
    let fx = Fixture::load();
    let knobs = Block {
        dilation: 1,
        ..Block::reference(&fx.geo)
    };
    assert_block_control_fails(&fx, knobs, "dilation 1");
}

/// Control (c): the gate is `sign(g) * sqrt(|g|)`. Dropping the sign restore
/// pushes every negative gate through `sigmoid` as positive, which inverts the
/// value's contribution for those streams.
#[test]
fn an_unsigned_sqrt_breaks_the_block_output() {
    let fx = Fixture::load();
    let knobs = Block {
        signed_sqrt: false,
        ..Block::reference(&fx.geo)
    };
    assert_block_control_fails(&fx, knobs, "unsigned sqrt");
}

fn assert_block_control_fails(fx: &Fixture, knobs: Block, label: &str) {
    let mut worst = 0f32;
    for case in &fx.cases {
        let got = ple_block(fx, case, knobs);
        let d = delta(&got, &case.out);
        worst = worst.max(d.max_abs);
        println!(
            "{}: out max_abs={:.3e} mean_abs={:.3e} differing={}/{} under {label} (tol {:.1e})",
            case.name,
            d.max_abs,
            d.mean_abs,
            d.total - d.identical,
            d.total,
            OUT_MAX_ABS
        );
        assert!(
            d.max_abs > OUT_MAX_ABS,
            "{}: {label} stayed within {:.3e} of the golden, so the block replay does not pin the \
             knob this control turns",
            case.name,
            OUT_MAX_ABS
        );
    }
    println!(
        "{label}: worst deviation {worst:.3e} = {:.0}x the tolerance",
        worst / OUT_MAX_ABS
    );
}
