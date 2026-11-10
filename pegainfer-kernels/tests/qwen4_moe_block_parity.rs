//! Gate for Qwen3.8-Flash-Next's routed experts: the fused expert-weight stack
//! and the per-token expert computation (`Qwen4ExpTextExperts`,
//! `modeling_qwen4_exp.py:859-895` in transformers 5.16.1).
//!
//! The comparison is against an **f64 CPU oracle that walks the reference's own
//! sequence**, rounding to bf16 at exactly the reference's rounding points:
//!
//! ```text
//! gate, up = project(x, gate_up_proj[e]).chunk(2)     // bf16 store
//! h        = act(gate) * up                            // silu, then bf16 product
//! y        = project(h, down_proj[e])                  // bf16 store
//! y       *= routing_weight[token, slot]               // bf16 store
//! final   .index_add_(0, token, y)                     // bf16 add, per contribution
//! ```
//!
//! It calls no kernel and shares no code with the device path, so agreement is
//! not a same-source comparison. The block is gated in the four parts the issue
//! asks for — router logits/weights, shared branch, routed experts, combine —
//! because a final-output mismatch cannot be attributed to one of them.
//!
//! **Numeric relationship.** The accumulation *order* and every bf16 rounding
//! point are reproduced exactly; the projections' internal f32 accumulation is
//! cuBLAS's own and is not the oracle's f64 sum, so above the fold agreement is
//! a bound rather than bit equality. The bound is derived, not tuned. An f32 sum
//! of `n` bf16 products differs from the f64 sum by at most `γ_n · Σ|x·w|` with
//! `γ_n ≈ n · 2^-24 ≈ 1.5e-4` at `n = 2560`; a dot of random-sign products
//! cancels by about `√n`, so on the loudest elements that error reaches ~7.5e-3
//! of the dot itself — just under one bf16 step (`2^-7` of the magnitude), which
//! is why the projections agree to **at most one step** rather than bitwise.
//! Two projection stores and two activation roundings then leave a handful of
//! steps of total slack, and the routed fold adds one step per contribution for
//! ten contributions. Deviations are measured in units of the **row's largest
//! reference output**, the scale a cancellation-prone reduction has to be judged
//! against (a per-element relative bound would explode wherever the reference
//! itself nearly cancels), and each comparison prints the fraction of its bound
//! it used.
//!
//! The *fold* alone — the stage the order lives in — is gated **bitwise** in
//! [`the_fold_reproduces_the_reference_accumulation_order`], on inputs built so
//! that slot order and expert order differ in the bits.
//!
//! Manual gate: no CI row enables `qwen4`. Run on a GPU box with
//!
//! ```text
//! cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_moe_block_parity -- --nocapture
//! ```
//!
//! `PEGAINFER_REQUIRE_GPU=1` turns a missing device into a failure rather than a
//! skip. The real-geometry case holds ~5.0 GB of expert weights, so the device
//! needs ~6 GB free; the cases in
//! [`the_routed_path_rejects_a_mismatched_call`] stay under 20 MB.

mod common;

use std::time::Instant;

use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use half::bf16;
use pegainfer_kernels::ops::Qwen4ExpertLayout;
use pegainfer_kernels::ops::Qwen4ExpertStacks;
use pegainfer_kernels::ops::Qwen4MoeExpertScratch;
use pegainfer_kernels::ops::Qwen4MoeGeometry;
use pegainfer_kernels::ops::Qwen4SharedBranchScratch;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::qwen4_moe_combine_into;
use pegainfer_kernels::ops::qwen4_moe_route_into;
use pegainfer_kernels::ops::qwen4_moe_route_scale_fold_into;
use pegainfer_kernels::ops::qwen4_moe_routed_experts_into;
use pegainfer_kernels::ops::qwen4_moe_shared_branch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

/// Repetitions behind the recorded timing.
const TIMED_REPEATS: usize = 20;

/// The frozen text geometry (`pegainfer-qwen4/tests/frozen_config.json`).
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;

// ── deterministic inputs ─────────────────────────────────────────────────

/// Deterministic noise in `[-scale, scale)`, distinct per element so a row read
/// from the wrong expert, column or token cannot coincide with the right one.
fn noise(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push(bf16::from_f32(
            (((state >> 40) as f32 / 8_388_608.0) - 1.0) * scale,
        ));
    }
    out
}

/// Projection weights are scaled by `1/sqrt(width)` so an output keeps O(1)
/// magnitude whatever the reduction length is.
fn stack_scale(width: usize) -> f32 {
    1.0 / (width as f32).sqrt()
}

// ── the reference's rounding primitives, in f64 ──────────────────────────

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

/// `value` rounded to bf16 and read back — one reference store.
fn rounded(value: f64) -> f64 {
    f64::from(bf16::from_f64(value).to_f32())
}

/// `y = F.linear(x, weight)` over one row of a row-major `[rows, cols]` weight
/// block starting at `row_offset`: f64 accumulation, one bf16 store.
fn project(weight: &[bf16], row_offset: usize, rows: usize, cols: usize, x: &[f64]) -> Vec<bf16> {
    assert_eq!(x.len(), cols, "input row is not {cols} wide");
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = (row_offset + row) * cols;
        let mut acc = 0.0f64;
        for col in 0..cols {
            acc += x[col] * f64::from(weight[base + col].to_f32());
        }
        out.push(bf16::from_f64(acc));
    }
    out
}

/// The reference's `act(gate) * up` over a fused `[2 * intermediate]` row: the
/// activation is rounded to bf16 before the product, as `torch.silu` on a bf16
/// tensor materialises it.
fn gated_activation(gate_up: &[bf16], intermediate: usize) -> Vec<bf16> {
    assert_eq!(gate_up.len(), 2 * intermediate);
    (0..intermediate)
        .map(|col| {
            let gate = rounded(silu(f64::from(gate_up[col].to_f32())));
            let up = f64::from(gate_up[intermediate + col].to_f32());
            bf16::from_f64(gate * up)
        })
        .collect()
}

fn row_f64(row: &[bf16]) -> Vec<f64> {
    row.iter().map(|value| f64::from(value.to_f32())).collect()
}

// ── the router's host canonical implementation ───────────────────────────

/// f64 softmax over the stored bf16 logits, top-k in the canonical order
/// (descending probability, ascending expert index inside a tie), optional
/// `norm_topk_prob` renormalization, one bf16 store per weight.
fn router_reference(
    logits: &[bf16],
    tokens: usize,
    experts: usize,
    top_k: usize,
    renorm: bool,
) -> (Vec<i32>, Vec<bf16>) {
    assert_eq!(logits.len(), tokens * experts);
    let mut ids = Vec::with_capacity(tokens * top_k);
    let mut weights = Vec::with_capacity(tokens * top_k);
    for token in 0..tokens {
        let row = &logits[token * experts..(token + 1) * experts];
        let peak = row
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = row
            .iter()
            .map(|value| (f64::from(value.to_f32()) - peak).exp())
            .collect();
        let total: f64 = probs.iter().sum();
        for prob in &mut probs {
            *prob /= total;
        }
        let mut taken = vec![false; experts];
        let mut picks = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            let mut best = usize::MAX;
            let mut best_prob = f64::NEG_INFINITY;
            for (expert, prob) in probs.iter().enumerate() {
                if !taken[expert] && (best == usize::MAX || *prob > best_prob) {
                    best = expert;
                    best_prob = *prob;
                }
            }
            taken[best] = true;
            picks.push((best, best_prob));
        }
        let selected: f64 = picks.iter().map(|(_, prob)| *prob).sum();
        for (expert, prob) in picks {
            ids.push(expert as i32);
            weights.push(bf16::from_f64(if renorm { prob / selected } else { prob }));
        }
    }
    (ids, weights)
}

// ── the routed experts' host reference ───────────────────────────────────

/// How one route batch is folded into per-token outputs. The variants exist so
/// the negative controls can state what a specific wrong fold would produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fold {
    /// The reference: the routing weight applied first, then contributions
    /// visited in ascending expert index with a bf16 rounding after every add.
    Reference,
    /// Slot order instead — what a port that folds the router's descending
    /// probabilities in place produces.
    SlotOrder,
    /// One f64 sum rounded once: the accumulator kept in f64 instead of being a
    /// bf16 tensor.
    NoStepRounding,
    /// The routing weight dropped.
    NoWeight,
}

/// One route batch's down-projection rows, `[tokens * top_k, hidden]`.
///
/// `swap_halves` reads each fused projection the other way round (the second
/// half as the gate), the control for "gate|up ordering".
#[allow(clippy::too_many_arguments)]
fn routed_rows_reference(
    x: &[bf16],
    gate_up: &[bf16],
    down: &[bf16],
    ids: &[i32],
    tokens: usize,
    top_k: usize,
    hidden: usize,
    intermediate: usize,
    swap_halves: bool,
) -> Vec<bf16> {
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(ids.len(), tokens * top_k);
    let gate_up_rows = 2 * intermediate;
    let mut rows = Vec::with_capacity(tokens * top_k * hidden);
    for token in 0..tokens {
        let x_row = row_f64(&x[token * hidden..(token + 1) * hidden]);
        for slot in 0..top_k {
            let expert = ids[token * top_k + slot] as usize;
            let fused = project(gate_up, expert * gate_up_rows, gate_up_rows, hidden, &x_row);
            let fused = if swap_halves {
                let mut swapped = Vec::with_capacity(gate_up_rows);
                swapped.extend_from_slice(&fused[intermediate..]);
                swapped.extend_from_slice(&fused[..intermediate]);
                swapped
            } else {
                fused
            };
            let act: Vec<f64> = gated_activation(&fused, intermediate)
                .iter()
                .map(|value| f64::from(value.to_f32()))
                .collect();
            rows.extend(project(down, expert * hidden, hidden, intermediate, &act));
        }
    }
    rows
}

/// The reference's fold, with the behavioural variants the controls name.
fn fold_reference(
    rows: &[bf16],
    weights: &[bf16],
    ids: &[i32],
    tokens: usize,
    top_k: usize,
    hidden: usize,
    variant: Fold,
) -> Vec<bf16> {
    assert_eq!(rows.len(), tokens * top_k * hidden);
    let mut out = vec![bf16::from_f32(0.0); tokens * hidden];
    for token in 0..tokens {
        let mut order: Vec<usize> = (0..top_k).collect();
        if variant == Fold::Reference {
            order.sort_by_key(|slot| ids[token * top_k + *slot]);
        }
        for col in 0..hidden {
            let mut acc = 0.0f64;
            for slot in &order {
                let route = token * top_k + slot;
                let value = f64::from(rows[route * hidden + col].to_f32());
                let contribution = if variant == Fold::NoWeight {
                    value
                } else {
                    rounded(value * f64::from(weights[route].to_f32()))
                };
                if variant == Fold::NoStepRounding {
                    acc += contribution;
                } else {
                    acc = rounded(acc + contribution);
                }
            }
            out[token * hidden + col] = bf16::from_f64(acc);
        }
    }
    out
}

// ── the shared branch and the block sum ──────────────────────────────────

/// The shared expert's reference: `down(silu(gate(x)) · up(x))`, its own bf16
/// store at each of the reference's five rounding points.
fn shared_reference(
    x: &[bf16],
    gate: &[bf16],
    up: &[bf16],
    down: &[bf16],
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Vec<bf16> {
    let mut out = Vec::with_capacity(tokens * hidden);
    for token in 0..tokens {
        let x_row = row_f64(&x[token * hidden..(token + 1) * hidden]);
        let gate_row = project(gate, 0, intermediate, hidden, &x_row);
        let up_row = project(up, 0, intermediate, hidden, &x_row);
        let mut fused = Vec::with_capacity(2 * intermediate);
        fused.extend_from_slice(&gate_row);
        fused.extend_from_slice(&up_row);
        let act: Vec<f64> = gated_activation(&fused, intermediate)
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect();
        out.extend(project(down, 0, hidden, intermediate, &act));
    }
    out
}

/// The block sum's reference:
/// `out = bf16(expert_out + bf16(bf16(sigmoid(gate_logit)) * shared_out))`.
fn combine_reference(
    gate_logit: &[bf16],
    routed: &[bf16],
    shared: &[bf16],
    tokens: usize,
    hidden: usize,
    plain_sum: bool,
) -> Vec<bf16> {
    let mut out = Vec::with_capacity(tokens * hidden);
    for (index, (routed, shared)) in routed.iter().zip(shared).enumerate() {
        let token = index / hidden;
        let logit = f64::from(gate_logit[token].to_f32());
        let gate = if plain_sum {
            1.0
        } else {
            rounded(1.0 / (1.0 + (-logit).exp()))
        };
        let gated = rounded(gate * f64::from(shared.to_f32()));
        out.push(bf16::from_f64(rounded(f64::from(routed.to_f32()) + gated)));
    }
    out
}

// ── comparison ───────────────────────────────────────────────────────────

/// What one comparison found, in units of each row's largest reference output.
#[derive(Debug, Clone, Copy)]
struct Check {
    worst: f32,
    at: usize,
    absolute: f32,
    row_scale: f32,
    differing_bits: usize,
    elements: usize,
}

impl Check {
    fn fraction(&self, bound: f32) -> f32 {
        self.worst / bound
    }

    fn report(&self, what: &str, bound: Option<f32>) {
        let bound = bound.map_or("-".to_string(), |bound| {
            format!("{bound:.3e} ({:.1}% used)", 100.0 * self.fraction(bound))
        });
        eprintln!(
            "  {what}: {} elements, worst {:.3e} at {} (bound {bound}), absolute {:.3e}, \
             row scale {:.3e}, {} elements differ bitwise",
            self.elements, self.worst, self.at, self.absolute, self.row_scale, self.differing_bits,
        );
    }
}

/// Compare `got` against `want` over `rows` rows of `cols`, normalizing each
/// element's deviation by **that row's largest reference magnitude**. Non-finite
/// values are rejected rather than ranked: a NaN compares false against
/// everything, so ranking one would let an all-NaN output through.
fn compare(got: &[bf16], want: &[bf16], rows: usize, cols: usize) -> Check {
    assert_eq!(got.len(), want.len(), "compared runs differ in length");
    assert_eq!(
        got.len(),
        rows * cols,
        "compared runs are not {rows}x{cols}"
    );
    let mut check = Check {
        worst: 0.0,
        at: 0,
        absolute: 0.0,
        row_scale: 0.0,
        differing_bits: 0,
        elements: rows * cols,
    };
    for row in 0..rows {
        let base = row * cols;
        let scale = want[base..base + cols]
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0f32, f32::max);
        check.row_scale = check.row_scale.max(scale);
        let denominator = scale.max(f32::MIN_POSITIVE);
        for col in 0..cols {
            let (got, want) = (got[base + col], want[base + col]);
            let (got, want) = (got.to_f32(), want.to_f32());
            assert!(
                got.is_finite() && want.is_finite(),
                "element {} is not finite: device {got}, oracle {want}",
                base + col
            );
            let deviation = (got - want).abs();
            check.absolute = check.absolute.max(deviation);
            check.differing_bits += usize::from(got.to_bits() != want.to_bits());
            let normalized = deviation / denominator;
            if normalized > check.worst {
                check.worst = normalized;
                check.at = base + col;
            }
        }
    }
    check
}

/// Assert a comparison is inside its bound.
fn assert_within(what: &str, got: &[bf16], want: &[bf16], rows: usize, cols: usize, bound: f32) {
    let check = compare(got, want, rows, cols);
    check.report(what, Some(bound));
    assert!(
        check.worst <= bound,
        "{what}: element {} deviates by {:.3e} in units of its row's largest output \
         ({:.3e} absolute), which is {:.1}% of the bound {bound:.3e}",
        check.at,
        check.worst,
        check.absolute,
        100.0 * check.fraction(bound)
    );
}

/// Assert a comparison **fails** on magnitude, and report the floor the fault
/// cleared the bound by. A control that stays inside the bound proves nothing
/// and fails here.
fn assert_rejected(
    what: &str,
    got: &[bf16],
    want: &[bf16],
    rows: usize,
    cols: usize,
    bound: f32,
) -> f32 {
    let check = compare(got, want, rows, cols);
    check.report(what, Some(bound));
    assert!(
        check.worst > bound,
        "{what}: the deliberate fault stayed inside the bound {bound:.3e} (worst {:.3e}), \
         so this control proves nothing",
        check.worst
    );
    check.worst / bound
}

/// Assert a comparison **fails** bitwise — the shape of an order-only control,
/// whose fault is a few bf16 steps and whose point is that it is observable at
/// all. Returns the deviation in units of the row scale.
fn assert_moves_bits(what: &str, got: &[bf16], want: &[bf16], rows: usize, cols: usize) -> f32 {
    let check = compare(got, want, rows, cols);
    check.report(what, None);
    assert!(
        check.differing_bits > 0,
        "{what}: the deliberate fault moved no bits, so this data cannot show it is observable"
    );
    check.worst
}

// ── device helpers ───────────────────────────────────────────────────────

fn upload(ctx: &DeviceContext, host: &[bf16], hidden_dim: usize, seq_len: usize) -> HiddenStates {
    HiddenStates::from_host(ctx, host, hidden_dim, seq_len).expect("H2D")
}

fn read_back(ctx: &DeviceContext, states: &HiddenStates) -> Vec<bf16> {
    let host = ctx.stream.clone_dtoh(&states.data).expect("D2H");
    ctx.sync().expect("sync");
    host
}

fn read_slice<T: DeviceRepr>(ctx: &DeviceContext, slice: &CudaSlice<T>) -> Vec<T> {
    let host = ctx.stream.clone_dtoh(slice).expect("D2H");
    ctx.sync().expect("sync");
    host
}

/// The MoE block's weights: the two fused expert stacks plus the router, the
/// shared branch and its gate. Deterministic, so a rerun on another card reads
/// the same bytes.
struct Weights {
    gate_up: Vec<bf16>,
    down: Vec<bf16>,
    router: Vec<bf16>,
    shared_gate: Vec<bf16>,
    shared_up: Vec<bf16>,
    shared_down: Vec<bf16>,
    shared_expert_gate: Vec<bf16>,
}

fn weights() -> Weights {
    Weights {
        gate_up: noise(
            0x51EE_D0001,
            EXPERTS * 2 * INTERMEDIATE * HIDDEN,
            stack_scale(HIDDEN),
        ),
        down: noise(
            0x51EE_D0002,
            EXPERTS * HIDDEN * INTERMEDIATE,
            stack_scale(INTERMEDIATE),
        ),
        router: noise(0x51EE_D0003, EXPERTS * HIDDEN, stack_scale(HIDDEN)),
        shared_gate: noise(0x51EE_D0004, INTERMEDIATE * HIDDEN, stack_scale(HIDDEN)),
        shared_up: noise(0x51EE_D0005, INTERMEDIATE * HIDDEN, stack_scale(HIDDEN)),
        shared_down: noise(
            0x51EE_D0006,
            HIDDEN * INTERMEDIATE,
            stack_scale(INTERMEDIATE),
        ),
        shared_expert_gate: noise(0x51EE_D0007, HIDDEN, stack_scale(HIDDEN)),
    }
}

/// Measured bounds, set from the derivation in the module docs; each comparison
/// prints the fraction it used.
mod bounds {
    /// The router's bf16 weights: the reference is an f64 softmax, the kernel an
    /// f32 one with a tree sum, and the store is the only rounding point.
    pub(super) const ROUTER_WEIGHT: f32 = 1.0 / 256.0;
    /// Two projection stores plus the activation's two roundings.
    pub(super) const SHARED: f32 = 1.0 / 64.0;
    /// Two batched projection stores, the activation's two roundings, the
    /// routing scale, and ten bf16-rounded adds.
    pub(super) const ROUTED: f32 = 1.0 / 32.0;
    /// The combine is elementwise: one sigmoid store, one product, one add.
    pub(super) const COMBINED: f32 = 1.0 / 64.0;
}

/// One case of the case matrix.
struct Case {
    name: &'static str,
    tokens: usize,
    /// Re-upload the router's picks sorted by *descending* expert id, so slot
    /// order is the reverse of the order the reference folds in.
    permuted: bool,
    /// Make every token's input row identical, so all tokens route to the same
    /// ten experts and the other 502 stay empty.
    skewed: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "decode, one token",
        tokens: 1,
        permuted: false,
        skewed: false,
    },
    Case {
        name: "prefill, 17 tokens",
        tokens: 17,
        permuted: false,
        skewed: false,
    },
    Case {
        name: "skewed routing, 12 identical tokens (502 experts empty)",
        tokens: 12,
        permuted: false,
        skewed: true,
    },
    Case {
        name: "prefill, 9 tokens, slots in descending expert order",
        tokens: 9,
        permuted: true,
        skewed: false,
    },
];

/// The four-way gate at the frozen geometry, plus the negative controls.
///
/// One test drives every case so the 5.0 GB of expert weights is built, uploaded
/// and dropped once.
#[test]
fn the_moe_block_matches_the_four_way_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let geometry = Qwen4MoeGeometry::qwen38_flash_next();
    geometry.validate().expect("the frozen geometry");
    let host = weights();

    let router = DeviceMatrix::from_host(&ctx, &host.router, EXPERTS, HIDDEN).expect("router up");
    let shared_gate =
        DeviceMatrix::from_host(&ctx, &host.shared_gate, INTERMEDIATE, HIDDEN).expect("gate up");
    let shared_up =
        DeviceMatrix::from_host(&ctx, &host.shared_up, INTERMEDIATE, HIDDEN).expect("up up");
    let shared_down =
        DeviceMatrix::from_host(&ctx, &host.shared_down, HIDDEN, INTERMEDIATE).expect("down up");
    let shared_expert_gate =
        DeviceMatrix::from_host(&ctx, &host.shared_expert_gate, 1, HIDDEN).expect("se gate up");
    let stacks =
        Qwen4ExpertStacks::from_host(&ctx, &geometry, INTERMEDIATE, &host.gate_up, &host.down)
            .expect("expert stacks up");

    let mut timing = Vec::new();
    for case in CASES {
        let tokens = case.tokens;
        let x = if case.skewed {
            let row = noise(0x51EE_D1000 ^ tokens as u64, HIDDEN, 1.0);
            let mut x = Vec::with_capacity(tokens * HIDDEN);
            for _ in 0..tokens {
                x.extend_from_slice(&row);
            }
            x
        } else {
            noise(0x51EE_D1000 ^ tokens as u64, tokens * HIDDEN, 1.0)
        };
        let hidden = upload(&ctx, &x, HIDDEN, tokens);
        eprintln!("case: {} ({tokens} tokens)", case.name);

        // 1. Router logits and weights.
        let mut logits = HiddenStates::zeros(&ctx, EXPERTS, tokens).expect("logits");
        gemm_into_checked(&ctx, &router, &hidden, &mut logits).expect("router GEMM");
        let logits_host = read_back(&ctx, &logits);
        let mut ids = ctx.stream.alloc_zeros::<i32>(tokens * TOP_K).expect("ids");
        let mut weights_dev = ctx
            .stream
            .alloc_zeros::<bf16>(tokens * TOP_K)
            .expect("weights");
        qwen4_moe_route_into(&ctx, &geometry, true, &logits, &mut weights_dev, &mut ids)
            .expect("route");
        let mut ids_host = read_slice(&ctx, &ids);
        let mut weights_host = read_slice(&ctx, &weights_dev);

        // The router's own reference reads the same stored logits, so the
        // routing stage is gated independently of the router GEMM's rounding.
        let (want_ids, want_weights) = router_reference(&logits_host, tokens, EXPERTS, TOP_K, true);
        assert_eq!(
            ids_host, want_ids,
            "case {}: the router's expert ids are not the host canonical top-k",
            case.name
        );
        eprintln!(
            "  router ids: all {} match the host canonical top-k",
            ids_host.len()
        );
        assert_within(
            "router weights",
            &weights_host,
            &want_weights,
            tokens,
            TOP_K,
            bounds::ROUTER_WEIGHT,
        );
        let logits_reference: Vec<bf16> = (0..tokens)
            .flat_map(|token| {
                let row = row_f64(&x[token * HIDDEN..(token + 1) * HIDDEN]);
                project(&host.router, 0, EXPERTS, HIDDEN, &row)
            })
            .collect();
        assert_within(
            "router logits",
            &logits_host,
            &logits_reference,
            tokens,
            EXPERTS,
            bounds::SHARED,
        );

        if case.permuted {
            // Reverse each token's picks by expert id, carrying the weights, so
            // slot order is descending expert order and the fold has to reorder.
            for token in 0..tokens {
                let base = token * TOP_K;
                let mut pairs: Vec<(i32, bf16)> = (0..TOP_K)
                    .map(|slot| (ids_host[base + slot], weights_host[base + slot]))
                    .collect();
                pairs.sort_by_key(|pick| std::cmp::Reverse(pick.0));
                for (slot, (id, weight)) in pairs.into_iter().enumerate() {
                    ids_host[base + slot] = id;
                    weights_host[base + slot] = weight;
                }
            }
            ctx.stream
                .memcpy_htod(&ids_host, &mut ids)
                .expect("ids down");
            ctx.stream
                .memcpy_htod(&weights_host, &mut weights_dev)
                .expect("weights down");
        }

        // 2. The routed experts.
        let mut scratch =
            Qwen4MoeExpertScratch::new(&ctx, &geometry, INTERMEDIATE, tokens).expect("scratch");
        let mut routed = HiddenStates::zeros(&ctx, HIDDEN, tokens).expect("routed");
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &stacks,
            &hidden,
            &weights_dev,
            &ids,
            &mut scratch,
            &mut routed,
        )
        .expect("routed experts");
        let routed_host = read_back(&ctx, &routed);
        let routed_rows = routed_rows_reference(
            &x,
            &host.gate_up,
            &host.down,
            &ids_host,
            tokens,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            false,
        );
        let routed_reference = fold_reference(
            &routed_rows,
            &weights_host,
            &ids_host,
            tokens,
            TOP_K,
            HIDDEN,
            Fold::Reference,
        );
        assert_within(
            "routed experts",
            &routed_host,
            &routed_reference,
            tokens,
            HIDDEN,
            bounds::ROUTED,
        );

        // 3. The shared branch.
        let mut branch_scratch =
            Qwen4SharedBranchScratch::new(&ctx, &geometry, tokens).expect("branch scratch");
        let mut shared = HiddenStates::zeros(&ctx, HIDDEN, tokens).expect("shared");
        qwen4_moe_shared_branch_into(
            &ctx,
            &geometry,
            &hidden,
            &shared_gate,
            &shared_up,
            &shared_down,
            &mut branch_scratch,
            &mut shared,
        )
        .expect("shared branch");
        let shared_host = read_back(&ctx, &shared);
        let shared_output = shared_reference(
            &x,
            &host.shared_gate,
            &host.shared_up,
            &host.shared_down,
            tokens,
            HIDDEN,
            INTERMEDIATE,
        );
        assert_within(
            "shared branch",
            &shared_host,
            &shared_output,
            tokens,
            HIDDEN,
            bounds::SHARED,
        );

        // 4. The combine, and the composed block.
        let mut gate_logit = HiddenStates::zeros(&ctx, 1, tokens).expect("gate logit");
        gemm_into_checked(&ctx, &shared_expert_gate, &hidden, &mut gate_logit)
            .expect("shared expert gate");
        let gate_logit_host = read_back(&ctx, &gate_logit);
        let mut combined = HiddenStates::zeros(&ctx, HIDDEN, tokens).expect("combined");
        qwen4_moe_combine_into(
            &ctx,
            &geometry,
            &gate_logit.data,
            &routed,
            &shared,
            &mut combined,
        )
        .expect("combine");
        let combined_host = read_back(&ctx, &combined);
        // Against the device's own two halves: this isolates the combine.
        let combine_of_device = combine_reference(
            &gate_logit_host,
            &routed_host,
            &shared_host,
            tokens,
            HIDDEN,
            false,
        );
        assert_within(
            "combine of the device's own halves",
            &combined_host,
            &combine_of_device,
            tokens,
            HIDDEN,
            bounds::COMBINED,
        );
        // Against the host's reference of the whole block: router, both branches
        // and the combine, accumulated end to end.
        let gate_logit_reference: Vec<bf16> = (0..tokens)
            .flat_map(|token| {
                let row = row_f64(&x[token * HIDDEN..(token + 1) * HIDDEN]);
                project(&host.shared_expert_gate, 0, 1, HIDDEN, &row)
            })
            .collect();
        let composed = combine_reference(
            &gate_logit_reference,
            &routed_reference,
            &shared_output,
            tokens,
            HIDDEN,
            false,
        );
        assert_within(
            "composed block output",
            &combined_host,
            &composed,
            tokens,
            HIDDEN,
            bounds::ROUTED,
        );

        // Timing is recorded, not optimized: the routed path is the new work.
        for _ in 0..3 {
            qwen4_moe_routed_experts_into(
                &ctx,
                &geometry,
                &stacks,
                &hidden,
                &weights_dev,
                &ids,
                &mut scratch,
                &mut routed,
            )
            .expect("routed experts warm");
        }
        ctx.sync().expect("sync");
        let start = Instant::now();
        for _ in 0..TIMED_REPEATS {
            qwen4_moe_routed_experts_into(
                &ctx,
                &geometry,
                &stacks,
                &hidden,
                &weights_dev,
                &ids,
                &mut scratch,
                &mut routed,
            )
            .expect("routed experts timed");
        }
        ctx.sync().expect("sync");
        let per_call = start.elapsed().as_secs_f64() * 1.0e3 / TIMED_REPEATS as f64;
        timing.push((tokens, per_call));
        eprintln!(
            "  routed expert path, {tokens} tokens ({} routes): {per_call:.3} ms per call \
             (wall clock around {TIMED_REPEATS} launches and one sync, warm)",
            tokens * TOP_K
        );

        // Negative controls on this case's own outputs. Magnitude faults first:
        // each has to land outside the bound, and the report states how far.
        let swapped = routed_rows_reference(
            &x,
            &host.gate_up,
            &host.down,
            &ids_host,
            tokens,
            TOP_K,
            HIDDEN,
            INTERMEDIATE,
            true,
        );
        assert_rejected(
            "control: fused projection halves swapped",
            &routed_host,
            &fold_reference(
                &swapped,
                &weights_host,
                &ids_host,
                tokens,
                TOP_K,
                HIDDEN,
                Fold::Reference,
            ),
            tokens,
            HIDDEN,
            bounds::ROUTED,
        );
        assert_rejected(
            "control: routing weight dropped",
            &routed_host,
            &fold_reference(
                &routed_rows,
                &weights_host,
                &ids_host,
                tokens,
                TOP_K,
                HIDDEN,
                Fold::NoWeight,
            ),
            tokens,
            HIDDEN,
            bounds::ROUTED,
        );
        assert_rejected(
            "control: shared branch summed instead of gated by sigmoid",
            &combined_host,
            &combine_reference(
                &gate_logit_host,
                &routed_host,
                &shared_host,
                tokens,
                HIDDEN,
                true,
            ),
            tokens,
            HIDDEN,
            bounds::COMBINED,
        );
        assert_rejected(
            "control: router without norm_topk_prob",
            &weights_host,
            &router_reference(&logits_host, tokens, EXPERTS, TOP_K, false).1,
            tokens,
            TOP_K,
            bounds::ROUTER_WEIGHT,
        );

        // Order-only faults: a few bf16 steps each, so the assertion is that
        // they move bits at all, with the magnitude reported.
        assert_moves_bits(
            "control: fold in slot order, not ascending expert",
            &routed_host,
            &fold_reference(
                &routed_rows,
                &weights_host,
                &ids_host,
                tokens,
                TOP_K,
                HIDDEN,
                Fold::SlotOrder,
            ),
            tokens,
            HIDDEN,
        );
        assert_moves_bits(
            "control: fold without the per-add bf16 rounding",
            &routed_host,
            &fold_reference(
                &routed_rows,
                &weights_host,
                &ids_host,
                tokens,
                TOP_K,
                HIDDEN,
                Fold::NoStepRounding,
            ),
            tokens,
            HIDDEN,
        );
    }

    eprintln!("timing (ms per routed-expert call, wall clock with sync):");
    for (tokens, per_call) in &timing {
        eprintln!("  {tokens} tokens: {per_call:.3}");
    }
}

/// The fold's order is the one thing above the projections that has to be
/// reproduced exactly, so it is gated **bitwise** on inputs built for it: three
/// tokens, ten picks each, magnitudes spread by factors of two and one negative
/// contribution per element so that adding in the wrong order changes the bits.
#[test]
fn the_fold_reproduces_the_reference_accumulation_order() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let geometry = Qwen4MoeGeometry::qwen38_flash_next();
    // Only three tokens of ten picks, so the route rows stay tiny even at the
    // real hidden width.
    let hidden = HIDDEN;
    let tokens = 3;
    let routes = tokens * TOP_K;
    let ids: Vec<i32> = (0..TOP_K as i32).map(|slot| 500 - slot).collect();
    let mut ids_host = Vec::with_capacity(routes);
    let mut weights_host = Vec::with_capacity(routes);
    let mut rows_host = Vec::with_capacity(routes * hidden);
    for token in 0..tokens {
        for (slot, id) in ids.iter().copied().enumerate() {
            ids_host.push(id);
            weights_host.push(bf16::from_f32(0.1 * (slot + 1) as f32));
            for col in 0..hidden {
                // A magnitude walk that makes the order observable: a small term
                // added to a large one can vanish (1 + 2^-8 rounds back to 1 in
                // bf16), so the other order keeps it.
                let magnitude = (1u32 << (slot % 5)) as f32 * 0.125;
                let sign = if (slot + col) % 3 == 0 { -1.0 } else { 1.0 };
                rows_host.push(bf16::from_f32(
                    sign * magnitude * (1.0 + col as f32 * 0.5) + token as f32,
                ));
            }
        }
    }
    let rows = upload(&ctx, &rows_host, hidden, routes);
    let weights_dev = ctx.stream.clone_htod(&weights_host).expect("weights up");
    let ids_dev = ctx.stream.clone_htod(&ids_host).expect("ids up");
    let mut out = HiddenStates::zeros(&ctx, hidden, tokens).expect("out");
    qwen4_moe_route_scale_fold_into(&ctx, &geometry, &rows, &weights_dev, &ids_dev, &mut out)
        .expect("fold");
    let got = read_back(&ctx, &out);

    let reference = fold_reference(
        &rows_host,
        &weights_host,
        &ids_host,
        tokens,
        TOP_K,
        hidden,
        Fold::Reference,
    );
    let differing = got
        .iter()
        .zip(&reference)
        .filter(|(got, want)| got.to_bits() != want.to_bits())
        .count();
    eprintln!(
        "  fold vs the reference over {} elements: {differing} differ bitwise",
        got.len()
    );
    assert_eq!(
        differing, 0,
        "the fold does not reproduce the reference's accumulation order bitwise"
    );

    // Both wrong folds have to move bits on this data, or the bitwise gate above
    // would pass without the order mattering at all.
    for (what, variant) in [
        ("slot order", Fold::SlotOrder),
        ("no per-add rounding", Fold::NoStepRounding),
    ] {
        let faulted = fold_reference(
            &rows_host,
            &weights_host,
            &ids_host,
            tokens,
            TOP_K,
            hidden,
            variant,
        );
        let moved = got
            .iter()
            .zip(&faulted)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        let worst = got
            .iter()
            .zip(&faulted)
            .map(|(got, want)| (got.to_f32() - want.to_f32()).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  control: fold in {what} moves {moved} elements (worst {worst:.3e} absolute)");
        assert!(
            moved > 0,
            "the {what} control moved nothing, so this data cannot show the order matters"
        );
    }

    // An id outside the stack fails that token closed on the device instead of
    // reading another expert's rows.
    let mut poisoned = ids_host.clone();
    poisoned[0] = EXPERTS as i32 + 7;
    let poisoned_dev = ctx.stream.clone_htod(&poisoned).expect("poisoned ids up");
    qwen4_moe_route_scale_fold_into(
        &ctx,
        &geometry,
        &rows,
        &weights_dev,
        &poisoned_dev,
        &mut out,
    )
    .expect("fold with a poisoned id");
    let poisoned_out = read_back(&ctx, &out);
    assert!(
        poisoned_out[..hidden].iter().all(|value| value.is_nan())
            && poisoned_out[hidden..].iter().all(|value| !value.is_nan()),
        "an out-of-range expert id must fail only its own token, with a NaN row"
    );
}

/// The stack's layout is the device-side half of the tensor contract: two fused
/// tensors per layer, the expert as the outermost axis, `2 * intermediate` rows
/// for gate|up and `hidden` for down. Every refusal here is device-free.
#[test]
fn only_the_checkpoint_layout_is_accepted() {
    let geometry = Qwen4MoeGeometry::qwen38_flash_next();
    let layout = Qwen4ExpertLayout::new(&geometry, INTERMEDIATE).expect("the frozen layout");
    assert_eq!(layout.experts(), EXPERTS);
    assert_eq!(layout.hidden(), HIDDEN);
    assert_eq!(layout.intermediate(), INTERMEDIATE);
    assert_eq!(layout.gate_up_rows(), 2 * INTERMEDIATE);
    // The two fused tensors, at the checkpoint's own element counts: ~3.36 GB and
    // ~1.68 GB, the layer's ~5.03 GB.
    assert_eq!(layout.gate_up_len(), EXPERTS * 2 * INTERMEDIATE * HIDDEN);
    assert_eq!(layout.down_len(), EXPERTS * HIDDEN * INTERMEDIATE);
    // The expert is the outermost axis, so the last expert's range ends exactly
    // at the stack's end and starts one expert's worth in.
    let last = layout.gate_up_of(EXPERTS - 1).expect("last expert");
    assert_eq!(last.row_offset, (EXPERTS - 1) * 2 * INTERMEDIATE);
    assert_eq!(last.rows, 2 * INTERMEDIATE);
    assert_eq!(last.cols, HIDDEN);
    assert_eq!(
        (last.row_offset + last.rows) * last.cols,
        layout.gate_up_len()
    );
    let last_down = layout.down_of(EXPERTS - 1).expect("last expert's down");
    assert_eq!(last_down.row_offset, (EXPERTS - 1) * HIDDEN);
    assert_eq!(last_down.rows, HIDDEN);
    assert_eq!(last_down.cols, INTERMEDIATE);
    assert_eq!(
        (last_down.row_offset + last_down.rows) * last_down.cols,
        layout.down_len()
    );

    // A zero-width expert, a geometry the routing kernel was not compiled for,
    // and an expert index past the stack are all refused up front.
    assert!(
        Qwen4ExpertLayout::new(&geometry, 0).is_err(),
        "a zero-width expert intermediate must be refused"
    );
    for (field, mutated) in [
        (
            "experts",
            Qwen4MoeGeometry {
                experts: 256,
                ..geometry
            },
        ),
        (
            "top_k",
            Qwen4MoeGeometry {
                top_k: 8,
                ..geometry
            },
        ),
        (
            "hidden",
            Qwen4MoeGeometry {
                hidden: 0,
                ..geometry
            },
        ),
    ] {
        assert!(
            Qwen4ExpertLayout::new(&mutated, INTERMEDIATE).is_err(),
            "a geometry with another {field} must be refused"
        );
    }
    assert!(
        layout.gate_up_of(EXPERTS).is_err() && layout.down_of(EXPERTS).is_err(),
        "a view of a non-existent expert must be refused"
    );
}

/// The routed path's own buffers and geometry are refused before any pointer is
/// formed: a stack of another width, a weight array one route short, a scratch
/// built for another token count, and an output of the wrong width.
#[test]
fn the_routed_path_rejects_a_mismatched_call() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // Small widths: the geometry pins the experts and top-k, not the widths.
    let hidden = 8;
    let intermediate = 2;
    let tokens = 2;
    let geometry = Qwen4MoeGeometry {
        hidden,
        ..Qwen4MoeGeometry::qwen38_flash_next()
    };
    geometry.validate().expect("a small-width geometry");
    let routes = tokens * TOP_K;
    let stacks = Qwen4ExpertStacks::from_host(
        &ctx,
        &geometry,
        intermediate,
        &noise(0x51EE_D2001, EXPERTS * 2 * intermediate * hidden, 0.5),
        &noise(0x51EE_D2002, EXPERTS * hidden * intermediate, 0.5),
    )
    .expect("small stacks");
    assert_eq!(
        stacks.layout().gate_up_len(),
        EXPERTS * 2 * intermediate * hidden
    );
    let x = noise(0x51EE_D2003, tokens * hidden, 1.0);
    let hidden_states = upload(&ctx, &x, hidden, tokens);
    let mut scratch =
        Qwen4MoeExpertScratch::new(&ctx, &geometry, intermediate, tokens).expect("small scratch");
    assert_eq!(scratch.tokens(), tokens);
    assert_eq!(scratch.routes(), routes);
    let mut routed = HiddenStates::zeros(&ctx, hidden, tokens).expect("routed");

    let ids_host: Vec<i32> = (0..routes as i32).collect();
    let weights_host = vec![bf16::from_f32(0.5); routes];
    let ids = ctx.stream.clone_htod(&ids_host).expect("ids");
    let weights = ctx.stream.clone_htod(&weights_host).expect("weights");

    // A stack of another expert width, used with this call's scratch.
    let wide = Qwen4ExpertStacks::from_host(
        &ctx,
        &geometry,
        intermediate + 1,
        &noise(0x51EE_D2004, EXPERTS * 2 * (intermediate + 1) * hidden, 0.5),
        &noise(0x51EE_D2005, EXPERTS * hidden * (intermediate + 1), 0.5),
    )
    .expect("wide stacks");
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &wide,
            &hidden_states,
            &weights,
            &ids,
            &mut scratch,
            &mut routed,
        )
        .is_err(),
        "a stack built for another expert width must be refused against this scratch"
    );

    // A stack one expert short, and a geometry the stack does not match.
    let layout = Qwen4ExpertLayout::new(&geometry, intermediate).expect("layout");
    assert!(
        Qwen4ExpertStacks::from_host(
            &ctx,
            &geometry,
            intermediate,
            &noise(
                0x51EE_D2006,
                layout.gate_up_len() - 2 * intermediate * hidden,
                0.5
            ),
            &noise(0x51EE_D2007, layout.down_len(), 0.5),
        )
        .is_err(),
        "a fused gate|up stack missing one expert must be refused"
    );
    assert!(
        Qwen4ExpertStacks::from_host(
            &ctx,
            &geometry,
            intermediate,
            &noise(0x51EE_D2008, layout.gate_up_len(), 0.5),
            &noise(0x51EE_D2009, layout.down_len() + 1, 0.5),
        )
        .is_err(),
        "a down stack with one element too many must be refused"
    );
    let other = Qwen4MoeGeometry {
        hidden: hidden + 1,
        ..geometry
    };
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &other,
            &stacks,
            &hidden_states,
            &weights,
            &ids,
            &mut scratch,
            &mut routed,
        )
        .is_err(),
        "a call whose geometry disagrees with the stack must be refused"
    );

    // One route short of ids and of weights.
    let short_ids = ctx
        .stream
        .clone_htod(&ids_host[..routes - 1])
        .expect("short ids");
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &stacks,
            &hidden_states,
            &weights,
            &short_ids,
            &mut scratch,
            &mut routed,
        )
        .is_err(),
        "a route batch without all of its ids must be refused"
    );
    let short_weights = ctx
        .stream
        .clone_htod(&weights_host[..routes - 1])
        .expect("short weights");
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &stacks,
            &hidden_states,
            &short_weights,
            &ids,
            &mut scratch,
            &mut routed,
        )
        .is_err(),
        "a route batch without all of its weights must be refused"
    );

    // Scratch built for another token count, and an output of another width.
    let mut wide_scratch = Qwen4MoeExpertScratch::new(&ctx, &geometry, intermediate, tokens + 1)
        .expect("wide scratch");
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &stacks,
            &hidden_states,
            &weights,
            &ids,
            &mut wide_scratch,
            &mut routed,
        )
        .is_err(),
        "a scratch sized for another token count must be refused"
    );
    let mut wrong_out = HiddenStates::zeros(&ctx, hidden + 1, tokens).expect("wrong out");
    assert!(
        qwen4_moe_routed_experts_into(
            &ctx,
            &geometry,
            &stacks,
            &hidden_states,
            &weights,
            &ids,
            &mut scratch,
            &mut wrong_out,
        )
        .is_err(),
        "an output of another width must be refused"
    );
    assert!(
        Qwen4MoeExpertScratch::new(&ctx, &geometry, intermediate, 0).is_err(),
        "a scratch for zero tokens must be refused"
    );

    // The correct call runs; an id past the stack fails only its own token
    // closed rather than reading another expert's rows.
    let out_of_range_host: Vec<i32> = (0..routes as i32)
        .map(|route| {
            if route == 2 {
                EXPERTS as i32 + 1
            } else {
                route
            }
        })
        .collect();
    let out_of_range = ctx
        .stream
        .clone_htod(&out_of_range_host)
        .expect("out of range ids");
    qwen4_moe_routed_experts_into(
        &ctx,
        &geometry,
        &stacks,
        &hidden_states,
        &weights,
        &out_of_range,
        &mut scratch,
        &mut routed,
    )
    .expect("routed path with one bad id");
    let out = read_back(&ctx, &routed);
    assert!(
        out[..hidden].iter().all(|value| value.is_nan())
            && out[hidden..].iter().all(|value| value.is_finite()),
        "only the token with the out-of-range id may be poisoned: {out:?}"
    );
}
