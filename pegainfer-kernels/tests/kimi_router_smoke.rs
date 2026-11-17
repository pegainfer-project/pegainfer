//! Device gate for the Kimi-K2 router: topk_idx must match the sequential
//! argmax reference (choice desc, expert asc) slot for slot, weights must
//! match to f32 rounding, and rows past active_tokens must stay untouched.

#![cfg(feature = "kimi-k2")]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod common;

use half::bf16;
use pegainfer_kernels::ops::KimiRouterBatch;
use pegainfer_kernels::ops::KimiRouterConfig;
use pegainfer_kernels::ops::KimiRouterOutput;
use pegainfer_kernels::ops::kimi_router_noaux_tc_launch;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::GpuTensor;
use pegainfer_kernels::tensor::GpuWeight;

const HIDDEN: usize = 7168;
const EXPERTS: usize = 384;
const TOPK: usize = 8;
// KimiRouterConfig::kimi_k2() route_scale; the field is private.
const ROUTE_SCALE: f32 = 1.0;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Selection reference from glm52_router_smoke.rs, computed on the GPU's own
/// f32 logits.
fn host_select(
    gpu_logits: &[f32],
    bias: &[f32],
    active_tokens: usize,
    route_scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut topk_idx = vec![0i32; active_tokens * TOPK];
    let mut topk_weight = vec![0.0f32; active_tokens * TOPK];
    for t in 0..active_tokens {
        let scores: Vec<f32> = (0..EXPERTS)
            .map(|e| 1.0 / (1.0 + (-gpu_logits[t * EXPERTS + e]).exp()))
            .collect();
        let choice: Vec<f32> = (0..EXPERTS)
            .map(|e| match scores[e] + bias[e] {
                c if c.is_nan() => f32::NEG_INFINITY,
                c => c,
            })
            .collect();
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        order.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap().then(a.cmp(&b)));
        let picks = &order[..TOPK];
        let sum: f32 = picks.iter().map(|&e| scores[e]).sum();
        let scale = if sum > 0.0 { route_scale / sum } else { 0.0 };
        for (r, &e) in picks.iter().enumerate() {
            topk_idx[t * TOPK + r] = e as i32;
            topk_weight[t * TOPK + r] = scores[e] * scale;
        }
    }
    (topk_idx, topk_weight)
}

#[allow(clippy::float_cmp)]
#[test]
fn router_select_matches_host_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut rng = Lcg(0x4b49_4d49_2026_1123);
    let gate: Vec<bf16> = (0..EXPERTS * HIDDEN)
        .map(|_| bf16::from_f32(rng.unit_f32() * 0.05))
        .collect();
    let bias: Vec<f32> = (0..EXPERTS).map(|_| rng.unit_f32() * 0.01).collect();
    let zero_hidden = vec![bf16::ZERO; 5 * HIDDEN];
    let zero_bias = vec![0.0f32; EXPERTS];
    let nan_hidden = vec![bf16::NAN; 5 * HIDDEN];
    let mut nan_gate = gate.clone();
    nan_gate[5 * HIDDEN..6 * HIDDEN].fill(bf16::NAN);

    // sigmoid(30) + 0 == sigmoid(0) + 0.5 == 1.0 exactly, so each token gets a
    // tie group of 9..=19 experts with mixed scores above every other choice.
    // Token t reads gate column t through a one-hot hidden row.
    let tie_bias: Vec<f32> = (0..EXPERTS)
        .map(|_| (rng.next_u32() & 1) as f32 * 0.5)
        .collect();
    let mut tie_hidden = vec![bf16::ZERO; 11 * HIDDEN];
    let mut tie_gate = vec![bf16::ZERO; EXPERTS * HIDDEN];
    for t in 0..11 {
        tie_hidden[t * HIDDEN + t] = bf16::ONE;
        for e in 0..EXPERTS {
            let below = if tie_bias[e] == 0.0 {
                rng.unit_f32() * 6.0
            } else {
                -0.5 - (rng.unit_f32() + 1.0) * 2.5
            };
            tie_gate[e * HIDDEN + t] = bf16::from_f32(below);
        }
        let start = rng.next_u32() as usize;
        for k in 0..9 + t {
            // 37 is coprime with 384, so the group has no repeats.
            let e = (start + k * 37) % EXPERTS;
            let logit = if tie_bias[e] == 0.0 { 30.0 } else { 0.0 };
            tie_gate[e * HIDDEN + t] = bf16::from_f32(logit);
        }
    }

    // (hidden, gate, bias, active, padded): decode batch sizes 1/8/32,
    // active < padded, a prefill batch past the 64-row decode cap, every
    // choice tied (zero logits and bias), tie groups, all-NaN rows and one NaN
    // expert.
    let mut cases = Vec::new();
    for &(active, padded) in &[
        (1, 1),
        (8, 8),
        (32, 32),
        (1, 2),
        (3, 5),
        (7, 8),
        (29, 32),
        (70, 72),
    ] {
        let hidden: Vec<bf16> = (0..padded * HIDDEN)
            .map(|_| bf16::from_f32(rng.unit_f32()))
            .collect();
        cases.push((hidden, &gate[..], &bias[..], active, padded));
    }
    cases.push((zero_hidden, &gate[..], &zero_bias[..], 3, 5));
    cases.push((tie_hidden, &tie_gate[..], &tie_bias[..], 8, 11));
    cases.push((nan_hidden, &gate[..], &bias[..], 3, 5));
    let nan_gate_hidden: Vec<bf16> = (0..32 * HIDDEN)
        .map(|_| bf16::from_f32(rng.unit_f32()))
        .collect();
    cases.push((nan_gate_hidden, &nan_gate[..], &bias[..], 29, 32));

    for (hidden, gate, bias, active, padded) in cases {
        let hidden = GpuTensor::<HIDDEN>::from_device_matrix_rows(
            DeviceMatrix::from_host(&ctx, &hidden, padded, HIDDEN).expect("hidden H2D"),
        )
        .expect("hidden tensor");
        let gate = GpuWeight::<EXPERTS, HIDDEN>::from_device_matrix(
            DeviceMatrix::from_host(&ctx, gate, EXPERTS, HIDDEN).expect("gate H2D"),
        )
        .expect("gate weight");
        let bias_dev = ctx.stream.clone_htod(bias).expect("bias H2D");
        let mut logits_dev = ctx
            .stream
            .alloc_zeros::<f32>(padded * EXPERTS)
            .expect("logits alloc");
        let mut weight_dev = ctx
            .stream
            .clone_htod(&vec![-1.0f32; padded * TOPK])
            .expect("weight H2D");
        let mut idx_dev = ctx
            .stream
            .clone_htod(&vec![-1i32; padded * TOPK])
            .expect("idx H2D");
        kimi_router_noaux_tc_launch(
            &ctx,
            KimiRouterConfig::kimi_k2(),
            KimiRouterBatch {
                batch_size: padded,
                active_tokens: active,
                padded_tokens: padded,
            },
            &hidden,
            &gate,
            &bias_dev,
            &mut logits_dev,
            &mut KimiRouterOutput {
                topk_weight: &mut weight_dev,
                topk_idx: &mut idx_dev,
            },
        )
        .expect("router launch");

        let logits = ctx.stream.clone_dtoh(&logits_dev).expect("logits D2H");
        let idx = ctx.stream.clone_dtoh(&idx_dev).expect("idx D2H");
        let weight = ctx.stream.clone_dtoh(&weight_dev).expect("weight D2H");

        let routed = active * TOPK;
        assert!(
            idx[routed..].iter().all(|&i| i == -1) && weight[routed..].iter().all(|&w| w == -1.0),
            "active={active} padded={padded}: padded rows were written"
        );
        let (ref_idx, ref_weight) = host_select(&logits, bias, active, ROUTE_SCALE);
        assert_eq!(
            idx[..routed],
            ref_idx[..],
            "active={active} padded={padded}: topk_idx mismatch"
        );
        for (r, (&gpu, &host)) in weight.iter().zip(&ref_weight).enumerate() {
            assert!(
                (gpu - host).abs() < 1e-4 || (gpu.is_nan() && host.is_nan()),
                "active={active} padded={padded}: topk_weight[{r}] gpu {gpu} vs host {host}"
            );
        }
    }
}
