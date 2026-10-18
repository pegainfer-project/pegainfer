//! Device gate for the GLM5.2 router: min-latency logits GEMV (which
//! replaced the cublas splitK plan) + noaux-tc top-k selection, checked
//! against an f64 host reference at every padded_tokens the runtime
//! dispatch instantiates (1..=8; production buckets are 1/2/4/8).
//!
//! The selection contract is exact: topk_idx must match the reference
//! sequential-argmax order position-for-position away from score near-ties,
//! and normalized weights must agree to f32 rounding. Logits themselves are
//! checked against the f64 dot product at bf16-input accumulation tolerance.

#![cfg(feature = "glm52")]

use half::bf16;
use pegainfer_kernels::ops::Glm52RouterBatch;
use pegainfer_kernels::ops::Glm52RouterConfig;
use pegainfer_kernels::ops::Glm52RouterOutput;
use pegainfer_kernels::ops::glm52_router_noaux_tc_launch;
use pegainfer_kernels::tensor::DeviceContext;

const HIDDEN: usize = 6144;
const EXPERTS: usize = 256;
const TOPK: usize = 8;

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

/// f64 dot-product reference for the GEMV half of the router.
fn host_logits(hidden: &[bf16], gate: &[bf16], active_tokens: usize) -> Vec<f64> {
    let mut logits = vec![0.0f64; active_tokens * EXPERTS];
    for t in 0..active_tokens {
        for e in 0..EXPERTS {
            let mut acc = 0.0f64;
            for h in 0..HIDDEN {
                acc += f64::from(hidden[t * HIDDEN + h].to_f32())
                    * f64::from(gate[e * HIDDEN + h].to_f32());
            }
            logits[t * EXPERTS + e] = acc;
        }
    }
    logits
}

/// Selection reference computed from the GPU's own f32 logits, so the
/// exact-order assertion tests the select kernel without near-tie flake
/// from f64-vs-f32 sigmoid differences.
fn host_select(
    gpu_logits: &[f32],
    bias: &[f32],
    active_tokens: usize,
    route_scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut topk_idx = vec![0i32; active_tokens * TOPK];
    let mut topk_weight = vec![0.0f32; active_tokens * TOPK];
    for t in 0..active_tokens {
        // Sequential argmax over sigmoid(logit) + bias in the kernel's
        // (value desc, index asc) order; weights normalize the un-biased
        // sigmoid scores of the picks.
        let scores: Vec<f32> = (0..EXPERTS)
            .map(|e| 1.0 / (1.0 + (-gpu_logits[t * EXPERTS + e]).exp()))
            .collect();
        let choice: Vec<f32> = (0..EXPERTS).map(|e| scores[e] + bias[e]).collect();
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

#[test]
fn router_gemv_and_select_match_host_reference() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut rng = Lcg(0x5157_2026_0713);
    let gate: Vec<bf16> = (0..EXPERTS * HIDDEN)
        .map(|_| bf16::from_f32(rng.unit_f32() * 0.05))
        .collect();
    let bias: Vec<f32> = (0..EXPERTS).map(|_| rng.unit_f32() * 0.01).collect();
    let bias_bytes = as_bytes(&bias).to_vec();
    let gate_dev = ctx.stream.clone_htod(&gate).expect("gate H2D");
    let bias_dev = ctx.stream.clone_htod(&bias_bytes).expect("bias H2D");

    // Every instantiation the runtime switch dispatches, with active < padded
    // coverage on the padded rows staying select-silent.
    for &(active, padded) in &[
        (1usize, 1usize),
        (2, 2),
        (2, 3),
        (4, 4),
        (3, 5),
        (6, 6),
        (5, 7),
        (8, 8),
    ] {
        let hidden: Vec<bf16> = (0..padded * HIDDEN)
            .map(|_| bf16::from_f32(rng.unit_f32()))
            .collect();
        let hidden_dev = ctx.stream.clone_htod(&hidden).expect("hidden H2D");
        let mut logits_dev = ctx
            .stream
            .alloc_zeros::<f32>(padded * EXPERTS)
            .expect("logits alloc");
        let mut weight_dev = ctx
            .stream
            .alloc_zeros::<f32>(active * TOPK)
            .expect("weight alloc");
        let mut idx_dev = ctx
            .stream
            .alloc_zeros::<i32>(active * TOPK)
            .expect("idx alloc");
        let mut out = Glm52RouterOutput {
            topk_weight: &mut weight_dev,
            topk_idx: &mut idx_dev,
        };
        glm52_router_noaux_tc_launch(
            &ctx,
            Glm52RouterConfig::glm52(),
            Glm52RouterBatch {
                active_tokens: active,
                padded_tokens: padded,
            },
            &hidden_dev,
            &gate_dev,
            &bias_dev,
            &mut logits_dev,
            &mut out,
        )
        .expect("router launch");

        let logits = ctx.stream.clone_dtoh(&logits_dev).expect("logits D2H");
        let idx = ctx.stream.clone_dtoh(&idx_dev).expect("idx D2H");
        let weight = ctx.stream.clone_dtoh(&weight_dev).expect("weight D2H");

        let ref_logits = host_logits(&hidden, &gate, active);
        let mut max_logit_err = 0.0f64;
        for t in 0..active {
            for e in 0..EXPERTS {
                let err = (f64::from(logits[t * EXPERTS + e]) - ref_logits[t * EXPERTS + e]).abs();
                max_logit_err = max_logit_err.max(err);
            }
        }
        // bf16 inputs, f32 accumulation over 6144 terms with |x| <= 1, |w| <= 0.05.
        assert!(
            max_logit_err < 5e-3,
            "padded={padded}: max logit err {max_logit_err}"
        );

        let (ref_idx, ref_weight) = host_select(
            &logits,
            &bias,
            active,
            Glm52RouterConfig::glm52().route_scale,
        );
        assert_eq!(idx, ref_idx, "padded={padded}: topk_idx mismatch");
        for r in 0..active * TOPK {
            let err = (weight[r] - ref_weight[r]).abs();
            assert!(
                err < 1e-4,
                "padded={padded}: topk_weight[{r}] gpu {} vs host {}",
                weight[r],
                ref_weight[r]
            );
        }
    }
}

/// Plain byte view of a POD slice (the launch API takes weight/bias as raw
/// byte buffers).
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
