//! Numeric gate for the capsule-vendored vLLM router top-k
//! (`cubin/k3/single_group_topk_e512t22_sm103.cubin`) against the native
//! `k3_router_topk` kernel on identical logits.
//!
//! Both kernels take raw router logits, apply f32 sigmoid, select top-k over
//! `sigmoid + bias`, and renormalize the *un-biased* sigmoid scores of the
//! picks to the routed scale. They differ in output order (native emits
//! selection order, the capsule emits its own sort order) and in the exact
//! renorm-eps spelling, so the contract under test is order-free: per row,
//! the selected expert *sets* must be equal and each expert's weight must
//! agree to f32 rounding. Downstream (MegaMoE routing) treats the pairs as
//! unordered, so this is exactly the production contract.
//!
//! Manual gate: CI compiles this but never runs it. Run on a Blackwell box
//! (set `PEGAINFER_REQUIRE_GPU=1` to turn a missing device into a failure).

#![cfg(feature = "k3")]

mod common;

use std::collections::HashMap;

use half::bf16;
use pegainfer_kernels::ops::k3_capsule_router_topk_launch;
use pegainfer_kernels::ops::k3_router_topk_batched_launch;

/// The pruned dev checkpoint's expert count — the only K3 shape inside the
/// capsule tier (<=512 experts; the 896-expert full model stays native).
const EXPERTS: usize = 224;
const TOPK: usize = 16;
const ROUTED_SCALE: f32 = 2.5;

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

#[test]
fn capsule_topk_matches_native_selection() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut rng = Lcg(0x4b33_2026_0828);
    // Match the native kernel's exact scale spelling: it reads rs as bf16 and
    // widens; hand the capsule the identical widened value.
    let rs = bf16::from_f32(ROUTED_SCALE);
    let rs_dev = ctx.stream.clone_htod(&[rs]).expect("rs H2D");
    let bias: Vec<f32> = (0..EXPERTS).map(|_| rng.unit_f32() * 0.02).collect();
    let bias_dev = ctx.stream.clone_htod(&bias).expect("bias H2D");

    // Production decode buckets plus an odd row count for the capsule's
    // 8-rows-per-block grid tail.
    for &b in &[1usize, 3, 8, 32] {
        let logits: Vec<f32> = (0..b * EXPERTS).map(|_| rng.unit_f32() * 4.0).collect();
        let logits_dev = ctx.stream.clone_htod(&logits).expect("logits H2D");

        let mut native_idx = ctx.stream.alloc_zeros::<i32>(b * TOPK).expect("idx alloc");
        let mut native_wts = ctx.stream.alloc_zeros::<f32>(b * TOPK).expect("wts alloc");
        k3_router_topk_batched_launch(
            &ctx,
            b,
            EXPERTS,
            TOPK,
            &logits_dev,
            &bias_dev,
            &rs_dev,
            &mut native_idx,
            &mut native_wts,
        )
        .expect("native topk launch");

        let mut cap_idx = ctx.stream.alloc_zeros::<i32>(b * TOPK).expect("idx alloc");
        let mut cap_wts = ctx.stream.alloc_zeros::<f32>(b * TOPK).expect("wts alloc");
        k3_capsule_router_topk_launch(
            &ctx,
            b,
            EXPERTS,
            TOPK,
            &logits_dev,
            &bias_dev,
            rs.to_f32(),
            &mut cap_idx,
            &mut cap_wts,
        )
        .expect("capsule topk launch");

        let n_idx = ctx.stream.clone_dtoh(&native_idx).expect("D2H");
        let n_wts = ctx.stream.clone_dtoh(&native_wts).expect("D2H");
        let c_idx = ctx.stream.clone_dtoh(&cap_idx).expect("D2H");
        let c_wts = ctx.stream.clone_dtoh(&cap_wts).expect("D2H");

        for t in 0..b {
            let native: HashMap<i32, f32> = (0..TOPK)
                .map(|r| (n_idx[t * TOPK + r], n_wts[t * TOPK + r]))
                .collect();
            let capsule: HashMap<i32, f32> = (0..TOPK)
                .map(|r| (c_idx[t * TOPK + r], c_wts[t * TOPK + r]))
                .collect();
            assert_eq!(native.len(), TOPK, "b={b} row {t}: native emitted a dup");
            assert_eq!(capsule.len(), TOPK, "b={b} row {t}: capsule emitted a dup");
            let mut native_experts: Vec<i32> = native.keys().copied().collect();
            let mut capsule_experts: Vec<i32> = capsule.keys().copied().collect();
            native_experts.sort_unstable();
            capsule_experts.sort_unstable();
            assert_eq!(
                native_experts, capsule_experts,
                "b={b} row {t}: expert sets diverge"
            );
            for (&e, &nw) in &native {
                let cw = capsule[&e];
                // Weights are O(rs / topk); the only legal differences are
                // sigmoid/renorm rounding and the 1e-20 eps spelling.
                assert!(
                    (nw - cw).abs() <= 1e-5 * nw.abs().max(1.0),
                    "b={b} row {t} expert {e}: native weight {nw} vs capsule {cw}"
                );
            }
        }
    }
    println!("capsule topk == native topk for b in [1, 3, 8, 32] at E={EXPERTS}, topk={TOPK}");
}
