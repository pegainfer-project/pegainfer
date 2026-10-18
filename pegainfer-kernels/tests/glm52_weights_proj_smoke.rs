//! Device gate for the GLM5.2 indexer weights_proj min-latency GEMV (which
//! replaced the cublas splitK plan): bf16 [tokens,6144] x [32,6144]^T ->
//! bf16 [tokens,32], checked against an f64 host reference at every
//! dispatched tokens count (1..=8).

#![cfg(feature = "glm52")]

use half::bf16;
use pegainfer_kernels::ops::GLM52_MIN_GEMV_MAX_TOKENS;
use pegainfer_kernels::ops::glm52_indexer_weights_proj_launch;
use pegainfer_kernels::tensor::DeviceContext;

const HIDDEN: usize = 6144;
const HEADS: usize = 32;

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
fn weights_proj_gemv_matches_host_reference() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut w_rng = Lcg(0x1d78_2026_0713);
    let mut h_rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let weights: Vec<bf16> = (0..HEADS * HIDDEN)
        .map(|_| bf16::from_f32(w_rng.unit_f32() * 0.05))
        .collect();
    let weights_dev = ctx.stream.clone_htod(&weights).expect("weights H2D");

    for tokens in 1..=GLM52_MIN_GEMV_MAX_TOKENS {
        // 0.25 scale keeps |dot| ~ 1: bf16 half-ulp 3.9e-3 vs the 8e-3 assert.
        let hidden: Vec<bf16> = (0..tokens * HIDDEN)
            .map(|_| bf16::from_f32(h_rng.unit_f32() * 0.25))
            .collect();
        let hidden_dev = ctx.stream.clone_htod(&hidden).expect("hidden H2D");
        let mut out_dev = ctx
            .stream
            .alloc_zeros::<bf16>(tokens * HEADS)
            .expect("out alloc");
        glm52_indexer_weights_proj_launch(
            &ctx,
            &hidden_dev,
            &weights_dev,
            tokens,
            HEADS,
            HIDDEN,
            &mut out_dev,
        )
        .expect("weights_proj launch");
        let out = ctx.stream.clone_dtoh(&out_dev).expect("out D2H");

        let mut max_err = 0.0f64;
        for t in 0..tokens {
            for h in 0..HEADS {
                let mut acc = 0.0f64;
                for k in 0..HIDDEN {
                    acc += f64::from(hidden[t * HIDDEN + k].to_f32())
                        * f64::from(weights[h * HIDDEN + k].to_f32());
                }
                let err = (f64::from(out[t * HEADS + h].to_f32()) - acc).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(max_err < 8e-3, "tokens={tokens}: max err {max_err}");
    }
}
