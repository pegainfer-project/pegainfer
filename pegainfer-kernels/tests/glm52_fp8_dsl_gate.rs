//! Numerical gate for the CuTe DSL tcgen05 AOT fp8 GEMM dispatch.
//!
//! Runs every wide-route projection shape at a split-K-free, a split-K, and
//! a multi-M-tile bucket through both routes on identical inputs — the DSL table via
//! `glm52_fp8_groupwise_gemm_sm100_offset_launch` (which dispatches when the
//! AOT modules are loaded) and CUTLASS via the bank variant (which never
//! dispatches) — and demands agreement. Both kernels sit within rel_l2
//! 9.1e-5 of the same f32 reference in kernel_lab, so 2e-4 mutual tolerance
//! has ~2x headroom while any wrong pointer, stride, or partial-reduce bug
//! lands orders of magnitude outside it.

#![cfg(feature = "glm52")]

use half::bf16;
use pegainfer_kernels::ops::glm52_flashinfer_sparse_mla_supported;
use pegainfer_kernels::ops::glm52_fp8_dsl_gemm_ready;
use pegainfer_kernels::ops::glm52_fp8_dsl_preload;
use pegainfer_kernels::ops::glm52_fp8_groupwise_gemm_sm100_bank_launch;
use pegainfer_kernels::ops::glm52_fp8_groupwise_gemm_sm100_offset_launch;
use pegainfer_kernels::tensor::DeviceContext;

/// e4m3 bytes with NaN encodings (0x7F/0xFF) excluded, deterministic LCG.
fn e4m3_bytes(len: usize, seed: &mut u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let magnitude = ((*seed >> 33) % 127) as u8;
            let sign = ((*seed >> 17) & 1) as u8;
            magnitude | (sign << 7)
        })
        .collect()
}

fn scales(len: usize, seed: &mut u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.5 + ((*seed >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

#[test]
fn dsl_route_matches_cutlass_on_every_wide_route_shape() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    if !glm52_flashinfer_sparse_mla_supported(16).expect("query SM support") {
        eprintln!("skip: needs SM100/SM103");
        return;
    }
    glm52_fp8_dsl_preload();
    if !glm52_fp8_dsl_gemm_ready() {
        eprintln!("skip: CuTe DSL AOT not built (PEGAINFER_CUTEDSL_PYTHON unset)");
        return;
    }

    let mut seed = 0x5EEDu64;
    for (n, k) in [
        (16384usize, 2048usize),
        (6144, 16384),
        (4096, 6144),
        (6144, 2048),
    ] {
        for m in [16usize, 64, 96] {
            let act = e4m3_bytes(m * k, &mut seed);
            let act_s = scales(m * (k / 128), &mut seed);
            let weight = e4m3_bytes(n * k, &mut seed);
            let w_s = scales((n / 128) * (k / 128), &mut seed);

            let act_dev = ctx.stream.clone_htod(&act).expect("upload activation");
            let act_s_dev = ctx.stream.clone_htod(&act_s).expect("upload act scales");
            let weight_dev = ctx.stream.clone_htod(&weight).expect("upload weight");
            let w_s_f32_dev = ctx.stream.clone_htod(&w_s).expect("upload scales f32");
            let w_s_bytes: Vec<u8> = w_s.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let w_s_u8_dev = ctx.stream.clone_htod(&w_s_bytes).expect("upload scales u8");
            let mut out_dsl = ctx
                .stream
                .alloc_zeros::<bf16>(m * n)
                .expect("alloc dsl out");
            let mut out_cut = ctx
                .stream
                .alloc_zeros::<bf16>(m * n)
                .expect("alloc cutlass out");
            let mut workspace = ctx
                .stream
                .alloc_zeros::<u8>(32 << 20)
                .expect("alloc workspace");

            glm52_fp8_groupwise_gemm_sm100_offset_launch(
                &ctx,
                m,
                n,
                k,
                &act_dev,
                &act_s_dev,
                &weight_dev,
                0,
                &w_s_u8_dev,
                0,
                &mut out_dsl,
                &mut workspace,
            )
            .expect("DSL-dispatched launch");
            glm52_fp8_groupwise_gemm_sm100_bank_launch(
                &ctx,
                m,
                n,
                k,
                &act_dev,
                &act_s_dev,
                &weight_dev,
                0,
                &w_s_f32_dev,
                0,
                &mut out_cut,
                &mut workspace,
            )
            .expect("CUTLASS launch");

            let dsl_host = ctx.stream.clone_dtoh(&out_dsl).expect("download dsl");
            let cut_host = ctx.stream.clone_dtoh(&out_cut).expect("download cutlass");
            let mut num = 0f64;
            let mut den = 0f64;
            for (a, b) in dsl_host.iter().zip(cut_host.iter()) {
                let (a, b) = (a.to_f64(), b.to_f64());
                num += (a - b) * (a - b);
                den += b * b;
            }
            let rel_l2 = (num / den.max(f64::MIN_POSITIVE)).sqrt();
            assert!(
                rel_l2 <= 2e-4,
                "DSL vs CUTLASS diverge at m={m} n={n} k={k}: rel_l2={rel_l2:.3e}"
            );
        }
    }
}
