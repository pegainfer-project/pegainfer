//! Fused KCP package forward equivalence gate.
//!
//! The fused `k3_flash_kda_fwd_md_launch` must reproduce the two standalone
//! doctored calls it replaces — `M` from (v = 0, state = I) and `D` from
//! (real v, state = 0) — bit-for-bit: the derived kernel keeps the upstream
//! GEMM shapes and accumulation order per state, so any drift is a bug, not
//! rounding.
//!
//! Needs one GPU with the accelerated FlashKDA build (SM90+):
//! `cargo test --release -p pegainfer-kernels --features k3 \
//!    --test k3_flash_kda_md_equiv -- --ignored --nocapture`

#![cfg(feature = "k3")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3FlashKdaSpan;
use pegainfer_kernels::ops::k3_flash_kda_fwd_launch;
use pegainfer_kernels::ops::k3_flash_kda_fwd_md_launch;
use pegainfer_kernels::ops::k3_flash_kda_workspace_bytes;
use pegainfer_kernels::tensor::DeviceContext;

const HEADS: usize = 96;
const D: usize = 128;
const SCALE: f32 = 0.088_388_35; // 128^-0.5
const LOWER_BOUND: f32 = -5.0;

fn xorshift_fill(seed: &mut u64, len: usize, amp: f32) -> Vec<bf16> {
    (0..len)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let unit = (*seed >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            bf16::from_f32((unit - 0.5) * 2.0 * amp)
        })
        .collect()
}

fn identity_state(heads: usize) -> Vec<f32> {
    let mut state = vec![0f32; heads * D * D];
    for h in 0..heads {
        for i in 0..D {
            state[(h * D + i) * D + i] = 1.0;
        }
    }
    state
}

fn dtoh(ctx: &DeviceContext, slice: &CudaSlice<f32>) -> Vec<f32> {
    let host = ctx.stream.clone_dtoh(slice).expect("dtoh");
    ctx.stream.synchronize().expect("dtoh sync");
    host
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0f32;
    let mut mismatches = 0usize;
    for (x, y) in a.iter().zip(b) {
        let diff = (x - y).abs();
        if diff > 0.0 {
            mismatches += 1;
        }
        if diff > worst {
            worst = diff;
        }
    }
    (worst, mismatches)
}

#[test]
#[ignore = "needs one GPU with the accelerated FlashKDA build"]
fn fused_md_vs_two_calls_timing() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut seed = 0x7137_0f_u64;
    let width = HEADS * D;
    let state = HEADS * D * D;

    for t in [2112usize, 4224] {
        const SETS: usize = 8;
        let mut ops = Vec::new();
        for _ in 0..SETS {
            ops.push((
                ctx.stream
                    .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
                    .expect("q"),
                ctx.stream
                    .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
                    .expect("k"),
                ctx.stream
                    .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
                    .expect("v"),
                ctx.stream
                    .clone_htod(&xorshift_fill(&mut seed, t * width, 1.0))
                    .expect("g"),
                ctx.stream
                    .clone_htod(&xorshift_fill(&mut seed, t * HEADS, 1.0))
                    .expect("beta"),
            ));
        }
        let zero_v = ctx.stream.alloc_zeros::<bf16>(t * width).expect("zero v");
        let mut beta_scratch = ctx
            .stream
            .alloc_zeros::<bf16>(t * HEADS)
            .expect("beta scratch");
        let a_log = ctx.stream.clone_htod(&vec![0.5f32; HEADS]).expect("a_log");
        let dt_bias = ctx.stream.alloc_zeros::<f32>(width).expect("dt_bias");
        let identity = ctx
            .stream
            .clone_htod(&identity_state(HEADS))
            .expect("identity");
        let zero_state = ctx.stream.alloc_zeros::<f32>(state).expect("zero state");
        let mut out = ctx.stream.alloc_zeros::<bf16>(t * width).expect("out");
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(k3_flash_kda_workspace_bytes(t, HEADS).max(16))
            .expect("workspace");
        let mut m_slab = ctx.stream.alloc_zeros::<f32>(state).expect("m slab");
        let mut d_slab = ctx.stream.alloc_zeros::<f32>(state).expect("d slab");

        let iters = 200usize;
        let mut two_call = |ctx: &DeviceContext| {
            for i in 0..iters {
                let (q, k, v, g, beta) = &ops[i % SETS];
                k3_flash_kda_fwd_launch(
                    ctx,
                    t,
                    HEADS,
                    K3FlashKdaSpan::default(),
                    q,
                    k,
                    &zero_v,
                    g,
                    beta,
                    &mut beta_scratch,
                    &a_log,
                    &dt_bias,
                    &identity,
                    &mut m_slab,
                    &mut out,
                    &mut workspace,
                    SCALE,
                    LOWER_BOUND,
                )
                .expect("M forward");
                k3_flash_kda_fwd_launch(
                    ctx,
                    t,
                    HEADS,
                    K3FlashKdaSpan::default(),
                    q,
                    k,
                    v,
                    g,
                    beta,
                    &mut beta_scratch,
                    &a_log,
                    &dt_bias,
                    &zero_state,
                    &mut d_slab,
                    &mut out,
                    &mut workspace,
                    SCALE,
                    LOWER_BOUND,
                )
                .expect("D forward");
            }
            ctx.stream.synchronize().expect("sync");
        };
        two_call(&ctx); // warmup
        let start = std::time::Instant::now();
        two_call(&ctx);
        let two_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let mut fused = |ctx: &DeviceContext| {
            for i in 0..iters {
                let (q, k, v, g, beta) = &ops[i % SETS];
                k3_flash_kda_fwd_md_launch(
                    ctx,
                    t,
                    HEADS,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    &mut beta_scratch,
                    &a_log,
                    &dt_bias,
                    &mut d_slab,
                    &mut m_slab,
                    &mut workspace,
                    SCALE,
                    LOWER_BOUND,
                )
                .expect("fused forward");
            }
            ctx.stream.synchronize().expect("sync");
        };
        fused(&ctx); // warmup
        let start = std::time::Instant::now();
        fused(&ctx);
        let fused_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        eprintln!(
            "T={t:>5}: two calls {two_ms:.4} ms, fused {fused_ms:.4} ms ({:.2}x)",
            two_ms / fused_ms
        );
    }
}

#[test]
#[ignore = "needs one GPU with the accelerated FlashKDA build"]
fn fused_md_matches_the_two_doctored_calls() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut seed = 0x0dd5_eed5_0f_u64;
    let width = HEADS * D;
    let state = HEADS * D * D;

    // Tail (14), verify pack (224), chunk divisions (1056, 4224), and an
    // uneven segment (4223) — the shape every real uneven CP split has.
    for t in [14usize, 224, 1056, 4223, 4224] {
        let q = ctx
            .stream
            .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
            .expect("q");
        let k = ctx
            .stream
            .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
            .expect("k");
        let v = ctx
            .stream
            .clone_htod(&xorshift_fill(&mut seed, t * width, 0.5))
            .expect("v");
        let g = ctx
            .stream
            .clone_htod(&xorshift_fill(&mut seed, t * width, 1.0))
            .expect("g");
        let beta = ctx
            .stream
            .clone_htod(&xorshift_fill(&mut seed, t * HEADS, 1.0))
            .expect("beta");
        let zero_v = ctx.stream.alloc_zeros::<bf16>(t * width).expect("zero v");
        let mut beta_scratch = ctx
            .stream
            .alloc_zeros::<bf16>(t * HEADS)
            .expect("beta scratch");
        let a_log = ctx.stream.clone_htod(&vec![0.5f32; HEADS]).expect("a_log");
        let dt_bias = ctx.stream.alloc_zeros::<f32>(width).expect("dt_bias");
        let identity = ctx
            .stream
            .clone_htod(&identity_state(HEADS))
            .expect("identity");
        let zero_state = ctx.stream.alloc_zeros::<f32>(state).expect("zero state");
        let mut out = ctx.stream.alloc_zeros::<bf16>(t * width).expect("out");
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(k3_flash_kda_workspace_bytes(t, HEADS).max(16))
            .expect("workspace");

        // Reference: the two standalone doctored calls.
        let mut ref_m = ctx.stream.alloc_zeros::<f32>(state).expect("ref m");
        let mut ref_d = ctx.stream.alloc_zeros::<f32>(state).expect("ref d");
        k3_flash_kda_fwd_launch(
            &ctx,
            t,
            HEADS,
            K3FlashKdaSpan::default(),
            &q,
            &k,
            &zero_v,
            &g,
            &beta,
            &mut beta_scratch,
            &a_log,
            &dt_bias,
            &identity,
            &mut ref_m,
            &mut out,
            &mut workspace,
            SCALE,
            LOWER_BOUND,
        )
        .expect("reference M forward");
        k3_flash_kda_fwd_launch(
            &ctx,
            t,
            HEADS,
            K3FlashKdaSpan::default(),
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut beta_scratch,
            &a_log,
            &dt_bias,
            &zero_state,
            &mut ref_d,
            &mut out,
            &mut workspace,
            SCALE,
            LOWER_BOUND,
        )
        .expect("reference D forward");

        // Fused pass.
        let mut fused_d = ctx.stream.alloc_zeros::<f32>(state).expect("fused d");
        let mut fused_m = ctx.stream.alloc_zeros::<f32>(state).expect("fused m");
        k3_flash_kda_fwd_md_launch(
            &ctx,
            t,
            HEADS,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut beta_scratch,
            &a_log,
            &dt_bias,
            &mut fused_d,
            &mut fused_m,
            &mut workspace,
            SCALE,
            LOWER_BOUND,
        )
        .expect("fused MD forward");

        let (m_diff, m_bad) = max_abs_diff(&dtoh(&ctx, &ref_m), &dtoh(&ctx, &fused_m));
        let (d_diff, d_bad) = max_abs_diff(&dtoh(&ctx, &ref_d), &dtoh(&ctx, &fused_d));
        eprintln!(
            "T={t:>5}: M max|diff| {m_diff:.3e} ({m_bad} cells), D max|diff| {d_diff:.3e} ({d_bad} cells)"
        );
        assert_eq!(
            (m_diff, d_diff),
            (0.0, 0.0),
            "fused MD diverged from the doctored calls at T={t}"
        );
    }
}
