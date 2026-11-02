//! Times the Qwen4 QSA sparse-attention forward at prefill shapes.
//!
//!   cargo run --release --example qwen4_qsa_attn_bench -- [rows,]tokens
//!
//! One process, one arm. The kernel always scores every token, so an all-visible
//! mask measures the same work as a causal one while keeping the setup simple.
//! `BENCH_ITERS` / `BENCH_ROUNDS` override the sample protocol.

use std::time::Instant;

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::QsaAttnGeometry;
use pegainfer_kernels::ops::qsa_attn_forward_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n.is_multiple_of(2) {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    } else {
        xs[n / 2]
    }
}

fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "2048".into());
    let (rows, tokens) = if let Some((rows, tokens)) = arg.split_once(',') {
        (rows.parse()?, tokens.parse()?)
    } else {
        let tokens: usize = arg.parse()?;
        (tokens, tokens)
    };
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let rounds: usize = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let ctx = DeviceContext::new()?;
    let geometry = QsaAttnGeometry::qwen38_flash_next();

    let queries = ctx
        .stream
        .alloc_zeros::<bf16>(rows * geometry.query_width())?;
    let keys = ctx
        .stream
        .alloc_zeros::<bf16>(tokens * geometry.kv_width())?;
    let values = ctx
        .stream
        .alloc_zeros::<bf16>(tokens * geometry.kv_width())?;
    let mask = ctx.stream.clone_htod(&vec![1u8; rows * tokens])?;
    let rows_dev = ctx
        .stream
        .clone_htod(&(0..rows as i32).collect::<Vec<_>>())?;
    let positions = ctx
        .stream
        .clone_htod(&(0..tokens as i32).collect::<Vec<_>>())?;
    let q_norm = ctx.stream.alloc_zeros::<bf16>(geometry.head_dim)?;
    let k_norm = ctx.stream.alloc_zeros::<bf16>(geometry.head_dim)?;
    let cos = DeviceVec::from_host(&ctx, &vec![bf16::ZERO; tokens * geometry.rotary_dim])?;
    let sin = DeviceVec::from_host(&ctx, &vec![bf16::ZERO; tokens * geometry.rotary_dim])?;
    let mut weights = ctx
        .stream
        .alloc_zeros::<bf16>(geometry.heads * rows * tokens)?;
    let mut out = ctx
        .stream
        .alloc_zeros::<bf16>(rows * geometry.out_width())?;
    let mut normed_keys = ctx
        .stream
        .alloc_zeros::<bf16>(tokens * geometry.kv_width())?;

    let mut run = || -> Result<()> {
        qsa_attn_forward_into(
            &ctx,
            &geometry,
            &queries,
            &keys,
            &values,
            &mask,
            &rows_dev,
            &positions,
            &q_norm,
            &k_norm,
            &cos,
            &sin,
            &mut weights,
            &mut out,
            &mut normed_keys,
        )
    };
    run()?;
    ctx.stream.synchronize()?;

    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iters {
            run()?;
        }
        ctx.stream.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    let ms = median(samples);
    // Every CTA scores and reduces all tokens for every head: two 256-wide
    // dots per (row, head, token), two flops each.
    let flops =
        4.0 * rows as f64 * tokens as f64 * geometry.heads as f64 * geometry.head_dim as f64;
    println!(
        "rows {rows} tokens {tokens}: {ms:.3} ms median, {:.2} TFLOP/s, {} weight MiB",
        flops / (ms / 1e3) / 1e12,
        geometry.heads * rows * tokens * 2 / (1 << 20)
    );
    Ok(())
}
