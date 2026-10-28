//! Microbenchmarks for the Qwen4 grouped RMSNorm and QSA indexer scoring
//! kernels at production shapes. Not a correctness gate; run with:
//! `cargo test -q -p pegainfer-kernels --release --features qwen4 \
//!  --test qwen4_operator_microbench -- --ignored --nocapture`

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::qsa_indexer_block_scores_into;
use pegainfer_kernels::ops::rms_norm_grouped_offset_batch_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HIDDEN: usize = 10_240;
const GROUP: usize = 2_560;
const HEAD_DIM: usize = 128;
const FUSED_WIDTH: usize = 640;
const ROTARY_DIM: usize = 64;
const COMPRESS_RATIO: usize = 4;
const EPS: f32 = 1e-6;

fn events(ctx: &DeviceContext) -> (cudarc::driver::CudaEvent, cudarc::driver::CudaEvent) {
    let flag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    (
        ctx.ctx.new_event(flag).expect("event"),
        ctx.ctx.new_event(flag).expect("event"),
    )
}

fn percentile(samples: &mut [f32], fraction: f64) -> f32 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[(samples.len() as f64 * fraction) as usize]
}

fn bench(ctx: &DeviceContext, label: &str, iters: usize, mut run: impl FnMut()) -> (f32, f32) {
    let (start, end) = events(ctx);
    for _ in 0..10 {
        run();
    }
    ctx.sync().expect("sync");
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        start.record(&ctx.stream).expect("record");
        run();
        end.record(&ctx.stream).expect("record");
        samples.push(start.elapsed_ms(&end).expect("elapsed"));
    }
    let p50 = percentile(&mut samples, 0.50);
    let p95 = percentile(&mut samples.clone(), 0.95);
    eprintln!("{label:36} p50={:9.4} ms  p95={:9.4} ms", p50, p95);
    (p50, p95)
}

fn values(len: usize, seed: u64) -> Vec<bf16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            bf16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect()
}

fn hidden(ctx: &DeviceContext, len: usize, seed: u64) -> HiddenStates {
    HiddenStates {
        data: ctx.stream.clone_htod(&values(len, seed)).expect("H2D"),
        hidden_dim: len,
        seq_len: 1,
    }
}

fn vector(ctx: &DeviceContext, len: usize, seed: u64) -> DeviceVec {
    DeviceVec {
        data: ctx.stream.clone_htod(&values(len, seed)).expect("H2D"),
        len,
    }
}

fn bench_grouped_norm(ctx: &DeviceContext) -> Result<()> {
    let weight = vector(ctx, HIDDEN, 0x1158);
    for tokens in [1usize, 16, 64, 128, 512] {
        let mut x = hidden(ctx, tokens * HIDDEN, 0x2000 + tokens as u64);
        x.hidden_dim = HIDDEN;
        x.seq_len = tokens;
        let mut out = HiddenStates::zeros(ctx, HIDDEN, tokens)?;
        let (p50, _) = bench(ctx, &format!("grouped_norm tokens={tokens}"), 100, || {
            rms_norm_grouped_offset_batch_into(ctx, &x, &weight, GROUP, EPS, &mut out).unwrap();
        });
        eprintln!("grouped_norm tokens={tokens}: {:.3} GB/s", {
            // Algorithmic minimum: one input read, one output write, weight once.
            let bytes = tokens as f64 * HIDDEN as f64 * 2.0 * 2.0 + HIDDEN as f64 * 2.0;
            bytes / (p50 as f64 * 1e-3) / 1e9
        });
    }
    Ok(())
}

fn bench_qsa_case(ctx: &DeviceContext, tokens: usize, rows: &[i32], iters: usize) -> Result<()> {
    let query_rows = rows.len();
    let score_cols = tokens / COMPRESS_RATIO;
    let visible: Vec<i32> = rows.iter().map(|row| row + 1).collect();
    let fused: CudaSlice<bf16> = ctx
        .stream
        .clone_htod(&values(tokens * FUSED_WIDTH, 0x1160 + tokens as u64))?;
    let q_norm = vector(ctx, HEAD_DIM, 0x1161);
    let k_norm = vector(ctx, HEAD_DIM, 0x1162);
    let cos = vector(ctx, tokens * ROTARY_DIM, 0x1163);
    let sin = vector(ctx, tokens * ROTARY_DIM, 0x1164);
    let rows_dev = ctx.stream.clone_htod(rows)?;
    let visible_dev = ctx.stream.clone_htod(&visible)?;
    let mut scores: CudaSlice<f32> = ctx.stream.alloc_zeros(query_rows * score_cols)?;
    let mut pooled: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros((tokens / COMPRESS_RATIO) * HEAD_DIM)?;
    let label = format!("qsa tokens={tokens} rows={query_rows}");
    let (p50, _) = bench(ctx, &label, iters, || {
        qsa_indexer_block_scores_into(
            ctx,
            &fused,
            &q_norm.data,
            &k_norm.data,
            &cos,
            &sin,
            &rows_dev,
            &visible_dev,
            tokens,
            &mut scores,
            score_cols,
            &mut pooled,
            EPS,
        )
        .unwrap();
    });
    let scores = rows
        .iter()
        .map(|row| (row + 1) as usize / COMPRESS_RATIO)
        .sum::<usize>();
    eprintln!(
        "qsa tokens={tokens} rows={query_rows}: {:.3} scores/us",
        scores as f64 / (p50 as f64 * 1e3)
    );
    Ok(())
}

fn bench_qsa(ctx: &DeviceContext) -> Result<()> {
    let decode_6: Vec<i32> = (2049..2055).collect();
    let prefill_512: Vec<i32> = (0..512).collect();
    let prefill_64: Vec<i32> = (0..64).collect();
    bench_qsa_case(ctx, 2055, &[2054], 200)?;
    bench_qsa_case(ctx, 2055, &decode_6, 200)?;
    bench_qsa_case(ctx, 512, &[511], 300)?;
    bench_qsa_case(ctx, 512, &prefill_512, 100)?;
    bench_qsa_case(ctx, 64, &prefill_64, 500)?;
    Ok(())
}

#[test]
#[ignore = "microbenchmark; requires a GPU and the qwen4 feature"]
fn qwen4_operator_microbench() -> Result<()> {
    let ctx = DeviceContext::new()?;
    bench_grouped_norm(&ctx)?;
    bench_qsa(&ctx)?;
    Ok(())
}
