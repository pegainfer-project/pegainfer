//! Microbenchmark for the DSpine bridge kernels at production shapes.
//! Not a correctness gate; run with:
//! `cargo test -q -p pegainfer-kernels --release --test dspine_microbench -- --ignored --nocapture`

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::dspine_feature_finish_into;
use pegainfer_kernels::ops::dspine_feature_finish_norm_into;
use pegainfer_kernels::ops::dspine_feature_message_into;
use pegainfer_kernels::ops::dspine_normalize_codes_into;
use pegainfer_kernels::ops::dspine_prepare_read_into;
use pegainfer_kernels::ops::dspine_serial_chain_into;
use pegainfer_kernels::ops::dspine_serial_table_into;
use pegainfer_kernels::ops::dspine_topk_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const VOCAB: usize = 151_936;
const HIDDEN: usize = 4096;
const CODE: usize = 1024;
const RANK: usize = 512;
const BLOCK: usize = 16;
const K: usize = 16;
const MAX_BATCH: usize = 256;
const EPS: f32 = 1e-6;

fn hidden(ctx: &DeviceContext, dim: usize, rows: usize, value: f32) -> HiddenStates {
    HiddenStates {
        data: ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(value); dim * rows])
            .expect("H2D"),
        hidden_dim: dim,
        seq_len: rows,
    }
}

fn matrix(ctx: &DeviceContext, rows: usize, cols: usize, value: f32) -> DeviceMatrix {
    DeviceMatrix {
        data: ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(value); rows * cols])
            .expect("H2D"),
        rows,
        cols,
    }
}

fn vector(ctx: &DeviceContext, len: usize, value: f32) -> DeviceVec {
    DeviceVec {
        data: ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(value); len])
            .expect("H2D"),
        len,
    }
}

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
    for _ in 0..5 {
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
    eprintln!("{label:28} p50={p50:9.3} ms  p95={p95:9.3} ms");
    (p50, p95)
}

#[test]
#[ignore = "microbenchmark; requires a GPU"]
fn dspine_microbench() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let max_rows = MAX_BATCH * BLOCK;
    let max_proposals = MAX_BATCH * (BLOCK - 1);

    let mut projected = hidden(&ctx, CODE, max_rows, 0.01);
    let mut normed = hidden(&ctx, HIDDEN, max_rows, 0.01);
    let codebook = matrix(&ctx, VOCAB, CODE, 0.01);
    let mut codes = hidden(&ctx, CODE, max_rows, 0.01);
    let ids: CudaSlice<u32> = ctx.stream.alloc_zeros(max_rows)?;
    let mut packed = hidden(&ctx, 3 * RANK, max_rows, 0.01);
    let bias = vector(&ctx, RANK, 0.01);
    let mut message = hidden(&ctx, RANK, max_rows, 0.01);
    let mut raw = hidden(&ctx, HIDDEN, max_rows, 0.01);
    let mut delta = hidden(&ctx, HIDDEN, max_rows, 0.01);
    let mut out = hidden(&ctx, HIDDEN, max_rows, 0.01);
    let mut rms: CudaSlice<f32> = ctx.stream.alloc_zeros(max_rows)?;
    let lm_head = matrix(&ctx, VOCAB, HIDDEN, 0.01);
    let norm = vector(&ctx, HIDDEN, 0.01);
    let tg = matrix(&ctx, VOCAB, RANK, 0.01);
    let tsv = matrix(&ctx, VOCAB, 2 * RANK, 0.01);
    let mut candidates: CudaSlice<i32> = ctx.stream.alloc_zeros(max_proposals * K)?;
    let mut idx: CudaSlice<i32> = ctx.stream.alloc_zeros(max_proposals * K)?;
    let mut tokens: CudaSlice<u32> = ctx.stream.alloc_zeros(max_rows)?;
    let mut logits = hidden(&ctx, VOCAB, max_rows, 0.5);

    for batch in [1usize, 8, 64, 256] {
        let rows = batch * BLOCK;
        let proposals = batch * (BLOCK - 1);
        eprintln!("--- batch {batch} (rows {rows}, proposals {proposals})");
        projected.seq_len = rows;
        normed.seq_len = rows;
        codes.seq_len = rows;
        packed.seq_len = rows;
        message.seq_len = rows;
        raw.seq_len = rows;
        delta.seq_len = rows;
        out.seq_len = rows;
        logits.seq_len = rows;

        bench(&ctx, "prepare_read", 30, || {
            dspine_prepare_read_into(&ctx, &raw, &mut normed, &mut rms, EPS).unwrap();
        });
        bench(&ctx, "normalize_codes", 30, || {
            dspine_normalize_codes_into(&ctx, &projected, &codebook, &ids, &mut codes, BLOCK, EPS)
                .unwrap();
        });
        bench(&ctx, "feature_message", 30, || {
            dspine_feature_message_into(&ctx, &packed, &bias, &mut message, BLOCK).unwrap();
        });
        bench(&ctx, "feature_finish", 30, || {
            dspine_feature_finish_into(&ctx, &raw, &delta, &rms, &mut out, BLOCK).unwrap();
        });
        bench(&ctx, "feature_finish_norm", 30, || {
            dspine_feature_finish_norm_into(
                &ctx,
                &raw,
                &delta,
                &rms,
                &mut out,
                &mut normed,
                &norm,
                BLOCK,
                EPS,
            )
            .unwrap();
        });
        bench(&ctx, "topk", 20, || {
            dspine_topk_into(&ctx, &logits, batch, BLOCK, K, &mut candidates).unwrap();
        });
        bench(&ctx, "serial_table", 10, || {
            dspine_serial_table_into(
                &ctx,
                &packed,
                &rms,
                &raw,
                &lm_head,
                &norm,
                &tg,
                &tsv,
                &bias,
                &candidates,
                &ids,
                &mut idx,
                batch,
                BLOCK,
                K,
            )
            .unwrap();
        });
        bench(&ctx, "serial_chain", 30, || {
            dspine_serial_chain_into(&ctx, &idx, &candidates, &ids, &mut tokens, batch, BLOCK, K)
                .unwrap();
        });
    }
    Ok(())
}
