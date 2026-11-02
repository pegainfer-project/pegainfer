//! Times both hd512 global-attention decodes at the context lengths the fair
//! A/B probes, the generated one over both pool row formats. Same instrument
//! as `hd512_prefill_bench`. A decode step is one query row over the whole
//! context, so what is measured is the streaming of one layer's global KV,
//! which is what the folded format shrinks; its arm runs over its own pool.
//!
//!   cargo run --release --features gemma4 --example hd512_decode_bench -- <kv_len>[,<kv_len>...]
use std::fmt::Write as _;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GlobalDecodeAttend;
use pegainfer_kernels::ops::Hd512DecodeMetadata;
use pegainfer_kernels::ops::gemma4_hd512_decode_split_kv_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::paged_attention_batch_decode_split_kv_hd512_into;
use pegainfer_kernels::paged_kv::KvFormat;
use pegainfer_kernels::paged_kv::KvStorage;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
/// The global family's rotated columns, which the folded format keeps of K.
const ROTARY: usize = 128;
const PAGE: usize = 64;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const LAYERS: usize = 10;
/// The serving path's split chunk.
const CHUNK: usize = 256;

struct Arm {
    name: &'static str,
    attend: GlobalDecodeAttend,
    layout: PagedKvLayout,
}

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
    let arg = std::env::args().nth(1).unwrap_or_else(|| "32768".into());
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let rounds: usize = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let ctx = DeviceContext::new()?;
    let split = PagedKvLayout::new(LAYERS, KV_HEADS, HD, PAGE);
    let folded = PagedKvLayout::with_storage_and_format(
        LAYERS,
        KV_HEADS,
        HD,
        PAGE,
        KvStorage::Bf16,
        KvFormat::Folded { rotary: ROTARY },
    );
    let mut arms = vec![Arm {
        name: "incumbent",
        attend: paged_attention_batch_decode_split_kv_hd512_into,
        layout: split,
    }];
    if gemma4_hd512_prefill_is_built() {
        arms.push(Arm {
            name: "tilelang",
            attend: gemma4_hd512_decode_split_kv_into,
            layout: split,
        });
        arms.push(Arm {
            name: "tilelang640",
            attend: gemma4_hd512_decode_split_kv_into,
            layout: folded,
        });
    } else {
        println!("note: this build carries the stub, so only the incumbent runs");
    }
    let mut header = format!("{:>9} {:>6}", "kv_len", "slots");
    for arm in &arms {
        let us = format!("{} us", arm.name);
        write!(header, " {us:>12} {:>7}", "GB/s").expect("String write");
    }
    for arm in &arms[1..] {
        let speedup = format!("x {}", arm.name);
        write!(header, " {speedup:>13}").expect("String write");
    }
    println!("{header}");
    for token in arg.split(',') {
        let kv_len: usize = token.trim().parse()?;
        let pages = kv_len.div_ceil(PAGE);
        let pools: Vec<CudaSlice<bf16>> = arms
            .iter()
            .map(|arm| Ok(ctx.stream.alloc_zeros(pages * arm.layout.page_stride)?))
            .collect::<Result<_>>()?;
        let page_indices: Vec<i32> = (0..pages as i32).collect();
        let chunks = kv_len.div_ceil(CHUNK);
        let up = |v: &[i32]| -> Result<CudaSlice<i32>> { Ok(ctx.stream.clone_htod(v)?) };
        let page_indices_d = up(&page_indices)?;
        let page_indptr_d = up(&[0, pages as i32])?;
        let last_d = up(&[((kv_len - 1) % PAGE + 1) as i32])?;
        let req_d = up(&vec![0i32; chunks])?;
        let tile_d = up(&(0..chunks as i32).collect::<Vec<_>>())?;
        let chunk_d = up(&[CHUNK as i32])?;
        let o_indptr_d = up(&[0, chunks as i32])?;
        let valid_d: CudaSlice<u8> = ctx.stream.clone_htod(&vec![1u8; chunks])?;
        let mut tmp_v: CudaSlice<bf16> = ctx.stream.alloc_zeros(chunks * Q_HEADS * HD)?;
        let mut tmp_s: CudaSlice<f32> = ctx.stream.alloc_zeros(chunks * Q_HEADS)?;
        let q = HiddenStates::zeros(&ctx, Q_HEADS * HD, 1)?;
        let mut out = HiddenStates::zeros(&ctx, Q_HEADS * HD, 1)?;
        let meta = Hd512DecodeMetadata::new(
            &page_indices_d,
            &page_indptr_d,
            &last_d,
            &req_d,
            &tile_d,
            &chunk_d,
            CHUNK,
        );
        let time = |arm: &Arm,
                    pool: &CudaSlice<bf16>,
                    out: &mut HiddenStates,
                    tmp_v: &mut CudaSlice<bf16>,
                    tmp_s: &mut CudaSlice<f32>|
         -> Result<f64> {
            let run = |out: &mut HiddenStates,
                       tmp_v: &mut CudaSlice<bf16>,
                       tmp_s: &mut CudaSlice<f32>| {
                (arm.attend)(
                    &ctx,
                    &q,
                    0,
                    pool,
                    &arm.layout,
                    LAYERS / 2,
                    &meta,
                    &o_indptr_d,
                    &valid_d,
                    tmp_v,
                    tmp_s,
                    chunks,
                    out,
                    Q_HEADS,
                    1.0,
                )
            };
            run(out, tmp_v, tmp_s)?;
            ctx.stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                run(out, tmp_v, tmp_s)?;
            }
            ctx.stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
        };
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            let order: Vec<usize> = if round.is_multiple_of(2) {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                samples[i].push(time(&arms[i], &pools[i], &mut out, &mut tmp_v, &mut tmp_s)?);
            }
        }
        let us: Vec<f64> = samples.into_iter().map(median).collect();
        let mut line = format!("{kv_len:>9} {chunks:>6}");
        for (arm, &u) in arms.iter().zip(&us) {
            // One layer's rows over the whole context, at this arm's width.
            let bytes = (KV_HEADS * arm.layout.format.values_per_token(HD) * 2 * kv_len) as f64;
            write!(line, " {u:>12.1} {:>7.0}", bytes / (u * 1e-6) / 1e9).expect("String write");
        }
        for &u in &us[1..] {
            write!(line, " {:>13.3}", us[0] / u).expect("String write");
        }
        println!("{line}");
    }
    Ok(())
}
