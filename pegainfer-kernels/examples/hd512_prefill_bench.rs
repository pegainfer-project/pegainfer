//! Times both hd512 global-attention prefills at the shapes the chunked walk
//! launches, and the generated one over both pool row formats. One kernel per
//! process, so `ncu` needs no filter; the arms share one plan and alternate
//! which leads each round.
//!
//!   cargo run --release --example hd512_prefill_bench -- <kv_len>[,<kv_len>...]
//!
//! Each kv_len is one chunk: the query block is CHUNK rows ending at kv_len,
//! which is what lines the causal mask up with the engine's.
use std::fmt::Write as _;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_hd512_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::gemma4_hd512_prefill_varlen_into;
#[cfg(feature = "gemma4")]
use pegainfer_kernels::paged_kv::KvFormat;
#[cfg(feature = "gemma4")]
use pegainfer_kernels::paged_kv::KvStorage;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
/// The global family's rotated columns, which the folded format keeps of K.
#[cfg(feature = "gemma4")]
const ROTARY: usize = 128;
// The global family's page size. The generated kernel refuses anything else,
// and the incumbent at page 16 was measured to be worth 1.02-1.04x, so a
// comparison at 16 would be measuring a shape neither side ships.
const PAGE: usize = 64;
const CHUNK: usize = 8192;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const LAYERS: usize = 10;

/// The seam the serving path switches on, so the bench cannot drift from it.
type Attend = fn(
    &DeviceContext,
    &HiddenStates,
    &CudaSlice<bf16>,
    &PagedKvLayout,
    usize,
    &PrefillPagedPlan,
    &mut HiddenStates,
    usize,
    f32,
) -> Result<()>;

struct Arm {
    name: &'static str,
    attend: Attend,
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
    let arg = std::env::args().nth(1).unwrap_or_else(|| "8192".into());
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let rounds: usize = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let ctx = DeviceContext::new()?;

    let split = PagedKvLayout::new(LAYERS, KV_HEADS, HD, PAGE);
    let mut arms = vec![Arm {
        name: "incumbent",
        attend: batch_prefill_paged_hd512_into,
        layout: split,
    }];
    #[cfg(feature = "gemma4")]
    if gemma4_hd512_prefill_is_built() {
        arms.push(Arm {
            name: "tilelang",
            attend: gemma4_hd512_prefill_varlen_into,
            layout: split,
        });
        arms.push(Arm {
            name: "tilelang640",
            attend: gemma4_hd512_prefill_varlen_into,
            layout: PagedKvLayout::with_storage_and_format(
                LAYERS,
                KV_HEADS,
                HD,
                PAGE,
                KvStorage::Bf16,
                KvFormat::Folded { rotary: ROTARY },
            ),
        });
    } else {
        println!("note: this build carries the stub, so only the incumbent runs");
    }

    let mut header = format!("{:>9} {:>7}", "kv_len", "pages");
    for arm in &arms {
        let ms = format!("{} ms", arm.name);
        write!(header, " {ms:>13} {:>7}", "TF").expect("String write");
    }
    for arm in &arms[1..] {
        let speedup = format!("x {}", arm.name);
        write!(header, " {speedup:>13}").expect("String write");
    }
    write!(header, " {:>12}", "req GiB").expect("String write");
    println!("{header}");
    for token in arg.split(',') {
        let kv_len: usize = token.trim().parse()?;
        anyhow::ensure!(
            kv_len >= CHUNK && kv_len.is_multiple_of(PAGE),
            "kv_len {kv_len} must be >= {CHUNK} and a multiple of {PAGE}"
        );
        let pages = kv_len.div_ceil(PAGE);
        // One page holds every layer's rows for PAGE tokens, which is what
        // puts a single layer's pages apart in the real pool.
        let pools: Vec<CudaSlice<bf16>> = arms
            .iter()
            .map(|arm| Ok(ctx.stream.alloc_zeros(pages * arm.layout.page_stride)?))
            .collect::<Result<_>>()?;
        let page_indices: Vec<i32> = (0..pages as i32).collect();
        let plan = PrefillPagedPlan::new_with_cta_tile_q(
            &ctx,
            &page_indices,
            (kv_len - 1) % PAGE + 1,
            kv_len - CHUNK,
            CHUNK,
            Q_HEADS,
            KV_HEADS,
            HD,
            0,
        )?;
        let q = HiddenStates::zeros(&ctx, Q_HEADS * HD, CHUNK)?;
        let mut out = HiddenStates::zeros(&ctx, Q_HEADS * HD, CHUNK)?;
        let scale = (HD as f32).powf(-0.5);
        let time = |arm: &Arm, pool: &CudaSlice<bf16>, out: &mut HiddenStates| -> Result<f64> {
            let run = |out: &mut HiddenStates| {
                (arm.attend)(
                    &ctx,
                    &q,
                    pool,
                    &arm.layout,
                    LAYERS / 2,
                    &plan,
                    out,
                    Q_HEADS,
                    scale,
                )
            };
            run(out)?;
            ctx.stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                run(out)?;
            }
            ctx.stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
        };

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            // Alternate which arm leads, so whichever penalty falls on going
            // first is not always paid by the same one.
            let order: Vec<usize> = if round.is_multiple_of(2) {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                samples[i].push(time(&arms[i], &pools[i], &mut out)?);
            }
        }

        // Causal, bottom-right aligned: the mean key count per query row is
        // kv_len - CHUNK/2, and each of the two matmuls costs two flops per MAC.
        let flops =
            4.0 * CHUNK as f64 * (kv_len as f64 - CHUNK as f64 / 2.0) * Q_HEADS as f64 * HD as f64;
        // Every incumbent CTA rescans its kv head's history, so its requested
        // bytes scale with the CTA count, which is the whole point of the
        // comparison.
        let ctas = (CHUNK * Q_HEADS) as f64 / 32.0;
        let req = ctas * (kv_len as f64 - CHUNK as f64 / 2.0) * (HD * 2 * 2) as f64;
        let ms: Vec<f64> = samples.into_iter().map(median).collect();
        let tf = |m: f64| flops / (m / 1e3) / 1e12;
        let mut line = format!("{kv_len:>9} {pages:>7}");
        for &m in &ms {
            write!(line, " {m:>13.3} {:>7.1}", tf(m)).expect("String write");
        }
        for &m in &ms[1..] {
            write!(line, " {:>13.3}", ms[0] / m).expect("String write");
        }
        write!(line, " {:>12.1}", req / (1u64 << 30) as f64).expect("String write");
        println!("{line}");
    }
    Ok(())
}
