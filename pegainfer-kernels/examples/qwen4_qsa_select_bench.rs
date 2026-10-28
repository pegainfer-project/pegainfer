//! Manual timing for the QSA indexer selection kernel.
//!
//! `cargo run --release -p pegainfer-kernels --features qwen4 \
//!   --example qwen4_qsa_select_bench`
//!
//! One row per case unless the case says otherwise. Each timed sample launches
//! the selector `reps` times, synchronizes once, and divides by `reps`, so the
//! short cases amortize the host round trip. The mask is downloaded after
//! timing and checked for the expected set-bit count. Set `QSA_BENCH_CASE` to a
//! case-name substring (for example `256k`) to run one case, which is what
//! `nsys profile` wants for a per-case kernel median.

use std::time::Instant;

use pegainfer_kernels::ops::qsa_indexer_select_into;
use pegainfer_kernels::tensor::DeviceContext;

/// The checkpoint's indexer budget, in tokens.
const BUDGET: i32 = 2048;

struct Case {
    name: &'static str,
    position: usize,
    rows: usize,
    reps: usize,
    iters: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "all-kept 2k x1",
        position: 2047,
        rows: 1,
        reps: 100,
        iters: 20,
    },
    Case {
        name: "all-kept 2k x64",
        position: 2047,
        rows: 64,
        reps: 20,
        iters: 20,
    },
    Case {
        name: "trunc 2k+ x1",
        position: 2051,
        rows: 1,
        reps: 100,
        iters: 20,
    },
    Case {
        name: "trunc 8k x1",
        position: 8191,
        rows: 1,
        reps: 20,
        iters: 20,
    },
    Case {
        name: "trunc 64k x1",
        position: 65535,
        rows: 1,
        reps: 5,
        iters: 20,
    },
    Case {
        name: "trunc 256k x1",
        position: 262143,
        rows: 1,
        reps: 1,
        iters: 20,
    },
    Case {
        name: "trunc 64k x8",
        position: 65535,
        rows: 8,
        reps: 1,
        iters: 20,
    },
];

fn synthetic_scores(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / 16_777_216.0
        })
        .collect()
}

fn summarize(mut samples: Vec<f32>) -> (f32, f32, f32, f32) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let at = |fraction: f64| {
        let index = ((samples.len() as f64 * fraction).ceil() as usize).min(samples.len() - 1);
        samples[index]
    };
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    (mean, at(0.50), at(0.95), at(0.99))
}

fn main() {
    let ctx = DeviceContext::new().expect("CUDA context");
    let filter = std::env::var("QSA_BENCH_CASE").ok();
    for case in CASES {
        if let Some(filter) = &filter {
            if !case.name.contains(filter.as_str()) {
                continue;
            }
        }
        let tokens = case.position + 1;
        let score_cols = tokens / 4;
        let blocks = tokens / 4;
        let keep = (BUDGET as usize / 4).min(blocks);
        let tail = tokens - blocks * 4;
        let scores = synthetic_scores(case.rows * score_cols, 0x51A5 + tokens as u64);
        let host_rows = vec![case.position as i32; case.rows];
        let scores_dev = ctx.stream.clone_htod(&scores).expect("scores H2D");
        let rows_dev = ctx.stream.clone_htod(&host_rows).expect("rows H2D");
        let mut mask_dev = ctx
            .stream
            .alloc_zeros::<u8>(case.rows * tokens)
            .expect("mask");

        let launch = |ctx: &DeviceContext, mask: &mut _| {
            qsa_indexer_select_into(
                ctx,
                &scores_dev,
                score_cols,
                &rows_dev,
                BUDGET,
                tokens,
                mask,
            )
            .expect("select launch");
        };
        for _ in 0..5 {
            launch(&ctx, &mut mask_dev);
        }
        ctx.sync().expect("sync");

        let mut samples = Vec::with_capacity(case.iters);
        for _ in 0..case.iters {
            let start = Instant::now();
            for _ in 0..case.reps {
                launch(&ctx, &mut mask_dev);
            }
            ctx.sync().expect("sync");
            samples.push(start.elapsed().as_secs_f64() as f32 * 1e3 / case.reps as f32);
        }

        let host_mask = ctx.stream.clone_dtoh(&mask_dev).expect("mask D2H");
        let set: usize = host_mask.iter().map(|cell| *cell as usize).sum();
        assert_eq!(
            set,
            case.rows * (keep * 4 + tail),
            "{}: set bits",
            case.name
        );

        let (mean, p50, p95, p99) = summarize(samples);
        let path = if keep == blocks { "fast" } else { "radix" };
        println!(
            "{:16} rows={:3} tokens={:7} blocks={:6} keep={:3} path={:5}  \
             mean={:8.3}us p50={:8.3}us p95={:8.3}us p99={:8.3}us",
            case.name,
            case.rows,
            tokens,
            blocks,
            keep,
            path,
            mean * 1e3,
            p50 * 1e3,
            p95 * 1e3,
            p99 * 1e3,
        );
    }
}
