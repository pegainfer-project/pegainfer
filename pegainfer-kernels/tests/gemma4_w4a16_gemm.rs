//! The W4A16 path against its definition, `y = x @ ((q - 8) * s)^T`, on every
//! 31B linear shape: the load-time rewrite and dequantization bit-exact, the
//! TileLang GEMMs at each row bucket (and at a count that pads into one, and
//! one past the largest, which dequantizes) against a host product, a row's
//! bits the same in every bucket, and a relaunch reproducing its output. The
//! gate|up shape's TileLang GEMMs write gelu(gate) * up, checked against the
//! same activation of the host product.
//!
//! Without a device, or in a build without the generated GEMMs, it skips;
//! `PEGAINFER_REQUIRE_GPU=1` turns either into a failure.

#![cfg(feature = "gemma4")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::W4a16Matrix;
use pegainfer_kernels::ops::W4a16Scratch;
use pegainfer_kernels::ops::gemma4_w4a16_gemm_into;
use pegainfer_kernels::ops::gemma4_w4a16_geometry;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

const GROUP: usize = 32;
/// gate|up: two stacked halves whose TileLang GEMMs write the MLP activation.
const GELU_MUL: (usize, usize) = (43008, 5376);
const SHAPES: [(usize, usize); 6] = [
    (16384, 5376),
    (18432, 5376),
    (5376, 8192),
    (5376, 16384),
    (43008, 5376),
    (5376, 21504),
];
/// Every `COL_STRIDE`-th output column is checked against the host product.
const COL_STRIDE: usize = 37;

struct Checkpoint {
    packed: Vec<u32>,
    scales: Vec<bf16>,
}

impl Checkpoint {
    fn random(seed: u64, rows: usize, cols: usize) -> Self {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let packed = (0..rows * cols / 8).map(|_| next() as u32).collect();
        let scales = (0..rows * cols / GROUP)
            .map(|_| bf16::from_f32(0.001 + (next() >> 40) as f32 / 16_777_216.0 * 0.02))
            .collect();
        Self { packed, scales }
    }

    /// `(q - 8) * s` rounded to bf16, the reference's weight.
    fn weight(&self, rows: usize, cols: usize) -> Vec<bf16> {
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let q = (self.packed[(r * cols + c) / 8] >> (4 * (c % 8))) & 0xF;
                let s = self.scales[r * (cols / GROUP) + c / GROUP].to_f32();
                out.push(bf16::from_f32((q as f32 - 8.0) * s));
            }
        }
        out
    }

    fn upload(&self, ctx: &DeviceContext, rows: usize, cols: usize, gelu_mul: bool) -> W4a16Matrix {
        let packed: Vec<u8> = self.packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = self
            .scales
            .iter()
            .flat_map(|s| s.to_bits().to_le_bytes())
            .collect();
        let packed = ctx.stream.clone_htod(&packed).expect("packed upload");
        let scales = ctx.stream.clone_htod(&scales).expect("scales upload");
        W4a16Matrix::from_checkpoint(ctx, &packed, &scales, rows, cols, gelu_mul).expect("rewrite")
    }
}

fn host(ctx: &DeviceContext, states: &HiddenStates, rows: usize) -> Vec<f32> {
    let all = ctx.stream.clone_dtoh(&states.data).expect("download");
    all[..rows * states.hidden_dim]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

/// gelu_pytorch_tanh(gate) * up from the two products, rounded as the
/// activation kernel rounds its bf16 inputs and output.
fn gelu_mul(gate: f32, up: f32) -> f32 {
    let g = bf16::from_f32(gate).to_f32();
    let u = bf16::from_f32(up).to_f32();
    let inner = 0.797_884_6_f32 * (g + 0.044_715 * g * g * g);
    let gelu = bf16::from_f32(0.5 * g * (1.0 + inner.tanh())).to_f32();
    bf16::from_f32(gelu * u).to_f32()
}

/// Output columns of a step: gate|up's TileLang GEMMs write half as many.
fn width(weight: &W4a16Matrix, rows: usize) -> usize {
    if weight.gelu_mul && W4a16Matrix::runs_tilelang(rows) {
        weight.rows / 2
    } else {
        weight.rows
    }
}

/// `rows` rows of `x` through the GEMM into a 16-row-capacity output.
fn run(
    ctx: &DeviceContext,
    weight: &W4a16Matrix,
    x: &[bf16],
    rows: usize,
    scratch: &mut W4a16Scratch,
) -> Vec<f32> {
    let capacity = rows.max(16);
    let mut padded = x[..rows * weight.cols].to_vec();
    padded.resize(capacity * weight.cols, bf16::from_f32(7.0));
    let input = HiddenStates {
        data: ctx.stream.clone_htod(&padded).expect("x upload"),
        hidden_dim: weight.cols,
        seq_len: rows,
    };
    let mut out = HiddenStates::zeros(ctx, width(weight, rows), capacity).expect("out");
    out.seq_len = rows;
    gemma4_w4a16_gemm_into(ctx, weight, &input, scratch, &mut out).expect("gemm");
    host(ctx, &out, rows)
}

#[test]
fn w4a16_gemm_matches_its_definition_on_every_31b_shape() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    if gemma4_w4a16_geometry().is_none() {
        assert!(
            std::env::var("PEGAINFER_REQUIRE_GPU").as_deref() != Ok("1"),
            "PEGAINFER_REQUIRE_GPU=1 but this build carries no W4A16 GEMMs"
        );
        eprintln!("skipping: this build carries no W4A16 GEMMs");
        return;
    }
    let largest = SHAPES.iter().map(|(n, k)| n * k).max().unwrap();
    let mut scratch = W4a16Scratch::new(&ctx, largest).expect("scratch");
    for (shape, &(n, k)) in SHAPES.iter().enumerate() {
        let checkpoint = Checkpoint::random(0x51A7 + shape as u64, n, k);
        let weight = checkpoint.upload(&ctx, n, k, (n, k) == GELU_MUL);
        let reference = checkpoint.weight(n, k);

        let mut dense = DeviceMatrix {
            data: ctx.stream.alloc_zeros(n * k).expect("dense"),
            rows: 0,
            cols: 0,
        };
        weight.dequant_into(&ctx, &mut dense).expect("dequant");
        let got = ctx.stream.clone_dtoh(&dense.data).expect("dense download");
        let mismatch = got
            .iter()
            .zip(&reference)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            mismatch.is_none(),
            "{n} x {k}: dequantized weight differs at {mismatch:?}"
        );

        let x = common::fill(0xF00D + shape as u64, 20 * k);
        let mut first_rows: Vec<Vec<f32>> = Vec::new();
        let product = |r: usize, c: usize| -> f32 {
            (0..k)
                .map(|i| x[r * k + i].to_f32() * reference[c * k + i].to_f32())
                .sum()
        };
        for rows in [1, 2, 3, 4, 8, 16, 20] {
            let y = run(&ctx, &weight, &x, rows, &mut scratch);
            let w = width(&weight, rows);
            for r in 0..rows {
                let mut worst = 0.0f32;
                let mut peak = 0.0f32;
                for c in (0..w).step_by(COL_STRIDE) {
                    let want = if w < n {
                        gelu_mul(product(r, c), product(r, c + w))
                    } else {
                        product(r, c)
                    };
                    worst = worst.max((y[r * w + c] - want).abs());
                    peak = peak.max(want.abs());
                }
                assert!(
                    worst <= 2e-2 * peak.max(1.0),
                    "{n} x {k} at {rows} rows, row {r}: worst {worst} against peak {peak}"
                );
            }
            if rows <= 16 {
                first_rows.push(y[..w].to_vec());
            }
            if rows == 16 {
                let again = run(&ctx, &weight, &x, rows, &mut scratch);
                assert_eq!(y, again, "{n} x {k}: a relaunch changed the output");
            }
        }
        for (i, row) in first_rows.iter().enumerate().skip(1) {
            assert!(
                row.iter()
                    .zip(&first_rows[0])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{n} x {k}: row 0 differs between bucket runs 0 and {i}"
            );
        }
    }
}
