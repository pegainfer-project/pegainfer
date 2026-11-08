//! Sparse proposal verification, including the target sampler's filter order.

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::BatchSamplingRow;
use pegainfer_kernels::ops::BatchSamplingScratch;
use pegainfer_kernels::ops::SparseVerifyRequest;
use pegainfer_kernels::ops::SparseVerifyScratch;
use pegainfer_kernels::ops::sparse_speculative_sample;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

#[test]
fn sparse_verify_accepts_corrects_and_preserves_the_target_distribution() -> Result<()> {
    const VOCAB: usize = 32;
    const BLOCK: usize = 4;
    const GRID: usize = 24;
    const SAMPLES: usize = GRID * GRID * GRID;
    let ctx = DeviceContext::new()?;
    let cases = [
        // (target, q by position, draft IDs, length, top-k, top-p, min-p, u, expected)
        (
            [0.25; 4],
            [[0.25; 4]; 3],
            [0, 1, 2],
            3,
            0,
            1.0,
            0.0,
            [0.5, 0.5, 0.5, 0.9],
            (3, 3),
        ),
        (
            [0.25; 4],
            [[0.75, 0.25, 0.0, 0.0]; 3],
            [0, 0, 0],
            3,
            0,
            1.0,
            0.0,
            [0.9, 0.5, 0.5, 0.75],
            (0, 3),
        ),
        (
            [0.25; 4],
            [[0.25; 4], [0.75, 0.25, 0.0, 0.0], [0.25; 4]],
            [0, 0, 2],
            3,
            0,
            1.0,
            0.0,
            [0.5, 0.9, 0.5, 0.25],
            (1, 2),
        ),
        (
            [0.25; 4],
            [[0.25; 4]; 3],
            [0, 1, 2],
            1,
            0,
            1.0,
            0.0,
            [0.5, 0.5, 0.5, 0.9],
            (1, 3),
        ),
        // The largest uniform times the rounded-up q rounds back to p. The
        // second draft is rejected with zero residual; draw token 2 from p.
        (
            [0.25; 4],
            [[0.25; 4], [0.25_f32.next_up(), 0.25, 0.25, 0.25], [0.25; 4]],
            [1, 0, 0],
            2,
            0,
            1.0,
            0.0,
            [0.5, 1.0_f32.next_down(), 0.5, 0.625],
            (1, 2),
        ),
        // Top-p keeps only token 1. A tiny proposal probability can coexist
        // with q[1] == 1 after rounding; correcting to draft 0 would violate top-p.
        (
            [0.25, 0.75, 0.0, 0.0],
            [[f32::EPSILON / 4.0, 1.0, 0.0, 0.0]; 3],
            [0, 0, 0],
            1,
            0,
            0.5,
            0.0,
            [0.5; 4],
            (0, 1),
        ),
        // An anchor-only span emits the target token without checking drafts.
        (
            [0.25; 4],
            [[0.25; 4]; 3],
            [0, 1, 2],
            0,
            0,
            1.0,
            0.0,
            [0.5, 0.5, 0.5, 0.9],
            (0, 3),
        ),
        // Joint top-k/top-p keeps tokens 0 and 1. Sequential top-k -> top-p
        // keeps only token 0, and is used only when min-p is enabled.
        (
            [0.4, 0.3, 0.2, 0.1],
            [[0.0, 1.0, 0.0, 0.0]; 3],
            [1, 1, 1],
            1,
            2,
            0.5,
            0.0,
            [0.2, 0.5, 0.5, 0.75],
            (1, 1),
        ),
        (
            [0.4, 0.3, 0.2, 0.1],
            [[0.0, 1.0, 0.0, 0.0]; 3],
            [1, 1, 1],
            1,
            2,
            0.5,
            0.01,
            [0.2, 0.5, 0.5, 0.75],
            (0, 0),
        ),
        // Top-k retains all ties at the threshold, like the existing sampler.
        (
            [0.25; 4],
            [[0.0, 0.0, 0.0, 1.0]; 3],
            [3, 3, 3],
            1,
            2,
            1.0,
            0.0,
            [0.2, 0.5, 0.5, 0.9],
            (1, 3),
        ),
    ];
    let mut logits = vec![bf16::NEG_INFINITY; cases.len() * BLOCK * VOCAB];
    let mut candidates = vec![0u32; cases.len() * BLOCK * 16];
    let mut q = vec![0.0; cases.len() * (BLOCK - 1) * 16];
    let mut drafts = Vec::new();
    let mut uniforms = Vec::new();
    let mut rows = Vec::new();
    let mut requests = Vec::new();
    for (i, (target, proposal, tokens, length, top_k, top_p, min_p, random, _)) in
        cases.iter().enumerate()
    {
        // Reverse source rows to make arena indices differ from compact rows.
        let source = cases.len() - 1 - i;
        requests.push(SparseVerifyRequest {
            logits_offset: rows.len(),
            draft_row: i,
            num_drafts: *length,
        });
        for pos in 0..BLOCK {
            for (token, &probability) in target.iter().enumerate() {
                logits[(source * BLOCK + pos) * VOCAB + token] =
                    bf16::from_f32(f32::ln(probability));
            }
            for k in 0..16 {
                candidates[(i * BLOCK + pos) * 16 + k] = k as u32;
            }
            if pos <= *length {
                rows.push(BatchSamplingRow {
                    row: source * BLOCK + pos,
                    temperature: 1.0,
                    top_k: *top_k,
                    top_p: *top_p,
                    min_p: *min_p,
                });
            }
        }
        for pos in 0..BLOCK - 1 {
            q[(i * (BLOCK - 1) + pos) * 16..(i * (BLOCK - 1) + pos) * 16 + 4]
                .copy_from_slice(&proposal[pos]);
        }
        drafts.extend(tokens);
        uniforms.extend(random);
    }
    let logits = HiddenStates::from_host(&ctx, &logits, VOCAB, cases.len() * BLOCK)?;
    let candidates = ctx.stream.clone_htod(&candidates)?;
    let q = ctx.stream.clone_htod(&q)?;
    let drafts = ctx.stream.clone_htod(&drafts)?;
    let mut scratch = SparseVerifyScratch::new(&ctx, cases.len(), BLOCK)?;
    let mut sampling = BatchSamplingScratch::new(&ctx, rows.len(), VOCAB)?;
    let actual = sparse_speculative_sample(
        &ctx,
        logits.as_ref(),
        &rows,
        &requests,
        &candidates,
        &q,
        &drafts,
        &uniforms,
        &mut scratch,
        &mut sampling,
    )?;
    for (i, (got, case)) in actual.iter().zip(&cases).enumerate() {
        assert_eq!(*got, case.8, "case {i}");
    }

    // Integrate proposal, accept and posterior uniforms on a fixed midpoint
    // grid. q=[3/4,1/4,0,0] must still emit the uniform target, including its
    // mass outside q's support. This catches a plausible-looking accept-only
    // implementation that never performs residual correction.
    let logits: Vec<_> = (0..2 * VOCAB)
        .map(|i| {
            if i % VOCAB < 4 {
                bf16::ZERO
            } else {
                bf16::NEG_INFINITY
            }
        })
        .collect();
    let logits = HiddenStates::from_host(&ctx, &logits, VOCAB, 2)?;
    let candidates: Vec<_> = (0..SAMPLES * 2 * 16).map(|i| (i % 16) as u32).collect();
    let candidates = ctx.stream.clone_htod(&candidates)?;
    let q: Vec<_> = (0..SAMPLES * 16)
        .map(|i| match i % 16 {
            0 => 0.75,
            1 => 0.25,
            _ => 0.0,
        })
        .collect();
    let q = ctx.stream.clone_htod(&q)?;
    let mut drafts = Vec::with_capacity(SAMPLES);
    let mut uniforms = Vec::with_capacity(SAMPLES * 2);
    let mut requests = Vec::with_capacity(SAMPLES);
    for proposal in 0..GRID {
        for accept in 0..GRID {
            for posterior in 0..GRID {
                requests.push(SparseVerifyRequest {
                    logits_offset: 0,
                    draft_row: drafts.len(),
                    num_drafts: 1,
                });
                drafts.push(u32::from(proposal >= GRID * 3 / 4));
                uniforms.extend([
                    (accept as f32 + 0.5) / GRID as f32,
                    (posterior as f32 + 0.5) / GRID as f32,
                ]);
            }
        }
    }
    let device_drafts = ctx.stream.clone_htod(&drafts)?;
    let rows: Vec<_> = (0..2)
        .map(|row| BatchSamplingRow {
            row,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        })
        .collect();
    let mut scratch = SparseVerifyScratch::new(&ctx, SAMPLES, 2)?;
    let mut sampling = BatchSamplingScratch::new(&ctx, SAMPLES * 2, VOCAB)?;
    let output = sparse_speculative_sample(
        &ctx,
        logits.as_ref(),
        &rows,
        &requests,
        &candidates,
        &q,
        &device_drafts,
        &uniforms,
        &mut scratch,
        &mut sampling,
    )?;
    let mut counts = [0usize; 4];
    for ((accepted, posterior), proposed) in output.into_iter().zip(drafts) {
        let token = if accepted == 0 { posterior } else { proposed };
        counts[token as usize] += 1;
    }
    for count in counts {
        assert!(
            (count as f64 / SAMPLES as f64 - 0.25).abs() < 0.01,
            "output counts {counts:?}"
        );
    }
    Ok(())
}
