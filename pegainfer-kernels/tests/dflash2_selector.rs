//! Actual GPU selector gates. Binary-fraction fixtures keep the CPU equation
//! exact while asymmetric codebooks expose row/operand/predecessor mistakes.
#![allow(
    clippy::float_cmp,
    reason = "Binary-fraction fixtures require exact scores and tie ordering"
)]

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::DFlash2Scratch;
use pegainfer_kernels::ops::dflash2_select_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::StreamOverrideGuard;

const N: usize = 16;
const BLOCK: usize = 4;
const LENGTH: usize = BLOCK - 1;
const K: usize = 16;
const RANK: usize = 5;
const VOCAB: usize = 1031;

struct Inputs {
    hidden: Vec<bf16>,
    logits: Vec<bf16>,
    a: Vec<bf16>,
    b: Vec<bf16>,
    anchors: Vec<u32>,
}

impl Inputs {
    fn new() -> Self {
        Self {
            hidden: (0..N * BLOCK * RANK)
                .map(|i| bf16::from_f32(((i % 9) as f32 - 4.0) / 8.0))
                .collect(),
            logits: (0..N * BLOCK * VOCAB)
                .map(|i| bf16::from_f32((((i % VOCAB) * 17 + (i / VOCAB) * 13) % 67) as f32 / 8.0))
                .collect(),
            a: (0..VOCAB * RANK)
                .map(|i| bf16::from_f32(((i % 11) as f32 - 5.0) / 8.0))
                .collect(),
            b: (0..VOCAB * RANK)
                .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) / 8.0))
                .collect(),
            anchors: (0..N).map(|i| (i * 31 + 2) as u32).collect(),
        }
    }

    fn poison_inactive(&mut self, active: usize) {
        for request in 0..N {
            // The discarded anchor slot is poisoned even for active
            // requests. It must never become a candidate or a gated hidden row.
            let row = request * BLOCK;
            self.logits[row * VOCAB..(row + 1) * VOCAB].fill(bf16::NAN);
            self.hidden[row * RANK..(row + 1) * RANK].fill(bf16::NAN);
        }
        self.logits[active * BLOCK * VOCAB..].fill(bf16::NAN);
        self.hidden[active * BLOCK * RANK..].fill(bf16::NAN);
        self.anchors[active..].fill(u32::MAX);
    }
}

struct Oracle {
    candidates: Vec<u32>,
    unary: Vec<f32>,
    edges: Vec<f32>,
    path: Vec<u32>,
}

fn oracle(input: &Inputs, active: usize) -> Oracle {
    let mut output = Oracle {
        candidates: Vec::new(),
        unary: Vec::new(),
        edges: Vec::new(),
        path: Vec::new(),
    };
    for request in 0..active {
        let mut previous_ids = vec![input.anchors[request]; K];
        let mut previous_index = 0;
        for position in 0..LENGTH {
            let row = request * BLOCK + 1 + position;
            let logits = &input.logits[row * VOCAB..(row + 1) * VOCAB];
            let mut ids: Vec<_> = (0..VOCAB).collect();
            ids.sort_by(|&left, &right| {
                let a = logits[left].to_f32();
                let b = logits[right].to_f32();
                if a == b {
                    left.cmp(&right)
                } else {
                    b.total_cmp(&a)
                }
            });
            ids.truncate(K);
            output.candidates.extend(ids.iter().map(|&id| id as u32));
            output
                .unary
                .extend(ids.iter().map(|&id| logits[id].to_f32()));

            let edge_start = output.edges.len();
            for &previous in &previous_ids {
                for &id in &ids {
                    let pair: f32 = (0..RANK)
                        .map(|r| {
                            input.a[previous as usize * RANK + r].to_f32()
                                * input.hidden[row * RANK + r].to_f32()
                                * input.b[id * RANK + r].to_f32()
                        })
                        .sum();
                    output.edges.push(logits[id].to_f32() + pair);
                }
            }

            let scores = &output.edges
                [edge_start + previous_index * K..edge_start + (previous_index + 1) * K];
            previous_index = (0..K)
                .min_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(ids[a].cmp(&ids[b])))
                .unwrap();
            output.path.push(ids[previous_index] as u32);
            previous_ids = ids.iter().map(|&id| id as u32).collect();
        }
    }
    output
}

fn verify(
    ctx: &DeviceContext,
    scratch: &mut DFlash2Scratch,
    input: &Inputs,
    active: usize,
) -> Result<()> {
    let h = HiddenStates::from_host(ctx, &input.hidden, RANK, N * BLOCK)?;
    let u = HiddenStates::from_host(ctx, &input.logits, VOCAB, N * BLOCK)?;
    let a = DeviceMatrix::from_host(ctx, &input.a, VOCAB, RANK)?;
    let b = DeviceMatrix::from_host(ctx, &input.b, VOCAB, RANK)?;
    let anchors = ctx.stream.clone_htod(&input.anchors)?;

    dflash2_select_into(ctx, &u, &h, &a, &b, &anchors, active, scratch)?;

    assert_eq!(ctx.stream.clone_dtoh(scratch.error_flag())?, [0]);

    let expected = oracle(input, active);
    let candidates = ctx.stream.clone_dtoh(scratch.candidate_ids())?;
    let unary = ctx.stream.clone_dtoh(scratch.unary_scores())?;
    let edges = ctx.stream.clone_dtoh(scratch.edge_scores())?;
    let path = ctx.stream.clone_dtoh(scratch.selected_ids())?;

    assert_eq!(
        &candidates[..expected.candidates.len()],
        expected.candidates
    );

    assert_eq!(&unary[..expected.unary.len()], expected.unary);

    assert_eq!(&edges[..expected.edges.len()], expected.edges);

    assert_eq!(&path[..expected.path.len()], expected.path);
    Ok(())
}

#[test]
fn strict_candidates_full_lattice_and_request_isolation() -> Result<()> {
    let ctx = DeviceContext::new()?;
    // Chunk 5 does not divide 48 rows or the active batch=1 tail (3 rows).
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK, 5)?;
    for active in [16, 1, 8, 16] {
        let mut input = Inputs::new();
        input.poison_inactive(active);
        verify(&ctx, &mut scratch, &input, active)?;
    }

    let mut input = Inputs::new();
    input.poison_inactive(1);
    input.logits[VOCAB..2 * VOCAB].fill(bf16::NEG_INFINITY);
    for id in 200..215 {
        input.logits[VOCAB + id] = bf16::from_f32(2.0);
    }
    for id in [1, 1024] {
        input.logits[VOCAB + id] = bf16::ONE;
    }
    verify(&ctx, &mut scratch, &input, 1)?;
    let candidates = ctx.stream.clone_dtoh(scratch.candidate_ids())?;

    assert_eq!(
        candidates[15], 1,
        "tie at the 16/17 boundary must retain the smaller ID"
    );

    for row in 1..BLOCK {
        input.hidden[row * RANK..(row + 1) * RANK].fill(bf16::ZERO);
        for id in 0..VOCAB {
            input.logits[row * VOCAB + id] = bf16::from_f32(if id % 2 == 0 { 0.0 } else { -0.0 });
        }
    }
    verify(&ctx, &mut scratch, &input, 1)?;

    assert_eq!(
        &ctx.stream.clone_dtoh(scratch.candidate_ids())?[..K],
        (0..K as u32).collect::<Vec<_>>()
    );

    assert_eq!(
        &ctx.stream.clone_dtoh(scratch.selected_ids())?[..LENGTH],
        [0; LENGTH]
    );

    // Minimum vocabulary and a non-tile-aligned vocabulary both use CUB.
    let mut minimal = DFlash2Scratch::new(&ctx, 1, 2, K, 1, 1)?;
    let h = HiddenStates::from_host(&ctx, &[bf16::ZERO; 2], 1, 2)?;
    let u = HiddenStates::from_host(&ctx, &[bf16::ZERO; 2 * K], K, 2)?;
    let a = DeviceMatrix::from_host(&ctx, &[bf16::ONE; K], K, 1)?;
    let anchor = ctx.stream.clone_htod(&[0u32])?;
    dflash2_select_into(&ctx, &u, &h, &a, &a, &anchor, 1, &mut minimal)?;

    assert_eq!(
        ctx.stream.clone_dtoh(minimal.candidate_ids())?,
        (0..K as u32).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn invalid_inputs_never_publish_and_next_call_recovers() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK, 2)?;
    for case in 0..5 {
        let mut input = Inputs::new();
        input.poison_inactive(1);
        let expected_flag = match case {
            0 => {
                input.logits[VOCAB] = bf16::NAN;
                1
            }
            1 => {
                input.logits[VOCAB] = bf16::INFINITY;
                1
            }
            2 => {
                input.logits[VOCAB..2 * VOCAB].fill(bf16::NEG_INFINITY);
                input.logits[VOCAB..VOCAB + 15].fill(bf16::ZERO);
                2
            }
            3 => {
                input.anchors[0] = VOCAB as u32;
                4
            }
            _ => {
                input.hidden[RANK] = bf16::NAN;
                8
            }
        };

        let h = HiddenStates::from_host(&ctx, &input.hidden, RANK, N * BLOCK)?;
        let u = HiddenStates::from_host(&ctx, &input.logits, VOCAB, N * BLOCK)?;
        let a = DeviceMatrix::from_host(&ctx, &input.a, VOCAB, RANK)?;
        let b = DeviceMatrix::from_host(&ctx, &input.b, VOCAB, RANK)?;
        let anchors = ctx.stream.clone_htod(&input.anchors)?;
        dflash2_select_into(&ctx, &u, &h, &a, &b, &anchors, 1, &mut scratch)?;
        assert_ne!(
            ctx.stream.clone_dtoh(scratch.error_flag())?[0] & expected_flag,
            0,
            "error case {case}"
        );
        assert_eq!(
            &ctx.stream.clone_dtoh(scratch.selected_ids())?[..LENGTH],
            [u32::MAX; LENGTH]
        );

        let mut valid = Inputs::new();
        valid.poison_inactive(1);
        verify(&ctx, &mut scratch, &valid, 1)?;
    }
    Ok(())
}

#[test]
fn foreign_streams_and_over_capacity_are_rejected_before_launch() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let input = Inputs::new();
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK, 1)?;
    let h = HiddenStates::from_host(&ctx, &input.hidden, RANK, N * BLOCK)?;
    let u = HiddenStates::from_host(&ctx, &input.logits, VOCAB, N * BLOCK)?;
    let a = DeviceMatrix::from_host(&ctx, &input.a, VOCAB, RANK)?;
    let b = DeviceMatrix::from_host(&ctx, &input.b, VOCAB, RANK)?;
    let anchors = ctx.stream.clone_htod(&input.anchors)?;
    assert!(dflash2_select_into(&ctx, &u, &h, &a, &b, &anchors, N + 1, &mut scratch).is_err());

    let other = ctx.ctx.new_stream()?;
    let foreign_anchors = other.clone_htod(&input.anchors)?;
    assert!(dflash2_select_into(&ctx, &u, &h, &a, &b, &foreign_anchors, 1, &mut scratch).is_err());

    // SAFETY: stream is from this same CUDA context and outlives the guard.
    let guard = unsafe { StreamOverrideGuard::activate(other.cu_stream()) };
    assert!(dflash2_select_into(&ctx, &u, &h, &a, &b, &anchors, 1, &mut scratch).is_err());
    drop(guard);

    assert!(DFlash2Scratch::new(&ctx, usize::MAX, BLOCK, VOCAB, RANK, 1).is_err());

    assert!(DFlash2Scratch::new(&ctx, 1, 1, VOCAB, RANK, 1).is_err());

    assert!(DFlash2Scratch::new(&ctx, 1, 2, 15, RANK, 1).is_err());
    Ok(())
}
