//! Actual GPU selector gates. Binary-fraction fixtures keep the CPU equation
//! exact while asymmetric codebooks expose row/operand/predecessor mistakes.
#![allow(
    clippy::float_cmp,
    reason = "Binary-fraction fixtures require exact scores"
)]

use anyhow::Result;
use cudarc::driver::sys::CUgraphInstantiate_flags;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::bf16;
use pegainfer_kernels::ops::DFlash2Scratch;
use pegainfer_kernels::ops::dflash2_select_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::StreamOverrideGuard;

const N: usize = 16;
const BLOCK: usize = 8;
const LENGTH: usize = BLOCK - 1;
const K: usize = 16;
const RANK: usize = 5;
// A production vocabulary exercises FlashInfer's multi-CTA radix path.
const VOCAB: usize = 151_936;

struct Inputs {
    hidden: Vec<bf16>,
    logits: Vec<bf16>,
    a: Vec<bf16>,
    b: Vec<bf16>,
    anchors: Vec<u32>,
}

impl Inputs {
    fn new() -> Self {
        let mut input = Self {
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
        };
        // Unique top-16 scores give an exact candidate-ID oracle, while tied
        // background values still require finding the right cutoff.
        for row in 0..N * BLOCK {
            for candidate in 0..K {
                let id = (row * 31 + candidate * 7919) % VOCAB;
                input.logits[row * VOCAB + id] = bf16::from_f32(32.0 - candidate as f32 / 8.0);
            }
        }
        input
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

fn oracle(input: &Inputs, active: usize, candidates: &[u32]) -> Oracle {
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
            let start = (request * LENGTH + position) * K;
            let ids: Vec<_> = candidates[start..start + K]
                .iter()
                .map(|&id| id as usize)
                .collect();
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
    exact_candidates: bool,
) -> Result<()> {
    let h = HiddenStates::from_host(ctx, &input.hidden, RANK, N * BLOCK)?;
    let u = HiddenStates::from_host(ctx, &input.logits, VOCAB, N * BLOCK)?;
    let a = DeviceMatrix::from_host(ctx, &input.a, VOCAB, RANK)?;
    let b = DeviceMatrix::from_host(ctx, &input.b, VOCAB, RANK)?;
    let anchors = ctx.stream.clone_htod(&input.anchors)?;

    dflash2_select_into(ctx, &u, &h, &a, &b, &anchors, active, scratch)?;

    assert_eq!(ctx.stream.clone_dtoh(scratch.error_flag())?, [0]);

    let candidates = ctx.stream.clone_dtoh(scratch.candidate_ids())?;
    let unary = ctx.stream.clone_dtoh(scratch.unary_scores())?;
    let edges = ctx.stream.clone_dtoh(scratch.edge_scores())?;
    let path = ctx.stream.clone_dtoh(scratch.selected_ids())?;

    for (row, ids) in candidates[..active * LENGTH * K]
        .as_chunks::<K>()
        .0
        .iter()
        .enumerate()
    {
        let source = (row / LENGTH) * BLOCK + 1 + row % LENGTH;
        let logits = &input.logits[source * VOCAB..(source + 1) * VOCAB];
        let mut ranked: Vec<_> = (0..VOCAB as u32).collect();
        let order = |&left: &u32, &right: &u32| {
            logits[right as usize]
                .to_f32()
                .partial_cmp(&logits[left as usize].to_f32())
                .unwrap()
                .then(left.cmp(&right))
        };
        ranked.select_nth_unstable_by(K - 1, order);
        ranked.truncate(K);
        ranked.sort_by(order);
        if exact_candidates {
            assert_eq!(ids.as_slice(), ranked.as_slice());
        }
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), K);
        for (&id, &expected_id) in ids.iter().zip(&ranked) {
            assert!((id as usize) < VOCAB);
            // Require every ranked top-16 value, allowing any equal-valued
            // member at the cutoff. Sorting 16 outputs alone cannot fix ties.
            assert_eq!(logits[id as usize], logits[expected_id as usize]);
        }
    }

    let expected = oracle(input, active, &candidates);
    assert_eq!(&unary[..expected.unary.len()], expected.unary);

    assert_eq!(&edges[..expected.edges.len()], expected.edges);

    assert_eq!(&path[..expected.path.len()], expected.path);

    // Warmup above precedes capture; repeated launches reuse the same radix
    // state and device pointers, including production-sized multi-CTA rows.
    ctx.stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let captured = dflash2_select_into(ctx, &u, &h, &a, &b, &anchors, active, scratch);
    let graph = ctx
        .stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
    captured?;
    let graph = graph.expect("selector capture contains GPU work");
    for _ in 0..3 {
        graph.launch()?;
        assert_eq!(ctx.stream.clone_dtoh(scratch.error_flag())?, [0]);
        assert_eq!(
            &ctx.stream.clone_dtoh(scratch.candidate_ids())?[..expected.candidates.len()],
            expected.candidates
        );
        assert_eq!(
            &ctx.stream.clone_dtoh(scratch.selected_ids())?[..expected.path.len()],
            expected.path
        );
    }
    Ok(())
}

#[test]
fn candidates_lattice_graph_replay_and_request_isolation() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK)?;
    for active in [16, 1, 8, 16] {
        let mut input = Inputs::new();
        input.poison_inactive(active);
        verify(&ctx, &mut scratch, &input, active, true)?;
    }

    let mut input = Inputs::new();
    input.poison_inactive(1);
    input.logits[VOCAB..2 * VOCAB].fill(bf16::NEG_INFINITY);
    for id in 200..215 {
        input.logits[VOCAB + id] = bf16::from_f32(2.0);
    }
    for id in [1, 1024, VOCAB - 1] {
        input.logits[VOCAB + id] = bf16::ONE;
    }
    verify(&ctx, &mut scratch, &input, 1, false)?;

    for row in 1..BLOCK {
        input.hidden[row * RANK..(row + 1) * RANK].fill(bf16::ZERO);
        for id in 0..VOCAB {
            input.logits[row * VOCAB + id] = bf16::from_f32(if id % 2 == 0 { 0.0 } else { -0.0 });
        }
    }
    verify(&ctx, &mut scratch, &input, 1, false)?;

    Ok(())
}

#[test]
fn invalid_inputs_never_publish_and_next_call_recovers() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK)?;
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
    }
    let mut valid = Inputs::new();
    valid.poison_inactive(1);
    verify(&ctx, &mut scratch, &valid, 1, true)?;
    Ok(())
}

#[test]
fn foreign_streams_and_over_capacity_are_rejected_before_launch() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let input = Inputs::new();
    let mut scratch = DFlash2Scratch::new(&ctx, N, BLOCK, VOCAB, RANK)?;
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

    Ok(())
}
