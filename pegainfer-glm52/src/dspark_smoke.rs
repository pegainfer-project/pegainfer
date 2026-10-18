//! DSpark draft perf smoke: measures `propose()` wall time (one draft round =
//! 7 draft tokens) against the bandwidth-bound floor, on a single GPU.
//!
//! Uses the zero-weight synthetic model at the checkpoint's exact geometry —
//! GPU kernel time is value-independent, so no checkpoint or 8-GPU box is
//! needed. Requires ~14 GiB free VRAM at peak (weights 3.9 GiB + embed/
//! lm_head 3.8 GiB + bs=8 slot states ~5.5 GiB + scratch; see
//! `glm52_dspark_arena_bytes`).
//!
//! Run (single GPU):
//! ```text
//! cargo test --release -p pegainfer-glm52 --lib \
//!   dspark_propose_smoke -- --nocapture --ignored
//! ```

use std::sync::Mutex;
use std::time::Instant;

/// Serializes the env-mutating tests (the graph kill-switches are process
/// globals) — `--test-threads=1` is recommended for GPU tests but this makes
/// the requirement local instead of assumed.
static ENV_LOCK: Mutex<()> = Mutex::new(());

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_VOCAB;
use crate::dspark::GLM52_DSPARK_CONTEXT_DIM;
use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::dspark::Glm52DsparkModel;
use crate::dspark::Glm52DsparkScratch;
use crate::dspark::Glm52DsparkSlotState;
use crate::dspark::dspark_cache_len;

const WARMUP: usize = 5;
const ITERS: usize = 50;

/// One propose round per iteration, one captured context row per round —
/// the bs=1 steady-state shape (accept length only changes the cheap fc
/// input rows). Returns per-round times.
fn bench_propose(
    ctx: &DeviceContext,
    model: &Glm52DsparkModel,
    embed: &DeviceMatrix,
    lm_head: &DeviceMatrix,
    scratch: &mut Glm52DsparkScratch,
    batch: usize,
) -> Result<Vec<f64>> {
    let mut states = (0..batch)
        .map(|_| Glm52DsparkSlotState::new(ctx, model.cache_len()))
        .collect::<Result<Vec<_>>>()?;
    let captured: cudarc::driver::CudaSlice<bf16> =
        ctx.stream.alloc_zeros(GLM52_DSPARK_CONTEXT_DIM)?;

    let mut times_ms = Vec::with_capacity(ITERS);
    for round in 0..WARMUP + ITERS {
        for state in &mut states {
            state.append_captured_row(ctx, &captured, 0)?;
        }
        // Each round drains 1 pending row into committed, so the anchor sits
        // at `round + 1`.
        let anchors = vec![(7u32, round + 1); batch];
        ctx.sync()?;
        let started = Instant::now();
        let mut state_refs: Vec<&mut Glm52DsparkSlotState> = states.iter_mut().collect();
        let drafts = model.propose(ctx, embed, lm_head, &mut state_refs, &anchors, scratch)?;
        ctx.sync()?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(drafts.len(), batch);
        assert!(drafts.iter().all(|d| d.len() == GLM52_DSPARK_DRAFTS));
        if round >= WARMUP {
            times_ms.push(elapsed);
        }
    }
    Ok(times_ms)
}

#[test]
#[ignore = "GPU perf smoke — needs a CUDA device with ~11 GB free VRAM"]
fn dspark_propose_smoke() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let cache_len = dspark_cache_len(4096);
    let model = Glm52DsparkModel::synthetic(&ctx, cache_len)?;
    let zero_head = || -> Result<DeviceMatrix> {
        Ok(DeviceMatrix {
            data: ctx.stream.alloc_zeros(GLM52_VOCAB * GLM52_HIDDEN)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        })
    };
    let embed = zero_head()?;
    let lm_head = zero_head()?;
    let mut scratch = Glm52DsparkScratch::new(&ctx, cache_len)?;

    for batch in [1usize, 8] {
        let mut times = bench_propose(&ctx, &model, &embed, &lm_head, &mut scratch, batch)?;
        times.sort_by(f64::total_cmp);
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let (min, p50, max) = (times[0], times[times.len() / 2], times[times.len() - 1]);
        println!(
            "dspark propose bs={batch}: avg {avg:.3} ms, min {min:.3} ms, p50 {p50:.3} ms, \
             max {max:.3} ms ({ITERS} rounds, 7 drafts/round)"
        );
    }
    Ok(())
}

/// Graph-vs-eager parity under the schedules serving actually produces:
/// mixed accept lengths (1..=8 pending rows per round — each `context_len` is
/// its own graph key, and the tail bounds must be re-set every round, not
/// baked at capture) and mixed row counts (a bs=2 round between bs=1 rounds —
/// bs>1 must not disturb the captured bs=1 graphs or the Markov buffer
/// orientation). The graphed run must emit exactly the eager run's tokens.
#[test]
#[ignore = "GPU parity test — needs a CUDA device with ~14 GiB free VRAM"]
fn dspark_propose_graph_parity() -> Result<()> {
    // Per-round (mode, len_a, len_b); mode 0 = slot A only, 1 = slot B only,
    // 2 = both (bs=2). Deliberate teeth: (A, len 8) warms at r3, captures at
    // r5, and REPLAYS at r7 right after a round whose tails were shorter —
    // pre-fix, the replay would consume the stale (smaller) tail seq_len and
    // trip the copy range check. The bs=2 rounds between bs=1 rounds pin the
    // Markov buffer orientation across row counts.
    const ROUNDS: usize = 14;
    const SCHEDULE: [(u8, usize, usize); ROUNDS] = [
        (0, 1, 0), // A warm (A,1)
        (1, 0, 3), // B warm (B,3)
        (2, 2, 4), // both, eager
        (0, 8, 0), // A warm (A,8)
        (1, 0, 5), // B warm (B,5)
        (0, 8, 0), // A capture (A,8)
        (1, 0, 3), // B capture (B,3) — leaves short tails
        (0, 8, 0), // A REPLAY (A,8): stale-short seq_len fires pre-fix
        (2, 1, 1), // both, eager
        (0, 1, 0), // A capture (A,1)
        (1, 0, 5), // B capture (B,5)
        (0, 8, 0), // A replay (A,8) again
        (1, 0, 3), // B replay (B,3)
        (0, 1, 0), // A replay (A,1)
    ];
    let _env = ENV_LOCK.lock().unwrap();
    let ctx = DeviceContext::new()?;
    let cache_len = dspark_cache_len(4096);
    let mut model = Glm52DsparkModel::synthetic(&ctx, cache_len)?;
    model.randomize_for_test(&ctx)?;
    let rand_head = |seed: u32| -> Result<DeviceMatrix> {
        let mut state = seed | 1;
        let host: Vec<half::bf16> = (0..GLM52_VOCAB * GLM52_HIDDEN)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                half::bf16::from_f32(((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02)
            })
            .collect();
        Ok(DeviceMatrix {
            data: ctx.stream.clone_htod(&host)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        })
    };
    let embed = rand_head(11)?;
    let lm_head = rand_head(23)?;
    let captured: cudarc::driver::CudaSlice<half::bf16> = {
        let host: Vec<half::bf16> = (0..GLM52_DSPARK_CONTEXT_DIM)
            .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) * 0.001))
            .collect();
        ctx.stream.clone_htod(&host)?
    };

    let run = |graphs: bool| -> Result<Vec<Vec<[u32; GLM52_DSPARK_DRAFTS]>>> {
        // SAFETY: ENV_LOCK serializes the env-mutating tests in this file.
        unsafe {
            if graphs {
                std::env::remove_var("DSPARK_NO_FORWARD_GRAPH");
                std::env::remove_var("DSPARK_NO_MARKOV_GRAPH");
            } else {
                std::env::set_var("DSPARK_NO_FORWARD_GRAPH", "1");
                std::env::set_var("DSPARK_NO_MARKOV_GRAPH", "1");
            }
        }
        let mut scratch = Glm52DsparkScratch::new(&ctx, cache_len)?;
        let mut slot_a = Glm52DsparkSlotState::new(&ctx, cache_len)?;
        let mut slot_b = Glm52DsparkSlotState::new(&ctx, cache_len)?;
        let mut pos = [0usize; 2];
        let mut all = Vec::with_capacity(ROUNDS);
        for (round, &(mode, len_a, len_b)) in SCHEDULE.iter().enumerate() {
            let mut round_drafts = Vec::new();
            if mode == 2 {
                for _ in 0..len_a {
                    slot_a.append_captured_row(&ctx, &captured, 0)?;
                }
                for _ in 0..len_b {
                    slot_b.append_captured_row(&ctx, &captured, 0)?;
                }
                pos[0] += len_a;
                pos[1] += len_b;
                let anchors = vec![(7 + round as u32, pos[0]), (91 + round as u32, pos[1])];
                let mut refs = vec![&mut slot_a, &mut slot_b];
                let drafts =
                    model.propose(&ctx, &embed, &lm_head, &mut refs, &anchors, &mut scratch)?;
                round_drafts.extend(drafts);
            } else {
                let which = mode as usize;
                let (state, len, anchor_base) = if which == 0 {
                    (&mut slot_a, len_a, 7u32)
                } else {
                    (&mut slot_b, len_b, 91u32)
                };
                for _ in 0..len {
                    state.append_captured_row(&ctx, &captured, 0)?;
                }
                pos[which] += len;
                let anchors = vec![(anchor_base + round as u32, pos[which])];
                let mut refs = vec![&mut *state];
                let drafts =
                    model.propose(&ctx, &embed, &lm_head, &mut refs, &anchors, &mut scratch)?;
                round_drafts.extend(drafts);
            }
            all.push(round_drafts);
        }
        unsafe {
            std::env::remove_var("DSPARK_NO_FORWARD_GRAPH");
            std::env::remove_var("DSPARK_NO_MARKOV_GRAPH");
        }
        Ok(all)
    };

    let graphed = run(true)?;
    let eager = run(false)?;
    assert_eq!(
        graphed, eager,
        "graphed propose diverged from the eager path"
    );
    assert!(
        graphed.iter().flatten().flatten().any(|&t| t != 0),
        "parity ran on degenerate logits (all drafts are token 0)"
    );
    Ok(())
}

/// Slot isolation under batching: with the shared tail scratch, each slot's
/// tail prep must be consumed before the next slot's prep overwrites it — a
/// prep-all-then-consume-all reorder makes earlier slots attend with the last
/// slot's K/V. Exact bs-vs-single equality is NOT promisable (cuBLAS picks
/// different algorithms per M, so bs=2 and bs=1 can legitimately differ in
/// the last ulp), but slot ISOLATION is exact: run the same slot-A inputs in
/// two bs=2 calls that differ only in slot B's content — the batch shape is
/// identical, so slot A's math is bit-identical unless B's data leaks in.
#[test]
#[ignore = "GPU isolation test — needs a CUDA device with ~11 GB free VRAM"]
fn dspark_propose_batched_slot_isolation() -> Result<()> {
    type SlotDrafts = Vec<[u32; GLM52_DSPARK_DRAFTS]>;
    const ROUNDS: usize = 6;
    let _env = ENV_LOCK.lock().unwrap();
    let ctx = DeviceContext::new()?;
    let cache_len = dspark_cache_len(4096);
    let mut model = Glm52DsparkModel::synthetic(&ctx, cache_len)?;
    model.randomize_for_test(&ctx)?;
    let rand_head = |seed: u32| -> Result<DeviceMatrix> {
        let mut state = seed | 1;
        let host: Vec<half::bf16> = (0..GLM52_VOCAB * GLM52_HIDDEN)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                half::bf16::from_f32(((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02)
            })
            .collect();
        Ok(DeviceMatrix {
            data: ctx.stream.clone_htod(&host)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        })
    };
    let embed = rand_head(11)?;
    let lm_head = rand_head(23)?;
    let captured_row = |salt: u32| -> Result<cudarc::driver::CudaSlice<half::bf16>> {
        let host: Vec<half::bf16> = (0..GLM52_DSPARK_CONTEXT_DIM)
            .map(|i| half::bf16::from_f32((((i as u32 + salt) % 97) as f32 - 48.0) * 0.001))
            .collect();
        Ok(ctx.stream.clone_htod(&host)?)
    };
    let cap_a = captured_row(3)?;

    let run = |cap_b_salt: u32, anchor_b_base: u32| -> Result<(SlotDrafts, SlotDrafts)> {
        let cap_b = captured_row(cap_b_salt)?;
        let mut scratch = Glm52DsparkScratch::new(&ctx, cache_len)?;
        let mut a = Glm52DsparkSlotState::new(&ctx, cache_len)?;
        let mut b = Glm52DsparkSlotState::new(&ctx, cache_len)?;
        let mut drafts_a = Vec::new();
        let mut drafts_b = Vec::new();
        for round in 0..ROUNDS {
            a.append_captured_row(&ctx, &cap_a, 0)?;
            b.append_captured_row(&ctx, &cap_b, 0)?;
            let anchors = vec![
                (11 + round as u32 * 7, round + 1),
                (anchor_b_base + round as u32 * 5, round + 1),
            ];
            let mut refs = vec![&mut a, &mut b];
            let drafts =
                model.propose(&ctx, &embed, &lm_head, &mut refs, &anchors, &mut scratch)?;
            drafts_a.push(drafts[0]);
            drafts_b.push(drafts[1]);
        }
        Ok((drafts_a, drafts_b))
    };

    let (a1, b1) = run(41, 900)?;
    let (a2, b2) = run(67, 5000)?;
    assert_eq!(
        a1, a2,
        "slot A's drafts changed when only slot B's content did"
    );
    assert_ne!(
        b1, b2,
        "slot B's inputs changed but its drafts did not (degenerate)"
    );
    assert!(
        a1.iter().flatten().any(|&t| t != 0),
        "isolation ran on degenerate logits"
    );
    Ok(())
}
