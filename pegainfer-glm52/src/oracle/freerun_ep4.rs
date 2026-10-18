//! Free-running DP go/no-go gates (docs/models/glm52/free-running-dp.md §8).
//!
//! The free-running design removes the coordinator's per-step agreements:
//! every rank enters each MoE collective with its OWN token count and the
//! conservative protocol-max `global_tokens` bound instead of a globally
//! negotiated bucket. These gates prove the kernel chain supports that on a
//! 4-GPU node (one GB300 NVL72 tray) before any architecture code is written:
//!
//! 1. [`freerun_hetero_traffic_gate`] — rank 0 replays the layer-6 oracle
//!    walk while ranks 1..3 push per-position varying token counts (including
//!    token-less positions) through the same collectives. Rank 0's output
//!    must stay on the oracle probes AND be bit-identical to a quiet-fleet
//!    pass — cross-rank traffic must not perturb another rank's rows.
//! 2. [`freerun_hetero_graph_gate`] — each rank captures the routed chain in
//!    its own CUDA graph at a DIFFERENT token count (1/2/4/8) and replays it;
//!    every replay must be bit-identical to that rank's eager reference. This
//!    is the "different ranks replay different buckets" claim.
//! 3. [`freerun_bound_tax_probe`] — measures the routed chain at the tight
//!    per-step-agreed `global_tokens` vs the free-running protocol max.
//!    The delta × 75 MoE layers is the per-step tax of deleting the global
//!    bucket agreement; the number feeds the design's go/no-go, so this
//!    probe prints measurements and never asserts.
//!
//! Run each gate in its OWN `cargo test` process: the DeepEP context is
//! once-per-process on this shim (a second `ctx_create` after a destroy
//! hits an NVLink barrier timeout — process exit is the release mechanism).
//! Results 2026-07-30 on GB300 tray03 (all GO) are recorded in
//! `docs/models/glm52/free-running-dp.md` §8.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_kernels::ops::Glm52Ep4DeepEpAbi;
use pegainfer_kernels::ops::glm52_ep_deepep_unique_id;
use pegainfer_kernels::tensor::DeviceContext;

use super::layer::GateLayerMlp;
use super::layer::LayerTensors;
use super::layer::MOE_ORACLE_CTX;
use super::layer::MOE_ORACLE_DEEPGEMM_LAYER_PROBES;
use super::layer::MOE_ORACLE_DEEPGEMM_LAYER_TOL;
use super::layer::MOE_ORACLE_HIDDEN_DIGEST;
use super::layer::MOE_ORACLE_INPUT_SCALE;
use super::layer::MOE_ORACLE_LAYER;
use super::layer::MOE_ORACLE_SEED;
use super::layer::assert_layer_probes;
use super::layer::checked_hidden;
use super::layer::load_decoder_layer;
use super::layer::load_rank_expert_bank;
use super::layer::model_path;
use super::layer::seeded_hidden;
use super::layer_ep4::run_layer_prefill_ep4;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::moe_decode::EXPERTS;
use crate::moe_decode::HIDDEN;
use crate::moe_decode::RoutedTopk;
use crate::moe_decode::TOPK;
use crate::moe_ep::Glm52MoeEpRankState;
use crate::moe_ep::glm52_moe_ep_routed_forward;

const EP_RANKS: usize = 4;
/// The free-running conservative bound: every rank always passes the
/// protocol max instead of a per-step-agreed bucket value.
const PROTOCOL_MAX: usize = EP_RANKS * GLM52_MAX_BATCH_PER_RANK;

/// Synthetic dispatch traffic for a side rank: seeded hidden rows plus a
/// hand-built valid top-k route (distinct global expert ids per row, uniform
/// weights). The VALUES are irrelevant — no gate reads a side rank's combined
/// output — but every id must be in range and every weight finite, because
/// these bytes go over the wire into other ranks' expert segments.
struct SyntheticTraffic {
    hidden: CudaSlice<bf16>,
    route: RoutedTopk,
}

impl SyntheticTraffic {
    fn new(ctx: &DeviceContext, rank: usize, rows: usize) -> Result<Self> {
        let hidden_host = seeded_hidden(
            0xF3EE_0000 + rank as u64,
            rows * HIDDEN,
            MOE_ORACLE_INPUT_SCALE,
        );
        let mut hidden = ctx.stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        ctx.stream.memcpy_htod(&hidden_host, &mut hidden)?;
        let mut idx_host = vec![0i32; rows * TOPK];
        let mut weight_host = vec![0f32; rows * TOPK];
        for row in 0..rows {
            for k in 0..TOPK {
                // k*17 is distinct for k in 0..8, so each row picks 8 distinct
                // global experts; rank/row offsets spread the load.
                idx_host[row * TOPK + k] = ((rank * 89 + row * 31 + k * 17) % EXPERTS) as i32;
                weight_host[row * TOPK + k] = 1.0 / TOPK as f32;
            }
        }
        let mut topk_idx = ctx.stream.alloc_zeros::<i32>(rows * TOPK)?;
        ctx.stream.memcpy_htod(&idx_host, &mut topk_idx)?;
        let mut topk_weight = ctx.stream.alloc_zeros::<f32>(rows * TOPK)?;
        ctx.stream.memcpy_htod(&weight_host, &mut topk_weight)?;
        Ok(Self {
            hidden,
            route: RoutedTopk {
                topk_idx,
                topk_weight,
            },
        })
    }
}

/// A side rank's per-position token count in the heterogeneous pass:
/// `0..=GLM52_MAX_BATCH_PER_RANK` cycling with a per-rank phase, where 0
/// means a token-less collective entry (`token: None`) — the fleet mixes
/// dispatching and quiet ranks at every position.
fn side_tokens(position: usize, rank: usize) -> usize {
    (position + rank * 3) % (GLM52_MAX_BATCH_PER_RANK + 1)
}

fn assert_bitwise_equal_f32(label: &str, quiet: &[f32], hetero: &[f32]) {
    assert_eq!(quiet.len(), hetero.len(), "{label}: length mismatch");
    let mut diffs = 0usize;
    let mut max_abs = 0f32;
    let mut first = None;
    for (i, (a, b)) in quiet.iter().zip(hetero).enumerate() {
        if a.to_bits() != b.to_bits() {
            diffs += 1;
            max_abs = max_abs.max((a - b).abs());
            first.get_or_insert(i);
        }
    }
    assert!(
        diffs == 0,
        "{label}: {diffs}/{} values differ (first at {first:?}, max |delta| {max_abs:.6e}) — \
         cross-rank traffic perturbed this rank's rows",
        quiet.len(),
    );
}

/// Gate 1: cross-rank traffic invariance under per-rank token counts.
///
/// Two passes over the same DeepEP contexts, both with the conservative
/// `PROTOCOL_MAX` bound. Pass A: side ranks enter every collective token-less
/// (the homogeneous oracle-gate shape). Pass B: side ranks push varying row
/// counts (0..=8, per-position, per-rank phase). Rank 0 walks the identical
/// layer-6 oracle input both times; its outputs must hit the oracle probes
/// AND be bit-identical across the passes — the free-running claim that a
/// rank's rows are computed independently of everyone else's traffic.
#[test]
#[ignore = "requires 4 GPUs + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn freerun_hetero_traffic_gate() -> Result<()> {
    let hidden_host = checked_hidden(
        MOE_ORACLE_SEED,
        MOE_ORACLE_CTX,
        MOE_ORACLE_INPUT_SCALE,
        MOE_ORACLE_HIDDEN_DIGEST,
    )?;
    let unique_id = glm52_ep_deepep_unique_id(EP_RANKS)?;
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);

    let handles: Vec<_> = (1..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("freerun-traffic-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, MOE_ORACLE_LAYER, rank, EP_RANKS)?;
                    let mut ep = Glm52MoeEpRankState::<Glm52Ep4DeepEpAbi>::new(
                        &ctx, &unique_id, EP_RANKS, rank,
                    )?;
                    // Pass A: quiet fleet — token-less entries only.
                    for _position in 0..MOE_ORACLE_CTX {
                        let dispatched =
                            glm52_moe_ep_routed_forward(&ctx, &mut ep, &bank, None, PROTOCOL_MAX)?;
                        ensure!(!dispatched, "token-less rank produced a combined output");
                    }
                    // Pass B: heterogeneous traffic.
                    let traffic = SyntheticTraffic::new(&ctx, rank, GLM52_MAX_BATCH_PER_RANK)?;
                    for position in 0..MOE_ORACLE_CTX {
                        let tokens = side_tokens(position, rank);
                        let token =
                            (tokens > 0).then_some((&traffic.hidden, &traffic.route, tokens));
                        let dispatched =
                            glm52_moe_ep_routed_forward(&ctx, &mut ep, &bank, token, PROTOCOL_MAX)?;
                        ensure!(dispatched == (tokens > 0), "dispatch/return mismatch");
                    }
                    Ok(())
                })
                .expect("spawn freerun traffic rank thread")
        })
        .collect();

    let ctx = DeviceContext::new_with_device(0)?;
    let w = load_decoder_layer(
        &ctx,
        &model_path(),
        MOE_ORACLE_LAYER,
        GateLayerMlp::MoeEp4Rank0,
    )?;
    let mut ep = Glm52MoeEpRankState::<Glm52Ep4DeepEpAbi>::new(&ctx, &unique_id, EP_RANKS, 0)?;
    let quiet = run_layer_prefill_ep4(
        &ctx,
        &w,
        &mut ep,
        &hidden_host,
        MOE_ORACLE_CTX,
        PROTOCOL_MAX,
    );
    let hetero = quiet.is_ok().then(|| {
        run_layer_prefill_ep4(
            &ctx,
            &w,
            &mut ep,
            &hidden_host,
            MOE_ORACLE_CTX,
            PROTOCOL_MAX,
        )
    });

    // The DeepEP context drop is collective: rank 0 must drop BEFORE joining
    // the side threads (see the EP8 gate).
    drop(ep);
    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("freerun traffic rank thread panicked")
            .with_context(|| format!("freerun traffic rank {}", rank + 1))?;
    }
    let quiet = quiet?;
    let hetero = hetero.context("rank 0 quiet pass failed before the hetero pass ran")??;

    assert_layer_probes(
        "layer6/moe/ep4/freerun-quiet",
        &quiet,
        MOE_ORACLE_DEEPGEMM_LAYER_PROBES,
        MOE_ORACLE_DEEPGEMM_LAYER_TOL,
        4,
    );
    assert_layer_probes(
        "layer6/moe/ep4/freerun-hetero",
        &hetero,
        MOE_ORACLE_DEEPGEMM_LAYER_PROBES,
        MOE_ORACLE_DEEPGEMM_LAYER_TOL,
        4,
    );
    assert_bitwise_equal_f32("layer6/moe/ep4/freerun-invariance", &quiet, &hetero);
    Ok(())
}

/// Gate 2: per-rank CUDA graphs at DIFFERENT token counts.
///
/// Each rank captures the routed chain (dispatch → tiles → GEMMs → combine)
/// in its own whole-chain graph at a rank-specific token count (1/2/4/8) and
/// the conservative bound, then replays it 16×. Every replay's combined rows
/// must be bit-identical to that rank's eager reference — proving the
/// collective has no cross-rank shape assumption that graph replay would
/// violate when ranks stop agreeing on a bucket.
#[test]
#[ignore = "requires 4 GPUs + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn freerun_hetero_graph_gate() -> Result<()> {
    const REPLAYS: usize = 16;
    let unique_id = glm52_ep_deepep_unique_id(EP_RANKS)?;
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);

    let handles: Vec<_> = (0..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("freerun-graph-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, MOE_ORACLE_LAYER, rank, EP_RANKS)?;
                    let mut ep = Glm52MoeEpRankState::<Glm52Ep4DeepEpAbi>::new(
                        &ctx, &unique_id, EP_RANKS, rank,
                    )?;
                    let tokens = 1 << rank; // 1, 2, 4, 8
                    let traffic = SyntheticTraffic::new(&ctx, rank, tokens)?;

                    // Eager reference: one collective execution.
                    let dispatched = glm52_moe_ep_routed_forward(
                        &ctx,
                        &mut ep,
                        &bank,
                        Some((&traffic.hidden, &traffic.route, tokens)),
                        PROTOCOL_MAX,
                    )?;
                    ensure!(dispatched, "eager reference produced no combined output");
                    ctx.stream.synchronize()?;
                    let eager = ctx.stream.clone_dtoh(ep.combined())?;
                    let eager = &eager[..tokens * HIDDEN];

                    // Capture + launch, then pure replays: run_or_capture
                    // executes the chain exactly once per call, so every rank
                    // performs 1 + REPLAYS paired collective executions.
                    let mut graph = CudaGraphState::new();
                    for replay in 0..=REPLAYS {
                        graph.run_or_capture(&ctx, || {
                            glm52_moe_ep_routed_forward(
                                &ctx,
                                &mut ep,
                                &bank,
                                Some((&traffic.hidden, &traffic.route, tokens)),
                                PROTOCOL_MAX,
                            )
                            .map(|_| ())
                        })?;
                        ctx.stream.synchronize()?;
                        let out = ctx.stream.clone_dtoh(ep.combined())?;
                        for (i, (a, b)) in eager.iter().zip(&out[..tokens * HIDDEN]).enumerate() {
                            ensure!(
                                a.to_bits() == b.to_bits(),
                                "rank {rank} replay {replay}: combined[{i}] {} != eager {} — \
                                 graph replay diverged from the eager chain",
                                b.to_f32(),
                                a.to_f32(),
                            );
                        }
                    }
                    Ok(())
                })
                .expect("spawn freerun graph rank thread")
        })
        .collect();

    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("freerun graph rank thread panicked")
            .with_context(|| format!("freerun graph rank {rank}"))?;
    }
    Ok(())
}

/// Probe 3: the conservative-bound tax, measured.
///
/// Every rank dispatches ONE decode token per collective (the steady-decode
/// shape that dominates TPOT) and the chain runs back-to-back at two bounds:
/// the tight per-step-agreed value (`EP_RANKS`, the retired coordinator's
/// bucket-1 agreement) and the free-running `PROTOCOL_MAX`. The per-call delta × 75
/// MoE layers is the step tax of deleting the bucket agreement. Prints
/// measurements; the go/no-go judgement lives in the design doc, not here.
#[test]
#[ignore = "requires 4 GPUs + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn freerun_bound_tax_probe() -> Result<()> {
    const WARMUP: usize = 32;
    const ITERS: usize = 256;
    let unique_id = glm52_ep_deepep_unique_id(EP_RANKS)?;
    let tensors = Arc::new(LayerTensors::load(&model_path(), MOE_ORACLE_LAYER)?);

    let handles: Vec<_> = (0..EP_RANKS)
        .map(|rank| {
            let tensors = Arc::clone(&tensors);
            std::thread::Builder::new()
                .name(format!("freerun-tax-rank-{rank}"))
                .spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let bank =
                        load_rank_expert_bank(&ctx, &tensors, MOE_ORACLE_LAYER, rank, EP_RANKS)?;
                    let mut ep = Glm52MoeEpRankState::<Glm52Ep4DeepEpAbi>::new(
                        &ctx, &unique_id, EP_RANKS, rank,
                    )?;
                    let traffic = SyntheticTraffic::new(&ctx, rank, 1)?;
                    for (label, global_tokens) in
                        [("tight", EP_RANKS), ("protocol-max", PROTOCOL_MAX)]
                    {
                        for _ in 0..WARMUP {
                            glm52_moe_ep_routed_forward(
                                &ctx,
                                &mut ep,
                                &bank,
                                Some((&traffic.hidden, &traffic.route, 1)),
                                global_tokens,
                            )?;
                        }
                        ctx.stream.synchronize()?;
                        let start = Instant::now();
                        for _ in 0..ITERS {
                            glm52_moe_ep_routed_forward(
                                &ctx,
                                &mut ep,
                                &bank,
                                Some((&traffic.hidden, &traffic.route, 1)),
                                global_tokens,
                            )?;
                        }
                        ctx.stream.synchronize()?;
                        let per_call = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                        println!(
                            "freerun-bound-tax rank {rank} {label} (global_tokens={global_tokens}): \
                             {per_call:.1} us/layer-call, x75 layers = {:.2} ms/step",
                            per_call * 75.0 / 1e3,
                        );
                    }
                    Ok(())
                })
                .expect("spawn freerun tax rank thread")
        })
        .collect();

    for (rank, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .expect("freerun tax rank thread panicked")
            .with_context(|| format!("freerun tax rank {rank}"))?;
    }
    Ok(())
}
