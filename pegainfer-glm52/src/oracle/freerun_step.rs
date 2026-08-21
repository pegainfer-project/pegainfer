//! Free-running DP whole-step gates 4–5
//! (`docs/models/glm52/free-running-dp.md` §8).
//!
//! Gate 1–3 (in `freerun_ep4.rs`) proved the isolated kernel chain; these
//! two probe the PRODUCTION step and MTP-round paths through a real EP4
//! engine launch on one GB300 tray:
//!
//! 4. [`freerun_padding_byte_constancy_gate`] — while one rank decodes and
//!    three ranks idle at the all-padding bucket-1 step, every idle rank's
//!    last routed layer's `topk_idx`/`topk_weight` bytes must be constant
//!    step over step (≥ 64 steps). Padding rows' routing goes over the
//!    DeepEP wire, so "constructively deterministic" is a protocol claim —
//!    this covers the indexer at seq_len=1 and the fp8 quant path, the two
//!    links the design flagged as unverified.
//! 5. [`freerun_mtp_fixed_chain_gate`] — with native MTP on and per-rank
//!    unequal work (one rank decoding+proposing, three ranks empty), every
//!    fleet round must run the fixed layer-78 chain on every rank, the
//!    decode output must equal the plain-decode trajectory of the
//!    production gate, and the empty ranks' rounds must not stretch a
//!    round by more than 0.5 ms vs the all-busy baseline.
//!
//! Run each gate in its OWN `cargo test` process (DeepEP context is
//! once-per-process — see `freerun_ep4.rs`).

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_sample::SamplingParams;

use super::mtp_production::PATHOLOGICAL_PROMPT;
use crate::Glm52LaunchOptions;
use crate::Glm52MoeTopo;
use crate::freerun_probe;

fn model_path() -> Result<PathBuf> {
    std::env::var_os("PEGAINFER_TEST_MODEL_PATH")
        .map(PathBuf::from)
        .context("PEGAINFER_TEST_MODEL_PATH must point to GLM-5.2-FP8")
}

fn launch_ep4(drafter: crate::Glm52Drafter) -> Result<EngineHandle> {
    crate::launch(
        &model_path()?,
        Glm52LaunchOptions {
            tp_size: 1,
            dp_size: 4,
            drafter,
            max_model_len: Some(4096),
            prefill_only: None,
            no_prefix_cache: true,
            kv_offload: None,
            moe_topo: Glm52MoeTopo::Ep4,
            weight_staging: true,
            dump_graph_png: None,
            ranks: None,
            rendezvous: None,
        },
    )
}

/// Run one request pinned to `rank` and return its completion.
/// `sampled` uses a seeded temperature-0.7 request, which blocks the
/// launch-ahead lease (see gate 4 — leased padding rows self-feed by
/// design, so byte constancy is a full-prologue claim).
fn run_request(
    engine: &EngineHandle,
    rank: usize,
    prompt_len: usize,
    max_tokens: usize,
    sampled: bool,
) -> Result<Vec<u32>> {
    let params = if sampled {
        SamplingParams {
            temperature: 0.7,
            seed: Some(0x5eed_f7ee),
            ignore_eos: true,
            ..SamplingParams::default()
        }
    } else {
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        }
    };
    let (token_tx, mut token_rx) = TokenSink::standalone();
    engine.submit(GenerateRequest {
        request_id: Some(format!("freerun-step-gate-rank-{rank}")),
        queued_at_unix_s: None,
        trace_parent: None,
        data_parallel_rank: Some(rank),
        prompt_tokens: PATHOLOGICAL_PROMPT[..prompt_len].to_vec(),
        params,
        stop_policy: StopPolicy {
            eos: EosPolicy::Ignore,
            token_ids: Vec::new(),
        },
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx,
        logprobs: 0,
        echo: false,
    })?;
    let mut completion = Vec::new();
    loop {
        let (_, event) = token_rx
            .blocking_recv()
            .context("GLM5.2 freerun gate token stream closed")?;
        match event {
            TokenEvent::Token { id, .. } => completion.push(id),
            TokenEvent::Finished {
                completion_tokens, ..
            } => {
                anyhow::ensure!(completion_tokens == max_tokens);
                break;
            }
            TokenEvent::Error { message, .. } | TokenEvent::Rejected { message, .. } => {
                anyhow::bail!("GLM5.2 freerun gate request failed: {message}")
            }
            TokenEvent::Scheduled { .. }
            | TokenEvent::PromptTokens { .. }
            | TokenEvent::KvTransfer { .. } => {}
        }
    }
    Ok(completion)
}

/// Gate 4: idle ranks' padding routing bytes are constant step over step.
///
/// Rank 0 decodes 96 tokens; ranks 1–3 never see a request, so each of
/// their ~96 steps is the all-padding bucket-1 step. The request is
/// SAMPLED, which withholds the launch-ahead lease — every step runs the
/// full prologue, whose padding contract (`GLM52_PADDING_STEP`: token 0,
/// position 0, padding page) is the claim under test. (Leased replays
/// deliberately let padding rows self-feed; their wire bytes evolve by
/// design and are covered by the lease invariants, not this gate.)
/// For every idle rank, group its recorded route snapshots by bucket and
/// assert bit-identity within each group — the contract upgraded from
/// "happens to be right" to a gated invariant.
#[test]
#[ignore = "requires 4 GPUs + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn freerun_padding_byte_constancy_gate() -> Result<()> {
    const MIN_PADDING_STEPS: usize = 64;
    let engine = launch_ep4(crate::Glm52Drafter::None)?;
    freerun_probe::set_enabled(true);
    let completion = run_request(&engine, 0, PATHOLOGICAL_PROMPT.len(), 96, true)?;
    freerun_probe::set_enabled(false);
    drop(engine);
    assert_eq!(completion.len(), 96);

    let records = freerun_probe::take_step_routes();
    for rank in 1..4 {
        // Idle ranks run only padding steps; group by bucket (prefill spans
        // on rank 0 no longer lift idle ranks' buckets, but the pre-capture
        // and any transient shapes may differ — the claim is per shape).
        let mut per_bucket: std::collections::BTreeMap<usize, Vec<(&Vec<i32>, &Vec<u32>)>> =
            std::collections::BTreeMap::new();
        for record in records.iter().filter(|record| record.rank == rank) {
            assert_eq!(
                record.active_rows, 0,
                "rank {rank} was expected to stay idle for the whole gate"
            );
            per_bucket
                .entry(record.bucket)
                .or_default()
                .push((&record.topk_idx, &record.topk_weight_bits));
        }
        let padding_steps: usize = per_bucket.values().map(Vec::len).sum();
        assert!(
            padding_steps >= MIN_PADDING_STEPS,
            "rank {rank} recorded only {padding_steps} padding steps (< {MIN_PADDING_STEPS})"
        );
        for (bucket, group) in per_bucket {
            let (first_idx, first_weight) = group[0];
            for (step, (idx, weight)) in group.iter().enumerate().skip(1) {
                assert_eq!(
                    (*idx, *weight),
                    (first_idx, first_weight),
                    "rank {rank} bucket {bucket} padding routing bytes drifted at \
                     recorded step {step} — padding inputs are not constructively \
                     deterministic"
                );
            }
        }
    }
    println!(
        "freerun-padding-constancy: {} route snapshots, idle ranks byte-stable",
        records.len()
    );
    Ok(())
}

/// Gate 5: the fixed MTP chain under per-rank unequal work.
///
/// Phase A (heterogeneous): rank 0 decodes 96 tokens with native MTP while
/// ranks 1–3 are empty — every fleet round runs the fixed layer-78 chain
/// with three all-padding ranks. Phase B (all-busy baseline): rank 0 runs
/// the IDENTICAL request while ranks 1–3 decode their own. Correctness:
/// rank 0's two completions must be identical — empty-rank padding rounds
/// must not perturb a proposing rank (the whole-step analog of gate 1's
/// traffic invariance; greedy MTP is lossless, so the trajectories compare
/// exactly). Performance: the per-round wall-time delta (hetero − busy) is
/// the empty-round overhead; accept at ≤ 0.5 ms.
#[test]
#[ignore = "requires 4 GPUs + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn freerun_mtp_fixed_chain_gate() -> Result<()> {
    let engine = launch_ep4(crate::Glm52Drafter::NativeMtp)?;

    // Phase A: heterogeneous (1 proposing rank, 3 empty ranks).
    freerun_probe::set_enabled(true);
    let hetero_completion = run_request(&engine, 0, PATHOLOGICAL_PROMPT.len(), 96, false)?;
    freerun_probe::set_enabled(false);
    let hetero = freerun_probe::take_mtp_rounds();
    assert!(!hetero.is_empty(), "no MTP rounds recorded in phase A");
    assert!(
        hetero
            .iter()
            .filter(|round| round.rank != 0)
            .all(|round| round.empty),
        "phase A expected ranks 1-3 empty in every round"
    );
    assert!(
        hetero.iter().any(|round| round.rank == 0 && !round.empty),
        "phase A expected rank 0 to enter its rounds with proposals"
    );

    // Phase B: all-busy baseline; rank 0 repeats the identical request
    // (prefix cache is off under native MTP, so it recomputes fully).
    freerun_probe::set_enabled(true);
    let handles: Vec<std::thread::JoinHandle<Result<Vec<u32>>>> = (0..4)
        .map(|rank| {
            let engine = engine.clone();
            let prompt_len = if rank == 0 {
                PATHOLOGICAL_PROMPT.len()
            } else {
                96
            };
            std::thread::spawn(move || run_request(&engine, rank, prompt_len, 96, false))
        })
        .collect();
    let mut busy_completions = Vec::with_capacity(4);
    for handle in handles {
        busy_completions.push(handle.join().expect("freerun gate request thread")?);
    }
    freerun_probe::set_enabled(false);
    let busy = freerun_probe::take_mtp_rounds();
    drop(engine);
    assert!(!busy.is_empty(), "no MTP rounds recorded in phase B");
    assert_eq!(
        busy_completions[0], hetero_completion,
        "rank 0's trajectory changed between empty-fleet and busy-fleet MTP rounds"
    );

    let mean_ms = |rounds: &[freerun_probe::MtpRoundRecord]| {
        rounds
            .iter()
            .map(|round| round.elapsed.as_secs_f64() * 1e3)
            .sum::<f64>()
            / rounds.len() as f64
    };
    // Compare the steady tail (skip warmup/capture rounds).
    let hetero_ms = mean_ms(&hetero[hetero.len() / 2..]);
    let busy_ms = mean_ms(&busy[busy.len() / 2..]);
    println!(
        "freerun-mtp-fixed-chain: hetero {hetero_ms:.3} ms/round ({} rounds), \
         busy {busy_ms:.3} ms/round ({} rounds), empty-round delta {:+.3} ms",
        hetero.len(),
        busy.len(),
        hetero_ms - busy_ms,
    );
    assert!(
        hetero_ms - busy_ms <= 0.5,
        "empty-rank MTP rounds cost {:.3} ms over the all-busy baseline (accept <= 0.5 ms)",
        hetero_ms - busy_ms,
    );
    Ok(())
}
