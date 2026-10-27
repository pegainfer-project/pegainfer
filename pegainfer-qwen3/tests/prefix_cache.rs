//! Behavioral gate for the Qwen3-4B prefix cache.
//!
//! `hf_golden_gate`'s cached-replay surfaces own the accuracy story (warm
//! logits vs the HF golden). This test pins the behavioral contract of the
//! cache itself — what gets matched, what gets recomputed:
//!   * exact cached-token counts: full prompt blocks hit, the partial tail
//!     is always recomputed
//!   * the full-block cap: a prompt whose length divides the block size and
//!     is fully cached still keeps one block of prefill, so the final chunk
//!     can emit a token
//!   * prefix extension: a longer prompt reuses a shorter prompt's blocks
//!   * mixed batch: a cold and a warm request share one prefill plan
//!     (per-request start positions) without corrupting each other
//!   * the unified prefill+decode path matches through the same code
//!
//! Each warm run is also bounded against the same executor's cache-off run:
//! the only legitimate difference is the prefill GEMM shrinking to the
//! uncached suffix, so logits must agree within bf16 reduction-order noise.
//! A wrong RoPE offset, causal-mask offset, or stale page is whole nats off
//! and cannot hide inside these tolerances.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when absent
//! (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::collections::HashMap;
use std::path::Path;

use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;
use pegainfer_qwen3::runtime::UnifiedPlan;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const BLOCK: usize = 16;
const LOGPROBS: usize = 16;
const MAX_OUTPUT: usize = 8;
/// Teacher-forced tokens fed after each prefill, so cold and warm runs are
/// compared on identical inputs at every decode position.
const DECODE_FED: [u32; 4] = [11, 22, 33, 44];

/// Warm-vs-cold self-consistency bounds, following the golden-gate
/// methodology: the warm argmax must sit within `REGRET_TOL` of the cold
/// argmax (structural wrong-token guard), and the *mean* head-token delta —
/// coverage-stable, unlike the absolute max — must stay at the bf16
/// reduction-order floor. Individual tail tokens can swing several ULPs
/// (≈0.6 nat observed on these synthetic prompts), so the per-token max is
/// printed, never asserted.
const REGRET_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 prefix_cache: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Deterministic synthetic prompt; different seeds share no prefix.
fn prompt(seed: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((seed * 100_003 + i * 17) % 50_000 + 1_000) as u32)
        .collect()
}

fn prefill_item(id: u64, prompt: &[u32]) -> PrefillStepItem {
    PrefillStepItem::new(
        RequestId::new(id),
        prompt.to_vec(),
        MAX_OUTPUT,
        SamplingParams::default(),
        Some(LOGPROBS),
        None,
    )
}

fn decode_item(id: u64, fed: u32) -> DecodeStepItem {
    DecodeStepItem::new(
        RequestId::new(id),
        fed,
        SamplingParams::default(),
        Some(LOGPROBS),
    )
}

fn top_logprobs(lp: Option<&TokenLogprob>) -> Vec<(u32, f32)> {
    lp.expect("logprobs requested but none returned")
        .top_logprobs
        .clone()
}

/// Prefill `prompt`, teacher-force `DECODE_FED`, drop the request. Returns
/// the cached-token count and the top-K logprobs at every position.
fn run_one(ex: &mut Qwen3Executor, id: u64, prompt: &[u32]) -> (usize, Vec<Vec<(u32, f32)>>) {
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(id, prompt)],
        })
        .expect("prefill");
    let cached = pr.requests[0].cached_tokens;
    let mut positions = vec![top_logprobs(pr.requests[0].first_token_logprob.as_ref())];
    for fed in DECODE_FED {
        let dr = ex
            .execute_decode(DecodePlan {
                sample_seed: 0,
                requests: &[decode_item(id, fed)],
            })
            .expect("decode");
        positions.push(top_logprobs(dr.requests[0].logprob.as_ref()));
    }
    ex.drop_request(RequestId::new(id)).expect("drop request");
    (cached, positions)
}

/// Warm logits must agree with cold logits up to bf16 reduction-order noise:
/// the warm argmax may sit at most `REGRET_TOL` below the cold argmax (in the
/// cold run's own logprobs), and the mean delta over head tokens common to
/// both top-Ks must stay under `MEAN_TOL`. The worst single delta is printed
/// but not asserted — it is coverage-unstable bf16 tail.
fn assert_close(label: &str, cold: &[Vec<(u32, f32)>], warm: &[Vec<(u32, f32)>]) {
    assert_eq!(cold.len(), warm.len(), "[{label}] position count mismatch");
    let mut deltas = Vec::new();
    for (pos, (c, w)) in cold.iter().zip(warm).enumerate() {
        let cold_map: HashMap<u32, f32> = c.iter().copied().collect();
        let cold_top = c[0].1;
        match cold_map.get(&w[0].0) {
            None => panic!(
                "[{label}] pos {pos}: warm argmax {} absent from cold top-{}",
                w[0].0,
                c.len()
            ),
            Some(&clp) => assert!(
                cold_top - clp <= REGRET_TOL,
                "[{label}] pos {pos}: warm argmax {} sits {:.4} nat below cold argmax",
                w[0].0,
                cold_top - clp
            ),
        }
        for &(token, wlp) in w.iter().take(8) {
            if let Some(&clp) = cold_map.get(&token) {
                deltas.push((wlp - clp).abs());
            }
        }
    }
    assert!(!deltas.is_empty(), "[{label}] no head-token overlap");
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let max = deltas.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "prefix_cache [{label}]: {} head deltas — mean {mean:.4} max {max:.4}",
        deltas.len()
    );
    assert!(
        mean <= MEAN_TOL,
        "[{label}] mean head logprob delta {mean:.4} > {MEAN_TOL} — warm logits drifted past bf16 noise"
    );
}

#[test]
fn prefix_cache_behavior() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let mut ex =
        Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("build eager executor");

    let a = prompt(1, 50); // 3 full blocks + 2-token tail
    let b = [a.clone(), prompt(2, 16)].concat(); // 66 = A extended; 4 full blocks + 2
    let c = prompt(3, 64); // exactly 4 blocks — the cap edge
    let d = prompt(4, 40); // never seen before the mixed batch

    // ── Phase 1: matching and completed-block retention disabled — true cold
    // baselines. Dropping each request resets its registered KV, so enabling
    // the cache later must start cold rather than resurrecting these blocks.
    ex.set_prefix_cache_enabled(false);
    let (cached, a_cold) = run_one(&mut ex, 1, &a);
    assert_eq!(
        cached, 0,
        "matching disabled must report zero cached tokens"
    );
    let (_, b_cold) = run_one(&mut ex, 2, &b);
    let (_, c_cold) = run_one(&mut ex, 3, &c);

    // ── Phase 2: enable retention and seed the cache explicitly. The first A
    // run must be cold: phase 1's disabled-mode request may not survive.
    ex.set_prefix_cache_enabled(true);
    let (cached, _) = run_one(&mut ex, 4, &a);
    assert_eq!(
        cached, 0,
        "cache-disabled requests must not seed a later enabled cache"
    );
    let _ = run_one(&mut ex, 5, &b);
    let _ = run_one(&mut ex, 6, &c);

    let (cached, a_warm) = run_one(&mut ex, 11, &a);
    assert_eq!(
        cached,
        3 * BLOCK,
        "A: 3 full blocks hit, 2-token tail recomputed"
    );
    assert_close("A warm", &a_cold, &a_warm);

    let (cached, b_warm) = run_one(&mut ex, 12, &b);
    assert_eq!(cached, 4 * BLOCK, "B: A's 3 blocks + B's own 4th block hit");
    assert_close("B warm (extension)", &b_cold, &b_warm);

    let (cached, c_warm) = run_one(&mut ex, 13, &c);
    assert_eq!(
        cached,
        3 * BLOCK,
        "C: all 4 blocks are cached but the cap must keep one block of prefill"
    );
    assert_close("C warm (full-block cap)", &c_cold, &c_warm);

    // ── Mixed batch: cold D and warm A share one prefill plan. Their start
    // positions differ (0 vs 48) inside a single kernel plan; each request's
    // logits must be unaffected by the neighbour.
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(21, &d), prefill_item(22, &a)],
        })
        .expect("mixed prefill");
    assert_eq!(pr.requests[0].cached_tokens, 0, "D is unseen — cold");
    assert_eq!(pr.requests[1].cached_tokens, 3 * BLOCK, "A is warm");
    let d_mixed = vec![top_logprobs(pr.requests[0].first_token_logprob.as_ref())];
    let a_mixed = vec![top_logprobs(pr.requests[1].first_token_logprob.as_ref())];
    assert_close("A in mixed batch", &a_cold[..1], &a_mixed);
    ex.drop_request(RequestId::new(21)).expect("drop");
    ex.drop_request(RequestId::new(22)).expect("drop");

    // D's baseline: a cache-off solo prefill. Differs from the mixed run only
    // by batching — so this bounds cross-request interference from the warm
    // neighbour's nonzero start position.
    ex.set_prefix_cache_enabled(false);
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(31, &d)],
        })
        .expect("D solo prefill");
    let d_solo = vec![top_logprobs(pr.requests[0].first_token_logprob.as_ref())];
    ex.drop_request(RequestId::new(31)).expect("drop");
    assert_close("D cold in mixed batch", &d_solo, &d_mixed);
    ex.set_prefix_cache_enabled(true);

    // Disabling the cache above drained the earlier A/B/C primaries. Seed B
    // again so the unified-path assertions below exercise warm page reuse.
    let (cached, _) = run_one(&mut ex, 32, &b);
    assert_eq!(
        cached, 0,
        "re-enabled cache starts cold after transition drain"
    );

    // ── Unified path: warm prefill of A while another request decodes.
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(41, &b)],
        })
        .expect("B prefill for unified decode");
    assert_eq!(pr.requests[0].cached_tokens, 4 * BLOCK);
    let ur = ex
        .execute_unified(UnifiedPlan {
            sample_seed: 0,
            prefill_requests: &[prefill_item(42, &a)],
            decode_requests: &[decode_item(41, DECODE_FED[0])],
        })
        .expect("unified");
    assert_eq!(
        ur.prefill_requests[0].cached_tokens,
        3 * BLOCK,
        "unified prefill matches through the same path"
    );
    let a_unified = vec![top_logprobs(
        ur.prefill_requests[0].first_token_logprob.as_ref(),
    )];
    assert_close("A in unified plan", &a_cold[..1], &a_unified);
    ex.drop_request(RequestId::new(41)).expect("drop");
    ex.drop_request(RequestId::new(42)).expect("drop");
}
