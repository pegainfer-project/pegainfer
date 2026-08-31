//! Decode-GEMM batch-invariance under CUDA-graph, per-policy isolated.
//!
//! The decode graph is captured once per (bucket, attention_path) and replayed; the cache key
//! excludes numeric_policy and replay skips the kernel closure, so each policy needs its own fresh
//! executor with the policy set before build. A (row 0) fixed; filler count varies A's bucket → its
//! decode-GEMM N (Tuned: a plan per bucket = drift; Pin: one algo = invariant).
//!
//!   PEGAINFER_TEST_MODEL_PATH=<Qwen3-4B-base> cargo test --release \
//!     -p pegainfer-qwen3 --test batch_invariance_decode_gemm_graph -- --nocapture

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::per_token_served;
use pegainfer_kernels::ops::pin_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4;
const SHORT_LEN: usize = 8; // short decode context; path now depends on bucket, not length (<=32 SplitKv, else NonPartition)

// (label, real_small, real_large). bucket_for: 1->1, 3->4, 7->8, 15->16.
// A (row 0) identical in all; only the bucket A pads into changes (decode GEMM N).
const PAIRS: [(&str, usize, usize); 6] = [
    ("ident_b8v8  ", 7, 7), // sanity: identical batch -> must be equal under every policy
    ("b1v8(1,7)   ", 1, 7), // bucket 1 vs 8
    ("b4v8(3,7)   ", 3, 7), // bucket 4 vs 8
    ("b4v16(3,15) ", 3, 15), // bucket 4 vs 16
    // Large buckets: the pin keys on {M,K} only, so A's algo must stay identical across two LARGE
    // decode buckets — where a capture/replay layout-lifetime bug would surface.
    ("b40v72     ", 40, 71),   // buckets 40 vs 72
    ("b200v256   ", 200, 255), // top buckets 200 vs 256
];

fn model_path_or_skip() -> Option<String> {
    let Ok(p) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3 batch_invariance_decode_gemm_graph: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base"
        );
        return None;
    };
    Some(p)
}

fn pitem(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        LOGPROBS,
        false,
    )
}

fn short_prompt(seed: u32) -> Vec<u32> {
    (0..SHORT_LEN as u32)
        .map(|i| 1000 + (seed * 131 + i * 7) % 50000)
        .collect()
}

/// A (request 0)'s `(prefill first_token, one-step decode top-K)` when co-batched with
/// `n_requests - 1` fillers. first_token lets the caller confirm prefill is unchanged.
fn a_first_and_decode(ex: &mut Qwen3Executor, n_requests: usize) -> (u32, Vec<(u32, f32)>) {
    let batch: Vec<(RequestId, Vec<u32>)> = (0..n_requests)
        .map(|i| (RequestId::new(1 + i as u64), short_prompt(i as u32)))
        .collect();
    let pitems: Vec<PrefillStepItem> = batch.iter().map(|(id, p)| pitem(*id, p.clone())).collect();
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &pitems,
            echo: false,
        })
        .expect("prefill");
    let a_first = pr.requests[0].first_token;
    let ditems: Vec<DecodeStepItem> = batch
        .iter()
        .zip(&pr.requests)
        .map(|((id, _), req)| {
            DecodeStepItem::new(*id, req.first_token, SamplingParams::default(), LOGPROBS)
        })
        .collect();
    let dr = ex
        .execute_decode(DecodePlan {
            sample_seed: 0,
            requests: &ditems,
        })
        .expect("decode");
    let topk = dr.requests[0]
        .logprob
        .as_ref()
        .expect("logprobs requested but none returned")
        .top_logprobs
        .clone();
    for (id, _) in &batch {
        ex.drop_request(*id).expect("drop request");
    }
    (a_first, topk)
}

fn run_policy(policy: NumericPolicy, model_path: &str) -> (Vec<(bool, bool, u64)>, Qwen3Executor) {
    set_numeric_policy(policy);
    let mut ex = Qwen3Executor::from_runtime(model_path, true, &[0]).expect("build executor");
    ex.set_prefix_cache_enabled(false);
    let mut out = Vec::new();
    for (_, small, large) in PAIRS {
        reset_numeric_policy_counters();
        let (ft_s, tk_s) = a_first_and_decode(&mut ex, small);
        let (ft_l, tk_l) = a_first_and_decode(&mut ex, large);
        let served = pin_served();
        let ft_eq = ft_s == ft_l;
        let tk_eq = tk_s == tk_l;
        // served counts the CAPTURE step only (replay skips the closure). Under baseline/per_token,
        // launch_gemm_pin is never called → served is structurally 0 (N/A).
        out.push((ft_eq, tk_eq, served));
    }
    (out, ex)
}

fn decode_batch(ex: &mut Qwen3Executor, ids: &[RequestId], tokens: &[u32]) -> Vec<u32> {
    let ditems: Vec<DecodeStepItem> = ids
        .iter()
        .zip(tokens.iter().copied())
        .map(|(id, token)| DecodeStepItem::new(*id, token, SamplingParams::default(), 0))
        .collect();

    ex.execute_decode(DecodePlan {
        sample_seed: 0,
        requests: &ditems,
    })
    .expect("decode for graph-mode probe")
    .requests
    .into_iter()
    .map(|request| request.token)
    .collect()
}

/// A graph replay does not invoke the GEMM closure again.
/// An eager decode invokes the PerToken GEMM path on every step.
fn per_token_counter_probe(ex: &mut Qwen3Executor, n_requests: usize) -> (u64, u64) {
    let batch: Vec<(RequestId, Vec<u32>)> = (0..n_requests)
        .map(|i| (RequestId::new(10_000 + i as u64), short_prompt(i as u32)))
        .collect();

    let pitems: Vec<PrefillStepItem> = batch.iter().map(|(id, p)| pitem(*id, p.clone())).collect();

    let prefill = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &pitems,
            echo: false,
        })
        .expect("prefill for graph-mode probe");

    let ids: Vec<RequestId> = batch.iter().map(|(id, _)| *id).collect();

    let mut tokens: Vec<u32> = prefill
        .requests
        .iter()
        .map(|request| request.first_token)
        .collect();

    // Clear the counters from the prefill phase and observe only the next two decode steps.
    reset_numeric_policy_counters();

    tokens = decode_batch(ex, &ids, &tokens);
    let after_first = per_token_served();

    let _ = decode_batch(ex, &ids, &tokens);
    let after_second = per_token_served();

    for id in ids {
        ex.drop_request(id).expect("drop graph-mode probe request");
    }

    (after_first, after_second)
}

/// Check graph replay at bucket 32 and eager execution at bucket 40
fn assert_pertoken_graph_cap_behavior(ex: &mut Qwen3Executor) {
    let (graph_first, graph_second) = per_token_counter_probe(ex, 32);
    assert!(graph_first > 0, "PerToken graph probe served no GEMM calls");
    assert_eq!(
        graph_second, graph_first,
        "PerToken bucket 32 ran the GEMM closure again; expected graph replay"
    );

    let (eager_first, eager_second) = per_token_counter_probe(ex, 33);

    assert!(eager_first > 0, "PerToken eager probe served no GEMM calls");
    assert!(
        eager_second > eager_first,
        "PerToken batch 33 did not execute eager GEMMs twice: \
        first={eager_first}, second={eager_second}"
    );
}

#[test]
fn batch_invariance_decode_gemm_graph() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let baseline = {
        let (result, executor) = run_policy(NumericPolicy::Tuned, &model_path);
        drop(executor);
        result
    };
    let pin = {
        let (result, executor) = run_policy(NumericPolicy::Pin, &model_path);
        drop(executor);
        result
    };
    let (pertoken, mut pertoken_executor) = run_policy(NumericPolicy::PerToken, &model_path);

    // Prefill must be identical for A in every batch/policy, else the decode comparison is
    // prefill-contaminated.
    for (name, rows) in [
        ("baseline", &baseline),
        ("pin", &pin),
        ("per_token", &pertoken),
    ] {
        for (i, (ft_eq, _, _)) in rows.iter().enumerate() {
            assert!(
                *ft_eq,
                "{name}: A's prefill first_token differs across batch on pair {} — decode \
                 comparison is prefill-contaminated, not a decode result",
                PAIRS[i].0.trim()
            );
        }
    }

    // Control: baseline must drift on at least one pair (else not reproduced here).
    let baseline_drifted = baseline.iter().any(|(_, tk_eq, _)| !tk_eq);
    assert!(
        baseline_drifted,
        "baseline did NOT drift on any pair — decode-GEMM coupling not reproduced; \
         the pin proof would be vacuous"
    );

    // Pin: every pair invariant, served>0 (pin ran at capture).
    for (i, (_, tk_eq, served)) in pin.iter().enumerate() {
        assert!(
            *tk_eq,
            "PIN: A's graph decode top-K changed on pair {} — pin did NOT make graph decode \
             batch-invariant (graph captured under Pin)",
            PAIRS[i].0.trim()
        );
        assert!(
            *served > 0,
            "PIN: pin did not run on pair {} (served=0) — vacuous",
            PAIRS[i].0.trim()
        );
    }
    for (i, (_, tk_eq, _)) in pertoken.iter().enumerate() {
        assert!(
            *tk_eq,
            "PER_TOKEN: A's graph decode top-K changed on pair {} — harness bug",
            PAIRS[i].0.trim()
        );
    }

    assert_pertoken_graph_cap_behavior(&mut pertoken_executor);
}
