//! Single-GPU gate: a decode row's output must be bit-identical across co-batched prefill lengths
//! (GEMM-N) and across the pure-decode and unified paths, under Pin and PerToken.
use std::sync::Mutex;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::per_token_served;
use pegainfer_kernels::ops::pin_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeRequestResult;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;
use pegainfer_qwen3::runtime::UnifiedPlan;

// Serialize the #[test]s — they share the process-global numeric policy.
static POLICY_LOCK: Mutex<()> = Mutex::new(());

fn model_path_or_skip() -> Option<String> {
    if let Ok(p) = std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Some(p)
    } else {
        eprintln!("skip batch_invariance_unified: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        None
    }
}

fn prefill_first(ex: &mut Qwen3Executor, id: RequestId, prompt: &[u32]) -> u32 {
    ex.execute_prefill(PrefillPlan {
        requests: &[PrefillStepItem::new(
            id,
            prompt.to_vec(),
            64,
            SamplingParams::default(),
            None,
            None,
        )],
        sample_seed: 0,
    })
    .expect("prefill")
    .requests[0]
        .first_token
}

/// A decode row's output: the sampled token and its top-K as `(id, logprob bits)`. The contract is
/// an ordered bit-equal top-K *and* the same sampled token — comparing logprob magnitudes alone
/// would pass a tie that reorders the top-K, or a ±0.0 flip.
type Row = (u32, Vec<(u32, u32)>);

fn row_of(r: &DecodeRequestResult) -> Row {
    let lp = r
        .logprob
        .as_ref()
        .expect("logprobs requested but none returned");
    assert!(
        !lp.top_logprobs.is_empty(),
        "empty top-K would make the comparison vacuous"
    );
    let topk = lp
        .top_logprobs
        .iter()
        .map(|&(id, lp)| (id, lp.to_bits()))
        .collect();
    (r.token, topk)
}

/// Nonzero proves the policy's own GEMM path ran, so a bit-equal result is not a silent fallback.
fn served_under(policy: NumericPolicy) -> u64 {
    match policy {
        NumericPolicy::Pin => pin_served(),
        NumericPolicy::PerToken => per_token_served(),
        NumericPolicy::Tuned => 0,
    }
}

fn unified_decode_row(
    ex: &mut Qwen3Executor,
    p: &[u32],
    cobatch: usize,
    id_dec: u64,
    id_pf: u64,
    policy: NumericPolicy,
) -> (Row, u64) {
    let id_a = RequestId::new(id_dec);
    let t0 = prefill_first(ex, id_a, p);
    let id_b = RequestId::new(id_pf);
    let chunk: Vec<u32> = (0..cobatch as u32).map(|i| (i % 1000) + 10).collect();
    reset_numeric_policy_counters();
    let ur = ex
        .execute_unified(UnifiedPlan {
            prefill_requests: &[PrefillStepItem::new(
                id_b,
                chunk,
                64,
                SamplingParams::default(),
                None,
                None,
            )],
            decode_requests: &[DecodeStepItem::new(
                id_a,
                t0,
                SamplingParams::default(),
                Some(64),
            )],
            sample_seed: 0,
        })
        .expect("unified");
    let row = row_of(&ur.decode_requests[0]);
    let served = served_under(policy);
    ex.drop_request(id_b).expect("drop prefill request");
    ex.drop_request(id_a).expect("drop decode request");
    (row, served)
}

fn pure_decode_row(ex: &mut Qwen3Executor, p: &[u32], id_dec: u64) -> Row {
    let id = RequestId::new(id_dec);
    let t0 = prefill_first(ex, id, p);
    reset_numeric_policy_counters();
    let dr = ex
        .execute_decode(DecodePlan {
            requests: &[DecodeStepItem::new(
                id,
                t0,
                SamplingParams::default(),
                Some(64),
            )],
            sample_seed: 0,
        })
        .expect("decode");
    let row = row_of(&dr.requests[0]);
    ex.drop_request(id).expect("drop decode request");
    row
}

const PROMPT: [u32; 8] = [9707, 785, 11, 1879, 13, 358, 1079, 264];
const COBATCH: [usize; 4] = [100, 200, 512, 1023];

// Both close the decode-vs-unified axis by routing unified decode rows through the decode ops.
const INVARIANT_POLICIES: [NumericPolicy; 2] = [NumericPolicy::Pin, NumericPolicy::PerToken];

#[test]
fn unified_within_path_gemm_n_invariant_under_pin() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _g = POLICY_LOCK.lock().unwrap();
    let p = PROMPT.to_vec();

    set_numeric_policy(NumericPolicy::Pin);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);
    let mut base: Option<Row> = None;
    for (i, &c) in COBATCH.iter().enumerate() {
        let n = c + 1;
        let (row, served) = unified_decode_row(
            &mut ex,
            &p,
            c,
            100 + i as u64 * 2,
            101 + i as u64 * 2,
            NumericPolicy::Pin,
        );
        assert!(served > 0, "Pin N={n}: served=0 — pin never ran (vacuous)");
        match &base {
            None => base = Some(row),
            Some(b) => assert_eq!(
                *b, row,
                "Pin: Unified decode row drifted at N={n} vs N=101 — GEMM-N not invariant within Unified"
            ),
        }
        eprintln!("[unified-gate] Pin N={n}: served={served} bit-eq-vs-N101=ok");
    }
    drop(ex);

    set_numeric_policy(NumericPolicy::Tuned);
    let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
    ex.set_prefix_cache_enabled(false);
    let mut tbase: Option<Row> = None;
    let mut drifted = false;
    for (i, &c) in COBATCH.iter().enumerate() {
        let (row, _) = unified_decode_row(
            &mut ex,
            &p,
            c,
            200 + i as u64 * 2,
            201 + i as u64 * 2,
            NumericPolicy::Tuned,
        );
        match &tbase {
            None => tbase = Some(row),
            Some(b) => drifted |= *b != row,
        }
    }
    eprintln!(
        "[unified-gate] Tuned within-Unified baseline drift: {}",
        if drifted {
            "drifts (reproduced)"
        } else {
            "STABLE"
        }
    );
    assert!(
        drifted,
        "Tuned within-Unified did not drift across N — batch-dependence not reproduced here, Pin pass vacuous"
    );
}

#[test]
fn cross_path_decode_vs_unified_bitequal_under_batch_invariant() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _g = POLICY_LOCK.lock().unwrap();
    let p = PROMPT.to_vec();
    for policy in INVARIANT_POLICIES {
        set_numeric_policy(policy);
        let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
        ex.set_prefix_cache_enabled(false);
        let dec = pure_decode_row(&mut ex, &p, 900);
        let (uni, served) = unified_decode_row(&mut ex, &p, 100, 901, 902, policy);
        assert!(
            served > 0,
            "[{policy:?}] cross-path: served=0 — the policy's GEMM path never ran (vacuous)"
        );
        assert_eq!(
            dec, uni,
            "[{policy:?}] cross-path residual reopened: pure-Decode and Unified disagree on the \
             decode row"
        );
        eprintln!("[cross-path] {policy:?}: pure-Decode vs Unified bit-equal, served={served}");
    }
}

// Exceed Tuned's 32-row cap (SPLIT_KV_MAX_BATCH_SIZE) to exercise Pin workspace/CSR sizing above it.
const PAST_SPLIT_CAP: usize = 33;

// Rows must differ from each other (a row-aliasing bug must not compare equal)
// with non-uniform lengths (irregular per-row page/chunk geometry); beyond that
// the values are arbitrary.
fn mixed_prompt(row: usize) -> Vec<u32> {
    let len = 50 + row * 3;
    (0..len as u32)
        .map(|i| (i + row as u32) % 1000 + 10)
        .collect()
}

// One request per prefill call: a single batched call would run the projection
// GEMM at N = sum of prompt lens (~3k) and overrun the pin envelope.
fn prefill_each(
    ex: &mut Qwen3Executor,
    prompts: &[Vec<u32>],
    id_base: u64,
) -> Vec<(RequestId, u32)> {
    prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let id = RequestId::new(id_base + i as u64);
            (id, prefill_first(ex, id, p))
        })
        .collect()
}

fn decode_items(decoders: &[(RequestId, u32)]) -> Vec<DecodeStepItem> {
    decoders
        .iter()
        .map(|&(id, tok)| DecodeStepItem::new(id, tok, SamplingParams::default(), Some(64)))
        .collect()
}

fn rows_of(results: &[DecodeRequestResult]) -> Vec<Row> {
    results.iter().map(row_of).collect()
}

fn pure_decode_batch(ex: &mut Qwen3Executor, prompts: &[Vec<u32>], id_base: u64) -> Vec<Row> {
    let decoders = prefill_each(ex, prompts, id_base);
    let items = decode_items(&decoders);
    let dr = ex
        .execute_decode(DecodePlan {
            requests: &items,
            sample_seed: 0,
        })
        .expect("decode batch");
    let out = rows_of(&dr.requests);
    for (id, _) in decoders {
        ex.drop_request(id).expect("drop decode request");
    }
    out
}

fn unified_decode_batch(
    ex: &mut Qwen3Executor,
    prompts: &[Vec<u32>],
    decode_id_base: u64,
    prefill_id: u64,
    policy: NumericPolicy,
) -> (Vec<Row>, u64) {
    let decoders = prefill_each(ex, prompts, decode_id_base);
    let decode_items = decode_items(&decoders);
    reset_numeric_policy_counters();
    let ur = ex
        .execute_unified(UnifiedPlan {
            prefill_requests: &[PrefillStepItem::new(
                RequestId::new(prefill_id),
                mixed_prompt(PAST_SPLIT_CAP + 1),
                64,
                SamplingParams::default(),
                None,
                None,
            )],
            decode_requests: &decode_items,
            sample_seed: 0,
        })
        .expect("unified batch");
    let out = rows_of(&ur.decode_requests);
    let served = served_under(policy);
    ex.drop_request(RequestId::new(prefill_id))
        .expect("drop prefill request");
    for (id, _) in decoders {
        ex.drop_request(id).expect("drop decode request");
    }
    (out, served)
}

#[test]
fn cross_path_decode_vs_unified_past_split_cap_bitequal_under_batch_invariant() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let _g = POLICY_LOCK.lock().unwrap();
    let prompts: Vec<Vec<u32>> = (0..PAST_SPLIT_CAP).map(mixed_prompt).collect();
    for policy in INVARIANT_POLICIES {
        set_numeric_policy(policy);
        let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("executor");
        ex.set_prefix_cache_enabled(false);

        let dec = pure_decode_batch(&mut ex, &prompts, 10_000);
        let (uni, served) = unified_decode_batch(&mut ex, &prompts, 20_000, 30_000, policy);
        assert!(
            served > 0,
            "[{policy:?}] past-cap cross-path: served=0 — the policy's GEMM path never ran (vacuous)"
        );
        assert_eq!(dec.len(), PAST_SPLIT_CAP, "pure-Decode row count changed");
        assert_eq!(
            uni.len(),
            PAST_SPLIT_CAP,
            "Unified decode row count changed"
        );
        for row in 0..PAST_SPLIT_CAP {
            assert_eq!(
                dec[row], uni[row],
                "[{policy:?}] past-cap cross-path row {row}: pure-Decode and Unified disagree"
            );
        }
        eprintln!(
            "[cross-path-past-cap] {policy:?}: {PAST_SPLIT_CAP} rows bit-equal, served={served}"
        );
    }
}
