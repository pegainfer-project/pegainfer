//! Gate: decode attention-path is batch-invariant under Pin.
//!
//! Tuned picks SplitKv (`bs <= 32`) vs NonPartition from the batch bucket — two numerically
//! different reductions. With the GEMM and split-chunk axes pinned and the request's prefill
//! isolated, the path choice is the only thing that can move its decode output across the boundary.
//! The isolation is asserted, not assumed.

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

/// `(token id, logprob)`, and the same with the logprob as its bit pattern.
type TopK = Vec<(u32, f32)>;
type TopKBits = Vec<(u32, u32)>;

const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4;
const A_LEN: usize = 2048; // long enough that SplitKv splits A into several chunks at bs=1
const FILLER_LEN: usize = 128;
const LARGE_COFILL: usize = 39; // decode bs = 40 > 32 -> NonPartition under Tuned; A alone is SplitKv
const PREFILL_CHUNK: usize = 1024; // chunk prefill within the pin's verified serve-N envelope, as the scheduler does

fn model_path_or_skip() -> Option<String> {
    let Ok(p) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!("skipping batch_invariance_attention_path: set PEGAINFER_TEST_MODEL_PATH");
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
        Some(LOGPROBS),
        None,
    )
}

fn filler(len: usize, stride: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 1000 + (i * stride) % 50000)
        .collect()
}

/// Prefill `prompt` in `PREFILL_CHUNK`-token chunks, as the scheduler does, so each chunk's GEMM N
/// stays inside the pin's verified serve-N envelope; returns the first decoded token and its top-K.
fn prefill_chunked(ex: &mut Qwen3Executor, id: RequestId, prompt: &[u32]) -> (u32, TopK) {
    let item = pitem(id, prompt.to_vec()).with_chunk_budget(PREFILL_CHUNK);
    loop {
        let pr = ex
            .execute_prefill(PrefillPlan {
                sample_seed: 0,
                requests: std::slice::from_ref(&item),
            })
            .expect("prefill chunk");
        let r = &pr.requests[0];
        if r.completed {
            let topk = r
                .first_token_logprob
                .as_ref()
                .expect("logprobs requested but none returned")
                .top_logprobs
                .clone();
            break (r.first_token, topk);
        }
    }
}

/// Prefill A alone (isolated KV), then decode it co-batched with `n_cofill` short fillers; return
/// A's prefill top-K (bit patterns) and its row-0 decode top-K. A's prefill is isolated, so only the
/// decode batch size — and thus `attention_path` — changes across calls.
fn a_decode_at_batch(
    ex: &mut Qwen3Executor,
    a_prompt: &[u32],
    n_cofill: usize,
) -> (TopKBits, TopK) {
    let id_a = RequestId::new(1);
    let (a_first, a_prefill_topk) = prefill_chunked(ex, id_a, a_prompt);

    let mut ditems = vec![DecodeStepItem::new(
        id_a,
        a_first,
        SamplingParams::default(),
        Some(LOGPROBS),
    )];
    let mut cofill_ids = Vec::with_capacity(n_cofill);
    for i in 0..n_cofill {
        let id = RequestId::new(100 + i as u64);
        let (f_first, _) = prefill_chunked(ex, id, &filler(FILLER_LEN, 7 + i as u32));
        ditems.push(DecodeStepItem::new(
            id,
            f_first,
            SamplingParams::default(),
            Some(LOGPROBS),
        ));
        cofill_ids.push(id);
    }

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

    ex.drop_request(id_a).expect("drop A");
    for id in cofill_ids {
        ex.drop_request(id).expect("drop filler");
    }
    let a_prefill_bits = a_prefill_topk
        .iter()
        .map(|&(id, lp)| (id, lp.to_bits()))
        .collect();
    (a_prefill_bits, topk)
}

/// Whether A's decode top-K is identical at bs=1 vs the boundary-crossing batch, under `policy`.
fn run_policy(policy: NumericPolicy, model_path: &str) -> bool {
    set_numeric_policy(policy);
    let mut ex = Qwen3Executor::from_runtime(model_path, true, &[0]).expect("build executor");
    ex.set_prefix_cache_enabled(false);
    let a = filler(A_LEN, 3);

    let (pf_small, tk_small) = a_decode_at_batch(&mut ex, &a, 0);
    let (pf_large, tk_large) = a_decode_at_batch(&mut ex, &a, LARGE_COFILL);
    assert_eq!(
        pf_small, pf_large,
        "[{policy:?}] A's isolated prefill is not bit-identical across the two runs: the decode \
         batch is no longer the only thing that changed, so neither the drifting baseline nor the \
         pinned result says anything about attention_path"
    );
    let eq = tk_small == tk_large;
    eprintln!(
        "batch_invariance_attention_path [{policy:?}]: bs=1(SplitKv) vs bs={} decode_topk_eq={eq} lp0(S={:.6},L={:.6})",
        1 + LARGE_COFILL,
        tk_small[0].1,
        tk_large[0].1
    );
    eq
}

#[test]
fn batch_invariance_attention_path() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tuned = run_policy(NumericPolicy::Tuned, &model_path);
    let pin = run_policy(NumericPolicy::Pin, &model_path);
    let pertoken = run_policy(NumericPolicy::PerToken, &model_path);
    eprintln!(
        "batch_invariance_attention_path: RESULT decode_topk_eq baseline={tuned} pin={pin} per_token={pertoken}"
    );

    assert!(
        !tuned,
        "baseline: A did not drift across the bucket boundary — control vacuous"
    );
    assert!(
        pin,
        "pin: A's decode top-K moved across the bucket boundary — attention_path not pinned under Pin"
    );
    assert!(
        pertoken,
        "per_token: A's decode top-K moved across the bucket boundary"
    );
}
