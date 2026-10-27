//! Gate: SplitKv decode batch-invariance. Co-batching A with a longer B moves A's chunk count;
//! Tuned drifts, Pin/PerToken stay bit-identical.

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;
use pegainfer_qwen3::runtime::split_chunk_size_for;

const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4;
// A long enough that A's Tuned chunk size exceeds the 64-token floor (ceil(A_LEN/64) > 64); B
// longer than A so the decode max_seq_len (= max KV length in the batch) rises from A_LEN to
// B_LEN, changing A's Tuned chunk count. Batch 2 <= 32 selects SplitKv for both calls (bucket 2).
const A_LEN: usize = 5000;
const B_LEN: usize = 8000;
const B_SHORT: usize = 100; // << A_LEN, so call C's decode max_seq_len = A_LEN

fn model_path_or_skip() -> Option<String> {
    let Ok(p) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3 batch_invariance_decode_splitkv_graph: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base"
        );
        return None;
    };
    Some(p)
}

// A_LEN/B_LEN exceed the pin's serve-N ceiling, so prefill chunked within the
// verified envelope, as the scheduler does — the gate is about decode split-KV.
const PREFILL_CHUNK: usize = 1024;

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

fn prefill_chunked(ex: &mut Qwen3Executor, id: RequestId, prompt: &[u32]) -> u32 {
    let item = pitem(id, prompt.to_vec()).with_chunk_budget(PREFILL_CHUNK);
    loop {
        let pr = ex
            .execute_prefill(PrefillPlan {
                sample_seed: 0,
                requests: std::slice::from_ref(&item),
            })
            .expect("prefill chunk");
        if pr.requests[0].completed {
            break pr.requests[0].first_token;
        }
    }
}

fn filler(len: usize, stride: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 1000 + (i * stride) % 50000)
        .collect()
}

/// Run A fixed, then decode A co-batched with B; returns A's `(prefill first_token, decode top-K)`.
fn a_decode_cobatched_with(
    ex: &mut Qwen3Executor,
    a_prompt: &[u32],
    b_prompt: &[u32],
) -> (u32, Vec<(u32, f32)>) {
    let id_a = RequestId::new(1);
    let id_b = RequestId::new(2);
    // Prefill A alone (batch 1); A's KV is identical regardless of B's length.
    let a_first = prefill_chunked(ex, id_a, a_prompt);
    let b_first = prefill_chunked(ex, id_b, b_prompt);
    // Decode A+B together (batch 2, A row 0); B's KV length sets the batch max_seq_len.
    let ditems = vec![
        DecodeStepItem::new(id_a, a_first, SamplingParams::default(), Some(LOGPROBS)),
        DecodeStepItem::new(id_b, b_first, SamplingParams::default(), Some(LOGPROBS)),
    ];
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
    ex.drop_request(id_b).expect("drop B");
    (a_first, topk)
}

/// Fresh executor with `policy` active before first decode; returns A's
/// `(first_token_eq, decode_topk_eq)` across capture vs replay of the same SplitKv graph.
fn run_policy(policy: NumericPolicy, model_path: &str) -> (bool, bool) {
    set_numeric_policy(policy);
    let mut ex = Qwen3Executor::from_runtime(model_path, true, &[0]).expect("build executor");
    ex.set_prefix_cache_enabled(false);
    let pl = match policy {
        NumericPolicy::Tuned => "baseline ",
        NumericPolicy::Pin => "pin      ",
        NumericPolicy::PerToken => "per_token",
    };
    let a = filler(A_LEN, 7);
    let b_short = filler(B_SHORT, 11);
    let b_long = filler(B_LEN, 13);

    let (ft_c, tk_c) = a_decode_cobatched_with(&mut ex, &a, &b_short); // capture @ max_seq=A_LEN
    let (ft_r, tk_r) = a_decode_cobatched_with(&mut ex, &a, &b_long); // replay  @ max_seq=B_LEN
    let ft_eq = ft_c == ft_r;
    let tk_eq = tk_c == tk_r;
    eprintln!(
        "batch_invariance_decode_splitkv_graph [{pl}]: path=SplitKv phase=capture@max_seq={}->replay@max_seq={} \
         A_prefill=isolated(batch1) decode_GEMM_N=2(fixed) \
         first_token_eq={ft_eq} decode_topk_eq={tk_eq} lp0(C={:.6},R={:.6})",
        A_LEN + 1,
        B_LEN + 1,
        tk_c[0].1,
        tk_r[0].1
    );
    (ft_eq, tk_eq)
}

#[test]
fn batch_invariance_decode_splitkv_graph() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    eprintln!(
        "batch_invariance_decode_splitkv_graph: A_LEN={A_LEN} B_LEN={B_LEN} cuda_graph=true, A's prefill isolated \
         (batch 1): SplitKv graph captured at max_seq=A_LEN, replayed at max_seq=B_LEN (same bucket 2). \
         Only A's Tuned split chunk size varies; Pin fixes it; A's prefill + decode GEMM N=2 are fixed."
    );

    let tuned_chunk_c = split_chunk_size_for(A_LEN + 1);
    let tuned_chunk_r = split_chunk_size_for(B_LEN + 1);
    assert_ne!(
        tuned_chunk_c, tuned_chunk_r,
        "Tuned chunk-size arithmetic did not drift; the split-KV control would be vacuous"
    );

    let (tuned_ft, tuned_tk) = run_policy(NumericPolicy::Tuned, &model_path);
    let (pin_ft, pin_tk) = run_policy(NumericPolicy::Pin, &model_path);
    let (pertoken_ft, pertoken_tk) = run_policy(NumericPolicy::PerToken, &model_path);

    eprintln!(
        "batch_invariance_decode_splitkv_graph: RESULT Tuned_chunk_tokens(C={tuned_chunk_c},R={tuned_chunk_r}) \
         | decode_topk_eq baseline={tuned_tk} pin={pin_tk} per_token={pertoken_tk} \
         | first_token_eq baseline={tuned_ft} pin={pin_ft} per_token={pertoken_ft}"
    );

    // A's prefill is isolated, so first_token must match C-vs-R under every policy (else isolation broke).
    assert!(
        tuned_ft,
        "baseline: A's prefill first_token differs C-vs-R, isolation broke"
    );
    assert!(
        pin_ft,
        "pin: A's prefill first_token differs C-vs-R, isolation broke"
    );
    assert!(
        pertoken_ft,
        "per_token: A's prefill first_token differs C-vs-R, isolation broke"
    );

    assert!(
        !tuned_tk,
        "baseline: A's decode top-K did not drift despite the Tuned chunk-size change; isolation suspect"
    );

    assert!(
        pin_tk,
        "pin: A's decode top-K changed across the SplitKv graph replay (capture@A_LEN, replay@B_LEN), \
         with prefill isolated, so (graph,SplitKv) is not batch-invariant under pin"
    );
    assert!(
        pertoken_tk,
        "per_token: A's decode top-K changed across the SplitKv graph replay; harness/fix bug"
    );

    eprintln!(
        "batch_invariance_decode_splitkv_graph: PASS with A's prefill isolated; Tuned chunk-size \
         arithmetic drifts and A's top-K follows, while Pin and PerToken replay bit-identically."
    );
}
