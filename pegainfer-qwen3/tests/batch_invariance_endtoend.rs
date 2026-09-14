//! End-to-end prefill repro: pinning the projection GEMMs makes prompt_a's next-token top-K stop
//! depending on its batch-mates in this tested prefill path. Scope is the GEMM-N reduction-order
//! axis of the prefill API — the decode axes belong to `batch_invariance_attention_path`
//! (attention-path) and `batch_invariance_unified` (decode-vs-unified).
//!
//! prompt_a's top-K alone vs co-batched with a filler, under three policies: baseline (Tuned,
//! drifts = the bug), pin (top-K bit-identical = the fix), per_token (oracle). Pass requires full
//! ordered top-K equality + a baseline-drift guard + served>0 (per-token fallback under Pin is
//! impossible — `launch_gemm_pin` bails, surfacing as a prefill `.expect` panic).
//!
//!   PEGAINFER_TEST_MODEL_PATH=<Qwen3-4B-base> cargo test --release \
//!     -p pegainfer-qwen3 --test batch_invariance_endtoend -- --nocapture

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::pin_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4;

fn model_path_or_skip() -> Option<String> {
    let Ok(p) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping qwen3 batch_invariance_endtoend: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base"
        );
        return None;
    };
    Some(p)
}

fn item(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        Some(LOGPROBS),
        None,
    )
}

/// prompt_a (request 0)'s next-token `(token, logprob)` top-K table; drops every
/// request afterward so ids can be reused for the next call.
fn dist_for_first(ex: &mut Qwen3Executor, batch: &[(RequestId, Vec<u32>)]) -> Vec<(u32, f32)> {
    let items: Vec<PrefillStepItem> = batch.iter().map(|(id, p)| item(*id, p.clone())).collect();
    let pr = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &items,
        })
        .expect("prefill");
    let out = pr.requests[0]
        .first_token_logprob
        .as_ref()
        .expect("logprobs requested but none returned")
        .top_logprobs
        .clone();
    for (id, _) in batch {
        ex.drop_request(*id).expect("drop request");
    }
    out
}

#[test]
fn batch_invariance_endtoend() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let prompt_a: Vec<u32> = vec![9707];
    let nbs: [usize; 5] = [1, 7, 15, 31, 63]; // total_N = 2,8,16,32,64 — straddles GEMM_LT_MAX_N
    let id_a = RequestId::new(1);

    let mut baseline_drifted = false;

    for policy in [
        NumericPolicy::Tuned,
        NumericPolicy::Pin,
        NumericPolicy::PerToken,
    ] {
        // Fresh executor per policy: the pin warmup must run under Pin, as production sets the policy
        // before construction. Switching policy on a live executor leaves base shapes unpinned, which
        // launch_gemm_pin now bails on.
        set_numeric_policy(policy);
        let mut ex = Qwen3Executor::from_runtime(&model_path, false, &[0]).expect("build executor");
        ex.set_prefix_cache_enabled(false);
        let alone = dist_for_first(&mut ex, &[(id_a, prompt_a.clone())]);
        for &nb in &nbs {
            reset_numeric_policy_counters();
            let id_b = RequestId::new(1000 + nb as u64);
            let batched = dist_for_first(
                &mut ex,
                &[(id_a, prompt_a.clone()), (id_b, vec![785u32; nb])],
            );
            let total_n = prompt_a.len() + nb;
            let topk_equal = alone == batched;

            match policy {
                NumericPolicy::Tuned => {
                    if !topk_equal {
                        baseline_drifted = true;
                    }
                }
                NumericPolicy::Pin => {
                    assert!(
                        topk_equal,
                        "PIN: prompt_a top-K changed when co-batched (+Nb={nb}, total_N={total_n}) \
                         — projection-GEMM reduction-order NOT invariant under the pin in this prefill path"
                    );
                    // served>0 also covers the small-N (total_N<=32) case this loop exercises.
                    assert!(
                        pin_served() > 0,
                        "PIN: no GEMM ran the pinned algo (+Nb={nb}) — vacuous"
                    );
                }
                NumericPolicy::PerToken => {
                    assert!(
                        topk_equal,
                        "PER_TOKEN oracle: top-K changed (+Nb={nb}) — harness bug, not a result"
                    );
                }
            }
        }
    }

    assert!(
        baseline_drifted,
        "baseline did NOT drift on any case — was not reproduced here, so the pin pass is vacuous"
    );
}
