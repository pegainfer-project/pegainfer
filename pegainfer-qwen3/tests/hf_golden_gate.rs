//! HuggingFace-golden logits gate for Qwen3 (size-keyed: 4B, 14B, …) — the
//! hardware-independent replacement for the old text-snapshot e2e and digest.
//!
//! HuggingFace is the numerical golden truth. A bit-wise check (exact text or a
//! logprob hash) is fragile: bf16 GEMM kernels differ per GPU, so the low bits —
//! and any frozen snapshot — drift across hardware and false-positive on a
//! different card. Instead this gate asserts pegainfer lands within the *bf16
//! noise floor* of a stored HF reference, which every numerically-correct GPU
//! satisfies; only a real regression escapes the tolerance.
//!
//! The reference (`test_data/qwen3-{size}-hf-golden.safetensors`, produced once
//! by `tools/accuracy/dump_qwen3_hf_golden.py`) pins a set of fixed token
//! sequences and HF's top-K next-token logprobs at each position. We replay the
//! *same fixed sequences* through pegainfer by teacher-forcing — prefill the
//! prompt, then decode feeding the reference's own tail tokens — so every
//! position is compared against the identical-input HF distribution.
//! Teacher-forcing (vs free greedy) is what makes this stable: one argmax flip
//! can't send the two engines down diverging sequences.
//!
//! Assertions:
//!   * argmax — wherever HF has a clear winner (top-1 over top-2 margin exceeds a
//!     few bf16 ULP), pegainfer must pick the same token. Below that margin it is
//!     a genuine tie with no correct answer, so it is not enforced.
//!   * logprobs — on the head tokens, |pegainfer − HF| is bounded in the mean
//!     (catches uniform drift) and the p99 (catches a noisier subset). Both are
//!     coverage-stable; the single worst delta is reported but not asserted,
//!     because it grows with sample count (irreducible bf16 tail) while mean and
//!     p99 do not. A padding leak, KV mixing, or logit drift blows past them.
//!
//! The golden is replayed several ways, holding the *invariant the user cares
//! about* — stable logits across prompt / hardware / batch size:
//!   * bs=1 sequential (eager) — the tightest comparison; also rerun once to
//!     assert determinism (identical inputs ⇒ bit-identical logprobs).
//!   * batched eager — all sequences advance as one batch. Eager runs at the
//!     exact batch width (no padding), so this is the cross-request isolation
//!     check: requests of differing lengths share each kernel launch, and KV
//!     mixing or a per-request indexing bug corrupts a neighbour. It replaces the
//!     old exact batch==sequential check, which mistook the batched decode path's
//!     benign reduction-order noise (within tolerance here) for a bug — batch
//!     composition changes the reduction order and drifts logits ~1 ULP.
//!   * batched CUDA graph — the captured decode path pads the batch up to its
//!     bucket, so this is where padding-slot leaks (and graph pointer/buffer
//!     bugs) surface. Run the bucket straddles (9→16, 5→8) that maximise the
//!     padding-slot count. Only for a compiled GQA group; an uncompiled group
//!     (Qwen3-14B, group 5) reroutes decode eager and skips this pass.
//!
//! On a machine with ≥2 CUDA devices the eager replays additionally run under
//! TP=2 (`device_ordinals = [0, 1]`) against the same golden and tolerances —
//! sharding must not move logits beyond the single-GPU bf16 noise floor.
//!
//! Requires a CUDA GPU and Qwen3 weights of a keyed size; skips cleanly when the
//! model is absent (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).
//! The TP pass skips on single-GPU hosts.

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
use safetensors::Dtype;
use safetensors::SafeTensors;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const GOLDEN_ENV: &str = "PEGAINFER_QWEN3_HF_GOLDEN";

/// Ask the executor for as many logprobs as the golden stores so the top-K sets
/// overlap; the comparison only leans on the head, but a wider request makes the
/// argmax/tie reasoning robust.
const LOGPROBS: usize = 64;
/// KV reservation; prompts are short and the tail is `DECODE_TOKENS` long.
const MAX_OUTPUT_TOKENS: usize = 64;

/// Max acceptable *regret*: how far below HF's own argmax (in HF's logprobs)
/// pegainfer's chosen token may sit. ~3 bf16 ULP at typical logit magnitudes —
/// genuine ties fall well under it; where HF has a clear winner the only token
/// within this regret is HF's argmax itself, so this still enforces exact
/// agreement there.
const MARGIN_TOL: f32 = 0.20;
/// Two coverage-stable, strict bounds on the head-delta distribution. They are
/// the right guards precisely because — unlike the single worst delta — they
/// barely move as coverage grows: widening the golden from 108 to 816 positions
/// left mean at 0.032 and p99 at 0.12, while the absolute max ran from 0.26 to
/// 0.44 (more samples ⇒ a fatter tail of irreducible bf16 rounding). So the max
/// is reported but *not* asserted — chasing it with a ceiling is a treadmill. A
/// wrong-*token* bug is caught structurally by the regret check; a systematic or
/// spread drift by these two.
///
/// `MEAN_TOL` 0.06 ≈ 2× the floor: a uniform logit drift of `d` nat shifts every
/// delta by ~`d`, so this trips on any systematic drift past ~0.03 nat (one bf16
/// ULP at logit magnitude ~8). That is the sensitivity the gate exists for.
const MEAN_TOL: f32 = 0.06;
/// `P99_TOL` 0.20 ≈ 1.6× the floor (0.12): catches *spread* inflation — a subset
/// of positions getting noisier — that the mean would average away. The single
/// worst token can't move it (it is one delta in thousands), so it stays put as
/// coverage grows, unlike the absolute max.
const P99_TOL: f32 = 0.20;
/// Head depth for the logprob tolerance check (tail tokens are inherently noisy).
const HEAD_K: usize = 8;

/// Batch sizes for the batched passes, chosen against the CUDA-graph decode
/// buckets `[1, 2, 4, 8, 16, 20, ...]` (`BATCH_BUCKETS` in
/// `batch_decode_buffers.rs`). A
/// batch pads up to the next bucket, and padding-slot isolation is stressed
/// hardest just *past* a boundary, where almost a whole bucket is padding: 9
/// pads to 16 (7 pad slots) and 5 pads to 8 (3 pad slots). Both are well within
/// the golden's sequence count. The eager pass reuses the larger size purely as
/// a count of concurrent mixed-length requests (eager does not pad).
const BUCKET_STRADDLES: [usize; 2] = [9, 5];

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 hf_golden_gate: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

fn config_json(model_path: &str) -> serde_json::Value {
    let config_path = Path::new(model_path).join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", config_path.display()))
}

fn fixture_size_name(model_path: &str) -> Option<&'static str> {
    let v = config_json(model_path);
    let t = v.get("text_config").unwrap_or(&v);
    let hidden = t.get("hidden_size").and_then(serde_json::Value::as_u64);
    let layers = t
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64);
    let (Some(hidden), Some(layers)) = (hidden, layers) else {
        panic!("{model_path}/config.json has no hidden_size/num_hidden_layers");
    };
    match (hidden, layers) {
        (1024, 28) => Some("0.6b"),
        (2048, 28) => Some("1.7b"),
        (2560, 36) => Some("4b"),
        (4096, 36) => Some("8b"),
        (5120, 40) => Some("14b"),
        (5120, 64) => Some("32b"),
        _ => None,
    }
}

const COMMITTED_FIXTURE_SIZES: &[&str] = &["0.6b", "1.7b", "4b", "8b", "14b", "32b"];

fn default_fixture_path(size: &str) -> String {
    format!(
        "{}/../test_data/qwen3-{size}-hf-golden.safetensors",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn decode_group_compiled(model_path: &str) -> bool {
    let v = config_json(model_path);
    let t = v.get("text_config").unwrap_or(&v);
    let heads = t
        .get("num_attention_heads")
        .and_then(serde_json::Value::as_u64);
    let kv = t
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64);
    let (Some(heads), Some(kv)) = (heads, kv) else {
        panic!("{model_path}/config.json has no num_attention_heads/num_key_value_heads");
    };
    assert!(
        kv > 0 && heads % kv == 0,
        "invalid GQA head counts {heads}/{kv}"
    );
    pegainfer_core::ops::SUPPORTED_GQA_GROUP_SIZES.contains(&((heads / kv) as usize))
}

fn as_i32(st: &SafeTensors, name: &str) -> (Vec<i32>, Vec<usize>) {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::I32, "{name} must be i32");
    let v = t
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    (v, t.shape().to_vec())
}

fn as_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    let t = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("golden missing {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::F32, "{name} must be f32");
    t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn top_logprobs(lp: Option<&TokenLogprob>) -> Vec<(u32, f32)> {
    lp.expect("logprobs requested but none returned")
        .top_logprobs
        .clone()
}

#[derive(Default)]
struct Stats {
    positions: usize,
    argmax_violations: Vec<String>,
    head_deltas: Vec<f32>, // |pega - HF| logprob on the head tokens
    worst: Option<(f32, usize, usize, u32, f32, f32)>, // delta, seq, pos, token, pega, hf
    /// Prompt tokens served from the prefix cache, summed over all prefills.
    /// Zero when caching is disabled; the cached-replay passes assert it is not.
    cached_tokens: usize,
}

/// Fold one position into `stats`. `hf` and `pega` are top-K `(token, logprob)`,
/// both sorted by descending logprob (rank 0 is the argmax).
fn check_position(
    stats: &mut Stats,
    seq: usize,
    pos: usize,
    pega: &[(u32, f32)],
    hf: &[(u32, f32)],
) {
    stats.positions += 1;
    let hf_top = hf[0].1;
    let pega_argmax = pega[0].0;
    let hf_map: HashMap<u32, f32> = hf.iter().copied().collect();

    // pega's chosen token must be one HF also ranks near its own best. "Regret"
    // is how far below HF's argmax (in HF's own logprobs) pegainfer's pick sits.
    // A genuine bf16 tie differs by a ULP or two — small regret, fine. But a pick
    // HF scores clearly worse, or one absent from HF's top-K entirely (pegainfer
    // confidently wrong on a token HF does not even rank), is a real wrong-token
    // bug. This one rule subsumes "match HF where it is sure" *and* closes the
    // tie-band hole where a garbage argmax would otherwise escape every check.
    match hf_map.get(&pega_argmax) {
        None => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer's argmax {pega_argmax} is absent from HF's top-{} — confidently wrong on a token HF does not rank",
            hf.len()
        )),
        Some(&hlp) if hf_top - hlp > MARGIN_TOL => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer chose {pega_argmax}, which HF scores {:.4} nat below its own argmax (> {MARGIN_TOL} tie tolerance)",
            hf_top - hlp
        )),
        Some(_) => {}
    }

    for &(token, plp) in pega.iter().take(HEAD_K) {
        if let Some(&hlp) = hf_map.get(&token) {
            let delta = (plp - hlp).abs();
            stats.head_deltas.push(delta);
            if stats.worst.is_none_or(|(w, ..)| delta > w) {
                stats.worst = Some((delta, seq, pos, token, plp, hlp));
            }
        }
    }
}

/// The stored HF reference, parsed into owned vectors (the safetensors view
/// borrows the file bytes, so we copy out and let both drop).
struct Golden {
    prompt_tokens: Vec<i32>, // ragged, sliced by `prompt_lens`
    prompt_lens: Vec<i32>,   // [S]
    decode_tokens: Vec<i32>, // [S, D] flattened
    topk_ids: Vec<i32>,      // [S, D+1, K] flattened
    topk_lp: Vec<f32>,       // [S, D+1, K] flattened
    num_seqs: usize,
    decode_len: usize, // D
    positions: usize,  // D + 1
    k: usize,
}

impl Golden {
    fn load_for(model_path: &str) -> Option<Golden> {
        let Some(size) = fixture_size_name(model_path) else {
            assert!(
                std::env::var(GOLDEN_ENV).is_err(),
                "{GOLDEN_ENV} is set but {model_path}/config.json geometry has no entry in the size table"
            );
            eprintln!(
                "skipping qwen3 hf_golden_gate: unrecognized model geometry in \
                 {model_path}/config.json; extend fixture_size_name to cover it"
            );
            return None;
        };
        let path = if let Ok(path) = std::env::var(GOLDEN_ENV) {
            path
        } else {
            let path = default_fixture_path(size);
            if !Path::new(&path).exists() {
                assert!(
                    !COMMITTED_FIXTURE_SIZES.contains(&size),
                    "committed golden fixture missing at {path}"
                );
                eprintln!(
                    "skipping qwen3 hf_golden_gate: no golden fixture for size {size} at {path}; \
                     generate one with tools/accuracy/dump_qwen3_hf_golden.py"
                );
                return None;
            }
            path
        };
        Some(Self::load_path(&path))
    }

    fn load_path(path: &str) -> Golden {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let st = SafeTensors::deserialize(&bytes).expect("parse golden safetensors");
        let (prompt_tokens, _) = as_i32(&st, "prompt_tokens");
        let (prompt_lens, _) = as_i32(&st, "prompt_lens");
        let (decode_tokens, dshape) = as_i32(&st, "decode_tokens");
        let (topk_ids, ishape) = as_i32(&st, "topk_ids");
        let topk_lp = as_f32(&st, "topk_logprobs");
        let num_seqs = prompt_lens.len();
        let decode_len = dshape[1];
        let positions = ishape[1];
        let k = ishape[2];
        assert_eq!(
            positions,
            decode_len + 1,
            "positions must be decode_len + 1"
        );
        Golden {
            prompt_tokens,
            prompt_lens,
            decode_tokens,
            topk_ids,
            topk_lp,
            num_seqs,
            decode_len,
            positions,
            k,
        }
    }

    fn prompt(&self, seq: usize) -> Vec<u32> {
        let off: usize = self.prompt_lens[..seq].iter().map(|&l| l as usize).sum();
        let len = self.prompt_lens[seq] as usize;
        self.prompt_tokens[off..off + len]
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    fn decode(&self, seq: usize, step: usize) -> u32 {
        self.decode_tokens[seq * self.decode_len + step] as u32
    }

    fn topk(&self, seq: usize, pos: usize) -> Vec<(u32, f32)> {
        let base = (seq * self.positions + pos) * self.k;
        (0..self.k)
            .map(|j| (self.topk_ids[base + j] as u32, self.topk_lp[base + j]))
            .collect()
    }
}

fn prefill_item(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        Some(LOGPROBS),
        None,
    )
}

fn decode_item(id: RequestId, fed: u32) -> DecodeStepItem {
    DecodeStepItem::new(id, fed, SamplingParams::default(), Some(LOGPROBS))
}

/// Teacher-force the golden sequences `seqs` through `ex` and fold every
/// position into a [`Stats`]. `batched = false` runs each sequence alone (bs=1,
/// the tightest comparison); `batched = true` advances them all as one batch.
/// Restricting `seqs` lets a caller hit a specific CUDA-graph bucket (e.g. 5
/// seqs → bucket 8) so more than one real/pad ratio is exercised. The returned
/// vector is pegainfer's own top-1 logprob at each evaluated position — a
/// fingerprint two identical runs must reproduce bit-for-bit (determinism).
fn run(g: &Golden, ex: &mut Qwen3Executor, seqs: &[usize], batched: bool) -> (Stats, Vec<f32>) {
    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(seq, pos));
    };

    if batched {
        let ids: Vec<RequestId> = (0..seqs.len())
            .map(|i| RequestId::new(1000 + i as u64))
            .collect();
        let items: Vec<PrefillStepItem> = seqs
            .iter()
            .zip(&ids)
            .map(|(&s, &id)| prefill_item(id, g.prompt(s)))
            .collect();
        let pr = ex
            .execute_prefill(PrefillPlan {
                sample_seed: 0,
                requests: &items,
            })
            .expect("prefill");
        for (i, &s) in seqs.iter().enumerate() {
            stats.cached_tokens += pr.requests[i].cached_tokens;
            fold(
                &mut stats,
                s,
                0,
                &top_logprobs(pr.requests[i].first_token_logprob.as_ref()),
            );
        }
        for step in 0..g.decode_len {
            let items: Vec<DecodeStepItem> = seqs
                .iter()
                .zip(&ids)
                .map(|(&s, &id)| decode_item(id, g.decode(s, step)))
                .collect();
            let dr = ex
                .execute_decode(DecodePlan {
                    sample_seed: 0,
                    requests: &items,
                })
                .expect("decode");
            for (i, &s) in seqs.iter().enumerate() {
                fold(
                    &mut stats,
                    s,
                    step + 1,
                    &top_logprobs(dr.requests[i].logprob.as_ref()),
                );
            }
        }
        for &id in &ids {
            ex.drop_request(id).expect("drop request");
        }
    } else {
        for &seq in seqs {
            let id = RequestId::new(seq as u64);
            let pr = ex
                .execute_prefill(PrefillPlan {
                    sample_seed: 0,
                    requests: &[prefill_item(id, g.prompt(seq))],
                })
                .expect("prefill");
            stats.cached_tokens += pr.requests[0].cached_tokens;
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[0].first_token_logprob.as_ref()),
            );
            for step in 0..g.decode_len {
                let dr = ex
                    .execute_decode(DecodePlan {
                        sample_seed: 0,
                        requests: &[decode_item(id, g.decode(seq, step))],
                    })
                    .expect("decode");
                fold(
                    &mut stats,
                    seq,
                    step + 1,
                    &top_logprobs(dr.requests[0].logprob.as_ref()),
                );
            }
            ex.drop_request(id).expect("drop request");
        }
    }
    (stats, fingerprint)
}

/// `(mean, p50, p99, max)` of a delta slice.
fn dist(deltas: &[f32]) -> (f32, f32, f32, f32) {
    let mut s = deltas.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| s[((s.len() as f64 * q) as usize).min(s.len() - 1)];
    (
        s.iter().sum::<f32>() / s.len() as f32,
        pct(0.50),
        pct(0.99),
        *s.last().unwrap(),
    )
}

fn report_and_assert(label: &str, stats: &Stats) {
    // Every position contributes at least its argmax token to `head_deltas` (the
    // regret check guarantees pega's argmax is in HF's top-K), so a count below
    // one-per-position means the top-K overlap collapsed and the mean is being
    // computed over too few samples to trust — fail loudly rather than pass on a
    // shrunken sample. (Also guards `dist` against an empty slice.)
    assert!(
        stats.head_deltas.len() >= stats.positions,
        "[{label}] only {} head deltas over {} positions — top-K overlap with HF collapsed",
        stats.head_deltas.len(),
        stats.positions
    );
    let (mean, p50, p99, max) = dist(&stats.head_deltas);
    eprintln!(
        "hf_golden_gate [{label}]: {} positions, {} head deltas — \
         mean {mean:.4} p50 {p50:.4} p99 {p99:.4} max {max:.4}",
        stats.positions,
        stats.head_deltas.len(),
    );
    if let Some((d, s, p, tok, plp, hlp)) = stats.worst {
        eprintln!(
            "hf_golden_gate [{label}]: worst head delta {d:.4} @ seq {s} pos {p} token {tok} (pega {plp:.4}, HF {hlp:.4})"
        );
    }

    assert!(
        stats.argmax_violations.is_empty(),
        "[{label}] pegainfer picked a token HF does not rank near its best:\n  {}",
        stats.argmax_violations.join("\n  ")
    );
    assert!(
        mean <= MEAN_TOL,
        "[{label}] mean head logprob delta {mean:.4} > {MEAN_TOL} — logits drifted uniformly across the distribution (see above)"
    );
    assert!(
        p99 <= P99_TOL,
        "[{label}] p99 head logprob delta {p99:.4} > {P99_TOL} — a subset of positions got noisier than bf16 rounding (see above)"
    );
    let _ = max; // reported above, but not asserted: the worst single delta grows with coverage
}

/// Number of visible CUDA devices; 0 when the driver cannot be queried.
fn cuda_device_count() -> usize {
    cudarc::driver::CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
}

/// The eager replay suite — bs=1 sequential (plus the bit-identical
/// determinism rerun), batched eager isolation, and the prefix-cached replays
/// — over a given device set, with `label` prefixing every report line.
fn run_eager_suite(golden: &Golden, model_path: &str, devices: &[usize], label: &str) {
    let all: Vec<usize> = (0..golden.num_seqs).collect();
    let mut ex = Qwen3Executor::from_runtime(model_path, false, devices)
        .unwrap_or_else(|e| panic!("build eager executor on {devices:?}: {e:#}"));
    // The determinism and isolation passes need bit-replayable prefill: a
    // prefix-cache hit shrinks the prefill GEMM to the uncached suffix,
    // which drifts logits by bf16 ULPs. Caching gets its own pass below.
    ex.set_prefix_cache_enabled(false);

    // bs=1 sequential over *every* golden sequence — this is the breadth of
    // the prompt/length coverage (one request's KV at a time, so it scales to
    // long prompts cheaply). Also the determinism anchor: a second identical
    // run must reproduce every logprob bit-for-bit. A nondeterministic kernel
    // or uninitialised decode scratch would not.
    let (stats, fp1) = run(golden, &mut ex, &all, false);
    report_and_assert(&format!("{label}sequential bs=1 eager"), &stats);
    let (_, fp2) = run(golden, &mut ex, &all, false);
    assert_eq!(
        fp1, fp2,
        "{label}sequential bs=1 eager: identical inputs must reproduce identical logprobs"
    );

    // A batch advanced together (eager runs at the exact width — no padding).
    // This is the cross-request isolation check: requests of differing
    // lengths/page layouts share each kernel launch, so KV mixing or a
    // per-request indexing bug corrupts a neighbour's logits. It replaces the
    // old exact batch==sequential check, which mistook the batched decode
    // path's benign reduction-order noise (within tolerance here) for a bug.
    // Breadth is the bs=1 pass's job; this just needs enough concurrent reqs.
    let n = BUCKET_STRADDLES[0];
    let (batched, _) = run(golden, &mut ex, &all[..n], true);
    report_and_assert(&format!("{label}batched eager ({n}, no pad)"), &batched);

    // Prefix-cached replay. Matching-disabled requests intentionally reset
    // their completed blocks, so first seed under enabled retention and then
    // rerun warm: prefill skips the cached prefix and attention spans cached
    // KV + recomputed suffix (kv_len > q_len). Same HF tolerances — a wrong
    // RoPE offset, mask, or stale page would blow far past them.
    ex.set_prefix_cache_enabled(true);
    let _ = run(golden, &mut ex, &all, false);
    let (cached, _) = run(golden, &mut ex, &all, false);
    assert!(
        cached.cached_tokens > 0,
        "{label}cached replay matched no prefix blocks — the cache path was not exercised"
    );
    report_and_assert(
        &format!("{label}sequential bs=1 eager cached replay"),
        &cached,
    );
    let (cached_batched, _) = run(golden, &mut ex, &all[..n], true);
    report_and_assert(
        &format!("{label}batched eager cached replay ({n})"),
        &cached_batched,
    );
}

#[test]
fn pega_logprobs_match_hf_golden_within_bf16_tolerance() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let Some(golden) = Golden::load_for(&model_path) else {
        return;
    };
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    // Debug knob: skip the single-GPU passes and run only the TP=2 suite.
    // The full gate builds TP executors after single-GPU executors in the
    // same process; running with this set isolates the TP pass from any
    // state they leave behind.
    if std::env::var("PEGAINFER_GOLDEN_TP_ONLY").is_ok() {
        let gpus = cuda_device_count();
        assert!(
            gpus >= 2,
            "PEGAINFER_GOLDEN_TP_ONLY needs >=2 GPUs, have {gpus}"
        );
        run_eager_suite(&golden, &model_path, &[0, 1], "tp2 ");
        return;
    }

    // Each executor owns its GPU memory; scope them so only one model is
    // resident at a time (drop frees it before the next load). Everything
    // stays in one #[test] for the same reason: two parallel tests would both
    // budget 85% of free VRAM for KV.
    run_eager_suite(&golden, &model_path, &[0], "");

    // CUDA-graph decode is captured per bucket and pads the batch up to it, so
    // this path is where padding-slot leaks (and graph pointer/buffer bugs)
    // surface. Run the bucket straddles, which maximise the padding-slot count.
    if decode_group_compiled(&model_path) {
        let mut ex = Qwen3Executor::from_runtime(&model_path, true, &[0])
            .expect("build cuda-graph executor");
        ex.set_prefix_cache_enabled(false);
        for n in BUCKET_STRADDLES {
            let (batched, _) = run(&golden, &mut ex, &all[..n], true);
            report_and_assert(&format!("batched cuda-graph ({n} padded)"), &batched);
        }

        // Graph decode over shared cached pages: the captured graph reads page
        // tables that now point at blocks written by earlier requests.
        ex.set_prefix_cache_enabled(true);
        let n = BUCKET_STRADDLES[1];
        let _ = run(&golden, &mut ex, &all[..n], true);
        let (cached, _) = run(&golden, &mut ex, &all[..n], true);
        assert!(
            cached.cached_tokens > 0,
            "cuda-graph cached replay matched no prefix blocks"
        );
        report_and_assert(&format!("batched cuda-graph cached replay ({n})"), &cached);
    } else {
        eprintln!(
            "hf_golden_gate: skipping CUDA-graph passes for {model_path} (uncompiled GQA group, reroutes to eager)"
        );
    }

    // TP=2: the same eager suite sharded over two GPUs. Same golden, same
    // tolerances — a reduction-order, shard-offset, or all-reduce bug drifts
    // logits far past the bf16 noise floor.
    let gpus = cuda_device_count();
    if gpus < 2 {
        eprintln!("skipping hf_golden_gate TP=2 pass: {gpus} CUDA device(s) visible, need 2");
        return;
    }
    run_eager_suite(&golden, &model_path, &[0, 1], "tp2 ");

    // TP=2, graph on: replay the pre-captured decode graphs against the same
    // golden, so a bad shard, pointer, or collective in a capture drifts logits.
    if decode_group_compiled(&model_path) {
        let mut ex = Qwen3Executor::from_runtime(&model_path, true, &[0, 1])
            .expect("build cuda-graph TP executor");
        ex.set_prefix_cache_enabled(false);
        for n in BUCKET_STRADDLES {
            let (batched, _) = run(&golden, &mut ex, &all[..n], true);
            report_and_assert(&format!("tp2 batched cuda-graph ({n} padded)"), &batched);
        }
    }
}
