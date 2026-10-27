//! PEFT-LoRA logits gate for Qwen3-4B. The zero-adapter `lora_smoke` test only
//! proves the route runs; this gate replays teacher-forced sequences with a non-zero
//! rank-1 q/v adapter against a PEFT reference, so a transposed, missing, or mis-scaled
//! delta fails. The fixture (from `tools/accuracy/dump_qwen3_4b_lora_golden.py`) embeds
//! the adapter tensors themselves — nothing reproduces PEFT or RNG at test time.
//!
//! Replay framework and tolerances are copied from `hf_golden_gate.rs` (tests cannot
//! share modules); see that header for the rationale.
//!
//! On a host with >=2 GPUs it also replays under TP=2 against the same reference: the
//! adapter is sharded per rank (B row-shard for q/k/v/gate/up, A col-shard for o/down),
//! so the col-shard deltas are summed by the same all-reduce as the base output — a path
//! single-GPU LoRA and the TP=2 base gate never exercise.
//!
//! Needs a CUDA GPU and Qwen3-4B weights (`PEGAINFER_TEST_MODEL_PATH`); skips cleanly
//! otherwise.

use std::collections::BTreeMap;
use std::path::Path;

use pegainfer_frontend::engine::LoadLoraAdapterRequest;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::lora_fixtures::FixtureTensor;
use pegainfer_qwen3::lora_fixtures::{self as fixtures};
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;
use safetensors::Dtype;
use safetensors::SafeTensors;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/qwen3-4b-lora-golden.safetensors"
);

const ADAPTER_NAME: &str = "golden-lora";
const LOGPROBS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 64;

// Tolerances as in hf_golden_gate.rs — the LoRA path stays in the same bf16 noise regime.
const MARGIN_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;
const P99_TOL: f32 = 0.20;
const HEAD_K: usize = 8;

/// The LoRA replay must differ from the base replay by more than this (mean |top-1
/// logprob| difference) — a silently ignored adapter fails here.
const CROSS_CHECK_FLOOR: f32 = 0.05;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 lora_golden_gate: {MODEL_PATH}/config.json is missing; set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
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
    head_deltas: Vec<f32>,
    worst: Option<(f32, usize, usize, u32, f32, f32)>,
}

fn check_position(
    stats: &mut Stats,
    seq: usize,
    pos: usize,
    pega: &[(u32, f32)],
    reference: &[(u32, f32)],
) {
    stats.positions += 1;

    // Regret in our own distribution, not the reference's: a benign bf16 tie is not a wrong token.
    let pega_top = pega[0].1;
    let ref_argmax = reference[0].0;
    match pega.iter().find(|&&(tok, _)| tok == ref_argmax) {
        None => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: reference argmax {ref_argmax} is absent from pegainfer's top-{}",
            pega.len()
        )),
        Some(&(_, plp)) if pega_top - plp > MARGIN_TOL => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer ranks {} over the reference argmax {ref_argmax} by {:.4} nat (> {MARGIN_TOL})",
            pega[0].0,
            pega_top - plp
        )),
        Some(_) => {}
    }

    for &(token, plp) in pega.iter().take(HEAD_K) {
        if let Some(&(_, rlp)) = reference.iter().find(|&&(t, _)| t == token) {
            let delta = (plp - rlp).abs();
            stats.head_deltas.push(delta);
            if stats.worst.is_none_or(|(w, ..)| delta > w) {
                stats.worst = Some((delta, seq, pos, token, plp, rlp));
            }
        }
    }
}

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
    assert!(
        stats.head_deltas.len() >= stats.positions,
        "[{label}] only {} head deltas over {} positions — top-K overlap with the reference collapsed",
        stats.head_deltas.len(),
        stats.positions
    );
    let (mean, p50, p99, max) = dist(&stats.head_deltas);
    eprintln!(
        "lora_golden_gate [{label}]: {} positions, {} head deltas — \
         mean {mean:.4} p50 {p50:.4} p99 {p99:.4} max {max:.4}",
        stats.positions,
        stats.head_deltas.len(),
    );
    if let Some((d, s, p, tok, plp, rlp)) = stats.worst {
        eprintln!(
            "lora_golden_gate [{label}]: worst head delta {d:.4} @ seq {s} pos {p} token {tok} (pega {plp:.4}, ref {rlp:.4})"
        );
    }
    assert!(
        stats.argmax_violations.is_empty(),
        "[{label}] pegainfer's argmax disagrees with the reference beyond tolerance:\n  {}",
        stats.argmax_violations.join("\n  ")
    );
    assert!(
        mean <= MEAN_TOL,
        "[{label}] mean head logprob delta {mean:.4} > {MEAN_TOL}"
    );
    assert!(
        p99 <= P99_TOL,
        "[{label}] p99 head logprob delta {p99:.4} > {P99_TOL}"
    );
}

struct Golden {
    prompt_tokens: Vec<i32>,
    prompt_lens: Vec<i32>,
    decode_tokens: Vec<i32>,
    base_topk_ids: Vec<i32>,
    base_topk_lp: Vec<f32>,
    lora_topk_ids: Vec<i32>,
    lora_topk_lp: Vec<f32>,
    adapter_tensors: BTreeMap<String, FixtureTensor>,
    adapter_config_json: String,
    num_seqs: usize,
    decode_len: usize,
    positions: usize,
    k: usize,
}

impl Golden {
    fn load() -> Golden {
        let bytes = std::fs::read(GOLDEN).unwrap_or_else(|e| panic!("read {GOLDEN}: {e}"));
        let (_, meta) =
            SafeTensors::read_metadata(&bytes).expect("read golden safetensors metadata");
        let md = meta
            .metadata()
            .as_ref()
            .expect("golden must carry adapter metadata")
            .clone();
        let rank: usize = md["rank"].parse().expect("rank metadata");
        let alpha: usize = md["lora_alpha"].parse().expect("lora_alpha metadata");
        let adapter_config_json = format!(
            r#"{{
  "peft_type": "LORA",
  "r": {rank},
  "lora_alpha": {alpha},
  "target_modules": {}
}}"#,
            md["target_modules"]
        );

        let st = SafeTensors::deserialize(&bytes).expect("parse golden safetensors");
        let (prompt_tokens, _) = as_i32(&st, "prompt_tokens");
        let (prompt_lens, _) = as_i32(&st, "prompt_lens");
        let (decode_tokens, dshape) = as_i32(&st, "decode_tokens");
        let (base_topk_ids, ishape) = as_i32(&st, "base_topk_ids");
        let base_topk_lp = as_f32(&st, "base_topk_logprobs");
        let (lora_topk_ids, _) = as_i32(&st, "lora_topk_ids");
        let lora_topk_lp = as_f32(&st, "lora_topk_logprobs");

        let mut adapter_tensors = BTreeMap::new();
        for name in st.names() {
            if let Some(tensor_name) = name.strip_prefix("adapter/") {
                let t = st.tensor(name).expect("adapter tensor");
                adapter_tensors.insert(
                    tensor_name.to_string(),
                    FixtureTensor {
                        dtype: t.dtype(),
                        shape: t.shape().to_vec(),
                        data: t.data().to_vec(),
                    },
                );
            }
        }

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
            base_topk_ids,
            base_topk_lp,
            lora_topk_ids,
            lora_topk_lp,
            adapter_tensors,
            adapter_config_json,
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

    fn topk(&self, lora: bool, seq: usize, pos: usize) -> Vec<(u32, f32)> {
        let (ids, lp) = if lora {
            (&self.lora_topk_ids, &self.lora_topk_lp)
        } else {
            (&self.base_topk_ids, &self.base_topk_lp)
        };
        let base = (seq * self.positions + pos) * self.k;
        (0..self.k)
            .map(|j| (ids[base + j] as u32, lp[base + j]))
            .collect()
    }

    /// Reconstruct the PEFT adapter directory the fixture embeds.
    fn write_adapter_dir(&self) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create temp adapter dir");
        fixtures::write_adapter_config_json(dir.path(), &self.adapter_config_json);
        fixtures::write_adapter_tensors(dir.path(), self.adapter_tensors.clone());
        dir
    }
}

fn prefill_item(id: RequestId, prompt: Vec<u32>, lora: bool) -> PrefillStepItem {
    PrefillStepItem::new(
        id,
        prompt,
        MAX_OUTPUT_TOKENS,
        SamplingParams::default(),
        Some(LOGPROBS),
        None,
    )
    .with_lora_adapter(lora.then(|| ADAPTER_NAME.to_string()))
}

fn decode_item(id: RequestId, fed: u32, lora: bool) -> DecodeStepItem {
    DecodeStepItem::new(id, fed, SamplingParams::default(), Some(LOGPROBS))
        .with_lora_adapter(lora.then(|| ADAPTER_NAME.to_string()))
}

/// Teacher-force `seqs`; `use_lora(seq)` picks each request's adapter and reference
/// grid. Returns the top-1 logprobs (the cross-check fingerprint).
fn run(
    g: &Golden,
    ex: &mut Qwen3Executor,
    seqs: &[usize],
    batched: bool,
    use_lora: impl Fn(usize) -> bool,
) -> (Stats, Vec<f32>) {
    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(use_lora(seq), seq, pos));
    };

    if batched {
        let ids: Vec<RequestId> = (0..seqs.len())
            .map(|i| RequestId::new(1000 + i as u64))
            .collect();
        let items: Vec<PrefillStepItem> = seqs
            .iter()
            .zip(&ids)
            .map(|(&s, &id)| prefill_item(id, g.prompt(s), use_lora(s)))
            .collect();
        let pr = ex
            .execute_prefill(PrefillPlan {
                sample_seed: 0,
                requests: &items,
            })
            .expect("prefill");
        for (i, &s) in seqs.iter().enumerate() {
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
                .map(|(&s, &id)| decode_item(id, g.decode(s, step), use_lora(s)))
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
                    requests: &[prefill_item(id, g.prompt(seq), use_lora(seq))],
                })
                .expect("prefill");
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
                        requests: &[decode_item(id, g.decode(seq, step), use_lora(seq))],
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

fn run_suite(
    golden: &Golden,
    model_path: &str,
    adapter_dir: &Path,
    devices: &[usize],
    label: &str,
) {
    let all: Vec<usize> = (0..golden.num_seqs).collect();
    let mut ex = Qwen3Executor::from_runtime(model_path, false, devices)
        .unwrap_or_else(|e| panic!("build executor on {devices:?}: {e:#}"));
    ex.set_prefix_cache_enabled(false);
    ex.load_lora_adapter(&LoadLoraAdapterRequest {
        lora_name: ADAPTER_NAME.to_string(),
        lora_path: adapter_dir.to_path_buf(),
        load_inplace: false,
    })
    .expect("load golden adapter");

    let (base_stats, base_fp) = run(golden, &mut ex, &all, false, |_| false);
    report_and_assert(
        &format!("{label}base bs=1 (adapter loaded, unused)"),
        &base_stats,
    );

    let (lora_stats, lora_fp) = run(golden, &mut ex, &all, false, |_| true);
    report_and_assert(&format!("{label}lora bs=1"), &lora_stats);

    let cross: f32 = base_fp
        .iter()
        .zip(&lora_fp)
        .map(|(b, l)| (b - l).abs())
        .sum::<f32>()
        / base_fp.len() as f32;
    eprintln!("lora_golden_gate [{label}cross-check]: mean |lora - base| top-1 logprob {cross:.4}");
    assert!(
        cross > CROSS_CHECK_FLOOR,
        "[{label}] LoRA replay is indistinguishable from base ({cross:.4} <= {CROSS_CHECK_FLOOR}) — the adapter never engaged"
    );

    let (mixed, _) = run(golden, &mut ex, &all, true, |s| s % 2 == 0);
    report_and_assert(&format!("{label}mixed lora/base batch"), &mixed);
}

#[test]
fn lora_logprobs_match_peft_golden_within_bf16_tolerance() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let golden = Golden::load();
    let adapter_dir = golden.write_adapter_dir();

    run_suite(&golden, &model_path, adapter_dir.path(), &[0], "");

    if cuda_device_count() >= 2 {
        run_suite(&golden, &model_path, adapter_dir.path(), &[0, 1], "tp2 ");
    } else {
        eprintln!("skipping lora_golden_gate TP=2 pass: <2 CUDA devices visible");
    }
}

fn cuda_device_count() -> usize {
    cudarc::driver::CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
}
