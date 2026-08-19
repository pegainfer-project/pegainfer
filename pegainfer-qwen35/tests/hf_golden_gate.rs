//! HuggingFace-golden logits gate for the Qwen3.5 line.
//!
//! This is the Qwen3.5 instance of the reusable logits-golden method in
//! `docs/subsystems/correctness/logits-golden-gate.md`: store HF bf16 top-K
//! logprobs for fixed teacher-forced token sequences, replay those sequences
//! through pegainfer, and compare bounded logprob drift instead of exact text.
//! The Qwen3.5 fixture is produced through HF's incremental `past_key_values`
//! path so the oracle matches pegainfer's prefill + decode shape.
//!
//! Qwen3.5 currently has no eager batched decode path; decode goes through the
//! CUDA-graph bucketed path. This gate therefore covers sequential bs=1,
//! bucket-straddling batched graph passes, and slot compaction after a request
//! is dropped mid-replay.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_qwen35::runtime::DecodePlan;
use pegainfer_qwen35::runtime::DecodeStepItem;
use pegainfer_qwen35::runtime::DropExpectation;
use pegainfer_qwen35::runtime::PrefillPlan;
use pegainfer_qwen35::runtime::PrefillStepItem;
use pegainfer_qwen35::runtime::Qwen35Executor;
use pegainfer_qwen35::runtime::Qwen35TpExecutor;
use pegainfer_qwen35::runtime::RequestId;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

mod common;

const GOLDEN_ENV: &str = "PEGAINFER_QWEN35_HF_GOLDEN";
const LONG_GOLDEN_ENV: &str = "PEGAINFER_QWEN35_HF_LONG_GOLDEN";

const LOGPROBS: usize = 64;
const MAX_EXECUTOR_BATCH: usize = 8;

const HEAD_K: usize = 8;

// Shared 4B calibration; split per-size only when a size needs its own.
const MARGIN_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;
const P99_TOL: f32 = 0.20;

/// Size key from config CONTENT, not the directory name; keep in sync with
/// `SIZE_NAMES` in `tools/accuracy/dump_qwen35_hf_golden.py`.
fn fixture_size_name(model_path: &str) -> Option<&'static str> {
    let config_path = Path::new(model_path).join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));
    let v: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("parse {}: {e}", config_path.display()));
    let t = v.get("text_config").unwrap_or(&v);
    let hidden = t.get("hidden_size").and_then(serde_json::Value::as_u64);
    let layers = t
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64);
    let (Some(hidden), Some(layers)) = (hidden, layers) else {
        panic!(
            "{} has no hidden_size/num_hidden_layers",
            config_path.display()
        );
    };
    match (hidden, layers) {
        (1024, 24) => Some("0.8b"),
        (2048, 24) => Some("2b"),
        (2560, 32) => Some("4b"),
        (4096, 32) => Some("9b"),
        (5120, 64) => Some("27b"),
        _ => None,
    }
}

/// Sizes whose fixtures are committed in `test_data/`; a missing file for
/// these is a broken checkout, not an ungenerated fixture.
const COMMITTED_FIXTURE_SIZES: &[&str] = &["0.8b", "2b", "4b", "9b", "27b"];

fn default_fixture_path(size: &str, long: bool) -> String {
    let kind = if long {
        "-hf-long-golden"
    } else {
        "-hf-golden"
    };
    format!(
        "{}/../test_data/qwen35-{size}{kind}.safetensors",
        env!("CARGO_MANIFEST_DIR")
    )
}

const BUCKET_STRADDLES: [usize; 2] = [5, 3];
const SLOT_COMPACTION_BATCH: usize = 5;
const SLOT_COMPACTION_DROP_INDEX: usize = 1;

fn sha256_file(path: impl AsRef<Path>) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Some(
        digest
            .finalize()
            .iter()
            .fold(String::new(), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            }),
    )
}

fn safetensors_metadata(bytes: &[u8]) -> HashMap<String, String> {
    let header_len_bytes: [u8; 8] = bytes[..8]
        .try_into()
        .expect("safetensors file missing 8-byte header length");
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let header = &bytes[8..8 + header_len];
    let value: serde_json::Value =
        serde_json::from_slice(header).expect("parse safetensors JSON header");
    value
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .map(|metadata| {
            metadata
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn model_revision(model_path: &str) -> Option<String> {
    if let Ok(value) = std::env::var("PEGAINFER_TEST_MODEL_REVISION") {
        return Some(value);
    }
    let path = Path::new(model_path);
    let metadata_path = path
        .join(".cache")
        .join("huggingface")
        .join("download")
        .join("config.json.metadata");
    if let Ok(content) = std::fs::read_to_string(metadata_path) {
        if let Some(first) = content.lines().next().map(str::trim) {
            if !first.is_empty() {
                return Some(first.to_string());
            }
        }
    }
    if path.join(".git").exists() {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(path)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .ok()?;
        if output.status.success() {
            return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
    }
    let parts: Vec<_> = path.components().collect();
    for window in parts.windows(2) {
        if window[0].as_os_str() == "snapshots" {
            return Some(window[1].as_os_str().to_string_lossy().to_string());
        }
    }
    None
}

fn require_metadata<'a>(metadata: &'a HashMap<String, String>, key: &str) -> &'a str {
    metadata
        .get(key)
        .unwrap_or_else(|| panic!("qwen35 hf_golden_gate fixture missing metadata key {key}"))
}

fn check_fixture_metadata(model_path: &str, golden: &Golden) -> bool {
    let metadata = &golden.metadata;
    assert_eq!(
        require_metadata(metadata, "dtype"),
        "bfloat16",
        "qwen35 hf_golden_gate fixture dtype mismatch; regenerate the fixture"
    );
    assert_eq!(
        require_metadata(metadata, "top_k"),
        LOGPROBS.to_string(),
        "qwen35 hf_golden_gate fixture top_k mismatch; regenerate the fixture"
    );

    let config = PathBuf::from(model_path).join("config.json");
    let actual_config_sha256 = sha256_file(&config).unwrap_or_else(|| {
        panic!(
            "qwen35 hf_golden_gate cannot read local config for metadata check: {}",
            config.display()
        )
    });
    assert_eq!(
        actual_config_sha256,
        require_metadata(metadata, "config_sha256"),
        "qwen35 hf_golden_gate config.json hash mismatch; regenerate the fixture for this model/config revision"
    );

    let expected_revision = require_metadata(metadata, "model_revision");
    assert_ne!(
        expected_revision, "unknown",
        "qwen35 hf_golden_gate fixture must record a pinned model_revision"
    );
    let Some(actual_revision) = model_revision(model_path) else {
        eprintln!(
            "skipping qwen35 hf_golden_gate: fixture requires model_revision={expected_revision}, but local model revision is unknown"
        );
        return false;
    };
    assert_eq!(
        actual_revision, expected_revision,
        "qwen35 hf_golden_gate model revision mismatch; set PEGAINFER_TEST_MODEL_REVISION or use the fixture's model snapshot"
    );

    if let Some(expected_tokenizer_revision) = metadata.get("tokenizer_revision") {
        assert_ne!(
            expected_tokenizer_revision, "unknown",
            "qwen35 hf_golden_gate fixture must record a pinned tokenizer_revision"
        );
    }
    true
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
    hf: &[(u32, f32)],
) {
    stats.positions += 1;
    let hf_top = hf[0].1;
    let pega_argmax = pega[0].0;
    let hf_map: HashMap<u32, f32> = hf.iter().copied().collect();

    match hf_map.get(&pega_argmax) {
        None => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer argmax {pega_argmax} absent from HF top-{}",
            hf.len()
        )),
        Some(&hlp) if hf_top - hlp > MARGIN_TOL => stats.argmax_violations.push(format!(
            "seq {seq} pos {pos}: pegainfer chose {pega_argmax}, HF scores it {:.4} nat below its argmax",
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

struct Golden {
    prompt_tokens: Vec<i32>,
    prompt_lens: Vec<i32>,
    decode_tokens: Vec<i32>,
    topk_ids: Vec<i32>,
    topk_lp: Vec<f32>,
    metadata: HashMap<String, String>,
    num_seqs: usize,
    decode_len: usize,
    positions: usize,
    k: usize,
}

impl Golden {
    /// An explicitly set env override must exist; a missing default keyed
    /// fixture is a clean skip (`None`).
    fn load_for(model_path: &str, long: bool) -> Option<Golden> {
        let env_key = if long { LONG_GOLDEN_ENV } else { GOLDEN_ENV };
        let Some(size) = fixture_size_name(model_path) else {
            assert!(
                std::env::var(env_key).is_err(),
                "{env_key} is set but the model geometry in {model_path}/config.json \
                 has no entry in the size table"
            );
            eprintln!(
                "skipping qwen35 hf_golden_gate: unrecognized model geometry in \
                 {model_path}/config.json; extend fixture_size_name to cover it"
            );
            return None;
        };
        let path = if let Ok(path) = std::env::var(env_key) {
            path
        } else {
            let path = default_fixture_path(size, long);
            if !Path::new(&path).exists() {
                assert!(
                    !COMMITTED_FIXTURE_SIZES.contains(&size),
                    "committed golden fixture missing at {path}"
                );
                eprintln!(
                    "skipping qwen35 hf_golden_gate: no golden fixture for this size at \
                     {path}; generate one with tools/accuracy/dump_qwen35_hf_golden.py"
                );
                return None;
            }
            path
        };
        Some(Self::load_path(path))
    }

    fn load_path(path: impl AsRef<Path>) -> Golden {
        let path = path.as_ref();
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let metadata = safetensors_metadata(&bytes);
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
        assert_eq!(positions, decode_len + 1);
        Golden {
            prompt_tokens,
            prompt_lens,
            decode_tokens,
            topk_ids,
            topk_lp,
            metadata,
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

    fn prompt_len(&self, seq: usize) -> usize {
        self.prompt_lens[seq] as usize
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

fn report_fixture_shape(golden: &Golden) {
    eprintln!(
        "qwen35 hf_golden_gate fixture: {} sequences, decode_len {}, prompt_lens [{}]",
        golden.num_seqs,
        golden.decode_len,
        prompt_lens_label(golden)
    );
}

fn prefill_item(id: RequestId, prompt: Vec<u32>) -> PrefillStepItem {
    PrefillStepItem::new(id, prompt, LOGPROBS)
}

fn decode_item(id: RequestId, fed: u32) -> DecodeStepItem {
    DecodeStepItem::new(id, fed, LOGPROBS)
}

fn run(g: &Golden, ex: &mut Qwen35Executor, seqs: &[usize], batched: bool) -> (Stats, Vec<f32>) {
    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(seq, pos));
    };

    if batched {
        let ids: Vec<RequestId> = seqs
            .iter()
            .map(|&seq| RequestId::new(1000 + seq as u64))
            .collect();
        let items: Vec<PrefillStepItem> = seqs
            .iter()
            .zip(&ids)
            .map(|(&seq, &id)| prefill_item(id, g.prompt(seq)))
            .collect();
        let pr = ex
            .execute_prefill(PrefillPlan { requests: &items })
            .expect("prefill");
        for (i, &seq) in seqs.iter().enumerate() {
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[i].first_token_logprob.as_ref()),
            );
        }

        for step in 0..g.decode_len {
            let items: Vec<DecodeStepItem> = seqs
                .iter()
                .zip(&ids)
                .map(|(&seq, &id)| decode_item(id, g.decode(seq, step)))
                .collect();
            let dr = ex
                .execute_decode(DecodePlan { requests: &items })
                .expect("decode");
            for (i, &seq) in seqs.iter().enumerate() {
                fold(
                    &mut stats,
                    seq,
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
                    requests: &[prefill_item(id, g.prompt(seq))],
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

fn run_tp(g: &Golden, ex: &Qwen35TpExecutor, seqs: &[usize], batched: bool) -> (Stats, Vec<f32>) {
    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(seq, pos));
    };

    if batched {
        let ids: Vec<RequestId> = seqs
            .iter()
            .map(|&seq| RequestId::new(10_000 + seq as u64))
            .collect();
        let items: Vec<PrefillStepItem> = seqs
            .iter()
            .zip(&ids)
            .map(|(&seq, &id)| prefill_item(id, g.prompt(seq)))
            .collect();
        let pr = ex
            .execute_prefill(PrefillPlan { requests: &items })
            .expect("TP2 prefill");
        for (i, &seq) in seqs.iter().enumerate() {
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[i].first_token_logprob.as_ref()),
            );
        }

        for step in 0..g.decode_len {
            let items: Vec<DecodeStepItem> = seqs
                .iter()
                .zip(&ids)
                .map(|(&seq, &id)| decode_item(id, g.decode(seq, step)))
                .collect();
            let dr = ex
                .execute_decode(DecodePlan { requests: &items })
                .expect("TP2 decode");
            for (i, &seq) in seqs.iter().enumerate() {
                fold(
                    &mut stats,
                    seq,
                    step + 1,
                    &top_logprobs(dr.requests[i].logprob.as_ref()),
                );
            }
        }

        for &id in &ids {
            ex.drop_request(id, DropExpectation::MustExist)
                .expect("TP2 drop request");
        }
    } else {
        for &seq in seqs {
            let id = RequestId::new(20_000 + seq as u64);
            let pr = ex
                .execute_prefill(PrefillPlan {
                    requests: &[prefill_item(id, g.prompt(seq))],
                })
                .expect("TP2 prefill");
            fold(
                &mut stats,
                seq,
                0,
                &top_logprobs(pr.requests[0].first_token_logprob.as_ref()),
            );
            for step in 0..g.decode_len {
                let dr = ex
                    .execute_decode(DecodePlan {
                        requests: &[decode_item(id, g.decode(seq, step))],
                    })
                    .expect("TP2 decode");
                fold(
                    &mut stats,
                    seq,
                    step + 1,
                    &top_logprobs(dr.requests[0].logprob.as_ref()),
                );
            }
            ex.drop_request(id, DropExpectation::MustExist)
                .expect("TP2 drop request");
        }
    }
    (stats, fingerprint)
}

fn prompt_lens_label(g: &Golden) -> String {
    (0..g.num_seqs)
        .map(|seq| format!("{seq}:{}", g.prompt_len(seq)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_with_slot_compaction(
    g: &Golden,
    ex: &mut Qwen35Executor,
    seqs: &[usize],
) -> (Stats, Vec<f32>) {
    assert!(
        seqs.len() > SLOT_COMPACTION_DROP_INDEX + 1,
        "slot-compaction replay needs a non-tail request to drop"
    );
    assert!(
        g.decode_len >= 2,
        "slot-compaction replay needs at least two decode tokens"
    );

    let mut stats = Stats::default();
    let mut fingerprint = Vec::new();
    let mut fold = |stats: &mut Stats, seq, pos, pega: &[(u32, f32)]| {
        fingerprint.push(pega[0].1);
        check_position(stats, seq, pos, pega, &g.topk(seq, pos));
    };

    let mut live: Vec<(usize, RequestId)> = seqs
        .iter()
        .map(|&seq| (seq, RequestId::new(2000 + seq as u64)))
        .collect();
    let items: Vec<PrefillStepItem> = live
        .iter()
        .map(|&(seq, id)| prefill_item(id, g.prompt(seq)))
        .collect();
    let pr = ex
        .execute_prefill(PrefillPlan { requests: &items })
        .expect("prefill");
    for (i, &(seq, _)) in live.iter().enumerate() {
        fold(
            &mut stats,
            seq,
            0,
            &top_logprobs(pr.requests[i].first_token_logprob.as_ref()),
        );
    }

    let step0: Vec<DecodeStepItem> = live
        .iter()
        .map(|&(seq, id)| decode_item(id, g.decode(seq, 0)))
        .collect();
    let dr = ex
        .execute_decode(DecodePlan { requests: &step0 })
        .expect("decode before compaction");
    for (i, &(seq, _)) in live.iter().enumerate() {
        fold(
            &mut stats,
            seq,
            1,
            &top_logprobs(dr.requests[i].logprob.as_ref()),
        );
    }

    let (_, dropped_id) = live[SLOT_COMPACTION_DROP_INDEX];
    ex.drop_request(dropped_id).expect("drop request");
    live.swap_remove(SLOT_COMPACTION_DROP_INDEX);

    for step in 1..g.decode_len {
        let items: Vec<DecodeStepItem> = live
            .iter()
            .map(|&(seq, id)| decode_item(id, g.decode(seq, step)))
            .collect();
        let dr = ex
            .execute_decode(DecodePlan { requests: &items })
            .expect("decode after compaction");
        for (i, &(seq, _)) in live.iter().enumerate() {
            fold(
                &mut stats,
                seq,
                step + 1,
                &top_logprobs(dr.requests[i].logprob.as_ref()),
            );
        }
    }

    for (_, id) in live {
        ex.drop_request(id).expect("drop request");
    }
    (stats, fingerprint)
}

fn dist(deltas: &[f32]) -> (f32, f32, f32, f32) {
    let mut s = deltas.to_vec();
    s.sort_by(f32::total_cmp);
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
        "[{label}] only {} head deltas over {} positions; top-K overlap collapsed",
        stats.head_deltas.len(),
        stats.positions
    );
    let (mean, p50, p99, max) = dist(&stats.head_deltas);
    eprintln!(
        "qwen35 hf_golden_gate [{label}]: {} positions, {} head deltas — mean {mean:.4} p50 {p50:.4} p99 {p99:.4} max {max:.4}",
        stats.positions,
        stats.head_deltas.len(),
    );
    if let Some((d, s, p, tok, plp, hlp)) = stats.worst {
        eprintln!(
            "qwen35 hf_golden_gate [{label}]: worst head delta {d:.4} @ seq {s} pos {p} token {tok} (pega {plp:.4}, HF {hlp:.4})"
        );
    }

    assert!(
        stats.argmax_violations.is_empty(),
        "[{label}] pegainfer picked a token HF does not rank near its best:\n  {}",
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
    let _ = max;
}

fn build_executor(model_path: &str) -> Qwen35Executor {
    Qwen35Executor::from_runtime(model_path, 0, MAX_EXECUTOR_BATCH)
        .expect("build Qwen3.5 logits executor")
}

fn build_tp2_executor(model_path: &str) -> Qwen35TpExecutor {
    let devices = common::tp2_device_ordinals();
    Qwen35TpExecutor::from_runtime_with_capacity(model_path, false, &devices, MAX_EXECUTOR_BATCH)
        .expect("build Qwen3.5 TP2 logits executor")
}

#[test]
fn pega_logprobs_match_hf_golden_within_qwen35_tolerance() {
    let Some(model_path) = common::model_path_or_skip("pega_logprobs_match_hf_golden") else {
        return;
    };
    let Some(golden) = Golden::load_for(&model_path, false) else {
        return;
    };
    if !check_fixture_metadata(&model_path, &golden) {
        return;
    }
    report_fixture_shape(&golden);
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    {
        let mut ex = build_executor(&model_path);

        let (stats, fp1) = run(&golden, &mut ex, &all, false);
        report_and_assert("sequential bs=1 graph", &stats);
        let (_, fp2) = run(&golden, &mut ex, &all, false);
        assert_eq!(
            fp1, fp2,
            "sequential Qwen3.5 replay must reproduce identical logprobs"
        );

        for n in BUCKET_STRADDLES {
            if all.len() >= n {
                let (batched, _) = run(&golden, &mut ex, &all[..n], true);
                report_and_assert(&format!("batched graph ({n} padded)"), &batched);
            } else {
                eprintln!(
                    "qwen35 hf_golden_gate: skipping batched graph ({n} padded); fixture has only {} sequence(s)",
                    all.len()
                );
            }
        }
    }

    if golden.num_seqs >= SLOT_COMPACTION_BATCH && golden.decode_len >= 2 {
        let fp1 = {
            let mut ex = build_executor(&model_path);
            let (compacted, fp) =
                run_with_slot_compaction(&golden, &mut ex, &all[..SLOT_COMPACTION_BATCH]);
            report_and_assert("slot-compaction graph", &compacted);
            fp
        };
        let fp2 = {
            let mut ex = build_executor(&model_path);
            let (_, fp) = run_with_slot_compaction(&golden, &mut ex, &all[..SLOT_COMPACTION_BATCH]);
            fp
        };
        assert_eq!(
            fp1, fp2,
            "slot-compaction Qwen3.5 replay must reproduce identical logprobs"
        );
    } else {
        eprintln!(
            "qwen35 hf_golden_gate: skipping slot-compaction graph; fixture has {} sequence(s), decode_len {}",
            golden.num_seqs, golden.decode_len
        );
    }
}

#[test]
fn pega_logprobs_match_hf_long_golden_within_qwen35_tolerance() {
    let Some(model_path) = common::model_path_or_skip("pega_logprobs_match_hf_long_golden") else {
        return;
    };
    let Some(golden) = Golden::load_for(&model_path, true) else {
        return;
    };
    if !check_fixture_metadata(&model_path, &golden) {
        return;
    }
    report_fixture_shape(&golden);
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    let mut ex = build_executor(&model_path);
    let (stats, fp1) = run(&golden, &mut ex, &all, false);
    report_and_assert("long sequential bs=1 graph", &stats);
    let (_, fp2) = run(&golden, &mut ex, &all, false);
    assert_eq!(
        fp1, fp2,
        "long sequential Qwen3.5 replay must reproduce identical logprobs"
    );
}

#[test]
#[ignore = "requires two CUDA devices, NCCL, and Qwen3.5 weights"]
fn pega_logprobs_match_hf_golden_within_qwen35_tolerance_tp2() {
    let Some(model_path) = common::model_path_or_skip("pega_logprobs_match_hf_golden_tp2") else {
        return;
    };
    let Some(golden) = Golden::load_for(&model_path, false) else {
        return;
    };
    if !check_fixture_metadata(&model_path, &golden) {
        return;
    }
    report_fixture_shape(&golden);
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    let ex = build_tp2_executor(&model_path);
    let (stats, fp1) = run_tp(&golden, &ex, &all, false);
    report_and_assert("TP2 sequential eager", &stats);
    let (_, fp2) = run_tp(&golden, &ex, &all, false);
    assert_eq!(
        fp1, fp2,
        "TP2 sequential Qwen3.5 replay must reproduce identical logprobs"
    );

    let batched_n = all.len().min(MAX_EXECUTOR_BATCH);
    let (batched, _) = run_tp(&golden, &ex, &all[..batched_n], true);
    report_and_assert("TP2 batched eager", &batched);
}

#[test]
#[ignore = "requires two CUDA devices, NCCL, and Qwen3.5 weights"]
fn pega_logprobs_match_hf_long_golden_within_qwen35_tolerance_tp2() {
    let Some(model_path) = common::model_path_or_skip("pega_logprobs_match_hf_long_golden_tp2")
    else {
        return;
    };
    let Some(golden) = Golden::load_for(&model_path, true) else {
        return;
    };
    if !check_fixture_metadata(&model_path, &golden) {
        return;
    }
    report_fixture_shape(&golden);
    let all: Vec<usize> = (0..golden.num_seqs).collect();

    let ex = build_tp2_executor(&model_path);
    let (stats, fp1) = run_tp(&golden, &ex, &all, false);
    report_and_assert("TP2 long sequential eager", &stats);
    let (_, fp2) = run_tp(&golden, &ex, &all, false);
    assert_eq!(
        fp1, fp2,
        "TP2 long sequential Qwen3.5 replay must reproduce identical logprobs"
    );
}
