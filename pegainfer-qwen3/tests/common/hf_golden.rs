//! The committed Qwen3 HF golden fixtures and the bounds every backend's gate
//! holds them to. Shared by `hf_golden_gate` (CUDA) and the `golden` test of
//! `pegainfer-rocm`, which includes this file by path; each uses a subset.
#![allow(dead_code)]

use safetensors::Dtype;
use safetensors::SafeTensors;

/// Max acceptable *regret*: how far below HF's own argmax (in HF's logprobs)
/// pegainfer's chosen token may sit. ~3 bf16 ULP at typical logit magnitudes —
/// genuine ties fall well under it; where HF has a clear winner the only token
/// within this regret is HF's argmax itself, so this still enforces exact
/// agreement there.
pub(crate) const MARGIN_TOL: f32 = 0.20;
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
pub(crate) const MEAN_TOL: f32 = 0.06;
/// `P99_TOL` 0.20 ≈ 1.6× the floor (0.12): catches *spread* inflation — a subset
/// of positions getting noisier — that the mean would average away. The single
/// worst token can't move it (it is one delta in thousands), so it stays put as
/// coverage grows, unlike the absolute max.
pub(crate) const P99_TOL: f32 = 0.20;
/// Head depth for the logprob tolerance check (tail tokens are inherently noisy).
pub(crate) const HEAD_K: usize = 8;

pub(crate) const COMMITTED_FIXTURE_SIZES: &[&str] = &["0.6b", "1.7b", "4b", "8b", "14b", "32b"];

/// The fixture size a checkpoint geometry maps to, by `hidden_size` and
/// `num_hidden_layers`.
pub(crate) fn size_name(hidden: u64, layers: u64) -> Option<&'static str> {
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

pub(crate) fn default_fixture_path(size: &str) -> String {
    format!(
        "{}/../test_data/qwen3-{size}-hf-golden.safetensors",
        env!("CARGO_MANIFEST_DIR")
    )
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

/// The stored HF reference, parsed into owned vectors (the safetensors view
/// borrows the file bytes, so we copy out and let both drop).
pub(crate) struct Golden {
    pub(crate) prompt_tokens: Vec<i32>, // ragged, sliced by `prompt_lens`
    pub(crate) prompt_lens: Vec<i32>,   // [S]
    pub(crate) decode_tokens: Vec<i32>, // [S, D] flattened
    pub(crate) topk_ids: Vec<i32>,      // [S, D+1, K] flattened
    pub(crate) topk_lp: Vec<f32>,       // [S, D+1, K] flattened
    pub(crate) num_seqs: usize,
    pub(crate) decode_len: usize, // D
    pub(crate) positions: usize,  // D + 1
    pub(crate) k: usize,
}

impl Golden {
    pub(crate) fn load_path(path: &str) -> Golden {
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

    pub(crate) fn prompt(&self, seq: usize) -> Vec<u32> {
        let off: usize = self.prompt_lens[..seq].iter().map(|&l| l as usize).sum();
        let len = self.prompt_lens[seq] as usize;
        self.prompt_tokens[off..off + len]
            .iter()
            .map(|&t| t as u32)
            .collect()
    }

    pub(crate) fn decode(&self, seq: usize, step: usize) -> u32 {
        self.decode_tokens[seq * self.decode_len + step] as u32
    }

    /// HF's top-K `(token, logprob)` at `pos` (0 is the prompt's last row, then
    /// one per decode step), in HF's descending order.
    pub(crate) fn topk(&self, seq: usize, pos: usize) -> Vec<(u32, f32)> {
        let base = (seq * self.positions + pos) * self.k;
        (0..self.k)
            .map(|j| (self.topk_ids[base + j] as u32, self.topk_lp[base + j]))
            .collect()
    }
}
