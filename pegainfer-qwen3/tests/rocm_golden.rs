//! Independent committed HF-golden gate for the production ROCm executor.
#![cfg(feature = "rocm")]
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use pegainfer_qwen3::rocm::Model;
use pegainfer_qwen3::rocm::ModelOptions;
use pegainfer_qwen3::rocm::TokenInput;
use safetensors::SafeTensors;

struct Golden {
    prompts: Vec<i64>,
    lengths: Vec<i64>,
    tails: Vec<i64>,
    ids: Vec<i64>,
    logprobs: Vec<f32>,
    sequences: usize,
    decode: usize,
    k: usize,
}
fn i64s(t: &safetensors::tensor::TensorView<'_>) -> Vec<i64> {
    if t.dtype() == safetensors::Dtype::I32 {
        return t
            .data()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .collect();
    }
    assert_eq!(t.dtype(), safetensors::Dtype::I64);
    t.data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
impl Golden {
    fn read(path: PathBuf) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let t = SafeTensors::deserialize(&bytes)?;
        let prompts = t.tensor("prompt_tokens")?;
        let tails = t.tensor("decode_tokens")?;
        let ids = t.tensor("topk_ids")?;
        let lp = t.tensor("topk_logprobs")?;
        ensure!(lp.dtype() == safetensors::Dtype::F32, "golden dtype");
        Ok(Self {
            prompts: i64s(&prompts),
            lengths: i64s(&t.tensor("prompt_lens")?),
            tails: i64s(&tails),
            ids: i64s(&ids),
            logprobs: lp
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            sequences: tails.shape()[0],
            decode: tails.shape()[1],
            k: ids.shape()[2],
        })
    }
    fn prompt(&self, s: usize) -> Vec<u32> {
        let offset = self.lengths[..s].iter().sum::<i64>() as usize;
        self.prompts[offset..offset + self.lengths[s] as usize]
            .iter()
            .map(|&x| x as u32)
            .collect()
    }
}
#[derive(Default)]
struct Stats {
    delta: Vec<f32>,
    positions: usize,
}
impl Stats {
    fn add(
        &mut self,
        g: &Golden,
        seq: usize,
        step: usize,
        logits: &[f32],
        prediction: u32,
    ) -> Result<()> {
        ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max as f64
            + logits
                .iter()
                .map(|&x| ((x - max) as f64).exp())
                .sum::<f64>()
                .ln();
        let mut top = (0..logits.len()).collect::<Vec<_>>();
        top.select_nth_unstable_by(8, |&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        top[..8].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        ensure!(
            prediction as usize == top[0],
            "GPU argmax/tie rule differs from CPU at seq={seq} step={step}"
        );
        let base = (seq * (g.decode + 1) + step) * g.k;
        let reference = &g.ids[base..base + g.k];
        let lp = &g.logprobs[base..base + g.k];
        let chosen = reference
            .iter()
            .position(|&x| x == top[0] as i64)
            .with_context(|| {
                format!(
                    "argmax {} absent from HF top64 at seq={seq} step={step}",
                    top[0]
                )
            })?;
        ensure!(
            lp[0] - lp[chosen] <= 0.20,
            "regret {} > .20 at seq={seq} step={step}",
            lp[0] - lp[chosen]
        );
        for &token in &top[..8] {
            let at = reference
                .iter()
                .position(|&x| x == token as i64)
                .context("engine head token absent from HF reference")?;
            self.delta
                .push(((logits[token] as f64 - lse) - lp[at] as f64).abs() as f32);
        }
        self.positions += 1;
        Ok(())
    }
    fn finish(mut self, label: &str) -> Result<()> {
        self.delta.sort_by(f32::total_cmp);
        let mean = self.delta.iter().sum::<f32>() / self.delta.len() as f32;
        let p99 = self.delta[((self.delta.len() - 1) as f64 * 0.99).ceil() as usize];
        println!(
            "{label}: positions={} values={} mean={mean} p99={p99} max={}",
            self.positions,
            self.delta.len(),
            self.delta.last().unwrap()
        );
        ensure!(mean <= 0.06 && p99 <= 0.20, "HF golden gate failed");
        Ok(())
    }
}
#[test]
#[ignore = "requires a ROCm GPU and PEGAINFER_TEST_MODEL_PATH"]
fn hf_golden_sequential_and_ragged_batches() -> Result<()> {
    let path = PathBuf::from(
        std::env::var("PEGAINFER_TEST_MODEL_PATH").context("set PEGAINFER_TEST_MODEL_PATH")?,
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let size = match (
        config["hidden_size"].as_u64(),
        config["num_hidden_layers"].as_u64(),
    ) {
        (Some(1024), Some(28)) => "0.6b",
        (Some(2560), Some(36)) => "4b",
        _ => anyhow::bail!("first release golden requires 0.6B or 4B"),
    };
    let g = Golden::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../test_data/qwen3-{size}-hf-golden.safetensors")),
    )?;
    let mut model = Model::load(
        path,
        ModelOptions {
            max_context: 512,
            slots: 4,
            chunk_size: 128,
            ..Default::default()
        },
    )?;
    let mut stats = Stats::default();
    let mut repeat = Vec::new();
    for seq in 0..g.sequences {
        model.reset_slot(0)?;
        let prompt = g.prompt(seq);
        let mut prediction = 0;
        for (chunk, tokens) in prompt.chunks(128).enumerate() {
            let input = tokens
                .iter()
                .enumerate()
                .map(|(j, &token)| TokenInput {
                    token,
                    slot: 0,
                    position: chunk * 128 + j,
                })
                .collect::<Vec<_>>();
            prediction = model.forward(&input, &[input.len() - 1])?[0];
        }
        let first = model.logits()?.remove(0);
        if seq == 0 {
            repeat = first.clone();
        }
        stats.add(&g, seq, 0, &first, prediction)?;
        for step in 0..g.decode {
            let token = g.tails[seq * g.decode + step] as u32;
            let ids = model.forward(
                &[TokenInput {
                    token,
                    slot: 0,
                    position: prompt.len() + step,
                }],
                &[0],
            )?;
            stats.add(&g, seq, step + 1, &model.logits()?.remove(0), ids[0])?;
        }
    }
    stats.finish(&format!("{size} sequential"))?;
    model.reset_slot(0)?;
    let prompt = g.prompt(0);
    for (chunk, tokens) in prompt.chunks(128).enumerate() {
        let input = tokens
            .iter()
            .enumerate()
            .map(|(j, &token)| TokenInput {
                token,
                slot: 0,
                position: chunk * 128 + j,
            })
            .collect::<Vec<_>>();
        model.forward(&input, &[input.len() - 1])?;
    }
    ensure!(
        model.logits()?.remove(0) == repeat,
        "fixed-shape replay is nondeterministic"
    );
    let mut stats = Stats::default();
    for base in (0..g.sequences).step_by(4) {
        let count = 4.min(g.sequences - base);
        let prompts = (0..count).map(|i| g.prompt(base + i)).collect::<Vec<_>>();
        let mut cursors = vec![0; count];
        for slot in 0..count {
            model.reset_slot(slot)?;
        }
        while cursors
            .iter()
            .enumerate()
            .any(|(s, &n)| n < prompts[s].len())
        {
            let mut rows = Vec::new();
            let mut output = Vec::new();
            let mut owners = Vec::new();
            for slot in 0..count {
                let end = (cursors[slot] + 32).min(prompts[slot].len());
                for pos in cursors[slot]..end {
                    rows.push(TokenInput {
                        token: prompts[slot][pos],
                        slot,
                        position: pos,
                    });
                }
                if end > cursors[slot] && end == prompts[slot].len() {
                    output.push(rows.len() - 1);
                    owners.push(slot);
                }
                cursors[slot] = end;
            }
            let selected = if output.is_empty() {
                vec![rows.len() - 1]
            } else {
                output
            };
            let ids = model.forward(&rows, &selected)?;
            for ((logits, slot), prediction) in model.logits()?.iter().zip(owners).zip(ids) {
                stats.add(&g, base + slot, 0, logits, prediction)?;
            }
        }
        for step in 0..g.decode {
            let input = (0..count)
                .map(|slot| TokenInput {
                    token: g.tails[(base + slot) * g.decode + step] as u32,
                    slot,
                    position: prompts[slot].len() + step,
                })
                .collect::<Vec<_>>();
            let ids = model.forward(&input, &(0..count).collect::<Vec<_>>())?;
            for (slot, logits) in model.logits()?.iter().enumerate() {
                stats.add(&g, base + slot, step + 1, logits, ids[slot])?;
            }
        }
    }
    stats.finish(&format!("{size} ragged batch4"))?;
    let mut stats = Stats::default();
    let mut mixed_steps = 0;
    for base in (0..g.sequences).step_by(4) {
        let count = 4.min(g.sequences - base);
        // Interleave fixture lengths instead of grouping adjacent length bands.
        let seqs = (0..count)
            .map(|s| (base / 4 + s * (g.sequences / 4)) % g.sequences)
            .collect::<Vec<_>>();
        let prompts = seqs.iter().map(|&s| g.prompt(s)).collect::<Vec<_>>();
        let mut cursors = vec![0; count];
        for slot in 0..count {
            model.reset_slot(slot)?;
        }
        while (0..count).any(|s| cursors[s] < prompts[s].len() + g.decode) {
            let mut input = Vec::new();
            let mut outputs = Vec::new();
            let mut scored = Vec::new();
            let (mut prefill, mut decode) = (false, false);
            for slot in 0..count {
                let cursor = cursors[slot];
                let prompt_len = prompts[slot].len();
                if cursor < prompt_len {
                    prefill = true;
                    let end = (cursor + 32).min(prompt_len);
                    for position in cursor..end {
                        input.push(TokenInput {
                            token: prompts[slot][position],
                            slot,
                            position,
                        });
                    }
                    cursors[slot] = end;
                    if end == prompt_len {
                        outputs.push(input.len() - 1);
                        scored.push((seqs[slot], 0));
                    }
                } else if cursor < prompt_len + g.decode {
                    decode = true;
                    let step = cursor - prompt_len;
                    input.push(TokenInput {
                        token: g.tails[seqs[slot] * g.decode + step] as u32,
                        slot,
                        position: cursor,
                    });
                    cursors[slot] += 1;
                    outputs.push(input.len() - 1);
                    scored.push((seqs[slot], step + 1));
                }
            }
            mixed_steps += usize::from(prefill && decode);
            if outputs.is_empty() {
                outputs.push(input.len() - 1);
            }
            let ids = model.forward(&input, &outputs)?;
            for ((logits, (seq, step)), prediction) in model.logits()?.iter().zip(scored).zip(ids) {
                stats.add(&g, seq, step, logits, prediction)?;
            }
        }
    }
    ensure!(
        mixed_steps > 0,
        "fixture did not exercise mixed prefill/decode"
    );
    println!("mixed prefill/decode steps={mixed_steps}");
    stats.finish(&format!("{size} mixed batch4"))?;
    Ok(())
}
