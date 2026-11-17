//! Independent committed HF-golden gate for the production ROCm executor.
#![cfg(feature = "rocm")]
#[path = "common/hf_golden.rs"]
mod hf_golden;

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use hf_golden::Golden;
use hf_golden::HEAD_K;
use hf_golden::MARGIN_TOL;
use hf_golden::MEAN_TOL;
use hf_golden::P99_TOL;
use pegainfer_qwen3::rocm::Model;
use pegainfer_qwen3::rocm::ModelOptions;
use pegainfer_qwen3::rocm::TokenInput;

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
        top.select_nth_unstable_by(HEAD_K, |&a, &b| {
            logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
        });
        top[..HEAD_K].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        ensure!(
            prediction as usize == top[0],
            "GPU argmax/tie rule differs from CPU at seq={seq} step={step}"
        );
        let reference = g.topk(seq, step);
        let chosen = reference
            .iter()
            .position(|&(id, _)| id as usize == top[0])
            .with_context(|| {
                format!(
                    "argmax {} absent from HF top{} at seq={seq} step={step}",
                    top[0],
                    reference.len()
                )
            })?;
        let regret = reference[0].1 - reference[chosen].1;
        ensure!(
            regret <= MARGIN_TOL,
            "regret {regret} > {MARGIN_TOL} at seq={seq} step={step}"
        );
        for &token in &top[..HEAD_K] {
            let &(_, hf) = reference
                .iter()
                .find(|&&(id, _)| id as usize == token)
                .context("engine head token absent from HF reference")?;
            self.delta
                .push(((logits[token] as f64 - lse) - hf as f64).abs() as f32);
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
        ensure!(mean <= MEAN_TOL && p99 <= P99_TOL, "HF golden gate failed");
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
    let size = config["hidden_size"]
        .as_u64()
        .zip(config["num_hidden_layers"].as_u64())
        .and_then(|(hidden, layers)| hf_golden::size_name(hidden, layers))
        .filter(|size| matches!(*size, "0.6b" | "4b"))
        .context("first release golden requires 0.6B or 4B")?;
    let g = Golden::load_path(&hf_golden::default_fixture_path(size));
    let mut model = Model::load(
        path,
        ModelOptions {
            device: 0,
            max_context: 512,
            slots: 4,
            chunk_size: 128,
        },
    )?;
    let mut stats = Stats::default();
    let mut repeat = Vec::new();
    for seq in 0..g.num_seqs {
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
        for step in 0..g.decode_len {
            let token = g.decode(seq, step);
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
    for base in (0..g.num_seqs).step_by(4) {
        let count = 4.min(g.num_seqs - base);
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
        for step in 0..g.decode_len {
            let input = (0..count)
                .map(|slot| TokenInput {
                    token: g.decode(base + slot, step),
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
    for base in (0..g.num_seqs).step_by(4) {
        let count = 4.min(g.num_seqs - base);
        // Interleave fixture lengths instead of grouping adjacent length bands.
        let seqs = (0..count)
            .map(|s| (base / 4 + s * (g.num_seqs / 4)) % g.num_seqs)
            .collect::<Vec<_>>();
        let prompts = seqs.iter().map(|&s| g.prompt(s)).collect::<Vec<_>>();
        let mut cursors = vec![0; count];
        for slot in 0..count {
            model.reset_slot(slot)?;
        }
        while (0..count).any(|s| cursors[s] < prompts[s].len() + g.decode_len) {
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
                } else if cursor < prompt_len + g.decode_len {
                    decode = true;
                    let step = cursor - prompt_len;
                    input.push(TokenInput {
                        token: g.decode(seqs[slot], step),
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
