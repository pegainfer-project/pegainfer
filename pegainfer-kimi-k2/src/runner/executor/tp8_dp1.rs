use std::cmp::Ordering;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use super::ForwardExecutor;
use crate::config::KIMI_K2_DENSE_LAYERS;
use crate::config::KIMI_K2_LAYERS;
use crate::config::KIMI_K2_MOE_LAYERS;
use crate::config::KIMI_K2_VOCAB;
use crate::runner::worker::KimiKvStepPages;
use crate::runner::worker::KimiOneTokenForwardReport;
use crate::runner::worker::KimiRankWeightLoadReport;
use crate::runner::worker::KimiRankWorker;
use crate::runner::worker::KimiRowOptions;

pub(in crate::runner) struct Tp8Dp1ForwardExecutor {
    pub(in crate::runner) workers: Vec<KimiRankWorker>,
    pub(in crate::runner) weight_reports: Vec<KimiRankWeightLoadReport>,
}

impl ForwardExecutor for Tp8Dp1ForwardExecutor {
    fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()> {
        self.ensure_decode_arena(decode_batch_size)
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        slot: usize,
        decode_batch_size: usize,
        cached_tokens: usize,
        _ep_max_seq_len: usize,
        kv_pages: &KimiKvStepPages,
        row: KimiRowOptions,
        _seed: u64,
    ) -> Result<KimiOneTokenForwardReport> {
        if self.workers.is_empty() {
            bail!("Kimi TP8 executor has no rank workers");
        }
        ensure_no_logprobs_tp8(row.logprobs.is_some())?;
        ensure_greedy_tp8(&row.sampling)?;
        // Every rank replicates the KV pool with identical geometry, so the
        // same page assignment is broadcast to all eight workers.
        let responses = self
            .workers
            .iter()
            .map(|worker| {
                worker.forward_prompt_next_token_async(
                    input_ids.to_vec(),
                    slot,
                    decode_batch_size,
                    cached_tokens,
                    0,
                    kv_pages.clone(),
                    KimiRowOptions::default(),
                    0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut reports = Vec::with_capacity(responses.len());
        for response in responses {
            reports.push(
                response.recv().map_err(|_| {
                    anyhow::anyhow!("Kimi-K2 rank worker dropped forward response")
                })??,
            );
        }
        let expected_input = *input_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt report validation requires input token"))?;
        for report in &reports {
            validate_one_token_report(report, "prompt", slot, expected_input)?;
        }
        reports
            .into_iter()
            .max_by(|left, right| {
                left.local_top_logit_f32
                    .partial_cmp(&right.local_top_logit_f32)
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("Kimi runtime produced no rank forward reports"))
    }

    fn forward_decode_batch(
        &self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
        kv_pages: &KimiKvStepPages,
        rows: &[KimiRowOptions],
        _seed: u64,
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        if self.workers.is_empty() {
            bail!("Kimi TP8 executor has no rank workers");
        }
        if token_ids.is_empty() {
            bail!("Kimi batch decode requires at least one token");
        }
        if token_ids.len() != append_positions.len() || token_ids.len() != slots.len() {
            bail!(
                "Kimi batch decode input mismatch: tokens={}, positions={}, slots={}",
                token_ids.len(),
                append_positions.len(),
                slots.len()
            );
        }
        ensure_no_logprobs_tp8(rows.iter().any(|r| r.logprobs.is_some()))?;
        for row in rows {
            ensure_greedy_tp8(&row.sampling)?;
        }
        let responses = self
            .workers
            .iter()
            .map(|worker| {
                worker.forward_decode_batch_next_tokens_async(
                    token_ids.to_vec(),
                    append_positions.to_vec(),
                    slots.to_vec(),
                    decode_batch_size,
                    kv_pages.clone(),
                    vec![KimiRowOptions::default(); token_ids.len()],
                    0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rank_reports = Vec::with_capacity(responses.len());
        for response in responses {
            rank_reports.push(response.recv().map_err(|_| {
                anyhow::anyhow!("Kimi-K2 rank worker dropped batch decode response")
            })??);
        }
        let shard_count = rank_reports.len();
        ensure!(
            shard_count == self.workers.len(),
            "Kimi batch decode received {} rank report sets for {} workers",
            shard_count,
            self.workers.len()
        );
        for (rank_idx, reports) in rank_reports.iter().enumerate() {
            ensure!(
                reports.len() == token_ids.len(),
                "Kimi rank {rank_idx} returned {} decode reports for {} active rows",
                reports.len(),
                token_ids.len()
            );
            for (row, report) in reports.iter().enumerate() {
                validate_one_token_report(report, "decode", slots[row], token_ids[row])?;
            }
        }
        let mut selected = Vec::with_capacity(token_ids.len());
        for row in 0..token_ids.len() {
            let best = rank_reports
                .iter()
                .map(|reports| reports[row].clone())
                .max_by(|left, right| {
                    left.local_top_logit_f32
                        .partial_cmp(&right.local_top_logit_f32)
                        .unwrap_or(Ordering::Equal)
                })
                .ok_or_else(|| anyhow::anyhow!("Kimi runtime produced no report for row {row}"))?;
            selected.push(best);
        }
        Ok(selected)
    }

    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn gpu_weight_ready_count(&self) -> usize {
        self.weight_reports
            .iter()
            .filter(|r| r.loaded_to_gpu && r.typed_view_validated)
            .count()
    }
}

impl Tp8Dp1ForwardExecutor {
    fn ensure_decode_arena(&self, decode_batch_size: usize) -> Result<()> {
        if self.workers.is_empty() {
            bail!("Kimi TP8 executor has no rank workers");
        }
        let responses = self
            .workers
            .iter()
            .map(|worker| worker.ensure_decode_arena_async(decode_batch_size))
            .collect::<Result<Vec<_>>>()?;
        for (rank, response) in responses.into_iter().enumerate() {
            response
                .recv()
                .map_err(|_| anyhow::anyhow!("Kimi-K2 rank {rank} dropped arena response"))?
                .with_context(|| {
                    format!("Kimi-K2 rank {rank} decode arena bs{decode_batch_size}")
                })?;
        }
        Ok(())
    }
}

impl Drop for Tp8Dp1ForwardExecutor {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            let _ = worker.shutdown();
        }
    }
}

fn ensure_no_logprobs_tp8(requested: bool) -> Result<()> {
    ensure!(
        !requested,
        "Kimi TP8 path does not support logprobs yet: each rank holds a vocab \
         shard and a shard-local logsumexp is not the global one — needs a \
         cross-rank merge (#236)"
    );
    Ok(())
}

fn ensure_greedy_tp8(sampling: &pegainfer_frontend::sampler::SamplingParams) -> Result<()> {
    ensure!(
        sampling.is_greedy(),
        "Kimi TP8 path does not support sampling yet: each rank holds a vocab \
         shard and cannot sample the global distribution — use TP1/DP8 or \
         temperature=0 (#237, #226)"
    );
    Ok(())
}

fn validate_one_token_report(
    report: &KimiOneTokenForwardReport,
    phase: &str,
    expected_slot: usize,
    expected_input_token: u32,
) -> Result<()> {
    ensure!(
        report.batch_slot == expected_slot,
        "Kimi {phase} rank {} report slot mismatch: got {}, expected {}",
        report.rank,
        report.batch_slot,
        expected_slot
    );
    ensure!(
        report.input_token_id == expected_input_token,
        "Kimi {phase} rank {} report input token mismatch: got {}, expected {}",
        report.rank,
        report.input_token_id,
        expected_input_token
    );
    ensure!(
        report.vocab_rows > 0 && report.vocab_start + report.vocab_rows <= KIMI_K2_VOCAB,
        "Kimi {phase} rank {} report invalid vocab shard: start={}, rows={}, vocab={}",
        report.rank,
        report.vocab_start,
        report.vocab_rows,
        KIMI_K2_VOCAB
    );
    ensure!(
        (report.local_next_token_id as usize) < report.vocab_rows,
        "Kimi {phase} rank {} local token {} outside shard rows {}",
        report.rank,
        report.local_next_token_id,
        report.vocab_rows
    );
    ensure!(
        report.local_next_token_global_id as usize
            == report.vocab_start + report.local_next_token_id as usize,
        "Kimi {phase} rank {} global token mismatch: got {}, expected {}",
        report.rank,
        report.local_next_token_global_id,
        report.vocab_start + report.local_next_token_id as usize
    );
    ensure!(
        report.local_top_logit_f32.is_finite(),
        "Kimi {phase} rank {} report has non-finite top logit {}",
        report.rank,
        report.local_top_logit_f32
    );
    ensure!(
        report.dense_layers_executed == KIMI_K2_DENSE_LAYERS,
        "Kimi {phase} rank {} dense layer count mismatch: got {}, expected {}",
        report.rank,
        report.dense_layers_executed,
        KIMI_K2_DENSE_LAYERS
    );
    ensure!(
        report.moe_layers_executed == KIMI_K2_MOE_LAYERS,
        "Kimi {phase} rank {} MoE layer count mismatch: got {}, expected {}",
        report.rank,
        report.moe_layers_executed,
        KIMI_K2_MOE_LAYERS
    );
    ensure!(
        report.dense_layers_executed + report.moe_layers_executed == KIMI_K2_LAYERS,
        "Kimi {phase} rank {} executed layer count mismatch: dense {} + moe {} != {}",
        report.rank,
        report.dense_layers_executed,
        report.moe_layers_executed,
        KIMI_K2_LAYERS
    );
    Ok(())
}
