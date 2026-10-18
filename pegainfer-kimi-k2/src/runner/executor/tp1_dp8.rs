use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use super::DP_MAX_BATCH_PER_RANK;
use super::ForwardExecutor;
use crate::config::KIMI_K2_VOCAB;
use crate::runner::worker::KimiKvStepPages;
use crate::runner::worker::KimiOneTokenForwardReport;
use crate::runner::worker::KimiRankWeightLoadReport;
use crate::runner::worker::KimiRankWorker;
use crate::runner::worker::KimiRowOptions;

pub(in crate::runner) struct Tp1Dp8ForwardExecutor {
    pub(in crate::runner) worker: KimiRankWorker,
    pub(in crate::runner) weight_report: KimiRankWeightLoadReport,
}

impl ForwardExecutor for Tp1Dp8ForwardExecutor {
    fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()> {
        ensure!(
            decode_batch_size <= DP_MAX_BATCH_PER_RANK,
            "Kimi TP1 decode batch size {decode_batch_size} exceeds per-rank arena capacity {DP_MAX_BATCH_PER_RANK}"
        );
        Ok(())
    }

    fn forward_prefill(
        &self,
        input_ids: &[u32],
        slot: usize,
        decode_batch_size: usize,
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: &KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    ) -> Result<KimiOneTokenForwardReport> {
        let response = self.worker.forward_prompt_next_token_async(
            input_ids.to_vec(),
            slot,
            decode_batch_size,
            cached_tokens,
            ep_max_seq_len,
            kv_pages.clone(),
            row,
            seed,
        )?;
        let report = response
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 TP1 worker dropped forward response"))??;
        validate_tp1_report(&report, "prefill", slot)?;
        Ok(report)
    }

    fn forward_decode_batch(
        &self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
        kv_pages: &KimiKvStepPages,
        rows: &[KimiRowOptions],
        seed: u64,
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        if token_ids.is_empty() {
            bail!("Kimi TP1 batch decode requires at least one token");
        }
        if token_ids.len() != append_positions.len() || token_ids.len() != slots.len() {
            bail!(
                "Kimi TP1 batch decode input mismatch: tokens={}, positions={}, slots={}",
                token_ids.len(),
                append_positions.len(),
                slots.len()
            );
        }
        let response = self.worker.forward_decode_batch_next_tokens_async(
            token_ids.to_vec(),
            append_positions.to_vec(),
            slots.to_vec(),
            decode_batch_size,
            kv_pages.clone(),
            rows.to_vec(),
            seed,
        )?;
        let reports = response
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 TP1 worker dropped batch decode response"))??;
        ensure!(
            reports.len() == token_ids.len(),
            "Kimi TP1 returned {} decode reports for {} active rows",
            reports.len(),
            token_ids.len()
        );
        for (row, report) in reports.iter().enumerate() {
            validate_tp1_report(report, "decode", slots[row])?;
        }
        Ok(reports)
    }

    fn worker_count(&self) -> usize {
        1
    }

    fn gpu_weight_ready_count(&self) -> usize {
        usize::from(self.weight_report.loaded_to_gpu && self.weight_report.typed_view_validated)
    }
}

impl Drop for Tp1Dp8ForwardExecutor {
    fn drop(&mut self) {
        let _ = self.worker.shutdown();
    }
}

fn validate_tp1_report(
    report: &KimiOneTokenForwardReport,
    phase: &str,
    expected_slot: usize,
) -> Result<()> {
    ensure!(
        report.batch_slot == expected_slot,
        "Kimi TP1 {phase} rank {} report slot mismatch: got {}, expected {}",
        report.rank,
        report.batch_slot,
        expected_slot
    );
    ensure!(
        report.vocab_start == 0 && report.vocab_rows == KIMI_K2_VOCAB,
        "Kimi TP1 {phase} rank {} expected full vocab: start={}, rows={}, vocab={}",
        report.rank,
        report.vocab_start,
        report.vocab_rows,
        KIMI_K2_VOCAB
    );
    ensure!(
        report.local_top_logit_f32.is_finite(),
        "Kimi TP1 {phase} rank {} report has non-finite top logit {}",
        report.rank,
        report.local_top_logit_f32
    );
    Ok(())
}
