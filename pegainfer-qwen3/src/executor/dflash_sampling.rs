//! Sampled DFlash2 verification. Proposal probabilities stay on the worker lane.

use anyhow::Result;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::BatchSamplingRow;
use pegainfer_kernels::ops::SparseVerifyRequest;
use pegainfer_kernels::ops::sparse_speculative_sample;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::LocalQwen3Lane;
use crate::speculative::VerifyRequestResult;
use crate::speculative::VerifyStepItem;
use crate::speculative::build_verify_results;

impl LocalQwen3Lane {
    pub(super) fn sample_selector_verify(
        &mut self,
        requests: &[VerifyStepItem],
        logits: &HiddenStates,
    ) -> Result<Option<Vec<VerifyRequestResult>>> {
        let Some(dflash) = self.dflash.as_ref() else {
            return Ok(None);
        };
        if dflash.proposal_rows.is_empty() {
            return Ok(None);
        }

        let block_size = dflash.model.block_size();
        let mut rows = Vec::new();
        let mut sampled = Vec::new();
        let mut request_indices = Vec::new();
        let mut uniforms = Vec::new();
        let mut offset = 0;
        for (i, req) in requests.iter().enumerate() {
            if let Some(&(draft_row, seed)) = dflash.proposal_rows.get(&req.request_id) {
                sampled.push(SparseVerifyRequest {
                    logits_offset: rows.len(),
                    draft_row,
                    num_drafts: req.token_ids.len() - 1,
                });
                request_indices.push(i);
                rows.extend(
                    (offset..offset + req.token_ids.len()).map(|row| BatchSamplingRow {
                        row,
                        temperature: req.params.temperature,
                        top_k: req.params.top_k,
                        top_p: req.params.top_p,
                        min_p: req.params.min_p,
                    }),
                );
                // A different stream from the proposal: acceptance must not
                // reuse the uniform that selected a candidate.
                let mut rng = StdRng::seed_from_u64(pegainfer_sample::mix_seed(seed, 1));
                uniforms.extend((0..block_size).map(|_| rng.random::<f32>()));
            }
            offset += req.token_ids.len();
        }

        // Mixed batches still use the shared batched argmax. The all-sampled
        // path skips it entirely, preparing target probabilities only once.
        let mut results = if sampled.len() < requests.len() {
            let greedy = SamplingParams::default();
            let params = vec![&greedy; offset];
            let tokens = self.select_step_tokens(logits, &params, 0)?;
            build_verify_results(requests, &tokens)?
        } else {
            requests
                .iter()
                .map(|req| VerifyRequestResult {
                    request_id: req.request_id,
                    matched_draft_tokens: 0,
                    accepted_tokens: Vec::new(),
                })
                .collect()
        };

        if rows.len() > self.sample_scratch.max_rows() {
            self.sample_scratch = pegainfer_sample::SampleScratch::with_selection_width(
                self.model.device_ctx(),
                self.model.config().logits_rows,
                self.model.config().decodable_vocab,
                rows.len(),
            )?;
        }
        let dflash = self
            .dflash
            .as_mut()
            .expect("sampled proposal requires DFlash");
        let (candidate_ids, probs, draft_ids) = dflash
            .scratch
            .selector_proposal()
            .expect("sampled proposal requires selector scratch");
        let accepted = sparse_speculative_sample(
            self.model.device_ctx(),
            logits.as_ref(),
            &rows,
            &sampled,
            candidate_ids,
            probs,
            draft_ids,
            &uniforms,
            dflash
                .sampling
                .as_mut()
                .expect("sampled proposal requires verifier scratch"),
            self.sample_scratch.sampling_scratch(),
        )?;
        for (&i, (count, token)) in request_indices.iter().zip(accepted) {
            let req = &requests[i];
            let mut tokens = req.token_ids[1..=count].to_vec();
            tokens.push(token);
            results[i] = VerifyRequestResult {
                request_id: req.request_id,
                matched_draft_tokens: count,
                accepted_tokens: tokens,
            };
        }
        Ok(Some(results))
    }
}
