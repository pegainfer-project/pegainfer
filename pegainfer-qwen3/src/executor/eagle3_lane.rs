//! Worker-side EAGLE-3 draft lane: the draft model plus per-request draft state.
//!
//! Like the DFlash lane it lives on the worker thread next to the target model,
//! because the drafter reads the target's embeddings and the hidden states the
//! target captures at prefill and verify. Those hidden states stay private to
//! this lane; the draft/verify boundary is a pure token span.

use std::collections::HashMap;

use anyhow::Result;
use pegainfer_core::tensor::HiddenStates;

use super::DraftMeta;
use super::LocalQwen3Lane;
use super::PrefillStepItem;
use super::RequestId;
use super::dflash_prefill::prefill_capture_supported;
use crate::eagle3::EAGLE3_CHAIN_LENGTH;
use crate::eagle3::EAGLE3_MAX_BATCH;
use crate::eagle3::Eagle3DraftModel;
use crate::eagle3::Eagle3RequestState;
use crate::eagle3::Eagle3Scratch;
use crate::eagle3::aux_hidden_state_layers;
use crate::speculative::DraftRequestResult;
use crate::speculative::DraftResult;
use crate::speculative::DraftStepItem;
use crate::speculative::VerifyRequestResult;
use crate::speculative::VerifyStepItem;

pub(super) struct Eagle3LaneState {
    model: Eagle3DraftModel,
    requests: HashMap<RequestId, Eagle3RequestState>,
    scratch: Eagle3Scratch,
    /// Target layers whose post-layer hidden states feed the drafter's `fc`.
    aux_layers: [usize; 3],
}

impl Eagle3LaneState {
    pub(super) fn aux_layers(&self) -> &[usize] {
        &self.aux_layers
    }
}

/// Whether a prefill step seeds the drafter for `req`. v1 is greedy only, and
/// `prefill_prompt` needs the whole prompt in one chunk; anything else decodes
/// without speculation.
fn eagle3_prefill_eligible(req: &PrefillStepItem) -> bool {
    prefill_capture_supported(req)
        && req.params.is_greedy()
        && req.chunk_start == 0
        && req.chunk_tokens == req.prompt_tokens.len()
        && req.chunk_tokens >= 2
}

impl LocalQwen3Lane {
    /// Load the EAGLE-3 draft model into this lane (primary rank only).
    pub(super) fn load_eagle3(&mut self, draft_path: &str) -> Result<DraftMeta> {
        let ctx = self.model.device_ctx();
        let model = Eagle3DraftModel::from_safetensors_for_target(ctx, draft_path, &self.model)?;
        let aux_layers = aux_hidden_state_layers(self.model.config().num_hidden_layers)?;
        let scratch = model.new_scratch(ctx)?;
        let meta = DraftMeta {
            block_size: EAGLE3_CHAIN_LENGTH + 1,
            num_spec_tokens: EAGLE3_CHAIN_LENGTH,
            max_position_embeddings: model.config.max_position_embeddings,
            max_draft_batch: Some(EAGLE3_MAX_BATCH),
        };
        self.eagle3 = Some(Eagle3LaneState {
            model,
            requests: HashMap::new(),
            scratch,
            aux_layers,
        });
        Ok(meta)
    }

    pub(super) fn should_capture_eagle3_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        self.eagle3.is_some() && requests.iter().any(eagle3_prefill_eligible)
    }

    /// Seed the drafter from the prompt's captured target hidden states.
    /// Returns the requests that now have draft state.
    pub(super) fn record_prefill_eagle3_context(
        &mut self,
        requests: &[PrefillStepItem],
        capture_requested: bool,
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<Vec<RequestId>> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::ensure!(
                !capture_requested,
                "EAGLE-3 prefill capture was requested but no hidden states were returned"
            );
            return Ok(Vec::new());
        };
        anyhow::ensure!(
            capture_requested,
            "EAGLE-3 prefill hidden states were returned without a capture request"
        );
        let Some(eagle3) = self.eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 prefill context record requested without EAGLE-3");
        };
        let expected_tokens: usize = requests.iter().map(|req| req.chunk_tokens).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 prefill captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let ctx = self.model.device_ctx();
        let mut captured_requests = Vec::new();
        let mut token_offset = 0usize;
        for req in requests {
            eagle3.requests.remove(&req.request_id);
            if eagle3_prefill_eligible(req) {
                // Room for every committed token plus one full round past it.
                let max_cache_len =
                    (req.prompt_tokens.len() + req.max_output_tokens + EAGLE3_CHAIN_LENGTH + 1)
                        .min(eagle3.model.config.max_position_embeddings);
                let mut state = eagle3.model.new_request_state(ctx, max_cache_len)?;
                eagle3.model.prefill_prompt(
                    &self.model,
                    &mut state,
                    captured_hidden,
                    token_offset,
                    req.as_slice(),
                )?;
                eagle3.requests.insert(req.request_id, state);
                captured_requests.push(req.request_id);
            }
            token_offset += req.chunk_tokens;
        }
        Ok(captured_requests)
    }

    /// Draft one chain per request: the verify span `[current_token, drafts…]`.
    pub(super) fn execute_eagle3_draft(
        &mut self,
        requests: &[DraftStepItem],
    ) -> Result<DraftResult> {
        let Some(eagle3) = self.eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 draft requested but EAGLE-3 is not loaded");
        };
        anyhow::ensure!(
            !requests.is_empty() && requests.len() <= EAGLE3_MAX_BATCH,
            "EAGLE-3 drafts 1..={EAGLE3_MAX_BATCH} requests per round, got {}",
            requests.len()
        );
        let Eagle3LaneState {
            model,
            requests: states,
            scratch,
            ..
        } = eagle3;
        let mut outputs = Vec::with_capacity(requests.len());
        for req in requests {
            let state = states
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing EAGLE-3 state for {:?}", req.request_id))?;
            let drafts = model.chain_round(
                &self.model,
                state,
                scratch,
                req.current_token,
                EAGLE3_CHAIN_LENGTH,
            )?;
            let mut token_ids = Vec::with_capacity(drafts.len() + 1);
            token_ids.push(req.current_token);
            token_ids.extend(drafts);
            outputs.push(DraftRequestResult {
                request_id: req.request_id,
                token_ids,
            });
        }
        Ok(DraftResult { requests: outputs })
    }

    /// Rebuild each request's draft KV over its committed span from the verify
    /// capture, and move its boundary feature to the last committed position.
    pub(super) fn record_verify_eagle3_context(
        &mut self,
        requests: &[VerifyStepItem],
        results: &[VerifyRequestResult],
        captured_hidden: &HiddenStates,
    ) -> Result<()> {
        let Some(eagle3) = self.eagle3.as_mut() else {
            anyhow::bail!("EAGLE-3 verify context record requested without EAGLE-3");
        };
        anyhow::ensure!(
            requests.len() == results.len(),
            "EAGLE-3 verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "EAGLE-3 verify captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let mut token_offset = 0usize;
        for (req, result) in requests.iter().zip(results) {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "EAGLE-3 verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            let state = eagle3.requests.get_mut(&req.request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing EAGLE-3 state after verify for {:?}",
                    req.request_id
                )
            })?;
            eagle3.model.rebuild_after_verify(
                &self.model,
                state,
                captured_hidden,
                token_offset,
                &req.token_ids,
                result.matched_draft_tokens,
            )?;
            token_offset += req.token_ids.len();
        }
        Ok(())
    }

    pub(super) fn drop_eagle3_request(&mut self, request_id: RequestId) {
        if let Some(eagle3) = self.eagle3.as_mut() {
            eagle3.requests.remove(&request_id);
        }
    }
}
