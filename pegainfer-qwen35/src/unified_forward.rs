//! Qwen3.5 batch prefill and unified step (prefill + decode combined).
//!
//! Linear attention (GDR chunkwise) does not have an efficient batched prefill
//! kernel, so `batch_prefill` runs each request's prefill serially. Full-attention
//! layers also run per-request to reuse the existing paged prefill path.
//!
//! `unified_step` combines:
//!   1. Serial `batch_prefill` for new requests entering the batch.
//!   2. `batch_decode_graph` for existing decode requests (CUDA Graph for
//!      compiled GQA groups; eager prefill fallback for uncompiled ones).

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaStream;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::panic_message;
use pegainfer_kernels::tensor::StreamOverrideGuard;

use super::batch_decode_graph::BatchDecodeGraphState;
use super::recurrent_state::RecurrentState;
use super::weights::Qwen35Model;

pub(crate) struct UnifiedStepOutput {
    pub(crate) prefill_logits: Option<HiddenStates>,
    pub(crate) decoded: bool,
}

impl Qwen35Model {
    /// Prefill `n` prompts sequentially, updating each request's KV and recurrent state.
    ///
    /// Returns batched last-token logits `[selection_vocab, n]` in request order.
    /// Requests are independent — there is no cross-request batching in the prefill pass.
    pub(crate) fn batch_prefill_logits(
        &self,
        prompts: &[&[u32]],
        kv_states: &mut [KvState],
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<HiddenStates> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "batch_prefill requires at least one prompt");
        anyhow::ensure!(n == kv_states.len(), "prompts / kv_states len mismatch");
        anyhow::ensure!(
            n == recurrent_states.len(),
            "prompts / recurrent_states len mismatch"
        );

        let mut last_hiddens = Vec::with_capacity(n);
        for i in 0..n {
            let last_hidden =
                self.prefill_last_hidden(prompts[i], &mut kv_states[i], recurrent_states[i])?;
            debug_assert_eq!(
                last_hidden.len, self.config.hidden_size,
                "Qwen3.5 prefill last hidden row must match request {i}"
            );
            last_hiddens.push(last_hidden);
        }
        self.batch_last_hidden_logits(&last_hiddens)
    }

    /// Run the complete prefill path on `stream`, including allocations, copies,
    /// kernels, and stream-ordered frees. The model is scheduler-thread owned, so
    /// the temporary context swap cannot race another host caller.
    pub(crate) fn batch_prefill_logits_on_stream(
        &mut self,
        stream: Arc<CudaStream>,
        prompts: &[&[u32]],
        kv_states: &mut [KvState],
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<HiddenStates> {
        let cu_stream = stream.cu_stream();
        let original_stream = std::mem::replace(&mut self.ctx.stream, stream);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _stream_override = unsafe { StreamOverrideGuard::activate_prefill(cu_stream) };
            self.batch_prefill_logits(prompts, kv_states, recurrent_states)
        }));
        self.ctx.stream = original_stream;

        match result {
            Ok(result) => result,
            Err(panic) => anyhow::bail!(
                "Qwen3.5 async prefill panicked: {}",
                panic_message(panic.as_ref())
            ),
        }
    }

    /// Unified step: prefill new requests and decode existing requests in one call.
    ///
    /// Prefill is run serially per-request (GDR chunkwise per request). Decode runs
    /// via CUDA Graph on the pre-allocated `graph_state` for the decode batch.
    ///
    /// Either `prefill_prompts` or `decode_tokens` may be empty (but not both).
    ///
    /// Prefill logits are returned as `[selection_vocab, n_prefill]` in request order.
    /// Decode logits remain in `graph_state.buffers.logits`; callers sample from
    /// that batched buffer directly to avoid per-request extraction.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [KvState],
        prefill_recurrent_states: &mut [&mut RecurrentState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<UnifiedStepOutput> {
        anyhow::ensure!(
            !prefill_prompts.is_empty() || !decode_tokens.is_empty(),
            "unified_step: both prefill and decode are empty"
        );

        // ── Prefill phase ─────────────────────────────────────────────────────
        let prefill_logits = if prefill_prompts.is_empty() {
            None
        } else {
            Some(self.batch_prefill_logits(
                prefill_prompts,
                prefill_kv_states,
                prefill_recurrent_states,
            )?)
        };

        // ── Decode phase ──────────────────────────────────────────────────────
        let decoded = if decode_tokens.is_empty() {
            false
        } else {
            self.batch_decode_graph(decode_tokens, decode_kv_states, graph_state)?;
            true
        };

        Ok(UnifiedStepOutput {
            prefill_logits,
            decoded,
        })
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_core::kv_pool::KvState;
    use pegainfer_core::tensor::HiddenStates;

    use super::*;

    fn greedy_sample_batch(model: &Qwen35Model, logits: &HiddenStates, rows: usize) -> Vec<u32> {
        let params = vec![pegainfer_frontend::sampler::SamplingParams::default(); rows];
        let params_refs: Vec<&pegainfer_frontend::sampler::SamplingParams> =
            params.iter().collect();
        let mut scratch =
            pegainfer_sample::SampleScratch::new(&model.ctx, model.config.selection_vocab, rows)
                .unwrap();
        let steps = vec![0u64; params_refs.len()];
        pegainfer_sample::select_batch(&model.ctx, logits, &params_refs, &steps, 0, &mut scratch)
            .unwrap()
    }

    /// Verify that unified_step decode output matches batch_decode_graph standalone.
    #[test]
    fn unified_step_decode_matches_graph_decode() {
        let Some(model_path) =
            crate::test_fixture::model_path_or_skip("unified_step_decode_matches_graph_decode")
        else {
            return;
        };
        let model = Qwen35Model::from_safetensors(&model_path, 0, 2).unwrap();

        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
        let num_steps = 5;

        // --- Reference: standalone batch_decode_graph ---
        let ref_tokens = {
            let prompts_ref: Vec<&[u32]> = vec![&prompt_a, &prompt_b];
            let mut kv_states: Vec<KvState> = vec![model.alloc_kv(), model.alloc_kv()];
            let mut rec_states: Vec<RecurrentState> = vec![
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
            ];
            let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();
            let first_logits = model
                .batch_prefill_logits(&prompts_ref, &mut kv_states, &mut rec_refs)
                .unwrap();
            let first = greedy_sample_batch(&model, &first_logits, 2);
            let first_a = first[0];
            let first_b = first[1];

            let mut gs = model.create_batch_decode_graph_state().unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_states[0], 0)
                .unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_states[1], 1)
                .unwrap();
            model.ctx.sync().unwrap();

            let mut tokens_a = vec![first_a];
            let mut tokens_b = vec![first_b];
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();

            for _ in 1..num_steps {
                let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
                model
                    .batch_decode_graph(&tids, &mut kv_refs, &mut gs)
                    .unwrap();
                let next = greedy_sample_batch(&model, &gs.buffers.logits, 2);
                tokens_a.push(next[0]);
                tokens_b.push(next[1]);
            }
            (tokens_a, tokens_b)
        };

        // --- unified_step path ---
        let unified_tokens = {
            let prompts_ref: Vec<&[u32]> = vec![&prompt_a, &prompt_b];
            let mut kv_states: Vec<KvState> = vec![model.alloc_kv(), model.alloc_kv()];
            let mut rec_states: Vec<RecurrentState> = vec![
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
            ];
            let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();

            let output = model
                .unified_step(
                    &prompts_ref,
                    &mut kv_states,
                    &mut rec_refs,
                    &[],
                    &mut [],
                    &mut model.create_batch_decode_graph_state().unwrap(),
                )
                .unwrap();
            let prefill_logits = output.prefill_logits.as_ref().unwrap();
            let first = greedy_sample_batch(&model, prefill_logits, 2);
            let first_a = first[0];
            let first_b = first[1];

            // Transfer prefill states to decode graph slots
            let mut gs = model.create_batch_decode_graph_state().unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_states[0], 0)
                .unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_states[1], 1)
                .unwrap();

            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();

            let mut tokens_a = vec![first_a];
            let mut tokens_b = vec![first_b];

            for _ in 1..num_steps {
                let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
                let output = model
                    .unified_step(&[], &mut [], &mut [], &tids, &mut kv_refs, &mut gs)
                    .unwrap();
                assert!(output.decoded);
                let next = greedy_sample_batch(&model, &gs.buffers.logits, 2);
                tokens_a.push(next[0]);
                tokens_b.push(next[1]);
            }
            (tokens_a, tokens_b)
        };

        assert_eq!(
            unified_tokens, ref_tokens,
            "unified_step decode mismatch:\n  unified: {:?}\n  ref:     {:?}",
            unified_tokens, ref_tokens
        );
    }
}
