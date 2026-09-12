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
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::panic_message;
use pegainfer_kernels::tensor::StreamOverrideGuard;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvView;

use super::batch_decode_graph::BatchDecodeGraphState;
use super::recurrent_state::RecurrentState;
use super::weights::Qwen35Model;
use crate::batch_decode::DecodeGraphUse;

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
        views: &[KvView],
        recurrent_states: &mut [&mut RecurrentState],
        kv_buffer: &KvBuffer,
    ) -> Result<HiddenStates> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "batch prefill requires prompts");
        anyhow::ensure!(n == views.len(), "prompts / KV views len mismatch");
        anyhow::ensure!(
            n == recurrent_states.len(),
            "prompts / recurrent states len mismatch"
        );
        let mut last_hiddens = Vec::with_capacity(n);
        for i in 0..n {
            last_hiddens.push(self.prefill_last_hidden(
                prompts[i],
                &views[i],
                kv_buffer,
                recurrent_states[i],
            )?);
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
        views: &[KvView],
        kv_buffer: &KvBuffer,
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<HiddenStates> {
        let cu_stream = stream.cu_stream();
        let original_stream = std::mem::replace(&mut self.ctx.stream, stream);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _stream_override = unsafe { StreamOverrideGuard::activate_prefill(cu_stream) };
            self.batch_prefill_logits(prompts, views, recurrent_states, kv_buffer)
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
        prefill_views: &[KvView],
        prefill_recurrent_states: &mut [&mut RecurrentState],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        kv_buffer: &KvBuffer,
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
                prefill_views,
                prefill_recurrent_states,
                kv_buffer,
            )?)
        };

        // ── Decode phase ──────────────────────────────────────────────────────
        let decoded = if decode_tokens.is_empty() {
            false
        } else {
            self.batch_decode_graph(
                decode_tokens,
                decode_views,
                kv_buffer,
                graph_state,
                DecodeGraphUse::Serve,
            )?;
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
    use pegainfer_core::tensor::HiddenStates;
    use pegainfer_kv_cache::KvCacheManager;

    use super::*;
    use crate::prefix_cache::Qwen35PrefixCache;

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

    fn run_decode_path(model: &Qwen35Model, unified: bool) -> (Vec<u32>, Vec<u32>) {
        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
        let prompts = [&prompt_a[..], &prompt_b[..]];
        let num_steps = 5;
        let manager =
            KvCacheManager::from_buffer(model.kv_buffer().clone(), model.kv_buffer().num_blocks())
                .unwrap();
        let cache = Qwen35PrefixCache::new(manager, 0).unwrap();
        let mut kvs = vec![
            cache.pool().new_request(prompt_a.clone(), num_steps, None),
            cache.pool().new_request(prompt_b.clone(), num_steps, None),
        ];
        for (kv, prompt) in kvs.iter_mut().zip(prompts) {
            cache.schedule_prefill(kv, prompt.len()).unwrap();
        }
        let views = kvs
            .iter()
            .zip(prompts)
            .map(|(kv, prompt)| cache.prefill_view(kv, prompt.len()))
            .collect::<Vec<_>>();
        let mut rec_states = [
            RecurrentState::new(&model.ctx, &model.config, model.geometry).unwrap(),
            RecurrentState::new(&model.ctx, &model.config, model.geometry).unwrap(),
        ];
        let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();
        let mut gs = model
            .create_batch_decode_graph_state(cache.pool().padding_block_id())
            .unwrap();
        let first_logits = if unified {
            model
                .unified_step(
                    &prompts,
                    &views,
                    &mut rec_refs,
                    &[],
                    &[],
                    cache.buffer(),
                    &mut gs,
                )
                .unwrap()
                .prefill_logits
                .unwrap()
        } else {
            model
                .batch_prefill_logits(&prompts, &views, &mut rec_refs, cache.buffer())
                .unwrap()
        };
        let first = greedy_sample_batch(model, &first_logits, 2);
        for (kv, token) in kvs.iter_mut().zip(&first) {
            cache.apply_prefill(kv, Some(*token)).unwrap();
        }
        gs.copy_state_to_slot(&model.ctx, &rec_states[0], 0)
            .unwrap();
        gs.copy_state_to_slot(&model.ctx, &rec_states[1], 1)
            .unwrap();
        let mut tokens_a = vec![first[0]];
        let mut tokens_b = vec![first[1]];
        for _ in 1..num_steps {
            for kv in &mut kvs {
                cache.schedule_decode(kv).unwrap();
            }
            let views = kvs
                .iter()
                .map(|kv| cache.decode_view(kv))
                .collect::<Vec<_>>();
            let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
            if unified {
                model
                    .unified_step(&[], &[], &mut [], &tids, &views, cache.buffer(), &mut gs)
                    .unwrap();
            } else {
                model
                    .batch_decode_graph(
                        &tids,
                        &views,
                        cache.buffer(),
                        &mut gs,
                        DecodeGraphUse::Serve,
                    )
                    .unwrap();
            }
            let next = greedy_sample_batch(model, &gs.buffers.logits, 2);
            for (kv, token) in kvs.iter_mut().zip(&next) {
                cache.apply_decode(kv, *token).unwrap();
            }
            tokens_a.push(next[0]);
            tokens_b.push(next[1]);
        }
        (tokens_a, tokens_b)
    }

    /// Verify that unified_step decode output matches batch_decode_graph standalone.
    #[test]
    fn unified_step_decode_matches_graph_decode() {
        let Some(model_path) =
            crate::test_fixture::model_path_or_skip("unified_step_decode_matches_graph_decode")
        else {
            return;
        };
        let model = Qwen35Model::from_safetensors(&model_path, 0, 2, 0).unwrap();
        let ref_tokens = run_decode_path(&model, false);
        let unified_tokens = run_decode_path(&model, true);

        assert_eq!(
            unified_tokens, ref_tokens,
            "unified_step decode mismatch:\n  unified: {:?}\n  ref:     {:?}",
            unified_tokens, ref_tokens
        );
    }
}
