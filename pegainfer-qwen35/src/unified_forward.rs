//! Qwen3.5 batch prefill and unified step (prefill + decode combined).
//!
//! Linear attention (GDR chunkwise) does not have an efficient batched prefill
//! kernel, so one request's prefill is a pass of its own. Passes for the requests
//! of one chunk run concurrently on alternate streams: a single request's chain
//! leaves the GPU idle between kernels and at the tail of its last GEMM, and
//! interleaving independent chains fills both. Full-attention layers also run
//! per-request to reuse the existing paged prefill path.
//!
//! `unified_step` combines:
//!   1. `batch_prefill` for new requests entering the batch.
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
use crate::prefill::SCRATCH_ESTIMATE_SEQ;
use crate::prefill_buffers::GdrChunkwiseScratch35;

/// Alternate streams used to overlap the requests of one prefill chunk. Four
/// covers the four 1024-token prompts the scheduler's default budget
/// (`--max-prefill-tokens 4096`) admits per step; a chunk with shorter prompts
/// reuses them in order.
const PREFILL_STREAMS: usize = 4;

pub(crate) struct UnifiedStepOutput {
    pub(crate) prefill_logits: Option<HiddenStates>,
    pub(crate) decoded: bool,
}

impl Qwen35Model {
    /// Prefill `n` prompts, updating each request's KV and recurrent state.
    ///
    /// Requests are independent, so their passes run on [`Self::prefill_streams`]
    /// when more than one is queued; the streams are joined back to the context
    /// stream before the last-token logits are computed. A single request, and TP
    /// (whose collectives must stay on one stream), keep the one-at-a-time pass.
    ///
    /// Returns batched last-token logits `[selection_vocab, n]` in request order.
    pub(crate) fn batch_prefill_logits(
        &mut self,
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

        // TP collectives assume one stream, and a lone request has nothing to
        // interleave: both keep the pass on the context stream.
        let overlap = n > 1 && !self.is_tp();
        if overlap {
            self.ensure_prefill_streams()?;
        }
        // Startup reserves one chunk of GDR scratch for prefill, so the passes in
        // flight have to fit inside that reservation together: a step of long
        // prompts overlaps fewer of them.
        let widest = prompts
            .iter()
            .map(|prompt| prompt.len().min(SCRATCH_ESTIMATE_SEQ))
            .max()
            .unwrap_or(1)
            .max(1);
        let pool = if overlap {
            let reserved = GdrChunkwiseScratch35::estimate_bytes(
                &self.config,
                self.geometry,
                SCRATCH_ESTIMATE_SEQ,
            );
            let per_pass =
                GdrChunkwiseScratch35::estimate_bytes(&self.config, self.geometry, widest);
            (reserved / per_pass.max(1)).min(self.prefill_streams.len())
        } else {
            0
        };

        // Request KV and recurrent state were staged on the context stream; order
        // them ahead of every pass without blocking the host.
        for stream in &self.prefill_streams[..pool] {
            stream
                .join(&self.ctx.stream)
                .map_err(|err| anyhow::anyhow!("join Qwen3.5 prefill stream: {err}"))?;
        }

        // Every pass that starts is joined back before any error escapes: the
        // caller reverts this chunk's KV schedule as soon as it sees the error, and
        // a pass still running past that point would keep writing pages the next
        // request can already be handed.
        let mut passes = Vec::with_capacity(n);
        for i in 0..n {
            if pool <= 1 {
                passes.push(self.prefill_last_hidden(
                    prompts[i],
                    &views[i],
                    kv_buffer,
                    recurrent_states[i],
                ));
                continue;
            }
            let stream = Arc::clone(&self.prefill_streams[i % pool]);
            passes.push(self.with_stream(stream, "overlapped prefill", |model| {
                model.prefill_last_hidden(prompts[i], &views[i], kv_buffer, recurrent_states[i])
            }));
        }
        for stream in &self.prefill_streams[..pool] {
            self.ctx
                .stream
                .join(stream)
                .map_err(|err| anyhow::anyhow!("join Qwen3.5 prefill result: {err}"))?;
        }

        let mut last_hiddens = Vec::with_capacity(n);
        for pass in passes {
            last_hiddens.push(pass?);
        }
        let logits = self.batch_last_hidden_logits(&last_hiddens)?;
        // Each row above was allocated on a prefill stream, so its stream-ordered
        // free follows this return. The context stream reads the rows for the
        // logits and the device context keeps no cross-stream events, so give the
        // reads back to the pool before the rows can be dropped.
        if pool > 1 {
            for stream in &self.prefill_streams[..pool] {
                stream
                    .join(&self.ctx.stream)
                    .map_err(|err| anyhow::anyhow!("join Qwen3.5 prefill read: {err}"))?;
            }
        }
        drop(last_hiddens);
        Ok(logits)
    }

    fn ensure_prefill_streams(&mut self) -> Result<()> {
        if !self.prefill_streams.is_empty() {
            return Ok(());
        }
        for _ in 0..PREFILL_STREAMS {
            let stream = self
                .ctx
                .ctx
                .new_stream()
                .map_err(|err| anyhow::anyhow!("create Qwen3.5 prefill stream: {err}"))?;
            self.prefill_streams.push(stream);
        }
        Ok(())
    }

    /// Run `body` with the context stream and the kernel stream override pointed
    /// at `stream`, allocations included.
    ///
    /// The context stream is swapped for the pass so every buffer it allocates is
    /// also released on the stream that used it: `cudaMallocAsync` orders a free
    /// against the stream it is queued on, and a free queued on the context
    /// stream would hand this pass's still-in-flight memory to the next request.
    /// The model is scheduler-thread owned, so the temporary swap cannot race
    /// another host caller.
    fn with_stream<T>(
        &mut self,
        stream: Arc<CudaStream>,
        what: &str,
        body: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let cu_stream = stream.cu_stream();
        let original_stream = std::mem::replace(&mut self.ctx.stream, stream);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _stream_override = unsafe { StreamOverrideGuard::activate_prefill(cu_stream) };
            body(self)
        }));
        self.ctx.stream = original_stream;

        match result {
            Ok(result) => result,
            Err(panic) => {
                anyhow::bail!("Qwen3.5 {what} panicked: {}", panic_message(panic.as_ref()))
            }
        }
    }

    /// Run the complete prefill path on `stream`, including allocations, copies,
    /// kernels, and stream-ordered frees.
    pub(crate) fn batch_prefill_logits_on_stream(
        &mut self,
        stream: Arc<CudaStream>,
        prompts: &[&[u32]],
        views: &[KvView],
        kv_buffer: &KvBuffer,
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<HiddenStates> {
        self.with_stream(stream, "async prefill", |model| {
            model.batch_prefill_logits(prompts, views, recurrent_states, kv_buffer)
        })
    }

    /// Unified step: prefill new requests and decode existing requests in one call.
    ///
    /// The chunk's prefills run per-request (GDR chunkwise per request), the
    /// requests overlapping on alternate streams. Decode runs via CUDA Graph on
    /// the pre-allocated `graph_state` for the decode batch.
    ///
    /// Either `prefill_prompts` or `decode_tokens` may be empty (but not both).
    ///
    /// Prefill logits are returned as `[selection_vocab, n_prefill]` in request order.
    /// Decode logits remain in `graph_state.buffers.logits`; callers sample from
    /// that batched buffer directly to avoid per-request extraction.
    pub(crate) fn unified_step(
        &mut self,
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
    use pegainfer_core::tensor::DeviceContext;
    use pegainfer_core::tensor::HiddenStates;
    use pegainfer_kv_cache::KvCacheManager;

    use super::*;
    use crate::prefix_cache::Qwen35PrefixCache;

    fn greedy_sample_batch(model: &Qwen35Model, logits: &HiddenStates, rows: usize) -> Vec<u32> {
        let params = vec![pegainfer_frontend::sampler::SamplingParams::default(); rows];
        let params_refs: Vec<&pegainfer_frontend::sampler::SamplingParams> =
            params.iter().collect();
        let mut scratch = pegainfer_sample::SampleScratch::with_selection_width(
            &model.ctx,
            model.config.selection_vocab,
            model.config.decodable_vocab,
            rows,
        )
        .unwrap();
        let steps = vec![0u64; params_refs.len()];
        pegainfer_sample::select_batch(&model.ctx, logits, &params_refs, &steps, 0, &mut scratch)
            .unwrap()
    }

    /// The alignment pad columns must come out of the model already masked, on
    /// both the prefill and the decode logits path. A logits site that ran the
    /// output projection without the mask leaves them selectable, and nothing
    /// else in the suite would notice: the pad rows are trained embeddings with
    /// plausible logits, so the failure shows up as an undecodable id on the
    /// wire rather than a crash.
    #[test]
    fn pad_columns_are_suppressed_on_the_model_logits_paths() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "pad_columns_are_suppressed_on_the_model_logits_paths",
        ) else {
            return;
        };
        let mut model = Qwen35Model::from_safetensors(&model_path, 0, 2, 0).unwrap();

        let decodable = model.config.decodable_vocab;
        let selection = model.config.selection_vocab;
        if decodable == selection {
            eprintln!("checkpoint needs no alignment pad; nothing to assert");
            return;
        }

        let pad_tail_is_suppressed = |ctx: &DeviceContext, logits: &HiddenStates, label: &str| {
            assert_eq!(logits.hidden_dim, selection, "{label}: arena width");
            let row = crate::ops::extract_vec(ctx, logits, 0).unwrap();
            let row = row.to_host(ctx).unwrap();
            for (id, value) in row.iter().enumerate().take(selection).skip(decodable) {
                assert!(
                    value.is_sign_negative() && value.is_infinite(),
                    "{label}: pad id {id} survived selection (want -inf, got {value})"
                );
            }
            assert!(
                row[..decodable].iter().any(|value| value.is_finite()),
                "{label}: decodable vocab came out entirely -inf"
            );
        };

        let prompt = vec![9707, 374, 220, 17];
        let prompt_refs = [&prompt[..]];
        let manager =
            KvCacheManager::from_buffer(model.kv_buffer().clone(), model.kv_buffer().num_blocks())
                .unwrap();
        let cache = Qwen35PrefixCache::new(manager, 0).unwrap();
        let mut kv = cache.pool().new_request(prompt.clone(), 2, None);
        cache.schedule_prefill(&mut kv, prompt.len()).unwrap();
        let views = [cache.prefill_view(&kv, prompt.len())];
        let mut rec_states =
            [RecurrentState::new(&model.ctx, &model.config, model.geometry).unwrap()];
        let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();
        let prefill_logits = model
            .batch_prefill_logits(&prompt_refs, &views, &mut rec_refs, cache.buffer())
            .unwrap();
        pad_tail_is_suppressed(&model.ctx, &prefill_logits, "prefill");

        let next = greedy_sample_batch(&model, &prefill_logits, 1)[0];
        cache.apply_prefill(&mut kv, Some(next)).unwrap();
        let mut graph_state = model
            .create_batch_decode_graph_state(cache.pool().padding_block_id())
            .unwrap();
        graph_state
            .copy_state_to_slot(&model.ctx, &rec_states[0], 0)
            .unwrap();
        cache.schedule_decode(&mut kv).unwrap();
        let decode_views = [cache.decode_view(&kv)];
        model
            .batch_decode_graph(
                &[next],
                &decode_views,
                cache.buffer(),
                &mut graph_state,
                DecodeGraphUse::Serve,
            )
            .unwrap();
        pad_tail_is_suppressed(&model.ctx, &graph_state.buffers.logits, "decode");
        let decoded = greedy_sample_batch(&model, &graph_state.buffers.logits, 1)[0];
        cache.apply_decode(&mut kv, decoded).unwrap();
    }

    fn run_decode_path(model: &mut Qwen35Model, unified: bool) -> (Vec<u32>, Vec<u32>) {
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
        let mut model = Qwen35Model::from_safetensors(&model_path, 0, 2, 0).unwrap();
        let ref_tokens = run_decode_path(&mut model, false);
        let unified_tokens = run_decode_path(&mut model, true);

        assert_eq!(
            unified_tokens, ref_tokens,
            "unified_step decode mismatch:\n  unified: {:?}\n  ref:     {:?}",
            unified_tokens, ref_tokens
        );
    }
}
