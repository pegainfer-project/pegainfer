//! Fixed, pre-allocated buffers for the DFlash speculative *verify* forward.
//!
//! The verify forward runs a target prefill over each active request's `span =
//! num_speculative_tokens + 1` token block (see [`super::executor`]'s
//! `SpeculativeVerify` handler). The default [`Qwen3Model::batch_prefill`] path
//! allocates fresh GPU scratch every step (`PrefillBuffers::new`, the embedding
//! `HiddenStates`, `all_logits`, and the `PrefillPagedPlan` upload). That churns
//! `cuMemAllocAsync`/`cuMemFreeAsync` and — more importantly — hands CUDA Graph
//! capture moving pointers.
//!
//! [`VerifyGraphBuffers`] pre-allocates all of that once at the worst-case shape
//! (`max_batch * span` rows) and refills it in place each step, then captures the
//! forward into a **piecewise** CUDA Graph: the dense ops (embedding, RMSNorm,
//! every GEMM, SwiGLU, residual adds — ~84% of the per-step kernel-launch gap)
//! are captured per segment and replayed, while the attention op runs EAGER
//! between segments. Attention must stay eager because FlashInfer's paged-prefill
//! kernel fixes its KV-iteration count when the graph is recorded; with the verify
//! context growing every step, a captured attention would under-read KV and
//! corrupt later tokens. The dense segments bake their row count, so a captured
//! segment is only ever replayed at the exact full `batch_size * span` shape it
//! was recorded at; a step whose span is truncated near a request's output
//! budget falls back to eager (see [`Qwen3Model::batch_prefill_into`]).

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kv_cache::KvView;

use crate::batch_decode_buffers::BATCH_BUCKETS;
use crate::config::PREFILL_ATTENTION_CTA_TILE_Q;
use crate::prefill::PrefillBuffers;
use crate::weights::Qwen3Model;

/// All GPU scratch the verify forward needs, sized once for `max_batch * span`
/// rows and reused (in place) every step. Pointer-stable for CUDA Graph capture.
pub(crate) struct VerifyGraphBuffers {
    /// Per-layer projection/attention scratch (reused exactly as the allocating
    /// prefill path uses it).
    prefill_bufs: PrefillBuffers,
    /// Residual-stream hidden states `[hidden_dim, max_total_rows]`.
    hidden: HiddenStates,
    /// Captured target hidden states for the DFlash layers,
    /// `[hidden_size * num_capture_layers, max_total_rows]`.
    captured_hidden: HiddenStates,
    /// RMS-norm output feeding the lm_head GEMM `[hidden_dim, max_total_rows]`.
    all_logits_normed: HiddenStates,
    /// All-position logits `[vocab, max_total_rows]` (the verify forward's output).
    all_logits: HiddenStates,
    /// Device-resident concatenated verify tokens `[max_total_rows]`.
    token_ids_d: CudaSlice<u32>,
    /// Paged-attention plan, refilled in place each step.
    plan: PrefillPagedPlan,
    /// Piecewise CUDA Graphs: `graphs[bucket_idx][segment]`. Each bucket's verify
    /// forward is split into `num_layers + 1` dense segments (attention runs eager
    /// between them); a segment is captured once at its exact-bucket batch and
    /// replayed thereafter. Empty (`CudaGraphState::new()`) until first captured.
    graphs: Vec<Vec<CudaGraphState>>,
    /// `NumericPolicy` when these graphs were built; asserted unchanged at capture/replay. They bake
    /// the policy-selected GEMM algo and are keyed `(bucket, segment)` without policy — the same
    /// policy-key-trap the decode graphs guard.
    policy_at_construction: NumericPolicy,
    max_batch: usize,
    span: usize,
}

impl VerifyGraphBuffers {
    /// Allocate verify scratch for up to `max_batch` requests, each a fixed
    /// `span`-token block. `num_capture_layers` is the DFlash target-layer count
    /// (the captured-hidden buffer holds one `hidden_size` slice per layer).
    /// `max_total_pages` bounds the paged-attention page list; pass the KV
    /// pool's total block count for a guaranteed worst case.
    pub(crate) fn new(
        model: &Qwen3Model,
        max_batch: usize,
        span: usize,
        num_capture_layers: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "verify buffers need max_batch >= 1");
        anyhow::ensure!(span > 0, "verify buffers need span >= 1");
        let ctx = model.device_ctx();
        let hidden_dim = model.config().hidden_size;
        let q_dim = model.local_q_dim();
        let kv_dim = model.local_kv_dim();
        let inter_dim = model.local_intermediate_size();
        let vocab = model.config().vocab_size;
        let max_total_rows = max_batch * span;

        // Each request's `span` query tokens fan out to `span * group_size`
        // packed-QO rows; with a CTA tile of at least 1, that bounds tiles per
        // request. `max_batch * span * group_size` is the conservative ceiling.
        let group_size = model.local_num_attention_heads() / model.local_num_key_value_heads();
        let max_tiles = max_batch * span * group_size.max(1);

        Ok(Self {
            prefill_bufs: PrefillBuffers::new(
                ctx,
                hidden_dim,
                q_dim,
                kv_dim,
                inter_dim,
                max_total_rows,
            )?,
            hidden: HiddenStates::zeros(ctx, hidden_dim, max_total_rows)?,
            captured_hidden: HiddenStates::zeros(
                ctx,
                hidden_dim * num_capture_layers.max(1),
                max_total_rows,
            )?,
            all_logits_normed: HiddenStates::zeros(ctx, hidden_dim, max_total_rows)?,
            all_logits: HiddenStates::zeros(ctx, vocab, max_total_rows)?,
            token_ids_d: ctx.stream.alloc_zeros(max_total_rows)?,
            plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_total_rows,
                max_total_pages,
                max_batch,
                max_tiles,
            )?,
            // num_layers + 1 dense segments per bucket (attention is eager between).
            graphs: BATCH_BUCKETS
                .iter()
                .map(|_| {
                    (0..=model.config().num_hidden_layers)
                        .map(|_| CudaGraphState::new())
                        .collect()
                })
                .collect(),
            policy_at_construction: numeric_policy(),
            max_batch,
            span,
        })
    }

    /// Point every buffer's logical extent at `total_rows` (`<= max capacity`).
    /// Like [`PrefillBuffers`] / [`super::batch_decode_buffers`], this only moves
    /// `seq_len`; it never reallocates.
    fn set_rows(&mut self, total_rows: usize) {
        let cap = self.max_batch * self.span;
        assert!(
            total_rows <= cap,
            "verify total_rows {total_rows} exceeds capacity {cap}"
        );
        self.prefill_bufs.set_rows(total_rows);
        self.hidden.seq_len = total_rows;
        self.captured_hidden.seq_len = total_rows;
        self.all_logits_normed.seq_len = total_rows;
        self.all_logits.seq_len = total_rows;
    }

    /// All-position logits `[vocab, total_rows]` from the last forward.
    pub(crate) fn all_logits(&self) -> &HiddenStates {
        &self.all_logits
    }

    /// Mutable view, for truncating `seq_len` back to the unexpanded row count
    /// after the hedge's winner rows have been compacted onto chain A's
    /// offsets.
    pub(crate) fn captured_hidden_mut(&mut self) -> &mut HiddenStates {
        &mut self.captured_hidden
    }

    /// Captured target hidden states `[hidden_size * num_capture_layers, total_rows]`.
    pub(crate) fn captured_hidden(&self) -> &HiddenStates {
        &self.captured_hidden
    }
}

impl Qwen3Model {
    /// Fixed-buffer, piecewise-CUDA-Graph twin of [`Qwen3Model::batch_prefill`]
    /// for the DFlash verify forward. Issues the same per-op kernels as the
    /// allocating path (split into `forward_layer_pre_attn` / `forward_layer_attn`
    /// / `forward_layer_post_attn`), so the all-position logits and captured hidden
    /// states match it; only the buffer *source* differs (reused vs. freshly
    /// allocated) and the dense ops replay from a graph. Results land in `bufs`
    /// (`all_logits()` / `captured_hidden()`).
    ///
    /// `capture_layer_ids` must be the strictly-increasing DFlash target layers
    /// whose count matches the `num_capture_layers` `bufs` was built with.
    pub(crate) fn batch_prefill_into(
        &self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        kv_buffer: &CudaSlice<half::bf16>,
        layout: &KvLayout,
        capture_layer_ids: &[usize],
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<()> {
        let batch_size = prompts.len();
        anyhow::ensure!(
            batch_size == kv_views.len(),
            "verify prompts ({batch_size}) and kv_views ({}) length mismatch",
            kv_views.len()
        );
        anyhow::ensure!(
            batch_size <= bufs.max_batch,
            "verify batch {batch_size} exceeds buffer capacity {}",
            bufs.max_batch
        );
        anyhow::ensure!(
            capture_layer_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "verify capture layer ids must be strictly increasing"
        );
        anyhow::ensure!(
            capture_layer_ids
                .iter()
                .all(|&layer| layer < self.layers.len()),
            "verify capture layer id out of range"
        );
        let expected_capture_dim = self.config().hidden_size * capture_layer_ids.len().max(1);
        anyhow::ensure!(
            bufs.captured_hidden.hidden_dim == expected_capture_dim,
            "verify capture buffer dim {} does not match {} capture layers",
            bufs.captured_hidden.hidden_dim,
            capture_layer_ids.len(),
        );

        let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let total_tokens: usize = seq_lens.iter().sum();
        anyhow::ensure!(total_tokens > 0, "verify forward has no tokens");
        let start_positions: Vec<usize> = kv_views
            .iter()
            .zip(prompts.iter())
            .map(|(v, p)| v.seq_len() - p.len())
            .collect();

        bufs.set_rows(total_tokens);

        // --- prep: H2D staging that MUST stay outside the graph capture (CUDA
        // Graph forbids host round-trips in a captured segment). The embedding
        // kernel itself runs inside graph segment 0 and reads this buffer. ---
        let all_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        anyhow::ensure!(
            all_tokens.len() == total_tokens,
            "verify token concat {} != total_tokens {total_tokens}",
            all_tokens.len()
        );
        let ctx = self.device_ctx();
        // Stage the active tokens into the front of the fixed device buffer; the
        // embedding kernel reads exactly `total_tokens` ids from its base pointer,
        // so the unused tail is never touched.
        ctx.stream.memcpy_htod(&all_tokens, &mut bufs.token_ids_d)?;

        // Refill the paged plan in place (same host math as the allocating path).
        let page_indices: Vec<Vec<i32>> =
            kv_views.iter().map(|v| v.page_indices().to_vec()).collect();
        let last_page_lens: Vec<usize> = kv_views
            .iter()
            .map(pegainfer_kv_cache::KvView::last_page_len)
            .collect();
        bufs.plan.update_batch_with_cta_tile_q(
            ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config().head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        // --- piecewise CUDA Graph: dense ops captured per segment, attention
        // EAGER between segments. FlashInfer's prefill attention freezes its KV
        // iteration count at capture time (it tracks the growing context), so
        // capturing it corrupts later tokens; every other op is shape-stable in
        // the fixed `span`-row layout. Segments: [embed + L0.pre] [L0.attn]
        // [L0.post + L1.pre] [L1.attn] ... [L_last.post + lm_head]. ---
        let num_layers = self.layers.len();
        // Each captured dense segment bakes its row count (`total_tokens`) into
        // every kernel launch. A request near its output budget shortens its span
        // (the scheduler truncates the verify span to the remaining budget), so
        // `total_tokens` varies at a fixed `batch_size`. Replaying a segment that
        // was captured at one row count at a *different* count processes the wrong
        // number of rows and leaves the tail rows stale — silently corrupting the
        // verify logits of the trailing requests. So only the full, maximal
        // `batch_size * span` shape (every request contributing a full span) uses
        // the graph; any truncated step runs eager. This makes
        // capture-shape == replay-shape an invariant by construction.
        let full_shape = total_tokens == batch_size * bufs.span;
        match BATCH_BUCKETS.iter().position(|&b| b == batch_size) {
            Some(bidx) if full_shape => {
                if bufs.graphs[bidx].is_empty() || !bufs.graphs[bidx][0].is_captured() {
                    log::info!(
                        "verify graph capturing: bucket {batch_size} span {span}",
                        span = bufs.span
                    );
                }
                assert_eq!(
                    numeric_policy(),
                    bufs.policy_at_construction,
                    "NumericPolicy changed after the verify graphs were captured; they are keyed (bucket, segment) without policy, so build a fresh executor per policy (policy-key-trap)"
                );
                // Take the bucket's segment graphs out so the capture closures can
                // borrow `bufs` mutably; restore them after (even on error).
                let mut segs = std::mem::take(&mut bufs.graphs[bidx]);
                let result = (|| -> Result<()> {
                    segs[0].run_or_capture(ctx, || self.verify_seg_embed_pre(bufs))?;
                    self.verify_attn(0, kv_buffer, layout, bufs)?;
                    for (i, segment) in segs.iter_mut().enumerate().take(num_layers).skip(1) {
                        segment.run_or_capture(ctx, || {
                            self.verify_seg_post_pre(i, capture_layer_ids, bufs)
                        })?;
                        self.verify_attn(i, kv_buffer, layout, bufs)?;
                    }
                    segs[num_layers].run_or_capture(ctx, || {
                        self.verify_seg_post_logits(capture_layer_ids, bufs)
                    })?;
                    Ok(())
                })();
                bufs.graphs[bidx] = segs;
                result?;
            }
            // Off-bucket batch, or a truncated (non-full-span) step: run the same
            // segments eager (no capture).
            _ => {
                self.verify_seg_embed_pre(bufs)?;
                self.verify_attn(0, kv_buffer, layout, bufs)?;
                for i in 1..num_layers {
                    self.verify_seg_post_pre(i, capture_layer_ids, bufs)?;
                    self.verify_attn(i, kv_buffer, layout, bufs)?;
                }
                self.verify_seg_post_logits(capture_layer_ids, bufs)?;
            }
        }

        Ok(())
    }

    /// Graph segment 0: embedding (reads the staged `token_ids_d`) plus layer 0's
    /// pre-attention dense ops. Verify never uses LoRA, so the LoRA group is empty.
    fn verify_seg_embed_pre(&self, bufs: &mut VerifyGraphBuffers) -> Result<()> {
        self.get_embeddings_batch_into(&bufs.token_ids_d, &mut bufs.hidden)?;
        self.forward_layer_pre_attn(
            0,
            &self.layers[0],
            &bufs.hidden,
            &[],
            &mut bufs.prefill_bufs,
        )
    }

    /// Eager attention for layer `i` — kept out of every graph (see
    /// [`Self::forward_layer_attn`]). Touches only the fixed `prefill_bufs` and
    /// the refilled `plan`.
    fn verify_attn(
        &self,
        i: usize,
        kv_buffer: &CudaSlice<half::bf16>,
        layout: &KvLayout,
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<()> {
        self.forward_layer_attn(
            i,
            &self.layers[i],
            kv_buffer,
            layout,
            &bufs.plan,
            &mut bufs.prefill_bufs,
        )
    }

    /// Middle graph segment `i` (`1..num_layers`): finish layer `i-1`
    /// (post-attention dense + DFlash-context capture), then start layer `i`
    /// (pre-attention dense).
    ///
    /// The ping-pong swap inside `post_attn` is graph-safe regardless of layer
    /// parity: `run_or_capture` runs the closure (and thus the CPU-side swap) only
    /// on the capture step, so the captured graph bakes the exact buffer pointers
    /// for every op; replay just relaunches them. Each step's segment-0 embedding
    /// overwrites the same baked buffer that this segment's first `post_attn`
    /// reads, so no stale residual can leak across steps. The only live (eager) op
    /// between segments — attention — touches just `q/k/v_batch` / `attn_output`,
    /// which never participate in the swap, so it is independent of `hidden`'s
    /// logical pointer. Parity only decides which physical buffer holds the final
    /// hidden; that choice is baked into the last segment either way.
    fn verify_seg_post_pre(
        &self,
        i: usize,
        capture_layer_ids: &[usize],
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<()> {
        let prev = i - 1;
        self.forward_layer_post_attn(
            prev,
            &self.layers[prev],
            &mut bufs.hidden,
            &[],
            &mut bufs.prefill_bufs,
        )?;
        self.verify_capture_if_needed(prev, capture_layer_ids, bufs)?;
        self.forward_layer_pre_attn(
            i,
            &self.layers[i],
            &bufs.hidden,
            &[],
            &mut bufs.prefill_bufs,
        )
    }

    /// Final graph segment: finish the last layer (post-attention + capture), then
    /// the all-position logits (final RMSNorm + lm_head GEMM) into `all_logits`.
    fn verify_seg_post_logits(
        &self,
        capture_layer_ids: &[usize],
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<()> {
        let last = self.layers.len() - 1;
        self.forward_layer_post_attn(
            last,
            &self.layers[last],
            &mut bufs.hidden,
            &[],
            &mut bufs.prefill_bufs,
        )?;
        self.verify_capture_if_needed(last, capture_layer_ids, bufs)?;
        let ctx = self.device_ctx();
        ops::rms_norm_batch_into(
            ctx,
            &bufs.hidden,
            &self.norm,
            self.config().rms_norm_eps,
            &mut bufs.all_logits_normed,
        );
        ops::gemm_into(
            ctx,
            self.output_projection(),
            &bufs.all_logits_normed,
            &mut bufs.all_logits,
        );
        Ok(())
    }

    /// Copy layer `layer_idx`'s residual-stream hidden into the captured-hidden
    /// buffer when that layer is a DFlash target. `capture_layer_ids` is strictly
    /// increasing, so its position is the capture slot.
    fn verify_capture_if_needed(
        &self,
        layer_idx: usize,
        capture_layer_ids: &[usize],
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<()> {
        if let Some(slot) = capture_layer_ids.iter().position(|&l| l == layer_idx) {
            let hidden_size = self.config().hidden_size;
            ops::copy_hidden_rows_into(
                self.device_ctx(),
                &bufs.hidden,
                &mut bufs.captured_hidden,
                slot * hidden_size,
            )?;
        }
        Ok(())
    }
}
