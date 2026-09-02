//! DSpark Markov head (Phase 1): semi-autoregressive draft sampling layered on
//! the shared DFlash backbone.
//!
//! DSpark = DFlash backbone + a low-rank Markov head. The backbone forward, the
//! verify span, and the optimistic KV transaction are reused verbatim from
//! [`crate::dflash`]; the *only* change is how draft tokens are selected from the
//! backbone's block logits. Where DFlash takes an independent argmax per block
//! position, DSpark adds a bigram-style logit bias `B(prev) = w2(w1[prev])` and
//! samples the block left-to-right, so each draft conditions on the previous one
//! (semi-autoregressive). In greedy decoding this is lossless — the bias only
//! reshapes the *draft proposal*; every token is still confirmed by the target
//! verify — but the proposals are higher quality, lifting accepted length.
//!
//! The released checkpoint (`deepseek-ai/dspark_qwen3_4b_block7`) stores, on top
//! of the DFlash backbone tensors:
//!   markov_head.markov_w1.weight  [vocab, rank]  prev-token embedding lookup
//!   markov_head.markov_w2.weight  [vocab, rank]  Linear(rank -> vocab) bias proj
//!   confidence_head.proj.{weight,bias}           Phase 2 (unused here)
//!   embed_tokens.weight / lm_head.weight         byte-identical to target (reused)
//!
//! Phase 1 ignores the confidence head: every block is verified in full (no
//! confidence-scheduled truncation). See docs/models/qwen3/dspark-integration.md.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::markov_step_argmax_into;
use pegainfer_kernels::ops::markov_step_argmax_partials_len;

use crate::config::DFlashConfig;

pub(crate) const MARKOV_W1_TENSOR: &str = "markov_head.markov_w1.weight";
pub(crate) const MARKOV_W2_TENSOR: &str = "markov_head.markov_w2.weight";

/// DSpark Markov head: a low-rank, previous-token-conditioned logit bias.
///
/// `w1` (`[vocab, rank]`) is an embedding table — row `t` is the rank-`r` code of
/// token `t`; `w2` (`[vocab, rank]`) projects that code back to vocab as the
/// additive bias `B(t) = w2 · w1[t]`. Both are stored row-major `[out, in]`, so
/// the gather is an embedding lookup and the projection is a plain GEMM.
pub(crate) struct MarkovHead {
    w1: DeviceMatrix,
    w2: DeviceMatrix,
}

impl MarkovHead {
    pub(crate) fn new(rank: usize, w1: DeviceMatrix, w2: DeviceMatrix) -> Result<Self> {
        anyhow::ensure!(rank > 0, "DSpark markov rank must be > 0");
        anyhow::ensure!(
            w1.cols == rank && w2.cols == rank,
            "DSpark markov weight rank mismatch: w1.cols={}, w2.cols={}, rank={}",
            w1.cols,
            w2.cols,
            rank
        );
        anyhow::ensure!(
            w1.rows == w2.rows,
            "DSpark markov w1/w2 vocab mismatch: {} vs {}",
            w1.rows,
            w2.rows
        );
        Ok(Self { w1, w2 })
    }

    /// Sample `block_size` draft tokens per request, left-to-right with the
    /// Markov bias.
    ///
    /// `base_logits` are the backbone draft logits `[rows*block_size, vocab]`
    /// (request-major: request `i` owns rows `[i*block_size, (i+1)*block_size)`),
    /// `current_tokens` are the per-request anchors (the verified token each block
    /// extends). Returns the `rows*block_size` request-major drafts, anchor-first:
    /// token `k` of request `i` is the draft read from backbone position `k`
    /// (position 0 included — DSpark's block input is anchor-first, so position 0
    /// already predicts the first draft, unlike DFlash which discards it).
    ///
    /// The loop is sequential across the `block_size` steps (step `k+1`'s prev
    /// token is step `k`'s output) but batched across requests; each step is one
    /// embedding gather + one GEMM + one strided argmax-with-bias kernel.
    pub(crate) fn sample_block(
        &self,
        ctx: &DeviceContext,
        base_logits: &HiddenStates,
        current_tokens: &[u32],
        block_size: usize,
        scratch: &mut MarkovScratch,
    ) -> Result<Vec<u32>> {
        let rows = current_tokens.len();
        anyhow::ensure!(rows > 0, "DSpark markov sample needs active requests");
        anyhow::ensure!(block_size > 0, "DSpark markov block_size must be > 0");
        let vocab = base_logits.hidden_dim;
        anyhow::ensure!(
            base_logits.seq_len == rows * block_size,
            "DSpark markov base logits rows {} != rows*block_size {}",
            base_logits.seq_len,
            rows * block_size
        );
        anyhow::ensure!(
            vocab == self.w2.rows,
            "DSpark markov vocab {} != w2 rows {}",
            vocab,
            self.w2.rows
        );
        scratch.activate(rows, vocab)?;

        // prev = anchors (only the active prefix; the kernels read the first
        // `rows` ids/rows of the max-batch buffers).
        {
            let mut prev_dst = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(current_tokens, &mut prev_dst)?;
        }

        let mut sampled = vec![0u32; rows * block_size];
        for step in 0..block_size {
            // w1emb[rows, rank] = markov_w1[prev]
            ops::embedding_batch(ctx, &self.w1, &scratch.prev_tokens, &mut scratch.w1emb)?;
            // bias[rows, vocab] = w1emb @ w2^T
            ops::gemm_into(ctx, &self.w2, &scratch.w1emb, &mut scratch.bias);
            // next[i] = argmax_v ( base_logits[(i*B+step), v] + bias[i, v] )
            markov_step_argmax_into(
                ctx,
                base_logits,
                &scratch.bias,
                block_size,
                step,
                rows,
                &mut scratch.partial_values,
                &mut scratch.partial_indices,
                &mut scratch.next_tokens,
                &mut scratch.sampled_tokens,
            )?;
            std::mem::swap(&mut scratch.prev_tokens, &mut scratch.next_tokens);
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block_size);
        sampled.copy_from_slice(&ctx.stream.clone_dtoh(&sampled_view)?);
        Ok(sampled)
    }

    /// Bytes occupied by the Markov head weights + sample scratch, for the memory
    /// reservation. `0` when the head is disabled.
    pub(crate) fn reservation_bytes(config: &DFlashConfig, max_decode_batch_size: usize) -> usize {
        const BF16: usize = 2;

        if !config.uses_markov_head() {
            return 0;
        }
        let vocab = config.vocab_size;
        let rank = config.markov_rank();
        let weights = 2 * vocab * rank * BF16;
        let scratch = MarkovScratch::bytes(vocab, rank, config.block_size, max_decode_batch_size);
        weights + scratch
    }
}

/// Scratch for the Markov sample loop, allocated once for the max decode batch.
/// `bias` is the per-step `[rows, vocab]` logit bias; `partial_*` back the
/// two-stage argmax; `prev`/`next` ping-pong the per-step token ids on device;
/// `sampled_tokens` stores the full request-major block so the host reads once.
pub(crate) struct MarkovScratch {
    max_batch: usize,
    w1emb: HiddenStates,
    bias: HiddenStates,
    partial_values: CudaSlice<f32>,
    partial_indices: CudaSlice<i32>,
    prev_tokens: CudaSlice<u32>,
    next_tokens: CudaSlice<u32>,
    sampled_tokens: CudaSlice<u32>,
}

impl MarkovScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_decode_batch_size > 0,
            "DSpark markov scratch needs a non-zero batch size"
        );
        let vocab = config.vocab_size;
        let rank = config.markov_rank();
        let partials = markov_step_argmax_partials_len(max_decode_batch_size, vocab);
        let sampled = max_decode_batch_size * config.block_size;
        Ok(Self {
            max_batch: max_decode_batch_size,
            w1emb: HiddenStates::zeros(ctx, rank, max_decode_batch_size)?,
            bias: HiddenStates::zeros(ctx, vocab, max_decode_batch_size)?,
            partial_values: ctx.stream.alloc_zeros(partials)?,
            partial_indices: ctx.stream.alloc_zeros(partials)?,
            prev_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            next_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            sampled_tokens: ctx.stream.alloc_zeros(sampled)?,
        })
    }

    /// Point the dense scratch at the active `rows` prefix. Allocated for the max
    /// decode batch, so this only shrinks `seq_len`; it never reallocates.
    fn activate(&mut self, rows: usize, vocab: usize) -> Result<()> {
        anyhow::ensure!(
            rows <= self.max_batch,
            "DSpark markov batch {} exceeds scratch capacity {}",
            rows,
            self.max_batch
        );
        anyhow::ensure!(
            self.bias.hidden_dim == vocab,
            "DSpark markov scratch vocab {} != base vocab {}",
            self.bias.hidden_dim,
            vocab
        );
        self.w1emb.seq_len = rows;
        self.bias.seq_len = rows;
        Ok(())
    }

    fn bytes(vocab: usize, rank: usize, block_size: usize, max_decode_batch_size: usize) -> usize {
        const BF16: usize = 2;
        let partials = markov_step_argmax_partials_len(max_decode_batch_size, vocab);
        let w1emb = max_decode_batch_size * rank * BF16;
        let bias = max_decode_batch_size * vocab * BF16;
        let partial_bytes = partials * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());
        let tokens = (2 * max_decode_batch_size + max_decode_batch_size * block_size)
            * std::mem::size_of::<u32>();
        w1emb + bias + partial_bytes + tokens
    }
}
