//! DSpark Markov head (Phase 1): semi-autoregressive draft sampling layered on
//! the shared DFlash backbone.
//!
//! DSpark = DFlash backbone + a low-rank Markov head. The backbone forward, the
//! verify span, and the optimistic KV transaction are reused verbatim from
//! [`crate::dflash`]; the only change on the default path is how draft tokens
//! are selected from the backbone's block logits. (With `PEGAINFER_SPEC_HEDGE`
//! set, the verify pass additionally carries alternative chains over lane-owned
//! scratch pages — opt-in, and still on the same transaction interface; see
//! `try_execute_hedged_verify`.) Where DFlash takes an independent argmax per block
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
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::hedge_ladder_force_into;
use pegainfer_kernels::ops::markov_step_argmax_into;
use pegainfer_kernels::ops::markov_step_argmax_mapped_into;
use pegainfer_kernels::ops::markov_step_argmax_partials_len;
use pegainfer_kernels::ops::markov_step_top2_into;

use crate::config::DFlashConfig;
use crate::sizing;

fn markov_graph_enabled() -> bool {
    std::env::var_os("DSPARK_NO_MARKOV_GRAPH").is_none()
}

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
        if markov_graph_enabled() && scratch.chain_warm[rows - 1] {
            let MarkovScratch {
                chain_graphs,
                w1emb,
                bias,
                partial_values,
                partial_indices,
                prev_tokens,
                next_tokens,
                sampled_tokens,
                ..
            } = scratch;
            chain_graphs[rows - 1].run_or_capture(ctx, || {
                for step in 0..block_size {
                    let (input, output): (&CudaSlice<u32>, &mut CudaSlice<u32>) = if step % 2 == 0 {
                        (&*prev_tokens, &mut *next_tokens)
                    } else {
                        (&*next_tokens, &mut *prev_tokens)
                    };
                    ops::embedding_batch(ctx, &self.w1, input, w1emb)?;
                    ops::gemm_into_checked(ctx, &self.w2, w1emb, bias)?;
                    markov_step_argmax_into(
                        ctx,
                        base_logits,
                        bias,
                        block_size,
                        step,
                        rows,
                        partial_values,
                        partial_indices,
                        output,
                        sampled_tokens,
                    )?;
                }
                Ok(())
            })?;
        } else {
            for step in 0..block_size {
                let input_is_prev = step % 2 == 0;
                {
                    let input = if input_is_prev {
                        &scratch.prev_tokens
                    } else {
                        &scratch.next_tokens
                    };
                    ops::embedding_batch(ctx, &self.w1, input, &mut scratch.w1emb)?;
                }
                ops::gemm_into_checked(ctx, &self.w2, &scratch.w1emb, &mut scratch.bias)?;
                let output = if input_is_prev {
                    &mut scratch.next_tokens
                } else {
                    &mut scratch.prev_tokens
                };
                markov_step_argmax_into(
                    ctx,
                    base_logits,
                    &scratch.bias,
                    block_size,
                    step,
                    rows,
                    &mut scratch.partial_values,
                    &mut scratch.partial_indices,
                    output,
                    &mut scratch.sampled_tokens,
                )?;
            }
            scratch.chain_warm[rows - 1] = true;
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block_size);
        sampled.copy_from_slice(&ctx.stream.clone_dtoh(&sampled_view)?);
        Ok(sampled)
    }

    /// Chain-A sample loop that also captures the exact Markov runner-up at
    /// each `branch_positions` step (given the chain's true parent at that
    /// step). Returns chain A's host tokens (request-major, needed for verify
    /// span assembly); the runner-ups stay device-resident in
    /// `MarkovScratch::runner_tokens` stripes (`[j * max_batch + i]` =
    /// request `i`'s rank-2 alternative at `branch_positions[j]`) for the
    /// ladder to consume. `branch_positions` must be sorted, deduped, and
    /// `< block_size`.
    pub(crate) fn sample_block_with_runners(
        &self,
        ctx: &DeviceContext,
        base_logits: &HiddenStates,
        current_tokens: &[u32],
        block_size: usize,
        branch_positions: &[usize],
        scratch: &mut MarkovScratch,
    ) -> Result<Vec<u32>> {
        let rows = current_tokens.len();
        anyhow::ensure!(rows > 0, "DSpark markov sample needs active requests");
        let vocab = base_logits.hidden_dim;
        anyhow::ensure!(
            base_logits.seq_len == rows * block_size,
            "DSpark markov base logits rows {} != rows*block_size {}",
            base_logits.seq_len,
            rows * block_size
        );
        // Checked, not just documented: an unsorted or out-of-block position
        // walks the runner stripes out of range, which would panic inside the
        // graph-capture closure below rather than return.
        anyhow::ensure!(
            branch_positions.windows(2).all(|w| w[0] < w[1])
                && branch_positions.last().is_none_or(|&p| p < block_size),
            "DSpark branch positions must be sorted, deduped and < block_size: {branch_positions:?}"
        );
        scratch.activate(rows, vocab)?;
        {
            let mut prev_dst = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(current_tokens, &mut prev_dst)?;
        }
        let max_batch = scratch.max_batch;
        if markov_graph_enabled() && scratch.runner_warm[rows - 1] {
            let MarkovScratch {
                runner_graphs,
                w1emb,
                bias,
                partial_values,
                partial_indices,
                partial_values2,
                partial_indices2,
                top2_tokens,
                runner_tokens,
                prev_tokens,
                next_tokens,
                sampled_tokens,
                ..
            } = scratch;
            runner_graphs[rows - 1].run_or_capture(ctx, || {
                let mut snap = 0usize;
                for step in 0..block_size {
                    let input_is_prev = step % 2 == 0;
                    {
                        let input: &CudaSlice<u32> = if input_is_prev {
                            prev_tokens
                        } else {
                            next_tokens
                        };
                        ops::embedding_batch(ctx, &self.w1, input, w1emb)?;
                    }
                    ops::gemm_into_checked(ctx, &self.w2, w1emb, bias)?;
                    let output: &mut CudaSlice<u32> = if input_is_prev {
                        next_tokens
                    } else {
                        prev_tokens
                    };
                    if branch_positions.contains(&step) {
                        markov_step_top2_into(
                            ctx,
                            base_logits,
                            bias,
                            block_size,
                            step,
                            rows,
                            partial_values2,
                            partial_indices2,
                            partial_values,
                            partial_indices,
                            output,
                            sampled_tokens,
                            top2_tokens,
                        )?;
                        let src = top2_tokens.slice(..rows);
                        let mut dst =
                            runner_tokens.slice_mut(snap * max_batch..snap * max_batch + rows);
                        ctx.stream.memcpy_dtod(&src, &mut dst)?;
                        snap += 1;
                    } else {
                        markov_step_argmax_into(
                            ctx,
                            base_logits,
                            bias,
                            block_size,
                            step,
                            rows,
                            partial_values,
                            partial_indices,
                            output,
                            sampled_tokens,
                        )?;
                    }
                }
                Ok(())
            })?;
        } else {
            let mut snap = 0usize;
            for step in 0..block_size {
                let input_is_prev = step % 2 == 0;
                {
                    let input = if input_is_prev {
                        &scratch.prev_tokens
                    } else {
                        &scratch.next_tokens
                    };
                    ops::embedding_batch(ctx, &self.w1, input, &mut scratch.w1emb)?;
                }
                ops::gemm_into_checked(ctx, &self.w2, &scratch.w1emb, &mut scratch.bias)?;
                if branch_positions.contains(&step) {
                    // One expression, three disjoint field borrows: splitting
                    // it would re-borrow `scratch` a second time.
                    let (output, top2, sampled) = if input_is_prev {
                        (
                            &mut scratch.next_tokens,
                            &mut scratch.top2_tokens,
                            &mut scratch.sampled_tokens,
                        )
                    } else {
                        (
                            &mut scratch.prev_tokens,
                            &mut scratch.top2_tokens,
                            &mut scratch.sampled_tokens,
                        )
                    };
                    markov_step_top2_into(
                        ctx,
                        base_logits,
                        &scratch.bias,
                        block_size,
                        step,
                        rows,
                        &mut scratch.partial_values2,
                        &mut scratch.partial_indices2,
                        &mut scratch.partial_values,
                        &mut scratch.partial_indices,
                        output,
                        sampled,
                        top2,
                    )?;
                    let src = scratch.top2_tokens.slice(..rows);
                    let mut dst = scratch
                        .runner_tokens
                        .slice_mut(snap * max_batch..snap * max_batch + rows);
                    ctx.stream.memcpy_dtod(&src, &mut dst)?;
                    snap += 1;
                } else {
                    let output = if input_is_prev {
                        &mut scratch.next_tokens
                    } else {
                        &mut scratch.prev_tokens
                    };
                    markov_step_argmax_into(
                        ctx,
                        base_logits,
                        &scratch.bias,
                        block_size,
                        step,
                        rows,
                        &mut scratch.partial_values,
                        &mut scratch.partial_indices,
                        output,
                        &mut scratch.sampled_tokens,
                    )?;
                }
            }
            scratch.runner_warm[rows - 1] = true;
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block_size);
        let sampled = ctx.stream.clone_dtoh(&sampled_view)?;
        Ok(sampled)
    }

    /// Batched hedge-chain ladder: for every request, one chain per branch
    /// position — chain `(i, j)` follows `chain_a[i]` up to `positions[j]`,
    /// takes the rank-2 alternative there (`runners[j][i]`), and continues
    /// greedily with the Markov loop over the same backbone logits. All
    /// `rows = n_requests × n_chains` chain-rows run in one batched loop; the
    /// kernels map chain-row `r` onto base block `r / n_chains` (`chains`
    /// divisor). Returns request-major, chain-minor `[rows * block_size]`
    /// tokens with the shared prefixes and branch tokens stamped in.
    pub(crate) fn sample_chain_ladder(
        &self,
        ctx: &DeviceContext,
        base_logits: &HiddenStates,
        chain_a: &[u32],
        req_map: &[u32],
        positions: &[usize],
        block_size: usize,
        scratch: &mut MarkovScratch,
    ) -> Result<Vec<u32>> {
        let c = positions.len();
        anyhow::ensure!(c > 0, "chain ladder needs positions");
        // `req_map[i]` = the hedged slot's ORIGINAL request index: hedge
        // eligibility can skip requests, so the hedged set is not a batch
        // prefix. `chain_a` and `base_logits` stay full-batch; every kernel
        // resolves its request through the map.
        let n = req_map.len();
        anyhow::ensure!(n > 0, "chain ladder needs hedged requests");
        let total = chain_a.len() / block_size;
        anyhow::ensure!(total * block_size == chain_a.len(), "chain_a shape");
        // Both kernels index by `req_map[i]`: the argmax needs it below `total`
        // (base blocks), the runner gather below the stripe width. Establish
        // the tighter of the two — the unsafe calls below rely on it.
        anyhow::ensure!(
            total <= scratch.max_batch,
            "chain ladder batch {total} exceeds runner stripe {}",
            scratch.max_batch
        );
        anyhow::ensure!(
            req_map.iter().all(|&r| (r as usize) < total),
            "chain ladder req_map exceeds chain_a requests"
        );
        let rows = n * c;
        let vocab = base_logits.hidden_dim;
        anyhow::ensure!(
            base_logits.seq_len == total * block_size,
            "chain ladder base logits rows {} != total*block_size {}",
            base_logits.seq_len,
            total * block_size
        );
        scratch.activate(rows, vocab)?;
        let max_batch = scratch.max_batch;
        {
            let map_i32: Vec<i32> = req_map.iter().map(|&r| r as i32).collect();
            let mut map_dst = scratch.ladder_map.slice_mut(..n);
            ctx.stream.memcpy_htod(&map_i32, &mut map_dst)?;
        }
        // Position-0 token per chain-row, seeded from chain A's host copy (it
        // is already on the host for verify-span assembly; the re-upload is
        // `rows` u32s). The force kernel then overwrites the position-0
        // branch chains from the device-resident runner snapshot.
        let prev0: Vec<u32> = (0..rows)
            .map(|r| chain_a[req_map[r / c] as usize * block_size])
            .collect();
        {
            let mut prev_dst = scratch.prev_tokens.slice_mut(..rows);
            ctx.stream.memcpy_htod(&prev0, &mut prev_dst)?;
        }
        if let Some(j) = positions.iter().position(|&p| p == 0) {
            // SAFETY: req_map indices are host-checked above.
            unsafe {
                hedge_ladder_force_into(
                    ctx,
                    &mut scratch.prev_tokens,
                    &mut scratch.sampled_tokens,
                    &scratch.runner_tokens,
                    &scratch.ladder_map,
                    n,
                    c,
                    j,
                    max_batch,
                    block_size,
                    0,
                )?;
            }
        }
        // A captured ladder graph bakes the `chains` divisor into its kernels;
        // `rows` alone is ambiguous at requests > 1 (2 req x 2 chains = 4 rows
        // = 1 req x 4 chains). Replay only when the baked divisor matches;
        // mismatched shapes run the eager loop forever (no recapture).
        if markov_graph_enabled()
            && scratch.ladder_warm[rows - 1]
            && scratch.ladder_baked_chains[rows - 1] == c
        {
            let MarkovScratch {
                ladder_graphs,
                ladder_map,
                w1emb,
                bias,
                partial_values,
                partial_indices,
                runner_tokens,
                prev_tokens,
                next_tokens,
                sampled_tokens,
                ..
            } = scratch;
            ladder_graphs[rows - 1].run_or_capture(ctx, || {
                for step in 1..block_size {
                    let input_is_prev = step % 2 == 1;
                    {
                        let input: &CudaSlice<u32> = if input_is_prev {
                            prev_tokens
                        } else {
                            next_tokens
                        };
                        ops::embedding_batch(ctx, &self.w1, input, w1emb)?;
                    }
                    ops::gemm_into_checked(ctx, &self.w2, w1emb, bias)?;
                    let output: &mut CudaSlice<u32> = if input_is_prev {
                        next_tokens
                    } else {
                        prev_tokens
                    };
                    // SAFETY: req_map indices are host-checked above.
                    unsafe {
                        markov_step_argmax_mapped_into(
                            ctx,
                            base_logits,
                            bias,
                            block_size,
                            step,
                            c,
                            ladder_map,
                            rows,
                            partial_values,
                            partial_indices,
                            output,
                            sampled_tokens,
                        )?;
                    }
                    // Chains branching AT this step take their runner-up in
                    // place of the recomputed greedy token, written into the
                    // step's output buffer and the sampled block.
                    if let Some(j) = positions.iter().position(|&p| p == step) {
                        let output: &mut CudaSlice<u32> = if input_is_prev {
                            next_tokens
                        } else {
                            prev_tokens
                        };
                        // SAFETY: req_map indices are host-checked above.
                        unsafe {
                            hedge_ladder_force_into(
                                ctx,
                                output,
                                sampled_tokens,
                                runner_tokens,
                                &*ladder_map,
                                n,
                                c,
                                j,
                                max_batch,
                                block_size,
                                step,
                            )?;
                        }
                    }
                }
                Ok(())
            })?;
        } else {
            for step in 1..block_size {
                let input_is_prev = step % 2 == 1;
                {
                    let input = if input_is_prev {
                        &scratch.prev_tokens
                    } else {
                        &scratch.next_tokens
                    };
                    ops::embedding_batch(ctx, &self.w1, input, &mut scratch.w1emb)?;
                }
                ops::gemm_into_checked(ctx, &self.w2, &scratch.w1emb, &mut scratch.bias)?;
                {
                    let output = if input_is_prev {
                        &mut scratch.next_tokens
                    } else {
                        &mut scratch.prev_tokens
                    };
                    // SAFETY: req_map indices are host-checked above.
                    unsafe {
                        markov_step_argmax_mapped_into(
                            ctx,
                            base_logits,
                            &scratch.bias,
                            block_size,
                            step,
                            c,
                            &scratch.ladder_map,
                            rows,
                            &mut scratch.partial_values,
                            &mut scratch.partial_indices,
                            output,
                            &mut scratch.sampled_tokens,
                        )?;
                    }
                }
                if let Some(j) = positions.iter().position(|&p| p == step) {
                    let output = if input_is_prev {
                        &mut scratch.next_tokens
                    } else {
                        &mut scratch.prev_tokens
                    };
                    // SAFETY: req_map indices are host-checked above.
                    unsafe {
                        hedge_ladder_force_into(
                            ctx,
                            output,
                            &mut scratch.sampled_tokens,
                            &scratch.runner_tokens,
                            &scratch.ladder_map,
                            n,
                            c,
                            j,
                            max_batch,
                            block_size,
                            step,
                        )?;
                    }
                }
            }
            if !scratch.ladder_warm[rows - 1] {
                scratch.ladder_warm[rows - 1] = true;
                scratch.ladder_baked_chains[rows - 1] = c;
            }
        }
        let sampled_view = scratch.sampled_tokens.slice(..rows * block_size);
        let mut sampled = ctx.stream.clone_dtoh(&sampled_view)?;
        // Stamp the shared prefixes from chain A (the loop recomputes them
        // identically by determinism; stamping makes it exact by
        // construction). Branch tokens were written on device by the force
        // kernel and arrive in the same single D2H.
        for (i, &orig) in req_map.iter().enumerate() {
            let orig = orig as usize;
            for (j, &p) in positions.iter().enumerate() {
                let row = i * c + j;
                sampled[row * block_size..row * block_size + p]
                    .copy_from_slice(&chain_a[orig * block_size..orig * block_size + p]);
            }
        }
        Ok(sampled)
    }

    /// Bytes occupied by the Markov head weights + sample scratch, for the memory
    /// reservation. `0` when the head is disabled.
    pub(crate) fn reservation_bytes(
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<usize> {
        const BF16: usize = 2;

        if !config.uses_markov_head() {
            return Ok(0);
        }
        let vocab = config.vocab_size;
        let rank = config.markov_rank;
        let weights = sizing::product(&[2, vocab, rank, BF16])?;
        let scratch = MarkovScratch::bytes(vocab, rank, config.block_size, max_decode_batch_size)?;
        sizing::sum(&[weights, scratch])
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
    /// Second partials pair + runner-up output for the hedge's top-2 scan
    /// (`markov_step_top2_into`); idle when the hedge is off. Allocated with
    /// the head regardless: ~307 KiB at the bs-256 ceiling, bounded and
    /// constant.
    partial_values2: CudaSlice<f32>,
    partial_indices2: CudaSlice<i32>,
    top2_tokens: CudaSlice<u32>,
    /// Device-resident runner-up snapshots for the hedge ladder, one
    /// `max_batch` stripe per branch position (`[j * max_batch + i]`), copied
    /// D2D from `top2_tokens` at each branch step. The runner tokens
    /// themselves never visit the host; chain A does (verify spans are
    /// assembled host-side), and the ladder re-uploads only its `rows`
    /// position-0 parents.
    runner_tokens: CudaSlice<u32>,
    prev_tokens: CudaSlice<u32>,
    next_tokens: CudaSlice<u32>,
    sampled_tokens: CudaSlice<u32>,
    /// Per-`rows` CUDA graphs for the three markov loops (glm52 #591
    /// pattern): warm flag = first call runs the plain loop so cuBLAS lazy
    /// init happens outside capture; capture on the second, replay after.
    /// `DSPARK_NO_MARKOV_GRAPH=1` restores the plain loops.
    chain_graphs: Vec<CudaGraphState>,
    chain_warm: Vec<bool>,
    runner_graphs: Vec<CudaGraphState>,
    runner_warm: Vec<bool>,
    ladder_graphs: Vec<CudaGraphState>,
    ladder_warm: Vec<bool>,
    /// Hedged-slot -> original-request index map for the ladder kernels
    /// (contents re-uploaded each round; the buffer pointer is stable so
    /// captured graphs stay valid).
    ladder_map: CudaSlice<i32>,
    /// `chains` divisor baked into each captured ladder graph (0 = unset):
    /// replay requires an exact match — see [`Self::sample_chain_ladder`].
    ladder_baked_chains: Vec<usize>,
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
        let rank = config.markov_rank;
        let partials = markov_step_argmax_partials_len(max_decode_batch_size, vocab);
        let sampled = sizing::product(&[max_decode_batch_size, config.block_size])?;
        Ok(Self {
            max_batch: max_decode_batch_size,
            w1emb: HiddenStates::zeros(ctx, rank, max_decode_batch_size)?,
            bias: HiddenStates::zeros(ctx, vocab, max_decode_batch_size)?,
            partial_values: ctx.stream.alloc_zeros(partials)?,
            partial_indices: ctx.stream.alloc_zeros(partials)?,
            partial_values2: ctx.stream.alloc_zeros(partials)?,
            partial_indices2: ctx.stream.alloc_zeros(partials)?,
            top2_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            runner_tokens: ctx.stream.alloc_zeros(sampled)?,
            prev_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            next_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            sampled_tokens: ctx.stream.alloc_zeros(sampled)?,
            chain_graphs: (0..max_decode_batch_size)
                .map(|_| CudaGraphState::new())
                .collect(),
            chain_warm: vec![false; max_decode_batch_size],
            runner_graphs: (0..max_decode_batch_size)
                .map(|_| CudaGraphState::new())
                .collect(),
            runner_warm: vec![false; max_decode_batch_size],
            ladder_graphs: (0..max_decode_batch_size)
                .map(|_| CudaGraphState::new())
                .collect(),
            ladder_warm: vec![false; max_decode_batch_size],
            ladder_baked_chains: vec![0; max_decode_batch_size],
            ladder_map: ctx.stream.alloc_zeros(max_decode_batch_size)?,
        })
    }

    /// Max chain-rows this scratch was allocated for.
    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
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

    fn bytes(
        vocab: usize,
        rank: usize,
        block_size: usize,
        max_decode_batch_size: usize,
    ) -> Result<usize> {
        const BF16: usize = 2;
        let partials = markov_step_argmax_partials_len(max_decode_batch_size, vocab);
        let w1emb = sizing::product(&[max_decode_batch_size, rank, BF16])?;
        let bias = sizing::product(&[max_decode_batch_size, vocab, BF16])?;
        // Two partials pairs: the argmax pair plus the hedge's top-2 pair.
        let partial_bytes = sizing::product(&[
            2,
            partials,
            std::mem::size_of::<f32>() + std::mem::size_of::<i32>(),
        ])?;
        let tokens = sizing::sum(&[
            sizing::product(&[
                sizing::sum(&[
                    sizing::product(&[3, max_decode_batch_size])?,
                    sizing::product(&[2, max_decode_batch_size, block_size])?,
                ])?,
                std::mem::size_of::<u32>(),
            ])?,
            sizing::product(&[max_decode_batch_size, std::mem::size_of::<i32>()])?,
        ])?;
        sizing::sum(&[w1emb, bias, partial_bytes, tokens])
    }
}
