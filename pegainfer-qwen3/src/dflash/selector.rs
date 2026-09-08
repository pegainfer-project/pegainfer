//! CUDA-backed DFlash2 candidate selection.
//!
//! DFlash2 selector weights and persistent scratch.
//!
//! CUDA performs bounded candidate and path selection.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::DFlash2SelectorScratch;
use pegainfer_kernels::ops::dflash2_selector_into;
use pegainfer_kernels::ops::dflash2_selector_scratch_bytes;
use pegainfer_kernels::ops::dflash2_selector_selected_host;

use crate::config::DFlashConfig;

/// DFlash2 selector weights.
pub(crate) struct SelectorWeights {
    hidden_projection: DeviceMatrix,
    predecessor_codebook: DeviceMatrix,
    successor_codebook: DeviceMatrix,
}

impl SelectorWeights {
    pub(crate) fn new(
        rank: usize,
        hidden_size: usize,
        vocab_size: usize,
        hidden_projection: DeviceMatrix,
        predecessor_codebook: DeviceMatrix,
        successor_codebook: DeviceMatrix,
    ) -> Result<Self> {
        ensure!(rank > 0, "DFlash selector rank must be positive");
        ensure!(
            hidden_size > 0,
            "DFlash selector hidden size must be positive"
        );
        ensure!(
            vocab_size > 0,
            "DFlash selector vocabulary must be positive"
        );
        ensure!(
            hidden_projection.rows == rank && hidden_projection.cols == hidden_size,
            "DFlash selector projection shape {}x{} does not match {}x{}",
            hidden_projection.rows,
            hidden_projection.cols,
            rank,
            hidden_size
        );
        ensure!(
            predecessor_codebook.rows == vocab_size && predecessor_codebook.cols == rank,
            "DFlash selector predecessor shape {}x{} does not match {}x{}",
            predecessor_codebook.rows,
            predecessor_codebook.cols,
            vocab_size,
            rank
        );
        ensure!(
            successor_codebook.rows == vocab_size && successor_codebook.cols == rank,
            "DFlash selector successor shape {}x{} does not match {}x{}",
            successor_codebook.rows,
            successor_codebook.cols,
            vocab_size,
            rank
        );
        Ok(Self {
            hidden_projection,
            predecessor_codebook,
            successor_codebook,
        })
    }

    pub(crate) fn select_block(
        &self,
        ctx: &DeviceContext,
        logits: &HiddenStates,
        logits_normed: &HiddenStates,
        current_tokens: &[u32],
        block_size: usize,
        anchor_first: bool,
        scratch: &mut SelectorScratch,
    ) -> Result<Vec<u32>> {
        ensure!(!current_tokens.is_empty(), "selector needs active requests");
        ensure!(block_size > 0, "selector block size must be positive");
        let rows = current_tokens
            .len()
            .checked_mul(block_size)
            .context("selector row count overflow")?;
        ensure!(
            logits.seq_len == rows,
            "selector logits rows {} != requests {} * block {}",
            logits.seq_len,
            current_tokens.len(),
            block_size
        );
        ensure!(
            logits_normed.seq_len == rows,
            "selector hidden rows {} != requests {} * block {}",
            logits_normed.seq_len,
            current_tokens.len(),
            block_size
        );
        ensure!(
            logits.hidden_dim == self.successor_codebook.rows,
            "selector logits vocabulary {} != codebook vocabulary {}",
            logits.hidden_dim,
            self.successor_codebook.rows
        );
        ensure!(
            logits_normed.hidden_dim == self.hidden_projection.cols,
            "selector hidden size {} != projection input {}",
            logits_normed.hidden_dim,
            self.hidden_projection.cols
        );
        scratch.activate(rows, current_tokens.len(), self.hidden_projection.rows)?;

        let mut anchor_dst = scratch.anchor_tokens.slice_mut(..current_tokens.len());
        ctx.stream.memcpy_htod(current_tokens, &mut anchor_dst)?;
        ops::gemm_into(
            ctx,
            &self.hidden_projection,
            logits_normed,
            &mut scratch.projected_hidden,
        );
        let (position_offset, positions_per_request) = if anchor_first {
            (0, block_size)
        } else {
            let positions = block_size
                .checked_sub(1)
                .context("anchor-drop selector block size must exceed one")?;
            (1, positions)
        };
        dflash2_selector_into(
            ctx,
            logits,
            &scratch.projected_hidden,
            &self.predecessor_codebook,
            &self.successor_codebook,
            &scratch.anchor_tokens,
            block_size,
            position_offset,
            positions_per_request,
            &mut scratch.selector,
        )?;

        let selected_rows = current_tokens
            .len()
            .checked_mul(positions_per_request)
            .context("selector selected row count overflow")?;
        let selected = dflash2_selector_selected_host(ctx, &scratch.selector, selected_rows)?;
        ctx.sync()?;
        ensure!(
            selected
                .iter()
                .all(|&token_id| u64::from(token_id) < logits.hidden_dim as u64),
            "DFlash2 selector produced a token outside vocabulary size {}",
            logits.hidden_dim
        );
        if anchor_first {
            ensure!(selected.len() == rows);
            return Ok(selected);
        }

        // Restore the existing anchor-inclusive draft contract.
        let mut output = Vec::with_capacity(rows);
        for (request_idx, &anchor) in current_tokens.iter().enumerate() {
            output.push(anchor);
            let start = request_idx * positions_per_request;
            output.extend_from_slice(&selected[start..start + positions_per_request]);
        }
        Ok(output)
    }
}

/// Persistent GPU scratch for a selector-enabled lane.
pub(crate) struct SelectorScratch {
    max_rows: usize,
    projected_hidden: HiddenStates,
    anchor_tokens: CudaSlice<u32>,
    selector: DFlash2SelectorScratch,
}

impl SelectorScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        let rank = match &config.proposal {
            crate::config::DFlashProposal::TopKSelector { rank, .. } => *rank,
            _ => {
                return Err(anyhow::anyhow!(
                    "selector scratch requested without selector"
                ));
            }
        };
        let max_rows = max_decode_batch_size
            .checked_mul(config.block_size)
            .context("selector scratch row count overflow")?;
        Ok(Self {
            max_rows,
            projected_hidden: HiddenStates::zeros(ctx, rank, max_rows)?,
            anchor_tokens: ctx.stream.alloc_zeros(max_decode_batch_size)?,
            selector: DFlash2SelectorScratch::new(ctx, max_rows)?,
        })
    }

    fn activate(&mut self, rows: usize, requests: usize, rank: usize) -> Result<()> {
        ensure!(
            rows <= self.max_rows,
            "selector rows {} exceed scratch capacity {}",
            rows,
            self.max_rows
        );
        ensure!(
            requests <= self.anchor_tokens.len(),
            "selector requests {} exceed anchor capacity {}",
            requests,
            self.anchor_tokens.len()
        );
        ensure!(
            self.projected_hidden.hidden_dim == rank,
            "selector scratch rank {} != projection rank {}",
            self.projected_hidden.hidden_dim,
            rank
        );
        self.projected_hidden.seq_len = rows;
        Ok(())
    }

    pub(crate) fn bytes(config: &DFlashConfig, max_decode_batch_size: usize) -> Result<usize> {
        const BF16: usize = 2;
        let rank = match &config.proposal {
            crate::config::DFlashProposal::TopKSelector { rank, .. } => *rank,
            _ => return Ok(0),
        };
        let rows = max_decode_batch_size
            .checked_mul(config.block_size)
            .context("selector scratch row count overflow")?;
        let projected = rows
            .checked_mul(rank)
            .and_then(|bytes| bytes.checked_mul(BF16))
            .context("selector projected-hidden scratch size overflow")?;
        let anchors = max_decode_batch_size
            .checked_mul(std::mem::size_of::<u32>())
            .context("selector anchor scratch size overflow")?;
        let selector = dflash2_selector_scratch_bytes(rows);
        projected
            .checked_add(anchors)
            .and_then(|bytes| bytes.checked_add(selector))
            .context("selector scratch size overflow")
    }
}
