//! Learned top-16 proposal selection on the existing DFlash backbone.

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::DFlash2Scratch;
use pegainfer_kernels::ops::dflash2_select_into;
use pegainfer_kernels::ops::f32_to_bf16_hidden_into;
use pegainfer_kernels::ops::gemm_bf16_f32;

use crate::config::DFlashConfig;
use crate::sizing;

// Pack a draft block per launch, capped to bound full-vocabulary keys.
// CUB still reuses one row's workspace.
const MAX_PACK_ROWS: usize = 16;

pub(super) struct SelectorHead {
    pub(super) projection: DeviceMatrix,
    pub(super) predecessor: DeviceMatrix,
    pub(super) successor: DeviceMatrix,
    pub(super) enabled: bool,
}

pub(super) struct SelectorScratch {
    // Graphs are destroyed before their captured buffers.
    graphs: Vec<CudaGraphState>,
    warmed: Vec<bool>,
    buffers: SelectorBuffers,
}

struct SelectorBuffers {
    projected_f32: CudaSlice<f32>,
    projected: HiddenStates,
    anchors: CudaSlice<u32>,
    lattice: DFlash2Scratch,
}

impl SelectorHead {
    pub(super) fn reservation_bytes(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        batch: usize,
    ) -> Result<usize> {
        let rank = config.selector_rank;
        if rank == 0 {
            return Ok(0);
        }
        let weights = sizing::product(&[
            2,
            rank,
            sizing::sum(&[
                config.hidden_size,
                sizing::product(&[2, config.vocab_size])?,
            ])?,
        ])?;
        sizing::sum(&[
            weights,
            sizing::product(&[batch, config.block_size, rank, 6])?,
            sizing::product(&[batch, 4])?,
            DFlash2Scratch::reservation_bytes(
                ctx,
                batch,
                config.block_size,
                config.vocab_size,
                rank,
                MAX_PACK_ROWS.min(config.block_size.saturating_sub(1)),
            )?,
        ])
    }

    /// Return `block_size - 1` draft tokens per request; the lane adds each anchor once.
    pub(super) fn select(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        logits: &HiddenStates,
        anchors: &[u32],
        block_size: usize,
        scratch: &mut SelectorScratch,
    ) -> Result<Vec<u32>> {
        let batch = anchors.len();
        ensure!(
            batch > 0 && batch <= scratch.graphs.len(),
            "DFlash2 invalid batch size"
        );
        let rows = batch * block_size;
        ensure!(
            hidden.seq_len == rows && hidden.hidden_dim == self.projection.cols,
            "DFlash2 backbone hidden shape does not match selector"
        );
        let SelectorScratch {
            graphs,
            warmed,
            buffers,
        } = scratch;
        buffers.projected.seq_len = rows;
        ctx.stream
            .memcpy_htod(anchors, &mut buffers.anchors.slice_mut(..batch))?;

        // Warm cuBLAS before capture, just as the existing DSpark proposal does.
        if warmed[batch - 1] {
            graphs[batch - 1]
                .run_or_capture(ctx, || self.enqueue(ctx, hidden, logits, batch, buffers))?;
        } else {
            self.enqueue(ctx, hidden, logits, batch, buffers)?;
            warmed[batch - 1] = true;
        }

        let count = batch * (block_size - 1);
        let selected = ctx
            .stream
            .clone_dtoh(&buffers.lattice.selected_ids().slice(..count))?;
        // The device walk poisons every active ID on error. Read the error
        // details only on failure, keeping the normal path to one D2H copy.
        if selected[0] == u32::MAX {
            let error = ctx.stream.clone_dtoh(buffers.lattice.error_flag())?[0];
            bail!("DFlash2 rejected invalid input or nonfinite scores ({error:#x})");
        }
        Ok(selected)
    }

    fn enqueue(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        logits: &HiddenStates,
        batch: usize,
        buffers: &mut SelectorBuffers,
    ) -> Result<()> {
        // Keep split-K partials/output in FP32, then round to BF16 exactly once.
        // A BF16-output GEMM can round its partial sums earlier.
        gemm_bf16_f32(
            ctx,
            true,
            false,
            self.projection.rows,
            hidden.seq_len,
            hidden.hidden_dim,
            &self.projection.data,
            hidden.hidden_dim,
            &hidden.data,
            hidden.hidden_dim,
            &mut buffers.projected_f32,
            self.projection.rows,
        )?;
        f32_to_bf16_hidden_into(ctx, &buffers.projected_f32, &mut buffers.projected)?;
        dflash2_select_into(
            ctx,
            logits,
            &buffers.projected,
            &self.predecessor,
            &self.successor,
            &buffers.anchors,
            batch,
            &mut buffers.lattice,
        )
    }
}

impl SelectorScratch {
    pub(super) fn new(ctx: &DeviceContext, config: &DFlashConfig, batch: usize) -> Result<Self> {
        let rows = sizing::product(&[batch, config.block_size])?;
        let elements = sizing::product(&[rows, config.selector_rank])?;
        ensure!(
            i32::try_from(elements).is_ok(),
            "DFlash2 projection exceeds int32 conversion extent"
        );
        Ok(Self {
            graphs: (0..batch).map(|_| CudaGraphState::new()).collect(),
            warmed: vec![false; batch],
            buffers: SelectorBuffers {
                projected_f32: ctx.stream.alloc_zeros(elements)?,
                projected: HiddenStates::zeros(ctx, config.selector_rank, rows)?,
                anchors: ctx.stream.alloc_zeros(batch)?,
                lattice: DFlash2Scratch::new(
                    ctx,
                    batch,
                    config.block_size,
                    config.vocab_size,
                    config.selector_rank,
                    MAX_PACK_ROWS.min(config.block_size.saturating_sub(1)),
                )?,
            },
        })
    }
}

#[cfg(test)]
mod tests;
