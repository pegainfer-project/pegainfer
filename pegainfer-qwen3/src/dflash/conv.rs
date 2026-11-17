//! Dynamic pre/post convolutions around the drafter attention and MLP.

use anyhow::Result;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::DFlashConfig;

pub(super) struct LayerConvs {
    pub(super) attention: GroupedConv,
    pub(super) mlp: GroupedConv,
}

pub(super) struct GroupedConv {
    pub(super) projection: DeviceMatrix,
    pub(super) base: DeviceVec,
    pub(super) block_size: usize,
    pub(super) group_size: usize,
}

pub(super) struct ConvScratch {
    dynamic: HiddenStates,
    output: HiddenStates,
}

impl ConvScratch {
    pub(super) fn new(ctx: &DeviceContext, config: &DFlashConfig, rows: usize) -> Result<Self> {
        let conv = config.conv.as_ref().expect("convolution config");
        let width = 2 * conv.taps * (config.hidden_size / conv.group_size);
        Ok(Self {
            dynamic: HiddenStates::zeros(ctx, width, rows)?,
            output: HiddenStates::zeros(ctx, config.hidden_size, rows)?,
        })
    }
}

impl GroupedConv {
    pub(super) fn prepare(
        &self,
        ctx: &DeviceContext,
        input: &mut HiddenStates,
        scratch: &mut ConvScratch,
    ) -> Result<()> {
        scratch.dynamic.seq_len = input.seq_len;
        ops::gemm_into(ctx, &self.projection, input, &mut scratch.dynamic);
        self.apply(ctx, input, scratch, 0)
    }

    pub(super) fn finish(
        &self,
        ctx: &DeviceContext,
        input: &mut HiddenStates,
        scratch: &mut ConvScratch,
    ) -> Result<()> {
        // The post side uses coefficients from before the sublayer, not its output.
        self.apply(ctx, input, scratch, 1)
    }

    fn apply(
        &self,
        ctx: &DeviceContext,
        input: &mut HiddenStates,
        scratch: &mut ConvScratch,
        side: usize,
    ) -> Result<()> {
        scratch.output.seq_len = input.seq_len;
        pegainfer_kernels::ops::dflash2_grouped_conv_into(
            ctx,
            input,
            &scratch.dynamic,
            &self.base.data,
            self.block_size,
            self.group_size,
            side,
            &mut scratch.output,
        )?;
        std::mem::swap(input, &mut scratch.output);
        Ok(())
    }
}
