//! Model-side semantic boundary for Qwen3.5 GDN prefill.
//!
//! CuTe/generated-symbol/TMA/module/workspace details belong exclusively to
//! `pegainfer-kernels`. This module owns only model policy, prepared tensors,
//! recurrent state, and observable backend evidence.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Qwen35GdnAot;
use pegainfer_kernels::ops::Qwen35GdnGeometry;
use pegainfer_kernels::ops::Qwen35GdnWorkspace;

use crate::config::Config35;
use crate::prefill_buffers::GdnPrepareScratch35;
use crate::weights::Qwen35Model;

/// Backend selected once at the production prefill boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GdnPrefillBackend {
    Triton,
    FlashInfer,
}

pub(crate) struct FlashInferGdnChunkResources {
    pub(crate) prepare: GdnPrepareScratch35,
    pub(crate) output: HiddenStates,
    launch: Qwen35GdnWorkspace,
}

impl FlashInferGdnChunkResources {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        backend: &Qwen35GdnAot,
        tokens: usize,
    ) -> Result<Self> {
        let geometry = model_geometry(config);
        ensure!(
            geometry == Qwen35GdnGeometry::PRODUCTION,
            "FlashInfer GDN is not supported for model geometry {geometry:?}"
        );
        Ok(Self {
            prepare: GdnPrepareScratch35::new(ctx, config, tokens)?,
            output: HiddenStates::zeros(ctx, geometry.h_v * geometry.head_dim, tokens)?,
            launch: backend.allocate_workspace(ctx, tokens)?,
        })
    }

    pub(crate) fn ensure_prepare_inputs_finite(&self, ctx: &DeviceContext) -> Result<()> {
        let status = ctx
            .stream
            .clone_dtoh(&self.prepare.non_finite_status)
            .map_err(|error| anyhow::anyhow!("read native GDN finite-status failed: {error}"))?;
        ctx.sync()?;
        ensure!(
            status == [0],
            "native GDN prepare rejected non-finite qkv/gate input"
        );
        Ok(())
    }

    pub(crate) fn launch_in_place(
        &mut self,
        ctx: &DeviceContext,
        backend: &Qwen35GdnAot,
        state: &mut CudaSlice<f32>,
    ) -> Result<()> {
        backend.launch_in_place(
            ctx,
            &self.prepare.q,
            &self.prepare.k,
            &self.prepare.v,
            &self.prepare.alpha,
            &self.prepare.beta,
            state,
            &mut self.output,
            &mut self.launch,
        )
    }
}

pub(crate) fn model_geometry(config: &Config35) -> Qwen35GdnGeometry {
    Qwen35GdnGeometry {
        h_q: config.linear_num_key_heads,
        h_k: config.linear_num_key_heads,
        h_v: config.linear_num_value_heads,
        head_dim: config.linear_key_head_dim,
    }
}

impl Qwen35Model {
    pub(crate) fn resolved_gdn_backend(&self) -> GdnPrefillBackend {
        if self.flashinfer_gdn.is_some() {
            GdnPrefillBackend::FlashInfer
        } else {
            GdnPrefillBackend::Triton
        }
    }

    pub(super) fn flashinfer_gdn(&self) -> Result<&Qwen35GdnAot> {
        self.flashinfer_gdn
            .as_ref()
            .context("FlashInfer GDN is not selected for this model capability")
    }
}
