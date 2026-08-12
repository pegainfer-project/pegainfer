//! Model-side semantic boundary for Qwen3.5 GDN prefill.
//!
//! CuTe/generated-symbol/TMA/module/workspace details belong exclusively to
//! `pegainfer-kernels`. This module owns only model policy, prepared tensors,
//! recurrent state, and observable backend evidence.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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

/// Internal policy injected at the production dispatch boundary. `Auto` is
/// used by serving; the forced variants exist only for same-path A/B gates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum GdnPrefillBackendSeam {
    #[default]
    Auto,
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
    pub(crate) fn resolved_gdn_backend(
        &self,
        requested: GdnPrefillBackendSeam,
    ) -> Result<GdnPrefillBackendSeam> {
        match requested {
            GdnPrefillBackendSeam::Auto => Ok(if self.flashinfer_gdn.is_some() {
                GdnPrefillBackendSeam::FlashInfer
            } else {
                GdnPrefillBackendSeam::Triton
            }),
            GdnPrefillBackendSeam::Triton => Ok(GdnPrefillBackendSeam::Triton),
            GdnPrefillBackendSeam::FlashInfer => {
                ensure!(
                    self.flashinfer_gdn.is_some(),
                    "forced FlashInfer GDN is unsupported for this device/model capability"
                );
                Ok(GdnPrefillBackendSeam::FlashInfer)
            }
        }
    }

    pub(super) fn flashinfer_gdn(&self) -> Result<&Qwen35GdnAot> {
        self.flashinfer_gdn
            .as_ref()
            .context("FlashInfer GDN is not selected for this model capability")
    }
}

/// Runtime proof for production dispatch and same-path A/B tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GdnPrefillRuntimeEvidence {
    pub selected_backend: String,
    pub artifact_sha256: String,
    pub artifact_size_bytes: u64,
    pub runtime_workspace_bytes: u64,
    pub successful_launches: u64,
}

#[derive(Clone, Debug)]
pub struct GdnPrefillRuntimeEvidenceHandle {
    selected_backend: &'static str,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    runtime_workspace_bytes: u64,
    successful_launches: Arc<AtomicU64>,
}

impl GdnPrefillRuntimeEvidenceHandle {
    pub fn snapshot(&self) -> GdnPrefillRuntimeEvidence {
        GdnPrefillRuntimeEvidence {
            selected_backend: self.selected_backend.to_owned(),
            artifact_sha256: self.artifact_sha256.clone(),
            artifact_size_bytes: self.artifact_size_bytes,
            runtime_workspace_bytes: self.runtime_workspace_bytes,
            successful_launches: self.successful_launches.load(Ordering::Relaxed),
        }
    }
}

impl Qwen35Model {
    pub fn flashinfer_gdn_runtime_evidence(&self) -> Result<GdnPrefillRuntimeEvidence> {
        Ok(self.flashinfer_gdn_runtime_evidence_handle()?.snapshot())
    }

    pub fn flashinfer_gdn_runtime_evidence_handle(
        &self,
    ) -> Result<GdnPrefillRuntimeEvidenceHandle> {
        let backend = self.flashinfer_gdn()?;
        Ok(GdnPrefillRuntimeEvidenceHandle {
            selected_backend: "flashinfer",
            artifact_sha256: backend.artifact_sha256().to_owned(),
            artifact_size_bytes: backend.artifact_size_bytes(),
            runtime_workspace_bytes: backend.workspace_bytes() as u64,
            successful_launches: backend.successful_launch_counter(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_geometry_is_exact_hv32() {
        assert_eq!(
            Qwen35GdnGeometry::PRODUCTION,
            Qwen35GdnGeometry {
                h_q: 16,
                h_k: 16,
                h_v: 32,
                head_dim: 128,
            }
        );
        assert_ne!(
            Qwen35GdnGeometry::PRODUCTION,
            Qwen35GdnGeometry {
                h_v: 48,
                ..Qwen35GdnGeometry::PRODUCTION
            }
        );
    }
}
