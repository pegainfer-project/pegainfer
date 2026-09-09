//! Model-side semantic boundary for Qwen3.5 GDN prefill.
//!
//! CuTe/generated-symbol/TMA/module/workspace details belong exclusively to
//! `pegainfer-kernels`. This module owns only model policy, prepared tensors,
//! and recurrent state.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Qwen35GdnAot;
use pegainfer_kernels::ops::Qwen35GdnGeometry;
use pegainfer_kernels::ops::Qwen35GdnWorkspace;

use crate::config::Config35;
use crate::prefill_buffers::GdnPrepareScratch35;

pub(crate) struct FlashInferGdnChunkResources {
    pub(crate) prepare: GdnPrepareScratch35,
    pub(crate) output: HiddenStates,
    launch: Qwen35GdnWorkspace,
}

impl FlashInferGdnChunkResources {
    /// One chunk's persistent buffers and maximum live layer temporaries, plus
    /// bounded paged-attention metadata. Module allocations already happened.
    pub(crate) fn estimate_bytes(
        config: &Config35,
        backend: &Qwen35GdnAot,
        tokens: usize,
        page_size: usize,
    ) -> usize {
        let geometry = model_geometry(config);
        let prepared_and_output = tokens
            * (geometry.h_q + geometry.h_k + 2 * geometry.h_v)
            * geometry.head_dim
            * size_of::<half::bf16>();
        let gates = tokens * geometry.h_v * 2 * size_of::<f32>();

        let hidden = config.hidden_size;
        let full_q = config.num_attention_heads * config.head_dim;
        let full_kv = config.num_key_value_heads * config.head_dim;
        let linear_qkv = (geometry.h_q + geometry.h_k + geometry.h_v) * geometry.head_dim;
        let linear_z = config.linear_attn_z_dim();
        // Both attention paths retain input hidden + normed + projected output.
        let full_attention = 3 * hidden + 4 * full_q + 2 * full_kv;
        let linear_attention = 3 * hidden + 2 * linear_qkv + 2 * linear_z + 2 * geometry.h_v;
        // At the final residual: input hidden, attention result, hidden+attention,
        // MLP normed input, MLP output and new residual output all remain live.
        // The fused gate/up and activated intermediate also survive until return.
        let mlp = 6 * hidden + 3 * config.intermediate_size;
        let layer_peak =
            tokens * full_attention.max(linear_attention).max(mlp) * size_of::<half::bf16>();

        // PrefillPagedPlan owns a page list, batch indices and positions (T each),
        // three tile arrays, and seven scalar/indptr elements. Bound page count
        // by the serving context ceiling and tile count by packed Q rows (each
        // tile consumes at least one row), independent of CUDA's tile choice.
        let pages = config.max_position_embeddings.div_ceil(page_size);
        let tile_bound = tokens * (config.num_attention_heads / config.num_key_value_heads);
        let plan_metadata = (pages + 2 * tokens + 3 * tile_bound + 7) * size_of::<i32>();
        let token_ids = tokens * size_of::<u32>();
        let attention_start_position = size_of::<i32>();

        prepared_and_output
            + gates
            + 2 * size_of::<i64>() // cu_seqlens for one sequence
            + backend.workspace_bytes()
            + layer_peak
            + plan_metadata
            + token_ids
            + attention_start_position
    }

    pub(crate) fn new(ctx: &DeviceContext, backend: &Qwen35GdnAot, tokens: usize) -> Result<Self> {
        let geometry = Qwen35GdnGeometry::PRODUCTION;
        Ok(Self {
            prepare: GdnPrepareScratch35::new(ctx, tokens)?,
            output: HiddenStates::zeros(ctx, geometry.h_v * geometry.head_dim, tokens)?,
            launch: backend.allocate_workspace(ctx, tokens)?,
        })
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
