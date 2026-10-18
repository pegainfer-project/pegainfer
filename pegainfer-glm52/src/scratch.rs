//! Persistent per-rank decode scratch: every intermediate of one decode step
//! lives in a buffer allocated once at model build.
//!
//! Two properties this buys (both prerequisites for the whole-step CUDA
//! graph, PR5c stage 3):
//!
//! - **No per-step allocator traffic.** PR5a made the MoE collective chain
//!   persistent (`Glm52MoeEp8State`); this extends the same treatment to the
//!   MLA/indexer/MLP spine (~5 ms/step of residual `alloc_zeros` at bs=1).
//! - **Pointer stability.** A captured graph replays the recorded device
//!   pointers; every kernel in the step must read/write the same buffers
//!   every step.
//!
//! Safety rule for reuse: every buffer is bound to ONE call-site purpose and
//! shared only ACROSS layers (the 78 layers have identical shapes, and layer
//! N's intermediates are dead once layer N consumed them). Stale content is
//! therefore always "the same semantic value from the previous step/layer" —
//! either fully overwritten before its consumer reads it, or never written by
//! anyone and still holding the build-time zero initialization (e.g. the
//! TMA-relayout pad rows). The one deliberate exception is
//! `idx.global_slots`, the cross-layer top-k carry: full-indexer layers
//! overwrite it, shared layers reuse it — exactly the DSA contract.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_TP_TOKENS;
use pegainfer_kernels::ops::Glm52DeepGemmMqaLogitsShape;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::argmax_batch_bf16_split_partials_len;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_DENSE_INTERMEDIATE;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_VOCAB;
use crate::dspark::GLM52_DSPARK_CONTEXT_DIM;
use crate::fp8::Glm52MlpScratch;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52LayerScratch;
use crate::mla_decode::Glm52MlaAttendScratch;
use crate::mla_decode::Glm52MlaBackend;
use crate::mla_front::Glm52MlaFront;
use crate::moe_decode::GLM52_SHARED_EXPERT_INTERMEDIATE;
use crate::moe_decode::Glm52RouterScratch;
use crate::rows::Rows;

/// Everything one decode step writes, allocated once per rank and sized for
/// the step's `tokens` rows.
pub(crate) struct Glm52DecodeScratch {
    pub(crate) mla_front: Glm52MlaFront,
    pub(crate) mla_attend: Glm52MlaAttendScratch,
    pub(crate) idx: Glm52IndexerScratch,
    pub(crate) dense_mlp: Glm52MlpScratch,
    pub(crate) shared_mlp: Glm52MlpScratch,
    pub(crate) router: Glm52RouterScratch,
    /// Fixed-row bridge used only by tensor-parallel MoE. TP4 can run the
    /// surrounding graph at bucket 1/2/4 while the phase chain retains its
    /// established eight-row memory contract.
    pub(crate) tp_normed2: CudaSlice<bf16>,
    pub(crate) tp_mlp_out: CudaSlice<bf16>,
    pub(crate) layer: Glm52LayerScratch,
    /// The residual stream: embed writes it, every layer reads and rewrites
    /// it, the final norm consumes it.
    pub(crate) hidden: Rows<GLM52_HIDDEN>,
    /// Optional DSpark aux-hidden capture: each row's residual stream after the
    /// [`crate::dspark::GLM52_DSPARK_AUX_LAYERS`] layers, concatenated per row
    /// (`[tokens, 5 * GLM52_HIDDEN]`). Present only when the drafter was
    /// requested at launch; otherwise neither the buffer nor its five graph
    /// copy nodes exist.
    pub(crate) captured: Option<Rows<GLM52_DSPARK_CONTEXT_DIM>>,
    pub(crate) final_normed: Rows<GLM52_HIDDEN>,
    pub(crate) logits: Rows<GLM52_VOCAB>,
    /// Device greedy argmax outputs: each row's top logit bf16 value (for the
    /// crash-early non-finite guard) and its index — the step's per-row
    /// 6-byte D2H egress. The two-stage argmax stages per-4096-tile partials
    /// first.
    pub(crate) argmax_partial_values: CudaSlice<f32>,
    pub(crate) argmax_partial_indices: CudaSlice<i32>,
    pub(crate) argmax_values: CudaSlice<bf16>,
    pub(crate) argmax_indices: CudaSlice<i32>,
}

impl Glm52DecodeScratch {
    #[cfg(test)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        mqa_shape: Glm52DeepGemmMqaLogitsShape,
        mla_heads: usize,
        dspark_enabled: bool,
    ) -> Result<Self> {
        Self::new_for_backend(
            ctx,
            contract,
            mqa_shape,
            mla_heads,
            Glm52MlaBackend::FlashMlaFp8Ds,
            dspark_enabled,
        )
    }

    pub(crate) fn new_for_backend(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        mqa_shape: Glm52DeepGemmMqaLogitsShape,
        mla_heads: usize,
        mla_backend: Glm52MlaBackend,
        dspark_enabled: bool,
    ) -> Result<Self> {
        let tokens = contract.batch_size;
        anyhow::ensure!(
            mqa_shape.batch_size == tokens,
            "GLM5.2 decode scratch: MQA batch {} != attend batch {tokens}",
            mqa_shape.batch_size
        );
        Ok(Self {
            mla_front: Glm52MlaFront::new(ctx, tokens, mla_heads)?,
            mla_attend: Glm52MlaAttendScratch::new_for_backend(
                ctx,
                contract,
                mla_heads,
                mla_backend,
            )?,
            idx: Glm52IndexerScratch::new(ctx, mqa_shape)?,
            dense_mlp: Glm52MlpScratch::new(ctx, GLM52_DENSE_INTERMEDIATE, tokens)?,
            shared_mlp: Glm52MlpScratch::new(ctx, GLM52_SHARED_EXPERT_INTERMEDIATE, tokens)?,
            router: Glm52RouterScratch::new(ctx, tokens)?,
            tp_normed2: ctx
                .stream
                .alloc_zeros::<bf16>(GLM52_TP_TOKENS * GLM52_HIDDEN)?,
            tp_mlp_out: ctx
                .stream
                .alloc_zeros::<bf16>(GLM52_TP_TOKENS * GLM52_HIDDEN)?,
            layer: Glm52LayerScratch::new(ctx, tokens)?,
            hidden: Rows::zeros(ctx, tokens)?,
            captured: dspark_enabled
                .then(|| Rows::zeros(ctx, tokens))
                .transpose()?,
            final_normed: Rows::zeros(ctx, tokens)?,
            logits: Rows::zeros(ctx, tokens)?,
            argmax_partial_values: ctx
                .stream
                .alloc_zeros::<f32>(argmax_batch_bf16_split_partials_len(tokens, GLM52_VOCAB))?,
            argmax_partial_indices: ctx
                .stream
                .alloc_zeros::<i32>(argmax_batch_bf16_split_partials_len(tokens, GLM52_VOCAB))?,
            argmax_values: ctx.stream.alloc_zeros::<bf16>(tokens)?,
            argmax_indices: ctx.stream.alloc_zeros::<i32>(tokens)?,
        })
    }
}
