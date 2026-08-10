//! GLM5.2 MTP layer-78 accuracy-oracle bookends.
//!
//! The checkpoint's MTP decoder block is the same concrete decoder-layer
//! implementation as the target stack. This module owns only the math unique
//! to MTP:
//!
//! ```text
//! embed = where(position == 0, 0, embed)
//! decoder_input = eh_proj(cat(enorm(embed), hnorm(previous_hidden)))
//! raw_hidden = decoder_layer_78(decoder_input)
//! recycle_hidden = shared_head.norm(raw_hidden)
//! logits = lm_head(shared_head.norm(raw_hidden))
//! ```
//!
//! `raw_hidden` must remain available for target-head logits. The normalized
//! value is recycled into the next draft iteration; normalizing in place
//! would apply the shared norm twice on the logits path.
//!
//! Production serving owns residency and state in `model::mtp`; the oracle
//! tests call these same bookend operations directly.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::copy_hidden_rows_raw_into;
use pegainfer_kernels::ops::gemm_strided_batched_bf16;
use pegainfer_kernels::ops::mask_position_zero_rows_into;
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_RMS_EPS;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::rows::Rows;

const MTP_FUSED_INPUT: usize = 2 * GLM52_HIDDEN;
pub(crate) const GLM52_MTP_DRAFTS: usize = 5;

/// The draft span length actually proposed and verified: `GLM52_MTP_DRAFTS`
/// (1..=5), default the full 5. Every wire/array shape stays at the compile
/// ceiling — a shorter span just leaves the tail rows unproposed, so P and D
/// may even disagree on this knob (D truncates to its own length; a longer
/// D span verifies zero-filled rows that simply fail the prefix match).
/// Wide-EP throughput deployments pair 2 drafts with 16 decode slots: fewer
/// sequential proposal forwards per step, same 48-row verify ceiling.
pub(crate) fn glm52_mtp_draft_len() -> usize {
    static LEN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LEN.get_or_init(|| {
        std::env::var("GLM52_MTP_DRAFTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, GLM52_MTP_DRAFTS))
            .unwrap_or(GLM52_MTP_DRAFTS)
    })
}

/// Context-scaled device memory the native MTP lane ADDS on top of
/// `glm52_arena_bytes` (which already charges the layer-78 committed mirrors
/// inside every slab page): the per-slot proposal scratch (EP: whole slab
/// pages past the registered pool region; TP4: rows of the dense cache),
/// TP4's dense FlashInfer execution cache, and one set of per-bucket indexer
/// logits/block tables. Fixed-size weights and scratch are accounted by the
/// post-build headroom probe; this function is the exact monotone term used
/// to derive the context cap before those arenas are allocated.
pub(crate) fn glm52_mtp_arena_bytes(
    max_model_len: usize,
    pool_blocks: usize,
    topology: crate::Glm52MoeTopo,
) -> Result<usize> {
    // The private per-slot pages hold unverified proposal KV. Committed
    // layer-78 rows ride the target BlockPool page ids and are transferable;
    // scratch pages sit beyond that registered range.
    let scratch_pages = crate::model::MTP_SCRATCH_PAGES_PER_SLOT
        .checked_mul(crate::model::glm52_decode_slots())
        .context("GLM5.2 MTP scratch page count overflow")?;
    let kv = if topology == crate::Glm52MoeTopo::Tp4 {
        // The dense FlashInfer execution cache (MLA + index-K co-allocation)
        // alone: the layer-78 wire mirrors commit into the slab pages that
        // `glm52_arena_bytes` already charges.
        let blocks = pool_blocks
            .checked_add(scratch_pages)
            .context("GLM5.2 MTP dense block count overflow")?;
        let per_block = GLM52_MODEL_LEN_ALIGN
            .checked_mul(pegainfer_kernels::ops::GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN)
            .and_then(|v| v.checked_add(crate::model::GLM52_KV_PAGE_IDXK_BYTES))
            .context("GLM5.2 MTP dense page byte count overflow")?;
        blocks
            .checked_mul(per_block)
            .context("GLM5.2 MTP dense cache byte count overflow")?
    } else {
        scratch_pages
            .checked_mul(crate::model::GLM52_KV_PAGE_STRIDE)
            .context("GLM5.2 MTP slab scratch byte count overflow")?
    };
    let rows: usize = GLM52_DECODE_BUCKETS.iter().sum();
    let indexer_logits = rows
        .checked_mul(max_model_len.next_multiple_of(256))
        .and_then(|v| v.checked_mul(size_of::<bf16>() + size_of::<f32>()))
        .context("GLM5.2 MTP indexer scratch byte count overflow")?;
    let block_tables = rows
        .checked_mul(max_model_len.div_ceil(GLM52_MODEL_LEN_ALIGN))
        .and_then(|v| v.checked_mul(size_of::<i32>()))
        .context("GLM5.2 MTP block-table byte count overflow")?;
    kv.checked_add(indexer_logits)
        .and_then(|v| v.checked_add(block_tables))
        .context("GLM5.2 MTP arena byte count overflow")
}

/// The four BF16 weights around the ordinary layer-78 decoder block.
pub(crate) struct Glm52MtpBookendWeights {
    enorm: DeviceVec,
    hnorm: DeviceVec,
    eh_proj: DeviceMatrix,
    shared_norm: DeviceVec,
}

impl Glm52MtpBookendWeights {
    pub(crate) fn new(
        enorm: DeviceVec,
        hnorm: DeviceVec,
        eh_proj: DeviceMatrix,
        shared_norm: DeviceVec,
    ) -> Result<Self> {
        ensure!(
            enorm.len == GLM52_HIDDEN,
            "GLM5.2 MTP enorm must be [{GLM52_HIDDEN}], got [{}]",
            enorm.len
        );
        ensure!(
            hnorm.len == GLM52_HIDDEN,
            "GLM5.2 MTP hnorm must be [{GLM52_HIDDEN}], got [{}]",
            hnorm.len
        );
        ensure!(
            eh_proj.rows == GLM52_HIDDEN && eh_proj.cols == MTP_FUSED_INPUT,
            "GLM5.2 MTP eh_proj must be [{GLM52_HIDDEN}, {MTP_FUSED_INPUT}], got [{}, {}]",
            eh_proj.rows,
            eh_proj.cols
        );
        ensure!(
            shared_norm.len == GLM52_HIDDEN,
            "GLM5.2 MTP shared norm must be [{GLM52_HIDDEN}], got [{}]",
            shared_norm.len
        );
        Ok(Self {
            enorm,
            hnorm,
            eh_proj,
            shared_norm,
        })
    }
}

/// Persistent MTP-only intermediates for one row bucket.
pub(crate) struct Glm52MtpScratch {
    masked_embed: Rows<GLM52_HIDDEN>,
    normed_embed: Rows<GLM52_HIDDEN>,
    normed_previous: Rows<GLM52_HIDDEN>,
    fused_input: HiddenStates,
}

pub(crate) struct Glm52MtpPrefillScratch {
    masked_embed: CudaSlice<bf16>,
    normed_embed: CudaSlice<bf16>,
    normed_previous: CudaSlice<bf16>,
    fused_input: CudaSlice<bf16>,
}

impl Glm52MtpPrefillScratch {
    pub(crate) fn new(ctx: &DeviceContext, rows: usize) -> Result<Self> {
        Ok(Self {
            masked_embed: ctx.stream.alloc_zeros(rows * GLM52_HIDDEN)?,
            normed_embed: ctx.stream.alloc_zeros(rows * GLM52_HIDDEN)?,
            normed_previous: ctx.stream.alloc_zeros(rows * GLM52_HIDDEN)?,
            fused_input: ctx.stream.alloc_zeros(rows * MTP_FUSED_INPUT)?,
        })
    }
}

impl Glm52MtpScratch {
    pub(crate) fn new(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        Ok(Self {
            masked_embed: Rows::zeros(ctx, tokens)?,
            normed_embed: Rows::zeros(ctx, tokens)?,
            normed_previous: Rows::zeros(ctx, tokens)?,
            fused_input: HiddenStates::zeros(ctx, MTP_FUSED_INPUT, tokens)?,
        })
    }
}

impl Glm52MtpBookendWeights {
    pub(crate) fn shared_norm(&self) -> &DeviceVec {
        &self.shared_norm
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_prefill_into(
        &self,
        ctx: &DeviceContext,
        positions: &CudaSlice<u32>,
        inputs_embeds: &CudaSlice<bf16>,
        previous_hidden: &CudaSlice<bf16>,
        rows: usize,
        scratch: &mut Glm52MtpPrefillScratch,
        decoder_input: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let hidden = rows * GLM52_HIDDEN;
        ensure!(
            positions.len() >= rows
                && inputs_embeds.len() >= hidden
                && previous_hidden.len() >= hidden
                && decoder_input.len() >= hidden,
            "GLM5.2 MTP prefill row buffers are smaller than {rows}"
        );
        mask_position_zero_rows_into(
            ctx,
            inputs_embeds,
            positions,
            GLM52_HIDDEN,
            rows,
            &mut scratch.masked_embed,
        )?;
        rms_norm_rows_into(
            ctx,
            &scratch.masked_embed,
            &self.enorm,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut scratch.normed_embed,
        )?;
        rms_norm_rows_into(
            ctx,
            previous_hidden,
            &self.hnorm,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut scratch.normed_previous,
        )?;
        copy_hidden_rows_raw_into(
            ctx,
            &scratch.normed_embed,
            GLM52_HIDDEN,
            &mut scratch.fused_input,
            MTP_FUSED_INPUT,
            0,
            rows,
        )?;
        copy_hidden_rows_raw_into(
            ctx,
            &scratch.normed_previous,
            GLM52_HIDDEN,
            &mut scratch.fused_input,
            MTP_FUSED_INPUT,
            GLM52_HIDDEN,
            rows,
        )?;
        gemm_strided_batched_bf16(
            ctx,
            true,
            false,
            GLM52_HIDDEN,
            rows,
            MTP_FUSED_INPUT,
            &self.eh_proj.data,
            MTP_FUSED_INPUT,
            0,
            &scratch.fused_input,
            MTP_FUSED_INPUT,
            0,
            decoder_input,
            GLM52_HIDDEN,
            0,
            1,
        )
    }
}

/// Build the ordinary layer-78 decoder input. One GEMM consumes the physical
/// concatenation so its accumulation and BF16 output boundary match vLLM's
/// `nn.Linear(torch.cat(...))`.
pub(crate) fn glm52_mtp_prepare_into(
    ctx: &DeviceContext,
    w: &Glm52MtpBookendWeights,
    positions: &CudaSlice<u32>,
    inputs_embeds: &Rows<GLM52_HIDDEN>,
    previous_hidden: &Rows<GLM52_HIDDEN>,
    s: &mut Glm52MtpScratch,
    decoder_input: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    let tokens = inputs_embeds.tokens();
    ensure!(
        previous_hidden.tokens() == tokens
            && s.masked_embed.tokens() == tokens
            && decoder_input.tokens() == tokens,
        "GLM5.2 MTP row bucket mismatch"
    );
    mask_position_zero_rows_into(
        ctx,
        inputs_embeds.data(),
        positions,
        GLM52_HIDDEN,
        tokens,
        s.masked_embed.data_mut(),
    )?;
    rms_norm_rows_into(
        ctx,
        s.masked_embed.data(),
        &w.enorm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        s.normed_embed.data_mut(),
    )?;
    rms_norm_rows_into(
        ctx,
        previous_hidden.data(),
        &w.hnorm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        tokens,
        s.normed_previous.data_mut(),
    )?;
    copy_hidden_rows_raw_into(
        ctx,
        s.normed_embed.data(),
        GLM52_HIDDEN,
        &mut s.fused_input.data,
        MTP_FUSED_INPUT,
        0,
        tokens,
    )?;
    copy_hidden_rows_raw_into(
        ctx,
        s.normed_previous.data(),
        GLM52_HIDDEN,
        &mut s.fused_input.data,
        MTP_FUSED_INPUT,
        GLM52_HIDDEN,
        tokens,
    )?;
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        GLM52_HIDDEN,
        tokens,
        MTP_FUSED_INPUT,
        &w.eh_proj.data,
        MTP_FUSED_INPUT,
        0,
        &s.fused_input.data,
        MTP_FUSED_INPUT,
        0,
        decoder_input.data_mut(),
        GLM52_HIDDEN,
        0,
        1,
    )
}

/// Normalize layer 78's raw residual output for the next MTP iteration.
/// Callers retain `raw_hidden` unchanged for the shared target lm_head path.
pub(crate) fn glm52_mtp_recycle_into(
    ctx: &DeviceContext,
    w: &Glm52MtpBookendWeights,
    raw_hidden: &Rows<GLM52_HIDDEN>,
    recycle_hidden: &mut Rows<GLM52_HIDDEN>,
) -> Result<()> {
    ensure!(
        raw_hidden.tokens() == recycle_hidden.tokens(),
        "GLM5.2 MTP recycle row bucket mismatch"
    );
    rms_norm_rows_into(
        ctx,
        raw_hidden.data(),
        &w.shared_norm,
        GLM52_RMS_EPS,
        GLM52_HIDDEN,
        raw_hidden.tokens(),
        recycle_hidden.data_mut(),
    )
}
