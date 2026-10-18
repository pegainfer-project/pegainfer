//! GLM5.2 native TP4 prefill executor, layer-outer over the prefill
//! chunk: every layer stage (norm, MLA front, KV pack, o_proj, MoE) runs at
//! chunk M so the fp8 GEMMs stay large and each MoE layer reads its expert
//! bank once per chunk (not once per tile). Only two stages sub-tile:
//! the DSA indexer (32 rows — the DeepGEMM paged-MQA AOT batch) and the
//! FlashMLA sparse attention (`PREFILL_ATTN_TILE_ROWS`). TP reductions ride
//! NCCL bf16 all-reduces (`Glm52MoeTpState::prefill_allreduce`).
//!
//! Causality note: packing the whole chunk's KV before attention is safe —
//! the indexer masks each query to `positions[row] + 1` keys, so keys packed
//! for later in-chunk positions are never selected for earlier queries.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::add_into;
use pegainfer_kernels::ops::argmax_batch_bf16_split_partials_len;
use pegainfer_kernels::ops::argmax_bf16_split_into;
use pegainfer_kernels::ops::embedding_rows_into;
use pegainfer_kernels::ops::fused_add_rms_norm_round_into;
use pegainfer_kernels::ops::gemm_strided_batched_bf16;
use pegainfer_kernels::ops::glm52_flashmla_sparse_prefill_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_ue8m0_launch;
use pegainfer_kernels::ops::glm52_mla_cache_pack_launch;
use pegainfer_kernels::ops::glm52_mla_front_pack_fp8_launch;
use pegainfer_kernels::ops::glm52_mla_query_assemble_launch;
use pegainfer_kernels::ops::glm52_prefill_moe_gather_rows_launch;
use pegainfer_kernels::ops::glm52_prefill_unpack_pages_launch;
use pegainfer_kernels::ops::glm52_vocab_parallel_pack_launch;
use pegainfer_kernels::ops::glm52_vocab_parallel_unpack_launch;
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStatesRef;
use pegainfer_sample::BatchSamplingRow;
use pegainfer_sample::BatchSamplingScratch;
use pegainfer_sample::gpu_sample_batch_into;
use pegainfer_sample::mix_seed;

use crate::bookend::glm52_final_norm_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_KV_A_OUT;
use crate::config::GLM52_KV_LORA_RANK;
use crate::config::GLM52_MTP_LAYER;
use crate::config::GLM52_QK_HEAD_DIM;
use crate::config::GLM52_QK_NOPE_HEAD_DIM;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_VOCAB;
use crate::dense::Glm52DenseMlpWeights;
use crate::dense::glm52_dense_mlp_prefill_into;
use crate::fp8::Glm52Fp8GemmScratch;
use crate::fp8::fp8_linear_large_m_into;
use crate::indexer::Glm52IndexerPrefillScratch;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52KvSlab;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerIndexer;
use crate::layer::Glm52LayerMlp;
use crate::mla_front::Glm52MlaFront;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::mla_front::glm52_mla_prefill_front_into;
use crate::model::GLM52_KV_PAGE_IDXK_BYTES;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::INDEX_CACHE_BLOCK;
use crate::moe_tp::Glm52MoeTpPrefillScratch;
use crate::moe_tp::Glm52MoeTpRank;
use crate::moe_tp::Glm52MoeTpState;
use crate::mtp::Glm52MtpBookendWeights;
use crate::mtp::Glm52MtpPrefillScratch;
use crate::rows::Rows;
use crate::runner::Glm52PrefillBatch;

/// FlashMLA sparse attention sub-tile (query rows per launch).
///
/// The SM100 sparse-prefill kernel accepts arbitrary positive query rows.
/// Keeping 4K rows in the temporary 64-head buffers cuts a 16K prefill
/// chunk from 32 launches to 4 while staying inside the fixed
/// prefill scratch reservation.
const PREFILL_ATTN_TILE_ROWS: usize = 4096;
/// Dense-MLP sub-tile: bounds the 12288-wide gate|up scratch.
const PREFILL_DENSE_TILE_ROWS: usize = 2048;

const GLM52_INDEXER_TOPK: usize = 2048;

fn mtp_shifted_tokens(batch: &Glm52PrefillBatch, boundary_outputs: &[u32]) -> Result<Vec<u32>> {
    ensure!(
        boundary_outputs.len() == batch.output_rows.len(),
        "GLM5.2 MTP boundary outputs {} != output rows {}",
        boundary_outputs.len(),
        batch.output_rows.len()
    );
    let mut shifted = vec![0u32; batch.token_ids.len()];
    let mut boundary = 0usize;
    for (request, range) in batch.request_indptr.windows(2).enumerate() {
        let start = range[0] as usize;
        let end = range[1] as usize;
        shifted[start..end - 1].copy_from_slice(&batch.token_ids[start + 1..end]);
        shifted[end - 1] = match batch.mtp_next_tokens[request] {
            Some(token) => token,
            None => {
                ensure!(
                    batch.output_rows.get(boundary).copied() == Some((end - 1) as u32),
                    "GLM5.2 MTP boundary row does not end request range {request}"
                );
                let token = boundary_outputs[boundary];
                boundary += 1;
                token
            }
        };
    }
    ensure!(
        boundary == boundary_outputs.len(),
        "GLM5.2 MTP consumed {boundary} boundary outputs, got {}",
        boundary_outputs.len()
    );
    Ok(shifted)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Glm52TpPrefillLayout {
    kv_slots: usize,
    table_width: usize,
    chunk_rows: usize,
}

impl Glm52TpPrefillLayout {
    /// `kv_slots` starts at 0: the pool size is decided by the measured
    /// launch fill AFTER the fixed buffers are built, and
    /// [`Glm52TpPrefillExecutor::attach_kv_pool`] fills it in.
    fn new(table_width: usize, chunk_rows: usize) -> Result<Self> {
        ensure!(
            table_width > 0 && chunk_rows > 0,
            "prefill capacities must be positive"
        );
        Ok(Self {
            kv_slots: 0,
            table_width,
            chunk_rows: chunk_rows.next_multiple_of(4),
        })
    }
}

/// Env-gated (`PEGAINFER_GLM52_PREFILL_PROFILE=1`) CUDA-event section
/// profile: per-section call counts and summed GPU ms per chunk forward.
struct Glm52PrefillProfiler {
    enabled: bool,
    sections: Vec<(&'static str, usize, Vec<(CudaEvent, CudaEvent)>)>,
}

impl Glm52PrefillProfiler {
    fn new() -> Self {
        Self {
            enabled: std::env::var("PEGAINFER_GLM52_PREFILL_PROFILE").as_deref() == Ok("1"),
            sections: Vec::new(),
        }
    }

    fn start(&self, ctx: &DeviceContext) -> Result<Option<CudaEvent>> {
        if !self.enabled {
            return Ok(None);
        }
        // Explicit default flags: cudarc's `None` means DISABLE_TIMING, which
        // would make `elapsed_ms` fail with CUDA_ERROR_INVALID_HANDLE.
        Ok(Some(ctx.stream.record_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ))?))
    }

    fn stop(
        &mut self,
        ctx: &DeviceContext,
        name: &'static str,
        begin: Option<CudaEvent>,
    ) -> Result<()> {
        let Some(begin) = begin else {
            return Ok(());
        };
        let end = ctx
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        match self.sections.iter_mut().find(|(n, _, _)| *n == name) {
            Some((_, count, pairs)) => {
                *count += 1;
                pairs.push((begin, end));
            }
            None => self.sections.push((name, 1, vec![(begin, end)])),
        }
        Ok(())
    }

    fn report(&mut self, ctx: &DeviceContext, rows: usize) -> Result<()> {
        if !self.enabled || self.sections.is_empty() {
            self.sections.clear();
            return Ok(());
        }
        ctx.stream.synchronize()?;
        let mut lines = Vec::with_capacity(self.sections.len());
        let mut total = 0.0f64;
        for (name, count, pairs) in &self.sections {
            let mut ms = 0.0f64;
            for (begin, end) in pairs {
                ms += f64::from(begin.elapsed_ms(end)?);
            }
            total += ms;
            lines.push(format!("\"{name}\": ({count}, {ms:.3})"));
        }
        log::info!(
            "GLM5.2 TP4 prefill CUDA-event profile: device={}, rows={rows}, \
             section_total_ms={total:.3}, sections={{{}}}",
            ctx.device_ordinal,
            lines.join(", ")
        );
        self.sections.clear();
        Ok(())
    }
}

pub(crate) struct Glm52TpPrefillExecutor {
    layout: Glm52TpPrefillLayout,
    profiler: Glm52PrefillProfiler,
    // ---- chunk-scale buffers ----
    token_ids: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    hidden: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    mla_front: Glm52MlaFront,
    ql_nope: CudaSlice<bf16>,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
    slot_mapping: CudaSlice<i64>,
    // ---- pool-scaled buffers, attached by `attach_kv_pool` once the
    // measured fill decides the block count (None only between build_fixed
    // and finish_kv — never during serving) ----
    block_ids: Option<CudaSlice<i32>>,
    unpacked_kv: Option<CudaSlice<bf16>>,
    /// The slab index-K layout (pool block count, page stride, offset 0);
    /// each full-indexer launch applies its layer's slice offset. The MTP
    /// proposal path builds its own dense layout instead.
    index_cache_layout: Option<Glm52IndexerCacheLayout>,
    fp8_gemm: Glm52Fp8GemmScratch,
    attention_v: CudaSlice<bf16>,
    attention_partial: CudaSlice<bf16>,
    attention_reduced: CudaSlice<bf16>,
    mlp_out: CudaSlice<bf16>,
    // Cross-layer indexer carry at chunk scale: a full-indexer layer fills
    // it in attention-tile slices; shared layers reuse it.
    carry_slots: CudaSlice<i32>,
    carry_lens: CudaSlice<i32>,
    // ---- chunk-scale DSA indexer (unpaged MQA) ----
    indexer: Glm52IndexerPrefillScratch,
    // ---- attention sub-tile buffers ----
    query_bf16: CudaSlice<bf16>,
    attention_out: CudaSlice<bf16>,
    attention_max: CudaSlice<f32>,
    attention_lse: CudaSlice<f32>,
    attention_v_sub: CudaSlice<bf16>,
    // ---- dense-MLP sub-tile buffers ----
    dense_gate_up: CudaSlice<bf16>,
    dense_silu: CudaSlice<bf16>,
    dense_out_sub: CudaSlice<bf16>,
    dense_gemm: Glm52Fp8GemmScratch,
    // ---- MoE (chunk-scale) ----
    moe: Glm52MoeTpPrefillScratch,
    // ---- output tail (fixed 32-row blocks) ----
    output_rows: CudaSlice<i32>,
    final_hidden: Rows<GLM52_HIDDEN>,
    final_normed: Rows<GLM52_HIDDEN>,
    logits: Rows<GLM52_VOCAB>,
    argmax_partial_values: CudaSlice<f32>,
    argmax_partial_indices: CudaSlice<i32>,
    argmax_values: CudaSlice<bf16>,
    argmax_indices: CudaSlice<i32>,
    // ---- native MTP context pass (chunk-scale) ----
    mtp_embeds: CudaSlice<bf16>,
    mtp_previous: CudaSlice<bf16>,
    mtp_decoder_input: CudaSlice<bf16>,
    mtp_bookend_scratch: Glm52MtpPrefillScratch,
    mtp_flashinfer_query: CudaSlice<u8>,
    /// Target final-normalized boundary rows retained for the small-M native
    /// proposal loop after the large-M committed context pass.
    mtp_target_boundary: Rows<GLM52_HIDDEN>,
    /// Large-M MTP boundary state seeds the proposal loop directly. Reusing
    /// it avoids recomputing the boundary through the decode attention path.
    mtp_proposal_boundary: Rows<GLM52_HIDDEN>,
}

/// The TP4 dense layer-78 caches, lent by `model::mtp` for one chunk. The
/// proposal slab is the FlashInfer execution cache (`proposal_caches` maps
/// its dense MLA region and tight index-K region); `slab_caches` are the
/// layer-78 mirror slices of a KV slab page, where the P/D wire rows
/// (fp8_ds_mla + index-K) commit — the slab is the only registered arena,
/// so this commit is what the decode side restores.
pub(crate) struct Glm52TpPrefillMtpView<'a> {
    pub(crate) bookend: &'a Glm52MtpBookendWeights,
    pub(crate) layer: &'a Glm52DecoderLayerWeights,
    pub(crate) slab_caches: Glm52LayerCaches,
    pub(crate) proposal: &'a mut Glm52KvSlab,
    pub(crate) proposal_caches: Glm52LayerCaches,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Glm52PrefillOutput {
    pub(crate) target_tokens: Vec<u32>,
    /// First native-MTP proposal token for every target boundary row.
    ///
    /// This is deliberately separate from `target_tokens`: the target token
    /// is the anchor emitted to the client, while this token is uncommitted
    /// proposal metadata for the decode worker's first verify span.
    pub(crate) mtp_draft1: Vec<u32>,
    /// Complete five-token proposal generated after the committed context
    /// pass. Empty only when native MTP is disabled or there is no boundary.
    pub(crate) mtp_drafts: Vec<[u32; crate::mtp::GLM52_MTP_DRAFTS]>,
}

pub(crate) struct Glm52TpPrefillModelView<'a> {
    pub(crate) layers: &'a [Glm52DecoderLayerWeights],
    /// The rank's page-first KV slab; `caches` carries each layer's slice
    /// offsets inside a page.
    pub(crate) slab: &'a mut Glm52KvSlab,
    pub(crate) caches: &'a [Glm52LayerCaches],
    pub(crate) embed: &'a DeviceMatrix,
    pub(crate) cos_table: &'a DeviceMatrix,
    pub(crate) sin_table: &'a DeviceMatrix,
    pub(crate) final_norm: &'a DeviceVec,
    pub(crate) shard_lm_head: &'a DeviceMatrix,
    pub(crate) full_lm_head: &'a DeviceMatrix,
    pub(crate) vocab_start: usize,
    pub(crate) sampling_scratch: &'a mut BatchSamplingScratch,
    pub(crate) mtp: Option<Glm52TpPrefillMtpView<'a>>,
}

impl Glm52TpPrefillExecutor {
    /// Build everything EXCEPT the pool-scaled buffers: the KV pool size is
    /// measured after this returns, and [`Self::attach_kv_pool`] installs
    /// the pool geometry (block ids, unpacked KV, index-K layout) before any
    /// forward.
    pub(crate) fn new(
        ctx: &DeviceContext,
        table_width: usize,
        chunk_rows: usize,
        topology: pegainfer_kernels::ops::Glm52TpTopology,
    ) -> Result<Self> {
        let layout = Glm52TpPrefillLayout::new(table_width, chunk_rows)?;
        let chunk = layout.chunk_rows;
        let attn = PREFILL_ATTN_TILE_ROWS;
        let dense = PREFILL_DENSE_TILE_ROWS.min(chunk);
        Ok(Self {
            profiler: Glm52PrefillProfiler::new(),
            token_ids: ctx.stream.alloc_zeros::<u32>(chunk)?,
            positions: ctx.stream.alloc_zeros::<u32>(chunk)?,
            hidden: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            normed: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            cos: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_ROPE_HALF)?,
            mla_front: Glm52MlaFront::new_prefill(ctx, chunk, 16)?,
            ql_nope: ctx
                .stream
                .alloc_zeros::<bf16>(chunk * 16 * GLM52_KV_LORA_RANK)?,
            ckv_fp8: ctx.stream.alloc_zeros::<u8>(chunk * GLM52_KV_LORA_RANK)?,
            ckv_scales: ctx.stream.alloc_zeros::<f32>(chunk * 4)?,
            slot_mapping: ctx.stream.alloc_zeros::<i64>(chunk)?,
            block_ids: None,
            unpacked_kv: None,
            index_cache_layout: None,
            fp8_gemm: Glm52Fp8GemmScratch::new(ctx, chunk, GLM52_HIDDEN)?,
            attention_v: ctx.stream.alloc_zeros::<bf16>(chunk * 16 * 256)?,
            attention_partial: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            attention_reduced: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            mlp_out: ctx.stream.alloc_zeros::<bf16>(chunk * GLM52_HIDDEN)?,
            carry_slots: ctx.stream.alloc_zeros::<i32>(chunk * GLM52_INDEXER_TOPK)?,
            carry_lens: ctx.stream.alloc_zeros::<i32>(chunk)?,
            indexer: Glm52IndexerPrefillScratch::new(
                ctx,
                chunk,
                PREFILL_ATTN_TILE_ROWS,
                table_width,
            )?,
            query_bf16: ctx.stream.alloc_zeros::<bf16>(attn * 64 * GLM52_KV_A_OUT)?,
            attention_out: ctx
                .stream
                .alloc_zeros::<bf16>(attn * 64 * GLM52_KV_LORA_RANK)?,
            attention_max: ctx.stream.alloc_zeros::<f32>(attn * 64)?,
            attention_lse: ctx.stream.alloc_zeros::<f32>(attn * 64)?,
            attention_v_sub: ctx.stream.alloc_zeros::<bf16>(attn * 16 * 256)?,
            dense_gate_up: ctx
                .stream
                .alloc_zeros::<bf16>(dense * 2 * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_silu: ctx
                .stream
                .alloc_zeros::<bf16>(dense * crate::config::GLM52_DENSE_INTERMEDIATE)?,
            dense_out_sub: ctx.stream.alloc_zeros::<bf16>(dense * GLM52_HIDDEN)?,
            dense_gemm: Glm52Fp8GemmScratch::new(
                ctx,
                dense,
                crate::config::GLM52_DENSE_INTERMEDIATE,
            )?,
            moe: Glm52MoeTpPrefillScratch::new(ctx, topology, chunk)?,
            output_rows: ctx.stream.alloc_zeros(32)?,
            final_hidden: Rows::zeros(ctx, 32)?,
            final_normed: Rows::zeros(ctx, 32)?,
            logits: Rows::zeros(ctx, 32)?,
            argmax_partial_values: ctx
                .stream
                .alloc_zeros(argmax_batch_bf16_split_partials_len(32, GLM52_VOCAB))?,
            argmax_partial_indices: ctx
                .stream
                .alloc_zeros(argmax_batch_bf16_split_partials_len(32, GLM52_VOCAB))?,
            argmax_values: ctx.stream.alloc_zeros(32)?,
            argmax_indices: ctx.stream.alloc_zeros(32)?,
            mtp_embeds: ctx.stream.alloc_zeros(chunk * GLM52_HIDDEN)?,
            mtp_previous: ctx.stream.alloc_zeros(chunk * GLM52_HIDDEN)?,
            mtp_decoder_input: ctx.stream.alloc_zeros(chunk * GLM52_HIDDEN)?,
            mtp_bookend_scratch: Glm52MtpPrefillScratch::new(ctx, chunk)?,
            mtp_flashinfer_query: ctx.stream.alloc_zeros(chunk * 16 * GLM52_KV_A_OUT)?,
            mtp_target_boundary: Rows::zeros(ctx, GLM52_MAX_BATCH_PER_RANK)?,
            mtp_proposal_boundary: Rows::zeros(ctx, GLM52_MAX_BATCH_PER_RANK)?,
            layout,
        })
    }

    /// Allocate the two pool-scaled buffers (the page-id upload scratch and
    /// the unpacked bf16 KV pool) and bind the launch-decided index-K layout,
    /// once the measured fill has fixed the pool block count. Must run before
    /// the first forward; the executor's other buffers are len/chunk-scaled
    /// and were already allocated by [`Self::new`].
    pub(crate) fn attach_kv_pool(
        &mut self,
        ctx: &DeviceContext,
        kv_slots: usize,
        index_cache_layout: Glm52IndexerCacheLayout,
    ) -> Result<()> {
        ensure!(
            self.layout.kv_slots == 0 && self.block_ids.is_none() && self.unpacked_kv.is_none(),
            "GLM5.2 prefill KV pool attached twice"
        );
        ensure!(kv_slots > 0, "GLM5.2 prefill KV pool must be non-empty");
        self.layout.kv_slots = kv_slots;
        self.block_ids = Some(ctx.stream.alloc_zeros::<i32>(kv_slots.div_ceil(64))?);
        self.unpacked_kv = Some(ctx.stream.alloc_zeros::<bf16>(kv_slots * GLM52_KV_A_OUT)?);
        self.index_cache_layout = Some(index_cache_layout);
        Ok(())
    }

    pub(crate) fn mtp_target_boundary(&self) -> &Rows<GLM52_HIDDEN> {
        &self.mtp_target_boundary
    }

    pub(crate) fn mtp_proposal_boundary(&self) -> &Rows<GLM52_HIDDEN> {
        &self.mtp_proposal_boundary
    }

    /// Run the complete TP4 prefill forward for one prefill batch,
    /// layer-outer: each of the 78 layers processes the whole chunk before
    /// the next layer starts. Returns tokens only for request boundary rows.
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        batch: &Glm52PrefillBatch,
        tp: &mut Glm52MoeTpRank,
        model: Glm52TpPrefillModelView<'_>,
    ) -> Result<Glm52PrefillOutput> {
        ensure!(
            model.layers.len() == model.caches.len() && !model.layers.is_empty(),
            "GLM5.2 TP prefill layer/cache layout is invalid"
        );
        let rows = batch.token_ids.len();
        ensure!(
            rows > 0 && rows <= self.layout.chunk_rows,
            "GLM5.2 TP prefill batch of {rows} rows exceeds the chunk capacity {}",
            self.layout.chunk_rows
        );
        let rows4 = rows.next_multiple_of(4);

        let mark = self.profiler.start(ctx)?;
        self.stage_chunk(ctx, batch, model.embed, model.cos_table, model.sin_table)?;
        self.indexer.stage_chunk(ctx, batch)?;
        self.profiler.stop(ctx, "embedding_rope_stage", mark)?;

        let mark = self.profiler.start(ctx)?;
        rms_norm_rows_into(
            ctx,
            &self.hidden,
            &model.layers[0].input_ln,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows4,
            &mut self.normed,
        )?;
        self.profiler.stop(ctx, "input_norm", mark)?;

        let mut carry_ready = false;
        for layer in 0..model.layers.len() {
            let weights = &model.layers[layer];
            let cache = model.caches[layer];

            let mark = self.profiler.start(ctx)?;
            glm52_mla_prefill_front_into(
                ctx,
                &weights.mla,
                rows4,
                &self.normed,
                &mut self.fp8_gemm,
                &mut self.mla_front,
            )?;
            self.profiler.stop(ctx, "mla_front", mark)?;

            let mark = self.profiler.start(ctx)?;
            self.pack_mla_cache(
                ctx,
                &weights.mla,
                &mut model.slab.slab,
                cache.mla_offset,
                model.slab.page_stride,
                model.slab.num_blocks,
                rows,
            )?;
            self.profiler.stop(ctx, "mla_pack_cache", mark)?;

            if !batch.block_ids.is_empty() {
                let mark = self.profiler.start(ctx)?;
                glm52_prefill_unpack_pages_launch(
                    ctx,
                    &model.slab.slab,
                    cache.mla_offset,
                    model.slab.page_stride,
                    pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
                    self.block_ids
                        .as_ref()
                        .context("GLM5.2 prefill KV pool is not attached")?,
                    batch.block_ids.len(),
                    self.unpacked_kv
                        .as_mut()
                        .context("GLM5.2 prefill KV pool is not attached")?,
                )?;
                self.profiler.stop(ctx, "kv_page_unpack", mark)?;
            }

            match &weights.indexer {
                Glm52LayerIndexer::Full(indexer) => {
                    let mark = self.profiler.start(ctx)?;
                    let index_k_offset = cache
                        .index_k_offset
                        .context("GLM5.2 full prefill indexer is missing its page slice")?;
                    let mut layout = self
                        .index_cache_layout
                        .context("GLM5.2 prefill KV pool is not attached")?;
                    layout.cache_layer_offset_bytes = index_k_offset;
                    self.indexer.run_layer(
                        ctx,
                        indexer,
                        &self.normed,
                        self.mla_front.q_resid.data(),
                        &self.cos,
                        &self.sin,
                        &mut model.slab.slab,
                        layout,
                        &self.slot_mapping,
                        rows,
                        &mut self.fp8_gemm,
                        &mut self.carry_slots,
                        &mut self.carry_lens,
                    )?;
                    carry_ready = true;
                    self.profiler.stop(ctx, "indexer_full", mark)?;
                }
                Glm52LayerIndexer::Shared => {
                    ensure!(
                        carry_ready,
                        "GLM5.2 shared prefill indexer has no top-k carry"
                    );
                }
            }

            let mark = self.profiler.start(ctx)?;
            self.attend_chunk(ctx, &weights.mla, rows)?;
            self.profiler.stop(ctx, "sparse_attention", mark)?;

            let mark = self.profiler.start(ctx)?;
            fp8_linear_large_m_into(
                ctx,
                &weights.mla.o_proj,
                rows4,
                &self.attention_v,
                &mut self.fp8_gemm,
                &mut self.attention_partial,
            )?;
            self.profiler.stop(ctx, "o_proj", mark)?;

            let mark = self.profiler.start(ctx)?;
            self.reduce_and_norm_attention(ctx, &mut tp.state, &weights.post_attn_ln, rows)?;
            self.profiler.stop(ctx, "attention_out_reduce_norm", mark)?;

            match &weights.mlp {
                Glm52LayerMlp::Dense(dense) => {
                    let mark = self.profiler.start(ctx)?;
                    self.dense_mlp(ctx, dense, rows4)?;
                    self.profiler.stop(ctx, "dense_mlp", mark)?;
                }
                Glm52LayerMlp::MoeTp(router) => {
                    let (state, _, bank) = tp.layer_bank(layer).with_context(|| {
                        format!("GLM5.2 TP prefill layer {layer} has no expert slice bank")
                    })?;
                    let mark = self.profiler.start(ctx)?;
                    self.moe.forward(
                        ctx,
                        state,
                        router,
                        bank,
                        &self.normed,
                        rows,
                        &mut self.mlp_out,
                    )?;
                    self.profiler.stop(ctx, "moe_mlp", mark)?;
                    let mark = self.profiler.start(ctx)?;
                    state.prefill_allreduce_in_place(ctx, rows, &mut self.mlp_out)?;
                    self.profiler.stop(ctx, "moe_reduce", mark)?;
                }
                Glm52LayerMlp::MoeEp8(_) => {
                    anyhow::bail!("GLM5.2 TP prefill layer {layer} has EP weights");
                }
            }

            let mark = self.profiler.start(ctx)?;
            self.finish_layer(
                ctx,
                model.layers.get(layer + 1).map(|next| &next.input_ln),
                rows,
            )?;
            self.profiler.stop(ctx, "residual_next_norm", mark)?;
        }

        let mut outputs = Vec::with_capacity(batch.output_rows.len());
        let local_outputs: Vec<i32> = batch.output_rows.iter().map(|&row| row as i32).collect();
        let mark = self.profiler.start(ctx)?;
        for rows_block in local_outputs.chunks(32) {
            let output_base = outputs.len();
            let sampling: Vec<_> = batch
                .sampling
                .iter()
                .filter(|sample| {
                    (output_base..output_base + rows_block.len()).contains(&sample.row)
                })
                .map(|sample| {
                    let mut sample = *sample;
                    sample.row -= output_base;
                    sample
                })
                .collect();
            outputs.extend(self.output_tokens(
                ctx,
                &mut tp.state,
                model.final_norm,
                model.shard_lm_head,
                model.full_lm_head,
                model.vocab_start,
                rows_block,
                &sampling,
                batch.seed,
                model.sampling_scratch,
            )?);
        }
        self.profiler.stop(ctx, "lm_head_sampling", mark)?;
        if !batch.output_rows.is_empty() {
            ensure!(
                batch.output_rows.len() <= GLM52_MAX_BATCH_PER_RANK,
                "GLM5.2 prefill boundary rows exceed native-MTP capacity"
            );
            ctx.stream.memcpy_dtod(
                &self
                    .final_normed
                    .data()
                    .slice(..batch.output_rows.len() * GLM52_HIDDEN),
                &mut self
                    .mtp_target_boundary
                    .data_mut()
                    .slice_mut(..batch.output_rows.len() * GLM52_HIDDEN),
            )?;
        }
        let mtp_draft1 = if let Some(mtp) = model.mtp {
            let mark = self.profiler.start(ctx)?;
            let drafts = self.run_mtp_context(
                ctx,
                batch,
                &outputs,
                tp,
                model.embed,
                model.final_norm,
                model.shard_lm_head,
                model.full_lm_head,
                model.vocab_start,
                model.sampling_scratch,
                mtp,
                model.slab,
                rows,
                rows4,
            )?;
            self.profiler.stop(ctx, "mtp_context", mark)?;
            drafts
        } else {
            Vec::new()
        };
        self.profiler.report(ctx, rows)?;
        Ok(Glm52PrefillOutput {
            target_tokens: outputs,
            mtp_draft1,
            mtp_drafts: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mtp_context(
        &mut self,
        ctx: &DeviceContext,
        batch: &Glm52PrefillBatch,
        boundary_outputs: &[u32],
        tp: &mut Glm52MoeTpRank,
        embed: &DeviceMatrix,
        final_norm: &DeviceVec,
        shard_lm_head: &DeviceMatrix,
        full_lm_head: &DeviceMatrix,
        vocab_start: usize,
        sampling_scratch: &mut BatchSamplingScratch,
        mtp: Glm52TpPrefillMtpView<'_>,
        slab: &mut Glm52KvSlab,
        rows: usize,
        rows4: usize,
    ) -> Result<Vec<u32>> {
        let shifted = mtp_shifted_tokens(batch, boundary_outputs)?;
        ctx.stream
            .memcpy_htod(&shifted, &mut self.token_ids.slice_mut(..rows))?;
        embedding_rows_into(ctx, embed, &self.token_ids, rows, &mut self.mtp_embeds)?;
        rms_norm_rows_into(
            ctx,
            &self.hidden,
            final_norm,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut self.mtp_previous,
        )?;
        mtp.bookend.prepare_prefill_into(
            ctx,
            &self.positions,
            &self.mtp_embeds,
            &self.mtp_previous,
            rows,
            &mut self.mtp_bookend_scratch,
            &mut self.mtp_decoder_input,
        )?;
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .mtp_decoder_input
                    .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
            )?;
        }
        ctx.stream.memcpy_dtod(
            &self.mtp_decoder_input.slice(..rows * GLM52_HIDDEN),
            &mut self.hidden.slice_mut(..rows * GLM52_HIDDEN),
        )?;
        rms_norm_rows_into(
            ctx,
            &self.hidden,
            &mtp.layer.input_ln,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows4,
            &mut self.normed,
        )?;

        glm52_mla_prefill_front_into(
            ctx,
            &mtp.layer.mla,
            rows4,
            &self.normed,
            &mut self.fp8_gemm,
            &mut self.mla_front,
        )?;
        // The P/D wire commit: layer-78 fp8_ds_mla rows go straight into the
        // KV slab's mirror slices — the page is the only registered arena,
        // so rows that miss it never reach the decode side's restore.
        self.pack_mla_cache(
            ctx,
            &mtp.layer.mla,
            &mut slab.slab,
            mtp.slab_caches.mla_offset,
            slab.page_stride,
            slab.num_blocks,
            rows,
        )?;
        glm52_mla_front_pack_fp8_launch(
            ctx,
            rows,
            16,
            &self.ql_nope,
            &self.mla_front.q_full,
            GLM52_QK_NOPE_HEAD_DIM,
            GLM52_QK_HEAD_DIM,
            &self.mla_front.ckv,
            &mtp.layer.mla.kv_a_ln.data,
            GLM52_RMS_EPS,
            &self.cos,
            &self.sin,
            &mut self.mtp_flashinfer_query,
            &mut mtp.proposal.slab,
            &self.slot_mapping,
        )?;
        if !batch.block_ids.is_empty() {
            glm52_prefill_unpack_pages_launch(
                ctx,
                &slab.slab,
                mtp.slab_caches.mla_offset,
                slab.page_stride,
                pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
                self.block_ids
                    .as_ref()
                    .context("GLM5.2 prefill KV pool is not attached")?,
                batch.block_ids.len(),
                self.unpacked_kv
                    .as_mut()
                    .context("GLM5.2 prefill KV pool is not attached")?,
            )?;
        }
        let Glm52LayerIndexer::Full(indexer) = &mtp.layer.indexer else {
            anyhow::bail!("GLM5.2 MTP layer 78 must own a full indexer")
        };
        let proposal_index_k_offset = mtp
            .proposal_caches
            .index_k_offset
            .context("GLM5.2 MTP layer 78 is missing index-K cache")?;
        let slab_index_k_offset = mtp
            .slab_caches
            .index_k_offset
            .context("GLM5.2 MTP layer 78 slab slices are missing index-K")?;
        // The slab's layer-78 slice is the indexer's primary cache: the
        // chunk's rows commit there (the P/D wire, like the MLA pack above),
        // and the topk gather sees host-restored prefix rows — the dense
        // proposal cache is never restored into, so gathering from it would
        // read zeros for every restored page.
        self.indexer.run_layer(
            ctx,
            indexer,
            &self.normed,
            self.mla_front.q_resid.data(),
            &self.cos,
            &self.sin,
            &mut slab.slab,
            Glm52IndexerCacheLayout {
                cache_blocks: slab.num_blocks,
                cache_block_size: INDEX_CACHE_BLOCK,
                cache_layer_offset_bytes: slab_index_k_offset,
                cache_block_stride_bytes: slab.page_stride,
            },
            &self.slot_mapping,
            rows,
            &mut self.fp8_gemm,
            &mut self.carry_slots,
            &mut self.carry_lens,
        )?;
        // Row-mirror the same rows into the dense proposal cache for the
        // proposal-decode rounds. Row-granular on purpose: a block-granular
        // copy would drag whole pages across and clobber restored slab
        // content for pages this chunk only partially wrote.
        self.indexer.commit_k_rows(
            ctx,
            &mut mtp.proposal.slab,
            Glm52IndexerCacheLayout {
                cache_blocks: mtp.proposal.num_blocks,
                cache_block_size: INDEX_CACHE_BLOCK,
                cache_layer_offset_bytes: proposal_index_k_offset,
                cache_block_stride_bytes: GLM52_KV_PAGE_IDXK_BYTES,
            },
            &self.slot_mapping,
            rows,
        )?;
        self.attend_chunk(ctx, &mtp.layer.mla, rows)?;
        fp8_linear_large_m_into(
            ctx,
            &mtp.layer.mla.o_proj,
            rows4,
            &self.attention_v,
            &mut self.fp8_gemm,
            &mut self.attention_partial,
        )?;
        self.reduce_and_norm_attention(ctx, &mut tp.state, &mtp.layer.post_attn_ln, rows)?;
        let Glm52LayerMlp::MoeTp(router) = &mtp.layer.mlp else {
            anyhow::bail!("GLM5.2 TP4 MTP layer 78 is not TP MoE")
        };
        let (state, _, bank) = tp
            .layer_bank(GLM52_MTP_LAYER)
            .context("GLM5.2 TP4 MTP layer 78 has no expert slice bank")?;
        self.moe.forward(
            ctx,
            state,
            router,
            bank,
            &self.normed,
            rows,
            &mut self.mlp_out,
        )?;
        state.prefill_allreduce_in_place(ctx, rows, &mut self.mlp_out)?;
        self.finish_layer(ctx, None, rows)?;
        if batch.output_rows.is_empty() {
            return Ok(Vec::new());
        }
        let boundary_rows: Vec<i32> = batch.output_rows.iter().map(|&row| row as i32).collect();
        let mut drafts = Vec::with_capacity(boundary_rows.len());
        for rows_block in boundary_rows.chunks(32) {
            drafts.extend(self.output_tokens(
                ctx,
                &mut tp.state,
                mtp.bookend.shared_norm(),
                shard_lm_head,
                full_lm_head,
                vocab_start,
                rows_block,
                &[],
                batch.seed,
                sampling_scratch,
            )?);
        }
        let boundary_count = boundary_rows.len();
        ctx.stream.memcpy_dtod(
            &self
                .final_normed
                .data()
                .slice(..boundary_count * GLM52_HIDDEN),
            &mut self
                .mtp_proposal_boundary
                .data_mut()
                .slice_mut(..boundary_count * GLM52_HIDDEN),
        )?;
        Ok(drafts)
    }

    /// Upload token ids/positions/slot mapping/block list and stage
    /// embeddings + rope rows for the whole chunk.
    fn stage_chunk(
        &mut self,
        ctx: &DeviceContext,
        batch: &Glm52PrefillBatch,
        embed: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
    ) -> Result<()> {
        let rows = batch.token_ids.len();
        ensure!(
            batch.positions.len() == rows && batch.slot_mapping.len() == rows,
            "prefill chunk rows/positions mismatch"
        );
        // The cache-pack kernel traps (unrecoverable launch failure with no
        // context) on any out-of-window slot. Catch a poisoned batch here
        // with context instead — this is the contract that the scheduler's
        // pool and the executor's cache slab are sized from the SAME slot
        // count (a drift once shipped page ids past the slab, #816).
        let kv_slots = self.layout.kv_slots as i64;
        for (row, &slot) in batch.slot_mapping.iter().enumerate() {
            ensure!(
                (0..kv_slots).contains(&slot),
                "prefill chunk slot_mapping[{row}] = {slot} outside the {kv_slots}-slot cache \
                 window: position={} request_indptr={:?} block_indptr={:?} block_ids(head)={:?}",
                batch.positions[row],
                batch.request_indptr,
                batch.block_indptr,
                &batch.block_ids[..batch.block_ids.len().min(24)],
            );
        }
        for (i, &page) in batch.block_ids.iter().enumerate() {
            ensure!(
                page >= 0 && (page as i64 + 1) * 64 <= kv_slots,
                "prefill chunk block_ids[{i}] = {page} outside the {kv_slots}-slot cache window \
                 (indptr {:?})",
                batch.block_indptr,
            );
        }
        ctx.stream
            .memcpy_htod(&batch.token_ids, &mut self.token_ids.slice_mut(..rows))?;
        ctx.stream
            .memcpy_htod(&batch.positions, &mut self.positions.slice_mut(..rows))?;
        ctx.stream.memcpy_htod(
            &batch.slot_mapping,
            &mut self.slot_mapping.slice_mut(..rows),
        )?;
        if !batch.block_ids.is_empty() {
            let block_ids = self
                .block_ids
                .as_mut()
                .context("GLM5.2 prefill KV pool is not attached")?;
            ensure!(
                batch.block_ids.len() <= block_ids.len(),
                "prefill block list exceeds scratch capacity"
            );
            ctx.stream.memcpy_htod(
                &batch.block_ids,
                &mut block_ids.slice_mut(..batch.block_ids.len()),
            )?;
        }
        embedding_rows_into(ctx, embed, &self.token_ids, rows, &mut self.hidden)?;
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .hidden
                    .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
            )?;
        }
        embedding_rows_into(ctx, cos_table, &self.positions, rows, &mut self.cos)?;
        embedding_rows_into(ctx, sin_table, &self.positions, rows, &mut self.sin)?;
        Ok(())
    }

    /// Per-layer chunk-scale MLA pack: the w_uk absorb bmm plus the fused
    /// canonical fp8_ds_mla pack that writes this layer's 656-byte KV rows at
    /// `slot_mapping`, into `packed_cache` at `layer_offset` with
    /// `block_stride` between pages (the slab passes its page geometry; the
    /// MTP wire mirror is dense). The bf16 attention query is assembled
    /// later, per attention sub-tile.
    #[allow(clippy::too_many_arguments)]
    fn pack_mla_cache(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        packed_cache: &mut CudaSlice<u8>,
        layer_offset: usize,
        block_stride: usize,
        num_blocks: usize,
        rows: usize,
    ) -> Result<()> {
        gemm_strided_batched_bf16(
            ctx,
            false,
            false,
            GLM52_KV_LORA_RANK,
            rows,
            GLM52_QK_NOPE_HEAD_DIM,
            &weights.w_uk,
            GLM52_KV_LORA_RANK,
            GLM52_QK_NOPE_HEAD_DIM * GLM52_KV_LORA_RANK,
            &self.mla_front.q_full,
            16 * GLM52_QK_HEAD_DIM,
            GLM52_QK_HEAD_DIM,
            &mut self.ql_nope,
            16 * GLM52_KV_LORA_RANK,
            GLM52_KV_LORA_RANK,
            16,
        )?;
        glm52_fp8_per_token_group_quant_bf16_ue8m0_launch(
            ctx,
            Glm52MoeQuantShape {
                rows,
                width: GLM52_KV_LORA_RANK,
                group_size: 128,
            },
            &self.mla_front.kv_c,
            &mut self.ckv_fp8,
            &mut self.ckv_scales,
        )?;
        glm52_mla_cache_pack_launch(
            ctx,
            rows,
            &self.ckv_fp8,
            &self.ckv_scales,
            &self.mla_front.k_pe,
            &self.cos,
            &self.sin,
            packed_cache,
            layer_offset,
            block_stride,
            num_blocks,
            &self.slot_mapping,
        )
    }

    /// Sparse attention over the chunk in `PREFILL_ATTN_TILE_ROWS` sub-tiles:
    /// query assembly (bf16), FlashMLA sparse prefill against the unpacked
    /// KV pool with the carried top-k slots, and the w_uv value bmm into the
    /// chunk-scale `attention_v`.
    fn attend_chunk(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52MlaLayerWeights,
        rows: usize,
    ) -> Result<()> {
        let unpacked_kv = self
            .unpacked_kv
            .as_ref()
            .context("GLM5.2 prefill KV pool is not attached")?;
        let mut sub = 0usize;
        while sub < rows {
            let t = (rows - sub).min(PREFILL_ATTN_TILE_ROWS);
            glm52_mla_query_assemble_launch(
                ctx,
                t,
                16,
                &self.ql_nope.slice(sub * 16 * GLM52_KV_LORA_RANK..),
                &self.mla_front.q_full.slice(sub * 16 * GLM52_QK_HEAD_DIM..),
                GLM52_QK_NOPE_HEAD_DIM,
                GLM52_QK_HEAD_DIM,
                &self.cos.slice(sub * GLM52_ROPE_HALF..),
                &self.sin.slice(sub * GLM52_ROPE_HALF..),
                &mut self.query_bf16,
            )?;
            let carry = self.carry_slots.slice(sub * GLM52_INDEXER_TOPK..);
            let lens = self.carry_lens.slice(sub..);
            glm52_flashmla_sparse_prefill_launch(
                ctx,
                t,
                self.layout.kv_slots,
                GLM52_INDEXER_TOPK,
                0.0625,
                &self.query_bf16,
                unpacked_kv,
                &carry,
                Some(&lens),
                &mut self.attention_out,
                &mut self.attention_max,
                &mut self.attention_lse,
            )?;
            // cuBLAS is column-major: token columns advance by `16 * 256`,
            // while each head batch starts 256 elements later. The resulting
            // address is `[token][head][value]`, matching `o_proj`'s
            // row-major input.
            gemm_strided_batched_bf16(
                ctx,
                true,
                false,
                256,
                t,
                GLM52_KV_LORA_RANK,
                &weights.w_uv,
                GLM52_KV_LORA_RANK,
                256 * GLM52_KV_LORA_RANK,
                &self.attention_out,
                64 * GLM52_KV_LORA_RANK,
                GLM52_KV_LORA_RANK,
                &mut self.attention_v_sub,
                16 * 256,
                256,
                16,
            )?;
            ctx.stream.memcpy_dtod(
                &self.attention_v_sub.slice(..t * 16 * 256),
                &mut self
                    .attention_v
                    .slice_mut(sub * 16 * 256..(sub + t) * 16 * 256),
            )?;
            sub += t;
        }
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .attention_v
                    .slice_mut(rows * 16 * 256..rows4 * 16 * 256),
            )?;
        }
        Ok(())
    }

    fn dense_mlp(
        &mut self,
        ctx: &DeviceContext,
        weights: &Glm52DenseMlpWeights,
        rows4: usize,
    ) -> Result<()> {
        let mut sub = 0usize;
        while sub < rows4 {
            let t = (rows4 - sub).min(PREFILL_DENSE_TILE_ROWS);
            let t4 = t.next_multiple_of(4);
            glm52_dense_mlp_prefill_into(
                ctx,
                weights,
                t4,
                &self.normed.slice(sub * GLM52_HIDDEN..),
                &mut self.dense_gemm,
                &mut self.dense_gate_up,
                &mut self.dense_silu,
                &mut self.dense_out_sub,
            )?;
            ctx.stream.memcpy_dtod(
                &self.dense_out_sub.slice(..t * GLM52_HIDDEN),
                &mut self
                    .mlp_out
                    .slice_mut(sub * GLM52_HIDDEN..(sub + t) * GLM52_HIDDEN),
            )?;
            sub += t;
        }
        Ok(())
    }

    fn reduce_and_norm_attention(
        &mut self,
        ctx: &DeviceContext,
        tp: &mut Glm52MoeTpState,
        post_attn_ln: &DeviceVec,
        rows: usize,
    ) -> Result<()> {
        tp.prefill_allreduce(
            ctx,
            rows,
            &self.attention_partial,
            &mut self.attention_reduced,
        )?;
        fused_add_rms_norm_round_into(
            ctx,
            &mut self.attention_reduced,
            &self.hidden,
            post_attn_ln,
            GLM52_RMS_EPS,
            GLM52_HIDDEN,
            rows,
            &mut self.normed,
        )?;
        let rows4 = rows.next_multiple_of(4);
        if rows4 > rows {
            ctx.stream.memset_zeros(
                &mut self
                    .normed
                    .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
            )?;
        }
        Ok(())
    }

    fn finish_layer(
        &mut self,
        ctx: &DeviceContext,
        next_input_ln: Option<&DeviceVec>,
        rows: usize,
    ) -> Result<()> {
        match next_input_ln {
            Some(weight) => {
                fused_add_rms_norm_round_into(
                    ctx,
                    &mut self.attention_reduced,
                    &self.mlp_out,
                    weight,
                    GLM52_RMS_EPS,
                    GLM52_HIDDEN,
                    rows,
                    &mut self.normed,
                )?;
                ctx.stream.memcpy_dtod(
                    &self.attention_reduced.slice(..rows * GLM52_HIDDEN),
                    &mut self.hidden.slice_mut(..rows * GLM52_HIDDEN),
                )?;
                let rows4 = rows.next_multiple_of(4);
                if rows4 > rows {
                    ctx.stream.memset_zeros(
                        &mut self
                            .normed
                            .slice_mut(rows * GLM52_HIDDEN..rows4 * GLM52_HIDDEN),
                    )?;
                }
            }
            None => {
                add_into(
                    ctx,
                    &self.attention_reduced,
                    &self.mlp_out,
                    rows * GLM52_HIDDEN,
                    &mut self.hidden,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn output_tokens(
        &mut self,
        ctx: &DeviceContext,
        tp: &mut Glm52MoeTpState,
        final_norm: &DeviceVec,
        shard_lm_head: &DeviceMatrix,
        full_lm_head: &DeviceMatrix,
        vocab_start: usize,
        rows: &[i32],
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
        sampling_scratch: &mut BatchSamplingScratch,
    ) -> Result<Vec<u32>> {
        ensure!(
            !rows.is_empty() && rows.len() <= 32 && rows.iter().all(|&row| row >= 0),
            "GLM5.2 prefill output row set is invalid"
        );
        ctx.stream
            .memcpy_htod(rows, &mut self.output_rows.slice_mut(..rows.len()))?;
        glm52_prefill_moe_gather_rows_launch(
            ctx,
            rows.len(),
            GLM52_HIDDEN,
            &self.hidden,
            &self.output_rows,
            self.final_hidden.data_mut(),
        )?;
        if rows.len() < 32 {
            ctx.stream.memset_zeros(
                &mut self
                    .final_hidden
                    .data_mut()
                    .slice_mut(rows.len() * GLM52_HIDDEN..),
            )?;
        }
        glm52_final_norm_into(ctx, &self.final_hidden, final_norm, &mut self.final_normed)?;
        glm52_lm_head_into(ctx, &self.final_normed, shard_lm_head, &mut self.logits)?;
        argmax_bf16_split_into(
            ctx,
            self.logits.data(),
            32,
            shard_lm_head.rows,
            &mut self.argmax_partial_values,
            &mut self.argmax_partial_indices,
            &mut self.argmax_values,
            &mut self.argmax_indices,
        )?;
        glm52_vocab_parallel_pack_launch(
            ctx,
            &self.argmax_values,
            &self.argmax_indices,
            &mut self.attention_partial,
            32,
            tp.rank(),
            vocab_start,
        )?;
        tp.prefill_allreduce(
            ctx,
            32,
            &self.attention_partial,
            &mut self.attention_reduced,
        )?;
        glm52_vocab_parallel_unpack_launch(
            ctx,
            &self.attention_reduced,
            &mut self.argmax_values,
            &mut self.argmax_indices,
            32,
            tp.ranks(),
        )?;
        let mut host = vec![0i32; 32];
        ctx.stream.memcpy_dtoh(&self.argmax_indices, &mut host)?;
        ctx.stream.synchronize()?;
        let mut outputs = host
            .into_iter()
            .take(rows.len())
            .map(|token| {
                ensure!(
                    (0..GLM52_VOCAB as i32).contains(&token),
                    "GLM5.2 prefill argmax token {token} is invalid"
                );
                Ok(token as u32)
            })
            .collect::<Result<Vec<_>>>()?;
        if sampling.is_empty() {
            return Ok(outputs);
        }

        glm52_lm_head_into(ctx, &self.final_normed, full_lm_head, &mut self.logits)?;
        let logits = HiddenStatesRef {
            data: self.logits.data(),
            hidden_dim: GLM52_VOCAB,
            seq_len: 32,
        };
        let as_row = |sample: &crate::runner::Glm52RowSample| BatchSamplingRow {
            row: sample.row,
            temperature: sample.params.temperature,
            top_k: sample.params.top_k,
            top_p: sample.params.top_p,
            min_p: sample.params.min_p,
        };
        let unseeded: Vec<_> = sampling
            .iter()
            .filter(|sample| sample.params.seed.is_none())
            .map(as_row)
            .collect();
        if !unseeded.is_empty() {
            let tokens = gpu_sample_batch_into(ctx, logits, &unseeded, seed, sampling_scratch)?;
            for (row, token) in unseeded.iter().zip(tokens) {
                outputs[row.row] = token;
            }
        }
        for sample in sampling {
            let Some(request_seed) = sample.params.seed else {
                continue;
            };
            let tokens = gpu_sample_batch_into(
                ctx,
                logits,
                &[as_row(sample)],
                mix_seed(request_seed, sample.step),
                sampling_scratch,
            )?;
            outputs[sample.row] = tokens[0];
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_shift_carries_across_chunks_and_uses_boundary_anchor() {
        let batch = Glm52PrefillBatch {
            token_ids: vec![10, 11, 20, 21, 22],
            positions: vec![0, 1, 4, 5, 6],
            request_indptr: vec![0, 2, 5],
            block_indptr: vec![0, 1, 2],
            block_ids: vec![1, 2],
            request_slots: vec![0, 1],
            padding_block: 0,
            slot_mapping: vec![0; 5],
            mtp_next_tokens: vec![Some(12), None],
            output_rows: vec![4],
            sampling: Vec::new(),
            seed: 0,
        };
        assert_eq!(
            mtp_shifted_tokens(&batch, &[99]).unwrap(),
            [11, 12, 21, 22, 99]
        );
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn w_uv_multirow_output_is_token_major() -> Result<()> {
        const ROWS: usize = 3;
        const HEADS: usize = 16;
        const SOURCE_HEADS: usize = 64;
        const K: usize = 512;
        const V: usize = 256;

        let ctx = DeviceContext::new_with_device(0)?;
        let mut weights = vec![bf16::ZERO; HEADS * V * K];
        for head in 0..HEADS {
            for value in 0..V {
                weights[head * V * K + value * K + value] = bf16::ONE;
            }
        }
        let mut latent = vec![bf16::ZERO; ROWS * SOURCE_HEADS * K];
        for token in 0..ROWS {
            for head in 0..HEADS {
                for value in 0..V {
                    latent[token * SOURCE_HEADS * K + head * K + value] =
                        bf16::from_f32((token * 64 + head * 2 + value % 2) as f32);
                }
            }
        }
        let weights = ctx.stream.clone_htod(&weights)?;
        let latent = ctx.stream.clone_htod(&latent)?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(ROWS * HEADS * V)?;
        gemm_strided_batched_bf16(
            &ctx,
            true,
            false,
            V,
            ROWS,
            K,
            &weights,
            K,
            V * K,
            &latent,
            SOURCE_HEADS * K,
            K,
            &mut output,
            HEADS * V,
            V,
            HEADS,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        for token in 0..ROWS {
            for head in 0..HEADS {
                for value in 0..V {
                    let offset = token * HEADS * V + head * V + value;
                    let expected = (token * 64 + head * 2 + value % 2) as f32;
                    assert_eq!(output[offset].to_f32(), expected);
                }
            }
        }
        Ok(())
    }
}
