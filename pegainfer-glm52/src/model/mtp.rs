//! Checkpoint-native GLM5.2 MTP serving lane.
//!
//! The target step keeps its final-normalized hidden rows resident. A draft round
//! packs only committed rows, shifts each sequence token one place left, and
//! runs checkpoint layer 78 once to synchronize MTP KV and produce draft 1.
//! Four single-token iterations then recycle the layer's shared-head-normalized
//! hidden. Rejected speculative KV is not copied back: the next committed
//! first pass overwrites it at the same positions.

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_kernels::ops::GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::Glm52TpTopology;
use pegainfer_kernels::ops::argmax_bf16_split_into;
use pegainfer_kernels::ops::copy_hidden_rows_raw_into;
use pegainfer_kernels::ops::embedding_rows_into;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use pegainfer_kernels::ops::rms_norm_rows_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;

use super::GLM52_DECODE_BUCKETS;
use super::GLM52_KV_PAGE_IDXK_BYTES;
use super::GLM52_KV_PAGE_STRIDE;
use super::GLM52_MAX_BATCH_PER_RANK;
use super::GLM52_MAX_STEP_ROWS;
use super::INDEX_CACHE_BLOCK;
use super::NUM_SMS;
use super::build;
#[cfg(test)]
use super::glm52_pool_blocks;
use super::glm52_table_width;
use super::step_body::glm52_moe_ep_layer;
use crate::bookend::glm52_embed_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_MTP_LAYER;
use crate::config::GLM52_RMS_EPS;
use crate::config::GLM52_SM_SCALE;
use crate::config::GLM52_VOCAB;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52KvSlab;
use crate::layer::Glm52LayerCaches;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_layer_attention_half;
use crate::layer::glm52_layer_finish;
#[cfg(test)]
use crate::mla_decode::Glm52MlaBackend;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_select_mla_backend;
use crate::moe_ep::Glm52MoeEpState;
use crate::moe_tp::Glm52MoeTpPrefillScratch;
use crate::moe_tp::Glm52MoeTpRank;
use crate::mtp::GLM52_MTP_DRAFTS;
use crate::mtp::Glm52MtpBookendWeights;
use crate::mtp::Glm52MtpScratch;
use crate::mtp::glm52_mtp_prepare_into;
use crate::mtp::glm52_mtp_recycle_into;
use crate::prefill_tp::Glm52TpPrefillMtpView;
use crate::rows::Rows;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52MtpRound;
use crate::scratch::Glm52DecodeScratch;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::retype_owned;

struct Glm52MtpBucket {
    rows: usize,
    sched: Glm52MlaSchedMetadata,
    scratch: Glm52DecodeScratch,
    bookend_scratch: Glm52MtpScratch,
    embeds: Rows<GLM52_HIDDEN>,
    previous: Rows<GLM52_HIDDEN>,
    decoder_input: Rows<GLM52_HIDDEN>,
    block_table: CudaSlice<i32>,
    compute_graph: CudaGraphState,
}

pub(super) struct Glm52MtpProposalSeed<'a> {
    pub(super) previous: &'a Rows<GLM52_HIDDEN>,
    pub(super) draft1: &'a [u32],
    /// Row in `previous` where this round's boundaries start. The TP4
    /// prefill caller splits one prefill batch's boundaries into
    /// `GLM52_TP_TOKENS`-row rounds, so a later round reads `previous`
    /// at its own offset while `draft1` arrives pre-sliced.
    pub(super) rows_before: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_scratch_backs_up_partial_committed_page() {
        let (table, copy) = proposal_page_table(&[11, 37], 127, [100, 101]).unwrap();
        assert_eq!(table, [11, 37, 101]);
        assert_eq!(copy, Some(37));
    }

    #[test]
    fn proposal_scratch_appends_after_aligned_committed_page() {
        let (table, copy) = proposal_page_table(&[11, 37], 128, [100, 101]).unwrap();
        assert_eq!(table, [11, 37, 100, 101]);
        assert_eq!(copy, None);
    }

    #[test]
    fn tp4_mtp_keeps_flashinfer_execution_separate_from_slab_wire_layout() {
        // The TP4 proposal executes over a dense 576 B/token FlashInfer
        // cache while the P/D wire rows commit into the slab's 656 B/token
        // fp8_ds_mla mirror slices — two layouts, never interchangeable.
        assert_eq!(Glm52MlaBackend::FlashInferFp8.cache_bytes_per_token(), 576);
        assert_eq!(GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN, 656);
    }

    #[test]
    fn tp4_mtp_arena_ledger_charges_the_dense_caches() {
        let cap = 16_384;
        let slots = crate::model::glm52_decode_slots();
        let pool = glm52_pool_blocks(cap, slots);
        let scratch_pages = MTP_SCRATCH_PAGES_PER_SLOT * slots;
        // TP4 charges the dense FlashInfer execution cache (MLA + index-K
        // co-allocation) for the whole committed+scratch range; EP charges
        // only its scratch slab pages. Neither charges the layer-78 wire
        // mirrors — they live inside the page content already charged by
        // `glm52_arena_bytes`.
        let dense = (pool + scratch_pages)
            * (GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN
                + GLM52_KV_PAGE_IDXK_BYTES);
        let ep_scratch = scratch_pages * GLM52_KV_PAGE_STRIDE;
        let tp4 = crate::mtp::glm52_mtp_arena_bytes(cap, pool, crate::Glm52MoeTopo::Tp4)
            .expect("TP4 arena bytes");
        let ep4 = crate::mtp::glm52_mtp_arena_bytes(cap, pool, crate::Glm52MoeTopo::Ep4)
            .expect("EP4 arena bytes");
        assert_eq!(tp4 - ep4, dense - ep_scratch);
    }
}

/// Private proposal scratch pages per decode slot: one page for the
/// partial-committed-page backup, one for the draft span's overflow writes.
pub(crate) const MTP_SCRATCH_PAGES_PER_SLOT: usize = 2;

fn proposal_page_table(
    committed_pages: &[i32],
    committed_len: usize,
    scratch: [i32; MTP_SCRATCH_PAGES_PER_SLOT],
) -> Result<(Vec<i32>, Option<i32>)> {
    let mut table = committed_pages.to_vec();
    let copy_source = if committed_len.is_multiple_of(GLM52_FLASHMLA_SPARSE_PAGE_SIZE) {
        table.push(scratch[0]);
        None
    } else {
        let source = *table
            .last()
            .context("GLM5.2 MTP partial committed context has no page")?;
        Some(source)
    };
    table.push(scratch[1]);
    Ok((table, copy_source))
}

/// Where the layer-78 KV rows live.
pub(super) enum Glm52MtpKv {
    /// EP decode: the committed mirrors ride the rank slab's pages at these
    /// slice offsets (pool block ids 1:1 with the target layers), and the
    /// per-slot proposal scratch pages sit past `pool_blocks` in the same
    /// slab. The slab itself is the rank model's; every cache-touching call
    /// receives it.
    Slab(Glm52LayerCaches),
    /// TP4 prefill: the FlashInfer execution cache and the fp8_ds_mla wire
    /// cache are packed by launches that address token slots dense from the
    /// buffer base (no layer offset / page stride), so neither can ride the
    /// page-first slab.
    Dense(Box<Glm52MtpDenseKv>),
}

/// TP4-only dense layer-78 caches.
pub(super) struct Glm52MtpDenseKv {
    /// FlashInfer proposal cache: `[num_blocks x 64 x 576 MLA | num_blocks x
    /// 8,448 index-K]` co-allocated so the shared attention half sees one
    /// buffer. The MLA region is addressed dense from the base
    /// (`mla_offset` 0, `page_stride` = one 64 x 576 page); the index-K
    /// region starts at `index_k_offset` with the tight 8,448-byte stride.
    proposal: Glm52KvSlab,
    proposal_caches: Glm52LayerCaches,
    /// The layer-78 mirror slices inside a KV slab page
    /// (`glm52_page_layout().mtp`). The prefill executor commits the P/D
    /// wire rows (fp8_ds_mla + index-K) straight into the rank slab at these
    /// offsets — the slab page is the ONLY registered arena, so KV that
    /// never reaches it never reaches the decode side (openinfer#850 review).
    slab_caches: Glm52LayerCaches,
}

pub(super) struct Glm52NativeMtp {
    bookend: Glm52MtpBookendWeights,
    layer: Glm52DecoderLayerWeights,
    kv: Glm52MtpKv,
    buckets: [Glm52MtpBucket; GLM52_DECODE_BUCKETS.len()],
    max_model_len: usize,
    table_width: usize,
    committed_blocks: usize,
    ep_ranks: usize,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    committed_lens: [usize; GLM52_MAX_BATCH_PER_RANK],
    /// TP4 only: the proposal layer-78 MoE runs through the prefill TP path
    /// (rank slice + NCCL all-reduce); the LL decode path is gone (#805).
    tp_moe: Option<Glm52MoeTpPrefillScratch>,
}

/// [`Glm52NativeMtp`] minus everything sized by the pool block count: the
/// two-phase build measures free VRAM after this exists, then
/// [`Self::attach_cache`] binds the layer-78 KV with the decided count.
pub(super) struct Glm52NativeMtpFixed {
    bookend: Glm52MtpBookendWeights,
    layer: Glm52DecoderLayerWeights,
    /// TP4 keeps dense layer-78 caches (FlashInfer execution + fp8_ds_mla
    /// wire); every EP topology rides the rank's page-first slab instead.
    dense_kv: bool,
    buckets: [Glm52MtpBucket; GLM52_DECODE_BUCKETS.len()],
    max_model_len: usize,
    table_width: usize,
    ep_ranks: usize,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    tp_moe: Option<Glm52MoeTpPrefillScratch>,
}

impl Glm52NativeMtpFixed {
    /// Whether the layer-78 KV rides the rank's page-first slab (every EP
    /// topology). `finish_kv` sizes the slab's scratch tail from this before
    /// [`Self::attach_cache`] binds the offsets.
    pub(super) fn slab_resident(&self) -> bool {
        !self.dense_kv
    }

    /// Bind the layer-78 KV for the launch-decided pool block count. The
    /// committed region mirrors the main pool's page ids 1:1 (radix hits
    /// reuse L78 KV by page id), so it must be the SAME block count as the
    /// pool — the per-slot scratch pair pages sit directly after it. Any
    /// other sizing desyncs the `glm52_mtp_arena_bytes` ledger.
    ///
    /// `offsets` are the layer-78 mirror slices inside a slab page
    /// (`glm52_page_layout().mtp`) — the slab-resident commit target for
    /// every topology. The TP4 dense caches additionally allocate their own
    /// FlashInfer execution buffer here.
    pub(super) fn attach_cache(
        mut self,
        ctx: &DeviceContext,
        pool_blocks: usize,
        offsets: Glm52LayerCaches,
    ) -> Result<Glm52NativeMtp> {
        let committed_blocks = pool_blocks;
        let num_blocks =
            committed_blocks + crate::model::glm52_decode_slots() * MTP_SCRATCH_PAGES_PER_SLOT;
        // The bucket sched/scratch were built against a placeholder count;
        // rebind them to the real cache geometry before anything launches.
        for bucket in &mut self.buckets {
            bucket.sched.set_num_blocks(num_blocks);
            bucket.scratch.idx.set_num_kv_blocks(num_blocks);
        }
        ensure!(
            offsets.index_k_offset.is_some(),
            "GLM5.2 MTP layer 78 page slices are missing the index-K mirror"
        );
        let kv = if self.dense_kv {
            let mla_bytes = num_blocks
                * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                * GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN;
            let idxk_bytes = num_blocks * GLM52_KV_PAGE_IDXK_BYTES;
            Glm52MtpKv::Dense(Box::new(Glm52MtpDenseKv {
                proposal: Glm52KvSlab {
                    slab: ctx.stream.alloc_zeros::<u8>(mla_bytes + idxk_bytes)?,
                    page_stride: GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                        * GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN,
                    num_blocks,
                },
                proposal_caches: Glm52LayerCaches {
                    mla_offset: 0,
                    index_k_offset: Some(mla_bytes),
                },
                slab_caches: offsets,
            }))
        } else {
            Glm52MtpKv::Slab(offsets)
        };
        Ok(Glm52NativeMtp {
            bookend: self.bookend,
            layer: self.layer,
            kv,
            buckets: self.buckets,
            max_model_len: self.max_model_len,
            table_width: self.table_width,
            committed_blocks,
            ep_ranks: self.ep_ranks,
            positions: self.positions,
            cos: self.cos,
            sin: self.sin,
            token_ids: self.token_ids,
            slot_mapping: self.slot_mapping,
            seq_lens: self.seq_lens,
            committed_lens: [0; GLM52_MAX_BATCH_PER_RANK],
            tp_moe: self.tp_moe,
        })
    }
}

impl Glm52NativeMtp {
    /// The TP4 prefill executor's view of the dense layer-78 caches.
    pub(super) fn prefill_view(&mut self) -> Glm52TpPrefillMtpView<'_> {
        let Glm52MtpKv::Dense(dense) = &mut self.kv else {
            panic!("MTP prefill view is TP4-only");
        };
        Glm52TpPrefillMtpView {
            bookend: &self.bookend,
            layer: &self.layer,
            slab_caches: dense.slab_caches,
            proposal: &mut dense.proposal,
            proposal_caches: dense.proposal_caches,
        }
    }

    /// Everything not sized by the pool block count: weights, per-bucket
    /// sched/scratch (len-scaled), and the fixed step-row buffers. The cache
    /// slabs follow in [`Glm52NativeMtpFixed::attach_cache`] once the
    /// measured launch fill decides the count.
    pub(super) fn build_fixed(
        ctx: &DeviceContext,
        weights: &mut Glm52RankGpuWeights,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        attn_shard: Option<usize>,
    ) -> Result<Glm52NativeMtpFixed> {
        let prefix = format!("model.layers.{GLM52_MTP_LAYER}");
        let enorm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.enorm.weight"),
            GLM52_HIDDEN,
        )?;
        let hnorm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.hnorm.weight"),
            GLM52_HIDDEN,
        )?;
        let eh_proj_raw = weights.take_tensor(&format!("{prefix}.eh_proj.weight"))?;
        ensure!(
            eh_proj_raw.len() == 2 * GLM52_HIDDEN * GLM52_HIDDEN * size_of::<bf16>(),
            "GLM5.2 MTP eh_proj byte length drifted"
        );
        let eh_proj = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, eh_proj_raw)?,
            rows: GLM52_HIDDEN,
            cols: 2 * GLM52_HIDDEN,
        };
        let shared_norm = build::take_bf16_vec(
            ctx,
            weights,
            &format!("{prefix}.shared_head.norm.weight"),
            GLM52_HIDDEN,
        )?;
        let bookend = Glm52MtpBookendWeights::new(enorm, hnorm, eh_proj, shared_norm)?;
        let layer =
            build::build_decoder_layer(ctx, weights, GLM52_MTP_LAYER, moe_topo, attn_shard)?;

        let table_width = glm52_table_width(max_model_len);
        let attention_heads = layer.mla.heads;
        let backend = glm52_select_mla_backend(attention_heads)?;
        let cache_bytes_per_token = backend.cache_bytes_per_token();
        // TP4 executes proposals over a dense FlashInfer cache; every
        // topology commits its layer-78 wire rows into the rank slab's
        // fp8_ds_mla mirror slices.
        let dense_kv = moe_topo == crate::Glm52MoeTopo::Tp4;
        // A slab-resident mirror must hold the slab's fp8_ds_mla rows.
        ensure!(
            dense_kv || cache_bytes_per_token == GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
            "GLM5.2 slab-resident MTP requires the {GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN}-byte \
             fp8_ds_mla layout, got {cache_bytes_per_token} bytes/token (backend {backend:?})"
        );
        // The pool block count is measured after this build; the bucket
        // sched/scratch below carry the count only as launch metadata, so
        // they are built against a 1-block placeholder that `attach_cache`
        // rebinds to the real geometry. Strides are final here: slab-resident
        // mirrors sit one page stride apart, the TP4 dense caches are tight.
        let index_layout = Glm52IndexerCacheLayout {
            cache_blocks: 1,
            cache_block_size: INDEX_CACHE_BLOCK,
            cache_layer_offset_bytes: 0,
            cache_block_stride_bytes: if dense_kv {
                GLM52_KV_PAGE_IDXK_BYTES
            } else {
                GLM52_KV_PAGE_STRIDE
            },
        };
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: GLM52_MAX_BATCH_PER_RANK,
            num_blocks: 1,
            kv_layer_offset_bytes: 0,
            // The TP4 dense stride is never consumed (FlashInfer addresses
            // its 576-byte cache dense) but must satisfy the fp8_ds contract.
            kv_block_stride_bytes: if dense_kv {
                super::GLM52_KV_PAGE_MLA_BYTES
            } else {
                GLM52_KV_PAGE_STRIDE
            },
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
            sm_scale: GLM52_SM_SCALE,
        };
        log::info!(
            "GLM5.2 native MTP cache: topology={moe_topo:?} \
             execution_backend={backend:?} execution_bytes/token={cache_bytes_per_token} \
             wire=slab fp8_ds_mla {GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN} bytes/token",
        );

        let tp_moe = match moe_topo {
            crate::Glm52MoeTopo::Tp4 => Some(Glm52MoeTpPrefillScratch::new(
                ctx,
                Glm52TpTopology::Tp4,
                GLM52_MAX_BATCH_PER_RANK,
            )?),
            _ => None,
        };

        let mut buckets = Vec::with_capacity(GLM52_DECODE_BUCKETS.len());
        for rows in GLM52_DECODE_BUCKETS {
            let row_contract = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract
            };
            let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
                rows,
                index_layout,
                table_width,
                NUM_SMS,
                max_model_len,
            );
            buckets.push(Glm52MtpBucket {
                rows,
                sched: Glm52MlaSchedMetadata::new_for_backend(
                    ctx,
                    row_contract,
                    attention_heads,
                    backend,
                )?,
                scratch: Glm52DecodeScratch::new_for_backend(
                    ctx,
                    &row_contract,
                    mqa_shape,
                    attention_heads,
                    backend,
                    false,
                )?,
                bookend_scratch: Glm52MtpScratch::new(ctx, rows)?,
                embeds: Rows::zeros(ctx, rows)?,
                previous: Rows::zeros(ctx, rows)?,
                decoder_input: Rows::zeros(ctx, rows)?,
                block_table: ctx.stream.alloc_zeros::<i32>(rows * table_width)?,
                compute_graph: CudaGraphState::new(),
            });
        }
        Ok(Glm52NativeMtpFixed {
            bookend,
            layer,
            dense_kv,
            buckets: buckets
                .try_into()
                .map_err(|_| anyhow::anyhow!("GLM5.2 MTP bucket count drifted"))?,
            max_model_len,
            table_width,
            ep_ranks: moe_topo.expected_ep_size(),
            positions: ctx.stream.alloc_zeros(GLM52_MAX_STEP_ROWS)?,
            cos: ctx
                .stream
                .alloc_zeros(GLM52_MAX_STEP_ROWS * crate::config::GLM52_ROPE_HALF)?,
            sin: ctx
                .stream
                .alloc_zeros(GLM52_MAX_STEP_ROWS * crate::config::GLM52_ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros(GLM52_MAX_STEP_ROWS)?,
            slot_mapping: ctx.stream.alloc_zeros(GLM52_MAX_STEP_ROWS)?,
            seq_lens: ctx.stream.alloc_zeros(GLM52_MAX_STEP_ROWS)?,
            tp_moe,
        })
    }

    pub(super) fn reset_slots(&mut self, resets: &[usize]) -> Result<()> {
        for &slot in resets {
            ensure!(
                slot < GLM52_MAX_BATCH_PER_RANK,
                "GLM5.2 MTP reset slot {slot} is outside \
                 0..{GLM52_MAX_BATCH_PER_RANK}"
            );
            self.committed_lens[slot] = 0;
        }
        Ok(())
    }

    pub(super) fn resume_reset_slots(
        &mut self,
        resets: &[usize],
        appends: &[Glm52MtpAppend],
    ) -> Result<()> {
        for &slot in resets {
            if let Some(first) = appends.iter().find(|append| append.slot == slot) {
                ensure!(
                    first.position <= self.max_model_len,
                    "GLM5.2 MTP restored position {} exceeds cap {}",
                    first.position,
                    self.max_model_len
                );
                self.committed_lens[slot] = first.position;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        mut ep: Option<&mut Glm52MoeEpState>,
        mut tp: Option<&mut Glm52MoeTpRank>,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
        slab: &mut Glm52KvSlab,
        target_final_normed: &Rows<GLM52_HIDDEN>,
        round: &Glm52MtpRound,
        seed: Option<Glm52MtpProposalSeed<'_>>,
    ) -> Result<Vec<[u32; GLM52_MTP_DRAFTS]>> {
        let context_bucket = round.context_bucket;
        let appends = round.appends.as_slice();
        let proposal_slots = round.proposal_slots.as_slice();
        ensure!(
            appends.len() <= context_bucket,
            "GLM5.2 MTP context rows {} exceed bucket {context_bucket}",
            appends.len(),
        );
        ensure!(
            proposal_slots.windows(2).all(|pair| pair[0] < pair[1]),
            "GLM5.2 MTP proposal slots must be strictly ascending"
        );
        let context_index = self.bucket_index(context_bucket)?;
        for (packed, append) in appends.iter().enumerate() {
            ensure!(
                append.slot < GLM52_MAX_BATCH_PER_RANK
                    && append.target_row < target_final_normed.tokens(),
                "GLM5.2 MTP append target row {} or slot {} is out of bounds \
                 (target rows {}, slots {})",
                append.target_row,
                append.slot,
                target_final_normed.tokens(),
                GLM52_MAX_BATCH_PER_RANK,
            );
            ensure!(
                append.position == self.committed_lens[append.slot],
                "GLM5.2 MTP slot {} first-pass position {} != committed {}",
                append.slot,
                append.position,
                self.committed_lens[append.slot]
            );
            if seed.is_none() {
                let src = target_final_normed.data().slice(
                    append.target_row * GLM52_HIDDEN..(append.target_row + 1) * GLM52_HIDDEN,
                );
                let mut dst = self.buckets[context_index]
                    .previous
                    .data_mut()
                    .slice_mut(packed * GLM52_HIDDEN..(packed + 1) * GLM52_HIDDEN);
                ctx.stream.memcpy_dtod(&src, &mut dst)?;
            }
            self.committed_lens[append.slot] += 1;
        }
        if seed.is_none() {
            // The context forward runs UNCONDITIONALLY, appends or not — it
            // is an EP collective, and the fixed-chain discipline forbids
            // host state deciding whether a collective happens. Rows without
            // an append are padding: token 0, position 0, page 0, and a
            // zeroed `previous` hidden — constructively deterministic bytes
            // on the wire, never capture-buffer residue.
            self.zero_padding_previous(ctx, context_index, appends.len())?;
            let context_inputs: Vec<(usize, u32, usize, Option<&[i32]>)> = appends
                .iter()
                .map(|append| {
                    (
                        append.slot,
                        append.input_token,
                        append.position,
                        Some(append.pages.as_slice()),
                    )
                })
                .collect();
            self.forward(
                ctx,
                aux,
                ep.as_deref_mut(),
                tp.as_deref_mut(),
                embed,
                lm_head,
                cos_table,
                sin_table,
                &mut *slab,
                context_index,
                &context_inputs,
            )
            .context("GLM5.2 MTP context forward")?;
        }
        let draft_bucket = round.draft_bucket;
        ensure!(
            proposal_slots.len() <= draft_bucket,
            "GLM5.2 MTP proposal rows {} exceed bucket {draft_bucket}",
            proposal_slots.len(),
        );
        let mut last_rows = Vec::with_capacity(proposal_slots.len());
        for &slot in proposal_slots {
            let row = appends
                .iter()
                .rposition(|append| append.slot == slot)
                .with_context(|| format!("GLM5.2 MTP proposal slot {slot} has no append"))?;
            last_rows.push(row);
        }
        let context_tokens = match seed.as_ref() {
            Some(seed) => {
                ensure!(
                    seed.draft1.len() == appends.len()
                        && seed.previous.tokens() >= seed.rows_before + appends.len(),
                    "GLM5.2 TP prefill MTP proposal seed shape mismatch"
                );
                seed.draft1.to_vec()
            }
            None => self
                .argmax_host(ctx, context_index)
                .context("GLM5.2 MTP context argmax")?,
        };
        let draft_index = self.bucket_index(draft_bucket)?;
        self.zero_padding_previous(ctx, draft_index, proposal_slots.len())?;
        let mut proposal_pages = Vec::with_capacity(proposal_slots.len());
        let mut partial_backups = Vec::with_capacity(proposal_slots.len());
        for (packed, (&slot, &context_row)) in proposal_slots.iter().zip(&last_rows).enumerate() {
            let src_hidden = match seed.as_ref() {
                Some(seed) => {
                    let row = seed.rows_before + packed;
                    seed.previous
                        .data()
                        .slice(row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN)
                }
                None => self.buckets[context_index]
                    .scratch
                    .final_normed
                    .data()
                    .slice(context_row * GLM52_HIDDEN..(context_row + 1) * GLM52_HIDDEN),
            };
            let mut dst_hidden = self.buckets[draft_index]
                .previous
                .data_mut()
                .slice_mut(packed * GLM52_HIDDEN..(packed + 1) * GLM52_HIDDEN);
            ctx.stream.memcpy_dtod(&src_hidden, &mut dst_hidden)?;
            ensure!(
                self.committed_lens[slot] < self.max_model_len,
                "GLM5.2 MTP slot {slot} exhausted its context cap"
            );
            let committed_len = self.committed_lens[slot];
            let scratch = [
                self.scratch_page(slot, 0) as i32,
                self.scratch_page(slot, 1) as i32,
            ];
            let (table, copy_source) =
                proposal_page_table(&appends[context_row].pages, committed_len, scratch)?;
            if let Some(source) = copy_source {
                self.copy_cache_page(ctx, &mut *slab, source as usize, scratch[0] as usize)?;
            }
            proposal_pages.push(table);
            partial_backups.push(copy_source.map(|source| (scratch[0], source)));
        }

        let mut spans = vec![[0u32; GLM52_MTP_DRAFTS]; proposal_slots.len()];
        for (span, &row) in spans.iter_mut().zip(&last_rows) {
            span[0] = context_tokens[row];
        }
        for iteration in 1..crate::mtp::glm52_mtp_draft_len() {
            let inputs: Vec<(usize, u32, usize, Option<&[i32]>)> = proposal_slots
                .iter()
                .enumerate()
                .map(|(row, &slot)| {
                    (
                        slot,
                        spans[row][iteration - 1],
                        self.committed_lens[slot] + iteration - 1,
                        Some(proposal_pages[row].as_slice()),
                    )
                })
                .collect();
            self.forward(
                ctx,
                aux,
                ep.as_deref_mut(),
                tp.as_deref_mut(),
                embed,
                lm_head,
                cos_table,
                sin_table,
                &mut *slab,
                draft_index,
                &inputs,
            )
            .with_context(|| format!("GLM5.2 MTP proposal iteration {iteration} forward"))?;
            let tokens = self
                .argmax_host(ctx, draft_index)
                .with_context(|| format!("GLM5.2 MTP proposal iteration {iteration} argmax"))?;
            for (row, span) in spans.iter_mut().enumerate() {
                span[iteration] = tokens[row];
                let src = self.buckets[draft_index]
                    .scratch
                    .final_normed
                    .data()
                    .slice(row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN);
                let mut dst = self.buckets[draft_index]
                    .previous
                    .data_mut()
                    .slice_mut(row * GLM52_HIDDEN..(row + 1) * GLM52_HIDDEN);
                ctx.stream.memcpy_dtod(&src, &mut dst)?;
            }
        }
        for backup in partial_backups.into_iter().flatten() {
            self.restore_cache_page(ctx, &mut *slab, backup.0 as usize, backup.1 as usize)?;
        }
        Ok(spans)
    }

    fn bucket_index(&self, rows: usize) -> Result<usize> {
        self.buckets
            .iter()
            .position(|bucket| bucket.rows == rows)
            .with_context(|| format!("GLM5.2 MTP bucket {rows} is not in {GLM52_DECODE_BUCKETS:?}"))
    }

    /// Zero the padding rows of a bucket's `previous` hidden buffer before a
    /// forward. `previous` persists across rounds, so without this a padding
    /// row would feed whatever hidden state the buffer held last — bytes that
    /// go over the DeepEP wire. The padding-as-protocol discipline requires
    /// every dummy row's input to be constructively deterministic.
    fn zero_padding_previous(
        &mut self,
        ctx: &DeviceContext,
        bucket_index: usize,
        real_rows: usize,
    ) -> Result<()> {
        let bucket = &mut self.buckets[bucket_index];
        if real_rows < bucket.rows {
            ctx.stream.memset_zeros(
                &mut bucket
                    .previous
                    .data_mut()
                    .slice_mut(real_rows * GLM52_HIDDEN..bucket.rows * GLM52_HIDDEN),
            )?;
        }
        Ok(())
    }

    fn scratch_page(&self, slot: usize, offset: usize) -> usize {
        self.committed_blocks + slot * MTP_SCRATCH_PAGES_PER_SLOT + offset
    }

    /// Back up a partial committed page into a proposal scratch page before
    /// the draft iterations write into it. Slab-resident: one whole-page
    /// content copy (the target layers' slices ride along; only layer 78
    /// writes during a proposal round, so the extra bytes are inert). TP4
    /// dense: the FlashInfer MLA block and the index-K block move within the
    /// proposal buffer.
    fn copy_cache_page(
        &mut self,
        ctx: &DeviceContext,
        slab: &mut Glm52KvSlab,
        source: usize,
        target: usize,
    ) -> Result<()> {
        match &mut self.kv {
            Glm52MtpKv::Slab(_) => {
                super::glm52_copy_page_content(&ctx.stream, slab, source, target)
            }
            Glm52MtpKv::Dense(dense) => {
                let idxk_offset = dense
                    .proposal_caches
                    .index_k_offset
                    .context("GLM5.2 MTP layer 78 has no index-K cache")?;
                super::copy_strided_block(
                    &ctx.stream,
                    &mut dense.proposal.slab,
                    0,
                    dense.proposal.page_stride,
                    dense.proposal.page_stride,
                    source,
                    target,
                )?;
                super::copy_strided_block(
                    &ctx.stream,
                    &mut dense.proposal.slab,
                    idxk_offset,
                    GLM52_KV_PAGE_IDXK_BYTES,
                    GLM52_KV_PAGE_IDXK_BYTES,
                    source,
                    target,
                )
            }
        }
    }

    /// Undo [`Self::copy_cache_page`] after the proposal round: the backed-up
    /// bytes return to the committed page (same copy shape, reversed pair).
    fn restore_cache_page(
        &mut self,
        ctx: &DeviceContext,
        slab: &mut Glm52KvSlab,
        source: usize,
        target: usize,
    ) -> Result<()> {
        self.copy_cache_page(ctx, slab, source, target)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        mut ep: Option<&mut Glm52MoeEpState>,
        mut tp: Option<&mut Glm52MoeTpRank>,
        embed: &DeviceMatrix,
        lm_head: &DeviceMatrix,
        cos_table: &DeviceMatrix,
        sin_table: &DeviceMatrix,
        slab: &mut Glm52KvSlab,
        bucket_index: usize,
        inputs: &[(usize, u32, usize, Option<&[i32]>)],
    ) -> Result<()> {
        let rows = self.buckets[bucket_index].rows;
        let mut tokens = [0u32; GLM52_MAX_STEP_ROWS];
        let mut positions = [0u32; GLM52_MAX_STEP_ROWS];
        let mut seq_lens = [1i32; GLM52_MAX_STEP_ROWS];
        let mut slot_mapping = [0i64; GLM52_MAX_STEP_ROWS];
        let mut pages = vec![0i32; rows * self.table_width];
        for (row, &(slot, token, position, committed_pages)) in inputs.iter().enumerate() {
            ensure!(
                row < rows && slot < GLM52_MAX_BATCH_PER_RANK && position < self.max_model_len,
                "GLM5.2 MTP input row {row}/{rows}, slot \
                 {slot}/{GLM52_MAX_BATCH_PER_RANK}, or position \
                 {position}/{} is out of bounds",
                self.max_model_len,
            );
            tokens[row] = token;
            positions[row] = position as u32;
            seq_lens[row] = (position + 1) as i32;
            let page_offset = position / GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
            let page = match committed_pages {
                Some(committed_pages) => {
                    ensure!(
                        committed_pages.len() > page_offset && page_offset < self.table_width,
                        "GLM5.2 MTP committed page table for slot {slot} has {} pages, \
                         but position {position} needs logical page {page_offset} \
                         within table width {}",
                        committed_pages.len(),
                        self.table_width,
                    );
                    // kvbm may eagerly own the dangling generation page at
                    // the context cap. It is not addressable by this
                    // position's max-model-len-wide attention table.
                    let page_count = committed_pages.len().min(self.table_width);
                    pages[row * self.table_width..row * self.table_width + page_count]
                        .copy_from_slice(&committed_pages[..page_count]);
                    committed_pages[page_offset] as usize
                }
                None => anyhow::bail!(
                    "GLM5.2 MTP input at slot {slot} position {position} has no page table"
                ),
            };
            slot_mapping[row] = (page * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
                + position % GLM52_FLASHMLA_SPARSE_PAGE_SIZE)
                as i64;
        }
        ctx.stream.memcpy_htod(&tokens, &mut self.token_ids)?;
        ctx.stream.memcpy_htod(&positions, &mut self.positions)?;
        ctx.stream.memcpy_htod(&seq_lens, &mut self.seq_lens)?;
        ctx.stream
            .memcpy_htod(&slot_mapping, &mut self.slot_mapping)?;
        embedding_rows_into(ctx, cos_table, &self.positions, rows, &mut self.cos)?;
        embedding_rows_into(ctx, sin_table, &self.positions, rows, &mut self.sin)?;
        ctx.stream
            .memcpy_htod(&pages, &mut self.buckets[bucket_index].block_table)?;
        // TP4 runs the body eagerly: its NCCL collectives (attention AR +
        // MoE reduce) stay out of CUDA graph capture, and proposal rounds on
        // the prefill engine are not a hot decode loop (#805).
        let tp_prefill = matches!(&self.layer.mlp, Glm52LayerMlp::MoeTp(_));
        let tp_moe = self.tp_moe.as_mut();
        // The layer-78 KV this forward writes/attends: the rank slab at the
        // mirror offsets (EP), or the private dense proposal cache (TP4).
        let (kv_slab, kv_caches) = match &mut self.kv {
            Glm52MtpKv::Slab(caches) => (slab, *caches),
            Glm52MtpKv::Dense(dense) => (&mut dense.proposal, dense.proposal_caches),
        };
        let bucket = &mut self.buckets[bucket_index];
        let Glm52MtpBucket {
            sched,
            scratch,
            bookend_scratch,
            embeds,
            previous,
            decoder_input,
            block_table,
            compute_graph,
            ..
        } = bucket;
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: sched,
            slot_mapping: &self.slot_mapping,
            block_table,
            seq_lens: &self.seq_lens,
        };
        let body = || {
            glm52_embed_into(ctx, embed, &self.token_ids, embeds)?;
            glm52_mtp_prepare_into(
                ctx,
                &self.bookend,
                &self.positions,
                embeds,
                previous,
                bookend_scratch,
                decoder_input,
            )?;
            ctx.stream
                .memcpy_dtod(decoder_input.data(), scratch.hidden.data_mut())?;
            rms_norm_rows_into(
                ctx,
                scratch.hidden.data(),
                &self.layer.input_ln,
                GLM52_RMS_EPS,
                GLM52_HIDDEN,
                rows,
                scratch.layer.normed.data_mut(),
            )?;
            let mut carry_ready = false;
            let tp_ar = match &self.layer.mlp {
                Glm52LayerMlp::MoeTp(_) => {
                    let rank = tp
                        .as_deref_mut()
                        .context("GLM5.2 TP MTP attention has no TP runtime")?;
                    Some(&mut rank.state)
                }
                _ => None,
            };
            glm52_layer_attention_half(
                ctx,
                Some(aux),
                &self.layer,
                kv_slab,
                kv_caches,
                &step,
                scratch,
                &mut carry_ready,
                0,
                true,
                tp_ar,
            )?;
            let mut tp_padded_mlp = false;
            match &self.layer.mlp {
                Glm52LayerMlp::MoeEp8(moe) => {
                    let ep = ep
                        .as_deref_mut()
                        .context("GLM5.2 EP MTP layer has no DeepEP runtime")?;
                    // Conservative protocol-max GEMM bound, same as the
                    // target step: MTP buckets are rank-local too.
                    glm52_moe_ep_layer(
                        ctx,
                        aux,
                        ep,
                        moe,
                        scratch,
                        rows,
                        self.ep_ranks * GLM52_MAX_STEP_ROWS,
                    )?;
                }
                Glm52LayerMlp::MoeTp(router) => {
                    let rank = tp
                        .as_deref_mut()
                        .context("GLM5.2 TP MTP layer has no TP runtime")?;
                    let (state, _slot, bank) = rank
                        .layer_bank(GLM52_MTP_LAYER)
                        .context("GLM5.2 TP MTP layer has no layer-78 slice bank")?;
                    let moe = tp_moe.context("GLM5.2 TP MTP layer has no prefill MoE scratch")?;
                    // The prefill MoE forward rounds rows up to 4: bridge
                    // through the fixed eight-row TP buffers, as the target
                    // layers do. Padding rows carry deterministic bytes
                    // (token 0 / position 0), identical on every rank.
                    copy_hidden_rows_raw_into(
                        ctx,
                        scratch.layer.normed2.data(),
                        GLM52_HIDDEN,
                        &mut scratch.tp_normed2,
                        GLM52_HIDDEN,
                        0,
                        rows,
                    )?;
                    moe.forward(
                        ctx,
                        state,
                        router,
                        bank,
                        &scratch.tp_normed2,
                        rows,
                        &mut scratch.tp_mlp_out,
                    )?;
                    state.prefill_allreduce_in_place(ctx, rows, &mut scratch.tp_mlp_out)?;
                    tp_padded_mlp = true;
                }
                Glm52LayerMlp::Dense(_) => {
                    anyhow::bail!("GLM5.2 MTP layer 78 unexpectedly has a dense MLP")
                }
            }
            glm52_layer_finish(ctx, scratch, 0, tp_padded_mlp)?;
            glm52_mtp_recycle_into(
                ctx,
                &self.bookend,
                &scratch.hidden,
                &mut scratch.final_normed,
            )?;
            glm52_lm_head_into(ctx, &scratch.final_normed, lm_head, &mut scratch.logits)?;
            argmax_bf16_split_into(
                ctx,
                scratch.logits.data(),
                rows,
                GLM52_VOCAB,
                &mut scratch.argmax_partial_values,
                &mut scratch.argmax_partial_indices,
                &mut scratch.argmax_values,
                &mut scratch.argmax_indices,
            )
        };
        if tp_prefill {
            body()
        } else {
            compute_graph.run_or_capture(ctx, body)
        }
    }

    fn argmax_host(&self, ctx: &DeviceContext, bucket_index: usize) -> Result<Vec<u32>> {
        let bucket = &self.buckets[bucket_index];
        let values = ctx.stream.clone_dtoh(&bucket.scratch.argmax_values)?;
        let indices = ctx.stream.clone_dtoh(&bucket.scratch.argmax_indices)?;
        values
            .iter()
            .zip(indices)
            .enumerate()
            .map(|(row, (value, index))| {
                ensure!(
                    value.to_f32().is_finite() && index >= 0,
                    "GLM5.2 MTP row {row} produced invalid argmax value {} at index {index}",
                    value.to_f32(),
                );
                u32::try_from(index).context("GLM5.2 MTP argmax does not fit u32")
            })
            .collect()
    }
}
