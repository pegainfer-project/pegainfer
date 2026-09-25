//! Sampling scratch and batched decode buffers for Qwen3.5.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kv_cache::KvView;

use super::config::Config35;
use super::config::LocalGeometry;

/// Decode buckets at or below this size read full attention through the split-KV
/// kernel instead of the FlashInfer batch-decode kernel.
///
/// FlashInfer launches one CTA per (request, kv head) and never grows that grid
/// with KV length, so a small decode leaves the device idle — four CTAs for a
/// single request, 64 for sixteen. Splitting the KV range instead wins at every
/// bucket this threshold admits, measured on one tree with two runs a side:
/// bs1 9.41/9.36 → 8.57/8.57 ms, c8 11.25/11.27 → 10.65/10.67, c16 13.06/13.08
/// → 12.80/12.82 mean TPOT, qps16 unchanged. It also covers the wide buckets a
/// rate-limited client reaches, where FlashInfer's grid grows one CTA per
/// request and the split form still reaches 932 GB/s at batch 64 against a
/// 1235 GB/s load-only floor. See `docs/models/qwen35/decode-kernel-attribution.md`.
pub(crate) const SPLIT_DECODE_MAX_BATCH: usize = 64;

/// KV splits per (kv head, request) CTA group, for a decode bucket.
///
/// Fixed per bucket rather than derived from KV length because the grid shape
/// has to be constant across a captured CUDA Graph bucket; an over-split request
/// simply leaves later CTAs empty. The counts are where each bucket's curve
/// flattens on the standalone harness: 32 splits is best at batch 1
/// (27.98 µs against 34.98 at 16), 16 is best at batch 8 (57.26 against 60.58 at
/// 32) and batch 16 (96.26 against 98.47), and 8 is best at batch 64 (288.03
/// against 296.35 at 16).
pub(crate) fn split_decode_splits(batch: usize) -> usize {
    if batch <= 2 {
        32
    } else if batch <= 16 {
        16
    } else {
        8
    }
}

/// Upper bound on [`split_decode_splits`], for sizing the partial buffers.
pub(crate) const SPLIT_DECODE_MAX_SPLITS: usize = 32;

/// Pre-allocated GPU buffers for Qwen3.5 batch decode (N requests, 1 token each).
pub(crate) struct BatchDecodeBuffers35 {
    pub(crate) max_batch_size: usize,

    // Shared hidden-state flow [dim, batch]
    pub(crate) hidden: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) attn_results: HiddenStates,
    pub(crate) hidden_mid: HiddenStates,
    pub(crate) gate_up_out: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) logits: HiddenStates,

    // Full attention [dim, batch]
    pub(crate) q_full: HiddenStates,
    pub(crate) q_attn: HiddenStates,
    pub(crate) k_attn: HiddenStates,
    pub(crate) v_attn: HiddenStates,
    pub(crate) attn_out_full: HiddenStates,

    // Split-KV decode scratch [splits, batch, qo_heads, head_dim] and the two
    // per-(split, request, head) softmax accumulators beside it. Sized for the
    // buckets the split kernel serves, not for the whole decode capacity.
    pub(crate) split_partial_o: CudaSlice<f32>,
    pub(crate) split_partial_m: CudaSlice<f32>,
    pub(crate) split_partial_l: CudaSlice<f32>,

    // Linear attention [dim, batch]. The two projections are the fused ones:
    // `qkvz` holds the qkv band below the z band, `ba` holds beta below alpha.
    pub(crate) qkvz: HiddenStates,
    pub(crate) ba: HiddenStates,
    pub(crate) qkv_conv: HiddenStates,
    pub(crate) gdr_out: HiddenStates,
    pub(crate) normed_gated: HiddenStates,

    // Metadata
    pub(crate) token_ids_d: CudaSlice<u32>,
    pub(crate) positions_d: CudaSlice<i32>,
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    // Sampling scratch
    pub(crate) sample: pegainfer_sample::SampleScratch,
    /// Request-local decode steps handed to `select_batch`, reused across
    /// steps to keep the sampling hot path allocation-free. All zeros until
    /// the scheduler wires generated counts through (sampling-parity 1b).
    pub(crate) steps: Vec<u64>,

    /// Page index reserved for CUDA Graph padding slots. Padding entries point
    /// here with seq_len=1 so FlashInfer accesses valid (but discarded) memory.
    padding_page_id: i32,
}

impl BatchDecodeBuffers35 {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
        max_batch_size: usize,
        page_size: usize,
        padding_page_id: i32,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let bs = max_batch_size;
        let max_view_pages = config.max_position_embeddings.div_ceil(page_size).max(1);
        let page_index_capacity = bs
            .checked_mul(max_view_pages)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 decode page-index capacity overflow"))?;
        let q_proj_dim = geometry.local_full_attn_gated_q_dim();
        let q_dim = geometry.local_full_attn_q_dim();
        let kv_dim = geometry.local_full_attn_kv_dim();
        let qkv_dim = geometry.local_linear_qkv_dim();
        let z_dim = geometry.local_linear_z_dim();
        let b_dim = geometry.local_linear_num_value_heads();
        let intermediate = geometry.local_intermediate_size();

        Ok(Self {
            max_batch_size: bs,
            hidden: HiddenStates::zeros(ctx, h, bs)?,
            normed: HiddenStates::zeros(ctx, h, bs)?,
            attn_results: HiddenStates::zeros(ctx, h, bs)?,
            hidden_mid: HiddenStates::zeros(ctx, h, bs)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * intermediate, bs)?,
            act_out: HiddenStates::zeros(ctx, intermediate, bs)?,
            mlp_out: HiddenStates::zeros(ctx, h, bs)?,
            logits: HiddenStates::zeros(ctx, config.selection_vocab, bs)?,

            q_full: HiddenStates::zeros(ctx, q_proj_dim, bs)?,
            q_attn: HiddenStates::zeros(ctx, q_dim, bs)?,
            k_attn: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v_attn: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out_full: HiddenStates::zeros(ctx, q_dim, bs)?,

            split_partial_o: ctx.stream.alloc_zeros(
                SPLIT_DECODE_MAX_SPLITS
                    * bs.min(SPLIT_DECODE_MAX_BATCH)
                    * geometry.local_num_attention_heads()
                    * 256,
            )?,
            split_partial_m: ctx.stream.alloc_zeros(
                SPLIT_DECODE_MAX_SPLITS
                    * bs.min(SPLIT_DECODE_MAX_BATCH)
                    * geometry.local_num_attention_heads(),
            )?,
            split_partial_l: ctx.stream.alloc_zeros(
                SPLIT_DECODE_MAX_SPLITS
                    * bs.min(SPLIT_DECODE_MAX_BATCH)
                    * geometry.local_num_attention_heads(),
            )?,

            qkvz: HiddenStates::zeros(ctx, qkv_dim + z_dim, bs)?,
            ba: HiddenStates::zeros(ctx, 2 * b_dim, bs)?,
            qkv_conv: HiddenStates::zeros(ctx, qkv_dim, bs)?,
            gdr_out: HiddenStates::zeros(ctx, z_dim, bs)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, bs)?,

            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            page_indices_d: ctx.stream.alloc_zeros(page_index_capacity)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,

            sample: pegainfer_sample::SampleScratch::with_selection_width(
                ctx,
                config.selection_vocab,
                config.decodable_vocab,
                bs,
            )?,
            steps: Vec::new(),

            padding_page_id,
        })
    }

    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        assert!(bs <= self.max_batch_size);
        self.hidden.seq_len = bs;
        self.normed.seq_len = bs;
        self.attn_results.seq_len = bs;
        self.hidden_mid.seq_len = bs;
        self.gate_up_out.seq_len = bs;
        self.act_out.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.logits.seq_len = bs;

        self.q_full.seq_len = bs;
        self.q_attn.seq_len = bs;
        self.k_attn.seq_len = bs;
        self.v_attn.seq_len = bs;
        self.attn_out_full.seq_len = bs;

        self.qkvz.seq_len = bs;
        self.ba.seq_len = bs;
        self.qkv_conv.seq_len = bs;
        self.gdr_out.seq_len = bs;
        self.normed_gated.seq_len = bs;
    }

    /// Sync paged attention metadata to GPU.
    ///
    /// `padded_bs` >= `views.len()`: padding slots (if any) point to the
    /// reserved padding page with seq_len=1 so FlashInfer accesses valid memory.
    pub(crate) fn sync_paged_views(
        &mut self,
        ctx: &DeviceContext,
        views: &[KvView],
        padded_bs: usize,
    ) -> Result<()> {
        let real_bs = views.len();
        debug_assert!(padded_bs >= real_bs);

        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(padded_bs);
        let mut chunk_sizes = Vec::with_capacity(padded_bs);

        for view in views {
            all_page_indices.extend_from_slice(view.page_indices());
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(view.last_page_len() as i32);
            chunk_sizes.push(view.seq_len() as i32);
        }

        // Padding slots: 1 page (the padding page), seq_len=1, last_page_len=1.
        for _ in real_bs..padded_bs {
            all_page_indices.push(self.padding_page_id);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(1);
            chunk_sizes.push(1);
        }

        let request_indices: Vec<i32> = (0..padded_bs as i32).collect();
        let kv_tile_indices = vec![0i32; padded_bs];

        anyhow::ensure!(
            all_page_indices.len() <= self.page_indices_d.len(),
            "Qwen3.5 decode page-index overflow: {} view pages exceed buffer capacity {}",
            all_page_indices.len(),
            self.page_indices_d.len()
        );
        ctx.stream
            .memcpy_htod(&all_page_indices, &mut self.page_indices_d)?;
        ctx.stream.memcpy_htod(&indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_lens, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&chunk_sizes, &mut self.kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;

        Ok(())
    }
}
