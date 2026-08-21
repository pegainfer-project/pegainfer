//! Pre-allocated GPU buffers for batched decode (multiple requests, 1 token each).

use anyhow::Result;
use cudarc::driver::CudaSlice;
use log::info;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::ops::build_split_kv_csr;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::gemm_lt_pin_warmup;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kv_cache::KvView;

use crate::split_kv::SplitKvConfig;

/// Bucket sizes for CUDA Graph capture. Actual batch is padded to the nearest bucket.
/// Based on vLLM's cudagraph capture list up to 256; graphs are captured lazily per
/// bucket, and activation buffers are shared (sized once at the largest bucket), so
/// extra buckets cost capture time on first hit, not memory.
///
/// Buckets 8/16 are viable only because decode GEMMs at N <= GEMM_LT_MAX_N run
/// tuned cublasLt algos: cuBLAS's GemmEx heuristic skips split-K for batch in
/// [8, 16] (RTX 5090 ctx1024: 9.2/9.3ms steps vs 7.9ms at bs20), while the Lt
/// heuristic list has full-speed candidates at every small N.
pub(crate) const BATCH_BUCKETS: &[usize] = &[
    1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
    160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
];
const DECODE_ATTENTION_PATH_COUNT: usize = 2;
// Split-KV decode attention: the non-partitioned kernel issues one CTA per
// (request x kv-head), starving SMs at small batch. The path is therefore
// chosen by batch (CTA count vs SM count), NOT context length — at bs=1 the
// 8 CTAs underfill the GPU at any seq_len, so SplitKv wins across the whole
// context range. SPLIT_KV_MAX_BATCH_SIZE caps it where NonPartition's CTAs
// already saturate the SMs (bs<=8 wins big, ~bs16 even, bs32 within ~1%).
// 64-token chunks measured fastest on RTX 5090 (128/256 are 1-7% slower, 32
// past the merge-overhead knee). Measurements: docs/models/qwen3/decode-attention.md.
pub(crate) const SPLIT_KV_CHUNK_TOKENS: usize = 64;
pub(crate) const SPLIT_KV_TUNED_MAX_CHUNKS: usize = 64; // Tuned adaptive-split count cap
const SPLIT_KV_MAX_CHUNKS_PER_REQUEST: usize = 256; // split-KV workspace/guard bound
const SPLIT_KV_MAX_BATCH_SIZE: usize = 32;

/// The split-KV config for `policy`: 64-token floor + per-policy cap (Tuned 64, Pin/PerToken 256).
const fn split_kv_config(policy: NumericPolicy) -> SplitKvConfig {
    let max_chunks = match policy {
        NumericPolicy::Tuned => SPLIT_KV_TUNED_MAX_CHUNKS,
        NumericPolicy::Pin | NumericPolicy::PerToken => SPLIT_KV_MAX_CHUNKS_PER_REQUEST,
    };
    SplitKvConfig::new(SPLIT_KV_CHUNK_TOKENS, max_chunks)
}

/// Tuned chunk size: 64-token floor, coarsened to keep a `basis`-token request within the cap.
pub fn split_chunk_size_for(basis: usize) -> usize {
    split_kv_config(NumericPolicy::Tuned).actual_chunk_size(basis)
}

/// Pin/PerToken fixed chunk size.
pub(crate) fn pin_chunk_size(max_context_tokens: usize) -> usize {
    split_kv_config(NumericPolicy::Pin).actual_chunk_size(max_context_tokens)
}

/// The decode projection GEMM `(M, K)` shapes the Pin path serves, lm_head last.
pub(crate) fn decode_projection_pin_shapes(
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    vocab: usize,
) -> [(usize, usize); 6] {
    [
        (q_dim, hidden),
        (kv_dim, hidden),
        (hidden, q_dim),
        (intermediate, hidden),
        (hidden, intermediate),
        (vocab, hidden),
    ]
}

/// Warm the pinned cuBLASLt algo for each `decode_projection_pin_shapes` `(M, K)`;
/// under `Pin`, `launch_gemm_pin` bails on an un-warmed shape.
pub(crate) fn warmup_decode_projection_pins(
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    vocab: usize,
) -> Result<()> {
    for (m, k) in decode_projection_pin_shapes(hidden, q_dim, kv_dim, intermediate, vocab) {
        let config = gemm_lt_pin_warmup(m, k)?;
        info!(
            "Qwen3 GEMM pin: m={m}, k={k}, splitk={}, reduction_scheme={}",
            config.splitk,
            config.reduction_scheme_name()
        );
    }
    Ok(())
}

fn active_split_kv_config() -> SplitKvConfig {
    split_kv_config(numeric_policy())
}

/// Per-request chunk cap (split-KV padded grid width + guard) for the active policy; Tuned stays at 64.
fn max_split_chunks() -> usize {
    active_split_kv_config().max_chunks_per_request
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeAttentionPath {
    NonPartition,
    SplitKv,
}

impl DecodeAttentionPath {
    fn graph_slot(self) -> usize {
        match self {
            Self::NonPartition => 0,
            Self::SplitKv => 1,
        }
    }
}

/// Find the smallest bucket >= `bs`. Panics if bs > largest bucket.
pub(crate) fn bucket_for(bs: usize) -> usize {
    for &b in BATCH_BUCKETS {
        if b >= bs {
            return b;
        }
    }
    panic!(
        "batch size {bs} exceeds largest bucket {}",
        BATCH_BUCKETS.last().unwrap()
    );
}

/// Pre-allocated buffers for batch decode. All tensors are sized for `max_batch_size`.
///
/// Uses `HiddenStates` (2D) instead of `DeviceVec` (1D) — the "seq_len" dimension
/// is actually the batch dimension (one token per request).
pub(crate) struct BatchDecodeBuffers {
    max_batch_size: usize,

    // Per-layer intermediates [dim, max_batch_size]
    pub(crate) normed: HiddenStates,
    pub(crate) q: HiddenStates,
    pub(crate) k: HiddenStates,
    pub(crate) v: HiddenStates,
    pub(crate) attn_out: HiddenStates,
    pub(crate) attn_proj: HiddenStates,
    /// Fused QKV projection output [q_dim + 2*kv_dim, bs]
    qkv_out: HiddenStates,
    /// Split MLP gate projection output [intermediate_size, bs].
    pub(crate) gate_out: HiddenStates,
    /// Split MLP up projection output [intermediate_size, bs].
    pub(crate) up_out: HiddenStates,
    pub(crate) mlp_act: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) hidden: HiddenStates,
    pub(crate) logits: HiddenStates,

    // GPU metadata
    pub(crate) token_ids_d: CudaSlice<u32>,
    pub(crate) positions_d: CudaSlice<i32>,
    pub(crate) lora_token_slots_d: CudaSlice<i32>,

    // Paged attention metadata (concatenated across requests, CSR format)
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    // Split-K paged attention metadata/workspace.
    pub(crate) split_request_indices_d: CudaSlice<i32>,
    pub(crate) split_kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) split_kv_chunk_size_d: CudaSlice<i32>,
    pub(crate) split_o_indptr_d: CudaSlice<i32>,
    pub(crate) split_block_valid_mask_d: CudaSlice<u8>,
    pub(crate) split_tmp_v: CudaSlice<half::bf16>,
    pub(crate) split_tmp_s: CudaSlice<f32>,
    pub(crate) split_padded_slots: usize,
    /// `(chunk_size, cap)` written by the latest SplitKv metadata sync;
    /// `None` outside SplitKv steps.
    #[cfg(feature = "kernel-call-trace")]
    synced_split_kv: Option<(usize, usize)>,
    max_seq_len: usize,
    /// Model context limit (`max_position_embeddings`) — the `Pin` split-KV chunk basis (see
    /// `split_chunk_size`).
    max_context_tokens: usize,

    /// `NumericPolicy` snapshot at construction, asserted unchanged at `batch_decode` entry. Sizes the
    /// split-KV workspace and feeds `attention_path`, so a later switch would overflow the workspace or
    /// replay a stale `(bucket, path)` graph — the policy-key-trap.
    pub(crate) policy_at_construction: NumericPolicy,

    /// Padding page index for bucket CUDA Graph. Padding slots point here.
    padding_page_id: i32,

    /// One CudaGraphState per `(bucket, attention_path)`, captured on the
    /// full-SM `ctx.stream` — used by the normal decode-only path.
    pub(crate) graphs: Vec<CudaGraphState>,

    /// Parallel cache captured on the Green Context decode-partition stream,
    /// used when decode runs concurrently with prefill (SplitConcurrent). A
    /// graph captured on `ctx.stream` would replay on all SMs, so the split
    /// path needs its own graphs whose nodes are pinned to the decode partition.
    pub(crate) graphs_split: Vec<CudaGraphState>,
}

impl BatchDecodeBuffers {
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_batch_size: usize,
        page_size: usize,
        padding_page_id: i32,
        num_qo_heads: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        let bs = max_batch_size;
        // One construction-time policy snapshot sizes the split-KV workspace and is recorded in
        // `policy_at_construction`; the `batch_decode` entry assert holds the live policy to it.
        let policy = numeric_policy();
        // Pin/PerToken pin SplitKv for every bucket, so the workspace must cover the full
        // max batch, not just Tuned's `<= SPLIT_KV_MAX_BATCH_SIZE` cap.
        let split_batch_cap = match policy {
            NumericPolicy::Pin | NumericPolicy::PerToken => bs,
            NumericPolicy::Tuned => bs.min(SPLIT_KV_MAX_BATCH_SIZE),
        };
        let max_split_slots = split_batch_cap * split_kv_config(policy).max_chunks_per_request;
        if matches!(policy, NumericPolicy::Pin | NumericPolicy::PerToken) {
            log::info!(
                "batch-invariant decode: attention pinned to SplitKv for every bucket; \
                 split-KV workspace sized for max batch {bs}"
            );
        }
        // The concatenated page-index list is counted by reference, not by
        // physical block: prefix-cached blocks are shared, so N views holding
        // the same cached prefix each list those page ids again. Sizing this
        // off the pool's physical block count therefore under-allocates under
        // prefix sharing (#403); the honest bound is every row presenting a
        // full-context view. Padding rows list 1 page each, so `bs *
        // max_view_pages` covers them too (max_view_pages >= 1).
        anyhow::ensure!(page_size > 0, "page_size must be > 0");
        let max_view_pages = max_context_tokens.div_ceil(page_size).max(1);

        Ok(Self {
            max_batch_size: bs,
            normed: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            q: HiddenStates::zeros(ctx, q_dim, bs)?,
            k: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, bs)?,
            attn_proj: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            qkv_out: HiddenStates::zeros(ctx, q_dim + 2 * kv_dim, bs)?,
            gate_out: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            up_out: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            mlp_act: HiddenStates::zeros(ctx, intermediate_size, bs)?,
            mlp_out: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            hidden: HiddenStates::zeros(ctx, hidden_dim, bs)?,
            logits: HiddenStates::zeros(ctx, vocab_size, bs)?,
            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            lora_token_slots_d: ctx.stream.alloc_zeros(bs)?,
            page_indices_d: ctx.stream.alloc_zeros(bs * max_view_pages)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,
            split_request_indices_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_kv_tile_indices_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_kv_chunk_size_d: ctx.stream.alloc_zeros(1)?,
            split_o_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            split_block_valid_mask_d: ctx.stream.alloc_zeros(max_split_slots)?,
            split_tmp_v: ctx.stream.alloc_zeros(max_split_slots * q_dim)?,
            split_tmp_s: ctx.stream.alloc_zeros(max_split_slots * num_qo_heads)?,
            split_padded_slots: 0,
            #[cfg(feature = "kernel-call-trace")]
            synced_split_kv: None,
            max_seq_len: 0,
            max_context_tokens,
            policy_at_construction: policy,
            padding_page_id,
            graphs: BATCH_BUCKETS
                .iter()
                .flat_map(|_| (0..DECODE_ATTENTION_PATH_COUNT).map(|_| CudaGraphState::new()))
                .collect(),
            graphs_split: BATCH_BUCKETS
                .iter()
                .flat_map(|_| (0..DECODE_ATTENTION_PATH_COUNT).map(|_| CudaGraphState::new()))
                .collect(),
        })
    }

    /// Set actual batch size for this step. Adjusts the seq_len field on all HiddenStates.
    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        assert!(bs <= self.max_batch_size);
        self.normed.seq_len = bs;
        self.q.seq_len = bs;
        self.k.seq_len = bs;
        self.v.seq_len = bs;
        self.attn_out.seq_len = bs;
        self.attn_proj.seq_len = bs;
        self.qkv_out.seq_len = bs;
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.mlp_act.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.hidden.seq_len = bs;
        self.logits.seq_len = bs;
    }

    /// Sync paged attention metadata from multiple KvViews to GPU buffers.
    ///
    /// `padded_bs` >= `kv_views.len()`: padding slots (if any) point to the
    /// reserved padding page with seq_len=1 so FlashInfer accesses valid memory.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_views: &[&KvView],
        padded_bs: usize,
    ) -> Result<()> {
        let real_bs = kv_views.len();
        debug_assert!(padded_bs >= real_bs);

        // Build concatenated page_indices and CSR indptr
        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(padded_bs);
        let mut chunk_sizes = Vec::with_capacity(padded_bs);
        self.max_seq_len = 0;

        for kv in kv_views {
            all_page_indices.extend_from_slice(kv.page_indices());
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
            self.max_seq_len = self.max_seq_len.max(kv.seq_len());
        }

        // Padding slots: 1 page (the padding page), seq_len=1, last_page_len=1
        for _ in real_bs..padded_bs {
            all_page_indices.push(self.padding_page_id);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(1);
            chunk_sizes.push(1);
        }

        let request_indices: Vec<i32> = (0..padded_bs as i32).collect();
        let kv_tile_indices = vec![0i32; padded_bs];

        // Fail loud instead of tripping cudarc's copy assert (a panic here
        // kills the worker thread and wedges the engine — #403). Reachable
        // only if a view exceeds the context limit the buffer was sized for.
        anyhow::ensure!(
            all_page_indices.len() <= self.page_indices_d.len(),
            "decode page-index overflow: {} view pages (bs={real_bs}, padded={padded_bs}) \
             exceed buffer capacity {}; a view is larger than the context limit \
             the buffers were sized for",
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
        self.sync_split_kv_meta(ctx, kv_views, padded_bs)?;

        Ok(())
    }

    /// Chunk count sets the online-softmax rescale order. `Pin`/`PerToken` fix the split SIZE so the
    /// count is request-local (batch-invariant); `Tuned` sizes off the live batch.
    fn split_chunk_size(&self) -> usize {
        match numeric_policy() {
            NumericPolicy::Tuned => split_chunk_size_for(self.max_seq_len),
            NumericPolicy::Pin | NumericPolicy::PerToken => pin_chunk_size(self.max_context_tokens),
        }
    }

    /// Reads back `synced_split_kv`.
    #[cfg(feature = "kernel-call-trace")]
    pub(crate) fn resolved_split_kv(&self) -> (usize, usize) {
        self.synced_split_kv
            .expect("resolved_split_kv read off a SplitKv step / before sync_split_kv_meta")
    }

    fn sync_split_kv_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_views: &[&KvView],
        padded_bs: usize,
    ) -> Result<()> {
        // Clear before the NonPartition early return so trace cannot reuse stale SplitKv metadata.
        #[cfg(feature = "kernel-call-trace")]
        {
            self.synced_split_kv = None;
        }
        // Tuned skips split metadata past the cap (NonPartition); Pin/PerToken build the CSR for
        // every bucket. The construction snapshot here and the live-policy `split_chunk_size`/
        // `max_split_chunks` below are held equal by the `batch_decode` entry assert, so the CSR fits.
        if padded_bs > SPLIT_KV_MAX_BATCH_SIZE
            && matches!(self.policy_at_construction, NumericPolicy::Tuned)
        {
            return Ok(());
        }
        let split_chunk_size = self.split_chunk_size();
        let cap = max_split_chunks();
        #[cfg(feature = "kernel-call-trace")]
        {
            self.synced_split_kv = Some((split_chunk_size, cap));
        }
        let kv_lens: Vec<usize> = kv_views.iter().map(|kv| kv.seq_len()).collect();
        let csr = build_split_kv_csr(split_chunk_size, cap, &kv_lens, padded_bs)?;

        let split_kv_chunk_size = [split_chunk_size as i32];
        ctx.stream
            .memcpy_htod(&csr.request_indices, &mut self.split_request_indices_d)?;
        ctx.stream
            .memcpy_htod(&csr.kv_tile_indices, &mut self.split_kv_tile_indices_d)?;
        ctx.stream
            .memcpy_htod(&split_kv_chunk_size, &mut self.split_kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&csr.o_indptr, &mut self.split_o_indptr_d)?;
        ctx.stream
            .memcpy_htod(&csr.block_valid_mask, &mut self.split_block_valid_mask_d)?;
        self.split_padded_slots = padded_bs * cap;

        Ok(())
    }

    /// Decode attention kernel. Tuned selects on batch (SplitKv fills the SMs below the cap,
    /// NonPartition above). Pin/PerToken pin SplitKv for every bucket so it never varies with
    /// co-batched load — the chunk count is already request-local (`pin_chunk_size`).
    pub(crate) fn attention_path(padded_bs: usize, policy: NumericPolicy) -> DecodeAttentionPath {
        if matches!(policy, NumericPolicy::Pin | NumericPolicy::PerToken) {
            return DecodeAttentionPath::SplitKv;
        }
        if padded_bs <= SPLIT_KV_MAX_BATCH_SIZE {
            DecodeAttentionPath::SplitKv
        } else {
            DecodeAttentionPath::NonPartition
        }
    }

    pub(crate) fn graph_index(bucket_idx: usize, path: DecodeAttentionPath) -> usize {
        bucket_idx * DECODE_ATTENTION_PATH_COUNT + path.graph_slot()
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_core::tensor::DeviceContext;
    use pegainfer_kv_cache::KvView;

    use super::BatchDecodeBuffers;
    use super::build_split_kv_csr;

    #[test]
    fn shared_prefix_views_are_counted_by_reference() {
        let ctx = DeviceContext::new().expect("CUDA context");
        let page_size = 16;
        let max_context_tokens = 64; // 4 pages per full-context view
        let bs = 4;
        let mut bufs = BatchDecodeBuffers::new(
            &ctx,
            8,
            8,
            8,
            8,
            16,
            bs,
            page_size,
            0,
            1,
            max_context_tokens,
        )
        .expect("buffer alloc");

        // Every view lists the SAME 4 physical pages (a shared cached prefix):
        // 16 listed entries backed by only 4 physical blocks. A buffer sized
        // off the physical pool count under-allocates exactly here (#403).
        let views: Vec<KvView> = (0..bs)
            .map(|_| KvView::new(vec![0, 1, 2, 3], 64, page_size))
            .collect();
        let refs: Vec<&KvView> = views.iter().collect();
        bufs.sync_paged_meta(&ctx, &refs, bs)
            .expect("shared views fit");

        // A view past the context limit the buffers were sized for must error
        // (fail loud), not trip cudarc's copy assert and kill the worker.
        let oversized: Vec<KvView> = (0..bs)
            .map(|_| KvView::new(vec![0, 1, 2, 3, 4], 80, page_size))
            .collect();
        let refs: Vec<&KvView> = oversized.iter().collect();
        let err = bufs
            .sync_paged_meta(&ctx, &refs, bs)
            .expect_err("oversized views must fail loud");
        assert!(err.to_string().contains("page-index overflow"), "{err}");
    }

    #[test]
    fn uniform_batch_csr() {
        // 2 requests at kv_len=100, chunk_size=64 -> 2 chunks each; cap=64, padded_bs=2.
        let csr = build_split_kv_csr(64, 64, &[100, 100], 2).unwrap();
        assert_eq!(csr.o_indptr, vec![0, 2, 4]);
        assert_eq!(csr.request_indices[..4].to_vec(), vec![0, 0, 1, 1]);
        assert_eq!(csr.kv_tile_indices[..4].to_vec(), vec![0, 1, 0, 1]);
        assert_eq!(csr.block_valid_mask[..4].to_vec(), vec![1u8, 1, 1, 1]);
        assert_eq!(csr.request_indices.len(), 2 * 64);
    }

    #[test]
    fn padding_requests_contribute_zero_chunks() {
        // 2 real requests (1 and 2 chunks) padded to 3 slots; cap=4.
        let csr = build_split_kv_csr(64, 4, &[40, 100], 3).unwrap();
        assert_eq!(csr.o_indptr, vec![0, 1, 3, 3]);
        assert_eq!(csr.request_indices[..3].to_vec(), vec![0, 1, 1]);
        assert_eq!(csr.kv_tile_indices[..3].to_vec(), vec![0, 0, 1]);
        assert_eq!(csr.request_indices.len(), 3 * 4);
    }

    #[test]
    fn errors_when_a_request_exceeds_the_cap() {
        assert!(build_split_kv_csr(64, 4, &[1000], 1).is_err());
    }
}
