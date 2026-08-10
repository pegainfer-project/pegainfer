//! GLM5.2 DP8/EP8 full-model decode: every rank owns the whole non-expert
//! path (embed → 78 decoder layers → final norm → lm_head → greedy argmax)
//! plus its 32 local experts, and forwards one of the
//! [`GLM52_DECODE_BUCKETS`] batch buckets per step (real request tokens in
//! occupied slots, padding rows elsewhere; prefill rides decode as *spans* —
//! several consecutive positions of one slot in a single step). KV lives in
//! a rank-wide pool of 64-token pages: the rank engine's `BlockPool` assigns
//! pages per request, and every step's [`Glm52StepKv`] carries each row's
//! page table row plus its flat cache write slot — padding rows ride the
//! pool's reserved padding page, whose garbage writes nobody reads.
//!
//! Every step, every rank runs the forward unconditionally with its OWN
//! bucket (one free-running engine per rank — idle ranks forward padding
//! rows), dispatching those rows into every MoE layer's DeepEP collective.
//! The collectives pair by entry count in the fixed layer order 3..=77 with
//! rank-local row counts under the conservative protocol-max bound
//! (`docs/models/glm52/free-running-dp.md` §2), and the per-bucket fixed
//! row count keeps every step's kernel shapes identical within a bucket
//! (the whole-step CUDA graphs' contract).

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr as _;
use cudarc::driver::PinnedHostSlice;
use half::bf16;
use pegainfer_core::cuda_graph::CudaGraphDumpSummary;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use pegainfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use pegainfer_kernels::ops::GLM52_TP_TOKENS;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::embedding_rows_into;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use pegainfer_kernels::ops::glm52_fp8_weight_only_gemv_launch;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStatesRef;
use pegainfer_kv_store::ArenaSpec;
use pegainfer_sample::BatchSamplingRow;
use pegainfer_sample::BatchSamplingScratch;
use pegainfer_sample::effectively_greedy;
use pegainfer_sample::gpu_sample_batch_into;
use pegainfer_sample::mix_seed;

use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_INDEX_TOPK;
use crate::config::GLM52_LAYERS;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::config::GLM52_VOCAB;
use crate::config::glm52_layer_has_full_indexer;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52KvSlab;
use crate::layer::Glm52LayerCaches;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_select_mla_backend;
use crate::moe_ep::Glm52MoeEpState;
use crate::moe_tp::Glm52MoeTpRank;
use crate::prefill_tp::Glm52TpPrefillExecutor;
use crate::prefill_tp::Glm52TpPrefillModelView;
use crate::scratch::Glm52DecodeScratch;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::retype_owned;

mod build;
mod launch_ahead;
mod mtp;
mod step_body;
use launch_ahead::Glm52SpeculatedStep;
use mtp::Glm52NativeMtp;
use mtp::Glm52NativeMtpFixed;
pub(crate) use mtp::MTP_SCRATCH_PAGES_PER_SLOT;
use step_body::run_step_body;

/// The compile-time slot-count CEILING: fixed arrays (`RankSlots`,
/// launch-ahead seen-sets, contiguity walks) are sized to it. A slot is a
/// batch lane (and the draft lane's state key), not a cache region — KV
/// pages come from the rank's shared pool. The count actually admitted and
/// sized for is [`glm52_decode_slots`], a startup knob; the pair
/// (slots, drafts) must satisfy `slots * (1 + drafts) <= GLM52_MAX_STEP_ROWS`
/// (validated at engine build), so the ceiling is only reachable with a
/// shortened draft span.
pub(crate) const GLM52_MAX_BATCH_PER_RANK: usize = 32;

/// The per-rank decode slot count: `GLM52_DECODE_SLOTS` (1..=ceiling),
/// default 8 — the latency-lean P/D profile. Wide-EP throughput deployments
/// run 32 slots with `GLM52_MTP_DRAFTS=2` to stay inside the 96-row step.
pub(crate) fn glm52_decode_slots() -> usize {
    static SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SLOTS.get_or_init(|| {
        std::env::var("GLM52_DECODE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, GLM52_MAX_BATCH_PER_RANK))
            .unwrap_or(8)
    })
}

/// Cache geometry is carved in units of the 64-token FlashMLA page (== the
/// index-K cache block), so the per-request context cap must sit on a page
/// boundary — the page-table width is `max_model_len / page`, and a
/// non-multiple cap would strand a partial page the table cannot address.
pub(crate) const GLM52_MODEL_LEN_ALIGN: usize = GLM52_FLASHMLA_SPARSE_PAGE_SIZE;

/// The rank-wide KV pool size for a given per-request cap: capacity for
/// every slot's full-lifetime draw plus the reserved padding page
/// (`BlockPool` block 0 — padding rows and CUDA-graph pre-capture write
/// there). A request's lifetime draw is `ceil((prompt + max_tokens)/page)`
/// with `prompt + max_tokens <= cap + 1` (`validate_request`) — one page
/// more than its KV ever writes, because kvbm appends the final generated
/// token and eagerly provisions its page (the dangling-token contract). The
/// engine's `BlockPool` and the rank arenas
/// ([`Glm52RankModel::finish_kv`]) MUST agree on this count: pool block ids
/// index the arenas directly.
pub(crate) fn glm52_pool_blocks(max_model_len: usize, pool_slots: usize) -> usize {
    pool_slots * (max_model_len + 1).div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE) + 1
}

/// Page-table width: the pages a single request at the full cap addresses.
/// Every per-row page table (bucket block tables) is this wide; rows with
/// fewer pages are padded with the padding page id.
pub(crate) fn glm52_table_width(max_model_len: usize) -> usize {
    max_model_len.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE)
}

/// One layer's MLA slice inside a slab page: 64 tokens x 656 B fp8_ds_mla.
/// Every topology persists this row — prefill-only is a P/D producer whose
/// wire format IS the EP decode consumer's cache format.
pub(crate) const GLM52_KV_PAGE_MLA_BYTES: usize =
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
/// One full-indexer layer's index-K slice inside a slab page: 64 tokens x
/// (128 B fp8 key + 4 B f32 scale) in the DeepGEMM interleaved block layout.
pub(crate) const GLM52_KV_PAGE_IDXK_BYTES: usize = INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4);
/// Bytes of one slab page's content ([`glm52_page_layout`] asserts the sum):
/// 78 MLA slices + 21 full-indexer index-K slices + the layer-78 MTP
/// committed mirrors. This is the save/load copy unit — the pad tail up to
/// [`GLM52_KV_PAGE_STRIDE`] never moves.
pub(crate) const GLM52_KV_PAGE_CONTENT_BYTES: usize = 3_502_592;
/// Byte distance between consecutive slab pages: content rounded up to the
/// 656-byte cache token row (the FlashMLA TMA derives its per-page step from
/// the stride, so it must stay token-row granular). THE wire-layout identity:
/// the offload namespace and the native P/D fingerprint both fold it.
pub(crate) const GLM52_KV_PAGE_STRIDE: usize = 3_503_040;

const _: () = assert!(GLM52_KV_PAGE_STRIDE % GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN == 0);
const _: () = assert!(
    GLM52_KV_PAGE_STRIDE >= GLM52_KV_PAGE_CONTENT_BYTES
        && GLM52_KV_PAGE_STRIDE - GLM52_KV_PAGE_CONTENT_BYTES
            < GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
);

/// The page-first slice map of one slab page: per-layer offsets in layer
/// order, then the layer-78 MTP committed-mirror slices. A pure function of
/// the architecture — the ONE construction shared by `finish_kv`, the MTP
/// attach, and the layout unit test, so the offset table cannot drift from
/// the wire constants.
struct Glm52PageLayout {
    layers: Vec<Glm52LayerCaches>,
    /// Layer-78 committed mirrors (native MTP): radix hits reuse L78 KV by
    /// pool page id, so the mirrors ride the same page as the target layers.
    mtp: Glm52LayerCaches,
}

fn glm52_page_layout() -> Glm52PageLayout {
    let mut offset = 0usize;
    let mut layers = Vec::with_capacity(GLM52_LAYERS);
    for layer in 0..GLM52_LAYERS {
        let mla_offset = offset;
        offset += GLM52_KV_PAGE_MLA_BYTES;
        let index_k_offset = glm52_layer_has_full_indexer(layer).then(|| {
            let at = offset;
            offset += GLM52_KV_PAGE_IDXK_BYTES;
            at
        });
        layers.push(Glm52LayerCaches {
            mla_offset,
            index_k_offset,
        });
    }
    let mtp = Glm52LayerCaches {
        mla_offset: offset,
        index_k_offset: Some(offset + GLM52_KV_PAGE_MLA_BYTES),
    };
    offset += GLM52_KV_PAGE_MLA_BYTES + GLM52_KV_PAGE_IDXK_BYTES;
    assert_eq!(
        offset, GLM52_KV_PAGE_CONTENT_BYTES,
        "GLM5.2 page layout drifted from its wire constant"
    );
    Glm52PageLayout { layers, mtp }
}

pub(crate) fn glm52_arena_bytes(
    max_model_len: usize,
    num_blocks: usize,
    prefill_only: bool,
) -> Result<usize> {
    // One page-first slab page per pool block carries every layer's MLA and
    // index-K slices, MTP committed mirrors included (the page layout is a
    // wire constant, drafter or not), plus the one-page-content tail slack
    // `finish_kv` allocates for the kernels' conservative extent checks. The
    // EP MTP scratch pages extend the slab past the pool and are charged by
    // `glm52_mtp_arena_bytes`.
    let slab = num_blocks
        .checked_mul(GLM52_KV_PAGE_STRIDE)
        .and_then(|bytes| bytes.checked_add(GLM52_KV_PAGE_CONTENT_BYTES))
        .context("GLM5.2 KV slab byte count overflow")?;
    let table_width = glm52_table_width(max_model_len);
    let rope_tables = 2 * max_model_len * GLM52_ROPE_HALF * size_of::<bf16>();
    let bucket_rows: usize = if prefill_only {
        0
    } else {
        GLM52_DECODE_BUCKETS.iter().sum()
    };
    let indexer_logits =
        bucket_rows * max_model_len.next_multiple_of(256) * (size_of::<bf16>() + size_of::<f32>());
    let block_tables = bucket_rows * table_width * size_of::<i32>();
    let prefill_unpacked = if prefill_only {
        num_blocks
            * GLM52_FLASHMLA_SPARSE_PAGE_SIZE
            * crate::config::GLM52_KV_A_OUT
            * size_of::<bf16>()
            + num_blocks * size_of::<i32>()
    } else {
        0
    };
    Ok(slab + rope_tables + indexer_logits + block_tables + prefill_unpacked)
}

/// The page-table width and index-K cache layout for a given cap — the ONE
/// construction shared by the two-phase build and the TP4 prefill executor,
/// so a layout change cannot drift between them. The stride is the slab
/// page stride; `cache_layer_offset_bytes` stays 0 here because one bucket
/// scratch serves every layer — each launch carries its layer's slice
/// offset (struct-update at the use site keeps this the single origin of
/// blocks/size/stride).
fn glm52_index_cache_layout(
    max_model_len: usize,
    num_blocks: usize,
) -> (usize, Glm52IndexerCacheLayout) {
    // The index-K slices are indexed by the same pool block ids as the MLA
    // slices, so the layout holds the same block count.
    let layout = Glm52IndexerCacheLayout {
        cache_blocks: num_blocks,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_layer_offset_bytes: 0,
        cache_block_stride_bytes: GLM52_KV_PAGE_STRIDE,
    };
    (glm52_table_width(max_model_len), layout)
}

fn glm52_persistent_mla_bytes_per_token(
    prefill_only: bool,
    backend: crate::mla_decode::Glm52MlaBackend,
) -> usize {
    if prefill_only {
        // P/D producer wire/storage format, independent of TP4's local
        // attention execution backend.
        GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
    } else {
        backend.cache_bytes_per_token()
    }
}

/// The step-row ceiling. Step rows and slots are SEPARATE dimensions — they
/// were both 8 until #812, which silently capped verify spans at 1 row/slot
/// under full occupancy and collapsed speculation (measured: TPOT == ITL at
/// c32 on EP4). Since the slot count and draft length became startup knobs
/// this is a standalone budget the runtime pair must fit
/// (`slots * (1 + drafts) <= 96`, validated at engine build): 16 slots ride
/// the full 5-draft span, 32 slots ride a 2-draft span — the same row
/// ceiling, so bucket/graph/kernel instantiations are shared across
/// profiles. 96 (not more) because the min-gemv token split, the MQA AOT
/// batch guard, and the decode-feed slot kernel are all extended exactly
/// this far (#817).
pub(crate) const GLM52_MAX_STEP_ROWS: usize = 96;

/// The decode batch buckets, ascending. Each bucket has its own captured
/// CUDA graphs, scratch arena, and FlashMLA plans, and the batched GEMV
/// kernel is instantiated for exactly these row counts (`kBatchedGemvBatch*`
/// in `glm52_moe_gemv.cu` — a drift crashes at the launch boundary). The
/// engine picks the smallest bucket covering its own demand, so a
/// lightly-loaded rank keeps the small-step cost; discrete buckets (not a
/// continuum) keep the whole-step graphs' fixed-shape contract.
pub(crate) const GLM52_DECODE_BUCKETS: [usize; 9] =
    [1, 2, 4, 8, 16, 32, 48, 64, GLM52_MAX_STEP_ROWS];

// NOTE (#812): the MTP round shares GLM52_DECODE_BUCKETS. The draft leg
// never exceeds one row per slot, but the CONTEXT leg ingests every
// committed token of a verify step — up to the full row ceiling — so its
// bucket list must match the decode buckets.

// DeepEP protocol worst-case: each source token contributes ≤1 row per
// local expert. The per-expert recv slab is sized at RUNTIME from the launch
// topology (`deepgemm_masked_cap(num_ranks)` = ranks × GLM52_MAX_STEP_ROWS,
// alignment-padded), so no compile-time rank×batch bound exists anymore —
// the EP8-flavored `GLM52_DEEPGEMM_MASKED_CAP` assert that stood here
// predated verify-span rows and under-counted every fleet wider than EP8.

// The min-latency GEMV (router logits, indexer weights_proj) dispatches
// tokens 1..=8 plus the decode bucket sizes; a bucket bump must extend
// glm52_min_gemv.cuh first (#812 added 16/32/48).
const _: () = assert!(GLM52_MAX_STEP_ROWS <= pegainfer_kernels::ops::GLM52_MIN_GEMV_MAX_TOKENS);

// The decode feed kernel runs one 32-thread block (`glm52_decode_feed.cu`);
// a batch-cap bump past it must widen the kernel, not silently truncate.
const _: () = assert!(GLM52_MAX_BATCH_PER_RANK <= 32);

/// The step's forward shape for one rank: `bucket` rows (a member of
/// [`GLM52_DECODE_BUCKETS`]; the MoE collectives pair by entry count with
/// rank-local row counts under the conservative protocol-max bound), with
/// `slots[row]` naming the cache slot each forwarded row addresses for
/// `row < bucket` (active slots first, padding rows parked on free slots
/// whose cache regions are dead).
///
/// A slot may own SEVERAL rows (a *span*): one contiguous run of rows walking
/// consecutive positions of that slot's sequence — how prompt tokens batch
/// through the decode path, and the shape a DSpark verify step reuses. Within
/// a step, a later row of a span attends to the earlier rows' KV through the
/// cache: per layer every row's cache write lands before any row's attention
/// launches, and row `k`'s `seq_len` admits exactly the positions before it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52StepShape {
    pub(crate) bucket: usize,
    #[serde(with = "serde_step_rows")]
    pub(crate) slots: [u8; GLM52_MAX_STEP_ROWS],
    /// Rows `0..active_rows` carry real requests; `active_rows..bucket` are
    /// padding. Carried explicitly because a padding input is NOT
    /// value-distinguishable from an active one (a single-token prompt `[0]`
    /// legally feeds `(token 0, position 0)`).
    pub(crate) active_rows: usize,
}

/// The step's KV paging, decided by the rank engine's `BlockPool`:
/// where each forwarded row's cache writes land and which pages its
/// attention/indexer walk. Uploaded by the step prologue into the bucket's
/// device block table / slot mapping (the captured graphs read only those
/// device buffers — a page's physical id is data, never baked).
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52StepKv {
    /// `[bucket, table_width]` row-major page ids. Row `r` holds the pages
    /// covering its request's KV through this step (span rows repeat their
    /// slot's row); entries past the covered pages — and every padding row —
    /// are the pool's padding page id.
    pub(crate) pages: Box<[i32]>,
    /// Per-row flat cache write slot: `pages[position/64]*64 + position%64`
    /// (the fp8_ds_mla packed cache and the index-K cache share this token
    /// index space). Padding rows point into the padding page.
    #[serde(with = "serde_step_rows")]
    pub(crate) slot_mapping: [i64; GLM52_MAX_STEP_ROWS],
    /// Native P/D boundary restores: `(src, dst)` pool page pairs copied
    /// across every KV arena before this step's kernels run.
    pub(crate) boundary_copies: Vec<(i32, i32)>,
}

/// serde's derive stops at 32-element arrays; the step-row arrays serialize
/// as plain sequences (the record/replay probes are the only consumer).
mod serde_step_rows {
    pub(crate) fn serialize<S, T>(value: &[T], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
        T: serde::Serialize,
    {
        serializer.collect_seq(value)
    }

    pub(crate) fn deserialize<'de, D, T, const N: usize>(
        deserializer: D,
    ) -> Result<[T; N], D::Error>
    where
        D: serde::Deserializer<'de>,
        T: serde::Deserialize<'de>,
    {
        let rows = <Vec<T> as serde::Deserialize>::deserialize(deserializer)?;
        rows.try_into()
            .map_err(|rows: Vec<T>| serde::de::Error::invalid_length(rows.len(), &"step-row array"))
    }
}

// There used to be a short-context attention tier here (topk 256 while every
// row's context fit in it — lossless, 1/8 the index walk). Dropped: the
// serving traffic is agent workloads whose contexts start well past 2048, so
// the tier was dead weight — 2x the pre-captured graphs and a second MLA
// schedule per bucket. To bring it back, restore the (bucket x tier) arrays
// from git history; both decode backends already accept any topk multiple
// of 64, and the checked-in FlashInfer cubin closure covers topk 256.

// The attention `topk` feeds the DSA indexer's top-k selection, whose
// buffers are sized for GLM52_INDEX_TOPK rows — pin the range here so the
// indexer forward never needs to re-check it per layer per step.
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK > 0);
const _: () = assert!(GLM52_FLASHMLA_SPARSE_TOPK <= GLM52_INDEX_TOPK);

/// DeepGEMM paged MQA requires BLOCK_KV=64 — a kernel constraint, not a
/// model property (kept here, not in config.rs).
pub(crate) const INDEX_CACHE_BLOCK: usize = 64;
/// The DeepGEMM MQA indexer's persistent-grid size. 132 is the H200 SM count
/// and is baked into the AOT instantiation (`kAotNumSms` in
/// glm52_deepgemm_mqa.cu, enforced by its `num_sms == kAotNumSms` gate), so
/// it deliberately stays 132 on 152-SM GB300 — the schedule metadata drives
/// correctness; "fixing" this to the live SM count breaks the AOT gate.
pub(crate) const NUM_SMS: usize = 132;

pub(crate) fn rope_tables(position: usize) -> (Vec<bf16>, Vec<bf16>) {
    let theta = crate::config::GLM52_ROPE_THETA as f32;
    (0..GLM52_ROPE_HALF)
        .map(|j| {
            let inv_freq = 1.0 / theta.powf(j as f32 / GLM52_ROPE_HALF as f32);
            let angle = position as f32 * inv_freq;
            (bf16::from_f32(angle.cos()), bf16::from_f32(angle.sin()))
        })
        .unzip()
}

/// One DP rank: the full non-expert model plus this rank's expert banks.
pub(crate) struct Glm52RankModel {
    layers: Vec<Glm52DecoderLayerWeights>,
    /// The rank's page-first KV slab: `pool_blocks` pages (plus the EP MTP
    /// scratch pages past them), each holding every layer's cache slices at
    /// the [`glm52_page_layout`] offsets in `caches`.
    kv_slab: Glm52KvSlab,
    caches: Vec<Glm52LayerCaches>,
    mtp: Option<Glm52NativeMtp>,
    embed: DeviceMatrix,
    final_norm: DeviceVec,
    /// Full vocabulary head retained for DSpark and non-greedy sampling.
    lm_head: DeviceMatrix,
    /// Contiguous vocabulary shard used only by attention-TP decode. EP8
    /// computes the full head directly and leaves this absent.
    decode_lm_head: Option<DeviceMatrix>,
    decode_vocab_start: usize,
    /// Per-bucket execution state, index-aligned with
    /// [`GLM52_DECODE_BUCKETS`]. Selecting one `Glm52BucketState` selects the
    /// plans, scratch, graphs, and block table together — a graph can never
    /// be taken from one shape and restored into another.
    buckets: Vec<Glm52BucketState>,
    /// Width of every per-row page-table row ([`glm52_table_width`]): the
    /// pages one request at the full cap addresses.
    table_width: usize,
    /// Per-request context cap: `prompt + max_tokens - 1 <= max_model_len`.
    /// Decided at launch (VRAM probe or `--max-model-len`); sized the
    /// rank-wide per-layer MLA and index-K page pools at build time
    /// ([`glm52_pool_blocks`]).
    max_model_len: usize,
    pool_blocks: usize,
    /// EP rank count of the launch topology (8 for EP8, 4 for EP4, 1 for the
    /// tensor-replicated topologies): the factor between the per-rank batch
    /// cap and the MoE collectives' protocol-max `global_tokens` bound.
    ep_ranks: usize,
    /// Built with `--moe-topo tp`: every MoE arm is `MoeTp`, bucket-8
    /// steps are span steps (all 8 rows one owner rank), and the
    /// engine must stage the span owner on every such step.
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    /// Device-resident rope tables for every position (`[max_model_len,
    /// GLM52_ROPE_HALF]` as a gatherable matrix); the prologue gathers each row's
    /// position row instead of recomputing on the host and copying up.
    cos_table: DeviceMatrix,
    sin_table: DeviceMatrix,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    /// FlashInfer batch-sampling buffers for the non-greedy rows, sized for
    /// the max bucket × vocab and shared by every bucket (the sampling pass
    /// runs outside the captured graphs, so pointer stability per bucket is
    /// not required). Allocated at build — a mid-serving step must never hit
    /// the allocator.
    sampling_scratch: Option<BatchSamplingScratch>,
    /// TP4 eager-prefill executor and reusable 32-row workspace. Persistent
    /// KV lives in `caches`.
    prefill: Option<Glm52TpPrefillExecutor>,
    /// In-flight speculative next-step replay, if any (see `decode_step`).
    speculated: Option<Glm52SpeculatedStep>,
    /// What the per-row `positions` device buffer currently holds (padding
    /// rows included): the feed kernel advances it without host readback,
    /// and a speculation must keep every row under the model-length cap.
    device_positions: [usize; GLM52_MAX_STEP_ROWS],
}

/// Everything one decode bucket owns: the MLA schedule and whole-step
/// graph, shared scratch, and the device block table.
struct Glm52BucketState {
    rows: usize,
    sched: Glm52MlaSchedMetadata,
    scratch: Glm52DecodeScratch,
    graph: CudaGraphState,
    block_table: CudaSlice<i32>,
    /// Pinned landing buffers for this bucket's argmax D2H, sized exactly
    /// `rows` (`memcpy_dtoh` copies the DESTINATION's byte count). Pinned
    /// memory keeps the copy asynchronous so the next step's replay can be
    /// enqueued launch-ahead before the host blocks on the result.
    argmax_values_host: PinnedHostSlice<bf16>,
    argmax_indices_host: PinnedHostSlice<i32>,
}

/// The output of [`Glm52RankModel::build_fixed`]: every len/chunk-scaled
/// piece of a rank model, built BEFORE the pool block count exists so launch
/// can measure the fixed footprint and size the pool from real free VRAM.
/// [`Glm52RankModel::finish_kv`] consumes it. Field meanings match the
/// [`Glm52RankModel`] fields of the same name.
pub(crate) struct Glm52RankModelFixed {
    layers: Vec<Glm52DecoderLayerWeights>,
    mtp: Option<Glm52NativeMtpFixed>,
    embed: DeviceMatrix,
    final_norm: DeviceVec,
    lm_head: DeviceMatrix,
    decode_lm_head: Option<DeviceMatrix>,
    decode_vocab_start: usize,
    /// Built against a 1-block placeholder cache geometry (nothing in a
    /// bucket's allocations scales with the pool); finish_kv rebinds the
    /// sched contract and MQA shape to the decided count.
    buckets: Vec<Glm52BucketState>,
    table_width: usize,
    max_model_len: usize,
    mla_backend: crate::mla_decode::Glm52MlaBackend,
    mla_cache_bytes_per_token: usize,
    ep_ranks: usize,
    slot_mapping: CudaSlice<i64>,
    seq_lens: CudaSlice<i32>,
    cos_table: DeviceMatrix,
    sin_table: DeviceMatrix,
    positions: CudaSlice<u32>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    token_ids: CudaSlice<u32>,
    sampling_scratch: Option<BatchSamplingScratch>,
    /// Chunk-scaled workspace only; its pool-scaled unpacked-KV buffers are
    /// attached in finish_kv.
    prefill: Option<Glm52TpPrefillExecutor>,
}

impl Glm52RankModel {
    /// Export one already pre-captured whole-step bucket graph. The scheduler
    /// selects the topology's serving shape; this method only enforces that
    /// the requested bucket belongs to this model and is ready for replay.
    pub(crate) fn dump_decode_graph_png(
        &self,
        bucket: usize,
        png_path: &std::path::Path,
        title: &str,
    ) -> Result<CudaGraphDumpSummary> {
        let state = self
            .buckets
            .iter()
            .find(|state| state.rows == bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 graph dump bucket {bucket} is not a member of {GLM52_DECODE_BUCKETS:?}"
                )
            })?;
        ensure!(
            state.graph.is_captured(),
            "GLM5.2 bucket-{bucket} graph dump requested before pre-capture"
        );
        state
            .graph
            .dump_png(png_path, title)
            .with_context(|| format!("dump GLM5.2 rank-0 bucket-{bucket} decode CUDA Graph"))
    }

    /// The token embedding table — the DSpark draft's block embedding reuses
    /// it (the draft checkpoint's copy is byte-identical and not loaded).
    pub(crate) fn embed(&self) -> &DeviceMatrix {
        &self.embed
    }

    /// The lm_head — the DSpark draft's logits reuse it (same reuse contract
    /// as [`Self::embed`]).
    pub(crate) fn lm_head(&self) -> &DeviceMatrix {
        &self.lm_head
    }

    /// The per-request context cap this rank's cache arenas were built for.
    pub(crate) fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// The last step's aux-hidden capture buffer for `bucket` (`[bucket,
    /// 5 * GLM52_HIDDEN]`, row = step row). Valid until the next step in the
    /// same bucket overwrites it — the draft lane consumes it between steps.
    pub(crate) fn captured(&self, bucket: usize) -> Result<&CudaSlice<bf16>> {
        let state = self
            .buckets
            .iter()
            .find(|state| state.rows == bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 capture bucket {bucket} is not a member of {GLM52_DECODE_BUCKETS:?}"
                )
            })?;
        Ok(state
            .scratch
            .captured
            .as_ref()
            .context("GLM5.2 DSpark context capture is disabled")?
            .data())
    }

    /// The single page-granular arena this rank registers with the KV
    /// offload tier: pool block `b`'s page at `b * page_stride` holds EVERY
    /// layer's MLA slice, the full-indexer index-K slices, and the layer-78
    /// MTP committed mirrors — per-layer co-movement is structural, one
    /// save/load moves a block's whole page. Only the pool region is
    /// registered: the EP MTP scratch pages past `pool_blocks` hold
    /// unverified proposal KV and never transfer, and the pad tail
    /// (stride − content) never moves.
    pub(crate) fn kv_arenas(&self, stream: &CudaStream) -> Result<Vec<ArenaSpec>> {
        ensure!(
            self.kv_slab.num_blocks >= self.pool_blocks
                && self.kv_slab.page_stride == GLM52_KV_PAGE_STRIDE,
            "GLM5.2 KV slab geometry drifted from the pool: {} pages x {} stride vs {} pool blocks",
            self.kv_slab.num_blocks,
            self.kv_slab.page_stride,
            self.pool_blocks,
        );
        let (base_ptr, _sync) = self.kv_slab.slab.device_ptr(stream);
        Ok(vec![ArenaSpec {
            name: "glm52.page".to_owned(),
            base_device_ptr: base_ptr,
            size_bytes: self.pool_blocks * GLM52_KV_PAGE_STRIDE,
            num_blocks: self.pool_blocks,
            segment_bytes: GLM52_KV_PAGE_CONTENT_BYTES,
            segments: 1,
            kv_stride_bytes: 0,
            block_stride_bytes: GLM52_KV_PAGE_STRIDE,
        }])
    }

    /// Native P/D boundary restore: one whole-page D2D from the restored
    /// shared page into the request's own page — every layer's slices (MTP
    /// mirrors included) move together by construction.
    fn copy_kv_page(&mut self, ctx: &DeviceContext, src: usize, dst: usize) -> Result<()> {
        ensure!(
            src < self.pool_blocks && dst < self.pool_blocks,
            "GLM5.2 boundary copy outside the pool: {src} -> {dst}, {} pool blocks",
            self.pool_blocks
        );
        glm52_copy_page_content(&ctx.stream, &mut self.kv_slab, src, dst)
    }

    /// Phase 1 of the two-phase build: everything NOT sized by the pool
    /// block count — weights, rope tables, per-bucket sched/scratch/logits/
    /// block tables (all len-scaled via [`glm52_table_width`]), and the
    /// chunk-scaled prefill workspace. Launch measures each rank's free VRAM
    /// after this returns and decides the pool size from it;
    /// [`Self::finish_kv`] then allocates every pool-scaled slab.
    pub(crate) fn build_fixed(
        ctx: &DeviceContext,
        w: &mut Glm52RankGpuWeights,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        attn_shard: Option<usize>,
        drafter: &crate::Glm52Drafter,
        prefill_chunk_size: Option<usize>,
    ) -> Result<Glm52RankModelFixed> {
        ensure!(
            moe_topo.uses_tensor_replicated_moe() == attn_shard.is_some(),
            "GLM5.2 attention-TP shard must ride a tensor-replicated topology (topo {moe_topo:?}, \
             shard {attn_shard:?})"
        );
        ensure!(
            max_model_len > 0 && max_model_len.is_multiple_of(GLM52_MODEL_LEN_ALIGN),
            "GLM5.2 max_model_len {max_model_len} must be a positive multiple of \
             {GLM52_MODEL_LEN_ALIGN} (the FlashMLA page / index-K block size)"
        );
        let batch = GLM52_MAX_STEP_ROWS;
        let mla_heads = if attn_shard.is_some() {
            crate::config::GLM52_HEADS / moe_topo.device_count()
        } else {
            crate::config::GLM52_HEADS
        };
        let mla_backend = glm52_select_mla_backend(mla_heads)?;
        let mla_cache_bytes_per_token =
            glm52_persistent_mla_bytes_per_token(prefill_chunk_size.is_some(), mla_backend);
        let indexer_slices = (0..GLM52_LAYERS)
            .filter(|&layer| glm52_layer_has_full_indexer(layer))
            .count();
        log::info!(
            "GLM5.2 KV cache: topology={moe_topo:?} backend={mla_backend:?} page-first slab \
             page_tokens={} page_stride={GLM52_KV_PAGE_STRIDE} \
             page_content={GLM52_KV_PAGE_CONTENT_BYTES} mla_slices={} x {GLM52_KV_PAGE_MLA_BYTES} \
             index_k_slices={indexer_slices} x {GLM52_KV_PAGE_IDXK_BYTES} \
             (fp8[64,128]+f32[64]) + L78 mirrors",
            GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
            GLM52_LAYERS,
        );
        log::info!(
            "GLM5.2 MLA execution: backend={:?} ({} heads/rank, persistent {} bytes/cache token)",
            mla_backend,
            mla_heads,
            mla_cache_bytes_per_token
        );
        let num_sm_parts = if attn_shard.is_some() {
            1
        } else {
            glm52_flashmla_sparse_decode_num_sm_parts()?
        };
        // The pool block count is decided by the measured launch fill AFTER
        // this build; the per-bucket sched/scratch below carry the count only
        // as launch metadata (nothing they allocate scales with it), so they
        // are built against a 1-block placeholder that `finish_kv` rebinds.
        // The layer offset stays 0 in the bucket-shared contract: one plan
        // serves all 78 layers, and the attend applies each layer's slab
        // offset per launch.
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: batch,
            num_blocks: 1,
            kv_layer_offset_bytes: 0,
            kv_block_stride_bytes: GLM52_KV_PAGE_STRIDE,
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts,
            sm_scale: GLM52_SM_SCALE,
        };
        let (table_width, index_cache_layout) = glm52_index_cache_layout(max_model_len, 1);

        let mut layers = Vec::with_capacity(GLM52_LAYERS);
        for layer in 0..GLM52_LAYERS {
            layers.push(
                build::build_decoder_layer(ctx, w, layer, moe_topo, attn_shard)
                    .with_context(|| format!("build GLM5.2 decoder layer {layer}"))?,
            );
        }
        let mtp = drafter
            .is_mtp()
            .then(|| Glm52NativeMtp::build_fixed(ctx, w, max_model_len, moe_topo, attn_shard))
            .transpose()?;

        let embed_raw = w.take_tensor("model.embed_tokens.weight")?;
        let lm_head_raw = w.take_tensor("lm_head.weight")?;
        ensure!(
            embed_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2
                && lm_head_raw.len() == GLM52_VOCAB * GLM52_HIDDEN * 2,
            "GLM5.2 embed/lm_head byte lengths unexpected"
        );
        let embed = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, embed_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let lm_head = DeviceMatrix {
            data: retype_owned::<bf16>(&ctx.stream, lm_head_raw)?,
            rows: GLM52_VOCAB,
            cols: GLM52_HIDDEN,
        };
        let (decode_lm_head, decode_vocab_start) = if let Some(rank) = attn_shard {
            let ranks = moe_topo.device_count();
            ensure!(
                rank < ranks && GLM52_VOCAB.is_multiple_of(ranks),
                "GLM5.2 vocab TP shard {rank}/{ranks} cannot partition {} rows",
                GLM52_VOCAB
            );
            let rows = GLM52_VOCAB / ranks;
            let start = rank * rows;
            let mut data = ctx.stream.alloc_zeros::<bf16>(rows * GLM52_HIDDEN)?;
            ctx.stream.memcpy_dtod(
                &lm_head
                    .data
                    .slice(start * GLM52_HIDDEN..(start + rows) * GLM52_HIDDEN),
                &mut data,
            )?;
            (
                Some(DeviceMatrix {
                    data,
                    rows,
                    cols: GLM52_HIDDEN,
                }),
                start,
            )
        } else {
            (None, 0)
        };
        let final_norm = build::take_bf16_vec(ctx, w, "model.norm.weight", GLM52_HIDDEN)?;
        w.ensure_consumed()?;

        let mut cos_host = Vec::with_capacity(max_model_len * GLM52_ROPE_HALF);
        let mut sin_host = Vec::with_capacity(max_model_len * GLM52_ROPE_HALF);
        for position in 0..max_model_len {
            let (cos_row, sin_row) = rope_tables(position);
            cos_host.extend_from_slice(&cos_row);
            sin_host.extend_from_slice(&sin_row);
        }
        let mut cos_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(max_model_len * GLM52_ROPE_HALF)?;
        let mut sin_table_data = ctx
            .stream
            .alloc_zeros::<bf16>(max_model_len * GLM52_ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos_table_data)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin_table_data)?;

        // One Glm52BucketState per decode bucket: batch-`rows` contracts
        // (num_blocks is cache geometry, not batch, so it carries over),
        // plans, scratch, and a zeroed block table (never read before the
        // first step prologue uploads the engine's page rows).
        // Attention-TP scratch follows the head shard and selected MLA cache
        // layout; both were fixed before the per-layer arenas were allocated.
        let mut buckets = Vec::with_capacity(if prefill_chunk_size.is_some() {
            0
        } else {
            GLM52_DECODE_BUCKETS.len()
        });
        for rows in prefill_chunk_size
            .is_none()
            .then_some(GLM52_DECODE_BUCKETS)
            .into_iter()
            .flatten()
        {
            let contract_rows = Glm52FlashMlaSparseDecode {
                batch_size: rows,
                ..contract
            };
            let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
                rows,
                index_cache_layout,
                table_width,
                NUM_SMS,
                max_model_len,
            );
            let bucket_table = ctx.stream.alloc_zeros::<i32>(rows * table_width)?;
            buckets.push(Glm52BucketState {
                rows,
                sched: Glm52MlaSchedMetadata::new_for_backend(
                    ctx,
                    contract_rows,
                    mla_heads,
                    mla_backend,
                )?,
                scratch: Glm52DecodeScratch::new_for_backend(
                    ctx,
                    &contract_rows,
                    mqa_shape,
                    mla_heads,
                    mla_backend,
                    drafter.is_dspark(),
                )?,
                graph: CudaGraphState::new(),
                block_table: bucket_table,
                // Read only after a D2H lands in them (the write-combined
                // pages start uninitialized).
                argmax_values_host: unsafe { ctx.ctx.alloc_pinned::<bf16>(rows)? },
                argmax_indices_host: unsafe { ctx.ctx.alloc_pinned::<i32>(rows)? },
            });
        }
        // Crash-early pre-flight: launch the batched weight-only GEMV once
        // per bucket, so a GLM52_DECODE_BUCKETS entry without a matching CUDA
        // template instantiation (`kBatchedGemvBatch*` in glm52_moe_gemv.cu)
        // fails at startup — not on the first mid-serving step that reaches
        // that bucket (graphs are lazily captured; nothing else exercises a
        // bucket before real traffic does). Zeroed dummy operands in the
        // smallest whitelisted linear shape (indexer wk, n=128 k=6144).
        if prefill_chunk_size.is_none() {
            let (n, k) = (128usize, 6144usize);
            let weight = ctx.stream.alloc_zeros::<u8>(n * k)?;
            let scale = ctx
                .stream
                .alloc_zeros::<u8>(n.div_ceil(128) * k.div_ceil(128) * 4)?;
            let activation = ctx.stream.alloc_zeros::<bf16>(GLM52_MAX_STEP_ROWS * k)?;
            let mut out = ctx.stream.alloc_zeros::<bf16>(GLM52_MAX_STEP_ROWS * n)?;
            let mut gemv_partial = ctx
                .stream
                .alloc_zeros::<f32>(GLM52_MAX_STEP_ROWS * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?;
            for rows in GLM52_DECODE_BUCKETS {
                glm52_fp8_weight_only_gemv_launch(
                    ctx,
                    rows,
                    n,
                    k,
                    &activation,
                    &weight,
                    &scale,
                    Some(&mut gemv_partial),
                    &mut out,
                )
                .with_context(|| {
                    format!(
                        "GLM5.2 decode bucket {rows} has no batched GEMV instantiation \
                         (GLM52_DECODE_BUCKETS drifted from kBatchedGemvBatch* in glm52_moe_gemv.cu)"
                    )
                })?;
            }
        }

        let prefill = prefill_chunk_size
            .map(|chunk_rows| {
                let topology = match moe_topo {
                    crate::Glm52MoeTopo::Tp4 => pegainfer_kernels::ops::Glm52TpTopology::Tp4,
                    other => anyhow::bail!(
                        "GLM5.2 prefill-only execution requires a TP topology, got {other:?}"
                    ),
                };
                Glm52TpPrefillExecutor::new(ctx, table_width, chunk_rows, topology)
            })
            .transpose()?;

        Ok(Glm52RankModelFixed {
            layers,
            mtp,
            embed,
            final_norm,
            lm_head,
            decode_lm_head,
            decode_vocab_start,
            buckets,
            table_width,
            max_model_len,
            mla_backend,
            mla_cache_bytes_per_token,
            ep_ranks: moe_topo.expected_ep_size(),
            slot_mapping: ctx.stream.alloc_zeros::<i64>(batch)?,
            seq_lens: ctx.stream.alloc_zeros::<i32>(batch)?,
            cos_table: DeviceMatrix {
                data: cos_table_data,
                rows: max_model_len,
                cols: GLM52_ROPE_HALF,
            },
            sin_table: DeviceMatrix {
                data: sin_table_data,
                rows: max_model_len,
                cols: GLM52_ROPE_HALF,
            },
            positions: ctx.stream.alloc_zeros::<u32>(batch)?,
            cos: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            sin: ctx.stream.alloc_zeros::<bf16>(batch * GLM52_ROPE_HALF)?,
            token_ids: ctx.stream.alloc_zeros::<u32>(batch)?,
            sampling_scratch: Some(BatchSamplingScratch::new(ctx, batch, GLM52_VOCAB)?),
            prefill,
        })
    }

    /// Phase 2 of the two-phase build: allocate the pool-scaled slabs for
    /// the launch-decided `pool_blocks` — the page-first KV slab (plus the
    /// EP MTP scratch pages past the pool region), the TP4 MTP dense caches,
    /// and the prefill unpacked-KV pool — and rebind the placeholder cache
    /// geometry the fixed buckets carry. The engine's `BlockPool` and the
    /// slab here MUST agree on this count: pool block ids index the slab
    /// pages directly.
    pub(crate) fn finish_kv(
        ctx: &DeviceContext,
        fixed: Glm52RankModelFixed,
        pool_blocks: usize,
    ) -> Result<Self> {
        let Glm52RankModelFixed {
            layers,
            mtp,
            embed,
            final_norm,
            lm_head,
            decode_lm_head,
            decode_vocab_start,
            mut buckets,
            table_width,
            max_model_len,
            mla_backend,
            mla_cache_bytes_per_token,
            ep_ranks,
            slot_mapping,
            seq_lens,
            cos_table,
            sin_table,
            positions,
            cos,
            sin,
            token_ids,
            sampling_scratch,
            mut prefill,
        } = fixed;
        ensure!(pool_blocks > 0, "GLM5.2 KV pool must be non-empty");
        // The slab page holds fp8_ds_mla rows; a 576-byte FlashInfer
        // persistent layout has no slab home. TP4 attention runs
        // prefill-only (its persistent rows are the 656-byte wire format),
        // so the only config this rejects is the removed TP4 decode role.
        ensure!(
            mla_cache_bytes_per_token == GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
            "GLM5.2 page-first KV slab requires the {GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN}-byte \
             fp8_ds_mla persistent layout, got {mla_cache_bytes_per_token} bytes/token \
             (backend {mla_backend:?})"
        );
        let (layout_width, index_cache_layout) =
            glm52_index_cache_layout(max_model_len, pool_blocks);
        ensure!(
            layout_width == table_width,
            "GLM5.2 table width drifted between build_fixed and finish_kv"
        );
        for bucket in &mut buckets {
            bucket.sched.set_num_blocks(pool_blocks);
            bucket.scratch.idx.set_num_kv_blocks(pool_blocks);
        }
        let page = glm52_page_layout();
        // EP MTP proposal scratch pages live in the same slab past the pool
        // region; the TP4 MTP caches are dense side allocations and add no
        // slab pages.
        let scratch_blocks = mtp.as_ref().map_or(0, |mtp| {
            if mtp.slab_resident() {
                glm52_decode_slots() * mtp::MTP_SCRATCH_PAGES_PER_SLOT
            } else {
                0
            }
        });
        let slab_blocks = pool_blocks + scratch_blocks;
        let slab_bytes = slab_blocks
            .checked_mul(GLM52_KV_PAGE_STRIDE)
            .and_then(|bytes| bytes.checked_add(GLM52_KV_PAGE_CONTENT_BYTES))
            .context("GLM5.2 KV slab byte count overflow")?;
        let kv_slab = Glm52KvSlab {
            slab: ctx.stream.alloc_zeros::<u8>(slab_bytes)?,
            page_stride: GLM52_KV_PAGE_STRIDE,
            num_blocks: slab_blocks,
        };
        let mtp = mtp
            .map(|fixed| fixed.attach_cache(ctx, pool_blocks, page.mtp))
            .transpose()?;
        if let Some(prefill) = prefill.as_mut() {
            prefill.attach_kv_pool(
                ctx,
                pool_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
                index_cache_layout,
            )?;
        }
        Ok(Self {
            layers,
            kv_slab,
            caches: page.layers,
            mtp,
            embed,
            final_norm,
            lm_head,
            decode_lm_head,
            decode_vocab_start,
            buckets,
            table_width,
            max_model_len,
            pool_blocks,
            ep_ranks,
            slot_mapping,
            seq_lens,
            cos_table,
            sin_table,
            positions,
            cos,
            sin,
            token_ids,
            sampling_scratch,
            prefill,
            speculated: None,
            device_positions: [0; GLM52_MAX_STEP_ROWS],
        })
    }

    /// Whether this rank was built for prefill-only execution (drives the
    /// prefill NCCL all-reduce bring-up in SetupComm).
    pub(crate) fn is_prefill_only(&self) -> bool {
        self.prefill.is_some()
    }

    pub(crate) fn prefill_chunk(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        batch: &crate::runner::Glm52PrefillBatch,
        tp: Option<&mut Glm52MoeTpRank>,
    ) -> Result<crate::prefill_tp::Glm52PrefillOutput> {
        let executor = self
            .prefill
            .as_mut()
            .context("GLM5.2 rank was not built for prefill-only execution")?;
        let tp = tp.context("GLM5.2 TP4 prefill is missing its TP runtime")?;
        let sampling_scratch = self
            .sampling_scratch
            .as_mut()
            .context("GLM5.2 prefill sampling scratch is missing")?;
        let lm_head = self
            .decode_lm_head
            .as_ref()
            .context("GLM5.2 TP4 prefill is missing its vocabulary shard")?;
        let mtp = self.mtp.as_mut().map(Glm52NativeMtp::prefill_view);
        let mut output = executor.forward(
            ctx,
            batch,
            tp,
            Glm52TpPrefillModelView {
                layers: &self.layers,
                slab: &mut self.kv_slab,
                caches: &self.caches,
                embed: &self.embed,
                cos_table: &self.cos_table,
                sin_table: &self.sin_table,
                final_norm: &self.final_norm,
                shard_lm_head: lm_head,
                full_lm_head: &self.lm_head,
                vocab_start: self.decode_vocab_start,
                sampling_scratch,
                mtp,
            },
        )?;
        let Some(mtp) = self.mtp.as_mut() else {
            return Ok(output);
        };
        if batch.output_rows.is_empty() {
            return Ok(output);
        }
        let mut appends = Vec::with_capacity(batch.output_rows.len());
        let mut proposal_slots = Vec::with_capacity(batch.output_rows.len());
        let mut boundary = 0usize;
        for request in 0..batch.mtp_next_tokens.len() {
            if batch.mtp_next_tokens[request].is_some() {
                continue;
            }
            let end = batch.request_indptr[request + 1] as usize;
            ensure!(
                batch.output_rows.get(boundary).copied() == Some((end - 1) as u32),
                "GLM5.2 MTP boundary row order drifted from request ranges"
            );
            let block_start = batch.block_indptr[request] as usize;
            let block_end = batch.block_indptr[request + 1] as usize;
            let slot = batch.request_slots[request];
            appends.push(crate::runner::Glm52MtpAppend {
                target_row: boundary,
                slot,
                input_token: output.target_tokens[boundary],
                position: batch.positions[end - 1] as usize,
                pages: batch.block_ids[block_start..block_end].to_vec(),
            });
            proposal_slots.push(slot);
            boundary += 1;
        }
        ensure!(
            boundary == output.target_tokens.len(),
            "GLM5.2 MTP boundary metadata count {boundary} != target outputs {}",
            output.target_tokens.len()
        );
        mtp.reset_slots(&proposal_slots)?;
        mtp.resume_reset_slots(&proposal_slots, &appends)?;
        // The proposal rounds run the decode-bucket machinery, whose TP MoE
        // path bridges through the fixed GLM52_TP_TOKENS-row buffers — but
        // one prefill batch can complete more boundaries than that. Split
        // the batch into bridge-sized rounds; the surplus boundaries simply
        // ride the later rounds. TP-safe: all four executors walk the same
        // deterministic split, so their collective chains stay aligned.
        let mut drafts = Vec::with_capacity(appends.len());
        for start in (0..appends.len()).step_by(GLM52_TP_TOKENS) {
            let end = appends.len().min(start + GLM52_TP_TOKENS);
            let bucket = GLM52_DECODE_BUCKETS
                .into_iter()
                .find(|&bucket| bucket >= end - start)
                .context("GLM5.2 TP4 prefill proposal exceeds decode bucket capacity")?;
            let round = crate::runner::Glm52MtpRound {
                source_bucket: bucket,
                context_bucket: bucket,
                draft_bucket: bucket,
                resets: Vec::new(),
                appends: appends[start..end].to_vec(),
                proposal_slots: proposal_slots[start..end].to_vec(),
            };
            drafts.extend(mtp.propose(
                ctx,
                aux,
                None,
                Some(&mut *tp),
                &self.embed,
                &self.lm_head,
                &self.cos_table,
                &self.sin_table,
                &mut self.kv_slab,
                executor.mtp_target_boundary(),
                &round,
                Some(mtp::Glm52MtpProposalSeed {
                    previous: executor.mtp_proposal_boundary(),
                    draft1: &output.mtp_draft1[start..end],
                    rows_before: start,
                }),
            )?);
        }
        output.mtp_drafts = drafts;
        ensure!(
            output
                .mtp_drafts
                .iter()
                .zip(&output.mtp_draft1)
                .all(|(span, &draft1)| span[0] == draft1),
            "GLM5.2 TP4 large-M and proposal-loop draft-1 diverged"
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mtp_propose(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep: Option<&mut Glm52MoeEpState>,
        tp: Option<&mut Glm52MoeTpRank>,
        round: &crate::runner::Glm52MtpRound,
    ) -> Result<Vec<[u32; crate::mtp::GLM52_MTP_DRAFTS]>> {
        let source_index = self
            .buckets
            .iter()
            .position(|bucket| bucket.rows == round.source_bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 MTP source bucket {} is not in {GLM52_DECODE_BUCKETS:?}",
                    round.source_bucket
                )
            })?;
        // Official vLLM feeds MTP the target model return, which is after
        // final RMSNorm for GLM5.2. The pre-norm residual is not an
        // interchangeable MTP input even when target top-1 is unchanged.
        let target_final_normed = &self.buckets[source_index].scratch.final_normed;
        let mtp = self
            .mtp
            .as_mut()
            .context("GLM5.2 native MTP command reached a model without MTP weights")?;
        mtp.reset_slots(&round.resets)?;
        mtp.resume_reset_slots(&round.resets, &round.appends)?;
        mtp.propose(
            ctx,
            aux,
            ep,
            tp,
            &self.embed,
            &self.lm_head,
            &self.cos_table,
            &self.sin_table,
            &mut self.kv_slab,
            target_final_normed,
            round,
            None,
        )
    }

    /// One rank's step: feed `inputs[row]` = the `(token, position)` each
    /// forwarded row carries, return the next-token id per ROW (the fused
    /// greedy argmax, overwritten for the engine's `sampling` rows by a
    /// post-graph FlashInfer sampling pass — see [`Self::sample_rows_into`]).
    /// Enters 75 MoE collectives unconditionally — every other rank's engine
    /// is stepping concurrently with its OWN bucket (`shape.bucket` is
    /// rank-local): the collectives pair by entry count with rank-local row
    /// counts under the conservative protocol-max bound
    /// (`docs/models/glm52/free-running-dp.md` §2). Row `r` writes and reads
    /// KV through `kv`'s page row / slot mapping; a slot's span rows walk
    /// consecutive positions (see [`Glm52StepShape`]); padding rows' cache
    /// writes land in the pool's padding page, which nobody reads
    /// meaningfully.
    ///
    /// The step body (embed → 78 layers → lm_head → argmax) is captured into
    /// a CUDA graph on the first call in each (attention tier × bucket) shape
    /// and replayed afterwards: one graph launch instead of ~4155 kernel
    /// launches per rank per step. The prologue rewrites the device input
    /// buffers the captured kernels read (per-row rope rows, slots, seq_lens,
    /// tokens, and — for partial buckets — each forwarded row's block-table
    /// row), and the epilogue reads back the per-row argmax results — both
    /// outside the graph. Capture-time safety: stream capture records without
    /// executing, so a capturing rank does NOT enter the collectives — its
    /// peers' pairing entries simply wait until this rank's first replay of
    /// the shape executes them (pairing is by entry order, not step index;
    /// the ceiling is the ~100 s DeepEP device timeout against a capture
    /// window of tens of ms — already proven by the mid-serving tier-crossing
    /// capture).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_step(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: Option<&mut Glm52MoeEpState>,
        tp: Option<&mut Glm52MoeTpRank>,
        inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
        shape: Glm52StepShape,
        kv: &Glm52StepKv,
        flags: crate::runner::Glm52StepFlags,
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
    ) -> Result<[u32; GLM52_MAX_STEP_ROWS]> {
        assert!(
            self.prefill.is_none(),
            "GLM5.2 prefill-only execution entered decode_step"
        );
        // A launch-ahead speculation feeds this step's ARGMAX token to the
        // next step, so it can never coexist with a sampled row — the
        // engine withholds the lease while any non-greedy request is
        // active, and a violation here is a protocol bug.
        ensure!(
            sampling.is_empty() || (!flags.consume && !flags.lease),
            "GLM5.2 sampling rows cannot ride a launch-ahead step (the speculation feeds the \
             argmax token, not the sampled one)"
        );
        for &(src, dst) in &kv.boundary_copies {
            ensure!(
                !flags.consume,
                "GLM5.2 boundary copy on a consumed lease step (kernels already enqueued)"
            );
            self.copy_kv_page(ctx, src as usize, dst as usize)?;
        }
        let batch = shape.bucket;
        if flags.consume {
            // Launch-ahead fast path: the engine says this step IS the
            // replay every rank speculatively enqueued last step. That claim
            // is global — a speculative replay is a full set of collectives,
            // so ranks must consume together or not at all. Any mismatch is
            // a protocol bug; failing the step beats a silent fallback that
            // would desync the collective pairing (measured as the ~100 s
            // DeepEP device-timeout trap).
            let speculated = self.speculated.take().context(
                "GLM5.2 launch-ahead desync: the engine consumed a speculation this rank \
                 never enqueued",
            )?;
            ensure!(
                speculated.bucket == batch
                    && speculated.active_rows == shape.active_rows
                    && speculated.slots[..batch] == shape.slots[..batch]
                    && speculated.expect[..batch] == inputs[..batch],
                "GLM5.2 launch-ahead desync: consumed speculation (bucket {}, slots {:?}, expect \
                 {:?}) does not match the step (bucket {batch}, slots {:?}, inputs {:?})",
                speculated.bucket,
                &speculated.slots[..speculated.bucket],
                &speculated.expect[..speculated.bucket],
                &shape.slots[..batch],
                &inputs[..batch],
            );
        } else {
            // Any stale speculation was enqueued by EVERY rank (the lease is
            // a global grant), so the stale replay's collectives pair up and
            // it degrades to a harmless recompute the prologue overwrites.
            self.speculated = None;
            self.decode_step_prologue_and_replay(
                ctx,
                aux,
                ep8,
                tp,
                inputs,
                shape,
                kv,
                flags.eager,
            )?;
        }
        let mut outputs = self.decode_step_harvest(ctx, inputs, shape, flags.lease)?;
        self.sample_rows_into(ctx, shape, sampling, seed, &mut outputs)?;
        Ok(outputs)
    }

    /// Overwrite the sampled rows' tokens: a non-greedy request's committed
    /// row takes a FlashInfer temperature/top-k/top-p/min_p pass over the
    /// step's logits instead of the fused argmax. Unseeded rows ride one
    /// batched call under the step seed; a seeded row is its own single-row
    /// call with `mix_seed(request_seed, step)` — the same replayable-stream
    /// contract as `pegainfer_sample::select_batch`.
    fn sample_rows_into(
        &mut self,
        ctx: &DeviceContext,
        shape: Glm52StepShape,
        sampling: &[crate::runner::Glm52RowSample],
        seed: u64,
        outputs: &mut [u32; GLM52_MAX_STEP_ROWS],
    ) -> Result<()> {
        if sampling.is_empty() {
            return Ok(());
        }
        let sampling_scratch = self
            .sampling_scratch
            .as_mut()
            .context("GLM5.2 sampling is unavailable in prefill-only mode")?;
        for pair in sampling.windows(2) {
            ensure!(
                pair[0].row < pair[1].row,
                "GLM5.2 sampling rows must be strictly ascending: {sampling:?}"
            );
        }
        for s in sampling {
            ensure!(
                s.row < shape.active_rows,
                "GLM5.2 sampling row {} outside the step's {} active rows",
                s.row,
                shape.active_rows
            );
            ensure!(
                !effectively_greedy(&s.params, GLM52_VOCAB),
                "GLM5.2 effectively-greedy row {} routed to the sampler (engine bug)",
                s.row
            );
        }
        let bucket = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.rows == shape.bucket)
            .expect("decode_step validated the bucket");
        // TP greedy decode writes compact shard logits into the shared
        // buffer. Sampling needs the full distribution, so only sampled
        // steps pay one eager full-head recompute after the graph.
        if self.decode_lm_head.is_some() {
            glm52_lm_head_into(
                ctx,
                &bucket.scratch.final_normed,
                &self.lm_head,
                &mut bucket.scratch.logits,
            )?;
        }
        let logits = HiddenStatesRef {
            data: bucket.scratch.logits.data(),
            hidden_dim: GLM52_VOCAB,
            seq_len: shape.bucket,
        };
        let as_row = |s: &crate::runner::Glm52RowSample| BatchSamplingRow {
            row: s.row,
            temperature: s.params.temperature,
            top_k: s.params.top_k,
            top_p: s.params.top_p,
            min_p: s.params.min_p,
        };
        let unseeded: Vec<BatchSamplingRow> = sampling
            .iter()
            .filter(|s| s.params.seed.is_none())
            .map(as_row)
            .collect();
        if !unseeded.is_empty() {
            let tokens = gpu_sample_batch_into(ctx, logits, &unseeded, seed, sampling_scratch)?;
            for (row, token) in unseeded.iter().zip(tokens) {
                outputs[row.row] = token;
            }
        }
        for s in sampling {
            let Some(request_seed) = s.params.seed else {
                continue;
            };
            let tokens = gpu_sample_batch_into(
                ctx,
                logits,
                &[as_row(s)],
                mix_seed(request_seed, s.step),
                sampling_scratch,
            )?;
            outputs[s.row] = tokens[0];
        }
        Ok(())
    }

    /// Test probe (free-running gate 4): D2H the last routed MoE layer's
    /// top-k output for one bucket's rows. The decode scratch is shared by
    /// every layer, so this is the layer-77 routing — a byte-constancy
    /// witness that transitively covers the whole step upstream of it
    /// (indexer seq_len=1, fp8 quant, attention, all 74 earlier routers).
    #[cfg(test)]
    pub(crate) fn probe_step_route(
        &self,
        ctx: &DeviceContext,
        bucket_rows: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let bucket = self
            .buckets
            .iter()
            .find(|bucket| bucket.rows == bucket_rows)
            .with_context(|| format!("GLM5.2 route probe: unknown bucket {bucket_rows}"))?;
        let rows = bucket_rows * crate::config::GLM52_TOPK;
        let idx = ctx
            .stream
            .clone_dtoh(&bucket.scratch.router.route.topk_idx)?;
        let weight = ctx
            .stream
            .clone_dtoh(&bucket.scratch.router.route.topk_weight)?;
        Ok((
            idx[..rows.min(idx.len())].to_vec(),
            weight[..rows.min(weight.len())]
                .iter()
                .map(|w| w.to_bits())
                .collect(),
        ))
    }

    /// The non-leased step path: validate the shape, rewrite every per-step
    /// device input buffer from the engine's `inputs`, and run (or lazily
    /// capture) the whole-step graph for the step's bucket × tier.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_prologue_and_replay(
        &mut self,
        ctx: &DeviceContext,
        aux: &DeviceContext,
        ep8: Option<&mut Glm52MoeEpState>,
        tp: Option<&mut Glm52MoeTpRank>,
        inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
        shape: Glm52StepShape,
        kv: &Glm52StepKv,
        eager: bool,
    ) -> Result<()> {
        // The bucket state's `rows` is the lookup key — an unknown bucket is
        // an engine bug and fails the step before touching the GPU.
        let bucket = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.rows == shape.bucket)
            .with_context(|| {
                format!(
                    "GLM5.2 step bucket {} is not a member of {GLM52_DECODE_BUCKETS:?}",
                    shape.bucket
                )
            })?;
        let batch = shape.bucket;
        // A slot's rows must form ONE contiguous run of consecutive
        // positions: a gap would leave positions the later rows attend to
        // unwritten this step (stale data from whatever request last held the
        // slot), and a second run would re-enter a region the first already
        // wrote. Single-row slots are the trivial run.
        // Only real rows carry span semantics; padding rows (>= active_rows)
        // reuse slot id 0 with padding inputs and are exempt (#812).
        let mut slot_last_row = [None::<usize>; GLM52_MAX_BATCH_PER_RANK];
        for row in 0..shape.active_rows {
            let slot = shape.slots[row] as usize;
            ensure!(
                slot < GLM52_MAX_BATCH_PER_RANK,
                "GLM5.2 step row {row} slot {slot} out of range in {:?}",
                &shape.slots[..batch]
            );
            match slot_last_row[slot] {
                None => {}
                Some(last) => {
                    ensure!(
                        last + 1 == row && inputs[last].1 + 1 == inputs[row].1,
                        "GLM5.2 step slot {slot} span is not one contiguous run of \
                         consecutive positions: rows {:?}, positions {:?}",
                        &shape.slots[..batch],
                        inputs[..batch].iter().map(|i| i.1).collect::<Vec<_>>()
                    );
                }
            }
            slot_last_row[slot] = Some(row);
        }
        ensure!(
            kv.pages.len() == batch * self.table_width,
            "GLM5.2 step KV pages {} != bucket {batch} x table width {}",
            kv.pages.len(),
            self.table_width
        );
        let mut tokens_host = [0u32; GLM52_MAX_STEP_ROWS];
        let mut positions_host = [0u32; GLM52_MAX_STEP_ROWS];
        let mut seq_lens_host = [0i32; GLM52_MAX_STEP_ROWS];
        for row in 0..batch {
            let slot = shape.slots[row] as usize;
            let (token, position) = inputs[row];
            ensure!(
                position < self.max_model_len,
                "GLM5.2 slot {slot} position {position} exceeds the model-length cap {}",
                self.max_model_len
            );
            // The engine's page row must place this row's write slot
            // inside the page covering its position — a drifted slot mapping
            // would write one row's KV into another request's page.
            let page =
                kv.pages[row * self.table_width + position / GLM52_FLASHMLA_SPARSE_PAGE_SIZE];
            let expect = page as i64 * GLM52_FLASHMLA_SPARSE_PAGE_SIZE as i64
                + (position % GLM52_FLASHMLA_SPARSE_PAGE_SIZE) as i64;
            ensure!(
                kv.slot_mapping[row] == expect,
                "GLM5.2 row {row} slot mapping {} does not match page {page} at position \
                 {position} (expect {expect})",
                kv.slot_mapping[row]
            );
            tokens_host[row] = token;
            positions_host[row] = position as u32;
            seq_lens_host[row] = (position + 1) as i32;
        }
        ctx.stream.memcpy_htod(&tokens_host, &mut self.token_ids)?;
        ctx.stream
            .memcpy_htod(&positions_host, &mut self.positions)?;
        ctx.stream
            .memcpy_htod(&kv.slot_mapping, &mut self.slot_mapping)?;
        ctx.stream.memcpy_htod(&seq_lens_host, &mut self.seq_lens)?;
        for (dst, &(_, position)) in self.device_positions.iter_mut().zip(&inputs[..batch]) {
            *dst = position;
        }
        // Gather each row's rotary table row (a bit-exact row copy).
        embedding_rows_into(ctx, &self.cos_table, &self.positions, batch, &mut self.cos)?;
        embedding_rows_into(ctx, &self.sin_table, &self.positions, batch, &mut self.sin)?;
        // Upload the step's page rows into the bucket's device block table —
        // device data, so the captured graphs replay against whichever pool
        // pages hold the requests (span rows repeat their slot's row, padding
        // rows ride the padding page).
        ctx.stream
            .memcpy_htod(&kv.pages[..], &mut bucket.block_table)?;
        // The bucket state selected above carries the plan, scratch, graph,
        // and block table together — one coherent shape.
        let step = Glm52DecodeStep {
            mla_cos: &self.cos,
            mla_sin: &self.sin,
            idx_cos: &self.cos,
            idx_sin: &self.sin,
            mla_sched: &bucket.sched,
            slot_mapping: &self.slot_mapping,
            block_table: &bucket.block_table,
            seq_lens: &self.seq_lens,
        };
        // The MoE collectives take this rank's real row count plus a GEMM
        // tile bound that only has to be conservative — every rank passes the
        // protocol max (`ep_ranks × GLM52_MAX_STEP_ROWS`) so the bound
        // never depends on other ranks' buckets. Measured at zero cost vs the
        // tight bound (free-running gate 3, `docs/models/glm52/
        // free-running-dp.md` §8); the recv-side truth comes from the device
        // expert counts, not this value. (`ep_ranks` is 1 on
        // tensor-replicated topologies, where the value is never consumed.)
        let global_tokens = self.ep_ranks * GLM52_MAX_STEP_ROWS;

        let s = &mut bucket.scratch;
        let decode_lm_head = self.decode_lm_head.as_ref().unwrap_or(&self.lm_head);
        if eager {
            return run_step_body(
                ctx,
                aux,
                ep8,
                tp,
                &self.layers,
                &mut self.kv_slab,
                &self.caches,
                &self.embed,
                &self.final_norm,
                decode_lm_head,
                self.decode_vocab_start,
                &self.token_ids,
                &step,
                s,
                global_tokens,
            );
        }
        let mut graph = std::mem::take(&mut bucket.graph);
        let result = graph.run_or_capture(ctx, || {
            run_step_body(
                ctx,
                aux,
                ep8,
                tp,
                &self.layers,
                &mut self.kv_slab,
                &self.caches,
                &self.embed,
                &self.final_norm,
                decode_lm_head,
                self.decode_vocab_start,
                &self.token_ids,
                &step,
                s,
                global_tokens,
            )
        });
        bucket.graph = graph;
        result
    }
}

#[cfg(test)]
mod cache_layout_tests {
    use super::*;
    use crate::mla_decode::Glm52MlaBackend;

    #[test]
    fn tp4_prefill_matches_ep_decode_persistent_mla_layout() {
        let tp4_prefill =
            glm52_persistent_mla_bytes_per_token(true, Glm52MlaBackend::FlashInferFp8);
        let ep_decode = glm52_persistent_mla_bytes_per_token(false, Glm52MlaBackend::FlashMlaFp8Ds);

        assert_eq!(tp4_prefill, 656);
        assert_eq!(tp4_prefill, ep_decode);
        assert_eq!(GLM52_KV_PAGE_MLA_BYTES, 41_984);
        assert_eq!(GLM52_KV_PAGE_IDXK_BYTES, 8_448);
    }

    #[test]
    fn page_layout_matches_the_wire_constants() {
        let layout = glm52_page_layout();
        assert_eq!(GLM52_KV_PAGE_CONTENT_BYTES, 3_502_592);
        assert_eq!(GLM52_KV_PAGE_STRIDE, 3_503_040);
        assert_eq!(
            GLM52_KV_PAGE_STRIDE,
            GLM52_KV_PAGE_CONTENT_BYTES.next_multiple_of(GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN)
        );
        assert_eq!(layout.layers.len(), GLM52_LAYERS);
        assert_eq!(
            layout
                .layers
                .iter()
                .filter(|caches| caches.index_k_offset.is_some())
                .count(),
            21
        );
        // Slices in declaration order, each 16-byte aligned, none crossing
        // into the next.
        let mut expected = 0usize;
        for caches in layout.layers.iter().chain([&layout.mtp]) {
            assert_eq!(caches.mla_offset, expected);
            assert_eq!(caches.mla_offset % 16, 0);
            expected += GLM52_KV_PAGE_MLA_BYTES;
            if let Some(index_k_offset) = caches.index_k_offset {
                assert_eq!(index_k_offset, expected);
                assert_eq!(index_k_offset % 16, 0);
                expected += GLM52_KV_PAGE_IDXK_BYTES;
            }
        }
        assert_eq!(expected, GLM52_KV_PAGE_CONTENT_BYTES);
        assert_eq!(layout.layers[0].mla_offset, 0);
        assert_eq!(layout.layers[0].index_k_offset, Some(41_984));
        assert_eq!(
            layout.mtp.index_k_offset,
            Some(GLM52_KV_PAGE_CONTENT_BYTES - GLM52_KV_PAGE_IDXK_BYTES)
        );
    }
}

/// Single-layer slab for oracle/unit paths: one page = `[MLA slice |
/// index-K slice]`, stride padded to the 656-byte cache token row the
/// FlashMLA TMA requires, with the same conservative-extent tail slack as
/// the production slab.
#[cfg(test)]
pub(crate) fn glm52_test_layer_slab(
    ctx: &DeviceContext,
    num_blocks: usize,
) -> Result<(Glm52KvSlab, Glm52LayerCaches)> {
    let content = GLM52_KV_PAGE_MLA_BYTES + GLM52_KV_PAGE_IDXK_BYTES;
    let page_stride = content.next_multiple_of(GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN);
    Ok((
        Glm52KvSlab {
            slab: ctx
                .stream
                .alloc_zeros::<u8>(num_blocks * page_stride + content)?,
            page_stride,
            num_blocks,
        },
        Glm52LayerCaches {
            mla_offset: 0,
            index_k_offset: Some(GLM52_KV_PAGE_MLA_BYTES),
        },
    ))
}

/// One whole-page D2D within a KV slab: the page CONTENT moves, the pad tail
/// does not. `src` and `dst` never overlap — the destination is always a
/// distinct page.
fn glm52_copy_page_content(
    stream: &Arc<CudaStream>,
    slab: &mut Glm52KvSlab,
    src: usize,
    dst: usize,
) -> Result<()> {
    copy_strided_block(
        stream,
        &mut slab.slab,
        0,
        slab.page_stride,
        GLM52_KV_PAGE_CONTENT_BYTES,
        src,
        dst,
    )
}

/// One block-granular D2D within a strided arena region: block `b` occupies
/// `copy_bytes` at `region_offset + b * block_stride`.
fn copy_strided_block(
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<u8>,
    region_offset: usize,
    block_stride: usize,
    copy_bytes: usize,
    src: usize,
    dst: usize,
) -> Result<()> {
    ensure!(src != dst, "arena block copy onto itself: page {src}");
    ensure!(
        copy_bytes <= block_stride,
        "arena block copy of {copy_bytes} bytes exceeds the {block_stride}-byte stride"
    );
    let start = |block: usize| region_offset + block * block_stride;
    let split = start(src.max(dst));
    let (mut low, mut high) = arena.split_at_mut(split);
    if src < dst {
        let src = low.slice(start(src)..start(src) + copy_bytes);
        let mut dst = high.slice_mut(0..copy_bytes);
        stream.memcpy_dtod(&src, &mut dst)?;
    } else {
        let src = high.slice(0..copy_bytes);
        let mut dst = low.slice_mut(start(dst)..start(dst) + copy_bytes);
        stream.memcpy_dtod(&src, &mut dst)?;
    }
    Ok(())
}
