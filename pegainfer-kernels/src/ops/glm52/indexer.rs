use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const GLM52_INDEXER_HEAD_DIM: usize = 128;
const GLM52_INDEXER_QUANT_BLOCK_SIZE: usize = 128;
const GLM52_INDEXER_SCALE_BYTES_PER_TOKEN: usize = 4;
pub const GLM52_INDEXER_TOPK: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheLayout {
    pub cache_blocks: usize,
    pub cache_block_size: usize,
    /// Byte offset of this layer's first index-K page inside the cache buffer
    /// (0 for a dedicated per-layer arena; the page-first slab passes the
    /// layer's slab offset).
    pub cache_layer_offset_bytes: usize,
    pub cache_block_stride_bytes: usize,
}

impl Glm52IndexerCacheLayout {
    fn min_block_stride_bytes(self) -> usize {
        self.cache_block_size * (GLM52_INDEXER_HEAD_DIM + GLM52_INDEXER_SCALE_BYTES_PER_TOKEN)
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.cache_blocks > 0,
            "GLM5.2 indexer cache_blocks must be positive"
        );
        ensure!(
            self.cache_block_size > 0,
            "GLM5.2 indexer cache_block_size must be positive"
        );
        ensure!(
            self.cache_block_stride_bytes >= self.min_block_stride_bytes(),
            "GLM5.2 indexer cache block stride too small: have {} bytes, need at least {}",
            self.cache_block_stride_bytes,
            self.min_block_stride_bytes()
        );
        Ok(())
    }

    pub fn min_cache_bytes(self) -> Result<usize> {
        self.validate()?;
        self.cache_blocks
            .checked_mul(self.cache_block_stride_bytes)
            .and_then(|extent| extent.checked_add(self.cache_layer_offset_bytes))
            .ok_or_else(|| anyhow!("GLM5.2 indexer cache byte size overflow: {self:?}"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerCacheInsert {
    pub tokens: usize,
    pub layout: Glm52IndexerCacheLayout,
}

impl Glm52IndexerCacheInsert {
    fn validate(self) -> Result<()> {
        ensure!(
            self.tokens > 0,
            "GLM5.2 indexer cache insert tokens must be positive"
        );
        self.layout.validate()
    }
}

pub fn glm52_indexer_k_quant_and_cache_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerCacheInsert,
    k: &CudaSlice<bf16>,
    indexer_cache: &mut CudaSlice<u8>,
    slot_mapping: &CudaSlice<i64>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        k.len() >= contract.tokens * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer K buffer too small: have {}, need {}",
        k.len(),
        contract.tokens * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        slot_mapping.len() >= contract.tokens,
        "GLM5.2 indexer slot_mapping too small: have {}, need {}",
        slot_mapping.len(),
        contract.tokens
    );
    let min_cache_bytes = contract.layout.min_cache_bytes()?;
    ensure!(
        indexer_cache.len() >= min_cache_bytes,
        "GLM5.2 indexer cache buffer too small: have {}, need {}",
        indexer_cache.len(),
        min_cache_bytes
    );

    let (k_ptr, _k_guard) = k.device_ptr(&ctx.stream);
    let (cache_base_ptr, _cache_guard) = indexer_cache.device_ptr_mut(&ctx.stream);
    let cache_ptr = cache_base_ptr + contract.layout.cache_layer_offset_bytes as u64;
    let (slot_ptr, _slot_guard) = slot_mapping.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_k_quant_and_cache_cuda(
            k_ptr as *const ffi::Half,
            cache_ptr as *mut u8,
            slot_ptr as *const i64,
            contract.tokens as i32,
            GLM52_INDEXER_HEAD_DIM as i32,
            GLM52_INDEXER_QUANT_BLOCK_SIZE as i32,
            contract.layout.cache_block_size as i32,
            contract.layout.cache_block_stride_bytes as i64,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer K quant/cache launch failed: {err}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerLocalTopKToSlots {
    pub num_tokens: usize,
    pub topk: usize,
    pub block_size: usize,
    pub block_table_cols: usize,
}

impl Glm52IndexerLocalTopKToSlots {
    fn validate(self) -> Result<()> {
        ensure!(
            self.num_tokens > 0,
            "GLM5.2 indexer local_topk_to_slots num_tokens must be positive"
        );
        ensure!(
            self.topk > 0,
            "GLM5.2 indexer local_topk_to_slots topk must be positive"
        );
        ensure!(
            self.block_size > 0,
            "GLM5.2 indexer local_topk_to_slots block_size must be positive"
        );
        ensure!(
            self.block_table_cols > 0,
            "GLM5.2 indexer local_topk_to_slots block_table_cols must be positive"
        );
        Ok(())
    }
}

/// Convert local top-k offsets (within a sequence's KV cache) to global KV
/// slot indices via the block table. Also writes `topk_lens` (valid slot
/// count per token). Ported from TokenSpeed Triton `local_topk_to_global_slots`.
///
/// - `local_topk_offsets`: `[num_tokens, topk]` int32, row-major.
/// - `seq_lens`: `[num_tokens]` int32, valid KV length per token (required,
///   matches vLLM `sparse_attn_indexer` which always passes `seq_lens` as
///   the top-k `lengths`).
/// - `block_table`: `[num_tokens, block_table_cols]` int32, row-major.
/// - `global_slots`: `[num_tokens, topk]` int32 output, `-1` for invalid slots.
/// - `topk_lens`: `[num_tokens]` int32 output, valid slot count per token.
pub fn glm52_indexer_local_topk_to_slots_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerLocalTopKToSlots,
    local_topk_offsets: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    block_table: &CudaSlice<i32>,
    global_slots: &mut CudaSlice<i32>,
    topk_lens: &mut CudaSlice<i32>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        local_topk_offsets.len() >= contract.num_tokens * contract.topk,
        "GLM5.2 indexer local_topk_to_slots offsets too small: have {}, need {}",
        local_topk_offsets.len(),
        contract.num_tokens * contract.topk
    );
    ensure!(
        seq_lens.len() >= contract.num_tokens,
        "GLM5.2 indexer local_topk_to_slots seq_lens too small: have {}, need {}",
        seq_lens.len(),
        contract.num_tokens
    );
    ensure!(
        block_table.len() >= contract.num_tokens * contract.block_table_cols,
        "GLM5.2 indexer local_topk_to_slots block_table too small: have {}, need {}",
        block_table.len(),
        contract.num_tokens * contract.block_table_cols
    );
    ensure!(
        global_slots.len() >= contract.num_tokens * contract.topk,
        "GLM5.2 indexer local_topk_to_slots global_slots too small: have {}, need {}",
        global_slots.len(),
        contract.num_tokens * contract.topk
    );
    ensure!(
        topk_lens.len() >= contract.num_tokens,
        "GLM5.2 indexer local_topk_to_slots topk_lens too small: have {}, need {}",
        topk_lens.len(),
        contract.num_tokens
    );

    let (offsets_ptr, _offsets_guard) = local_topk_offsets.device_ptr(&ctx.stream);
    let (seq_lens_ptr, _seq_lens_guard) = seq_lens.device_ptr(&ctx.stream);
    let (block_table_ptr, _block_table_guard) = block_table.device_ptr(&ctx.stream);
    let (global_slots_ptr, _global_slots_guard) = global_slots.device_ptr_mut(&ctx.stream);
    let (topk_lens_ptr, _topk_lens_guard) = topk_lens.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_local_topk_to_slots_cuda(
            global_slots_ptr as *mut i32,
            topk_lens_ptr as *mut i32,
            offsets_ptr as *const i32,
            contract.topk as i32,
            seq_lens_ptr as *const i32,
            block_table_ptr as *const i32,
            contract.block_table_cols as i32,
            contract.block_table_cols as i32,
            contract.block_size as i32,
            contract.topk as i32,
            contract.num_tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer local_topk_to_slots launch failed: {err}"))
}

/// Token bound baked into `glm52_min_gemv.cuh`'s `launch_tokens` switch.
pub const GLM52_MIN_GEMV_MAX_TOKENS: usize = 96;

/// The token counts `glm52_min_gemv.cuh` instantiates: 1..=8 plus the
/// verify-span decode bucket sizes (#812) — token counts always land on a
/// bucket member.
fn glm52_min_gemv_tokens_supported(tokens: usize) -> bool {
    (1..=8).contains(&tokens) || matches!(tokens, 16 | 32 | 48 | 64 | 96)
}

/// weights_proj min-latency GEMV: `out[t, h] = dot(hidden[t], weights[h])`,
/// bf16 in/out with fixed-order f32 accumulation. Replaces the cublas splitK
/// plan (GEMM + splitKreduce + workspace alloc/free per call).
pub fn glm52_indexer_weights_proj_launch(
    ctx: &DeviceContext,
    hidden: &CudaSlice<bf16>,
    weights_proj: &CudaSlice<bf16>,
    tokens: usize,
    heads: usize,
    hidden_dim: usize,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        glm52_min_gemv_tokens_supported(tokens),
        "GLM5.2 indexer weights_proj tokens {tokens} not instantiated \
         (1..=8 or a decode bucket size)"
    );
    ensure!(
        hidden.len() >= tokens * hidden_dim,
        "GLM5.2 indexer weights_proj hidden too small: have {}, need {}",
        hidden.len(),
        tokens * hidden_dim
    );
    ensure!(
        weights_proj.len() >= heads * hidden_dim,
        "GLM5.2 indexer weights_proj weights too small: have {}, need {}",
        weights_proj.len(),
        heads * hidden_dim
    );
    ensure!(
        out.len() >= tokens * heads,
        "GLM5.2 indexer weights_proj out too small: have {}, need {}",
        out.len(),
        tokens * heads
    );
    let (h_ptr, _g0) = hidden.device_ptr(&ctx.stream);
    let (w_ptr, _g1) = weights_proj.device_ptr(&ctx.stream);
    let (o_ptr, _g2) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_weights_proj_cuda(
            h_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            tokens as i32,
            heads as i32,
            hidden_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer weights_proj launch failed: {err}"))
}

/// Fold the per-head `weights_proj` output (bf16) with the per-head q quant
/// scale and the two attention scale constants into f32 weights for the
/// DeepGEMM MQA logits kernel: `out[h] = weights[h] * q_scale[h] *
/// softmax_scale * n_heads_scale` (left-to-right f32, bit-identical to the
/// retired host-side fold). Replaces two mid-step D2H readbacks + an H2D —
/// the DSA indexer chain stays on-device (CUDA-graph capturable).
pub fn glm52_indexer_weights_fold_launch(
    ctx: &DeviceContext,
    weights: &CudaSlice<bf16>,
    q_scale: &CudaSlice<f32>,
    softmax_scale: f32,
    n_heads_scale: f32,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    let heads = out.len();
    ensure!(heads > 0, "GLM5.2 indexer weights fold is empty");
    ensure!(
        weights.len() >= heads && q_scale.len() >= heads,
        "GLM5.2 indexer weights fold inputs too small: weights {}, q_scale {} (need {heads})",
        weights.len(),
        q_scale.len()
    );
    let (w_ptr, _g0) = weights.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q_scale.device_ptr(&ctx.stream);
    let (out_ptr, _g2) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_weights_fold_cuda(
            w_ptr as *const ffi::Half,
            q_ptr as *const f32,
            softmax_scale,
            n_heads_scale,
            out_ptr as *mut f32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer weights fold launch failed: {err}"))
}

/// Gather the paged fp8 indexer K cache into the compact unpaged layout
/// (`[total_kv, 128]` fp8 + `[total_kv]` f32 scales) the unpaged MQA logits
/// kernel consumes, and emit the position -> global-KV-slot LUT the top-k
/// slot conversion reads. Request `r` gathers `seq_lens[r]` tokens through
/// its `block_table` row into rows starting at `out_offsets[r]`.
#[allow(clippy::too_many_arguments)]
pub fn glm52_indexer_k_gather_launch(
    ctx: &DeviceContext,
    num_requests: usize,
    table_stride: usize,
    block_size: usize,
    layer_offset_bytes: usize,
    block_stride_bytes: usize,
    paged_cache: &CudaSlice<u8>,
    block_table: &impl DevicePtr<i32>,
    seq_lens: &impl DevicePtr<i32>,
    out_offsets: &impl DevicePtr<i32>,
    k_out: &mut CudaSlice<u8>,
    scale_out: &mut CudaSlice<f32>,
    slot_out: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        num_requests > 0
            && table_stride > 0
            && block_size > 0
            && block_stride_bytes >= block_size * (GLM52_INDEXER_HEAD_DIM + 4)
            && block_table.len() >= num_requests * table_stride
            && seq_lens.len() >= num_requests
            && out_offsets.len() >= num_requests,
        "GLM5.2 indexer K gather shape is invalid"
    );
    let (cache_base_ptr, _cache_guard) = paged_cache.device_ptr(&ctx.stream);
    let cache_ptr = cache_base_ptr + layer_offset_bytes as u64;
    let (table_ptr, _table_guard) = block_table.device_ptr(&ctx.stream);
    let (lens_ptr, _lens_guard) = seq_lens.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = out_offsets.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k_out.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scale_out.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_k_gather_cuda(
            cache_ptr as *const u8,
            table_ptr as *const i32,
            lens_ptr as *const i32,
            offsets_ptr as *const i32,
            k_ptr as *mut u8,
            scale_ptr as *mut f32,
            slot_ptr as *mut i32,
            num_requests as i32,
            table_stride as i32,
            block_size as i32,
            block_stride_bytes as i64,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer K gather launch failed: {err}"))
}

/// Convert per-query top-k offsets (relative to the query's request segment)
/// into global KV slots through the gather LUT; `-1`/out-of-range offsets
/// stay `-1` and `topk_lens` counts the valid picks (same semantics as the
/// paged block-table twin).
#[allow(clippy::too_many_arguments)]
pub fn glm52_indexer_topk_to_slots_lut_launch(
    ctx: &DeviceContext,
    rows: usize,
    topk: usize,
    topk_offsets: &impl DevicePtr<i32>,
    context_lens: &impl DevicePtr<i32>,
    cu_seqlen_ks: &impl DevicePtr<i32>,
    slot_lut: &impl DevicePtr<i32>,
    global_slots: &mut impl DevicePtrMut<i32>,
    topk_lens: &mut impl DevicePtrMut<i32>,
) -> Result<()> {
    ensure!(
        rows > 0
            && topk > 0
            && topk <= GLM52_INDEXER_TOPK
            && topk_offsets.len() >= rows * topk
            && context_lens.len() >= rows
            && cu_seqlen_ks.len() >= rows
            && slot_lut.len() > 0
            && global_slots.len() >= rows * topk
            && topk_lens.len() >= rows,
        "GLM5.2 indexer top-k LUT conversion buffers are invalid"
    );
    let (offsets_ptr, _offsets_guard) = topk_offsets.device_ptr(&ctx.stream);
    let (lens_ptr, _lens_guard) = context_lens.device_ptr(&ctx.stream);
    let (ks_ptr, _ks_guard) = cu_seqlen_ks.device_ptr(&ctx.stream);
    let (lut_ptr, _lut_guard) = slot_lut.device_ptr(&ctx.stream);
    let (slots_ptr, _slots_guard) = global_slots.device_ptr_mut(&ctx.stream);
    let (out_lens_ptr, _out_lens_guard) = topk_lens.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_topk_to_slots_lut_cuda(
            offsets_ptr as *const i32,
            lens_ptr as *const i32,
            ks_ptr as *const i32,
            lut_ptr as *const i32,
            slots_ptr as *mut i32,
            out_lens_ptr as *mut i32,
            rows as i32,
            topk as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer top-k LUT conversion launch failed: {err}"))
}
