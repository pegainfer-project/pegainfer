use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY: usize = 128;
const GLM52_FLASHMLA_SPARSE_HEADS: usize = 64;
const GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM: usize = 576;
const GLM52_FLASHMLA_SPARSE_V_HEAD_DIM: usize = 512;
pub const GLM52_FLASHMLA_SPARSE_PAGE_SIZE: usize = 64;
pub const GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN: usize = 656;
pub const GLM52_FLASHMLA_SPARSE_TOPK: usize = 2048;
const GLM52_FLASHMLA_SPARSE_TOPK_BLOCK: usize = 64;
const GLM52_FLASHMLA_SPARSE_SCHED_META_INTS: usize = 8;
const GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS: usize = 160;

#[allow(clippy::too_many_arguments)]
pub fn glm52_flashmla_sparse_prefill_launch(
    ctx: &DeviceContext,
    query_rows: usize,
    kv_rows: usize,
    topk: usize,
    sm_scale: f32,
    q: &impl DevicePtr<bf16>,
    kv: &CudaSlice<bf16>,
    topk_indices: &impl DevicePtr<i32>,
    topk_length: Option<&dyn DevicePtr<i32>>,
    out: &mut CudaSlice<bf16>,
    max_logits: &mut CudaSlice<f32>,
    lse: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        query_rows > 0
            && kv_rows > 0
            && topk > 0
            && topk <= GLM52_FLASHMLA_SPARSE_TOPK
            && topk.is_multiple_of(GLM52_FLASHMLA_SPARSE_TOPK_BLOCK)
            && sm_scale.is_finite()
            && sm_scale > 0.0,
        "GLM5.2 FlashMLA sparse prefill shape is invalid"
    );
    ensure!(
        q.len() >= query_rows * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
            && kv.len() >= kv_rows * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
            && topk_indices.len() >= query_rows * topk
            && topk_length.is_none_or(|length| length.len() >= query_rows)
            && out.len()
                >= query_rows * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
            && max_logits.len() >= query_rows * GLM52_FLASHMLA_SPARSE_HEADS
            && lse.len() >= query_rows * GLM52_FLASHMLA_SPARSE_HEADS,
        "GLM5.2 FlashMLA sparse prefill buffers are too small"
    );
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_ptr, _kv_guard) = kv.device_ptr(&ctx.stream);
    let (indices_ptr, _indices_guard) = topk_indices.device_ptr(&ctx.stream);
    let (length_ptr, _length_guard) = match topk_length {
        Some(length) => {
            let (ptr, guard) = length.device_ptr(&ctx.stream);
            (ptr as *const i32, Some(guard))
        }
        None => (std::ptr::null(), None),
    };
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let (max_ptr, _max_guard) = max_logits.device_ptr_mut(&ctx.stream);
    let (lse_ptr, _lse_guard) = lse.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashmla_sparse_prefill_launch_cuda(
            q_ptr as *const ffi::Half,
            kv_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            length_ptr,
            out_ptr as *mut ffi::Half,
            max_ptr as *mut f32,
            lse_ptr as *mut f32,
            query_rows as i32,
            kv_rows as i32,
            topk as i32,
            sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse prefill launch failed: {err}"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm52FlashMlaSparseDecode {
    pub batch_size: usize,
    pub num_blocks: usize,
    /// Byte offset of this layer's first page inside the KV slab the launch
    /// receives (0 when the cache is a dedicated per-layer arena).
    pub kv_layer_offset_bytes: usize,
    /// Byte distance between consecutive pages. The tight per-layer layout is
    /// `64 * 656`; the page-first slab passes the whole-page stride.
    pub kv_block_stride_bytes: usize,
    pub topk: usize,
    pub num_sm_parts: usize,
    pub sm_scale: f32,
}

impl Glm52FlashMlaSparseDecode {
    fn validate(self) -> Result<()> {
        ensure!(
            (1..=GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY).contains(&self.batch_size),
            "GLM5.2 FlashMLA sparse decode batch_size {} out of 1..={}",
            self.batch_size,
            GLM52_FLASHMLA_SPARSE_BATCH_CAPACITY
        );
        ensure!(
            self.num_blocks > 0,
            "GLM5.2 FlashMLA sparse decode num_blocks must be positive"
        );
        // The TMA tensormap derives its per-page step from this stride, so it
        // must stay token-row granular (the shim rejects violations too).
        ensure!(
            self.kv_block_stride_bytes
                >= GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
                && self
                    .kv_block_stride_bytes
                    .is_multiple_of(GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN),
            "GLM5.2 FlashMLA sparse decode kv_block_stride_bytes {} must be a \
             multiple of {} covering at least one {}-token page",
            self.kv_block_stride_bytes,
            GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
            GLM52_FLASHMLA_SPARSE_PAGE_SIZE
        );
        ensure!(
            self.topk > 0
                && self.topk <= GLM52_FLASHMLA_SPARSE_TOPK
                && self.topk.is_multiple_of(GLM52_FLASHMLA_SPARSE_TOPK_BLOCK),
            "GLM5.2 FlashMLA sparse decode topk must be a multiple of {} in 1..={}, got {}",
            GLM52_FLASHMLA_SPARSE_TOPK_BLOCK,
            GLM52_FLASHMLA_SPARSE_TOPK,
            self.topk
        );
        ensure!(
            (1..=GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS).contains(&self.num_sm_parts),
            "GLM5.2 FlashMLA sparse decode num_sm_parts {} out of 1..={}",
            self.num_sm_parts,
            GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS
        );
        ensure!(
            self.sm_scale.is_finite() && self.sm_scale > 0.0,
            "GLM5.2 FlashMLA sparse decode sm_scale must be finite and positive"
        );
        Ok(())
    }

    fn q_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
    }

    pub fn packed_kv_cache_len(self) -> usize {
        self.kv_layer_offset_bytes
            + (self.num_blocks - 1) * self.kv_block_stride_bytes
            + GLM52_FLASHMLA_SPARSE_PAGE_SIZE * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN
    }

    fn topk_indices_len(self) -> usize {
        self.batch_size * self.topk
    }

    pub fn tile_scheduler_metadata_len(self) -> usize {
        self.num_sm_parts * GLM52_FLASHMLA_SPARSE_SCHED_META_INTS
    }

    pub fn num_splits_len(self) -> usize {
        self.batch_size + 1
    }

    pub fn lse_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS
    }

    pub fn latent_len(self) -> usize {
        self.batch_size * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }

    fn split_count(self) -> usize {
        self.batch_size + self.num_sm_parts
    }

    pub fn lse_accum_len(self) -> usize {
        self.split_count() * GLM52_FLASHMLA_SPARSE_HEADS
    }

    pub fn o_accum_len(self) -> usize {
        self.split_count() * GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM
    }
}

pub fn glm52_flashmla_sparse_decode_num_sm_parts() -> Result<usize> {
    let mut num_sm_parts = 0i32;
    let result =
        unsafe { ffi::glm52_flashmla_sparse_decode_num_sm_parts_cuda(&raw mut num_sm_parts) };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse num_sm_parts query failed: {err}"))?;
    ensure!(
        (1..=GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS).contains(&(num_sm_parts as usize)),
        "GLM5.2 FlashMLA sparse num_sm_parts query returned {num_sm_parts}; supported range is 1..={}",
        GLM52_FLASHMLA_SPARSE_MAX_SM_PARTS
    );
    Ok(num_sm_parts as usize)
}

pub fn glm52_flashmla_sparse_decode_metadata_launch(
    ctx: &DeviceContext,
    batch_size: usize,
    topk: usize,
    num_sm_parts: usize,
    tile_scheduler_metadata: &mut CudaSlice<i32>,
    num_splits: &mut CudaSlice<i32>,
) -> Result<()> {
    // Metadata planning never touches the KV cache; a tight single-block
    // layout keeps validate() satisfied.
    let contract = Glm52FlashMlaSparseDecode {
        batch_size,
        num_blocks: 1,
        topk,
        num_sm_parts,
        sm_scale: 1.0,
        kv_layer_offset_bytes: 0,
        kv_block_stride_bytes: GLM52_FLASHMLA_SPARSE_PAGE_SIZE
            * GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
    };
    contract.validate()?;
    validate_metadata_buffers(contract, tile_scheduler_metadata, num_splits)?;

    let (sched_ptr, _sched_guard) = tile_scheduler_metadata.device_ptr_mut(&ctx.stream);
    let (splits_ptr, _splits_guard) = num_splits.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashmla_sparse_decode_metadata_cuda(
            sched_ptr as *mut i32,
            splits_ptr as *mut i32,
            batch_size as i32,
            topk as i32,
            num_sm_parts as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse metadata launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_flashmla_sparse_decode_launch(
    ctx: &DeviceContext,
    contract: Glm52FlashMlaSparseDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
    out_latent: &mut CudaSlice<bf16>,
    lse: &mut CudaSlice<f32>,
    lse_accum: &mut CudaSlice<f32>,
    o_accum: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_decode_buffers(
        contract,
        q,
        packed_kv_cache,
        topk_indices,
        tile_scheduler_metadata,
        num_splits,
        out_latent,
        lse,
        lse_accum,
        o_accum,
    )?;

    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (kv_base_ptr, _kv_guard) = packed_kv_cache.device_ptr(&ctx.stream);
    let kv_ptr = kv_base_ptr + contract.kv_layer_offset_bytes as u64;
    let (indices_ptr, _indices_guard) = topk_indices.device_ptr(&ctx.stream);
    let (sched_ptr, _sched_guard) = tile_scheduler_metadata.device_ptr(&ctx.stream);
    let (splits_ptr, _splits_guard) = num_splits.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out_latent.device_ptr_mut(&ctx.stream);
    let (lse_ptr, _lse_guard) = lse.device_ptr_mut(&ctx.stream);
    let (lse_accum_ptr, _lse_accum_guard) = lse_accum.device_ptr_mut(&ctx.stream);
    let (o_accum_ptr, _o_accum_guard) = o_accum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashmla_sparse_decode_launch_cuda(
            q_ptr as *const ffi::Half,
            kv_ptr as *const u8,
            indices_ptr as *const i32,
            sched_ptr as *const i32,
            splits_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            lse_ptr as *mut f32,
            lse_accum_ptr as *mut f32,
            o_accum_ptr as *mut f32,
            contract.batch_size as i32,
            contract.num_blocks as i32,
            contract.kv_block_stride_bytes as i64,
            contract.topk as i32,
            contract.num_sm_parts as i32,
            contract.sm_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FlashMLA sparse decode launch failed: {err}"))
}

fn validate_metadata_buffers(
    contract: Glm52FlashMlaSparseDecode,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        tile_scheduler_metadata.len() >= contract.tile_scheduler_metadata_len(),
        "GLM5.2 FlashMLA sparse scheduler metadata too small: have {}, need {}",
        tile_scheduler_metadata.len(),
        contract.tile_scheduler_metadata_len()
    );
    ensure!(
        num_splits.len() >= contract.num_splits_len(),
        "GLM5.2 FlashMLA sparse num_splits too small: have {}, need {}",
        num_splits.len(),
        contract.num_splits_len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_decode_buffers(
    contract: Glm52FlashMlaSparseDecode,
    q: &CudaSlice<bf16>,
    packed_kv_cache: &CudaSlice<u8>,
    topk_indices: &CudaSlice<i32>,
    tile_scheduler_metadata: &CudaSlice<i32>,
    num_splits: &CudaSlice<i32>,
    out_latent: &CudaSlice<bf16>,
    lse: &CudaSlice<f32>,
    lse_accum: &CudaSlice<f32>,
    o_accum: &CudaSlice<f32>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        q.len() >= contract.q_len(),
        "GLM5.2 FlashMLA sparse q too small: have {}, need {}",
        q.len(),
        contract.q_len()
    );
    ensure!(
        packed_kv_cache.len() >= contract.packed_kv_cache_len(),
        "GLM5.2 FlashMLA sparse packed kv cache too small: have {}, need {}",
        packed_kv_cache.len(),
        contract.packed_kv_cache_len()
    );
    ensure!(
        topk_indices.len() >= contract.topk_indices_len(),
        "GLM5.2 FlashMLA sparse topk_indices too small: have {}, need {}",
        topk_indices.len(),
        contract.topk_indices_len()
    );
    validate_metadata_buffers(contract, tile_scheduler_metadata, num_splits)?;
    ensure!(
        out_latent.len() >= contract.latent_len(),
        "GLM5.2 FlashMLA sparse latent output too small: have {}, need {}",
        out_latent.len(),
        contract.latent_len()
    );
    ensure!(
        lse.len() >= contract.lse_len(),
        "GLM5.2 FlashMLA sparse lse too small: have {}, need {}",
        lse.len(),
        contract.lse_len()
    );
    ensure!(
        lse_accum.len() >= contract.lse_accum_len(),
        "GLM5.2 FlashMLA sparse lse_accum too small: have {}, need {}",
        lse_accum.len(),
        contract.lse_accum_len()
    );
    ensure!(
        o_accum.len() >= contract.o_accum_len(),
        "GLM5.2 FlashMLA sparse o_accum too small: have {}, need {}",
        o_accum.len(),
        contract.o_accum_len()
    );
    Ok(())
}

#[cfg(test)]
mod prefill_tests {
    use super::*;

    #[test]
    #[ignore = "requires an SM100/103 CUDA device"]
    fn sparse_prefill_uniform_attention() -> Result<()> {
        const ROWS: usize = 64;
        const TOPK: usize = 64;
        let ctx = DeviceContext::new()?;
        let q = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM)?;
        let mut kv_host = vec![bf16::ZERO; ROWS * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM];
        for row in 0..ROWS {
            let value = bf16::from_f32((row % 4) as f32);
            kv_host[row * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM
                ..row * GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM + GLM52_FLASHMLA_SPARSE_V_HEAD_DIM]
                .fill(value);
        }
        let kv = ctx.stream.clone_htod(&kv_host)?;
        let indices = ctx
            .stream
            .clone_htod(&(0..TOPK as i32).collect::<Vec<_>>())?;
        let mut out = ctx
            .stream
            .alloc_zeros::<bf16>(GLM52_FLASHMLA_SPARSE_HEADS * GLM52_FLASHMLA_SPARSE_V_HEAD_DIM)?;
        let mut max_logits = ctx.stream.alloc_zeros::<f32>(GLM52_FLASHMLA_SPARSE_HEADS)?;
        let mut lse = ctx.stream.alloc_zeros::<f32>(GLM52_FLASHMLA_SPARSE_HEADS)?;
        glm52_flashmla_sparse_prefill_launch(
            &ctx,
            1,
            ROWS,
            TOPK,
            1.0,
            &q,
            &kv,
            &indices,
            None,
            &mut out,
            &mut max_logits,
            &mut lse,
        )?;
        let out = ctx.stream.clone_dtoh(&out)?;
        ensure!(
            out.iter().all(|value| (value.to_f32() - 1.5).abs() <= 0.01),
            "uniform sparse prefill did not average the selected values"
        );
        Ok(())
    }
}
