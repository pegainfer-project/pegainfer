use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::GpuTensor;
use crate::tensor::NormWeight;

pub const KIMI_K2_MLA_LOCAL_HEADS_TP8: usize = 8;
pub const KIMI_K2_MLA_Q_HEAD_DIM: usize = 192;
pub const KIMI_K2_MLA_V_HEAD_DIM: usize = 128;
pub const KIMI_K2_MLA_ROPE_DIM: usize = KIMI_K2_MLA_Q_HEAD_DIM - KIMI_K2_MLA_V_HEAD_DIM;
const KIMI_K2_MLA_Q_LORA_RANK: usize = 1536;
pub const KIMI_K2_MLA_KV_LORA_RANK: usize = 512;
pub const KIMI_K2_MLA_KV_A_OUT: usize = 576;
pub const KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8: usize = 2048;
pub const KIMI_K2_MLA_Q_LOCAL_OUT_TP8: usize = KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_Q_HEAD_DIM;
pub const KIMI_K2_MLA_O_LOCAL_IN_TP8: usize = KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_V_HEAD_DIM;
pub const KIMI_K2_MLA_QKV_A_OUT: usize = KIMI_K2_MLA_Q_LORA_RANK + KIMI_K2_MLA_KV_A_OUT;
pub const KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8: usize =
    KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_KV_LORA_RANK;
pub const KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8: usize =
    KIMI_K2_MLA_LOCAL_HEADS_TP8 * KIMI_K2_MLA_ROPE_DIM;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiMlaPagedKvLayout {
    max_pages: usize,
    pub(crate) page_size: usize,
    pub(crate) batch_size: usize,
    pub(crate) ckv_stride_page: usize,
    pub(crate) ckv_stride_n: usize,
    pub(crate) kpe_stride_page: usize,
    pub(crate) kpe_stride_n: usize,
}

impl KimiMlaPagedKvLayout {
    pub fn separate_contiguous(max_pages: usize, page_size: usize, batch_size: usize) -> Self {
        Self {
            max_pages,
            page_size,
            batch_size,
            ckv_stride_page: page_size * KIMI_K2_MLA_KV_LORA_RANK,
            ckv_stride_n: KIMI_K2_MLA_KV_LORA_RANK,
            kpe_stride_page: page_size * KIMI_K2_MLA_ROPE_DIM,
            kpe_stride_n: KIMI_K2_MLA_ROPE_DIM,
        }
    }

    pub fn required_ckv_len(&self) -> Result<usize> {
        required_cache_len(
            self.max_pages,
            self.page_size,
            self.ckv_stride_page,
            self.ckv_stride_n,
            KIMI_K2_MLA_KV_LORA_RANK,
        )
    }

    pub fn required_kpe_len(&self) -> Result<usize> {
        required_cache_len(
            self.max_pages,
            self.page_size,
            self.kpe_stride_page,
            self.kpe_stride_n,
            KIMI_K2_MLA_ROPE_DIM,
        )
    }
}

fn required_cache_len(
    max_pages: usize,
    page_size: usize,
    stride_page: usize,
    stride_n: usize,
    dim: usize,
) -> Result<usize> {
    if max_pages == 0 || page_size == 0 {
        return Ok(0);
    }
    let page_offset = (max_pages - 1)
        .checked_mul(stride_page)
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache page stride overflows"))?;
    let token_offset = (page_size - 1)
        .checked_mul(stride_n)
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache token stride overflows"))?;
    page_offset
        .checked_add(token_offset)
        .and_then(|offset| offset.checked_add(dim))
        .ok_or_else(|| anyhow::anyhow!("Kimi MLA paged cache length overflows"))
}

pub(super) fn validate_paged_layout(
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
) -> Result<()> {
    ensure!(layout.max_pages > 0, "Kimi MLA max_pages must be positive");
    ensure!(layout.page_size > 0, "Kimi MLA page_size must be positive");
    ensure!(
        layout.batch_size > 0,
        "Kimi MLA batch_size must be positive"
    );
    ensure!(
        layout.ckv_stride_n >= KIMI_K2_MLA_KV_LORA_RANK
            && layout.kpe_stride_n >= KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA cache token strides must cover ckv={} and kpe={}",
        KIMI_K2_MLA_KV_LORA_RANK,
        KIMI_K2_MLA_ROPE_DIM
    );
    ensure!(
        layout.ckv_stride_page >= layout.page_size * layout.ckv_stride_n
            && layout.kpe_stride_page >= layout.page_size * layout.kpe_stride_n,
        "Kimi MLA cache page strides must cover page_size * token_stride"
    );
    ensure!(
        !page_indices_d.is_empty(),
        "Kimi MLA page_indices must contain active decode pages"
    );
    ensure!(
        page_indptr_d.len() > layout.batch_size,
        "Kimi MLA page_indptr too small: got {}, need {}",
        page_indptr_d.len(),
        layout.batch_size + 1
    );
    ensure!(
        last_page_len_d.len() >= layout.batch_size,
        "Kimi MLA last_page_len too small: got {}, need {}",
        last_page_len_d.len(),
        layout.batch_size
    );
    Ok(())
}

pub fn kimi_mla_split_qkv_a(
    ctx: &DeviceContext,
    qkv_a: &GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    q_a: &mut GpuTensor<KIMI_K2_MLA_Q_LORA_RANK>,
    compressed: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    k_rope: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
) -> Result<()> {
    ensure!(
        q_a.seq_len == qkv_a.seq_len
            && compressed.seq_len == qkv_a.seq_len
            && k_rope.seq_len == qkv_a.seq_len,
        "Kimi MLA split seq_len mismatch: qkv_a={}, q_a={}, compressed={}, k_rope={}",
        qkv_a.seq_len,
        q_a.seq_len,
        compressed.seq_len,
        k_rope.seq_len
    );
    let (qkv_a_ptr, _qkv_a_guard) = qkv_a.data.device_ptr(&ctx.stream);
    let (q_a_ptr, _q_a_guard) = q_a.data.device_ptr_mut(&ctx.stream);
    let (compressed_ptr, _compressed_guard) = compressed.data.device_ptr_mut(&ctx.stream);
    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_split_qkv_a_cuda(
            qkv_a_ptr as *const ffi::Half,
            q_a_ptr as *mut ffi::Half,
            compressed_ptr as *mut ffi::Half,
            k_rope_ptr as *mut ffi::Half,
            qkv_a.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_mla_split_qkv_a_norm(
    ctx: &DeviceContext,
    qkv_a: &GpuTensor<KIMI_K2_MLA_QKV_A_OUT>,
    q_a_weight: &NormWeight<KIMI_K2_MLA_Q_LORA_RANK>,
    ckv_weight: &NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    q_a_normed: &mut GpuTensor<KIMI_K2_MLA_Q_LORA_RANK>,
    ckv_normed: &mut GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    k_rope: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    eps: f32,
) -> Result<()> {
    ensure!(
        q_a_normed.seq_len == qkv_a.seq_len
            && ckv_normed.seq_len == qkv_a.seq_len
            && k_rope.seq_len == qkv_a.seq_len,
        "Kimi MLA split+norm seq_len mismatch: qkv_a={}, q_a_normed={}, ckv_normed={}, k_rope={}",
        qkv_a.seq_len,
        q_a_normed.seq_len,
        ckv_normed.seq_len,
        k_rope.seq_len
    );
    let (qkv_a_ptr, _g0) = qkv_a.data.device_ptr(&ctx.stream);
    let (q_w_ptr, _g1) = q_a_weight.data.device_ptr(&ctx.stream);
    let (ckv_w_ptr, _g2) = ckv_weight.data.device_ptr(&ctx.stream);
    let (q_out_ptr, _g3) = q_a_normed.data.device_ptr_mut(&ctx.stream);
    let (ckv_out_ptr, _g4) = ckv_normed.data.device_ptr_mut(&ctx.stream);
    let (rope_ptr, _g5) = k_rope.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_mla_split_qkv_a_norm_cuda(
            qkv_a_ptr as *const ffi::Half,
            q_w_ptr as *const ffi::Half,
            ckv_w_ptr as *const ffi::Half,
            q_out_ptr as *mut ffi::Half,
            ckv_out_ptr as *mut ffi::Half,
            rope_ptr as *mut ffi::Half,
            eps,
            qkv_a.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_mla_rope_apply_kpe(
    ctx: &DeviceContext,
    k_rope: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    cos: &CudaSlice<half::bf16>,
    sin: &CudaSlice<half::bf16>,
    positions_d: &CudaSlice<i32>,
    append_kpe: &mut GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
) -> Result<()> {
    let seq_len = k_rope.seq_len;
    ensure!(
        append_kpe.seq_len == seq_len,
        "Kimi MLA apply KPE seq_len mismatch: k_rope={}, append_kpe={}",
        k_rope.seq_len,
        append_kpe.seq_len
    );
    ensure!(
        cos.len() >= seq_len * KIMI_K2_MLA_ROPE_DIM && sin.len() >= seq_len * KIMI_K2_MLA_ROPE_DIM,
        "Kimi MLA apply KPE RoPE cache too small: cos={}, sin={}, need {}",
        cos.len(),
        sin.len(),
        seq_len * KIMI_K2_MLA_ROPE_DIM
    );
    ensure!(
        positions_d.len() >= seq_len,
        "Kimi MLA apply KPE positions too small: got {}, need {}",
        positions_d.len(),
        seq_len
    );

    let (k_rope_ptr, _k_rope_guard) = k_rope.data.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let (positions_ptr, _positions_guard) = positions_d.device_ptr(&ctx.stream);
    let (append_kpe_ptr, _append_kpe_guard) = append_kpe.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_mla_rope_apply_kpe_cuda(
            k_rope_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            positions_ptr as *const i32,
            append_kpe_ptr as *mut ffi::Half,
            seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kimi_mla_paged_kv_append(
    ctx: &DeviceContext,
    ckv_cache: &mut CudaSlice<half::bf16>,
    kpe_cache: &mut CudaSlice<half::bf16>,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    append_ckv: &GpuTensor<KIMI_K2_MLA_KV_LORA_RANK>,
    append_kpe: &GpuTensor<KIMI_K2_MLA_ROPE_DIM>,
    batch_indices_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
) -> Result<()> {
    validate_paged_layout(layout, page_indices_d, page_indptr_d, last_page_len_d)?;
    ensure!(
        ckv_cache.len() >= layout.required_ckv_len()?,
        "Kimi MLA ckv_cache too small: got {}, need {}",
        ckv_cache.len(),
        layout.required_ckv_len()?
    );
    ensure!(
        kpe_cache.len() >= layout.required_kpe_len()?,
        "Kimi MLA kpe_cache too small: got {}, need {}",
        kpe_cache.len(),
        layout.required_kpe_len()?
    );
    ensure!(
        batch_indices_d.len() >= append_ckv.seq_len && positions_d.len() >= append_ckv.seq_len,
        "Kimi MLA append metadata too small for nnz={}",
        append_ckv.seq_len
    );
    ensure!(
        append_kpe.seq_len == append_ckv.seq_len,
        "Kimi MLA append seq_len mismatch: append_ckv={}, append_kpe={}",
        append_ckv.seq_len,
        append_kpe.seq_len
    );

    let (ckv_cache_ptr, _ckv_cache_guard) = ckv_cache.device_ptr_mut(&ctx.stream);
    let (kpe_cache_ptr, _kpe_cache_guard) = kpe_cache.device_ptr_mut(&ctx.stream);
    let (page_indices_ptr, _page_indices_guard) = page_indices_d.device_ptr(&ctx.stream);
    let (page_indptr_ptr, _page_indptr_guard) = page_indptr_d.device_ptr(&ctx.stream);
    let (last_page_len_ptr, _last_page_len_guard) = last_page_len_d.device_ptr(&ctx.stream);
    let (append_ckv_ptr, _append_ckv_guard) = append_ckv.data.device_ptr(&ctx.stream);
    let (append_kpe_ptr, _append_kpe_guard) = append_kpe.data.device_ptr(&ctx.stream);
    let (batch_indices_ptr, _batch_indices_guard) = batch_indices_d.device_ptr(&ctx.stream);
    let (positions_ptr, _positions_guard) = positions_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::kimi_mla_paged_kv_append_cuda(
            ckv_cache_ptr as *mut ffi::Half,
            kpe_cache_ptr as *mut ffi::Half,
            page_indices_ptr as *const i32,
            page_indptr_ptr as *const i32,
            last_page_len_ptr as *const i32,
            append_ckv_ptr as *const ffi::Half,
            append_kpe_ptr as *const ffi::Half,
            batch_indices_ptr as *const i32,
            positions_ptr as *const i32,
            append_ckv.seq_len as i32,
            layout.ckv_stride_page as i64,
            layout.ckv_stride_n as i64,
            layout.kpe_stride_page as i64,
            layout.kpe_stride_n as i64,
            layout.page_size as i32,
            layout.batch_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    if result != 0 {
        bail!(
            "kimi_mla_paged_kv_append_cuda failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}
