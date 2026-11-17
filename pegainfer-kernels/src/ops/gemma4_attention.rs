//! Gemma 4 attention wrappers: the global family at head dim 512 (split-KV
//! decode, paged prefill, QK prep) and the sliding family at head dim 256
//! (windowed paged prefill, plain-w QKV prep).

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::PrefillPagedPlan;
use super::attention::PagedGeometry;
use super::attention::checked_paged_geometry;
use super::attention::checked_row_offset;
use crate::ffi;
use crate::paged_kv::KvFormat;
use crate::paged_kv::KvStorage;
use crate::paged_kv::PagedKvLayout;
use crate::tensor::Columns;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

/// Byte offset of row `offset`'s first column of the band `c`, after checking
/// the row window and that the band lies within the row.
fn checked_band_offset(c: &Columns<'_>, offset: usize, span: usize, name: &str) -> Result<u64> {
    anyhow::ensure!(
        c.col
            .checked_add(c.width)
            .is_some_and(|end| end <= c.states.hidden_dim),
        "{name} columns [{}..+{}) exceed the row's {}",
        c.col,
        c.width,
        c.states.hidden_dim
    );
    let rows = checked_row_offset(c.states, offset, span, name)?;
    Ok(rows + (c.col * std::mem::size_of::<bf16>()) as u64)
}

/// `DeviceVec` fields are public too: reject a logical `len` past the backing
/// allocation before a kernel indexes through it.
fn ensure_vec_backed(v: &DeviceVec, name: &str) -> Result<()> {
    anyhow::ensure!(
        v.data.len() >= v.len,
        "{name} backing len {} < len {}",
        v.data.len(),
        v.len
    );
    Ok(())
}

/// Decode-metadata aggregate for the hd512 split-KV decode path, validated
/// at use: the wrapper calls [`Self::validate`] with the batch size it
/// derives.
///
/// `page_indptr` needs at least `batch_size + 1` elements and `last_page_len`
/// at least `batch_size`; `request_indices` and `kv_tile_indices` are
/// per-split-slot arrays sized by the caller's padded slot count, and
/// `kv_chunk_size` holds the one chunk-size entry the kernel reads.
/// `page_indices` must be non-empty (its per-request reach is device-side
/// data and cannot be verified here).
///
/// The pages referenced by `page_indices` must lie within `kv_buffer`; the
/// wrapper cannot verify device-side contents — caller contract.
pub struct Hd512DecodeMetadata<'a> {
    page_indices: &'a CudaSlice<i32>,
    page_indptr: &'a CudaSlice<i32>,
    last_page_len: &'a CudaSlice<i32>,
    request_indices: &'a CudaSlice<i32>,
    kv_tile_indices: &'a CudaSlice<i32>,
    kv_chunk_size: &'a CudaSlice<i32>,
    /// The chunk the tile indices count in, on the host as well: the device
    /// slot holds the same number for the incumbent's kernel, and a launcher
    /// that sizes its own grid from the tiles needs it where it can read it.
    chunk_tokens: usize,
}

impl<'a> Hd512DecodeMetadata<'a> {
    pub fn new(
        page_indices: &'a CudaSlice<i32>,
        page_indptr: &'a CudaSlice<i32>,
        last_page_len: &'a CudaSlice<i32>,
        request_indices: &'a CudaSlice<i32>,
        kv_tile_indices: &'a CudaSlice<i32>,
        kv_chunk_size: &'a CudaSlice<i32>,
        chunk_tokens: usize,
    ) -> Self {
        Self {
            page_indices,
            page_indptr,
            last_page_len,
            request_indices,
            kv_tile_indices,
            kv_chunk_size,
            chunk_tokens,
        }
    }

    pub fn page_indices(&self) -> &'a CudaSlice<i32> {
        self.page_indices
    }

    pub fn page_indptr(&self) -> &'a CudaSlice<i32> {
        self.page_indptr
    }

    pub fn last_page_len(&self) -> &'a CudaSlice<i32> {
        self.last_page_len
    }

    pub fn request_indices(&self) -> &'a CudaSlice<i32> {
        self.request_indices
    }

    pub fn kv_tile_indices(&self) -> &'a CudaSlice<i32> {
        self.kv_tile_indices
    }

    pub fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    pub fn validate(&self, batch_size: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.page_indices.is_empty(),
            "Hd512DecodeMetadata: page_indices must be non-empty"
        );
        anyhow::ensure!(
            self.page_indptr.len() > batch_size,
            "Hd512DecodeMetadata: page_indptr length {} must be >= batch_size + 1 = {}",
            self.page_indptr.len(),
            batch_size + 1
        );
        for (name, len) in [
            ("last_page_len", self.last_page_len.len()),
            ("request_indices", self.request_indices.len()),
            ("kv_tile_indices", self.kv_tile_indices.len()),
        ] {
            anyhow::ensure!(
                len >= batch_size,
                "Hd512DecodeMetadata: {name} length {len} must be >= batch_size {batch_size}",
            );
        }
        // The kernel reads one chunk size for the whole launch.
        anyhow::ensure!(
            !self.kv_chunk_size.is_empty(),
            "Hd512DecodeMetadata: kv_chunk_size must hold its one entry"
        );
        anyhow::ensure!(
            self.chunk_tokens > 0,
            "Hd512DecodeMetadata: chunk_tokens must be positive"
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_split_kv_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    meta: &Hd512DecodeMetadata,
    split_o_indptr_d: &CudaSlice<i32>,
    split_valid_mask_d: &CudaSlice<u8>,
    split_tmp_v: &mut CudaSlice<bf16>,
    split_tmp_s: &mut CudaSlice<f32>,
    split_padded_slots: usize,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    sm_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "paged_attention_batch_decode_hd512 sm_scale {sm_scale} must be finite"
    );
    let num_kv_heads = layout.num_kv_heads;
    let geometry = checked_paged_geometry(
        "hd512 split decode",
        layout,
        kv_buffer.len(),
        layer,
        512,
        num_kv_heads,
        false,
    )?;
    anyhow::ensure!(
        row_offset < q.seq_len,
        "hd512 split decode row_offset {row_offset} leaves no rows of {}",
        q.seq_len
    );
    let batch_size = q.seq_len - row_offset;
    meta.validate(batch_size)?;
    anyhow::ensure!(
        output.seq_len == q.seq_len,
        "hd512 decode output.seq_len {} != q.seq_len {}",
        output.seq_len,
        q.seq_len
    );
    let qo_dim = num_qo_heads.checked_mul(512).ok_or_else(|| {
        anyhow::anyhow!("hd512 split decode num_qo_heads {num_qo_heads} * 512 overflows")
    })?;
    anyhow::ensure!(
        q.hidden_dim == qo_dim,
        "hd512 decode q.hidden_dim {} != num_qo_heads {num_qo_heads} * 512",
        q.hidden_dim
    );
    anyhow::ensure!(
        output.hidden_dim == qo_dim,
        "hd512 decode output.hidden_dim {} != num_qo_heads {num_qo_heads} * 512",
        output.hidden_dim
    );
    anyhow::ensure!(
        split_padded_slots >= batch_size,
        "hd512 split decode padded_slots {split_padded_slots} < batch {batch_size}"
    );
    anyhow::ensure!(
        meta.request_indices.len() >= split_padded_slots
            && meta.kv_tile_indices.len() >= split_padded_slots
            && split_valid_mask_d.len() >= split_padded_slots,
        "hd512 split decode plan arrays shorter than padded_slots {split_padded_slots}"
    );
    anyhow::ensure!(
        split_o_indptr_d.len() > batch_size,
        "hd512 split decode o_indptr len {} < batch {batch_size} + 1",
        split_o_indptr_d.len()
    );
    let tmp_v_need = split_padded_slots.checked_mul(qo_dim).ok_or_else(|| {
        anyhow::anyhow!("hd512 split decode padded_slots {split_padded_slots} * {qo_dim} overflows")
    })?;
    let tmp_s_need = split_padded_slots
        .checked_mul(num_qo_heads)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "hd512 split decode padded_slots {split_padded_slots} * {num_qo_heads} overflows"
            )
        })?;
    anyhow::ensure!(
        split_tmp_v.len() >= tmp_v_need && split_tmp_s.len() >= tmp_s_need,
        "hd512 split decode workspace shorter than padded_slots {split_padded_slots}"
    );
    let q_elems = q.checked_extent("batch_decode_hd512 q")?;
    let out_elems = output.checked_extent("batch_decode_hd512 output")?;
    crate::ops::checked_i32(q_elems, "hd512 split decode q extent")?;
    crate::ops::checked_i32(out_elems, "hd512 split decode output extent")?;
    let batch_i32 = crate::ops::checked_i32(batch_size, "hd512 split decode batch")?;
    let padded_i32 =
        crate::ops::checked_i32(split_padded_slots, "hd512 split decode padded slots")?;
    let qo_heads_i32 = crate::ops::checked_i32(num_qo_heads, "hd512 split decode qo heads")?;
    let kv_heads_i32 = crate::ops::checked_i32(num_kv_heads, "hd512 split decode kv heads")?;

    let q_row_bytes = checked_row_offset(q, row_offset, batch_size, "hd512 split decode q")?;
    let out_row_bytes =
        checked_row_offset(output, row_offset, batch_size, "hd512 split decode output")?;
    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_row_bytes;
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let out_ptr = out_ptr + out_row_bytes;
    let (pi_ptr, _gpi) = meta.page_indices.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = meta.page_indptr.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = meta.last_page_len.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = meta.request_indices.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = meta.kv_tile_indices.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = meta.kv_chunk_size.device_ptr(&ctx.stream);
    let (soi_ptr, _gsoi) = split_o_indptr_d.device_ptr(&ctx.stream);
    let (svm_ptr, _gsvm) = split_valid_mask_d.device_ptr(&ctx.stream);
    let (stv_ptr, _gstv) = split_tmp_v.device_ptr_mut(&ctx.stream);
    let (sts_ptr, _gsts) = split_tmp_s.device_ptr_mut(&ctx.stream);

    let stream = crate::tensor::active_cu_stream(ctx);

    // No KV scatter here: the serving prep kernel has already written this
    // step's normed+roped K and its V fork; this only attends over the
    // resident pages.
    let result = unsafe {
        ffi::paged_attention_decode_split_kv_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            geometry.k_offset_elems,
            geometry.v_offset_elems,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            soi_ptr as *const i32,
            svm_ptr as *const u8,
            stv_ptr as *mut ffi::Half,
            sts_ptr as *mut f32,
            qo_heads_i32,
            kv_heads_i32,
            512,
            geometry.page_size,
            batch_i32,
            padded_i32,
            geometry.stride_page,
            sm_scale,
            stream,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "paged_attention_decode_split_kv_cuda_hd512 (batch) failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// The pages referenced by `plan.page_indices_d()` must lie within
/// `kv_buffer`; the wrapper cannot verify device-side contents — caller
/// contract.
pub fn batch_prefill_paged_hd512_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    sm_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "batch_prefill_paged_hd512 sm_scale {sm_scale} must be finite"
    );
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        head_dim == 512,
        "hd512 prefill expects head_dim 512, got {}",
        head_dim
    );
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * 512,
        "hd512 prefill q.hidden_dim {} != num_qo_heads {} * 512",
        q.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        output.hidden_dim == num_qo_heads * 512,
        "hd512 prefill output.hidden_dim {} != num_qo_heads {} * 512",
        output.hidden_dim,
        num_qo_heads
    );
    anyhow::ensure!(
        q.seq_len >= plan.total_tokens,
        "hd512 prefill q.seq_len {} < plan.total_tokens {}",
        q.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        output.seq_len >= plan.total_tokens,
        "hd512 prefill output.seq_len {} < plan.total_tokens {}",
        output.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        layer < layout.num_layers,
        "hd512 prefill layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    q.checked_extent("batch_prefill_paged_hd512 q")?;
    output.checked_extent("batch_prefill_paged_hd512 output")?;
    anyhow::ensure!(
        plan.head_dim == 512,
        "plan built for head_dim {}, expected 512",
        plan.head_dim
    );
    anyhow::ensure!(
        plan.num_kv_heads == num_kv_heads,
        "plan built for num_kv_heads {}, expected {}",
        plan.num_kv_heads,
        num_kv_heads
    );
    anyhow::ensure!(
        plan.num_qo_heads == num_qo_heads,
        "plan built for num_qo_heads {}, expected {}",
        plan.num_qo_heads,
        num_qo_heads
    );

    // Plan tiles laid out for a cta_tile_q override would be misread by the
    // kernel's automatic derivation.
    let kernel_cta_tile_q = unsafe {
        ffi::batch_prefill_cta_tile_q_with_override(
            plan.total_tokens as i32,
            plan.num_qo_heads as i32,
            plan.num_kv_heads as i32,
            plan.head_dim as i32,
            0, // auto-detect, matching the hd512 kernel
        )
    };
    anyhow::ensure!(
        kernel_cta_tile_q == plan.cta_tile_q(),
        "hd512 prefill cta_tile_q: plan recorded {}, kernel derives {}; \
         overridden plans are not supported at hd512",
        plan.cta_tile_q(),
        kernel_cta_tile_q
    );

    let PagedGeometry {
        k_offset_elems: k_offset,
        v_offset_elems: v_offset,
        stride_page,
        ..
    } = checked_paged_geometry(
        "hd512 prefill",
        layout,
        kv_buffer.len(),
        layer,
        head_dim,
        num_kv_heads,
        false,
    )?;

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _gqti) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _gtnr) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::batch_prefill_paged_cuda_hd512(
            q_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            buf_ptr as *const ffi::Half,
            k_offset,
            v_offset,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            qi_ptr as *const i32,
            ri_ptr as *const i32,
            qti_ptr as *const i32,
            kti_ptr as *const i32,
            kcs_ptr as *const i32,
            tnr_ptr as *const u32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            layout.page_size as i32,
            plan.total_tokens as i32,
            plan.batch_size(),
            plan.num_tiles,
            stride_page,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "batch_prefill_paged_cuda_hd512 (prefill) failed for layer {layer}, \
             bs={}, tiles={}, qo_heads={num_qo_heads}, kv_heads={num_kv_heads}: {result}{}",
            plan.batch_size(),
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

/// Windowed batch prefill over paged KV at head_dim 256 (Gemma 4 local
/// layers): attention read only — the prep kernel has already written K/V
/// into the pool. `window_left` is an inclusive distance: an N-token window
/// passes N - 1, and -1 degrades to full attention. `sm_scale` is the
/// caller's; Gemma 4 runs unscaled attention (1.0).
#[allow(clippy::too_many_arguments)]
pub fn batch_prefill_paged_window_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    sm_scale: f32,
    window_left: i32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "batch_prefill_paged_window_hd256 sm_scale {sm_scale} must be finite"
    );
    anyhow::ensure!(
        window_left >= -1,
        "batch_prefill_paged_window_hd256 window_left {window_left} must be >= -1"
    );
    let num_kv_heads = layout.num_kv_heads;
    let geometry = checked_paged_geometry(
        "hd256 window prefill",
        layout,
        kv_buffer.len(),
        layer,
        256,
        num_kv_heads,
        true,
    )?;
    anyhow::ensure!(
        plan.max_page_index < geometry.num_pages,
        "hd256 window prefill plan references page {} but the pool holds {} pages; the attention \
         kernel computes addresses from page ids with no device-side bounds check",
        plan.max_page_index,
        geometry.num_pages
    );
    anyhow::ensure!(
        plan.max_last_page_len <= geometry.page_size,
        "hd256 window prefill plan last-page length {} exceeds page_size {}; the kernel would \
         read a page-table entry that does not exist",
        plan.max_last_page_len,
        geometry.page_size
    );
    let qo_dim = num_qo_heads.checked_mul(256).ok_or_else(|| {
        anyhow::anyhow!("hd256 window prefill num_qo_heads {num_qo_heads} * 256 overflows")
    })?;
    anyhow::ensure!(
        q.hidden_dim == qo_dim,
        "hd256 window prefill q.hidden_dim {} != num_qo_heads {num_qo_heads} * 256",
        q.hidden_dim
    );
    anyhow::ensure!(
        output.hidden_dim == qo_dim,
        "hd256 window prefill output.hidden_dim {} != num_qo_heads {num_qo_heads} * 256",
        output.hidden_dim
    );
    anyhow::ensure!(
        q.seq_len >= plan.total_tokens,
        "hd256 window prefill q.seq_len {} < plan.total_tokens {}",
        q.seq_len,
        plan.total_tokens
    );
    anyhow::ensure!(
        output.seq_len >= plan.total_tokens,
        "hd256 window prefill output.seq_len {} < plan.total_tokens {}",
        output.seq_len,
        plan.total_tokens
    );
    let q_elems = q.checked_extent("batch_prefill_paged_window_hd256 q")?;
    let out_elems = output.checked_extent("batch_prefill_paged_window_hd256 output")?;
    crate::ops::checked_i32(q_elems, "hd256 window prefill q extent")?;
    crate::ops::checked_i32(out_elems, "hd256 window prefill output extent")?;
    anyhow::ensure!(
        plan.head_dim == 256,
        "plan built for head_dim {}, expected 256",
        plan.head_dim
    );
    anyhow::ensure!(
        plan.num_kv_heads == num_kv_heads,
        "plan built for num_kv_heads {}, expected {}",
        plan.num_kv_heads,
        num_kv_heads
    );
    anyhow::ensure!(
        plan.num_qo_heads == num_qo_heads,
        "plan built for num_qo_heads {}, expected {}",
        plan.num_qo_heads,
        num_qo_heads
    );

    // Plan tiles laid out for a cta_tile_q override would be misread by the
    // kernel's automatic derivation.
    let num_qo_heads_i32 =
        crate::ops::checked_i32(num_qo_heads, "hd256 window prefill num_qo_heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(num_kv_heads, "hd256 window prefill num_kv_heads")?;
    let total_tokens_i32 =
        crate::ops::checked_i32(plan.total_tokens, "hd256 window prefill total_tokens")?;
    let kernel_cta_tile_q = unsafe {
        ffi::batch_prefill_cta_tile_q_with_override(
            total_tokens_i32,
            num_qo_heads_i32,
            num_kv_heads_i32,
            256,
            0, // auto-detect, matching the hd256 kernel
        )
    };
    anyhow::ensure!(
        kernel_cta_tile_q == plan.cta_tile_q(),
        "hd256 window prefill cta_tile_q: plan recorded {}, kernel derives {}; \
         overridden plans are not supported here",
        plan.cta_tile_q(),
        kernel_cta_tile_q
    );

    let (buf_ptr, _gbuf) = kv_buffer.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d.device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d.device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d.device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d.device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = plan.request_indices_d.device_ptr(&ctx.stream);
    let (qti_ptr, _gqti) = plan.qo_tile_indices_d.device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = plan.kv_tile_indices_d.device_ptr(&ctx.stream);
    let (kcs_ptr, _gkcs) = plan.kv_chunk_size_d.device_ptr(&ctx.stream);
    let (tnr_ptr, _gtnr) = plan.total_num_rows_d.device_ptr(&ctx.stream);

    let (result, entry_point) = match layout.storage {
        KvStorage::E4m3 => {
            let result = unsafe {
                ffi::gemma4_batch_prefill_paged_window_hd256_fp8kv_cuda(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    buf_ptr as *const core::ffi::c_void,
                    geometry.k_offset_elems,
                    geometry.v_offset_elems,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    num_qo_heads_i32,
                    num_kv_heads_i32,
                    256,
                    geometry.page_size,
                    total_tokens_i32,
                    plan.batch_size(),
                    plan.num_tiles,
                    geometry.stride_page,
                    sm_scale,
                    0,
                    window_left,
                    crate::tensor::active_cu_stream(ctx),
                )
            };
            (result, "gemma4_batch_prefill_paged_window_hd256_fp8kv_cuda")
        }
        KvStorage::Bf16 => (
            unsafe {
                ffi::batch_prefill_paged_window_cuda_hd256(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    geometry.k_offset_elems,
                    geometry.v_offset_elems,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    num_qo_heads_i32,
                    num_kv_heads_i32,
                    256,
                    geometry.page_size,
                    total_tokens_i32,
                    plan.batch_size(),
                    plan.num_tiles,
                    geometry.stride_page,
                    sm_scale,
                    window_left,
                    crate::tensor::active_cu_stream(ctx),
                )
            },
            "batch_prefill_paged_window_cuda_hd256",
        ),
    };
    if result != 0 {
        anyhow::bail!(
            "{entry_point} failed for layer {layer}, \
             bs={}, tiles={}, qo_heads={num_qo_heads}, kv_heads={num_kv_heads}, \
             window_left={window_left}: {result}{}",
            plan.batch_size(),
            plan.num_tiles,
            crate::ops::ffi_exception_message(result)
        );
    }

    Ok(())
}

/// What the hd512 prep writes into, from the layout's format: the layer's
/// K and V blocks and the row's bands. For the split format the blocks are
/// the layer's two and the row is the head; for the folded one both offsets
/// name the layer's single block, the row is `head_dim + rotary` wide and
/// `fold_rotary` is the rotated columns the kernel keeps of K.
struct Hd512PrepTarget {
    k_offset_elems: i64,
    v_offset_elems: i64,
    row_width: i32,
    fold_rotary: i32,
    page_size: i32,
    num_pages: i32,
    stride_page: i64,
}

fn hd512_prep_target(
    what: &str,
    layout: &PagedKvLayout,
    pool_len: usize,
    layer: usize,
    num_kv_heads: usize,
) -> Result<Hd512PrepTarget> {
    anyhow::ensure!(
        layout.storage == KvStorage::Bf16,
        "{what} has no fp8 KV path; the layout carries {}-byte elements",
        layout.storage.elem_bytes()
    );
    anyhow::ensure!(
        layout.head_dim == 512,
        "{what} layout.head_dim {} != 512",
        layout.head_dim
    );
    anyhow::ensure!(
        layout.num_kv_heads == num_kv_heads,
        "{what} layout.num_kv_heads {} != num_kv_heads {num_kv_heads}",
        layout.num_kv_heads
    );
    let block = layout.block_geometry(pool_len, layer, what)?;
    let v_offset = match layout.format {
        KvFormat::Split => block
            .block_offset_elems
            .checked_add(layout.kv_block_len)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{what} K offset {} + kv_block_len {} overflows",
                    block.block_offset_elems,
                    layout.kv_block_len
                )
            })?,
        KvFormat::Folded { .. } => block.block_offset_elems,
    };
    let to_i64 = |value: usize, name: &str| {
        i64::try_from(value).map_err(|_| anyhow::anyhow!("{what} {name} {value} does not fit i64"))
    };
    Ok(Hd512PrepTarget {
        k_offset_elems: to_i64(block.block_offset_elems, "K offset")?,
        v_offset_elems: to_i64(v_offset, "V offset")?,
        row_width: crate::ops::checked_i32(
            layout.format.row_width(layout.head_dim),
            &format!("{what} row_width"),
        )?,
        fold_rotary: crate::ops::checked_i32(
            layout.format.fold_rotary(),
            &format!("{what} fold_rotary"),
        )?,
        page_size: crate::ops::checked_i32(block.page_size, &format!("{what} page_size"))?,
        num_pages: crate::ops::checked_i32(block.num_pages, &format!("{what} num_pages"))?,
        stride_page: to_i64(block.stride_page, "page_stride")?,
    })
}

/// Plain-w QKV prep at head_dim 256 (Gemma 4 local layers).
/// Q is normalised + rotated into a contiguous `q_out`; K is normalised +
/// rotated straight into the paged KV pool at layer `layer`'s K block, and
/// V — a separate v_proj head vector, unlike the hd512 K=V fork — is
/// weightless-normalised (never rotated) into the layer's V block, all in
/// one kernel with no intermediate scatter. `q`, `k` and `v` are column
/// bands read at their own row strides, so they may share one fused Q|K|V
/// projection row.
///
/// The kernel `__trap()`s on any out-of-range pos or page id as the second
/// layer of the host validation's defence. `page_indices` is the resident
/// page row and `page_origin` the absolute page its first entry covers, so
/// a caller that released the front passes the released count; the origin is
/// page-aligned, which keeps in-page offsets position-invariant and shifts
/// only the row index. RoPE still runs on absolute positions.
#[allow(clippy::too_many_arguments)]
pub fn qkv_norm_rope_paged_prefill_hd256_plain_into<'a>(
    ctx: &DeviceContext,
    q: impl Into<Columns<'a>>,
    k: impl Into<Columns<'a>>,
    v: impl Into<Columns<'a>>,
    q_out: &mut HiddenStates,
    row_offset: usize,
    kv_pool: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    layer: usize,
    page_indices: &CudaSlice<i32>,
    pages_offset: usize,
    page_origin: usize,
    start_pos: usize,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let (q, k, v) = (q.into(), k.into(), v.into());
    // The prompt segment is the row suffix `[row_offset..seq_len)` with its
    // page table `pages_offset` elements into the (possibly concatenated)
    // table — a multi-prompt mixed step parks earlier prompts and their
    // tables in the prefixes.
    anyhow::ensure!(
        row_offset < q.states.seq_len,
        "hd256 paged prep row_offset {row_offset} leaves no rows of {}",
        q.states.seq_len
    );
    let seq_len = q.states.seq_len - row_offset;
    anyhow::ensure!(
        pages_offset < page_indices.len(),
        "hd256 paged prep pages_offset {pages_offset} exceeds table len {}",
        page_indices.len()
    );
    let q_dim = num_q_heads.checked_mul(256).ok_or_else(|| {
        anyhow::anyhow!("hd256 paged prep num_q_heads {num_q_heads} * 256 overflows")
    })?;
    let kv_dim = num_kv_heads.checked_mul(256).ok_or_else(|| {
        anyhow::anyhow!("hd256 paged prep num_kv_heads {num_kv_heads} * 256 overflows")
    })?;
    anyhow::ensure!(
        q.width == q_dim,
        "hd256 paged prep q width {} != num_q_heads {num_q_heads} * 256",
        q.width
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.width,
        "hd256 paged prep q_out.hidden_dim {} != q width {}",
        q_out.hidden_dim,
        q.width
    );
    anyhow::ensure!(
        q_out.seq_len == q.states.seq_len,
        "hd256 paged prep q_out.seq_len {} != q rows {}",
        q_out.seq_len,
        q.states.seq_len
    );
    anyhow::ensure!(
        k.width == kv_dim,
        "hd256 paged prep k width {} != num_kv_heads {num_kv_heads} * 256",
        k.width
    );
    anyhow::ensure!(
        v.width == k.width,
        "hd256 paged prep v width {} != k width {}",
        v.width,
        k.width
    );
    anyhow::ensure!(
        k.states.seq_len == q.states.seq_len,
        "hd256 paged prep k rows {} != q rows {}",
        k.states.seq_len,
        q.states.seq_len
    );
    anyhow::ensure!(
        v.states.seq_len == q.states.seq_len,
        "hd256 paged prep v rows {} != q rows {}",
        v.states.seq_len,
        q.states.seq_len
    );
    let geometry = checked_paged_geometry(
        "hd256 paged prep",
        layout,
        kv_pool.len(),
        layer,
        256,
        num_kv_heads,
        true,
    )?;
    ensure_vec_backed(q_norm_weight, "hd256 paged prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd256 paged prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd256 paged prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd256 paged prep sin_cache")?;
    anyhow::ensure!(
        q_norm_weight.len == 256,
        "hd256 paged prep q_norm_weight len {} != 256",
        q_norm_weight.len
    );
    anyhow::ensure!(
        k_norm_weight.len == 256,
        "hd256 paged prep k_norm_weight len {} != 256",
        k_norm_weight.len
    );
    let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
        anyhow::anyhow!("hd256 paged prep start_pos {start_pos} + seq_len {seq_len} overflows")
    })?;
    let covered = (page_indices.len() - pages_offset)
        .checked_mul(layout.page_size)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "hd256 paged prep page window {} * page_size {} overflows",
                page_indices.len() - pages_offset,
                layout.page_size
            )
        })?;
    let row_start = page_origin.checked_mul(layout.page_size).ok_or_else(|| {
        anyhow::anyhow!(
            "hd256 paged prep page_origin {page_origin} * page_size {} overflows",
            layout.page_size
        )
    })?;
    let row_end = row_start.checked_add(covered).ok_or_else(|| {
        anyhow::anyhow!("hd256 paged prep row start {row_start} + span {covered} overflows")
    })?;
    anyhow::ensure!(
        row_start <= start_pos,
        "hd256 paged prep page_origin {page_origin} starts after start_pos {start_pos}"
    );
    anyhow::ensure!(
        row_end >= end_pos,
        "hd256 paged prep row covers tokens {row_start}..{row_end}, need start_pos \
         {start_pos} + seq_len {seq_len}"
    );
    anyhow::ensure!(
        end_pos <= cos_max_pos,
        "hd256 paged prep start_pos {start_pos} + seq_len {seq_len} > cos_max_pos {cos_max_pos}"
    );
    let table_len = cos_max_pos.checked_mul(rotary_dim).ok_or_else(|| {
        anyhow::anyhow!(
            "hd256 paged prep cos_max_pos {cos_max_pos} * rotary_dim {rotary_dim} overflows"
        )
    })?;
    anyhow::ensure!(
        cos_cache.len >= table_len,
        "hd256 paged prep cos_cache len {} < cos_max_pos {cos_max_pos} * rotary_dim {rotary_dim}",
        cos_cache.len
    );
    anyhow::ensure!(
        sin_cache.len >= table_len,
        "hd256 paged prep sin_cache len {} < cos_max_pos {cos_max_pos} * rotary_dim {rotary_dim}",
        sin_cache.len
    );
    let q_elems = q.states.checked_extent("hd256 paged prep q")?;
    q_out.checked_extent("hd256 paged prep q_out")?;
    let k_elems = k.states.checked_extent("hd256 paged prep k")?;
    v.states.checked_extent("hd256 paged prep v")?;
    crate::ops::checked_i32(q_elems, "hd256 paged prep q extent")?;
    crate::ops::checked_i32(k_elems, "hd256 paged prep k extent")?;
    crate::ops::checked_i32(table_len, "hd256 paged prep rope table extent")?;

    let page_origin_i32 = crate::ops::checked_i32(page_origin, "hd256 paged prep page_origin")?;
    let num_q_heads_i32 = crate::ops::checked_i32(num_q_heads, "hd256 paged prep num_q_heads")?;
    let num_kv_heads_i32 = crate::ops::checked_i32(num_kv_heads, "hd256 paged prep num_kv_heads")?;
    let seq_len_i32 = crate::ops::checked_i32(seq_len, "hd256 paged prep seq_len")?;
    let start_pos_i32 = crate::ops::checked_i32(start_pos, "hd256 paged prep start_pos")?;
    let cos_max_pos_i32 = crate::ops::checked_i32(cos_max_pos, "hd256 paged prep cos_max_pos")?;
    let rotary_dim_i32 = crate::ops::checked_i32(rotary_dim, "hd256 paged prep rotary_dim")?;

    let page_indices_len = crate::ops::checked_i32(
        page_indices.len() - pages_offset,
        "hd256 paged prep page window len",
    )?;
    let q_row_bytes = checked_band_offset(&q, row_offset, seq_len, "hd256 paged prep q")?;
    let k_row_bytes = checked_band_offset(&k, row_offset, seq_len, "hd256 paged prep k")?;
    let v_row_bytes = checked_band_offset(&v, row_offset, seq_len, "hd256 paged prep v")?;
    let qo_row_bytes = checked_row_offset(q_out, row_offset, seq_len, "hd256 paged prep q_out")?;
    let (q_ptr, _gq) = q.states.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_row_bytes;
    let (k_ptr, _gk) = k.states.data.device_ptr(&ctx.stream);
    let k_ptr = k_ptr + k_row_bytes;
    let (v_ptr, _gv) = v.states.data.device_ptr(&ctx.stream);
    let v_ptr = v_ptr + v_row_bytes;
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let qo_ptr = qo_ptr + qo_row_bytes;
    // The KV state owns mutation; this call borrows its shared pool handle.
    let (pool_ptr, _gp) = kv_pool.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices.device_ptr(&ctx.stream);
    let pi_ptr = pi_ptr + (pages_offset * std::mem::size_of::<i32>()) as u64;

    // Two `if`s, not a tuple-building match: a fn item only coerces to a
    // fn pointer across plain `if` arms.
    let fp8 = layout.storage == KvStorage::E4m3;
    let launch = if fp8 {
        ffi::qkv_norm_rope_paged_prefill_hd256_plain_fp8kv_cuda
    } else {
        ffi::qkv_norm_rope_paged_prefill_hd256_plain_cuda
    };
    let entry_point = if fp8 {
        "qkv_norm_rope_paged_prefill_hd256_plain_fp8kv_cuda"
    } else {
        "qkv_norm_rope_paged_prefill_hd256_plain_cuda"
    };
    let q_stride = crate::ops::checked_i32(q.states.hidden_dim, "hd256 paged prep q stride")?;
    let k_stride = crate::ops::checked_i32(k.states.hidden_dim, "hd256 paged prep k stride")?;
    let v_stride = crate::ops::checked_i32(v.states.hidden_dim, "hd256 paged prep v stride")?;
    let result = unsafe {
        launch(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_stride,
            k_stride,
            v_stride,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            pool_ptr as *mut ffi::Half,
            geometry.k_offset_elems,
            geometry.v_offset_elems,
            pi_ptr as *const i32,
            page_indices_len,
            page_origin_i32,
            num_q_heads_i32,
            num_kv_heads_i32,
            seq_len_i32,
            start_pos_i32,
            cos_max_pos_i32,
            rotary_dim_i32,
            rms_eps,
            geometry.page_size,
            geometry.num_pages,
            geometry.stride_page,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "{entry_point} failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Per-request hd256 prep; `page_origins` is the released local-window front.
/// The decode batch is the row suffix `[row_offset..seq_len)` — 0 covers the
/// whole tensor, a mixed step parks its prefill segment in the prefix.
/// Invalid device metadata traps before the first paged-pool access.
#[allow(clippy::too_many_arguments)]
pub fn qkv_norm_rope_paged_decode_hd256_plain_into<'a>(
    ctx: &DeviceContext,
    q: impl Into<Columns<'a>>,
    k: impl Into<Columns<'a>>,
    v: impl Into<Columns<'a>>,
    q_out: &mut HiddenStates,
    row_offset: usize,
    kv_pool: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    layer: usize,
    page_indices: &CudaSlice<i32>,
    page_indptr: &CudaSlice<i32>,
    page_origins: &CudaSlice<i32>,
    positions: &CudaSlice<i32>,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let (q, k, v) = (q.into(), k.into(), v.into());
    anyhow::ensure!(
        row_offset < q.states.seq_len,
        "hd256 paged decode prep row_offset {row_offset} leaves no rows of {}",
        q.states.seq_len
    );
    let batch = q.states.seq_len - row_offset;
    anyhow::ensure!(
        Some(q.width) == num_q_heads.checked_mul(256),
        "hd256 paged decode prep q width {} != num_q_heads {} * 256",
        q.width,
        num_q_heads
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.width && q_out.seq_len == q.states.seq_len,
        "hd256 paged decode prep q_out [{} x {}] != q [{} x {}]",
        q_out.hidden_dim,
        q_out.seq_len,
        q.width,
        q.states.seq_len
    );
    anyhow::ensure!(
        Some(k.width) == num_kv_heads.checked_mul(256) && k.states.seq_len == q.states.seq_len,
        "hd256 paged decode prep k [{} x {}] != [num_kv_heads {} * 256 x {}]",
        k.width,
        k.states.seq_len,
        num_kv_heads,
        q.states.seq_len
    );
    anyhow::ensure!(
        v.width == k.width && v.states.seq_len == q.states.seq_len,
        "hd256 paged decode prep v [{} x {}] != k [{} x {}]",
        v.width,
        v.states.seq_len,
        k.width,
        q.states.seq_len
    );
    let geometry = checked_paged_geometry(
        "hd256 paged decode prep",
        layout,
        kv_pool.len(),
        layer,
        256,
        num_kv_heads,
        true,
    )?;
    ensure_vec_backed(q_norm_weight, "hd256 paged decode prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd256 paged decode prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd256 paged decode prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd256 paged decode prep sin_cache")?;
    anyhow::ensure!(
        q_norm_weight.len == 256 && k_norm_weight.len == 256,
        "hd256 paged decode prep norm weight lens {} / {} != 256",
        q_norm_weight.len,
        k_norm_weight.len
    );
    let table_rows = cos_max_pos.checked_mul(rotary_dim).ok_or_else(|| {
        anyhow::anyhow!(
            "hd256 paged decode prep cos_max_pos {cos_max_pos} * rotary_dim {rotary_dim} overflows"
        )
    })?;
    crate::ops::checked_i32(table_rows, "hd256 paged decode prep rope table extent")?;
    anyhow::ensure!(
        cos_cache.len >= table_rows && sin_cache.len >= table_rows,
        "hd256 paged decode prep cos/sin lens {} / {} < cos_max_pos {cos_max_pos} * \
         rotary_dim {rotary_dim}",
        cos_cache.len,
        sin_cache.len
    );
    anyhow::ensure!(
        positions.len() >= batch && page_origins.len() >= batch && page_indptr.len() > batch,
        "hd256 paged decode prep metadata lens (positions {}, origins {}, indptr {}) \
         do not cover batch {batch}",
        positions.len(),
        page_origins.len(),
        page_indptr.len()
    );
    let page_indices_len = crate::ops::checked_i32(
        page_indices.len(),
        "hd256 paged decode prep page_indices len",
    )?;
    let q_elems = q.states.checked_extent("hd256 paged decode prep q")?;
    q_out.checked_extent("hd256 paged decode prep q_out")?;
    let k_elems = k.states.checked_extent("hd256 paged decode prep k")?;
    v.states.checked_extent("hd256 paged decode prep v")?;
    crate::ops::checked_i32(q_elems, "hd256 paged decode prep q extent")?;
    crate::ops::checked_i32(k_elems, "hd256 paged decode prep k extent")?;
    let num_q_heads_i32 = crate::ops::checked_i32(num_q_heads, "hd256 paged decode prep q heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(num_kv_heads, "hd256 paged decode prep kv heads")?;
    let batch_i32 = crate::ops::checked_i32(batch, "hd256 paged decode prep batch")?;
    let cos_max_pos_i32 =
        crate::ops::checked_i32(cos_max_pos, "hd256 paged decode prep cos_max_pos")?;
    let rotary_dim_i32 = crate::ops::checked_i32(rotary_dim, "hd256 paged decode prep rotary_dim")?;

    let q_row_bytes = checked_band_offset(&q, row_offset, batch, "hd256 paged decode prep q")?;
    let k_row_bytes = checked_band_offset(&k, row_offset, batch, "hd256 paged decode prep k")?;
    let v_row_bytes = checked_band_offset(&v, row_offset, batch, "hd256 paged decode prep v")?;
    let qo_row_bytes =
        checked_row_offset(q_out, row_offset, batch, "hd256 paged decode prep q_out")?;
    let (q_ptr, _gq) = q.states.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_row_bytes;
    let (k_ptr, _gk) = k.states.data.device_ptr(&ctx.stream);
    let k_ptr = k_ptr + k_row_bytes;
    let (v_ptr, _gv) = v.states.data.device_ptr(&ctx.stream);
    let v_ptr = v_ptr + v_row_bytes;
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let qo_ptr = qo_ptr + qo_row_bytes;
    // The KV states own mutation; this call borrows the shared pool handle.
    let (pool_ptr, _gp) = kv_pool.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices.device_ptr(&ctx.stream);
    let (ip_ptr, _gip) = page_indptr.device_ptr(&ctx.stream);
    let (og_ptr, _gog) = page_origins.device_ptr(&ctx.stream);
    let (ps_ptr, _gps) = positions.device_ptr(&ctx.stream);

    // Two `if`s, not a tuple-building match: a fn item only coerces to a
    // fn pointer across plain `if` arms.
    let fp8 = layout.storage == KvStorage::E4m3;
    let launch = if fp8 {
        ffi::qkv_norm_rope_paged_decode_hd256_plain_fp8kv_cuda
    } else {
        ffi::qkv_norm_rope_paged_decode_hd256_plain_cuda
    };
    let entry_point = if fp8 {
        "qkv_norm_rope_paged_decode_hd256_plain_fp8kv_cuda"
    } else {
        "qkv_norm_rope_paged_decode_hd256_plain_cuda"
    };
    let q_stride =
        crate::ops::checked_i32(q.states.hidden_dim, "hd256 paged decode prep q stride")?;
    let k_stride =
        crate::ops::checked_i32(k.states.hidden_dim, "hd256 paged decode prep k stride")?;
    let v_stride =
        crate::ops::checked_i32(v.states.hidden_dim, "hd256 paged decode prep v stride")?;
    let result = unsafe {
        launch(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_stride,
            k_stride,
            v_stride,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            pool_ptr as *mut ffi::Half,
            geometry.k_offset_elems,
            geometry.v_offset_elems,
            pi_ptr as *const i32,
            page_indices_len,
            ip_ptr as *const i32,
            og_ptr as *const i32,
            ps_ptr as *const i32,
            num_q_heads_i32,
            num_kv_heads_i32,
            batch_i32,
            cos_max_pos_i32,
            rotary_dim_i32,
            rms_eps,
            geometry.page_size,
            geometry.num_pages,
            geometry.stride_page,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "{entry_point} failed with error {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// QK RMSNorm + partial RoPE for the hd512 paged-prefill prep (Gemma 4
/// global layers). Q is normalised + partially rotated into a contiguous
/// `q_out`; K is normalised + partially rotated straight into the paged KV
/// pool at layer `layer` (feeds `batch_prefill_paged`); V — the K=V fork —
/// is the weightless RMS norm of the same raw K, written in the same pass.
/// No gate; plain-w RMSNorm.
///
/// The rotation is the engine's proportional one over `cos_max_pos` rows of
/// 512: pairs `(d, d + 256)` with the live angles first and the identity
/// past them. How the row is laid out — K and V as two blocks, or the
/// folded row of `KvFormat::Folded` — is the layout's format, and the
/// kernel is told it as bands rather than asked to know the format. `q` and
/// `k` are column bands read at their own row strides, so they may share one
/// fused Q|K projection row.
///
/// The kernel `__trap()`s on any out-of-range pos or page id as the
/// second layer of the host validation's defence.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_paged_prefill_hd512_into<'a>(
    ctx: &DeviceContext,
    q: impl Into<Columns<'a>>,
    k: impl Into<Columns<'a>>,
    q_out: &mut HiddenStates,
    row_offset: usize,
    kv_pool: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    layer: usize,
    page_indices: &CudaSlice<i32>,
    pages_offset: usize,
    start_pos: usize,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rms_eps: f32,
) -> Result<()> {
    let (q, k) = (q.into(), k.into());
    // Same suffix-window contract as the hd256 prefill prep: the segment
    // is the rows `[row_offset..seq_len)` with its table at `pages_offset`.
    anyhow::ensure!(
        row_offset < q.states.seq_len,
        "hd512 prefill prep row_offset {row_offset} leaves no rows of {}",
        q.states.seq_len
    );
    let seq_len = q.states.seq_len - row_offset;
    anyhow::ensure!(
        pages_offset < page_indices.len(),
        "hd512 prefill prep pages_offset {pages_offset} exceeds table len {}",
        page_indices.len()
    );
    let q_dim = num_q_heads.checked_mul(512).ok_or_else(|| {
        anyhow::anyhow!("hd512 prefill prep num_q_heads {num_q_heads} * 512 overflows")
    })?;
    let kv_dim = num_kv_heads.checked_mul(512).ok_or_else(|| {
        anyhow::anyhow!("hd512 prefill prep num_kv_heads {num_kv_heads} * 512 overflows")
    })?;
    anyhow::ensure!(
        q.width == q_dim,
        "hd512 prefill prep q width {} != num_q_heads {num_q_heads} * 512",
        q.width
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.width,
        "hd512 prefill prep q_out.hidden_dim {} != q width {}",
        q_out.hidden_dim,
        q.width
    );
    anyhow::ensure!(
        q_out.seq_len == q.states.seq_len,
        "hd512 prefill prep q_out.seq_len {} != q rows {}",
        q_out.seq_len,
        q.states.seq_len
    );
    anyhow::ensure!(
        k.width == kv_dim,
        "hd512 prefill prep k width {} != num_kv_heads {num_kv_heads} * 512",
        k.width
    );
    anyhow::ensure!(
        k.states.seq_len == q.states.seq_len,
        "hd512 prefill prep k rows {} != q rows {}",
        k.states.seq_len,
        q.states.seq_len
    );
    let target = hd512_prep_target(
        "hd512 prefill prep",
        layout,
        kv_pool.len(),
        layer,
        num_kv_heads,
    )?;
    anyhow::ensure!(
        q_norm_weight.len == 512,
        "hd512 prefill prep q_norm_weight len {} != 512",
        q_norm_weight.len
    );
    anyhow::ensure!(
        k_norm_weight.len == 512,
        "hd512 prefill prep k_norm_weight len {} != 512",
        k_norm_weight.len
    );
    let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
        anyhow::anyhow!("hd512 prefill prep start_pos {start_pos} + seq_len {seq_len} overflows")
    })?;
    let covered = (page_indices.len() - pages_offset)
        .checked_mul(layout.page_size)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "hd512 prefill prep page window {} * page_size {} overflows",
                page_indices.len() - pages_offset,
                layout.page_size
            )
        })?;
    anyhow::ensure!(
        covered >= end_pos,
        "hd512 prefill prep pages cover {covered} tokens, need start_pos {start_pos} + seq_len {seq_len}"
    );
    anyhow::ensure!(
        end_pos <= cos_max_pos,
        "hd512 prefill prep start_pos {start_pos} + seq_len {seq_len} > cos_max_pos {cos_max_pos}"
    );
    let table_len = cos_max_pos.checked_mul(512).ok_or_else(|| {
        anyhow::anyhow!("hd512 prefill prep cos_max_pos {cos_max_pos} * 512 overflows")
    })?;
    anyhow::ensure!(
        cos_cache.len >= table_len,
        "hd512 prefill prep cos_cache len {} < cos_max_pos {cos_max_pos} * 512",
        cos_cache.len
    );
    anyhow::ensure!(
        sin_cache.len >= table_len,
        "hd512 prefill prep sin_cache len {} < cos_max_pos {cos_max_pos} * 512",
        sin_cache.len
    );
    let q_elems = q.states.checked_extent("hd512 prefill prep q")?;
    q_out.checked_extent("hd512 prefill prep q_out")?;
    let k_elems = k.states.checked_extent("hd512 prefill prep k")?;
    crate::ops::checked_i32(q_elems, "hd512 prefill prep q extent")?;
    crate::ops::checked_i32(k_elems, "hd512 prefill prep k extent")?;
    crate::ops::checked_i32(table_len, "hd512 prefill prep rope table extent")?;
    ensure_vec_backed(q_norm_weight, "hd512 prefill prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd512 prefill prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd512 prefill prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd512 prefill prep sin_cache")?;

    let num_q_heads_i32 = crate::ops::checked_i32(num_q_heads, "hd512 prefill prep num_q_heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(num_kv_heads, "hd512 prefill prep num_kv_heads")?;
    let seq_len_i32 = crate::ops::checked_i32(seq_len, "hd512 prefill prep seq_len")?;
    let start_pos_i32 = crate::ops::checked_i32(start_pos, "hd512 prefill prep start_pos")?;
    let cos_max_pos_i32 = crate::ops::checked_i32(cos_max_pos, "hd512 prefill prep cos_max_pos")?;

    let page_indices_len = crate::ops::checked_i32(
        page_indices.len() - pages_offset,
        "hd512 paged prep page window len",
    )?;
    let q_row_bytes = checked_band_offset(&q, row_offset, seq_len, "hd512 prefill prep q")?;
    let k_row_bytes = checked_band_offset(&k, row_offset, seq_len, "hd512 prefill prep k")?;
    let qo_row_bytes = checked_row_offset(q_out, row_offset, seq_len, "hd512 prefill prep q_out")?;
    let (q_ptr, _gq) = q.states.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_row_bytes;
    let (k_ptr, _gk) = k.states.data.device_ptr(&ctx.stream);
    let k_ptr = k_ptr + k_row_bytes;
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let qo_ptr = qo_ptr + qo_row_bytes;
    // The KV state owns mutation; this call borrows its shared pool handle.
    let (pool_ptr, _gp) = kv_pool.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices.device_ptr(&ctx.stream);
    let pi_ptr = pi_ptr + (pages_offset * std::mem::size_of::<i32>()) as u64;

    let q_stride = crate::ops::checked_i32(q.states.hidden_dim, "hd512 prefill prep q stride")?;
    let k_stride = crate::ops::checked_i32(k.states.hidden_dim, "hd512 prefill prep k stride")?;
    let result = unsafe {
        ffi::qk_norm_partial_rope_paged_prefill_hd512_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            q_stride,
            k_stride,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            pool_ptr as *mut ffi::Half,
            target.k_offset_elems,
            target.v_offset_elems,
            pi_ptr as *const i32,
            page_indices_len,
            num_q_heads_i32,
            num_kv_heads_i32,
            seq_len_i32,
            start_pos_i32,
            cos_max_pos_i32,
            target.row_width,
            target.fold_rotary,
            rms_eps,
            target.page_size,
            target.num_pages,
            target.stride_page,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "qk_norm_partial_rope_paged_prefill_hd512_cuda failed with error \
             {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}

/// Per-request hd512 prep for the non-evicting global cache; V is the K fork.
/// Invalid device metadata traps before the first paged-pool access.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_partial_rope_paged_decode_hd512_into<'a>(
    ctx: &DeviceContext,
    q: impl Into<Columns<'a>>,
    k: impl Into<Columns<'a>>,
    q_out: &mut HiddenStates,
    row_offset: usize,
    kv_pool: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    layer: usize,
    page_indices: &CudaSlice<i32>,
    page_indptr: &CudaSlice<i32>,
    page_origins: &CudaSlice<i32>,
    positions: &CudaSlice<i32>,
    cos_max_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rms_eps: f32,
) -> Result<()> {
    let (q, k) = (q.into(), k.into());
    anyhow::ensure!(
        row_offset < q.states.seq_len,
        "hd512 paged decode prep row_offset {row_offset} leaves no rows of {}",
        q.states.seq_len
    );
    let batch = q.states.seq_len - row_offset;
    anyhow::ensure!(
        Some(q.width) == num_q_heads.checked_mul(512),
        "hd512 paged decode prep q width {} != num_q_heads {} * 512",
        q.width,
        num_q_heads
    );
    anyhow::ensure!(
        q_out.hidden_dim == q.width && q_out.seq_len == q.states.seq_len,
        "hd512 paged decode prep q_out [{} x {}] != q [{} x {}]",
        q_out.hidden_dim,
        q_out.seq_len,
        q.width,
        q.states.seq_len
    );
    anyhow::ensure!(
        Some(k.width) == num_kv_heads.checked_mul(512) && k.states.seq_len == q.states.seq_len,
        "hd512 paged decode prep k [{} x {}] != [num_kv_heads {} * 512 x {}]",
        k.width,
        k.states.seq_len,
        num_kv_heads,
        q.states.seq_len
    );
    let target = hd512_prep_target(
        "hd512 paged decode prep",
        layout,
        kv_pool.len(),
        layer,
        num_kv_heads,
    )?;
    ensure_vec_backed(q_norm_weight, "hd512 paged decode prep q_norm_weight")?;
    ensure_vec_backed(k_norm_weight, "hd512 paged decode prep k_norm_weight")?;
    ensure_vec_backed(cos_cache, "hd512 paged decode prep cos_cache")?;
    ensure_vec_backed(sin_cache, "hd512 paged decode prep sin_cache")?;
    anyhow::ensure!(
        q_norm_weight.len == 512 && k_norm_weight.len == 512,
        "hd512 paged decode prep norm weight lens {} / {} != 512",
        q_norm_weight.len,
        k_norm_weight.len
    );
    let table_rows = cos_max_pos.checked_mul(512).ok_or_else(|| {
        anyhow::anyhow!("hd512 paged decode prep cos_max_pos {cos_max_pos} * 512 overflows")
    })?;
    crate::ops::checked_i32(table_rows, "hd512 paged decode prep rope table extent")?;
    anyhow::ensure!(
        cos_cache.len >= table_rows && sin_cache.len >= table_rows,
        "hd512 paged decode prep cos/sin lens {} / {} < cos_max_pos {cos_max_pos} * 512",
        cos_cache.len,
        sin_cache.len
    );
    anyhow::ensure!(
        positions.len() >= batch && page_indptr.len() > batch && page_origins.len() >= batch,
        "hd512 paged decode prep metadata lens (positions {}, indptr {}, origins {}) do \
         not cover batch {batch}",
        positions.len(),
        page_indptr.len(),
        page_origins.len()
    );
    let page_indices_len = crate::ops::checked_i32(
        page_indices.len(),
        "hd512 paged decode prep page_indices len",
    )?;
    let q_elems = q.states.checked_extent("hd512 paged decode prep q")?;
    q_out.checked_extent("hd512 paged decode prep q_out")?;
    let k_elems = k.states.checked_extent("hd512 paged decode prep k")?;
    crate::ops::checked_i32(q_elems, "hd512 paged decode prep q extent")?;
    crate::ops::checked_i32(k_elems, "hd512 paged decode prep k extent")?;
    let num_q_heads_i32 = crate::ops::checked_i32(num_q_heads, "hd512 paged decode prep q heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(num_kv_heads, "hd512 paged decode prep kv heads")?;
    let batch_i32 = crate::ops::checked_i32(batch, "hd512 paged decode prep batch")?;
    let cos_max_pos_i32 =
        crate::ops::checked_i32(cos_max_pos, "hd512 paged decode prep cos_max_pos")?;
    let q_row_bytes = checked_band_offset(&q, row_offset, batch, "hd512 paged decode prep q")?;
    let k_row_bytes = checked_band_offset(&k, row_offset, batch, "hd512 paged decode prep k")?;
    let qo_row_bytes =
        checked_row_offset(q_out, row_offset, batch, "hd512 paged decode prep q_out")?;
    let (q_ptr, _gq) = q.states.data.device_ptr(&ctx.stream);
    let q_ptr = q_ptr + q_row_bytes;
    let (k_ptr, _gk) = k.states.data.device_ptr(&ctx.stream);
    let k_ptr = k_ptr + k_row_bytes;
    let (qo_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let qo_ptr = qo_ptr + qo_row_bytes;
    // The KV states own mutation; this call borrows the shared pool handle.
    let (pool_ptr, _gp) = kv_pool.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = page_indices.device_ptr(&ctx.stream);
    let (ip_ptr, _gip) = page_indptr.device_ptr(&ctx.stream);
    let (po_ptr, _gpo) = page_origins.device_ptr(&ctx.stream);
    let (ps_ptr, _gps) = positions.device_ptr(&ctx.stream);

    let q_stride =
        crate::ops::checked_i32(q.states.hidden_dim, "hd512 paged decode prep q stride")?;
    let k_stride =
        crate::ops::checked_i32(k.states.hidden_dim, "hd512 paged decode prep k stride")?;
    let result = unsafe {
        ffi::qk_norm_partial_rope_paged_decode_hd512_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            q_stride,
            k_stride,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            qo_ptr as *mut ffi::Half,
            pool_ptr as *mut ffi::Half,
            target.k_offset_elems,
            target.v_offset_elems,
            pi_ptr as *const i32,
            page_indices_len,
            ip_ptr as *const i32,
            po_ptr as *const i32,
            ps_ptr as *const i32,
            num_q_heads_i32,
            num_kv_heads_i32,
            batch_i32,
            cos_max_pos_i32,
            target.row_width,
            target.fold_rotary,
            rms_eps,
            target.page_size,
            target.num_pages,
            target.stride_page,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        anyhow::bail!(
            "qk_norm_partial_rope_paged_decode_hd512_cuda failed with error \
             {result}{}",
            crate::ops::ffi_exception_message(result)
        );
    }
    Ok(())
}
