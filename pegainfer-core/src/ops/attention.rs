use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::PrefillPagedPlan;
use crate::kv_pool::KvLayout;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_into(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    kv_buffer: &CudaSlice<bf16>,
    layout: &KvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    pegainfer_kernels::ops::prefill_attention_paged_into(
        ctx,
        q_batch,
        k_batch,
        v_batch,
        q_norm,
        k_norm,
        cos_cache,
        sin_cache,
        kv_buffer,
        &layout.kernel_layout(),
        layer,
        plan,
        output,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rms_eps,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &KvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    pegainfer_kernels::ops::paged_attention_batch_decode_into(
        ctx,
        q,
        k,
        v,
        kv_buffer,
        &layout.kernel_layout(),
        layer,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        positions_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        output,
        num_qo_heads,
        batch_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_split_kv_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    row_offset: usize,
    kv_buffer: &CudaSlice<bf16>,
    layout: &KvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    split_request_indices_d: &CudaSlice<i32>,
    split_kv_tile_indices_d: &CudaSlice<i32>,
    split_kv_chunk_size_d: &CudaSlice<i32>,
    split_o_indptr_d: &CudaSlice<i32>,
    split_block_valid_mask_d: &CudaSlice<u8>,
    split_tmp_v: &mut CudaSlice<bf16>,
    split_tmp_s: &mut CudaSlice<f32>,
    split_padded_slots: usize,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    pegainfer_kernels::ops::paged_attention_batch_decode_split_kv_into(
        ctx,
        q,
        k,
        v,
        row_offset,
        kv_buffer,
        &layout.kernel_layout(),
        layer,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        positions_d,
        request_indices_d,
        split_request_indices_d,
        split_kv_tile_indices_d,
        split_kv_chunk_size_d,
        split_o_indptr_d,
        split_block_valid_mask_d,
        split_tmp_v,
        split_tmp_s,
        split_padded_slots,
        output,
        num_qo_heads,
        batch_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &KvLayout,
    layer: usize,
    page_indices_d: &CudaSlice<i32>,
    page_indptr_d: &CudaSlice<i32>,
    last_page_len_d: &CudaSlice<i32>,
    positions_d: &CudaSlice<i32>,
    request_indices_d: &CudaSlice<i32>,
    kv_tile_indices_d: &CudaSlice<i32>,
    kv_chunk_size_d: &CudaSlice<i32>,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    batch_size: usize,
) -> Result<()> {
    pegainfer_kernels::ops::paged_attention_batch_decode_hd256_into(
        ctx,
        q,
        k,
        v,
        kv_buffer,
        &layout.kernel_layout(),
        layer,
        page_indices_d,
        page_indptr_d,
        last_page_len_d,
        positions_d,
        request_indices_d,
        kv_tile_indices_d,
        kv_chunk_size_d,
        output,
        num_qo_heads,
        batch_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_decode_via_prefill_hd256_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &KvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_qo_heads: usize,
) -> Result<()> {
    pegainfer_kernels::ops::paged_attention_batch_decode_via_prefill_hd256_into(
        ctx,
        q,
        k,
        v,
        kv_buffer,
        &layout.kernel_layout(),
        layer,
        plan,
        output,
        num_qo_heads,
    )
}
