use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;
use crate::tensor::HiddenStatesRef;

/// Batched element-wise add: out = a + b (same shape HiddenStates)
pub fn add_batch(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch_into(ctx, a, b, &mut out)?;
    Ok(out)
}

/// Batched element-wise add into pre-allocated output buffer (zero allocation).
pub fn add_batch_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.seq_len, b.seq_len);
    assert_eq!(out.hidden_dim, a.hidden_dim);
    assert_eq!(out.seq_len, a.seq_len);

    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Advance the per-row decode tables written by a regular graph replay.
pub fn advance_decode_metadata(
    ctx: &DeviceContext,
    positions: &mut CudaSlice<i32>,
    local_last: &mut CudaSlice<i32>,
    pseudo_last: &mut CudaSlice<i32>,
    kv_chunk: &mut CudaSlice<i32>,
    rows: usize,
    factor: usize,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "advance_decode_metadata: rows must be positive");
    anyhow::ensure!(
        factor > 0,
        "advance_decode_metadata: factor must be positive"
    );
    let pseudo_rows = rows
        .checked_mul(factor)
        .filter(|n| i32::try_from(*n).is_ok())
        .ok_or_else(|| anyhow!("advance_decode_metadata: rows x factor exceeds i32"))?;
    anyhow::ensure!(
        positions.len() >= rows
            && local_last.len() >= rows
            && kv_chunk.len() >= rows
            && pseudo_last.len() >= pseudo_rows,
        "advance_decode_metadata: {rows} rows x {factor} exceeds a table"
    );
    let rows = super::checked_i32(rows, "advance decode metadata rows")?;
    let factor = super::checked_i32(factor, "advance decode metadata factor")?;
    let (positions_ptr, _positions_guard) = positions.device_ptr_mut(&ctx.stream);
    let (local_ptr, _local_guard) = local_last.device_ptr_mut(&ctx.stream);
    let (pseudo_ptr, _pseudo_guard) = pseudo_last.device_ptr_mut(&ctx.stream);
    let (chunk_ptr, _chunk_guard) = kv_chunk.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::advance_decode_metadata_cuda(
            positions_ptr as *mut i32,
            local_ptr as *mut i32,
            pseudo_ptr as *mut i32,
            chunk_ptr as *mut i32,
            rows,
            factor,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Element-wise add of `n` bf16 elements into a pre-allocated output
/// (`out = a + b`). Slice-level twin of [`add_batch_into`] — same kernel —
/// for callers whose buffers live in a persistent decode arena rather than
/// owned `HiddenStates`.
pub fn add_into(
    ctx: &DeviceContext,
    a: &CudaSlice<half::bf16>,
    b: &CudaSlice<half::bf16>,
    n: usize,
    out: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    if a.len() < n || b.len() < n || out.len() < n {
        return Err(anyhow!(
            "add_into buffers too small for n={n}: a {}, b {}, out {}",
            a.len(),
            b.len(),
            out.len()
        ));
    }
    let (a_ptr, _ga) = a.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// `out = bf16(shared + bf16(scale * routed))`.
///
/// The intermediate BF16 narrowing matches vLLM's CUDA MoE path, which
/// scales the reduced routed output in place before adding the shared expert.
pub fn add_scaled_bf16_into(
    ctx: &DeviceContext,
    routed: &CudaSlice<half::bf16>,
    scale: f32,
    shared: &CudaSlice<half::bf16>,
    n: usize,
    out: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    if !scale.is_finite() {
        return Err(anyhow!("add_scaled_bf16_into scale must be finite"));
    }
    if routed.len() < n || shared.len() < n || out.len() < n {
        return Err(anyhow!(
            "add_scaled_bf16_into buffers too small for n={n}: routed {}, shared {}, out {}",
            routed.len(),
            shared.len(),
            out.len()
        ));
    }
    let (routed_ptr, _routed_guard) = routed.device_ptr(&ctx.stream);
    let (shared_ptr, _shared_guard) = shared.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::add_scaled_bf16_cuda(
            routed_ptr as *const ffi::Half,
            scale,
            shared_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// In-place scaled add into a row range of `out`: out[row_offset..] += scale * delta.
pub fn scaled_add_rows_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
    row_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_into scale must be finite"
    );
    assert_eq!(
        delta.seq_len, out.seq_len,
        "delta seq_len {} != out seq_len {}",
        delta.seq_len, out.seq_len
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scaled_add_rows_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            delta.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// In-place scaled add into a row range of a contiguous token range in `out`.
pub fn scaled_add_rows_token_range_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
    row_offset: usize,
    token_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_token_range_into scale must be finite"
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );
    assert!(
        token_offset + delta.seq_len <= out.seq_len,
        "token range [{}..{}) exceeds out seq_len {}",
        token_offset,
        token_offset + delta.seq_len,
        out.seq_len
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (out_base, _go) = out.data.device_ptr_mut(&ctx.stream);
    let out_ptr =
        out_base + (token_offset * out.hidden_dim * std::mem::size_of::<half::bf16>()) as u64;
    let result = unsafe {
        ffi::scaled_add_rows_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            delta.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

pub fn gather_hidden_tokens_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    token_indices: &impl DevicePtr<i32>,
    token_count: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(
        out.hidden_dim, input.hidden_dim,
        "gather output hidden_dim {} != input hidden_dim {}",
        out.hidden_dim, input.hidden_dim
    );
    assert_eq!(
        out.seq_len, token_count,
        "gather output seq_len {} != token_count {}",
        out.seq_len, token_count
    );
    assert!(
        token_count <= token_indices.len(),
        "token_count {} exceeds indices len {}",
        token_count,
        token_indices.len()
    );
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (indices_ptr, _gidx) = token_indices.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gather_hidden_tokens_cuda(
            input_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            input.hidden_dim as i32,
            token_count as i32,
            input.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn copy_hidden_rows_into(
    ctx: &DeviceContext,
    src: &HiddenStates,
    dst: &mut HiddenStates,
    row_offset: usize,
) -> Result<()> {
    assert_eq!(
        src.seq_len, dst.seq_len,
        "copy_hidden_rows_into seq_len mismatch: src {}, dst {}",
        src.seq_len, dst.seq_len
    );
    copy_hidden_rows_raw_into(
        ctx,
        &src.data,
        src.hidden_dim,
        &mut dst.data,
        dst.hidden_dim,
        row_offset,
        src.seq_len,
    )
}

/// [`copy_hidden_rows_into`] over raw device slices: per token, copy the
/// `src_dim` features of `src` (row stride `src_dim`) into `dst` at feature
/// offset `row_offset` (row stride `dst_dim`), for `tokens` tokens.
/// Inverse of [`copy_hidden_rows_raw_into`]: extract the column window
/// `[col_offset, col_offset + dst_dim)` of each token's `src_dim`-wide row
/// into a compact `dst_dim`-wide destination row — the packed-projection
/// output split (#812).
pub fn extract_hidden_rows_raw_into(
    ctx: &DeviceContext,
    src: &CudaSlice<half::bf16>,
    src_dim: usize,
    dst: &mut CudaSlice<half::bf16>,
    dst_dim: usize,
    col_offset: usize,
    tokens: usize,
) -> Result<()> {
    assert!(
        col_offset + dst_dim <= src_dim,
        "column window [{}..{}) exceeds source hidden_dim {}",
        col_offset,
        col_offset + dst_dim,
        src_dim
    );
    assert!(
        tokens * src_dim <= src.len() && tokens * dst_dim <= dst.len(),
        "extract_hidden_rows_raw_into token count {} exceeds src {} / dst {} capacity",
        tokens,
        src.len(),
        dst.len()
    );

    let (src_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::extract_hidden_rows_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            src_dim as i32,
            dst_dim as i32,
            col_offset as i32,
            dst_dim as i32,
            tokens as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("extract_hidden_rows launch failed: {err}"))
}

pub fn copy_hidden_rows_raw_into(
    ctx: &DeviceContext,
    src: &CudaSlice<half::bf16>,
    src_dim: usize,
    dst: &mut CudaSlice<half::bf16>,
    dst_dim: usize,
    row_offset: usize,
    tokens: usize,
) -> Result<()> {
    assert!(
        row_offset + src_dim <= dst_dim,
        "row range [{}..{}) exceeds destination hidden_dim {}",
        row_offset,
        row_offset + src_dim,
        dst_dim
    );
    assert!(
        tokens * src_dim <= src.len() && tokens * dst_dim <= dst.len(),
        "copy_hidden_rows_raw_into token count {} exceeds src {} / dst {} capacity",
        tokens,
        src.len(),
        dst.len()
    );

    let (src_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::copy_hidden_rows_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            src_dim as i32,
            dst_dim as i32,
            row_offset as i32,
            src_dim as i32,
            tokens as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Copy `[rows, hidden_dim]`, replacing every row whose position is zero with
/// zeros. This is the MTP embedding-mask boundary: position zero has no
/// previous token and must not contribute its embedding.
pub fn mask_position_zero_rows_into(
    ctx: &DeviceContext,
    src: &CudaSlice<half::bf16>,
    positions: &CudaSlice<u32>,
    hidden_dim: usize,
    rows: usize,
    dst: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    assert!(
        rows * hidden_dim <= src.len() && rows * hidden_dim <= dst.len(),
        "mask_position_zero_rows_into rows {rows} x hidden_dim {hidden_dim} exceed src {} / dst {}",
        src.len(),
        dst.len()
    );
    assert!(
        rows <= positions.len(),
        "mask_position_zero_rows_into rows {rows} exceed positions {}",
        positions.len()
    );

    let (src_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (positions_ptr, _gp) = positions.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::mask_position_zero_rows_cuda(
            src_ptr as *const ffi::Half,
            positions_ptr as *const u32,
            dst_ptr as *mut ffi::Half,
            hidden_dim as i32,
            rows as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn copy_hidden_token_range_into(
    ctx: &DeviceContext,
    src: &HiddenStates,
    src_token_offset: usize,
    dst: &mut HiddenStates,
    dst_token_offset: usize,
    token_count: usize,
) -> Result<()> {
    assert_eq!(
        src.hidden_dim, dst.hidden_dim,
        "copy_hidden_token_range_into hidden_dim mismatch: src {}, dst {}",
        src.hidden_dim, dst.hidden_dim
    );
    assert!(
        src_token_offset + token_count <= src.seq_len,
        "source token range [{}..{}) exceeds seq_len {}",
        src_token_offset,
        src_token_offset + token_count,
        src.seq_len
    );
    assert!(
        dst_token_offset + token_count <= dst.seq_len,
        "destination token range [{}..{}) exceeds seq_len {}",
        dst_token_offset,
        dst_token_offset + token_count,
        dst.seq_len
    );

    let (src_ptr, _gs) = src.data.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::copy_hidden_token_range_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            src.hidden_dim as i32,
            src_token_offset as i32,
            dst_token_offset as i32,
            token_count as i32,
            src.seq_len as i32,
            dst.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn scaled_add_rows_indexed_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    token_indices: &CudaSlice<i32>,
    token_count: usize,
    out: &mut HiddenStates,
    row_offset: usize,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "scaled_add_rows_indexed_into scale must be finite"
    );
    assert_eq!(
        delta.seq_len, token_count,
        "delta seq_len {} != token_count {}",
        delta.seq_len, token_count
    );
    assert!(
        token_count <= token_indices.len(),
        "token_count {} exceeds indices len {}",
        token_count,
        token_indices.len()
    );
    assert!(
        row_offset + delta.hidden_dim <= out.hidden_dim,
        "row range [{}..{}) exceeds out hidden_dim {}",
        row_offset,
        row_offset + delta.hidden_dim,
        out.hidden_dim
    );

    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (indices_ptr, _gidx) = token_indices.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scaled_add_rows_indexed_cuda(
            delta_ptr as *const ffi::Half,
            scale,
            indices_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            out.hidden_dim as i32,
            row_offset as i32,
            delta.hidden_dim as i32,
            token_count as i32,
            out.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// In-place scaled add for tensors with identical shape.
pub fn scaled_add_batch_into(
    ctx: &DeviceContext,
    delta: &HiddenStates,
    scale: f32,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(delta.hidden_dim, out.hidden_dim);
    scaled_add_rows_into(ctx, delta, scale, out, 0)
}

pub fn bf16_hidden_to_f32_into(
    ctx: &DeviceContext,
    input: &HiddenStates,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    assert!(
        output.len() >= input.data.len(),
        "f32 output len {} < bf16 input len {}",
        output.len(),
        input.data.len()
    );
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (output_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::bf16_to_f32_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut f32,
            input.data.len() as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Cast bf16 logits (as raw `CudaSlice<u8>`) to f32.
/// Used by the indexer forward to convert DeepGEMM's bf16 logits output
/// before FlashInfer top-k (which expects f32).
pub fn bf16_bytes_to_f32_into(
    ctx: &DeviceContext,
    input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    let n = input.len() / 2;
    anyhow::ensure!(
        output.len() >= n,
        "f32 output len {} < bf16 input len {}",
        output.len(),
        n
    );
    let (input_ptr, _gi) = input.device_ptr(&ctx.stream);
    let (output_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::bf16_to_f32_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut f32,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn f32_to_bf16_hidden_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    output: &mut HiddenStates,
) -> Result<()> {
    assert!(
        input.len() >= output.data.len(),
        "f32 input len {} < bf16 output len {}",
        input.len(),
        output.data.len()
    );
    let n = output.data.len();
    let (input_ptr, _gi) = input.device_ptr(&ctx.stream);
    let (output_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::f32_to_bf16_cuda(
            input_ptr as *const f32,
            output_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn scale_f32_in_place(
    ctx: &DeviceContext,
    values: &mut CudaSlice<f32>,
    len: usize,
    scale: f32,
) -> Result<()> {
    assert!(
        len <= values.len(),
        "scale_f32_in_place len {} exceeds values len {}",
        len,
        values.len()
    );
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::scale_f32_cuda(
            values_ptr as *mut f32,
            scale,
            len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn accumulate_bf16_token_scaled_to_f32_into(
    ctx: &DeviceContext,
    token: &HiddenStates,
    scale: f32,
    token_idx: usize,
    seq_len: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    assert!(
        scale.is_finite(),
        "accumulate_bf16_token_scaled_to_f32_into scale must be finite"
    );
    assert_eq!(
        token.seq_len, 1,
        "accumulate_bf16_token_scaled_to_f32_into expects one token, got seq_len={}",
        token.seq_len
    );
    assert!(
        token_idx < seq_len,
        "accumulate token_idx {} exceeds seq_len {}",
        token_idx,
        seq_len
    );
    assert!(
        out.len() >= token.hidden_dim * seq_len,
        "f32 output len {} < hidden_dim {} * seq_len {}",
        out.len(),
        token.hidden_dim,
        seq_len
    );
    let (token_ptr, _gt) = token.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::accumulate_bf16_token_scaled_to_f32_cuda(
            token_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut f32,
            token.hidden_dim as i32,
            token_idx as i32,
            seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn repeat_f32_for_reduce_scatter_into(
    ctx: &DeviceContext,
    local: &CudaSlice<f32>,
    repeated: &mut CudaSlice<f32>,
    local_elems: usize,
    world_size: usize,
) -> Result<()> {
    assert!(
        local_elems <= local.len(),
        "repeat_f32 local_elems {} exceeds local len {}",
        local_elems,
        local.len()
    );
    assert!(
        repeated.len() >= local_elems * world_size,
        "repeat_f32 repeated len {} < local_elems {} * world_size {}",
        repeated.len(),
        local_elems,
        world_size
    );
    let (local_ptr, _local_guard) = local.device_ptr(&ctx.stream);
    let (repeated_ptr, _repeated_guard) = repeated.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::repeat_f32_for_reduce_scatter_cuda(
            local_ptr as *const f32,
            repeated_ptr as *mut f32,
            local_elems as i32,
            world_size as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched SiLU+mul: out[i] = silu(gate[i]) * up[i]
pub fn silu_mul_batch(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    silu_mul_batch_into(ctx, gate, up, &mut out)?;
    Ok(out)
}

/// Batched SiLU+mul into pre-allocated output buffer (zero allocation).
pub fn silu_mul_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    assert_eq!(out.hidden_dim, gate.hidden_dim);
    assert_eq!(out.seq_len, gate.seq_len);

    let n = gate.hidden_dim * gate.seq_len;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_triton_aot_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Fused SiLU-mul from combined [2*I, bs] gate+up buffer → [I, bs] output.
/// Reads gate and up from interleaved column-major layout, no deinterleave needed.
pub fn silu_mul_fused_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let intermediate_size = out.hidden_dim;
    let bs = gate_up.seq_len;
    assert_eq!(
        gate_up.hidden_dim,
        2 * intermediate_size,
        "gate_up dim {} != 2 * out dim {}",
        gate_up.hidden_dim,
        intermediate_size
    );
    assert_eq!(out.seq_len, bs);

    let (gu_ptr, _g0) = gate_up.data.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            intermediate_size as i32,
            bs as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if result != 0 {
        return Err(anyhow!(
            "silu_mul_fused CUDA launch failed: cuda_status={}, intermediate={}, batch={}",
            result,
            intermediate_size,
            bs
        ));
    }
    Ok(())
}

/// Split contiguous `[Q; K; V]` projection output into compact Q/K/V tensors.
///
/// This is a BF16 bitwise copy. The checked shape boundary matters because a
/// wrong local TP dimension would otherwise corrupt the following attention
/// buffers and surface only at a later CUDA call.
pub fn split_qkv_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    v: &mut HiddenStates,
) -> Result<()> {
    anyhow::ensure!(
        q.hidden_dim > 0 && k.hidden_dim > 0 && qkv.seq_len > 0,
        "split_qkv requires non-empty Q, KV, and token dimensions; got q_dim={}, kv_dim={}, tokens={}",
        q.hidden_dim,
        k.hidden_dim,
        qkv.seq_len
    );
    anyhow::ensure!(
        k.hidden_dim == v.hidden_dim,
        "K/V dimensions must match: k_dim={}, v_dim={}",
        k.hidden_dim,
        v.hidden_dim
    );
    anyhow::ensure!(
        qkv.hidden_dim == q.hidden_dim + k.hidden_dim + v.hidden_dim,
        "QKV input dimension {} must equal Q + K + V ({} + {} + {})",
        qkv.hidden_dim,
        q.hidden_dim,
        k.hidden_dim,
        v.hidden_dim
    );
    anyhow::ensure!(
        qkv.seq_len == q.seq_len,
        "QKV/Q token count mismatch: qkv={}, q={}",
        qkv.seq_len,
        q.seq_len
    );
    anyhow::ensure!(
        qkv.seq_len == k.seq_len,
        "QKV/K token count mismatch: qkv={}, k={}",
        qkv.seq_len,
        k.seq_len
    );
    anyhow::ensure!(
        qkv.seq_len == v.seq_len,
        "QKV/V token count mismatch: qkv={}, v={}",
        qkv.seq_len,
        v.seq_len
    );

    let q_dim = i32::try_from(q.hidden_dim)
        .map_err(|_| anyhow!("Q dimension {} exceeds i32", q.hidden_dim))?;
    let kv_dim = i32::try_from(k.hidden_dim)
        .map_err(|_| anyhow!("KV dimension {} exceeds i32", k.hidden_dim))?;
    let tokens = i32::try_from(qkv.seq_len)
        .map_err(|_| anyhow!("QKV token count {} exceeds i32", qkv.seq_len))?;
    i32::try_from(qkv.hidden_dim)
        .map_err(|_| anyhow!("QKV dimension {} exceeds i32", qkv.hidden_dim))?;
    i32::try_from(
        qkv.hidden_dim
            .checked_mul(qkv.seq_len)
            .ok_or_else(|| anyhow!("QKV element count overflow"))?,
    )
    .map_err(|_| {
        anyhow!(
            "QKV element count {}×{} exceeds CUDA kernel i32 indexing",
            qkv.hidden_dim,
            qkv.seq_len
        )
    })?;

    let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
    let (q_ptr, _gq) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::split_qkv_cuda(
            qkv_ptr as *const ffi::Half,
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *mut ffi::Half,
            q_dim,
            kv_dim,
            tokens,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if status != 0 {
        return Err(anyhow!(
            "split_qkv CUDA launch failed: cuda_status={status}, q_dim={}, kv_dim={}, tokens={}",
            q.hidden_dim,
            k.hidden_dim,
            qkv.seq_len
        ));
    }
    Ok(())
}

/// Extract a single token's vector from a HiddenStates batch (GPU copy)
pub fn extract_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<DeviceVec> {
    extract_vec_ref(ctx, batch.as_ref(), token_idx)
}

/// Extract a single token's vector from a borrowed HiddenStates batch.
pub fn extract_vec_ref(
    ctx: &DeviceContext,
    batch: HiddenStatesRef<'_>,
    token_idx: usize,
) -> Result<DeviceVec> {
    let len = batch.hidden_dim;
    let mut out = DeviceVec::zeros(ctx, len)?;
    extract_vec_ref_into(ctx, batch, token_idx, &mut out)?;
    Ok(out)
}

/// Copy one column from `batch` into a pre-allocated `out`.
pub fn extract_vec_into(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    extract_vec_ref_into(ctx, batch.as_ref(), token_idx, out)
}

/// Copy one column from a borrowed `batch` into a pre-allocated `out`.
pub fn extract_vec_ref_into(
    ctx: &DeviceContext,
    batch: HiddenStatesRef<'_>,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    let len = batch.hidden_dim;
    anyhow::ensure!(out.len == len, "extract_vec_into len mismatch");
    anyhow::ensure!(
        token_idx < batch.seq_len,
        "extract_vec_into token index {token_idx} out of bounds for seq_len {}",
        batch.seq_len
    );
    let offset = token_idx * batch.hidden_dim;
    let src_view = batch.data.slice(offset..offset + len);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

/// Copy `src` into one column of `batch`.
pub fn write_vec_into(
    ctx: &DeviceContext,
    src: &DeviceVec,
    batch: &mut HiddenStates,
    token_idx: usize,
) -> Result<()> {
    anyhow::ensure!(src.len == batch.hidden_dim, "write_vec_into len mismatch");
    let offset = token_idx * batch.hidden_dim;
    let mut dst_view = batch.data.slice_mut(offset..offset + src.len);
    ctx.stream
        .memcpy_dtod(&src.data, &mut dst_view)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

/// Batched GELU-tanh+mul into a pre-allocated output buffer:
/// `out = gelu_tanh(gate) * up` (Gemma 4 MLP, `gelu_pytorch_tanh`). The
/// kernel matches HF's op-sequence rounding: activation in f32, cast to
/// bf16, then multiplied in f32 and rounded once more.
pub fn gelu_tanh_mul_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    anyhow::ensure!(
        gate.hidden_dim == up.hidden_dim && gate.seq_len == up.seq_len,
        "gelu_tanh_mul gate {}x{} != up {}x{}",
        gate.hidden_dim,
        gate.seq_len,
        up.hidden_dim,
        up.seq_len
    );
    anyhow::ensure!(
        out.hidden_dim == gate.hidden_dim && out.seq_len == gate.seq_len,
        "gelu_tanh_mul out {}x{} != gate {}x{}",
        out.hidden_dim,
        out.seq_len,
        gate.hidden_dim,
        gate.seq_len
    );
    let n = gate.checked_extent("gelu_tanh_mul gate")?;
    up.checked_extent("gelu_tanh_mul up")?;
    out.checked_extent("gelu_tanh_mul out")?;
    let n = super::checked_i32(n, "gelu_tanh_mul extent")?;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gelu_tanh_mul_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result
        .result()
        .map_err(|e| anyhow!("gelu_tanh_mul_cuda failed: {e}"))?;

    Ok(())
}

/// In-place final-logit softcap: `x = cap * tanh(x / cap)` in f32 with a
/// single rounding back to bf16, matching the reference's compute-dtype
/// application. Gemma 4 declares `final_logit_softcapping` 30.0 at every
/// published size.
pub fn softcap_bf16_in_place(ctx: &DeviceContext, buf: &mut HiddenStates, cap: f32) -> Result<()> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(anyhow!(
            "softcap_bf16_in_place cap {cap} must be positive and finite"
        ));
    }
    let n = buf.checked_extent("softcap_bf16 buf")?;
    let n = super::checked_i32(n, "softcap_bf16 extent")?;
    let (buf_ptr, _gb) = buf.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::softcap_bf16_in_place_cuda(
            buf_ptr as *mut ffi::Half,
            cap,
            n,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result
        .result()
        .map_err(|e| anyhow!("softcap_bf16_in_place_cuda failed: {e}"))?;

    Ok(())
}

/// Device-resident suppression ids that were held against a head width when
/// they were uploaded. The kernel indexes `logits` by these ids without
/// re-reading them on the host, so the bound has to be structural: this type
/// is the only way to reach it, and it cannot be built without the check.
pub struct SuppressIds {
    ids: CudaSlice<u32>,
    vocab: usize,
}

impl SuppressIds {
    /// Upload `ids` for a head that spans `vocab` columns, refusing any id
    /// the head does not contain.
    pub fn upload(ctx: &DeviceContext, ids: &[u32], vocab: usize) -> Result<Self> {
        for &id in ids {
            if id as usize >= vocab {
                bail!("suppression id {id} is outside the {vocab} columns the head spans");
            }
        }
        let ids = ctx
            .stream
            .clone_htod(ids)
            .map_err(|e| anyhow!("uploading suppression ids failed: {e}"))?;
        Ok(Self { ids, vocab })
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Write negative infinity into every `(row, id)` slot in `logits` with one
/// launch.
pub fn suppress_logits_bf16_in_place(
    ctx: &DeviceContext,
    logits: &mut HiddenStates,
    ids: &SuppressIds,
) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    if logits.hidden_dim != ids.vocab {
        bail!(
            "suppression ids were checked against {} columns but these logits span {}",
            ids.vocab,
            logits.hidden_dim
        );
    }
    logits.checked_extent("suppress_logits_bf16 logits")?;
    let vocab = super::checked_i32(logits.hidden_dim, "suppression vocabulary")?;
    let rows = super::checked_i32(logits.seq_len, "suppression rows")?;
    let id_count = super::checked_i32(ids.ids.len(), "suppression id count")?;
    let (logits_ptr, _logits_guard) = logits.data.device_ptr_mut(&ctx.stream);
    let (ids_ptr, _ids_guard) = ids.ids.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::suppress_logits_bf16_in_place_cuda(
            logits_ptr as *mut ffi::Half,
            ids_ptr as *const u32,
            vocab,
            rows,
            id_count,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result
        .result()
        .map_err(|e| anyhow!("suppress_logits_bf16_in_place_cuda failed: {e}"))
}

/// In-place multiply by a host scalar — Gemma 4's per-layer `layer_scalar`,
/// a `[1]` weight the model reads to the host at load.
pub fn scale_bf16_in_place(ctx: &DeviceContext, buf: &mut HiddenStates, scale: f32) -> Result<()> {
    if !scale.is_finite() {
        return Err(anyhow!("scale_bf16_in_place scale must be finite"));
    }
    let n = buf.checked_extent("scale_bf16 buf")?;
    let n = super::checked_i32(n, "scale_bf16 extent")?;
    let (buf_ptr, _gb) = buf.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::scale_bf16_in_place_cuda(
            buf_ptr as *mut ffi::Half,
            scale,
            n,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result
        .result()
        .map_err(|e| anyhow!("scale_bf16_in_place_cuda failed: {e}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    /// The suppression contract, beside the kernel that owns it: writes are
    /// exactly the given ids, the upload refuses an id at the head's width,
    /// and ids validated against a different head never reach these logits.
    #[test]
    #[ignore = "requires a GPU"]
    fn the_suppression_mask_writes_only_the_ids_it_is_given() {
        let ctx = DeviceContext::new().expect("GPU required");
        let (vocab, rows) = (8usize, 2usize);
        let mut logits = HiddenStates::zeros(&ctx, vocab, rows).expect("logits");
        let suppress_ids = SuppressIds::upload(&ctx, &[3u32, 5], vocab).expect("ids");
        suppress_logits_bf16_in_place(&ctx, &mut logits, &suppress_ids).expect("suppress");

        let host = logits.to_host(&ctx).expect("D2H");
        for row in 0..rows {
            for id in 0..vocab {
                let value = host[row * vocab + id];
                if id == 3 || id == 5 {
                    assert!(
                        value == f32::NEG_INFINITY,
                        "row {row} id {id} is {value}, not suppressed"
                    );
                } else {
                    assert!(value == 0.0, "row {row} id {id} moved to {value}");
                }
            }
        }

        // The bound is structural: an id the head does not span cannot reach
        // the kernel, and neither can ids checked against a different head.
        let past_the_head = SuppressIds::upload(&ctx, &[vocab as u32], vocab);
        assert!(
            past_the_head.is_err(),
            "an id at the head's width must be refused at upload"
        );
        let other_head = SuppressIds::upload(&ctx, &[1u32], vocab + 1).expect("ids");
        assert!(
            suppress_logits_bf16_in_place(&ctx, &mut logits, &other_head).is_err(),
            "ids checked against a wider head must not be applied to these logits"
        );
    }

    fn hidden_from_host(
        ctx: &DeviceContext,
        data: &[bf16],
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        Ok(HiddenStates {
            data: ctx.stream.clone_htod(data)?,
            hidden_dim,
            seq_len,
        })
    }

    fn hidden_to_host(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<bf16>> {
        let host = ctx.stream.clone_dtoh(&hidden.data)?;
        ctx.sync()?;
        Ok(host)
    }

    fn assert_bf16_bits_eq(actual: &[bf16], expected: &[bf16], context: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: element count mismatch"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{context}: BF16 bits mismatch at index {index}"
            );
        }
    }

    #[test]
    fn silu_mul_fused_matches_split_bf16_rounding() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden_dim = 4;
        let seq_len = 3;
        let gate: Vec<_> = [
            -3.0, -1.5, 0.0, 2.0, 0.25, 1.0, 3.5, -0.75, 8.0, -8.0, 0.125, -0.5,
        ]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
        let up: Vec<_> = [
            0.5, -2.0, 4.0, 1.25, -1.0, 0.75, 2.5, -3.0, 0.25, 1.5, -0.625, 5.0,
        ]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
        let mut gate_up = Vec::with_capacity(2 * hidden_dim * seq_len);
        for token in 0..seq_len {
            let offset = token * hidden_dim;
            gate_up.extend_from_slice(&gate[offset..offset + hidden_dim]);
            gate_up.extend_from_slice(&up[offset..offset + hidden_dim]);
        }

        let gate_hidden = hidden_from_host(&ctx, &gate, hidden_dim, seq_len)?;
        let up_hidden = hidden_from_host(&ctx, &up, hidden_dim, seq_len)?;
        let gate_up_hidden = hidden_from_host(&ctx, &gate_up, 2 * hidden_dim, seq_len)?;
        let split = silu_mul_batch(&ctx, &gate_hidden, &up_hidden)?;
        let mut fused = HiddenStates::zeros(&ctx, hidden_dim, seq_len)?;

        silu_mul_fused_batch_into(&ctx, &gate_up_hidden, &mut fused)?;

        let split_host = hidden_to_host(&ctx, &split)?;
        let fused_host = hidden_to_host(&ctx, &fused)?;
        assert_eq!(split_host.len(), fused_host.len());
        for (idx, (split_value, fused_value)) in
            split_host.iter().zip(fused_host.iter()).enumerate()
        {
            assert_eq!(
                split_value.to_bits(),
                fused_value.to_bits(),
                "fused/split silu_mul mismatch at index {idx}"
            );
        }
        Ok(())
    }

    #[test]
    fn split_qkv_is_a_bitwise_copy_for_tp_shapes_and_tails() -> Result<()> {
        let ctx = DeviceContext::new()?;
        for (q_dim, kv_dim, tokens) in [
            // Qwen3-4B/8B production decode geometry at the largest target batch.
            (4096, 1024, 8),
            // Deliberately not aligned to the 256-thread launch width.
            (5, 3, 7),
        ] {
            let qkv_dim = q_dim + 2 * kv_dim;
            let source: Vec<_> = (0..qkv_dim * tokens)
                .map(|idx| bf16::from_bits((idx as u16).wrapping_mul(977).wrapping_add(13)))
                .collect();
            let qkv = hidden_from_host(&ctx, &source, qkv_dim, tokens)?;
            let mut q = HiddenStates::zeros(&ctx, q_dim, tokens)?;
            let mut k = HiddenStates::zeros(&ctx, kv_dim, tokens)?;
            let mut v = HiddenStates::zeros(&ctx, kv_dim, tokens)?;

            split_qkv_into(&ctx, &qkv, &mut q, &mut k, &mut v)?;

            let q_host = hidden_to_host(&ctx, &q)?;
            let k_host = hidden_to_host(&ctx, &k)?;
            let v_host = hidden_to_host(&ctx, &v)?;
            for token in 0..tokens {
                let src = token * qkv_dim;
                let q_start = token * q_dim;
                let kv_start = token * kv_dim;
                assert_bf16_bits_eq(
                    &q_host[q_start..q_start + q_dim],
                    &source[src..src + q_dim],
                    &format!("Q q_dim={q_dim} kv_dim={kv_dim} tokens={tokens} token={token}"),
                );
                assert_bf16_bits_eq(
                    &k_host[kv_start..kv_start + kv_dim],
                    &source[src + q_dim..src + q_dim + kv_dim],
                    &format!("K q_dim={q_dim} kv_dim={kv_dim} tokens={tokens} token={token}"),
                );
                assert_bf16_bits_eq(
                    &v_host[kv_start..kv_start + kv_dim],
                    &source[src + q_dim + kv_dim..src + qkv_dim],
                    &format!("V q_dim={q_dim} kv_dim={kv_dim} tokens={tokens} token={token}"),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn split_qkv_rejects_mismatched_shapes_without_launching() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let qkv = HiddenStates::zeros(&ctx, 12, 1)?;
        let mut q = HiddenStates::zeros(&ctx, 5, 1)?;
        let mut k = HiddenStates::zeros(&ctx, 3, 1)?;
        let mut v = HiddenStates::zeros(&ctx, 3, 1)?;

        let error = split_qkv_into(&ctx, &qkv, &mut q, &mut k, &mut v)
            .expect_err("mismatched QKV dimensions must fail before launch");
        assert!(
            error.to_string().contains("must equal Q + K + V"),
            "unexpected shape error: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn extract_vec_ref_rejects_out_of_bounds_token() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = hidden_from_host(&ctx, &[bf16::from_f32(1.0), bf16::from_f32(2.0)], 2, 1)?;
        let mut out = DeviceVec::zeros(&ctx, 2)?;

        let err = extract_vec_ref_into(&ctx, hidden.as_ref(), 1, &mut out).unwrap_err();

        assert!(
            err.to_string().contains("out of bounds"),
            "unexpected error: {err}"
        );
        Ok(())
    }
}
