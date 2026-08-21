use std::ptr::null;
use std::ptr::null_mut;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

pub struct LoraDecodeGroupedProjection<'a> {
    pub a_packed: &'a CudaSlice<half::bf16>,
    pub b_packed: &'a CudaSlice<half::bf16>,
    pub scales: &'a CudaSlice<f32>,
    pub out: &'a mut HiddenStates,
    pub max_loras: usize,
    pub max_rank: usize,
    pub rank: usize,
    pub out_dim: usize,
}

pub fn pack_lora_b_rows_into(
    ctx: &DeviceContext,
    src: &CudaSlice<half::bf16>,
    dst: &mut CudaSlice<half::bf16>,
    dst_offset: usize,
    rank: usize,
    max_rank: usize,
    out_dim: usize,
) -> Result<()> {
    assert!(rank > 0, "LoRA rank must be > 0");
    assert!(rank <= max_rank, "LoRA rank exceeds packed max_rank");
    assert!(
        src.len() >= out_dim * rank,
        "LoRA B source is smaller than out_dim * rank"
    );
    assert!(
        dst.len() >= dst_offset + out_dim * max_rank,
        "LoRA B destination slot exceeds packed storage"
    );

    let (src_ptr, _gs) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.device_ptr_mut(&ctx.stream);
    let dst_ptr = dst_ptr + (dst_offset * std::mem::size_of::<half::bf16>()) as u64;
    let result = unsafe {
        ffi::lora_pack_b_rows_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            rank as i32,
            max_rank as i32,
            out_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn lora_decode_fused_delta_into(
    ctx: &DeviceContext,
    a_packed: &CudaSlice<half::bf16>,
    b_packed: &CudaSlice<half::bf16>,
    scales: &CudaSlice<f32>,
    token_slots: &CudaSlice<i32>,
    input: &HiddenStates,
    out: &mut HiddenStates,
    max_loras: usize,
    max_rank: usize,
    rank: usize,
    out_dim: usize,
    row_offset: usize,
) -> Result<()> {
    assert!(rank > 0, "LoRA rank must be > 0");
    assert!(rank <= max_rank, "LoRA rank exceeds packed max_rank");
    assert!(
        row_offset + out_dim <= out.hidden_dim,
        "LoRA output row range exceeds output hidden dimension"
    );
    assert_eq!(
        input.seq_len, out.seq_len,
        "LoRA decode input/output batch mismatch"
    );

    let (a_ptr, _ga) = a_packed.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_packed.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (slots_ptr, _gslot) = token_slots.device_ptr(&ctx.stream);
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::lora_decode_fused_delta_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            scales_ptr as *const f32,
            slots_ptr as *const i32,
            input_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            input.seq_len as i32,
            max_loras as i32,
            max_rank as i32,
            rank as i32,
            input.hidden_dim as i32,
            out_dim as i32,
            out.hidden_dim as i32,
            row_offset as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn lora_decode_fused_delta_group3_into(
    ctx: &DeviceContext,
    token_slots: &CudaSlice<i32>,
    input: &HiddenStates,
    mut p0: Option<LoraDecodeGroupedProjection<'_>>,
    mut p1: Option<LoraDecodeGroupedProjection<'_>>,
    mut p2: Option<LoraDecodeGroupedProjection<'_>>,
) -> Result<()> {
    let Some((max_loras, max_rank)) = p0.as_ref().or(p1.as_ref()).or(p2.as_ref()).map(|p| {
        assert!(p.rank > 0, "LoRA rank must be > 0");
        assert!(p.rank <= p.max_rank, "LoRA rank exceeds packed max_rank");
        (p.max_loras, p.max_rank)
    }) else {
        return Ok(());
    };
    validate_group_projection(p0.as_ref(), input, max_loras, max_rank);
    validate_group_projection(p1.as_ref(), input, max_loras, max_rank);
    validate_group_projection(p2.as_ref(), input, max_loras, max_rank);

    let (slots_ptr, _gslot) = token_slots.device_ptr(&ctx.stream);
    let (input_ptr, _gi) = input.data.device_ptr(&ctx.stream);

    let mut _p0_a_guard = None;
    let mut _p0_b_guard = None;
    let mut _p0_scales_guard = None;
    let mut _p0_out_guard = None;
    let (p0_a, p0_b, p0_scales, p0_out, p0_rank, p0_out_dim, p0_out_hidden_dim) =
        if let Some(projection) = p0.as_mut() {
            let (a_ptr, ga) = projection.a_packed.device_ptr(&ctx.stream);
            _p0_a_guard = Some(ga);
            let (b_ptr, gb) = projection.b_packed.device_ptr(&ctx.stream);
            _p0_b_guard = Some(gb);
            let (scales_ptr, gs) = projection.scales.device_ptr(&ctx.stream);
            _p0_scales_guard = Some(gs);
            let (out_ptr, go) = projection.out.data.device_ptr_mut(&ctx.stream);
            _p0_out_guard = Some(go);
            (
                a_ptr as *const half::bf16,
                b_ptr as *const half::bf16,
                scales_ptr as *const f32,
                out_ptr as *mut half::bf16,
                projection.rank as i32,
                projection.out_dim as i32,
                projection.out.hidden_dim as i32,
            )
        } else {
            (
                null::<half::bf16>(),
                null::<half::bf16>(),
                null::<f32>(),
                null_mut(),
                0,
                0,
                0,
            )
        };

    let mut _p1_a_guard = None;
    let mut _p1_b_guard = None;
    let mut _p1_scales_guard = None;
    let mut _p1_out_guard = None;
    let (p1_a, p1_b, p1_scales, p1_out, p1_rank, p1_out_dim, p1_out_hidden_dim) =
        if let Some(projection) = p1.as_mut() {
            let (a_ptr, ga) = projection.a_packed.device_ptr(&ctx.stream);
            _p1_a_guard = Some(ga);
            let (b_ptr, gb) = projection.b_packed.device_ptr(&ctx.stream);
            _p1_b_guard = Some(gb);
            let (scales_ptr, gs) = projection.scales.device_ptr(&ctx.stream);
            _p1_scales_guard = Some(gs);
            let (out_ptr, go) = projection.out.data.device_ptr_mut(&ctx.stream);
            _p1_out_guard = Some(go);
            (
                a_ptr as *const half::bf16,
                b_ptr as *const half::bf16,
                scales_ptr as *const f32,
                out_ptr as *mut half::bf16,
                projection.rank as i32,
                projection.out_dim as i32,
                projection.out.hidden_dim as i32,
            )
        } else {
            (
                null::<half::bf16>(),
                null::<half::bf16>(),
                null::<f32>(),
                null_mut(),
                0,
                0,
                0,
            )
        };

    let mut _p2_a_guard = None;
    let mut _p2_b_guard = None;
    let mut _p2_scales_guard = None;
    let mut _p2_out_guard = None;
    let (p2_a, p2_b, p2_scales, p2_out, p2_rank, p2_out_dim, p2_out_hidden_dim) =
        if let Some(projection) = p2.as_mut() {
            let (a_ptr, ga) = projection.a_packed.device_ptr(&ctx.stream);
            _p2_a_guard = Some(ga);
            let (b_ptr, gb) = projection.b_packed.device_ptr(&ctx.stream);
            _p2_b_guard = Some(gb);
            let (scales_ptr, gs) = projection.scales.device_ptr(&ctx.stream);
            _p2_scales_guard = Some(gs);
            let (out_ptr, go) = projection.out.data.device_ptr_mut(&ctx.stream);
            _p2_out_guard = Some(go);
            (
                a_ptr as *const half::bf16,
                b_ptr as *const half::bf16,
                scales_ptr as *const f32,
                out_ptr as *mut half::bf16,
                projection.rank as i32,
                projection.out_dim as i32,
                projection.out.hidden_dim as i32,
            )
        } else {
            (
                null::<half::bf16>(),
                null::<half::bf16>(),
                null::<f32>(),
                null_mut(),
                0,
                0,
                0,
            )
        };

    let result = unsafe {
        ffi::lora_decode_fused_delta_group3_cuda(
            p0_a.cast::<ffi::Half>(),
            p0_b.cast::<ffi::Half>(),
            p0_scales,
            p0_out.cast::<ffi::Half>(),
            p0_rank,
            p0_out_dim,
            p0_out_hidden_dim,
            p1_a.cast::<ffi::Half>(),
            p1_b.cast::<ffi::Half>(),
            p1_scales,
            p1_out.cast::<ffi::Half>(),
            p1_rank,
            p1_out_dim,
            p1_out_hidden_dim,
            p2_a.cast::<ffi::Half>(),
            p2_b.cast::<ffi::Half>(),
            p2_scales,
            p2_out.cast::<ffi::Half>(),
            p2_rank,
            p2_out_dim,
            p2_out_hidden_dim,
            slots_ptr as *const i32,
            input_ptr as *const ffi::Half,
            input.seq_len as i32,
            max_loras as i32,
            max_rank as i32,
            input.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

fn validate_group_projection(
    projection: Option<&LoraDecodeGroupedProjection<'_>>,
    input: &HiddenStates,
    max_loras: usize,
    max_rank: usize,
) {
    let Some(projection) = projection else {
        return;
    };
    assert!(projection.rank > 0, "LoRA rank must be > 0");
    assert!(
        projection.rank <= projection.max_rank,
        "LoRA rank exceeds packed max_rank"
    );
    assert_eq!(
        projection.max_loras, max_loras,
        "grouped LoRA projection max_loras mismatch"
    );
    assert_eq!(
        projection.max_rank, max_rank,
        "grouped LoRA projection max_rank mismatch"
    );
    assert_eq!(
        input.seq_len, projection.out.seq_len,
        "LoRA decode input/output batch mismatch"
    );
    assert!(
        projection.out_dim <= projection.out.hidden_dim,
        "LoRA output row range exceeds output hidden dimension"
    );
}
