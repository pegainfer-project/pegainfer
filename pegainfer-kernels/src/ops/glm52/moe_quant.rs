use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_MOE_QUANT_GROUP_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52MoeQuantShape {
    pub rows: usize,
    pub width: usize,
    pub group_size: usize,
}

impl Glm52MoeQuantShape {
    fn scale_cols(self) -> Result<usize> {
        self.validate()?;
        Ok(self.width / self.group_size)
    }

    fn validate(self) -> Result<()> {
        ensure!(self.rows > 0, "GLM5.2 MoE quant rows must be positive");
        ensure!(self.width > 0, "GLM5.2 MoE quant width must be positive");
        ensure!(
            self.group_size == GLM52_MOE_QUANT_GROUP_SIZE,
            "GLM5.2 MoE quant group_size must be {GLM52_MOE_QUANT_GROUP_SIZE}, got {}",
            self.group_size
        );
        ensure!(
            self.width.is_multiple_of(self.group_size),
            "GLM5.2 MoE quant width {} is not divisible by group_size {}",
            self.width,
            self.group_size
        );
        Ok(())
    }
}

pub fn glm52_fp8_per_token_group_quant_bf16_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &impl DevicePtr<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_quant_buffers(shape, input, output, scales)?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 per-token-group quant launch failed: {err}"))
}

/// UE8M0-scale (next power of two) per-token-group quant — the FlashMLA V3.2
/// fp8 sparse KV-cache contract. The sm100 decode kernel truncates the stored
/// f32 scales to e8m0 (round-toward-zero) for its block-scaled MMA, so a
/// non-power-of-two scale is read up to 2x too small on Blackwell; sm90 reads
/// f32 scales exactly. Power-of-two scales are exact on both. Use this for
/// every write into the 656-byte fp8_ds_mla cache and nowhere else (the MoE
/// and dense GEMM activations keep amax/448 scales).
pub fn glm52_fp8_per_token_group_quant_bf16_ue8m0_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_quant_buffers(shape, input, output, scales)?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _scale_guard) = scales.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_ue8m0_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 per-token-group UE8M0 quant launch failed: {err}"))
}

fn validate_quant_buffers(
    shape: Glm52MoeQuantShape,
    input: &impl DevicePtr<bf16>,
    output: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
) -> Result<()> {
    shape.validate()?;
    let scale_elems = shape.rows * shape.scale_cols()?;
    ensure!(
        input.len() >= shape.rows * shape.width,
        "GLM5.2 MoE quant input too small: have {}, need {}",
        input.len(),
        shape.rows * shape.width
    );
    ensure!(
        output.len() >= shape.rows * shape.width,
        "GLM5.2 MoE quant output too small: have {}, need {}",
        output.len(),
        shape.rows * shape.width
    );
    ensure!(
        scales.len() >= scale_elems,
        "GLM5.2 MoE quant scales too small: have {}, need {scale_elems}",
        scales.len()
    );
    Ok(())
}

/// Bounded UE8M0 re-quant writing the DeepGEMM SM100 masked grouped layout:
/// the loop
/// space stays the aligned recv rows (`shape.rows` capacity, device bound),
/// `row_map` redirects each row to its masked slot (skipping alignment
/// gaps), values land in `[groups, masked_cap, width]` and scales in the
/// mn-major `[groups, width/128, masked_cap]` layout the masked GEMM's SFA
/// descriptor reads. Active-row scales are powers of two; pack them with
/// [`glm52_fp8_scale_pack_ue8m0_launch`] before the GEMM.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_per_token_group_quant_bf16_masked_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    masked_groups: usize,
    masked_cap: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
    row_map: &CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;
    let masked_rows = masked_groups * masked_cap;
    ensure!(
        input.len() >= shape.rows * shape.width
            && output.len() >= masked_rows * shape.width
            && scales.len() >= masked_rows * shape.scale_cols()?
            && row_map.len() >= shape.rows,
        "GLM5.2 FP8 masked quant buffers too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 FP8 masked quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (output_ptr, _g1) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g3) = row_bound.device_ptr(&ctx.stream);
    let (map_ptr, _g4) = row_map.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_per_token_group_quant_bf16_masked_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            map_ptr as *const i32,
            masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 masked group quant launch failed: {err}"))
}

/// Bounded UE8M0 SwiGLU quant for the masked layout: the gate|up input rows
/// are already masked (the W13 masked GEMM wrote them), and output/scales
/// land in the masked layouts (see the quant twin above). Router weights are
/// applied after W2.
#[allow(clippy::too_many_arguments)]
pub fn glm52_silu_and_mul_per_token_group_quant_bf16_masked_launch(
    ctx: &DeviceContext,
    shape: Glm52MoeQuantShape,
    masked_groups: usize,
    masked_cap: usize,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    row_bound: &CudaSlice<i64>,
    bound_index: usize,
    row_map: &CudaSlice<i32>,
) -> Result<()> {
    shape.validate()?;
    let masked_rows = masked_groups * masked_cap;
    ensure!(
        input.len() >= masked_rows * shape.width * 2
            && output.len() >= masked_rows * shape.width
            && scales.len() >= masked_rows * shape.scale_cols()?
            && row_map.len() >= shape.rows,
        "GLM5.2 SiLU masked quant buffers too small"
    );
    ensure!(
        row_bound.len() > bound_index,
        "GLM5.2 SiLU masked quant row_bound index {bound_index} outside buffer of {}",
        row_bound.len()
    );
    let (input_ptr, _g0) = input.device_ptr(&ctx.stream);
    let (output_ptr, _g1) = output.device_ptr_mut(&ctx.stream);
    let (scale_ptr, _g2) = scales.device_ptr_mut(&ctx.stream);
    let (bound_ptr, _g3) = row_bound.device_ptr(&ctx.stream);
    let (map_ptr, _g4) = row_map.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_silu_and_mul_per_token_group_quant_bf16_masked_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut u8,
            scale_ptr as *mut f32,
            shape.rows as i32,
            shape.width as i32,
            shape.group_size as i32,
            (bound_ptr as *const i64).wrapping_add(bound_index),
            map_ptr as *const i32,
            masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 SiLU masked quant launch failed: {err}"))
}

/// Pack power-of-two f32 scales from `[groups, scale_cols, cap]` into the
/// DeepGEMM SM100 MN-major UE8M0 layout `[groups, scale_cols / 4, cap]`.
pub fn glm52_fp8_scale_pack_ue8m0_launch(
    ctx: &DeviceContext,
    groups: usize,
    scale_cols: usize,
    cap: usize,
    scales: &CudaSlice<f32>,
    packed: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && cap > 0 && scale_cols > 0 && scale_cols.is_multiple_of(4),
        "GLM5.2 UE8M0 scale pack needs positive groups/cap and scale_cols divisible by 4, got groups={groups}, scale_cols={scale_cols}, cap={cap}"
    );
    let input_len = groups * scale_cols * cap;
    let output_len = groups * (scale_cols / 4) * cap;
    ensure!(
        scales.len() >= input_len && packed.len() >= output_len,
        "GLM5.2 UE8M0 scale pack buffers too small: scales {}, packed {}, need {input_len}/{output_len}",
        scales.len(),
        packed.len()
    );
    let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
    let (packed_ptr, _packed_guard) = packed.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_scale_pack_ue8m0_cuda(
            scales_ptr as *const f32,
            packed_ptr as *mut i32,
            groups as i32,
            scale_cols as i32,
            cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 UE8M0 scale pack launch failed: {err}"))
}

/// Convert one expert bank in place from arbitrary f32 block scales to the
/// power-of-two UE8M0 contract required by the SM100 DeepGEMM kernel, and
/// pack those scales into `[groups, k / 512, n]`.
pub fn glm52_fp8_weight_ue8m0_prepare_launch(
    ctx: &DeviceContext,
    groups: usize,
    n: usize,
    k: usize,
    weight: &mut CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    packed_scales: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0
            && n > 0
            && n.is_multiple_of(GLM52_MOE_QUANT_GROUP_SIZE)
            && k.is_multiple_of(4 * GLM52_MOE_QUANT_GROUP_SIZE),
        "GLM5.2 weight UE8M0 prepare needs groups>0, n%128=0, k%512=0; got groups={groups}, n={n}, k={k}"
    );
    let weight_len = groups * n * k;
    let scale_len = groups * (n / GLM52_MOE_QUANT_GROUP_SIZE) * (k / GLM52_MOE_QUANT_GROUP_SIZE);
    let packed_len = groups * (k / (4 * GLM52_MOE_QUANT_GROUP_SIZE)) * n;
    ensure!(
        weight.len() >= weight_len
            && scales.len() >= scale_len
            && packed_scales.len() >= packed_len,
        "GLM5.2 weight UE8M0 buffers too small: weight {}, scales {}, packed {}, need {weight_len}/{scale_len}/{packed_len}",
        weight.len(),
        scales.len(),
        packed_scales.len()
    );
    let (weight_ptr, _weight_guard) = weight.device_ptr_mut(&ctx.stream);
    let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
    let (packed_ptr, _packed_guard) = packed_scales.device_ptr_mut(&ctx.stream);
    let requant = unsafe {
        ffi::glm52_fp8_weight_ue8m0_requant_cuda(
            weight_ptr as *mut u8,
            scales_ptr as *const f32,
            groups as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    requant
        .result()
        .map_err(|err| anyhow!("GLM5.2 weight UE8M0 requant launch failed: {err}"))?;
    let pack = unsafe {
        ffi::glm52_fp8_weight_scale_pack_ue8m0_cuda(
            scales_ptr as *const f32,
            packed_ptr as *mut i32,
            groups as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    pack.result()
        .map_err(|err| anyhow!("GLM5.2 weight UE8M0 scale pack launch failed: {err}"))
}
