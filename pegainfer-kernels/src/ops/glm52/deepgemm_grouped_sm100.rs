//! GLM5.2 EP4/Blackwell routed-expert GEMM: DeepGEMM SM100 MGroupedMasked
//! fp8 blockscale (tcgen05), AOT-instantiated (no JIT, no torch). Same
//! metadata/remap bridging as the SM90 EP8 chain (`deepgemm_grouped.rs`),
//! but scales are the Blackwell packed-UE8M0 i32 layout and the per-expert
//! capacity is a runtime multiple of 128. See
//! `csrc/glm52/glm52_deepgemm_grouped_sm100.cu`.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::deepgemm_grouped::Glm52DeepGemmGroupedFp8Kind;
use crate::ffi;
use crate::tensor::DeviceContext;

/// Alignment required by the SM100 masked layout and packed scale factors.
pub const GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT: usize = 128;
/// Per-expert row alignment of the DeepEP recv segment layout.
const GLM52_DEEPGEMM_SM100_EXPERT_ALIGNMENT: usize = 64;

/// psum (i32 aligned running ends) → aligned segment starts (`expert_offsets`,
/// with `[groups]` = the aligned end), per-expert real row counts
/// (`masked_m`), and the aligned-row → masked-slot map (`row_map`, -1 across
/// alignment gaps). `m_capacity` is the row bound the caller's quant kernels
/// cover; the kernel device-traps if any segment ends past it (a cross-rank
/// token-count disagreement) or exceeds the masked per-expert capacity.
pub fn glm52_deepgemm_sm100_grouped_fp8_metadata_launch(
    ctx: &DeviceContext,
    groups: usize,
    m_capacity: usize,
    masked_cap: usize,
    psum_expert: &CudaSlice<i32>,
    expert_offsets: &mut CudaSlice<i64>,
    masked_m: &mut CudaSlice<i32>,
    row_map: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0
            && m_capacity > 0
            && masked_cap.is_multiple_of(GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT),
        "GLM5.2 SM100 DeepGEMM metadata needs groups/m_capacity>0 and masked_cap%128=0, got groups={groups}, m_capacity={m_capacity}, masked_cap={masked_cap}"
    );
    ensure!(
        psum_expert.len() >= groups
            && expert_offsets.len() > groups
            && masked_m.len() >= groups
            && row_map.len() >= m_capacity,
        "GLM5.2 SM100 DeepGEMM metadata buffers too small for {groups} groups / {m_capacity} rows: psum={}, offsets={}, masked_m={}, row_map={}",
        psum_expert.len(),
        expert_offsets.len(),
        masked_m.len(),
        row_map.len()
    );
    let (psum_ptr, _psum_guard) = psum_expert.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr_mut(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr_mut(&ctx.stream);
    let (map_ptr, _map_guard) = row_map.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_sm100_grouped_fp8_metadata_cuda(
            psum_ptr as *const i32,
            offsets_ptr as *mut i64,
            masked_ptr as *mut i32,
            map_ptr as *mut i32,
            groups as i32,
            m_capacity as i32,
            GLM52_DEEPGEMM_SM100_EXPERT_ALIGNMENT as i32,
            masked_cap as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 SM100 DeepGEMM metadata launch failed: {err}"))
}

/// Masked grouped fp8 GEMM over the rank's local experts:
/// `out[g, :masked_m[g], n] = deq(weight[g]) @ deq(activation[g])`.
/// Activation `[groups, masked_cap, k]` fp8, activation scale packed-UE8M0
/// i32 `[groups, k/512, masked_cap]` (MN-major, 4 exponent bytes per i32), weight
/// `[groups, n, k]` fp8 (bank layout as-is), weight scale packed-UE8M0 i32
/// `[groups, k/512, n]`, out `[groups, masked_cap, n]` bf16. `groups` dispatches
/// over the EP widths {64,32,16,8,4,2}. Requires sm_100f (NOT_SUPPORTED
/// elsewhere).
pub fn glm52_deepgemm_sm100_masked_grouped_fp8_launch(
    ctx: &DeviceContext,
    kind: Glm52DeepGemmGroupedFp8Kind,
    groups: usize,
    masked_cap: usize,
    num_sms: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<i32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<i32>,
    masked_m: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    let (n, k) = kind.shape();
    ensure!(
        matches!(groups, 64 | 32 | 16 | 8 | 4 | 2),
        "GLM5.2 SM100 masked grouped FP8 needs groups in {{64,32,16,8,4,2}}, got {groups}"
    );
    ensure!(
        matches!(num_sms, 148 | 152),
        "GLM5.2 SM100 masked grouped FP8 supports B200/GB300 SM counts {{148,152}}, got {num_sms}"
    );
    ensure!(
        masked_cap > 0 && masked_cap.is_multiple_of(GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT),
        "GLM5.2 SM100 masked grouped FP8 needs masked_cap divisible by 128, got {masked_cap}"
    );
    ensure!(
        activation.len() >= groups * masked_cap * k
            && activation_scale.len() >= groups * (k / 512) * masked_cap
            && weight.len() >= groups * n * k
            && weight_scale.len() >= groups * (k / 512) * n
            && masked_m.len() >= groups
            && output.len() >= groups * masked_cap * n,
        "GLM5.2 SM100 masked grouped FP8 {kind:?} buffers too small: act {}, act_scale {}, w {}, w_scale {}, masked_m {}, out {}",
        activation.len(),
        activation_scale.len(),
        weight.len(),
        weight_scale.len(),
        masked_m.len(),
        output.len()
    );
    let (act_ptr, _act_guard) = activation.device_ptr(&ctx.stream);
    let (act_scale_ptr, _act_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = weight.device_ptr(&ctx.stream);
    let (w_scale_ptr, _w_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let abi_kind: i32 = match kind {
        Glm52DeepGemmGroupedFp8Kind::W13 => 1,
        Glm52DeepGemmGroupedFp8Kind::W2 => 2,
    };
    let result = unsafe {
        ffi::glm52_deepgemm_sm100_masked_grouped_fp8_launch_cuda(
            abi_kind,
            act_ptr as *const u8,
            act_scale_ptr as *const i32,
            w_ptr as *const u8,
            w_scale_ptr as *const i32,
            masked_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            groups as i32,
            n as i32,
            k as i32,
            masked_cap as i32,
            num_sms as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 SM100 masked grouped FP8 {kind:?} launch failed: {err}"))
}

/// Masked W2 output `[groups, masked_cap, n]` → the aligned recv slots
/// `decode_combine` addresses, applying each aligned row's router weight
/// after the GEMM (rows `offsets[g] + r` for `r < masked_m[g]`).
pub fn glm52_deepgemm_sm100_masked_out_to_aligned_launch(
    ctx: &DeviceContext,
    groups: usize,
    masked_cap: usize,
    n: usize,
    masked_out: &CudaSlice<bf16>,
    masked_m: &CudaSlice<i32>,
    expert_offsets: &CudaSlice<i64>,
    row_weights: &CudaSlice<f32>,
    aligned_out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        masked_cap > 0
            && masked_cap.is_multiple_of(GLM52_DEEPGEMM_SM100_MASKED_ALIGNMENT)
            && n > 0
            && n.is_multiple_of(4)
            && aligned_out.len().is_multiple_of(n),
        "GLM5.2 SM100 masked-out remap needs masked_cap%128=0, n%4=0 and whole output rows, got cap {masked_cap}, n {n}, output {}",
        aligned_out.len()
    );
    let aligned_rows = i32::try_from(aligned_out.len() / n)
        .map_err(|_| anyhow!("GLM5.2 SM100 masked-out remap output row count exceeds i32"))?;
    ensure!(
        masked_out.len() >= groups * masked_cap * n
            && masked_m.len() >= groups
            && expert_offsets.len() > groups
            && aligned_rows > 0
            && row_weights.len() >= aligned_rows as usize,
        "GLM5.2 SM100 masked-out remap buffers too small: masked {}, masked_m {}, offsets {}, weights {}, output rows {aligned_rows}",
        masked_out.len(),
        masked_m.len(),
        expert_offsets.len(),
        row_weights.len()
    );
    let (src_ptr, _src_guard) = masked_out.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (weights_ptr, _weights_guard) = row_weights.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = aligned_out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_sm100_masked_out_to_aligned_cuda(
            src_ptr as *const ffi::Half,
            masked_ptr as *const i32,
            offsets_ptr as *const i64,
            weights_ptr as *const f32,
            dst_ptr as *mut ffi::Half,
            groups as i32,
            masked_cap as i32,
            aligned_rows,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 SM100 masked-out remap launch failed: {err}"))
}
