use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUresult;

use crate::tensor::DeviceContext;

pub(super) struct MarlinGemmBuffers<'a, Scale> {
    pub input: &'a CudaSlice<half::bf16>,
    pub output: &'a mut CudaSlice<half::bf16>,
    pub c_tmp: &'a mut CudaSlice<f32>,
    pub qweight: &'a CudaSlice<u8>,
    pub scales: &'a CudaSlice<Scale>,
    pub workspace: &'a mut CudaSlice<i32>,
    pub sorted_token_ids: &'a CudaSlice<i32>,
    pub expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_padded: &'a CudaSlice<i32>,
    pub topk_weights: &'a CudaSlice<f32>,
}

pub(super) struct MarlinGemmPointers<Scale> {
    pub input: *const u16,
    pub output: *mut u16,
    pub c_tmp: *mut f32,
    pub qweight: *const u8,
    pub scales: *const Scale,
    pub workspace: *mut i32,
    pub sorted_token_ids: *const i32,
    pub expert_ids: *const i32,
    pub num_tokens_post_padded: *const i32,
    pub topk_weights: *const f32,
}

pub(super) fn launch_marlin_gemm<Scale>(
    ctx: &DeviceContext,
    buffers: MarlinGemmBuffers<'_, Scale>,
    launch: impl FnOnce(MarlinGemmPointers<Scale>) -> anyhow::Result<CUresult>,
) -> anyhow::Result<CUresult> {
    let MarlinGemmBuffers {
        input,
        output,
        c_tmp,
        qweight,
        scales,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
    } = buffers;
    let (input, _input_guard) = input.device_ptr(&ctx.stream);
    let (output, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (c_tmp, _c_tmp_guard) = c_tmp.device_ptr_mut(&ctx.stream);
    let (qweight, _qweight_guard) = qweight.device_ptr(&ctx.stream);
    let (scales, _scales_guard) = scales.device_ptr(&ctx.stream);
    let (workspace, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let (sorted_token_ids, _sorted_guard) = sorted_token_ids.device_ptr(&ctx.stream);
    let (expert_ids, _expert_guard) = expert_ids.device_ptr(&ctx.stream);
    let (num_tokens_post_padded, _padded_guard) = num_tokens_post_padded.device_ptr(&ctx.stream);
    let (topk_weights, _weights_guard) = topk_weights.device_ptr(&ctx.stream);
    launch(MarlinGemmPointers {
        input: input as *const u16,
        output: output as *mut u16,
        c_tmp: c_tmp as *mut f32,
        qweight: qweight as *const u8,
        scales: scales as *const Scale,
        workspace: workspace as *mut i32,
        sorted_token_ids: sorted_token_ids as *const i32,
        expert_ids: expert_ids as *const i32,
        num_tokens_post_padded: num_tokens_post_padded as *const i32,
        topk_weights: topk_weights as *const f32,
    })
}

pub(super) struct MarlinAlignBuffers<'a> {
    pub topk_idx: &'a CudaSlice<i32>,
    pub sorted_token_ids: &'a mut CudaSlice<i32>,
    pub expert_ids: &'a mut CudaSlice<i32>,
    pub num_tokens_post_padded: &'a mut CudaSlice<i32>,
    pub expert_offsets: &'a mut CudaSlice<u32>,
}

pub(super) fn launch_marlin_align(
    ctx: &DeviceContext,
    buffers: MarlinAlignBuffers<'_>,
    launch: impl FnOnce(*const i32, *mut i32, *mut i32, *mut i32, *mut u32) -> anyhow::Result<CUresult>,
) -> anyhow::Result<CUresult> {
    let MarlinAlignBuffers {
        topk_idx,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_offsets,
    } = buffers;
    let (topk_idx, _topk_guard) = topk_idx.device_ptr(&ctx.stream);
    let (sorted_token_ids, _sorted_guard) = sorted_token_ids.device_ptr_mut(&ctx.stream);
    let (expert_ids, _expert_guard) = expert_ids.device_ptr_mut(&ctx.stream);
    let (num_tokens_post_padded, _padded_guard) =
        num_tokens_post_padded.device_ptr_mut(&ctx.stream);
    let (expert_offsets, _offsets_guard) = expert_offsets.device_ptr_mut(&ctx.stream);
    launch(
        topk_idx as *const i32,
        sorted_token_ids as *mut i32,
        expert_ids as *mut i32,
        num_tokens_post_padded as *mut i32,
        expert_offsets as *mut u32,
    )
}
