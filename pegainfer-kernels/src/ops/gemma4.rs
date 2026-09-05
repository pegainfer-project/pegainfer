use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::ops::marlin_face::MarlinAlignBuffers;
use crate::ops::marlin_face::MarlinGemmBuffers;
use crate::ops::marlin_face::launch_marlin_align;
use crate::ops::marlin_face::launch_marlin_gemm;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

const GEMMA4_ROUTER_EXPERTS: usize = 128;
const GEMMA4_ROUTER_MAX_TOP_K: usize = 32;

/// Softmax the router logits over every expert, take the top `top_k`,
/// renormalize those among themselves, and apply the per-expert scale.
pub fn gemma4_moe_router_topk_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    per_expert_scale: &DeviceVec,
    top_k: usize,
    index_out: &mut CudaSlice<i32>,
    weight_out: &mut CudaSlice<f32>,
) -> Result<()> {
    let rows = logits.seq_len;
    let experts = logits.hidden_dim;
    ensure!(
        experts == GEMMA4_ROUTER_EXPERTS && (1..=GEMMA4_ROUTER_MAX_TOP_K).contains(&top_k),
        "gemma4_moe_router_topk_into: the 128-expert register-router contract requires exactly \
         128 experts and 1..=32 picks, got {experts} experts and top {top_k}"
    );
    ensure!(
        rows > 0 && top_k > 0 && top_k <= experts,
        "gemma4_moe_router_topk_into: {rows} rows and top {top_k} of {experts} experts is not \
         a routing problem"
    );
    // `DeviceVec::len` and `HiddenStates` dims are labels; the kernel reads
    // the allocations, so the allocations are what is checked.
    ensure!(
        per_expert_scale.len == experts && per_expert_scale.data.len() >= experts,
        "gemma4_moe_router_topk_into: the per-expert scale holds {} over {} allocated, not \
         {experts}",
        per_expert_scale.len,
        per_expert_scale.data.len()
    );
    let slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("gemma4_moe_router_topk_into: {rows} x {top_k} overflows usize"))?;
    logits.checked_extent("gemma4_moe_router_topk_into logits")?;
    ensure!(
        index_out.len() >= slots && weight_out.len() >= slots,
        "gemma4_moe_router_topk_into outputs too small for {rows} x {top_k}: index {}, weight {}",
        index_out.len(),
        weight_out.len()
    );
    let rows_i32 = i32::try_from(rows)
        .map_err(|_| anyhow!("gemma4_moe_router_topk_into: {rows} rows exceed the kernel's i32"))?;
    let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = per_expert_scale.data.device_ptr(&ctx.stream);
    let (index_ptr, _index_guard) = index_out.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight_out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_moe_router_topk_cuda(
            logits_ptr as *const ffi::Half,
            scale_ptr as *const ffi::Half,
            rows_i32,
            experts as i32,
            top_k as i32,
            index_ptr as *mut i32,
            weight_ptr as *mut f32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Fold a routed GEMM's `[rows * top_k, hidden]` result back onto its tokens.
pub fn gemma4_moe_sum_topk_into(
    ctx: &DeviceContext,
    routed: &HiddenStates,
    top_k: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let rows = out.seq_len;
    let hidden = out.hidden_dim;
    ensure!(
        rows > 0 && top_k > 0 && routed.hidden_dim == hidden,
        "gemma4_moe_sum_topk_into: routed is {} wide, out is {hidden}",
        routed.hidden_dim
    );
    ensure!(
        routed.seq_len >= rows * top_k,
        "gemma4_moe_sum_topk_into: routed holds {} rows, not {}",
        routed.seq_len,
        rows * top_k
    );
    let (routed_ptr, _routed_guard) = routed.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_moe_sum_topk_cuda(
            routed_ptr as *const ffi::Half,
            i32::try_from(rows)?,
            i32::try_from(top_k)?,
            i32::try_from(hidden)?,
            out_ptr as *mut ffi::Half,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Everything one Marlin NVFP4 GEMM needs beyond its operands.
pub struct MarlinDispatch<'a> {
    pub sorted_token_ids: &'a CudaSlice<i32>,
    pub expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_padded: &'a CudaSlice<i32>,
    pub topk_weights: &'a CudaSlice<f32>,
    pub block_size: usize,
    /// `top_k` for the first projection, which gathers each token once per
    /// pick; `1` for the second, whose rows are already per pick.
    pub top_k: usize,
    pub mul_topk_weights: bool,
}

/// One expert-blocked NVFP4 GEMM. `rows` is the A operand's row count, which
/// the dispatch's `top_k` expands to the output's.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_marlin_nvfp4_moe(
    ctx: &DeviceContext,
    input: &HiddenStates,
    qweight: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    global_scale: &CudaSlice<f32>,
    workspace: &mut CudaSlice<i32>,
    c_tmp: &mut CudaSlice<f32>,
    dispatch: &MarlinDispatch<'_>,
    rows: usize,
    size_n: usize,
    size_k: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        rows > 0 && size_n > 0 && size_k > 0,
        "gemma4_marlin_nvfp4_moe: {rows} x {size_n} x {size_k} is not a GEMM"
    );
    ensure!(
        input.hidden_dim == size_k && input.seq_len >= rows,
        "gemma4_marlin_nvfp4_moe: input is {} x {}, needs {rows} x {size_k}",
        input.seq_len,
        input.hidden_dim
    );
    ensure!(
        out.hidden_dim == size_n && out.seq_len >= rows * dispatch.top_k,
        "gemma4_marlin_nvfp4_moe: out is {} x {}, needs {} x {size_n}",
        out.seq_len,
        out.hidden_dim,
        rows * dispatch.top_k
    );
    let (global_ptr, _global_guard) = global_scale.device_ptr(&ctx.stream);
    let workspace_len = workspace.len();
    let sorted_len = dispatch.sorted_token_ids.len();
    let buffers = MarlinGemmBuffers {
        input: &input.data,
        output: &mut out.data,
        c_tmp,
        qweight,
        scales,
        workspace,
        sorted_token_ids: dispatch.sorted_token_ids,
        expert_ids: dispatch.expert_ids,
        num_tokens_post_padded: dispatch.num_tokens_post_padded,
        topk_weights: dispatch.topk_weights,
    };
    let result = launch_marlin_gemm(ctx, buffers, |ptrs| {
        Ok(unsafe {
            ffi::gemma4_marlin_nvfp4_moe_cuda(
                ptrs.input,
                ptrs.output,
                ptrs.c_tmp,
                ptrs.qweight,
                ptrs.scales,
                global_ptr as *const f32,
                ptrs.workspace,
                ptrs.sorted_token_ids,
                ptrs.expert_ids,
                ptrs.num_tokens_post_padded,
                ptrs.topk_weights,
                i32::try_from(workspace_len)?,
                i32::try_from(sorted_len)?,
                i32::try_from(dispatch.block_size)?,
                i32::try_from(dispatch.top_k)?,
                dispatch.mul_topk_weights,
                i32::try_from(rows)?,
                i32::try_from(size_n)?,
                i32::try_from(size_k)?,
                0,
                crate::tensor::active_cu_stream(ctx),
            )
        })
    })?;
    result.result()?;
    Ok(())
}

/// Rewrite a stacked projection's block scales into Marlin's order.
pub fn gemma4_marlin_nvfp4_prepare_scales(
    ctx: &DeviceContext,
    checkpoint: &CudaSlice<u8>,
    prepared: &mut CudaSlice<u8>,
    experts: usize,
    in_dim: usize,
    out_dim: usize,
    rescale: f32,
) -> Result<()> {
    let expected = experts * out_dim * (in_dim / 16);
    ensure!(
        in_dim.is_multiple_of(16) && checkpoint.len() >= expected && prepared.len() >= expected,
        "gemma4_marlin_nvfp4_prepare_scales: {experts} x {out_dim} x {in_dim} needs {expected} \
         bytes, have {} and {}",
        checkpoint.len(),
        prepared.len()
    );
    let (src_ptr, _src_guard) = checkpoint.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = prepared.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_marlin_nvfp4_prepare_scales_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            i32::try_from(experts)?,
            i32::try_from(in_dim)?,
            i32::try_from(out_dim)?,
            rescale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Rewrite a stacked four-bit projection into Marlin's B layout.
pub fn marlin_repack_4bit(
    ctx: &DeviceContext,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    experts: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    let expected = experts * out_dim * in_dim / 2;
    ensure!(
        src.len() >= expected && dst.len() >= expected,
        "marlin_repack_4bit: {experts} x {out_dim} x {in_dim} needs {expected} bytes, have {} \
         and {}",
        src.len(),
        dst.len()
    );
    let (src_ptr, _src_guard) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::marlin_repack_4bit_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            i32::try_from(experts)?,
            i32::try_from(in_dim)?,
            i32::try_from(out_dim)?,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Scratch the device-side dispatch builder writes through.
pub struct MoeAlignScratch<'a> {
    pub sorted_token_ids: &'a mut CudaSlice<i32>,
    pub expert_ids: &'a mut CudaSlice<i32>,
    pub num_tokens_post_padded: &'a mut CudaSlice<i32>,
    pub expert_offsets: &'a mut CudaSlice<u32>,
    /// Ignored: the alignment pass no longer writes a cursor. The field stays
    /// so struct literals written against the earlier shape keep compiling.
    pub expert_cursor: &'a mut CudaSlice<u32>,
}

/// Group the routed slots into the kernel's fixed-width blocks, on the device.
///
/// Entries beyond `num_tokens_post_padded` may remain stale; consumers use
/// that device-resident extent.
pub fn marlin_moe_align_block_size(
    ctx: &DeviceContext,
    topk_idx: &CudaSlice<i32>,
    rows: usize,
    top_k: usize,
    experts: usize,
    block_size: usize,
    out: &mut MoeAlignScratch<'_>,
) -> Result<()> {
    let routes = rows
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("marlin_moe_align_block_size: {rows} x {top_k} overflows usize"))?;
    let max_padded = out.sorted_token_ids.len();
    let max_blocks = out.expert_ids.len();
    ensure!(
        topk_idx.len() >= routes
            && !out.num_tokens_post_padded.is_empty()
            && out.expert_offsets.len() > experts,
        "marlin_moe_align_block_size scratch too small for {routes} routes over {experts} experts"
    );
    ensure!(
        max_padded >= routes + experts * (block_size - 1),
        "marlin_moe_align_block_size: {max_padded} padded slots cannot hold {routes} routes with \
         one part filled block per expert"
    );
    let buffers = MarlinAlignBuffers {
        topk_idx,
        sorted_token_ids: out.sorted_token_ids,
        expert_ids: out.expert_ids,
        num_tokens_post_padded: out.num_tokens_post_padded,
        expert_offsets: out.expert_offsets,
    };
    let result = launch_marlin_align(
        ctx,
        buffers,
        |idx_ptr, sorted_ptr, expert_ptr, padded_ptr, offsets_ptr| {
            Ok(unsafe {
                ffi::marlin_moe_align_block_size_cuda(
                    idx_ptr,
                    sorted_ptr,
                    expert_ptr,
                    padded_ptr,
                    offsets_ptr,
                    std::ptr::null_mut(),
                    i32::try_from(rows)?,
                    i32::try_from(top_k)?,
                    0,
                    i32::try_from(experts)?,
                    i32::try_from(block_size)?,
                    i32::try_from(max_padded)?,
                    i32::try_from(max_blocks)?,
                    crate::tensor::active_cu_stream(ctx),
                )
            })
        },
    )?;
    result.result()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    const EXPERTS: usize = GEMMA4_ROUTER_EXPERTS;
    const TOP_K: usize = 8;
    const SELECTED: [usize; TOP_K] = [3, 10, 34, 41, 65, 72, 96, 127];
    const SELECTED_LOGIT: f32 = 1.0;
    const BASE_SCALE: f32 = 0.5;
    const SCALE_STEP: f32 = 0.125;
    const TOP_K_F32: f32 = 8.0;
    const WEIGHT_TOLERANCE: f32 = 2.0e-6;

    fn scale_host() -> Vec<bf16> {
        (0..EXPERTS)
            .map(|expert| {
                let bucket = u8::try_from(expert % TOP_K).expect("scale bucket");
                bf16::from_f32(BASE_SCALE + f32::from(bucket) * SCALE_STEP)
            })
            .collect()
    }

    fn finite_row() -> Vec<bf16> {
        let mut row = vec![bf16::ZERO; EXPERTS];
        for expert in SELECTED {
            row[expert] = bf16::from_f32(SELECTED_LOGIT);
        }
        row
    }

    fn run_router(
        ctx: &DeviceContext,
        scale: &DeviceVec,
        logits_host: &[bf16],
    ) -> (Vec<i32>, Vec<f32>) {
        let rows = logits_host.len() / EXPERTS;
        let logits = HiddenStates::from_host(ctx, logits_host, EXPERTS, rows).expect("logits");
        let mut index = ctx.stream.alloc_zeros::<i32>(rows * TOP_K).expect("index");
        let mut weight = ctx.stream.alloc_zeros::<f32>(rows * TOP_K).expect("weight");
        gemma4_moe_router_topk_into(ctx, &logits, scale, TOP_K, &mut index, &mut weight)
            .expect("router");
        (
            ctx.stream.clone_dtoh(&index).expect("index D2H"),
            ctx.stream.clone_dtoh(&weight).expect("weight D2H"),
        )
    }

    fn assert_non_finite_rows_fail_closed(
        ctx: &DeviceContext,
        scale: &DeviceVec,
        alone: &(Vec<i32>, Vec<f32>),
    ) {
        let finite = finite_row();
        let all_negative_infinity = vec![bf16::from_f32(f32::NEG_INFINITY); EXPERTS];
        let mut positive_infinity = finite.clone();
        positive_infinity[SELECTED[0]] = bf16::from_f32(f32::INFINITY);
        let mut nan = finite.clone();
        nan[SELECTED[0]] = bf16::from_f32(f32::NAN);
        // A lone -inf beside finite logits is the case a plain softmax would
        // mask into ordinary weights.
        let mut one_negative_infinity = finite.clone();
        one_negative_infinity[SELECTED[0] + 1] = bf16::from_f32(f32::NEG_INFINITY);
        let non_finite_rows = [
            ("all -inf row", all_negative_infinity),
            ("+inf row", positive_infinity),
            ("NaN row", nan),
            ("one -inf beside finite row", one_negative_infinity),
        ];
        let logits_host = std::iter::once(finite.as_slice())
            .chain(non_finite_rows.iter().map(|(_, row)| row.as_slice()))
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let (batched_index, batched_weight) = run_router(ctx, scale, &logits_host);
        let (alone_index, alone_weight) = alone;
        assert_eq!(&batched_index[..TOP_K], alone_index);
        assert!(
            batched_weight[..TOP_K]
                .iter()
                .map(|weight| weight.to_bits())
                .eq(alone_weight.iter().map(|weight| weight.to_bits()))
        );
        let expected_indices = (0..TOP_K as i32).collect::<Vec<_>>();
        for (offset, (name, _)) in non_finite_rows.iter().enumerate() {
            let row = offset + 1;
            let slots = row * TOP_K..(row + 1) * TOP_K;
            assert_eq!(
                &batched_index[slots.clone()],
                expected_indices.as_slice(),
                "{name} (row {row}) did not emit indices 0..TOP_K"
            );
            assert!(
                batched_weight[slots].iter().all(|weight| weight.is_nan()),
                "{name} (row {row}) did not emit all-NaN weights"
            );
        }
    }

    fn assert_finite_contract(
        ctx: &DeviceContext,
        scale: &DeviceVec,
        scale_host: &[bf16],
    ) -> (Vec<i32>, Vec<f32>) {
        let finite = finite_row();
        let logits_host = finite
            .iter()
            .chain(std::iter::repeat_n(&bf16::ZERO, EXPERTS))
            .copied()
            .collect::<Vec<_>>();
        let (index_host, weight_host) = run_router(ctx, scale, &logits_host);
        let expected_index = SELECTED
            .into_iter()
            .chain(0..TOP_K)
            .map(|expert| i32::try_from(expert).expect("expert index"))
            .collect::<Vec<_>>();
        assert_eq!(index_host, expected_index);
        for (slot, (&expert, &actual)) in index_host.iter().zip(&weight_host).enumerate() {
            let expert = usize::try_from(expert).expect("nonnegative expert");
            let expected = scale_host[expert].to_f32() / TOP_K_F32;
            assert!(
                (actual - expected).abs() <= WEIGHT_TOLERANCE,
                "slot {slot}: {actual} != {expected}"
            );
        }
        (index_host[..TOP_K].to_vec(), weight_host[..TOP_K].to_vec())
    }

    fn assert_invalid_expert_contract(ctx: &DeviceContext) {
        let invalid_experts = EXPERTS - 1;
        let invalid_logits =
            HiddenStates::from_host(ctx, &vec![bf16::ZERO; invalid_experts], invalid_experts, 1)
                .expect("invalid logits");
        let invalid_scale =
            DeviceVec::from_host(ctx, &vec![bf16::from_f32(SELECTED_LOGIT); invalid_experts])
                .expect("invalid scale");
        let mut invalid_index = ctx.stream.alloc_zeros::<i32>(TOP_K).expect("invalid index");
        let mut invalid_weight = ctx
            .stream
            .alloc_zeros::<f32>(TOP_K)
            .expect("invalid weight");
        let error = gemma4_moe_router_topk_into(
            ctx,
            &invalid_logits,
            &invalid_scale,
            TOP_K,
            &mut invalid_index,
            &mut invalid_weight,
        );
        let message = error.expect_err("127 experts must be rejected").to_string();
        assert!(
            message.contains("128-expert register-router contract") && message.contains("127"),
            "unexpected rejection: {message}"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn router_topk_matches_the_exact_128_expert_contract() {
        let ctx = DeviceContext::new().expect("CUDA context");
        let scale_host = scale_host();
        let scale = DeviceVec::from_host(&ctx, &scale_host).expect("scale");
        let alone = assert_finite_contract(&ctx, &scale, &scale_host);

        assert_non_finite_rows_fail_closed(&ctx, &scale, &alone);
        assert_invalid_expert_contract(&ctx);
    }
}
