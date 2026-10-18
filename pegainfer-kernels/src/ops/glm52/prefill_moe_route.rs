//! GLM5.2 TP prefill MoE device-side routing surface: expert-sorted route
//! metadata, fp8/bf16 row gathers, and the deterministic weighted combine.
//! Kernels live in `csrc/glm52/glm52_prefill_moe_route.cu` (hand-written
//! glue; the expert GEMMs themselves run through the FlashInfer CUTLASS
//! grouped template in `glm52_fp8_gemm.cu`).
//!
//! Determinism contract: the slot order inside one expert segment is
//! atomicAdd-order (nondeterministic), but every downstream value is
//! deterministic — each grouped-GEMM row depends only on its own gathered
//! source row, and the combine reads through `route_slot` in fixed
//! `(row, j)` order with f32 accumulation, so no result depends on slot
//! order or atomics.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Build expert-sorted route metadata on device from the router's
/// `topk_idx` (`[rows, topk]` i32). Outputs: `m_indptr` (`[num_experts+1]`,
/// exclusive prefix ends over per-expert route counts), `gather_rows`
/// (`[rows*topk]`, slot → source row) and `route_slot` (`[rows*topk]`,
/// `(row, j)` → slot). `expert_counts` is a `[num_experts]` scratch.
#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_moe_route_launch(
    ctx: &DeviceContext,
    rows: usize,
    topk: usize,
    num_experts: usize,
    topk_idx: &impl DevicePtr<i32>,
    expert_counts: &mut CudaSlice<i32>,
    m_indptr: &mut CudaSlice<i32>,
    gather_rows: &mut CudaSlice<i32>,
    route_slot: &mut CudaSlice<i32>,
) -> Result<()> {
    let routes = rows * topk;
    ensure!(
        rows > 0 && topk > 0 && num_experts > 0,
        "GLM5.2 prefill MoE route shape is invalid"
    );
    ensure!(
        topk_idx.len() >= routes
            && expert_counts.len() >= num_experts
            && m_indptr.len() > num_experts
            && gather_rows.len() >= routes
            && route_slot.len() >= routes,
        "GLM5.2 prefill MoE route buffers are too small for {rows}x{topk} over {num_experts}"
    );
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (counts_ptr, _counts_guard) = expert_counts.device_ptr_mut(&ctx.stream);
    let (indptr_ptr, _indptr_guard) = m_indptr.device_ptr_mut(&ctx.stream);
    let (gather_ptr, _gather_guard) = gather_rows.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = route_slot.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_prefill_moe_route_cuda(
            idx_ptr as *const i32,
            rows as i32,
            topk as i32,
            num_experts as i32,
            counts_ptr as *mut i32,
            indptr_ptr as *mut i32,
            gather_ptr as *mut i32,
            slot_ptr as *mut i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 prefill MoE route metadata launch failed: {err}"))
}

/// Gather bf16 rows into slot order: `output[slot] = input[gather_rows[slot]]`.
pub fn glm52_prefill_moe_gather_rows_launch(
    ctx: &DeviceContext,
    total: usize,
    hidden: usize,
    input: &impl DevicePtr<bf16>,
    gather_rows: &impl DevicePtr<i32>,
    output: &mut impl DevicePtrMut<bf16>,
) -> Result<()> {
    ensure!(
        total > 0
            && hidden > 0
            && hidden.is_multiple_of(8)
            && gather_rows.len() >= total
            && input.len() >= hidden
            && output.len() >= total * hidden,
        "GLM5.2 prefill MoE bf16 gather buffers are invalid"
    );
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = gather_rows.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_prefill_moe_gather_rows_cuda(
            input_ptr as *const ffi::Half,
            rows_ptr as *const i32,
            output_ptr as *mut ffi::Half,
            total as i32,
            hidden as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 prefill MoE bf16 gather launch failed: {err}"))
}

/// Gather pre-quantized fp8 rows plus their per-128-group f32 scales into
/// slot order (the chunk is quantized once; routes reuse the source row's
/// quantization).
#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_moe_gather_fp8_launch(
    ctx: &DeviceContext,
    total: usize,
    k: usize,
    input: &impl DevicePtr<u8>,
    input_scale: &impl DevicePtr<f32>,
    gather_rows: &impl DevicePtr<i32>,
    output: &mut impl DevicePtrMut<u8>,
    output_scale: &mut impl DevicePtrMut<f32>,
) -> Result<()> {
    ensure!(
        total > 0
            && k > 0
            && k.is_multiple_of(128)
            && gather_rows.len() >= total
            && input.len() >= k
            && input_scale.len() >= k / 128
            && output.len() >= total * k
            && output_scale.len() >= total * (k / 128),
        "GLM5.2 prefill MoE fp8 gather buffers are invalid"
    );
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = input_scale.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = gather_rows.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let (out_scale_ptr, _out_scale_guard) = output_scale.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_prefill_moe_gather_fp8_cuda(
            input_ptr as *const u8,
            scale_ptr as *const f32,
            rows_ptr as *const i32,
            output_ptr as *mut u8,
            out_scale_ptr as *mut f32,
            total as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 prefill MoE fp8 gather launch failed: {err}"))
}

/// Deterministic weighted combine of the grouped W2 output back into row
/// order: `output[row] = shared_out[row] + Σ_j topk_weight[row,j] *
/// w2_out[route_slot[row,j]]`, f32 accumulation, one bf16 round.
#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_moe_combine_launch(
    ctx: &DeviceContext,
    rows: usize,
    topk: usize,
    hidden: usize,
    w2_out: &impl DevicePtr<bf16>,
    route_slot: &impl DevicePtr<i32>,
    topk_weight: &impl DevicePtr<f32>,
    shared_out: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
) -> Result<()> {
    ensure!(
        rows > 0
            && topk > 0
            && hidden > 0
            && hidden.is_multiple_of(8)
            && w2_out.len() >= hidden
            && route_slot.len() >= rows * topk
            && topk_weight.len() >= rows * topk
            && shared_out.len() >= rows * hidden
            && output.len() >= rows * hidden,
        "GLM5.2 prefill MoE combine buffers are invalid"
    );
    let (w2_ptr, _w2_guard) = w2_out.device_ptr(&ctx.stream);
    let (slot_ptr, _slot_guard) = route_slot.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (shared_ptr, _shared_guard) = shared_out.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_prefill_moe_combine_cuda(
            w2_ptr as *const ffi::Half,
            slot_ptr as *const i32,
            weight_ptr as *const f32,
            shared_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            rows as i32,
            topk as i32,
            hidden as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 prefill MoE combine launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Route metadata invariants + fp8 gather + deterministic combine on a
    /// small hand-checkable case.
    #[test]
    #[ignore = "requires a GPU"]
    fn route_gather_combine_roundtrip() -> Result<()> {
        const ROWS: usize = 5;
        const TOPK: usize = 3;
        const EXPERTS: usize = 8;
        const HIDDEN: usize = 128;
        let routes = ROWS * TOPK;

        let ctx = DeviceContext::new()?;
        let topk_host: Vec<i32> = vec![
            0, 3, 7, // row 0
            3, 3, 1, // row 1 (duplicate expert on purpose)
            7, 0, 3, // row 2
            2, 4, 5, // row 3
            3, 7, 0, // row 4
        ];
        let weights_host: Vec<f32> = (0..routes).map(|i| 0.125 * (i as f32 + 1.0)).collect();
        let topk_idx = ctx.stream.clone_htod(&topk_host)?;
        let topk_weight = ctx.stream.clone_htod(&weights_host)?;
        let mut counts = ctx.stream.alloc_zeros::<i32>(EXPERTS)?;
        let mut m_indptr = ctx.stream.alloc_zeros::<i32>(EXPERTS + 1)?;
        let mut gather_rows = ctx.stream.alloc_zeros::<i32>(routes)?;
        let mut route_slot = ctx.stream.alloc_zeros::<i32>(routes)?;
        glm52_prefill_moe_route_launch(
            &ctx,
            ROWS,
            TOPK,
            EXPERTS,
            &topk_idx,
            &mut counts,
            &mut m_indptr,
            &mut gather_rows,
            &mut route_slot,
        )?;
        let indptr = ctx.stream.clone_dtoh(&m_indptr)?;
        let gather = ctx.stream.clone_dtoh(&gather_rows)?;
        let slots = ctx.stream.clone_dtoh(&route_slot)?;

        let mut want_counts = [0i32; EXPERTS];
        for &expert in &topk_host {
            want_counts[expert as usize] += 1;
        }
        let mut want_indptr = vec![0i32; EXPERTS + 1];
        for expert in 0..EXPERTS {
            want_indptr[expert + 1] = want_indptr[expert] + want_counts[expert];
        }
        ensure!(
            indptr == want_indptr,
            "m_indptr {indptr:?} != {want_indptr:?}"
        );
        let mut seen = vec![false; routes];
        for row in 0..ROWS {
            for j in 0..TOPK {
                let expert = topk_host[row * TOPK + j] as usize;
                let slot = slots[row * TOPK + j];
                ensure!(
                    (want_indptr[expert]..want_indptr[expert + 1]).contains(&slot),
                    "slot {slot} outside expert {expert} segment"
                );
                ensure!(!seen[slot as usize], "slot {slot} assigned twice");
                seen[slot as usize] = true;
                ensure!(
                    gather[slot as usize] == row as i32,
                    "gather_rows[{slot}] = {} != {row}",
                    gather[slot as usize]
                );
            }
        }

        // fp8 gather: distinct byte pattern per source row.
        let input_host: Vec<u8> = (0..ROWS * HIDDEN)
            .map(|i| ((i / HIDDEN) * 16 + 7) as u8)
            .collect();
        let scale_host: Vec<f32> = (0..ROWS).map(|row| row as f32 + 0.5).collect();
        let input = ctx.stream.clone_htod(&input_host)?;
        let input_scale = ctx.stream.clone_htod(&scale_host)?;
        let mut gathered = ctx.stream.alloc_zeros::<u8>(routes * HIDDEN)?;
        let mut gathered_scale = ctx.stream.alloc_zeros::<f32>(routes)?;
        glm52_prefill_moe_gather_fp8_launch(
            &ctx,
            routes,
            HIDDEN,
            &input,
            &input_scale,
            &gather_rows,
            &mut gathered,
            &mut gathered_scale,
        )?;
        let gathered = ctx.stream.clone_dtoh(&gathered)?;
        let gathered_scale = ctx.stream.clone_dtoh(&gathered_scale)?;
        for slot in 0..routes {
            let row = gather[slot] as usize;
            ensure!(
                gathered[slot * HIDDEN] == (row * 16 + 7) as u8
                    && gathered_scale[slot] == row as f32 + 0.5,
                "fp8 gather slot {slot} mismatch"
            );
        }

        // Combine: w2_out[slot] rows encode their source row; expected
        // output is shared + sum over the row's routes in fixed j order.
        let w2_host: Vec<bf16> = (0..routes * HIDDEN)
            .map(|i| bf16::from_f32((gather[i / HIDDEN] as f32 + 1.0) * 0.5))
            .collect();
        let shared_host: Vec<bf16> = (0..ROWS * HIDDEN)
            .map(|i| bf16::from_f32((i / HIDDEN) as f32 * 0.25))
            .collect();
        let w2_out = ctx.stream.clone_htod(&w2_host)?;
        let shared = ctx.stream.clone_htod(&shared_host)?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
        glm52_prefill_moe_combine_launch(
            &ctx,
            ROWS,
            TOPK,
            HIDDEN,
            &w2_out,
            &route_slot,
            &topk_weight,
            &shared,
            &mut output,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        for row in 0..ROWS {
            let mut want = row as f32 * 0.25;
            for j in 0..TOPK {
                want += weights_host[row * TOPK + j] * (row as f32 + 1.0) * 0.5;
            }
            let got = output[row * HIDDEN].to_f32();
            ensure!(
                (got - want).abs() <= want.abs() * 0.01 + 0.01,
                "combine row {row}: {got} != {want}"
            );
        }
        Ok(())
    }
}
