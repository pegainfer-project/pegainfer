use pegainfer_kernels::ffi;

use super::*;

fn all_reduce_hidden_via_f32_in_place<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &mut GpuTensor<DIM>,
    f32_scratch: &mut CudaSlice<f32>,
    comm: &Comm,
) -> Result<()> {
    typed_ops::bf16_to_f32_into(ctx, hidden, f32_scratch)?;
    all_reduce_f32_bulk_in_place(f32_scratch, hidden.seq_len, DIM, comm)?;
    typed_ops::f32_to_bf16_into(ctx, f32_scratch, hidden)
}

pub(in crate::runner) fn maybe_all_reduce_hidden_via_f32_in_place<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &mut GpuTensor<DIM>,
    f32_scratch: &mut CudaSlice<f32>,
    comm: Option<&Comm>,
) -> Result<()> {
    if let Some(comm) = comm {
        all_reduce_hidden_via_f32_in_place(ctx, hidden, f32_scratch, comm)
    } else {
        Ok(())
    }
}

// The `pub(in crate::runner)` collective + Marlin helpers below are re-exported
// by `worker.rs` for the sibling `moe_nccl` backend, which is why their
// visibility is widened past the usual `pub(super)`.
pub(in crate::runner) fn all_reduce_f32_in_place(
    values: &mut CudaSlice<f32>,
    comm: &Comm,
) -> Result<()> {
    comm.all_reduce_in_place(values, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi TP/EP f32 sum failed: status={:?}", err.0))
}

fn all_reduce_f32_bulk_in_place(
    values: &mut CudaSlice<f32>,
    rows: usize,
    row_len: usize,
    comm: &Comm,
) -> Result<()> {
    ensure!(
        values.len() >= rows * row_len,
        "Kimi bulk f32 all-reduce len {} < rows {} * row_len {}",
        values.len(),
        rows,
        row_len
    );
    let mut view = values.slice_mut(0..rows * row_len);
    comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi bulk f32 all-reduce failed: status={:?}", err.0))
}

pub(in crate::runner) fn reduce_scatter_f32_hidden_into(
    global: &CudaSlice<f32>,
    global_rows: usize,
    row_len: usize,
    local: &mut CudaSlice<f32>,
    local_rows: usize,
    world_size: usize,
    comm: &Comm,
) -> Result<()> {
    ensure!(world_size > 0, "Kimi RS world size must be positive");
    ensure!(
        global_rows == local_rows * world_size,
        "Kimi RS rows mismatch: global_rows={global_rows}, local_rows={local_rows}, world_size={world_size}"
    );
    ensure!(
        global.len() >= global_rows * row_len,
        "Kimi RS global buffer too small: have {}, need {}",
        global.len(),
        global_rows * row_len
    );
    ensure!(
        local.len() >= local_rows * row_len,
        "Kimi RS local buffer too small: have {}, need {}",
        local.len(),
        local_rows * row_len
    );
    let global_view = global.slice(0..global_rows * row_len);
    let mut local_view = local.slice_mut(0..local_rows * row_len);
    comm.reduce_scatter(&global_view, &mut local_view, &ReduceOp::Sum)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("Kimi routed RS f32 hidden failed: status={:?}", err.0))
}

pub(super) fn kimi_mla_softmax_scale() -> f32 {
    let base = (KIMI_K2_MLA_Q_HEAD_DIM as f32).sqrt().recip();
    let mscale = yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0);
    base * mscale * mscale
}

pub(in crate::runner) fn kimi_marlin_block_size(active_tokens: usize) -> usize {
    for block_size in [8usize, 16, 32, 48, KIMI_MARLIN_MAX_BLOCK_SIZE] {
        let routes_per_expert_block = active_tokens as f32 * KIMI_K2_TOPK as f32
            / KIMI_K2_LOCAL_EXPERTS as f32
            / block_size as f32;
        if routes_per_expert_block < 0.9 {
            return block_size;
        }
    }
    KIMI_MARLIN_MAX_BLOCK_SIZE
}

pub(super) fn build_yarn_rope_cache(seq_len: usize) -> (Vec<half::bf16>, Vec<half::bf16>) {
    let dim = KIMI_K2_QK_ROPE_HEAD_DIM;
    let half_dim = dim / 2;
    let (low, high) = yarn_find_correction_range(
        KIMI_K2_YARN_BETA_FAST,
        KIMI_K2_YARN_BETA_SLOW,
        dim,
        KIMI_K2_ROPE_THETA,
        KIMI_K2_YARN_ORIGINAL_MAX_POS,
    );
    let rope_mscale =
        yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0) / yarn_get_mscale(KIMI_K2_YARN_FACTOR, 1.0);
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / dim as f32;
        let freq_extra = 1.0 / KIMI_K2_ROPE_THETA.powf(exponent);
        let freq_inter = 1.0 / (KIMI_K2_YARN_FACTOR * KIMI_K2_ROPE_THETA.powf(exponent));
        let ramp = yarn_linear_ramp_mask(i as f32, low as f32, high as f32);
        inv_freq.push(freq_inter * ramp + freq_extra * (1.0 - ramp));
    }

    let mut cos = vec![half::bf16::from_f32(0.0); seq_len * dim];
    let mut sin = vec![half::bf16::from_f32(0.0); seq_len * dim];
    for token in 0..seq_len {
        for i in 0..half_dim {
            let freq = token as f32 * inv_freq[i];
            let c = half::bf16::from_f32(freq.cos() * rope_mscale);
            let s = half::bf16::from_f32(freq.sin() * rope_mscale);
            cos[token * dim + i] = c;
            sin[token * dim + i] = s;
            cos[token * dim + half_dim + i] = c;
            sin[token * dim + half_dim + i] = s;
        }
    }
    (cos, sin)
}

fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> (usize, usize) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
    (
        low.max(0.0) as usize,
        (high as usize).min(dim.saturating_sub(1)),
    )
}

fn yarn_find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32
        * (max_position_embeddings as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_linear_ramp_mask(value: f32, min: f32, max: f32) -> f32 {
    let denom = if (max - min).abs() < f32::EPSILON {
        0.001
    } else {
        max - min
    };
    ((value - min) / denom).clamp(0.0, 1.0)
}

pub(super) fn sample_local_top1_with_value(
    ctx: &DeviceContext,
    logits: &DeviceVec,
) -> Result<(u32, f32)> {
    let mut top1_value_scratch = ctx.stream.alloc_zeros::<half::bf16>(1)?;
    let mut row_states_scratch = ctx
        .stream
        .alloc_zeros::<u8>(flashinfer_top1_row_states_bytes())?;
    let mut out = ctx.stream.alloc_zeros::<i32>(1)?;
    sample_local_top1_with_value_reuse(
        ctx,
        logits,
        &mut top1_value_scratch,
        &mut row_states_scratch,
        &mut out,
    )
}

fn sample_local_top1_with_value_reuse(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<(u32, f32)> {
    ensure!(
        !top1_value_scratch.is_empty() && !out.is_empty(),
        "Kimi top1 scratch must have scalar value/id outputs"
    );
    ensure!(
        row_states_scratch.len() >= flashinfer_top1_row_states_bytes(),
        "Kimi top1 row-states scratch too small: have {}, need {}",
        row_states_scratch.len(),
        flashinfer_top1_row_states_bytes()
    );
    {
        let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
        let (value_ptr, _value_guard) = top1_value_scratch.device_ptr_mut(&ctx.stream);
        let (row_states_ptr, _row_states_guard) = row_states_scratch.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::flashinfer_top1_cuda(
                logits_ptr as *const ffi::Half,
                value_ptr as *mut ffi::Half,
                row_states_ptr as *mut u8,
                out_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
    ctx.sync()?;
    let top_id = ctx
        .stream
        .clone_dtoh(&*out)
        .map_err(|err| anyhow::anyhow!("D2H Kimi top1 id read failed: {err}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Kimi top1 id output was empty"))?;
    let top_value = ctx
        .stream
        .clone_dtoh(&*top1_value_scratch)
        .map_err(|err| anyhow::anyhow!("D2H Kimi top1 value read failed: {err}"))?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Kimi top1 value output was empty"))?
        .to_f32();
    ensure!(
        top_id >= 0 && (top_id as usize) < logits.len,
        "Kimi local top1 id {} out of logits range {}",
        top_id,
        logits.len
    );
    Ok((top_id as u32, top_value))
}

pub(super) fn launch_local_top1_batch(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    active_rows: usize,
    top1_values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        active_rows > 0 && active_rows <= logits.seq_len,
        "Kimi batched top1 active_rows {active_rows} must be in 1..={}",
        logits.seq_len
    );
    ensure!(
        top1_values.len() >= active_rows && out.len() >= active_rows,
        "Kimi batched top1 scratch too small: values={}, out={}, active_rows={}",
        top1_values.len(),
        out.len(),
        active_rows
    );
    let partials = pegainfer_kernels::ops::argmax_batch_bf16_split_partials_len(
        active_rows,
        logits.hidden_dim,
    );
    ensure!(
        partial_values.len() >= partials && partial_indices.len() >= partials,
        "Kimi batched top1 partial scratch too small: values={}, indices={}, need={}",
        partial_values.len(),
        partial_indices.len(),
        partials
    );
    {
        let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
        let (value_ptr, _value_guard) = top1_values.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let (partial_values_ptr, _partial_values_guard) =
            partial_values.device_ptr_mut(&ctx.stream);
        let (partial_indices_ptr, _partial_indices_guard) =
            partial_indices.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_batch_bf16_split_cuda(
                logits_ptr as *const ffi::Half,
                value_ptr as *mut ffi::Half,
                out_ptr as *mut i32,
                partial_values_ptr as *mut f32,
                partial_indices_ptr as *mut i32,
                active_rows as i32,
                logits.hidden_dim as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(())
}
