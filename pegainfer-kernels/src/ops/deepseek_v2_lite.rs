use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;
use crate::tensor::HiddenStatesRef;

const DSV2_LITE_ACCUM_THREADS: usize = 256;

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| anyhow!("DSV2-Lite {label} element count overflow"))
}

fn as_i32(label: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| anyhow!("DSV2-Lite {label} exceeds i32 range: {value}"))
}

fn as_accum_launch_i32(label: &str, len: usize) -> Result<i32> {
    let max_len = i32::MAX as usize - (DSV2_LITE_ACCUM_THREADS - 1);
    ensure!(
        len <= max_len,
        "DSV2-Lite {label} exceeds CUDA int launch rounding range: {len}"
    );
    as_i32(label, len)
}

fn ensure_hidden_backing_len(
    label: &str,
    data_len: usize,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<usize> {
    let needed = checked_len(label, hidden_dim, seq_len)?;
    as_i32(label, needed)?;
    ensure!(
        data_len >= needed,
        "DSV2-Lite {label} backing buffer too small: have {}, need {needed}",
        data_len
    );
    Ok(needed)
}

fn ensure_matrix_backing_len(label: &str, data_len: usize, rows: usize, cols: usize) -> Result<()> {
    let needed = checked_len(label, rows, cols)?;
    as_i32(label, needed)?;
    ensure!(
        data_len >= needed,
        "DSV2-Lite {label} backing buffer too small: have {}, need {needed}",
        data_len
    );
    Ok(())
}

pub struct Dsv2LiteRouterOutput<'a> {
    pub topk_weight: &'a mut CudaSlice<f32>,
    pub topk_idx: &'a mut CudaSlice<i32>,
}

pub fn dsv2_lite_router_logits_into(
    ctx: &DeviceContext,
    hidden: HiddenStatesRef<'_>,
    gate_weight: &DeviceMatrix,
    logits: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == gate_weight.cols,
        "DSV2-Lite router hidden_dim {} must match gate cols {}",
        hidden.hidden_dim,
        gate_weight.cols
    );
    ensure_hidden_backing_len(
        "router hidden",
        hidden.data.len(),
        hidden.hidden_dim,
        hidden.seq_len,
    )?;
    ensure_matrix_backing_len(
        "router gate",
        gate_weight.data.len(),
        gate_weight.rows,
        gate_weight.cols,
    )?;
    let logits_elems = checked_len("router logits", hidden.seq_len, gate_weight.rows)?;
    as_i32("router logits", logits_elems)?;
    ensure!(
        logits.len() >= logits_elems,
        "DSV2-Lite router logits output too small: have {}, need {logits_elems}",
        logits.len()
    );
    let seq_len = as_i32("router seq_len", hidden.seq_len)?;
    let hidden_dim = as_i32("router hidden_dim", hidden.hidden_dim)?;
    let n_experts = as_i32("router n_experts", gate_weight.rows)?;

    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.data.device_ptr(&ctx.stream);
    let (logits_ptr, _logits_guard) = logits.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_router_logits_cuda(
            hidden_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            logits_ptr as *mut f32,
            seq_len,
            hidden_dim,
            n_experts,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite router logits CUDA launch failed: {err}"))
}

pub fn dsv2_lite_accumulate_route_row_into(
    ctx: &DeviceContext,
    rows: HiddenStatesRef<'_>,
    source_row: usize,
    scale: f32,
    token_idx: usize,
    seq_len: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(scale.is_finite(), "DSV2-Lite route scale must be finite");
    ensure!(
        source_row < rows.seq_len,
        "DSV2-Lite route source row {source_row} exceeds seq_len {}",
        rows.seq_len
    );
    ensure!(
        token_idx < seq_len,
        "DSV2-Lite route token {token_idx} exceeds seq_len {seq_len}"
    );
    ensure_hidden_backing_len("route rows", rows.data.len(), rows.hidden_dim, rows.seq_len)?;
    let output_elems = checked_len("route accumulation output", rows.hidden_dim, seq_len)?;
    as_i32("route accumulation output", output_elems)?;
    ensure!(
        out.len() >= output_elems,
        "DSV2-Lite route accumulation output too small: have {}, need {output_elems}",
        out.len()
    );
    let source_offset = source_row
        .checked_mul(rows.hidden_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite route source offset overflow"))?;
    let source_end = source_offset
        .checked_add(rows.hidden_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite route source row end overflow"))?;
    ensure!(
        source_end <= rows.data.len(),
        "DSV2-Lite route source row backing buffer too small: have {}, need {source_end}",
        rows.data.len()
    );
    let hidden_dim = as_accum_launch_i32("route hidden_dim", rows.hidden_dim)?;
    let token_idx = as_i32("route token_idx", token_idx)?;
    let seq_len = as_i32("route seq_len", seq_len)?;
    let source_view = rows.data.slice(source_offset..source_end);
    let (source_ptr, _source_guard) = source_view.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::accumulate_bf16_token_scaled_to_f32_cuda(
            source_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut f32,
            hidden_dim,
            token_idx,
            seq_len,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite route accumulation CUDA launch failed: {err}"))
}

#[derive(Clone, Copy, Debug)]
pub struct Dsv2LiteAttentionConfig {
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<Dsv2LiteRopeScalingConfig>,
}

#[derive(Clone, Copy, Debug)]
pub struct Dsv2LiteRopeScalingConfig {
    pub factor: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position_embeddings: usize,
}

pub fn dsv2_lite_router_softmax_topk_into(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    gate_weight: &DeviceMatrix,
    topk: usize,
    output: &mut Dsv2LiteRouterOutput<'_>,
) -> Result<()> {
    dsv2_lite_router_softmax_topk_ref_into(ctx, hidden.as_ref(), gate_weight, topk, output)
}

pub fn dsv2_lite_router_softmax_topk_ref_into(
    ctx: &DeviceContext,
    hidden: HiddenStatesRef<'_>,
    gate_weight: &DeviceMatrix,
    topk: usize,
    output: &mut Dsv2LiteRouterOutput<'_>,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == gate_weight.cols,
        "DSV2-Lite router hidden_dim {} must match gate cols {}",
        hidden.hidden_dim,
        gate_weight.cols
    );
    ensure!(
        gate_weight.rows > 0 && topk > 0 && topk <= gate_weight.rows,
        "DSV2-Lite router invalid n_experts={} topk={topk}",
        gate_weight.rows
    );
    ensure_hidden_backing_len(
        "router hidden",
        hidden.data.len(),
        hidden.hidden_dim,
        hidden.seq_len,
    )?;
    ensure_matrix_backing_len(
        "router gate",
        gate_weight.data.len(),
        gate_weight.rows,
        gate_weight.cols,
    )?;
    let route_elems = checked_len("router route", hidden.seq_len, topk)?;
    as_i32("router route", route_elems)?;
    ensure!(
        output.topk_weight.len() >= route_elems,
        "DSV2-Lite router topk_weight too small: have {}, need {route_elems}",
        output.topk_weight.len()
    );
    ensure!(
        output.topk_idx.len() >= route_elems,
        "DSV2-Lite router topk_idx too small: have {}, need {route_elems}",
        output.topk_idx.len()
    );
    let seq_len = as_i32("router seq_len", hidden.seq_len)?;
    let hidden_dim = as_i32("router hidden_dim", hidden.hidden_dim)?;
    let n_experts = as_i32("router n_experts", gate_weight.rows)?;
    let topk = as_i32("router topk", topk)?;

    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _idx_guard) = output.topk_idx.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_router_softmax_topk_cuda(
            hidden_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            weight_ptr as *mut f32,
            idx_ptr as *mut i32,
            seq_len,
            hidden_dim,
            n_experts,
            topk,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite router CUDA launch failed: {err}"))
}

pub fn dsv2_lite_accumulate_fixed_expert_into(
    ctx: &DeviceContext,
    expert_output: &HiddenStates,
    topk_weight: &CudaSlice<f32>,
    topk_idx: &CudaSlice<i32>,
    global_expert: usize,
    topk: usize,
    accum: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        expert_output.hidden_dim > 0 && expert_output.seq_len > 0,
        "DSV2-Lite fixed-expert accumulate requires non-empty expert output"
    );
    let route_elems = checked_len("fixed-expert route", expert_output.seq_len, topk)?;
    as_i32("fixed-expert route", route_elems)?;
    let hidden_elems = ensure_hidden_backing_len(
        "fixed-expert output",
        expert_output.data.len(),
        expert_output.hidden_dim,
        expert_output.seq_len,
    )?;
    ensure!(
        topk_weight.len() >= route_elems && topk_idx.len() >= route_elems,
        "DSV2-Lite fixed-expert route buffers too small: weights={}, idx={}, need {route_elems}",
        topk_weight.len(),
        topk_idx.len()
    );
    ensure!(
        accum.len() >= hidden_elems,
        "DSV2-Lite fixed-expert accum too small: have {}, need {hidden_elems}",
        accum.len()
    );
    as_accum_launch_i32("fixed-expert hidden", hidden_elems)?;
    let global_expert = as_i32("fixed-expert global_expert", global_expert)?;
    let seq_len = as_i32("fixed-expert seq_len", expert_output.seq_len)?;
    let hidden_dim = as_i32("fixed-expert hidden_dim", expert_output.hidden_dim)?;
    let topk = as_i32("fixed-expert topk", topk)?;

    let (expert_ptr, _expert_guard) = expert_output.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (accum_ptr, _accum_guard) = accum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_accumulate_fixed_expert_cuda(
            expert_ptr as *const ffi::Half,
            weight_ptr as *const f32,
            idx_ptr as *const i32,
            accum_ptr as *mut f32,
            global_expert,
            seq_len,
            hidden_dim,
            topk,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite fixed-expert accumulate CUDA launch failed: {err}"))
}

pub fn dsv2_lite_kv_norm_into(
    ctx: &DeviceContext,
    kv_a: &HiddenStates,
    norm_weight: &CudaSlice<bf16>,
    kv_lora_rank: usize,
    eps: f32,
    compressed: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        kv_a.hidden_dim >= kv_lora_rank && compressed.hidden_dim == kv_lora_rank,
        "DSV2-Lite kv norm shape mismatch: kv_a hidden_dim={}, kv_lora_rank={kv_lora_rank}, compressed hidden_dim={}",
        kv_a.hidden_dim,
        compressed.hidden_dim
    );
    ensure!(
        compressed.seq_len == kv_a.seq_len,
        "DSV2-Lite kv norm seq_len mismatch: kv_a={}, compressed={}",
        kv_a.seq_len,
        compressed.seq_len
    );
    ensure!(
        norm_weight.len() >= kv_lora_rank,
        "DSV2-Lite kv norm weight too small: have {}, need {kv_lora_rank}",
        norm_weight.len()
    );
    ensure_hidden_backing_len(
        "kv norm kv_a",
        kv_a.data.len(),
        kv_a.hidden_dim,
        kv_a.seq_len,
    )?;
    ensure_hidden_backing_len(
        "kv norm compressed",
        compressed.data.len(),
        compressed.hidden_dim,
        compressed.seq_len,
    )?;
    let kv_lora_rank = as_i32("kv norm kv_lora_rank", kv_lora_rank)?;
    let kv_a_rows = as_i32("kv norm kv_a rows", kv_a.hidden_dim)?;
    let seq_len = as_i32("kv norm seq_len", kv_a.seq_len)?;
    let (kv_a_ptr, _kv_a_guard) = kv_a.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = norm_weight.device_ptr(&ctx.stream);
    let (compressed_ptr, _compressed_guard) = compressed.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_kv_norm_cuda(
            kv_a_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            compressed_ptr as *mut ffi::Half,
            kv_lora_rank,
            kv_a_rows,
            seq_len,
            eps,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite kv norm CUDA launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn dsv2_lite_decode_attention_into(
    ctx: &DeviceContext,
    cfg: Dsv2LiteAttentionConfig,
    q: &HiddenStates,
    kv_a: &HiddenStates,
    kv_b: &HiddenStates,
    position: usize,
    key_cache: &mut CudaSlice<f32>,
    value_cache: &mut CudaSlice<f32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let query_head_dim = cfg
        .qk_nope_head_dim
        .checked_add(cfg.qk_rope_head_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite attention query head dim overflow"))?;
    let kv_b_stride = cfg
        .qk_nope_head_dim
        .checked_add(cfg.v_head_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite attention kv_b stride overflow"))?;
    let expected_q_dim = cfg
        .num_heads
        .checked_mul(query_head_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite attention q dim overflow"))?;
    let expected_kv_a_dim = cfg
        .kv_lora_rank
        .checked_add(cfg.qk_rope_head_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite attention kv_a dim overflow"))?;
    let expected_kv_b_dim = cfg
        .num_heads
        .checked_mul(kv_b_stride)
        .ok_or_else(|| anyhow!("DSV2-Lite attention kv_b dim overflow"))?;
    let expected_out_dim = cfg
        .num_heads
        .checked_mul(cfg.v_head_dim)
        .ok_or_else(|| anyhow!("DSV2-Lite attention out dim overflow"))?;
    ensure!(
        q.hidden_dim == expected_q_dim && q.seq_len == 1,
        "DSV2-Lite attention q shape mismatch: got [{} x {}], expected [{} x 1]",
        q.hidden_dim,
        q.seq_len,
        expected_q_dim
    );
    ensure!(
        kv_a.hidden_dim == expected_kv_a_dim && kv_a.seq_len == 1,
        "DSV2-Lite attention kv_a shape mismatch: got [{} x {}]",
        kv_a.hidden_dim,
        kv_a.seq_len
    );
    ensure!(
        kv_b.hidden_dim == expected_kv_b_dim && kv_b.seq_len == 1,
        "DSV2-Lite attention kv_b shape mismatch: got [{} x {}]",
        kv_b.hidden_dim,
        kv_b.seq_len
    );
    ensure!(
        out.hidden_dim == expected_out_dim && out.seq_len == 1,
        "DSV2-Lite attention out shape mismatch: got [{} x {}]",
        out.hidden_dim,
        out.seq_len
    );
    ensure!(
        position < cfg.max_seq_len,
        "DSV2-Lite attention position {position} exceeds max_seq_len {}",
        cfg.max_seq_len
    );
    let key_cache_rows = checked_len("attention key cache rows", cfg.max_seq_len, cfg.num_heads)?;
    let key_elems = checked_len("attention key cache", key_cache_rows, query_head_dim)?;
    as_i32("attention key cache", key_elems)?;
    let value_cache_rows =
        checked_len("attention value cache rows", cfg.max_seq_len, cfg.num_heads)?;
    let value_elems = checked_len("attention value cache", value_cache_rows, cfg.v_head_dim)?;
    as_i32("attention value cache", value_elems)?;
    ensure!(
        key_cache.len() >= key_elems,
        "DSV2-Lite attention key cache too small: have {}, need {key_elems}",
        key_cache.len()
    );
    ensure!(
        value_cache.len() >= value_elems,
        "DSV2-Lite attention value cache too small: have {}, need {value_elems}",
        value_cache.len()
    );
    ensure_hidden_backing_len("attention q", q.data.len(), q.hidden_dim, q.seq_len)?;
    ensure_hidden_backing_len(
        "attention kv_a",
        kv_a.data.len(),
        kv_a.hidden_dim,
        kv_a.seq_len,
    )?;
    ensure_hidden_backing_len(
        "attention kv_b",
        kv_b.data.len(),
        kv_b.hidden_dim,
        kv_b.seq_len,
    )?;
    ensure_hidden_backing_len("attention out", out.data.len(), out.hidden_dim, out.seq_len)?;

    let (
        rope_factor,
        rope_mscale,
        rope_mscale_all_dim,
        rope_beta_fast,
        rope_beta_slow,
        rope_original,
        has_rope_scaling,
    ) = cfg
        .rope_scaling
        .map_or((1.0, 1.0, 1.0, 1.0, 1.0, cfg.max_seq_len, 0), |rope| {
            (
                rope.factor,
                rope.mscale,
                rope.mscale_all_dim,
                rope.beta_fast,
                rope.beta_slow,
                rope.original_max_position_embeddings,
                1,
            )
        });
    let position = as_i32("attention position", position)?;
    let num_heads = as_i32("attention num_heads", cfg.num_heads)?;
    let qk_nope_head_dim = as_i32("attention qk_nope_head_dim", cfg.qk_nope_head_dim)?;
    let qk_rope_head_dim = as_i32("attention qk_rope_head_dim", cfg.qk_rope_head_dim)?;
    let v_head_dim = as_i32("attention v_head_dim", cfg.v_head_dim)?;
    let kv_lora_rank = as_i32("attention kv_lora_rank", cfg.kv_lora_rank)?;
    let kv_a_rows = as_i32("attention kv_a rows", kv_a.hidden_dim)?;
    let kv_b_rows = as_i32("attention kv_b rows", kv_b.hidden_dim)?;
    let max_seq_len = as_i32("attention max_seq_len", cfg.max_seq_len)?;
    let rope_original = as_i32(
        "attention rope original max position embeddings",
        rope_original,
    )?;

    let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
    let (kv_a_ptr, _kv_a_guard) = kv_a.data.device_ptr(&ctx.stream);
    let (kv_b_ptr, _kv_b_guard) = kv_b.data.device_ptr(&ctx.stream);
    let (key_cache_ptr, _key_cache_guard) = key_cache.device_ptr_mut(&ctx.stream);
    let (value_cache_ptr, _value_cache_guard) = value_cache.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::dsv2_lite_decode_attention_cuda(
            q_ptr as *const ffi::Half,
            kv_a_ptr as *const ffi::Half,
            kv_b_ptr as *const ffi::Half,
            key_cache_ptr as *mut f32,
            value_cache_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            position,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            kv_a_rows,
            kv_b_rows,
            max_seq_len,
            cfg.rope_theta,
            rope_factor,
            rope_mscale,
            rope_mscale_all_dim,
            rope_beta_fast,
            rope_beta_slow,
            rope_original,
            has_rope_scaling,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("DSV2-Lite decode attention CUDA launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_len_and_as_i32_reject_cuda_int_index_overflow() {
        let max = i32::MAX as usize;

        let max_len = checked_len("unit", max, 1).expect("i32 max len");
        assert_eq!(as_i32("unit", max_len).expect("i32 max"), i32::MAX);
        assert!(as_i32("unit", max + 1).is_err());
        assert!(as_i32("unit", checked_len("unit", max / 2 + 1, 2).unwrap()).is_err());
        assert!(checked_len("unit", usize::MAX, 2).is_err());
    }

    #[test]
    fn accum_launch_i32_rejects_cuda_grid_rounding_overflow() {
        let max = i32::MAX as usize - (DSV2_LITE_ACCUM_THREADS - 1);
        let max_i32 = i32::try_from(max).expect("max rounded fits i32");

        assert_eq!(
            as_accum_launch_i32("unit", max).expect("max rounded"),
            max_i32
        );
        assert!(as_accum_launch_i32("unit", max + 1).is_err());
    }
}
