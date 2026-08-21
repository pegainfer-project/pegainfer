#[cfg(feature = "kernel-call-trace")]
use anyhow::Result;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_kernels::ops::NumericPolicy;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_kernels::ops::numeric_policy;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_kernels::tensor::KernelCall;

#[cfg(feature = "kernel-call-trace")]
use crate::batch_decode_buffers::BatchDecodeBuffers;
#[cfg(feature = "kernel-call-trace")]
use crate::weights::ModelRuntimeConfig;
#[cfg(feature = "kernel-call-trace")]
use crate::weights::Qwen3Model;

pub const MODEL: &str = "qwen3";
pub const PHASE_DECODE: &str = "decode";
pub const NUM_LAYERS: usize = 36;
pub const NUM_Q_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM_VALUE: usize = 128;
pub const KV_DIM_VALUE: usize = NUM_KV_HEADS * HEAD_DIM_VALUE;
pub const RMS_NORM_EPS: f32 = 1.0e-6;

#[cfg(feature = "kernel-call-trace")]
pub fn trace_decode_kernel_calls(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    anyhow::ensure!(batch_size > 0, "batch_size must be greater than zero");
    anyhow::ensure!(kv_len > 0, "kv_len must be greater than zero");

    let model = Qwen3Model::from_safetensors_with_runtime(
        model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: false,
            tensor_parallel: None,
            device_ordinal: 0,
            ..Default::default()
        },
    )?;
    let budget = model.kv_budget();
    let kv_mgr = pegainfer_kv_cache::KvCacheManager::new(
        &model.device_ctx().stream,
        budget.num_layers,
        budget.num_kv_heads,
        budget.head_dim,
        budget.block_size,
        budget.num_blocks,
    )?;
    let layout = pegainfer_core::kv_pool::KvLayout::new(
        budget.num_layers,
        budget.num_kv_heads,
        budget.head_dim,
        budget.block_size,
    )
    .expect("kv layout geometry");

    // Build dummy RequestKvs with the target kv_len.
    //
    // max_output_tokens must be at least 2: `apply_prefill` emits the first
    // generated token (counted as 1 toward the budget), and tracing a decode
    // step needs one more. With max_output_tokens = 1 the sequence is already
    // complete after prefill, so `schedule_decode` fails with GenerationComplete.
    let dummy_prompt_len = if kv_len > 1 { kv_len - 1 } else { 1 };
    let rkvs = (0..batch_size)
        .map(|_| {
            let mut rkv = kv_mgr
                .pool()
                .new_request(vec![0; dummy_prompt_len], 2, None);
            rkv.schedule_prefill(dummy_prompt_len, kv_mgr.pool())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            rkv.apply_prefill(0, kv_mgr.pool())?;
            // Now kv_position == dummy_prompt_len. Schedule one decode step.
            rkv.schedule_decode(kv_mgr.pool())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(rkv)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut bufs = BatchDecodeBuffers::new(
        model.device_ctx(),
        model.config().hidden_size,
        model.local_q_dim(),
        model.local_kv_dim(),
        model.local_intermediate_size(),
        model.config().vocab_size,
        batch_size,
        layout.page_size,
        kv_mgr.pool().padding_block_id(),
        model.local_num_attention_heads(),
        model.config().max_position_embeddings,
    )?;
    // This trace path bypasses the serving executor (which warms the pinned shapes at startup);
    // warm here, outside `collect_result` below so it isn't recorded as a kernel call.
    if numeric_policy() == NumericPolicy::Pin {
        crate::batch_decode_buffers::warmup_decode_projection_pins(
            model.config().hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config().vocab_size,
        )?;
    }
    let token_ids = vec![0_u32; batch_size];
    let views: Vec<_> = rkvs.iter().map(|r| r.decode_view()).collect();
    let ((), calls) = call_trace::collect_result(|| {
        model.batch_decode(
            &token_ids,
            &views,
            &vec![None; batch_size],
            kv_mgr.buffer().buffer(),
            &layout,
            &mut bufs,
            crate::batch_decode::DecodeGraphUse::Serve,
        )
    })?;
    Ok(calls)
}

pub fn normalize_call_site(label: &str) -> String {
    let Some(rest) = label.strip_prefix('L') else {
        return label.to_string();
    };
    let digit_count = rest
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 || rest.as_bytes().get(digit_count) != Some(&b'.') {
        return label.to_string();
    }
    format!("layer.*{}", &rest[digit_count..])
}
