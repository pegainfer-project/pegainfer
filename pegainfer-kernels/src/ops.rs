//! GPU operations on device tensors.

mod attention;
#[cfg(feature = "moe")]
mod deepep;
#[cfg(feature = "deepseek-v2-lite")]
mod deepseek_v2_lite;
mod elementwise;
mod embedding;
#[cfg(feature = "glm52")]
mod glm52;
#[cfg(feature = "k3")]
mod k3;
#[cfg(feature = "k3")]
mod k3_tilelang;
#[cfg(feature = "kimi-k2")]
mod kimi_k2;
mod linear;
mod lora;
mod norm;
mod sampling;

pub use attention::Hd512DecodeMetadata;
pub use attention::PrefillPagedPlan;
pub use attention::SUPPORTED_GQA_GROUP_SIZES;
pub use attention::batch_prefill_paged_hd512_into;
pub use attention::batch_prefill_paged_window_hd256_into;
pub use attention::dflash_qk_norm_rope_into;
pub use attention::eagle3_rope_into;
pub use attention::paged_attention_batch_decode_hd256_into;
pub use attention::paged_attention_batch_decode_into;
pub use attention::paged_attention_batch_decode_split_kv_hd512_into;
pub use attention::paged_attention_batch_decode_split_kv_into;
pub use attention::paged_attention_batch_decode_via_prefill_hd256_into;
pub use attention::paged_attention_batch_decode_via_prefill_hd512_into;
pub use attention::prefill_attention_paged_into;
pub use attention::qk_norm_partial_rope_batched_decode_hd256_into;
pub use attention::qk_norm_partial_rope_batched_decode_hd512_into;
pub use attention::qk_norm_partial_rope_paged_decode_hd512_into;
pub use attention::qk_norm_partial_rope_paged_prefill_hd512_into;
pub use attention::qk_norm_rope_batch_decode_into;
pub use attention::qk_norm_rope_prefill_hd256_plain_into;
pub use attention::qkv_norm_rope_paged_decode_hd256_plain_into;
pub use attention::qkv_norm_rope_paged_prefill_hd256_plain_into;
pub use attention::single_decode_nhd_into;
pub use attention::single_prefill_hd256_into;
pub use attention::single_prefill_hd512_into;
pub use attention::single_prefill_nhd_causal_into;
pub use attention::single_prefill_nhd_noncausal_into;
#[cfg(feature = "moe")]
pub use deepep::DeepEp;
#[cfg(feature = "moe")]
pub use deepep::DeepEpAbi;
#[cfg(feature = "moe")]
pub use deepep::DeepEpBase;
#[cfg(feature = "moe")]
pub use deepep::DeepEpDispatchScratch;
#[cfg(feature = "moe")]
pub use deepep::DeepEpPrefillCounts;
#[cfg(feature = "glm52")]
pub use deepep::Glm52DeepEp;
#[cfg(feature = "glm52")]
pub use deepep::Glm52DeepEpAbi;
#[cfg(feature = "glm52")]
pub use deepep::Glm52Ep4DeepEpAbi;
#[cfg(feature = "glm52")]
pub use deepep::Glm52Ep16DeepEpAbi;
#[cfg(feature = "glm52")]
pub use deepep::Glm52Ep32DeepEpAbi;
#[cfg(feature = "glm52")]
pub use deepep::Glm52Ep64DeepEpAbi;
#[cfg(feature = "moe")]
pub use deepep::deepep_info;
#[cfg(feature = "moe")]
pub use deepep::deepep_unique_id;
#[cfg(feature = "glm52")]
pub use deepep::glm52_deepep_info;
#[cfg(feature = "glm52")]
pub use deepep::glm52_ep_deepep_unique_id;
#[cfg(feature = "deepseek-v2-lite")]
pub use deepseek_v2_lite::*;
pub use elementwise::accumulate_bf16_token_scaled_to_f32_into;
pub use elementwise::add_batch;
pub use elementwise::add_batch_into;
pub use elementwise::add_into;
pub use elementwise::add_scaled_bf16_into;
pub use elementwise::bf16_bytes_to_f32_into;
pub use elementwise::bf16_hidden_to_f32_into;
pub use elementwise::copy_hidden_rows_into;
pub use elementwise::copy_hidden_rows_raw_into;
pub use elementwise::copy_hidden_token_range_into;
pub use elementwise::extract_hidden_rows_raw_into;
pub use elementwise::extract_vec;
pub use elementwise::extract_vec_into;
pub use elementwise::extract_vec_ref;
pub use elementwise::extract_vec_ref_into;
pub use elementwise::f32_to_bf16_hidden_into;
pub use elementwise::gather_hidden_tokens_into;
pub use elementwise::gelu_tanh_mul_batch_into;
pub use elementwise::mask_position_zero_rows_into;
pub use elementwise::repeat_f32_for_reduce_scatter_into;
pub use elementwise::scale_bf16_in_place;
pub use elementwise::scale_f32_in_place;
pub use elementwise::scaled_add_batch_into;
pub use elementwise::scaled_add_rows_indexed_into;
pub use elementwise::scaled_add_rows_into;
pub use elementwise::scaled_add_rows_token_range_into;
pub use elementwise::silu_mul_batch;
pub use elementwise::silu_mul_batch_into;
pub use elementwise::silu_mul_fused_batch_into;
pub use elementwise::softcap_bf16_in_place;
pub use elementwise::write_vec_into;
pub use embedding::embedding_batch;
pub use embedding::embedding_batch_vocab_shard;
pub use embedding::embedding_decode_into;
pub use embedding::embedding_rows_into;
#[cfg(feature = "glm52")]
pub use glm52::*;
#[cfg(feature = "k3")]
pub use k3::*;
#[cfg(feature = "k3")]
pub use k3_tilelang::*;
#[cfg(feature = "kimi-k2")]
pub use kimi_k2::*;
pub use linear::GEMM_LT_MAX_N;
pub use linear::NumericPolicy;
pub use linear::PinAlgoConfig;
pub(crate) use linear::ensure_tuned_policy;
pub use linear::gemm;
pub use linear::gemm_bf16_f32;
pub use linear::gemm_graphsafe_into_checked;
pub use linear::gemm_graphsafe_ref_into_checked;
pub use linear::gemm_into;
pub use linear::gemm_into_checked;
pub use linear::gemm_lt_pin_check;
pub use linear::gemm_lt_pin_into_checked;
pub use linear::gemm_lt_pin_tune;
pub use linear::gemm_lt_pin_warmup;
pub use linear::gemm_lt_tune;
pub use linear::gemm_per_token;
pub use linear::gemm_per_token_into_checked;
pub use linear::gemm_rows_into;
pub use linear::gemm_rows_into_checked;
pub use linear::gemm_strided_batched_bf16;
pub use linear::gemm_token_range_into_checked;
pub use linear::gemv;
pub use linear::linear;
pub use linear::numeric_policy;
pub use linear::per_token_served;
pub use linear::pin_served;
pub use linear::reset_numeric_policy_counters;
pub use linear::set_numeric_policy;
pub use linear::with_gemm_lt_disabled;
pub use lora::LoraDecodeGroupedProjection;
pub use lora::lora_decode_fused_delta_group3_into;
pub use lora::lora_decode_fused_delta_into;
pub use lora::pack_lora_b_rows_into;
pub use norm::fused_add_rms_norm_batch_into;
pub use norm::fused_add_rms_norm_into;
pub use norm::fused_add_rms_norm_round_batch_into;
pub use norm::fused_add_rms_norm_round_into;
pub use norm::layer_norm_into;
pub use norm::rms_norm;
pub use norm::rms_norm_batch_into;
pub use norm::rms_norm_batch_offset_into;
pub use norm::rms_norm_gated_batch_into;
pub use norm::rms_norm_into;
pub use norm::rms_norm_offset_into;
pub use norm::rms_norm_rows_into;
pub use sampling::BatchSamplingRow;
pub use sampling::BatchSamplingScratch;
pub use sampling::argmax;
pub use sampling::argmax_batch_bf16_into;
pub use sampling::argmax_batch_bf16_split_indexed_into;
pub use sampling::argmax_batch_bf16_split_partials_len;
pub use sampling::argmax_bf16_split_into;
pub use sampling::flashinfer_top1_batch_into;
pub use sampling::flashinfer_top1_row_states_bytes;
pub use sampling::gpu_sample_batch_into;
pub use sampling::logprob_topk_batch_bf16_into;
pub use sampling::markov_step_argmax_into;
pub use sampling::markov_step_argmax_partials_len;

pub(crate) fn checked_i32(value: usize, what: &str) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("{what} {value} does not fit i32"))
}

pub(crate) fn checked_u32(value: usize, what: &str) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("{what} {value} does not fit u32"))
}

/// Calling thread's last FFI exception message, ready to append to an error;
/// empty unless `result` is the -1 sentinel set by the C++ guard. Public for
/// crates that call guarded FFI entries directly instead of through a wrapper.
pub fn ffi_exception_message(result: i32) -> String {
    if result != -1 {
        return String::new();
    }
    let ptr = unsafe { crate::ffi::pegainfer_kernels_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    let text = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy();
    if text.is_empty() {
        String::new()
    } else {
        format!(": {text}")
    }
}
