//! Qwen3.5 GPU operation wrappers.

pub(crate) use pegainfer_core::ops::GEMM_LT_MAX_N;
pub(crate) use pegainfer_core::ops::PrefillPagedPlan;
pub(crate) use pegainfer_core::ops::add_batch;
pub(crate) use pegainfer_core::ops::add_batch_into;
pub(crate) use pegainfer_core::ops::copy_hidden_token_range_into;
pub(crate) use pegainfer_core::ops::embedding_batch;
pub(crate) use pegainfer_core::ops::extract_vec;
pub(crate) use pegainfer_core::ops::gemm;
pub(crate) use pegainfer_core::ops::gemm_into;
pub(crate) use pegainfer_core::ops::gemm_into_checked;
pub(crate) use pegainfer_core::ops::gemm_lt_tune;
pub(crate) use pegainfer_core::ops::gemm_rows_into_checked;
pub(crate) use pegainfer_core::ops::paged_attention_batch_decode_hd256_into;
pub(crate) use pegainfer_core::ops::paged_attention_batch_decode_via_prefill_hd256_into;
pub(crate) use pegainfer_core::ops::qk_norm_partial_rope_batched_decode_hd256_into;
pub use pegainfer_core::ops::rms_norm_batch_offset_into;
pub(crate) use pegainfer_core::ops::rms_norm_gated_batch_into;
pub use pegainfer_core::ops::rms_norm_offset_into;
pub(crate) use pegainfer_core::ops::silu_mul_fused_batch_into;
pub(crate) use pegainfer_core::ops::write_vec_into;
pub(crate) use recurrent::conv1d_decode_batch_into;
pub(crate) use recurrent::conv1d_prefill_batch_into;
pub(crate) use recurrent::gated_delta_rule_decode_batch_into;
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;

use crate::recurrent;
