//! Kimi-K2 text geometry. Not behind the `kimi-k2` feature, so the model crate's config probe,
//! which builds without it, checks a checkpoint against the values the kernels compute with.

pub const KIMI_K2_HIDDEN: usize = 7168;
pub const KIMI_K2_ROUTED_EXPERTS: usize = 384;
pub const KIMI_K2_TOPK: usize = 8;
pub const KIMI_K2_EXPERT_INTERMEDIATE: usize = 2048;
pub const KIMI_K2_INT4_GROUP_SIZE: usize = 32;
pub const KIMI_K2_Q_LORA_RANK: usize = 1536;
pub const KIMI_K2_KV_LORA_RANK: usize = 512;
pub const KIMI_K2_QK_NOPE_HEAD_DIM: usize = 128;
pub const KIMI_K2_QK_ROPE_HEAD_DIM: usize = 64;
pub const KIMI_K2_V_HEAD_DIM: usize = 128;
pub const KIMI_K2_ROUTED_SCALING_FACTOR: f32 = 2.827;
