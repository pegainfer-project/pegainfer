//! First ROCm execution profile: single GPU, BF16, eager ragged batches.
//! Uses the common Qwen3 config and frontend contract. KV capacity is reserved
//! in fixed per-request slots; advanced CUDA-only capabilities are rejected.
mod model;
pub mod model_line;
mod scheduler;
pub use model::Model;
pub use model::ModelOptions;
pub use model::TokenInput;
