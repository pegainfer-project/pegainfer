//! First Ascend execution profile: single device, BF16, eager ragged batches.
//! Reads the Qwen3 config as the CUDA line does and serves through the frontend
//! contract. KV capacity is reserved in fixed per-request slots; advanced
//! CUDA-only capabilities are rejected.
mod config;
mod model;
pub mod model_line;
mod scheduler;
pub use model::Model;
pub use model::ModelOptions;
pub use model::TokenInput;
