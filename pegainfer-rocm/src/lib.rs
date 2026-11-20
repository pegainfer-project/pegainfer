#![cfg(feature = "rocm")]
//! ROCm backend: HIP device resources, BF16 eager operators, and the Qwen3 line
//! that runs on them. The CUDA crates never reach this crate; the server links
//! it through its `rocm` feature and finds the line through `ModelLine`.

pub mod device;
pub mod ops;
pub mod qwen3;
