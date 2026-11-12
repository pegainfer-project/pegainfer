//! Qwen3 model line, selected at build time for CUDA or ROCm.
#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("select exactly one of cuda or rocm (disable default features for ROCm)");
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
compile_error!("Qwen3 requires the cuda or rocm backend");
mod config;
#[cfg(feature = "cuda")]
include!("cuda.rs");
#[cfg(feature = "rocm")]
pub mod rocm;
#[cfg(feature = "rocm")]
pub use rocm::model_line;
