//! The Qwen4 architecture preview (`qwen4_exp`): line detection, validated geometry,
//! the tensor contract and, behind the `qwen4` feature, the device half. First
//! checkpoint `Qwen/Qwen3.8-Flash-Next`, recorded in `docs/models/qwen4/bring-up.md`.
//!
//! Without the feature the crate has no CUDA dependency, so its probe, config and
//! manifest tests run with no device, no weights and no Triton build.

mod config;
#[cfg(feature = "qwen4")]
pub mod layer;
#[cfg(any(feature = "qwen4", test))]
mod manifest;
#[cfg(feature = "qwen4")]
pub mod model;
pub mod model_line;
#[cfg(feature = "qwen4")]
pub mod moe;
// The PLE block's n-gram addressing: pure integer host math, so it belongs to
// this crate's CUDA-free half and the device gather consumes what it produces.
pub mod ngram;
mod probe;
#[cfg(feature = "qwen4")]
pub mod weights;

pub use probe::MODEL_TYPE;
pub use probe::TEXT_MODEL_TYPE;
