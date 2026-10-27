//! The Qwen4 architecture preview (`qwen4_exp`): line detection, validated geometry
//! and the tensor contract. First checkpoint `Qwen/Qwen3.8-Flash-Next`, recorded in
//! `docs/models/qwen4/bring-up.md`.
//!
//! No CUDA dependency at all, so its probe, config and manifest tests run with no
//! device, no weights and no Triton build. Slice B adds the device side behind a
//! feature of its own, and these modules then move behind
//! `#[cfg(any(<that feature>, test))]` the way `pegainfer-gemma4/src/lib.rs` does.
//! The Gated DeltaNet decode wrappers this line will drive already live in
//! `pegainfer_kernels::ops` under the kernels' `qwen35` feature.

mod config;
// Only tests read the contract until slice B's loader checks a checkpoint with it.
#[cfg(test)]
mod manifest;
pub mod model_line;
mod probe;

pub use probe::MODEL_TYPE;
pub use probe::TEXT_MODEL_TYPE;
