//! The Qwen4 architecture preview (`qwen4_exp`): line detection, validated geometry
//! and the tensor contract. First checkpoint `Qwen/Qwen3.8-Flash-Next`, recorded in
//! `docs/models/qwen4/bring-up.md`.
//!
//! No CUDA dependency in the default build, so the probe, config and manifest tests
//! run with no device, no weights and no Triton build. The `qwen4` feature turns on
//! the device side: this line's Gated DeltaNet layers reuse the Qwen3.5 decode
//! kernels through `pegainfer-core::ops`, and `tests/gdn_decode_reuse.rs` proves that
//! reuse against layer 0 of the real checkpoint. Further device modules move behind
//! `#[cfg(any(feature = "qwen4", test))]` the way `pegainfer-gemma4/src/lib.rs` does.

mod config;
// Only tests read the contract until slice B's loader checks a checkpoint with it.
#[cfg(test)]
mod manifest;
pub mod model_line;
mod probe;

pub use probe::MODEL_TYPE;
pub use probe::TEXT_MODEL_TYPE;
