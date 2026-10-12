//! The Qwen4 architecture preview (`qwen4_exp`): line detection, validated geometry
//! and the tensor contract. First checkpoint `Qwen/Qwen3.8-Flash-Next`, recorded in
//! `docs/models/qwen4/bring-up.md`.
//!
//! No CUDA dependency at all, so its probe, config and manifest tests run with no
//! device, no weights and no Triton build. Slice B adds the device side behind a
//! feature of its own, and these modules then move behind
//! `#[cfg(any(<that feature>, test))]` the way `pegainfer-gemma4/src/lib.rs` does.

mod config;
// The loader that reads a checkpoint against the contract arrives with slice B.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the checkpoint loader is the first runtime caller"
    )
)]
mod manifest;
pub mod model_line;
mod probe;

pub use probe::MODEL_TYPE;
pub use probe::TEXT_MODEL_TYPE;
