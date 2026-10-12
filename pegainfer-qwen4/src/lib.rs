//! The Qwen4 architecture preview (`qwen4_exp`): line detection, validated
//! geometry and the tensor contract. Its first checkpoint is Qwen3.8-Flash-Next;
//! the frozen revision and the reference release are recorded in
//! `docs/models/qwen4/bring-up.md`.
//!
//! This crate carries **no CUDA dependency at all**, which is why its probe,
//! config and manifest tests run with no device, no weights and no
//! Triton-equipped build. Slice B (text-graph assembly and the decode step) adds
//! the device side behind a crate feature of its own; the modules below then move
//! behind `#[cfg(any(<that feature>, test))]` the way `pegainfer-gemma4/src/lib.rs`
//! does, so this property survives.

mod config;
// The loader that reads a checkpoint against the contract arrives with slice B;
// until then only this crate's own tests call into the manifest.
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
