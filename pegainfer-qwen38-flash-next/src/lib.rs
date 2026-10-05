//! Qwen3.8-Flash-Next (`qwen4_exp`): line detection and validated geometry.
//!
//! This crate carries **no CUDA dependency at all**, which is why its probe and
//! config tests run with no device, no weights and no Triton-equipped build.
//! Slice B (text-graph assembly and the decode step) adds the device side; when it
//! does, the modules below must move behind
//! `#[cfg(any(feature = "qwen38-flash-next", test))]` the way
//! `pegainfer-gemma4/src/lib.rs` does, so this property survives.
//!
//! Frozen target: `Qwen/Qwen3.8-Flash-Next` at revision
//! `de4b8e4d43b917e7706784d8bb445c9af86a3540` (public, non-gated,
//! `license: other`). See [`config`] for the geometry that revision implies.

// Nothing outside this crate's own test suite consumes the geometry yet: the
// tensor contract that reads a checkpoint against it, and the loader behind that,
// arrive next. Until then the non-test build has no runtime caller, so
// `dead_code` would fire on the entire crate. Test builds keep the lint, so
// genuinely unused items still surface.
#![cfg_attr(not(test), allow(dead_code))]

mod config;
pub mod model_line;
mod probe;

pub use config::FROZEN_CONFIG_SHA256;
pub use config::FROZEN_REVISION;
pub use config::PINNED_TRANSFORMERS;
