//! Higgs Audio model-line scaffolding.
//!
//! This crate starts with the artifact and golden-contract boundary for the
//! zero-shot, one-step `[8, 1026]` audio-logits gate. Runtime execution is added
//! behind this boundary in later slices so stale fixture assumptions cannot leak
//! into the model implementation.

pub mod compare;
pub mod config;
pub mod kernel_plan;
pub mod layer_dump;
pub mod load_plan;
pub mod materialize_qwen3;
pub mod one_step_actual;
pub mod one_step_golden;
#[cfg(feature = "runtime-qwen3")]
pub mod runtime_bridge;
pub mod runtime_source;
pub mod weights;

pub use kernel_plan::kernel_plan;
