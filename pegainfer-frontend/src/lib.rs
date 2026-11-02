//! Serving frontend: the engine request/event contract plus the protocol
//! stacks that sit on top of it.
//!
//! The contract half (`engine`, `sampler`, `parallel`, `tracing_state`) is
//! what model crates implement against: a scheduler that reports
//! `StepOutputs` each step, or on the lines not yet migrated a
//! `(GenerateRequest, KvPrefix)` submit channel and a `TokenSink` event
//! stream, with no CUDA types anywhere. The protocol half (`vllm`, later `dynamo`) translates that
//! contract to a concrete HTTP serving stack. `model_line` is the seam the
//! server binary uses to dispatch a detected model to its crate.

pub mod engine;
pub mod model_line;
pub mod parallel;
pub mod sampler;
pub mod tracing_state;
pub mod vllm;
