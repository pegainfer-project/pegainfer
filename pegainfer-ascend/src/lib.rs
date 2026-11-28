#![cfg(feature = "ascend")]
//! Ascend (CANN) backend: AscendCL device resources, ACLNN operators, and the
//! Qwen3 line that runs on them. The CUDA crates never reach this crate; the
//! server links it through its `ascend` feature and finds the line through
//! `ModelLine`.

pub mod device;
pub mod ops;
pub mod qwen3;
#[allow(dead_code)]
mod sys;
