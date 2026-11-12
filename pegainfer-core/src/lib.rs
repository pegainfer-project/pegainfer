//! Shared runtime API used by pegainfer model crates.

#[cfg(feature = "cuda")]
pub mod cpu_topology;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
pub mod decodable_vocab;
pub mod eos;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod kv_pool;
pub mod logging;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod page_pool;
#[cfg(feature = "cuda")]
pub mod rope;
#[cfg(feature = "cuda")]
pub mod tensor;
pub mod tracing;
#[cfg(feature = "cuda")]
pub mod weight_loader;

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("core CUDA and ROCm backends are mutually exclusive");
