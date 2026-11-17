#![allow(incomplete_features)]
#![cfg_attr(feature = "cuda", feature(generic_const_exprs))]

#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub mod forward_pass;
#[cfg(feature = "cuda")]
pub mod gpu_buffers;
#[cfg(feature = "cuda")]
pub mod green_ctx;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "tvm-ffi-triton-cubin")]
#[cfg(feature = "cuda")]
pub mod triton_cubin;
#[cfg(feature = "cuda")]
pub mod typed_ops;

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("kernel CUDA and ROCm backends are mutually exclusive");
/// DFlash2 checkpoint selection width, shared by host config validation.
pub const DFLASH2_CANDIDATE_K: usize = 16;
#[cfg(feature = "rocm")]
pub mod rocm;
