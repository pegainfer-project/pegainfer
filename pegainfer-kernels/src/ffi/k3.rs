//! Kimi-K3 CUDA entry points.
//!
//! The masked grouped GEMM is DeepGEMM's SM100 `MGroupedMasked` FP8 x FP4
//! kernel, AOT-instantiated (no JIT, no torch). Both scale operands are the
//! Blackwell packed-UE8M0 i32 layout (`[groups, ceil(k / gran_k / 4), mn]`
//! MN-major, 4 exponent bytes per i32); the activation side uses a per-1x128
//! granularity and the FP4 weight side a group-32 granularity, so their packed
//! K extents differ (`k / 512` vs `k / 128`).
//!
//! See `csrc/k3/k3_deepgemm_fp8_fp4_grouped_sm100.cu`.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Checkpoint MXFP4 weight scales (`[groups, n, k / 32]` u8 UE8M0 exponent
    /// bytes, K-major) -> the runtime SFB tensor (`[groups, k / 128, n]` i32,
    /// MN-major, 4 exponent bytes per word LSB-first). Loader-time helper.
    pub fn k3_fp4_sf_prepare_cuda(
        sf: *const u8,
        packed: *mut i32,
        groups: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Masked grouped FP8 x FP4 GEMM over the rank's local experts.
    /// `operand_kind` is 1 for the fused W13 gate|up projection and 2 for W2.
    /// Requires sm_100f (returns `CUDA_ERROR_NOT_SUPPORTED` elsewhere).
    pub fn k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale: *const i32,
        b: *const u8,
        b_scale: *const i32,
        masked_m: *const i32,
        out: *mut Half,
        groups: i32,
        n: i32,
        k: i32,
        masked_cap: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;
}
