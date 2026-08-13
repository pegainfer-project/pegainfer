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

    /// Local routing metadata for one rank's experts: per-expert row counts
    /// (`masked_m[groups]`) and the expanded-entry -> masked-slot map
    /// (`slot_map[tokens * topk]`, `-1` for inactive entries). Entry order
    /// (`token * topk + slot`) fixes the row assignment deterministically.
    pub fn k3_moe_local_route_metadata_cuda(
        topk_idx: *const i32,
        masked_m: *mut i32,
        slot_map: *mut i32,
        tokens: i32,
        topk: i32,
        groups: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Local gather fused with the W13 A-operand quant: token-major bf16
    /// latents `[tokens, hidden]` -> fp8 e4m3 `[groups * masked_cap, hidden]`
    /// plus MN-major UE8M0 f32 group scales `[groups, hidden / 128,
    /// masked_cap]`.
    pub fn k3_moe_gather_fp8_quant_masked_cuda(
        latent: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        output: *mut u8,
        scales: *mut f32,
        tokens: i32,
        topk: i32,
        hidden: i32,
        groups: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// K3 situ activation over the masked gate|up rows
    /// (`4 * tanh(g / 4) * sigmoid(g) * 25 * tanh(u / 25)`, f32 over the bf16
    /// W13 output) followed by the W2 A-operand quant.
    pub fn k3_situ_and_mul_fp8_quant_masked_cuda(
        gate_up: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        output: *mut u8,
        scales: *mut f32,
        tokens: i32,
        topk: i32,
        inter: i32,
        groups: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// f32 MN-major group scales `[groups, scale_cols, cap]` -> the packed
    /// UE8M0 i32 SFA tensor `[groups, scale_cols / 4, cap]`.
    pub fn k3_fp8_scale_pack_ue8m0_cuda(
        scales: *const f32,
        packed: *mut i32,
        groups: i32,
        scale_cols: i32,
        cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Weighted combine: masked W2 rows -> token-major bf16 hidden states,
    /// f32 accumulation in topk-slot order (no atomics), one bf16 round.
    pub fn k3_moe_weighted_combine_cuda(
        expert_out: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        topk_weight: *const f32,
        out: *mut Half,
        tokens: i32,
        topk: i32,
        hidden: i32,
        groups: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;
}
