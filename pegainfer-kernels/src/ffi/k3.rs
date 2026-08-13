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

    /// Routing metadata for one rank's expert window: per-expert row counts
    /// (`masked_m[groups]`) and the expanded-entry -> masked-slot map
    /// (`slot_map[tokens * topk]`, `-1` for inactive entries). Entry order
    /// (`token * topk + slot`) fixes the row assignment deterministically.
    /// `topk_idx` carries GLOBAL expert ids; an entry is active when
    /// `topk_idx - local_expert_base` lands in `[0, groups)`, so a single-rank
    /// chain passes `local_expert_base = 0`.
    pub fn k3_moe_local_route_metadata_cuda(
        topk_idx: *const i32,
        masked_m: *mut i32,
        slot_map: *mut i32,
        tokens: i32,
        topk: i32,
        groups: i32,
        masked_cap: i32,
        local_expert_base: i32,
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
        local_expert_base: i32,
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
        local_expert_base: i32,
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

    /// This rank's fixed-shape contribution to the per-layer EP allgather:
    /// latent rows plus the two router arrays, with every row whose
    /// `row_active` entry is negative written as padding (zero latent, `-1`
    /// expert id, zero weight).
    pub fn k3_moe_ep_pack_dispatch_cuda(
        latent: *const Half,
        topk_idx: *const i32,
        topk_weight: *const f32,
        row_active: *const i32,
        latent_out: *mut Half,
        idx_out: *mut i32,
        weight_out: *mut f32,
        rows: i32,
        topk: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Masked W2 rows -> entry-major staging `[entries, hidden]` bf16. Dense
    /// full-cover pass: entries this rank does not own become exact zeros, so
    /// the following sum all-reduce adds each entry's row to zeros.
    pub fn k3_moe_entry_scatter_cuda(
        expert_out: *const Half,
        slot_map: *const i32,
        staging: *mut Half,
        entries: i32,
        hidden: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Weighted combine over the reduced entry-major staging buffer: same f32
    /// accumulation in topk-slot order and same single bf16 round as
    /// [`k3_moe_weighted_combine_cuda`], for the `tokens_out` global token rows
    /// starting at `token_base`.
    pub fn k3_moe_entry_combine_cuda(
        staging: *const Half,
        topk_idx: *const i32,
        topk_weight: *const f32,
        out: *mut Half,
        token_base: i32,
        tokens_out: i32,
        topk: i32,
        hidden: i32,
        experts: i32,
        stream: CUstream,
    ) -> CUresult;
}
