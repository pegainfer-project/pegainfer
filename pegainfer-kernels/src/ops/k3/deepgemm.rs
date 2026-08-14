//! Kimi-K3 routed-expert GEMM: DeepGEMM SM100 `MGroupedMasked` FP8 x FP4
//! (tcgen05), AOT-instantiated (no JIT, no torch). The activation side is the
//! FP8 e4m3 / per-1x128 UE8M0 recipe GLM5.2 uses; the weight side is MXFP4
//! (e2m1, K-major, 2 values per byte) with group-32 UE8M0 scale factors.
//! See `csrc/k3/k3_deepgemm_fp8_fp4_grouped_sm100.cu` for the full layout
//! contract.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Alignment required by the SM100 masked layout and packed scale factors.
pub const K3_DEEPGEMM_SM100_MASKED_ALIGNMENT: usize = 128;
/// MXFP4 scale-factor group size along K.
pub const K3_FP4_SF_GROUP_K: usize = 32;
/// K elements covered by one packed i32 scale word on the FP4 weight side.
const K3_FP4_SF_WORD_K: usize = K3_FP4_SF_GROUP_K * 4;
/// K elements covered by one packed i32 scale word on the FP8 activation side.
const K3_FP8_SF_WORD_K: usize = 128 * 4;

/// Which per-expert projection a masked grouped GEMM call targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum K3DeepGemmFp8Fp4Kind {
    /// Fused gate|up projection: `n = 6144`, `k = 3584`.
    W13,
    /// Down projection: `n = 3584`, `k = 3072`.
    W2,
}

impl K3DeepGemmFp8Fp4Kind {
    /// `(n, k)` for one expert.
    #[must_use]
    pub const fn shape(self) -> (usize, usize) {
        match self {
            Self::W13 => (6144, 3584),
            Self::W2 => (3584, 3072),
        }
    }

    const fn abi_kind(self) -> i32 {
        match self {
            Self::W13 => 1,
            Self::W2 => 2,
        }
    }
}

/// Local-expert counts the GEMM is instantiated for: 56 (EP4 dev / EP16 full),
/// 112 (EP8 full), 224 (single-GPU bring-up).
pub const K3_DEEPGEMM_SM100_GROUPS: [usize; 3] = [56, 112, 224];

/// Checkpoint MXFP4 weight scales -> the runtime SFB tensor.
///
/// Input `sf` is `[groups, n, k / 32]` u8 UE8M0 exponent bytes in K-major
/// order (matching the K-major packed FP4 weight bank). Output `packed` is
/// `[groups, k / 128, n]` i32, MN-major, with four consecutive K-group
/// exponents per word LSB-first. Loader-time helper, not a step-time kernel.
pub fn k3_fp4_sf_prepare_launch(
    ctx: &DeviceContext,
    groups: usize,
    n: usize,
    k: usize,
    sf: &CudaSlice<u8>,
    packed: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && n > 0 && k > 0 && n.is_multiple_of(4) && k.is_multiple_of(K3_FP4_SF_WORD_K),
        "K3 FP4 SF prepare needs groups/n/k>0, n%4=0 and k%128=0, got groups={groups}, n={n}, k={k}"
    );
    ensure!(
        sf.len() >= groups * n * (k / K3_FP4_SF_GROUP_K)
            && packed.len() >= groups * (k / K3_FP4_SF_WORD_K) * n,
        "K3 FP4 SF prepare buffers too small for {groups} groups (n={n}, k={k}): sf {}, packed {}",
        sf.len(),
        packed.len()
    );
    let (sf_ptr, _sf_guard) = sf.device_ptr(&ctx.stream);
    let (packed_ptr, _packed_guard) = packed.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_fp4_sf_prepare_cuda(
            sf_ptr as *const u8,
            packed_ptr as *mut i32,
            groups as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 FP4 SF prepare launch failed: {err}"))
}

/// Masked grouped FP8 x FP4 GEMM over the rank's local experts:
/// `out[g, :masked_m[g], n] = deq(weight[g]) @ deq(activation[g])`.
///
/// Activation `[groups, masked_cap, k]` fp8 e4m3, activation scale packed
/// UE8M0 i32 `[groups, k/512, masked_cap]` (MN-major, 4 exponent bytes per
/// i32), weight `[groups, n, k]` fp4 e2m1 packed 2-per-byte, weight scale
/// packed UE8M0 i32 `[groups, k/128, n]`, out `[groups, masked_cap, n]` bf16.
/// `groups` dispatches over `K3_DEEPGEMM_SM100_GROUPS`. Requires sm_100f
/// (`NOT_SUPPORTED` elsewhere).
#[allow(clippy::too_many_arguments)]
pub fn k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch(
    ctx: &DeviceContext,
    kind: K3DeepGemmFp8Fp4Kind,
    groups: usize,
    masked_cap: usize,
    num_sms: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<i32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<i32>,
    masked_m: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    let (n, k) = kind.shape();
    ensure!(
        K3_DEEPGEMM_SM100_GROUPS.contains(&groups),
        "K3 SM100 masked grouped FP8xFP4 needs groups in {K3_DEEPGEMM_SM100_GROUPS:?}, got {groups}"
    );
    ensure!(
        matches!(num_sms, 148 | 152),
        "K3 SM100 masked grouped FP8xFP4 supports B200/GB300 SM counts {{148,152}}, got {num_sms}"
    );
    ensure!(
        masked_cap > 0 && masked_cap.is_multiple_of(K3_DEEPGEMM_SM100_MASKED_ALIGNMENT),
        "K3 SM100 masked grouped FP8xFP4 needs masked_cap divisible by 128, got {masked_cap}"
    );
    ensure!(
        activation.len() >= groups * masked_cap * k
            && activation_scale.len() >= groups * (k / K3_FP8_SF_WORD_K) * masked_cap
            // FP4 packs two values per byte.
            && weight.len() >= groups * n * k / 2
            && weight_scale.len() >= groups * (k / K3_FP4_SF_WORD_K) * n
            && masked_m.len() >= groups
            && output.len() >= groups * masked_cap * n,
        "K3 SM100 masked grouped FP8xFP4 {kind:?} buffers too small: act {}, act_scale {}, w {}, w_scale {}, masked_m {}, out {}",
        activation.len(),
        activation_scale.len(),
        weight.len(),
        weight_scale.len(),
        masked_m.len(),
        output.len()
    );
    let (act_ptr, _act_guard) = activation.device_ptr(&ctx.stream);
    let (act_scale_ptr, _act_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = weight.device_ptr(&ctx.stream);
    let (w_scale_ptr, _w_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda(
            kind.abi_kind(),
            act_ptr as *const u8,
            act_scale_ptr as *const i32,
            w_ptr as *const u8,
            w_scale_ptr as *const i32,
            masked_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            groups as i32,
            n as i32,
            k as i32,
            masked_cap as i32,
            num_sms as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 SM100 masked grouped FP8xFP4 {kind:?} launch failed: {err}"))
}
