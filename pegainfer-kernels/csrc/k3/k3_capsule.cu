// K3 capsule loader: binds vendored external decode kernels (cubin/k3/,
// embedded at build time via k3_capsule_cubins.inc) to extern "C" launchers
// the Rust side calls like any other kernel. The framework owns the op
// contract and the reference twin; these cubins are replaceable artifacts.
//
// Loading is per-thread (K3 executor threads each own one device/context,
// same discipline as the thread_local cuBLAS handle in shared/linear.cu).
// Each capsule is loaded lazily on first launch and sanity-checked
// fail-closed: cuFuncGetParamInfo must report the exact staged parameter
// layout recorded at capture time, or the launcher refuses to run.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "k3_capsule_cubins.inc"

namespace {

struct Capsule {
  CUmodule module = nullptr;
  CUfunction func = nullptr;
  CUresult status = CUDA_ERROR_NOT_INITIALIZED;
  bool tried = false;
};

// Expected staged-parameter layout, from cuFuncGetParamInfo against the
// vendored cubin (recorded in the capture's port_abi manifest).
struct ParamSpec {
  int count;
  size_t total_bytes;  // offset+size of the last parameter
};

CUresult capsule_load(Capsule& cap, const unsigned char* image,
                      const char* entry, const ParamSpec& spec) {
  if (cap.tried) {
    return cap.status;
  }
  cap.tried = true;
  cap.status = cuModuleLoadData(&cap.module, image);
  if (cap.status != CUDA_SUCCESS) {
    fprintf(stderr, "[k3-capsule] cuModuleLoadData failed for %s: %d\n", entry,
            (int)cap.status);
    return cap.status;
  }
  cap.status = cuModuleGetFunction(&cap.func, cap.module, entry);
  if (cap.status != CUDA_SUCCESS) {
    fprintf(stderr, "[k3-capsule] entry %s not found: %d\n", entry,
            (int)cap.status);
    return cap.status;
  }
  // Static ABI check: parameter count and packed size must match the layout
  // this launcher stages. A drifted cubin fails here, not at launch.
  size_t offset = 0, size = 0;
  int count = 0;
  while (count < 64 &&
         cuFuncGetParamInfo(cap.func, (size_t)count, &offset, &size) ==
             CUDA_SUCCESS) {
    count++;
  }
  if (count != spec.count || offset + size != spec.total_bytes) {
    fprintf(stderr,
            "[k3-capsule] ABI mismatch for %s: %d params/%zu bytes, expected "
            "%d/%zu\n",
            entry, count, offset + size, spec.count, spec.total_bytes);
    cap.status = CUDA_ERROR_INVALID_IMAGE;
    return cap.status;
  }
  cap.status = CUDA_SUCCESS;
  return cap.status;
}

thread_local Capsule g_topk;
thread_local Capsule g_kda;

constexpr char kTopkEntry[] =
    "_ZN4vllm3moe17single_group_topk6detail29single_group_topk_warp_kernelIffi"
    "LNS0_11ScoringFuncE1ELi512ELi22EEEvPKT_PfPT1_PKT0_lllbfb";
constexpr char kKdaEntry[] =
    "_ZN44_GLOBAL__N__7c05f95b_9_kda_tu_cu_2dd95ecb_4135kda_decode_fusion_"
    "many_heads_kernelILb1ELb1ELi96ELi96ELb1ELb0ELb0ELb0ELb0ELb0ELb1ELb1ELb1E"
    "Lb1ELb1ELb1EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S5_S5_S5_PS1_S6_S6_S5_S3_"
    "S5_S3_S3_S5_PKiS8_PfS6_iiifffNS_16KdaDecodeStridesE";

// Mirror of the kernel's aggregate parameter (fused_kda_decode_kernel.cu:26).
struct KdaDecodeStrides {
  int64_t x_row;
  int64_t beta_row;
  int64_t onorm_row;
  int64_t conv_slot;
  int64_t state_slot;
};

}  // namespace

// vLLM v0.28.0 single_group_topk warp kernel <float, float, int, SIGMOID,
// MaxExperts=512, MaxTopK=22>: sigmoid scoring with bias-corrected selection,
// weights from unbiased scores, renormalized and scaled on device. One warp
// per token, 8 warps per block.
extern "C" CUresult k3_capsule_router_topk_cuda(
    const float* scores, const float* bias, int* topk_idx, float* topk_wts,
    int b, int num_experts, int topk, float routed_scaling,
    cudaStream_t stream) {
  // 10 staged params, packed tail (..., bool@56, float@60, bool@64) = 65
  // bytes, from the captured cuFuncGetParamInfo walk.
  CUresult rc = capsule_load(g_topk, kK3CapsuleTopk, kTopkEntry,
                             ParamSpec{10, 65});
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  int64_t num_tokens = b;
  int64_t experts64 = num_experts;
  int64_t topk64 = topk;
  bool renormalize = true;
  bool enable_pdl = false;
  void* params[] = {
      (void*)&scores,  (void*)&topk_wts,   (void*)&topk_idx,
      (void*)&bias,    (void*)&num_tokens, (void*)&experts64,
      (void*)&topk64,  (void*)&renormalize, (void*)&routed_scaling,
      (void*)&enable_pdl,
  };
  constexpr unsigned kBlock = 256;  // WarpTopKLaunchConfig<512>::BlockDim
  constexpr unsigned kWarpsPerBlock = kBlock / 32;
  unsigned grid = (unsigned)((b + kWarpsPerBlock - 1) / kWarpsPerBlock);
  return cuLaunchKernel(g_topk.func, grid, 1, 1, kBlock, 1, 1, 0,
                        (CUstream)stream, params, nullptr);
}

// vLLM v0.28.0 fused KDA decode, 96-head static-layout head-grid variant with
// conv-state update, lower bound and beta sigmoid. One block per (token,
// value head); consumes projected x_q|x_k|x_v rows, updates the bf16 conv
// windows and the fp32 recurrent state in place, applies the gated output
// norm, writes bf16 out rows.
extern "C" CUresult k3_capsule_kda_decode_cuda(
    const void* x_q, const void* x_k, const void* x_v, const float* w_q_t,
    const float* w_k_t, const float* w_v_t, const float* bias_q,
    const float* bias_k, const float* bias_v, void* cs_q, void* cs_k,
    void* cs_v, const float* a_log, const void* g, const float* dt_bias,
    const void* beta, const void* onorm_g, const float* onorm_weight,
    const int* ssm_state_indices, const int* cu_seqlens, float* state,
    void* out, int b, int heads, int value_heads, float lower_bound,
    float scale, float onorm_eps, int64_t x_row, int64_t beta_row,
    int64_t onorm_row, int64_t conv_slot, int64_t state_slot,
    cudaStream_t stream) {
  // 29 staged params: 22 pointers, 3 ints, 3 floats, one 40-byte struct at
  // offset 200 -> 240 bytes, from the captured cuFuncGetParamInfo walk.
  CUresult rc = capsule_load(g_kda, kK3CapsuleKdaDecode, kKdaEntry,
                             ParamSpec{29, 240});
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  KdaDecodeStrides strides{x_row, beta_row, onorm_row, conv_slot, state_slot};
  void* params[] = {
      (void*)&x_q,     (void*)&x_k,        (void*)&x_v,
      (void*)&w_q_t,   (void*)&w_k_t,      (void*)&w_v_t,
      (void*)&bias_q,  (void*)&bias_k,     (void*)&bias_v,
      (void*)&cs_q,    (void*)&cs_k,       (void*)&cs_v,
      (void*)&a_log,   (void*)&g,          (void*)&dt_bias,
      (void*)&beta,    (void*)&onorm_g,    (void*)&onorm_weight,
      (void*)&ssm_state_indices, (void*)&cu_seqlens, (void*)&state,
      (void*)&out,     (void*)&b,          (void*)&heads,
      (void*)&value_heads, (void*)&lower_bound, (void*)&scale,
      (void*)&onorm_eps, (void*)&strides,
  };
  return cuLaunchKernel(g_kda.func, (unsigned)b, (unsigned)value_heads, 1, 256,
                        1, 1, 0, (CUstream)stream, params, nullptr);
}
