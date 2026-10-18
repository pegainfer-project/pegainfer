#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>

#include "../shared/ffi_guard.cuh"

namespace {

using DType = __nv_bfloat16;

constexpr int kKvLoraRank = 512;
constexpr int kNopeDim = 128;
constexpr int kVHeadDim = 128;
constexpr int kKvBHeadDim = kNopeDim + kVHeadDim;
constexpr int kMlaLtLocalHeads = 64;
constexpr int kMlaLtMaxBatch = 8;

struct MlaLtPlan {
  int batch_size = 0;
  bool v_up = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool ready = false;
};

thread_local cublasLtHandle_t g_mla_lt_handle = nullptr;
thread_local MlaLtPlan g_absorb_q_nope_plans[kMlaLtMaxBatch];
thread_local MlaLtPlan g_v_up_plans[kMlaLtMaxBatch];

int cublas_status_to_error(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaSuccess);
  }
  return PEGAINFER_CUBLAS_STATUS_BASE + static_cast<int>(status);
}

void destroy_mla_lt_plan(MlaLtPlan& plan) {
  if (plan.d != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.d);
  }
  if (plan.c != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.c);
  }
  if (plan.b != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.b);
  }
  if (plan.a != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.a);
  }
  if (plan.op != nullptr) {
    cublasLtMatmulDescDestroy(plan.op);
  }
  plan = MlaLtPlan{};
}

int set_batched_layout(cublasLtMatrixLayout_t layout,
                       int batch_count,
                       int64_t stride_elements) {
  cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_elements,
      sizeof(stride_elements));
  return cublas_status_to_error(status);
}

int preferred_mla_lt_candidate(bool v_up, int batch_size, int returned) {
  int preferred = 0;
  if (v_up) {
    preferred = batch_size == 1 ? 1 : 0;
  } else {
    preferred = batch_size == 1 ? 2 : 1;
  }
  return preferred < returned ? preferred : 0;
}

int build_mla_lt_plan(cublasLtHandle_t handle,
                      MlaLtPlan& plan,
                      int batch_size,
                      bool v_up) {
  destroy_mla_lt_plan(plan);
  plan.batch_size = batch_size;
  plan.v_up = v_up;

  const cublasOperation_t transa = v_up ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &transb, sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }

  if (v_up) {
    status = cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, kKvLoraRank, kVHeadDim,
                                        kKvLoraRank);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, kKvLoraRank, batch_size,
                                        kMlaLtLocalHeads * kKvLoraRank);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, kVHeadDim, batch_size,
                                        kMlaLtLocalHeads * kVHeadDim);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_16BF, kVHeadDim, batch_size,
                                        kMlaLtLocalHeads * kVHeadDim);
  } else {
    status = cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, kKvLoraRank, kNopeDim,
                                        kKvLoraRank);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, kNopeDim, batch_size,
                                        kMlaLtLocalHeads * kNopeDim);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, kKvLoraRank, batch_size,
                                        kMlaLtLocalHeads * kKvLoraRank);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_mla_lt_plan(plan);
      return cublas_status_to_error(status);
    }
    status = cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_16BF, kKvLoraRank, batch_size,
                                        kMlaLtLocalHeads * kKvLoraRank);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }

  const int64_t weight_stride = static_cast<int64_t>(kKvBHeadDim) * kKvLoraRank;
  int result = set_batched_layout(plan.a, kMlaLtLocalHeads, weight_stride);
  if (result != static_cast<int>(cudaSuccess)) {
    destroy_mla_lt_plan(plan);
    return result;
  }
  result = set_batched_layout(plan.b, kMlaLtLocalHeads, v_up ? kKvLoraRank : kNopeDim);
  if (result != static_cast<int>(cudaSuccess)) {
    destroy_mla_lt_plan(plan);
    return result;
  }
  result = set_batched_layout(plan.c, kMlaLtLocalHeads, v_up ? kVHeadDim : kKvLoraRank);
  if (result != static_cast<int>(cudaSuccess)) {
    destroy_mla_lt_plan(plan);
    return result;
  }
  result = set_batched_layout(plan.d, kMlaLtLocalHeads, v_up ? kVHeadDim : kKvLoraRank);
  if (result != static_cast<int>(cudaSuccess)) {
    destroy_mla_lt_plan(plan);
    return result;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }
  std::size_t workspace_bytes = 0;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }

  cublasLtMatmulHeuristicResult_t heuristics[8] = {};
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b, plan.c, plan.d,
                                          preference, 8, heuristics, &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(status);
  }
  if (returned <= 0) {
    destroy_mla_lt_plan(plan);
    return cublas_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  plan.algo = heuristics[preferred_mla_lt_candidate(v_up, batch_size, returned)].algo;
  plan.ready = true;
  return static_cast<int>(cudaSuccess);
}

MlaLtPlan* find_mla_lt_plan(MlaLtPlan* plans, int batch_size) {
  if (batch_size < 1 || batch_size > kMlaLtMaxBatch) {
    return nullptr;
  }
  MlaLtPlan& plan = plans[batch_size - 1];
  if (plan.ready && plan.batch_size == batch_size) {
    return &plan;
  }
  return nullptr;
}

int ensure_mla_lt_initialized() {
  if (g_mla_lt_handle == nullptr) {
    cublasStatus_t status = cublasLtCreate(&g_mla_lt_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return cublas_status_to_error(status);
    }
  }
  for (int batch_size = 1; batch_size <= kMlaLtMaxBatch; ++batch_size) {
    int status =
        build_mla_lt_plan(g_mla_lt_handle, g_absorb_q_nope_plans[batch_size - 1],
                          batch_size, /*v_up=*/false);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = build_mla_lt_plan(g_mla_lt_handle, g_v_up_plans[batch_size - 1],
                               batch_size, /*v_up=*/true);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

int run_absorb_q_nope(const DType* kv_b_proj,
                      const DType* q_nope,
                      DType* q_abs_nope,
                      int batch_size,
                      cudaStream_t stream) {
  if (g_mla_lt_handle == nullptr) {
    int status = ensure_mla_lt_initialized();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  MlaLtPlan* plan = find_mla_lt_plan(g_absorb_q_nope_plans, batch_size);
  if (plan == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(g_mla_lt_handle,
                                         plan->op,
                                         &alpha,
                                         kv_b_proj,
                                         plan->a,
                                         q_nope,
                                         plan->b,
                                         &beta,
                                         q_abs_nope,
                                         plan->c,
                                         q_abs_nope,
                                         plan->d,
                                         &plan->algo,
                                         nullptr,
                                         0,
                                         stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int run_v_up(const DType* kv_b_proj,
             const DType* latent,
             DType* output,
             int batch_size,
             cudaStream_t stream) {
  if (g_mla_lt_handle == nullptr) {
    int status = ensure_mla_lt_initialized();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  MlaLtPlan* plan = find_mla_lt_plan(g_v_up_plans, batch_size);
  if (plan == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const DType* w_uv = kv_b_proj + static_cast<int64_t>(kNopeDim) * kKvLoraRank;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(g_mla_lt_handle,
                                         plan->op,
                                         &alpha,
                                         w_uv,
                                         plan->a,
                                         latent,
                                         plan->b,
                                         &beta,
                                         output,
                                         plan->c,
                                         output,
                                         plan->d,
                                         &plan->algo,
                                         nullptr,
                                         0,
                                         stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

extern "C" {

int kimi_mla_cublaslt_init_cuda() {
  return ensure_mla_lt_initialized();
}

void kimi_mla_cublaslt_destroy_cuda() {
  for (int i = 0; i < kMlaLtMaxBatch; ++i) {
    destroy_mla_lt_plan(g_absorb_q_nope_plans[i]);
    destroy_mla_lt_plan(g_v_up_plans[i]);
  }
  if (g_mla_lt_handle != nullptr) {
    cublasLtDestroy(g_mla_lt_handle);
    g_mla_lt_handle = nullptr;
  }
}

int kimi_mla_absorb_q_nope_cublaslt_cuda(const DType* kv_b_proj,
                                         const DType* q_nope,
                                         DType* q_abs_nope,
                                         int batch_size,
                                         int local_heads,
                                         cudaStream_t stream) {
  if (local_heads != kMlaLtLocalHeads) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return run_absorb_q_nope(kv_b_proj, q_nope, q_abs_nope, batch_size, stream);
}

int kimi_mla_v_up_cublaslt_cuda(const DType* kv_b_proj,
                                const DType* latent,
                                DType* output,
                                int batch_size,
                                int local_heads,
                                cudaStream_t stream) {
  if (local_heads != kMlaLtLocalHeads) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return run_v_up(kv_b_proj, latent, output, batch_size, stream);
}

}  // extern "C"
