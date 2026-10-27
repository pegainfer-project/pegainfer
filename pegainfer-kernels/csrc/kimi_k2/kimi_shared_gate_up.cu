#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>

#include "../shared/ffi_guard.cuh"

static constexpr int KIMI_SHARED_GATE_UP_M = 4096;
static constexpr int KIMI_SHARED_GATE_UP_K = 7168;
static constexpr int KIMI_SHARED_GATE_UP_MAX_BATCH = 64;

static int cublas_status_to_error(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaSuccess);
  }
  return PEGAINFER_CUBLAS_STATUS_BASE + static_cast<int>(status);
}

struct KimiSharedGateUpLtPlan {
  int batch_size = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool ready = false;
};

thread_local cublasLtHandle_t g_kimi_shared_gate_up_lt = nullptr;
thread_local KimiSharedGateUpLtPlan g_kimi_shared_gate_up_plans[KIMI_SHARED_GATE_UP_MAX_BATCH];

static void destroy_plan(KimiSharedGateUpLtPlan &plan) {
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
  plan = KimiSharedGateUpLtPlan{};
}

static int build_plan(cublasLtHandle_t handle, KimiSharedGateUpLtPlan &plan, int batch_size) {
  destroy_plan(plan);
  plan.batch_size = batch_size;

  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &transb, sizeof(transb));
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }

  status = cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, KIMI_SHARED_GATE_UP_K,
                                      KIMI_SHARED_GATE_UP_M, KIMI_SHARED_GATE_UP_K);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, KIMI_SHARED_GATE_UP_K, batch_size,
                                      KIMI_SHARED_GATE_UP_K);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16BF, KIMI_SHARED_GATE_UP_M, batch_size,
                                      KIMI_SHARED_GATE_UP_M);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  status = cublasLtMatrixLayoutCreate(&plan.d, CUDA_R_16BF, KIMI_SHARED_GATE_UP_M, batch_size,
                                      KIMI_SHARED_GATE_UP_M);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  std::size_t workspace_bytes = 0;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b, plan.c, plan.d,
                                          preference, 1, &heuristic, &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(plan);
    return cublas_status_to_error(status);
  }
  if (returned != 1) {
    destroy_plan(plan);
    return cublas_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  plan.algo = heuristic.algo;
  plan.ready = true;
  return static_cast<int>(cudaSuccess);
}

static KimiSharedGateUpLtPlan *find_plan(int batch_size) {
  if (batch_size < 1 || batch_size > KIMI_SHARED_GATE_UP_MAX_BATCH) {
    return nullptr;
  }
  KimiSharedGateUpLtPlan &plan = g_kimi_shared_gate_up_plans[batch_size - 1];
  if (plan.ready && plan.batch_size == batch_size) {
    return &plan;
  }
  return nullptr;
}

extern "C" {

int kimi_shared_gate_up_cublaslt_init_cuda() {
  if (g_kimi_shared_gate_up_lt == nullptr) {
    cublasStatus_t status = cublasLtCreate(&g_kimi_shared_gate_up_lt);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return cublas_status_to_error(status);
    }
  }
  for (int batch_size = 1; batch_size <= KIMI_SHARED_GATE_UP_MAX_BATCH; ++batch_size) {
    int status = build_plan(g_kimi_shared_gate_up_lt,
                            g_kimi_shared_gate_up_plans[batch_size - 1], batch_size);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

void kimi_shared_gate_up_cublaslt_destroy_cuda() {
  for (int i = 0; i < KIMI_SHARED_GATE_UP_MAX_BATCH; ++i) {
    destroy_plan(g_kimi_shared_gate_up_plans[i]);
  }
  if (g_kimi_shared_gate_up_lt != nullptr) {
    cublasLtDestroy(g_kimi_shared_gate_up_lt);
    g_kimi_shared_gate_up_lt = nullptr;
  }
}

int kimi_shared_gate_up_cublaslt_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X,
                                      __nv_bfloat16 *Y, int M, int batch_size, int K,
                                      cudaStream_t stream) {
  if (W == nullptr || X == nullptr || Y == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (M != KIMI_SHARED_GATE_UP_M || K != KIMI_SHARED_GATE_UP_K) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (g_kimi_shared_gate_up_lt == nullptr) {
    return static_cast<int>(cudaErrorInvalidResourceHandle);
  }
  KimiSharedGateUpLtPlan *plan = find_plan(batch_size);
  if (plan == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status =
      cublasLtMatmul(g_kimi_shared_gate_up_lt, plan->op, &alpha, W, plan->a, X, plan->b,
                     &beta, Y, plan->c, Y, plan->d, &plan->algo, nullptr, 0, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

} // extern "C"
