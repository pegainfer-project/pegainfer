// Included by shared/linear.cu only for Qwen3.5. The model owns the copied
// algorithm; descriptors and the decode workspace remain thread-local. Never
// publish this choice to the shared tuner or its persistent store.
#pragma once

static_assert(sizeof(cublasLtMatmulAlgo_t) == 8 * sizeof(uint64_t),
              "Qwen3.5 decode GEMM recipe storage must match cuBLASLt");

extern "C" int pegainfer_qwen35_decode_gemm_prepare(
    const __nv_bfloat16 *weights, int rows, int batch, int cols,
    uint64_t *algorithm, cudaStream_t stream) {
  const auto found = g_lt_plans.find({rows, batch, cols});
  if (!g_lt_handle || !g_lt_workspace || found == g_lt_plans.end()) {
    return cublas_status_to_error(CUBLAS_STATUS_NOT_INITIALIZED);
  }
  LtGemmPlan plan = found->second;
  uint32_t split_k = 0;
  size_t written = 0;
  auto status = cublasLtMatmulAlgoConfigGetAttribute(
      &plan.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &written);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  if (written != sizeof(split_k)) {
    return cublas_status_to_error(CUBLAS_STATUS_INTERNAL_ERROR);
  }
  if (split_k > 1) {
    // The shared descriptor already computes in FP32 with BF16 A/B/C/D.
    // COMPUTE_TYPE keeps split partials and their reduction in FP32; only D
    // is converted to BF16. Tile, split count and every other attribute stay put.
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    status = cublasLtMatmulAlgoConfigSetAttribute(
        &plan.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction));
    if (status != CUBLAS_STATUS_SUCCESS) {
      return cublas_status_to_error(status);
    }
  }
  if (!lt_algo_usable(plan, plan.algo)) {
    return cublas_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  // Execute before graph capture, including a recipe loaded from disk.
  if (!lt_algo_warm(plan, weights, rows, batch, cols, stream)) {
    return cublas_status_to_error(CUBLAS_STATUS_EXECUTION_FAILED);
  }
  std::memcpy(algorithm, &plan.algo, sizeof(plan.algo));
  return 0;
}

extern "C" int pegainfer_qwen35_decode_gemm_launch(
    const uint64_t *algorithm, const __nv_bfloat16 *weights,
    const __nv_bfloat16 *input, __nv_bfloat16 *output,
    int rows, int batch, int cols, cudaStream_t stream) {
  const auto found = g_lt_plans.find({rows, batch, cols});
  if (!g_lt_handle || !g_lt_workspace || found == g_lt_plans.end()) {
    return cublas_status_to_error(CUBLAS_STATUS_NOT_INITIALIZED);
  }
  const auto &plan = found->second;
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, algorithm, sizeof(algo));
  const float alpha = 1.0f, beta = 0.0f;
  const auto status = cublasLtMatmul(
      g_lt_handle, plan.op, &alpha, weights, plan.a, input, plan.b,
      &beta, output, plan.c, output, plan.c, &algo,
      g_lt_workspace, LT_WORKSPACE_SIZE, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status_to_error(status);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
