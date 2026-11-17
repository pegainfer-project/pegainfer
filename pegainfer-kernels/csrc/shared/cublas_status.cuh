#pragma once

#include <cublas_api.h>
#include <cuda_runtime_api.h>

#include "ffi_guard.cuh"

inline int cublas_status_to_error(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return static_cast<int>(cudaSuccess);
  }
  return PEGAINFER_CUBLAS_STATUS_BASE + static_cast<int>(status);
}
