#pragma once
#include <cstdio>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

struct PgiDevice {
  int ordinal;
  hipStream_t stream = nullptr;
  rocblas_handle blas = nullptr;
};
extern thread_local char pgi_error[1024];
inline int pgi_hip(hipError_t status, const char *operation) {
  if (status == hipSuccess)
    return 0;
  std::snprintf(pgi_error, sizeof(pgi_error), "%s: %s", operation,
                hipGetErrorString(status));
  return static_cast<int>(status);
}
inline int pgi_blas(rocblas_status status, const char *operation) {
  if (status == rocblas_status_success)
    return 0;
  std::snprintf(pgi_error, sizeof(pgi_error), "%s: rocBLAS status %d",
                operation, static_cast<int>(status));
  return 10000 + static_cast<int>(status);
}
#define PGI_HIP(call)                                                          \
  do {                                                                         \
    int s = pgi_hip((call), #call);                                            \
    if (s)                                                                     \
      return s;                                                                \
  } while (0)
#define PGI_BLAS(call)                                                         \
  do {                                                                         \
    int s = pgi_blas((call), #call);                                           \
    if (s)                                                                     \
      return s;                                                                \
  } while (0)
