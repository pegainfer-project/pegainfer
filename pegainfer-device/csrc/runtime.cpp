#include "runtime.h"
#include <cstdlib>
#include <cstring>
#include <new>

thread_local char pgi_error[1024];
extern "C" const char *pgi_last_error() { return pgi_error; }
extern "C" int pgi_create(int ordinal, void **result) {
  *result = nullptr;
  if (const char *expected = std::getenv("PEGAINFER_EXPECTED_PCI_BUS_ID")) {
    char bus[64]{};
    PGI_HIP(hipDeviceGetPCIBusId(bus, sizeof(bus), ordinal));
    if (std::strcmp(bus, expected) != 0) {
      std::snprintf(pgi_error, sizeof(pgi_error),
                    "device PCI mismatch: got %s, expected %s", bus, expected);
      return -1;
    }
  }
  PGI_HIP(hipSetDevice(ordinal));
  auto *d = new (std::nothrow) PgiDevice{ordinal};
  if (!d)
    return pgi_hip(hipErrorOutOfMemory, "create runtime");
  int status =
      pgi_hip(hipStreamCreateWithFlags(&d->stream, hipStreamNonBlocking),
              "create stream");
  if (!status)
    status = pgi_blas(rocblas_create_handle(&d->blas), "create BLAS handle");
  if (!status)
    status =
        pgi_blas(rocblas_set_stream(d->blas, d->stream), "set BLAS stream");
  if (status) {
    if (d->blas)
      rocblas_destroy_handle(d->blas);
    if (d->stream)
      hipStreamDestroy(d->stream);
    delete d;
    return status;
  }
  *result = d;
  return 0;
}
extern "C" int pgi_bind(void *ptr) {
  return pgi_hip(hipSetDevice(static_cast<PgiDevice *>(ptr)->ordinal),
                 "bind device");
}
extern "C" int pgi_sync(void *ptr) {
  auto *d = static_cast<PgiDevice *>(ptr);
  PGI_HIP(hipSetDevice(d->ordinal));
  return pgi_hip(hipStreamSynchronize(d->stream), "stream synchronize");
}
extern "C" void pgi_destroy(void *ptr) {
  auto *d = static_cast<PgiDevice *>(ptr);
  hipSetDevice(d->ordinal);
  hipStreamSynchronize(d->stream);
  rocblas_destroy_handle(d->blas);
  hipStreamDestroy(d->stream);
  delete d;
}
extern "C" int pgi_alloc(void *ptr, size_t bytes, void **out) {
  auto *d = static_cast<PgiDevice *>(ptr);
  PGI_HIP(hipSetDevice(d->ordinal));
  PGI_HIP(hipMalloc(out, bytes));
  int status =
      pgi_hip(hipMemsetAsync(*out, 0, bytes, d->stream), "zero allocation");
  if (status) {
    hipFree(*out);
    *out = nullptr;
  }
  return status;
}
extern "C" int pgi_free(void *ptr, void *allocation) {
  auto *d = static_cast<PgiDevice *>(ptr);
  PGI_HIP(hipSetDevice(d->ordinal));
  int sync =
      pgi_hip(hipStreamSynchronize(d->stream), "synchronize before free");
  int freed = pgi_hip(hipFree(allocation), "free");
  return sync ? sync : freed;
}
extern "C" int pgi_copy(void *ptr, void *dst, const void *src, size_t bytes,
                        int to_device) {
  auto *d = static_cast<PgiDevice *>(ptr);
  PGI_HIP(hipSetDevice(d->ordinal));
  PGI_HIP(hipMemcpyAsync(
      dst, src, bytes,
      to_device ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, d->stream));
  return pgi_hip(hipStreamSynchronize(d->stream), "complete host copy");
}
extern "C" int pgi_info(void *ptr, char *name, size_t name_len, char *arch,
                        size_t arch_len, size_t *free, size_t *total) {
  auto *d = static_cast<PgiDevice *>(ptr);
  PGI_HIP(hipSetDevice(d->ordinal));
  hipDeviceProp_t props{};
  PGI_HIP(hipGetDeviceProperties(&props, d->ordinal));
  std::snprintf(name, name_len, "%s", props.name);
  std::snprintf(arch, arch_len, "%s", props.gcnArchName);
  return pgi_hip(hipMemGetInfo(free, total), "memory info");
}
