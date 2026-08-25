// Whale doorbell ring: SM-path stores into peer inbox flags.
//
// Waits use stream memops (`cuStreamWaitValue64`) on this rank's own slab.
// Writes cannot: the GB300 memops engine rejects fabric-imported VAs, so the
// ring is a one-thread kernel taking the NVLink store path instead (probe
// and rationale: docs/models/k3/cp-lane-design.md).
//
// Ordering: the announced bytes are written by earlier work on THIS stream
// and are complete when this kernel launches; the 8-byte-aligned flag stores
// cannot tear. Publication and its doorbell must stay on one stream.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// One publish or consume beat rings at most every other gang member, so this
// caps the CP width at kMaxDoorbellTargets + 1.
constexpr int kMaxDoorbellTargets = 64;

struct DoorbellArgs {
  unsigned long long flag_addrs[kMaxDoorbellTargets];
  unsigned long long value;
  int flag_count;
};

__global__ void whale_doorbell_kernel(DoorbellArgs args) {
  __threadfence_system();
  for (int i = 0; i < args.flag_count; ++i) {
    *reinterpret_cast<volatile unsigned long long*>(args.flag_addrs[i]) = args.value;
  }
}

}  // namespace

extern "C" {

// Ring `value` into `flag_count` flag addresses (device VAs of 8-byte-aligned
// u64 slots, local or fabric-imported), stream-ordered after preceding work.
CUresult k3_whale_doorbell_ring(const unsigned long long* flag_addrs, int flag_count,
                                unsigned long long value, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (flag_addrs == nullptr || flag_count <= 0 || flag_count > kMaxDoorbellTargets) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  DoorbellArgs args{};
  for (int i = 0; i < flag_count; ++i) {
    // An unaligned or null store from the kernel would kill the CUDA context
    // instead of returning an error; refuse it host-side.
    if (flag_addrs[i] == 0 || flag_addrs[i] % 8 != 0) return CUDA_ERROR_INVALID_VALUE;
    args.flag_addrs[i] = flag_addrs[i];
  }
  args.value = value;
  args.flag_count = flag_count;
  whale_doorbell_kernel<<<1, 1, 0, stream>>>(args);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return CUDA_SUCCESS;
  return err == cudaErrorInvalidValue ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_LAUNCH_FAILED;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
