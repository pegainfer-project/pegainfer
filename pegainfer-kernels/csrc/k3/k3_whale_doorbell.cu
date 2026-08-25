// Whale doorbell ring: SM-path stores into peer inbox flags.
//
// The whale data plane orders exchange windows with 64-bit doorbell flags in
// each rank's fabric slab. Waits run on the stream memops engine
// (`cuStreamWaitValue64`) against this rank's OWN slab — a local allocation,
// which memops accept. Writes cannot: the targets are peer slabs reached
// through `cuMemImportFromShareableHandle` fabric mappings, and the memops
// engine rejects imported fabric VAs outright (`CUDA_ERROR_INVALID_VALUE`,
// verified on GB300 with a two-process probe; a plain DtoD copy or an SM
// store to the same mapping succeeds — the MegaMoE kernel writes through
// these mappings on every step). So the ring is a one-thread kernel: SM
// stores take the NVLink path that fabric mappings were built for.
//
// Ordering: the publication bytes this doorbell announces were written by
// operations enqueued earlier on the same stream, complete before this
// kernel launches. The system fence before the stores is the release for
// anything this SM could still hold; the stores themselves are 8-byte
// aligned and land atomically.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// One publish or consume beat rings at most every other gang member.
constexpr int kMaxDoorbellTargets = 64;

struct DoorbellArgs {
  unsigned long long addrs[kMaxDoorbellTargets];
  unsigned long long value;
  int count;
};

__global__ void whale_doorbell_kernel(DoorbellArgs args) {
  __threadfence_system();
  for (int i = 0; i < args.count; ++i) {
    *reinterpret_cast<volatile unsigned long long*>(args.addrs[i]) = args.value;
  }
}

inline CUresult doorbell_map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

extern "C" {

// Ring `value` into `count` flag addresses (device VAs of 8-byte-aligned
// u64 slots, local or fabric-imported), stream-ordered after preceding work.
CUresult k3_whale_doorbell_ring(const unsigned long long* addrs, int count,
                                unsigned long long value, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (addrs == nullptr || count <= 0 || count > kMaxDoorbellTargets) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  DoorbellArgs args;
  for (int i = 0; i < count; ++i) {
    if (addrs[i] == 0 || addrs[i] % 8 != 0) return CUDA_ERROR_INVALID_VALUE;
    args.addrs[i] = addrs[i];
  }
  args.value = value;
  args.count = count;
  whale_doorbell_kernel<<<1, 1, 0, stream>>>(args);
  return doorbell_map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
