#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

namespace {

constexpr int kPage = 64;
constexpr int kLatent = 576;

__global__ void unpack_pages(const unsigned char* packed, const int* block_ids,
                             int blocks, int packed_bytes,
                             long long packed_block_stride, long long max_slots,
                             __nv_bfloat16* unpacked) {
  const int item = blockIdx.x;
  const int block = block_ids[item / kPage];
  const int token = item % kPage;
  const long long slot = static_cast<long long>(block) * kPage + token;
  if (block < 0 || slot >= max_slots) __trap();
  // Page-first slab: pages sit packed_block_stride apart (tight layout passes
  // kPage * packed_bytes). The unpacked buffer stays dense.
  const unsigned char* src = packed +
                             static_cast<long long>(block) * packed_block_stride +
                             static_cast<long long>(token) * packed_bytes;
  __nv_bfloat16* dst = unpacked + slot * kLatent;
  for (int dim = threadIdx.x; dim < kLatent; dim += blockDim.x) {
    if (packed_bytes == 576) {
      dst[dim] = __float2bfloat16(
          __half2float(__nv_cvt_fp8_to_halfraw(src[dim], __NV_E4M3)));
    } else if (dim < 512) {
      const float value =
          __half2float(__nv_cvt_fp8_to_halfraw(src[dim], __NV_E4M3));
      const float scale = reinterpret_cast<const float*>(src + 512)[dim / 128];
      dst[dim] = __float2bfloat16(value * scale);
    } else {
      dst[dim] =
          reinterpret_cast<const __nv_bfloat16*>(src + 528)[dim - 512];
    }
  }
}

}  // namespace

extern "C" CUresult glm52_prefill_unpack_pages_cuda(
    const unsigned char* packed, const int* block_ids, int blocks,
    int packed_bytes, long long packed_block_stride, long long max_slots,
    __nv_bfloat16* unpacked, CUstream stream) {
  if (!packed || !block_ids || !unpacked || blocks <= 0 ||
      (packed_bytes != 576 && packed_bytes != 656) || max_slots <= 0 ||
      packed_block_stride < static_cast<long long>(kPage) * packed_bytes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  PEGAINFER_FFI_GUARD_BEGIN
  unpack_pages<<<blocks * kPage, 256, 0,
                 reinterpret_cast<cudaStream_t>(stream)>>>(
      packed, block_ids, blocks, packed_bytes, packed_block_stride, max_slots,
      unpacked);
  return map_cuda_error(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
