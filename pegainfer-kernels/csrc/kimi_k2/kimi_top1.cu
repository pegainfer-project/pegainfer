#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int TOP1_PACK_THREADS = 128;

struct Top1Packet {
  int32_t id;
  __nv_bfloat16 value;
};

static_assert(sizeof(Top1Packet) == 8);
static_assert(offsetof(Top1Packet, id) == 0);
static_assert(offsetof(Top1Packet, value) == sizeof(int32_t));

__global__ void pack_top1_packets_kernel(const int32_t* ids,
                                        const __nv_bfloat16* values,
                                        Top1Packet* packets, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    packets[row].id = ids[row];
    packets[row].value = values[row];
  }
}

}

extern "C" cudaError_t kimi_pack_top1_packets_cuda(
    const int32_t* ids, const __nv_bfloat16* values, void* packets, int rows,
    cudaStream_t stream) {
  const int blocks = (rows + TOP1_PACK_THREADS - 1) / TOP1_PACK_THREADS;
  pack_top1_packets_kernel<<<blocks, TOP1_PACK_THREADS, 0, stream>>>(
      ids, values, static_cast<Top1Packet*>(packets), rows);
  return cudaGetLastError();
}
