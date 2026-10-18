// Device-side decode-step input feed: advance the per-row step inputs in
// place so the next whole-step graph replay can be enqueued without a host
// round-trip. Runs as an eager launch between two replays on the decode
// stream; on a non-leased step the host prologue rewrites all four buffers,
// so an unused advance is harmless by construction.
#include <cstdint>
#include <climits>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 6144;
constexpr int kMaxRanks = 8;
constexpr int kCandidateFields = 4;

// Each rank owns four unique bf16 fields in the hidden-width carrier:
// [top value, global token byte 0, byte 1, byte 2]. Bytes are exactly
// representable in bf16, so the fixed-order all-reduce transports them
// losslessly while preserving negative logit values at their unique slot.
__global__ void glm52_vocab_parallel_pack_kernel(
    const __nv_bfloat16* __restrict__ local_values,
    const int32_t* __restrict__ local_indices,
    __nv_bfloat16* __restrict__ partial, int rank, int vocab_start) {
  const int row = static_cast<int>(blockIdx.x);
  uint4* row_words = reinterpret_cast<uint4*>(partial + (size_t)row * kHidden);
  constexpr int kWords = kHidden * sizeof(__nv_bfloat16) / sizeof(uint4);
  for (int i = static_cast<int>(threadIdx.x); i < kWords; i += blockDim.x) {
    row_words[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t token = static_cast<uint32_t>(vocab_start + local_indices[row]);
    __nv_bfloat16* fields = partial + (size_t)row * kHidden + rank * kCandidateFields;
    fields[0] = local_values[row];
    fields[1] = __float2bfloat16_rn(static_cast<float>(token & 0xffu));
    fields[2] = __float2bfloat16_rn(static_cast<float>((token >> 8) & 0xffu));
    fields[3] = __float2bfloat16_rn(static_cast<float>((token >> 16) & 0xffu));
  }
}

__global__ void glm52_vocab_parallel_unpack_kernel(
    const __nv_bfloat16* __restrict__ gathered,
    __nv_bfloat16* __restrict__ values, int32_t* __restrict__ indices,
    int rows, int ranks) {
  const int row = static_cast<int>(threadIdx.x);
  if (row >= rows) return;

  float best = -__int_as_float(0x7f800000);
  uint32_t best_token = static_cast<uint32_t>(INT_MAX);
  for (int rank = 0; rank < ranks; ++rank) {
    const __nv_bfloat16* fields =
        gathered + (size_t)row * kHidden + rank * kCandidateFields;
    const float value = __bfloat162float(fields[0]);
    const uint32_t token =
        static_cast<uint32_t>(__bfloat162float(fields[1])) |
        (static_cast<uint32_t>(__bfloat162float(fields[2])) << 8) |
        (static_cast<uint32_t>(__bfloat162float(fields[3])) << 16);
    if (value == value && (value > best || (value == best && token < best_token))) {
      best = value;
      best_token = token;
    }
  }
  // All-NaN row: no candidate was accepted. Degrade to token 0 like the
  // non-sharded argmax path — INT_MAX would be read back as an embedding row
  // index by the speculative decode feed. (Packed tokens are < 2^24, so
  // INT_MAX can never be a legitimate winner.)
  if (best_token == static_cast<uint32_t>(INT_MAX)) best_token = 0;
  values[row] = __float2bfloat16_rn(best);
  indices[row] = static_cast<int32_t>(best_token);
}

}  // namespace

extern "C" cudaError_t glm52_vocab_parallel_pack_cuda(
    const __nv_bfloat16* local_values, const int32_t* local_indices,
    __nv_bfloat16* partial, int rows, int rank, int vocab_start,
    cudaStream_t stream) {
  if (rows <= 0 || rows > 32 || rank < 0 || rank >= kMaxRanks ||
      vocab_start < 0) {
    return cudaErrorInvalidValue;
  }
  glm52_vocab_parallel_pack_kernel<<<rows, 256, 0, stream>>>(
      local_values, local_indices, partial, rank, vocab_start);
  return cudaGetLastError();
}

extern "C" cudaError_t glm52_vocab_parallel_unpack_cuda(
    const __nv_bfloat16* gathered, __nv_bfloat16* values, int32_t* indices,
    int rows, int ranks, cudaStream_t stream) {
  if (rows <= 0 || rows > 32 || ranks <= 0 || ranks > kMaxRanks) {
    return cudaErrorInvalidValue;
  }
  glm52_vocab_parallel_unpack_kernel<<<1, 32, 0, stream>>>(
      gathered, values, indices, rows, ranks);
  return cudaGetLastError();
}

__global__ void glm52_decode_feed_kernel(const int32_t* __restrict__ argmax_indices,
                                         uint32_t* __restrict__ token_ids,
                                         uint32_t* __restrict__ positions,
                                         int64_t* __restrict__ slot_mapping,
                                         int32_t* __restrict__ seq_lens,
                                         int rows) {
    const int row = static_cast<int>(threadIdx.x);
    if (row >= rows) {
        return;
    }
    token_ids[row] = static_cast<uint32_t>(argmax_indices[row]);
    positions[row] += 1u;
    slot_mapping[row] += 1;
    seq_lens[row] += 1;
}

extern "C" cudaError_t glm52_decode_feed_launch_cuda(const int32_t* argmax_indices,
                                                     uint32_t* token_ids,
                                                     uint32_t* positions,
                                                     int64_t* slot_mapping,
                                                     int32_t* seq_lens,
                                                     int rows,
                                                     cudaStream_t stream) {
    if (rows <= 0 || rows > 32) {
        return cudaErrorInvalidValue;
    }
    glm52_decode_feed_kernel<<<1, 32, 0, stream>>>(argmax_indices, token_ids, positions,
                                                   slot_mapping, seq_lens, rows);
    return cudaGetLastError();
}
