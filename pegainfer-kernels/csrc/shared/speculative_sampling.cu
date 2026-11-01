#include "ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <stdint.h>

#include <flashinfer/sampling.cuh>

namespace {

constexpr int THREADS = 1024;
constexpr int CANDIDATES = 16;
using Reduce = cub::BlockReduce<float, THREADS>;
using SamplingStorage = flashinfer::sampling::SamplingTempStorage<
    THREADS, cub::BLOCK_SCAN_WARP_SCANS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;

__global__ void normalize_min_p(float* probs, const float* min_p, int vocab) {
  __shared__ Reduce::TempStorage storage;
  __shared__ float threshold;
  __shared__ float mass;
  float* row = probs + static_cast<size_t>(blockIdx.x) * vocab;
  float maximum = 0.0f;
  for (int token = threadIdx.x; token < vocab; token += THREADS) {
    maximum = fmaxf(maximum, row[token]);
  }
  maximum = Reduce(storage).Reduce(maximum, cuda::maximum<>{});
  if (threadIdx.x == 0) threshold = maximum * min_p[blockIdx.x];
  __syncthreads();

  float sum = 0.0f;
  for (int token = threadIdx.x; token < vocab; token += THREADS) {
    float value = row[token] >= threshold ? row[token] : 0.0f;
    row[token] = value;
    sum += value;
  }
  sum = Reduce(storage).Sum(sum);
  if (threadIdx.x == 0) mass = sum;
  __syncthreads();
  for (int token = threadIdx.x; token < vocab; token += THREADS) row[token] /= mass;
}

// One CTA owns a request's first rejection and final sample. Only the 16
// proposal entries change in the residual row; q never spans the vocabulary.
template <int VEC_SIZE>
__global__ void sparse_verify(float* probs, const int* requests, const uint32_t* candidates,
                              const float* q, const uint32_t* drafts, const float* uniforms,
                              uint32_t* output, int block_size, int vocab) {
  __shared__ SamplingStorage storage;
  __shared__ int accepted;
  __shared__ float mass;
  int request = blockIdx.x;
  int row_start = requests[request * 3];
  int draft_row = requests[request * 3 + 1];
  int length = requests[request * 3 + 2];
  int draft_stride = block_size - 1;
  if (threadIdx.x == 0) {
    accepted = 0;
    for (int position = 0; position < length; ++position) {
      uint32_t token = drafts[draft_row * draft_stride + position];
      const uint32_t* ids = candidates + (draft_row * block_size + position + 1) * CANDIDATES;
      const float* proposal = q + (draft_row * draft_stride + position) * CANDIDATES;
      float proposed_prob = 0.0f;
      for (int k = 0; k < CANDIDATES; ++k) {
        if (ids[k] == token) proposed_prob = proposal[k];
      }
      float* target = probs + static_cast<size_t>(row_start + position) * vocab;
      float threshold = flashinfer::sampling::ieee_mul(
          uniforms[request * block_size + position], proposed_prob);
      if (!(threshold < target[token])) {
        for (int k = 0; k < CANDIDATES; ++k) {
          target[ids[k]] = fmaxf(target[ids[k]] - proposal[k], 0.0f);
        }
        break;
      }
      ++accepted;
    }
    output[request * 2] = accepted;
  }
  __syncthreads();

  float* row = probs + static_cast<size_t>(row_start + accepted) * vocab;
  float sum = 0.0f;
  for (int token = threadIdx.x; token < vocab; token += THREADS) sum += row[token];
  sum = Reduce(storage.block_prim.reduce).Sum(sum);
  if (threadIdx.x == 0) {
    mass = sum;
    storage.sampled_id = vocab;
    storage.last_valid_id = -1;
  }
  __syncthreads();
  if (!(mass > 0.0f) || !isfinite(mass)) {
    if (threadIdx.x == 0) output[request * 2 + 1] = UINT32_MAX;
    return;
  }

  // Scaling u by the actual residual mass also handles tiny positive
  // residuals, without an epsilon floor or another full-vocabulary buffer.
  float u = flashinfer::sampling::ieee_mul(uniforms[request * block_size + block_size - 1], mass);
  float aggregate = 0.0f;
  for (int i = 0; i * THREADS * VEC_SIZE < vocab; ++i) {
    int token = (i * THREADS + threadIdx.x) * VEC_SIZE;
    flashinfer::vec_t<float, VEC_SIZE> value;
    value.fill(0.0f);
    if (token < vocab) value.cast_load(row + token);
    flashinfer::sampling::DeviceSamplingFromProb<
        VEC_SIZE, THREADS, cub::BLOCK_SCAN_WARP_SCANS, cub::BLOCK_REDUCE_WARP_REDUCTIONS, true>(
        i, vocab, [](float x) { return x > 0.0f; }, u, value, aggregate, &storage);
    if (aggregate > u) break;
  }
  if (threadIdx.x == 0) {
    // Match FlashInfer's endpoint handling for the final floating-point CDF.
    output[request * 2 + 1] = storage.sampled_id == vocab
                                  ? storage.last_valid_id
                                  : storage.sampled_id;
  }
}

}  // namespace

extern "C" int sparse_speculative_sample_cuda(
    float* probs, const float* min_p, const int* requests, const uint32_t* candidates,
    const float* q, const uint32_t* drafts, const float* uniforms, uint32_t* output,
    int num_rows, int num_requests, int block_size, int vocab, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (min_p != nullptr) {
    normalize_min_p<<<num_rows, THREADS, 0, stream>>>(probs, min_p, vocab);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
  }
  const uint32_t vec_size = std::gcd(16 / sizeof(float), static_cast<size_t>(vocab));
  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    sparse_verify<VEC_SIZE><<<num_requests, THREADS, 0, stream>>>(
        probs, requests, candidates, q, drafts, uniforms, output, block_size, vocab);
  });
  return static_cast<int>(cudaGetLastError());
  PEGAINFER_FFI_GUARD_END(-1)
}
