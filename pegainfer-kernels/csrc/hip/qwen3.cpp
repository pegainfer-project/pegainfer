#include "runtime.h"
#include <cmath>
#include <cstdint>
#include <hip/hip_bfloat16.h>
using B = hip_bfloat16;

__device__ float reduce_sum(float value, float *tmp) {
  int t = threadIdx.x;
  tmp[t] = value;
  __syncthreads();
  for (int n = blockDim.x / 2; n; n /= 2) {
    if (t < n)
      tmp[t] += tmp[t + n];
    __syncthreads();
  }
  return tmp[0];
}
__global__ void embedding(const B *weights, const uint32_t *ids, B *output,
                          int rows, int hidden, int vocab) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(rows) * hidden) {
    auto id = ids[i / hidden];
    output[i] = id < uint32_t(vocab) ? weights[size_t(id) * hidden + i % hidden]
                                     : B(0.f);
  }
}
__global__ void norm(const B *x, const B *weight, B *out, int width,
                     float eps) {
  __shared__ float tmp[256];
  int row = blockIdx.x;
  float sum = 0;
  for (int d = threadIdx.x; d < width; d += blockDim.x) {
    float v = float(x[size_t(row) * width + d]);
    sum += v * v;
  }
  float inv = rsqrtf(reduce_sum(sum, tmp) / width + eps);
  for (int d = threadIdx.x; d < width; d += blockDim.x)
    out[size_t(row) * width + d] =
        B(float(x[size_t(row) * width + d]) * inv * float(weight[d]));
}
__global__ void add(const B *a, const B *b, B *out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = B(float(a[i]) + float(b[i]));
}
__global__ void silu(const B *gate, const B *up, B *out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = float(gate[i]);
    float activated = float(B(g / (1.f + expf(-g))));
    out[i] = B(activated * float(up[i]));
  }
}
__global__ void rope(B *q, const int32_t *positions, int rows, int heads,
                     int dim, float theta) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t n = size_t(rows) * heads * (dim / 2);
  if (i >= n)
    return;
  int pair = i % (dim / 2);
  size_t base = (i / (dim / 2)) * dim;
  int row = i / (size_t(heads) * (dim / 2));
  float angle = float(positions[row]) * powf(theta, -2.f * pair / dim);
  float c = float(B(cosf(angle))), s = float(B(sinf(angle)));
  float a = float(q[base + pair]), b = float(q[base + pair + dim / 2]);
  q[base + pair] = B(float(B(a * c)) - float(B(b * s)));
  q[base + pair + dim / 2] = B(float(B(b * c)) + float(B(a * s)));
}
__global__ void cache(const B *k, const B *v, B *keys, B *values,
                      const int32_t *slots, const int32_t *positions, int rows,
                      int kv_heads, int dim, int context, int capacity) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t width = size_t(kv_heads) * dim;
  if (i >= size_t(rows) * width)
    return;
  int row = i / width, slot = slots[row], pos = positions[row];
  if (slot < 0 || slot >= capacity || pos < 0 || pos >= context)
    return;
  size_t dst = (size_t(slot) * context + pos) * width + i % width;
  keys[dst] = k[i];
  values[dst] = v[i];
}
__global__ void attention(const B *q, const B *keys, const B *values, B *out,
                          const int32_t *slots, const int32_t *positions,
                          int rows, int heads, int kv_heads, int dim,
                          int context, int capacity) {
  int row = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
  int slot = slots[row], pos = positions[row];
  if (slot < 0 || slot >= capacity || pos < 0 || pos >= context)
    return;
  int kv = head / (heads / kv_heads), lane = t % warpSize, wave = t / warpSize;
  extern __shared__ float scores[];
  __shared__ float tmp[256];
  size_t qbase = (size_t(row) * heads + head) * dim;
  for (int token = wave; token <= pos; token += blockDim.x / warpSize) {
    size_t kb = ((size_t(slot) * context + token) * kv_heads + kv) * dim;
    float score = 0;
    for (int d = lane; d < dim; d += warpSize)
      score += float(q[qbase + d]) * float(keys[kb + d]);
    for (int n = warpSize / 2; n; n /= 2)
      score += __shfl_down(score, n, warpSize);
    if (lane == 0)
      scores[token] = score * rsqrtf(float(dim));
  }
  __syncthreads();
  float m = -INFINITY;
  for (int token = t; token <= pos; token += blockDim.x)
    m = fmaxf(m, scores[token]);
  tmp[t] = m;
  __syncthreads();
  for (int n = blockDim.x / 2; n; n /= 2) {
    if (t < n)
      tmp[t] = fmaxf(tmp[t], tmp[t + n]);
    __syncthreads();
  }
  m = tmp[0];
  __syncthreads();
  float sum = 0;
  for (int token = t; token <= pos; token += blockDim.x) {
    float p = expf(scores[token] - m);
    scores[token] = p;
    sum += p;
  }
  float denom = reduce_sum(sum, tmp);
  for (int token = t; token <= pos; token += blockDim.x)
    scores[token] /= denom;
  __syncthreads();
  for (int d = t; d < dim; d += blockDim.x) {
    float value = 0;
    for (int token = 0; token <= pos; ++token) {
      size_t vb = ((size_t(slot) * context + token) * kv_heads + kv) * dim;
      value += scores[token] * float(values[vb + d]);
    }
    out[qbase + d] = B(value);
  }
}
__global__ void top1(const B *logits, uint32_t *ids, int vocab) {
  __shared__ float values[256];
  __shared__ uint32_t indices[256];
  int row = blockIdx.x, t = threadIdx.x;
  float best = -INFINITY;
  uint32_t chosen = 0;
  for (int i = t; i < vocab; i += blockDim.x) {
    float x = float(logits[size_t(row) * vocab + i]);
    if (x > best || (x == best && uint32_t(i) < chosen)) {
      best = x;
      chosen = i;
    }
  }
  values[t] = best;
  indices[t] = chosen;
  __syncthreads();
  for (int n = blockDim.x / 2; n; n /= 2) {
    if (t < n && (values[t + n] > values[t] || (values[t + n] == values[t] &&
                                                indices[t + n] < indices[t]))) {
      values[t] = values[t + n];
      indices[t] = indices[t + n];
    }
    __syncthreads();
  }
  if (!t)
    ids[row] = indices[0];
}
extern "C" int pgi_embedding(void *device, const B *w, const uint32_t *ids,
                             B *out, int rows, int h, int vocab) {
  auto *d = (PgiDevice *)device;
  embedding<<<(size_t(rows) * h + 255) / 256, 256, 0, d->stream>>>(
      w, ids, out, rows, h, vocab);
  return pgi_hip(hipGetLastError(), "embedding");
}
extern "C" int pgi_norm(void *device, const B *x, const B *w, B *out, int rows,
                        int h, float eps) {
  auto *d = (PgiDevice *)device;
  norm<<<rows, 256, 0, d->stream>>>(x, w, out, h, eps);
  return pgi_hip(hipGetLastError(), "norm");
}
extern "C" int pgi_add(void *device, const B *a, const B *b, B *out, size_t n) {
  auto *d = (PgiDevice *)device;
  add<<<(n + 255) / 256, 256, 0, d->stream>>>(a, b, out, n);
  return pgi_hip(hipGetLastError(), "add");
}
extern "C" int pgi_silu(void *device, const B *a, const B *b, B *out,
                        size_t n) {
  auto *d = (PgiDevice *)device;
  silu<<<(n + 255) / 256, 256, 0, d->stream>>>(a, b, out, n);
  return pgi_hip(hipGetLastError(), "silu");
}
extern "C" int pgi_rope(void *device, B *q, const int32_t *p, int rows,
                        int heads, int dim, float theta) {
  auto *d = (PgiDevice *)device;
  rope<<<(size_t(rows) * heads * (dim / 2) + 255) / 256, 256, 0, d->stream>>>(
      q, p, rows, heads, dim, theta);
  return pgi_hip(hipGetLastError(), "rope");
}
extern "C" int pgi_cache(void *device, const B *k, const B *v, B *keys,
                         B *values, const int32_t *slots, const int32_t *pos,
                         int rows, int heads, int dim, int context,
                         int capacity) {
  auto *d = (PgiDevice *)device;
  cache<<<(size_t(rows) * heads * dim + 255) / 256, 256, 0, d->stream>>>(
      k, v, keys, values, slots, pos, rows, heads, dim, context, capacity);
  return pgi_hip(hipGetLastError(), "KV append");
}
extern "C" int pgi_attention(void *device, const B *q, const B *keys,
                             const B *values, B *out, const int32_t *slots,
                             const int32_t *pos, int rows, int heads,
                             int kv_heads, int dim, int context, int capacity) {
  auto *d = (PgiDevice *)device;
  attention<<<dim3(rows, heads), 256, size_t(context) * sizeof(float),
              d->stream>>>(q, keys, values, out, slots, pos, rows, heads,
                           kv_heads, dim, context, capacity);
  return pgi_hip(hipGetLastError(), "attention");
}
extern "C" int pgi_top1(void *device, const B *x, uint32_t *ids, int rows,
                        int vocab) {
  auto *d = (PgiDevice *)device;
  top1<<<rows, 256, 0, d->stream>>>(x, ids, vocab);
  return pgi_hip(hipGetLastError(), "argmax");
}
extern "C" int pgi_gemm(void *device, const B *w, const B *x, B *out, int rows,
                        int n, int k) {
  auto *d = (PgiDevice *)device;
  float alpha = 1.f, beta = 0.f;
  PGI_BLAS(rocblas_gemm_ex(
      d->blas, rocblas_operation_transpose, rocblas_operation_none, n, rows, k,
      &alpha, w, rocblas_datatype_bf16_r, k, x, rocblas_datatype_bf16_r, k,
      &beta, out, rocblas_datatype_bf16_r, n, out, rocblas_datatype_bf16_r, n,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  return 0;
}
