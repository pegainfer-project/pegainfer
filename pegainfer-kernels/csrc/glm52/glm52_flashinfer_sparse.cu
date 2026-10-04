// GLM5.2 TP4 sparse MLA through FlashInfer's TRTLLM-generation FP8 kernels.
// The runner is header-only; the minimal SM100-family selector closure for
// our fixed decode shapes is embedded in the pegainfer-kernels archive.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <sstream>
#include <string>

#include "glm52_trtllm_fmha_cubins.inc"
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>

namespace {

constexpr int kHeads = 16;
constexpr int kHeadDimQk = 576;
constexpr int kHeadDimV = 512;
constexpr int kPageSize = 64;
constexpr size_t kCounterBytes = 8192ull * 256 * sizeof(uint32_t);

CUresult consume_last_cuda_error() {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue) return CUDA_ERROR_INVALID_VALUE;
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  return CUDA_ERROR_LAUNCH_FAILED;
}

}  // namespace

namespace flashinfer::trtllm_cubin_loader {

std::string getCubin(const std::string& kernel_name, const std::string& sha256) {
  auto load = [&](const char* name, const char* expected_sha, const unsigned char* data,
                  unsigned int size) -> std::string {
    if (kernel_name.find(name) == std::string::npos) return {};
    if (sha256 != expected_sha) {
      throw std::runtime_error("GLM5.2 FlashInfer cubin SHA-256 mismatch for " + kernel_name);
    }
    return std::string(reinterpret_cast<const char*>(data), size);
  };

  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
          "97b3d2de9c2025e967e54ce3986aa3c88954b13e7dde4927c4a672e54f4b68a0",
          kGlm52FmhaSparseSeedQ8, kGlm52FmhaSparseSeedQ8Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
          "6544178cddeffe987d4d7774549a7e0d8823815afa66e1fdc2131381c050ed41",
          kGlm52FmhaSparsePersistentSeedQ8,
          kGlm52FmhaSparsePersistentSeedQ8Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
          "b1640b8068bc12de38d3f2950b7af7b53e818d9fb6d9a07b2e630bb968007df3",
          kGlm52FmhaSparseShortQ8, kGlm52FmhaSparseShortQ8Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
          "eddd15b3ae26ac40468a61f1947a64f189c7aca0120ceb385f8ff1e1c634f08f",
          kGlm52FmhaSparseLongQ8, kGlm52FmhaSparseLongQ8Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
          "ca45c6611280408f82bd613411b0e25075833348ddac5dd0af24937cca2dd509",
          kGlm52FmhaSparseLongQ16, kGlm52FmhaSparseLongQ16Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
          "3e39ced03725e3a8d03f79a22ddc704d2d059cf3256692affae23759264c3db7",
          kGlm52FmhaSparseLongQ16V256, kGlm52FmhaSparseLongQ16V256Size);
      !cubin.empty()) {
    return cubin;
  }
  if (auto cubin = load(
          "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
          "2166c91e4c23c51222af9dfd30372daedf1a74c8727289168ffa8c21c552adeb",
          kGlm52FmhaSparseLongQ16V512, kGlm52FmhaSparseLongQ16V512Size);
      !cubin.empty()) {
    return cubin;
  }
  throw std::runtime_error("GLM5.2 FlashInfer requested an unembedded cubin: " + kernel_name);
}

}  // namespace flashinfer::trtllm_cubin_loader

namespace tensorrt_llm::kernels {

void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernel_meta,
                      KernelParams const&, int32_t, bool, cudaStream_t) {
  // The embedded kernels use either no multi-CTA reduction or in-kernel
  // global-memory reduction. A separate reduction cubin is never selected.
  if (kernel_meta.mMultiCtasKvMode ==
      static_cast<int>(MultiCtasKvMode::GmemReductionWithSeparateKernel)) {
    throw std::runtime_error("GLM5.2 FlashInfer selected an unsupported separate reduction");
  }
}

}  // namespace tensorrt_llm::kernels

extern "C" {

CUresult glm52_flashinfer_sparse_mla_supported_cuda(int heads, int* supported) {
  if (supported == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *supported = 0;
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return consume_last_cuda_error();
  }
  *supported = heads == kHeads && prop.major == 10 &&
               (prop.minor == 0 || prop.minor == 3);
  return CUDA_SUCCESS;
}

int glm52_flashinfer_sparse_mla_fp8_cuda(
    const unsigned char* query, const unsigned char* cache,
    const int* topk_indices, const int* seq_lens, __nv_bfloat16* out,
    unsigned char* workspace, size_t workspace_bytes, int tokens, int heads,
    int num_blocks, int topk, float sm_scale, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (query == nullptr || cache == nullptr || topk_indices == nullptr ||
      seq_lens == nullptr || out == nullptr || workspace == nullptr || heads != kHeads ||
      (tokens != 1 && tokens != 2 && tokens != 4 && tokens != 8) ||
      num_blocks <= 0 || (topk != 256 && topk != 2048) ||
      !std::isfinite(sm_scale) || sm_scale <= 0.0f ||
      workspace_bytes <= kCounterBytes) {
    return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
  }

  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return static_cast<int>(consume_last_cuda_error());
  }
  if (prop.major != 10 || (prop.minor != 0 && prop.minor != 3)) {
    return static_cast<int>(CUDA_ERROR_NOT_SUPPORTED);
  }

  using namespace tensorrt_llm::kernels;
  TllmGenFmhaRunner runner(DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
                           DATA_TYPE_BF16);
  TllmGenFmhaRunnerParams params;
  params.qPtr = const_cast<unsigned char*>(query);
  params.kPtr = const_cast<unsigned char*>(cache);
  params.vPtr = const_cast<unsigned char*>(cache);
  params.kvPageIdxPtr = const_cast<int*>(topk_indices);
  params.seqLensKvPtr = const_cast<int*>(seq_lens);
  params.oPtr = out;
  params.mHeadDimQk = kHeadDimQk;
  params.mHeadDimV = kHeadDimV;
  params.mNumHeadsQ = heads;
  params.mNumHeadsKv = 1;
  params.mNumHeadsQPerKv = heads;
  params.mBatchSize = tokens;
  params.mMaxSeqLenKv = topk;
  params.mMaxNumPagesPerSeqKv = topk;
  params.mNumTokensPerPage = kPageSize;
  params.mQkvLayout = QkvLayout::PagedKv;
  params.mNumPagesInMemPool = num_blocks;
  params.mMultiProcessorCount = prop.multiProcessorCount;
  params.qStrideTokens = heads * kHeadDimQk;
  params.qStrideHeads = kHeadDimQk;
  params.kStrideKeysValues = kHeadDimQk;
  params.kStrideHeads = kPageSize * kHeadDimQk;
  params.kStrideBatch = kPageSize * kHeadDimQk;
  params.vStrideKeysValues = kHeadDimQk;
  params.vStrideHeads = kPageSize * kHeadDimQk;
  params.vStrideBatch = kPageSize * kHeadDimQk;
  params.outputScale = 1.0f;
  params.mScaleSfKv = 1.0f;
  params.scaleSoftmaxLog2 = sm_scale * static_cast<float>(M_LOG2E);
  params.mChunkedAttentionSize = INT_MAX;
  params.mAttentionWindowSize = INT_MAX;
  params.mMaxSeqLenQ = 1;
  params.mSumOfSeqLensQ = tokens;
  params.mUsesSharedPagedKvIdx = true;
  params.enable_pdl = true;
  params.mMaskType = TrtllmGenAttentionMaskType::Dense;
  params.mKernelType = FmhaKernelType::Generation;
  params.mTileScheduler = TileScheduler::Static;
  params.mMultiCtasKvMode = true;
  params.mSparseMlaType = TrtllmGenSparseMlaType::StaticTokenSparse;
  params.mSparseMlaTopK = topk;
  params.sparseMlaTopKLensPtr = nullptr;
  params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace);
  params.multiCtasKvScratchPtr = workspace + kCounterBytes;
  params.stream = stream;

  auto [supported, info] = runner.isSupportedWithInfo(params);
  if (!supported) {
    throw std::runtime_error("missing GLM5.2 FlashInfer sparse MLA kernel: " + info);
  }
  runner.run(params);
  return static_cast<int>(consume_last_cuda_error());
  PEGAINFER_FFI_GUARD_END(-1)
}

}  // extern "C"
