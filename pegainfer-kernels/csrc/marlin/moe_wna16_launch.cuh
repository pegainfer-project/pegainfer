// Host-side launch machinery for the vendored Marlin MoE kernel.
//
// The thread-config search and the shared-memory accounting below have to
// agree with `marlin_template.h` exactly — a second copy that drifts from the
// template does not fail loudly, it launches with too little shared memory.
// So the arithmetic lives here once, and each weight type contributes only the
// kernel table it can instantiate.
//
// Include it after `MARLIN_NAMESPACE_NAME` is defined and after the template
// headers: everything here lands in the includer's namespace, so two weight
// types in one build get separate tables without colliding.

#pragma once

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {}

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

/// Answers with `MarlinDefault` when no kernel is compiled for the shape.
using MarlinKernelGetter = MarlinFuncPtr (*)(
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    int group_blocks,
    int num_threads);

struct ThreadConfig {
  int thread_k;
  int thread_n;
  int num_threads;
};

struct ExecConfig {
  int blocks_per_sm;
  ThreadConfig tb_cfg;
};

constexpr int kMarlinMinThreadN = min_thread_n;
constexpr int kMarlinMaxThreadN = max_thread_n;
constexpr int kMarlinPipeStages = pipe_stages;

/// The shapes a caller is willing to tile with. They stay with the caller
/// because a table entry is only usable if the kernel table has the matching
/// instantiation, and because widening one model's search would silently
/// change which kernel another model picks.
struct ThreadConfigs {
  const ThreadConfig* small_batch;
  int small_count;
  const ThreadConfig* large_batch;
  int large_count;
};

inline CUresult last_error_to_cu(cudaError_t err) {
  return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

inline int get_scales_cache_size(
    ThreadConfig const& th_config,
    int prob_k,
    int group_size,
    bool has_act_order,
    bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * kMarlinPipeStages * 2;
    load_groups = max(load_groups, 32);
    return load_groups * tb_n * 2;
  }
  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * kMarlinPipeStages;
}

inline int get_kernel_cache_size(
    ThreadConfig const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float) {
  int pack_factor = 32 / num_bits;
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  int sh_block_meta_size = tb_m * 4;
  int sh_a_size = kMarlinPipeStages * (tb_m * tb_k) * 2;
  int sh_b_size = kMarlinPipeStages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = max(max(sh_b_size, sh_red_size), sh_red_size + sh_bias_size);
  int sh_s_size =
      get_scales_cache_size(th_config, prob_k, num_bits == 4 ? group_size : -1,
                            has_act_order, is_k_full);
  int sh_g_idx_size = has_act_order && !is_k_full ? kMarlinPipeStages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float) {
      sh_zp_size = sh_s_size;
    } else if (num_bits == 4) {
      sh_zp_size = sh_s_size / 4;
    } else if (num_bits == 8) {
      sh_zp_size = sh_s_size / 2;
    }
  }
  (void)m_block_size_8;
  return tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size +
         sh_block_meta_size;
}

inline bool is_valid_config(
    ThreadConfig const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }
  if (th_config.num_threads < 128) {
    return false;
  }
  int cache_size = get_kernel_cache_size(
      th_config, m_block_size_8, thread_m_blocks, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float);
  return cache_size + 512 <= max_shared_mem;
}

inline ExecConfig determine_exec_config(
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int group_size,
    int max_shared_mem,
    ThreadConfigs const& tables,
    MarlinKernelGetter get_kernel) {
  ExecConfig exec_cfg{1, ThreadConfig{-1, -1, -1}};
  const ThreadConfig* configs =
      thread_m_blocks > 1 ? tables.large_batch : tables.small_batch;
  int config_count = thread_m_blocks > 1 ? tables.large_count : tables.small_count;
  constexpr int device_max_reg_size = 255 * 1024;
  int best_count = 0;
  for (int i = 0; i < config_count; ++i) {
    ThreadConfig cfg = configs[i];
    if (!is_valid_config(cfg, m_block_size_8, thread_m_blocks, prob_n, prob_k,
                         4, group_size, false, true, false, false,
                         max_shared_mem)) {
      continue;
    }
    MarlinFuncPtr kernel = get_kernel(
        thread_m_blocks, cfg.thread_n / 16, cfg.thread_k / 16,
        m_block_size_8, group_size / 16, cfg.num_threads);
    if (kernel == MarlinDefault) {
      continue;
    }
    if (thread_m_blocks > 1) {
      exec_cfg = ExecConfig{1, cfg};
      break;
    }
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    if (err != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    int reg_size = max(attr.numRegs, 1) * cfg.num_threads * 4;
    int cache_size = get_kernel_cache_size(cfg, m_block_size_8, thread_m_blocks,
                                           prob_k, 4, group_size, false,
                                           true, false, false);
    int allow_count = min(device_max_reg_size / reg_size,
                          max_shared_mem / (cache_size + 1024));
    allow_count = max(min(allow_count, 4), 1);
    if (allow_count > best_count) {
      best_count = allow_count;
      exec_cfg = ExecConfig{allow_count, cfg};
    }
  }
  return exec_cfg;
}

/// One expert-blocked Marlin GEMM. `b_scales` is whatever the instantiated
/// `s_type` reads, and `global_scale` is the per-tensor factor an FP4 weight
/// type folds in; a type that scales per block alone passes null.
inline CUresult launch_marlin_moe_gemm(
    const void* input,
    void* output,
    float* c_tmp,
    const void* b_qweight,
    const void* b_scales,
    const float* global_scale,
    int* workspace,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    const float* topk_weights,
    int workspace_len,
    int sorted_token_ids_len,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    int size_m,
    int size_n,
    int size_k,
    int group_size,
    int sm_count,
    ThreadConfigs const& tables,
    MarlinKernelGetter get_kernel,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || b_qweight == nullptr ||
      b_scales == nullptr || workspace == nullptr || sorted_token_ids == nullptr ||
      expert_ids == nullptr || num_tokens_post_padded == nullptr ||
      topk_weights == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr bool use_atomic_add = false;
  constexpr bool use_fp32_reduce = true;
  if (use_fp32_reduce && !use_atomic_add && c_tmp == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // The kernel table is keyed on whole 16-wide K blocks per group, so a
  // group that is not a multiple of 16 would truncate on lookup and run a
  // kernel whose compile-time scale geometry disagrees with the group count
  // the launch reports.
  if (top_k <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0 ||
      sorted_token_ids_len <= 0 || group_size <= 0 || group_size % 16 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!(moe_block_size == 8 ||
        (moe_block_size >= 16 && moe_block_size <= 64 && moe_block_size % 16 == 0))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (size_k % tile_size != 0 || size_n % min_thread_n != 0 ||
      size_k % group_size != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;
  if (sm_count <= 0) {
    err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess || sm_count <= 0) return CUDA_ERROR_INVALID_VALUE;
  }
  int max_shared_mem = 0;
  err = cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (err != cudaSuccess || max_shared_mem <= 0) return CUDA_ERROR_INVALID_VALUE;

  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  ExecConfig exec_cfg = determine_exec_config(
      size_n, size_k, thread_m_blocks, m_block_size_8, group_size,
      max_shared_mem, tables, get_kernel);
  ThreadConfig cfg = exec_cfg.tb_cfg;
  if (cfg.thread_k == -1) return CUDA_ERROR_NOT_SUPPORTED;

  int max_n_tiles = size_n / min_thread_n;
  int min_workspace_size =
      min(max_n_tiles * (sorted_token_ids_len / moe_block_size), sm_count * 4);
  if (workspace_len < min_workspace_size) return CUDA_ERROR_INVALID_VALUE;

  int thread_k_blocks = cfg.thread_k / 16;
  int thread_n_blocks = cfg.thread_n / 16;
  int blocks = sm_count * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1) {
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
  }
  if (!is_valid_config(cfg, m_block_size_8, thread_m_blocks, size_n, size_k, 4,
                       group_size, false, true, false, false, max_shared_mem)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  MarlinFuncPtr kernel = get_kernel(
      thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8,
      group_size / 16, cfg.num_threads);
  if (kernel == MarlinDefault) return CUDA_ERROR_NOT_SUPPORTED;

  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared_mem);
  if (err != cudaSuccess) return CUDA_ERROR_INVALID_VALUE;

  const int4* A_ptr = reinterpret_cast<const int4*>(input);
  const int4* B_ptr = reinterpret_cast<const int4*>(b_qweight);
  int4* C_ptr = reinterpret_cast<int4*>(output);
  int4* C_tmp_ptr = reinterpret_cast<int4*>(c_tmp);
  const int4* scales_ptr = reinterpret_cast<const int4*>(b_scales);
  int* locks = workspace;

  kernel<<<blocks, cfg.num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, nullptr, nullptr, scales_ptr, global_scale,
      nullptr, nullptr, sorted_token_ids, expert_ids, num_tokens_post_padded,
      topk_weights, top_k, mul_topk_weights, size_k / group_size, size_m,
      size_n, size_k, locks, false, use_atomic_add, use_fp32_reduce);
  return last_error_to_cu(cudaPeekAtLastError());
}

}  // namespace MARLIN_NAMESPACE_NAME
