/* Generation-only HVK oracle. This translation unit is never a serving input. */
#include <cuda_runtime.h>
#include <stdint.h>
#include "pegainfer_qwen35_gdn_upstream_hvk.h"

#define GDN(name) pegainfer_qwen35_gdn_upstream_hvk_##name

int upstream_in_place(int32_t tokens, void **p, int32_t workspace_bytes,
                      void *stream) {
    static GDN(Kernel_Module_t) module;
    static int loaded = 0;
    if (!loaded) {
        cudaError_t ret = cudaSuccess;
        cudaLibrary_t *library = &module.module;
        struct { cudaLibrary_t **library; cudaError_t *ret; }
            init = {&library, &ret};
        _mlir_pegainfer_qwen35_gdn_upstream_hvk_cuda_init((void **)&init);
        if (ret != cudaSuccess) return (int)ret;
        int32_t device = 0;
        struct { cudaLibrary_t **library; int32_t *device; cudaError_t *ret; }
            load = {&library, &device, &ret};
        _mlir_pegainfer_qwen35_gdn_upstream_hvk_cuda_load_to_device((void **)&load);
        if (ret != cudaSuccess) return (int)ret;
        loaded = 1;
    }
    GDN(Tensor_g_q_t) q = {p[0], {tokens}};
    GDN(Tensor_g_k_t) k = {p[1], {tokens}};
    GDN(Tensor_g_v_t) v = {p[2], {tokens}};
    GDN(Tensor_g_o_t) output = {p[3], {tokens}};
    GDN(Tensor_g_alpha_t) alpha = {p[4], {tokens * 32}};
    GDN(Tensor_g_beta_t) beta = {p[5], {tokens * 32}};
    GDN(Tensor_g_state_t) state = {p[6]};
    GDN(Tensor_g_init_state_t) initial = {p[6]};
    GDN(Tensor_g_tensormaps_t) workspace = {p[7], {workspace_bytes}};
    GDN(Tensor_cu_seqlens_t) cu_seqlens = {p[8], {2}};
    return cute_dsl_pegainfer_qwen35_gdn_upstream_hvk_wrapper(
        &module, &q, &k, &v, &output, &alpha, &beta, &state, &initial,
        &workspace, &cu_seqlens, 0.08838834764831845f,
        16, 16, 32, 32, 1, 1, 0, 32, (cudaStream_t)stream);
}
