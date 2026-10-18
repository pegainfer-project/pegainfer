/* GLM5.2 CuTe DSL tcgen05 fp8 blockwise GEMM — AOT dispatch shim.
 *
 * The launchable artifacts (host entry + embedded cubin) are exported at
 * build time by tools/cutedsl/export_glm52_fp8_dsl.py; the generated
 * glm52_fp8_dsl_gen.h wraps them into G52DSL_TABLE, keyed on the exact
 * (m, n, k) of the four wide-route projections per decode bucket (16-96).
 * Anything off-table (prefill shards, MoE banks) stays on the CUTLASS
 * entry. Without PEGAINFER_CUTEDSL_PYTHON the stub half compiles instead and
 * the Rust side sees an always-empty table.
 *
 * Split-K entries write f32 partials into the caller's workspace (the CUTLASS
 * workspace, unused on this path) and chain the reduce kernel on the same
 * stream. Loading must happen before CUDA-graph capture; the launch entries
 * themselves are capture-safe.
 */

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef GLM52_CUTEDSL_AOT

#include "glm52_fp8_dsl_gen.h"

static const g52dsl_entry_t *g52dsl_find(int32_t m, int32_t n, int32_t k) {
    for (size_t i = 0; i < G52DSL_TABLE_LEN; i++) {
        const g52dsl_entry_t *e = &G52DSL_TABLE[i];
        if (e->m == m && e->n == n && e->k == k) {
            return e;
        }
    }
    return NULL;
}

int32_t glm52_fp8_dsl_gemm_load_cuda(void) {
    /* The generated Module_Load walks every device; restore the caller's
     * current device afterwards — EP rank threads allocate on whatever is
     * current, and a leaked switch lands their buffers on the last device. */
    int device = 0;
    cudaError_t device_err = cudaGetDevice(&device);
    (void)cudaGetLastError();
    for (size_t i = 0; i < G52DSL_TABLE_LEN; i++) {
        G52DSL_TABLE[i].load();
        if (G52DSL_TABLE[i].red_load) {
            G52DSL_TABLE[i].red_load();
        }
    }
    cudaError_t load_err = cudaGetLastError();
    if (device_err == cudaSuccess) {
        cudaError_t restore_err = cudaSetDevice(device);
        if (load_err == cudaSuccess) {
            load_err = restore_err;
        }
    }
    return (int32_t)load_err;
}

int32_t glm52_fp8_dsl_gemm_supported_cuda(int32_t m, int32_t n, int32_t k) {
    return g52dsl_find(m, n, k) != NULL;
}

int32_t glm52_fp8_dsl_gemm_cuda(const void *activation,
                                const void *activation_scale,
                                const void *weight, const void *weight_scale,
                                void *output, void *workspace,
                                size_t workspace_bytes, int32_t m, int32_t n,
                                int32_t k, cudaStream_t stream) {
    const g52dsl_entry_t *e = g52dsl_find(m, n, k);
    if (!e) {
        return (int32_t)cudaErrorNotSupported;
    }
    if (e->split_k > 1) {
        size_t partial_bytes =
            (size_t)e->split_k * (size_t)m * (size_t)n * sizeof(float);
        if (partial_bytes > workspace_bytes) {
            return (int32_t)cudaErrorMemoryAllocation;
        }
        int32_t rc = e->run((void *)activation, (void *)weight,
                            (void *)activation_scale, (void *)weight_scale,
                            workspace, stream);
        if (rc) {
            return rc;
        }
        return e->red_run(workspace, output, stream);
    }
    return e->run((void *)activation, (void *)weight, (void *)activation_scale,
                  (void *)weight_scale, output, stream);
}

#else /* !GLM52_CUTEDSL_AOT */

int32_t glm52_fp8_dsl_gemm_load_cuda(void) { return 0; }

int32_t glm52_fp8_dsl_gemm_supported_cuda(int32_t m, int32_t n, int32_t k) {
    (void)m;
    (void)n;
    (void)k;
    return 0;
}

int32_t glm52_fp8_dsl_gemm_cuda(const void *activation,
                                const void *activation_scale,
                                const void *weight, const void *weight_scale,
                                void *output, void *workspace,
                                size_t workspace_bytes, int32_t m, int32_t n,
                                int32_t k, cudaStream_t stream) {
    (void)activation;
    (void)activation_scale;
    (void)weight;
    (void)weight_scale;
    (void)output;
    (void)workspace;
    (void)workspace_bytes;
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    return (int32_t)cudaErrorNotSupported;
}

#endif
