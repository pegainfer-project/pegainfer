#include "flashinfer_gdn_aot.h"

#include <cuda_runtime.h>
#include <stdlib.h>

#include "flashinfer_gdn_build_config.h"

#ifdef PEGAINFER_QWEN35_GDN_AOT
static int32_t status_from_cuda(cudaError_t error) {
    if (error == cudaSuccess) return PEGAINFER_QWEN35_GDN_OK;
    if (error == cudaErrorNotSupported)
        return PEGAINFER_QWEN35_GDN_NOT_SUPPORTED;
    if (error == cudaErrorInvalidValue)
        return PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT;
    return PEGAINFER_QWEN35_GDN_CUDA_ERROR;
}
#include "kernel.h"

typedef struct {
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Kernel_Module_t module;
    int32_t device;
    size_t workspace_bytes;
} gdn_handle_t;

static int32_t load_current_device(
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Kernel_Module_t *module,
    int32_t device) {
    cudaError_t ret = cudaSuccess;
    cudaLibrary_t *library = &module->module;
    struct {
        cudaLibrary_t **library;
        cudaError_t *ret;
    } init_args = {&library, &ret};
    _mlir_pegainfer_qwen35_gdn_qwen35_4b_candidate_cuda_init(
        (void **)&init_args);
    if (ret != cudaSuccess) return (int32_t)ret;
    struct {
        cudaLibrary_t **library;
        int32_t *device;
        cudaError_t *ret;
    } load_args = {&library, &device, &ret};
    _mlir_pegainfer_qwen35_gdn_qwen35_4b_candidate_cuda_load_to_device(
        (void **)&load_args);
    return (int32_t)ret;
}
#endif

uint32_t pegainfer_qwen35_gdn_abi_version(void) {
    return PEGAINFER_QWEN35_GDN_ABI_VERSION;
}

const char *pegainfer_qwen35_gdn_artifact_sha256(void) {
    return PEGAINFER_QWEN35_GDN_ARTIFACT_SHA256;
}

int32_t pegainfer_qwen35_gdn_aot_available(void) {
#ifdef PEGAINFER_QWEN35_GDN_AOT
    return 1;
#else
    return 0;
#endif
}

int32_t pegainfer_qwen35_gdn_create(void **handle, int32_t device) {
    if (handle == NULL) return PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT;
    *handle = NULL;
#ifdef PEGAINFER_QWEN35_GDN_AOT
    cudaError_t ret = cudaSetDevice(device);
    if (ret != cudaSuccess) return status_from_cuda(ret);
    int32_t major = 0, minor = 0;
    ret = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                 device);
    if (ret != cudaSuccess) return status_from_cuda(ret);
    ret = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                 device);
    if (ret != cudaSuccess) return status_from_cuda(ret);
    if (major != 12 || minor != 0) return PEGAINFER_QWEN35_GDN_NOT_SUPPORTED;
    int32_t sm_count = 0;
    ret = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                 device);
    if (ret != cudaSuccess) return status_from_cuda(ret);
    gdn_handle_t *owner = (gdn_handle_t *)calloc(1, sizeof(*owner));
    if (owner == NULL) return PEGAINFER_QWEN35_GDN_CUDA_ERROR;
    owner->device = device;
    owner->workspace_bytes =
        (size_t)sm_count * PEGAINFER_QWEN35_GDN_WORKSPACE_BYTES_PER_SM;
    int32_t rc = load_current_device(&owner->module, device);
    if (rc != (int32_t)cudaSuccess) {
        free(owner);
        return PEGAINFER_QWEN35_GDN_CUDA_ERROR;
    }
    *handle = owner;
    return PEGAINFER_QWEN35_GDN_OK;
#else
    (void)device;
    return PEGAINFER_QWEN35_GDN_NOT_SUPPORTED;
#endif
}

int32_t pegainfer_qwen35_gdn_workspace_bytes(void *handle,
                                             size_t *workspace_bytes) {
    if (handle == NULL || workspace_bytes == NULL)
        return PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT;
#ifdef PEGAINFER_QWEN35_GDN_AOT
    gdn_handle_t *owner = (gdn_handle_t *)handle;
    *workspace_bytes = owner->workspace_bytes;
    return PEGAINFER_QWEN35_GDN_OK;
#else
    return PEGAINFER_QWEN35_GDN_NOT_SUPPORTED;
#endif
}

int32_t pegainfer_qwen35_gdn_launch(void *handle,
                                   const pegainfer_qwen35_gdn_args_t *args) {
#ifdef PEGAINFER_QWEN35_GDN_AOT
    if (handle == NULL || args == NULL ||
        args->struct_size != sizeof(*args) || args->tokens == 0 ||
        args->q == NULL || args->k == NULL || args->v == NULL ||
        args->output == NULL || args->alpha == NULL || args->beta == NULL ||
        args->state == NULL || args->initial_state == NULL ||
        args->workspace == NULL || args->cu_seqlens == NULL ||
        args->stream == NULL) {
        return PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT;
    }
    if (args->abi_version != PEGAINFER_QWEN35_GDN_ABI_VERSION)
        return PEGAINFER_QWEN35_GDN_ABI_MISMATCH;
    gdn_handle_t *owner = (gdn_handle_t *)handle;
    cudaError_t cuda_rc = cudaSetDevice(owner->device);
    if (cuda_rc != cudaSuccess) return status_from_cuda(cuda_rc);
    if (args->workspace_bytes < owner->workspace_bytes)
        return PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT;

    int32_t tokens = (int32_t)args->tokens;
    int32_t gates = tokens * 32;
    int32_t workspace_bytes = (int32_t)args->workspace_bytes;
    int32_t cu_count = 2;
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_q_t q = {
        (void *)args->q, {tokens}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_k_t k = {
        (void *)args->k, {tokens}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_v_t v = {
        (void *)args->v, {tokens}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_o_t output = {
        args->output, {tokens}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_alpha_t alpha = {
        (void *)args->alpha, {gates}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_beta_t beta = {
        (void *)args->beta, {gates}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_state_t state = {
        args->state};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_init_state_t initial = {
        (void *)args->initial_state};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_g_tensormaps_t workspace = {
        args->workspace, {workspace_bytes}};
    pegainfer_qwen35_gdn_qwen35_4b_candidate_Tensor_cu_seqlens_t cu_seqlens = {
        (void *)args->cu_seqlens, {cu_count}};
    int32_t rc = cute_dsl_pegainfer_qwen35_gdn_qwen35_4b_candidate_wrapper(
        &owner->module, &q, &k, &v, &output, &alpha, &beta, &state,
        &initial, &workspace, &cu_seqlens, 0.08838834764831845f,
        16, 16, 32, 32, 1, 1, 0, 32, (cudaStream_t)args->stream);
    return rc == 0 ? PEGAINFER_QWEN35_GDN_OK
                   : PEGAINFER_QWEN35_GDN_CUDA_ERROR;
#else
    (void)handle;
    (void)args;
    return PEGAINFER_QWEN35_GDN_NOT_SUPPORTED;
#endif
}

void pegainfer_qwen35_gdn_destroy(void *handle) {
#ifdef PEGAINFER_QWEN35_GDN_AOT
    if (handle != NULL) {
        gdn_handle_t *owner = (gdn_handle_t *)handle;
        cudaSetDevice(owner->device);
        cudaLibraryUnload(owner->module.module);
        free(owner);
    }
#else
    (void)handle;
#endif
}
