#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PEGAINFER_QWEN35_GDN_ABI_VERSION 1u

typedef enum {
    PEGAINFER_QWEN35_GDN_OK = 0,
    PEGAINFER_QWEN35_GDN_NOT_SUPPORTED = 1,
    PEGAINFER_QWEN35_GDN_INVALID_ARGUMENT = 2,
    PEGAINFER_QWEN35_GDN_ABI_MISMATCH = 3,
    PEGAINFER_QWEN35_GDN_CUDA_ERROR = 4,
} pegainfer_qwen35_gdn_status_t;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t sm;
    uint32_t h_q;
    uint32_t h_k;
    uint32_t h_v;
    uint32_t head_dim;
    uint32_t qkv_dtype;
    uint32_t state_dtype;
    uint32_t state_layout;
} pegainfer_qwen35_gdn_spec_t;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const void *q;
    const void *k;
    const void *v;
    void *output;
    const void *alpha;
    const void *beta;
    void *state;
    const void *initial_state;
    void *workspace;
    size_t workspace_bytes;
    const int64_t *cu_seqlens;
    uint32_t cu_seqlens_len;
    uint32_t tokens;
    uint32_t h_q;
    uint32_t h_k;
    uint32_t h_v;
    uint32_t head_dim;
    void *stream;
} pegainfer_qwen35_gdn_args_t;

uint32_t pegainfer_qwen35_gdn_abi_version(void);
const char *pegainfer_qwen35_gdn_artifact_sha256(void);
uint64_t pegainfer_qwen35_gdn_artifact_size_bytes(void);
int32_t pegainfer_qwen35_gdn_aot_available(void);
int32_t pegainfer_qwen35_gdn_supported(
    const pegainfer_qwen35_gdn_spec_t *spec);
int32_t pegainfer_qwen35_gdn_workspace_bytes(void *handle,
                                             size_t *workspace_bytes);
int32_t pegainfer_qwen35_gdn_create(void **handle, int32_t device);
int32_t pegainfer_qwen35_gdn_launch(void *handle,
                                   const pegainfer_qwen35_gdn_args_t *args);
void pegainfer_qwen35_gdn_destroy(void *handle);

#ifdef __cplusplus
}
#endif
