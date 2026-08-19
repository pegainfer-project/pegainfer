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

#if defined(__cplusplus)
#define PEGAINFER_GDN_STATIC_ASSERT static_assert
#define PEGAINFER_GDN_ALIGNOF alignof
#else
#define PEGAINFER_GDN_STATIC_ASSERT _Static_assert
#define PEGAINFER_GDN_ALIGNOF _Alignof
#endif

#define PEGAINFER_GDN_ASSERT_OFFSET(type, field, expected)                  \
    PEGAINFER_GDN_STATIC_ASSERT(offsetof(type, field) == (expected),        \
                                #type "." #field " ABI offset changed")

PEGAINFER_GDN_STATIC_ASSERT(sizeof(pegainfer_qwen35_gdn_spec_t) == 40,
                            "GDN spec ABI size changed");
PEGAINFER_GDN_STATIC_ASSERT(PEGAINFER_GDN_ALIGNOF(pegainfer_qwen35_gdn_spec_t) == 4,
                            "GDN spec ABI alignment changed");
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, abi_version, 0);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, struct_size, 4);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, sm, 8);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, h_q, 12);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, h_k, 16);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, h_v, 20);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, head_dim, 24);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, qkv_dtype, 28);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, state_dtype, 32);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_spec_t, state_layout, 36);

PEGAINFER_GDN_STATIC_ASSERT(sizeof(pegainfer_qwen35_gdn_args_t) == 128,
                            "GDN args ABI size changed");
PEGAINFER_GDN_STATIC_ASSERT(PEGAINFER_GDN_ALIGNOF(pegainfer_qwen35_gdn_args_t) == 8,
                            "GDN args ABI alignment changed");
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, abi_version, 0);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, struct_size, 4);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, q, 8);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, k, 16);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, v, 24);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, output, 32);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, alpha, 40);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, beta, 48);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, state, 56);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, initial_state, 64);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, workspace, 72);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, workspace_bytes, 80);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, cu_seqlens, 88);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, cu_seqlens_len, 96);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, tokens, 100);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, h_q, 104);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, h_k, 108);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, h_v, 112);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, head_dim, 116);
PEGAINFER_GDN_ASSERT_OFFSET(pegainfer_qwen35_gdn_args_t, stream, 120);

#undef PEGAINFER_GDN_ASSERT_OFFSET
#undef PEGAINFER_GDN_STATIC_ASSERT
#undef PEGAINFER_GDN_ALIGNOF

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
