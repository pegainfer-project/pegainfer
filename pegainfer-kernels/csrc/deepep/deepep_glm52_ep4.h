// C ABI for the GLM5.2 EP4 instantiation of the DeepEP elastic shim (DP1/EP4,
// 256 experts / 64 local / topk 8 / hidden 6144 / expert alignment 64).
//
// Same contract as deepep.h with `glm52_ep4_deepep_` symbols and its own
// opaque context tag; DeepEpInfo is shared. See deepep.h for the full
// semantics.
#pragma once

#include "deepep.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Glm52Ep4DeepEpCtx Glm52Ep4DeepEpCtx;

const char* glm52_ep4_deepep_last_error(void);

void glm52_ep4_deepep_info(DeepEpInfo* out);

int glm52_ep4_deepep_unique_id(uint8_t out[128]);

int glm52_ep4_deepep_ctx_create(const uint8_t unique_id[128], int32_t num_ranks, int32_t rank_idx,
                                Glm52Ep4DeepEpCtx** out);

int glm52_ep4_deepep_ctx_destroy(Glm52Ep4DeepEpCtx* ctx);

int glm52_ep4_deepep_decode_dispatch(
    Glm52Ep4DeepEpCtx* ctx, void* stream,
    const void* x,
    const int32_t* topk_idx,
    const float* topk_weights,
    int32_t num_tokens,
    int32_t* rank_count_scratch,
    int32_t* dst_slot_scratch,
    int32_t* psum_rank,
    int32_t* psum_expert,
    void* recv_x,
    float* recv_topk_weights,
    int32_t* recv_src_metadata);

int glm52_ep4_deepep_decode_combine(
    Glm52Ep4DeepEpCtx* ctx, void* stream,
    const void* x,
    const int32_t* src_metadata,
    const int32_t* psum_rank,
    const int32_t* combined_topk_idx,
    int32_t num_tokens,
    void* combined_x);

#ifdef __cplusplus
}  // extern "C"
#endif
