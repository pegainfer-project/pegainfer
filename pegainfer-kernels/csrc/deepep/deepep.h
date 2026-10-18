// C ABI for the DeepEP elastic EP shim (Kimi-K2 single-node 8-rank config).
//
// AOT replacement for DeepEP's NVRTC JIT + torch host layer: the kernel
// template specializations for the Kimi-K2 config are instantiated at build
// time (deepep_shim.cu) and launched through cudaLaunchKernelExC with the
// same grid/cluster/cooperative/PDL attributes the upstream JIT runtime uses.
//
// Threading model: one host thread per rank, each with its CUDA device set
// before any call. ctx_create / ctx_destroy / barrier are collective — all
// ranks must call them. All kernel launches are stream-ordered on the
// caller's stream; nothing here creates streams or syncs the device except
// where documented.
//
// All functions return 0 on success. On failure they return -1 and the
// message is available via deepep_last_error() (thread-local).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeepEpCtx DeepEpCtx;

// Compile-time capacities of the baked config. Rust sizes all device
// allocations from this — there is no other sizing contract.
typedef struct {
    int32_t num_ranks;
    int32_t num_experts;
    int32_t num_local_experts;
    int32_t num_topk;
    int32_t hidden;            // elements, bf16
    int32_t expert_alignment;  // per-expert segment alignment in expanded layout

    // Decode path: do_expand=true, do_cpu_sync=false, deterministic.
    // Fixed worst-case shapes; CUDA-graph capturable.
    int32_t decode_max_tokens_per_rank;
    int32_t decode_worst_recv_tokens;      // recv_src_metadata rows
    int32_t decode_worst_expanded_tokens;  // recv_x / recv_topk_weights rows

    // Prefill path: do_expand=true, do_cpu_sync=true, deterministic.
    // Actual counts come back through deepep_prefill_wait_counts.
    int32_t prefill_max_tokens_per_rank;
    int32_t prefill_worst_recv_tokens;

    // Scratch sizing (i32 element counts).
    int32_t prologue_rank_count_len;  // deterministic-prologue per-SM rank counters

    int64_t buffer_bytes;     // symmetric data buffer (informational)
    int64_t workspace_bytes;  // symmetric workspace (informational)
} DeepEpInfo;

const char* deepep_last_error(void);

void deepep_info(DeepEpInfo* out);

// NCCL unique id for rank-0 to generate and share (128 bytes).
int deepep_unique_id(uint8_t out[128]);

// Collective. Creates the NCCL communicator + device communicator + symmetric
// window for this rank. Requires NCCL >= 2.30.4 (device API) and all ranks in
// one NVLink domain. Internally barriers (window registration is collective).
int deepep_ctx_create(const uint8_t unique_id[128], int32_t num_ranks, int32_t rank_idx,
                      DeepEpCtx** out);

// Collective. Synchronizes the device, barriers across ranks, releases
// everything including the NCCL communicator.
int deepep_ctx_destroy(DeepEpCtx* ctx);

// ---------------------------------------------------------------------------
// Decode: deterministic prologue + dispatch + copy epilogue, one call.
// All outputs caller-allocated at the worst-case sizes from DeepEpInfo.
// Expanded layout: recv_x rows are per-(expert, token) slots, per-local-expert
// segments contiguous and aligned to expert_alignment; psum_expert[i] are the
// running aligned offsets (exclusive form, first num_local_experts entries).
// ---------------------------------------------------------------------------
int deepep_decode_dispatch(
    DeepEpCtx* ctx, void* stream,
    const void* x,                // [num_tokens, hidden] bf16
    const int32_t* topk_idx,      // [num_tokens, num_topk] global expert ids
    const float* topk_weights,    // [num_tokens, num_topk]
    int32_t num_tokens,           // <= decode_max_tokens_per_rank (0 allowed)
    int32_t* rank_count_scratch,  // [prologue_rank_count_len]
    int32_t* dst_slot_scratch,    // [decode_max_tokens_per_rank * num_topk]
    int32_t* psum_rank,           // out [num_ranks]
    int32_t* psum_expert,         // out [num_local_experts + 1]
    void* recv_x,                 // out [decode_worst_expanded_tokens, hidden] bf16
    float* recv_topk_weights,     // out [decode_worst_expanded_tokens]
    int32_t* recv_src_metadata);  // out [decode_worst_recv_tokens, num_topk + 2]

// Decode combine + reduce epilogue, one call. x carries the expert outputs in
// the expanded slots (router weights already applied by the caller); the
// reduction sums each source token's slots and writes [num_tokens, hidden].
int deepep_decode_combine(
    DeepEpCtx* ctx, void* stream,
    const void* x,                   // [decode_worst_expanded_tokens, hidden] bf16
    const int32_t* src_metadata,     // from dispatch
    const int32_t* psum_rank,        // from dispatch
    const int32_t* combined_topk_idx,  // original [num_tokens, num_topk]
    int32_t num_tokens,
    void* combined_x);  // out [num_tokens, hidden] bf16

// ---------------------------------------------------------------------------
// Prefill: same kernels with CPU-synced counts so recv buffers can be
// allocated at actual size. Three phases:
//   1. dispatch_send   — prologue + dispatch (stream-ordered)
//   2. wait_counts     — CPU spin on pinned counters (no stream involvement)
//   3. dispatch_recv   — copy epilogue into actual-size buffers
// ---------------------------------------------------------------------------
int deepep_prefill_dispatch_send(
    DeepEpCtx* ctx, void* stream,
    const void* x,                // [num_tokens, hidden] bf16
    const int32_t* topk_idx,      // [num_tokens, num_topk]
    const float* topk_weights,    // [num_tokens, num_topk]
    int32_t num_tokens,           // <= prefill_max_tokens_per_rank
    int32_t* rank_count_scratch,  // [prologue_rank_count_len]
    int32_t* dst_slot_scratch,    // [prefill_max_tokens_per_rank * num_topk]
    int32_t* psum_rank,           // out [num_ranks]
    int32_t* psum_expert);        // out [num_local_experts + 1]

// Blocks the CPU until this rank's receive counts arrive (pinned-memory spin,
// ~100 s timeout). num_expanded_tokens is already segment-aligned: it is the
// exact recv_x row count.
int deepep_prefill_wait_counts(
    DeepEpCtx* ctx,
    int32_t* num_recv_tokens,        // out
    int32_t* num_expanded_tokens);   // out

int deepep_prefill_dispatch_recv(
    DeepEpCtx* ctx, void* stream,
    int32_t num_recv_tokens,      // from wait_counts
    const int32_t* psum_rank,
    const int32_t* psum_expert,
    void* recv_x,                 // out [num_expanded_tokens, hidden] bf16
    float* recv_topk_weights,     // out [num_expanded_tokens]
    int32_t* recv_src_metadata);  // out [num_recv_tokens, num_topk + 2]

int deepep_prefill_combine(
    DeepEpCtx* ctx, void* stream,
    const void* x,                // [num_expanded_tokens, hidden] bf16
    const int32_t* src_metadata,
    const int32_t* psum_rank,
    int32_t num_recv_tokens,      // from wait_counts
    const int32_t* combined_topk_idx,  // original [num_tokens, num_topk]
    int32_t num_tokens,
    void* combined_x);  // out [num_tokens, hidden] bf16

#ifdef __cplusplus
}  // extern "C"
#endif
