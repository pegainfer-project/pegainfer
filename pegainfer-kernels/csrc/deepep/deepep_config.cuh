// Kimi-K2 single-node 8×H200 DeepEP config: compile-time constants plus the
// shared warp-count derivations (deepep_config_derived.cuh).
#pragma once

#include <cstdint>

namespace deepep_shim::cfg {

// Model / topology (Kimi-K2, TP1/DP8/EP8).
inline constexpr int kNumRanks = 8;
inline constexpr int kNumExperts = 384;
inline constexpr int kNumLocalExperts = kNumExperts / kNumRanks;  // 48
inline constexpr int kNumTopk = 8;
inline constexpr int kHidden = 7168;
inline constexpr int kHiddenBytes = kHidden * 2;  // bf16, no FP8 SF
inline constexpr int kExpertAlignment = 8;        // Marlin routing block size

// Comm tuning, mirroring upstream defaults for direct (single-node) mode:
// kernel QPs = min(num_sms, 8 + 1); allocated GIN contexts = 17 (non-hybrid).
inline constexpr int kKernelQPs = 9;
inline constexpr int kAllocatedQPs = 17;
// GPU-side timeout in cycles. Upstream bakes 100 s × device clock rate at JIT
// time; we fix ~100 s at the H200's ~2 GHz. Only affects hang detection.
inline constexpr int64_t kTimeoutCycles = 200'000'000'000;

// Device facts baked at compile time (H200 SXM, sm_90). ctx_create asserts
// the actual device meets these; epilogue/prologue grids are templated on
// kDeviceSms so it must be <= the real SM count for cooperative launches.
inline constexpr int kDeviceSms = 132;
inline constexpr int kSmemBytes = 232448;  // sharedMemPerBlockOptin

// Per-path configs. Decode runs without CPU sync at fixed worst-case shapes
// (graph-capturable); prefill syncs the CPU on receive counts so buffers are
// allocated at actual size. Prefill max tokens matches the retired PPLX cap
// (PPLX_MAX_DISPATCH_TOKENS = 2048).
inline constexpr int kDecodeMaxTokens = 128;
inline constexpr int kDecodeNumSms = 32;
inline constexpr int kPrefillMaxTokens = 2048;
inline constexpr int kPrefillNumSms = 64;

#include "deepep_config_derived.cuh"

}  // namespace deepep_shim::cfg
