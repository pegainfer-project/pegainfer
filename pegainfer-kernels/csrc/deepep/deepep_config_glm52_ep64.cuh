// GLM5.2 sixty-four-GPU DeepEP config (DP1/EP64, 16 GB300 NVL72 trays): 256
// routed experts place 4 whole experts per rank — the decode-island target.
// kNumLocalExperts (4) is below kNumTopk (8); the derived worst-case
// expanded-token capacity already clamps with min(topk, local).
//
// Comm/device facts stay at the EP8 values: kDeviceSms=132 is a compile-time
// floor ctx_create checks with `>=` (GB300 has 152, H200 exactly 132), and
// GB300's sharedMemPerBlockOptin equals H200's 232448. The decode collective
// kernels use kDecodeNumSms regardless.
#pragma once

#include <cstdint>

namespace deepep_shim::cfg_glm52_ep64 {

// Model / topology (GLM5.2, DP1/EP64).
inline constexpr int kNumRanks = 64;
inline constexpr int kNumExperts = 256;
inline constexpr int kNumLocalExperts = kNumExperts / kNumRanks;  // 4
inline constexpr int kNumTopk = 8;
inline constexpr int kHidden = 6144;
inline constexpr int kHiddenBytes = kHidden * 2;  // bf16 payload, no FP8 SF
inline constexpr int kExpertAlignment = 64;       // masked-DeepGEMM M granularity

// Comm tuning and device facts: identical to the EP8 config (see header
// comment for why the H200-derived values also hold on GB300).
inline constexpr int kKernelQPs = 9;
inline constexpr int kAllocatedQPs = 17;
inline constexpr int64_t kTimeoutCycles = 200'000'000'000;
inline constexpr int kDeviceSms = 132;
inline constexpr int kSmemBytes = 232448;

// 128, not the current 8-slot protocol cap: deliberate headroom for
// unibatch prefill tokens riding the decode dispatch (the NVL72 plan).
// Fixed buffer cost scales with kNumRanks * kDecodeMaxTokens; revisit
// per width if the headroom is never used.
inline constexpr int kDecodeMaxTokens = 128;
inline constexpr int kDecodeNumSms = 32;

#include "deepep_config_derived.cuh"

}  // namespace deepep_shim::cfg_glm52_ep64
