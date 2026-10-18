// GLM5.2 four-GPU DeepEP config (DP1/EP4, 4xGB300 bring-up target): 256
// routed experts place 64 whole experts per rank instead of EP8's 32.
//
// Comm/device facts stay at the EP8 values: kDeviceSms=132 is a compile-time
// floor ctx_create checks with `>=` (GB300 has 152, H200 exactly 132), and
// GB300's sharedMemPerBlockOptin equals H200's 232448. The decode collective
// kernels use kDecodeNumSms regardless.
#pragma once

#include <cstdint>

namespace deepep_shim::cfg_glm52_ep4 {

// Model / topology (GLM5.2, DP1/EP4).
inline constexpr int kNumRanks = 4;
inline constexpr int kNumExperts = 256;
inline constexpr int kNumLocalExperts = kNumExperts / kNumRanks;  // 64
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

inline constexpr int kDecodeMaxTokens = 128;
inline constexpr int kDecodeNumSms = 32;

#include "deepep_config_derived.cuh"

}  // namespace deepep_shim::cfg_glm52_ep4
