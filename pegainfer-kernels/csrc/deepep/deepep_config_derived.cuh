// Derivations shared by every DeepEP shim config: constexpr mirrors of the
// upstream JIT host logic (csrc/kernels/elastic/{dispatch,combine}.hpp).
//
// Include this INSIDE a config namespace, after defining the base constants:
//   kNumRanks, kNumExperts, kNumLocalExperts, kNumTopk, kHidden, kHiddenBytes,
//   kExpertAlignment, kKernelQPs, kAllocatedQPs, kTimeoutCycles, kDeviceSms,
//   kSmemBytes, kDecodeMaxTokens, kDecodeNumSms. Prefill-capable instances
//   additionally define kPrefillMaxTokens and kPrefillNumSms.
//
// deepep_ctx_create runtime-asserts these mirrors against the real layout
// classes so upstream layout changes fail loudly instead of corrupting
// buffers.

#ifndef DEEPEP_SHIM_HAS_PREFILL
#error "instantiation must define DEEPEP_SHIM_HAS_PREFILL before its config"
#endif

inline constexpr int kTMAAlign = 32;  // ptx::kNumTMAAlignBytes

constexpr int align_up(int a, int b) { return (a + b - 1) / b * b; }
constexpr int min_i(int a, int b) { return a < b ? a : b; }

// layout::TokenLayout::get_num_bytes<kWithMBarrier>() for our shapes
// (sf bytes are always 0; mbarrier is 8 bytes aligned up to 32).
constexpr int token_smem_bytes(int hidden_bytes, int topk, bool with_metadata,
                               bool with_mbarrier) {
    const int metadata_bytes = topk * 8 + (with_metadata ? (1 + topk) * 4 : 0);
    return align_up(hidden_bytes, kTMAAlign) + align_up(metadata_bytes, kTMAAlign) +
           (with_mbarrier ? kTMAAlign : 0);
}

inline constexpr int kDispatchTokenSmem = token_smem_bytes(kHiddenBytes, kNumTopk, true, true);
inline constexpr int kCombineTokenSmem = token_smem_bytes(kHiddenBytes, kNumTopk, false, true);
inline constexpr int kReduceTokenSmem = align_up(kHiddenBytes, kTMAAlign);

// get_num_notify_smem_bytes(num_ranks, num_experts) with kNumNotifyWarps = 4.
inline constexpr int kNumNotifyWarps = 4;
inline constexpr int kNotifySmemBytes =
    align_up(kNumRanks + kNumExperts, kNumNotifyWarps * 32) * static_cast<int>(sizeof(int));

// launch_dispatch warp derivation (direct mode).
constexpr int dispatch_warps(int num_sms) {
    return min_i(min_i((kSmemBytes - kNotifySmemBytes) / kDispatchTokenSmem,
                       32 - kNumNotifyWarps),
                 (512 + num_sms - 1) / num_sms);
}

inline constexpr int kCombineWarps = min_i(kSmemBytes / kCombineTokenSmem, 32);
inline constexpr int kCopyEpilogueWarps = min_i(kSmemBytes / kDispatchTokenSmem, 32);
inline constexpr int kReduceEpilogueWarps = min_i(kSmemBytes / kReduceTokenSmem, 32);
inline constexpr int kPrologueWarps = 8;
inline constexpr int kBarrierThreads = 512;

// Worst-case decode capacities (no CPU sync ⇒ fixed shapes), mirroring
// buffer.hpp's non-cached/no-sync branch.
inline constexpr int kDecodeWorstRecvTokens = kNumRanks * kDecodeMaxTokens;
inline constexpr int kDecodeWorstExpandedTokens = align_up(
    kNumRanks * kDecodeMaxTokens * min_i(kNumTopk, kNumLocalExperts) +
        (kExpertAlignment - 1) * kNumLocalExperts,
    kExpertAlignment);

#if DEEPEP_SHIM_HAS_PREFILL
inline constexpr int kPrefillWorstRecvTokens = kNumRanks * kPrefillMaxTokens;
#endif
