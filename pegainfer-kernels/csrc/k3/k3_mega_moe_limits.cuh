#pragma once

namespace k3_mega {

// Token capacity one rank's slab and kernel instantiation carry
// (`num_max_tokens_per_rank`): the chunked-prefill ceiling, 44x the upstream
// token alignment (`layout::kLCMCandidateBlockM`, 384). This is the ONLY value
// any launch accepts: the ring capacities derived from it are kernel template
// parameters, so a slab allocated for any other value addresses the rings
// wrong. Exported to the Rust side through `k3_mega_max_tokens_per_rank`.
//
// 16896 = 4 x the historical 4224 ceiling: a 16k-class chunk lands the MoE
// GEMMs in their efficient shape range and lets a CP gang swallow a 64k
// prompt in one superstep at CP4 (16 x 16896 at CP16). The bill is the slab —
// the rings scale linearly with the protocol max (~6.3 GiB at EP4/224,
// ~19.6 GiB at EP16/896, measured via `k3_mega_symm_buffer_layout`).
constexpr int kProtocolMaxTokensPerRank = 16896;

}  // namespace k3_mega
