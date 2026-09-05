//! Kimi-K3 CUDA entry points.
//!
//! The masked grouped GEMM is DeepGEMM's SM100 `MGroupedMasked` FP8 x FP4
//! kernel, AOT-instantiated (no JIT, no torch). Both scale operands are the
//! Blackwell packed-UE8M0 i32 layout (`[groups, ceil(k / gran_k / 4), mn]`
//! MN-major, 4 exponent bytes per i32); the activation side uses a per-1x128
//! granularity and the FP4 weight side a group-32 granularity, so their packed
//! K extents differ (`k / 512` vs `k / 128`).
//!
//! See `csrc/k3/k3_deepgemm_fp8_fp4_grouped_sm100.cu`.

use core::ffi::c_void;

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// Checkpoint MXFP4 weight scales (`[groups, n, k / 32]` u8 UE8M0 exponent
    /// bytes, K-major) -> the runtime SFB tensor (`[groups, k / 128, n]` i32,
    /// MN-major, 4 exponent bytes per word LSB-first). Loader-time helper.
    pub fn k3_fp4_sf_prepare_cuda(
        sf: *const u8,
        packed: *mut i32,
        groups: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Masked grouped FP8 x FP4 GEMM over the rank's local experts.
    /// `operand_kind` is 1 for the fused W13 gate|up projection and 2 for W2.
    /// Requires sm_100f (returns `CUDA_ERROR_NOT_SUPPORTED` elsewhere).
    pub fn k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch_cuda(
        operand_kind: i32,
        a: *const u8,
        a_scale: *const i32,
        b: *const u8,
        b_scale: *const i32,
        masked_m: *const i32,
        out: *mut Half,
        groups: i32,
        n: i32,
        k: i32,
        masked_cap: i32,
        num_sms: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Routing metadata for one rank's expert window: per-expert row counts
    /// (`masked_m[groups]`) and the expanded-entry -> masked-slot map
    /// (`slot_map[tokens * topk]`, `-1` for inactive entries). Entry order
    /// (`token * topk + slot`) fixes the row assignment deterministically.
    /// `topk_idx` carries GLOBAL expert ids; an entry is active when
    /// `topk_idx - local_expert_base` lands in `[0, groups)`; the chain passes
    /// `local_expert_base = 0`.
    pub fn k3_moe_local_route_metadata_cuda(
        topk_idx: *const i32,
        masked_m: *mut i32,
        slot_map: *mut i32,
        tokens: i32,
        topk: i32,
        groups: i32,
        masked_cap: i32,
        local_expert_base: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Local gather fused with the W13 A-operand quant: token-major bf16
    /// latents `[tokens, hidden]` -> fp8 e4m3 `[groups * masked_cap, hidden]`
    /// plus MN-major UE8M0 f32 group scales `[groups, hidden / 128,
    /// masked_cap]`.
    pub fn k3_moe_gather_fp8_quant_masked_cuda(
        latent: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        output: *mut u8,
        scales: *mut f32,
        tokens: i32,
        topk: i32,
        hidden: i32,
        groups: i32,
        masked_cap: i32,
        local_expert_base: i32,
        stream: CUstream,
    ) -> CUresult;

    /// K3 situ activation over the masked gate|up rows
    /// (`4 * tanh(g / 4) * sigmoid(g) * 25 * tanh(u / 25)`, f32 over the bf16
    /// W13 output) followed by the W2 A-operand quant.
    pub fn k3_situ_and_mul_fp8_quant_masked_cuda(
        gate_up: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        output: *mut u8,
        scales: *mut f32,
        tokens: i32,
        topk: i32,
        inter: i32,
        groups: i32,
        masked_cap: i32,
        local_expert_base: i32,
        stream: CUstream,
    ) -> CUresult;

    /// f32 MN-major group scales `[groups, scale_cols, cap]` -> the packed
    /// UE8M0 i32 SFA tensor `[groups, scale_cols / 4, cap]`.
    pub fn k3_fp8_scale_pack_ue8m0_cuda(
        scales: *const f32,
        packed: *mut i32,
        groups: i32,
        scale_cols: i32,
        cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Weighted combine: masked W2 rows -> token-major bf16 hidden states,
    /// f32 accumulation in topk-slot order (no atomics), one bf16 round.
    pub fn k3_moe_weighted_combine_cuda(
        expert_out: *const Half,
        topk_idx: *const i32,
        slot_map: *const i32,
        topk_weight: *const f32,
        out: *mut Half,
        tokens: i32,
        topk: i32,
        hidden: i32,
        groups: i32,
        masked_cap: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Absorbed-MLA decode over the paged latent cache
    /// (`csrc/k3/k3_mla_paged_attn.cu`). One (row, head) block absorbs the
    /// query against `w_kv_b`'s W_UK, walks the row's block table, and expands
    /// the attended latent with W_UV. `layer_offset`/`page_stride` are in
    /// elements; `table` is `[b, max_pages]` i32 (`-1` = unmapped, read as
    /// zero latent) and `n` the per-row device context length.
    pub fn k3_mla_paged_attn_cuda(
        q: *const Half,
        w_kv_b: *const Half,
        cache: *const Half,
        layer_offset: i64,
        page_stride: i64,
        table: *const i32,
        max_pages: i32,
        n: *const i32,
        scale: *const Half,
        o: *mut Half,
        b: i32,
        num_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Sigmoid router plus biased top-k over merged f32 score rows
    /// (`csrc/k3/k3_router_topk.cu`): `s [b, num_experts]` f32,
    /// `bias [num_experts]` f32, the bf16 routed scale `rs [1]`; writes
    /// `idx [b, topk]` i32 and `wts [b, topk]` f32. Block-parallel argmax
    /// with the serial kernel's lowest-index tie-break; shapes are runtime
    /// values (no per-bucket instantiation).
    pub fn k3_router_topk_cuda(
        s: *const f32,
        bias: *const f32,
        rs: *const c_void,
        idx: *mut i32,
        wts: *mut f32,
        b: i32,
        num_experts: i32,
        topk: i32,
        stream: CUstream,
    ) -> CUresult;

    // --- fused MegaMoE (see `csrc/k3/k3_mega_moe_sm100.cu`) ---

    /// Token-count alignment the MegaMoE API enforces on
    /// `num_max_tokens_per_rank` (`layout::kLCMCandidateBlockM`).
    pub fn k3_mega_token_alignment() -> i32;

    /// Token capacity one rank's slab and the AOT kernels are built for
    /// (`num_max_tokens_per_rank`). The launch accepts exactly this value.
    pub fn k3_mega_max_tokens_per_rank() -> i32;

    /// Whether the AOT matrix carries a MegaMoE kernel for this world (GLOBAL
    /// expert count x rank count, situ activation). Non-zero means supported.
    pub fn k3_mega_world_supported(num_experts: i32, num_ranks: i32) -> i32;

    /// Open the device pair `(self_ordinal, peer_ordinal)` for the kernel's
    /// cross-rank addressing: peer access from this device's context, plus a
    /// memory-pool access grant so this device's stream-ordered allocations are
    /// visible from the peer. Must precede the allocations it protects.
    /// Idempotent, and a no-op for the self pair. In-process groups only —
    /// fabric mappings carry their own access grants.
    pub fn k3_mega_open_peer_access(self_ordinal: i32, peer_ordinal: i32) -> CUresult;

    /// Whether `device_ordinal` supports `CU_MEM_HANDLE_TYPE_FABRIC`
    /// allocations (driver + IMEX). Writes 0 or 1 into `out_supported`.
    pub fn k3_mega_fabric_supported(device_ordinal: i32, out_supported: *mut i32) -> CUresult;

    /// Store `value` into `flag_count` 8-byte-aligned u64 flag slots (local
    /// or fabric-imported device VAs) from a kernel on `stream`; see
    /// `csrc/k3/k3_whale_doorbell.cu`.
    pub fn k3_whale_doorbell_ring(
        flag_addrs: *const u64,
        flag_count: i32,
        value: u64,
        stream: CUstream,
    ) -> CUresult;

    /// Allocate a fabric-exportable symmetric slab of `num_bytes` on
    /// `device_ordinal`, mapped and access-granted for every local device,
    /// zeroed and synchronized. Writes the device pointer and the 64-byte
    /// `CUmemFabricHandle`. Process-lifetime: nothing frees it.
    pub fn k3_mega_fabric_slab_alloc(
        device_ordinal: i32,
        num_bytes: u64,
        out_ptr: *mut i64,
        out_handle: *mut u8,
    ) -> CUresult;

    /// Import a peer rank's 64-byte fabric handle and map it for every local
    /// device. `num_bytes` is the slab size before granularity rounding.
    /// Process-lifetime: nothing unmaps it.
    pub fn k3_mega_fabric_slab_import(
        handle: *const u8,
        num_bytes: u64,
        device_ordinal: i32,
        out_ptr: *mut i64,
    ) -> CUresult;

    /// Symmetric-buffer sizing: total bytes plus the 12 sub-buffer byte offsets
    /// in the order `x, x_sf, topk_idx, topk_weights, shared_l1_acts,
    /// shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf, l1_acts,
    /// l1_acts_sf, l2_acts, l2_acts_sf`. `out_offsets` must have room for 12.
    pub fn k3_mega_symm_buffer_layout_cuda(
        num_ranks: i32,
        num_experts: i32,
        num_max_tokens_per_rank: i32,
        num_topk: i32,
        hidden: i32,
        intermediate_hidden: i32,
        num_sms: i32,
        out_num_bytes: *mut u64,
        out_offsets: *mut u64,
        out_ring_tokens: *mut i32,
        out_sf_ring_tokens: *mut i32,
    ) -> CUresult;

    /// Gate/up interleave (granularity 8) over the packed-FP4 W13 bytes:
    /// `[groups, n, k / 2]` u8 in, same shape out, rows permuted from
    /// split-half `[gate | up]` into `[gate 0..7, up 0..7, gate 8..15, ...]`.
    pub fn k3_mega_prepare_l1_weights_cuda(
        src: *const u8,
        dst: *mut u8,
        groups: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Checkpoint UE8M0 scales (`[groups, n, k / 32]` u8, K-major) -> the
    /// MegaMoE weight SF tensor (`[groups, k / 128, n]` i32, MN-major, four
    /// exponents per word LSB-first) with the UTCCP row transpose applied, and
    /// additionally the gate/up interleave when `interleave != 0` (W13).
    pub fn k3_mega_prepare_sf_cuda(
        sf: *const u8,
        out: *mut i32,
        groups: i32,
        n: i32,
        k: i32,
        interleave: i32,
        stream: CUstream,
    ) -> CUresult;

    /// bf16 `[num_tokens, hidden]` -> e4m3 plus packed UE8M0 group-32 scales,
    /// written into the symmetric buffer's `x` / `x_sf` regions. Bit-for-bit
    /// DeepGEMM's `per_token_cast_to_fp8(use_ue8m0=True, gran_k=32,
    /// use_packed_ue8m0=True)`.
    pub fn k3_mega_quant_x_cuda(
        x: *const Half,
        x_fp8: *mut u8,
        x_sf: *mut i32,
        num_tokens: i32,
        hidden: i32,
        x_stride: i32,
        x_sf_stride: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Routing pair into the symmetric buffer, widening K3's i32 expert ids to
    /// the i64 the kernel reads.
    pub fn k3_mega_write_routing_cuda(
        topk_idx: *const i32,
        topk_weight: *const f32,
        dst_idx: *mut i64,
        dst_weight: *mut f32,
        num_tokens: i32,
        num_topk: i32,
        stream: CUstream,
    ) -> CUresult;

    /// The fused MegaMoE launch. `symm_ptrs` is the per-rank base-pointer table
    /// (one entry at `ep_size == 1`) and `symm_offsets` the 12 offsets from
    /// [`k3_mega_symm_buffer_layout_cuda`]. `activation` is 0 for swiglu and 1
    /// for K3's situ. Requires sm_100f (`NOT_SUPPORTED` elsewhere).
    pub fn k3_mega_moe_launch_cuda(
        y: *mut Half,
        l1_weights: *const u8,
        l1_weights_sf: *const i32,
        l2_weights: *const u8,
        l2_weights_sf: *const i32,
        symm_ptrs: *const i64,
        symm_offsets: *const u64,
        num_ranks: i32,
        rank_idx: i32,
        num_max_tokens_per_rank: i32,
        num_tokens: i32,
        num_experts: i32,
        num_topk: i32,
        hidden: i32,
        intermediate_hidden: i32,
        num_sms: i32,
        activation: i32,
        cumulative_stats: *mut i32,
        stream: CUstream,
    ) -> CUresult;

    /// Workspace bytes the FlashKDA forward needs for one sequence of
    /// `t_total` tokens at `h` heads. Pure host arithmetic.
    pub fn k3_flash_kda_workspace_bytes(t_total: i32, h: i32) -> i64;

    /// beta `[T, H]` bf16 -> `[H, T]` bf16, the layout FlashKDA's 1D TMA
    /// loads.
    pub fn k3_flash_kda_beta_transpose(
        beta_th: *const Half,
        beta_ht: *mut Half,
        t_total: i32,
        h: i32,
        stream: CUstream,
    ) -> CUresult;

    /// One sequence through the vendored FlashKDA chunkwise forward
    /// (third_party/flash-kda, MIT, MoonshotAI): q/k/v/g/out `[T, H, 128]`
    /// bf16, g pre-activation (dt_bias/exp(A_log)/sigmoid/lower-bound applied
    /// in-kernel), beta `[H, T]` bf16 logits, f32 recurrent state
    /// `[H, 128, 128]` carried in and out. Requires an accelerated SM90+
    /// build (`NOT_SUPPORTED` elsewhere).
    pub fn k3_flash_kda_fwd(
        q: *const Half,
        k: *const Half,
        v: *const Half,
        g: *const Half,
        beta_ht: *const Half,
        a_log: *const f32,
        dt_bias: *const f32,
        state_in: *const f32,
        state_out: *mut f32,
        out: *mut Half,
        workspace: *mut core::ffi::c_void,
        t_total: i32,
        h: i32,
        scale: f32,
        lower_bound: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Fused KCP package forward: one kernel-1 pass plus a dual-state
    /// kernel 2 producing the segment's affine package in one sweep —
    /// `state_out_d` = D (real v from zero state), `state_out_m` = M (v = 0
    /// from identity state), both `[H, 128, 128]` f32. No token output.
    /// Same operand contract and workspace as [`k3_flash_kda_fwd`].
    pub fn k3_flash_kda_fwd_md(
        q: *const Half,
        k: *const Half,
        v: *const Half,
        g: *const Half,
        beta_ht: *const Half,
        a_log: *const f32,
        dt_bias: *const f32,
        state_out_d: *mut f32,
        state_out_m: *mut f32,
        workspace: *mut core::ffi::c_void,
        t_total: i32,
        h: i32,
        scale: f32,
        lower_bound: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Chunked-prefill MLA context gather: walk one block-table row and split
    /// `t_total` cached 576-wide latent rows into dense `[t, 512]` latent and
    /// `[t, 64]` rope halves. Strides/offsets are in elements.
    pub fn k3_mla_prefill_gather(
        slab: *const Half,
        table: *const i32,
        page_stride: i64,
        layer_offset: i64,
        t_total: i32,
        latent_out: *mut Half,
        rope_out: *mut Half,
        stream: CUstream,
    ) -> CUresult;

    /// Assemble per-head K rows `[t, heads, 192]` from the kv_b expansion
    /// `[t, heads, 256]` (nope half) and the shared per-token rope `[t, 64]`
    /// broadcast across heads.
    pub fn k3_mla_prefill_expand_k(
        nope_v: *const Half,
        rope: *const Half,
        k_out: *mut Half,
        t_total: i32,
        heads: i32,
        stream: CUstream,
    ) -> CUresult;

    /// FlashMLA SM100 dense FMHA forward (third_party/FlashMLA), one
    /// sequence, bottom-right-aligned causal: q `[t_q, h, 192]`,
    /// k `[t_kv, h, 192]`, v a strided `[t_kv, h, 128]` view, out
    /// `[t_q, h, 128]`, all bf16; strides in elements. `lse_out`, when
    /// non-null, receives f32 `[h, t_q]` log-sum-exp (natural log, softmax
    /// scale absorbed). Requires an sm_100f build (`NOT_SUPPORTED`
    /// elsewhere).
    pub fn k3_flash_mla_prefill_fwd(
        q: *const Half,
        q_stride_tok: i64,
        q_stride_head: i64,
        k: *const Half,
        k_stride_tok: i64,
        k_stride_head: i64,
        v: *const Half,
        v_stride_tok: i64,
        v_stride_head: i64,
        out: *mut Half,
        o_stride_tok: i64,
        o_stride_head: i64,
        lse_out: *mut f32,
        t_q: i32,
        t_kv: i32,
        heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// The full-visibility twin of `k3_flash_mla_prefill_fwd` for context
    /// windows entirely in the queries' past: no causal mask, `t_q` and
    /// `t_kv` unrelated. Same layouts and LSE contract.
    pub fn k3_flash_mla_prefill_fwd_dense(
        q: *const Half,
        q_stride_tok: i64,
        q_stride_head: i64,
        k: *const Half,
        k_stride_tok: i64,
        k_stride_head: i64,
        v: *const Half,
        v_stride_tok: i64,
        v_stride_head: i64,
        out: *mut Half,
        o_stride_tok: i64,
        o_stride_head: i64,
        lse_out: *mut f32,
        t_q: i32,
        t_kv: i32,
        heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Fold one window's FMHA output `[t_q, h, 128]` bf16 + `[h, t_q]` f32
    /// LSE into the f32 running accumulator pair via the log-sum-exp
    /// identity; `reset != 0` starts a fresh accumulation.
    pub fn k3_mla_prefill_lse_merge(
        o_win: *const Half,
        lse_win: *const f32,
        o_acc: *mut f32,
        lse_acc: *mut f32,
        t_q: i32,
        heads: i32,
        reset: i32,
        stream: CUstream,
    ) -> CUresult;

    /// Convert the merged f32 accumulator back to the bf16 attention output
    /// `[t_q, h, 128]`.
    pub fn k3_mla_prefill_o_finalize(
        o_acc: *const f32,
        out: *mut Half,
        t_q: i32,
        heads: i32,
        stream: CUstream,
    ) -> CUresult;
}
