// Fused KCP package forward: one FlashKDA recurrence pass producing both the
// segment transition M and offset D.
//
// A CP middle rank exports its KDA package `(M, D)` — `S_out = S_in·M + D` —
// which the plain path derives with two full doctored forwards: `M` from
// (v = 0, state = I) and `D` from (real v, state = 0). Both discard the token
// output (the rank's real forward runs later from the merged state), and both
// consume the *same* kernel-1 workspace: k_decayed/k_restored/g_total/INV
// depend only on q/k/g/beta, never on v or the state. So the fused pass runs
// kernel 1 once and a derived kernel 2 that walks the chunk tiles carrying
// TWO state accumulators:
//
//   u_d = INV @ ((v − k_decayed @ S_D)·β)     S_D ← S_D·g + k_restored^T @ u_d
//   u_m = INV @ ((    − k_decayed @ S_M)·β)   S_M ← S_M·g + k_restored^T @ u_m
//
// with S_D seeded 0 and S_M seeded I in smem (no state TMA loads), and the
// q_decayed/Mqk loads, the q@S and Mqk@U GEMMs, and the entire output store
// path removed. Per tile that is 6 GEMMs against the two calls' 10, one
// workspace read instead of two, and zero output traffic.
//
// Derived from third_party/flash-kda csrc/smxx/fwd_kernel2.cuh +
// fwd_launch.cu (MIT, MoonshotAI — see PROVENANCE.md); the per-state tile
// math keeps the upstream GEMM shapes and accumulation order exactly, so
// each state is bit-identical to its standalone doctored call. Included by
// k3_flash_kda.cu inside the K3_FLASH_KDA_SM90A guard — one TU, implicit
// instantiation, D = 128 / fp32 states / non-varlen (N = 1) only.

#pragma once

template <class Layouts, int InputStages>
struct SharedStorageK2MD {
    using BF16 = cutlass::bfloat16_t;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_d;
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_m;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
    };

    // The fp32 conversion buffer overlays the pipeline stages: state stores
    // happen strictly after the tile loop, one state at a time.
    union {
        InputStorage input[InputStages];
        alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];
    };

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
};

// ==================== Kernel 2-MD: dual-state recurrence ====================
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadWsKD, class TmaLoadWsKR,
    class TmaLoadWsGT, class TmaLoadWsINV,
    class TmaStoreStateD,
    class TmaStoreStateM,
    int CHUNK,
    int D,
    int InputStages,
    int NumThreads
>
__global__ void __launch_bounds__(NumThreads) _k3_flash_kda_fwd_md_recurrence(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadWsKD const tma_load_ws_kd,
    CUTE_GRID_CONSTANT TmaLoadWsKR const tma_load_ws_kr,
    CUTE_GRID_CONSTANT TmaLoadWsGT const tma_load_ws_gt,
    CUTE_GRID_CONSTANT TmaLoadWsINV const tma_load_ws_inv,
    CUTE_GRID_CONSTANT TmaStoreStateD const tma_store_state_d,
    CUTE_GRID_CONSTANT TmaStoreStateM const tma_store_state_m,
    int T_total,
    int H,
    int total_tiles
) {
    using BF16 = cutlass::bfloat16_t;
    using Layouts = K2Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMALMLayout = typename Layouts::TMALMLayout;
    using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
    using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
    using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;
    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = 128;

    // Transaction bytes: v + beta + k_decayed + k_restored + g_total + INV
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +
        uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 2 +
        uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16));

    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK2MD<Layouts, InputStages>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    int warp_id = threadIdx.x / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    }

    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads
    );

    // Non-varlen N = 1: one sequence, whole chunk.
    int head_idx = blockIdx.y;
    int seq_len = T_total;
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    bool lane_predicate = cute::elect_one_sync();

    // --- Seed the states: S_D = 0, S_M = I (no state TMA loads).
    {
        BF16* buf_d = shared_storage.state_d.begin();
        BF16* buf_m = shared_storage.state_m.begin();
        constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
        for (int i = threadIdx.x; i < kTotal; i += NumThreads) {
            buf_d[i] = BF16(0);
            buf_m[i] = BF16(0);
        }
    }
    __syncthreads();
    {
        Tensor s_m = make_tensor(
            make_smem_ptr(shared_storage.state_m.begin()), StateSmemLayout{});
        for (int i = threadIdx.x; i < D; i += NumThreads) {
            s_m(i, i) = BF16(1);
        }
    }
    // generic writes -> visible to async proxy (TMA state store covers t_tiles==0)
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    // --- LOAD warp: TMA loads for v, beta, and the v/state-independent
    // workspace subset (k_decayed, k_restored, g_total, INV).
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));
        auto g_ws_kd = tma_load_ws_kd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_kr = tma_load_ws_kr.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_gt = tma_load_ws_gt.get_tma_tensor(make_shape(H * total_tiles, D));
        auto g_ws_inv = tma_load_ws_inv.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));

        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        auto cta_ws_kd = tma_load_ws_kd.get_slice(Int<0>{});
        auto cta_ws_kr = tma_load_ws_kr.get_slice(Int<0>{});
        auto cta_ws_gt = tma_load_ws_gt.get_slice(Int<0>{});
        auto cta_ws_inv = tma_load_ws_inv.get_slice(Int<0>{});

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            int ws_idx = head_idx * total_tiles + t;

            {
                auto v_off = g_v.layout()(head_idx, t * CHUNK, 0);
                Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_v.layout())));
                Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
                cute::copy(tma_load_v.with(*tma_barrier),
                    cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));
            }
            {
                int beta_linear = head_idx * T_total + t * CHUNK;
                int beta_aligned = beta_linear & ~7;
                auto beta_off = g_beta.layout()(beta_aligned);
                Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
                Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
                cute::copy(tma_load_beta.with(*tma_barrier),
                    cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));
            }
            {
                auto off = g_ws_kd.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_kd.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kd.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_decayed.begin()), TMAVOLayout{});
                cute::copy(tma_load_ws_kd.with(*tma_barrier), cta_ws_kd.partition_S(g_tile), cta_ws_kd.partition_D(s_tile));
            }
            {
                auto off = g_ws_kr.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_kr.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kr.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_restored.begin()), TMAVOLayout{});
                cute::copy(tma_load_ws_kr.with(*tma_barrier), cta_ws_kr.partition_S(g_tile), cta_ws_kr.partition_D(s_tile));
            }
            {
                auto off = g_ws_gt.layout()(ws_idx, 0);
                Tensor g_tile = make_tensor(g_ws_gt.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_ws_gt.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), TMAGTotalSmemLayout{});
                cute::copy(tma_load_ws_gt.with(*tma_barrier), cta_ws_gt.partition_S(g_tile), cta_ws_gt.partition_D(s_tile));
            }
            {
                auto off = g_ws_inv.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_inv.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws_inv.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].INV.begin()), TMALMLayout{});
                cute::copy(tma_load_ws_inv.with(*tma_barrier), cta_ws_inv.partition_S(g_tile), cta_ws_inv.partition_D(s_tile));
            }

            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }

    // --- MMA warps
    if (warp_role == WarpRole::MMA) {
        cutlass::arch::NamedBarrier compute_barrier(kComputeThreads, 0);
        LoadPipelineState load_read;
        int compute_tid = threadIdx.x;

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.consumer_wait(load_read);
            int load_stage = load_read.index();

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + t * CHUNK) & 7;

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});

            Tensor s_d = make_tensor(make_smem_ptr(shared_storage.state_d.begin()), StateSmemLayout{});
            Tensor s_d_T = make_tensor(make_smem_ptr(shared_storage.state_d.begin()), TransposedStateSmemLayout{});
            Tensor s_m = make_tensor(make_smem_ptr(shared_storage.state_m.begin()), StateSmemLayout{});
            Tensor s_m_T = make_tensor(make_smem_ptr(shared_storage.state_m.begin()), TransposedStateSmemLayout{});

            {
            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id_c = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);

            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);

            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_d, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            // Accumulators: [state 0 = D, state 1 = M][column block]
            AccFragT u_acc[2][2];
            #pragma unroll
            for (int st = 0; st < 2; ++st)
                #pragma unroll
                for (int i = 0; i < 2; ++i) { u_acc[st][i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[st][i]); }

            // ======== Phase 1: k@S for BOTH states (k-loop, 2 blocks per warp) ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    // A(k) is consumed into registers; prefetch A(k+1) after use.
                }

                #pragma unroll
                for (int st = 0; st < 2; ++st) {
                    auto& s_state = (st == 0) ? s_d : s_m;
                    #pragma unroll
                    for (int bi = 0; bi < 2; ++bi) {
                        copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                            local_tile(s_state, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id_c * 2 + bi, k))), tCrBi_view);
                        cute::transform(tCrBi, tCrB, cute::identity{});
                        gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[st][bi]);
                    }
                }

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                }
            }

            // ======== Phase 2: load v (D path only) and INV, beta ========
            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id_c * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));

            // ======== Phase 3: u = INV @ ((v − u)·β) per state (v ≡ 0 for M),
            // then MOVM_T into B fragments for Phase 6 ========
            BFragT_u tCrB_u_arr[2][2];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int st = 0; st < 2; ++st) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    SFragT u_bf16;
                    cute::transform(u_acc[st][i], u_bf16, [] __device__ (float x) { return BF16(x); });

                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            BF16 v0 = (st == 0) ? v_bf16[i](c0) : BF16(0);
                            BF16 v1 = (st == 0) ? v_bf16[i](c1) : BF16(0);
                            u_bf16(c0) = (v0 - u_bf16(c0)) * beta0;
                            u_bf16(c1) = (v1 - u_bf16(c1)) * beta1;
                        }
                    }

                    uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));
                    SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                    SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                    SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                    SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                    auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                    uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                    b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                    b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                    clear(u_acc[st][i]);
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[st][i]);

                    cute::transform(u_acc[st][i], u_bf16, [] __device__ (float x) { return BF16(x); });

                    u_c = reinterpret_cast<uint32_t*>(&u_bf16(0));
                    SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                    SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                    SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                    SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                    tCrB_u_arr[st][i] = thr_mma.partition_fragment_B(B_ref);
                    b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[st][i](0));
                    b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                    b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];
                }
            }

            // ======== Phase 6: S ← S·g + k_restored^T @ U for BOTH states ========
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[2][2][PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                #pragma unroll
                for (int st = 0; st < 2; ++st) {
                    auto& s_state_T = (st == 0) ? s_d_T : s_m_T;
                    #pragma unroll
                    for (int bi = 0; bi < 2; ++bi) {
                        Tensor s_block = local_tile(s_state_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, warp_id_c * 2 + bi));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[st][bi][i]));
                    }
                }

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                AccFragT s_upd[2][2];
                #pragma unroll
                for (int st = 0; st < 2; ++st)
                    #pragma unroll
                    for (int bi = 0; bi < 2; ++bi) {
                        s_upd[st][bi] = thr_mma.make_fragment_C(tCrC_ref);
                        clear(s_upd[st][bi]);
                        gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[st][bi](_,_,Int<0>{}), s_upd[st][bi]);
                    }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int st = 0; st < 2; ++st) {
                    auto& s_state_T = (st == 0) ? s_d_T : s_m_T;
                    #pragma unroll
                    for (int bi = 0; bi < 2; ++bi) {
                        #pragma unroll
                        for (int a = 0; a < 2; ++a) {
                            #pragma unroll
                            for (int d = 0; d < 2; ++d) {
                                auto c0 = make_coord(make_coord(a, 0), 0, d);
                                auto c1 = make_coord(make_coord(a, 1), 0, d);
                                ring_S_acc[st][bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[st][bi][slot](c0)) * g0 + s_upd[st][bi](c0));
                                ring_S_acc[st][bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[st][bi][slot](c1)) * g1 + s_upd[st][bi](c1));
                            }
                        }

                        Tensor s_block = local_tile(s_state_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id_c * 2 + bi));
                        copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[st][bi][slot]), smem_thr_store_C_T.partition_D(s_block));

                        if (m + PREFETCH < S_M_BLOCKS) {
                            Tensor s_next = local_tile(s_state_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, warp_id_c * 2 + bi));
                            copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[st][bi][slot]));
                        }
                    }
                }
            }
            }
            compute_barrier.arrive_and_wait();

            cutlass::arch::fence_view_async_shared();
            load_pipeline.consumer_release(load_read);
            ++load_read;
        }
    }

    // --- Epilogue: fp32-convert and TMA-store both states, sequentially
    // through the shared conversion buffer (pipeline smem is dead now).
    __syncthreads();

    auto store_state = [&](BF16 const* state_smem, auto const& tma_store) {
        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>(
            const_cast<BF16*>(state_smem),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            Tensor g_final = tma_store.get_tma_tensor(make_shape(H, D, D));
            auto state_off = g_final.layout()(head_idx, 0, 0);
            Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_final.layout())));
            Tensor s_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                TMAFP32StateSmemLayout{});

            auto cta_tma_store_state = tma_store.get_slice(Int<0>{});
            cute::copy(
                tma_store,
                cta_tma_store_state.partition_S(s_fp32),
                cta_tma_store_state.partition_D(g_final_tile)
            );
            tma_store_arrive();
            tma_store_wait<0>();
        }
        __syncthreads();
    };

    store_state(shared_storage.state_d.begin(), tma_store_state_d);
    store_state(shared_storage.state_m.begin(), tma_store_state_m);
}

// ==================== launch_fwd_md ====================
// Kernel 1 verbatim from fwd_launch.cu, then the dual-state kernel 2.
template <int D>
void launch_fwd_md(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    float scale,
    float* state_d_ptr,
    float* state_m_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kInputStages = 3;
    constexpr int CHUNK = 16;

    using K1L = K1Layouts<D, CHUNK>;
    using K2L = K2Layouts<D, CHUNK>;
    using WS = WorkspaceSizes<CHUNK, D>;

    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAVOLayout = typename K1L::TMAVOLayout;
    using TMALMLayout = typename K1L::TMALMLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;
    using TMAFP32StateSmemLayout = typename K2L::TMAFP32StateSmemLayout;

    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(H, D, D), LayoutRight{});

    Tensor m_q = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_v = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);

    int64_t n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    BF16*  ws_kd  = reinterpret_cast<BF16*>(ws);
    BF16*  ws_qd  = reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed);
    BF16*  ws_kr  = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
    float* ws_gt  = reinterpret_cast<float*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    BF16*  ws_inv = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal));
    BF16*  ws_mqk = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV));

    auto ws_kd_gmem_layout = make_layout(make_shape(int(n_ht), CHUNK, D), LayoutRight{});
    auto ws_qd_gmem_layout = ws_kd_gmem_layout;
    auto ws_kr_gmem_layout = ws_kd_gmem_layout;
    auto ws_gt_gmem_layout = make_layout(make_shape(int(n_ht), D), LayoutRight{});
    auto ws_lm_gmem_layout = make_layout(make_shape(int(n_ht), CHUNK, CHUNK), LayoutRight{});

    Tensor m_ws_kd  = make_tensor(make_gmem_ptr(ws_kd), ws_kd_gmem_layout);
    Tensor m_ws_qd  = make_tensor(make_gmem_ptr(ws_qd), ws_qd_gmem_layout);
    Tensor m_ws_kr  = make_tensor(make_gmem_ptr(ws_kr), ws_kr_gmem_layout);
    Tensor m_ws_gt  = make_tensor(make_gmem_ptr(ws_gt), ws_gt_gmem_layout);
    Tensor m_ws_inv = make_tensor(make_gmem_ptr(ws_inv), ws_lm_gmem_layout);
    Tensor m_ws_mqk = make_tensor(make_gmem_ptr(ws_mqk), ws_lm_gmem_layout);

    // Kernel 1 TMA descriptors (loads: q,k,beta,g,dt_bias; stores: workspace).
    auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
    auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
    auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

    auto tma_store_ws_kd  = make_tma_copy(SM90_TMA_STORE{}, m_ws_kd, TMAVOLayout{});
    auto tma_store_ws_qd  = make_tma_copy(SM90_TMA_STORE{}, m_ws_qd, TMAVOLayout{});
    auto tma_store_ws_kr  = make_tma_copy(SM90_TMA_STORE{}, m_ws_kr, TMAVOLayout{});
    auto tma_store_ws_gt  = make_tma_copy(SM90_TMA_STORE{}, m_ws_gt, TMAGTotalSmemLayout{});
    auto tma_store_ws_inv = make_tma_copy(SM90_TMA_STORE{}, m_ws_inv, TMALMLayout{});
    auto tma_store_ws_mqk = make_tma_copy(SM90_TMA_STORE{}, m_ws_mqk, TMALMLayout{});

    // Kernel 2-MD TMA descriptors.
    auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, typename K2L::TMAVOLayout{});
    auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, typename K2L::TMABetaSmemLayout{});
    auto tma_load_ws_kd  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_kd, typename K2L::TMAVOLayout{});
    auto tma_load_ws_kr  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_kr, typename K2L::TMAVOLayout{});
    auto tma_load_ws_gt  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_gt, typename K2L::TMAGTotalSmemLayout{});
    auto tma_load_ws_inv = make_tma_copy(SM90_TMA_LOAD{}, m_ws_inv, typename K2L::TMALMLayout{});

    auto m_state_d = make_tensor(make_gmem_ptr(state_d_ptr), state_gmem_layout);
    auto m_state_m = make_tensor(make_gmem_ptr(state_m_ptr), state_gmem_layout);
    auto tma_store_state_d = make_tma_copy(SM90_TMA_STORE{}, m_state_d, TMAFP32StateSmemLayout{});
    auto tma_store_state_m = make_tma_copy(SM90_TMA_STORE{}, m_state_m, TMAFP32StateSmemLayout{});

    // ===== Kernel 1 (prepare), verbatim =====
    {
        constexpr int kK1Threads = 256;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare<
            decltype(tma_load_q), decltype(tma_load_k),
            decltype(tma_load_beta),
            decltype(tma_load_g), decltype(tma_load_dt_bias),
            decltype(tma_store_ws_kd), decltype(tma_store_ws_qd), decltype(tma_store_ws_kr),
            decltype(tma_store_ws_gt), decltype(tma_store_ws_inv), decltype(tma_store_ws_mqk),
            CHUNK, D, kK1Threads, /*IsVarlen=*/false
        >;

        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1Threads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            tma_load_q, tma_load_k, tma_load_beta,
            tma_load_g, tma_load_dt_bias,
            tma_store_ws_kd, tma_store_ws_qd, tma_store_ws_kr,
            tma_store_ws_gt, tma_store_ws_inv, tma_store_ws_mqk,
            scale, T_total, H, /*N=*/1, /*cu_seqlens=*/nullptr, total_tiles,
            A_log_ptr, gate_scale, /*ws_tile_prefix=*/nullptr
        );
    }

    // ===== Kernel 2-MD (dual-state recurrence) =====
    {
        constexpr int kK2Threads = 128 + 32;
        using SharedStorageK2T = SharedStorageK2MD<K2L, kInputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _k3_flash_kda_fwd_md_recurrence<
            decltype(tma_load_v), decltype(tma_load_beta2),
            decltype(tma_load_ws_kd), decltype(tma_load_ws_kr),
            decltype(tma_load_ws_gt), decltype(tma_load_ws_inv),
            decltype(tma_store_state_d),
            decltype(tma_store_state_m),
            CHUNK, D, kInputStages, kK2Threads
        >;

        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        dim3 grid_k2(1, H);
        dim3 block_k2(kK2Threads);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_ws_kd, tma_load_ws_kr,
            tma_load_ws_gt, tma_load_ws_inv,
            tma_store_state_d, tma_store_state_m,
            T_total, H, total_tiles
        );
    }
}
