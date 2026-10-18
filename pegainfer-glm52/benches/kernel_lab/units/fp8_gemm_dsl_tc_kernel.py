"""CuTe DSL FP8 groupwise blockscaled GEMM kernel for sm_103/sm_100a (tcgen05).

Same semantics as the production CUTLASS wide route:

    D[m, n] = bf16( sum_kb  blockacc[m, n, kb] * SFA[m, kb] * SFB[n//128, kb] )

where blockacc[..., kb] = A_tile[m, kb*128:(kb+1)*128] x B[n, same]^T is
accumulated by tcgen05.mma in TMEM inside one 128-K block only (first MMA of
the block issues with ACCUMULATE=False), then scaled and folded into the f32
total accumulator held in registers (software blockscale, bit-isomorphic to
CUTLASS Sm100BlockwiseScaleConfig<1,128,128,K,K>).

- A (act):    e4m3 [m, k] row-major (K-major)     SFA: f32 [m, k/128]
- B (weight): e4m3 [n, k] row-major (K-major, TN) SFB: f32 [n/128, k/128]
- D (out):    bf16 [m, n] row-major
- any m (grid.x tiles the rows axis in 64-row MMA tiles; TMA zero-fills OOB
  rows of the last partial tile on load, epilogue stores are row-predicated),
  k % 128 == 0, n % 128 == 0.

Structure (stripped-down version of the vendored CUTLASS CuTeDSL example
`blackwell/blockwise_gemm/blockwise_gemm.py` — persistent tile scheduler,
2-SM MMA, clusters/multicast, scale smem staging and TMA-store epilogue all
removed; the shapes here are small and single-wave):

- 160 threads: warps 0-3 = acc-update/epilogue, warp 4 = TMA + MMA issue.
- TMA loads of A/B tiles (64x128 / 128x128 e4m3) into a multi-stage smem
  pipeline (`PipelineTmaUmma`), tcgen05.mma (M=64/N=128/K=128 tile, 1-CTA,
  cta_group ONE) into a double-buffered TMEM block accumulator
  (`PipelineUmmaAsync`, 2 stages), so block MMA k+1 overlaps the tmem->reg
  fold of block k.
- Scales: each CTA preloads its SFA/SFB K-strip into smem once, then the
  per-k-tile fold reads smem instead of re-issuing guarded LDGs.
- Epilogue: registers -> predicated SIMT stores straight to gmem.

Split-K (`split_k > 1`): grid.z runs `split_k` CTAs per (m, n) tile; CTA z
folds the K range [z*kb_local, (z+1)*kb_local) (kb_local = (k/128)/split_k,
required integral) with its own scale sub-strip, and stores its f32 partial
to a (m, n, split_k) tensor (natural: the existing RestL mode of mD carries
the split id — the L mode of A/B/SFA/SFB stays fixed at 0). A second tiny
elementwise kernel (`_SplitKReduce`) then reduces the f32 partials to the
bf16 output on the same stream. Purpose: raise CTA occupancy for
CTA-starved shapes (gate_up/down at 32/48 CTAs vs 148 SMs); the main
kernel's per-CTA TMA issue rate is the limiter there, so more CTAs with
shorter K strips buy bandwidth the epilogue vectorization can't.

Imported lazily by `units/fp8_gemm_dsl_tc.py`; imports the DSL at module
level on purpose (never import on CPU-only boxes).
"""
from __future__ import annotations

import cuda.bindings.driver as cuda_drv

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils

KB_TILE = 128  # scale granularity along K — MMA tile K must equal this.


class Fp8BlockwiseGemmTcgen05:
    def __init__(
        self,
        block_n: int = 128,
        split_k: int = 1,
        num_ab_stage: int = 0,
        num_acc_stage: int = 2,
    ):
        self.ab_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.sfa_dtype = cutlass.Float32
        self.mma_tiler_mn = (64, block_n)
        self.split_k = split_k
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.occupancy = 1
        # warps 0-3: acc update + epilogue; warp 4: TMA + MMA issue.
        self.mma_warp_id = 4
        self.num_acc_warps = 4
        self.threads_per_cta = 32 * (self.num_acc_warps + 1)
        self.num_acc_stage = num_acc_stage
        self.num_ab_stage = num_ab_stage  # 0 = auto from smem capacity

    def _setup_attributes(self, n: int, k: int):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),  # A K-major
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),  # B K-major
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            mma_inst_shape_k * (KB_TILE // mma_inst_shape_k),
        )
        assert self.mma_tiler[2] == KB_TILE
        # N tile 64 or 128; either way the CTA lies inside one 128-N scale
        # block (SFB granularity), indexed separately below.
        assert self.mma_tiler[1] in (64, 128)
        assert n % self.mma_tiler[1] == 0 and k % KB_TILE == 0
        # Split-K partitions the 128-K-block count across grid.z CTAs.
        assert self.split_k >= 1 and (k // KB_TILE) % self.split_k == 0
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            False,
            utils.LayoutEnum.ROW_MAJOR,  # C n-major
            self.c_dtype,
        )
        self.a_smem_layout_staged = None  # filled below once stage count is set
        a_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, 1
        )
        b_stage_one = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, 1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            self.ab_dtype, a_stage_one
        ) + cute.size_in_bytes(self.ab_dtype, b_stage_one)
        if self.num_ab_stage == 0:
            smem_capacity = utils.get_smem_capacity_in_bytes()
            kb_local = (k // KB_TILE) // self.split_k
            scale_smem_bytes = (self.mma_tiler_mn[0] * kb_local + kb_local + 16) * 4
            mbar_reserve = 2048
            self.num_ab_stage = (
                smem_capacity - mbar_reserve - scale_smem_bytes
            ) // ab_bytes_per_stage
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)))
            + cute.size_in_bytes(self.ab_dtype, cute.slice_(self.b_smem_layout_staged, (None, None, None, 0)))
        ) * cute.size(tiled_mma.thr_id.shape)
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,      # e4m3 (m, k, 1)  k-major
        mB: cute.Tensor,      # e4m3 (n, k, 1)  k-major
        mSFA: cute.Tensor,    # f32  (m, kb, 1)
        mSFB: cute.Tensor,    # f32  (n/128, kb, 1)
        mD: cute.Tensor,      # out  (m, n, L)  n-major — bf16 with L=1 when
                              # split_k == 1, else f32 partials with L == split_k
        stream: cuda_drv.CUstream,
    ):
        # D dtype comes from the tensor: bf16 final output, or f32 partials
        # under split-K (epilogue skips the bf16 narrowing in that case).
        self.c_dtype = mD.element_type
        tiled_mma = self._setup_attributes(
            cute.size(mB, mode=[0]), cute.size(mA, mode=[1])
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        grid = (
            cute.ceil_div(cute.size(mD, mode=[0]), self.mma_tiler[0]),
            cute.ceil_div(cute.size(mD, mode=[1]), self.mma_tiler[1]),
            self.split_k,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mSFA,
            mSFB,
            mD,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # grid.z is the split-K id (1 when split_k == 1); A/B/SFA/SFB L is
        # always 0, but mD's RestL mode carries the split id for partials.
        split_id = cutlass.Int32(bidz)
        mma_tile_coord_mnl = (bidx, bidy, split_id)
        # 128-N scale block index of this CTA (block_n in {64, 128} divides a
        # single SFB granularity block).
        nb = (bidy * self.mma_tiler[1]) // 128
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_acc_warps
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )

        # Cluster arrive after barrier init (no-op for 1x1, kept for parity).
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Scale staging: SFA/SFB are tiny (<=64 x kb_local + kb_local f32);
        # every CTA preloads only its own split-K strip once, so the
        # per-k-tile fold reads scales from smem instead of re-issuing
        # guarded LDGs each tile.
        kb_local = cute.size(mSFA_mkl, mode=[1]) // self.split_k
        sSFA = smem.allocate_tensor(
            element_type=self.sfa_dtype,
            layout=cute.make_layout((self.mma_tiler[0], kb_local), stride=(kb_local, 1)),
            byte_alignment=16,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sfa_dtype,
            layout=cute.make_layout((kb_local,)),
            byte_alignment=16,
        )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # This CTA's K range within the tile sweep: [kb_base, kb_base + k_tile_cnt).
        k_tile_cnt = cute.size(gA_mkl, mode=[3]) // self.split_k
        kb_base = split_id * k_tile_cnt

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE) / ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # Slice to this CTA's mma tile (m is always 0, one M tile). A/B batch
        # mode L is fixed at 0 (bidz is the split-K id, carried on mD only).
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, 0)]

        #
        # Specialized TMA load + MMA warp
        #
        if warp_idx == self.mma_warp_id:
            prefetch_k_tile_cnt = cutlass.min(self.num_ab_stage - 2, k_tile_cnt)
            for k_tile_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, kb_base + k_tile_idx)],
                    tAsA[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, kb_base + k_tile_idx)],
                    tBsB[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                )

            peek_ab_full_status = ab_consumer.try_wait()
            peek_ab_empty_status = ab_producer.try_acquire()
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                acc_producer_state
            )

            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                if k_tile < k_tile_cnt - prefetch_k_tile_cnt:
                    producer_handle = ab_producer.acquire_and_advance(
                        peek_ab_empty_status
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, kb_base + producer_handle.count)],
                        tAsA[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, kb_base + producer_handle.count)],
                        tBsB[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                    )

                # Wait block-acc tmem stage free (released by acc warps).
                acc_pipeline.producer_acquire(
                    acc_producer_state, peek_acc_empty_status
                )
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Fresh block accumulator every 128-K tile.
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                num_kblocks = cute.size(tCrA, mode=[2])
                for kblk_idx in cutlass.range_constexpr(num_kblocks):
                    kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kblk_crd],
                        tCrB[kblk_crd],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                consumer_handle.release()

                acc_pipeline.producer_commit(acc_producer_state)

                if k_tile + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()
                if k_tile + 1 < k_tile_cnt:
                    peek_ab_full_status = ab_consumer.try_wait()
                acc_producer_state.advance()
                if acc_producer_state.count < k_tile_cnt:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                        acc_producer_state
                    )

            acc_pipeline.producer_tail(acc_producer_state)
            ab_producer.tail()

        #
        # Acc-update + epilogue warps
        #
        if warp_idx < self.num_acc_warps:
            epi_tidx = tidx  # 0..127

            # tmem -> rmem copy setup (M=64: 16dp256b atom family).
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
            gC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, None, None, None)], epi_tile
            )
            cC_epi = cute.flat_divide(
                cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]), epi_tile
            )
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
            tTR_gC_all = thr_copy_t2r.partition_D(gC_epi)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)
            tTR_rAcc = [
                cute.make_rmem_tensor(
                    tTR_gC_all[(None, None, None, 0, 0, 0, 0, 0)].shape,
                    self.acc_dtype,
                )
                for _ in range(cute.size(tTR_gC_all.shape, mode=[3]) * cute.size(tTR_gC_all.shape, mode=[4]))
            ]
            # Persistent f32 total accumulator for the whole K sweep.
            tTR_rAcc_final = cute.make_rmem_tensor(
                tTR_gC_all[(None, None, None, None, None, 0, 0, 0)].shape,
                self.acc_dtype,
            )
            tTR_rAcc_final.fill(0.0)

            num_epi_m = cute.size(tTR_rAcc_final.shape, mode=[3])
            num_epi_n = cute.size(tTR_rAcc_final.shape, mode=[4])
            frag_len = cute.size(tTR_rAcc[0])

            m_total = cutlass.Int32(cute.size(mC_mnl, mode=[0]))
            # Rows this CTA's M tile starts at; smem/identity coords are
            # tile-local, gmem reads and row predicates need the offset.
            m_base = cutlass.Int32(bidx) * self.mma_tiler[0]

            #
            # Preload this split's SFA/SFB strip into smem (once per CTA);
            # global gmem reads are offset by kb_base.
            #
            kb_local_cnt = cute.size(sSFB)
            scale_threads = 32 * self.num_acc_warps
            for i in cutlass.range_constexpr(
                (self.mma_tiler[0] * kb_local_cnt + scale_threads - 1) // scale_threads
            ):
                idx = tidx + i * scale_threads
                if idx < self.mma_tiler[0] * kb_local_cnt:
                    row = idx // kb_local_cnt
                    kc = idx % kb_local_cnt
                    val = cutlass.Float32(0.0)
                    if cute.elem_less(m_base + row, m_total):
                        val = mSFA_mkl[m_base + row, kb_base + kc, 0]
                    sSFA[row, kc] = val
            if tidx < kb_local_cnt:
                sSFB[tidx] = mSFB_nkl[nb, kb_base + tidx, 0]
            scale_barrier = pipeline.NamedBarrier(
                barrier_id=2, num_threads=scale_threads
            )
            scale_barrier.arrive_and_wait()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                acc_consumer_state
            )

            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                kb = cutlass.Int32(k_tile)
                sfb = sSFB[kb]  # uniform per (N tile, k tile)

                acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)
                stage = acc_consumer_state.index

                # Issue every subtile's tmem->rmem load back-to-back (distinct
                # register destinations), then fold: one tcgen05.wait_ld covers
                # all loads instead of one round trip per subtile.
                for epi_m in cutlass.range_constexpr(num_epi_m):
                    for epi_n in cutlass.range_constexpr(num_epi_n):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, epi_m, epi_n, stage)],
                            tTR_rAcc[epi_m * num_epi_n + epi_n],
                        )
                for epi_m in cutlass.range_constexpr(num_epi_m):
                    for epi_n in cutlass.range_constexpr(num_epi_n):
                        tTR_rAcc_sub = tTR_rAcc[epi_m * num_epi_n + epi_n]
                        tTR_cC_sub = tTR_cC[(None, None, None, epi_m, epi_n)]
                        tTR_final_sub = tTR_rAcc_final[(None, None, None, epi_m, epi_n)]
                        for i in cutlass.range_constexpr(frag_len):
                            sfa = sSFA[tTR_cC_sub[i][0], kb]
                            tTR_final_sub[i] = (
                                tTR_final_sub[i] + tTR_rAcc_sub[i] * (sfa * sfb)
                            )

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(
                        acc_consumer_state
                    )

            # Epilogue: convert to mD dtype (bf16 final, or f32 partial under
            # split-K) + row-predicated SIMT stores to gmem. mD's RestL mode
            # is indexed by the split id (0 when split_k == 1).
            n_tile = mma_tile_coord_mnl[1]
            for epi_m in cutlass.range_constexpr(num_epi_m):
                for epi_n in cutlass.range_constexpr(num_epi_n):
                    tTR_final_sub = tTR_rAcc_final[(None, None, None, epi_m, epi_n)]
                    tTR_rC = cute.make_rmem_tensor(tTR_final_sub.shape, self.c_dtype)
                    tTR_rC.store(tTR_final_sub.load().to(self.c_dtype))
                    tTR_cC_sub = tTR_cC[(None, None, None, epi_m, epi_n)]
                    tTR_gC_sub = tTR_gC_all[
                        (None, None, None, epi_m, epi_n, mma_tile_coord_mnl[0], n_tile, bidz)
                    ]
                    for i in cutlass.range_constexpr(frag_len):
                        if cute.elem_less(m_base + tTR_cC_sub[i][0], m_total):
                            tTR_gC_sub[i] = tTR_rC[i]

        # Whole-CTA sync before tensor memory dealloc.
        cute.arch.barrier()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class _SplitKReduce:
    """out[r, c] = out_dtype( sum_s partials[r, c, s] ) — split-K epilogue.

    One thread folds `vec` contiguous N columns of one row; grid =
    (ceil(n / vec / threads), rows, 1). Traffic is split_k*rows*n f32 reads
    plus rows*n bf16 writes (<= 8 MB at our shapes) — scalar LDG/STG suffice.
    """

    def __init__(self, split_k: int, vec: int = 8, threads: int = 128):
        self.split_k = split_k
        self.vec = vec
        self.threads = threads
        self.c_dtype = cutlass.BFloat16  # overwritten from mOut at trace time

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,      # f32 (rows, n, split_k) — main kernel partials
        mOut: cute.Tensor,    # (rows, n, 1)
        stream: cuda_drv.CUstream,
    ):
        self.c_dtype = mOut.element_type
        n_chunks = cute.size(mOut, mode=[1]) // self.vec
        grid = (
            cute.ceil_div(n_chunks, self.threads),
            cute.size(mOut, mode=[0]),
            1,
        )
        self.kernel(mP, mOut).launch(
            grid=grid,
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mP: cute.Tensor, mOut: cute.Tensor):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        c0 = (bidx * self.threads + tidx) * self.vec
        if c0 < cute.size(mOut, mode=[1]):
            row = cutlass.Int32(bidy)
            acc = cute.make_rmem_tensor(cute.make_layout(self.vec), cutlass.Float32)
            acc.fill(0.0)
            for s in cutlass.range_constexpr(self.split_k):
                for i in cutlass.range_constexpr(self.vec):
                    acc[i] = acc[i] + mP[row, c0 + i, s]
            out_v = cute.make_rmem_tensor(cute.make_layout(self.vec), self.c_dtype)
            out_v.store(acc.load().to(self.c_dtype))
            for i in cutlass.range_constexpr(self.vec):
                mOut[row, c0 + i, 0] = out_v[i]


_CACHE: dict = {}       # shape key -> JitExecutor (main GEMM)
_RED_CACHE: dict = {}   # (rows, n, split_k) -> JitExecutor (split-K reduce)
_CALL_CACHE: dict = {}  # (key, ptr signature, stream) -> launch entries
_PARTIALS: dict = {}    # (rows, n, split_k) -> torch f32 (split_k, rows, n)


def run_fp8_blockwise_gemm_tc(
    rows: int,
    n: int,
    k: int,
    act_q,      # torch uint8 [rows, k] (e4m3 bytes)
    act_s,      # torch f32 [rows, k/128]
    weight,     # torch uint8 [n, k]
    w_scales,   # torch f32 [n/128, k/128]
    out,        # torch bf16 [rows, n]
    stream_int: int,
    block_n: int = 128,
    split_k: int = 1,
    num_ab_stage: int = 0,
    num_acc_stage: int = 4,
) -> None:
    """Compile-once-per-shape + launch on `stream_int` (raw cudaStream_t).

    from_dlpack wrapping costs ~70us CPU per call, so wrapped views are
    cached on
    (shape key, data_ptrs, stream). `block_n` picks the MMA N tile (128 = full
    SFB block; 64 doubles CTA count for CTA-starved shapes). `split_k` > 1
    runs grid.z CTAs per (m, n) tile on disjoint K ranges, writes f32
    partials into a cached zeroed (split_k, rows, n) buffer (kept alive in
    `_PARTIALS` so its wrapped view is pointer-stable), then launches
    `_SplitKReduce` on the same stream — stream order is the dependency."""
    import torch as _torch

    key = (rows, n, k, block_n, split_k, num_ab_stage, num_acc_stage)

    def _wrap(t, dtype):
        # Append a size-1 batch mode L: the tcgen05/TMA machinery is built for
        # (M, K, L) ranked tensors.
        return from_dlpack(t.view(dtype).unsqueeze(-1), assumed_align=16)

    sig = (
        key,
        act_q.data_ptr(), act_s.data_ptr(), weight.data_ptr(),
        w_scales.data_ptr(), out.data_ptr(), stream_int,
    )
    entry = _CALL_CACHE.get(sig)
    if entry is None:
        mA = _wrap(act_q, _torch.float8_e4m3fn)
        mB = _wrap(weight, _torch.float8_e4m3fn)
        mSFA = _wrap(act_s, _torch.float32)
        mSFB = _wrap(w_scales, _torch.float32)
        stream = cuda_drv.CUstream(stream_int)
        compiled = _CACHE.get(key)
        red_exec = None
        red_args = None
        if split_k > 1:
            pkey = (rows, n, split_k)
            partials = _PARTIALS.get(pkey)
            if partials is None:
                # Zeroed so rows the predicated main kernel skips contribute
                # exact 0.0 to the reduction (deterministic dead rows).
                partials = _torch.zeros(
                    (split_k, rows, n), dtype=_torch.float32, device=out.device
                )
                _PARTIALS[pkey] = partials
            # (rows, n, split_k) n-major view: RestL carries the split id.
            mD = from_dlpack(partials.permute(1, 2, 0), assumed_align=16)
            mOut = _wrap(out, _torch.bfloat16)
            red_exec = _RED_CACHE.get(pkey)
            if red_exec is None:
                red_exec = cute.compile(_SplitKReduce(split_k), mD, mOut, stream)
                _RED_CACHE[pkey] = red_exec
            red_args = (mD, mOut, stream)
        else:
            mD = _wrap(out, _torch.bfloat16)
        if compiled is None:
            gemm = Fp8BlockwiseGemmTcgen05(
                block_n=block_n, split_k=split_k, num_ab_stage=num_ab_stage,
                num_acc_stage=num_acc_stage,
            )
            compiled = cute.compile(gemm, mA, mB, mSFA, mSFB, mD, stream)
            _CACHE[key] = compiled
        entry = (compiled, (mA, mB, mSFA, mSFB, mD, stream), red_exec, red_args)
        if len(_CALL_CACHE) > 512:  # unbounded-growth guard
            _CALL_CACHE.clear()
        _CALL_CACHE[sig] = entry
    compiled, args, red_exec, red_args = entry
    compiled(*args)
    if red_exec is not None:
        red_exec(*red_args)
