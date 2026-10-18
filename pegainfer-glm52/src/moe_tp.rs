//! GLM5.2 tensor-parallel MoE for **TP4 prefill-only**: per-rank slices of
//! all 257 experts (shared at bank index 256) plus NCCL bf16 all-reduce.
//! Decode-side LL packet MoE/attention kernels are gone.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_FP8_GROUPED_GEMM_WORKSPACE_BYTES;
use pegainfer_kernels::ops::GLM52_TP_BANK_EXPERTS;
use pegainfer_kernels::ops::GLM52_TP_HIDDEN;
use pegainfer_kernels::ops::GLM52_TP_MAX_RANKS;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::Glm52TpTopology;
use pegainfer_kernels::ops::glm52_fp8_grouped_gemm_sm100_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_launch;
use pegainfer_kernels::ops::glm52_prefill_moe_combine_launch;
use pegainfer_kernels::ops::glm52_prefill_moe_gather_fp8_launch;
use pegainfer_kernels::ops::glm52_prefill_moe_route_launch;
use pegainfer_kernels::ops::glm52_silu_and_mul_bf16_launch;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_EXPERT_INTERMEDIATE as INTERMEDIATE;
use crate::fp8::Glm52Fp8GemmScratch;
use crate::fp8::fp8_linear_large_m_bank_into;
use crate::moe_decode::EXPERTS;
use crate::moe_decode::Glm52MoeRouterWeights;
use crate::moe_decode::Glm52RouterScratch;
use crate::moe_decode::QUANT_GROUP;
use crate::moe_decode::TOPK;
use crate::moe_decode::W2_K;
use crate::moe_decode::W2_N;
use crate::moe_decode::W2_SCALE_COLS;
use crate::moe_decode::W2_SCALE_ROWS;
use crate::moe_decode::run_router_rows_into;
use crate::weights::Glm52WeightManifest;
use crate::weights::expected_tensor_contract;
use crate::weights::mmap_file;
use crate::weights::retype_owned;

const H: usize = GLM52_TP_HIDDEN;
const RANKS: usize = GLM52_TP_MAX_RANKS;

const BANK: usize = GLM52_TP_BANK_EXPERTS;
#[cfg(test)]
const SLICE_ROWS: usize = Glm52TpTopology::Tp4.slice_rows();
#[cfg(test)]
const SLICE_I: usize = Glm52TpTopology::Tp4.slice_i();

/// One pilot layer's TP slice bank: this rank's intermediate rows of all 257
/// experts, in the layout the cooperative kernel consumes.
pub(crate) struct Glm52MoeTpSliceBank {
    tp_ranks: usize,
    slice_i: usize,
    slice_rows: usize,
    w13: CudaSlice<u8>,        // fp8 [257, 512, 6144]
    w13_scale: CudaSlice<f32>, // f32 [257, 4, 48]
    w2: CudaSlice<u8>,         // fp8 [257, 6144, 256]
    w2_scale: CudaSlice<f32>,  // f32 [257, 48, 2]
}

/// Slice one expert's checkpoint tensors into the rank-r staging bank.
/// `bank_idx` is the destination expert slot (routed id, or 256 for shared).
struct SliceStaging {
    rank: usize,
    tp_ranks: usize,
    slice_i: usize,
    slice_rows: usize,
    w13: Vec<u8>,
    w13_scale: Vec<u8>,
    w2: Vec<u8>,
    w2_scale: Vec<u8>,
}

/// Projection kind for one checkpoint tensor loaded into a TP slice bank.
#[derive(Clone, Copy)]
enum SliceKind {
    Gate,
    Up,
    Down,
    GateScale,
    UpScale,
    DownScale,
}

impl SliceStaging {
    fn new(rank: usize, tp_ranks: usize) -> Result<Self> {
        ensure!(
            tp_ranks > 0 && INTERMEDIATE.is_multiple_of(tp_ranks),
            "GLM5.2 TP slice count {tp_ranks} must divide expert intermediate {INTERMEDIATE}"
        );
        ensure!(
            rank < tp_ranks,
            "GLM5.2 TP slice rank {rank} out of range for {tp_ranks} ranks"
        );
        let slice_i = INTERMEDIATE / tp_ranks;
        let slice_rows = 2 * slice_i;
        ensure!(
            slice_i.is_multiple_of(QUANT_GROUP) && slice_rows.is_multiple_of(QUANT_GROUP),
            "GLM5.2 TP slice geometry must align to FP8 quant group {QUANT_GROUP}: \
             slice_i={slice_i}, slice_rows={slice_rows}"
        );
        Ok(Self {
            rank,
            tp_ranks,
            slice_i,
            slice_rows,
            w13: vec![0u8; BANK * slice_rows * H],
            w13_scale: vec![0u8; BANK * (slice_rows / QUANT_GROUP) * (H / QUANT_GROUP) * 4],
            w2: vec![0u8; BANK * H * slice_i],
            w2_scale: vec![0u8; BANK * (H / QUANT_GROUP) * (slice_i / QUANT_GROUP) * 4],
        })
    }

    /// gate/up [2048, 6144]: rows r*256..(r+1)*256 land at slice rows 0..256
    /// (gate) / 256..512 (up) — one contiguous copy each.
    fn put_w13_weight(&mut self, bank_idx: usize, is_up: bool, src: &[u8]) {
        debug_assert_eq!(src.len(), INTERMEDIATE * H);
        let rows = self.slice_i; // rows per projection per rank
        let src_off = self.rank * rows * H;
        let dst_off = bank_idx * self.slice_rows * H + if is_up { self.slice_i * H } else { 0 };
        self.w13[dst_off..dst_off + rows * H].copy_from_slice(&src[src_off..src_off + rows * H]);
    }

    /// gate/up scale f32 [16, 48]: row blocks 2r..2r+2 land at slice blocks
    /// 0..2 (gate) / 2..4 (up).
    fn put_w13_scale(&mut self, bank_idx: usize, is_up: bool, src: &[u8]) {
        debug_assert_eq!(
            src.len(),
            (INTERMEDIATE / QUANT_GROUP) * (H / QUANT_GROUP) * 4
        );
        let row_bytes = (H / QUANT_GROUP) * 4; // 48 f32
        let blocks = self.slice_i / QUANT_GROUP;
        let src_off = self.rank * blocks * row_bytes;
        let dst_off = (bank_idx * (self.slice_rows / QUANT_GROUP) + if is_up { blocks } else { 0 })
            * row_bytes;
        self.w13_scale[dst_off..dst_off + blocks * row_bytes]
            .copy_from_slice(&src[src_off..src_off + blocks * row_bytes]);
    }

    /// down [6144, 2048]: columns r*256..(r+1)*256 of every row — strided
    /// gather into [6144, 256].
    fn put_w2_weight(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_N * W2_K);
        let dst_base = bank_idx * H * self.slice_i;
        let src_col = self.rank * self.slice_i;
        for row in 0..H {
            let dst = dst_base + row * self.slice_i;
            let src_off = row * W2_K + src_col;
            self.w2[dst..dst + self.slice_i].copy_from_slice(&src[src_off..src_off + self.slice_i]);
        }
    }

    /// down scale f32 [48, 16]: column blocks 2r..2r+2 of every row block.
    fn put_w2_scale(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_SCALE_ROWS * W2_SCALE_COLS * 4);
        let blocks = self.slice_i / QUANT_GROUP;
        let dst_base = bank_idx * W2_SCALE_ROWS * blocks * 4;
        let src_col = self.rank * blocks * 4;
        for row in 0..W2_SCALE_ROWS {
            let dst = dst_base + row * blocks * 4;
            let src_off = row * W2_SCALE_COLS * 4 + src_col;
            self.w2_scale[dst..dst + blocks * 4]
                .copy_from_slice(&src[src_off..src_off + blocks * 4]);
        }
    }

    fn upload(self, ctx: &DeviceContext) -> Result<Glm52MoeTpSliceBank> {
        let htod = |host: &[u8]| -> Result<CudaSlice<u8>> {
            // SAFETY: fully written by the memcpy below before use.
            let mut dst = unsafe { ctx.stream.alloc::<u8>(host.len()) }?;
            ctx.stream.memcpy_htod(host, &mut dst)?;
            Ok(dst)
        };
        Ok(Glm52MoeTpSliceBank {
            tp_ranks: self.tp_ranks,
            slice_i: self.slice_i,
            slice_rows: self.slice_rows,
            w13: htod(&self.w13)?,
            w13_scale: retype_owned::<f32>(&ctx.stream, htod(&self.w13_scale)?)?,
            w2: htod(&self.w2)?,
            w2_scale: retype_owned::<f32>(&ctx.stream, htod(&self.w2_scale)?)?,
        })
    }
}

/// Second-pass load of one tensor-replicated MoE slice bank for `rank`.
/// TP4 uses 1/4-intermediate slices.
pub(crate) fn load_tp_slice_layer(
    ctx: &DeviceContext,
    model_path: &Path,
    manifest: &Glm52WeightManifest,
    rank: usize,
    tp_ranks: usize,
    layer: usize,
) -> Result<Glm52MoeTpSliceBank> {
    ensure!(
        rank < tp_ranks,
        "TP rank {rank} out of range for {tp_ranks}"
    );
    // (name, bank_idx, projection kind) for all 257 experts x 6 tensors.
    let mut wanted: Vec<(String, usize, SliceKind)> = Vec::with_capacity(BANK * 6);
    let prefix = format!("model.layers.{layer}.mlp");
    let push_expert =
        |stem: String, bank_idx: usize, wanted: &mut Vec<(String, usize, SliceKind)>| {
            wanted.push((
                format!("{stem}.gate_proj.weight"),
                bank_idx,
                SliceKind::Gate,
            ));
            wanted.push((format!("{stem}.up_proj.weight"), bank_idx, SliceKind::Up));
            wanted.push((
                format!("{stem}.down_proj.weight"),
                bank_idx,
                SliceKind::Down,
            ));
            wanted.push((
                format!("{stem}.gate_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::GateScale,
            ));
            wanted.push((
                format!("{stem}.up_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::UpScale,
            ));
            wanted.push((
                format!("{stem}.down_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::DownScale,
            ));
        };
    for expert in 0..BANK - 1 {
        push_expert(format!("{prefix}.experts.{expert}"), expert, &mut wanted);
    }
    push_expert(format!("{prefix}.shared_experts"), BANK - 1, &mut wanted);

    let mut by_shard: BTreeMap<String, Vec<(String, usize, SliceKind)>> = BTreeMap::new();
    for (name, bank_idx, kind) in wanted {
        let shard = manifest.shard_for(&name)?.to_owned();
        by_shard
            .entry(shard)
            .or_default()
            .push((name, bank_idx, kind));
    }

    let mut staging = SliceStaging::new(rank, tp_ranks)?;
    let mut placed = 0usize;
    for (shard, tensors) in by_shard {
        let path = model_path.join(&shard);
        let mmap = mmap_file(&path)?;
        let safetensors = safetensors::SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for (name, bank_idx, kind) in tensors {
            let view = safetensors
                .tensor(&name)
                .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
            let contract = expected_tensor_contract(&name)?;
            ensure!(
                view.dtype() == contract.dtype && view.shape() == contract.shape.as_slice(),
                "GLM5.2 TP tensor {name} contract mismatch: got {:?} {:?}, expected {:?} {:?}",
                view.dtype(),
                view.shape(),
                contract.dtype,
                contract.shape
            );
            let data = view.data();
            match kind {
                SliceKind::Gate => staging.put_w13_weight(bank_idx, false, data),
                SliceKind::Up => staging.put_w13_weight(bank_idx, true, data),
                SliceKind::Down => staging.put_w2_weight(bank_idx, data),
                SliceKind::GateScale => staging.put_w13_scale(bank_idx, false, data),
                SliceKind::UpScale => staging.put_w13_scale(bank_idx, true, data),
                SliceKind::DownScale => staging.put_w2_scale(bank_idx, data),
            }
            placed += 1;
        }
    }
    ensure!(
        placed == BANK * 6,
        "GLM5.2 TP layer {layer} slice load placed {placed} tensors, expected {}",
        BANK * 6
    );
    staging.upload(ctx)
}

/// Max rows per prefill MoE sub-block. Bounds routed gather/GEMM scratch
/// buffers (`block * TOPK` rows) while keeping per-layer expert weight
/// re-reads to `ceil(chunk / block)` passes.
const GLM52_PREFILL_MOE_BLOCK_ROWS: usize = 8192;

/// Chunk-scale TP prefill MoE scratch: the router runs over the whole
/// chunk once; expert compute walks the chunk in blocks of this size.
pub(crate) struct Glm52MoeTpPrefillScratch {
    chunk_rows: usize,
    block_rows: usize,
    router: Glm52RouterScratch,
    act_fp8: CudaSlice<u8>,
    act_scale: CudaSlice<f32>,
    expert_counts: CudaSlice<i32>,
    m_indptr: CudaSlice<i32>,
    gather_rows: CudaSlice<i32>,
    route_slot: CudaSlice<i32>,
    routed_fp8: CudaSlice<u8>,
    routed_scale: CudaSlice<f32>,
    gate_up: CudaSlice<bf16>,
    silu: CudaSlice<bf16>,
    silu_fp8: CudaSlice<u8>,
    silu_scale: CudaSlice<f32>,
    w2_out: CudaSlice<bf16>,
    shared_gate_up: CudaSlice<bf16>,
    shared_silu: CudaSlice<bf16>,
    shared_out: CudaSlice<bf16>,
    grouped_workspace: CudaSlice<u8>,
    shared_gemm: Glm52Fp8GemmScratch,
}

impl Glm52MoeTpPrefillScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        topology: Glm52TpTopology,
        chunk_rows: usize,
    ) -> Result<Self> {
        ensure!(chunk_rows > 0, "GLM5.2 TP prefill MoE needs positive rows");
        let chunk = chunk_rows.next_multiple_of(4);
        let block = chunk.min(GLM52_PREFILL_MOE_BLOCK_ROWS);
        let routes = block * TOPK;
        let slice_rows = topology.slice_rows();
        let slice_i = topology.slice_i();
        Ok(Self {
            chunk_rows: chunk,
            block_rows: block,
            router: Glm52RouterScratch::new(ctx, chunk)?,
            act_fp8: ctx.stream.alloc_zeros::<u8>(chunk * H)?,
            act_scale: ctx.stream.alloc_zeros::<f32>(chunk * (H / QUANT_GROUP))?,
            expert_counts: ctx.stream.alloc_zeros::<i32>(EXPERTS)?,
            m_indptr: ctx.stream.alloc_zeros::<i32>(EXPERTS + 1)?,
            gather_rows: ctx.stream.alloc_zeros::<i32>(routes)?,
            route_slot: ctx.stream.alloc_zeros::<i32>(routes)?,
            routed_fp8: ctx.stream.alloc_zeros::<u8>(routes * H)?,
            routed_scale: ctx.stream.alloc_zeros::<f32>(routes * (H / QUANT_GROUP))?,
            gate_up: ctx.stream.alloc_zeros::<bf16>(routes * slice_rows)?,
            silu: ctx.stream.alloc_zeros::<bf16>(routes * slice_i)?,
            silu_fp8: ctx.stream.alloc_zeros::<u8>(routes * slice_i)?,
            silu_scale: ctx
                .stream
                .alloc_zeros::<f32>(routes * slice_i.div_ceil(QUANT_GROUP))?,
            w2_out: ctx.stream.alloc_zeros::<bf16>(routes * H)?,
            shared_gate_up: ctx.stream.alloc_zeros::<bf16>(block * slice_rows)?,
            shared_silu: ctx.stream.alloc_zeros::<bf16>(block * slice_i)?,
            shared_out: ctx.stream.alloc_zeros::<bf16>(block * H)?,
            grouped_workspace: ctx
                .stream
                .alloc_zeros::<u8>(GLM52_FP8_GROUPED_GEMM_WORKSPACE_BYTES)?,
            shared_gemm: Glm52Fp8GemmScratch::new(ctx, block, H)?,
        })
    }

    /// Chunk-scale MoE forward: `output[..active * H]` receives this rank\'s
    /// UNREDUCED partial (shared + routed slice contributions); the caller
    /// all-reduces. Row blocks bound scratch, not correctness.
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        state: &mut Glm52MoeTpState,
        router: &Glm52MoeRouterWeights,
        bank: &Glm52MoeTpSliceBank,
        normed: &CudaSlice<bf16>,
        active: usize,
        output: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let rows = active.next_multiple_of(4);
        ensure!(
            active > 0
                && rows <= self.chunk_rows
                && bank.tp_ranks == state.ranks()
                && bank.slice_i == INTERMEDIATE / state.ranks()
                && normed.len() >= rows * H
                && output.len() >= active * H,
            "GLM5.2 TP prefill MoE shape is invalid"
        );
        run_router_rows_into(ctx, router, normed, active, rows, &mut self.router)?;
        // One quantization of the whole chunk; routed gathers reuse it.
        glm52_fp8_per_token_group_quant_bf16_launch(
            ctx,
            Glm52MoeQuantShape {
                rows,
                width: H,
                group_size: QUANT_GROUP,
            },
            normed,
            &mut self.act_fp8,
            &mut self.act_scale,
        )?;
        let mut start = 0usize;
        while start < active {
            let block = (active - start).min(self.block_rows);
            self.forward_block(ctx, bank, normed, start, block, output)?;
            start += block;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_block(
        &mut self,
        ctx: &DeviceContext,
        bank: &Glm52MoeTpSliceBank,
        normed: &CudaSlice<bf16>,
        start: usize,
        block: usize,
        output: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let routes = block * TOPK;
        let topk_idx = self
            .router
            .route
            .topk_idx
            .slice(start * TOPK..(start + block) * TOPK);
        let topk_weight = self
            .router
            .route
            .topk_weight
            .slice(start * TOPK..(start + block) * TOPK);
        glm52_prefill_moe_route_launch(
            ctx,
            block,
            TOPK,
            EXPERTS,
            &topk_idx,
            &mut self.expert_counts,
            &mut self.m_indptr,
            &mut self.gather_rows,
            &mut self.route_slot,
        )?;
        glm52_prefill_moe_gather_fp8_launch(
            ctx,
            routes,
            H,
            &self.act_fp8.slice(start * H..),
            &self.act_scale.slice(start * (H / QUANT_GROUP)..),
            &self.gather_rows,
            &mut self.routed_fp8,
            &mut self.routed_scale,
        )?;
        glm52_fp8_grouped_gemm_sm100_launch(
            ctx,
            routes,
            bank.slice_rows,
            H,
            EXPERTS,
            &self.routed_fp8,
            &self.routed_scale,
            &bank.w13,
            &bank.w13_scale,
            &self.m_indptr,
            &mut self.gate_up,
            &mut self.grouped_workspace,
        )?;
        glm52_silu_and_mul_bf16_launch(ctx, routes, bank.slice_i, &self.gate_up, &mut self.silu)?;
        glm52_fp8_per_token_group_quant_bf16_launch(
            ctx,
            Glm52MoeQuantShape {
                rows: routes,
                width: bank.slice_i,
                group_size: QUANT_GROUP,
            },
            &self.silu,
            &mut self.silu_fp8,
            &mut self.silu_scale,
        )?;
        glm52_fp8_grouped_gemm_sm100_launch(
            ctx,
            routes,
            H,
            bank.slice_i,
            EXPERTS,
            &self.silu_fp8,
            &self.silu_scale,
            &bank.w2,
            &bank.w2_scale,
            &self.m_indptr,
            &mut self.w2_out,
            &mut self.grouped_workspace,
        )?;

        // Shared expert (bank index 256): every row, dense large-M chain.
        let shared = EXPERTS;
        let block4 = block.next_multiple_of(4);
        fp8_linear_large_m_bank_into(
            ctx,
            block4,
            bank.slice_rows,
            H,
            &normed.slice(start * H..),
            &bank.w13,
            shared * bank.slice_rows * H,
            &bank.w13_scale,
            shared * (bank.slice_rows / QUANT_GROUP) * (H / QUANT_GROUP),
            &mut self.shared_gemm,
            &mut self.shared_gate_up,
        )?;
        glm52_silu_and_mul_bf16_launch(
            ctx,
            block4,
            bank.slice_i,
            &self.shared_gate_up,
            &mut self.shared_silu,
        )?;
        fp8_linear_large_m_bank_into(
            ctx,
            block4,
            H,
            bank.slice_i,
            &self.shared_silu,
            &bank.w2,
            shared * H * bank.slice_i,
            &bank.w2_scale,
            shared * (H / QUANT_GROUP) * (bank.slice_i / QUANT_GROUP),
            &mut self.shared_gemm,
            &mut self.shared_out,
        )?;

        let mut out_view = output.slice_mut(start * H..);
        glm52_prefill_moe_combine_launch(
            ctx,
            block,
            TOPK,
            H,
            &self.w2_out,
            &self.route_slot,
            &topk_weight,
            &self.shared_out,
            &mut out_view,
        )?;
        Ok(())
    }
}

/// In-process rendezvous for TP4 prefill NCCL unique id (+ optional
/// shutdown barrier so rank teardown stays ordered).
pub(crate) struct Glm52TpExchange {
    rank_count: usize,
    nccl_id: Mutex<Option<[core::ffi::c_char; 128]>>,
    nccl_ready: Condvar,
    departed: Mutex<usize>,
    all_out: Condvar,
}

impl Glm52TpExchange {
    pub(crate) fn new(rank_count: usize) -> Self {
        assert!(
            rank_count > 0 && rank_count <= RANKS,
            "TP exchange rank count out of range"
        );
        Self {
            rank_count,
            nccl_id: Mutex::new(None),
            nccl_ready: Condvar::new(),
            departed: Mutex::new(0),
            all_out: Condvar::new(),
        }
    }

    /// Share one NCCL unique id across the in-process rank threads: rank 0
    /// mints it, everyone returns the same bytes. Prefill all-reduce bring-up
    /// — all ranks must call this concurrently.
    fn nccl_id_rendezvous(&self, rank: usize) -> Result<cudarc::nccl::Id> {
        let mut slot = self.nccl_id.lock().expect("TP exchange poisoned");
        if rank == 0 {
            ensure!(slot.is_none(), "TP NCCL id minted twice");
            let id = cudarc::nccl::Id::new()
                .map_err(|err| anyhow::anyhow!("NCCL unique id creation failed: {:?}", err.0))?;
            *slot = Some(*id.internal());
            self.nccl_ready.notify_all();
            return Ok(id);
        }
        while slot.is_none() {
            let (guard, timeout) = self
                .nccl_ready
                .wait_timeout(slot, Duration::from_secs(120))
                .expect("TP exchange poisoned");
            slot = guard;
            if timeout.timed_out() && slot.is_none() {
                bail!("TP NCCL id rendezvous timed out after 120s — rank 0 never published");
            }
        }
        Ok(cudarc::nccl::Id::uninit(
            *slot.as_ref().expect("checked above"),
        ))
    }

    /// Shutdown-side barrier: optional ordering before dropping NCCL comms.
    pub(crate) fn teardown_rendezvous(&self, rank: usize) {
        let mut departed = self.departed.lock().expect("TP exchange poisoned");
        *departed += 1;
        self.all_out.notify_all();
        while *departed < self.rank_count {
            let (guard, timeout) = self
                .all_out
                .wait_timeout(departed, Duration::from_secs(120))
                .expect("TP exchange poisoned");
            departed = guard;
            if timeout.timed_out() && *departed < self.rank_count {
                log::warn!(
                    "GLM5.2 rank {rank} TP teardown rendezvous timed out ({}/{} arrived) \
                     — continuing; a peer rank likely died",
                    *departed,
                    self.rank_count
                );
                return;
            }
        }
    }
}

/// A rank's complete tensor-replicated MoE runtime: the state plus the
/// per-layer slice banks (keyed by absolute layer index).
pub(crate) struct Glm52MoeTpRank {
    pub(crate) state: Glm52MoeTpState,
    pub(crate) slices: BTreeMap<usize, Glm52MoeTpSliceBank>,
}

impl Glm52MoeTpRank {
    /// This layer's TP pieces: runtime state, slot index among sliced layers,
    /// and slice bank.
    pub(crate) fn layer_bank(
        &mut self,
        layer: usize,
    ) -> Option<(&mut Glm52MoeTpState, usize, &Glm52MoeTpSliceBank)> {
        let slot = self.slices.range(..layer).count();
        let bank = self.slices.get(&layer)?;
        Some((&mut self.state, slot, bank))
    }
}

/// cudarc's `Comm` holds a raw `ncclComm_t` and is `!Send`; this rank's
/// runtime (and therefore the comm) is created and used on its own worker
/// thread only, so moving the containing runtime between threads before
/// first use is safe. NCCL comms are not used concurrently from two threads.
pub(crate) struct Glm52PrefillNccl(cudarc::nccl::Comm);

unsafe impl Send for Glm52PrefillNccl {}

/// Per-rank tensor-parallel runtime for **TP4 prefill-only** (NCCL all-reduce).
pub(crate) struct Glm52MoeTpState {
    topology: Glm52TpTopology,
    rank: usize,
    /// Prefill NCCL communicator over the TP fleet.
    nccl: Option<Glm52PrefillNccl>,
}

impl Glm52MoeTpState {
    pub(crate) fn new(topology: Glm52TpTopology, rank: usize) -> Result<Self> {
        ensure!(
            rank < topology.ranks(),
            "{topology:?} rank {rank} out of range"
        );
        Ok(Self {
            topology,
            rank,
            nccl: None,
        })
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn ranks(&self) -> usize {
        self.topology.ranks()
    }

    /// Collective NCCL bring-up for the prefill-only path: all TP ranks must
    /// call this concurrently (rank 0 mints the unique id via the exchange).
    pub(crate) fn init_prefill_nccl(
        &mut self,
        ctx: &DeviceContext,
        exchange: &Glm52TpExchange,
    ) -> Result<()> {
        ensure!(self.nccl.is_none(), "GLM5.2 prefill NCCL initialized twice");
        let id = exchange.nccl_id_rendezvous(self.rank)?;
        let comm =
            cudarc::nccl::Comm::from_rank(ctx.stream.clone(), self.rank, self.topology.ranks(), id)
                .map_err(|err| {
                    anyhow::anyhow!("GLM5.2 prefill NCCL comm init failed: {:?}", err.0)
                })?;
        self.nccl = Some(Glm52PrefillNccl(comm));
        Ok(())
    }

    /// Sum-all-reduce `partial[..rows * H]` into `out[..rows * H]` across the
    /// TP fleet on this rank's stream (ncclAllReduce, bf16).
    pub(crate) fn prefill_allreduce(
        &mut self,
        _ctx: &DeviceContext,
        rows: usize,
        partial: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && partial.len() >= rows * H && out.len() >= rows * H,
            "GLM5.2 prefill all-reduce buffers are invalid"
        );
        let comm = self
            .nccl
            .as_ref()
            .context("GLM5.2 prefill NCCL communicator was never initialized")?;
        let send = partial.slice(..rows * H);
        let mut recv = out.slice_mut(..rows * H);
        comm.0
            .all_reduce(&send, &mut recv, &cudarc::nccl::ReduceOp::Sum)
            .map_err(|err| anyhow::anyhow!("GLM5.2 prefill NCCL all-reduce failed: {:?}", err.0))?;
        Ok(())
    }

    /// In-place variant of [`Self::prefill_allreduce`].
    pub(crate) fn prefill_allreduce_in_place(
        &mut self,
        _ctx: &DeviceContext,
        rows: usize,
        buffer: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && buffer.len() >= rows * H,
            "GLM5.2 prefill all-reduce buffer is invalid"
        );
        let comm = self
            .nccl
            .as_ref()
            .context("GLM5.2 prefill NCCL communicator was never initialized")?;
        let mut view = buffer.slice_mut(..rows * H);
        comm.0
            .all_reduce_in_place(&mut view, &cudarc::nccl::ReduceOp::Sum)
            .map_err(|err| anyhow::anyhow!("GLM5.2 prefill NCCL all-reduce failed: {:?}", err.0))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_staging_geometry() {
        // A synthetic expert with row-index-stamped bytes must land at the
        // right slice offsets for every rank.
        let mut w13_src = vec![0u8; INTERMEDIATE * H];
        for (row, chunk) in w13_src.chunks_mut(H).enumerate() {
            chunk.fill((row / SLICE_I) as u8); // stamp = owning rank
        }
        let mut w2_src = vec![0u8; W2_N * W2_K];
        for (row, chunk) in w2_src.chunks_mut(W2_K).enumerate() {
            for (col_block, seg) in chunk.chunks_mut(SLICE_I).enumerate() {
                seg.fill((row % 251) as u8 ^ (col_block as u8));
            }
        }
        for rank in 0..RANKS {
            let mut s = SliceStaging::new(rank, RANKS).expect("TP slice staging");
            s.put_w13_weight(3, false, &w13_src);
            s.put_w13_weight(3, true, &w13_src);
            s.put_w2_weight(3, &w2_src);
            let base = 3 * SLICE_ROWS * H;
            assert!(
                s.w13[base..base + SLICE_ROWS * H]
                    .iter()
                    .all(|&b| b == rank as u8)
            );
            let w2_base = 3 * H * SLICE_I;
            for row in [0usize, 17, H - 1] {
                let expect = (row % 251) as u8 ^ (rank as u8);
                assert!(
                    s.w2[w2_base + row * SLICE_I..w2_base + (row + 1) * SLICE_I]
                        .iter()
                        .all(|&b| b == expect),
                    "rank {rank} row {row}"
                );
            }
        }
    }

    #[test]
    fn tp4_slice_staging_geometry() {
        let mut w13_src = vec![0u8; INTERMEDIATE * H];
        let tp4_slice_i = INTERMEDIATE / 4;
        for (row, chunk) in w13_src.chunks_mut(H).enumerate() {
            chunk.fill((row / tp4_slice_i) as u8);
        }
        let mut w2_src = vec![0u8; W2_N * W2_K];
        for chunk in w2_src.chunks_mut(W2_K) {
            for (col_block, seg) in chunk.chunks_mut(tp4_slice_i).enumerate() {
                seg.fill(col_block as u8);
            }
        }

        for rank in 0..4 {
            let mut s = SliceStaging::new(rank, 4).expect("TP4 slice staging");
            assert_eq!(s.slice_i, 512);
            assert_eq!(s.slice_rows, 1024);
            s.put_w13_weight(2, false, &w13_src);
            s.put_w13_weight(2, true, &w13_src);
            s.put_w2_weight(2, &w2_src);

            let base = 2 * s.slice_rows * H;
            assert!(
                s.w13[base..base + s.slice_rows * H]
                    .iter()
                    .all(|&b| b == rank as u8)
            );
            let w2_base = 2 * H * s.slice_i;
            for row in [0usize, 17, H - 1] {
                assert!(
                    s.w2[w2_base + row * s.slice_i..w2_base + (row + 1) * s.slice_i]
                        .iter()
                        .all(|&b| b == rank as u8),
                    "rank {rank} row {row}"
                );
            }
        }
    }
}

#[cfg(test)]
mod prefill_nccl_tests {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::Mutex;

    use pegainfer_kernels::tensor::DeviceContext;

    use super::*;

    /// Minimal in-process 4-rank cudarc NCCL smoke: Id share + from_rank +
    /// bf16 sum all-reduce, mirroring the prefill bring-up exactly.
    #[test]
    #[ignore = "requires 4 CUDA devices"]
    fn tp4_nccl_allreduce_smoke() -> Result<()> {
        const RANKS: usize = 4;
        const ELEMS: usize = 6144 * 32;
        let id_slot: Arc<(Mutex<Option<[core::ffi::c_char; 128]>>, Condvar)> =
            Arc::new((Mutex::new(None), Condvar::new()));
        let barrier = Arc::new(Barrier::new(RANKS));
        let handles: Vec<_> = (0..RANKS)
            .map(|rank| {
                let id_slot = Arc::clone(&id_slot);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || -> Result<()> {
                    let ctx = DeviceContext::new_with_device(rank)?;
                    let id = if rank == 0 {
                        let id = cudarc::nccl::Id::new()
                            .map_err(|err| anyhow::anyhow!("id: {:?}", err.0))?;
                        *id_slot.0.lock().unwrap() = Some(*id.internal());
                        id_slot.1.notify_all();
                        id
                    } else {
                        let mut slot = id_slot.0.lock().unwrap();
                        while slot.is_none() {
                            slot = id_slot.1.wait(slot).unwrap();
                        }
                        cudarc::nccl::Id::uninit(slot.unwrap())
                    };
                    barrier.wait();
                    let comm = cudarc::nccl::Comm::from_rank(ctx.stream.clone(), rank, RANKS, id)
                        .map_err(|err| anyhow::anyhow!("init: {:?}", err.0))?;
                    let send_host = vec![bf16::from_f32((rank + 1) as f32); ELEMS];
                    let send = ctx.stream.clone_htod(&send_host)?;
                    let mut recv = ctx.stream.alloc_zeros::<bf16>(ELEMS)?;
                    let send_view = send.slice(..ELEMS);
                    let mut recv_view = recv.slice_mut(..ELEMS);
                    comm.all_reduce(&send_view, &mut recv_view, &cudarc::nccl::ReduceOp::Sum)
                        .map_err(|err| anyhow::anyhow!("allreduce: {:?}", err.0))?;
                    drop(recv_view);
                    let host = ctx.stream.clone_dtoh(&recv)?;
                    ensure!(
                        host.iter().all(|v| v.to_f32() == 10.0),
                        "rank {rank} sum mismatch: {}",
                        host[0].to_f32()
                    );
                    barrier.wait();
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("nccl smoke rank panicked")?;
        }
        Ok(())
    }
}
