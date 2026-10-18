//! Single-layer GLM5.2 MLA decode forward, row-batched:
//! `hidden[T, 6144] -> o[T, 6144]` (each row is an independent token).
//!
//! Composes the oracle-validated GPU ops into one callable forward — the
//! attention half of a decode layer. The pieces are each gated against the HF
//! MLA oracle in `tests/mla_decode_oracle.rs` (front projections, the rope/query/
//! cache-pack assembly, FlashMLA sparse decode, the back-half v_up/o_proj); this
//! module wires them with no new math.
//!
//! Weights are taken as raw fp8 bytes (`from_host`) and uploaded once — the module
//! is loader-agnostic (functional core). kv_b is pre-dequantized into the bf16
//! absorb factors W_UK / W_UV at construction; the fp8 projection weights stay
//! as-loaded and every projection relays its activation scale into the TRTLLM
//! col-major TMA layout before the blockscale linear (the documented footgun).

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN;
use pegainfer_kernels::ops::GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN;
#[cfg(test)]
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW;
use pegainfer_kernels::ops::Glm52FlashInferSparseDecode;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::Glm52MoeQuantShape;
use pegainfer_kernels::ops::gemm_strided_batched_bf16;
use pegainfer_kernels::ops::glm52_flashinfer_sparse_mla_fp8_launch;
use pegainfer_kernels::ops::glm52_flashinfer_sparse_mla_supported;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_launch;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_metadata_launch;
use pegainfer_kernels::ops::glm52_fp8_per_token_group_quant_bf16_ue8m0_launch;
use pegainfer_kernels::ops::glm52_mla_cache_pack_launch;
use pegainfer_kernels::ops::glm52_mla_front_pack_fp8_launch;
use pegainfer_kernels::ops::glm52_mla_query_assemble_launch;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_HEADS;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_KV_LORA_RANK;
use crate::config::GLM52_QK_HEAD_DIM;
use crate::config::GLM52_QK_NOPE_HEAD_DIM;
use crate::config::GLM52_QK_ROPE_HEAD_DIM;
use crate::config::GLM52_RMS_EPS as RMS_EPS;
use crate::config::GLM52_V_HEAD_DIM;
use crate::fp8::FP8_BLOCK;
use crate::fp8::fp8_linear_into;
use crate::mla_front::Glm52MlaFront;
use crate::mla_front::Glm52MlaLayerWeights;
#[cfg(test)]
use crate::mla_front::glm52_mla_front_into;
use crate::rows::Rows;

// Local short names for the config-owned architecture constants (the module
// is dense with shape math; the values live in one place).
const HEADS: usize = GLM52_HEADS;
const HIDDEN: usize = GLM52_HIDDEN;
const QK_NOPE: usize = GLM52_QK_NOPE_HEAD_DIM; // absorbed q nope width per head
const Q_HEAD: usize = GLM52_QK_HEAD_DIM; // qk_nope(192) + qk_rope(64)
const ROPE_DIM: usize = GLM52_QK_ROPE_HEAD_DIM;
const KV_LORA: usize = GLM52_KV_LORA_RANK;
const V_HEAD: usize = GLM52_V_HEAD_DIM;
const QUERY_DIM: usize = KV_LORA + ROPE_DIM; // 576

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52MlaBackend {
    FlashMlaFp8Ds,
    FlashInferFp8,
}

impl Glm52MlaBackend {
    pub(crate) fn cache_bytes_per_token(self) -> usize {
        match self {
            Self::FlashMlaFp8Ds => GLM52_FLASHMLA_SPARSE_BYTES_PER_TOKEN,
            Self::FlashInferFp8 => GLM52_FLASHINFER_SPARSE_BYTES_PER_TOKEN,
        }
    }
}

pub(crate) fn glm52_select_mla_backend(heads: usize) -> Result<Glm52MlaBackend> {
    if heads == 16 && glm52_flashinfer_sparse_mla_supported(heads)? {
        Ok(Glm52MlaBackend::FlashInferFp8)
    } else {
        Ok(Glm52MlaBackend::FlashMlaFp8Ds)
    }
}

/// Persistent scratch for the MLA attend half: absorb/query-assemble/cache-
/// pack intermediates and the FlashMLA output + split accumulators, sized for
/// the contract's `batch_size` rows. Shared across all 78 layers, written in
/// place every step.
pub(crate) struct Glm52MlaAttendScratch {
    // Compact head-shard buffers ([T, heads, .]): the absorb GEMM output and
    // the W_UV output feeding o_proj.
    ql_nope: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    backend: Glm52MlaBackendScratch,
    heads: usize,
    // Owned mma partial buffer for the o_proj projection (see Glm52MlaFront).
    gemv_partial: CudaSlice<f32>,
    // Wide-bucket route (#812): o_proj runs the fp8 GEMM past the GEMV row
    // ceiling. Owned per scratch, like `gemv_partial`.
    wide: Option<crate::fp8::Glm52Fp8GemmScratch>,
}

enum Glm52MlaBackendScratch {
    Fp8Ds(Box<Glm52Fp8DsScratch>),
    FlashInfer(Box<Glm52FlashInferScratch>),
}

struct Glm52Fp8DsScratch {
    query: CudaSlice<bf16>,
    latent: CudaSlice<bf16>,
    attend: Glm52Fp8DsAttendScratch,
    ckv_fp8: CudaSlice<u8>,
    ckv_scales: CudaSlice<f32>,
}

enum Glm52Fp8DsAttendScratch {
    FlashMla {
        lse: CudaSlice<f32>,
        lse_accum: CudaSlice<f32>,
        o_accum: CudaSlice<f32>,
    },
}

struct Glm52FlashInferScratch {
    query: CudaSlice<u8>,
    latent: CudaSlice<bf16>,
    workspace: CudaSlice<u8>,
}

impl Glm52MlaAttendScratch {
    #[cfg(test)]
    fn new(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        heads: usize,
    ) -> Result<Self> {
        Self::new_for_backend(ctx, contract, heads, Glm52MlaBackend::FlashMlaFp8Ds)
    }

    pub(crate) fn new_for_backend(
        ctx: &DeviceContext,
        contract: &Glm52FlashMlaSparseDecode,
        heads: usize,
        backend: Glm52MlaBackend,
    ) -> Result<Self> {
        ensure!(
            (1..=HEADS).contains(&heads),
            "GLM5.2 MLA attend heads {heads} out of 1..={HEADS}"
        );
        let t = contract.batch_size;
        let backend = match backend {
            Glm52MlaBackend::FlashMlaFp8Ds => {
                Glm52MlaBackendScratch::Fp8Ds(Box::new(Glm52Fp8DsScratch {
                    query: ctx.stream.alloc_zeros::<bf16>(t * HEADS * QUERY_DIM)?,
                    latent: ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?,
                    attend: Glm52Fp8DsAttendScratch::FlashMla {
                        lse: ctx.stream.alloc_zeros::<f32>(contract.lse_len())?,
                        lse_accum: ctx.stream.alloc_zeros::<f32>(contract.lse_accum_len())?,
                        o_accum: ctx.stream.alloc_zeros::<f32>(contract.o_accum_len())?,
                    },
                    ckv_fp8: ctx.stream.alloc_zeros::<u8>(t * KV_LORA)?,
                    ckv_scales: ctx.stream.alloc_zeros::<f32>(t * (KV_LORA / FP8_BLOCK))?,
                }))
            }
            Glm52MlaBackend::FlashInferFp8 => {
                Glm52MlaBackendScratch::FlashInfer(Box::new(Glm52FlashInferScratch {
                    query: ctx.stream.alloc_zeros::<u8>(t * heads * QUERY_DIM)?,
                    latent: ctx.stream.alloc_zeros::<bf16>(t * heads * KV_LORA)?,
                    workspace: ctx
                        .stream
                        .alloc_zeros::<u8>(GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES)?,
                }))
            }
        };
        Ok(Self {
            ql_nope: ctx.stream.alloc_zeros::<bf16>(t * heads * KV_LORA)?,
            v: ctx.stream.alloc_zeros::<bf16>(t * heads * V_HEAD)?,
            backend,
            heads,
            gemv_partial: ctx
                .stream
                .alloc_zeros::<f32>(t * GLM52_GEMV_MMA_SCRATCH_FLOATS_PER_ROW)?,
            wide: (t > crate::fp8::FP8_GEMV_MAX_ROWS)
                .then(|| {
                    // o_proj consumes the [T, heads * V_HEAD] attention out.
                    crate::fp8::Glm52Fp8GemmScratch::new(ctx, t.next_multiple_of(4), heads * V_HEAD)
                })
                .transpose()?,
        })
    }
}

/// MLA decode forward for one token (bs=1): runs the projections, assembles the
/// FlashMLA query, writes the new token into the paged cache at `position`,
/// attends over the cached context, and projects back to `o[6144]`.
///
/// Allocating convenience over the `_into` halves for the oracle-gate/test
/// paths (per-call scratch). `cache` is the fp8_ds_mla paged cache (656
/// bytes/token); `cos`/`sin` are the position's rotary table first half
/// (`[32]`); `topk` is the (fixed-2048, -1-padded) sparse index list; `sched`
/// carries the FlashMLA launch sizing (its contract) plus the precomputed
/// tile-scheduler plan.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm52_mla_decode_forward(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    hidden: &Rows<HIDDEN>,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    position: usize,
    topk: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
) -> Result<Rows<HIDDEN>> {
    ensure!(
        position < sched.contract.num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        "GLM5.2 MLA position {position} outside paged cache ({} blocks x {GLM52_FLASHMLA_SPARSE_PAGE_SIZE})",
        sched.contract.num_blocks
    );
    // Front, attend scratch, and output all sized from the plan's contract —
    // the same one-construction-point coherence the production bucket state
    // provides.
    let mut front = Glm52MlaFront::new(ctx, sched.batch(), w.heads)?;
    let mut attend = Glm52MlaAttendScratch::new(ctx, &sched.contract, w.heads)?;
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let seq_lens = ctx.stream.clone_htod(&[(position + 1) as i32])?;
    ctx.stream
        .memcpy_htod(&[position as i64], &mut slot_mapping)?;
    let mut o = Rows::zeros(ctx, sched.batch())?;
    glm52_mla_front_into(ctx, w, hidden, &mut front, sched.folds_kv_pack())?;
    glm52_mla_attend_into(
        ctx,
        w,
        &front,
        cos,
        sin,
        cache,
        0,
        &slot_mapping,
        topk,
        &seq_lens,
        sched,
        &mut attend,
        &mut o,
    )?;
    Ok(o)
}

/// A FlashMLA sparse decode contract paired with its tile-scheduler plan. The
/// plan depends only on `batch_size`, `topk` and `num_sm_parts` — not on
/// position, sequence length, or layer — so it is computed once (model build
/// time) instead of per layer per step (78 × ~25 µs/step at bs=1). Owning the
/// contract makes a plan/contract mismatch unrepresentable: every consumer
/// reads both from the same object.
pub(crate) struct Glm52MlaSchedMetadata {
    contract: Glm52FlashMlaSparseDecode,
    backend: Glm52MlaBackendSchedule,
}

enum Glm52MlaBackendSchedule {
    FlashMla {
        tile_scheduler_metadata: CudaSlice<i32>,
        num_splits: CudaSlice<i32>,
    },
    FlashInfer,
}

impl Glm52MlaSchedMetadata {
    #[cfg(test)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        contract: Glm52FlashMlaSparseDecode,
        heads: usize,
    ) -> Result<Self> {
        Self::new_for_backend(ctx, contract, heads, Glm52MlaBackend::FlashMlaFp8Ds)
    }

    pub(crate) fn new_for_backend(
        ctx: &DeviceContext,
        contract: Glm52FlashMlaSparseDecode,
        _heads: usize,
        backend: Glm52MlaBackend,
    ) -> Result<Self> {
        let backend = match backend {
            Glm52MlaBackend::FlashMlaFp8Ds => {
                let mut tile_scheduler_metadata = ctx
                    .stream
                    .alloc_zeros::<i32>(contract.tile_scheduler_metadata_len())?;
                let mut num_splits = ctx.stream.alloc_zeros::<i32>(contract.num_splits_len())?;
                glm52_flashmla_sparse_decode_metadata_launch(
                    ctx,
                    contract.batch_size,
                    contract.topk,
                    contract.num_sm_parts,
                    &mut tile_scheduler_metadata,
                    &mut num_splits,
                )?;
                Glm52MlaBackendSchedule::FlashMla {
                    tile_scheduler_metadata,
                    num_splits,
                }
            }
            Glm52MlaBackend::FlashInferFp8 => Glm52MlaBackendSchedule::FlashInfer,
        };
        Ok(Self { contract, backend })
    }

    /// Rebind the plan to the launch-decided pool block count. The tile
    /// scheduler plan depends only on `batch_size`/`topk`/`num_sm_parts`, so
    /// the plan built before the measured KV fill stays valid — only the
    /// cache-geometry field the attend launches read needs the real count.
    pub(crate) fn set_num_blocks(&mut self, num_blocks: usize) {
        self.contract.num_blocks = num_blocks;
    }

    /// The sparse index-list length this plan was built for. The DSA indexer
    /// must produce its top-k with the same k — reading it from the plan makes
    /// an indexer/attend mismatch unrepresentable.
    pub(crate) fn topk(&self) -> usize {
        self.contract.topk
    }

    /// The decode row count this plan was built for — the single source of
    /// truth for a step's batch shape (every consumer reads it from here).
    pub(crate) fn batch(&self) -> usize {
        self.contract.batch_size
    }

    /// Whether the attend half packs the cache with the fused front kernel
    /// (query assemble + kv_a RMSNorm + cache pack in one launch): the MLA
    /// front must then leave `front.ckv` raw and skip the split/norm.
    pub(crate) fn folds_kv_pack(&self) -> bool {
        matches!(self.backend, Glm52MlaBackendSchedule::FlashInfer)
    }
}

/// MLA attend half over the plan's `batch()` rows: consumes the front
/// projections + the per-row sparse top-k, packs each row's new token into
/// its paged-cache slot, runs FlashMLA sparse decode, and projects back into
/// `out[T, 6144]`. Every intermediate lives in the persistent attend scratch
/// — the chain is allocation-free. `cos`/`sin` carry one `[32]` row per token
/// (each row sits at its own position); `slot_mapping`/`topk` are `[T]` /
/// `[T, topk]`. `kv_layer_offset` is this layer's MLA slice offset inside
/// `cache` (0 for a dedicated arena); it rides each call because one plan
/// serves every layer, and the plan's contract carries the block stride.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm52_mla_attend_into(
    ctx: &DeviceContext,
    w: &Glm52MlaLayerWeights,
    front: &Glm52MlaFront,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
    cache: &mut CudaSlice<u8>,
    kv_layer_offset: usize,
    slot_mapping: &CudaSlice<i64>,
    topk: &CudaSlice<i32>,
    seq_lens: &CudaSlice<i32>,
    sched: &Glm52MlaSchedMetadata,
    s: &mut Glm52MlaAttendScratch,
    out: &mut Rows<HIDDEN>,
) -> Result<()> {
    let contract = Glm52FlashMlaSparseDecode {
        kv_layer_offset_bytes: kv_layer_offset,
        ..sched.contract
    };
    let t = contract.batch_size;
    ensure!(
        w.heads == front.heads && w.heads == s.heads,
        "GLM5.2 MLA attend head mismatch: weights {}, front {}, scratch {}",
        w.heads,
        front.heads,
        s.heads
    );
    let heads = w.heads;
    // Each row's new token is written to cache slot `slot_mapping[row]`
    // (device data, so the launch replays under CUDA graph capture); the
    // cache-pack kernel traps on a slot outside the paged window. The
    // every-step host guard is the caller's position bound (`decode_step`
    // prologue: position < max_model_len and each row confined to its
    // own slot region by construction).

    // ---- absorb: ql_nope[T,heads,512] = q_pass @ W_UK ----
    // cuBLAS batches over this instance's heads (the full 64 or an
    // attention-TP shard); the T rows ride the GEMM's n dimension — column t
    // of head h reads q_full[t, h, 0..192] (ldb = the compact [T,heads,256]
    // token stride) and writes ql_nope[t, h, 0..512] (ldc = the compact
    // [T,heads,512] token stride).
    gemm_strided_batched_bf16(
        ctx,
        false,
        false,
        KV_LORA,
        t,
        QK_NOPE,
        &w.w_uk,
        KV_LORA,
        QK_NOPE * KV_LORA,
        &front.q_full,
        heads * Q_HEAD,
        Q_HEAD,
        &mut s.ql_nope,
        heads * KV_LORA,
        KV_LORA,
        heads,
    )?;

    let (latent, latent_token_stride) = match (&sched.backend, &mut s.backend) {
        (
            schedule @ Glm52MlaBackendSchedule::FlashMla { .. },
            Glm52MlaBackendScratch::Fp8Ds(scratch),
        ) => {
            glm52_mla_query_assemble_launch(
                ctx,
                t,
                heads,
                &s.ql_nope,
                &front.q_full,
                QK_NOPE,
                Q_HEAD,
                cos,
                sin,
                &mut scratch.query,
            )?;
            // UE8M0 scales: the FlashMLA fp8_ds_mla cache contract — the sm100
            // kernel truncates stored scales to powers of two (see the quant
            // kernel's comment); amax/448 scales silently lose up to 1 bit of
            // V/K magnitude per group on GB300.
            glm52_fp8_per_token_group_quant_bf16_ue8m0_launch(
                ctx,
                Glm52MoeQuantShape {
                    rows: t,
                    width: KV_LORA,
                    group_size: FP8_BLOCK,
                },
                &front.kv_c,
                &mut scratch.ckv_fp8,
                &mut scratch.ckv_scales,
            )?;
            glm52_mla_cache_pack_launch(
                ctx,
                t,
                &scratch.ckv_fp8,
                &scratch.ckv_scales,
                &front.k_pe,
                cos,
                sin,
                cache,
                contract.kv_layer_offset_bytes,
                contract.kv_block_stride_bytes,
                contract.num_blocks,
                slot_mapping,
            )?;
            match (schedule, &mut scratch.attend) {
                (
                    Glm52MlaBackendSchedule::FlashMla {
                        tile_scheduler_metadata,
                        num_splits,
                    },
                    Glm52Fp8DsAttendScratch::FlashMla {
                        lse,
                        lse_accum,
                        o_accum,
                    },
                ) => glm52_flashmla_sparse_decode_launch(
                    ctx,
                    contract,
                    &scratch.query,
                    cache,
                    topk,
                    tile_scheduler_metadata,
                    num_splits,
                    &mut scratch.latent,
                    lse,
                    lse_accum,
                    o_accum,
                )?,
                _ => anyhow::bail!("GLM5.2 FP8-DS MLA schedule/scratch backend mismatch"),
            }
            (&scratch.latent, HEADS * KV_LORA)
        }
        (Glm52MlaBackendSchedule::FlashInfer, Glm52MlaBackendScratch::FlashInfer(scratch)) => {
            // The FlashInfer pack/attend launches address the cache dense
            // from its base — this backend's cache never rides a strided
            // slab layout.
            ensure!(
                contract.kv_layer_offset_bytes == 0,
                "GLM5.2 FlashInfer sparse decode requires a dense cache (layer offset {})",
                contract.kv_layer_offset_bytes
            );
            // One fused launch: query assemble + kv_a RMSNorm + cache pack
            // (the split/norm were skipped in `glm52_mla_front_rest_into` for
            // this backend — the raw kv_a output is consumed directly).
            glm52_mla_front_pack_fp8_launch(
                ctx,
                t,
                heads,
                &s.ql_nope,
                &front.q_full,
                QK_NOPE,
                Q_HEAD,
                &front.ckv,
                &w.kv_a_ln.data,
                RMS_EPS,
                cos,
                sin,
                &mut scratch.query,
                cache,
                slot_mapping,
            )?;
            glm52_flashinfer_sparse_mla_fp8_launch(
                ctx,
                Glm52FlashInferSparseDecode {
                    batch_size: t,
                    heads,
                    num_blocks: contract.num_blocks,
                    topk: contract.topk,
                    sm_scale: contract.sm_scale,
                },
                &scratch.query,
                cache,
                topk,
                seq_lens,
                &mut scratch.latent,
                &mut scratch.workspace,
            )?;
            (&scratch.latent, heads * KV_LORA)
        }
        _ => anyhow::bail!("GLM5.2 MLA schedule/scratch backend mismatch"),
    };

    // ---- back: v[T,heads,256] = latent @ W_UV, then o_proj ----
    // latent stays full-width [T,64,512] (FlashMLA output); the batch count
    // reads only the shard's head slots. v is compact — exactly the sliced
    // o_proj's k = heads * 256 input.
    gemm_strided_batched_bf16(
        ctx,
        true,
        false,
        V_HEAD,
        t,
        KV_LORA,
        &w.w_uv,
        KV_LORA,
        V_HEAD * KV_LORA,
        latent,
        latent_token_stride,
        KV_LORA,
        &mut s.v,
        heads * V_HEAD,
        V_HEAD,
        heads,
    )?;
    if let Some(wide) = s.wide.as_mut() {
        return crate::fp8::fp8_linear_large_m_into(ctx, &w.o_proj, t, &s.v, wide, out.data_mut());
    }
    fp8_linear_into(
        ctx,
        &w.o_proj,
        t,
        &s.v,
        Some(&mut s.gemv_partial),
        out.data_mut(),
    ) // [T, 6144]
}
