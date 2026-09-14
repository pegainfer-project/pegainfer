use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use bytesize::ByteSize;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use crossbeam_channel::unbounded;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::nccl::ReduceOp;
use cudarc::nccl::safe::Comm;
use cudarc::nccl::safe::Id;
use log::debug;
use pegainfer_core::cuda_graph::CudaGraphState;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_kernels::ops::KIMI_K2_LOCAL_EXPERTS;
use pegainfer_kernels::ops::KIMI_K2_MLA_KV_A_OUT;
use pegainfer_kernels::ops::KIMI_K2_MLA_KV_LORA_RANK;
use pegainfer_kernels::ops::KIMI_K2_MLA_Q_HEAD_DIM;
use pegainfer_kernels::ops::KIMI_K2_MLA_QKV_A_OUT;
use pegainfer_kernels::ops::KIMI_K2_MLA_ROPE_DIM;
use pegainfer_kernels::ops::KIMI_K2_MLA_V_HEAD_DIM;
use pegainfer_kernels::ops::KIMI_O_PROJ_CUBLASLT_INPUT;
use pegainfer_kernels::ops::KimiMarlinRouteWorkspace;
use pegainfer_kernels::ops::KimiMarlinWna16Workspace;
use pegainfer_kernels::ops::KimiMlaPagedKvLayout;
use pegainfer_kernels::ops::flashinfer_top1_row_states_bytes;
use pegainfer_kernels::ops::kimi_flashinfer_batch_decode_mla_rt;
use pegainfer_kernels::ops::kimi_flashinfer_single_prefill_mla_rt;
use pegainfer_kernels::ops::kimi_mla_absorb_q_nope_rt;
use pegainfer_kernels::ops::kimi_mla_assemble_cached_kv_rt;
use pegainfer_kernels::ops::kimi_mla_gather_cached_ckv_rt;
use pegainfer_kernels::ops::kimi_mla_paged_kv_append;
use pegainfer_kernels::ops::kimi_mla_rope_apply_kpe;
use pegainfer_kernels::ops::kimi_mla_rope_assemble_prefill_rt;
use pegainfer_kernels::ops::kimi_mla_rope_split_decode_rt;
use pegainfer_kernels::ops::kimi_mla_split_qkv_a;
use pegainfer_kernels::ops::kimi_mla_split_qkv_a_norm;
use pegainfer_kernels::ops::kimi_mla_v_up_rt;
use pegainfer_kernels::ops::kimi_o_proj_cublaslt_into;
use pegainfer_kernels::ops::kimi_o_proj_cublaslt_supports_batch_size;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::GpuTensor;
use pegainfer_kernels::tensor::GpuWeight;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::NormWeight;
use pegainfer_kernels::typed_ops;

use crate::config::KIMI_K2_DENSE_LAYERS;
use crate::config::KIMI_K2_HIDDEN;
use crate::config::KIMI_K2_LAYERS;
use crate::config::KIMI_K2_MOE_LAYERS;
use crate::config::KIMI_K2_Q_LORA_RANK;
use crate::config::KIMI_K2_QK_ROPE_HEAD_DIM;
use crate::config::KIMI_K2_RMS_NORM_EPS;
use crate::config::KIMI_K2_ROPE_THETA;
use crate::config::KIMI_K2_TOPK;
use crate::config::KIMI_K2_YARN_BETA_FAST;
use crate::config::KIMI_K2_YARN_BETA_SLOW;
use crate::config::KIMI_K2_YARN_FACTOR;
use crate::config::KIMI_K2_YARN_ORIGINAL_MAX_POS;
use crate::runner::affinity::KimiRankThreadPlacement;
use crate::runner::affinity::pin_rank_worker_thread;
pub(super) use crate::typed_scratch::KimiWorkerDecodeScratch;
pub(super) use crate::typed_scratch::MARLIN_W13_OUT_DIM;
use crate::weights::KimiGpuRawTensor;
use crate::weights::KimiLayerWeightKindNames;
use crate::weights::KimiLayerWeightNames;
use crate::weights::KimiRankExpertMarlinWeights;
use crate::weights::KimiRankGpuContext;
use crate::weights::KimiRankGpuWeights;
use crate::weights::KimiRankSlicedLoadPlan;
use crate::weights::KimiRankWeightNames;
use crate::weights::KimiRouterDeviceWeights;
use crate::weights::KimiRouterGpuWeights;
use crate::weights::load_rank_sliced_weights_to_gpu;

const KIMI_MARLIN_MAX_BLOCK_SIZE: usize = 64;
const KIMI_DECODE_MAX_BATCH: usize = 64;
const KIMI_DECODE_BATCH_BUCKETS: [usize; 7] = [1, 2, 4, 8, 16, 32, KIMI_DECODE_MAX_BATCH];
/// KV page granularity shared by the worker page tables and the scheduler's
/// logical `BlockPool` (one kvbm block = one physical page).
pub(crate) const KIMI_KV_PAGE_SIZE: usize = 16;
/// Per-request KV capacity: prompt + generated tokens. Sets the decode RoPE
/// table length; the scheduler rejects requests over this at admission.
pub(crate) const KIMI_MAX_REQUEST_TOKENS: usize = 8192;

/// Physical KV page assignments for one forward step.
///
/// `pages`/`indptr` form a row-major CSR over the step's rows (prefill: one
/// row for the target slot; decode: one row per active request, same order
/// as `slots`). Page IDs index the rank's shared KV pool. `padding_page`
/// backs idle slots and CUDA-graph padding rows — those rows write/read
/// garbage on a page no live request owns, which is benign by construction.
#[derive(Clone, Debug)]
pub(crate) struct KimiKvStepPages {
    pages: Vec<i32>,
    indptr: Vec<i32>,
    padding_page: i32,
}

impl KimiKvStepPages {
    pub(crate) fn new(rows: Vec<Vec<i32>>, padding_page: i32) -> Self {
        let mut pages = Vec::with_capacity(rows.iter().map(Vec::len).sum());
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        indptr.push(0i32);
        for row in rows {
            pages.extend_from_slice(&row);
            indptr.push(pages.len() as i32);
        }
        Self {
            pages,
            indptr,
            padding_page,
        }
    }

    pub(crate) fn single(row: Vec<i32>, padding_page: i32) -> Self {
        Self::new(vec![row], padding_page)
    }

    pub(super) fn rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    pub(super) fn row(&self, row: usize) -> Result<&[i32]> {
        ensure!(
            row + 1 < self.indptr.len(),
            "Kimi KV step pages row {row} out of range ({} rows)",
            self.rows()
        );
        let start = self.indptr[row] as usize;
        let end = self.indptr[row + 1] as usize;
        ensure!(
            start <= end && end <= self.pages.len(),
            "Kimi KV step pages row {row} CSR is corrupt: start={start}, end={end}, pages={}",
            self.pages.len()
        );
        Ok(&self.pages[start..end])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KimiK2RankPlacement {
    pub rank: usize,
    pub device_ordinal: usize,
}

impl KimiK2RankPlacement {
    fn new(rank: usize, device_ordinal: usize) -> Result<Self> {
        ensure!(rank < 8, "Kimi-K2 rank must be < 8, got {rank}");
        Ok(Self {
            rank,
            device_ordinal,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct KimiRankWeightLoadReport {
    pub rank: usize,
    pub tensor_count: usize,
    pub total_bytes: usize,
    pub expert_kernel_layers: usize,
    pub expert_kernel_total_bytes: usize,
    pub loaded_to_gpu: bool,
    pub typed_view_validated: bool,
    pub expert_kernel_weights_packaged: bool,
}

/// Per-row token-selection options carried through a forward call: how the
/// next token is picked (greedy argmax vs temperature/top-k/top-p sampling)
/// and how many logprobs to report for it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KimiRowOptions {
    pub(crate) logprobs: Option<usize>,
    pub(crate) sampling: pegainfer_frontend::sampler::SamplingParams,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct KimiOneTokenForwardReport {
    pub rank: usize,
    pub batch_slot: usize,
    pub input_token_id: u32,
    pub local_next_token_id: u32,
    pub local_next_token_global_id: u32,
    pub local_top_logit_f32: f32,
    pub vocab_start: usize,
    pub vocab_rows: usize,
    pub dense_layers_executed: usize,
    pub moe_layers_executed: usize,
    /// Exact log-softmax of the picked token plus the top-K, computed on the
    /// host from the full-vocab logits row. `Some` only when the request
    /// asked for logprobs (`GenerateRequest::logprobs` is `Some`); the serving
    /// path never pays for it.
    pub logprob: Option<TokenLogprob>,
}

enum KimiRankCommand {
    LoadSlicedWeights {
        model_path: PathBuf,
        resp: Sender<Result<KimiRankWeightLoadReport>>,
    },
    InitTpComm {
        id: Id,
        world_size: usize,
        resp: Sender<Result<()>>,
    },
    EnsureDecodeArena {
        decode_batch_size: usize,
        resp: Sender<Result<()>>,
    },
    ForwardPromptNextToken {
        slot: usize,
        decode_batch_size: usize,
        input_ids: Vec<u32>,
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
        resp: Sender<Result<KimiOneTokenForwardReport>>,
    },
    ForwardDecodeBatchNextTokens {
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        kv_pages: KimiKvStepPages,
        rows: Vec<KimiRowOptions>,
        seed: u64,
        resp: Sender<Result<Vec<KimiOneTokenForwardReport>>>,
    },
    EnableDeepEp {
        unique_id: [u8; 128],
        num_ranks: usize,
        resp: Sender<Result<()>>,
    },
    Shutdown,
}

pub(super) struct KimiRankWorker {
    placement: KimiK2RankPlacement,
    tx: Sender<KimiRankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl KimiRankWorker {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spawn(
        placement: KimiK2RankPlacement,
        weight_names: KimiRankWeightNames,
        sliced_load_plan: KimiRankSlicedLoadPlan,
        thread_placement: KimiRankThreadPlacement,
        local_dims: crate::config::KimiLocalDims,
        ctx: KimiRankGpuContext,
        collective_barrier: Arc<Barrier>,
        enable_cuda_graph: bool,
        kv_pool_pages: usize,
    ) -> Result<Self> {
        ensure!(
            weight_names.rank == placement.rank,
            "Kimi rank weight names {} do not match placement {}",
            weight_names.rank,
            placement.rank
        );
        ensure!(
            sliced_load_plan.rank == placement.rank,
            "Kimi rank sliced load plan {} does not match placement {}",
            sliced_load_plan.rank,
            placement.rank
        );
        ensure!(
            thread_placement.rank == placement.rank,
            "Kimi rank thread placement {} does not match placement {}",
            thread_placement.rank,
            placement.rank
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("kimi-k2-rank-{}", placement.rank))
            .spawn(move || {
                pin_rank_worker_thread(&thread_placement);
                match bind_rank_thread(
                    ctx,
                    weight_names,
                    sliced_load_plan,
                    local_dims,
                    collective_barrier,
                    enable_cuda_graph,
                    kv_pool_pages,
                ) {
                    Ok(state) => {
                        let _ = startup_tx.send(Ok(()));
                        rank_worker_loop(rx, state);
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 rank worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker exited during startup"))??;
        Ok(Self {
            placement,
            tx,
            handle: Some(handle),
        })
    }

    pub(super) fn placement(&self) -> KimiK2RankPlacement {
        self.placement
    }

    pub(super) fn load_sliced_weights_async(
        &self,
        model_path: &Path,
    ) -> Result<Receiver<Result<KimiRankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::LoadSlicedWeights {
                model_path: model_path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn init_tp_comm_async(
        &self,
        id: Id,
        world_size: usize,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::InitTpComm {
                id,
                world_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn ensure_decode_arena_async(
        &self,
        decode_batch_size: usize,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::EnsureDecodeArena {
                decode_batch_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_prompt_next_token_async(
        &self,
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    ) -> Result<Receiver<Result<KimiOneTokenForwardReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                cached_tokens,
                ep_max_seq_len,
                kv_pages,
                row,
                seed,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_decode_batch_next_tokens_async(
        &self,
        token_ids: Vec<u32>,
        append_positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        kv_pages: KimiKvStepPages,
        rows: Vec<KimiRowOptions>,
        seed: u64,
    ) -> Result<Receiver<Result<Vec<KimiOneTokenForwardReport>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                kv_pages,
                rows,
                seed,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    /// Collective DeepEP context creation: send to every rank first, then
    /// wait — each worker thread blocks inside the NCCL init until all ranks
    /// have joined.
    pub(super) fn enable_deepep_async(
        &self,
        unique_id: [u8; 128],
        num_ranks: usize,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(KimiRankCommand::EnableDeepEp {
                unique_id,
                num_ranks,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(super) fn shutdown(&mut self) -> Result<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        self.tx
            .send(KimiRankCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker channel closed"))?;
        let handle = self.handle.take().expect("Kimi rank handle must exist");
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("Kimi-K2 rank worker panicked"))?;
        Ok(())
    }
}

impl Drop for KimiRankWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct KimiRankThreadState {
    ctx: KimiRankGpuContext,
    decode_aux_ctx: DeviceContext,
    _cublas: KimiCublasThreadGuard,
    tp_comm: Option<OwnedRankComm>,
    weight_names: KimiRankWeightNames,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    local_dims: crate::config::KimiLocalDims,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
    kv_pool_pages: usize,
    loaded: Option<KimiRankLoadedWeights>,
    deepep: Option<super::moe_deepep::KimiMoeDeepEpState>,
}

struct OwnedRankComm(Comm);

// SAFETY: each NCCL communicator is moved into exactly one persistent Kimi rank
// worker and is only used from that worker thread on its owning CUDA stream.
unsafe impl Send for OwnedRankComm {}

impl OwnedRankComm {
    fn get(&self) -> &Comm {
        &self.0
    }
}

struct KimiRankLoadedWeights {
    gpu: KimiRankGpuWeights,
    expert_kernels: KimiRankExpertMarlinWeights,
    one_token_cache: KimiOneTokenForwardCache,
    kv_pool: KimiWorkerKvPool,
    decode_arenas: KimiWorkerDecodeArenas,
}

struct KimiWorkerDecodeArenas {
    arenas: Vec<Option<KimiWorkerDecodeArena>>,
    vocab_rows: usize,
    dims: crate::config::KimiLocalDims,
    kv_pool_pages: usize,
}

impl KimiWorkerDecodeArenas {
    fn new(vocab_rows: usize, dims: &crate::config::KimiLocalDims, kv_pool_pages: usize) -> Self {
        let arenas = KIMI_DECODE_BATCH_BUCKETS.iter().map(|_| None).collect();
        Self {
            arenas,
            vocab_rows,
            dims: *dims,
            kv_pool_pages,
        }
    }

    fn get_mut(
        &mut self,
        ctx: &DeviceContext,
        decode_batch_size: usize,
    ) -> Result<&mut KimiWorkerDecodeArena> {
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        let (idx, arena_batch_size) = decode_batch_bucket(decode_batch_size)?;
        if self.arenas[idx].is_none() {
            self.arenas[idx] = Some(
                KimiWorkerDecodeArena::new(
                    ctx,
                    arena_batch_size,
                    self.kv_pool_pages,
                    self.vocab_rows,
                    &self.dims,
                )
                .with_context(|| {
                    format!(
                        "failed to allocate Kimi bs{arena_batch_size} decode arena for requested bs{decode_batch_size}"
                    )
                })?,
            );
        }
        self.arenas[idx]
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi bs{arena_batch_size} decode arena missing"))
    }
}

fn decode_batch_bucket(decode_batch_size: usize) -> Result<(usize, usize)> {
    ensure!(
        (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
        "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
    );
    KIMI_DECODE_BATCH_BUCKETS
        .iter()
        .copied()
        .enumerate()
        .find(|(_, bucket)| decode_batch_size <= *bucket)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Kimi decode batch size {decode_batch_size} has no arena bucket up to {KIMI_DECODE_MAX_BATCH}"
            )
        })
}

struct KimiOneTokenForwardCache {
    vocab_start: usize,
    vocab_rows: usize,
    token_embedding: GpuTensor<KIMI_K2_HIDDEN>,
    final_norm: NormWeight<KIMI_K2_HIDDEN>,
    lm_head: GpuTensor<KIMI_K2_HIDDEN>,
    layers: Vec<KimiLayerForwardCache>,
}

struct KimiLayerForwardCache {
    layer_idx: usize,
    attention: KimiAttentionForwardCache,
    kind: KimiLayerForwardKindCache,
}

struct KimiAttentionForwardCache {
    input_norm: NormWeight<KIMI_K2_HIDDEN>,
    fused_qkv_a_proj: GpuWeight<KIMI_K2_MLA_QKV_A_OUT, KIMI_K2_HIDDEN>,
    q_a_norm: NormWeight<KIMI_K2_Q_LORA_RANK>,
    q_b_proj: DeviceMatrix,
    kv_a_norm: NormWeight<KIMI_K2_MLA_KV_LORA_RANK>,
    kv_b_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    post_attention_norm: NormWeight<KIMI_K2_HIDDEN>,
}

enum KimiLayerForwardKindCache {
    Dense(KimiDenseForwardCache),
    Moe(KimiMoeForwardCache),
}

struct KimiDenseForwardCache {
    gate_up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

pub(super) struct KimiMoeForwardCache {
    pub(super) router: KimiRouterDeviceWeights,
    pub(super) shared_gate_up_proj: DeviceMatrix,
    pub(super) shared_down_proj: DeviceMatrix,
}

/// Per-bucket decode metadata + scratch. KV data itself lives in the rank's
/// shared [`KimiWorkerKvPool`]; the arena only carries the page-table device
/// buffers (fixed addresses, contents re-uploaded each step — CUDA Graph
/// replays stay valid) plus per-bucket compute scratch.
struct KimiWorkerDecodeArena {
    batch_size: usize,
    page_size: usize,
    pool_pages: usize,
    page_table_capacity: usize,
    append_capacity: usize,
    layout: KimiMlaPagedKvLayout,
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    batch_indices_d: CudaSlice<i32>,
    positions_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    token_ids_d: CudaSlice<u32>,
    cos_d: CudaSlice<half::bf16>,
    sin_d: CudaSlice<half::bf16>,
    scratch: KimiWorkerDecodeScratch,
    logits: HiddenStates,
    graph: CudaGraphState,
}

/// Rank-wide paged KV storage shared by every decode bucket and the prefill
/// path: one ckv + kpe buffer pair per layer, indexed by pool page IDs the
/// scheduler assigns. Allocated once at weight load (crash early on OOM);
/// the stable base pointers are what keep captured CUDA graphs valid.
struct KimiWorkerKvPool {
    layers: Vec<KimiWorkerMlaLayerCache>,
}

struct KimiWorkerMlaLayerCache {
    ckv_cache: CudaSlice<half::bf16>,
    kpe_cache: CudaSlice<half::bf16>,
}

struct KimiCublasThreadGuard;

impl Drop for KimiCublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            pegainfer_kernels::ffi::kimi_mla_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::kimi_o_proj_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::kimi_shared_gate_up_cublaslt_destroy_cuda();
            pegainfer_kernels::ffi::cublas_destroy();
        }
    }
}

fn bind_rank_thread(
    ctx: KimiRankGpuContext,
    weight_names: KimiRankWeightNames,
    sliced_load_plan: KimiRankSlicedLoadPlan,
    local_dims: crate::config::KimiLocalDims,
    collective_barrier: Arc<Barrier>,
    enable_cuda_graph: bool,
    kv_pool_pages: usize,
) -> Result<KimiRankThreadState> {
    ctx.set_current()?;
    let decode_aux_ctx = ctx.auxiliary_device_context("decode aux")?;
    unsafe {
        pegainfer_kernels::ffi::cublas_init();
        let status = pegainfer_kernels::ffi::kimi_shared_gate_up_cublaslt_init_cuda();
        if status != 0 {
            if status >= 100_000 {
                anyhow::bail!(
                    "Kimi shared_gate_up cuBLASLt init failed: cublas_status={}",
                    status - 100_000
                );
            }
            anyhow::bail!(
                "Kimi shared_gate_up cuBLASLt init failed: cuda_status={}",
                status
            );
        }
        let status = pegainfer_kernels::ffi::kimi_mla_cublaslt_init_cuda();
        if status != 0 {
            if status >= 100_000 {
                anyhow::bail!(
                    "Kimi MLA cuBLASLt init failed: cublas_status={}",
                    status - 100_000
                );
            }
            anyhow::bail!("Kimi MLA cuBLASLt init failed: cuda_status={}", status);
        }
        if local_dims.o_proj_in == KIMI_O_PROJ_CUBLASLT_INPUT {
            let status = pegainfer_kernels::ffi::kimi_o_proj_cublaslt_init_cuda();
            if status != 0 {
                if status >= 100_000 {
                    anyhow::bail!(
                        "Kimi o_proj cuBLASLt init failed: cublas_status={}",
                        status - 100_000
                    );
                }
                anyhow::bail!("Kimi o_proj cuBLASLt init failed: cuda_status={}", status);
            }
        }
    }
    Ok(KimiRankThreadState {
        ctx,
        decode_aux_ctx,
        _cublas: KimiCublasThreadGuard,
        tp_comm: None,
        weight_names,
        sliced_load_plan,
        local_dims,
        collective_barrier,
        enable_cuda_graph,
        kv_pool_pages,
        loaded: None,
        deepep: None,
    })
}

// The worker owns its command channel for the lifetime of the loop: taking
// `&Receiver` would leave the channel alive in the caller and break the
// "senders dropped → loop exits" shutdown signal.
#[allow(clippy::needless_pass_by_value)]
fn rank_worker_loop(rx: Receiver<KimiRankCommand>, mut state: KimiRankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            KimiRankCommand::LoadSlicedWeights { model_path, resp } => {
                let result = state.load_sliced_weights(&model_path);
                let _ = resp.send(result);
            }
            KimiRankCommand::InitTpComm {
                id,
                world_size,
                resp,
            } => {
                let result = state.init_tp_comm(id, world_size);
                let _ = resp.send(result);
            }
            KimiRankCommand::EnsureDecodeArena {
                decode_batch_size,
                resp,
            } => {
                let result = state.ensure_decode_arena(decode_batch_size);
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardPromptNextToken {
                slot,
                decode_batch_size,
                input_ids,
                cached_tokens,
                ep_max_seq_len,
                kv_pages,
                row,
                seed,
                resp,
            } => {
                let result = state.forward_prompt_next_token(
                    slot,
                    decode_batch_size,
                    &input_ids,
                    cached_tokens,
                    ep_max_seq_len,
                    &kv_pages,
                    row,
                    seed,
                );
                let _ = resp.send(result);
            }
            KimiRankCommand::ForwardDecodeBatchNextTokens {
                token_ids,
                append_positions,
                slots,
                decode_batch_size,
                kv_pages,
                rows,
                seed,
                resp,
            } => {
                let result = state.forward_decode_batch_next_tokens(
                    &token_ids,
                    &append_positions,
                    &slots,
                    decode_batch_size,
                    &kv_pages,
                    &rows,
                    seed,
                );
                let _ = resp.send(result);
            }
            KimiRankCommand::EnableDeepEp {
                unique_id,
                num_ranks,
                resp,
            } => {
                let result = state.enable_deepep(&unique_id, num_ranks);
                let _ = resp.send(result);
            }
            KimiRankCommand::Shutdown => break,
        }
    }
}

mod cache;
mod load;
mod runtime;
// Collective + Marlin helpers shared with the sibling `moe_nccl` backend.
pub(super) use runtime::all_reduce_f32_in_place;
pub(super) use runtime::kimi_marlin_block_size;
pub(super) use runtime::maybe_all_reduce_hidden_via_f32_in_place;
pub(super) use runtime::reduce_scatter_f32_hidden_into;
mod forward;
mod state;

impl KimiRankWeightLoadReport {
    fn from_loaded_weights(
        tensor_count: usize,
        total_bytes: usize,
        expert_kernel_weights: &KimiRankExpertMarlinWeights,
    ) -> Self {
        Self {
            rank: expert_kernel_weights.rank,
            tensor_count,
            total_bytes,
            expert_kernel_layers: expert_kernel_weights.layers.len(),
            expert_kernel_total_bytes: expert_kernel_weights.total_bytes,
            loaded_to_gpu: true,
            typed_view_validated: true,
            expert_kernel_weights_packaged: true,
        }
    }
}

pub(super) fn build_placements(device_ordinals: &[usize]) -> Result<Vec<KimiK2RankPlacement>> {
    ensure!(
        !device_ordinals.is_empty(),
        "Kimi-K2 requires at least one device ordinal"
    );
    device_ordinals
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, device_ordinal)| KimiK2RankPlacement::new(rank, device_ordinal))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::KimiKvStepPages;
    use super::cache::build_slot_page_table;
    use super::decode_batch_bucket;

    #[test]
    fn decode_batch_bucket_rounds_up_to_power_of_two_buckets() {
        let cases = [
            (1, (0, 1)),
            (2, (1, 2)),
            (3, (2, 4)),
            (4, (2, 4)),
            (5, (3, 8)),
            (17, (5, 32)),
            (33, (6, 64)),
            (64, (6, 64)),
        ];
        for (requested, expected) in cases {
            assert_eq!(decode_batch_bucket(requested).unwrap(), expected);
        }
    }

    #[test]
    fn decode_batch_bucket_rejects_out_of_range_sizes() {
        assert!(decode_batch_bucket(0).is_err());
        assert!(decode_batch_bucket(65).is_err());
    }

    #[test]
    fn slot_page_table_scatters_rows_and_pads_idle_slots() {
        // Row 0 → slot 2 with 26 KV tokens (2 pool pages); row 1 → slot 0
        // with 1 token. Slots 1 and 3 are idle → padding page 7.
        let kv_pages = KimiKvStepPages::new(vec![vec![40, 41], vec![9]], 7);
        let (page_indices, page_indptr, last_page_len) =
            build_slot_page_table(4, 16, &kv_pages, &[2, 0], &[26, 1]).unwrap();
        assert_eq!(page_indices, vec![9, 7, 40, 41, 7]);
        assert_eq!(page_indptr, vec![0, 1, 2, 4, 5]);
        assert_eq!(last_page_len, vec![1, 1, 10, 1]);
    }

    #[test]
    fn slot_page_table_last_page_len_tracks_page_boundary() {
        // 16 tokens fill one page exactly; 17 spill one token into a second.
        let full = KimiKvStepPages::single(vec![3], 0);
        let (_, _, last_page_len) = build_slot_page_table(1, 16, &full, &[0], &[16]).unwrap();
        assert_eq!(last_page_len, vec![16]);

        let spill = KimiKvStepPages::single(vec![3, 4], 0);
        let (_, _, last_page_len) = build_slot_page_table(1, 16, &spill, &[0], &[17]).unwrap();
        assert_eq!(last_page_len, vec![1]);
    }

    #[test]
    fn slot_page_table_rejects_page_count_mismatch() {
        // 26 tokens need 2 pages; offering 1 or 3 must fail loudly — the
        // scheduler's block accounting and the kernel view must agree.
        let short = KimiKvStepPages::single(vec![40], 0);
        assert!(build_slot_page_table(2, 16, &short, &[0], &[26]).is_err());
        let long = KimiKvStepPages::single(vec![40, 41, 42], 0);
        assert!(build_slot_page_table(2, 16, &long, &[0], &[26]).is_err());
    }

    #[test]
    fn slot_page_table_rejects_out_of_range_slot_and_row_mismatch() {
        let kv_pages = KimiKvStepPages::single(vec![1], 0);
        assert!(build_slot_page_table(2, 16, &kv_pages, &[2], &[1]).is_err());
        assert!(build_slot_page_table(2, 16, &kv_pages, &[0, 1], &[1, 1]).is_err());
    }
}
