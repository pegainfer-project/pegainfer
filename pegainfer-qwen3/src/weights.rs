use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use cudarc::nccl::safe::Comm;
use cudarc::nccl::safe::ReduceOp;
use half::bf16;
use log::debug;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kv_cache::KvBuffer;

use super::config::Config;
use super::config::TensorParallelConfig;
use crate::batch_decode_buffers::BatchDecodeBuffers;
use crate::lora::DeviceLoraAdapter;
use crate::lora::DeviceLoraLayer;
use crate::lora::DeviceLoraProjection;
use crate::lora::DeviceLoraTokenGroup;
use crate::lora::LoraProjectionKind;
use crate::lora::apply_lora_projection_delta_indexed;
use crate::lora::apply_lora_projection_delta_range;

mod load;
pub(crate) use load::ModelRuntimeConfig;

pub(crate) const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.90;
pub const DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES: usize = 150 * 1024 * 1024;
/// Default KV cache page (block) size in tokens.
pub const DEFAULT_KV_PAGE_SIZE: usize = 16;
/// Page sizes FlashInfer's paged attention kernels accept (see #545).
const VALID_KV_PAGE_SIZES: &[usize] = &[16, 64];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3MemoryOptions {
    /// Mirrors vLLM's `gpu_memory_utilization`: the KV pool gets what remains
    /// inside this requested budget after weights, profiled non-KV runtime
    /// memory, and a small safety margin are accounted for.
    gpu_memory_utilization: f64,
    /// Extra bytes held back after the profile result to cover allocator
    /// fragmentation and small unprofiled runtime drift.
    pub(crate) kv_cache_memory_margin_bytes: usize,
    /// KV cache page (block) size in tokens (`--kv-page-size`). FlashInfer
    /// constrains this to [`VALID_KV_PAGE_SIZES`]; 16 by default.
    pub(crate) page_size: usize,
}

impl Qwen3MemoryOptions {
    pub const fn new(
        gpu_memory_utilization: f64,
        kv_cache_memory_margin_bytes: usize,
        page_size: usize,
    ) -> Self {
        Self {
            gpu_memory_utilization,
            kv_cache_memory_margin_bytes,
            page_size,
        }
    }

    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.gpu_memory_utilization > 0.0 && self.gpu_memory_utilization <= 1.0,
            "gpu_memory_utilization must be in (0, 1], got {}",
            self.gpu_memory_utilization
        );
        anyhow::ensure!(
            VALID_KV_PAGE_SIZES.contains(&self.page_size),
            "page_size must be one of {:?}, got {}",
            VALID_KV_PAGE_SIZES,
            self.page_size
        );
        Ok(self)
    }
}

impl Default for Qwen3MemoryOptions {
    fn default() -> Self {
        Self {
            gpu_memory_utilization: DEFAULT_GPU_MEMORY_UTILIZATION,
            kv_cache_memory_margin_bytes: DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            page_size: DEFAULT_KV_PAGE_SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KvBudget {
    pub(crate) num_layers: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) block_size: usize,
    pub(crate) num_blocks: usize,
}

/// Attention layer weights.
/// QKV stored as a single concatenated matrix [q_dim + 2*kv_dim, hidden_size].
/// Individual projections accessed via row offsets (zero extra memory).
pub(super) struct Attention {
    /// Fused [q_proj; k_proj; v_proj] row-major
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
    pub(super) q_dim: usize,
    pub(super) kv_dim: usize,
}

/// MLP layer weights.
/// Gate+Up stored as a single concatenated matrix [2*intermediate_size, hidden_size].
#[allow(clippy::upper_case_acronyms, clippy::struct_field_names)]
pub(super) struct MLP {
    /// Fused [gate_proj; up_proj] row-major
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP,
}

pub(crate) struct PackedLoraProjection {
    pub(crate) a: cudarc::driver::CudaSlice<bf16>,
    pub(crate) b: cudarc::driver::CudaSlice<bf16>,
    pub(crate) scales: cudarc::driver::CudaSlice<f32>,
    pub(crate) max_loras: usize,
    pub(crate) max_rank: usize,
    pub(crate) rank: usize,
    in_dim: usize,
    pub(crate) out_dim: usize,
    slot_ranks: Vec<usize>,
}

impl PackedLoraProjection {
    fn new(
        ctx: &DeviceContext,
        max_loras: usize,
        max_rank: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let a_elems = max_loras * max_rank * in_dim;
        let b_elems = max_loras * out_dim * max_rank;
        let a = ctx
            .stream
            .alloc_zeros(a_elems)
            .map_err(|e| anyhow::anyhow!("packed LoRA A alloc failed: {e}"))?;
        let b = ctx
            .stream
            .alloc_zeros(b_elems)
            .map_err(|e| anyhow::anyhow!("packed LoRA B alloc failed: {e}"))?;
        let scales = ctx
            .stream
            .alloc_zeros(max_loras)
            .map_err(|e| anyhow::anyhow!("packed LoRA scales alloc failed: {e}"))?;
        Ok(Self {
            a,
            b,
            scales,
            max_loras,
            max_rank,
            rank: 0,
            in_dim,
            out_dim,
            slot_ranks: vec![0; max_loras],
        })
    }

    fn write_slot(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        projection: &DeviceLoraProjection,
        scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(slot < self.max_loras, "packed LoRA slot out of range");
        anyhow::ensure!(
            projection.a.rows <= self.max_rank && projection.a.cols == self.in_dim,
            "inconsistent LoRA A shape in packed projection"
        );
        anyhow::ensure!(
            projection.b.rows == self.out_dim && projection.b.cols == projection.a.rows,
            "inconsistent LoRA B shape in packed projection"
        );
        self.clear_slot(ctx, slot)?;

        let rank = projection.a.rows;
        let a_src = projection.a.data.slice(..rank * self.in_dim);
        let a_offset = slot * self.max_rank * self.in_dim;
        let mut a_dst = self.a.slice_mut(a_offset..a_offset + rank * self.in_dim);
        ctx.stream
            .memcpy_dtod(&a_src, &mut a_dst)
            .map_err(|e| anyhow::anyhow!("packed LoRA A copy failed: {e}"))?;

        let b_offset = slot * self.out_dim * self.max_rank;
        pegainfer_core::ops::pack_lora_b_rows_into(
            ctx,
            &projection.b.data,
            &mut self.b,
            b_offset,
            rank,
            self.max_rank,
            self.out_dim,
        )
        .map_err(|e| anyhow::anyhow!("packed LoRA B copy failed: {e}"))?;

        let mut scale_slot = self.scales.slice_mut(slot..=slot);
        ctx.stream
            .memcpy_htod(&[scale], &mut scale_slot)
            .map_err(|e| anyhow::anyhow!("packed LoRA scale copy failed: {e}"))?;
        self.slot_ranks[slot] = rank;
        self.refresh_rank();
        Ok(())
    }

    fn clear_slot(&mut self, ctx: &DeviceContext, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.max_loras, "packed LoRA slot out of range");

        let zero_a = vec![bf16::ZERO; self.max_rank * self.in_dim];
        let a_offset = slot * self.max_rank * self.in_dim;
        let mut a_dst = self
            .a
            .slice_mut(a_offset..a_offset + self.max_rank * self.in_dim);
        ctx.stream
            .memcpy_htod(&zero_a, &mut a_dst)
            .map_err(|e| anyhow::anyhow!("packed LoRA A clear failed: {e}"))?;

        let zero_b = vec![bf16::ZERO; self.out_dim * self.max_rank];
        let b_offset = slot * self.out_dim * self.max_rank;
        let mut b_dst = self
            .b
            .slice_mut(b_offset..b_offset + self.out_dim * self.max_rank);
        ctx.stream
            .memcpy_htod(&zero_b, &mut b_dst)
            .map_err(|e| anyhow::anyhow!("packed LoRA B clear failed: {e}"))?;

        let mut scale_slot = self.scales.slice_mut(slot..=slot);
        ctx.stream
            .memcpy_htod(&[0.0f32], &mut scale_slot)
            .map_err(|e| anyhow::anyhow!("packed LoRA scale clear failed: {e}"))?;
        self.slot_ranks[slot] = 0;
        self.refresh_rank();
        Ok(())
    }

    fn refresh_rank(&mut self) {
        self.rank = self.slot_ranks.iter().copied().max().unwrap_or(0);
    }
}

pub(crate) struct PackedLoraLayer {
    projections: Vec<Option<PackedLoraProjection>>,
}

impl PackedLoraLayer {
    fn empty() -> Self {
        Self {
            projections: (0..LoraProjectionKind::ALL.len())
                .map(|_| None)
                .collect::<Vec<Option<PackedLoraProjection>>>(),
        }
    }

    fn projection(&self, kind: LoraProjectionKind) -> Option<&PackedLoraProjection> {
        self.projections
            .get(kind.index())
            .and_then(Option::as_ref)
            .filter(|projection| projection.rank > 0)
    }
}

pub(crate) struct PackedLoraRegistry {
    slots_by_name: HashMap<String, usize>,
    slot_names: Vec<Option<String>>,
    packed_layers: Vec<PackedLoraLayer>,
}

impl PackedLoraRegistry {
    fn empty(max_loras: usize, num_layers: usize) -> Self {
        Self {
            slots_by_name: HashMap::new(),
            slot_names: vec![None; max_loras],
            packed_layers: (0..num_layers).map(|_| PackedLoraLayer::empty()).collect(),
        }
    }

    fn slot_for(&self, name: &str) -> Option<usize> {
        self.slots_by_name.get(name).copied()
    }

    fn layer(&self, layer_idx: usize) -> Option<&PackedLoraLayer> {
        self.packed_layers.get(layer_idx)
    }

    fn slot_for_install(&self, name: &str, load_inplace: bool) -> Result<usize> {
        if let Some(slot) = self.slot_for(name) {
            anyhow::ensure!(load_inplace, "Qwen3 LoRA adapter {name} is already loaded");
            return Ok(slot);
        }
        self.slot_names
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| anyhow::anyhow!("Qwen3 LoRA adapter capacity exceeded"))
    }

    fn bind_slot(&mut self, slot: usize, name: &str) {
        if let Some(previous_name) = self.slot_names[slot].replace(name.to_string()) {
            self.slots_by_name.remove(&previous_name);
        }
        self.slots_by_name.insert(name.to_string(), slot);
    }

    fn release_slot(&mut self, name: &str) -> Result<usize> {
        let slot = self
            .slots_by_name
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("Qwen3 LoRA adapter {name} is not loaded"))?;
        self.slot_names[slot] = None;
        Ok(slot)
    }
}

/// Qwen3 model — weights and config only. Request state is owned by the executor.
pub(crate) struct Qwen3Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config,
    pub(super) embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
    pub(super) tensor_parallel: TensorParallelConfig,
    pub(super) decode_projection_path: crate::projection_fusion::DecodeProjectionPath,
    tp_comm: Option<Comm>,
    lora_adapters: HashMap<String, DeviceLoraAdapter>,
    packed_lora: PackedLoraRegistry,
    max_loras: usize,
    max_lora_rank: usize,
}

// SAFETY: Each model instance is pinned to a single CUDA device and is only
// driven from one worker thread at a time. The TP path creates one model per
// rank and never shares a single rank-local model concurrently across threads.
unsafe impl Send for Qwen3Model {}
unsafe impl Sync for Qwen3Model {}

impl Qwen3Model {
    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn device_ctx(&self) -> &pegainfer_core::tensor::DeviceContext {
        &self.ctx
    }

    pub(crate) fn local_num_attention_heads(&self) -> usize {
        self.config.local_num_attention_heads(self.tensor_parallel)
    }

    pub(crate) fn local_num_key_value_heads(&self) -> usize {
        self.config.local_num_key_value_heads(self.tensor_parallel)
    }

    pub(crate) fn local_intermediate_size(&self) -> usize {
        self.config.local_intermediate_size(self.tensor_parallel)
    }

    pub(crate) fn local_q_dim(&self) -> usize {
        self.config.local_q_dim(self.tensor_parallel)
    }

    pub(crate) fn local_kv_dim(&self) -> usize {
        self.config.local_kv_dim(self.tensor_parallel)
    }

    pub(crate) const fn fused_decode_qkv(&self) -> bool {
        self.decode_projection_path.fuses_qkv()
    }

    pub(crate) fn attach_tp_comm(&mut self, comm: Comm) {
        self.tp_comm = Some(comm);
    }

    /// Whether decode steps replay pre-captured graphs that record NCCL
    /// collectives — i.e. CUDA Graph on a tensor-parallel model.
    pub(crate) fn tp_graph_enabled(&self) -> bool {
        self.enable_cuda_graph && self.tp_comm.is_some()
    }

    /// Force NCCL connect before any capture records a collective (lazy connect
    /// inside `cuStreamBeginCapture` is the classic hazard). NCCL >= 2.22
    /// connects per size-selected algorithm, so warm one all-reduce at every
    /// bucket's message size. No-op without a TP communicator.
    pub(crate) fn warmup_tp_collective(&self) -> Result<()> {
        if let Some(comm) = &self.tp_comm {
            let buckets = crate::batch_decode_buffers::BATCH_BUCKETS;
            let max_elems = buckets.last().unwrap() * self.config.hidden_size;
            let mut scratch = self.ctx.stream.alloc_zeros::<bf16>(max_elems)?;
            for &bucket in buckets {
                let mut view = scratch.slice_mut(0..bucket * self.config.hidden_size);
                comm.all_reduce_in_place(&mut view, &ReduceOp::Sum)
                    .map_err(|e| anyhow::anyhow!("nccl warm-up all-reduce failed: {e:?}"))?;
            }
            self.ctx.stream.synchronize()?;
        }
        Ok(())
    }

    pub(crate) fn install_lora_adapter(
        &mut self,
        adapter: DeviceLoraAdapter,
        load_inplace: bool,
    ) -> Result<()> {
        debug!(
            "Installing Qwen3 LoRA adapter {} from {}",
            adapter.name,
            adapter.manifest.path.display()
        );
        let name = adapter.name.clone();
        let slot = self.packed_lora.slot_for_install(&name, load_inplace)?;
        if let Err(err) = self.update_packed_lora_slot(slot, &adapter) {
            self.lora_adapters.remove(&name);
            if self.packed_lora.slot_for(&name) == Some(slot) {
                let _ = self.packed_lora.release_slot(&name);
            }
            return Err(err).with_context(|| {
                format!(
                    "failed to update packed LoRA slot {slot} for adapter {name}; adapter was removed to keep packed decode state consistent"
                )
            });
        }
        install_lora_adapter_in_registry(&mut self.lora_adapters, adapter, load_inplace)?;
        self.packed_lora.bind_slot(slot, &name);
        Ok(())
    }

    pub(crate) fn uninstall_lora_adapter(&mut self, name: &str) -> Result<()> {
        let slot = self
            .packed_lora
            .slot_for(name)
            .ok_or_else(|| anyhow::anyhow!("Qwen3 LoRA adapter {name} is not loaded"))?;
        self.clear_packed_lora_slot(slot)?;
        self.packed_lora.release_slot(name)?;
        self.lora_adapters
            .remove(name)
            .expect("packed LoRA slot map and adapter registry must be consistent");
        Ok(())
    }

    pub(crate) fn discard_lora_adapter(&mut self, name: &str) -> Result<()> {
        if self.packed_lora.slot_for(name).is_some() {
            self.packed_lora.release_slot(name)?;
        }
        self.lora_adapters.remove(name);
        Ok(())
    }

    fn lora_layer_for(&self, name: &str, layer_idx: usize) -> Option<(&DeviceLoraLayer, f32)> {
        self.lora_adapters.get(name).and_then(|adapter| {
            adapter
                .layers
                .get(layer_idx)
                .map(|layer| (layer, adapter.scale))
        })
    }

    pub(crate) fn apply_lora_projection_ranges(
        &self,
        layer_idx: usize,
        groups: &[DeviceLoraTokenGroup<'_>],
        projection: impl for<'a> Fn(&'a DeviceLoraLayer) -> Option<&'a DeviceLoraProjection>,
        input: &HiddenStates,
        out: &mut HiddenStates,
        row_offset: usize,
    ) -> Result<()> {
        for group in groups {
            let Some((layer, scale)) = self.lora_layer_for(group.adapter, layer_idx) else {
                anyhow::bail!("Qwen3 LoRA adapter {} is not loaded", group.adapter);
            };
            if let Some(projection) = projection(layer) {
                if group.ranges.len() == 1 {
                    let range = group.ranges[0];
                    apply_lora_projection_delta_range(
                        &self.ctx,
                        projection,
                        input,
                        out,
                        row_offset,
                        range.token_offset,
                        range.token_len,
                        scale,
                    )?;
                } else {
                    let token_indices_d = group
                        .token_indices_d
                        .as_ref()
                        .expect("non-contiguous LoRA token group must have device indices");
                    apply_lora_projection_delta_indexed(
                        &self.ctx,
                        projection,
                        input,
                        out,
                        row_offset,
                        token_indices_d,
                        group.token_count,
                        scale,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn decode_lora_slots(&self, adapters: &[Option<&str>]) -> Result<Option<Vec<i32>>> {
        let mut slots = Vec::with_capacity(adapters.len());
        let mut any_lora = false;
        for adapter in adapters {
            match adapter {
                Some(name) => {
                    let Some(slot) = self.packed_lora.slot_for(name) else {
                        anyhow::bail!("Qwen3 LoRA adapter {name} is not loaded");
                    };
                    slots.push(slot as i32);
                    any_lora = true;
                }
                None => slots.push(-1),
            }
        }
        Ok(any_lora.then_some(slots))
    }

    pub(crate) fn packed_lora_projection(
        &self,
        layer_idx: usize,
        kind: LoraProjectionKind,
    ) -> Option<&PackedLoraProjection> {
        self.packed_lora
            .layer(layer_idx)
            .and_then(|layer| layer.projection(kind))
    }

    fn update_packed_lora_slot(&mut self, slot: usize, adapter: &DeviceLoraAdapter) -> Result<()> {
        self.clear_packed_lora_slot(slot)?;
        for layer_idx in 0..self.config.num_hidden_layers {
            let Some(layer) = adapter.layers.get(layer_idx) else {
                continue;
            };
            for kind in LoraProjectionKind::ALL {
                let Some(projection) = layer.projection(kind) else {
                    continue;
                };
                self.pack_lora_projection_slot(slot, projection, adapter.scale, layer_idx, kind)?;
            }
        }
        Ok(())
    }

    fn clear_packed_lora_projection_slot(
        &mut self,
        slot: usize,
        layer_idx: usize,
        kind: LoraProjectionKind,
    ) -> Result<()> {
        if let Some(packed) =
            self.packed_lora.packed_layers[layer_idx].projections[kind.index()].as_mut()
        {
            packed.clear_slot(&self.ctx, slot)?;
        }
        Ok(())
    }

    fn clear_packed_lora_slot(&mut self, slot: usize) -> Result<()> {
        for layer_idx in 0..self.config.num_hidden_layers {
            for kind in LoraProjectionKind::ALL {
                self.clear_packed_lora_projection_slot(slot, layer_idx, kind)?;
            }
        }
        Ok(())
    }

    fn pack_lora_projection_slot(
        &mut self,
        slot: usize,
        projection: &DeviceLoraProjection,
        scale: f32,
        layer_idx: usize,
        kind: LoraProjectionKind,
    ) -> Result<()> {
        let max_loras = self.max_loras;
        let max_rank = self.max_lora_rank;
        let packed_slot = &mut self.packed_lora.packed_layers[layer_idx].projections[kind.index()];
        if packed_slot.is_none() {
            *packed_slot = Some(PackedLoraProjection::new(
                &self.ctx,
                max_loras,
                max_rank,
                projection.a.cols,
                projection.b.rows,
            )?);
        }
        let packed = packed_slot
            .as_mut()
            .expect("packed LoRA projection was initialized");
        packed.write_slot(&self.ctx, slot, projection, scale)
    }

    pub(crate) fn all_reduce_hidden(
        &self,
        hidden: &mut pegainfer_core::tensor::HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        if pegainfer_core::ops::call_trace::is_enabled() {
            let label = pegainfer_core::ops::call_trace::current_label("all_reduce_hidden");
            pegainfer_core::ops::call_trace::record_call(
                pegainfer_core::ops::call_spec::all_reduce_hidden_call(
                    label,
                    hidden.hidden_dim,
                    hidden.seq_len,
                ),
            );
        }
        self.all_reduce_hidden_untraced(hidden)
    }

    pub(crate) fn all_reduce_hidden_untraced(
        &self,
        hidden: &mut pegainfer_core::tensor::HiddenStates,
    ) -> Result<()> {
        if let Some(comm) = &self.tp_comm {
            comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
                .map_err(|e| anyhow::anyhow!("nccl all-reduce failed: {e:?}"))?;
        }
        Ok(())
    }

    /// KV cache geometry and budget for kernel-call tracing.
    #[cfg(feature = "kernel-call-trace")]
    pub(crate) fn kv_budget(&self) -> KvBudget {
        let geometry = self.kv_budget_geometry(DEFAULT_KV_PAGE_SIZE);
        let bytes_per_block = Self::kv_bytes_per_block(&geometry).expect("KV bytes per block");
        let (free_bytes, _) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo failed");
        let kv_budget_bytes = (free_bytes as f64 * 0.85) as usize;
        Self::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            0,
            kv_budget_bytes,
            free_bytes,
            "heuristic",
        )
        .expect("KV budget")
    }

    pub(crate) fn profiled_kv_budget(
        &self,
        max_prefill_tokens: usize,
        max_decode_batch_size: usize,
        dflash_kv_bytes_per_token: usize,
        memory_options: Qwen3MemoryOptions,
    ) -> Result<KvBudget> {
        let memory_options = memory_options.validate()?;
        let geometry = self.kv_budget_geometry(memory_options.page_size);
        let bytes_per_block = Self::kv_bytes_per_block(&geometry)?;
        let (initial_free_bytes, total_bytes) = mem_info_bytes()?;
        let requested_bytes =
            (total_bytes as f64 * memory_options.gpu_memory_utilization).ceil() as usize;
        let initial_used_bytes = total_bytes.saturating_sub(initial_free_bytes);
        anyhow::ensure!(
            initial_used_bytes < requested_bytes,
            "Qwen3 requested GPU memory is already exhausted before KV allocation: \
             used={} MiB, requested={} MiB (utilization {:.2})",
            initial_used_bytes / (1024 * 1024),
            requested_bytes / (1024 * 1024),
            memory_options.gpu_memory_utilization
        );

        anyhow::ensure!(
            max_prefill_tokens > 0,
            "Qwen3 memory profile requires positive max_prefill_tokens"
        );
        let profile_prefill_rows = 1;
        let profile_decode_rows = max_decode_batch_size.saturating_sub(profile_prefill_rows);
        anyhow::ensure!(
            profile_decode_rows > 0,
            "Qwen3 memory profile requires decode capacity above prefill rows"
        );
        let profile_rows = profile_prefill_rows + profile_decode_rows;
        let profile_blocks =
            profile_temp_blocks(max_prefill_tokens, profile_decode_rows, geometry.block_size);
        let profile_kv_bytes = crate::sizing::product(&[profile_blocks, bytes_per_block])?;
        let mut peak_used_bytes = initial_used_bytes;
        let mut record_peak = || -> Result<()> {
            let (free_bytes, total_bytes) = mem_info_bytes()?;
            peak_used_bytes = peak_used_bytes.max(total_bytes.saturating_sub(free_bytes));
            Ok(())
        };

        let profile_kv = KvBuffer::new(
            &self.ctx.stream,
            geometry.num_layers,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.block_size,
            profile_blocks,
        )
        .context("Qwen3 memory profile temp KV alloc failed")?;
        record_peak()?;

        let mut decode_bufs = BatchDecodeBuffers::new(
            self.device_ctx(),
            self.config.hidden_size,
            self.local_q_dim(),
            self.local_kv_dim(),
            self.local_intermediate_size(),
            self.config.vocab_size,
            max_decode_batch_size,
            geometry.block_size,
            0,
            self.local_num_attention_heads(),
            self.config.max_position_embeddings,
            self.fused_decode_qkv(),
        )
        .context("Qwen3 memory profile decode buffer alloc failed")?;
        record_peak()?;

        let mut sample_scratch = pegainfer_sample::SampleScratch::new(
            self.device_ctx(),
            self.config.vocab_size,
            profile_rows,
        )
        .context("Qwen3 memory profile sampling scratch alloc failed")?;
        record_peak()?;

        self.profile_unified_step_memory(
            max_prefill_tokens,
            profile_decode_rows,
            &profile_kv,
            &mut decode_bufs,
            &mut sample_scratch,
            &mut record_peak,
        )?;
        record_peak()?;

        // `peak_used_bytes` includes the temporary KV buffer used only to make
        // the dummy step legal. The final KV pool is sized separately below, so
        // remove that profile-only backing store from the measured non-KV peak.
        let profile_peak_increase = peak_used_bytes.saturating_sub(initial_used_bytes);
        let non_kv_peak_increase = profile_peak_increase.saturating_sub(profile_kv_bytes);
        let non_kv_bytes = initial_used_bytes
            .saturating_add(non_kv_peak_increase)
            .saturating_add(memory_options.kv_cache_memory_margin_bytes);
        anyhow::ensure!(
            requested_bytes > non_kv_bytes,
            "Qwen3 memory profile leaves no room for KV cache: requested={} MiB, \
             non_kv={} MiB, margin={} MiB",
            requested_bytes / (1024 * 1024),
            non_kv_bytes / (1024 * 1024),
            memory_options.kv_cache_memory_margin_bytes / (1024 * 1024)
        );
        let kv_budget_bytes = requested_bytes - non_kv_bytes;
        log::info!(
            "memory profile: total={} MiB requested={} MiB ({:.0}%) initial_used={} MiB \
             peak_non_kv_increase={} MiB margin={} MiB -> KV budget={} MiB",
            total_bytes / (1024 * 1024),
            requested_bytes / (1024 * 1024),
            memory_options.gpu_memory_utilization * 100.0,
            initial_used_bytes / (1024 * 1024),
            non_kv_peak_increase / (1024 * 1024),
            memory_options.kv_cache_memory_margin_bytes / (1024 * 1024),
            kv_budget_bytes / (1024 * 1024),
        );
        Self::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            dflash_kv_bytes_per_token,
            kv_budget_bytes,
            initial_free_bytes,
            "profiled",
        )
    }

    /// Bytes one KV page costs in the pool, for a given page size. The single
    /// owner of this formula — callers that bill KV pages (hedge scratch,
    /// budgets) must not re-derive it.
    pub(crate) fn kv_page_bytes(&self, page_size: usize) -> Result<usize> {
        Self::kv_bytes_per_block(&self.kv_budget_geometry(page_size))
    }

    fn kv_budget_geometry(&self, page_size: usize) -> KvBudget {
        let num_kv_heads = self.local_num_key_value_heads();
        KvBudget {
            num_layers: self.config.num_hidden_layers,
            num_kv_heads,
            head_dim: self.config.head_dim,
            block_size: page_size,
            num_blocks: 0,
        }
    }

    fn kv_bytes_per_block(geometry: &KvBudget) -> Result<usize> {
        // Checked BEFORE constructing the layout: `KvLayout::new` derives its
        // strides with plain products, and this full product dominates every
        // partial one, so clearing it clears them all.
        let bytes = crate::sizing::product(&[
            geometry.num_layers,
            2,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.block_size,
            std::mem::size_of::<half::bf16>(),
        ])?;
        let layout = pegainfer_kv_cache::KvLayout::new(
            geometry.num_layers,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.block_size,
        );
        debug_assert_eq!(
            bytes,
            layout.page_stride * std::mem::size_of::<half::bf16>()
        );
        Ok(bytes)
    }

    /// Smallest pool the scheduler can make progress with.
    const MIN_KV_BLOCKS: usize = 64;

    fn kv_budget_from_bytes(
        mut geometry: KvBudget,
        bytes_per_block: usize,
        dflash_kv_bytes_per_token: usize,
        kv_budget_bytes: usize,
        free_bytes: usize,
        source: &'static str,
    ) -> Result<KvBudget> {
        // DFlash keeps its own per-request KV (plus prompt-scaling scratch) outside
        // the paged pool, scaling with the same token count. Charge it as extra
        // bytes per pool token so the target block count shrinks to leave room; the
        // pool itself is still allocated at the target-only `bytes_per_block`.
        //
        // Checked: the draft config is not shape-validated against the draft
        // weights until the model loads, which is after this budget is fixed.
        let effective_bytes_per_block = crate::sizing::sum(&[
            bytes_per_block,
            crate::sizing::product(&[dflash_kv_bytes_per_token, geometry.block_size])?,
        ])?;
        anyhow::ensure!(
            effective_bytes_per_block > 0,
            "KV budget ({source}): degenerate geometry gives zero bytes per block"
        );
        // Against `effective`, not `bytes_per_block`: a budget that affords 64
        // target blocks can still be short once the draft's out-of-pool KV is
        // charged, and forcing the floor anyway defers the shortfall into a
        // draft-growth OOM at serving time.
        let affordable = kv_budget_bytes / effective_bytes_per_block;
        anyhow::ensure!(
            affordable >= Self::MIN_KV_BLOCKS,
            "KV cache ({source}) affords only {affordable} blocks, below the \
             {}-block minimum: budget={} MiB, needed={} MiB \
             ({} KiB per block = {} target + {} draft)",
            Self::MIN_KV_BLOCKS,
            kv_budget_bytes / (1024 * 1024),
            crate::sizing::product(&[Self::MIN_KV_BLOCKS, effective_bytes_per_block])?
                / (1024 * 1024),
            effective_bytes_per_block / 1024,
            bytes_per_block / 1024,
            (effective_bytes_per_block - bytes_per_block) / 1024,
        );
        let num_blocks = affordable;
        // Saturating, not checked: log-only figures must not abort startup.
        // Reported on the same basis the blocks were derived from — pool and
        // draft split out, since only the pool shows up in the KV allocation.
        let pool_mb = num_blocks.saturating_mul(bytes_per_block) / (1024 * 1024);
        let draft_mb =
            num_blocks.saturating_mul(effective_bytes_per_block - bytes_per_block) / (1024 * 1024);
        let spent = num_blocks.saturating_mul(effective_bytes_per_block);
        let page_size = geometry.block_size;
        log::info!(
            "KV cache ({source}): {num_blocks} blocks (pool {pool_mb} MB + draft {draft_mb} MB, \
             page size {page_size}, {:.0}% of {:.0} MB free)",
            spent as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        geometry.num_blocks = num_blocks;
        Ok(geometry)
    }
}

fn mem_info_bytes() -> Result<(usize, usize)> {
    let (free, total) = cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("cuMemGetInfo failed: {e:?}"))?;
    Ok((free, total))
}

fn profile_temp_blocks(
    max_prefill_tokens: usize,
    profile_decode_rows: usize,
    block_size: usize,
) -> usize {
    // Block 0 is reserved as the decode padding block. Worst-case scheduling can
    // run one max-sized prefill row plus the remaining decode rows.
    1 + max_prefill_tokens.div_ceil(block_size) + profile_decode_rows
}

fn install_lora_adapter_in_registry(
    lora_adapters: &mut HashMap<String, DeviceLoraAdapter>,
    adapter: DeviceLoraAdapter,
    load_inplace: bool,
) -> Result<()> {
    if !load_inplace {
        anyhow::ensure!(
            !lora_adapters.contains_key(&adapter.name),
            "Qwen3 LoRA adapter {} is already loaded",
            adapter.name
        );
    }
    lora_adapters.insert(adapter.name.clone(), adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::DeviceLoraLayer;
    use crate::lora::LoraAdapterManifest;

    /// The draft config drives `dflash_kv_bytes_per_token`, and it is not
    /// shape-validated against the draft weights until the model loads — after
    /// this budget is fixed. A malformed value must fail the budget instead of
    /// wrapping into an under-reservation.
    #[test]
    fn a_malformed_draft_config_fails_the_kv_budget_instead_of_wrapping() {
        let geometry = KvBudget {
            num_layers: 36,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            num_blocks: 0,
        };
        let bytes_per_block = Qwen3Model::kv_bytes_per_block(&geometry).expect("sane geometry");
        let sane = Qwen3Model::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            65_536,
            8 << 30,
            16 << 30,
            "test",
        )
        .expect("sane draft config budgets");
        assert!(sane.num_blocks >= 64);

        let err = Qwen3Model::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            usize::MAX / 8,
            8 << 30,
            16 << 30,
            "test",
        )
        .expect_err("overflowing draft bytes-per-token must fail closed");
        assert!(
            err.to_string().contains("overflow"),
            "expected a sizing-overflow error, got: {err}"
        );
    }

    /// The pool costs `bytes_per_block`, but each pool block also drags the
    /// draft's out-of-pool KV along. A budget that affords the minimum in
    /// target bytes alone must still be rejected when it cannot afford it in
    /// effective bytes — otherwise the shortfall lands as a draft-growth OOM
    /// once requests actually fill the pool.
    #[test]
    fn a_budget_that_affords_the_floor_only_without_the_draft_charge_is_rejected() {
        let geometry = KvBudget {
            num_layers: 36,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            num_blocks: 0,
        };
        let bytes_per_block = Qwen3Model::kv_bytes_per_block(&geometry).expect("sane geometry");
        let per_token = 65_536;
        let effective = bytes_per_block + per_token * geometry.block_size;
        assert!(bytes_per_block < effective);

        let between = (Qwen3Model::MIN_KV_BLOCKS * bytes_per_block + 1)
            .max((Qwen3Model::MIN_KV_BLOCKS * effective) / 2);
        let err = Qwen3Model::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            per_token,
            between,
            16 << 30,
            "test",
        )
        .expect_err("a budget short of the effective floor must be rejected");
        assert!(
            err.to_string().contains("below the"),
            "expected a minimum-blocks error, got: {err}"
        );

        let exact = Qwen3Model::MIN_KV_BLOCKS * effective;
        let budget = Qwen3Model::kv_budget_from_bytes(
            geometry,
            bytes_per_block,
            per_token,
            exact,
            16 << 30,
            "test",
        )
        .expect("the effective floor itself is affordable");
        assert_eq!(budget.num_blocks, Qwen3Model::MIN_KV_BLOCKS);
    }

    fn test_device_adapter(name: &str, path: &Path) -> DeviceLoraAdapter {
        DeviceLoraAdapter {
            name: name.to_string(),
            manifest: LoraAdapterManifest {
                path: path.to_path_buf(),
                rank: 1,
                alpha: 1,
                target_modules: vec!["q_proj".to_string()],
                tensor_count: 0,
            },
            scale: 1.0,
            layers: vec![DeviceLoraLayer::default()],
        }
    }

    #[test]
    fn install_lora_adapter_requires_load_inplace_to_replace_existing_name() {
        let mut adapters = HashMap::new();
        let first_path = Path::new("adapters/replace-first");
        let second_path = Path::new("adapters/replace-second");

        let first = test_device_adapter("adapter-a", first_path);
        install_lora_adapter_in_registry(&mut adapters, first, false)
            .expect("install first adapter");
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(first_path),
        );

        let duplicate = test_device_adapter("adapter-a", second_path);
        let error = install_lora_adapter_in_registry(&mut adapters, duplicate, false)
            .expect_err("duplicate adapter without load_inplace should fail")
            .to_string();
        assert!(error.contains("already loaded"));
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(first_path),
        );

        let replacement = test_device_adapter("adapter-a", second_path);
        install_lora_adapter_in_registry(&mut adapters, replacement, true)
            .expect("replace adapter");
        assert_eq!(
            adapters
                .get("adapter-a")
                .map(|adapter| adapter.manifest.path.as_path()),
            Some(second_path),
        );
    }

    #[test]
    fn packed_lora_registry_keeps_fixed_slots() {
        let mut registry = PackedLoraRegistry::empty(2, 1);

        let slot_a = registry
            .slot_for_install("adapter-a", false)
            .expect("first slot");
        assert_eq!(slot_a, 0);
        registry.bind_slot(slot_a, "adapter-a");
        assert_eq!(registry.slot_for("adapter-a"), Some(0));

        let slot_b = registry
            .slot_for_install("adapter-b", false)
            .expect("second slot");
        assert_eq!(slot_b, 1);
        registry.bind_slot(slot_b, "adapter-b");

        let replacement_slot = registry
            .slot_for_install("adapter-a", true)
            .expect("replacement slot");
        assert_eq!(replacement_slot, 0);

        let duplicate = registry
            .slot_for_install("adapter-a", false)
            .expect_err("duplicate without load_inplace should fail")
            .to_string();
        assert!(duplicate.contains("already loaded"));

        assert_eq!(registry.release_slot("adapter-a").expect("release"), 0);
        let slot_c = registry
            .slot_for_install("adapter-c", false)
            .expect("released slot should be reused");
        assert_eq!(slot_c, 0);
    }

    #[test]
    fn memory_options_rejects_invalid_page_sizes() {
        // FlashInfer only accepts 16 and 64; anything else must fail validation.
        for &invalid in &[0usize, 1, 15, 17, 32, 63, 65, 128] {
            let err = Qwen3MemoryOptions::new(0.9, 0, invalid)
                .validate()
                .expect_err(&format!("page_size {invalid} should be rejected"));
            assert!(
                err.to_string().contains("page_size"),
                "error should mention page_size, got: {err}"
            );
        }
    }
}
