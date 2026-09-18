//! Native DFlash2's model-owned selector component. Serving remains gated until
//! the native convolution and attention backbone is implemented.

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::PinnedHostSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_core::weight_loader::tensor_descriptors;
use pegainfer_kernels::ops::DFlash2Scratch;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::dflash2_select_into;
use pegainfer_kernels::ops::f32_to_bf16_hidden_into;
use pegainfer_kernels::ops::gemm_bf16_f32;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kernels::tensor::has_stream_override;

use super::config::NativeDFlash2Config;
use super::manifest::inspect_config;
use super::manifest::validate_tensors;

/// Deterministic native candidate-path selection, independent of target sampling.
///
/// Load the three trained selector tensors, then feed normalized backbone hidden
/// states and corresponding unary logits. This does not run the native backbone
/// or turn an inspected checkpoint into a supported serving model.
pub struct DFlash2Sampler {
    ctx: DeviceContext,
    config: NativeDFlash2Config,

    projection: DeviceMatrix,
    predecessor: DeviceMatrix,
    successor: DeviceMatrix,

    projected_f32: CudaSlice<f32>,
    projected: HiddenStates,
    scratch: DFlash2Scratch,

    selected_host: PinnedHostSlice<u32>,
    error_host: PinnedHostSlice<u32>,

    max_batch: usize,
    device_bytes: usize,
}

impl DFlash2Sampler {
    /// Validate the full native manifest before allocating any selector weights.
    ///
    /// `rows_per_chunk` bounds the packed vocabulary input; it is fixed for
    /// the component's lifetime. Only W/A/B are uploaded, not the backbone.
    pub fn from_safetensors(
        ctx: &DeviceContext,
        model_path: &str,
        max_batch: usize,
        rows_per_chunk: usize,
    ) -> Result<Self> {
        ensure!(
            !has_stream_override(),
            "DFlash2 loading requires the base stream"
        );
        ensure!(
            numeric_policy() == NumericPolicy::Tuned,
            "DFlash2 FP32 projection does not support Pin/PerToken numeric policies"
        );
        ensure!(max_batch > 0, "DFlash2 max_batch must be positive");

        let config = inspect_config(model_path)?;
        let rows = max_batch
            .checked_mul(config.block_size)
            .context("DFlash2 row capacity overflow")?;
        let projected_elements = rows
            .checked_mul(config.selector_rank)
            .context("DFlash2 projection capacity overflow")?;
        ensure!(
            i32::try_from(projected_elements).is_ok(),
            "DFlash2 projection exceeds the shared conversion's int32 extent"
        );
        let selected_elements = max_batch
            .checked_mul(config.block_size - 1)
            .context("DFlash2 path capacity overflow")?;
        let weight_bytes = config.selector_weight_bytes()?;
        let projected_bytes = projected_elements
            .checked_mul(size_of::<f32>() + size_of::<half::bf16>())
            .context("DFlash2 projection byte size overflow")?;

        // Validate the same mapped shards that remain alive through upload.
        let (paths, weight_map) = load_shard_info(model_path)?;
        let mmaps = mmap_shards(&paths)?;
        let shards = deserialize_shards(&mmaps)?;
        let descriptors = tensor_descriptors(&shards, &weight_map)?;
        validate_tensors(&config, &descriptors)?;

        let scratch = DFlash2Scratch::new(
            ctx,
            max_batch,
            config.block_size,
            config.vocab_size,
            config.selector_rank,
            rows_per_chunk,
        )?;
        let device_bytes = weight_bytes
            .checked_add(projected_bytes)
            .and_then(|n| n.checked_add(scratch.total_bytes()))
            .context("DFlash2 resident byte size overflow")?;

        let mut loader = StagedWeightLoader::new(ctx, &shards, &weight_map)?;
        let projection = loader.matrix(
            "candidate_selector.hidden_projection.weight",
            config.selector_rank,
            config.hidden_size,
        )?;
        let predecessor = loader.matrix(
            "candidate_selector.predecessor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?;
        let successor = loader.matrix(
            "candidate_selector.successor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?;

        loader.finish()?;
        let projection = loader.take(projection);
        let predecessor = loader.take(predecessor);
        let successor = loader.take(successor);

        Ok(Self {
            ctx: ctx.clone(),
            projected_f32: ctx.stream.alloc_zeros(projected_elements)?,
            projected: HiddenStates::zeros(ctx, config.selector_rank, rows)?,
            // SAFETY: never read before an event-tracked pinned D2H completes.
            selected_host: unsafe { ctx.ctx.alloc_pinned(selected_elements)? },
            // SAFETY: overwritten by error_flag's D2H before read.
            error_host: unsafe { ctx.ctx.alloc_pinned(1)? },
            config,
            projection,
            predecessor,
            successor,
            scratch,
            max_batch,
            device_bytes,
        })
    }

    pub fn config(&self) -> &NativeDFlash2Config {
        &self.config
    }

    /// Resident device weights and scratch, excluding caller-owned inputs and
    /// cuBLAS context workspace. Pinned host readback is reported separately.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub fn pinned_host_bytes(&self) -> usize {
        (self.max_batch * (self.config.block_size - 1) + 1) * size_of::<u32>()
    }

    /// Enqueue projection, top-k, edge scoring and a request-local greedy walk.
    ///
    /// Inputs are contiguous `[requests * block_size, hidden/vocab]` arenas.
    /// Row zero of each block is the discarded anchor slot. The returned borrow
    /// prevents scratch reuse until collection (or explicit discard). The device
    /// work is capturable; collection and input upload belong outside capture.
    /// The FP32-output projection is not covered by Pin/PerToken BF16 plans;
    /// unsupported numeric policies fail explicitly instead of bypassing them.
    pub fn enqueue<'a>(
        &'a mut self,
        hidden: &HiddenStates,
        logits: &HiddenStates,
        anchors: &CudaSlice<u32>,
        requests: usize,
    ) -> Result<DFlash2Selection<'a>> {
        ensure!(
            !has_stream_override(),
            "DFlash2 selector requires the base stream"
        );
        ensure!(
            numeric_policy() == NumericPolicy::Tuned,
            "DFlash2 FP32 projection does not support Pin/PerToken numeric policies"
        );
        ensure!(
            requests <= self.max_batch,
            "DFlash2 requests exceed sampler capacity"
        );

        let rows = requests
            .checked_mul(self.config.block_size)
            .context("DFlash2 active row overflow")?;
        ensure!(
            hidden.seq_len == rows && logits.seq_len == rows,
            "DFlash2 hidden/logits must contain one full block per request"
        );
        ensure!(
            hidden.hidden_dim == self.config.hidden_size,
            "DFlash2 hidden width does not match checkpoint"
        );
        ensure!(
            logits.hidden_dim == self.config.vocab_size,
            "DFlash2 vocabulary does not match checkpoint"
        );
        ensure!(
            anchors.len() >= requests,
            "DFlash2 anchor buffer is too small"
        );

        ensure!(
            Arc::ptr_eq(hidden.data.stream(), &self.ctx.stream)
                && Arc::ptr_eq(logits.data.stream(), &self.ctx.stream)
                && Arc::ptr_eq(anchors.stream(), &self.ctx.stream),
            "DFlash2 inputs must belong to the sampler base stream"
        );

        ensure!(
            hidden.data.len()
                >= rows
                    .checked_mul(hidden.hidden_dim)
                    .context("DFlash2 hidden extent overflow")?,
            "DFlash2 hidden backing is too small"
        );
        ensure!(
            logits.data.len()
                >= rows
                    .checked_mul(logits.hidden_dim)
                    .context("DFlash2 logits extent overflow")?,
            "DFlash2 logits backing is too small"
        );

        self.projected.seq_len = rows;
        if requests != 0 {
            // BF16-output GEMM may reduce split-K partials in BF16 even with
            // COMPUTE_32F. Keep partials/output in FP32, then narrow exactly once.
            // Column-major output [rank, rows] is row-major [rows, rank].
            gemm_bf16_f32(
                &self.ctx,
                true,
                false,
                self.config.selector_rank,
                rows,
                self.config.hidden_size,
                &self.projection.data,
                self.config.hidden_size,
                &hidden.data,
                self.config.hidden_size,
                &mut self.projected_f32,
                self.config.selector_rank,
            )?;
            f32_to_bf16_hidden_into(&self.ctx, &self.projected_f32, &mut self.projected)?;

            dflash2_select_into(
                &self.ctx,
                logits,
                &self.projected,
                &self.predecessor,
                &self.successor,
                anchors,
                requests,
                &mut self.scratch,
            )?;
        }

        Ok(DFlash2Selection {
            sampler: self,
            requests,
        })
    }
}

/// An in-flight path and diagnostic device views, borrowing the sampler scratch.
/// Values are only valid after the associated stream work completes. CUDA graphs
/// do not own these allocations: keep the sampler and all inputs alive for replay.
/// Views cover the full capacity; only the active request prefix is valid.
pub struct DFlash2Selection<'a> {
    sampler: &'a mut DFlash2Sampler,
    requests: usize,
}

impl DFlash2Selection<'_> {
    pub fn projected_hidden(&self) -> &HiddenStates {
        &self.sampler.projected
    }

    pub fn candidate_ids(&self) -> &CudaSlice<u32> {
        self.sampler.scratch.candidate_ids()
    }

    pub fn unary_scores(&self) -> &CudaSlice<f32> {
        self.sampler.scratch.unary_scores()
    }

    pub fn edge_scores(&self) -> &CudaSlice<f32> {
        self.sampler.scratch.edge_scores()
    }

    pub fn selected_ids(&self) -> &CudaSlice<u32> {
        self.sampler.scratch.selected_ids()
    }

    /// Pinned readback for the path and its error flag; no host roundtrip
    /// per draft position. The pinned buffer waits on its copy's completion.
    pub fn collect(self) -> Result<Vec<u32>> {
        if self.requests == 0 {
            return Ok(Vec::new());
        }

        ensure!(
            !has_stream_override(),
            "DFlash2 collection requires the base stream"
        );

        let sampler = self.sampler;
        sampler
            .ctx
            .stream
            .memcpy_dtoh(sampler.scratch.error_flag(), &mut sampler.error_host)?;
        sampler
            .ctx
            .stream
            .memcpy_dtoh(sampler.scratch.selected_ids(), &mut sampler.selected_host)?;

        let error = sampler.error_host.as_slice()?[0];
        // Drain both copies even if the device reported invalid inputs, so a
        // subsequent call cannot overwrite an unfinished host landing buffer.
        let selected = sampler.selected_host.as_slice()?;
        ensure!(
            error == 0,
            "DFlash2 selector rejected invalid input or non-finite score (device flags: {error:#x})"
        );

        let count = self.requests * (sampler.config.block_size - 1);
        ensure!(
            selected[..count]
                .iter()
                .all(|&id| (id as usize) < sampler.config.vocab_size),
            "DFlash2 path contains an invalid token ID"
        );

        Ok(selected[..count].to_vec())
    }
}

#[cfg(test)]
mod tests;
