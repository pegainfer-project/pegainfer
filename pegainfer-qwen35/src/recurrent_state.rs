//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [local_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [local_qkv_dim × (conv_kernel_dim - 1)] bf16
//!
//! Under TP both are rank-local: Phase 2b shards value heads (and fused qkv
//! channels) across ranks. Recurrent and conv state are NEVER all-reduced.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;

use super::config::Config35;
use super::config::TensorParallelConfig;

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [local_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config. Rank-local under
    /// TP (local dims equal global dims at world_size 1). Never all-reduced.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [local_linear_qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d. Rank-local.
    pub(crate) conv_state: DeviceVec,
}

/// Recurrent state for all linear attention layers.
pub(crate) struct RecurrentState {
    pub(crate) layers: Vec<LayerRecurrentState>,
    /// Number of tokens processed so far (for prefill/decode tracking).
    pub(crate) seq_len: usize,
}

/// Device-side tables of per-slot recurrent-state pointers.
///
/// Batched linear decode kernels take a device array of pointers per linear
/// layer. The underlying `CudaSlice` allocations inside `RecurrentState` stay
/// at fixed device addresses for the request lifetime, so these tables should
/// be built once per slot/request and then reused across decode tokens.
pub(crate) struct LinearStatePointerTables {
    pub(crate) state_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) conv_state_ptrs: Vec<CudaSlice<u64>>,
    batch_size: usize,
}

/// Per-layer element counts shared by allocation and reservation:
/// (linear layers, f32 state elements, bf16 conv elements). Sizes are
/// rank-local under TP; at world_size 1 they equal the global dims.
fn per_layer_dims(config: &Config35, tp: TensorParallelConfig) -> (usize, usize, usize) {
    let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
    let state_size = config.local_linear_num_value_heads(tp)
        * config.linear_key_head_dim
        * config.linear_value_head_dim;
    let conv_state_size = config.local_linear_qkv_dim(tp) * (config.linear_conv_kernel_dim - 1);
    (num_linear_layers, state_size, conv_state_size)
}

impl RecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        tp: TensorParallelConfig,
    ) -> Result<Self> {
        let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config, tp);

        let mut layers = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            let state: CudaSlice<f32> = ctx
                .stream
                .alloc_zeros(state_size)
                .map_err(|e| anyhow::anyhow!("Alloc recurrent state failed: {}", e))?;
            layers.push(LayerRecurrentState {
                state,
                conv_state: DeviceVec::zeros(ctx, conv_state_size)?,
            });
        }

        Ok(Self { layers, seq_len: 0 })
    }
}

impl LinearStatePointerTables {
    pub(crate) fn from_recurrent_refs(
        ctx: &DeviceContext,
        config: &Config35,
        recurrent_states: &mut [&mut RecurrentState],
        batch_size: usize,
        label: &str,
    ) -> Result<Self> {
        anyhow::ensure!(
            batch_size <= recurrent_states.len(),
            "{label} pointer table batch {batch_size} exceeds recurrent refs {}",
            recurrent_states.len()
        );
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let mut linear_state_ptrs = Vec::with_capacity(num_linear_layers);
        let mut linear_conv_state_ptrs = Vec::with_capacity(num_linear_layers);
        for layer_idx in 0..num_linear_layers {
            let mut state_ptrs = Vec::with_capacity(batch_size);
            let mut conv_state_ptrs = Vec::with_capacity(batch_size);
            for slot in recurrent_states.iter_mut().take(batch_size) {
                let state_ptr = {
                    let (ptr, _guard) = slot.layers[layer_idx].state.device_ptr_mut(&ctx.stream);
                    ptr
                };
                let conv_ptr = {
                    let (ptr, _guard) = slot.layers[layer_idx]
                        .conv_state
                        .data
                        .device_ptr_mut(&ctx.stream);
                    ptr
                };
                state_ptrs.push(state_ptr);
                conv_state_ptrs.push(conv_ptr);
            }
            linear_state_ptrs.push(ctx.stream.clone_htod(&state_ptrs).map_err(|e| {
                anyhow::anyhow!("copy {label} linear state pointer table {layer_idx}: {e}")
            })?);
            linear_conv_state_ptrs.push(ctx.stream.clone_htod(&conv_state_ptrs).map_err(|e| {
                anyhow::anyhow!("copy {label} conv state pointer table {layer_idx}: {e}")
            })?);
        }

        Ok(Self {
            state_ptrs: linear_state_ptrs,
            conv_state_ptrs: linear_conv_state_ptrs,
            batch_size,
        })
    }

    pub(crate) fn validate_for(
        &self,
        config: &Config35,
        batch_size: usize,
        label: &str,
    ) -> Result<()> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        anyhow::ensure!(
            self.batch_size >= batch_size,
            "{label} pointer table capacity {} is smaller than batch {batch_size}",
            self.batch_size
        );
        anyhow::ensure!(
            self.state_ptrs.len() == num_linear_layers
                && self.conv_state_ptrs.len() == num_linear_layers,
            "{label} pointer table layer count mismatch: state={}, conv={}, expected={num_linear_layers}",
            self.state_ptrs.len(),
            self.conv_state_ptrs.len()
        );
        Ok(())
    }
}

/// Device bytes of one request's (rank-local) recurrent state.
pub(crate) fn bytes_per_request(config: &Config35, tp: TensorParallelConfig) -> usize {
    let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config, tp);
    num_linear_layers
        * (state_size * std::mem::size_of::<f32>()
            + conv_state_size * std::mem::size_of::<half::bf16>())
}

impl RecurrentState {
    pub(crate) fn allocation_bytes(config: &Config35, tp: TensorParallelConfig) -> usize {
        bytes_per_request(config, tp)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn qwen35_4b_recurrent_allocation_is_49_125_mib() {
        let bytes = 24
            * (32 * 128 * 128 * std::mem::size_of::<f32>()
                + 8192 * 3 * std::mem::size_of::<half::bf16>());
        assert_eq!(bytes, 49 * 1024 * 1024 + 128 * 1024);
    }
}
