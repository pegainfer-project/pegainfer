//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [local_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [local_qkv_dim × (conv_kernel_dim - 1)] bf16
//!
//! Under TP, value heads (and fused qkv channels) are sharded across ranks,
//! so every rank owns its own recurrent/conv state; these states are never
//! all-reduced.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;

use super::config::Config35;
use super::config::LocalGeometry;

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [local_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [local_linear_qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d.
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
///
/// Where the row set changes between steps (eager TP decode), allocate once at
/// capacity with `with_capacity` and `refill_from_recurrent_refs` the live
/// rows each step: H2D copies into the existing tables, no per-step allocation.
pub(crate) struct LinearStatePointerTables {
    pub(crate) state_ptrs: Vec<CudaSlice<u64>>,
    pub(crate) conv_state_ptrs: Vec<CudaSlice<u64>>,
    capacity: usize,
}

/// Per-layer element counts shared by allocation and reservation:
/// (linear layers, f32 state elements, bf16 conv elements).
fn per_layer_dims(config: &Config35, geometry: LocalGeometry) -> (usize, usize, usize) {
    let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
    let state_size = geometry.local_linear_num_value_heads()
        * config.linear_key_head_dim
        * config.linear_value_head_dim;
    let conv_state_size = geometry.local_linear_qkv_dim() * (config.linear_conv_kernel_dim - 1);
    (num_linear_layers, state_size, conv_state_size)
}

impl RecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        geometry: LocalGeometry,
    ) -> Result<Self> {
        let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config, geometry);

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

    /// Copy one complete target-model recurrent state into this allocation.
    pub(crate) fn copy_from(&mut self, ctx: &DeviceContext, src: &Self) -> Result<()> {
        anyhow::ensure!(
            self.layers.len() == src.layers.len(),
            "Qwen3.5 recurrent copy layer mismatch: dst={}, src={}",
            self.layers.len(),
            src.layers.len()
        );
        for (layer_idx, (dst, src)) in self.layers.iter_mut().zip(&src.layers).enumerate() {
            ctx.stream
                .memcpy_dtod(&src.state, &mut dst.state)
                .map_err(|e| anyhow::anyhow!("copy recurrent layer {layer_idx}: {e}"))?;
            ctx.stream
                .memcpy_dtod(&src.conv_state.data, &mut dst.conv_state.data)
                .map_err(|e| anyhow::anyhow!("copy conv state layer {layer_idx}: {e}"))?;
        }
        self.seq_len = src.seq_len;
        Ok(())
    }
}

impl LinearStatePointerTables {
    /// Allocate zeroed per-layer tables with room for `capacity` rows. Rows
    /// carry no pointers until `refill_from_recurrent_refs` writes them.
    pub(crate) fn with_capacity(
        ctx: &DeviceContext,
        config: &Config35,
        capacity: usize,
        label: &str,
    ) -> Result<Self> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let mut state_ptrs = Vec::with_capacity(num_linear_layers);
        let mut conv_state_ptrs = Vec::with_capacity(num_linear_layers);
        for layer_idx in 0..num_linear_layers {
            state_ptrs.push(ctx.stream.alloc_zeros::<u64>(capacity).map_err(|e| {
                anyhow::anyhow!("alloc {label} linear state pointer table {layer_idx}: {e}")
            })?);
            conv_state_ptrs.push(ctx.stream.alloc_zeros::<u64>(capacity).map_err(|e| {
                anyhow::anyhow!("alloc {label} conv state pointer table {layer_idx}: {e}")
            })?);
        }
        Ok(Self {
            state_ptrs,
            conv_state_ptrs,
            capacity,
        })
    }

    /// Tables sized exactly to `batch_size` rows, filled once.
    pub(crate) fn from_recurrent_refs(
        ctx: &DeviceContext,
        config: &Config35,
        recurrent_states: &mut [&mut RecurrentState],
        batch_size: usize,
        label: &str,
    ) -> Result<Self> {
        let mut tables = Self::with_capacity(ctx, config, batch_size, label)?;
        tables.refill_from_recurrent_refs(ctx, recurrent_states, batch_size, label)?;
        Ok(tables)
    }

    /// Overwrite rows `0..batch_size` of every layer table with the current
    /// state addresses of `recurrent_states`. Copies into the existing
    /// allocations only; rows past `batch_size` keep whatever they held and
    /// must stay unaddressed (the kernels are handed `batch_size`).
    pub(crate) fn refill_from_recurrent_refs(
        &mut self,
        ctx: &DeviceContext,
        recurrent_states: &mut [&mut RecurrentState],
        batch_size: usize,
        label: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            batch_size <= recurrent_states.len(),
            "{label} pointer table batch {batch_size} exceeds recurrent refs {}",
            recurrent_states.len()
        );
        anyhow::ensure!(
            batch_size <= self.capacity,
            "{label} pointer table batch {batch_size} exceeds capacity {}",
            self.capacity
        );
        let tables = self
            .state_ptrs
            .iter_mut()
            .zip(self.conv_state_ptrs.iter_mut());
        for (layer_idx, (state_table, conv_table)) in tables.enumerate() {
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
            ctx.stream
                .memcpy_htod(state_ptrs.as_slice(), state_table)
                .map_err(|e| {
                    anyhow::anyhow!("copy {label} linear state pointer table {layer_idx}: {e}")
                })?;
            ctx.stream
                .memcpy_htod(conv_state_ptrs.as_slice(), conv_table)
                .map_err(|e| {
                    anyhow::anyhow!("copy {label} conv state pointer table {layer_idx}: {e}")
                })?;
        }
        Ok(())
    }

    pub(crate) fn validate_for(
        &self,
        config: &Config35,
        batch_size: usize,
        label: &str,
    ) -> Result<()> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        anyhow::ensure!(
            self.capacity >= batch_size,
            "{label} pointer table capacity {} is smaller than batch {batch_size}",
            self.capacity
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

/// Device bytes of one request's recurrent state.
pub(crate) fn bytes_per_request(config: &Config35, geometry: LocalGeometry) -> usize {
    let (num_linear_layers, state_size, conv_state_size) = per_layer_dims(config, geometry);
    num_linear_layers
        * (state_size * std::mem::size_of::<f32>()
            + conv_state_size * std::mem::size_of::<half::bf16>())
}

impl RecurrentState {
    pub(crate) fn allocation_bytes(config: &Config35, geometry: LocalGeometry) -> usize {
        bytes_per_request(config, geometry)
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
