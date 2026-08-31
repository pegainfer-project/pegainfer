//! Resident Gemma 4 text-tower weights.

use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;

use crate::config::Gemma4Config;

mod load;

pub(crate) struct Gemma4Weights {
    pub(crate) config: Gemma4Config,
    pub(crate) embed_tokens: DeviceMatrix,
    pub(crate) norm: DeviceVec,
    pub(crate) layers: Vec<Gemma4Layer>,
}

pub(crate) struct Gemma4Layer {
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) post_attention_layernorm: DeviceVec,
    pub(crate) pre_feedforward_layernorm: DeviceVec,
    pub(crate) post_feedforward_layernorm: DeviceVec,
    /// The per-layer output multiplier, read to the host at load: it is a
    /// one-element constant consumed as a kernel scalar every step, and a
    /// device read here would put a synchronous D2H in every layer of every
    /// decode token.
    pub(crate) layer_scalar: f32,
    pub(crate) attention: Gemma4Attention,
    pub(crate) mlp: Gemma4Mlp,
    /// Present only on the routing size, beside the dense MLP rather than
    /// instead of it.
    pub(crate) moe: Option<Gemma4Moe>,
}

/// Every expert's copy of one projection, stacked into one buffer: expert `e`
/// owns rows `[e * rows, (e + 1) * rows)`.
///
/// Stored packed, as the checkpoint ships it. Widening the experts to bf16
/// here would cost 45.7 GB against the 11.4 GB they occupy packed, which does
/// not fit the card this line serves on, and the FP4 kernel this is heading
/// for wants them packed regardless. Stacking is what lets one batched call
/// address all of them, and it turns 128 allocations per projection into one.
pub(crate) struct StackedProjection {
    /// Marlin's B order, which the checkpoint's is not: the loader rewrites it
    /// once so a step never has to.
    pub(crate) qweight: CudaSlice<u8>,
    /// The block scales in Marlin's order and its S0E5M3 encoding.
    pub(crate) scales: CudaSlice<u8>,
    /// One per expert, carrying the checkpoint's per-tensor scale, the
    /// exponent bias the encoding above owes, and the shared normalization
    /// the block scales were rescaled by.
    pub(crate) global_scales: CudaSlice<f32>,
    pub(crate) rows: usize,
    /// Logical values per row, before packing.
    pub(crate) values: usize,
}

pub(crate) struct Gemma4Moe {
    pub(crate) pre_feedforward_layernorm_2: DeviceVec,
    pub(crate) post_feedforward_layernorm_1: DeviceVec,
    pub(crate) post_feedforward_layernorm_2: DeviceVec,
    pub(crate) router_proj: DeviceMatrix,
    pub(crate) router_scale: DeviceVec,
    pub(crate) router_per_expert_scale: DeviceVec,
    pub(crate) gate: StackedProjection,
    pub(crate) up: StackedProjection,
    pub(crate) down: StackedProjection,
}

pub(crate) struct Gemma4Attention {
    pub(crate) q_proj: DeviceMatrix,
    pub(crate) k_proj: DeviceMatrix,
    /// Absent on global layers, which the checkpoint ships without one.
    pub(crate) v_proj: Option<DeviceMatrix>,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

pub(crate) struct Gemma4Mlp {
    pub(crate) gate: DeviceMatrix,
    pub(crate) up: DeviceMatrix,
    pub(crate) down: DeviceMatrix,
}
