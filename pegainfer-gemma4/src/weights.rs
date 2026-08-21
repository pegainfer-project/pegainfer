//! Resident Gemma 4 text-tower weights.

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
