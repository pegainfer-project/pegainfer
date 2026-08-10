//! Resident Gemma 4 text-tower weights.

use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;

use crate::config::Gemma4Config;

mod load;

pub(crate) struct Gemma4Weights {
    config: Gemma4Config,
    embed_tokens: DeviceMatrix,
    norm: DeviceVec,
    layers: Vec<Gemma4Layer>,
}

pub(crate) struct Gemma4Layer {
    input_layernorm: DeviceVec,
    post_attention_layernorm: DeviceVec,
    pre_feedforward_layernorm: DeviceVec,
    post_feedforward_layernorm: DeviceVec,
    layer_scalar: DeviceVec,
    attention: Gemma4Attention,
    mlp: Gemma4Mlp,
}

pub(crate) struct Gemma4Attention {
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    /// Absent on global layers, which the checkpoint ships without one.
    v_proj: Option<DeviceMatrix>,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

pub(crate) struct Gemma4Mlp {
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
}
