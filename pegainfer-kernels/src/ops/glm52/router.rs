use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_ROUTER_HIDDEN: usize = 6144;
const GLM52_ROUTER_EXPERTS: usize = 256;
const GLM52_ROUTER_TOPK: usize = 8;
/// `routed_scaling_factor` from the GLM5.2 checkpoint config. The default
/// TP path folds it into top-k weights; the EP path uses normalized weights
/// and applies the factor after expert reduction, matching vLLM.
const GLM52_ROUTED_RESIDUAL_SCALE: f32 = 2.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm52RouterConfig {
    hidden_dim: usize,
    n_experts: usize,
    topk: usize,
    pub route_scale: f32,
}

impl Glm52RouterConfig {
    pub const fn glm52() -> Self {
        Self {
            hidden_dim: GLM52_ROUTER_HIDDEN,
            n_experts: GLM52_ROUTER_EXPERTS,
            topk: GLM52_ROUTER_TOPK,
            route_scale: GLM52_ROUTED_RESIDUAL_SCALE,
        }
    }

    pub const fn glm52_unscaled() -> Self {
        Self {
            hidden_dim: GLM52_ROUTER_HIDDEN,
            n_experts: GLM52_ROUTER_EXPERTS,
            topk: GLM52_ROUTER_TOPK,
            route_scale: 1.0,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.hidden_dim == GLM52_ROUTER_HIDDEN,
            "GLM5.2 router hidden_dim must be {GLM52_ROUTER_HIDDEN}, got {}",
            self.hidden_dim
        );
        ensure!(
            self.n_experts == GLM52_ROUTER_EXPERTS,
            "GLM5.2 router n_experts must be {GLM52_ROUTER_EXPERTS}, got {}",
            self.n_experts
        );
        ensure!(
            self.topk == GLM52_ROUTER_TOPK,
            "GLM5.2 router topk must be {GLM52_ROUTER_TOPK}, got {}",
            self.topk
        );
        ensure!(
            self.route_scale.is_finite() && self.route_scale > 0.0,
            "GLM5.2 router route_scale must be finite and positive"
        );
        Ok(())
    }
}

impl Default for Glm52RouterConfig {
    fn default() -> Self {
        Self::glm52()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52RouterBatch {
    pub active_tokens: usize,
    pub padded_tokens: usize,
}

impl Glm52RouterBatch {
    fn validate(self) -> Result<()> {
        ensure!(
            self.active_tokens > 0,
            "GLM5.2 router active_tokens must be positive"
        );
        ensure!(
            self.padded_tokens >= self.active_tokens,
            "GLM5.2 router padded_tokens={} is smaller than active_tokens={}",
            self.padded_tokens,
            self.active_tokens
        );
        Ok(())
    }
}

pub struct Glm52RouterOutput<'a> {
    pub topk_weight: &'a mut CudaSlice<f32>,
    pub topk_idx: &'a mut CudaSlice<i32>,
}

fn validate_glm52_router_shapes(
    config: Glm52RouterConfig,
    batch: Glm52RouterBatch,
    hidden: &CudaSlice<bf16>,
    gate_weight: &CudaSlice<bf16>,
    e_score_correction_bias: &CudaSlice<u8>,
    logits: &CudaSlice<f32>,
    output: &Glm52RouterOutput<'_>,
) -> Result<()> {
    config.validate()?;
    batch.validate()?;
    let hidden_elems = batch.padded_tokens * config.hidden_dim;
    ensure!(
        hidden.len() >= hidden_elems,
        "GLM5.2 router hidden too small: have {}, need {}",
        hidden.len(),
        hidden_elems
    );
    let gate_elems = config.n_experts * config.hidden_dim;
    ensure!(
        gate_weight.len() >= gate_elems,
        "GLM5.2 router gate_weight too small: have {}, need {gate_elems}",
        gate_weight.len()
    );
    let bias_bytes = config.n_experts * std::mem::size_of::<f32>();
    ensure!(
        e_score_correction_bias.len() >= bias_bytes,
        "GLM5.2 router correction bias too small: have {} bytes, need {bias_bytes}",
        e_score_correction_bias.len()
    );
    let score_elems = batch.padded_tokens * config.n_experts;
    ensure!(
        logits.len() >= score_elems,
        "GLM5.2 router logits scratch too small: have {}, need {score_elems}",
        logits.len()
    );
    let route_elems = batch.active_tokens * config.topk;
    ensure!(
        output.topk_weight.len() >= route_elems,
        "GLM5.2 router topk_weight too small: have {}, need {route_elems}",
        output.topk_weight.len()
    );
    ensure!(
        output.topk_idx.len() >= route_elems,
        "GLM5.2 router topk_idx too small: have {}, need {route_elems}",
        output.topk_idx.len()
    );
    Ok(())
}

pub fn glm52_router_noaux_tc_launch(
    ctx: &DeviceContext,
    config: Glm52RouterConfig,
    batch: Glm52RouterBatch,
    hidden: &CudaSlice<bf16>,
    gate_weight: &CudaSlice<bf16>,
    e_score_correction_bias: &CudaSlice<u8>,
    logits: &mut CudaSlice<f32>,
    output: &mut Glm52RouterOutput<'_>,
) -> Result<()> {
    validate_glm52_router_shapes(
        config,
        batch,
        hidden,
        gate_weight,
        e_score_correction_bias,
        logits,
        output,
    )?;

    let (hidden_ptr, _hidden_guard) = hidden.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = e_score_correction_bias.device_ptr(&ctx.stream);
    let (logits_ptr, _logits_guard) = logits.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _weight_guard) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _idx_guard) = output.topk_idx.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::glm52_router_noaux_tc_cuda(
            hidden_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            bias_ptr as *const f32,
            logits_ptr as *mut f32,
            weight_ptr as *mut f32,
            idx_ptr as *mut i32,
            batch.active_tokens as i32,
            batch.padded_tokens as i32,
            config.hidden_dim as i32,
            config.n_experts as i32,
            config.topk as i32,
            config.route_scale,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 router CUDA launch failed: {err}"))
}

pub fn glm52_router_select_launch(
    ctx: &DeviceContext,
    config: Glm52RouterConfig,
    batch: Glm52RouterBatch,
    logits: &CudaSlice<f32>,
    e_score_correction_bias: &CudaSlice<u8>,
    output: &mut Glm52RouterOutput<'_>,
) -> Result<()> {
    config.validate()?;
    batch.validate()?;
    ensure!(
        logits.len() >= batch.padded_tokens * config.n_experts
            && e_score_correction_bias.len() >= config.n_experts * std::mem::size_of::<f32>()
            && output.topk_weight.len() >= batch.active_tokens * config.topk
            && output.topk_idx.len() >= batch.active_tokens * config.topk,
        "GLM5.2 router selection buffers are too small"
    );
    let (logits_ptr, _gl) = logits.device_ptr(&ctx.stream);
    let (bias_ptr, _gb) = e_score_correction_bias.device_ptr(&ctx.stream);
    let (weight_ptr, _gw) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _gi) = output.topk_idx.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::glm52_router_select_cuda(
            logits_ptr as *const f32,
            bias_ptr as *const f32,
            weight_ptr as *mut f32,
            idx_ptr as *mut i32,
            batch.active_tokens as i32,
            batch.padded_tokens as i32,
            config.n_experts as i32,
            config.topk as i32,
            config.route_scale,
            ctx.stream.cu_stream(),
        )
    }
    .result()
    .map_err(|err| anyhow!("GLM5.2 router selection failed: {err}"))
}
