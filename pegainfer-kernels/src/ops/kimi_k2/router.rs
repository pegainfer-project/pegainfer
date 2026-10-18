use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::GpuTensor;
use crate::tensor::GpuWeight;

const KIMI_K2_ROUTER_HIDDEN: usize = 7168;
const KIMI_K2_ROUTER_EXPERTS: usize = 384;
const KIMI_K2_ROUTER_TOPK: usize = 8;
const KIMI_K2_ROUTER_N_GROUP: usize = 1;
const KIMI_K2_ROUTER_TOPK_GROUP: usize = 1;
pub const KIMI_K2_ROUTER_SCALE: f32 = 2.827;
const KIMI_K2_ROUTER_WEIGHT_SCALE: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KimiRouterConfig {
    hidden_dim: usize,
    n_experts: usize,
    topk: usize,
    n_group: usize,
    topk_group: usize,
    route_scale: f32,
}

impl KimiRouterConfig {
    pub const fn kimi_k2() -> Self {
        Self {
            hidden_dim: KIMI_K2_ROUTER_HIDDEN,
            n_experts: KIMI_K2_ROUTER_EXPERTS,
            topk: KIMI_K2_ROUTER_TOPK,
            n_group: KIMI_K2_ROUTER_N_GROUP,
            topk_group: KIMI_K2_ROUTER_TOPK_GROUP,
            route_scale: KIMI_K2_ROUTER_WEIGHT_SCALE,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.hidden_dim > 0,
            "Kimi router hidden_dim must be positive"
        );
        ensure!(self.n_experts > 0, "Kimi router n_experts must be positive");
        ensure!(self.topk > 0, "Kimi router topk must be positive");
        ensure!(
            self.topk <= self.n_experts,
            "Kimi router topk={} exceeds n_experts={}",
            self.topk,
            self.n_experts
        );
        ensure!(
            self.n_group == 1 && self.topk_group == 1,
            "Kimi noaux_tc router currently requires n_group=1/topk_group=1, got {}/{}",
            self.n_group,
            self.topk_group
        );
        ensure!(
            self.route_scale.is_finite() && self.route_scale > 0.0,
            "Kimi router route_scale must be finite and positive"
        );
        Ok(())
    }
}

impl Default for KimiRouterConfig {
    fn default() -> Self {
        Self::kimi_k2()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiRouterBatch {
    pub batch_size: usize,
    pub active_tokens: usize,
    pub padded_tokens: usize,
}

impl KimiRouterBatch {
    fn validate(self) -> Result<()> {
        ensure!(
            self.batch_size > 0,
            "Kimi router batch_size must be positive"
        );
        ensure!(
            self.active_tokens > 0,
            "Kimi router active_tokens must be positive"
        );
        ensure!(
            self.padded_tokens >= self.active_tokens,
            "Kimi router padded_tokens={} is smaller than active_tokens={}",
            self.padded_tokens,
            self.active_tokens
        );
        Ok(())
    }
}

pub struct KimiRouterOutput<'a> {
    pub topk_weight: &'a mut CudaSlice<f32>,
    pub topk_idx: &'a mut CudaSlice<i32>,
}

fn validate_kimi_router_shapes<const DIM: usize>(
    config: KimiRouterConfig,
    batch: KimiRouterBatch,
    hidden: &GpuTensor<DIM>,
    _gate_weight: &GpuWeight<KIMI_K2_ROUTER_EXPERTS, KIMI_K2_ROUTER_HIDDEN>,
    e_score_correction_bias: &CudaSlice<f32>,
    logits: &CudaSlice<f32>,
    output: &KimiRouterOutput<'_>,
) -> Result<()> {
    config.validate()?;
    batch.validate()?;
    ensure!(
        DIM == config.hidden_dim,
        "Kimi router hidden dim mismatch: got {}, expected {}",
        DIM,
        config.hidden_dim
    );
    ensure!(
        hidden.seq_len == batch.padded_tokens,
        "Kimi router hidden seq_len={} must equal padded_tokens={}",
        hidden.seq_len,
        batch.padded_tokens
    );
    ensure!(
        config.n_experts == KIMI_K2_ROUTER_EXPERTS && config.hidden_dim == KIMI_K2_ROUTER_HIDDEN,
        "Kimi router config must match typed gate weight [{}, {}], got [{}, {}]",
        KIMI_K2_ROUTER_EXPERTS,
        KIMI_K2_ROUTER_HIDDEN,
        config.n_experts,
        config.hidden_dim
    );

    let score_elems = batch.padded_tokens * config.n_experts;
    ensure!(
        e_score_correction_bias.len() >= config.n_experts,
        "Kimi router e_score_correction_bias too small: have {}, need {}",
        e_score_correction_bias.len(),
        config.n_experts
    );
    ensure!(
        logits.len() >= score_elems,
        "Kimi router logits scratch too small: have {}, need {}",
        logits.len(),
        score_elems
    );

    let route_elems = batch.active_tokens * config.topk;
    ensure!(
        output.topk_weight.len() >= route_elems,
        "Kimi router topk_weight too small: have {}, need {}",
        output.topk_weight.len(),
        route_elems
    );
    ensure!(
        output.topk_idx.len() >= route_elems,
        "Kimi router topk_idx too small: have {}, need {}",
        output.topk_idx.len(),
        route_elems
    );
    Ok(())
}

/// Shape-checked CUDA launch for Kimi-K2 `noaux_tc` routing.
///
/// Launch contract for decode graph readiness:
/// - all scratch and outputs are caller-owned, preallocated device buffers;
/// - `hidden` has `padded_tokens` rows, while route outputs are emitted only for
///   the first `active_tokens` rows;
/// - no host-visible route metadata, D2H transfer, stream sync, or per-step
///   allocation is performed by this wrapper/body;
/// - CUDA computes `scores = sigmoid(hidden @ gate_weight.T)`, selects top-k
///   over `scores + e_score_correction_bias`, then gathers unbiased `scores`,
///   normalizes them per active token. Kimi's routed output scale is applied
///   after the routed expert sum to match vLLM's rounding boundary.
pub fn kimi_router_noaux_tc_launch<const DIM: usize>(
    ctx: &DeviceContext,
    config: KimiRouterConfig,
    batch: KimiRouterBatch,
    hidden: &GpuTensor<DIM>,
    gate_weight: &GpuWeight<KIMI_K2_ROUTER_EXPERTS, KIMI_K2_ROUTER_HIDDEN>,
    e_score_correction_bias: &CudaSlice<f32>,
    logits: &mut CudaSlice<f32>,
    output: &mut KimiRouterOutput<'_>,
) -> Result<()> {
    validate_kimi_router_shapes(
        config,
        batch,
        hidden,
        gate_weight,
        e_score_correction_bias,
        logits,
        output,
    )?;

    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = gate_weight.data.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = e_score_correction_bias.device_ptr(&ctx.stream);
    let (logits_ptr, _logits_guard) = logits.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _weight_guard) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (idx_ptr, _idx_guard) = output.topk_idx.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_k2_router_noaux_tc_cuda(
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
        .map_err(|err| anyhow!("Kimi router CUDA launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KimiRouterConfig {
        KimiRouterConfig {
            hidden_dim: 4,
            n_experts: 12,
            topk: 4,
            n_group: 1,
            topk_group: 1,
            route_scale: KIMI_K2_ROUTER_SCALE,
        }
    }

    #[test]
    fn config_rejects_grouped_topk_until_kimi_needs_it() {
        let bad = KimiRouterConfig {
            n_group: 2,
            topk_group: 1,
            ..test_config()
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("n_group=1/topk_group=1"));
    }
}
