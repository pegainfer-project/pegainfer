//! Resident Gemma 4 text-tower weights.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::W4a16Matrix;
use pegainfer_kernels::ops::W4a16Scratch;

use crate::config::Gemma4Config;
use crate::config::TensorParallelConfig;

mod load;

pub(crate) struct Gemma4Weights {
    pub(crate) config: Gemma4Config,
    /// The shard these weights are, and the only source of the tensor-parallel
    /// geometry: a serve built around them reads its rank and world size here
    /// instead of taking them as a second argument, so the two cannot disagree.
    pub(crate) tp: TensorParallelConfig,
    pub(crate) embed_tokens: DeviceMatrix,
    pub(crate) norm: DeviceVec,
    pub(crate) layers: Vec<Gemma4Layer>,
}

impl Gemma4Weights {
    /// The largest W4A16 linear's `rows * cols`, zero for a bf16 checkpoint.
    pub(crate) fn w4a16_values(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| {
                [
                    &layer.attention.qkv,
                    &layer.attention.o_proj,
                    &layer.mlp.gate_up,
                    &layer.mlp.down,
                ]
            })
            .map(|linear| match linear {
                Linear::Bf16(_) => 0,
                Linear::W4a16(m) => m.rows * m.cols,
            })
            .max()
            .unwrap_or(0)
    }
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
/// here would take about four times what they occupy packed, which does not
/// fit the card this line serves on, and the Marlin FP4 kernel reads them
/// packed. Stacking is what lets one batched call
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

/// A text-tower linear as the checkpoint ships it.
pub(crate) enum Linear {
    Bf16(DeviceMatrix),
    /// In the TileLang GEMMs' fragment layout; only whole-matrix products
    /// read it.
    W4a16(W4a16Matrix),
}

impl Linear {
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Bf16(m) => m.rows,
            Self::W4a16(m) => m.rows,
        }
    }

    /// The bf16 matrix a row-range product reads.
    pub(crate) fn bf16(&self) -> Result<&DeviceMatrix> {
        match self {
            Self::Bf16(m) => Ok(m),
            Self::W4a16(_) => {
                anyhow::bail!("a W4A16 linear projects whole, never through a row range")
            }
        }
    }

    /// `out = gelu(x @ gate^T) * (x @ up^T)` from this gate|up stack in one
    /// GEMM, where its kernels fuse the two (W4A16, a step the TileLang GEMMs
    /// run). False leaves `out` untouched for the caller's two-step path.
    pub(crate) fn gelu_mul_into(
        &self,
        ctx: &DeviceContext,
        x: &HiddenStates,
        scratch: &LinearScratch,
        out: &mut HiddenStates,
    ) -> Result<bool> {
        match (self, &scratch.0) {
            (Self::W4a16(m), Some(w4)) if m.gelu_mul && W4a16Matrix::runs_tilelang(x.seq_len) => {
                pegainfer_kernels::ops::gemma4_w4a16_gemm_into(ctx, m, x, &mut *lock(w4)?, out)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// `out = x @ self^T` over every row.
    pub(crate) fn project_into(
        &self,
        ctx: &DeviceContext,
        x: &HiddenStates,
        scratch: &LinearScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        match (self, &scratch.0) {
            (Self::Bf16(m), _) => ops::gemm_rows_into_checked(ctx, m, 0, m.rows, x, out),
            (Self::W4a16(m), Some(w4)) => {
                pegainfer_kernels::ops::gemma4_w4a16_gemm_into(ctx, m, x, &mut *lock(w4)?, out)
            }
            (Self::W4a16(_), None) => {
                anyhow::bail!("a W4A16 linear needs the scratch built for W4A16 weights")
            }
        }
    }
}

/// The W4A16 GEMMs' stream-K scratch and dequantization matrix, held only
/// when the weights are W4A16. One per engine, shared by every step's tower:
/// the matrix is as large as the largest linear, and every W4A16 step runs on
/// the stream that built the pools (the async lane, the one step on another
/// stream, refuses W4A16 checkpoints), so the calls are in stream order.
#[derive(Clone)]
pub(crate) struct LinearScratch(Option<Arc<Mutex<W4a16Scratch>>>);

impl LinearScratch {
    /// `w4a16_values` is the largest W4A16 linear's `rows * cols`, zero for a
    /// bf16 checkpoint.
    pub(crate) fn new(ctx: &DeviceContext, w4a16_values: usize) -> Result<Self> {
        Ok(Self(if w4a16_values > 0 {
            Some(Arc::new(Mutex::new(W4a16Scratch::new(ctx, w4a16_values)?)))
        } else {
            None
        }))
    }

    pub(crate) fn is_w4a16(&self) -> bool {
        self.0.is_some()
    }
}

fn lock(scratch: &Mutex<W4a16Scratch>) -> Result<MutexGuard<'_, W4a16Scratch>> {
    scratch
        .lock()
        .map_err(|_| anyhow::anyhow!("a W4A16 GEMM panicked while holding the scratch"))
}

pub(crate) struct Gemma4Attention {
    /// Q, K and, on sliding layers, V stacked along rows in that order, so a
    /// step projects them all with one GEMM or each through its row range.
    /// Global layers ship no V: it is the K fork.
    pub(crate) qkv: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

pub(crate) struct Gemma4Mlp {
    /// gate then up, stacked along rows.
    pub(crate) gate_up: Linear,
    pub(crate) down: Linear,
}
