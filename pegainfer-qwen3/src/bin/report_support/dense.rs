//! Dense (non-attention) forward-op bench: weight-free synthetic buffers at
//! production shapes, one launch per measured iteration, cold L2 via the
//! streaming sweep in [`super::common`].

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Duration;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::sys;
use half::bf16;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

use super::common::HEAD_DIM;
use super::common::L2CacheClear;
use super::common::NUM_KV_HEADS;
use super::common::NUM_QO_HEADS;
use super::common::cache_clear_bytes;
use super::common::patterned_bf16;

/// Qwen3-4B dense-op dimensions. The dense benches are weight-free (synthetic
/// buffers at production shapes), so the model facts live here as constants —
/// same convention as the attention constants in `common`.
pub(crate) const HIDDEN_SIZE: usize = 2560;
pub(crate) const INTERMEDIATE_SIZE: usize = 9728;
pub(crate) const VOCAB_SIZE: usize = 151_936;
const Q_DIM: usize = NUM_QO_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
/// Position span for the decode qk-norm-rope bench: mid-context decode is the
/// common case, and the cache read is position-indexed, so the span only has
/// to be large enough that positions don't all hit one cache line.
const DENSE_ROPE_CACHE_TOKENS: usize = 8192;
/// Model fact (config.json `rms_norm_eps`), mirrored here like the head
/// counts so the weight-free benches launch the production epsilon.
const RMS_NORM_EPS: f32 = 1.0e-6;
/// Device-memory cap for the `gemm_lt_tune` weight-rotation copies of a
/// projection-GEMM dense case; the actual copy count is derived from the L2
/// sweep size so the tuner stays DRAM-cold, and this cap only protects
/// small-VRAM cards from the lm_head shape.
const TUNE_ROTATION_BUDGET_BYTES: usize = 2 * (1 << 30);

/// The projection GEMM (out_dim, in_dim) shapes production launches — the same
/// set `decode_projection_pin_shapes` warms for the Pin policy. Gate and up
/// share a shape, so one variant covers both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GemmProjection {
    QProj,
    KvProj,
    OProj,
    GateUpHalf,
    DownProj,
    LmHead,
}

impl GemmProjection {
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "q_proj" => Self::QProj,
            "kv_proj" => Self::KvProj,
            "o_proj" => Self::OProj,
            "gate_up_half" => Self::GateUpHalf,
            "down_proj" => Self::DownProj,
            "lm_head" => Self::LmHead,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KvProj => "kv_proj",
            Self::OProj => "o_proj",
            Self::GateUpHalf => "gate_up_half",
            Self::DownProj => "down_proj",
            Self::LmHead => "lm_head",
        }
    }

    /// (out_dim, in_dim) — cuBLAS N is the token/row count of the step.
    pub(crate) const fn out_in(self) -> (usize, usize) {
        match self {
            Self::QProj => (Q_DIM, HIDDEN_SIZE),
            Self::KvProj => (KV_DIM, HIDDEN_SIZE),
            Self::OProj => (HIDDEN_SIZE, Q_DIM),
            Self::GateUpHalf => (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            Self::DownProj => (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            Self::LmHead => (VOCAB_SIZE, HIDDEN_SIZE),
        }
    }
}

/// One dense (non-attention) forward op at production shape. `rows` is the
/// step's token/row count: decode batch size, or prefill token count.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DenseKernelKind {
    ProjectionGemm(GemmProjection),
    RmsNorm,
    FusedAddRmsNorm,
    QkNormRopeDecode,
    SiluMul,
    Embedding,
    Sampling { greedy: bool },
}

impl DenseKernelKind {
    pub(crate) fn label(self) -> String {
        match self {
            Self::ProjectionGemm(projection) => projection.label().to_string(),
            Self::Sampling { greedy: true } => "argmax".to_string(),
            Self::Sampling { greedy: false } => "sampling".to_string(),
            Self::RmsNorm
            | Self::FusedAddRmsNorm
            | Self::QkNormRopeDecode
            | Self::SiluMul
            | Self::Embedding => "default".to_string(),
        }
    }
}

/// The buffers a dense case owns, one variant per kind — which buffers exist
/// for which op is a type-level fact, not a runtime assertion. One instance
/// per case, never stored in collections, so the variant size spread is
/// irrelevant and boxing the large ones would only add indirection.
#[allow(clippy::large_enum_variant)]
enum DenseBuffers {
    Gemm {
        weight: DeviceMatrix,
        x: HiddenStates,
        out: HiddenStates,
    },
    Norm {
        weight: DeviceVec,
        x: HiddenStates,
        out: HiddenStates,
    },
    FusedAddNorm {
        weight: DeviceVec,
        hidden: HiddenStates,
        residual: HiddenStates,
        out: HiddenStates,
    },
    QkRope {
        q: HiddenStates,
        k: HiddenStates,
        q_norm: DeviceVec,
        k_norm: DeviceVec,
        cos_cache: DeviceVec,
        sin_cache: DeviceVec,
        positions: CudaSlice<i32>,
    },
    SiluMul {
        gate: HiddenStates,
        up: HiddenStates,
        out: HiddenStates,
    },
    Embedding {
        table: DeviceMatrix,
        token_ids: CudaSlice<u32>,
        out: HiddenStates,
    },
    Sampling {
        logits: HiddenStates,
        scratch: pegainfer_sample::SampleScratch,
        params: Vec<pegainfer_frontend::sampler::SamplingParams>,
        seed: u64,
    },
}

/// Bench harness for the dense forward ops, mirroring the attention cases:
/// synthetic buffers at production shapes, one launch per measured iteration,
/// cold L2 via the streaming sweep. Launches go through the same
/// `pegainfer_kernels::ops` entry points as `BatchDecodeDag` / the prefill
/// path, so cuBLAS algo selection matches production steady state after the
/// pre-measure launch.
pub(crate) struct DenseCase {
    pub(crate) ctx: DeviceContext,
    buffers: DenseBuffers,
    start: CudaEvent,
    end: CudaEvent,
}

fn zeros_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    Ok(DeviceMatrix {
        data: ctx.stream.alloc_zeros(rows * cols)?,
        rows,
        cols,
    })
}

fn ones_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &vec![bf16::ONE; len])
}

/// Sampling-case logits: production distributions are sharply peaked, and the
/// FlashInfer rejection sampler's round count depends on that peakedness — a
/// flat synthetic vocabulary would overstate its cost. Each row gets a few
/// dominant logits (top-1 mass ~0.5 after softmax) over a low-noise floor, at
/// row-varying positions.
fn peaked_logits(ctx: &DeviceContext, rows: usize) -> Result<HiddenStates> {
    let mut host = patterned_bf16(VOCAB_SIZE * rows, 0.001);
    for row in 0..rows {
        for peak in 0..8 {
            let token = (row * 48_271 + peak * 15_485_863) % VOCAB_SIZE;
            host[row * VOCAB_SIZE + token] = bf16::from_f32(10.0 - peak as f32);
        }
    }
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(&host)?,
        hidden_dim: VOCAB_SIZE,
        seq_len: rows,
    })
}

/// Build the projection weight and tune its cuBLASLt plan the way the
/// executor does. Production decode GEMMs at N <= GEMM_LT_MAX_N run the algo
/// `gemm_lt_tune` selected at startup over every layer's weights — an L2-cold
/// rotation — and an untuned context falls back to GemmEx, mis-ranking the
/// small-N projections. The rotation here is sized off the L2 sweep size, so
/// the tuner times DRAM-cold candidates even for the small kv_proj weight;
/// the copies are dropped afterwards (the tuned plan is keyed by shape, not
/// pointer).
fn gemm_weight_tuned(
    ctx: &DeviceContext,
    out_dim: usize,
    in_dim: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    // Zero weights: cuBLAS HMMA does no zero-skipping, and the lm_head table
    // is too large to build patterned on the host.
    let weight = zeros_matrix(ctx, out_dim, in_dim)?;
    if rows <= pegainfer_kernels::ops::GEMM_LT_MAX_N {
        let weight_bytes = out_dim * in_dim * size_of::<bf16>();
        let l2_bytes = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
            as usize;
        let cold_copies = cache_clear_bytes(l2_bytes).div_ceil(weight_bytes).max(1);
        let budget_copies = (TUNE_ROTATION_BUDGET_BYTES / weight_bytes).max(1);
        let extra_copies = cold_copies.min(budget_copies) - 1;
        let rotation: Vec<DeviceMatrix> = (0..extra_copies)
            .map(|_| zeros_matrix(ctx, out_dim, in_dim))
            .collect::<Result<_>>()?;
        let samples: Vec<(&DeviceMatrix, usize)> = std::iter::once((&weight, 0))
            .chain(rotation.iter().map(|weight| (weight, 0)))
            .collect();
        pegainfer_kernels::ops::gemm_lt_tune(ctx, &samples, out_dim, rows)?;
    }
    Ok(weight)
}

impl DenseCase {
    pub(crate) fn new(kind: DenseKernelKind, rows: usize) -> Result<Self> {
        anyhow::ensure!(rows > 0, "dense case rows must be greater than zero");
        let ctx = DeviceContext::new()?;

        let buffers = match kind {
            DenseKernelKind::ProjectionGemm(projection) => {
                let (out_dim, in_dim) = projection.out_in();
                DenseBuffers::Gemm {
                    weight: gemm_weight_tuned(&ctx, out_dim, in_dim, rows)?,
                    x: hidden_of(&ctx, in_dim, rows, 0.01)?,
                    out: HiddenStates::zeros(&ctx, out_dim, rows)?,
                }
            }
            DenseKernelKind::RmsNorm => DenseBuffers::Norm {
                weight: ones_vec(&ctx, HIDDEN_SIZE)?,
                x: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.01)?,
                out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
            },
            DenseKernelKind::FusedAddRmsNorm => DenseBuffers::FusedAddNorm {
                weight: ones_vec(&ctx, HIDDEN_SIZE)?,
                hidden: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.01)?,
                residual: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.02)?,
                out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
            },
            DenseKernelKind::QkNormRopeDecode => {
                let positions: Vec<i32> = (0..rows)
                    .map(|i| ((i * 997) % DENSE_ROPE_CACHE_TOKENS) as i32)
                    .collect();
                let (cos_cache, sin_cache) = precompute_rope(
                    &ctx,
                    &RopeTableSpec {
                        rotary_dim: HEAD_DIM,
                        frequency_dim: HEAD_DIM,
                        max_seq_len: DENSE_ROPE_CACHE_TOKENS,
                        theta: 1e6,
                    },
                )?;
                DenseBuffers::QkRope {
                    q: hidden_of(&ctx, Q_DIM, rows, 0.01)?,
                    k: hidden_of(&ctx, KV_DIM, rows, 0.01)?,
                    q_norm: ones_vec(&ctx, HEAD_DIM)?,
                    k_norm: ones_vec(&ctx, HEAD_DIM)?,
                    cos_cache,
                    sin_cache,
                    positions: ctx.stream.clone_htod(&positions)?,
                }
            }
            DenseKernelKind::SiluMul => DenseBuffers::SiluMul {
                gate: hidden_of(&ctx, INTERMEDIATE_SIZE, rows, 0.01)?,
                up: hidden_of(&ctx, INTERMEDIATE_SIZE, rows, 0.02)?,
                out: HiddenStates::zeros(&ctx, INTERMEDIATE_SIZE, rows)?,
            },
            DenseKernelKind::Embedding => {
                let token_ids: Vec<u32> = (0..rows)
                    .map(|i| ((i * 7919) % VOCAB_SIZE) as u32)
                    .collect();
                DenseBuffers::Embedding {
                    table: zeros_matrix(&ctx, VOCAB_SIZE, HIDDEN_SIZE)?,
                    token_ids: ctx.stream.clone_htod(&token_ids)?,
                    out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
                }
            }
            DenseKernelKind::Sampling { greedy } => {
                let params = if greedy {
                    pegainfer_frontend::sampler::SamplingParams::default()
                } else {
                    pegainfer_frontend::sampler::SamplingParams {
                        temperature: 0.8,
                        top_k: 50,
                        top_p: 0.9,
                        min_p: 0.0,
                        seed: None,
                        ignore_eos: true,
                    }
                };
                DenseBuffers::Sampling {
                    logits: peaked_logits(&ctx, rows)?,
                    scratch: pegainfer_sample::SampleScratch::new(&ctx, VOCAB_SIZE, rows)?,
                    params: vec![params; rows],
                    seed: 0x5eed,
                }
            }
        };

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let case = Self {
            ctx,
            buffers,
            start,
            end,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub(crate) fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub(crate) fn pre_measure(&mut self) -> Result<()> {
        self.launch_once()?;
        self.ctx.sync()
    }

    pub(crate) fn launch_once(&mut self) -> Result<()> {
        use pegainfer_kernels::ops as kops;
        match &mut self.buffers {
            DenseBuffers::Gemm { weight, x, out } => {
                kops::gemm_into(&self.ctx, weight, x, out);
                Ok(())
            }
            DenseBuffers::Norm { weight, x, out } => {
                kops::rms_norm_batch_into(&self.ctx, x, weight, RMS_NORM_EPS, out);
                Ok(())
            }
            DenseBuffers::FusedAddNorm {
                weight,
                hidden,
                residual,
                out,
            } => kops::fused_add_rms_norm_round_batch_into(
                &self.ctx,
                hidden,
                residual,
                weight,
                RMS_NORM_EPS,
                out,
            ),
            DenseBuffers::QkRope {
                q,
                k,
                q_norm,
                k_norm,
                cos_cache,
                sin_cache,
                positions,
            } => {
                kops::qk_norm_rope_batch_decode_into(
                    &self.ctx,
                    q,
                    k,
                    0,
                    q.seq_len,
                    q_norm,
                    k_norm,
                    cos_cache,
                    sin_cache,
                    positions,
                    NUM_QO_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    RMS_NORM_EPS,
                )?;
                Ok(())
            }
            DenseBuffers::SiluMul { gate, up, out } => {
                kops::silu_mul_batch_into(&self.ctx, gate, up, out)
            }
            DenseBuffers::Embedding {
                table,
                token_ids,
                out,
            } => kops::embedding_batch(&self.ctx, table, token_ids, out),
            DenseBuffers::Sampling {
                logits,
                scratch,
                params,
                seed,
            } => {
                let param_refs: Vec<&pegainfer_frontend::sampler::SamplingParams> =
                    params.iter().collect();
                let steps = vec![0u64; param_refs.len()];
                *seed = seed.wrapping_add(1);
                pegainfer_sample::select_batch(
                    &self.ctx,
                    logits,
                    &param_refs,
                    &steps,
                    *seed,
                    scratch,
                )?;
                Ok(())
            }
        }
    }

    /// Cold-L2 latency, same protocol as the attention cases. The sampling
    /// case's measured span includes its device-to-host token readback and
    /// stream sync — that is the production step-tail cost, not overhead.
    pub(crate) fn measure_cold_l2(
        &mut self,
        criterion_iters: u64,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;
        for _ in 0..criterion_iters {
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_once()?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }
        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

fn hidden_of(ctx: &DeviceContext, dim: usize, rows: usize, scale: f32) -> Result<HiddenStates> {
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(&patterned_bf16(dim * rows, scale))?,
        hidden_dim: dim,
        seq_len: rows,
    })
}
