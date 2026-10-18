//! Device model build: one rank's loaded checkpoint tensors become the
//! per-layer, kernel-ready weight bundles a decode step reads.
//!
//! The bundles mirror the certified reference engine's per-layer weight struct
//! field for field, under the same names, so the executor's step reads the same
//! things the reference step reads. What the build does — and deliberately does
//! not do — to each tensor is spelled out in [`plan`]: adopt unchanged, row-fuse
//! into one GEMV operand, or fold a norm gamma into the projection it always
//! multiplies. Routed experts are adopted by move from the loader's packed
//! regions: the MXFP4 payloads are already the masked grouped GEMM's B operand
//! byte for byte, and only their e8m0 scale factors are relaid out, on-device,
//! into the GEMM's packed-i32 SF layout.
//!
//! Dense projections keep the checkpoint's `[out, in]` row-major orientation —
//! the executor's cuBLASLt GEMMs take it with `OP_T`. The one genuine transpose
//! is the KDA conv taps, which `conv_silu` reads as `[taps, inner]`; at ~196
//! KiB per tensor the build does it on a host round trip. See [`plan`] for the
//! full slot-by-slot contract.
//!
//! `build` accepts a layer count below the architecture's so bring-up can run a
//! shallow model. A truncated build discards the layers it does not take
//! (freeing their device buffers immediately) and then holds the loader to the
//! same full invariant: nothing resident may go unaccounted for.

// Bring-up: the model build lands ahead of the decode executor that reads it.
#![allow(dead_code)]

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use log::info;
use pegainfer_kernels::ops::K3DeepGemmFp8Fp4Kind;
use pegainfer_kernels::ops::k3_fp4_sf_prepare_launch;
use pegainfer_kernels::ops::k3_mega_prepare_l1_weights_launch;
use pegainfer_kernels::ops::k3_mega_prepare_sf_launch;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;

use crate::config::K3_ATTN_INNER;
use crate::config::K3_DENSE_INTERMEDIATE;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HEADS;
use crate::config::K3_HIDDEN;
use crate::config::K3_KDA_CONV_KERNEL;
use crate::config::K3_KDA_GATE_RANK;
use crate::config::K3_KV_B_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_LAYERS;
use crate::config::K3_MXFP4_GROUP;
use crate::config::K3_O_PROJ_IN;
use crate::config::K3_Q_B_OUT;
use crate::config::K3_Q_LORA_RANK;
use crate::config::K3_ROUTED_EXPERT_HIDDEN;
use crate::config::K3_SHARED_INTERMEDIATE;
use crate::config::K3_VOCAB;
use crate::config::K3LayerKind;
use crate::config::K3MoeTopo;
use crate::config::K3RoutedExperts;
use crate::config::k3_layer_is_moe;
use crate::config::k3_layer_kind;
use crate::weights::K3ExpertLayerRegions;
use crate::weights::K3RankGpuWeights;

mod build;
mod plan;

use build::K3SlotBuffers;
use plan::K3_KDA_BIG_ROWS;
use plan::K3_KDA_WSM_ROWS;
use plan::K3_MLA_FUSED_ROWS;
use plan::K3LayerGeometry;
use plan::k3_bookend_slots;
use plan::k3_layer_geometry;
use plan::k3_layer_slots;
pub(crate) use plan::k3_mla_scale;

/// K elements one packed i32 scale-factor word covers on the FP4 weight side:
/// four consecutive group-32 exponents, LSB first.
const K3_FP4_SF_WORD_K: usize = 4 * K3_MXFP4_GROUP;

/// KDA (linear-attention) layer weights.
pub(crate) struct K3KdaWeights {
    /// Fused q/k/v/gate projection, bf16 `[4 * inner, hidden]`.
    pub(crate) wbig: DeviceMatrix,
    /// Fused per-head beta + low-rank forget gate, bf16
    /// `[wsm_rows, hidden]`, tail rows zero.
    pub(crate) wsm: DeviceMatrix,
    /// Forget-gate up projection, bf16 `[inner, gate_rank]`.
    pub(crate) w_f_b: DeviceMatrix,
    /// Short depthwise conv taps, f32 `[taps, inner]` (transposed out of the
    /// checkpoint's `[inner, 1, taps]`), one per q/k/v stream.
    pub(crate) cw_q: CudaSlice<f32>,
    pub(crate) cw_k: CudaSlice<f32>,
    pub(crate) cw_v: CudaSlice<f32>,
    /// f32 `[inner]`.
    pub(crate) dt_bias: CudaSlice<f32>,
    /// Per-head decay, f32 `[heads]` (the loader trimmed the stored padding).
    pub(crate) a_log: CudaSlice<f32>,
    /// Output RMSNorm gamma, f32 `[head_dim]`.
    pub(crate) gamma_o: CudaSlice<f32>,
    /// Output projection, bf16 `[hidden, inner]`.
    pub(crate) w_o: DeviceMatrix,
}

/// MLA (full-attention) layer weights.
pub(crate) struct K3MlaWeights {
    /// Fused q_a / kv_a-with-MQA / gate projection, bf16 `[fused, hidden]`.
    pub(crate) wfu: DeviceMatrix,
    pub(crate) gamma_q_a: DeviceVec,
    pub(crate) gamma_kv_a: DeviceVec,
    pub(crate) w_q_b: DeviceMatrix,
    pub(crate) w_kv_b: DeviceMatrix,
    pub(crate) w_o: DeviceMatrix,
    /// Softmax scale as a device scalar, bf16 `[1]`.
    pub(crate) scale: DeviceVec,
}

/// Latent-MoE layer weights: router, the latent down/up projections around the
/// routed-expert stack, the fused shared expert, and this rank's MXFP4 experts.
pub(crate) struct K3MoeWeights {
    pub(crate) w_router: DeviceMatrix,
    /// Router score correction bias, f32 `[experts]`.
    pub(crate) bias: CudaSlice<f32>,
    /// Routed scaling factor as a device scalar, bf16 `[1]`.
    pub(crate) rs: DeviceVec,
    pub(crate) w_lat_down: DeviceMatrix,
    pub(crate) w_lat_up: DeviceMatrix,
    pub(crate) gamma_lat: DeviceVec,
    /// Fused shared-expert gate+up, bf16 `[2 * shared_inter, hidden]`.
    pub(crate) wsh: DeviceMatrix,
    pub(crate) sh_down: DeviceMatrix,
    /// This rank's routed experts, ready for the masked grouped GEMM.
    pub(crate) experts: K3ExpertBank,
}

/// One layer's rank-local routed experts in the layout the DeepGEMM masked
/// FP8xFP4 grouped GEMM takes.
///
/// The MXFP4 payloads are the loader's buffers adopted by move: the GEMM's B
/// operand is K-major packed `[experts, n, k / 2]`, byte-identical to the
/// checkpoint's N-row-major storage, and the gate|up call's `n` is exactly the
/// fused `[gate; up]` the loader already writes. Only the scale factors change
/// shape — the checkpoint's K-major e8m0 bytes are repacked on-device into the
/// MN-major i32 words the GEMM's SF TMA descriptor reads.
pub(crate) struct K3ExpertBank {
    /// Fused gate|up weights, fp4 e2m1 `[experts, 2 * moe_inter, latent / 2]`.
    /// Row order is the checkpoint's split-half `[gate | up]` for
    /// [`K3ExpertBankForm::MaskedChain`] and granularity-8 interleaved for
    /// [`K3ExpertBankForm::Mega`].
    pub(crate) w13_weight: CudaSlice<u8>,
    /// Their scale factors, i32 `[experts, latent / 128, 2 * moe_inter]`.
    pub(crate) w13_scale: CudaSlice<i32>,
    /// Down weights, fp4 e2m1 `[experts, latent, moe_inter / 2]`.
    pub(crate) w2_weight: CudaSlice<u8>,
    /// Their scale factors, i32 `[experts, moe_inter / 128, latent]`.
    pub(crate) w2_scale: CudaSlice<i32>,
    pub(crate) local_experts: usize,
    /// Which consumer this bank was laid out for.
    pub(crate) form: K3ExpertBankForm,
}

/// The two mutually exclusive layouts a bank can carry.
///
/// A rank holds 84-189 GiB of experts, so the two forms are never resident at
/// once: the build picks one and the executor's routed-expert path must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum K3ExpertBankForm {
    /// What the masked grouped GEMM chain reads: split-half gate|up rows and
    /// straight MN-major packed scale factors.
    MaskedChain,
    /// What the fused MegaMoE kernel reads: gate|up rows interleaved at
    /// granularity 8, and scale factors additionally UTCCP-transposed (the L1
    /// side after the same interleave).
    Mega,
}

impl K3ExpertBank {
    /// Adopt one layer's packed regions, repacking the two scale regions into
    /// the layout `form` asks for. The e8m0 source bytes are freed as soon as
    /// their packed form exists, so the bank never holds both.
    ///
    /// The mega form additionally permutes the gate|up payload, which needs a
    /// second full-size buffer for the duration of one layer — the source is
    /// dropped before the next layer is built, so the transient peak is one
    /// extra layer's W13, not the whole model's.
    fn from_regions(
        ctx: &DeviceContext,
        regions: K3ExpertLayerRegions,
        local_experts: usize,
        form: K3ExpertBankForm,
    ) -> Result<Self> {
        match form {
            K3ExpertBankForm::MaskedChain => {
                let w13_scale = prepare_expert_scales(
                    ctx,
                    local_experts,
                    K3DeepGemmFp8Fp4Kind::W13,
                    &regions.w13_scale,
                )?;
                let w2_scale = prepare_expert_scales(
                    ctx,
                    local_experts,
                    K3DeepGemmFp8Fp4Kind::W2,
                    &regions.w2_scale,
                )?;
                drop(regions.w13_scale);
                drop(regions.w2_scale);
                Ok(Self {
                    w13_weight: regions.w13_weight,
                    w13_scale,
                    w2_weight: regions.w2_weight,
                    w2_scale,
                    local_experts,
                    form,
                })
            }
            K3ExpertBankForm::Mega => {
                let w13_scale = prepare_mega_expert_scales(
                    ctx,
                    local_experts,
                    K3DeepGemmFp8Fp4Kind::W13,
                    &regions.w13_scale,
                )?;
                let w2_scale = prepare_mega_expert_scales(
                    ctx,
                    local_experts,
                    K3DeepGemmFp8Fp4Kind::W2,
                    &regions.w2_scale,
                )?;
                drop(regions.w13_scale);
                drop(regions.w2_scale);
                let (n, k) = K3DeepGemmFp8Fp4Kind::W13.shape();
                ensure!(
                    regions.w13_weight.len() == local_experts * n * (k / 2),
                    "K3 W13 weight region is {} bytes, expected {local_experts} x [{n}, {}]",
                    regions.w13_weight.len(),
                    k / 2
                );
                let mut w13_weight = ctx
                    .stream
                    .alloc_zeros::<u8>(regions.w13_weight.len())
                    .context("alloc K3 MegaMoE interleaved gate|up bank")?;
                k3_mega_prepare_l1_weights_launch(
                    ctx,
                    local_experts,
                    n,
                    k,
                    &regions.w13_weight,
                    &mut w13_weight,
                )
                .context("K3 MegaMoE gate|up interleave")?;
                // Free the split-half source before the next layer allocates.
                ctx.sync()
                    .context("sync after K3 MegaMoE gate|up interleave")?;
                drop(regions.w13_weight);
                Ok(Self {
                    w13_weight,
                    w13_scale,
                    w2_weight: regions.w2_weight,
                    w2_scale,
                    local_experts,
                    form,
                })
            }
        }
    }

    fn bytes(&self) -> usize {
        self.w13_weight.len()
            + self.w13_scale.len() * 4
            + self.w2_weight.len()
            + self.w2_scale.len() * 4
    }
}

/// Repack one region of checkpoint e8m0 scale bytes `[experts, n, k / 32]` into
/// the GEMM's `[experts, k / 128, n]` i32 words.
fn prepare_expert_scales(
    ctx: &DeviceContext,
    local_experts: usize,
    kind: K3DeepGemmFp8Fp4Kind,
    source: &CudaSlice<u8>,
) -> Result<CudaSlice<i32>> {
    let (n, k) = kind.shape();
    ensure!(
        source.len() == local_experts * n * (k / K3_MXFP4_GROUP),
        "K3 {kind:?} scale region is {} bytes, expected {local_experts} x [{n}, {}]",
        source.len(),
        k / K3_MXFP4_GROUP
    );
    let words = local_experts * (k / K3_FP4_SF_WORD_K) * n;
    let mut packed = ctx
        .stream
        .alloc_zeros::<i32>(words)
        .context("alloc K3 expert scale-factor tensor")?;
    k3_fp4_sf_prepare_launch(ctx, local_experts, n, k, source, &mut packed)
        .with_context(|| format!("K3 {kind:?} scale-factor prepare"))?;
    Ok(packed)
}

/// Repack one region of checkpoint e8m0 scale bytes `[experts, n, k / 32]` into
/// the MegaMoE `[experts, k / 128, n]` i32 words: the same 4:1 pack as
/// [`prepare_expert_scales`] plus the UTCCP row transpose, and for W13 the
/// gate|up interleave that must precede it.
fn prepare_mega_expert_scales(
    ctx: &DeviceContext,
    local_experts: usize,
    kind: K3DeepGemmFp8Fp4Kind,
    source: &CudaSlice<u8>,
) -> Result<CudaSlice<i32>> {
    let (n, k) = kind.shape();
    ensure!(
        source.len() == local_experts * n * (k / K3_MXFP4_GROUP),
        "K3 {kind:?} scale region is {} bytes, expected {local_experts} x [{n}, {}]",
        source.len(),
        k / K3_MXFP4_GROUP
    );
    let words = local_experts * (k / K3_FP4_SF_WORD_K) * n;
    let mut packed = ctx
        .stream
        .alloc_zeros::<i32>(words)
        .context("alloc K3 MegaMoE expert scale-factor tensor")?;
    k3_mega_prepare_sf_launch(
        ctx,
        local_experts,
        n,
        k,
        kind == K3DeepGemmFp8Fp4Kind::W13,
        source,
        &mut packed,
    )
    .with_context(|| format!("K3 MegaMoE {kind:?} scale-factor prepare"))?;
    Ok(packed)
}

/// Dense MLP weights (layer 0 only).
pub(crate) struct K3DenseMlpWeights {
    /// Fused gate+up, bf16 `[2 * intermediate, hidden]`.
    pub(crate) wgu: DeviceMatrix,
    pub(crate) w_dn: DeviceMatrix,
}

pub(crate) enum K3LayerAttention {
    Kda(Box<K3KdaWeights>),
    Mla(Box<K3MlaWeights>),
}

pub(crate) enum K3LayerMlp {
    Dense(Box<K3DenseMlpWeights>),
    Moe(Box<K3MoeWeights>),
}

/// One decoder layer's device weights plus its place in the attention-residual
/// stream.
pub(crate) struct K3LayerWeights {
    pub(crate) geometry: K3LayerGeometry,
    /// Attention-input RMSNorm gamma, bf16 `[hidden]`.
    pub(crate) gamma_in: DeviceVec,
    /// MLP-input RMSNorm gamma, bf16 `[hidden]`.
    pub(crate) gamma_post: DeviceVec,
    /// Attention-residual scoring vector for the pre-attention mix, f32
    /// `[hidden]` (norm gamma already folded into the projection).
    pub(crate) sw_attn: CudaSlice<f32>,
    /// Same, for the pre-MLP mix.
    pub(crate) sw_mlp: CudaSlice<f32>,
    pub(crate) attn: K3LayerAttention,
    pub(crate) mlp: K3LayerMlp,
}

/// Device bytes the built model holds, by class.
///
/// * `backbone` — every per-layer weight: attention, MLP backbone, norms, the
///   folded scoring vectors and the two per-layer device scalars.
/// * `experts` — the rank-local routed experts' MXFP4 payload and scale bytes.
/// * `other` — the model-level bookends: embedding, final norm, output scoring
///   vector, LM head.
///
/// Counts are logical buffer bytes, not the allocator's rounded reservations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct K3ModelVram {
    pub(crate) backbone: usize,
    pub(crate) experts: usize,
    pub(crate) other: usize,
}

impl K3ModelVram {
    pub(crate) fn total_bytes(self) -> usize {
        self.backbone + self.experts + self.other
    }
}

/// One rank's device-resident model.
pub(crate) struct K3RankModel {
    pub(crate) rank: usize,
    pub(crate) topo: K3MoeTopo,
    /// Token embedding, bf16 `[vocab, hidden]`.
    pub(crate) embed: DeviceMatrix,
    /// Final RMSNorm gamma, bf16 `[hidden]`.
    pub(crate) gamma_final: DeviceVec,
    /// Output-side attention-residual scoring vector, f32 `[hidden]`.
    pub(crate) sw_out: CudaSlice<f32>,
    /// LM head, bf16 `[vocab, hidden]`.
    pub(crate) w_lm: DeviceMatrix,
    pub(crate) layers: Vec<K3LayerWeights>,
    /// Attention-residual blocks the output mix sees.
    pub(crate) blocks: usize,
    pub(crate) vram: K3ModelVram,
}

impl K3RankModel {
    /// Build this rank's model out of its loaded tensors.
    ///
    /// `num_layers` may be below [`K3_LAYERS`] for bring-up: the layers above
    /// it are discarded rather than built, and their device buffers are freed
    /// before this returns. Either way the loader's resident set must come out
    /// empty — a truncated build relaxes what the model *takes*, never the
    /// invariant that everything uploaded is accounted for.
    pub(crate) fn build(
        ctx: &DeviceContext,
        mut weights: K3RankGpuWeights,
        topo: K3MoeTopo,
        rank: usize,
        num_layers: usize,
        form: K3ExpertBankForm,
    ) -> Result<Self> {
        ensure!(
            (1..=K3_LAYERS).contains(&num_layers),
            "K3 model must have 1..={K3_LAYERS} layers, got {num_layers}"
        );
        ensure!(
            rank < topo.device_count(),
            "K3 rank must be in 0..{}, got {rank}",
            topo.device_count()
        );
        let routed_experts = topo.routed_experts();

        let mut bookends = K3SlotBuffers::materialize(ctx, &mut weights, &k3_bookend_slots())?;
        let mut vram = K3ModelVram {
            other: bookends.bytes,
            ..K3ModelVram::default()
        };
        let embed = bookends.matrix("embed", K3_VOCAB, K3_HIDDEN)?;
        let gamma_final = bookends.vector("gamma_final", K3_HIDDEN)?;
        let sw_out = bookends.f32("sw_out", K3_HIDDEN, 1)?;
        let w_lm = bookends.matrix("w_lm", K3_VOCAB, K3_HIDDEN)?;
        bookends.ensure_drained("bookend")?;

        let (geometry, blocks) = k3_layer_geometry(num_layers);
        let mut layers = Vec::with_capacity(num_layers);
        for geom in geometry {
            let (layer, layer_vram) =
                build_layer(ctx, &mut weights, geom, topo, routed_experts, form)?;
            vram.backbone += layer_vram.backbone;
            vram.experts += layer_vram.experts;
            layers.push(layer);
        }

        let discarded = discard_layers_above(&mut weights, num_layers, routed_experts);
        weights.ensure_consumed()?;

        info!(
            "K3 rank {rank} model built: layers={num_layers}/{K3_LAYERS}, local_experts={}, \
             vram backbone={:.2} GiB, experts={:.2} GiB, other={:.2} GiB, total={:.2} GiB\
             {}",
            topo.local_experts(),
            gib(vram.backbone),
            gib(vram.experts),
            gib(vram.other),
            gib(vram.total_bytes()),
            if discarded == 0 {
                String::new()
            } else {
                format!(
                    " (truncated: freed {:.2} GiB of unbuilt layers)",
                    gib(discarded)
                )
            }
        );

        Ok(Self {
            rank,
            topo,
            embed,
            gamma_final,
            sw_out,
            w_lm,
            layers,
            blocks,
            vram,
        })
    }
}

fn build_layer(
    ctx: &DeviceContext,
    weights: &mut K3RankGpuWeights,
    geometry: K3LayerGeometry,
    topo: K3MoeTopo,
    routed_experts: K3RoutedExperts,
    form: K3ExpertBankForm,
) -> Result<(K3LayerWeights, K3ModelVram)> {
    let layer = geometry.layer;
    let mut slots =
        K3SlotBuffers::materialize(ctx, weights, &k3_layer_slots(layer, routed_experts))?;
    let mut vram = K3ModelVram {
        backbone: slots.bytes,
        ..K3ModelVram::default()
    };

    let gamma_in = slots.vector("gamma_in", K3_HIDDEN)?;
    let gamma_post = slots.vector("gamma_post", K3_HIDDEN)?;
    let sw_attn = slots.f32("sw_attn", K3_HIDDEN, 1)?;
    let sw_mlp = slots.f32("sw_mlp", K3_HIDDEN, 1)?;

    let attn = match k3_layer_kind(layer) {
        K3LayerKind::Kda => K3LayerAttention::Kda(Box::new(K3KdaWeights {
            wbig: slots.matrix("wbig", K3_KDA_BIG_ROWS, K3_HIDDEN)?,
            wsm: slots.matrix("wsm", K3_KDA_WSM_ROWS, K3_HIDDEN)?,
            w_f_b: slots.matrix("w_f_b", K3_ATTN_INNER, K3_KDA_GATE_RANK)?,
            cw_q: slots.f32("cw_q", K3_KDA_CONV_KERNEL, K3_ATTN_INNER)?,
            cw_k: slots.f32("cw_k", K3_KDA_CONV_KERNEL, K3_ATTN_INNER)?,
            cw_v: slots.f32("cw_v", K3_KDA_CONV_KERNEL, K3_ATTN_INNER)?,
            dt_bias: slots.f32("dt_bias", K3_ATTN_INNER, 1)?,
            a_log: slots.f32("a_log", K3_HEADS, 1)?,
            gamma_o: slots.f32("gamma_o", K3_HEAD_DIM, 1)?,
            w_o: slots.matrix("w_o", K3_HIDDEN, K3_O_PROJ_IN)?,
        })),
        K3LayerKind::Mla => K3LayerAttention::Mla(Box::new(K3MlaWeights {
            wfu: slots.matrix("wfu", K3_MLA_FUSED_ROWS, K3_HIDDEN)?,
            gamma_q_a: slots.vector("gamma_q_a", K3_Q_LORA_RANK)?,
            gamma_kv_a: slots.vector("gamma_kv_a", K3_KV_LORA_RANK)?,
            w_q_b: slots.matrix("w_q_b", K3_Q_B_OUT, K3_Q_LORA_RANK)?,
            w_kv_b: slots.matrix("w_kv_b", K3_KV_B_OUT, K3_KV_LORA_RANK)?,
            w_o: slots.matrix("w_o", K3_HIDDEN, K3_O_PROJ_IN)?,
            scale: slots.vector("scale", 1)?,
        })),
    };

    let mlp = if k3_layer_is_moe(layer) {
        let experts = K3ExpertBank::from_regions(
            ctx,
            weights.take_expert_layer(layer)?,
            topo.local_experts(),
            form,
        )?;
        vram.experts += experts.bytes();
        K3LayerMlp::Moe(Box::new(K3MoeWeights {
            w_router: slots.matrix("w_router", routed_experts.count(), K3_HIDDEN)?,
            bias: slots.f32("bias", routed_experts.count(), 1)?,
            rs: slots.vector("rs", 1)?,
            w_lat_down: slots.matrix("w_lat_down", K3_ROUTED_EXPERT_HIDDEN, K3_HIDDEN)?,
            w_lat_up: slots.matrix("w_lat_up", K3_HIDDEN, K3_ROUTED_EXPERT_HIDDEN)?,
            gamma_lat: slots.vector("gamma_lat", K3_ROUTED_EXPERT_HIDDEN)?,
            wsh: slots.matrix("wsh", 2 * K3_SHARED_INTERMEDIATE, K3_HIDDEN)?,
            sh_down: slots.matrix("sh_down", K3_HIDDEN, K3_SHARED_INTERMEDIATE)?,
            experts,
        }))
    } else {
        K3LayerMlp::Dense(Box::new(K3DenseMlpWeights {
            wgu: slots.matrix("wgu", 2 * K3_DENSE_INTERMEDIATE, K3_HIDDEN)?,
            w_dn: slots.matrix("w_dn", K3_HIDDEN, K3_DENSE_INTERMEDIATE)?,
        }))
    };
    slots.ensure_drained(&format!("layer {layer}"))?;

    Ok((
        K3LayerWeights {
            geometry,
            gamma_in,
            gamma_post,
            sw_attn,
            sw_mlp,
            attn,
            mlp,
        },
        vram,
    ))
}

fn expert_region_bytes(regions: &K3ExpertLayerRegions) -> usize {
    regions.w13_weight.len()
        + regions.w13_scale.len()
        + regions.w2_weight.len()
        + regions.w2_scale.len()
}

/// Drop the resident tensors of the layers a truncated build does not take,
/// returning the bytes freed. Taking them by name (rather than leaving them in
/// the map) keeps the loader's "everything resident was consumed" invariant
/// exact: an unexpected leftover still fails `ensure_consumed`, and the device
/// memory is released here instead of at the end of the load bundle's life.
///
/// A load bundle planned for the same truncation never made these resident in
/// the first place, so an absent name is not an error here — anything else left
/// over is still caught by `ensure_consumed`.
fn discard_layers_above(
    weights: &mut K3RankGpuWeights,
    num_layers: usize,
    routed_experts: K3RoutedExperts,
) -> usize {
    let mut bytes = 0usize;
    for layer in num_layers..K3_LAYERS {
        for plan in k3_layer_slots(layer, routed_experts) {
            for name in &plan.sources {
                if let Ok(tensor) = weights.take_tensor(name) {
                    bytes += tensor.len();
                }
            }
        }
        if k3_layer_is_moe(layer)
            && let Ok(regions) = weights.take_expert_layer(layer)
        {
            bytes += expert_region_bytes(&regions);
        }
    }
    bytes
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::K3_EXPERT_INTERMEDIATE;
    use crate::config::K3_KV_A_OUT;
    use crate::config::K3_MXFP4_GROUP;
    use crate::config::K3_MXFP4_PACK;
    use crate::weights::K3RankGpuContext;
    use crate::weights::K3WeightManifest;
    use crate::weights::load_rank_weights_to_gpu;
    use crate::weights::mmap_file;

    const BF16: usize = 2;
    const F32: usize = 4;

    fn env_usize(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(default)
    }

    /// Per-class bytes of a `num_layers`-deep model, derived from the
    /// architecture constants independently of the build plan.
    fn expected_vram(num_layers: usize, experts: usize, local_experts: usize) -> K3ModelVram {
        let h = K3_HIDDEN;
        // gamma_in + gamma_post (bf16) and the two folded scoring vectors (f32).
        let common = 2 * h * BF16 + 2 * h * F32;
        let kda = 4 * K3_ATTN_INNER * h * BF16
            + K3_KDA_WSM_ROWS * h * BF16
            + K3_ATTN_INNER * K3_KDA_GATE_RANK * BF16
            + 3 * K3_ATTN_INNER * K3_KDA_CONV_KERNEL * F32
            + K3_ATTN_INNER * F32
            + K3_HEADS * F32
            + K3_HEAD_DIM * F32
            + h * K3_O_PROJ_IN * BF16;
        let mla = (K3_Q_LORA_RANK + K3_KV_A_OUT + K3_ATTN_INNER) * h * BF16
            + K3_Q_LORA_RANK * BF16
            + K3_KV_LORA_RANK * BF16
            + K3_Q_B_OUT * K3_Q_LORA_RANK * BF16
            + K3_KV_B_OUT * K3_KV_LORA_RANK * BF16
            + h * K3_O_PROJ_IN * BF16
            + BF16;
        let dense = 2 * K3_DENSE_INTERMEDIATE * h * BF16 + h * K3_DENSE_INTERMEDIATE * BF16;
        let moe = experts * h * BF16
            + experts * F32
            + BF16
            + 2 * K3_ROUTED_EXPERT_HIDDEN * h * BF16
            + K3_ROUTED_EXPERT_HIDDEN * BF16
            + 2 * K3_SHARED_INTERMEDIATE * h * BF16
            + h * K3_SHARED_INTERMEDIATE * BF16;
        // One expert: [gate; up] MXFP4 payload + scales, then w2 alone.
        let expert = 2 * K3_EXPERT_INTERMEDIATE * (K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_PACK)
            + 2 * K3_EXPERT_INTERMEDIATE * (K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_GROUP)
            + K3_ROUTED_EXPERT_HIDDEN * (K3_EXPERT_INTERMEDIATE / K3_MXFP4_PACK)
            + K3_ROUTED_EXPERT_HIDDEN * (K3_EXPERT_INTERMEDIATE / K3_MXFP4_GROUP);

        let mut vram = K3ModelVram {
            other: 2 * K3_VOCAB * h * BF16 + h * BF16 + h * F32,
            ..K3ModelVram::default()
        };
        for layer in 0..num_layers {
            vram.backbone += common;
            vram.backbone += match k3_layer_kind(layer) {
                K3LayerKind::Kda => kda,
                K3LayerKind::Mla => mla,
            };
            if k3_layer_is_moe(layer) {
                vram.backbone += moe;
                vram.experts += local_experts * expert;
            } else {
                vram.backbone += dense;
            }
        }
        vram
    }

    /// Byte accounting of the 4-layer bring-up model, on the host. Pins the
    /// per-class totals the GPU build test asserts, so a plan change that
    /// silently resizes a slot fails without a GPU.
    #[test]
    fn truncated_model_vram_accounting_is_pinned() {
        let vram = expected_vram(4, 224, 56);
        // Layers 0-2 are KDA, layer 3 is MLA; layer 0 is the one dense MLP.
        assert_eq!(vram.backbone, 5_693_503_752);
        assert_eq!(vram.experts, 3 * 56 * 17_547_264);
        assert_eq!(vram.other, 4_697_663_488);
        // The plan's own byte sum must agree with the independent derivation.
        let experts = K3RoutedExperts::new(224).unwrap();
        let planned: usize = plan::k3_model_slots(4, experts)
            .iter()
            .map(plan::K3SlotPlan::bytes)
            .sum();
        assert_eq!(planned, vram.backbone + vram.other);
    }

    /// Build a truncated rank-0 model from the real checkpoint and check the
    /// per-class byte totals. Requires a GPU and the mounted checkpoint.
    ///
    /// The loader plans the FULL 93-layer rank, so the peak device footprint is
    /// the whole rank's weights regardless of the truncation; raise
    /// `PEGAINFER_K3_TEST_EP` to shrink the rank's expert shard on a busy GPU.
    #[test]
    #[ignore = "requires a GPU and the K3 checkpoint"]
    fn builds_a_truncated_rank_model_from_the_checkpoint() {
        let model_dir =
            std::env::var("PEGAINFER_K3_TEST_224").unwrap_or_else(|_| "models/Kimi-K3-224".into());
        let dir = Path::new(&model_dir);
        assert!(dir.exists(), "{model_dir} is not mounted");
        let num_layers = env_usize("PEGAINFER_K3_TEST_LAYERS", 4);
        let ep_size = env_usize("PEGAINFER_K3_TEST_EP", 4);
        let device = env_usize("PEGAINFER_K3_TEST_DEVICE", 0);

        let manifest = K3WeightManifest::from_model_dir(dir).unwrap();
        let topo = K3MoeTopo::new(manifest.routed_experts(), ep_size).unwrap();
        let bundle = manifest.rank_load_bundle(0, topo).unwrap();
        let gpu = K3RankGpuContext::new(device).unwrap();
        let ctx = gpu.device_context().unwrap();
        let loaded = load_rank_weights_to_gpu(&gpu, dir, &bundle, false).unwrap();

        let model = K3RankModel::build(
            &ctx,
            loaded.weights,
            topo,
            0,
            num_layers,
            K3ExpertBankForm::MaskedChain,
        )
        .unwrap();
        gpu.sync().unwrap();

        let expected = expected_vram(
            num_layers,
            manifest.routed_experts().count(),
            topo.local_experts(),
        );
        eprintln!(
            "K3 rank 0 model ({num_layers} layers, EP{ep_size}): backbone={:.2} GiB, \
             experts={:.2} GiB, other={:.2} GiB, total={:.2} GiB",
            gib(model.vram.backbone),
            gib(model.vram.experts),
            gib(model.vram.other),
            gib(model.vram.total_bytes()),
        );
        assert_eq!(model.vram, expected);
        assert_eq!(model.layers.len(), num_layers);
        assert_eq!(model.blocks, num_layers.div_ceil(12));
        assert!(matches!(model.layers[0].mlp, K3LayerMlp::Dense(_)));
        assert!(matches!(model.layers[0].attn, K3LayerAttention::Kda(_)));
        if num_layers > 3 {
            assert!(matches!(model.layers[3].attn, K3LayerAttention::Mla(_)));
            let K3LayerMlp::Moe(moe) = &model.layers[3].mlp else {
                panic!("layer 3 must be a MoE layer");
            };
            // The expert bank must match the masked grouped GEMM's operand
            // contract: K-major fp4 payload adopted as-is, scale factors
            // repacked into MN-major i32 words.
            let local = topo.local_experts();
            let bank = &moe.experts;
            assert_eq!(bank.local_experts, local);
            for (kind, weight, scale) in [
                (
                    K3DeepGemmFp8Fp4Kind::W13,
                    bank.w13_weight.len(),
                    bank.w13_scale.len(),
                ),
                (
                    K3DeepGemmFp8Fp4Kind::W2,
                    bank.w2_weight.len(),
                    bank.w2_scale.len(),
                ),
            ] {
                let (n, k) = kind.shape();
                assert_eq!(weight, local * n * (k / K3_MXFP4_PACK), "{kind:?} payload");
                assert_eq!(scale, local * (k / K3_FP4_SF_WORD_K) * n, "{kind:?} scales");
            }
        }
        assert_eq!(model.embed.rows, K3_VOCAB);
        assert_eq!(model.sw_out.len(), K3_HIDDEN);
        // The conv taps are transposed at build: check the built buffer
        // element for element against the checkpoint tensor it came from —
        // the transpose is not size-detectable, so only values prove it.
        let K3LayerAttention::Kda(kda) = &model.layers[0].attn else {
            panic!("layer 0 must be a KDA layer");
        };
        let name = "language_model.model.layers.0.self_attn.q_conv1d.weight";
        let mmap = mmap_file(&dir.join(manifest.shard_for(name).unwrap())).unwrap();
        let store = safetensors::SafeTensors::deserialize(&mmap).unwrap();
        let src = store
            .tensor(name)
            .unwrap()
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes).to_bits())
            .collect::<Vec<u32>>();
        let built = ctx.stream.clone_dtoh(&kda.cw_q).unwrap();
        gpu.sync().unwrap();
        assert_eq!(built.len(), K3_KDA_CONV_KERNEL * K3_ATTN_INNER);
        assert_eq!(src.len(), built.len());
        // Bit patterns: the transpose must move values, not round them.
        for tap in 0..K3_KDA_CONV_KERNEL {
            for lane in 0..K3_ATTN_INNER {
                assert_eq!(
                    built[tap * K3_ATTN_INNER + lane].to_bits(),
                    src[lane * K3_KDA_CONV_KERNEL + tap],
                    "cw_q[{tap}, {lane}]"
                );
            }
        }
    }
}
