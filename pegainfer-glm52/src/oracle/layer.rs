//! HF-oracle gates for the full decoder-layer composition.
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py --stage layer` (official
//! `GlmMoeDsaDecoderLayer`, fp8sim precision): it runs one whole decoder layer
//! (norms + MLA/DSA attention + residuals + dense-or-MoE MLP) on a seeded input
//! and emits `layer_out` probe constants. These tests replay the same input
//! through `glm52_decoder_layer_forward` position by position (prefill-via-
//! decode, real DSA indexer — at ctx <= 2048 its top-k equals full top-k) and
//! assert the outputs land on the probes.
//!
//! Layer 0 exercises the dense MLP and full indexer. Layer 6's generated
//! probes are shared by the EP8 and TP8 topology gates in the sibling modules;
//! the obsolete EP1 local-MoE implementation is not retained here.
//!
//! Run (H200 + checkpoint + DeepGEMM env):
//! ```text
//! PEGAINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//! PEGAINFER_DEEPGEMM_ROOT=pegainfer-kernels/third_party/DeepGEMM/deep_gemm \
//! CUDA_HOME=/usr/local/cuda \
//!   cargo test --release -p pegainfer-glm52 --features glm52 --lib layer_oracle -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use sha2::Digest;
use sha2::Sha256;

use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::fp8::Glm52ProjBytes;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::indexer::Glm52IndexerScratch;
use crate::layer::Glm52DecodeStep;
use crate::layer::Glm52DecoderLayerWeights;
use crate::layer::Glm52LayerIndexer;
use crate::layer::Glm52LayerMlp;
use crate::layer::glm52_decoder_layer_forward;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::model::INDEX_CACHE_BLOCK;
use crate::model::NUM_SMS;
use crate::model::rope_tables;
use crate::moe_decode::Glm52MoeRoutedExpertBytes;
use crate::scratch::Glm52DecodeScratch;

// ---- BEGIN GENERATED: glm52_oracle layer probes (dense, layer 0) ----
// uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 \
//     --ctx 200 --seed 0x5eed604d --layer 0 --precision fp8sim \
//     --stage layer --input-scale 0.02
// transformers=5.13.0.dev0 torch=2.12.1+cu130
const DENSE_ORACLE_SEED: u64 = 0x5eed604d;
const DENSE_ORACLE_CTX: usize = 200;
const DENSE_ORACLE_LAYER: usize = 0;
const DENSE_ORACLE_INPUT_SCALE: f64 = 0.02;
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const DENSE_ORACLE_HIDDEN_DIGEST: &str = "d39daa8ba2c7f939";
// tap `layer_out` [200, 6144] bf16 digest=d70e6acd35fe115f (provenance only)
// tol = max(rel_tol 0.05 x delta_rms 8.119e-04, 3 x bf16-ulp 9.021e-05) — see emit_rust_layer.
const DENSE_ORACLE_LAYER_TOL: f32 = 2.706195228e-04;
const DENSE_ORACLE_LAYER_PROBES: &[(usize, f32)] = &[
    (7504, 3.881835938e-02),
    (10832, 3.100585938e-02),
    (30355, 3.515625000e-02),
    (33148, -3.027343750e-02),
    (69206, 2.600097656e-02),
    (146761, 3.393554688e-02),
    (156574, 3.906250000e-02),
    (161844, 2.490234375e-02),
    (240978, -1.855468750e-02),
    (307757, -2.136230469e-02),
    (319510, -2.941894531e-02),
    (333821, 2.868652344e-02),
    (337363, 6.370544434e-04),
    (345826, 1.330566406e-02),
    (368340, -3.588867188e-02),
    (377565, -2.792358398e-03),
    (387659, 2.099609375e-02),
    (432017, 3.631591797e-03),
    (442664, 3.466796875e-02),
    (446114, 2.587890625e-02),
    (468571, 1.733398438e-02),
    (471935, 1.214599609e-02),
    (488799, 8.544921875e-03),
    (520950, -3.759765625e-02),
    (530739, -2.636718750e-02),
    (534505, -3.662109375e-02),
    (534971, 3.906250000e-02),
    (577397, 1.251220703e-03),
    (604084, -3.222656250e-02),
    (636056, -2.258300781e-02),
    (668274, 2.014160156e-02),
    (672858, -3.295898438e-02),
    (714313, -6.408691406e-03),
    (743834, -2.587890625e-02),
    (791113, 1.300048828e-02),
    (802252, -1.635742188e-02),
    (807243, 1.843261719e-02),
    (818652, -1.745605469e-02),
    (878485, 1.611328125e-02),
    (879770, 3.369140625e-02),
    (880514, -3.125000000e-02),
    (903613, 3.930664062e-02),
    (915339, -3.753662109e-03),
    (931272, -2.697753906e-02),
    (943182, 1.501464844e-02),
    (949584, -1.686096191e-03),
    (980538, 1.818847656e-02),
    (980931, -2.197265625e-03),
    (1022303, -2.075195312e-02),
    (1023279, -8.483886719e-03),
    (1091288, 3.015136719e-02),
    (1092832, -5.584716797e-03),
    (1094227, 2.331542969e-02),
    (1094427, 6.347656250e-03),
    (1102135, -2.856445312e-02),
    (1104580, -2.709960938e-02),
    (1120355, 1.831054688e-02),
    (1158451, -1.708984375e-02),
    (1167014, -3.857421875e-02),
    (1176953, 3.125000000e-02),
    (1181822, 2.517700195e-03),
    (1209754, 3.967285156e-03),
    (1216021, 4.003906250e-02),
    (1218315, 3.271484375e-02),
];
// ---- END GENERATED ----

// Shared seeded-input metadata for the MoE layer oracle below.
pub(crate) const MOE_ORACLE_SEED: u64 = 0x5eed604d;
pub(crate) const MOE_ORACLE_CTX: usize = 200;
pub(crate) const MOE_ORACLE_LAYER: usize = 6;
pub(crate) const MOE_ORACLE_INPUT_SCALE: f64 = 0.02;
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
pub(crate) const MOE_ORACLE_HIDDEN_DIGEST: &str = "d39daa8ba2c7f939";

// ---- BEGIN GENERATED: glm52_oracle layer probes (moe, layer 6, deepgemm precision) ----
// Non-expert projections use the engine's bf16-activation GEMV precision.
// Routed experts use loader-time UE8M0 weight requantization and UE8M0
// activation quantization before W13 and W2, matching the SM100 DeepGEMM path.
// uv run tools/accuracy/glm52_oracle.py --model-path <GLM-5.2-FP8> \
//     --ctx 200 --seed 0x5eed604d --layer 6 --precision deepgemm \
//     --stage layer --input-scale 0.02
// transformers=5.13.1 torch=2.11.0+cu130
// tap `layer_out` [200, 6144] bf16 digest=765173988a1424b2 (provenance only)
// tol = max(rel_tol 0.05 x delta_rms 1.945e-03, 3 x bf16-ulp 9.021e-05) — see emit_rust_layer.
pub(crate) const MOE_ORACLE_DEEPGEMM_LAYER_TOL: f32 = 2.706195228e-04;
pub(crate) const MOE_ORACLE_DEEPGEMM_LAYER_PROBES: &[(usize, f32)] = &[
    (7504, 3.369140625e-02),
    (10832, 1.977539062e-02),
    (30355, 4.052734375e-02),
    (33148, -2.648925781e-02),
    (69206, 2.172851562e-02),
    (146761, 3.100585938e-02),
    (156574, 4.028320312e-02),
    (161844, 2.575683594e-02),
    (240978, -1.855468750e-02),
    (307757, -2.380371094e-02),
    (319510, -2.880859375e-02),
    (333821, 2.734375000e-02),
    (337363, -8.506774902e-04),
    (345826, 1.135253906e-02),
    (368340, -3.540039062e-02),
    (377565, -2.136230469e-03),
    (387659, 1.965332031e-02),
    (432017, 3.219604492e-03),
    (442664, 3.808593750e-02),
    (446114, 2.526855469e-02),
    (468571, 2.160644531e-02),
    (471935, 1.275634766e-02),
    (488799, 9.948730469e-03),
    (520950, -3.784179688e-02),
    (530739, -2.697753906e-02),
    (534505, -3.564453125e-02),
    (534971, 3.710937500e-02),
    (577397, 2.380371094e-03),
    (604084, -3.369140625e-02),
    (636056, -2.441406250e-02),
    (668274, 1.953125000e-02),
    (672858, -3.417968750e-02),
    (714313, -8.056640625e-03),
    (743834, -2.648925781e-02),
    (791113, 1.226806641e-02),
    (802252, -1.403808594e-02),
    (807243, 1.611328125e-02),
    (818652, -1.818847656e-02),
    (878485, 1.544189453e-02),
    (879770, 3.320312500e-02),
    (880514, -3.344726562e-02),
    (903613, 3.759765625e-02),
    (915339, -1.892089844e-03),
    (931272, -2.758789062e-02),
    (943182, 1.672363281e-02),
    (949584, -3.448486328e-03),
    (980538, 2.038574219e-02),
    (980931, -3.860473633e-03),
    (1022303, -1.855468750e-02),
    (1023279, -8.544921875e-03),
    (1091288, 3.063964844e-02),
    (1092832, -3.784179688e-03),
    (1094227, 2.001953125e-02),
    (1094427, 1.422119141e-02),
    (1102135, -2.807617188e-02),
    (1104580, -2.648925781e-02),
    (1120355, 1.977539062e-02),
    (1158451, -1.586914062e-02),
    (1167014, -3.735351562e-02),
    (1176953, 3.247070312e-02),
    (1181822, -5.187988281e-04),
    (1209754, 3.082275391e-03),
    (1216021, 4.052734375e-02),
    (1218315, 3.173828125e-02),
];
// ---- END GENERATED ----

const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const DENSE_INTERMEDIATE: usize = 12288;
const MOE_INTERMEDIATE: usize = 2048;
const EXPERTS: usize = 256;

/// Mirror of the Python splitmix64 generator (see `mla_oracle_gate`), with the
/// layer stage's input scale applied in f64 exactly like the harness.
pub(crate) fn seeded_hidden(seed: u64, count: usize, scale: f64) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            bf16::from_f32(((u - 0.5) * 4.0 * scale) as f32)
        })
        .collect()
}

fn bf16_digest(data: &[bf16]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

/// Owned copies of every `model.layers.{L}.` tensor (attention + indexer +
/// layernorms + MLP/MoE). For the MoE layer this is ~10 GB of host copies — a
/// test-only cost that keeps the borrow story trivial.
pub(crate) struct LayerTensors {
    by_name: BTreeMap<String, Vec<u8>>,
}

impl LayerTensors {
    pub(crate) fn load(model_path: &Path, layer: usize) -> Result<Self> {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            model_path.join("model.safetensors.index.json"),
        )?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("weight_map missing")?;
        let prefix = format!("model.layers.{layer}.");
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::default();
        for (name, shard) in weight_map {
            if name.starts_with(&prefix) {
                by_shard
                    .entry(shard.as_str().context("shard not a string")?.to_owned())
                    .or_default()
                    .push(name.clone());
            }
        }
        let mut by_name = BTreeMap::new();
        for (shard, names) in by_shard {
            let mmap = crate::weights::mmap_file(&model_path.join(&shard))?;
            let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
            for name in names {
                by_name.insert(name.clone(), st.tensor(&name)?.data().to_vec());
            }
        }
        Ok(Self { by_name })
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.by_name
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("layer tensor {name} not loaded"))
    }

    fn proj(&self, stem: &str, n: usize, k: usize) -> Result<Glm52ProjBytes<'_>> {
        Ok(Glm52ProjBytes {
            weight: self.bytes(&format!("{stem}.weight"))?,
            scale: self.bytes(&format!("{stem}.weight_scale_inv"))?,
            n,
            k,
        })
    }
}

/// Which live MLP half the gate loads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GateLayerMlp {
    Dense,
    MoeEp8Rank0,
    MoeEp4Rank0,
}

/// Pack one EP rank's local experts from the layer's host tensors
/// (`ranks` = 8 → 32 experts, 4 → 64 experts).
pub(crate) fn load_rank_expert_bank(
    ctx: &DeviceContext,
    t: &LayerTensors,
    layer: usize,
    rank: usize,
    ranks: usize,
) -> Result<crate::moe_decode::Glm52MoeExpertBank> {
    let mp = format!("model.layers.{layer}.mlp");
    let local = EXPERTS / ranks;
    let experts: Vec<Glm52MoeRoutedExpertBytes<'_>> = (rank * local..(rank + 1) * local)
        .map(|e| {
            let ep = format!("{mp}.experts.{e}");
            Ok(Glm52MoeRoutedExpertBytes {
                gate: t.proj(&format!("{ep}.gate_proj"), MOE_INTERMEDIATE, HIDDEN)?,
                up: t.proj(&format!("{ep}.up_proj"), MOE_INTERMEDIATE, HIDDEN)?,
                down: t.proj(&format!("{ep}.down_proj"), HIDDEN, MOE_INTERMEDIATE)?,
            })
        })
        .collect::<Result<_>>()?;
    crate::moe_decode::Glm52MoeExpertBank::pack_from_host(ctx, &experts)
}

fn upload_u8(ctx: &DeviceContext, host: &[u8]) -> Result<cudarc::driver::CudaSlice<u8>> {
    let mut dev = ctx.stream.alloc_zeros::<u8>(host.len())?;
    ctx.stream.memcpy_htod(host, &mut dev)?;
    Ok(dev)
}

pub(crate) fn load_decoder_layer(
    ctx: &DeviceContext,
    model_path: &Path,
    layer: usize,
    mlp_kind: GateLayerMlp,
) -> Result<Glm52DecoderLayerWeights> {
    let t = LayerTensors::load(model_path, layer)?;
    let p = format!("model.layers.{layer}");

    let mla = Glm52MlaLayerWeights::from_host(
        ctx,
        &t.proj(&format!("{p}.self_attn.q_a_proj"), Q_LORA, HIDDEN)?,
        t.bytes(&format!("{p}.self_attn.q_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.self_attn.q_b_proj"), 16384, Q_LORA)?,
        &t.proj(&format!("{p}.self_attn.kv_a_proj_with_mqa"), 576, HIDDEN)?,
        t.bytes(&format!("{p}.self_attn.kv_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.self_attn.kv_b_proj"), 28672, 512)?,
        &t.proj(&format!("{p}.self_attn.o_proj"), HIDDEN, 16384)?,
    )?;

    let ip = format!("{p}.self_attn.indexer");
    let indexer = Glm52IndexerLayerWeights::from_host(
        ctx,
        &t.proj(&format!("{ip}.wq_b"), 32 * GLM52_INDEX_HEAD_DIM, Q_LORA)?,
        &t.proj(&format!("{ip}.wk"), GLM52_INDEX_HEAD_DIM, HIDDEN)?,
        t.bytes(&format!("{ip}.weights_proj.weight"))?,
        t.bytes(&format!("{ip}.k_norm.weight"))?,
        t.bytes(&format!("{ip}.k_norm.bias"))?,
    )?;

    let mp = format!("{p}.mlp");
    let mlp = match mlp_kind {
        GateLayerMlp::Dense => {
            Glm52LayerMlp::Dense(Box::new(crate::dense::Glm52DenseMlpWeights::from_host(
                ctx,
                &t.proj(&format!("{mp}.gate_proj"), DENSE_INTERMEDIATE, HIDDEN)?,
                &t.proj(&format!("{mp}.up_proj"), DENSE_INTERMEDIATE, HIDDEN)?,
                &t.proj(&format!("{mp}.down_proj"), HIDDEN, DENSE_INTERMEDIATE)?,
            )?))
        }
        GateLayerMlp::MoeEp8Rank0 | GateLayerMlp::MoeEp4Rank0 => {
            let ep_ranks = if mlp_kind == GateLayerMlp::MoeEp8Rank0 {
                8
            } else {
                4
            };
            Glm52LayerMlp::MoeEp8(Box::new(crate::moe_ep8::Glm52MoeEp8LayerWeights {
                router: crate::moe_decode::Glm52MoeRouterWeights::new(
                    ctx,
                    upload_u8(ctx, t.bytes(&format!("{mp}.gate.weight"))?)?,
                    upload_u8(ctx, t.bytes(&format!("{mp}.gate.e_score_correction_bias"))?)?,
                )?,
                shared: crate::moe_decode::Glm52MoeSharedExpert::new(
                    ctx,
                    &crate::fp8::ProjWeight::upload(
                        ctx,
                        &t.proj(
                            &format!("{mp}.shared_experts.gate_proj"),
                            MOE_INTERMEDIATE,
                            HIDDEN,
                        )?,
                    )?,
                    &crate::fp8::ProjWeight::upload(
                        ctx,
                        &t.proj(
                            &format!("{mp}.shared_experts.up_proj"),
                            MOE_INTERMEDIATE,
                            HIDDEN,
                        )?,
                    )?,
                    crate::fp8::ProjWeight::upload(
                        ctx,
                        &t.proj(
                            &format!("{mp}.shared_experts.down_proj"),
                            HIDDEN,
                            MOE_INTERMEDIATE,
                        )?,
                    )?,
                )?,
                bank: load_rank_expert_bank(ctx, &t, layer, 0, ep_ranks)?,
            }))
        }
    };

    Ok(Glm52DecoderLayerWeights {
        input_ln: DeviceVec::from_safetensors(
            ctx,
            t.bytes(&format!("{p}.input_layernorm.weight"))?,
        )?,
        post_attn_ln: DeviceVec::from_safetensors(
            ctx,
            t.bytes(&format!("{p}.post_attention_layernorm.weight"))?,
        )?,
        mla,
        indexer: Glm52LayerIndexer::Full(Box::new(indexer)),
        mlp,
    })
}

/// Drive one full prefill-via-decode pass through the layer; returns the
/// concatenated f32 outputs `[ctx * HIDDEN]`.
fn run_layer_prefill(
    ctx: &DeviceContext,
    w: &Glm52DecoderLayerWeights,
    hidden_host: &[bf16],
    oracle_ctx: usize,
) -> Result<Vec<f32>> {
    // Single-layer slab: one `[MLA | index-K]` page per 64-token block, so
    // the contract/layout strides are the test slab's page stride.
    let num_blocks = oracle_ctx.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE);
    let (mut slab, caches) = crate::model::glm52_test_layer_slab(ctx, num_blocks)?;
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks,
        kv_layer_offset_bytes: 0,
        kv_block_stride_bytes: slab.page_stride,
        topk: GLM52_FLASHMLA_SPARSE_TOPK,
        num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
        sm_scale: GLM52_SM_SCALE,
    };
    let index_blocks = oracle_ctx.div_ceil(INDEX_CACHE_BLOCK);
    let index_cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: index_blocks,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_layer_offset_bytes: 0,
        cache_block_stride_bytes: slab.page_stride,
    };

    let block_table_host: Vec<i32> = (0..index_blocks as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(index_blocks)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;
    let mut slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    let mut cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mut sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let mla_sched = Glm52MlaSchedMetadata::new(ctx, contract, w.mla.heads)?;

    let mqa_shape = Glm52IndexerScratch::paged_mqa_shape(
        1,
        index_cache_layout,
        index_blocks,
        NUM_SMS,
        oracle_ctx,
    );
    let mut scratch =
        Glm52DecodeScratch::new(ctx, &contract, mqa_shape, crate::config::GLM52_HEADS, false)?;

    let mut outputs = Vec::with_capacity(oracle_ctx * HIDDEN);
    for position in 0..oracle_ctx {
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            scratch.hidden.data_mut(),
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;
        ctx.stream
            .memcpy_htod(&[position as i64], &mut slot_mapping)?;
        ctx.stream
            .memcpy_htod(&[(position + 1) as i32], &mut seq_lens)?;

        let step = Glm52DecodeStep {
            mla_cos: &cos,
            mla_sin: &sin,
            idx_cos: &cos,
            idx_sin: &sin,
            mla_sched: &mla_sched,
            slot_mapping: &slot_mapping,
            block_table: &block_table,
            seq_lens: &seq_lens,
        };
        let mut carry_ready = false;
        glm52_decoder_layer_forward(
            ctx,
            w,
            &mut slab,
            caches,
            &step,
            &mut scratch,
            &mut carry_ready,
        )?;
        let out_host = ctx.stream.clone_dtoh(scratch.hidden.data())?;
        outputs.extend(out_host.iter().map(|v| v.to_f32()));
    }
    // Debugging aid: dump the engine outputs (f32 LE) for offline diffing
    // against the harness's safetensors taps.
    if let Some(dump) = std::env::var_os("PEGAINFER_GLM52_LAYER_DUMP") {
        let bytes: Vec<u8> = outputs.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&dump, bytes)?;
    }
    Ok(outputs)
}

/// Probe assertion with a bounded router-tie allowance.
///
/// On MoE layers a handful of positions sit on a near-tie between the 8th and
/// 9th biased router scores (measured on layer 6: the divergent positions'
/// selection margins are 1.0-1.7e-4 vs a 1.8e-3 median — 10x smaller), and the
/// engine's fp8 router logits legitimately flip that pick vs the fp8sim
/// oracle. A flip perturbs only that position by roughly one weighted expert
/// contribution. So: allow up to `allowed_outliers` failing probes, but cap
/// each outlier's deviation at 8x tol — a systematic bug (dropped x2.5,
/// swapped gate/up, wrong expert weights) shifts probes by orders more and
/// still fails. Dense layers have no router and use zero allowance.
pub(crate) fn assert_layer_probes(
    label: &str,
    outputs: &[f32],
    probes: &[(usize, f32)],
    tol: f32,
    allowed_outliers: usize,
) {
    assert!(
        !probes.is_empty(),
        "{label}: probe block is the ungenerated placeholder — run the harness and paste"
    );
    let failures: Vec<_> = probes
        .iter()
        .filter(|&&(idx, expected)| (outputs[idx] - expected).abs() > tol)
        .collect();
    println!(
        "{label}: {}/{} probes within tol={tol:.6e} ({} tie-flip outliers allowed)",
        probes.len() - failures.len(),
        probes.len(),
        allowed_outliers
    );
    for &&(idx, expected) in failures.iter().take(10) {
        println!(
            "  probe[{idx}]: oracle {expected:.6} vs engine {:.6}",
            outputs[idx]
        );
    }
    assert!(
        failures.len() <= allowed_outliers,
        "{label}: {} probes out of tolerance (> {allowed_outliers} allowed)",
        failures.len()
    );
    let cap = 8.0 * tol;
    for &&(idx, expected) in &failures {
        let dev = (outputs[idx] - expected).abs();
        assert!(
            dev <= cap,
            "{label}: probe[{idx}] deviation {dev:.6e} exceeds the tie-flip cap {cap:.6e} — this is not a router tie, investigate"
        );
    }
}

pub(crate) fn model_path() -> PathBuf {
    std::env::var_os("PEGAINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from)
}

pub(crate) fn checked_hidden(
    seed: u64,
    ctxlen: usize,
    scale: f64,
    digest: &str,
) -> Result<Vec<bf16>> {
    let hidden = seeded_hidden(seed, ctxlen * HIDDEN, scale);
    let got = bf16_digest(&hidden);
    ensure!(
        got == digest,
        "input digest {got} != oracle {digest}: PRNG drift or stale probes — regenerate"
    );
    Ok(hidden)
}

#[test]
#[ignore = "requires H200 + GLM-5.2-FP8 checkpoint + DeepGEMM env"]
fn layer_dense_oracle_gate() -> Result<()> {
    let hidden_host = checked_hidden(
        DENSE_ORACLE_SEED,
        DENSE_ORACLE_CTX,
        DENSE_ORACLE_INPUT_SCALE,
        DENSE_ORACLE_HIDDEN_DIGEST,
    )?;
    let ctx = DeviceContext::new()?;
    let w = load_decoder_layer(&ctx, &model_path(), DENSE_ORACLE_LAYER, GateLayerMlp::Dense)?;
    let outputs = run_layer_prefill(&ctx, &w, &hidden_host, DENSE_ORACLE_CTX)?;
    assert_layer_probes(
        "layer0/dense",
        &outputs,
        DENSE_ORACLE_LAYER_PROBES,
        DENSE_ORACLE_LAYER_TOL,
        0,
    );
    Ok(())
}
