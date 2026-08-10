use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use memmap2::Mmap;
use safetensors::Dtype;
use serde_json::Value;

use crate::config::GLM52_DENSE_INTERMEDIATE;
use crate::config::GLM52_DENSE_LAYERS;
use crate::config::GLM52_EXPERT_INTERMEDIATE;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_INDEX_HEADS;
use crate::config::GLM52_KV_A_OUT;
use crate::config::GLM52_KV_B_OUT;
use crate::config::GLM52_KV_LORA_RANK;
use crate::config::GLM52_LAYERS;
use crate::config::GLM52_MTP_LAYER;
use crate::config::GLM52_O_PROJ_IN;
use crate::config::GLM52_Q_B_OUT;
use crate::config::GLM52_Q_LORA_RANK;
use crate::config::GLM52_ROUTED_EXPERTS;
use crate::config::GLM52_VOCAB;

mod context;
mod load;
mod staging;

pub(crate) use context::Glm52RankGpuContext;
pub(crate) use load::Glm52ExpertLayerRegions;
pub(crate) use load::Glm52RankGpuWeights;
pub(crate) use load::load_rank_weights_to_gpu;

const GLM52_WEIGHT_INDEX: &str = "model.safetensors.index.json";
/// The EP8 production partition (8 ranks × 32 experts). Serving-path code
/// derives rank/expert counts from the launch topology
/// (`Glm52MoeTopo::ep_local_experts`); these constants remain the manifest
/// coverage partition (`rank_tensor_names` / `validate_rank_coverage`, which
/// only needs SOME partition covering every checkpoint tensor) and the EP8
/// chain's shim contract check.
pub(crate) const GLM52_EP_RANKS: usize = 8;
const GLM52_LOCAL_EXPERTS: usize = GLM52_ROUTED_EXPERTS / GLM52_EP_RANKS;
const FP8_BLOCK_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// Expert packed placement: routed-expert tensors are written into their FINAL
// expert-major layout at H2D time (per expert: [gate; up] rows, scales
// likewise), byte-identical to `Glm52MoeExpertBank::pack_from_host` packing.
// Repacking after load cannot work — a rank's expert slab (~85 GiB) plus its
// packed copy exceeds the 141 GiB HBM — so placement happens in the loader.
// ---------------------------------------------------------------------------

/// Per-expert byte strides of the packed regions (expert intermediate 2048,
/// hidden 6144, fp8 weights + f32 128×128-block scales).
const EXPERT_PROJ_W13_BYTES: usize = GLM52_EXPERT_INTERMEDIATE * GLM52_HIDDEN; // one of gate|up
const EXPERT_W13_WEIGHT_STRIDE: usize = 2 * EXPERT_PROJ_W13_BYTES;
const EXPERT_PROJ_W13_SCALE_BYTES: usize =
    GLM52_EXPERT_INTERMEDIATE.div_ceil(FP8_BLOCK_SIZE) * GLM52_HIDDEN.div_ceil(FP8_BLOCK_SIZE) * 4;
const EXPERT_W13_SCALE_STRIDE: usize = 2 * EXPERT_PROJ_W13_SCALE_BYTES;
const EXPERT_W2_WEIGHT_STRIDE: usize = GLM52_HIDDEN * GLM52_EXPERT_INTERMEDIATE;
const EXPERT_W2_SCALE_STRIDE: usize =
    GLM52_HIDDEN.div_ceil(FP8_BLOCK_SIZE) * GLM52_EXPERT_INTERMEDIATE.div_ceil(FP8_BLOCK_SIZE) * 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) enum Glm52ExpertRegionKind {
    W13Weight,
    W13Scale,
    W2Weight,
    W2Scale,
}

impl Glm52ExpertRegionKind {
    const ALL: [Self; 4] = [
        Self::W13Weight,
        Self::W13Scale,
        Self::W2Weight,
        Self::W2Scale,
    ];

    /// Total bytes of this region for one layer's rank-local experts
    /// (`local_experts` = 32 for EP8, 64 for EP4).
    fn region_bytes(self, local_experts: usize) -> usize {
        local_experts * self.expert_stride()
    }

    pub(crate) fn expert_stride(self) -> usize {
        match self {
            Self::W13Weight => EXPERT_W13_WEIGHT_STRIDE,
            Self::W13Scale => EXPERT_W13_SCALE_STRIDE,
            Self::W2Weight => EXPERT_W2_WEIGHT_STRIDE,
            Self::W2Scale => EXPERT_W2_SCALE_STRIDE,
        }
    }
}

/// Destination of one routed-expert tensor inside its layer's packed regions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52ExpertPlacement {
    layer: usize,
    region: Glm52ExpertRegionKind,
    offset: usize,
}

/// Classify a tensor name: `Some(placement)` for routed-expert tensors (the
/// expert index must fall in this rank's range), `None` for everything else
/// (own-region tensors). Fails loudly on a malformed expert name or an expert
/// outside the rank's range — either means the load plan is corrupt.
pub(crate) fn expert_placement(
    name: &str,
    rank_experts: &std::ops::Range<usize>,
) -> Result<Option<Glm52ExpertPlacement>> {
    use Glm52ExpertRegionKind::W2Scale;
    use Glm52ExpertRegionKind::W2Weight;
    use Glm52ExpertRegionKind::W13Scale;
    use Glm52ExpertRegionKind::W13Weight;

    let Some((layer, rest)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once(".mlp.experts."))
    else {
        return Ok(None);
    };
    let layer = layer
        .parse::<usize>()
        .with_context(|| format!("GLM5.2 expert tensor has invalid layer index: {name}"))?;
    let (expert, proj) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 expert tensor has malformed name: {name}"))?;
    let expert = expert
        .parse::<usize>()
        .with_context(|| format!("GLM5.2 expert tensor has invalid expert index: {name}"))?;
    ensure!(
        rank_experts.contains(&expert),
        "GLM5.2 expert tensor {name} is outside this rank's expert range {rank_experts:?}"
    );
    let local = expert - rank_experts.start;

    let (region, offset) = match proj {
        "gate_proj.weight" => (W13Weight, local * EXPERT_W13_WEIGHT_STRIDE),
        "up_proj.weight" => (
            W13Weight,
            local * EXPERT_W13_WEIGHT_STRIDE + EXPERT_PROJ_W13_BYTES,
        ),
        "gate_proj.weight_scale_inv" => (W13Scale, local * EXPERT_W13_SCALE_STRIDE),
        "up_proj.weight_scale_inv" => (
            W13Scale,
            local * EXPERT_W13_SCALE_STRIDE + EXPERT_PROJ_W13_SCALE_BYTES,
        ),
        "down_proj.weight" => (W2Weight, local * EXPERT_W2_WEIGHT_STRIDE),
        "down_proj.weight_scale_inv" => (W2Scale, local * EXPERT_W2_SCALE_STRIDE),
        other => anyhow::bail!("GLM5.2 expert tensor {name} has unknown projection {other}"),
    };
    Ok(Some(Glm52ExpertPlacement {
        layer,
        region,
        offset,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TensorLoadSpec {
    name: String,
    shard: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52ShardLoadPlan {
    shard: String,
    tensors: Vec<Glm52TensorLoadSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankWeightPlan {
    pub(crate) rank: usize,
    pub(crate) expert_range: std::ops::Range<usize>,
    pub(crate) tensor_count: usize,
    pub(crate) native_mtp: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankLoadBundle {
    pub(crate) plan: Glm52RankWeightPlan,
    shards: Vec<Glm52ShardLoadPlan>,
}

impl Glm52RankLoadBundle {
    fn planned_total_bytes(&self) -> Result<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.tensors.iter())
            .try_fold(0usize, |total, spec| {
                total
                    .checked_add(expected_tensor_contract(&spec.name)?.byte_len()?)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "GLM5.2 rank {} planned byte count overflow",
                            self.plan.rank
                        )
                    })
            })
    }
}

pub(crate) struct Glm52WeightManifest {
    weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52TensorContract {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
}

impl Glm52TensorContract {
    fn byte_len(&self) -> Result<usize> {
        let elements = self.shape.iter().try_fold(1usize, |total, dim| {
            total.checked_mul(*dim).ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM5.2 tensor shape {:?} element count overflow",
                    self.shape
                )
            })
        })?;
        elements
            .checked_mul(dtype_element_bytes(self.dtype)?)
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 tensor {:?} byte count overflow", self.shape))
    }
}

impl Glm52WeightManifest {
    pub(crate) fn from_model_dir(model_path: &Path) -> Result<Self> {
        let index_path = model_path.join(GLM52_WEIGHT_INDEX);
        let content = std::fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let json: Value = serde_json::from_str(&content)
            .with_context(|| format!("parse {}", index_path.display()))?;
        Self::from_index_json(&json)
    }

    fn from_index_json(json: &Value) -> Result<Self> {
        let weight_map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("GLM5.2 safetensors index missing weight_map"))?;
        let mut out = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("weight_map entry {name} is not a shard string"))?;
            out.insert(name.clone(), shard.to_owned());
        }
        let manifest = Self { weight_map: out };
        manifest.validate_rank_coverage()?;
        Ok(manifest)
    }

    pub(crate) fn all_rank_load_bundles(
        &self,
        moe_topo: crate::Glm52MoeTopo,
        native_mtp: bool,
    ) -> Result<Vec<Glm52RankLoadBundle>> {
        (0..moe_topo.device_count())
            .map(|rank| self.rank_load_bundle(rank, moe_topo, native_mtp))
            .collect()
    }

    /// Shard file holding `name` (TP slice second-pass loads resolve every
    /// expert's tensors, not just this rank's 32-expert bundle).
    pub(crate) fn shard_for(&self, name: &str) -> Result<&str> {
        self.weight_map
            .get(name)
            .map(String::as_str)
            .with_context(|| format!("GLM5.2 safetensors index missing tensor {name}"))
    }

    fn rank_load_bundle(
        &self,
        rank: usize,
        moe_topo: crate::Glm52MoeTopo,
        native_mtp: bool,
    ) -> Result<Glm52RankLoadBundle> {
        let names = self.rank_resident_tensor_names(rank, moe_topo, native_mtp)?;
        let mut by_shard: BTreeMap<String, Vec<Glm52TensorLoadSpec>> = BTreeMap::new();
        for name in names {
            let shard = self
                .weight_map
                .get(&name)
                .with_context(|| format!("GLM5.2 safetensors index missing tensor {name}"))?;
            by_shard
                .entry(shard.clone())
                .or_default()
                .push(Glm52TensorLoadSpec {
                    name,
                    shard: shard.clone(),
                });
        }
        let tensor_count = by_shard.values().map(Vec::len).sum();
        // Tensor-replicated topologies carry no expert tensors in the
        // resident plan; their bundles get an empty range.
        let expert_range = if moe_topo.uses_ep_expert_bundles() {
            let local = moe_topo.ep_local_experts();
            rank * local..(rank + 1) * local
        } else {
            0..0
        };
        Ok(Glm52RankLoadBundle {
            plan: Glm52RankWeightPlan {
                rank,
                expert_range,
                tensor_count,
                native_mtp,
            },
            shards: by_shard
                .into_iter()
                .map(|(shard, tensors)| Glm52ShardLoadPlan { shard, tensors })
                .collect(),
        })
    }

    /// Full checkpoint coverage, including the MTP accuracy-oracle layer.
    /// This is a manifest invariant only: resident load plans below
    /// deliberately omit tensors that the serving model never consumes.
    fn rank_tensor_names(&self, rank: usize) -> Result<Vec<String>> {
        ensure!(
            rank < GLM52_EP_RANKS,
            "GLM5.2 rank must be in 0..{GLM52_EP_RANKS}, got {rank}"
        );
        let mut names = Vec::new();
        self.push_checkpoint_non_expert_names(&mut names);
        let expert_start = rank * GLM52_LOCAL_EXPERTS;
        let expert_range = expert_start..expert_start + GLM52_LOCAL_EXPERTS;
        for layer_idx in GLM52_DENSE_LAYERS..=GLM52_MTP_LAYER {
            push_routed_experts(&mut names, layer_idx, expert_range.clone());
        }
        Ok(names)
    }

    /// Tensors that must become GPU-resident for one serving rank. MTP layer
    /// 78 enters the EP plan only when native MTP is enabled. TP4 gets all
    /// routed + shared experts from `load_tp_slice_layer`, so its first pass
    /// loads routers but not duplicate full shared-expert projections.
    fn rank_resident_tensor_names(
        &self,
        rank: usize,
        moe_topo: crate::Glm52MoeTopo,
        native_mtp: bool,
    ) -> Result<Vec<String>> {
        ensure!(
            rank < moe_topo.device_count(),
            "GLM5.2 rank must be in 0..{}, got {rank}",
            moe_topo.device_count()
        );
        let mut names = Vec::new();
        self.push_resident_non_expert_names(&mut names, moe_topo.uses_ep_expert_bundles());
        if moe_topo.uses_ep_expert_bundles() {
            let local = moe_topo.ep_local_experts();
            let expert_range = rank * local..(rank + 1) * local;
            for layer_idx in GLM52_DENSE_LAYERS..GLM52_LAYERS {
                push_routed_experts(&mut names, layer_idx, expert_range.clone());
            }
            if native_mtp {
                self.push_mtp_non_expert_names(&mut names, true);
                push_routed_experts(&mut names, GLM52_MTP_LAYER, expert_range);
            }
        } else {
            if native_mtp {
                self.push_mtp_non_expert_names(&mut names, false);
            }
        }
        Ok(names)
    }

    fn push_resident_non_expert_names(
        &self,
        names: &mut Vec<String>,
        include_shared_experts: bool,
    ) {
        names.push("model.embed_tokens.weight".to_owned());
        names.push("model.norm.weight".to_owned());
        names.push("lm_head.weight".to_owned());

        for layer_idx in 0..GLM52_LAYERS {
            self.push_attention_names(names, layer_idx);
            if layer_idx < GLM52_DENSE_LAYERS {
                push_dense_mlp(names, layer_idx);
            } else {
                push_moe_non_expert(names, layer_idx, include_shared_experts);
            }
        }
    }

    fn push_checkpoint_non_expert_names(&self, names: &mut Vec<String>) {
        self.push_resident_non_expert_names(names, true);
        self.push_mtp_non_expert_names(names, true);
    }

    fn push_mtp_non_expert_names(&self, names: &mut Vec<String>, include_shared_experts: bool) {
        names.push(format!("model.layers.{GLM52_MTP_LAYER}.enorm.weight"));
        names.push(format!("model.layers.{GLM52_MTP_LAYER}.hnorm.weight"));
        names.push(format!("model.layers.{GLM52_MTP_LAYER}.eh_proj.weight"));
        names.push(format!(
            "model.layers.{GLM52_MTP_LAYER}.shared_head.norm.weight"
        ));
        self.push_attention_names(names, GLM52_MTP_LAYER);
        push_moe_non_expert(names, GLM52_MTP_LAYER, include_shared_experts);
    }

    fn push_attention_names(&self, names: &mut Vec<String>, layer_idx: usize) {
        let prefix = format!("model.layers.{layer_idx}");
        names.push(format!("{prefix}.input_layernorm.weight"));
        push_fp8(names, &format!("{prefix}.self_attn.q_a_proj"));
        names.push(format!("{prefix}.self_attn.q_a_layernorm.weight"));
        push_fp8(names, &format!("{prefix}.self_attn.q_b_proj"));
        push_fp8(names, &format!("{prefix}.self_attn.kv_a_proj_with_mqa"));
        names.push(format!("{prefix}.self_attn.kv_a_layernorm.weight"));
        push_fp8(names, &format!("{prefix}.self_attn.kv_b_proj"));
        push_fp8(names, &format!("{prefix}.self_attn.o_proj"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));

        let indexer = format!("{prefix}.self_attn.indexer");
        if self
            .weight_map
            .contains_key(&format!("{indexer}.k_norm.weight"))
        {
            names.push(format!("{indexer}.k_norm.weight"));
            names.push(format!("{indexer}.k_norm.bias"));
            names.push(format!("{indexer}.weights_proj.weight"));
            push_fp8(names, &format!("{indexer}.wk"));
            push_fp8(names, &format!("{indexer}.wq_b"));
        }
    }

    fn validate_rank_coverage(&self) -> Result<()> {
        let mut generated = BTreeSet::new();
        for rank in 0..GLM52_EP_RANKS {
            for name in self.rank_tensor_names(rank)? {
                generated.insert(name);
            }
        }
        let checkpoint = self.weight_map.keys().cloned().collect::<BTreeSet<_>>();
        let missing = checkpoint
            .difference(&generated)
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        let extra = generated
            .difference(&checkpoint)
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        ensure!(
            missing.is_empty() && extra.is_empty(),
            "GLM5.2 rank load plan does not exactly cover checkpoint tensors: missing_sample={missing:?}, extra_sample={extra:?}, checkpoint={}, generated={}",
            checkpoint.len(),
            generated.len()
        );
        Ok(())
    }
}

fn push_dense_mlp(names: &mut Vec<String>, layer_idx: usize) {
    let prefix = format!("model.layers.{layer_idx}.mlp");
    push_fp8(names, &format!("{prefix}.gate_proj"));
    push_fp8(names, &format!("{prefix}.up_proj"));
    push_fp8(names, &format!("{prefix}.down_proj"));
}

fn push_moe_non_expert(names: &mut Vec<String>, layer_idx: usize, include_shared_experts: bool) {
    let prefix = format!("model.layers.{layer_idx}.mlp");
    names.push(format!("{prefix}.gate.weight"));
    names.push(format!("{prefix}.gate.e_score_correction_bias"));
    if include_shared_experts {
        push_fp8(names, &format!("{prefix}.shared_experts.gate_proj"));
        push_fp8(names, &format!("{prefix}.shared_experts.up_proj"));
        push_fp8(names, &format!("{prefix}.shared_experts.down_proj"));
    }
}

fn push_routed_experts(names: &mut Vec<String>, layer_idx: usize, experts: std::ops::Range<usize>) {
    let prefix = format!("model.layers.{layer_idx}.mlp.experts");
    for expert_idx in experts {
        let expert = format!("{prefix}.{expert_idx}");
        push_fp8(names, &format!("{expert}.gate_proj"));
        push_fp8(names, &format!("{expert}.up_proj"));
        push_fp8(names, &format!("{expert}.down_proj"));
    }
}

fn push_fp8(names: &mut Vec<String>, prefix: &str) {
    names.push(format!("{prefix}.weight"));
    names.push(format!("{prefix}.weight_scale_inv"));
}

pub(crate) fn expected_tensor_contract(name: &str) -> Result<Glm52TensorContract> {
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return Ok(contract(Dtype::BF16, [GLM52_VOCAB, GLM52_HIDDEN]));
    }
    if name == "model.norm.weight" {
        return Ok(contract(Dtype::BF16, [GLM52_HIDDEN]));
    }

    let layer_idx = parse_layer_index(name)?;
    ensure!(
        layer_idx <= GLM52_MTP_LAYER,
        "GLM5.2 tensor contract excludes layer {layer_idx}: {name}"
    );

    if layer_idx == GLM52_MTP_LAYER {
        if name.ends_with(".enorm.weight")
            || name.ends_with(".hnorm.weight")
            || name.ends_with(".shared_head.norm.weight")
        {
            return Ok(contract(Dtype::BF16, [GLM52_HIDDEN]));
        }
        if name.ends_with(".eh_proj.weight") {
            return Ok(contract(Dtype::BF16, [GLM52_HIDDEN, 2 * GLM52_HIDDEN]));
        }
    }

    if name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
    {
        return Ok(contract(Dtype::BF16, [GLM52_HIDDEN]));
    }
    if name.ends_with(".self_attn.q_a_layernorm.weight") {
        return Ok(contract(Dtype::BF16, [GLM52_Q_LORA_RANK]));
    }
    if name.ends_with(".self_attn.kv_a_layernorm.weight") {
        return Ok(contract(Dtype::BF16, [GLM52_KV_LORA_RANK]));
    }

    if let Some(projection) = attention_projection_contract(name) {
        return Ok(projection);
    }
    if let Some(indexer) = indexer_contract(name) {
        return Ok(indexer);
    }
    if let Some(mlp) = mlp_contract(layer_idx, name) {
        return Ok(mlp);
    }

    anyhow::bail!("no GLM5.2 tensor contract for {name}")
}

fn attention_projection_contract(name: &str) -> Option<Glm52TensorContract> {
    projection_contract(name, ".self_attn.q_a_proj", GLM52_Q_LORA_RANK, GLM52_HIDDEN)
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.q_b_proj",
                GLM52_Q_B_OUT,
                GLM52_Q_LORA_RANK,
            )
        })
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.kv_a_proj_with_mqa",
                GLM52_KV_A_OUT,
                GLM52_HIDDEN,
            )
        })
        .or_else(|| {
            projection_contract(
                name,
                ".self_attn.kv_b_proj",
                GLM52_KV_B_OUT,
                GLM52_KV_LORA_RANK,
            )
        })
        .or_else(|| projection_contract(name, ".self_attn.o_proj", GLM52_HIDDEN, GLM52_O_PROJ_IN))
}

fn indexer_contract(name: &str) -> Option<Glm52TensorContract> {
    if name.ends_with(".self_attn.indexer.k_norm.weight")
        || name.ends_with(".self_attn.indexer.k_norm.bias")
    {
        return Some(contract(Dtype::BF16, [GLM52_INDEX_HEAD_DIM]));
    }
    if name.ends_with(".self_attn.indexer.weights_proj.weight") {
        return Some(contract(Dtype::BF16, [GLM52_INDEX_HEADS, GLM52_HIDDEN]));
    }
    projection_contract(
        name,
        ".self_attn.indexer.wk",
        GLM52_INDEX_HEAD_DIM,
        GLM52_HIDDEN,
    )
    .or_else(|| {
        projection_contract(
            name,
            ".self_attn.indexer.wq_b",
            GLM52_INDEX_HEADS * GLM52_INDEX_HEAD_DIM,
            GLM52_Q_LORA_RANK,
        )
    })
}

fn mlp_contract(layer_idx: usize, name: &str) -> Option<Glm52TensorContract> {
    if name.ends_with(".mlp.gate.weight") {
        return Some(contract(Dtype::BF16, [GLM52_ROUTED_EXPERTS, GLM52_HIDDEN]));
    }
    if name.ends_with(".mlp.gate.e_score_correction_bias") {
        return Some(contract(Dtype::F32, [GLM52_ROUTED_EXPERTS]));
    }

    let intermediate = if layer_idx < GLM52_DENSE_LAYERS {
        GLM52_DENSE_INTERMEDIATE
    } else {
        GLM52_EXPERT_INTERMEDIATE
    };
    projection_contract(name, ".gate_proj", intermediate, GLM52_HIDDEN)
        .or_else(|| projection_contract(name, ".up_proj", intermediate, GLM52_HIDDEN))
        .or_else(|| projection_contract(name, ".down_proj", GLM52_HIDDEN, intermediate))
}

fn projection_contract(
    name: &str,
    stem: &str,
    rows: usize,
    cols: usize,
) -> Option<Glm52TensorContract> {
    if name.ends_with(&format!("{stem}.weight")) {
        return Some(contract(Dtype::F8_E4M3, [rows, cols]));
    }
    if name.ends_with(&format!("{stem}.weight_scale_inv")) {
        return Some(contract(
            Dtype::F32,
            [rows.div_ceil(FP8_BLOCK_SIZE), cols.div_ceil(FP8_BLOCK_SIZE)],
        ));
    }
    None
}

fn contract<const N: usize>(dtype: Dtype, shape: [usize; N]) -> Glm52TensorContract {
    Glm52TensorContract {
        dtype,
        shape: shape.to_vec(),
    }
}

fn dtype_element_bytes(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F8_E4M3 => Ok(1),
        Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        other => anyhow::bail!("GLM5.2 loader does not support dtype {:?}", other),
    }
}

fn parse_layer_index(name: &str) -> Result<usize> {
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 tensor is not a layer tensor: {name}"))?;
    let (idx, _) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 tensor has malformed layer prefix: {name}"))?;
    idx.parse::<usize>()
        .map_err(|err| anyhow::anyhow!("GLM5.2 tensor has invalid layer index in {name}: {err}"))
}

/// Reinterpret an owned device byte buffer as a typed slice (no copy). The
/// loader keeps every region as raw `u8`; consumers retype at construction.
pub(crate) fn retype_owned<T>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    bytes: cudarc::driver::CudaSlice<u8>,
) -> Result<cudarc::driver::CudaSlice<T>> {
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<T>()),
        "GLM5.2 retype: {} bytes is not a multiple of element size {}",
        bytes.len(),
        std::mem::size_of::<T>()
    );
    let len = bytes.len() / std::mem::size_of::<T>();
    let ptr = bytes.leak();
    // SAFETY: ptr is a live device allocation of exactly len*size_of::<T>()
    // bytes (leaked just above); cudaMalloc alignment (256B) covers any T we
    // use (f32/bf16/i32).
    Ok(unsafe { stream.upgrade_device_ptr::<T>(ptr, len) })
}

pub(crate) fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path)
        .map_err(|err| anyhow::anyhow!("open {}: {err}", path.display()))?;
    unsafe { Mmap::map(&file) }.map_err(|err| anyhow::anyhow!("mmap {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn rank_expert_ranges_cover_all_routed_experts() {
        let ranges = (0..GLM52_EP_RANKS)
            .map(|rank| rank * GLM52_LOCAL_EXPERTS..(rank + 1) * GLM52_LOCAL_EXPERTS)
            .collect::<Vec<_>>();

        assert_eq!(ranges[0], 0..32);
        assert_eq!(ranges[7], 224..256);
        assert_eq!(ranges.iter().map(std::ops::Range::len).sum::<usize>(), 256);
    }

    #[test]
    fn expert_placement_matches_from_host_packing() {
        // The packed layout must stay byte-identical to
        // `Glm52MoeExpertBank::pack_from_host` (per expert: gate bytes then up
        // bytes; scales likewise; down alone). Walk rank 1's experts in
        // checkpoint order and require contiguous, gap-free regions.
        let rank_experts = 32..64usize;
        let mut cursor: BTreeMap<Glm52ExpertRegionKind, usize> = BTreeMap::new();
        for expert in rank_experts.clone() {
            for suffix in [
                "gate_proj.weight",
                "up_proj.weight",
                "gate_proj.weight_scale_inv",
                "up_proj.weight_scale_inv",
                "down_proj.weight",
                "down_proj.weight_scale_inv",
            ] {
                let name = format!("model.layers.7.mlp.experts.{expert}.{suffix}");
                let placement = expert_placement(&name, &rank_experts).unwrap().unwrap();
                assert_eq!(placement.layer, 7, "{suffix}");
                let next = cursor.entry(placement.region).or_default();
                assert_eq!(placement.offset, *next, "{suffix} expert {expert}");
                *next += expected_tensor_contract(&name).unwrap().byte_len().unwrap();
            }
        }
        for kind in Glm52ExpertRegionKind::ALL {
            assert_eq!(
                cursor[&kind],
                kind.region_bytes(rank_experts.len()),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn official_attention_shapes_are_not_provider_4x_shapes() {
        assert_eq!(
            expected_tensor_contract("model.layers.0.self_attn.q_b_proj.weight").unwrap(),
            contract(Dtype::F8_E4M3, [16_384, 2_048])
        );
        assert_eq!(
            expected_tensor_contract("model.layers.0.self_attn.kv_b_proj.weight").unwrap(),
            contract(Dtype::F8_E4M3, [28_672, 512])
        );
    }

    #[test]
    fn resident_load_plans_exclude_validation_only_and_duplicate_tensors() {
        let manifest = Glm52WeightManifest {
            weight_map: BTreeMap::new(),
        };
        let ep8 = manifest
            .rank_resident_tensor_names(0, crate::Glm52MoeTopo::Ep8, false)
            .unwrap();
        let tp4 = manifest
            .rank_resident_tensor_names(0, crate::Glm52MoeTopo::Tp4, false)
            .unwrap();

        let mtp_prefix = format!("model.layers.{GLM52_MTP_LAYER}.");
        assert!(ep8.iter().all(|name| !name.starts_with(&mtp_prefix)));
        assert!(tp4.iter().all(|name| !name.starts_with(&mtp_prefix)));
        assert!(ep8.iter().any(|name| name.contains(".mlp.shared_experts.")));
        assert!(
            tp4.iter()
                .all(|name| !name.contains(".mlp.shared_experts."))
        );
        assert!(
            ep8.iter()
                .any(|name| name.contains(".mlp.experts.0.gate_proj.weight"))
        );
        assert!(tp4.iter().all(|name| !name.contains(".mlp.experts.")));
    }

    #[test]
    fn native_mtp_ep8_plan_adds_the_local_layer_78_partition() {
        let manifest = Glm52WeightManifest {
            weight_map: BTreeMap::new(),
        };
        let plain = manifest
            .rank_resident_tensor_names(3, crate::Glm52MoeTopo::Ep8, false)
            .unwrap();
        let native_mtp = manifest
            .rank_resident_tensor_names(3, crate::Glm52MoeTopo::Ep8, true)
            .unwrap();
        let mtp_prefix = format!("model.layers.{GLM52_MTP_LAYER}.");

        assert!(plain.iter().all(|name| !name.starts_with(&mtp_prefix)));
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}eh_proj.weight"))
        );
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}mlp.experts.96.gate_proj.weight"))
        );
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}mlp.experts.127.down_proj.weight"))
        );
        assert!(native_mtp.iter().all(|name| {
            if !name.starts_with(&format!("{mtp_prefix}mlp.experts.")) {
                return true;
            }
            name.split(".mlp.experts.")
                .nth(1)
                .and_then(|rest| rest.split('.').next())
                .and_then(|expert| expert.parse::<usize>().ok())
                .is_some_and(|expert| (96..128).contains(&expert))
        }));
    }

    #[test]
    fn native_mtp_ep4_plan_adds_the_local_layer_78_partition() {
        let manifest = Glm52WeightManifest {
            weight_map: BTreeMap::new(),
        };
        let plain = manifest
            .rank_resident_tensor_names(3, crate::Glm52MoeTopo::Ep4, false)
            .unwrap();
        let native_mtp = manifest
            .rank_resident_tensor_names(3, crate::Glm52MoeTopo::Ep4, true)
            .unwrap();
        let mtp_prefix = format!("model.layers.{GLM52_MTP_LAYER}.");

        assert!(plain.iter().all(|name| !name.starts_with(&mtp_prefix)));
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}eh_proj.weight"))
        );
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}mlp.experts.192.gate_proj.weight"))
        );
        assert!(
            native_mtp
                .iter()
                .any(|name| name == &format!("{mtp_prefix}mlp.experts.255.down_proj.weight"))
        );
        assert!(native_mtp.iter().all(|name| {
            if !name.starts_with(&format!("{mtp_prefix}mlp.experts.")) {
                return true;
            }
            name.split(".mlp.experts.")
                .nth(1)
                .and_then(|rest| rest.split('.').next())
                .and_then(|expert| expert.parse::<usize>().ok())
                .is_some_and(|expert| (192..256).contains(&expert))
        }));
    }

    #[test]
    fn native_mtp_tp4_plan_loads_bookends_and_tp_slice_layer() {
        let manifest = Glm52WeightManifest {
            weight_map: BTreeMap::new(),
        };
        let names = manifest
            .rank_resident_tensor_names(0, crate::Glm52MoeTopo::Tp4, true)
            .unwrap();
        let prefix = format!("model.layers.{GLM52_MTP_LAYER}.");
        assert!(
            names
                .iter()
                .any(|name| name == &format!("{prefix}eh_proj.weight"))
        );
        assert!(
            names
                .iter()
                .all(|name| !name.starts_with(&format!("{prefix}mlp.experts.")))
        );
    }

    #[test]
    fn ep4_resident_plan_carries_64_expert_bundles() {
        let manifest = Glm52WeightManifest {
            weight_map: BTreeMap::new(),
        };
        // Rank 1 of EP4 owns whole experts 64..128 on every MoE layer.
        let ep4 = manifest
            .rank_resident_tensor_names(1, crate::Glm52MoeTopo::Ep4, false)
            .unwrap();
        assert!(
            ep4.iter()
                .any(|name| name.contains(".mlp.experts.64.gate_proj.weight"))
        );
        assert!(
            ep4.iter()
                .any(|name| name.contains(".mlp.experts.127.down_proj.weight"))
        );
        assert!(ep4.iter().all(|name| {
            name.split(".mlp.experts.")
                .nth(1)
                .and_then(|rest| rest.split('.').next())
                .and_then(|e| e.parse::<usize>().ok())
                .is_none_or(|e| (64..128).contains(&e))
        }));
        assert!(ep4.iter().any(|name| name.contains(".mlp.shared_experts.")));
        // Four EP4 ranks cover the full routed set.
        assert!(
            manifest
                .rank_resident_tensor_names(4, crate::Glm52MoeTopo::Ep4, false)
                .is_err()
        );
    }
}
