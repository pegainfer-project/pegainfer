//! K3 weight manifest, tensor contract and rank load planning.
//!
//! Structure follows `pegainfer-glm52/src/weights.rs` (manifest / load
//! executor / stager / rank context). K3-specific departures:
//!
//! * The checkpoint is a multimodal wrapper: every text tensor is named
//!   `language_model.<text name>`, and the vision tower / projector families
//!   are present but never loaded (we serve text only). Plans carry checkpoint
//!   names; the contract, the expert placement and the resident map are all
//!   keyed by the STRIPPED text name, so nothing downstream of the loader has
//!   to know the wrapper exists.
//! * Routed experts are MXFP4 (`weight_packed` + e8m0 `weight_scale`, both
//!   `u8`). Payload and scale bytes are uploaded RAW: the packed layout is
//!   byte-isomorphic to the FP4 B operand the grouped GEMM wants, and scale
//!   decode/relayout happens at model build, not at load.
//! * `self_attn.A_log` is stored padded to `head_dim` lanes; only the leading
//!   `num_heads` values are real, so the contract carries a resident element
//!   count and the loader uploads that prefix (same semantics as the
//!   reference `a_log_weight_loader`).

// Bring-up: the loader lands ahead of the model builder that consumes it.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use memmap2::Mmap;
use safetensors::Dtype;
use serde_json::Value;

use crate::config::K3_ATTN_INNER;
use crate::config::K3_DENSE_INTERMEDIATE;
use crate::config::K3_EXPERT_INTERMEDIATE;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HEADS;
use crate::config::K3_HIDDEN;
use crate::config::K3_KDA_CONV_KERNEL;
use crate::config::K3_KDA_GATE_RANK;
use crate::config::K3_KV_A_OUT;
use crate::config::K3_KV_B_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_LAYERS;
use crate::config::K3_MXFP4_GROUP;
use crate::config::K3_MXFP4_PACK;
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

mod context;
mod load;
mod staging;

pub(crate) use context::K3RankGpuContext;
// The load executor's surface: re-exported now, consumed by the model builder.
#[allow(unused_imports)]
pub(crate) use load::K3ExpertLayerRegions;
#[allow(unused_imports)]
pub(crate) use load::K3RankGpuWeights;
#[allow(unused_imports)]
pub(crate) use load::load_rank_weights_to_gpu;

const K3_WEIGHT_INDEX: &str = "model.safetensors.index.json";
const K3_CONFIG: &str = "config.json";

/// Name prefix the multimodal wrapper puts on every text-tower tensor.
pub(crate) const K3_TEXT_PREFIX: &str = "language_model.";
/// Checkpoint tensor families the text-only serving path never loads. Coverage
/// validation requires every non-text checkpoint key to fall in one of these,
/// so a new family shows up as a hard failure rather than a silent skip.
const K3_NON_TEXT_PREFIXES: [&str; 2] = ["vision_tower.", "mm_projector."];

/// EP size used by `validate_rank_coverage` only. It divides both supported
/// routed-expert counts, so the manifest invariant ("some partition covers
/// every text tensor exactly once") is proven with real sharding math while
/// serving-path code derives its own ranks from the launch topology.
const K3_COVERAGE_EP_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// Expert packed placement: routed-expert tensors are written into their FINAL
// expert-major layout at H2D time (per expert: w1/gate payload rows then
// w3/up payload rows — [gate; up] along N; scales likewise; w2 alone).
// Repacking after load cannot work — a rank's expert slab plus a packed copy
// does not fit in HBM — so placement happens in the loader, as in glm52.
// ---------------------------------------------------------------------------

/// Per-expert byte strides of the packed regions. MXFP4 payload rows are
/// `K / 2` bytes wide, scale rows `K / 32` bytes wide; both are copied
/// verbatim from the checkpoint.
const EXPERT_W13_PROJ_WEIGHT_BYTES: usize =
    K3_EXPERT_INTERMEDIATE * (K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_PACK);
const EXPERT_W13_WEIGHT_STRIDE: usize = 2 * EXPERT_W13_PROJ_WEIGHT_BYTES;
const EXPERT_W13_PROJ_SCALE_BYTES: usize =
    K3_EXPERT_INTERMEDIATE * (K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_GROUP);
const EXPERT_W13_SCALE_STRIDE: usize = 2 * EXPERT_W13_PROJ_SCALE_BYTES;
const EXPERT_W2_WEIGHT_STRIDE: usize =
    K3_ROUTED_EXPERT_HIDDEN * (K3_EXPERT_INTERMEDIATE / K3_MXFP4_PACK);
const EXPERT_W2_SCALE_STRIDE: usize =
    K3_ROUTED_EXPERT_HIDDEN * (K3_EXPERT_INTERMEDIATE / K3_MXFP4_GROUP);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) enum K3ExpertRegionKind {
    W13Weight,
    W13Scale,
    W2Weight,
    W2Scale,
}

impl K3ExpertRegionKind {
    pub(crate) const ALL: [Self; 4] = [
        Self::W13Weight,
        Self::W13Scale,
        Self::W2Weight,
        Self::W2Scale,
    ];

    /// Total bytes of this region for one layer's rank-local experts.
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
pub(crate) struct K3ExpertPlacement {
    pub(crate) layer: usize,
    pub(crate) region: K3ExpertRegionKind,
    pub(crate) offset: usize,
}

/// Strip the multimodal wrapper prefix. Every tensor the loader touches is a
/// text tensor; a non-text name reaching here means the plan is corrupt.
pub(crate) fn text_tensor_name(name: &str) -> Result<&str> {
    name.strip_prefix(K3_TEXT_PREFIX)
        .ok_or_else(|| anyhow::anyhow!("K3 tensor {name} is not a text-tower tensor"))
}

/// Classify a TEXT tensor name: `Some(placement)` for routed-expert tensors
/// (the expert index must fall in this rank's range), `None` for everything
/// else (own-region tensors). Fails loudly on a malformed expert name or an
/// expert outside the rank's range — either means the load plan is corrupt.
pub(crate) fn expert_placement(
    name: &str,
    rank_experts: &std::ops::Range<usize>,
) -> Result<Option<K3ExpertPlacement>> {
    use K3ExpertRegionKind::W2Scale;
    use K3ExpertRegionKind::W2Weight;
    use K3ExpertRegionKind::W13Scale;
    use K3ExpertRegionKind::W13Weight;

    let Some((layer, rest)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once(".block_sparse_moe.experts."))
    else {
        return Ok(None);
    };
    let layer = layer
        .parse::<usize>()
        .with_context(|| format!("K3 expert tensor has invalid layer index: {name}"))?;
    let (expert, proj) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("K3 expert tensor has malformed name: {name}"))?;
    let expert = expert
        .parse::<usize>()
        .with_context(|| format!("K3 expert tensor has invalid expert index: {name}"))?;
    ensure!(
        rank_experts.contains(&expert),
        "K3 expert tensor {name} is outside this rank's expert range {rank_experts:?}"
    );
    let local = expert - rank_experts.start;

    // w1 is the gate half and w3 the up half of the fused W13 region.
    let (region, offset) = match proj {
        "w1.weight_packed" => (W13Weight, local * EXPERT_W13_WEIGHT_STRIDE),
        "w3.weight_packed" => (
            W13Weight,
            local * EXPERT_W13_WEIGHT_STRIDE + EXPERT_W13_PROJ_WEIGHT_BYTES,
        ),
        "w1.weight_scale" => (W13Scale, local * EXPERT_W13_SCALE_STRIDE),
        "w3.weight_scale" => (
            W13Scale,
            local * EXPERT_W13_SCALE_STRIDE + EXPERT_W13_PROJ_SCALE_BYTES,
        ),
        "w2.weight_packed" => (W2Weight, local * EXPERT_W2_WEIGHT_STRIDE),
        "w2.weight_scale" => (W2Scale, local * EXPERT_W2_SCALE_STRIDE),
        other => anyhow::bail!("K3 expert tensor {name} has unknown projection {other}"),
    };
    Ok(Some(K3ExpertPlacement {
        layer,
        region,
        offset,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct K3TensorLoadSpec {
    /// Checkpoint name, wrapper prefix included.
    name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct K3ShardLoadPlan {
    shard: String,
    tensors: Vec<K3TensorLoadSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct K3RankWeightPlan {
    pub(crate) rank: usize,
    pub(crate) expert_range: std::ops::Range<usize>,
    pub(crate) routed_experts: K3RoutedExperts,
    pub(crate) tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct K3RankLoadBundle {
    pub(crate) plan: K3RankWeightPlan,
    shards: Vec<K3ShardLoadPlan>,
}

impl K3RankLoadBundle {
    /// Bytes this rank will actually place on the device — the resident length,
    /// which is smaller than the checkpoint length for padded tensors.
    fn planned_total_bytes(&self) -> Result<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.tensors.iter())
            .try_fold(0usize, |total, spec| {
                let contract = expected_tensor_contract(
                    text_tensor_name(&spec.name)?,
                    self.plan.routed_experts,
                )?;
                total
                    .checked_add(contract.resident_byte_len()?)
                    .ok_or_else(|| {
                        anyhow::anyhow!("K3 rank {} planned byte count overflow", self.plan.rank)
                    })
            })
    }
}

pub(crate) struct K3WeightManifest {
    weight_map: BTreeMap<String, String>,
    routed_experts: K3RoutedExperts,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct K3TensorContract {
    pub(crate) dtype: Dtype,
    /// Shape exactly as stored in the checkpoint.
    pub(crate) shape: Vec<usize>,
    /// Leading elements that become GPU-resident. Equal to the shape's element
    /// count except for tensors the checkpoint pads (`A_log`).
    pub(crate) resident_elements: usize,
}

impl K3TensorContract {
    fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |total, dim| {
            total.checked_mul(*dim).ok_or_else(|| {
                anyhow::anyhow!("K3 tensor shape {:?} element count overflow", self.shape)
            })
        })
    }

    pub(crate) fn byte_len(&self) -> Result<usize> {
        self.element_count()?
            .checked_mul(dtype_element_bytes(self.dtype)?)
            .ok_or_else(|| anyhow::anyhow!("K3 tensor {:?} byte count overflow", self.shape))
    }

    pub(crate) fn resident_byte_len(&self) -> Result<usize> {
        self.resident_elements
            .checked_mul(dtype_element_bytes(self.dtype)?)
            .ok_or_else(|| {
                anyhow::anyhow!("K3 tensor {:?} resident byte count overflow", self.shape)
            })
    }

    fn is_padded(&self) -> Result<bool> {
        Ok(self.resident_elements != self.element_count()?)
    }
}

impl K3WeightManifest {
    /// Read `config.json` (for the routed-expert count) and the safetensors
    /// index out of a model directory.
    pub(crate) fn from_model_dir(model_path: &Path) -> Result<Self> {
        let routed_experts =
            K3RoutedExperts::from_config_json(&read_json(&model_path.join(K3_CONFIG))?)?;
        Self::from_index_json(
            &read_json(&model_path.join(K3_WEIGHT_INDEX))?,
            routed_experts,
        )
    }

    fn from_index_json(json: &Value, routed_experts: K3RoutedExperts) -> Result<Self> {
        let weight_map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("K3 safetensors index missing weight_map"))?;
        let mut out = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("weight_map entry {name} is not a shard string"))?;
            out.insert(name.clone(), shard.to_owned());
        }
        let manifest = Self {
            weight_map: out,
            routed_experts,
        };
        manifest.validate_rank_coverage()?;
        Ok(manifest)
    }

    pub(crate) fn routed_experts(&self) -> K3RoutedExperts {
        self.routed_experts
    }

    /// Shard file holding `name` (checkpoint name, wrapper prefix included).
    pub(crate) fn shard_for(&self, name: &str) -> Result<&str> {
        self.weight_map
            .get(name)
            .map(String::as_str)
            .with_context(|| format!("K3 safetensors index missing tensor {name}"))
    }

    pub(crate) fn rank_load_bundle(
        &self,
        rank: usize,
        topo: K3MoeTopo,
    ) -> Result<K3RankLoadBundle> {
        self.rank_load_bundle_for_layers(rank, topo, K3_LAYERS)
    }

    /// A rank load that covers only the model's first `num_layers` layers.
    ///
    /// A truncated build discards the layers it does not take, but only after
    /// they are resident: at low expert parallelism a whole rank's weights do
    /// not fit on one device, so bring-up has to plan the shallow model rather
    /// than trim it afterwards. `num_layers == K3_LAYERS` is the whole rank.
    pub(crate) fn rank_load_bundle_for_layers(
        &self,
        rank: usize,
        topo: K3MoeTopo,
        num_layers: usize,
    ) -> Result<K3RankLoadBundle> {
        ensure!(
            (1..=K3_LAYERS).contains(&num_layers),
            "K3 rank load must cover 1..={K3_LAYERS} layers, got {num_layers}"
        );
        let expert_range = topo.rank_expert_range(rank)?;
        let names = Self::rank_tensor_names_for_layers(rank, topo, num_layers)?;
        let mut by_shard: BTreeMap<String, Vec<K3TensorLoadSpec>> = BTreeMap::new();
        for name in names {
            let shard = self
                .weight_map
                .get(&name)
                .with_context(|| format!("K3 safetensors index missing tensor {name}"))?;
            by_shard
                .entry(shard.clone())
                .or_default()
                .push(K3TensorLoadSpec { name });
        }
        let tensor_count = by_shard.values().map(Vec::len).sum();
        Ok(K3RankLoadBundle {
            plan: K3RankWeightPlan {
                rank,
                expert_range,
                routed_experts: self.routed_experts,
                tensor_count,
            },
            shards: by_shard
                .into_iter()
                .map(|(shard, tensors)| K3ShardLoadPlan { shard, tensors })
                .collect(),
        })
    }

    /// Checkpoint names one rank must make GPU-resident. Text-only: the
    /// backbone is replicated on every rank, routed experts are EP-sharded.
    /// Names are GENERATED from the architecture constants, never enumerated
    /// from the index — `validate_rank_coverage` proves the generator and the
    /// checkpoint agree.
    fn rank_tensor_names(rank: usize, topo: K3MoeTopo) -> Result<Vec<String>> {
        Self::rank_tensor_names_for_layers(rank, topo, K3_LAYERS)
    }

    /// The same list truncated to the first `num_layers`, for a bring-up build
    /// that only makes part of the stack resident.
    fn rank_tensor_names_for_layers(
        rank: usize,
        topo: K3MoeTopo,
        num_layers: usize,
    ) -> Result<Vec<String>> {
        let expert_range = topo.rank_expert_range(rank)?;
        let mut names = Vec::new();
        push_bookends(&mut names);
        for layer in 0..num_layers {
            push_layer_backbone(&mut names, layer);
            if k3_layer_is_moe(layer) {
                push_moe_backbone(&mut names, layer);
                push_routed_experts(&mut names, layer, expert_range.clone());
            } else {
                push_dense_mlp(&mut names, layer);
            }
        }
        Ok(names)
    }

    fn validate_rank_coverage(&self) -> Result<()> {
        let topo = K3MoeTopo::new(self.routed_experts, K3_COVERAGE_EP_SIZE)?;
        let mut generated = BTreeSet::new();
        for rank in 0..topo.device_count() {
            for name in Self::rank_tensor_names(rank, topo)? {
                generated.insert(name);
            }
        }

        let mut text = BTreeSet::new();
        let mut unknown = Vec::new();
        for name in self.weight_map.keys() {
            if name.starts_with(K3_TEXT_PREFIX) {
                text.insert(name.clone());
            } else if !K3_NON_TEXT_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
            {
                unknown.push(name.clone());
            }
        }
        ensure!(
            unknown.is_empty(),
            "K3 checkpoint has {} tensors in no known family (text {K3_TEXT_PREFIX:?}, non-text {K3_NON_TEXT_PREFIXES:?}): sample={:?}",
            unknown.len(),
            unknown.iter().take(5).collect::<Vec<_>>()
        );

        let missing = text
            .difference(&generated)
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        let extra = generated
            .difference(&text)
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        ensure!(
            missing.is_empty() && extra.is_empty(),
            "K3 rank load plan does not exactly cover the checkpoint's text tensors: missing_sample={missing:?}, extra_sample={extra:?}, checkpoint_text={}, generated={}",
            text.len(),
            generated.len()
        );
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parse {}", path.display()))
}

fn push_bookends(names: &mut Vec<String>) {
    names.push(text_name("model.embed_tokens.weight"));
    names.push(text_name("model.norm.weight"));
    names.push(text_name("lm_head.weight"));
    // Output side of the attention-residual stream.
    names.push(text_name("model.output_attn_res_norm.weight"));
    names.push(text_name("model.output_attn_res_proj.weight"));
}

/// Per-layer tensors present on every layer regardless of attention kind, plus
/// the kind-specific attention projections.
fn push_layer_backbone(names: &mut Vec<String>, layer: usize) {
    let prefix = layer_prefix(layer);
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        // Attention-residual and MLP-residual stream taps.
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        // Output gate + output projection are shared by both attention kinds.
        "self_attn.g_proj.weight",
        "self_attn.o_proj.weight",
    ] {
        names.push(format!("{prefix}.{suffix}"));
    }
    let kind_suffixes: &[&str] = match k3_layer_kind(layer) {
        K3LayerKind::Kda => &[
            "self_attn.A_log",
            "self_attn.dt_bias",
            "self_attn.b_proj.weight",
            "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight",
            "self_attn.o_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
        ],
        K3LayerKind::Mla => &[
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
        ],
    };
    for suffix in kind_suffixes {
        names.push(format!("{prefix}.{suffix}"));
    }
}

fn push_dense_mlp(names: &mut Vec<String>, layer: usize) {
    let prefix = layer_prefix(layer);
    for suffix in [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        names.push(format!("{prefix}.{suffix}"));
    }
}

/// Router, latent down/up projections and the two fused shared experts. All
/// bf16 and replicated on every rank — only the routed experts are sharded.
fn push_moe_backbone(names: &mut Vec<String>, layer: usize) {
    let prefix = format!("{}.block_sparse_moe", layer_prefix(layer));
    for suffix in [
        "gate.weight",
        "gate.e_score_correction_bias",
        "routed_expert_down_proj.weight",
        "routed_expert_norm.weight",
        "routed_expert_up_proj.weight",
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    ] {
        names.push(format!("{prefix}.{suffix}"));
    }
}

fn push_routed_experts(names: &mut Vec<String>, layer: usize, experts: std::ops::Range<usize>) {
    let prefix = format!("{}.block_sparse_moe.experts", layer_prefix(layer));
    for expert_idx in experts {
        for proj in ["w1", "w2", "w3"] {
            names.push(format!("{prefix}.{expert_idx}.{proj}.weight_packed"));
            names.push(format!("{prefix}.{expert_idx}.{proj}.weight_scale"));
        }
    }
}

fn layer_prefix(layer: usize) -> String {
    format!("{K3_TEXT_PREFIX}model.layers.{layer}")
}

fn text_name(name: &str) -> String {
    format!("{K3_TEXT_PREFIX}{name}")
}

/// Pure name → (dtype, checkpoint shape, resident elements) oracle over TEXT
/// names (wrapper prefix already stripped). Used both to precompute planned
/// bytes and to validate every tensor at load.
pub(crate) fn expected_tensor_contract(
    name: &str,
    routed_experts: K3RoutedExperts,
) -> Result<K3TensorContract> {
    match name {
        "model.embed_tokens.weight" | "lm_head.weight" => {
            return Ok(contract(Dtype::BF16, [K3_VOCAB, K3_HIDDEN]));
        }
        "model.norm.weight" | "model.output_attn_res_norm.weight" => {
            return Ok(contract(Dtype::BF16, [K3_HIDDEN]));
        }
        "model.output_attn_res_proj.weight" => {
            return Ok(contract(Dtype::BF16, [1, K3_HIDDEN]));
        }
        _ => {}
    }

    let (layer, rest) = split_layer(name)?;
    ensure!(
        layer < K3_LAYERS,
        "K3 tensor contract excludes layer {layer}: {name}"
    );
    let kind = k3_layer_kind(layer);

    if let Some(common) = layer_common_contract(rest) {
        return Ok(common);
    }
    if let Some(attention) = attention_contract(kind, rest) {
        return Ok(attention);
    }
    if let Some(mlp) = mlp_contract(layer, rest, routed_experts) {
        return Ok(mlp);
    }
    anyhow::bail!("no K3 tensor contract for {name}")
}

fn layer_common_contract(rest: &str) -> Option<K3TensorContract> {
    match rest {
        "input_layernorm.weight"
        | "post_attention_layernorm.weight"
        | "self_attention_res_norm.weight"
        | "mlp_res_norm.weight" => Some(contract(Dtype::BF16, [K3_HIDDEN])),
        // Residual-stream taps project hidden down to a single row.
        "self_attention_res_proj.weight" | "mlp_res_proj.weight" => {
            Some(contract(Dtype::BF16, [1, K3_HIDDEN]))
        }
        "self_attn.g_proj.weight" => Some(contract(Dtype::BF16, [K3_ATTN_INNER, K3_HIDDEN])),
        "self_attn.o_proj.weight" => Some(contract(Dtype::BF16, [K3_HIDDEN, K3_O_PROJ_IN])),
        _ => None,
    }
}

fn attention_contract(kind: K3LayerKind, rest: &str) -> Option<K3TensorContract> {
    match (kind, rest) {
        // ---- KDA ----------------------------------------------------------
        // Stored padded to head_dim lanes; the leading `K3_HEADS` entries are
        // the real per-head decay values and the only ones uploaded.
        (K3LayerKind::Kda, "self_attn.A_log") => {
            Some(padded_contract(Dtype::F32, [K3_HEAD_DIM], K3_HEADS))
        }
        // f32 in the checkpoint and consumed as f32 downstream — do NOT
        // narrow these to bf16 at load.
        (K3LayerKind::Kda, "self_attn.dt_bias") => Some(contract(Dtype::F32, [K3_ATTN_INNER])),
        (K3LayerKind::Kda, "self_attn.o_norm.weight") => Some(contract(Dtype::F32, [K3_HEAD_DIM])),
        (
            K3LayerKind::Kda,
            "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight",
        ) => Some(contract(Dtype::F32, [K3_ATTN_INNER, 1, K3_KDA_CONV_KERNEL])),
        (
            K3LayerKind::Kda,
            "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight",
        ) => Some(contract(Dtype::BF16, [K3_ATTN_INNER, K3_HIDDEN])),
        // Per-head beta.
        (K3LayerKind::Kda, "self_attn.b_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_HEADS, K3_HIDDEN]))
        }
        // Low-rank forget gate.
        (K3LayerKind::Kda, "self_attn.f_a_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_KDA_GATE_RANK, K3_HIDDEN]))
        }
        (K3LayerKind::Kda, "self_attn.f_b_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_ATTN_INNER, K3_KDA_GATE_RANK]))
        }
        // ---- MLA ----------------------------------------------------------
        (K3LayerKind::Mla, "self_attn.q_a_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_Q_LORA_RANK, K3_HIDDEN]))
        }
        (K3LayerKind::Mla, "self_attn.q_a_layernorm.weight") => {
            Some(contract(Dtype::BF16, [K3_Q_LORA_RANK]))
        }
        (K3LayerKind::Mla, "self_attn.q_b_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_Q_B_OUT, K3_Q_LORA_RANK]))
        }
        (K3LayerKind::Mla, "self_attn.kv_a_proj_with_mqa.weight") => {
            Some(contract(Dtype::BF16, [K3_KV_A_OUT, K3_HIDDEN]))
        }
        (K3LayerKind::Mla, "self_attn.kv_a_layernorm.weight") => {
            Some(contract(Dtype::BF16, [K3_KV_LORA_RANK]))
        }
        (K3LayerKind::Mla, "self_attn.kv_b_proj.weight") => {
            Some(contract(Dtype::BF16, [K3_KV_B_OUT, K3_KV_LORA_RANK]))
        }
        _ => None,
    }
}

fn mlp_contract(
    layer: usize,
    rest: &str,
    routed_experts: K3RoutedExperts,
) -> Option<K3TensorContract> {
    if !k3_layer_is_moe(layer) {
        return match rest {
            "mlp.gate_proj.weight" | "mlp.up_proj.weight" => {
                Some(contract(Dtype::BF16, [K3_DENSE_INTERMEDIATE, K3_HIDDEN]))
            }
            "mlp.down_proj.weight" => {
                Some(contract(Dtype::BF16, [K3_HIDDEN, K3_DENSE_INTERMEDIATE]))
            }
            _ => None,
        };
    }

    let rest = rest.strip_prefix("block_sparse_moe.")?;
    let experts = routed_experts.count();
    match rest {
        "gate.weight" => return Some(contract(Dtype::BF16, [experts, K3_HIDDEN])),
        "gate.e_score_correction_bias" => return Some(contract(Dtype::F32, [experts])),
        // Latent projections around the routed-expert stack.
        "routed_expert_down_proj.weight" => {
            return Some(contract(Dtype::BF16, [K3_ROUTED_EXPERT_HIDDEN, K3_HIDDEN]));
        }
        "routed_expert_up_proj.weight" => {
            return Some(contract(Dtype::BF16, [K3_HIDDEN, K3_ROUTED_EXPERT_HIDDEN]));
        }
        "routed_expert_norm.weight" => {
            return Some(contract(Dtype::BF16, [K3_ROUTED_EXPERT_HIDDEN]));
        }
        // The two shared experts are stored fused into one bf16 MLP on hidden.
        "shared_experts.gate_proj.weight" | "shared_experts.up_proj.weight" => {
            return Some(contract(Dtype::BF16, [K3_SHARED_INTERMEDIATE, K3_HIDDEN]));
        }
        "shared_experts.down_proj.weight" => {
            return Some(contract(Dtype::BF16, [K3_HIDDEN, K3_SHARED_INTERMEDIATE]));
        }
        _ => {}
    }

    let (_, proj) = rest.strip_prefix("experts.")?.split_once('.')?;
    routed_expert_contract(proj)
}

/// MXFP4 routed-expert contracts. `w1`/`w3` map latent -> intermediate and
/// `w2` maps intermediate -> latent; payload rows are `K / 2` bytes and e8m0
/// scale rows `K / 32` bytes, both stored as u8.
fn routed_expert_contract(proj: &str) -> Option<K3TensorContract> {
    match proj {
        "w1.weight_packed" | "w3.weight_packed" => Some(contract(
            Dtype::U8,
            [
                K3_EXPERT_INTERMEDIATE,
                K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_PACK,
            ],
        )),
        "w1.weight_scale" | "w3.weight_scale" => Some(contract(
            Dtype::U8,
            [
                K3_EXPERT_INTERMEDIATE,
                K3_ROUTED_EXPERT_HIDDEN / K3_MXFP4_GROUP,
            ],
        )),
        "w2.weight_packed" => Some(contract(
            Dtype::U8,
            [
                K3_ROUTED_EXPERT_HIDDEN,
                K3_EXPERT_INTERMEDIATE / K3_MXFP4_PACK,
            ],
        )),
        "w2.weight_scale" => Some(contract(
            Dtype::U8,
            [
                K3_ROUTED_EXPERT_HIDDEN,
                K3_EXPERT_INTERMEDIATE / K3_MXFP4_GROUP,
            ],
        )),
        _ => None,
    }
}

fn contract<const N: usize>(dtype: Dtype, shape: [usize; N]) -> K3TensorContract {
    let resident_elements = shape.iter().product();
    K3TensorContract {
        dtype,
        shape: shape.to_vec(),
        resident_elements,
    }
}

fn padded_contract<const N: usize>(
    dtype: Dtype,
    shape: [usize; N],
    resident_elements: usize,
) -> K3TensorContract {
    K3TensorContract {
        dtype,
        shape: shape.to_vec(),
        resident_elements,
    }
}

fn dtype_element_bytes(dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::U8 => Ok(1),
        Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        other => anyhow::bail!("K3 loader does not support dtype {other:?}"),
    }
}

fn split_layer(name: &str) -> Result<(usize, &str)> {
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| anyhow::anyhow!("K3 tensor is not a layer tensor: {name}"))?;
    let (idx, rest) = rest
        .split_once('.')
        .ok_or_else(|| anyhow::anyhow!("K3 tensor has malformed layer prefix: {name}"))?;
    let layer = idx
        .parse::<usize>()
        .map_err(|err| anyhow::anyhow!("K3 tensor has invalid layer index in {name}: {err}"))?;
    Ok((layer, rest))
}

/// Reinterpret an owned device byte buffer as a typed slice (no copy). The
/// loader keeps every region as raw `u8`; consumers retype at construction.
pub(crate) fn retype_owned<T>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    bytes: cudarc::driver::CudaSlice<u8>,
) -> Result<cudarc::driver::CudaSlice<T>> {
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<T>()),
        "K3 retype: {} bytes is not a multiple of element size {}",
        bytes.len(),
        std::mem::size_of::<T>()
    );
    let len = bytes.len() / std::mem::size_of::<T>();
    let ptr = bytes.leak();
    // SAFETY: ptr is a live device allocation of exactly len*size_of::<T>()
    // bytes (leaked just above); cudaMalloc alignment (256B) covers any T we
    // use (f32/bf16/u8/i32).
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

    /// Checkpoints the manifest tests read when present (skipped otherwise),
    /// with the routed-expert count each one must report. Override the default
    /// repo-relative locations with `PEGAINFER_K3_TEST_224` /
    /// `PEGAINFER_K3_TEST_896`.
    fn local_checkpoints() -> [(String, usize); 2] {
        let path =
            |var: &str, default: &str| std::env::var(var).unwrap_or_else(|_| default.to_string());
        [
            (path("PEGAINFER_K3_TEST_224", "models/Kimi-K3-224"), 224),
            (path("PEGAINFER_K3_TEST_896", "models/Kimi-K3"), 896),
        ]
    }

    /// The 224-expert checkpoint: small enough to plan against in a unit test.
    fn dev_model_dir() -> String {
        local_checkpoints()[0].0.clone()
    }

    fn experts224() -> K3RoutedExperts {
        K3RoutedExperts::new(224).unwrap()
    }

    #[test]
    fn expert_placement_matches_from_host_packing() {
        // The packed layout must stay contiguous and gap-free per layer: for
        // each local expert, w1 (gate) payload rows then w3 (up) payload rows;
        // scales likewise; w2 alone. Walk rank 1's experts in packing order and
        // require every region to fill exactly.
        let rank_experts = 56..112usize;
        let mut cursor: BTreeMap<K3ExpertRegionKind, usize> = BTreeMap::new();
        for expert in rank_experts.clone() {
            for proj in [
                "w1.weight_packed",
                "w3.weight_packed",
                "w1.weight_scale",
                "w3.weight_scale",
                "w2.weight_packed",
                "w2.weight_scale",
            ] {
                let name = format!("model.layers.7.block_sparse_moe.experts.{expert}.{proj}");
                let placement = expert_placement(&name, &rank_experts).unwrap().unwrap();
                assert_eq!(placement.layer, 7, "{proj}");
                let next = cursor.entry(placement.region).or_default();
                assert_eq!(placement.offset, *next, "{proj} expert {expert}");
                *next += expected_tensor_contract(&name, experts224())
                    .unwrap()
                    .byte_len()
                    .unwrap();
            }
        }
        for kind in K3ExpertRegionKind::ALL {
            assert_eq!(
                cursor[&kind],
                kind.region_bytes(rank_experts.len()),
                "{kind:?}"
            );
        }
        // Non-expert tensors and out-of-range experts.
        assert!(
            expert_placement("model.layers.7.self_attn.o_proj.weight", &rank_experts)
                .unwrap()
                .is_none()
        );
        assert!(
            expert_placement(
                "model.layers.7.block_sparse_moe.experts.5.w1.weight_packed",
                &rank_experts
            )
            .is_err()
        );
    }

    #[test]
    fn tensor_contract_spot_checks() {
        let checks: [(&str, Dtype, &[usize]); 14] = [
            ("model.embed_tokens.weight", Dtype::BF16, &[163_840, 7168]),
            ("lm_head.weight", Dtype::BF16, &[163_840, 7168]),
            ("model.output_attn_res_proj.weight", Dtype::BF16, &[1, 7168]),
            // Layer 0: KDA attention + the only dense MLP.
            (
                "model.layers.0.self_attn.q_proj.weight",
                Dtype::BF16,
                &[12_288, 7168],
            ),
            (
                "model.layers.0.self_attn.k_conv1d.weight",
                Dtype::F32,
                &[12_288, 1, 4],
            ),
            ("model.layers.0.self_attn.o_norm.weight", Dtype::F32, &[128]),
            (
                "model.layers.0.self_attn.f_b_proj.weight",
                Dtype::BF16,
                &[12_288, 128],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                Dtype::BF16,
                &[7168, 33_792],
            ),
            // Layer 3: MLA.
            (
                "model.layers.3.self_attn.q_b_proj.weight",
                Dtype::BF16,
                &[18_432, 1536],
            ),
            (
                "model.layers.3.self_attn.kv_b_proj.weight",
                Dtype::BF16,
                &[24_576, 512],
            ),
            (
                "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
                Dtype::BF16,
                &[576, 7168],
            ),
            // MoE backbone + MXFP4 experts.
            (
                "model.layers.3.block_sparse_moe.shared_experts.down_proj.weight",
                Dtype::BF16,
                &[7168, 6144],
            ),
            (
                "model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
                Dtype::U8,
                &[3072, 1792],
            ),
            (
                "model.layers.3.block_sparse_moe.experts.0.w2.weight_scale",
                Dtype::U8,
                &[3584, 96],
            ),
        ];
        for (name, dtype, shape) in checks {
            let got = expected_tensor_contract(name, experts224()).unwrap();
            assert_eq!(got.dtype, dtype, "{name}");
            assert_eq!(got.shape.as_slice(), shape, "{name}");
            assert!(!got.is_padded().unwrap(), "{name}");
        }

        // The router width is the one contract that tracks the expert count.
        for count in [224usize, 896] {
            let experts = K3RoutedExperts::new(count).unwrap();
            assert_eq!(
                expected_tensor_contract("model.layers.1.block_sparse_moe.gate.weight", experts)
                    .unwrap()
                    .shape,
                vec![count, 7168]
            );
            assert_eq!(
                expected_tensor_contract(
                    "model.layers.1.block_sparse_moe.gate.e_score_correction_bias",
                    experts
                )
                .unwrap()
                .shape,
                vec![count]
            );
        }

        // A_log is stored padded to head_dim; only the leading heads land.
        let a_log =
            expected_tensor_contract("model.layers.0.self_attn.A_log", experts224()).unwrap();
        assert_eq!(a_log.shape, vec![128]);
        assert_eq!(a_log.resident_elements, 96);
        assert_eq!(a_log.byte_len().unwrap(), 512);
        assert_eq!(a_log.resident_byte_len().unwrap(), 384);
        assert!(a_log.is_padded().unwrap());

        // Attention tensors are kind-specific: MLA names must not resolve on a
        // KDA layer and vice versa.
        assert!(
            expected_tensor_contract("model.layers.0.self_attn.q_b_proj.weight", experts224())
                .is_err()
        );
        assert!(
            expected_tensor_contract("model.layers.3.self_attn.q_proj.weight", experts224())
                .is_err()
        );
        assert!(
            expected_tensor_contract("model.layers.93.self_attn.o_proj.weight", experts224())
                .is_err()
        );
        // Layer 0 has no MoE tensors; layers >= 1 have no dense MLP.
        assert!(
            expected_tensor_contract("model.layers.0.block_sparse_moe.gate.weight", experts224())
                .is_err()
        );
        assert!(
            expected_tensor_contract("model.layers.1.mlp.down_proj.weight", experts224()).is_err()
        );
    }

    #[test]
    fn rank_plans_shard_experts_and_replicate_the_backbone() {
        let topo = K3MoeTopo::new(experts224(), 4).unwrap();
        let rank1 = K3WeightManifest::rank_tensor_names(1, topo).unwrap();

        assert!(rank1.iter().any(|name| name
            == "language_model.model.layers.1.block_sparse_moe.experts.56.w1.weight_packed"));
        assert!(rank1.iter().any(|name| name
            == "language_model.model.layers.92.block_sparse_moe.experts.111.w2.weight_scale"));
        assert!(rank1.iter().all(|name| {
            name.split(".block_sparse_moe.experts.")
                .nth(1)
                .and_then(|rest| rest.split('.').next())
                .and_then(|expert| expert.parse::<usize>().ok())
                .is_none_or(|expert| (56..112).contains(&expert))
        }));
        // The MoE backbone (router, latent projections, shared experts) is
        // replicated, not sharded.
        assert!(rank1.iter().any(|name| name
            == "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"));
        assert!(
            rank1
                .iter()
                .any(|name| name == "language_model.model.layers.1.block_sparse_moe.gate.weight")
        );
        assert!(
            rank1
                .iter()
                .any(|name| name == "language_model.lm_head.weight")
        );
        assert!(K3WeightManifest::rank_tensor_names(4, topo).is_err());

        // Single-GPU bring-up keeps every expert on rank 0.
        let ep1 = K3MoeTopo::new(experts224(), 1).unwrap();
        let all = K3WeightManifest::rank_tensor_names(0, ep1).unwrap();
        assert!(all.iter().any(|name| name
            == "language_model.model.layers.1.block_sparse_moe.experts.223.w3.weight_scale"));
    }

    /// Coverage against the real checkpoint index: the union of the EP ranks'
    /// generated names must equal the checkpoint's text-tensor key set exactly.
    /// Skipped when the checkpoint is not mounted.
    #[test]
    fn manifest_covers_the_checkpoint_index() {
        let dev_model_dir = dev_model_dir();
        let dir = Path::new(&dev_model_dir);
        if !dir.join(K3_WEIGHT_INDEX).exists() {
            eprintln!("skipping: {dev_model_dir} is not mounted");
            return;
        }
        let manifest = K3WeightManifest::from_model_dir(dir).unwrap();
        assert_eq!(manifest.routed_experts().count(), 224);

        // Every planned tensor resolves to a shard, and the contract matches
        // for every name the plan generates.
        let topo = K3MoeTopo::new(manifest.routed_experts(), 4).unwrap();
        let bundles: Vec<_> = (0..topo.device_count())
            .map(|rank| manifest.rank_load_bundle(rank, topo).unwrap())
            .collect();
        assert_eq!(bundles.len(), 4);
        let mut total = 0usize;
        for bundle in &bundles {
            assert!(bundle.plan.tensor_count > 0);
            assert!(bundle.planned_total_bytes().unwrap() > 0);
            total += bundle.plan.tensor_count;
            for shard in &bundle.shards {
                for spec in &shard.tensors {
                    assert_eq!(manifest.shard_for(&spec.name).unwrap(), shard.shard);
                }
            }
        }
        // 4 ranks x (2460 backbone tensors + 92 MoE layers x 56 experts x 6).
        assert_eq!(total, 4 * (2460 + 92 * 56 * 6));
    }

    /// Both published checkpoints' configs must satisfy `probe_config_json`,
    /// differing only in the routed-expert count. Skipped when unmounted.
    #[test]
    fn checkpoint_configs_match_the_architecture_constants() {
        let checkpoints = local_checkpoints();
        let mut probed = 0usize;
        for (dir, experts) in &checkpoints {
            let config_path = Path::new(dir).join(K3_CONFIG);
            if !config_path.exists() {
                eprintln!("skipping: {dir} is not mounted");
                continue;
            }
            let json = read_json(&config_path).unwrap();
            crate::config::probe_config_json(&json)
                .unwrap_or_else(|error| panic!("{dir} must probe as K3: {error:#}"));
            assert_eq!(
                K3RoutedExperts::from_config_json(&json).unwrap().count(),
                *experts,
                "{dir}"
            );
            probed += 1;
        }
        eprintln!("probed {probed} of {} checkpoints", checkpoints.len());
    }
}
