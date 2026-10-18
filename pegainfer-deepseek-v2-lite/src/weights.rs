use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::Config;
use crate::ep::ExpertParallelLayout;

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct ModelManifest {
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorRequirement {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct RankLoadPlan {
    tensors: Vec<TensorRequirement>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl ModelManifest {
    pub(crate) fn from_model_dir(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let shard_paths = shard_paths(model_path)?;
        let index = read_index(model_path)?;
        let mut tensors = BTreeMap::new();

        for shard_path in shard_paths {
            let mmap = mmap_file(&shard_path)?;
            let safetensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("deserialize {}", shard_path.display()))?;
            for name in safetensors.names() {
                let view = safetensors
                    .tensor(name)
                    .with_context(|| format!("read tensor metadata {name}"))?;
                let indexed_shard = index.as_ref().and_then(|weight_map| weight_map.get(name));
                if let Some(indexed_shard) = indexed_shard {
                    let actual = shard_path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or_default();
                    ensure!(
                        indexed_shard == actual,
                        "tensor {name} expected in shard {indexed_shard}, found in {actual}"
                    );
                }
                tensors.insert(
                    name.to_string(),
                    TensorInfo {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                    },
                );
            }
        }

        Ok(Self { tensors })
    }

    fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn validate_rank_plan(&self, plan: &RankLoadPlan) -> Result<()> {
        for req in &plan.tensors {
            let info = self
                .get(&req.name)
                .ok_or_else(|| anyhow::anyhow!("missing tensor {}", req.name))?;
            ensure!(
                info.dtype == req.dtype,
                "tensor {} dtype mismatch: expected {:?}, got {:?}",
                req.name,
                req.dtype,
                info.dtype
            );
            ensure!(
                info.shape == req.shape,
                "tensor {} shape mismatch: expected {:?}, got {:?}",
                req.name,
                req.shape,
                info.shape
            );
        }
        Ok(())
    }
}

impl RankLoadPlan {
    pub(crate) fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn for_driver_rank(config: &Config, layout: &ExpertParallelLayout) -> Self {
        let mut tensors = Vec::new();
        push_top_level(config, &mut tensors);
        for layer in 0..config.num_hidden_layers {
            push_attention_layer(config, layer, &mut tensors);
            if config.is_moe_layer(layer) {
                push_moe_layer(config, layout, layer, &mut tensors);
            } else {
                push_dense_layer(config, layer, &mut tensors);
            }
        }

        Self { tensors }
    }

    pub(crate) fn for_expert_rank(config: &Config, layout: &ExpertParallelLayout) -> Self {
        let mut tensors = Vec::new();
        for layer in 0..config.num_hidden_layers {
            if config.is_moe_layer(layer) {
                push_moe_gate(config, layer, &mut tensors);
                push_owned_experts(config, layout, layer, &mut tensors);
            }
        }

        Self { tensors }
    }
}

fn push_req(
    tensors: &mut Vec<TensorRequirement>,
    name: impl Into<String>,
    dtype: Dtype,
    shape: impl Into<Vec<usize>>,
) {
    tensors.push(TensorRequirement {
        name: name.into(),
        dtype,
        shape: shape.into(),
    });
}

fn push_top_level(config: &Config, tensors: &mut Vec<TensorRequirement>) {
    push_req(
        tensors,
        "model.embed_tokens.weight",
        Dtype::BF16,
        [config.vocab_size, config.hidden_size],
    );
    if !config.tie_word_embeddings {
        push_req(
            tensors,
            "lm_head.weight",
            Dtype::BF16,
            [config.vocab_size, config.hidden_size],
        );
    }
    push_req(
        tensors,
        "model.norm.weight",
        Dtype::BF16,
        [config.hidden_size],
    );
}

fn push_attention_layer(config: &Config, layer: usize, tensors: &mut Vec<TensorRequirement>) {
    let prefix = format!("model.layers.{layer}");
    push_req(
        tensors,
        format!("{prefix}.input_layernorm.weight"),
        Dtype::BF16,
        [config.hidden_size],
    );
    push_req(
        tensors,
        format!("{prefix}.post_attention_layernorm.weight"),
        Dtype::BF16,
        [config.hidden_size],
    );
    let attn = format!("{prefix}.self_attn");
    push_req(
        tensors,
        format!("{attn}.q_proj.weight"),
        Dtype::BF16,
        [config.q_proj_rows(), config.hidden_size],
    );
    push_req(
        tensors,
        format!("{attn}.kv_a_proj_with_mqa.weight"),
        Dtype::BF16,
        [config.kv_a_proj_rows(), config.hidden_size],
    );
    push_req(
        tensors,
        format!("{attn}.kv_a_layernorm.weight"),
        Dtype::BF16,
        [config.kv_lora_rank],
    );
    push_req(
        tensors,
        format!("{attn}.kv_b_proj.weight"),
        Dtype::BF16,
        [config.kv_b_proj_rows(), config.kv_lora_rank],
    );
    push_req(
        tensors,
        format!("{attn}.o_proj.weight"),
        Dtype::BF16,
        [config.hidden_size, config.o_proj_cols()],
    );
}

fn push_dense_layer(config: &Config, layer: usize, tensors: &mut Vec<TensorRequirement>) {
    let prefix = format!("model.layers.{layer}.mlp");
    push_req(
        tensors,
        format!("{prefix}.gate_proj.weight"),
        Dtype::BF16,
        [config.intermediate_size, config.hidden_size],
    );
    push_req(
        tensors,
        format!("{prefix}.up_proj.weight"),
        Dtype::BF16,
        [config.intermediate_size, config.hidden_size],
    );
    push_req(
        tensors,
        format!("{prefix}.down_proj.weight"),
        Dtype::BF16,
        [config.hidden_size, config.intermediate_size],
    );
}

fn push_moe_layer(
    config: &Config,
    layout: &ExpertParallelLayout,
    layer: usize,
    tensors: &mut Vec<TensorRequirement>,
) {
    let prefix = format!("model.layers.{layer}.mlp");
    push_moe_gate(config, layer, tensors);
    let shared = format!("{prefix}.shared_experts");
    push_req(
        tensors,
        format!("{shared}.gate_proj.weight"),
        Dtype::BF16,
        [config.shared_moe_intermediate(), config.hidden_size],
    );
    push_req(
        tensors,
        format!("{shared}.up_proj.weight"),
        Dtype::BF16,
        [config.shared_moe_intermediate(), config.hidden_size],
    );
    push_req(
        tensors,
        format!("{shared}.down_proj.weight"),
        Dtype::BF16,
        [config.hidden_size, config.shared_moe_intermediate()],
    );
    push_owned_experts(config, layout, layer, tensors);
}

fn push_moe_gate(config: &Config, layer: usize, tensors: &mut Vec<TensorRequirement>) {
    let prefix = format!("model.layers.{layer}.mlp");
    push_req(
        tensors,
        format!("{prefix}.gate.weight"),
        Dtype::BF16,
        [config.n_routed_experts, config.hidden_size],
    );
}

fn push_owned_experts(
    config: &Config,
    layout: &ExpertParallelLayout,
    layer: usize,
    tensors: &mut Vec<TensorRequirement>,
) {
    let prefix = format!("model.layers.{layer}.mlp");
    for global_expert in layout.owned_experts() {
        let expert = format!("{prefix}.experts.{global_expert}");
        push_req(
            tensors,
            format!("{expert}.gate_proj.weight"),
            Dtype::BF16,
            [config.moe_intermediate_size, config.hidden_size],
        );
        push_req(
            tensors,
            format!("{expert}.up_proj.weight"),
            Dtype::BF16,
            [config.moe_intermediate_size, config.hidden_size],
        );
        push_req(
            tensors,
            format!("{expert}.down_proj.weight"),
            Dtype::BF16,
            [config.hidden_size, config.moe_intermediate_size],
        );
    }
}

fn shard_paths(model_path: &Path) -> Result<Vec<PathBuf>> {
    let single = model_path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index_path = model_path.join("model.safetensors.index.json");
    let index = read_index(model_path)?
        .ok_or_else(|| anyhow::anyhow!("missing {}", index_path.display()))?;
    let unique: BTreeSet<_> = index.into_values().collect();
    Ok(unique
        .into_iter()
        .map(|shard| model_path.join(shard))
        .collect())
}

fn read_index(model_path: &Path) -> Result<Option<HashMap<String, String>>> {
    let index_path = model_path.join("model.safetensors.index.json");
    match fs::read_to_string(&index_path) {
        Ok(content) => {
            let index: SafetensorsIndex = serde_json::from_str(&content)
                .with_context(|| format!("parse {}", index_path.display()))?;
            Ok(Some(index.weight_map))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: The safetensors shard is opened read-only and only read for
    // metadata while the mmap is alive.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::config::test_lite_config;
    use crate::ep::ExpertParallelConfig;

    #[test]
    fn driver_rank0_load_plan_keeps_routed_experts_rank_local() {
        let config = test_lite_config();
        let rank0 = ExpertParallelConfig::ep2(0).validate_for(&config).unwrap();
        let rank0_plan = RankLoadPlan::for_driver_rank(&config, &rank0);
        let expected_routed_tensors =
            (config.num_hidden_layers - config.first_k_dense_replace) * 32 * 3;

        assert_eq!(
            routed_expert_tensor_count(&rank0_plan),
            expected_routed_tensors
        );
        assert_eq!(routed_experts(&rank0_plan), (0..32).collect());
    }

    #[test]
    fn expert_rank_load_plan_requires_gate_and_owned_routed_experts() {
        let config = test_lite_config();
        let rank1 = ExpertParallelConfig::ep2(1).validate_for(&config).unwrap();
        let plan = RankLoadPlan::for_expert_rank(&config, &rank1);
        let moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
        let expected_routed_tensors = moe_layers * (config.n_routed_experts / rank1.ep_size()) * 3;

        assert_eq!(plan.tensors.len(), expected_routed_tensors + moe_layers);
        assert_eq!(routed_expert_tensor_count(&plan), expected_routed_tensors);
        assert_eq!(moe_gate_tensor_count(&plan), moe_layers);
        assert!(
            plan.tensors
                .iter()
                .all(|req| req.name.ends_with(".gate.weight") || req.name.contains(".experts."))
        );
        assert_eq!(routed_experts(&plan), (32..64).collect());
    }

    fn routed_expert_tensor_count(plan: &RankLoadPlan) -> usize {
        plan.tensors
            .iter()
            .filter(|req| req.name.contains(".experts."))
            .count()
    }

    fn moe_gate_tensor_count(plan: &RankLoadPlan) -> usize {
        plan.tensors
            .iter()
            .filter(|req| req.name.ends_with(".mlp.gate.weight"))
            .count()
    }

    fn routed_experts(plan: &RankLoadPlan) -> BTreeSet<usize> {
        plan.tensors
            .iter()
            .filter_map(|req| {
                let (_, suffix) = req.name.split_once(".experts.")?;
                suffix.split('.').next()?.parse().ok()
            })
            .collect()
    }
}
