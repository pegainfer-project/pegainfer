use std::collections::BTreeMap;
use std::collections::BTreeSet;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

use crate::config::HiggsConfig;
use crate::weights::BODY_NORM;
use crate::weights::FUSED_MODALITY_EMBEDDING;
use crate::weights::HiggsWeightManifest;
use crate::weights::TEXT_EMBEDDING;
use crate::weights::TensorHeaderSpec;
use crate::weights::expected_checkpoint_tensors;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    TextEmbedding,
    FusedAudioHead,
    BodyNorm,
    LayerInputLayernorm,
    LayerPostAttentionLayernorm,
    LayerQProj,
    LayerKProj,
    LayerVProj,
    LayerOProj,
    LayerQNorm,
    LayerKNorm,
    LayerGateProj,
    LayerUpProj,
    LayerDownProj,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub checkpoint_name: String,
    pub shard_file: String,
    pub role: TensorRole,
    pub loader_slot: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlanSummary {
    pub tensors: usize,
    pub shard_files: usize,
    pub bf16_bytes: usize,
    pub body_tensors: usize,
    pub qwen3_backbone_tensors: usize,
    pub higgs_head_tensors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiggsRuntimeLoadPlan {
    pub tensors: Vec<PlannedTensor>,
}

impl HiggsRuntimeLoadPlan {
    pub fn from_manifest(config: &HiggsConfig, manifest: &HiggsWeightManifest) -> Result<Self> {
        manifest.validate_for_config(config)?;
        let tensors = expected_checkpoint_tensors(config)
            .into_iter()
            .map(|spec| planned_tensor(spec, manifest))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tensors })
    }

    pub fn summary(&self) -> LoadPlanSummary {
        let shard_files: BTreeSet<_> = self
            .tensors
            .iter()
            .map(|tensor| tensor.shard_file.as_str())
            .collect();
        LoadPlanSummary {
            tensors: self.tensors.len(),
            shard_files: shard_files.len(),
            bf16_bytes: self
                .tensors
                .iter()
                .filter(|tensor| tensor.dtype == "BF16")
                .map(|tensor| tensor.bytes)
                .sum(),
            body_tensors: self
                .tensors
                .iter()
                .filter(|tensor| tensor.checkpoint_name.starts_with("body."))
                .count(),
            qwen3_backbone_tensors: self
                .tensors
                .iter()
                .filter(|tensor| tensor.loader_slot.starts_with("qwen3."))
                .count(),
            higgs_head_tensors: self
                .tensors
                .iter()
                .filter(|tensor| tensor.loader_slot.starts_with("higgs."))
                .count(),
        }
    }

    pub fn tensor(&self, checkpoint_name: &str) -> Option<&PlannedTensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.checkpoint_name == checkpoint_name)
    }

    pub fn qwen3_tensor_aliases(&self) -> Result<BTreeMap<String, String>> {
        self.tensors
            .iter()
            .filter(|tensor| tensor.loader_slot.starts_with("qwen3."))
            .map(|tensor| {
                Ok((
                    qwen3_tensor_name(&tensor.loader_slot)?,
                    tensor.checkpoint_name.clone(),
                ))
            })
            .collect()
    }
}

fn planned_tensor(spec: TensorHeaderSpec, manifest: &HiggsWeightManifest) -> Result<PlannedTensor> {
    let role = tensor_role(&spec.name)?;
    let loader_slot = loader_slot(&spec.name, role)?;
    let shard_file = manifest
        .weight_map
        .get(&spec.name)
        .with_context(|| format!("manifest missing tensor {}", spec.name))?
        .clone();
    let elements = spec.shape.iter().product::<usize>();
    let bytes_per_element = dtype_bytes(spec.dtype)?;
    Ok(PlannedTensor {
        checkpoint_name: spec.name,
        shard_file,
        role,
        loader_slot,
        dtype: spec.dtype,
        shape: spec.shape,
        elements,
        bytes: elements * bytes_per_element,
    })
}

fn tensor_role(name: &str) -> Result<TensorRole> {
    if name == TEXT_EMBEDDING {
        return Ok(TensorRole::TextEmbedding);
    }
    if name == FUSED_MODALITY_EMBEDDING {
        return Ok(TensorRole::FusedAudioHead);
    }
    if name == BODY_NORM {
        return Ok(TensorRole::BodyNorm);
    }
    let suffix = layer_suffix(name)?;
    match suffix {
        "input_layernorm.weight" => Ok(TensorRole::LayerInputLayernorm),
        "post_attention_layernorm.weight" => Ok(TensorRole::LayerPostAttentionLayernorm),
        "self_attn.q_proj.weight" => Ok(TensorRole::LayerQProj),
        "self_attn.k_proj.weight" => Ok(TensorRole::LayerKProj),
        "self_attn.v_proj.weight" => Ok(TensorRole::LayerVProj),
        "self_attn.o_proj.weight" => Ok(TensorRole::LayerOProj),
        "self_attn.q_norm.weight" => Ok(TensorRole::LayerQNorm),
        "self_attn.k_norm.weight" => Ok(TensorRole::LayerKNorm),
        "mlp.gate_proj.weight" => Ok(TensorRole::LayerGateProj),
        "mlp.up_proj.weight" => Ok(TensorRole::LayerUpProj),
        "mlp.down_proj.weight" => Ok(TensorRole::LayerDownProj),
        _ => bail!("unsupported Higgs layer tensor suffix {suffix} in {name}"),
    }
}

fn loader_slot(name: &str, role: TensorRole) -> Result<String> {
    match role {
        TensorRole::TextEmbedding => Ok("qwen3.embed_tokens".to_string()),
        TensorRole::FusedAudioHead => Ok("higgs.fused_audio_head".to_string()),
        TensorRole::BodyNorm => Ok("qwen3.norm".to_string()),
        _ => {
            let layer = layer_index(name)?;
            Ok(format!("qwen3.layers.{layer}.{}", layer_suffix(name)?))
        }
    }
}

pub fn qwen3_tensor_name(loader_slot: &str) -> Result<String> {
    match loader_slot {
        "qwen3.embed_tokens" => Ok("model.embed_tokens.weight".to_string()),
        "qwen3.norm" => Ok("model.norm.weight".to_string()),
        slot if slot.starts_with("qwen3.layers.") => {
            Ok(format!("model.layers.{}", &slot["qwen3.layers.".len()..]))
        }
        _ => bail!("loader slot {loader_slot} is not part of the Qwen3 body view"),
    }
}

fn layer_index(name: &str) -> Result<usize> {
    let rest = name
        .strip_prefix("body.layers.")
        .with_context(|| format!("{name} is not a body layer tensor"))?;
    let Some((layer, _suffix)) = rest.split_once('.') else {
        bail!("{name} is missing layer suffix");
    };
    layer
        .parse::<usize>()
        .with_context(|| format!("{name} has invalid layer index"))
}

fn layer_suffix(name: &str) -> Result<&str> {
    let rest = name
        .strip_prefix("body.layers.")
        .with_context(|| format!("{name} is not a body layer tensor"))?;
    let Some((_layer, suffix)) = rest.split_once('.') else {
        bail!("{name} is missing layer suffix");
    };
    Ok(suffix)
}

fn dtype_bytes(dtype: &str) -> Result<usize> {
    match dtype {
        "BF16" => Ok(2),
        "F32" | "I32" => Ok(4),
        "I64" => Ok(8),
        _ => bail!("unsupported dtype in Higgs runtime load plan: {dtype}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXPECTED_ARCHITECTURE;
    use crate::config::EXPECTED_AUDIO_ENCODER_TYPE;
    use crate::config::EXPECTED_HEAD_DIM;
    use crate::config::EXPECTED_INTERMEDIATE_SIZE;
    use crate::config::EXPECTED_MODEL_TYPE;
    use crate::config::EXPECTED_NUM_ATTENTION_HEADS;
    use crate::config::EXPECTED_NUM_KV_HEADS;
    use crate::config::EXPECTED_NUM_LAYERS;
    use crate::config::EXPECTED_ROPE_THETA;
    use crate::config::EXPECTED_TEXT_VOCAB_SIZE;
    use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
    use crate::one_step_golden::HIDDEN_SIZE;
    use crate::one_step_golden::NUM_CODEBOOKS;
    use crate::weights::fused_modality_shape;
    use crate::weights::required_body_tensors;

    fn config() -> HiggsConfig {
        HiggsConfig::from_json(&serde_json::json!({
            "architectures": [EXPECTED_ARCHITECTURE],
            "audio_token_id": -100,
            "model_type": EXPECTED_MODEL_TYPE,
            "text_config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
                "num_hidden_layers": EXPECTED_NUM_LAYERS,
                "num_attention_heads": EXPECTED_NUM_ATTENTION_HEADS,
                "num_key_value_heads": EXPECTED_NUM_KV_HEADS,
                "head_dim": EXPECTED_HEAD_DIM,
                "vocab_size": EXPECTED_TEXT_VOCAB_SIZE,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "tie_word_embeddings": true,
                "rope_parameters": {"rope_theta": EXPECTED_ROPE_THETA}
            },
            "audio_encoder_config": {
                "encoder_type": EXPECTED_AUDIO_ENCODER_TYPE,
                "num_codebooks": NUM_CODEBOOKS,
                "vocab_size": CODEBOOK_VOCAB_SIZE,
                "out_dim": HIDDEN_SIZE,
                "tie_word_embeddings": true,
                "use_delay_pattern": true
            }
        }))
        .unwrap()
    }

    fn manifest() -> HiggsWeightManifest {
        let cfg = config();
        let mut weight_map = serde_json::Map::new();
        weight_map.insert(
            TEXT_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        weight_map.insert(
            FUSED_MODALITY_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        for name in required_body_tensors(&cfg) {
            weight_map.insert(name, serde_json::json!("model.safetensors"));
        }
        HiggsWeightManifest::from_json(&serde_json::json!({
            "metadata": {"total_size": "8489763794"},
            "weight_map": weight_map
        }))
        .unwrap()
    }

    #[test]
    fn builds_runtime_load_plan_for_higgs_backbone_and_audio_head() {
        let cfg = config();
        let plan = HiggsRuntimeLoadPlan::from_manifest(&cfg, &manifest()).unwrap();
        let summary = plan.summary();

        assert_eq!(summary.tensors, 399);
        assert_eq!(summary.shard_files, 1);
        assert_eq!(summary.body_tensors, 397);
        assert_eq!(summary.qwen3_backbone_tensors, 398);
        assert_eq!(summary.higgs_head_tensors, 1);

        let text = plan.tensor(TEXT_EMBEDDING).unwrap();
        assert_eq!(text.role, TensorRole::TextEmbedding);
        assert_eq!(text.loader_slot, "qwen3.embed_tokens");
        assert_eq!(text.shape, [EXPECTED_TEXT_VOCAB_SIZE, HIDDEN_SIZE]);

        let audio = plan.tensor(FUSED_MODALITY_EMBEDDING).unwrap();
        assert_eq!(audio.role, TensorRole::FusedAudioHead);
        assert_eq!(audio.loader_slot, "higgs.fused_audio_head");
        assert_eq!(audio.shape, fused_modality_shape());

        let q_proj = plan
            .tensor("body.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj.role, TensorRole::LayerQProj);
        assert_eq!(q_proj.loader_slot, "qwen3.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn load_plan_rejects_missing_required_manifest_tensor() {
        let cfg = config();
        let mut manifest = manifest();
        manifest.weight_map.remove(TEXT_EMBEDDING);
        let err = HiggsRuntimeLoadPlan::from_manifest(&cfg, &manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains(TEXT_EMBEDDING));
    }

    #[test]
    fn builds_qwen3_tensor_aliases_without_audio_head() {
        let cfg = config();
        let plan = HiggsRuntimeLoadPlan::from_manifest(&cfg, &manifest()).unwrap();
        let aliases = plan.qwen3_tensor_aliases().unwrap();

        assert_eq!(aliases.len(), 398);
        assert_eq!(
            aliases.get("model.embed_tokens.weight").unwrap(),
            TEXT_EMBEDDING
        );
        assert_eq!(aliases.get("model.norm.weight").unwrap(), BODY_NORM);
        assert_eq!(
            aliases
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
            "body.layers.0.self_attn.q_proj.weight"
        );
        assert!(
            !aliases
                .values()
                .any(|checkpoint_name| checkpoint_name == FUSED_MODALITY_EMBEDDING)
        );
    }
}
