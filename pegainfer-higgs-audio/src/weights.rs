use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

use crate::config::HiggsConfig;
use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

pub const TEXT_EMBEDDING: &str = "tied.embedding.text_embedding.weight";
pub const FUSED_MODALITY_EMBEDDING: &str = "tied.embedding.modality_embeddings.0.embedding.weight";
pub const BODY_NORM: &str = "body.norm.weight";

#[derive(Debug, Clone)]
pub struct HiggsWeightManifest {
    pub total_size: Option<u64>,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSummary {
    pub total_tensors: usize,
    pub body_tensors: usize,
    pub decoder_only_tensors: usize,
    pub has_text_embedding: bool,
    pub has_fused_modality_embedding: bool,
    pub has_separate_audio_head: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeaderSpec {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHeaderSummary {
    pub files_checked: usize,
    pub tensors_checked: usize,
    pub bf16_tensors_checked: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    byte_range: [usize; 2],
}

impl HiggsWeightManifest {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("model.safetensors.index.json");
        let value: Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        Self::from_json(&value)
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let total_size = value
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(|total| match total {
                Value::Number(n) => n.as_u64(),
                Value::String(s) => s.parse().ok(),
                _ => None,
            });
        let raw = value
            .get("weight_map")
            .and_then(Value::as_object)
            .context("index missing weight_map")?;
        let mut weight_map = HashMap::with_capacity(raw.len());
        for (key, value) in raw {
            let file = value
                .as_str()
                .with_context(|| format!("weight_map entry {key} is not a string"))?;
            weight_map.insert(key.clone(), file.to_string());
        }
        Ok(Self {
            total_size,
            weight_map,
        })
    }

    pub fn summary(&self) -> ManifestSummary {
        ManifestSummary {
            total_tensors: self.weight_map.len(),
            body_tensors: self
                .weight_map
                .keys()
                .filter(|name| name.starts_with("body."))
                .count(),
            decoder_only_tensors: self
                .weight_map
                .keys()
                .filter(|name| is_decoder_only_tensor(name))
                .count(),
            has_text_embedding: self.weight_map.contains_key(TEXT_EMBEDDING),
            has_fused_modality_embedding: self.weight_map.contains_key(FUSED_MODALITY_EMBEDDING),
            has_separate_audio_head: self.weight_map.keys().any(|name| {
                name.starts_with("tied.head.modality") || name.starts_with("tied.head.audio")
            }),
        }
    }

    pub fn validate_for_config(&self, config: &HiggsConfig) -> Result<ManifestSummary> {
        let summary = self.summary();
        if !summary.has_text_embedding {
            bail!("Higgs manifest missing {TEXT_EMBEDDING}");
        }
        if !summary.has_fused_modality_embedding {
            bail!("Higgs manifest missing {FUSED_MODALITY_EMBEDDING}");
        }
        if summary.has_separate_audio_head {
            bail!(
                "Higgs manifest unexpectedly contains a separate audio head; current checkpoint ties the fused modality head"
            );
        }
        let required = required_body_tensors(config);
        let missing: Vec<_> = required
            .iter()
            .filter(|name| !self.weight_map.contains_key(*name))
            .cloned()
            .collect();
        if !missing.is_empty() {
            bail!(
                "Higgs manifest missing {} required body tensor(s): {:?}",
                missing.len(),
                &missing[..missing.len().min(8)]
            );
        }
        if summary.body_tensors != required.len() {
            bail!(
                "Higgs manifest body tensor count mismatch: expected {}, got {}",
                required.len(),
                summary.body_tensors
            );
        }
        Ok(summary)
    }
}

pub fn validate_checkpoint_headers(
    model_dir: impl AsRef<Path>,
    config: &HiggsConfig,
    manifest: &HiggsWeightManifest,
) -> Result<CheckpointHeaderSummary> {
    let expected = expected_checkpoint_tensors(config);
    let mut by_file: BTreeMap<String, Vec<&TensorHeaderSpec>> = BTreeMap::new();
    for spec in &expected {
        let file = manifest
            .weight_map
            .get(&spec.name)
            .with_context(|| format!("manifest missing expected tensor {}", spec.name))?;
        by_file.entry(file.clone()).or_default().push(spec);
    }

    let mut tensors_checked = 0usize;
    let mut bf16_tensors_checked = 0usize;
    for (file, specs) in &by_file {
        let path = model_dir.as_ref().join(file);
        let header = read_safetensors_header(&path)?;
        for spec in specs {
            let actual = header
                .get(&spec.name)
                .with_context(|| format!("{} missing tensor {}", path.display(), spec.name))?;
            ensure!(
                actual.dtype == spec.dtype,
                "{} dtype mismatch: expected {}, got {}",
                spec.name,
                spec.dtype,
                actual.dtype
            );
            ensure!(
                actual.shape == spec.shape,
                "{} shape mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.shape,
                actual.shape
            );
            ensure!(
                actual.byte_range[0] <= actual.byte_range[1],
                "{} has invalid data_offsets {:?}",
                spec.name,
                actual.byte_range
            );
            tensors_checked += 1;
            if actual.dtype == "BF16" {
                bf16_tensors_checked += 1;
            }
        }
    }

    Ok(CheckpointHeaderSummary {
        files_checked: by_file.len(),
        tensors_checked,
        bf16_tensors_checked,
    })
}

pub fn expected_checkpoint_tensors(config: &HiggsConfig) -> Vec<TensorHeaderSpec> {
    let hidden = config.text.hidden_size;
    let head_dim = config.text.head_dim;
    let q_dim = config.text.num_attention_heads * head_dim;
    let kv_dim = config.text.num_key_value_heads * head_dim;
    let intermediate = config.text.intermediate_size;
    let mut specs = Vec::with_capacity(2 + 1 + config.text.num_hidden_layers * 11);
    specs.push(TensorHeaderSpec {
        name: TEXT_EMBEDDING.to_string(),
        dtype: "BF16",
        shape: vec![config.text.vocab_size, hidden],
    });
    specs.push(TensorHeaderSpec {
        name: FUSED_MODALITY_EMBEDDING.to_string(),
        dtype: "BF16",
        shape: fused_modality_shape().to_vec(),
    });
    specs.push(TensorHeaderSpec {
        name: BODY_NORM.to_string(),
        dtype: "BF16",
        shape: vec![hidden],
    });
    for layer in 0..config.text.num_hidden_layers {
        let prefix = format!("body.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("self_attn.q_proj.weight", vec![q_dim, hidden]),
            ("self_attn.k_proj.weight", vec![kv_dim, hidden]),
            ("self_attn.v_proj.weight", vec![kv_dim, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, q_dim]),
            ("self_attn.q_norm.weight", vec![head_dim]),
            ("self_attn.k_norm.weight", vec![head_dim]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            specs.push(TensorHeaderSpec {
                name: format!("{prefix}.{suffix}"),
                dtype: "BF16",
                shape,
            });
        }
    }
    specs
}

pub fn required_body_tensors(config: &HiggsConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert(BODY_NORM.to_string());
    for layer in 0..config.text.num_hidden_layers {
        let prefix = format!("body.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.insert(format!("{prefix}.{suffix}"));
        }
    }
    names
}

pub fn fused_modality_shape() -> [usize; 2] {
    [NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE, HIDDEN_SIZE]
}

fn is_decoder_only_tensor(name: &str) -> bool {
    name.starts_with("tied.embedding.modality_embeddings.0.model.quantizer")
        || name.starts_with("tied.embedding.modality_embeddings.0.model.fc2")
        || name.starts_with("tied.embedding.modality_embeddings.0.model.acoustic_decoder")
}

fn read_safetensors_header(path: &Path) -> Result<HashMap<String, TensorHeader>> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("read safetensors header length from {}", path.display()))?;
    let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).with_context(|| {
        format!(
            "{} safetensors header length does not fit usize",
            path.display()
        )
    })?;
    ensure!(
        header_len < 512 * 1024 * 1024,
        "{} safetensors header is unexpectedly large: {} bytes",
        path.display(),
        header_len
    );
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("read safetensors header from {}", path.display()))?;
    let value: Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;
    let object = value
        .as_object()
        .with_context(|| format!("{} safetensors header is not a JSON object", path.display()))?;
    let mut tensors = HashMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = str_field(value, "dtype")?.to_string();
        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing shape"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .context("shape dim is not u64")
                    .and_then(|dim| usize::try_from(dim).context("shape dim does not fit usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing data_offsets"))?;
        ensure!(offsets.len() == 2, "{name} data_offsets must have length 2");
        let start = offsets[0]
            .as_u64()
            .with_context(|| format!("{name} data_offsets[0] is not u64"))
            .and_then(|offset| usize::try_from(offset).context("offset does not fit usize"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("{name} data_offsets[1] is not u64"))
            .and_then(|offset| usize::try_from(offset).context("offset does not fit usize"))?;
        tensors.insert(
            name.clone(),
            TensorHeader {
                dtype,
                shape,
                byte_range: [start, end],
            },
        );
    }
    Ok(tensors)
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {key}"))
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

    fn manifest_json(include_fused_head: bool) -> Value {
        let cfg = config();
        let mut weight_map = serde_json::Map::new();
        weight_map.insert(
            TEXT_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        if include_fused_head {
            weight_map.insert(
                FUSED_MODALITY_EMBEDDING.to_string(),
                serde_json::json!("model.safetensors"),
            );
        }
        for name in required_body_tensors(&cfg) {
            weight_map.insert(name, serde_json::json!("model.safetensors"));
        }
        serde_json::json!({"metadata": {"total_size": "8489763794"}, "weight_map": weight_map})
    }

    #[test]
    fn validates_required_higgs_manifest_surface() {
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(true)).unwrap();
        let summary = manifest.validate_for_config(&cfg).unwrap();
        assert_eq!(summary.body_tensors, 397);
        assert_eq!(summary.total_tensors, 399);
        assert_eq!(fused_modality_shape(), [8208, 2560]);
    }

    #[test]
    fn rejects_manifest_without_fused_modality_head_weight() {
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(false)).unwrap();
        let err = manifest.validate_for_config(&cfg).unwrap_err().to_string();
        assert!(err.contains(FUSED_MODALITY_EMBEDDING));
    }

    #[test]
    fn expected_checkpoint_tensor_shapes_cover_higgs_body() {
        let specs = expected_checkpoint_tensors(&config());
        assert_eq!(specs.len(), 399);
        assert!(specs.iter().any(|spec| {
            spec.name == "body.layers.0.self_attn.q_proj.weight"
                && spec.shape == [4096, HIDDEN_SIZE]
        }));
        assert!(specs.iter().any(|spec| {
            spec.name == "body.layers.0.self_attn.k_proj.weight"
                && spec.shape == [1024, HIDDEN_SIZE]
        }));
        assert!(specs.iter().any(|spec| {
            spec.name == "body.layers.0.self_attn.q_norm.weight"
                && spec.shape == [EXPECTED_HEAD_DIM]
        }));
        assert!(specs.iter().any(|spec| {
            spec.name == "body.layers.0.mlp.down_proj.weight"
                && spec.shape == [HIDDEN_SIZE, EXPECTED_INTERMEDIATE_SIZE]
        }));
    }

    #[test]
    fn validates_checkpoint_header_without_reading_payload() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(true)).unwrap();
        write_fake_safetensors_header(dir.path().join("model.safetensors"), &cfg);

        let summary = validate_checkpoint_headers(dir.path(), &cfg, &manifest).unwrap();
        assert_eq!(summary.files_checked, 1);
        assert_eq!(summary.tensors_checked, 399);
        assert_eq!(summary.bf16_tensors_checked, 399);
    }

    #[test]
    fn rejects_checkpoint_header_shape_drift() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(true)).unwrap();
        let mut specs = expected_checkpoint_tensors(&cfg);
        let q_proj = specs
            .iter_mut()
            .find(|spec| spec.name == "body.layers.0.self_attn.q_proj.weight")
            .unwrap();
        q_proj.shape = vec![1, HIDDEN_SIZE];
        write_fake_safetensors_header_from_specs(dir.path().join("model.safetensors"), &specs);

        let err = validate_checkpoint_headers(dir.path(), &cfg, &manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains("q_proj"));
        assert!(err.contains("shape mismatch"));
    }

    fn write_fake_safetensors_header(path: impl AsRef<Path>, config: &HiggsConfig) {
        write_fake_safetensors_header_from_specs(path, &expected_checkpoint_tensors(config));
    }

    fn write_fake_safetensors_header_from_specs(
        path: impl AsRef<Path>,
        specs: &[TensorHeaderSpec],
    ) {
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for spec in specs {
            let bytes = spec.shape.iter().product::<usize>() * 2;
            header.insert(
                spec.name.clone(),
                serde_json::json!({
                    "dtype": spec.dtype,
                    "shape": spec.shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        std::fs::write(path, bytes).unwrap();
    }
}
