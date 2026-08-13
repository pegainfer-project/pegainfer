use std::collections::BTreeMap;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

use crate::config::HiggsConfig;
use crate::load_plan::HiggsRuntimeLoadPlan;
use crate::load_plan::PlannedTensor;
use crate::load_plan::qwen3_tensor_name;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeSummary {
    pub output_dir: PathBuf,
    pub tensors: usize,
    pub payload_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigViewSummary {
    pub output_dir: PathBuf,
    pub alias_manifest: PathBuf,
    pub aliases: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub fn materialize_qwen3_body_view(
    source_model_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    config: &HiggsConfig,
    load_plan: &HiggsRuntimeLoadPlan,
) -> Result<MaterializeSummary> {
    let source_model_dir = source_model_dir.as_ref();
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create {}", output_dir.display()))?;

    write_qwen3_config(output_dir, config)?;
    write_generation_config(output_dir, config)?;

    let qwen_tensors: Vec<_> = load_plan
        .tensors
        .iter()
        .filter(|tensor| tensor.loader_slot.starts_with("qwen3."))
        .collect();
    ensure!(
        qwen_tensors.len() == 398,
        "expected 398 Qwen3 backbone tensors, got {}",
        qwen_tensors.len()
    );

    let shard_files = qwen_tensors
        .iter()
        .map(|tensor| tensor.shard_file.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        shard_files.len() == 1,
        "Qwen3 body materializer currently expects one Higgs source shard, got {}",
        shard_files.len()
    );
    let shard_file = shard_files.iter().next().context("missing source shard")?;
    let source_safetensors = source_model_dir.join(shard_file);
    let output_safetensors = output_dir.join("model.safetensors");
    materialize_safetensors_alias(&source_safetensors, &output_safetensors, &qwen_tensors)?;

    Ok(MaterializeSummary {
        output_dir: output_dir.to_path_buf(),
        tensors: qwen_tensors.len(),
        payload_bytes: qwen_tensors.iter().map(|tensor| tensor.bytes).sum(),
    })
}

pub fn write_qwen3_config_view(
    output_dir: impl AsRef<Path>,
    config: &HiggsConfig,
    load_plan: &HiggsRuntimeLoadPlan,
) -> Result<ConfigViewSummary> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create {}", output_dir.display()))?;
    write_qwen3_config(output_dir, config)?;
    write_generation_config(output_dir, config)?;

    let aliases = load_plan.qwen3_tensor_aliases()?;
    let alias_manifest = output_dir.join("higgs-qwen3-tensor-aliases.json");
    write_json(
        &alias_manifest,
        &json!({
            "format": "higgs-qwen3-tensor-aliases-v1",
            "requested_to_stored": aliases,
        }),
    )?;
    Ok(ConfigViewSummary {
        output_dir: output_dir.to_path_buf(),
        alias_manifest,
        aliases: load_plan.qwen3_tensor_aliases()?.len(),
    })
}

fn write_qwen3_config(output_dir: &Path, config: &HiggsConfig) -> Result<()> {
    let value = json!({
        "hidden_size": config.text.hidden_size,
        "intermediate_size": config.text.intermediate_size,
        "num_hidden_layers": config.text.num_hidden_layers,
        "num_attention_heads": config.text.num_attention_heads,
        "num_key_value_heads": config.text.num_key_value_heads,
        "head_dim": config.text.head_dim,
        "vocab_size": config.text.vocab_size,
        "rms_norm_eps": config.text.rms_norm_eps,
        "rope_theta": config.text.rope_theta as f32,
        "eos_token_id": config.text.eos_token_id,
        "tie_word_embeddings": true
    });
    write_json(output_dir.join("config.json"), &value)
}

fn write_generation_config(output_dir: &Path, config: &HiggsConfig) -> Result<()> {
    write_json(
        output_dir.join("generation_config.json"),
        &json!({"eos_token_id": config.text.eos_token_id}),
    )
}

fn write_json(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize JSON")?;
    std::fs::write(path.as_ref(), [bytes, b"\n".to_vec()].concat())
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn materialize_safetensors_alias(
    source_path: &Path,
    output_path: &Path,
    tensors: &[&PlannedTensor],
) -> Result<()> {
    let mut source = std::fs::File::open(source_path)
        .with_context(|| format!("open {}", source_path.display()))?;
    let (source_data_start, source_headers) = read_source_headers(&mut source, source_path)?;
    let mut aliases = Vec::with_capacity(tensors.len());
    let mut output_offset = 0usize;
    for tensor in tensors {
        let source_header = source_headers
            .get(&tensor.checkpoint_name)
            .with_context(|| format!("source header missing {}", tensor.checkpoint_name))?;
        ensure!(
            source_header.dtype == tensor.dtype,
            "{} dtype drift: plan {} source {}",
            tensor.checkpoint_name,
            tensor.dtype,
            source_header.dtype
        );
        ensure!(
            source_header.shape == tensor.shape,
            "{} shape drift: plan {:?} source {:?}",
            tensor.checkpoint_name,
            tensor.shape,
            source_header.shape
        );
        let alias = qwen3_tensor_name(&tensor.loader_slot)?;
        aliases.push((
            alias,
            tensor.checkpoint_name.clone(),
            source_header.data_offsets,
            output_offset,
            output_offset + tensor.bytes,
            source_header.dtype.clone(),
            source_header.shape.clone(),
        ));
        output_offset += tensor.bytes;
    }

    let mut header = serde_json::Map::new();
    for (alias, _source_name, _source_offsets, start, end, dtype, shape) in &aliases {
        header.insert(
            alias.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let mut header_bytes =
        serde_json::to_vec(&Value::Object(header)).context("serialize header")?;
    while !(8 + header_bytes.len()).is_multiple_of(std::mem::align_of::<half::bf16>()) {
        header_bytes.push(b' ');
    }
    let mut output = std::fs::File::create(output_path)
        .with_context(|| format!("create {}", output_path.display()))?;
    output
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .context("write safetensors header length")?;
    output
        .write_all(&header_bytes)
        .context("write safetensors header")?;

    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for (_alias, source_name, source_offsets, _start, _end, _dtype, _shape) in &aliases {
        let len = source_offsets[1] - source_offsets[0];
        copy_exact_range(
            &mut source,
            source_data_start + source_offsets[0] as u64,
            len,
            &mut output,
            &mut buffer,
        )
        .with_context(|| format!("copy tensor payload for {source_name}"))?;
    }
    Ok(())
}

fn read_source_headers(
    source: &mut std::fs::File,
    path: &Path,
) -> Result<(u64, BTreeMap<String, SourceTensorHeader>)> {
    let mut len_bytes = [0u8; 8];
    source
        .read_exact(&mut len_bytes)
        .with_context(|| format!("read header length from {}", path.display()))?;
    let header_len = usize::try_from(u64::from_le_bytes(len_bytes))
        .with_context(|| format!("{} header length does not fit usize", path.display()))?;
    let mut header_bytes = vec![0u8; header_len];
    source
        .read_exact(&mut header_bytes)
        .with_context(|| format!("read header from {}", path.display()))?;
    let value: Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;
    let object = value
        .as_object()
        .with_context(|| format!("{} header is not an object", path.display()))?;
    let mut headers = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(Value::as_str)
            .with_context(|| format!("{name} missing dtype"))?
            .to_string();
        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing shape"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .context("shape dim not u64")
                    .and_then(|dim| usize::try_from(dim).context("shape dim does not fit usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing data_offsets"))?;
        ensure!(offsets.len() == 2, "{name} data_offsets length must be 2");
        let start = offsets[0]
            .as_u64()
            .context("start offset not u64")
            .and_then(|offset| usize::try_from(offset).context("start does not fit usize"))?;
        let end = offsets[1]
            .as_u64()
            .context("end offset not u64")
            .and_then(|offset| usize::try_from(offset).context("end does not fit usize"))?;
        ensure!(start <= end, "{name} invalid data_offsets [{start}, {end}]");
        headers.insert(
            name.clone(),
            SourceTensorHeader {
                dtype,
                shape,
                data_offsets: [start, end],
            },
        );
    }
    Ok((8 + header_len as u64, headers))
}

fn copy_exact_range(
    input: &mut std::fs::File,
    start: u64,
    len: usize,
    output: &mut std::fs::File,
    buffer: &mut [u8],
) -> Result<()> {
    input.seek(SeekFrom::Start(start))?;
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(buffer.len());
        input.read_exact(&mut buffer[..chunk])?;
        output.write_all(&buffer[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_plan::TensorRole;
    use crate::weights::BODY_NORM;
    use crate::weights::TEXT_EMBEDDING;

    #[test]
    fn qwen3_tensor_name_maps_loader_slots() {
        assert_eq!(
            qwen3_tensor_name("qwen3.embed_tokens").unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            qwen3_tensor_name("qwen3.layers.0.self_attn.q_proj.weight").unwrap(),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn materializes_qwen3_alias_safetensors_from_small_payload() {
        let tmp = tempfile::TempDir::new().unwrap();
        let source = tmp.path().join("higgs.safetensors");
        let out = tmp.path().join("qwen3.safetensors");
        let planned = vec![
            planned(
                TEXT_EMBEDDING,
                TensorRole::TextEmbedding,
                "qwen3.embed_tokens",
                [2, 2],
            ),
            planned(BODY_NORM, TensorRole::BodyNorm, "qwen3.norm", [2, 1]),
            planned(
                "body.layers.0.self_attn.q_proj.weight",
                TensorRole::LayerQProj,
                "qwen3.layers.0.self_attn.q_proj.weight",
                [2, 2],
            ),
        ];
        write_small_higgs_safetensors(&source, &planned);
        let refs: Vec<_> = planned.iter().collect();

        materialize_safetensors_alias(&source, &out, &refs).unwrap();

        let bytes = std::fs::read(out).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + header_len) % std::mem::align_of::<half::bf16>(), 0);
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(tensors.names().len(), 3);
        assert_eq!(
            tensors.tensor("model.embed_tokens.weight").unwrap().data(),
            &[0; 8]
        );
        assert_eq!(tensors.tensor("model.norm.weight").unwrap().data(), &[1; 4]);
        assert_eq!(
            tensors
                .tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .data(),
            &[2; 8]
        );
        assert!(tensors.tensor(TEXT_EMBEDDING).is_err());
    }

    fn planned(
        checkpoint_name: &str,
        role: TensorRole,
        loader_slot: &str,
        shape: [usize; 2],
    ) -> PlannedTensor {
        let elements = shape.iter().product::<usize>();
        PlannedTensor {
            checkpoint_name: checkpoint_name.to_string(),
            shard_file: "model.safetensors".to_string(),
            role,
            loader_slot: loader_slot.to_string(),
            dtype: "BF16",
            shape: shape.to_vec(),
            elements,
            bytes: elements * 2,
        }
    }

    fn write_small_higgs_safetensors(path: &Path, planned: &[PlannedTensor]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut offset = 0usize;
        for (idx, tensor) in planned.iter().enumerate() {
            header.insert(
                tensor.checkpoint_name.clone(),
                json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [offset, offset + tensor.bytes]
                }),
            );
            payload.extend(std::iter::repeat_n(idx as u8, tensor.bytes));
            offset += tensor.bytes;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&payload);
        std::fs::write(path, out).unwrap();
    }
}
