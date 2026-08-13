use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use safetensors::Dtype;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

pub const FIXTURE_KIND: &str = "higgs-one-step-audio-logits-golden";
pub const MODEL_ID: &str = "bosonai/higgs-tts-3-4b";
pub const MODEL_REVISION: &str = "7556c17e05201fccd9c8cc120bc216dcc7b5d561";
pub const NUM_CODEBOOKS: usize = 8;
pub const CODEBOOK_VOCAB_SIZE: usize = 1026;
pub const HIDDEN_SIZE: usize = 2560;
pub const TOP_K: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub dtype: Dtype,
    pub shape: &'static [usize],
}

pub const REQUIRED_TENSORS: &[TensorSpec] = &[
    TensorSpec {
        name: "prompt.input_ids_padded",
        dtype: Dtype::I64,
        shape: &[1, 10],
    },
    TensorSpec {
        name: "prompt.attention_mask",
        dtype: Dtype::I64,
        shape: &[1, 10],
    },
    TensorSpec {
        name: "prompt.lengths",
        dtype: Dtype::I64,
        shape: &[1],
    },
    TensorSpec {
        name: "final_hidden.bf16",
        dtype: Dtype::BF16,
        shape: &[1, HIDDEN_SIZE],
    },
    TensorSpec {
        name: "audio_logits.f32",
        dtype: Dtype::F32,
        shape: &[1, NUM_CODEBOOKS, CODEBOOK_VOCAB_SIZE],
    },
    TensorSpec {
        name: "audio_top64.ids",
        dtype: Dtype::I64,
        shape: &[1, NUM_CODEBOOKS, TOP_K],
    },
    TensorSpec {
        name: "audio_top64.logprobs.f32",
        dtype: Dtype::F32,
        shape: &[1, NUM_CODEBOOKS, TOP_K],
    },
    TensorSpec {
        name: "audio_argmax.ids",
        dtype: Dtype::I64,
        shape: &[1, NUM_CODEBOOKS],
    },
];

#[derive(Debug, Clone)]
pub struct GoldenContract {
    pub metadata: HashMap<String, String>,
    pub sha256: String,
    pub bytes: usize,
}

pub fn load_and_validate(path: impl AsRef<Path>) -> Result<GoldenContract> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let sha256 = sha256_hex(&bytes);
    let metadata = safetensors_metadata(&bytes)?;
    validate_metadata(&metadata)?;
    let st = SafeTensors::deserialize(&bytes).context("parse Higgs golden safetensors")?;
    validate_required_tensors(&st, "golden")?;
    Ok(GoldenContract {
        metadata,
        sha256,
        bytes: bytes.len(),
    })
}

fn validate_metadata(metadata: &HashMap<String, String>) -> Result<()> {
    require_metadata(metadata, "fixture_kind", FIXTURE_KIND)?;
    require_metadata(metadata, "model_id", MODEL_ID)?;
    require_metadata(metadata, "model_revision", MODEL_REVISION)?;
    require_metadata(metadata, "schema_version", "1")?;
    require_metadata(metadata, "num_codebooks", &NUM_CODEBOOKS.to_string())?;
    require_metadata(
        metadata,
        "codebook_vocab_size",
        &CODEBOOK_VOCAB_SIZE.to_string(),
    )?;
    require_metadata(metadata, "hidden_size", &HIDDEN_SIZE.to_string())?;
    let reference = metadata
        .get("reference")
        .context("golden metadata missing reference")?;
    if !reference.contains("SGLang-Omni Higgs prompt builder") {
        bail!("golden reference must record SGLang-Omni Higgs prompt semantics");
    }
    let files = metadata
        .get("sglang_omni_reference_files")
        .context("golden metadata missing sglang_omni_reference_files")?;
    for expected in [
        "sglang_omni/models/higgs_tts/text_tokenizer.py",
        "sglang_omni/models/higgs_tts/modeling.py",
        "sglang_omni/models/higgs_tts/model.py",
    ] {
        if !files.contains(expected) {
            bail!("golden metadata reference files missing {expected}");
        }
    }
    Ok(())
}

pub fn validate_required_tensors(st: &SafeTensors, label: &str) -> Result<()> {
    for spec in REQUIRED_TENSORS {
        let tensor = st
            .tensor(spec.name)
            .with_context(|| format!("{label} missing tensor {}", spec.name))?;
        if tensor.dtype() != spec.dtype {
            bail!(
                "{label} tensor {} dtype mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.dtype,
                tensor.dtype()
            );
        }
        if tensor.shape() != spec.shape {
            bail!(
                "{label} tensor {} shape mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.shape,
                tensor.shape()
            );
        }
    }
    Ok(())
}

fn require_metadata(metadata: &HashMap<String, String>, key: &str, expected: &str) -> Result<()> {
    let actual = metadata
        .get(key)
        .with_context(|| format!("golden metadata missing {key}"))?;
    if actual != expected {
        bail!("golden metadata {key} mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn safetensors_metadata(bytes: &[u8]) -> Result<HashMap<String, String>> {
    let header_len_bytes: [u8; 8] = bytes
        .get(..8)
        .context("safetensors file missing 8-byte header length")?
        .try_into()
        .expect("slice length checked");
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let header = bytes
        .get(8..8 + header_len)
        .context("safetensors file missing JSON header")?;
    let value: serde_json::Value =
        serde_json::from_slice(header).context("parse safetensors JSON header")?;
    Ok(value
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .map(|metadata| {
            metadata
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/higgs-one-step-audio-logits.safetensors"
    );

    #[test]
    fn committed_higgs_one_step_golden_has_expected_contract() {
        let contract = load_and_validate(GOLDEN).expect("validate committed Higgs golden");
        assert_eq!(
            contract.sha256,
            "bbaae8018759b7e8f26d2acfb1aefb5bee3e5099d47573bb4ce3c980b6096684"
        );
        assert_eq!(contract.bytes, 46_064);
    }
}
