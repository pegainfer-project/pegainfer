//! Synthetic PEFT adapter fixtures shared by the LoRA unit and integration
//! tests; integration tests reach it through the `test-fixtures` feature,
//! which the self dev-dependency enables for all test builds.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use half::bf16;
use half::f16;
use safetensors::Dtype;
use safetensors::View;

use super::ADAPTER_CONFIG_FILE;
use super::ADAPTER_WEIGHTS_FILE;
use super::tensor_name;

#[derive(Clone)]
pub struct FixtureTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl FixtureTensor {
    pub(crate) fn filled(dtype: Dtype, shape: Vec<usize>, value: f32) -> Self {
        let elems = shape.iter().product::<usize>();
        let data = match dtype {
            Dtype::BF16 => bf16::from_f32(value).to_bits().to_le_bytes().repeat(elems),
            Dtype::F16 => f16::from_f32(value).to_bits().to_le_bytes().repeat(elems),
            Dtype::F32 => value.to_le_bytes().repeat(elems),
            _ => panic!("unsupported fixture dtype {dtype:?}"),
        };
        Self { dtype, shape, data }
    }
}

impl View for FixtureTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Zero-filled bf16 `lora_A`/`lora_B` pair for one projection.
pub fn push_projection(
    tensors: &mut BTreeMap<String, FixtureTensor>,
    layer_idx: usize,
    path_segment: &str,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
) {
    tensors.insert(
        tensor_name(layer_idx, path_segment, "lora_A"),
        FixtureTensor::filled(Dtype::BF16, vec![rank, in_dim], 0.0),
    );
    tensors.insert(
        tensor_name(layer_idx, path_segment, "lora_B"),
        FixtureTensor::filled(Dtype::BF16, vec![out_dim, rank], 0.0),
    );
}

pub fn write_adapter_config(dir: &Path, rank: usize, alpha: usize, targets: &[&str]) {
    let targets = targets
        .iter()
        .map(|target| format!("\"{target}\""))
        .collect::<Vec<_>>()
        .join(", ");
    write_adapter_config_json(
        dir,
        &format!(
            r#"{{
  "peft_type": "LORA",
  "r": {rank},
  "lora_alpha": {alpha},
  "target_modules": [{targets}]
}}"#
        ),
    );
}

pub fn write_adapter_config_json(dir: &Path, json: &str) {
    fs::write(dir.join(ADAPTER_CONFIG_FILE), json).expect("write adapter_config.json");
}

pub fn write_adapter_tensors(dir: &Path, tensors: BTreeMap<String, FixtureTensor>) {
    safetensors::serialize_to_file(tensors, None, &dir.join(ADAPTER_WEIGHTS_FILE))
        .expect("write adapter_model.safetensors");
}
