//! The tensor contract, ported from `pegainfer-gemma4/src/manifest/`: [`schema`]
//! derives it from a config, and [`validate`] holds a checkpoint's shard headers
//! against it.

use safetensors::Dtype;

pub(crate) mod schema;
#[cfg(test)]
pub(crate) mod validate;

/// Every shard header of the frozen revision, as
/// `tools/accuracy/dump_qwen4_frozen_tensors.py` writes them, plus the two n-gram
/// partition vectors the checkpoint stores.
#[cfg(test)]
#[derive(serde::Deserialize)]
pub(crate) struct FrozenTensors {
    pub(crate) tensors: std::collections::BTreeMap<String, (Dtype, Vec<usize>)>,
    pub(crate) values: std::collections::BTreeMap<String, Vec<usize>>,
}

#[cfg(test)]
pub(crate) fn frozen_tensors() -> FrozenTensors {
    serde_json::from_str(include_str!("../../tests/frozen_tensors.json")).unwrap()
}

/// Every weight tensor is BF16, the GDN scalars included.
pub(crate) const EXPECTED_DTYPE: Dtype = Dtype::BF16;

/// The n-gram partition vectors are stored as integers.
pub(crate) const INTEGER_INDEX_DTYPE: Dtype = Dtype::I64;
