//! CPU preflight for native DFlash2. Uses the shared safetensors reader and
//! validates the complete backbone even when only the selector will be uploaded.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use pegainfer_core::weight_loader::TensorDescriptor;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;
use pegainfer_core::weight_loader::tensor_descriptors;
use safetensors::Dtype;

use super::config::NativeDFlash2Config;

/// A validated checkpoint manifest, without a CUDA context or GPU allocations.
#[derive(Debug)]
pub struct NativeArtifact {
    pub(crate) config: NativeDFlash2Config,
    pub(crate) tensor_manifest: BTreeMap<String, TensorDescriptor>,
}

impl NativeArtifact {
    pub fn inspect(model_path: &str) -> Result<Self> {
        let config = inspect_config(model_path)?;
        let (paths, weight_map) =
            load_shard_info(model_path).context("native DFlash2 inspect: shard index")?;
        let mmaps = mmap_shards(&paths).context("native DFlash2 inspect: mapping shards")?;
        let shards = deserialize_shards(&mmaps).context("native DFlash2 inspect: shard headers")?;

        let tensor_manifest = tensor_descriptors(&shards, &weight_map)
            .context("native DFlash2 inspect: tensor index")?;
        validate_tensors(&config, &tensor_manifest)?;

        Ok(Self {
            config,
            tensor_manifest,
        })
    }

    pub fn config(&self) -> &NativeDFlash2Config {
        &self.config
    }

    pub fn tensor_manifest(&self) -> &BTreeMap<String, TensorDescriptor> {
        &self.tensor_manifest
    }

    pub fn weight_bytes(&self) -> Result<usize> {
        self.config.weight_bytes()
    }

    pub fn selector_weight_bytes(&self) -> Result<usize> {
        self.config.selector_weight_bytes()
    }
}

pub(crate) fn inspect_config(model_path: &str) -> Result<NativeDFlash2Config> {
    let config = NativeDFlash2Config::from_file(model_path)?;
    let root = Path::new(model_path);

    ensure!(
        !(root.join("model.safetensors").exists()
            && root.join("model.safetensors.index.json").exists()),
        "native DFlash2 inspect: ambiguous checkpoint has both model.safetensors and a shard index"
    );

    for file in ["mask_embedding.pt", "d2t.pt", "t2d.pt"] {
        ensure!(
            !root.join(file).exists(),
            "native DFlash2 inspect: unsupported separate weight source {file}"
        );
    }

    Ok(config)
}

pub(crate) fn validate_tensors(
    config: &NativeDFlash2Config,
    descriptors: &BTreeMap<String, TensorDescriptor>,
) -> Result<()> {
    let count = config
        .num_hidden_layers
        .checked_mul(15)
        .and_then(|n| n.checked_add(6))
        .context("native DFlash2 inspect: tensor count overflow")?;

    // Bound expansion by the actual, validated safetensors header. A corrupt
    // billion-layer config must not allocate a billion expected tensor names.
    ensure!(
        count <= descriptors.len().saturating_add(1),
        "native DFlash2 inspect: incomplete checkpoint, expected {count} tensors, got {}",
        descriptors.len()
    );

    let expected = config.expected_tensors()?;
    for (name, shape) in &expected {
        let descriptor = descriptors.get(name).with_context(|| {
            format!("native DFlash2 inspect: missing tensor '{name}' (expected BF16 {shape:?})")
        })?;
        ensure!(
            descriptor.dtype == Dtype::BF16,
            "native DFlash2 inspect tensor '{name}': expected BF16, got {:?}",
            descriptor.dtype
        );
        ensure!(
            &descriptor.shape == shape,
            "native DFlash2 inspect tensor '{name}': expected shape {shape:?}, got {:?}",
            descriptor.shape
        );

        let bytes = shape
            .iter()
            .try_fold(2usize, |size, &dim| size.checked_mul(dim))
            .context("native DFlash2 inspect: tensor size overflow")?;
        ensure!(
            descriptor.byte_len == bytes,
            "native DFlash2 inspect tensor '{name}': expected {bytes} payload bytes, got {}",
            descriptor.byte_len
        );
    }

    for name in descriptors.keys() {
        ensure!(
            expected.contains_key(name),
            "native DFlash2 inspect: unsupported tensor '{name}'; expected the full-vocabulary profile with target-owned embedding/lm_head"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests;
