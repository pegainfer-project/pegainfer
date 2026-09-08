//! Preflight validation for DFlash2 tensors.

use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

pub(crate) const HIDDEN_PROJECTION_TENSOR: &str = "candidate_selector.hidden_projection.weight";
pub(crate) const PREDECESSOR_CODEBOOK_TENSOR: &str = "candidate_selector.predecessor_codebook";
pub(crate) const SUCCESSOR_CODEBOOK_TENSOR: &str = "candidate_selector.successor_codebook";
pub(crate) const DRAFT_LM_HEAD_TENSOR: &str = "lm_head.weight";

/// Validate selector dtype and shapes before GPU upload.
pub(crate) fn validate_selector_tensors(
    shards: &[SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    rank: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<()> {
    validate_matrix(
        shards,
        weight_map,
        HIDDEN_PROJECTION_TENSOR,
        rank,
        hidden_size,
    )?;
    validate_matrix(
        shards,
        weight_map,
        PREDECESSOR_CODEBOOK_TENSOR,
        vocab_size,
        rank,
    )?;
    validate_matrix(
        shards,
        weight_map,
        SUCCESSOR_CODEBOOK_TENSOR,
        vocab_size,
        rank,
    )?;

    Ok(())
}

/// Validate the native DFlash2 output head before uploading it.
pub(crate) fn validate_native_output_head(
    shards: &[SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<()> {
    validate_matrix(
        shards,
        weight_map,
        DRAFT_LM_HEAD_TENSOR,
        vocab_size,
        hidden_size,
    )?;
    Ok(())
}

fn validate_matrix(
    shards: &[SafeTensors<'_>],
    weight_map: &HashMap<String, usize>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    let tensor = find_tensor(shards, weight_map, name)?;
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "DFlash2 tensor {name:?} must be BF16, got {:?}",
        tensor.dtype()
    );
    ensure!(
        tensor.shape() == [expected_rows, expected_cols],
        "DFlash2 tensor {name:?} has shape {:?}, expected [{expected_rows}, {expected_cols}]",
        tensor.shape()
    );
    Ok(())
}

fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<TensorView<'a>> {
    if let Some(&shard_idx) = weight_map.get(name) {
        let shard = shards.get(shard_idx).with_context(|| {
            format!("DFlash2 tensor {name:?} references missing shard index {shard_idx}")
        })?;
        return shard
            .tensor(name)
            .with_context(|| format!("load DFlash2 tensor {name:?}"));
    }

    for shard in shards {
        if let Ok(tensor) = shard.tensor(name) {
            return Ok(tensor);
        }
    }

    anyhow::bail!("DFlash2 tensor {name:?} is missing")
}
