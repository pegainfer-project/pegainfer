use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;

use crate::compare::FINAL_HIDDEN_BF16;
use crate::compare::PROMPT_ATTENTION_MASK;
use crate::compare::PROMPT_INPUT_IDS;
use crate::compare::PROMPT_LENGTHS;
use crate::one_step_actual::PromptTensors;
use crate::one_step_actual::owned_bf16;
use crate::one_step_actual::owned_i64;
use crate::one_step_golden::HIDDEN_SIZE;

pub const NUM_LAYERS: usize = 36;
pub const EMBEDDING_HIDDEN_BF16: &str = "embedding.last_hidden.bf16";

#[derive(Debug, Clone, PartialEq)]
pub struct LayerHiddenDumpSummary {
    pub output_path: PathBuf,
    pub prompt_tokens: usize,
    pub layers: usize,
    pub hidden_values_per_layer: usize,
}

pub fn layer_hidden_tensor_name(layer_idx: usize) -> String {
    format!("layer.{layer_idx:02}.last_hidden.bf16")
}

pub fn write_layer_hidden_dump(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    embedding_hidden: &[bf16],
    layer_hidden: &[Vec<bf16>],
    final_normed: &[bf16],
) -> Result<LayerHiddenDumpSummary> {
    ensure!(
        embedding_hidden.len() == HIDDEN_SIZE,
        "embedding hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
        embedding_hidden.len()
    );
    ensure!(
        layer_hidden.len() == NUM_LAYERS,
        "expected {NUM_LAYERS} layer snapshots, got {}",
        layer_hidden.len()
    );
    ensure!(
        final_normed.len() == HIDDEN_SIZE,
        "final normed hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
        final_normed.len()
    );

    let mut tensors = BTreeMap::from([
        (
            PROMPT_INPUT_IDS.to_string(),
            owned_i64(
                &[1, prompt.input_ids_padded.len()],
                &prompt.input_ids_padded,
            ),
        ),
        (
            PROMPT_ATTENTION_MASK.to_string(),
            owned_i64(&[1, prompt.attention_mask.len()], &prompt.attention_mask),
        ),
        (
            PROMPT_LENGTHS.to_string(),
            owned_i64(&[prompt.lengths.len()], &prompt.lengths),
        ),
        (
            EMBEDDING_HIDDEN_BF16.to_string(),
            owned_bf16(&[1, HIDDEN_SIZE], embedding_hidden),
        ),
        (
            FINAL_HIDDEN_BF16.to_string(),
            owned_bf16(&[1, HIDDEN_SIZE], final_normed),
        ),
    ]);
    for (layer_idx, hidden) in layer_hidden.iter().enumerate() {
        ensure!(
            hidden.len() == HIDDEN_SIZE,
            "layer {layer_idx} hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
            hidden.len()
        );
        tensors.insert(
            layer_hidden_tensor_name(layer_idx),
            owned_bf16(&[1, HIDDEN_SIZE], hidden),
        );
    }

    let output_path = output_path.as_ref();
    let metadata = HashMap::from([(
        "fixture_kind".to_string(),
        "higgs-prefill-layer-hidden-actual".to_string(),
    )]);
    safetensors::serialize_to_file(tensors, Some(metadata), output_path)
        .with_context(|| format!("write {}", output_path.display()))?;

    Ok(LayerHiddenDumpSummary {
        output_path: output_path.to_path_buf(),
        prompt_tokens: prompt.prompt_ids()?.len(),
        layers: layer_hidden.len(),
        hidden_values_per_layer: HIDDEN_SIZE,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageDumpSummary {
    pub output_path: PathBuf,
    pub prompt_tokens: usize,
    pub stages: usize,
    pub values: usize,
}

pub fn write_stage_dump(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    stages: &[(String, Vec<bf16>)],
) -> Result<StageDumpSummary> {
    ensure!(!stages.is_empty(), "stage dump requires at least one stage");
    let mut tensors = BTreeMap::from([
        (
            PROMPT_INPUT_IDS.to_string(),
            owned_i64(
                &[1, prompt.input_ids_padded.len()],
                &prompt.input_ids_padded,
            ),
        ),
        (
            PROMPT_ATTENTION_MASK.to_string(),
            owned_i64(&[1, prompt.attention_mask.len()], &prompt.attention_mask),
        ),
        (
            PROMPT_LENGTHS.to_string(),
            owned_i64(&[prompt.lengths.len()], &prompt.lengths),
        ),
    ]);
    let mut values = 0usize;
    for (name, stage_values) in stages {
        ensure!(!stage_values.is_empty(), "stage {name} must not be empty");
        values += stage_values.len();
        tensors.insert(
            name.clone(),
            owned_bf16(&[1, stage_values.len()], stage_values),
        );
    }

    let output_path = output_path.as_ref();
    let metadata = HashMap::from([(
        "fixture_kind".to_string(),
        "higgs-layer0-stage-actual".to_string(),
    )]);
    safetensors::serialize_to_file(tensors, Some(metadata), output_path)
        .with_context(|| format!("write {}", output_path.display()))?;

    Ok(StageDumpSummary {
        output_path: output_path.to_path_buf(),
        prompt_tokens: prompt.prompt_ids()?.len(),
        stages: stages.len(),
        values,
    })
}
