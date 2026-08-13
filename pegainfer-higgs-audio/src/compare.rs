use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;
use crate::one_step_golden::TOP_K;

pub const PROMPT_INPUT_IDS: &str = "prompt.input_ids_padded";
pub const PROMPT_ATTENTION_MASK: &str = "prompt.attention_mask";
pub const PROMPT_LENGTHS: &str = "prompt.lengths";
pub const FINAL_HIDDEN_BF16: &str = "final_hidden.bf16";
pub const AUDIO_LOGITS_F32: &str = "audio_logits.f32";
pub const AUDIO_TOP64_IDS: &str = "audio_top64.ids";
pub const AUDIO_TOP64_LOGPROBS_F32: &str = "audio_top64.logprobs.f32";
pub const AUDIO_ARGMAX_IDS: &str = "audio_argmax.ids";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneStepTolerances {
    pub hidden_abs_tol: f32,
    pub hidden_mean_abs_tol: f32,
    pub logits_abs_tol: f32,
    pub logits_mean_abs_tol: f32,
    pub top_logprobs_abs_tol: f32,
    pub top_logprobs_mean_abs_tol: f32,
}

impl Default for OneStepTolerances {
    fn default() -> Self {
        Self {
            hidden_abs_tol: 0.03125,
            hidden_mean_abs_tol: 0.003,
            logits_abs_tol: 0.05,
            logits_mean_abs_tol: 0.005,
            top_logprobs_abs_tol: 0.05,
            top_logprobs_mean_abs_tol: 0.005,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneStepSemanticTolerances {
    pub hidden_cosine_min: f32,
    pub logits_cosine_min: f32,
    pub argmax_regret_tol: f32,
    pub top64_min_overlap: usize,
}

impl Default for OneStepSemanticTolerances {
    fn default() -> Self {
        Self {
            hidden_cosine_min: 0.9998,
            logits_cosine_min: 0.99999,
            argmax_regret_tol: 0.20,
            top64_min_overlap: 40,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorComparison {
    pub name: &'static str,
    pub elements: usize,
    pub exact_mismatches: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rmse: f32,
    pub p99_abs: f32,
    pub abs_tol: f32,
    pub mean_abs_tol: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneStepComparison {
    pub tensors: Vec<TensorComparison>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneStepSemanticComparison {
    pub prompt_exact: bool,
    pub audio_argmax_exact: bool,
    pub hidden_cosine: f32,
    pub logits_cosine: f32,
    pub max_argmax_regret: f32,
    pub top64_min_overlap: usize,
    pub top64_mean_overlap: f32,
    pub tolerances: OneStepSemanticTolerances,
}

impl OneStepComparison {
    pub fn passed(&self) -> bool {
        self.tensors.iter().all(|tensor| tensor.passed)
    }
}

impl OneStepSemanticComparison {
    pub fn passed(&self) -> bool {
        self.prompt_exact
            && self.audio_argmax_exact
            && self.hidden_cosine >= self.tolerances.hidden_cosine_min
            && self.logits_cosine >= self.tolerances.logits_cosine_min
            && self.max_argmax_regret <= self.tolerances.argmax_regret_tol
            && self.top64_min_overlap >= self.tolerances.top64_min_overlap
    }
}

pub fn compare_one_step_files(
    golden: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerances: OneStepTolerances,
) -> Result<OneStepComparison> {
    let golden_path = golden.as_ref();
    let actual_path = actual.as_ref();
    let golden_bytes =
        std::fs::read(golden_path).with_context(|| format!("read {}", golden_path.display()))?;
    let actual_bytes =
        std::fs::read(actual_path).with_context(|| format!("read {}", actual_path.display()))?;
    let golden_st =
        SafeTensors::deserialize(&golden_bytes).context("parse Higgs golden safetensors")?;
    let actual_st =
        SafeTensors::deserialize(&actual_bytes).context("parse Higgs actual safetensors")?;
    compare_one_step_safetensors(&golden_st, &actual_st, tolerances)
}

pub fn compare_one_step_semantic_files(
    golden: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerances: OneStepSemanticTolerances,
) -> Result<OneStepSemanticComparison> {
    let golden_path = golden.as_ref();
    let actual_path = actual.as_ref();
    let golden_bytes =
        std::fs::read(golden_path).with_context(|| format!("read {}", golden_path.display()))?;
    let actual_bytes =
        std::fs::read(actual_path).with_context(|| format!("read {}", actual_path.display()))?;
    let golden_st =
        SafeTensors::deserialize(&golden_bytes).context("parse Higgs golden safetensors")?;
    let actual_st =
        SafeTensors::deserialize(&actual_bytes).context("parse Higgs actual safetensors")?;
    compare_one_step_semantic_safetensors(&golden_st, &actual_st, tolerances)
}

pub fn compare_one_step_safetensors(
    golden: &SafeTensors,
    actual: &SafeTensors,
    tolerances: OneStepTolerances,
) -> Result<OneStepComparison> {
    validate_comparison_schema(golden, actual)?;

    let tensors = vec![
        compare_i64_exact(golden, actual, PROMPT_INPUT_IDS)?,
        compare_i64_exact(golden, actual, PROMPT_ATTENTION_MASK)?,
        compare_i64_exact(golden, actual, PROMPT_LENGTHS)?,
        compare_bf16(
            golden,
            actual,
            FINAL_HIDDEN_BF16,
            tolerances.hidden_abs_tol,
            tolerances.hidden_mean_abs_tol,
        )?,
        compare_f32(
            golden,
            actual,
            AUDIO_LOGITS_F32,
            tolerances.logits_abs_tol,
            tolerances.logits_mean_abs_tol,
        )?,
        compare_i64_exact(golden, actual, AUDIO_TOP64_IDS)?,
        compare_f32(
            golden,
            actual,
            AUDIO_TOP64_LOGPROBS_F32,
            tolerances.top_logprobs_abs_tol,
            tolerances.top_logprobs_mean_abs_tol,
        )?,
        compare_i64_exact(golden, actual, AUDIO_ARGMAX_IDS)?,
    ];

    Ok(OneStepComparison { tensors })
}

pub fn compare_one_step_semantic_safetensors(
    golden: &SafeTensors,
    actual: &SafeTensors,
    tolerances: OneStepSemanticTolerances,
) -> Result<OneStepSemanticComparison> {
    validate_comparison_schema(golden, actual)?;

    let prompt_exact = i64_values(tensor(golden, PROMPT_INPUT_IDS)?)?
        == i64_values(tensor(actual, PROMPT_INPUT_IDS)?)?
        && i64_values(tensor(golden, PROMPT_ATTENTION_MASK)?)?
            == i64_values(tensor(actual, PROMPT_ATTENTION_MASK)?)?
        && i64_values(tensor(golden, PROMPT_LENGTHS)?)?
            == i64_values(tensor(actual, PROMPT_LENGTHS)?)?;
    let golden_argmax = i64_values(tensor(golden, AUDIO_ARGMAX_IDS)?)?;
    let actual_argmax = i64_values(tensor(actual, AUDIO_ARGMAX_IDS)?)?;
    let audio_argmax_exact = golden_argmax == actual_argmax;

    let golden_hidden = bf16_values(tensor(golden, FINAL_HIDDEN_BF16)?)?;
    let actual_hidden = bf16_values(tensor(actual, FINAL_HIDDEN_BF16)?)?;
    let hidden_cosine = cosine_similarity(&golden_hidden, &actual_hidden)?;

    let golden_logits = f32_values(tensor(golden, AUDIO_LOGITS_F32)?)?;
    let actual_logits = f32_values(tensor(actual, AUDIO_LOGITS_F32)?)?;
    let logits_cosine = cosine_similarity(&golden_logits, &actual_logits)?;

    let max_argmax_regret = max_argmax_regret(&golden_logits, &actual_argmax)?;
    let (top64_min_overlap, top64_mean_overlap) =
        topk_overlap_from_logits(&golden_logits, &actual_logits, actual_argmax.len(), 64)?;

    Ok(OneStepSemanticComparison {
        prompt_exact,
        audio_argmax_exact,
        hidden_cosine,
        logits_cosine,
        max_argmax_regret,
        top64_min_overlap,
        top64_mean_overlap,
        tolerances,
    })
}

pub fn ensure_comparison_passed(comparison: &OneStepComparison) -> Result<()> {
    if comparison.passed() {
        return Ok(());
    }
    let failing: Vec<_> = comparison
        .tensors
        .iter()
        .filter(|tensor| !tensor.passed)
        .map(|tensor| tensor.name)
        .collect();
    bail!("Higgs one-step comparison failed for tensor(s): {failing:?}");
}

pub fn ensure_semantic_comparison_passed(comparison: &OneStepSemanticComparison) -> Result<()> {
    if comparison.passed() {
        return Ok(());
    }
    bail!(
        "Higgs one-step semantic comparison failed: prompt_exact={} argmax_exact={} hidden_cosine={:.9} logits_cosine={:.9} max_argmax_regret={:.6} top64_min_overlap={} top64_mean_overlap={:.2}",
        comparison.prompt_exact,
        comparison.audio_argmax_exact,
        comparison.hidden_cosine,
        comparison.logits_cosine,
        comparison.max_argmax_regret,
        comparison.top64_min_overlap,
        comparison.top64_mean_overlap
    );
}

fn compare_i64_exact(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::I64, "{name} golden must be I64");
    ensure!(actual.dtype() == Dtype::I64, "{name} actual must be I64");
    let golden_values = i64_values(golden)?;
    let actual_values = i64_values(actual)?;
    ensure!(
        golden_values.len() == actual_values.len(),
        "{name} element count mismatch: golden {} actual {}",
        golden_values.len(),
        actual_values.len()
    );

    let diffs: Vec<f32> = golden_values
        .iter()
        .zip(&actual_values)
        .map(|(golden, actual)| (*golden - *actual).unsigned_abs() as f32)
        .collect();
    let exact_mismatches = diffs.iter().filter(|diff| **diff != 0.0).count();
    let stats = stats(&diffs);
    Ok(TensorComparison {
        name,
        elements: diffs.len(),
        exact_mismatches,
        max_abs: stats.max_abs,
        mean_abs: stats.mean_abs,
        rmse: stats.rmse,
        p99_abs: stats.p99_abs,
        abs_tol: 0.0,
        mean_abs_tol: 0.0,
        passed: exact_mismatches == 0,
    })
}

fn compare_f32(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::F32, "{name} golden must be F32");
    ensure!(actual.dtype() == Dtype::F32, "{name} actual must be F32");
    compare_float_values(
        name,
        &f32_values(golden)?,
        &f32_values(actual)?,
        abs_tol,
        mean_abs_tol,
    )
}

fn validate_comparison_schema(golden: &SafeTensors, actual: &SafeTensors) -> Result<()> {
    let prompt_shape = require_matching_tensor(golden, actual, PROMPT_INPUT_IDS, Dtype::I64)?;
    ensure!(
        prompt_shape.len() == 2,
        "{PROMPT_INPUT_IDS} must be rank-2 [batch, seq], got {prompt_shape:?}"
    );
    let batch = prompt_shape[0];
    let seq = prompt_shape[1];
    ensure!(batch > 0, "{PROMPT_INPUT_IDS} batch must be non-zero");
    ensure!(seq > 0, "{PROMPT_INPUT_IDS} seq must be non-zero");

    let attention_shape =
        require_matching_tensor(golden, actual, PROMPT_ATTENTION_MASK, Dtype::I64)?;
    ensure!(
        attention_shape == prompt_shape,
        "{PROMPT_ATTENTION_MASK} shape {attention_shape:?} must match {PROMPT_INPUT_IDS} shape {prompt_shape:?}"
    );

    let lengths_shape = require_matching_tensor(golden, actual, PROMPT_LENGTHS, Dtype::I64)?;
    ensure!(
        lengths_shape == [batch],
        "{PROMPT_LENGTHS} shape {lengths_shape:?} must be [batch={batch}]"
    );
    validate_prompt_surface(golden, "golden", batch, seq)?;
    validate_prompt_surface(actual, "actual", batch, seq)?;

    let hidden_shape = require_matching_tensor(golden, actual, FINAL_HIDDEN_BF16, Dtype::BF16)?;
    ensure!(
        hidden_shape == [batch, HIDDEN_SIZE],
        "{FINAL_HIDDEN_BF16} shape {hidden_shape:?} must be [batch={batch}, hidden={HIDDEN_SIZE}]"
    );

    let logits_shape = require_matching_tensor(golden, actual, AUDIO_LOGITS_F32, Dtype::F32)?;
    ensure!(
        logits_shape == [batch, NUM_CODEBOOKS, CODEBOOK_VOCAB_SIZE],
        "{AUDIO_LOGITS_F32} shape {logits_shape:?} must be [batch={batch}, codebooks={NUM_CODEBOOKS}, vocab={CODEBOOK_VOCAB_SIZE}]"
    );

    let top_ids_shape = require_matching_tensor(golden, actual, AUDIO_TOP64_IDS, Dtype::I64)?;
    ensure!(
        top_ids_shape == [batch, NUM_CODEBOOKS, TOP_K],
        "{AUDIO_TOP64_IDS} shape {top_ids_shape:?} must be [batch={batch}, codebooks={NUM_CODEBOOKS}, top_k={TOP_K}]"
    );

    let top_logprobs_shape =
        require_matching_tensor(golden, actual, AUDIO_TOP64_LOGPROBS_F32, Dtype::F32)?;
    ensure!(
        top_logprobs_shape == [batch, NUM_CODEBOOKS, TOP_K],
        "{AUDIO_TOP64_LOGPROBS_F32} shape {top_logprobs_shape:?} must be [batch={batch}, codebooks={NUM_CODEBOOKS}, top_k={TOP_K}]"
    );

    let argmax_shape = require_matching_tensor(golden, actual, AUDIO_ARGMAX_IDS, Dtype::I64)?;
    ensure!(
        argmax_shape == [batch, NUM_CODEBOOKS],
        "{AUDIO_ARGMAX_IDS} shape {argmax_shape:?} must be [batch={batch}, codebooks={NUM_CODEBOOKS}]"
    );

    Ok(())
}

fn require_matching_tensor(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
    dtype: Dtype,
) -> Result<Vec<usize>> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(
        golden.dtype() == dtype,
        "{name} golden dtype mismatch: expected {:?}, got {:?}",
        dtype,
        golden.dtype()
    );
    ensure!(
        actual.dtype() == dtype,
        "{name} actual dtype mismatch: expected {:?}, got {:?}",
        dtype,
        actual.dtype()
    );
    ensure!(
        golden.shape() == actual.shape(),
        "{name} shape mismatch: golden {:?} actual {:?}",
        golden.shape(),
        actual.shape()
    );
    Ok(golden.shape().to_vec())
}

fn validate_prompt_surface(st: &SafeTensors, label: &str, batch: usize, seq: usize) -> Result<()> {
    let lengths = i64_values(tensor(st, PROMPT_LENGTHS)?)?;
    ensure!(
        lengths.len() == batch,
        "{label} prompt lengths element count {} must equal batch {batch}",
        lengths.len()
    );
    let attention = i64_values(tensor(st, PROMPT_ATTENTION_MASK)?)?;
    ensure!(
        attention.len() == batch * seq,
        "{label} attention mask element count {} must equal batch*seq {}",
        attention.len(),
        batch * seq
    );

    for (row_idx, length) in lengths.iter().enumerate() {
        ensure!(
            *length > 0,
            "{label} prompt length at row {row_idx} must be positive, got {length}"
        );
        ensure!(
            (*length as usize) <= seq,
            "{label} prompt length at row {row_idx} exceeds seq {seq}: {length}"
        );
        let row = &attention[row_idx * seq..(row_idx + 1) * seq];
        let mut mask_sum = 0i64;
        for (col_idx, value) in row.iter().enumerate() {
            ensure!(
                *value == 0 || *value == 1,
                "{label} attention mask at row {row_idx} col {col_idx} must be 0/1, got {value}"
            );
            mask_sum += *value;
        }
        ensure!(
            mask_sum == *length,
            "{label} attention mask sum at row {row_idx} must equal prompt length {length}, got {mask_sum}"
        );
    }

    Ok(())
}

fn compare_bf16(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::BF16, "{name} golden must be BF16");
    ensure!(actual.dtype() == Dtype::BF16, "{name} actual must be BF16");
    compare_float_values(
        name,
        &bf16_values(golden)?,
        &bf16_values(actual)?,
        abs_tol,
        mean_abs_tol,
    )
}

fn compare_float_values(
    name: &'static str,
    golden: &[f32],
    actual: &[f32],
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    ensure!(
        golden.len() == actual.len(),
        "{name} element count mismatch: golden {} actual {}",
        golden.len(),
        actual.len()
    );
    let mut non_finite = 0usize;
    let diffs: Vec<f32> = golden
        .iter()
        .zip(actual)
        .map(|(golden, actual)| {
            let diff = (*golden - *actual).abs();
            if diff.is_finite() {
                diff
            } else {
                non_finite += 1;
                f32::INFINITY
            }
        })
        .collect();
    let stats = stats(&diffs);
    Ok(TensorComparison {
        name,
        elements: diffs.len(),
        exact_mismatches: non_finite,
        max_abs: stats.max_abs,
        mean_abs: stats.mean_abs,
        rmse: stats.rmse,
        p99_abs: stats.p99_abs,
        abs_tol,
        mean_abs_tol,
        passed: non_finite == 0 && stats.max_abs <= abs_tol && stats.mean_abs <= mean_abs_tol,
    })
}

#[derive(Debug, Clone, Copy)]
struct FloatStats {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    p99_abs: f32,
}

fn stats(diffs: &[f32]) -> FloatStats {
    if diffs.is_empty() {
        return FloatStats {
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
            p99_abs: 0.0,
        };
    }
    let mut sorted = diffs.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = diffs.iter().map(|value| f64::from(*value)).sum();
    let sum_sq: f64 = diffs
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum();
    let p99_idx = ((sorted.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    FloatStats {
        max_abs: *sorted.last().expect("non-empty"),
        mean_abs: (sum / diffs.len() as f64) as f32,
        rmse: (sum_sq / diffs.len() as f64).sqrt() as f32,
        p99_abs: sorted[p99_idx],
    }
}

fn cosine_similarity(golden: &[f32], actual: &[f32]) -> Result<f32> {
    ensure!(
        golden.len() == actual.len(),
        "cosine element count mismatch: golden {} actual {}",
        golden.len(),
        actual.len()
    );
    ensure!(!golden.is_empty(), "cosine requires at least one element");

    let mut dot = 0.0f64;
    let mut golden_norm_sq = 0.0f64;
    let mut actual_norm_sq = 0.0f64;
    for (golden, actual) in golden.iter().zip(actual) {
        ensure!(
            golden.is_finite() && actual.is_finite(),
            "cosine inputs must be finite"
        );
        let golden = f64::from(*golden);
        let actual = f64::from(*actual);
        dot += golden * actual;
        golden_norm_sq += golden * golden;
        actual_norm_sq += actual * actual;
    }
    ensure!(
        golden_norm_sq > 0.0 && actual_norm_sq > 0.0,
        "cosine inputs must have non-zero norm"
    );
    Ok((dot / (golden_norm_sq.sqrt() * actual_norm_sq.sqrt())) as f32)
}

fn max_argmax_regret(golden_logits: &[f32], actual_argmax: &[i64]) -> Result<f32> {
    let rows = actual_argmax.len();
    ensure!(rows > 0, "argmax regret requires at least one row");
    ensure!(
        golden_logits.len().is_multiple_of(rows),
        "golden logits length {} is not divisible by argmax rows {}",
        golden_logits.len(),
        rows
    );
    let vocab = golden_logits.len() / rows;
    ensure!(vocab > 0, "argmax regret requires non-empty vocab");

    let mut max_regret = 0.0f32;
    for (row_idx, actual_id) in actual_argmax.iter().enumerate() {
        ensure!(
            *actual_id >= 0 && (*actual_id as usize) < vocab,
            "actual argmax id {} out of range for vocab {} at row {}",
            actual_id,
            vocab,
            row_idx
        );
        let row = &golden_logits[row_idx * vocab..(row_idx + 1) * vocab];
        let golden_best = row
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .context("argmax regret row is empty")?;
        let actual_score = row[*actual_id as usize];
        let regret = golden_best - actual_score;
        if regret > max_regret {
            max_regret = regret;
        }
    }
    Ok(max_regret)
}

fn topk_overlap_from_logits(
    golden_logits: &[f32],
    actual_logits: &[f32],
    rows: usize,
    k: usize,
) -> Result<(usize, f32)> {
    ensure!(rows > 0, "top-k overlap requires at least one row");
    ensure!(k > 0, "top-k overlap requires k > 0");
    ensure!(
        golden_logits.len() == actual_logits.len(),
        "top-k overlap element count mismatch: golden {} actual {}",
        golden_logits.len(),
        actual_logits.len()
    );
    ensure!(
        golden_logits.len().is_multiple_of(rows),
        "top-k overlap logits length {} is not divisible by rows {}",
        golden_logits.len(),
        rows
    );
    let vocab = golden_logits.len() / rows;
    ensure!(
        k <= vocab,
        "top-k overlap k {} exceeds per-codebook vocab {}",
        k,
        vocab
    );

    let mut min_overlap = usize::MAX;
    let mut overlap_sum = 0usize;
    for row_idx in 0..rows {
        let start = row_idx * vocab;
        let end = start + vocab;
        let golden_top = topk_indices(&golden_logits[start..end], k)?;
        let actual_top = topk_indices(&actual_logits[start..end], k)?;
        let overlap = actual_top
            .iter()
            .filter(|idx| golden_top.contains(idx))
            .count();
        min_overlap = min_overlap.min(overlap);
        overlap_sum += overlap;
    }

    Ok((min_overlap, overlap_sum as f32 / rows as f32))
}

fn topk_indices(values: &[f32], k: usize) -> Result<Vec<usize>> {
    ensure!(
        k <= values.len(),
        "top-k k {} exceeds row length {}",
        k,
        values.len()
    );
    let mut indexed: Vec<_> = values.iter().copied().enumerate().collect();
    for (idx, value) in &indexed {
        ensure!(
            value.is_finite(),
            "top-k value at index {idx} must be finite"
        );
    }
    indexed.sort_by(|(left_idx, left), (right_idx, right)| {
        right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
    });
    indexed.truncate(k);
    Ok(indexed.into_iter().map(|(idx, _)| idx).collect())
}

fn tensor<'a>(st: &'a SafeTensors, name: &str) -> Result<TensorView<'a>> {
    st.tensor(name)
        .with_context(|| format!("safetensors missing tensor {name}"))
}

fn i64_values(tensor: TensorView<'_>) -> Result<Vec<i64>> {
    bytes_to_chunks(tensor.data(), 8, "i64")?;
    Ok(tensor
        .data()
        .chunks_exact(8)
        .map(|bytes| {
            i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect())
}

fn f32_values(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    bytes_to_chunks(tensor.data(), 4, "f32")?;
    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn bf16_values(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    bytes_to_chunks(tensor.data(), 2, "bf16")?;
    Ok(tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect())
}

fn bytes_to_chunks(bytes: &[u8], chunk: usize, dtype: &str) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(chunk),
        "{dtype} tensor byte length {} is not divisible by {chunk}",
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::BTreeMap;

    use safetensors::tensor::View;

    use super::*;

    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/higgs-one-step-audio-logits.safetensors"
    );

    #[test]
    fn golden_compares_equal_to_itself() {
        let comparison =
            compare_one_step_files(GOLDEN, GOLDEN, OneStepTolerances::default()).unwrap();
        assert!(comparison.passed());
        assert_eq!(comparison.tensors.len(), 8);
        for tensor in comparison.tensors {
            assert_eq!(tensor.max_abs, 0.0, "{}", tensor.name);
            assert_eq!(tensor.mean_abs, 0.0, "{}", tensor.name);
        }
    }

    #[test]
    fn semantic_comparator_accepts_golden_self_comparison() {
        let comparison =
            compare_one_step_semantic_files(GOLDEN, GOLDEN, OneStepSemanticTolerances::default())
                .unwrap();
        assert!(comparison.passed());
        assert!(comparison.prompt_exact);
        assert!(comparison.audio_argmax_exact);
        assert_eq!(comparison.hidden_cosine, 1.0);
        assert_eq!(comparison.logits_cosine, 1.0);
        assert_eq!(comparison.max_argmax_regret, 0.0);
        assert_eq!(comparison.top64_min_overlap, 64);
    }

    #[test]
    fn comparison_accepts_dynamic_batch_schema() {
        let bytes = dynamic_one_step_bytes(2, 3);
        let st = SafeTensors::deserialize(&bytes).unwrap();

        let strict = compare_one_step_safetensors(&st, &st, OneStepTolerances::default()).unwrap();
        assert!(strict.passed());

        let semantic =
            compare_one_step_semantic_safetensors(&st, &st, OneStepSemanticTolerances::default())
                .unwrap();
        assert!(semantic.passed());
        assert_eq!(semantic.top64_min_overlap, TOP_K);
    }

    #[test]
    fn comparison_rejects_prompt_length_out_of_range() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_length = tensor_data_offset(&bytes, PROMPT_LENGTHS);
        bytes[first_length..first_length + 8].copy_from_slice(&11i64.to_le_bytes());
        std::fs::write(tmp.path(), bytes).unwrap();

        let err = compare_one_step_files(GOLDEN, tmp.path(), OneStepTolerances::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("actual prompt length at row 0 exceeds seq 10"));
    }

    #[test]
    fn comparison_rejects_attention_mask_length_mismatch() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_mask = tensor_data_offset(&bytes, PROMPT_ATTENTION_MASK);
        bytes[first_mask..first_mask + 8].copy_from_slice(&0i64.to_le_bytes());
        std::fs::write(tmp.path(), bytes).unwrap();

        let err = compare_one_step_files(GOLDEN, tmp.path(), OneStepTolerances::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("actual attention mask sum at row 0"));
    }

    #[test]
    fn semantic_comparator_rejects_argmax_mismatch() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_argmax = tensor_data_offset(&bytes, AUDIO_ARGMAX_IDS);
        let original =
            i64::from_le_bytes(bytes[first_argmax..first_argmax + 8].try_into().unwrap());
        bytes[first_argmax..first_argmax + 8].copy_from_slice(&(original + 1).to_le_bytes());
        std::fs::write(tmp.path(), bytes).unwrap();

        let comparison = compare_one_step_semantic_files(
            GOLDEN,
            tmp.path(),
            OneStepSemanticTolerances::default(),
        )
        .unwrap();
        assert!(!comparison.audio_argmax_exact);
        assert!(!comparison.passed());
        let err = ensure_semantic_comparison_passed(&comparison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("argmax_exact=false"));
    }

    #[test]
    fn semantic_comparator_rejects_prompt_mismatch() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_prompt_id = tensor_data_offset(&bytes, PROMPT_INPUT_IDS);
        let original = i64::from_le_bytes(
            bytes[first_prompt_id..first_prompt_id + 8]
                .try_into()
                .unwrap(),
        );
        bytes[first_prompt_id..first_prompt_id + 8].copy_from_slice(&(original + 1).to_le_bytes());
        std::fs::write(tmp.path(), bytes).unwrap();

        let comparison = compare_one_step_semantic_files(
            GOLDEN,
            tmp.path(),
            OneStepSemanticTolerances::default(),
        )
        .unwrap();
        assert!(!comparison.prompt_exact);
        assert!(comparison.audio_argmax_exact);
        assert!(!comparison.passed());
        let err = ensure_semantic_comparison_passed(&comparison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prompt_exact=false"));
        assert!(err.contains("top64_mean_overlap="));
    }

    #[test]
    fn topk_overlap_reports_min_and_mean_by_codebook() {
        let mut golden = Vec::new();
        let mut actual = Vec::new();
        for row in 0..8 {
            for col in 0..8 {
                golden.push((8 - col) as f32);
                actual.push(if row == 0 {
                    col as f32
                } else {
                    (8 - col) as f32
                });
            }
        }
        let (min_overlap, mean_overlap) = topk_overlap_from_logits(&golden, &actual, 8, 4).unwrap();
        assert_eq!(min_overlap, 0);
        assert_eq!(mean_overlap, 3.5);
    }

    #[test]
    fn topk_overlap_uses_dynamic_row_count() {
        let mut golden = Vec::new();
        let mut actual = Vec::new();
        for row in 0..16 {
            for col in 0..8 {
                golden.push((8 - col) as f32);
                actual.push(if row < 2 {
                    col as f32
                } else {
                    (8 - col) as f32
                });
            }
        }
        let (min_overlap, mean_overlap) =
            topk_overlap_from_logits(&golden, &actual, 16, 4).unwrap();
        assert_eq!(min_overlap, 0);
        assert_eq!(mean_overlap, 3.5);
    }

    #[test]
    fn comparator_rejects_logit_drift_beyond_tolerance() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_logit = tensor_data_offset(&bytes, AUDIO_LOGITS_F32);
        bytes[first_logit + 3] ^= 0x40;
        std::fs::write(tmp.path(), bytes).unwrap();

        let comparison =
            compare_one_step_files(GOLDEN, tmp.path(), OneStepTolerances::default()).unwrap();
        let logits = comparison
            .tensors
            .iter()
            .find(|tensor| tensor.name == AUDIO_LOGITS_F32)
            .unwrap();
        assert!(!logits.passed);
        assert!(!comparison.passed());
        let err = ensure_comparison_passed(&comparison)
            .unwrap_err()
            .to_string();
        assert!(err.contains(AUDIO_LOGITS_F32));
    }

    fn tensor_data_offset(bytes: &[u8], name: &str) -> usize {
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let start = header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        8 + header_len + start
    }

    fn dynamic_one_step_bytes(batch: usize, seq: usize) -> Vec<u8> {
        let rows = batch * NUM_CODEBOOKS;
        let mut prompt = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for row in 0..batch {
            for col in 0..seq {
                prompt.push((row * seq + col + 1) as i64);
                mask.push(1);
            }
        }
        let lengths = vec![seq as i64; batch];

        let hidden: Vec<_> = (0..batch * HIDDEN_SIZE)
            .map(|idx| bf16::from_f32((idx % 17 + 1) as f32 / 17.0))
            .collect();
        let mut logits = vec![0.0f32; rows * CODEBOOK_VOCAB_SIZE];
        let mut argmax = Vec::with_capacity(rows);
        let mut top_ids = Vec::with_capacity(rows * TOP_K);
        let mut top_logprobs = Vec::with_capacity(rows * TOP_K);
        for row in 0..rows {
            let best = row % CODEBOOK_VOCAB_SIZE;
            logits[row * CODEBOOK_VOCAB_SIZE + best] = 1.0;
            argmax.push(best as i64);
            for offset in 0..TOP_K {
                top_ids.push(((best + offset) % CODEBOOK_VOCAB_SIZE) as i64);
                top_logprobs.push(-(offset as f32));
            }
        }

        let tensors = BTreeMap::from([
            (
                PROMPT_INPUT_IDS.to_string(),
                test_i64(&[batch, seq], &prompt),
            ),
            (
                PROMPT_ATTENTION_MASK.to_string(),
                test_i64(&[batch, seq], &mask),
            ),
            (PROMPT_LENGTHS.to_string(), test_i64(&[batch], &lengths)),
            (
                FINAL_HIDDEN_BF16.to_string(),
                test_bf16(&[batch, HIDDEN_SIZE], &hidden),
            ),
            (
                AUDIO_LOGITS_F32.to_string(),
                test_f32(&[batch, NUM_CODEBOOKS, CODEBOOK_VOCAB_SIZE], &logits),
            ),
            (
                AUDIO_TOP64_IDS.to_string(),
                test_i64(&[batch, NUM_CODEBOOKS, TOP_K], &top_ids),
            ),
            (
                AUDIO_TOP64_LOGPROBS_F32.to_string(),
                test_f32(&[batch, NUM_CODEBOOKS, TOP_K], &top_logprobs),
            ),
            (
                AUDIO_ARGMAX_IDS.to_string(),
                test_i64(&[batch, NUM_CODEBOOKS], &argmax),
            ),
        ]);
        safetensors::serialize(tensors, None).unwrap()
    }

    #[derive(Clone)]
    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestTensor {
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

    fn test_i64(shape: &[usize], values: &[i64]) -> TestTensor {
        TestTensor {
            dtype: Dtype::I64,
            shape: shape.to_vec(),
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn test_f32(shape: &[usize], values: &[f32]) -> TestTensor {
        TestTensor {
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn test_bf16(shape: &[usize], values: &[bf16]) -> TestTensor {
        TestTensor {
            dtype: Dtype::BF16,
            shape: shape.to_vec(),
            data: values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        }
    }
}
