use std::fmt::Display;
use std::fmt::Write as _;
use std::fs;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

pub const ENGINE_PROFILE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineProfile {
    pub schema_version: u32,
    pub calibration: CalibrationReference,
    pub scheduler: SchedulerProfile,
    pub predictor: PredictorProfile,
}

impl EngineProfile {
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).context("failed to parse engine profile JSON")?;
        if value.get("profile_id").is_some()
            || value.get("provenance").is_some()
            || value.get("timing").is_some()
        {
            bail!(
                "legacy engine profile fields 'profile_id'/'provenance'/'timing' are unsupported; migrate to schema_version {ENGINE_PROFILE_SCHEMA_VERSION} with a calibration manifest reference and explicit predictor coverage"
            );
        }
        let profile: Self =
            serde_json::from_value(value).context("failed to deserialize engine profile schema")?;
        profile.validate()?;
        Ok(profile)
    }

    /// Load a profile and its calibration manifest from one relocatable bundle.
    /// The manifest digest is checked before the loaded profile is returned.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<LoadedEngineProfile> {
        let profile_path = path.as_ref().to_path_buf();
        let profile_bytes = fs::read(&profile_path)
            .with_context(|| format!("failed to read engine profile {}", profile_path.display()))?;
        let profile = Self::from_json_slice(&profile_bytes)
            .with_context(|| format!("failed to load engine profile {}", profile_path.display()))?;

        let profile_dir = profile_path.parent().unwrap_or_else(|| Path::new("."));
        let manifest_path = profile_dir.join(&profile.calibration.manifest);
        let manifest_bytes = fs::read(&manifest_path).with_context(|| {
            format!(
                "failed to read calibration manifest {} referenced by {}",
                manifest_path.display(),
                profile_path.display()
            )
        })?;
        let actual_sha256 = sha256_hex(&manifest_bytes);
        ensure!(
            actual_sha256.eq_ignore_ascii_case(&profile.calibration.sha256),
            "calibration manifest digest mismatch for {}: profile references {}, actual {}",
            manifest_path.display(),
            profile.calibration.sha256,
            actual_sha256
        );
        let manifest: CalibrationManifest =
            serde_json::from_slice(&manifest_bytes).with_context(|| {
                format!(
                    "failed to parse calibration manifest {}",
                    manifest_path.display()
                )
            })?;
        manifest.validate()?;

        Ok(LoadedEngineProfile {
            profile,
            manifest,
            profile_path,
            manifest_path,
        })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == ENGINE_PROFILE_SCHEMA_VERSION,
            "unsupported engine profile schema_version {}; expected {}",
            self.schema_version,
            ENGINE_PROFILE_SCHEMA_VERSION
        );
        self.calibration.validate()?;
        self.scheduler.validate()?;
        self.predictor.validate(&self.scheduler)
    }

    pub fn estimate_step(&self, shape: StepShape) -> Result<StepTimingEstimate> {
        self.scheduler.validate_shape(shape)?;
        let domain = self.predictor.coverage.domain();
        ensure!(
            self.predictor.coverage.contains(shape),
            "calibration manifest '{}' (sha256 {}) does not cover step shape {shape:?}; supported predictor coverage: {domain:?}",
            self.calibration.manifest,
            self.calibration.sha256,
        );
        let duration_us = self
            .predictor
            .grid
            .interpolate(shape)?
            .context("predictor coverage contains a shape with no grid duration")?;
        Ok(StepTimingEstimate {
            duration_us,
            source: StepTimingSource::GridInterpolation,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationReference {
    pub manifest: String,
    pub sha256: String,
}

impl CalibrationReference {
    fn validate(&self) -> Result<()> {
        ensure_nonempty("calibration.manifest", &self.manifest)?;
        ensure!(
            !Path::new(&self.manifest).is_absolute(),
            "calibration.manifest must be a relative path"
        );
        ensure!(
            is_sha256(&self.sha256),
            "calibration.sha256 must be 64 hexadecimal characters"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationManifest {
    pub target_engine: String,
    pub engine_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_config_sha256: String,
    pub gpu: String,
    pub server_flags: Vec<String>,
    pub collection_command: String,
    pub source_artifacts: Vec<String>,
}

impl CalibrationManifest {
    fn validate(&self) -> Result<()> {
        ensure_nonempty("manifest.target_engine", &self.target_engine)?;
        ensure_nonempty("manifest.engine_version", &self.engine_version)?;
        ensure_nonempty("manifest.model_id", &self.model_id)?;
        ensure_nonempty("manifest.model_revision", &self.model_revision)?;
        ensure_nonempty("manifest.gpu", &self.gpu)?;
        ensure!(
            is_sha256(&self.model_config_sha256),
            "manifest.model_config_sha256 must be 64 hexadecimal characters"
        );
        for flag in &self.server_flags {
            ensure_nonempty("manifest.server_flags entry", flag)?;
        }
        ensure_nonempty("manifest.collection_command", &self.collection_command)?;
        for artifact in &self.source_artifacts {
            ensure_nonempty("manifest.source_artifacts entry", artifact)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedEngineProfile {
    pub profile: EngineProfile,
    pub manifest: CalibrationManifest,
    pub profile_path: PathBuf,
    pub manifest_path: PathBuf,
}

impl LoadedEngineProfile {
    pub fn validate(&self) -> Result<()> {
        self.profile.validate()?;
        self.manifest.validate()
    }

    pub fn model_id(&self) -> &str {
        &self.manifest.model_id
    }
}

impl Deref for LoadedEngineProfile {
    type Target = EngineProfile;

    fn deref(&self) -> &Self::Target {
        &self.profile
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerPolicy {
    VllmV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerProfile {
    pub policy: SchedulerPolicy,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    pub max_model_len: u32,
    pub prefill: PrefillPolicy,
}

impl SchedulerProfile {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_num_seqs > 0,
            "scheduler.max_num_seqs must be positive"
        );
        ensure!(
            self.max_num_batched_tokens > 0,
            "scheduler.max_num_batched_tokens must be positive"
        );
        ensure!(
            self.max_model_len > 0,
            "scheduler.max_model_len must be positive"
        );
        ensure!(
            self.max_num_seqs <= self.max_num_batched_tokens,
            "scheduler.max_num_seqs cannot exceed max_num_batched_tokens because every decoding request consumes one token per step"
        );
        if let PrefillPolicy::Chunked { max_chunk_tokens } = self.prefill {
            ensure!(
                max_chunk_tokens > 0,
                "scheduler.prefill.max_chunk_tokens must be positive"
            );
            ensure!(
                max_chunk_tokens <= self.max_num_batched_tokens,
                "scheduler.prefill.max_chunk_tokens cannot exceed max_num_batched_tokens"
            );
        }
        Ok(())
    }

    fn validate_shape(&self, shape: StepShape) -> Result<()> {
        ensure!(
            shape.decode_reqs > 0 || shape.prefill_tokens_in_step > 0,
            "step shape must contain prefill or decode work"
        );
        ensure!(
            shape.decode_reqs <= self.max_num_seqs,
            "step shape decode_reqs {} exceed scheduler max_num_seqs {}",
            shape.decode_reqs,
            self.max_num_seqs
        );
        let step_tokens = shape
            .decode_reqs
            .checked_add(shape.prefill_tokens_in_step)
            .context("step token count overflow")?;
        ensure!(
            step_tokens <= self.max_num_batched_tokens,
            "step shape token count {step_tokens} exceeds scheduler max_num_batched_tokens {}",
            self.max_num_batched_tokens
        );
        if shape.decode_reqs == 0 {
            ensure!(
                shape.sum_decode_ctx_tokens == 0,
                "sum_decode_ctx_tokens must be zero when decode_reqs is zero"
            );
        }
        let max_decode_ctx = u64::from(shape.decode_reqs)
            .checked_mul(u64::from(self.max_model_len))
            .context("maximum decode context overflow")?;
        ensure!(
            shape.sum_decode_ctx_tokens <= max_decode_ctx,
            "step shape sum_decode_ctx_tokens {} exceed the per-request context ceiling {}",
            shape.sum_decode_ctx_tokens,
            max_decode_ctx
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrefillPolicy {
    Whole,
    Chunked { max_chunk_tokens: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepShape {
    pub decode_reqs: u32,
    pub sum_decode_ctx_tokens: u64,
    pub prefill_tokens_in_step: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorProfile {
    pub coverage: PredictorCoverage,
    pub grid: TimingGrid,
}

impl PredictorProfile {
    fn validate(&self, scheduler: &SchedulerProfile) -> Result<()> {
        self.grid.validate(scheduler)?;
        self.coverage.validate(scheduler, &self.grid)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorCoverage {
    pub decode_reqs: [u32; 2],
    pub sum_decode_ctx_tokens: [u64; 2],
    pub prefill_tokens_in_step: [u32; 2],
}

impl PredictorCoverage {
    fn validate(&self, scheduler: &SchedulerProfile, grid: &TimingGrid) -> Result<()> {
        validate_range("predictor.coverage.decode_reqs", self.decode_reqs)?;
        validate_range(
            "predictor.coverage.sum_decode_ctx_tokens",
            self.sum_decode_ctx_tokens,
        )?;
        validate_range(
            "predictor.coverage.prefill_tokens_in_step",
            self.prefill_tokens_in_step,
        )?;
        ensure!(
            self.decode_reqs[1] <= scheduler.max_num_seqs,
            "predictor.coverage.decode_reqs exceed scheduler.max_num_seqs"
        );
        ensure!(
            self.prefill_tokens_in_step[1] <= scheduler.max_num_batched_tokens,
            "predictor.coverage.prefill_tokens_in_step exceed scheduler.max_num_batched_tokens"
        );
        let max_context = u64::from(scheduler.max_num_seqs)
            .checked_mul(u64::from(scheduler.max_model_len))
            .context("scheduler context domain overflow")?;
        ensure!(
            self.sum_decode_ctx_tokens[1] <= max_context,
            "predictor.coverage.sum_decode_ctx_tokens exceed the scheduler context domain"
        );
        let grid_domain = grid
            .domain()
            .context("timing grid must contain all predictor coverage axes")?;
        ensure!(
            self.domain() == grid_domain,
            "predictor coverage must match the timing grid domain"
        );
        Ok(())
    }

    fn contains(&self, shape: StepShape) -> bool {
        self.decode_reqs[0] <= shape.decode_reqs
            && shape.decode_reqs <= self.decode_reqs[1]
            && self.sum_decode_ctx_tokens[0] <= shape.sum_decode_ctx_tokens
            && shape.sum_decode_ctx_tokens <= self.sum_decode_ctx_tokens[1]
            && self.prefill_tokens_in_step[0] <= shape.prefill_tokens_in_step
            && shape.prefill_tokens_in_step <= self.prefill_tokens_in_step[1]
    }

    pub fn domain(&self) -> TimingDomain {
        TimingDomain {
            min_decode_reqs: self.decode_reqs[0],
            max_decode_reqs: self.decode_reqs[1],
            min_sum_decode_ctx_tokens: self.sum_decode_ctx_tokens[0],
            max_sum_decode_ctx_tokens: self.sum_decode_ctx_tokens[1],
            min_prefill_tokens_in_step: self.prefill_tokens_in_step[0],
            max_prefill_tokens_in_step: self.prefill_tokens_in_step[1],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimingGrid {
    pub decode_reqs: Vec<u32>,
    pub sum_decode_ctx_tokens: Vec<u64>,
    pub prefill_tokens_in_step: Vec<u32>,
    /// Row-major order: decode request, decode context, then prefill token axis.
    pub step_duration_us: Vec<u64>,
}

impl TimingGrid {
    fn validate(&self, scheduler: &SchedulerProfile) -> Result<()> {
        validate_axis("timing.grid.decode_reqs", &self.decode_reqs)?;
        validate_axis(
            "timing.grid.sum_decode_ctx_tokens",
            &self.sum_decode_ctx_tokens,
        )?;
        validate_axis(
            "timing.grid.prefill_tokens_in_step",
            &self.prefill_tokens_in_step,
        )?;
        let expected_values = self
            .decode_reqs
            .len()
            .checked_mul(self.sum_decode_ctx_tokens.len())
            .and_then(|value| value.checked_mul(self.prefill_tokens_in_step.len()))
            .context("timing grid dimensions overflow")?;
        ensure!(
            self.step_duration_us.len() == expected_values,
            "timing.grid.step_duration_us contains {} values; expected {expected_values}",
            self.step_duration_us.len()
        );
        ensure!(
            self.decode_reqs
                .last()
                .copied()
                .expect("validated non-empty axis")
                <= scheduler.max_num_seqs,
            "timing.grid.decode_reqs exceed scheduler.max_num_seqs"
        );
        ensure!(
            self.prefill_tokens_in_step
                .last()
                .copied()
                .expect("validated non-empty axis")
                <= scheduler.max_num_batched_tokens,
            "timing.grid.prefill_tokens_in_step exceed scheduler.max_num_batched_tokens"
        );
        let max_context = u64::from(scheduler.max_num_seqs)
            .checked_mul(u64::from(scheduler.max_model_len))
            .context("scheduler context domain overflow")?;
        ensure!(
            self.sum_decode_ctx_tokens
                .last()
                .copied()
                .expect("validated non-empty axis")
                <= max_context,
            "timing.grid.sum_decode_ctx_tokens exceed the scheduler context domain"
        );
        Ok(())
    }

    pub fn domain(&self) -> Option<TimingDomain> {
        Some(TimingDomain {
            min_decode_reqs: *self.decode_reqs.first()?,
            max_decode_reqs: *self.decode_reqs.last()?,
            min_sum_decode_ctx_tokens: *self.sum_decode_ctx_tokens.first()?,
            max_sum_decode_ctx_tokens: *self.sum_decode_ctx_tokens.last()?,
            min_prefill_tokens_in_step: *self.prefill_tokens_in_step.first()?,
            max_prefill_tokens_in_step: *self.prefill_tokens_in_step.last()?,
        })
    }

    fn interpolate(&self, shape: StepShape) -> Result<Option<u64>> {
        let Some(decode) = bracket(&self.decode_reqs, shape.decode_reqs) else {
            return Ok(None);
        };
        let Some(context) = bracket(&self.sum_decode_ctx_tokens, shape.sum_decode_ctx_tokens)
        else {
            return Ok(None);
        };
        let Some(prefill) = bracket(&self.prefill_tokens_in_step, shape.prefill_tokens_in_step)
        else {
            return Ok(None);
        };

        let d0c0 = interpolate_bracket(
            self.value_at(decode.lower, context.lower, prefill.lower)?,
            self.value_at(decode.lower, context.lower, prefill.upper)?,
            prefill,
        )?;
        let d0c1 = interpolate_bracket(
            self.value_at(decode.lower, context.upper, prefill.lower)?,
            self.value_at(decode.lower, context.upper, prefill.upper)?,
            prefill,
        )?;
        let d1c0 = interpolate_bracket(
            self.value_at(decode.upper, context.lower, prefill.lower)?,
            self.value_at(decode.upper, context.lower, prefill.upper)?,
            prefill,
        )?;
        let d1c1 = interpolate_bracket(
            self.value_at(decode.upper, context.upper, prefill.lower)?,
            self.value_at(decode.upper, context.upper, prefill.upper)?,
            prefill,
        )?;
        let d0 = interpolate_bracket(d0c0, d0c1, context)?;
        let d1 = interpolate_bracket(d1c0, d1c1, context)?;
        Ok(Some(interpolate_bracket(d0, d1, decode)?))
    }

    fn value_at(&self, decode: usize, context: usize, prefill: usize) -> Result<u64> {
        let index = decode
            .checked_mul(self.sum_decode_ctx_tokens.len())
            .and_then(|value| value.checked_add(context))
            .and_then(|value| value.checked_mul(self.prefill_tokens_in_step.len()))
            .and_then(|value| value.checked_add(prefill))
            .context("timing grid index overflow")?;
        self.step_duration_us
            .get(index)
            .copied()
            .context("timing grid is missing a duration value")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimingDomain {
    pub min_decode_reqs: u32,
    pub max_decode_reqs: u32,
    pub min_sum_decode_ctx_tokens: u64,
    pub max_sum_decode_ctx_tokens: u64,
    pub min_prefill_tokens_in_step: u32,
    pub max_prefill_tokens_in_step: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepTimingSource {
    GridInterpolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepTimingEstimate {
    pub duration_us: u64,
    pub source: StepTimingSource,
}

#[derive(Clone, Copy)]
struct AxisBracket {
    lower: usize,
    upper: usize,
    numerator: u64,
    denominator: u64,
}

fn bracket<T>(axis: &[T], value: T) -> Option<AxisBracket>
where
    T: Copy + Ord + Into<u64>,
{
    match axis.binary_search(&value) {
        Ok(index) => Some(AxisBracket {
            lower: index,
            upper: index,
            numerator: 0,
            denominator: 1,
        }),
        Err(0) => None,
        Err(index) if index == axis.len() => None,
        Err(index) => {
            let lower_value = axis[index - 1].into();
            let upper_value = axis[index].into();
            Some(AxisBracket {
                lower: index - 1,
                upper: index,
                numerator: value.into() - lower_value,
                denominator: upper_value - lower_value,
            })
        }
    }
}

fn interpolate_bracket(lower: u64, upper: u64, bracket: AxisBracket) -> Result<u64> {
    if bracket.lower == bracket.upper || lower == upper {
        return Ok(lower);
    }
    let delta = lower.abs_diff(upper);
    let scaled = u128::from(delta)
        .checked_mul(u128::from(bracket.numerator))
        .context("timing interpolation multiplication overflow")?;
    let rounded = scaled
        .checked_add(u128::from(bracket.denominator / 2))
        .context("timing interpolation rounding overflow")?
        / u128::from(bracket.denominator);
    let adjustment = u64::try_from(rounded).context("timing interpolation result overflow")?;
    if upper >= lower {
        lower
            .checked_add(adjustment)
            .context("timing interpolation addition overflow")
    } else {
        lower
            .checked_sub(adjustment)
            .context("timing interpolation subtraction overflow")
    }
}

fn ensure_nonempty(field: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{field} must not be empty");
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn validate_axis<T>(name: &str, axis: &[T]) -> Result<()>
where
    T: Copy + Ord + Display,
{
    ensure!(!axis.is_empty(), "{name} must not be empty");
    for pair in axis.windows(2) {
        ensure!(
            pair[0] < pair[1],
            "{name} must be strictly increasing; found {} then {}",
            pair[0],
            pair[1]
        );
    }
    Ok(())
}

fn validate_range<T>(name: &str, range: [T; 2]) -> Result<()>
where
    T: Copy + Ord + Display,
{
    ensure!(
        range[0] <= range[1],
        "{name} must be ordered, found {} then {}",
        range[0],
        range[1]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::*;

    fn profile_json() -> Value {
        json!({
            "schema_version": 1,
            "calibration": {
                "manifest": "calibration.manifest.json",
                "sha256": "11".repeat(32)
            },
            "scheduler": {
                "policy": "vllm_v1",
                "max_num_seqs": 4,
                "max_num_batched_tokens": 16,
                "max_model_len": 32,
                "prefill": {
                    "mode": "chunked",
                    "max_chunk_tokens": 8
                }
            },
            "predictor": {
                "coverage": {
                    "decode_reqs": [0, 2],
                    "sum_decode_ctx_tokens": [0, 20],
                    "prefill_tokens_in_step": [0, 10]
                },
                "grid": {
                    "decode_reqs": [0, 2],
                    "sum_decode_ctx_tokens": [0, 20],
                    "prefill_tokens_in_step": [0, 10],
                    "step_duration_us": [10, 80, 70, 140, 210, 280, 270, 340]
                }
            }
        })
    }

    fn parse(value: &Value) -> Result<EngineProfile> {
        EngineProfile::from_json_slice(&serde_json::to_vec(value)?)
    }

    #[test]
    fn profile_rejects_invalid_scheduler_and_grid() {
        let mut bad_chunk = profile_json();
        bad_chunk["scheduler"]["prefill"]["max_chunk_tokens"] = json!(17);
        assert!(
            parse(&bad_chunk)
                .unwrap_err()
                .to_string()
                .contains("cannot exceed max_num_batched_tokens")
        );

        let mut unordered_axis = profile_json();
        unordered_axis["predictor"]["grid"]["decode_reqs"] = json!([2, 0]);
        assert!(
            parse(&unordered_axis)
                .unwrap_err()
                .to_string()
                .contains("must be strictly increasing")
        );

        let mut missing_value = profile_json();
        missing_value["predictor"]["grid"]["step_duration_us"] = json!([1, 2]);
        assert!(
            parse(&missing_value)
                .unwrap_err()
                .to_string()
                .contains("contains 2 values; expected 8")
        );
    }

    #[test]
    fn grid_returns_exact_points_and_deterministic_interpolation() {
        let profile = parse(&profile_json()).unwrap();
        let exact = profile
            .estimate_step(StepShape {
                decode_reqs: 2,
                sum_decode_ctx_tokens: 20,
                prefill_tokens_in_step: 10,
            })
            .unwrap();
        assert_eq!(
            exact,
            StepTimingEstimate {
                duration_us: 340,
                source: StepTimingSource::GridInterpolation,
            }
        );

        let interpolated = profile
            .estimate_step(StepShape {
                decode_reqs: 1,
                sum_decode_ctx_tokens: 10,
                prefill_tokens_in_step: 5,
            })
            .unwrap();
        assert_eq!(interpolated.duration_us, 175);
        assert_eq!(interpolated.source, StepTimingSource::GridInterpolation);
    }

    #[test]
    fn out_of_domain_shape_fails_with_coverage_details() {
        let profile = parse(&profile_json()).unwrap();
        for shape in [
            StepShape {
                decode_reqs: 3,
                sum_decode_ctx_tokens: 10,
                prefill_tokens_in_step: 2,
            },
            StepShape {
                decode_reqs: 1,
                sum_decode_ctx_tokens: 21,
                prefill_tokens_in_step: 2,
            },
            StepShape {
                decode_reqs: 1,
                sum_decode_ctx_tokens: 10,
                prefill_tokens_in_step: 12,
            },
        ] {
            let error = profile.estimate_step(shape).unwrap_err().to_string();
            assert!(error.contains("does not cover step shape"));
            assert!(error.contains("supported predictor coverage"));
            assert!(error.contains("calibration.manifest.json"));
        }
    }

    #[test]
    fn invalid_step_shapes_fail_before_coverage() {
        let profile = parse(&profile_json()).unwrap();
        let no_work = profile
            .estimate_step(StepShape {
                decode_reqs: 0,
                sum_decode_ctx_tokens: 0,
                prefill_tokens_in_step: 0,
            })
            .unwrap_err();
        assert!(
            no_work
                .to_string()
                .contains("must contain prefill or decode")
        );

        let over_budget = profile
            .estimate_step(StepShape {
                decode_reqs: 4,
                sum_decode_ctx_tokens: 16,
                prefill_tokens_in_step: 13,
            })
            .unwrap_err();
        assert!(over_budget.to_string().contains("exceeds scheduler"));

        let context_without_decode = profile
            .estimate_step(StepShape {
                decode_reqs: 0,
                sum_decode_ctx_tokens: 1,
                prefill_tokens_in_step: 1,
            })
            .unwrap_err();
        assert!(
            context_without_decode
                .to_string()
                .contains("must be zero when decode_reqs is zero")
        );
    }

    #[test]
    fn zero_cost_profile_is_valid_and_out_of_domain_fails() {
        let mut zero_cost = profile_json();
        zero_cost["predictor"]["grid"]["step_duration_us"] = json!([0, 0, 0, 0, 0, 0, 0, 0]);
        let profile = parse(&zero_cost).unwrap();
        assert_eq!(
            profile
                .estimate_step(StepShape {
                    decode_reqs: 1,
                    sum_decode_ctx_tokens: 10,
                    prefill_tokens_in_step: 10,
                },)
                .unwrap()
                .duration_us,
            0
        );
        let error = profile
            .estimate_step(StepShape {
                decode_reqs: 1,
                sum_decode_ctx_tokens: 10,
                prefill_tokens_in_step: 11,
            })
            .unwrap_err();
        assert!(error.to_string().contains("supported predictor coverage"));
    }
}
