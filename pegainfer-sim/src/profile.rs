use std::collections::BTreeSet;
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
use thiserror::Error;

pub const ENGINE_PROFILE_SCHEMA_VERSION: u32 = 3;
pub const CALIBRATION_MANIFEST_SCHEMA_VERSION: u32 = 2;
const LEGACY_CALIBRATION_MANIFEST_SCHEMA_VERSION: u32 = 1;

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
        if let Some(version) = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
        {
            ensure!(
                version == u64::from(ENGINE_PROFILE_SCHEMA_VERSION),
                "unsupported engine profile schema_version {version}; migrate to schema_version {ENGINE_PROFILE_SCHEMA_VERSION} with a calibration manifest reference and explicit predictor coverage"
            );
        }
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
        manifest.validate_for_profile(&profile)?;

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
        if !self.predictor.coverage.contains(shape) {
            return Err(PredictorError::OutOfDomain {
                shape,
                coverage: domain,
                manifest: self.calibration.manifest.clone(),
                manifest_sha256: self.calibration.sha256.clone(),
            }
            .into());
        }
        let (duration_us, source) = self.predictor.model.estimate(shape)?;
        Ok(StepTimingEstimate {
            duration_us,
            source,
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

#[derive(Clone, Debug, Deserialize, Eq, Error, PartialEq, Serialize)]
pub enum ProfileContractError {
    #[error(
        "unsupported calibration manifest schema_version {0}; expected {CALIBRATION_MANIFEST_SCHEMA_VERSION} or legacy {LEGACY_CALIBRATION_MANIFEST_SCHEMA_VERSION}"
    )]
    UnsupportedManifestSchema(u32),
    #[error("calibration manifest scheduler identity does not match the profile scheduler")]
    SchedulerIdentityMismatch,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum StepShapeError {
    #[error("step shape {shape:?} must contain prefill or decode work")]
    NoWork { shape: StepShape },
    #[error("step shape decode_reqs {decode_reqs} exceeds scheduler max_num_seqs {max_num_seqs}")]
    DecodeRequestsExceedCapacity { decode_reqs: u32, max_num_seqs: u32 },
    #[error(
        "step shape token count {step_tokens} exceeds scheduler max_num_batched_tokens {max_num_batched_tokens}"
    )]
    TokenBudgetExceeded {
        step_tokens: u32,
        max_num_batched_tokens: u32,
    },
    #[error(
        "step shape sum_decode_ctx_tokens {sum_decode_ctx_tokens} must be zero when decode_reqs is zero"
    )]
    ContextWithoutDecode { sum_decode_ctx_tokens: u64 },
    #[error(
        "step shape sum_decode_ctx_tokens {sum_decode_ctx_tokens} exceeds the per-request context ceiling {max_decode_ctx}"
    )]
    ContextExceedsCapacity {
        sum_decode_ctx_tokens: u64,
        max_decode_ctx: u64,
    },
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum PredictorError {
    #[error(transparent)]
    InvalidShape(#[from] StepShapeError),
    #[error(
        "calibration manifest '{manifest}' (sha256 {manifest_sha256}) does not cover step shape {shape:?}; supported predictor coverage: {coverage:?}"
    )]
    OutOfDomain {
        shape: StepShape,
        coverage: TimingDomain,
        manifest: String,
        manifest_sha256: String,
    },
    #[error("selected predictor cannot price step shape {shape:?}: {reason}")]
    UnsupportedShape {
        shape: StepShape,
        reason: &'static str,
    },
    #[error("selected predictor model is missing a value for step shape {shape:?}: {model}")]
    ModelCoverageGap {
        shape: StepShape,
        model: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionKind {
    BlackBoxEffective,
    TraceBacked,
    AttributionUncertain,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadFamily {
    ColdPrefill,
    SteadyDecode,
    CachedPrefill,
    Mixed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestSchedulerIdentity {
    pub policy: SchedulerPolicy,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    pub max_model_len: u32,
    pub prefill: PrefillPolicy,
}

impl From<&SchedulerProfile> for ManifestSchedulerIdentity {
    fn from(scheduler: &SchedulerProfile) -> Self {
        Self {
            policy: scheduler.policy,
            max_num_seqs: scheduler.max_num_seqs,
            max_num_batched_tokens: scheduler.max_num_batched_tokens,
            max_model_len: scheduler.max_model_len,
            prefill: scheduler.prefill,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheIdentity {
    pub enabled: bool,
    pub policy: String,
    pub block_size: u32,
    pub alignment: u32,
    pub hit_semantics: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientIdentity {
    pub name: String,
    pub version: String,
    pub tokenizer_config_sha256: String,
    pub seed: Option<u64>,
    pub sampling: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadCell {
    pub id: String,
    pub family: WorkloadFamily,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub concurrency: u32,
    pub arrival_pattern: String,
    pub attribution: AttributionKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadMatrix {
    pub cells: Vec<WorkloadCell>,
    pub warmup_repeats: u32,
    pub repeats: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSplit {
    pub train: Vec<String>,
    pub threshold_validation: Vec<String>,
    pub held_out_mixed: Vec<String>,
    pub sealed_test: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestProvenance {
    pub collector_revision: String,
    pub fitter_revision: String,
    pub artifact_uri: String,
    pub source_digest: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationManifest {
    /// Legacy manifests omit this field and are read as schema version 1.
    #[serde(default = "legacy_calibration_manifest_schema_version")]
    pub schema_version: u32,
    pub target_engine: String,
    pub engine_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_config_sha256: String,
    pub gpu: String,
    pub server_flags: Vec<String>,
    pub collection_command: String,
    pub source_artifacts: Vec<String>,
    /// Expanded identity and campaign metadata introduced by W0. These are
    /// optional only for legacy v1 fixtures; v2 manifests require every field.
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub driver: Option<String>,
    #[serde(default)]
    pub cuda: Option<String>,
    #[serde(default)]
    pub parallelism: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub scheduler: Option<ManifestSchedulerIdentity>,
    #[serde(default)]
    pub cache: Option<CacheIdentity>,
    #[serde(default)]
    pub client: Option<ClientIdentity>,
    #[serde(default)]
    pub workload: Option<WorkloadMatrix>,
    #[serde(default)]
    pub split: Option<CampaignSplit>,
    #[serde(default)]
    pub provenance: Option<ManifestProvenance>,
}

impl CalibrationManifest {
    fn validate(&self) -> Result<()> {
        match self.schema_version {
            LEGACY_CALIBRATION_MANIFEST_SCHEMA_VERSION | CALIBRATION_MANIFEST_SCHEMA_VERSION => {}
            version => return Err(ProfileContractError::UnsupportedManifestSchema(version).into()),
        }
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
        for (name, value) in [
            ("manifest.backend", self.backend.as_deref()),
            ("manifest.driver", self.driver.as_deref()),
            ("manifest.cuda", self.cuda.as_deref()),
            ("manifest.parallelism", self.parallelism.as_deref()),
            ("manifest.quantization", self.quantization.as_deref()),
        ] {
            if let Some(value) = value {
                ensure_nonempty(name, value)?;
            }
        }
        if let Some(scheduler) = &self.scheduler {
            SchedulerProfile {
                policy: scheduler.policy,
                max_num_seqs: scheduler.max_num_seqs,
                max_num_batched_tokens: scheduler.max_num_batched_tokens,
                max_model_len: scheduler.max_model_len,
                prefill: scheduler.prefill,
            }
            .validate()?;
        }
        if let Some(cache) = &self.cache {
            ensure_nonempty("manifest.cache.policy", &cache.policy)?;
            ensure_nonempty("manifest.cache.hit_semantics", &cache.hit_semantics)?;
            if cache.enabled {
                ensure!(
                    cache.block_size > 0,
                    "manifest.cache.block_size must be positive when cache is enabled"
                );
                ensure!(
                    cache.alignment > 0,
                    "manifest.cache.alignment must be positive when cache is enabled"
                );
            }
        }
        if let Some(client) = &self.client {
            ensure_nonempty("manifest.client.name", &client.name)?;
            ensure_nonempty("manifest.client.version", &client.version)?;
            ensure!(
                is_sha256(&client.tokenizer_config_sha256),
                "manifest.client.tokenizer_config_sha256 must be 64 hexadecimal characters"
            );
            ensure_nonempty("manifest.client.sampling", &client.sampling)?;
        }
        if let Some(workload) = &self.workload {
            workload.validate()?;
        }
        if let Some(split) = &self.split {
            split.validate(self.workload.as_ref())?;
        }
        if let Some(provenance) = &self.provenance {
            ensure_nonempty(
                "manifest.provenance.collector_revision",
                &provenance.collector_revision,
            )?;
            ensure_nonempty(
                "manifest.provenance.fitter_revision",
                &provenance.fitter_revision,
            )?;
            ensure_nonempty("manifest.provenance.artifact_uri", &provenance.artifact_uri)?;
            ensure!(
                is_sha256(&provenance.source_digest),
                "manifest.provenance.source_digest must be 64 hexadecimal characters"
            );
        }
        if self.schema_version == CALIBRATION_MANIFEST_SCHEMA_VERSION {
            for (name, present) in [
                ("backend", self.backend.is_some()),
                ("driver", self.driver.is_some()),
                ("cuda", self.cuda.is_some()),
                ("parallelism", self.parallelism.is_some()),
                ("quantization", self.quantization.is_some()),
                ("scheduler", self.scheduler.is_some()),
                ("cache", self.cache.is_some()),
                ("client", self.client.is_some()),
                ("workload", self.workload.is_some()),
                ("split", self.split.is_some()),
                ("provenance", self.provenance.is_some()),
            ] {
                ensure!(
                    present,
                    "manifest schema_version 2 requires manifest.{name}"
                );
            }
        }
        Ok(())
    }

    fn validate_for_profile(&self, profile: &EngineProfile) -> Result<()> {
        self.validate()?;
        if let Some(scheduler) = &self.scheduler {
            ensure!(
                scheduler == &ManifestSchedulerIdentity::from(&profile.scheduler),
                ProfileContractError::SchedulerIdentityMismatch
            );
        }
        Ok(())
    }
}

impl WorkloadMatrix {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.repeats > 0,
            "manifest.workload.repeats must be positive"
        );
        let mut ids = BTreeSet::new();
        for cell in &self.cells {
            ensure_nonempty("manifest.workload.cells.id", &cell.id)?;
            ensure!(
                ids.insert(&cell.id),
                "manifest.workload cell id '{}' is duplicated",
                cell.id
            );
            ensure!(
                cell.concurrency > 0,
                "manifest.workload cell '{}' concurrency must be positive",
                cell.id
            );
            ensure_nonempty(
                "manifest.workload.cells.arrival_pattern",
                &cell.arrival_pattern,
            )?;
            cell.input_tokens
                .checked_add(cell.output_tokens)
                .context("manifest workload token count overflow")?;
        }
        Ok(())
    }
}

impl CampaignSplit {
    fn validate(&self, workload: Option<&WorkloadMatrix>) -> Result<()> {
        let mut seen = BTreeSet::new();
        for (split, cells) in [
            ("train", &self.train),
            ("threshold_validation", &self.threshold_validation),
            ("held_out_mixed", &self.held_out_mixed),
            ("sealed_test", &self.sealed_test),
        ] {
            for cell in cells {
                ensure_nonempty(&format!("manifest.split.{split} cell"), cell)?;
                ensure!(
                    seen.insert(cell),
                    "manifest.split cell '{}' appears in more than one split",
                    cell
                );
            }
        }
        if let Some(workload) = workload {
            let known: BTreeSet<_> = workload.cells.iter().map(|cell| &cell.id).collect();
            for cell in &seen {
                ensure!(
                    known.contains(cell),
                    "manifest.split references unknown workload cell '{cell}'"
                );
            }
        }
        Ok(())
    }
}

fn legacy_calibration_manifest_schema_version() -> u32 {
    LEGACY_CALIBRATION_MANIFEST_SCHEMA_VERSION
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileIdentity {
    pub manifest_sha256: String,
    pub target_engine: String,
    pub engine_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_config_sha256: String,
    pub gpu: String,
    pub backend: Option<String>,
    pub driver: Option<String>,
    pub cuda: Option<String>,
    pub parallelism: Option<String>,
    pub quantization: Option<String>,
    pub scheduler: SchedulerProfile,
    pub cache: Option<CacheIdentity>,
    pub client: Option<ClientIdentity>,
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
        self.manifest.validate_for_profile(&self.profile)
    }

    pub fn model_id(&self) -> &str {
        &self.manifest.model_id
    }

    #[must_use]
    pub fn identity(&self) -> ProfileIdentity {
        ProfileIdentity {
            manifest_sha256: self.profile.calibration.sha256.clone(),
            target_engine: self.manifest.target_engine.clone(),
            engine_version: self.manifest.engine_version.clone(),
            model_id: self.manifest.model_id.clone(),
            model_revision: self.manifest.model_revision.clone(),
            model_config_sha256: self.manifest.model_config_sha256.clone(),
            gpu: self.manifest.gpu.clone(),
            backend: self.manifest.backend.clone(),
            driver: self.manifest.driver.clone(),
            cuda: self.manifest.cuda.clone(),
            parallelism: self.manifest.parallelism.clone(),
            quantization: self.manifest.quantization.clone(),
            scheduler: self.profile.scheduler.clone(),
            cache: self.manifest.cache.clone(),
            client: self.manifest.client.clone(),
        }
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

    fn validate_shape(&self, shape: StepShape) -> std::result::Result<(), StepShapeError> {
        if shape.decode_reqs == 0 && shape.prefill_tokens_in_step == 0 {
            return Err(StepShapeError::NoWork { shape });
        }
        if shape.decode_reqs > self.max_num_seqs {
            return Err(StepShapeError::DecodeRequestsExceedCapacity {
                decode_reqs: shape.decode_reqs,
                max_num_seqs: self.max_num_seqs,
            });
        }
        let step_tokens = shape
            .decode_reqs
            .checked_add(shape.prefill_tokens_in_step)
            .ok_or(StepShapeError::TokenBudgetExceeded {
                step_tokens: u32::MAX,
                max_num_batched_tokens: self.max_num_batched_tokens,
            })?;
        if step_tokens > self.max_num_batched_tokens {
            return Err(StepShapeError::TokenBudgetExceeded {
                step_tokens,
                max_num_batched_tokens: self.max_num_batched_tokens,
            });
        }
        if shape.decode_reqs == 0 {
            if shape.sum_decode_ctx_tokens != 0 {
                return Err(StepShapeError::ContextWithoutDecode {
                    sum_decode_ctx_tokens: shape.sum_decode_ctx_tokens,
                });
            }
        }
        let max_decode_ctx =
            u64::from(shape.decode_reqs).saturating_mul(u64::from(self.max_model_len));
        if shape.sum_decode_ctx_tokens > max_decode_ctx {
            return Err(StepShapeError::ContextExceedsCapacity {
                sum_decode_ctx_tokens: shape.sum_decode_ctx_tokens,
                max_decode_ctx,
            });
        }
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
    pub model: PredictorModel,
}

impl PredictorProfile {
    fn validate(&self, scheduler: &SchedulerProfile) -> Result<()> {
        self.coverage.validate(scheduler)?;
        self.model.validate()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PredictorModel {
    FixedStep {
        duration_us: u64,
    },
    Functional {
        prefill: Option<PiecewiseLinear>,
        decode: Option<DecodeSurface>,
        mixed: Option<MixedAdditive>,
    },
}

impl PredictorModel {
    fn validate(&self) -> Result<()> {
        match self {
            Self::FixedStep { .. } => Ok(()),
            Self::Functional {
                prefill,
                decode,
                mixed,
            } => {
                ensure!(
                    prefill.is_some() || decode.is_some(),
                    "functional predictor must define prefill or decode"
                );
                if let Some(prefill) = prefill {
                    prefill.validate()?;
                }
                if let Some(decode) = decode {
                    decode.validate()?;
                }
                if let Some(mixed) = mixed {
                    ensure!(
                        prefill.is_some() && decode.is_some(),
                        "functional mixed predictor requires both prefill and decode models"
                    );
                    mixed.validate()?;
                }
                Ok(())
            }
        }
    }

    fn estimate(&self, shape: StepShape) -> Result<(u64, StepTimingSource)> {
        match self {
            Self::FixedStep { duration_us } => Ok((*duration_us, StepTimingSource::FixedStep)),
            Self::Functional {
                prefill,
                decode,
                mixed,
            } => {
                let has_prefill = shape.prefill_tokens_in_step > 0;
                let has_decode = shape.decode_reqs > 0;
                match (has_prefill, has_decode) {
                    (true, false) => {
                        let prefill = prefill.as_ref().ok_or(PredictorError::UnsupportedShape {
                            shape,
                            reason: "prefill work is not modeled",
                        })?;
                        let duration = prefill.estimate(shape.prefill_tokens_in_step)?;
                        Ok((duration, StepTimingSource::PrefillPiecewise))
                    }
                    (false, true) => {
                        let decode = decode.as_ref().ok_or(PredictorError::UnsupportedShape {
                            shape,
                            reason: "decode work is not modeled",
                        })?;
                        let duration = decode.estimate(shape)?;
                        Ok((duration, StepTimingSource::DecodeSurface))
                    }
                    (true, true) => {
                        let mixed = mixed.as_ref().ok_or(PredictorError::UnsupportedShape {
                            shape,
                            reason: "mixed work has no validated composition model",
                        })?;
                        let prefill = prefill.as_ref().ok_or(PredictorError::UnsupportedShape {
                            shape,
                            reason: "mixed work is missing its prefill model",
                        })?;
                        let decode = decode.as_ref().ok_or(PredictorError::UnsupportedShape {
                            shape,
                            reason: "mixed work is missing its decode model",
                        })?;
                        let prefill_us = prefill.estimate(shape.prefill_tokens_in_step)?;
                        let decode_us = decode.estimate(shape)?;
                        let duration = prefill_us
                            .checked_add(decode_us)
                            .and_then(|value| value.checked_add(mixed.intercept_us))
                            .context("mixed predictor duration overflow")?;
                        Ok((duration, StepTimingSource::MixedAdditive))
                    }
                    (false, false) => unreachable!("scheduler validation rejects empty work"),
                }
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PiecewiseLinear {
    pub input_tokens: Vec<u32>,
    pub duration_us: Vec<u64>,
}

impl PiecewiseLinear {
    fn validate(&self) -> Result<()> {
        validate_axis("predictor.model.prefill.input_tokens", &self.input_tokens)?;
        ensure!(
            self.input_tokens.len() == self.duration_us.len(),
            "predictor.model.prefill.duration_us contains {} values; expected {}",
            self.duration_us.len(),
            self.input_tokens.len()
        );
        Ok(())
    }

    fn estimate(&self, input_tokens: u32) -> Result<u64> {
        let bracket =
            bracket(&self.input_tokens, input_tokens).ok_or(PredictorError::ModelCoverageGap {
                shape: StepShape {
                    decode_reqs: 0,
                    sum_decode_ctx_tokens: 0,
                    prefill_tokens_in_step: input_tokens,
                },
                model: "prefill_piecewise",
            })?;
        interpolate_bracket(
            self.duration_us[bracket.lower],
            self.duration_us[bracket.upper],
            bracket,
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeSurface {
    pub decode_reqs: Vec<u32>,
    pub sum_decode_ctx_tokens: Vec<u64>,
    /// Row-major order: decode request, then decode context.
    pub duration_us: Vec<u64>,
}

impl DecodeSurface {
    fn validate(&self) -> Result<()> {
        validate_axis("predictor.model.decode.decode_reqs", &self.decode_reqs)?;
        validate_axis(
            "predictor.model.decode.sum_decode_ctx_tokens",
            &self.sum_decode_ctx_tokens,
        )?;
        let expected_values = self
            .decode_reqs
            .len()
            .checked_mul(self.sum_decode_ctx_tokens.len())
            .context("decode surface dimensions overflow")?;
        ensure!(
            self.duration_us.len() == expected_values,
            "predictor.model.decode.duration_us contains {} values; expected {expected_values}",
            self.duration_us.len()
        );
        Ok(())
    }

    fn estimate(&self, shape: StepShape) -> Result<u64> {
        let decode = bracket(&self.decode_reqs, shape.decode_reqs).ok_or(
            PredictorError::ModelCoverageGap {
                shape,
                model: "decode_surface",
            },
        )?;
        let context = bracket(&self.sum_decode_ctx_tokens, shape.sum_decode_ctx_tokens).ok_or(
            PredictorError::ModelCoverageGap {
                shape,
                model: "decode_surface",
            },
        )?;
        let lower = self.value_at(decode.lower, context.lower)?;
        let upper = self.value_at(decode.upper, context.lower)?;
        let lower_context =
            interpolate_bracket(lower, self.value_at(decode.lower, context.upper)?, context)?;
        let upper_context =
            interpolate_bracket(upper, self.value_at(decode.upper, context.upper)?, context)?;
        interpolate_bracket(lower_context, upper_context, decode)
    }

    fn value_at(&self, decode: usize, context: usize) -> Result<u64> {
        let index = decode
            .checked_mul(self.sum_decode_ctx_tokens.len())
            .and_then(|value| value.checked_add(context))
            .context("decode surface index overflow")?;
        self.duration_us
            .get(index)
            .copied()
            .context("decode surface is missing a duration value")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MixedAdditive {
    pub intercept_us: u64,
    #[serde(default)]
    pub kind: Option<MixedAdditiveKind>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MixedAdditiveKind {
    CompositionAdditive,
}

impl MixedAdditive {
    fn validate(&self) -> Result<()> {
        if let Some(kind) = self.kind {
            ensure!(
                kind == MixedAdditiveKind::CompositionAdditive,
                "unsupported mixed predictor kind {kind:?}"
            );
        }
        Ok(())
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
    fn validate(&self, scheduler: &SchedulerProfile) -> Result<()> {
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
    FixedStep,
    PrefillPiecewise,
    DecodeSurface,
    MixedAdditive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepTimingEstimate {
    pub duration_us: u64,
    pub source: StepTimingSource,
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
        .context("predictor interpolation multiplication overflow")?;
    let rounded = scaled
        .checked_add(u128::from(bracket.denominator / 2))
        .context("predictor interpolation rounding overflow")?
        / u128::from(bracket.denominator);
    let adjustment = u64::try_from(rounded).context("predictor interpolation result overflow")?;
    if upper >= lower {
        lower
            .checked_add(adjustment)
            .context("predictor interpolation addition overflow")
    } else {
        lower
            .checked_sub(adjustment)
            .context("predictor interpolation subtraction overflow")
    }
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
            "schema_version": 3,
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
                "model": {
                    "kind": "fixed_step",
                    "duration_us": 175
                }
            }
        })
    }

    fn parse(value: &Value) -> Result<EngineProfile> {
        EngineProfile::from_json_slice(&serde_json::to_vec(value)?)
    }

    #[test]
    fn profile_rejects_invalid_scheduler_and_grid_payload() {
        let mut bad_chunk = profile_json();
        bad_chunk["scheduler"]["prefill"]["max_chunk_tokens"] = json!(17);
        assert!(
            parse(&bad_chunk)
                .unwrap_err()
                .to_string()
                .contains("cannot exceed max_num_batched_tokens")
        );

        let mut grid_payload = profile_json();
        grid_payload["predictor"]["grid"] = json!({
            "decode_reqs": [0, 2],
            "sum_decode_ctx_tokens": [0, 20],
            "prefill_tokens_in_step": [0, 10],
            "step_duration_us": [10, 80, 70, 140, 210, 280, 270, 340]
        });
        assert!(parse(&grid_payload).is_err());

        grid_payload["schema_version"] = json!(1);
        let error = parse(&grid_payload).unwrap_err().to_string();
        assert!(error.contains("schema_version 1"));
        assert!(error.contains("migrate to schema_version 3"));
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
}
