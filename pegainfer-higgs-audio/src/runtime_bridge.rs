use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use half::bf16;
use pegainfer_core::weight_loader::TensorNameAliases;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

use crate::config::HiggsConfig;
use crate::load_plan::HiggsRuntimeLoadPlan;
use crate::materialize_qwen3::write_qwen3_config_view;
use crate::one_step_actual::OneStepActualSummary;
use crate::one_step_actual::OneStepAudioPrediction;
use crate::one_step_actual::PromptTensors;
use crate::one_step_actual::compute_one_step_audio_prediction;
use crate::one_step_actual::compute_one_step_audio_prediction_gpu_bf16;
use crate::one_step_actual::load_fused_audio_head_bf16;
use crate::one_step_actual::load_prompt_from_golden;
use crate::one_step_actual::write_one_step_actual_prediction;
use crate::weights::HiggsWeightManifest;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioHeadBackend {
    CudaBf16,
    CpuFp32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HiggsRuntimeSource<'a> {
    Qwen3BodyView { qwen3_body_dir: &'a Path },
    Qwen3ConfigAlias { qwen3_config_dir: &'a Path },
    AutoConfigAlias { qwen3_config_dir: &'a Path },
}

/// Higgs Audio runtime surface backed by the existing Qwen3 executor.
///
/// The current implementation owns prefill and prompt-session smoke paths. Full
/// audio decode continuation is intentionally not exposed until the Higgs crate
/// owns the audio-codebook feedback semantics.
pub struct HiggsAudioRuntime {
    executor: Qwen3Executor,
    audio_head: Vec<bf16>,
    audio_head_backend: AudioHeadBackend,
    device_ordinal: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsAudioPrefill {
    pub prompt_tokens: usize,
    pub final_hidden_bf16: Vec<bf16>,
    pub audio: OneStepAudioPrediction,
}

/// Compatibility alias for early one-step gate callers.
pub type HiggsOneStepRuntime = HiggsAudioRuntime;

/// Compatibility alias for early one-step gate callers.
pub type HiggsOneStepPrefill = HiggsAudioPrefill;

/// Higgs-owned handle for a retained prompt KV session.
///
/// The backing executor currently stores the session under a Qwen3 request id,
/// but callers should treat this as a Higgs Audio session id. Audio-codebook
/// continuation is intentionally not exposed through this handle yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HiggsPromptSession {
    request_id: RequestId,
}

impl HiggsPromptSession {
    pub fn new(id: u64) -> Self {
        Self {
            request_id: RequestId::new(id),
        }
    }

    pub fn id(self) -> u64 {
        self.request_id.get()
    }

    fn request_id(self) -> RequestId {
        self.request_id
    }
}

impl From<RequestId> for HiggsPromptSession {
    fn from(request_id: RequestId) -> Self {
        Self { request_id }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsPromptSessionPrefill {
    pub session: HiggsPromptSession,
    pub prompt_tokens: usize,
    pub final_hidden_bf16: Vec<bf16>,
    pub audio: OneStepAudioPrediction,
}

impl HiggsAudioRuntime {
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        source: HiggsRuntimeSource<'_>,
        audio_head_backend: AudioHeadBackend,
        device_ordinal: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let executor = load_qwen3_executor(model_dir, source, device_ordinal)?;
        let audio_head = load_fused_audio_head_bf16(model_dir)?;
        Ok(Self {
            executor,
            audio_head,
            audio_head_backend,
            device_ordinal,
        })
    }

    pub fn dump_one_step_actual(
        &mut self,
        golden: impl AsRef<Path>,
        out: impl AsRef<Path>,
    ) -> Result<OneStepActualSummary> {
        let prompt = load_prompt_from_golden(golden)?;
        let prompt_ids = prompt.prompt_ids()?;
        let prefill = self.prefill_audio_from_prompt_ids(&prompt_ids)?;
        write_one_step_actual_prediction(out, &prompt, &prefill.final_hidden_bf16, &prefill.audio)
    }

    pub fn prefill_audio_from_prompt(
        &mut self,
        prompt: &PromptTensors,
    ) -> Result<HiggsAudioPrefill> {
        let prompt_ids = prompt.prompt_ids()?;
        self.prefill_audio_from_prompt_ids(&prompt_ids)
    }

    pub fn prefill_audio_from_prompt_ids(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<HiggsAudioPrefill> {
        let hidden = self
            .executor
            .prefill_last_hidden_bf16(prompt_ids.to_vec())?
            .hidden_bf16;
        let audio = match self.audio_head_backend {
            AudioHeadBackend::CudaBf16 => compute_one_step_audio_prediction_gpu_bf16(
                &hidden,
                &self.audio_head,
                self.device_ordinal,
            )?,
            AudioHeadBackend::CpuFp32 => {
                compute_one_step_audio_prediction(&hidden, &self.audio_head)?
            }
        };
        Ok(HiggsAudioPrefill {
            prompt_tokens: prompt_ids.len(),
            final_hidden_bf16: hidden,
            audio,
        })
    }

    pub fn prefill_prompt_session_from_prompt_ids(
        &mut self,
        request_id: RequestId,
        prompt_ids: &[u32],
    ) -> Result<HiggsPromptSessionPrefill> {
        self.prefill_prompt_session(request_id.into(), prompt_ids)
    }

    pub fn prefill_prompt_session(
        &mut self,
        session: HiggsPromptSession,
        prompt_ids: &[u32],
    ) -> Result<HiggsPromptSessionPrefill> {
        let retained = self
            .executor
            .prefill_last_hidden_bf16_retained_prompt(session.request_id(), prompt_ids.to_vec())?;
        let audio = match self.audio_head_backend {
            AudioHeadBackend::CudaBf16 => compute_one_step_audio_prediction_gpu_bf16(
                &retained.hidden_bf16,
                &self.audio_head,
                self.device_ordinal,
            )?,
            AudioHeadBackend::CpuFp32 => {
                compute_one_step_audio_prediction(&retained.hidden_bf16, &self.audio_head)?
            }
        };
        Ok(HiggsPromptSessionPrefill {
            session: retained.request_id.into(),
            prompt_tokens: prompt_ids.len(),
            final_hidden_bf16: retained.hidden_bf16,
            audio,
        })
    }

    pub fn drop_prompt_session(&mut self, session: impl Into<HiggsPromptSession>) -> Result<()> {
        self.executor.drop_request(session.into().request_id())
    }
}

fn load_qwen3_executor(
    model_dir: &Path,
    source: HiggsRuntimeSource<'_>,
    device_ordinal: usize,
) -> Result<Qwen3Executor> {
    match source {
        HiggsRuntimeSource::Qwen3BodyView { qwen3_body_dir } => {
            let qwen3_body_dir = path_str(qwen3_body_dir, "qwen3 body dir")?;
            Qwen3Executor::from_runtime(qwen3_body_dir, false, &[device_ordinal])
        }
        HiggsRuntimeSource::Qwen3ConfigAlias { qwen3_config_dir } => {
            load_qwen3_executor_from_alias_config(model_dir, qwen3_config_dir, device_ordinal)
        }
        HiggsRuntimeSource::AutoConfigAlias { qwen3_config_dir } => {
            prepare_qwen3_config_view(model_dir, qwen3_config_dir)?;
            load_qwen3_executor_from_alias_config(model_dir, qwen3_config_dir, device_ordinal)
        }
    }
}

fn load_qwen3_executor_from_alias_config(
    model_dir: &Path,
    qwen3_config_dir: &Path,
    device_ordinal: usize,
) -> Result<Qwen3Executor> {
    let qwen3_config_dir = path_str(qwen3_config_dir, "qwen3 config dir")?;
    let model_dir_str = path_str(model_dir, "model dir")?;
    Qwen3Executor::from_runtime_with_weight_source(
        qwen3_config_dir,
        Some(model_dir_str),
        qwen3_tensor_name_aliases(model_dir)?,
        false,
        &[device_ordinal],
    )
}

fn prepare_qwen3_config_view(model_dir: &Path, qwen3_config_dir: &Path) -> Result<()> {
    let config = HiggsConfig::from_model_dir(model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(model_dir)?;
    let plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    write_qwen3_config_view(qwen3_config_dir, &config, &plan)?;
    Ok(())
}

fn qwen3_tensor_name_aliases(model_dir: &Path) -> Result<TensorNameAliases> {
    let config = HiggsConfig::from_model_dir(model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(model_dir)?;
    let plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    Ok(TensorNameAliases::new(
        plan.qwen3_tensor_aliases()?.into_iter().collect(),
    ))
}

fn path_str<'a>(path: &'a Path, label: &str) -> Result<&'a str> {
    path.to_str()
        .with_context(|| format!("{label} must be valid UTF-8"))
}
