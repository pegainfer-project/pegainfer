//! Model setup and identity checks used only by the crate's acceptance tests.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::parse_for_line;

use crate::Qwen35DecodeOverlap;
use crate::Qwen35SchedulerPolicy;
pub(crate) use crate::test_fixture_common::load_tokenizer;
pub(crate) use crate::test_fixture_common::model_path_or_skip;
pub(crate) use crate::test_fixture_common::tp2_device_ordinals;
use crate::weights::ModelRuntimeConfig;
use crate::weights::Qwen35Model;

#[derive(Debug)]
pub(crate) enum GdnAcceptance {
    Triton,
    Candidate { object_sha256: String },
}

impl GdnAcceptance {
    pub(crate) fn candidate() -> Result<Self> {
        Self::candidate_with_identity(std::env::var("PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256").ok())
    }

    pub(crate) fn candidate_with_identity(identity: Option<String>) -> Result<Self> {
        let object_sha256 = identity
            .context("candidate acceptance requires PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256")?;
        ensure!(
            object_sha256.len() == 64 && object_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "candidate acceptance requires a 64-character hexadecimal object SHA256"
        );
        Ok(Self::Candidate { object_sha256 })
    }

    pub(crate) fn is_candidate(&self) -> bool {
        matches!(self, Self::Candidate { .. })
    }

    pub(crate) fn model_path(&self, test_name: &str) -> Result<Option<String>> {
        const ENV: &str = "PEGAINFER_TEST_MODEL_PATH";
        if !self.is_candidate() {
            return Ok(model_path_or_skip(test_name));
        }
        let path = std::env::var(ENV).context("candidate acceptance requires a model path")?;
        crate::test_fixture_common::model_fixture::validated_fixture_path(ENV, path).map(Some)
    }

    pub(crate) fn load_model(
        &self,
        model_path: &str,
        max_batch: usize,
        max_prefill_tokens: usize,
        policy: Qwen35SchedulerPolicy,
        overlap: Qwen35DecodeOverlap,
    ) -> Result<Qwen35Model> {
        let backend = match self {
            Self::Triton => "triton",
            Self::Candidate { .. } => "flashinfer-candidate",
        };
        let overlap = match overlap {
            Qwen35DecodeOverlap::Off => "off",
            Qwen35DecodeOverlap::SharedSm => "stream",
        };
        let scheduler_policy = match policy {
            Qwen35SchedulerPolicy::Off => "off",
            Qwen35SchedulerPolicy::Auto => "auto",
        };
        let max_batch = max_batch.to_string();
        let max_prefill_tokens = max_prefill_tokens.to_string();
        let (shared, matches, provided) = parse_for_line(
            &crate::model_line::MODEL_LINE,
            &[
                "pegainfer",
                "--model-path",
                model_path,
                "--max-batch",
                &max_batch,
                "--max-prefill-tokens",
                &max_prefill_tokens,
                "--decode-overlap",
                overlap,
                "--qwen35-gdn-backend",
                backend,
                "--qwen35-scheduler-policy",
                scheduler_policy,
            ],
        )?;
        let path = std::path::Path::new(model_path);
        let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        crate::model_line::MODEL_LINE
            .probe(&config)
            .map_err(anyhow::Error::msg)?;
        let ctx = LaunchContext {
            model_path: path,
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        crate::model_line::MODEL_LINE.validate(&ctx, &provided)?;
        let options = crate::model_line::launch_options(&ctx);
        ensure!(
            options.tp_size == 1,
            "rank-local model loading requires tp_size=1"
        );
        let model = Qwen35Model::from_safetensors_with_runtime_and_capacity(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.cuda_graph,
                device_ordinal: options.device_ordinal,
                gdn_backend: options.gdn_backend,
                tensor_parallel: None,
            },
            options.max_batch,
        )?;
        self.validate_model(&model)?;
        Ok(model)
    }

    pub(crate) fn validate_model(&self, model: &Qwen35Model) -> Result<()> {
        match self {
            Self::Triton => ensure!(
                model.flashinfer_gdn.is_none(),
                "Triton acceptance unexpectedly loaded a FlashInfer candidate"
            ),
            Self::Candidate { object_sha256 } => {
                let backend = model
                    .flashinfer_gdn
                    .as_ref()
                    .context("candidate acceptance loaded no FlashInfer backend")?;
                ensure!(
                    backend.artifact_sha256() == object_sha256,
                    "candidate acceptance artifact identity mismatch: expected {object_sha256}, loaded {}",
                    backend.artifact_sha256()
                );
                eprintln!(
                    "CANDIDATE_MODEL_IDENTITY_OK object_sha256={}",
                    backend.artifact_sha256()
                );
            }
        }
        Ok(())
    }

    pub(crate) fn launch_engine(
        &self,
        model_path: &str,
        max_batch: usize,
        max_prefill_tokens: usize,
        policy: Qwen35SchedulerPolicy,
        overlap: Qwen35DecodeOverlap,
    ) -> Result<EngineHandle> {
        // The identity-checked model is the one moved into the real scheduler.
        let model = self.load_model(model_path, max_batch, max_prefill_tokens, policy, overlap)?;
        crate::scheduler::start_with_capacity_and_policy(
            model,
            42,
            max_batch,
            max_prefill_tokens,
            policy,
            overlap,
        )
    }
}
