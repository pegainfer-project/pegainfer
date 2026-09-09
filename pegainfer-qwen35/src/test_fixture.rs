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
use crate::weights::Qwen35Model;

pub(crate) enum GdnAcceptance {
    Triton,
    Candidate { object_sha256: String },
}

impl GdnAcceptance {
    /// Candidate entry points never infer selection from an optional variable.
    pub(crate) fn candidate() -> Self {
        const ENV: &str = "PEGAINFER_TEST_QWEN35_GDN_OBJECT_SHA256";
        let object_sha256 = std::env::var(ENV)
            .unwrap_or_else(|error| panic!("candidate acceptance requires {ENV}: {error}"));
        assert!(
            object_sha256.len() == 64 && object_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "candidate acceptance requires a 64-character hexadecimal object SHA256"
        );
        Self::Candidate { object_sha256 }
    }

    pub(crate) fn is_candidate(&self) -> bool {
        matches!(self, Self::Candidate { .. })
    }

    pub(crate) fn model_path(&self, test_name: &str) -> Option<String> {
        let path = model_path_or_skip(test_name);
        assert!(
            !self.is_candidate() || path.is_some(),
            "candidate acceptance requires a readable Qwen3.5 model fixture"
        );
        path
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
        let model = Qwen35Model::from_safetensors_with_launch_options(
            model_path,
            &crate::model_line::launch_options(&ctx),
        )?;
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
        Ok(model)
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
