use std::sync::Arc;

use vllm_text::Error;
use vllm_text::Result;
use vllm_text::backend::hf::ResolvedModelFiles;
use vllm_text::backend::hf::TokenizerSource;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::tokenizer::HuggingFaceTokenizer;
use vllm_text::tokenizer::TekkenTokenizer;
use vllm_text::tokenizer::TiktokenTokenizer;

pub(crate) mod model_fixture;

pub(crate) use model_fixture::model_path_or_skip;

#[allow(dead_code)]
pub(crate) fn gdn_backend() -> String {
    match std::env::var("PEGAINFER_TEST_QWEN35_GDN_BACKEND") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "triton".to_string(),
        Err(error) => panic!("invalid PEGAINFER_TEST_QWEN35_GDN_BACKEND: {error}"),
    }
}

#[allow(dead_code)]
pub(crate) fn with_launch_context<T>(
    model_path: &str,
    max_batch: usize,
    max_prefill_tokens: usize,
    overlap: pegainfer_qwen35::Qwen35DecodeOverlap,
    operation: impl FnOnce(&pegainfer_frontend::model_line::LaunchContext<'_>) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    use pegainfer_frontend::model_line::LaunchContext;
    use pegainfer_frontend::model_line::ModelLine;
    use pegainfer_frontend::model_line::parse_for_line;
    use pegainfer_qwen35::model_line::MODEL_LINE;

    let backend = gdn_backend();
    let max_batch = max_batch.to_string();
    let max_prefill_tokens = max_prefill_tokens.to_string();
    let overlap = match overlap {
        pegainfer_qwen35::Qwen35DecodeOverlap::Off => "off",
        pegainfer_qwen35::Qwen35DecodeOverlap::SharedSm => "stream",
    };
    let (shared, matches, provided) = parse_for_line(
        &MODEL_LINE,
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
            &backend,
        ],
    )?;
    let model_path = std::path::Path::new(model_path);
    let config = serde_json::from_slice(&std::fs::read(model_path.join("config.json"))?)?;
    MODEL_LINE.probe(&config).map_err(anyhow::Error::msg)?;
    let ctx = LaunchContext {
        model_path,
        config: &config,
        shared: &shared,
        matches: &matches,
    };
    MODEL_LINE.validate(&ctx, &provided)?;
    operation(&ctx)
}

#[allow(dead_code)]
pub(crate) fn launch_engine(
    model_path: &str,
    max_batch: usize,
    max_prefill_tokens: usize,
    overlap: pegainfer_qwen35::Qwen35DecodeOverlap,
) -> anyhow::Result<pegainfer_frontend::engine::EngineHandle> {
    use pegainfer_frontend::engine::LaunchedEngine;
    use pegainfer_frontend::model_line::ModelLine;

    with_launch_context(model_path, max_batch, max_prefill_tokens, overlap, |ctx| {
        match pegainfer_qwen35::model_line::MODEL_LINE.launch(ctx)? {
            LaunchedEngine::Handle(handle) => Ok(handle),
            LaunchedEngine::Stepped(_) => anyhow::bail!("Qwen3.5 must launch its scheduler handle"),
        }
    })
}

#[allow(dead_code)]
pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    try_load_tokenizer(model_path)
        .unwrap_or_else(|err| panic!("Failed to load tokenizer for {model_path}: {err}"))
}

// vllm-text exposes model-file resolution as async even though the local
// directory path (all we ever use) is synchronous; bridge it with a throwaway
// current-thread runtime.
#[allow(dead_code)]
fn try_load_tokenizer(model_path: &str) -> Result<DynTokenizer> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(Error::Tokenizer(
            "load_tokenizer cannot be called from inside an active Tokio runtime".to_string(),
        ));
    }
    let files = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            Error::Tokenizer(format!("failed to build tokenizer resolver runtime: {err}"))
        })?
        .block_on(ResolvedModelFiles::new(model_path))?;
    match &files.tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

#[allow(dead_code)]
pub(crate) fn tp2_device_ordinals() -> Vec<usize> {
    const ENV: &str = "PEGAINFER_TEST_TP_DEVICES";
    let Ok(value) = std::env::var(ENV) else {
        return vec![0, 1];
    };

    let devices: Vec<usize> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .unwrap_or_else(|err| panic!("{ENV} must be comma-separated CUDA ordinals: {err}"))
        })
        .collect();

    assert_eq!(
        devices.len(),
        2,
        "{ENV} must specify exactly two CUDA ordinals for TP2, e.g. 0,1 or 2,3"
    );
    assert_ne!(
        devices[0], devices[1],
        "{ENV} must specify two distinct CUDA ordinals for TP2"
    );
    devices
}
