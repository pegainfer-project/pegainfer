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
