use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

const MODEL_PATH_ENV: &str = "PEGAINFER_TEST_MODEL_PATH";
#[allow(dead_code)]
const FRONTEND_MODEL_PATH_ENV: &str = "PEGAINFER_TEST_FRONTEND_MODEL_PATH";

pub(crate) fn model_path_or_skip(test_name: &str) -> Option<String> {
    fixture_path_from_env_or_skip(MODEL_PATH_ENV, test_name)
}

#[allow(dead_code)]
pub(crate) fn frontend_model_path_or_skip(
    engine_model_path: &Path,
    test_name: &str,
) -> Option<String> {
    match std::env::var(FRONTEND_MODEL_PATH_ENV) {
        Ok(path) => validated_fixture_path_or_skip(FRONTEND_MODEL_PATH_ENV, path, test_name),
        Err(std::env::VarError::NotPresent) => {
            Some(engine_model_path.to_string_lossy().into_owned())
        }
        Err(std::env::VarError::NotUnicode(_)) => skip(
            test_name,
            &format!("{FRONTEND_MODEL_PATH_ENV} is not valid UTF-8"),
        ),
    }
}

fn fixture_path_from_env_or_skip(env: &str, test_name: &str) -> Option<String> {
    match std::env::var(env) {
        Ok(path) => validated_fixture_path_or_skip(env, path, test_name),
        Err(std::env::VarError::NotPresent) => skip(
            test_name,
            &format!("{env} is not set; point it at a public Qwen3.5 model fixture"),
        ),
        Err(std::env::VarError::NotUnicode(_)) => {
            skip(test_name, &format!("{env} is not valid UTF-8"))
        }
    }
}

fn validated_fixture_path_or_skip(env: &str, path: String, test_name: &str) -> Option<String> {
    match validated_fixture_path(env, path) {
        Ok(path) => Some(path),
        Err(error) => skip(test_name, &format!("{error:#}")),
    }
}

pub(crate) fn validated_fixture_path(env: &str, path: String) -> Result<String> {
    ensure!(!path.trim().is_empty(), "{env} is empty");
    let config_path = Path::new(&path).join("config.json");
    let raw = std::fs::read(&config_path)
        .with_context(|| format!("cannot read {} from {env}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&raw)
        .with_context(|| format!("{} from {env} is not valid JSON", config_path.display()))?;
    let root_model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let text_model_type = config
        .pointer("/text_config/model_type")
        .and_then(serde_json::Value::as_str);
    ensure!(
        root_model_type == Some("qwen3_5") || text_model_type == Some("qwen3_5_text"),
        "{} from {env} is not a Qwen3.5 config",
        config_path.display()
    );
    Ok(path)
}

fn skip<T>(test_name: &str, reason: &str) -> Option<T> {
    eprintln!("SKIP {test_name}: {reason}");
    None
}
