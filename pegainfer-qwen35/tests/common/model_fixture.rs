use std::path::Path;

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
    if path.trim().is_empty() {
        return skip(test_name, &format!("{env} is empty"));
    }

    let config_path = Path::new(&path).join("config.json");
    let raw = match std::fs::read(&config_path) {
        Ok(raw) => raw,
        Err(err) => {
            return skip(
                test_name,
                &format!("cannot read {} from {env}: {err}", config_path.display()),
            );
        }
    };
    let config: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(config) => config,
        Err(err) => {
            return skip(
                test_name,
                &format!(
                    "{} from {env} is not valid JSON: {err}",
                    config_path.display()
                ),
            );
        }
    };
    let root_model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let text_model_type = config
        .pointer("/text_config/model_type")
        .and_then(serde_json::Value::as_str);
    if root_model_type != Some("qwen3_5") && text_model_type != Some("qwen3_5_text") {
        return skip(
            test_name,
            &format!(
                "{} from {env} is not a Qwen3.5 config",
                config_path.display()
            ),
        );
    }

    Some(path)
}

fn skip<T>(test_name: &str, reason: &str) -> Option<T> {
    eprintln!("SKIP {test_name}: {reason}");
    None
}
