//! The checkpoint's EOS set, resolved the same way for every model line.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

/// EOS ids from `generation_config.json`, then `config.json`, then its
/// `text_config`; the first that declares `eos_token_id` wins. The field may
/// be one id or a list, duplicates are dropped in order, and the load fails on
/// a checkpoint that declares none, an empty list, an entry that is not a u32
/// or an id outside `vocab_size`.
pub fn eos_token_ids(model_path: impl AsRef<Path>, vocab_size: usize) -> Result<Vec<u32>> {
    let model_path = model_path.as_ref();
    let generation_path = model_path.join("generation_config.json");
    let config_path = model_path.join("config.json");
    let generation = read_optional_json(&generation_path)?;
    let config = read_optional_json(&config_path)?;
    let text_config = config.as_ref().and_then(|json| json.get("text_config"));
    let sources = [
        (generation_path.display().to_string(), generation.as_ref()),
        (config_path.display().to_string(), config.as_ref()),
        (
            format!("{} text_config", config_path.display()),
            text_config,
        ),
    ];
    let Some((source, field)) = sources.into_iter().find_map(|(source, json)| {
        let field = json?.get("eos_token_id")?;
        (!field.is_null()).then_some((source, field))
    }) else {
        bail!("{} declares no eos_token_id", model_path.display());
    };

    let entries = match field {
        Value::Array(entries) => entries.iter().collect::<Vec<_>>(),
        other => vec![other],
    };
    ensure!(!entries.is_empty(), "{source} lists no eos_token_id");
    let mut ids = Vec::with_capacity(entries.len());
    for entry in entries {
        let id = entry
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .with_context(|| format!("{source} eos_token_id entry {entry} is not a u32"))?;
        ensure!(
            (id as usize) < vocab_size,
            "{source} eos_token_id {id} is outside the vocabulary of {vocab_size}"
        );
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn read_optional_json(path: &Path) -> Result<Option<Value>> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .with_context(|| format!("parse {}", path.display())),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).with_context(|| format!("read {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::eos_token_ids;

    fn checkpoint(
        generation: Option<serde_json::Value>,
        config: &serde_json::Value,
    ) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        if let Some(generation) = generation {
            fs::write(
                dir.path().join("generation_config.json"),
                generation.to_string(),
            )
            .unwrap();
        }
        fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        dir
    }

    #[test]
    fn the_first_declaring_file_wins_in_every_shape_the_lines_load() {
        let cases = [
            // generation_config over a config that names a different id (Qwen3, Kimi-K2).
            (
                Some(json!({"eos_token_id": [7, 5, 7]})),
                json!({"eos_token_id": 6}),
                vec![7, 5],
            ),
            (
                Some(json!({"eos_token_id": 5})),
                json!({"eos_token_id": 6}),
                vec![5],
            ),
            (None, json!({"eos_token_id": [1, 2]}), vec![1, 2]),
            // A null top level falls through to text_config (Qwen3.5 without generation_config).
            (
                None,
                json!({"eos_token_id": null, "text_config": {"eos_token_id": 9}}),
                vec![9],
            ),
            // A generation_config without the field, or with null, falls through.
            (
                Some(json!({"bos_token_id": 0})),
                json!({"eos_token_id": 4}),
                vec![4],
            ),
            (
                Some(json!({"eos_token_id": null})),
                json!({"eos_token_id": 4}),
                vec![4],
            ),
        ];
        for (generation, config, want) in cases {
            let dir = checkpoint(generation, &config);
            assert_eq!(eos_token_ids(dir.path(), 16).unwrap(), want);
        }
    }

    #[test]
    fn a_missing_empty_malformed_or_out_of_vocab_set_fails_the_load() {
        let cases = [
            (None, json!({}), "declares no eos_token_id"),
            (
                Some(json!({"eos_token_id": []})),
                json!({"eos_token_id": 4}),
                "lists no eos_token_id",
            ),
            (
                Some(json!({"eos_token_id": [3, "eos"]})),
                json!({}),
                "is not a u32",
            ),
            (Some(json!({"eos_token_id": -1})), json!({}), "is not a u32"),
            (
                None,
                json!({"eos_token_id": 16}),
                "outside the vocabulary of 16",
            ),
        ];
        for (generation, config, want) in cases {
            let dir = checkpoint(generation, &config);
            let err = eos_token_ids(dir.path(), 16).unwrap_err().to_string();
            assert!(err.contains(want), "{err}");
        }
    }
}
