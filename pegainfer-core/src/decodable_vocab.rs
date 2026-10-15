//! Frontend tokenizer schema and decodable-vocabulary width.
//!
//! The pinned frontend serves from the join of `tokenizer.json` (vocab +
//! `added_tokens`) and `tokenizer_config.json` (`added_tokens_decoder`). We
//! mirror that merge to compute the widest decodable token id, then bound output
//! selection to a dense prefix of that id space.

use std::collections::HashSet;
use std::fs;

use anyhow::Result;
use log::warn;
use serde::Deserialize;

/// One `added_tokens_decoder` entry in `tokenizer_config.json`.
///
/// An explicitly-owned frontend compatibility contract. The frontend's decoder
/// schema is bool-heavy, and we parse it with these typed fields purely to make
/// the whole-file typed parse fail-closed on a malformed entry (e.g. a `special`
/// that is not a bool): then every decoder token is dropped, mirroring the
/// frontend. None of the fields are read by [`decodable_vocab_width`] — the
/// contract's entire job is the fail-closed type-check, so this schema is
/// intentionally dead in the code path and its fields exist to shape the parse.
#[allow(clippy::struct_excessive_bools)]
#[allow(dead_code)]
#[derive(Deserialize)]
struct FrontendAddedToken {
    #[serde(default)]
    id: Option<u32>,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct TokenizerJsonIds {
    model: TokenizerModelIds,
    #[serde(default)]
    added_tokens: Vec<FrontendAddedToken>,
}

#[derive(Deserialize)]
struct TokenizerModelIds {
    vocab: std::collections::HashMap<String, u32>,
}

#[derive(Deserialize)]
struct TokenizerConfigIds {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, FrontendAddedToken>,
}

/// Width of the frontend-decodable id space, mirroring the pinned frontend's
/// merge: `tokenizer.json` vocab and added_tokens (fatal on parse failure -
/// the frontend cannot serve without it) plus `tokenizer_config.json`
/// added_tokens_decoder (whole-file typed parse; failure drops all decoder
/// tokens with a warning, unparsable keys are skipped per entry). The ids
/// must form a dense prefix - a row-range selection bound cannot mask holes -
/// so a sparse id space fails the load instead of silently truncating the
/// output space.
pub fn decodable_vocab_width(model_path: &str) -> Result<usize> {
    let path = format!("{model_path}/tokenizer.json");
    let content =
        fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
    let tj: TokenizerJsonIds =
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("cannot parse {path}: {e}"))?;
    anyhow::ensure!(!tj.model.vocab.is_empty(), "{path} model.vocab is empty");
    let mut ids: HashSet<u32> = tj.model.vocab.into_values().collect();
    ids.extend(tj.added_tokens.iter().filter_map(|t| t.id));

    let config_path = format!("{model_path}/tokenizer_config.json");
    if let Ok(text) = fs::read_to_string(&config_path) {
        match serde_json::from_str::<TokenizerConfigIds>(&text) {
            Ok(cfg) => ids.extend(
                cfg.added_tokens_decoder
                    .keys()
                    .filter_map(|k| k.parse::<u32>().ok()),
            ),
            Err(e) => warn!(
                "cannot parse {config_path}: {e}; skipping its added tokens like the frontend does"
            ),
        }
    }

    let width = ids.len();
    let max_id =
        *ids.iter()
            .max()
            .ok_or_else(|| anyhow::anyhow!("tokenizer vocab must be non-empty"))? as usize;
    anyhow::ensure!(
        max_id + 1 == width,
        "tokenizer id space is not dense (max id {max_id}, {width} distinct ids); \
         a row-range selection bound cannot mask holes"
    );
    Ok(width)
}

/// Rows the output projection computes for a `decodable` prefix of a
/// `vocab`-row checkpoint. The GEMM's M dimension and the logits buffer's
/// leading dimension both take this width, and an unaligned one (Qwen3.5-4B
/// decodes 248077 ids) forces cublasLt onto an align-1 kernel, so it is rounded
/// up to a 128-token tile and capped at the checkpoint's rows. The rows past
/// `decodable` must be masked before selection.
pub fn aligned_logits_rows(decodable: usize, vocab: usize) -> usize {
    decodable.next_multiple_of(128).min(vocab)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::decodable_vocab_width;

    fn dir_with(json: &str, config: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        fs::write(dir.path().join("tokenizer_config.json"), config).unwrap();
        dir
    }

    #[test]
    fn effective_vocab_is_the_dense_decodable_width() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1, "c": 2 } }, "added_tokens": [ { "id": 3, "content": "<x>" } ] }"#,
            r#"{ "added_tokens_decoder": { "4": { "content": "<z>" }, "5": { "content": "<w>" }, "x": { "content": "<bad-key>" } } }"#,
        );
        assert_eq!(
            decodable_vocab_width(dir.path().to_str().unwrap()).unwrap(),
            6
        );
    }

    #[test]
    fn effective_vocab_fails_on_a_sparse_id_space() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#,
            r#"{ "added_tokens_decoder": { "5": { "content": "<z>" } } }"#,
        );
        assert!(decodable_vocab_width(dir.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn one_invalid_decoder_entry_drops_all_decoder_tokens() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#,
            r#"{ "added_tokens_decoder": { "2": { "content": "<z>" }, "3": { "content": "<w>", "special": "not-a-bool" } } }"#,
        );
        assert_eq!(
            decodable_vocab_width(dir.path().to_str().unwrap()).unwrap(),
            2
        );
    }
}
