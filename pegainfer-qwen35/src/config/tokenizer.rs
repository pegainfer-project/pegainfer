//! Frontend tokenizer schema and decodable-vocab width for Qwen3.5.
//!
//! The pinned frontend serves from the join of `tokenizer.json` (vocab +
//! `added_tokens`) and `tokenizer_config.json` (`added_tokens_decoder`). We
//! mirror that merge to compute the widest decodable token id, then bound the
//! output selection to a dense prefix of that id space.

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
/// frontend. None of the fields are read by [`tokenizer_effective_vocab`] —
/// the contract's entire job is the fail-closed type-check, so this schema is
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
pub(crate) fn tokenizer_effective_vocab(model_path: &str) -> Result<usize> {
    let path = format!("{}/tokenizer.json", model_path);
    let content =
        fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
    let tj: TokenizerJsonIds =
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("cannot parse {path}: {e}"))?;
    anyhow::ensure!(!tj.model.vocab.is_empty(), "{path} model.vocab is empty");
    let mut ids: HashSet<u32> = tj.model.vocab.into_values().collect();
    ids.extend(tj.added_tokens.iter().filter_map(|t| t.id));

    let config_path = format!("{}/tokenizer_config.json", model_path);
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
