//! Validated geometry for the frozen checkpoint, and the derivations the tensor
//! contract depends on.

use anyhow::Result;
use anyhow::bail;
use serde::Deserialize;

use crate::probe::probe_config_json;

/// The immutable revision every constant and derivation in this crate was
/// measured against: `Qwen/Qwen3.8-Flash-Next`, 131 shards, 360.0 GB,
/// `license: other`. A different revision is a different contract.
pub const FROZEN_REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";

/// sha256 of the frozen revision's `config.json`, which is committed verbatim at
/// `tests/frozen_config.json` and re-hashed by a test so the fixture cannot drift
/// from the revision it claims to be.
pub const FROZEN_CONFIG_SHA256: &str =
    "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b";

/// The transformers release the reference implementation is read from.
///
/// `qwen4_exp` first ships in **5.16.0**: the `v5.8.0`…`v5.15.0` tags carry no
/// `src/transformers/models/qwen4_exp/`, and `v5.16.0` does. The checkpoint's own
/// declared `5.8.0.dev0` is a different fact — the version it was authored
/// against, not one that carries the reference — and conflating the two is what
/// put `5.8.0` here originally.
pub const PINNED_TRANSFORMERS: &str = "5.16.0";

/// `norm_topk_prob` is **absent** from the frozen checkpoint's `config.json` and
/// comes from the upstream default. It decides whether the router renormalizes
/// its top-k weights, so a port that reads only `config.json` silently drops the
/// renormalization and gets plausible, wrong routing. Modelled explicitly here
/// rather than left to serde's silence.
const UPSTREAM_DEFAULT_NORM_TOPK_PROB: bool = true;

/// `seed` is likewise absent upstream-defaulted. It only matters if the n-gram
/// hash multipliers are recomputed rather than read from the checkpoint; this
/// line reads the stored tensor, so the value is recorded, not relied on.
const UPSTREAM_DEFAULT_SEED: i64 = 1234;

/// The nested rope object, which is where the reference reads every rope value
/// from (`modeling_qwen4_exp.py` takes `config.rope_parameters["rope_theta"]`
/// and `[..., "partial_rotary_factor"]`) and therefore the only canonical copy.
///
/// The checkpoint *also* carries a legacy top-level `partial_rotary_factor` that
/// nothing upstream reads. Modelling the nested one is what keeps this layer and
/// the probe on the same value; the probe refuses a checkpoint where the two
/// copies disagree rather than letting readers pick differently.
#[derive(Debug, Deserialize)]
pub(crate) struct RopeParameters {
    pub(crate) partial_rotary_factor: f64,
    /// The rope base. The reference reads it from this same nested object with a
    /// direct index (modeling_qwen4_exp.py:108), so it is required here and
    /// carried, keeping a future rope consumer off a second read of
    /// `config.json` — the same rule `num_experts_per_tok` follows.
    pub(crate) rope_theta: f64,
}

/// Everything the engine reads lives in `text_config`; the outer config carries
/// only `architectures`, the token ids, `tie_word_embeddings` and `vision_config`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct TextConfig {
    pub(crate) num_hidden_layers: usize,
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) layer_types: Vec<String>,

    pub(crate) head_dim: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rope_parameters: RopeParameters,

    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_conv_kernel_dim: usize,

    pub(crate) indexer_n_heads: usize,
    pub(crate) indexer_kv_heads: usize,
    pub(crate) indexer_head_dim: usize,

    pub(crate) num_experts: usize,
    /// Top-k the router keeps. The reference validates `0 < k <= num_experts`
    /// (`configuration_qwen4_exp.py`, `:200`) and the probe mirrors that rule —
    /// carrying the field here is what gives that check a consumer, since a
    /// router that re-read `config.json` on its own would be a second source.
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,

    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,

    pub(crate) ple_layer_ids: Vec<usize>,
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) ple_embed_dim: usize,
    pub(crate) ple_conv_kernel_size: usize,
    pub(crate) ngram_vocab_size_base: usize,
    pub(crate) split_ngram_parts: usize,
    pub(crate) make_ngram_vocab_size_divisible_by: usize,

    #[serde(default = "default_seed")]
    pub(crate) seed: i64,
}

fn default_norm_topk_prob() -> bool {
    UPSTREAM_DEFAULT_NORM_TOPK_PROB
}

fn default_seed() -> i64 {
    UPSTREAM_DEFAULT_SEED
}

/// Validated geometry plus the derivations the tensor contract needs.
#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) text: TextConfig,
    /// Per-layer kind, indexed by layer number.
    pub(crate) layer_kinds: Vec<LayerKind>,
    /// The n-gram table geometry, derived rather than read.
    pub(crate) ngram: NgramTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    LinearAttention,
    FullAttention,
}

impl Config {
    /// Probes first, so a config can never be parsed unprobed.
    pub(crate) fn from_json(json: &serde_json::Value) -> Result<Self> {
        probe_config_json(json)?;
        let text_config = json
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing text_config"))?;
        let text: TextConfig = serde_json::from_value(text_config.clone()).map_err(|e| {
            anyhow::anyhow!("qwen4_exp: text_config is missing a required field: {e}")
        })?;

        let layer_kinds = text
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => LayerKind::FullAttention,
                _ => LayerKind::LinearAttention,
            })
            .collect::<Vec<_>>();

        let ngram = NgramTable::derive(&text)?;
        Ok(Self {
            text,
            layer_kinds,
            ngram,
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.text.hidden_size
    }

    /// Width of the hyper-connection residual stream: every sublayer reads and
    /// writes this, not `hidden_size`.
    pub(crate) fn hc_hidden_size(&self) -> usize {
        self.text.hc_count * self.text.hidden_size
    }

    pub(crate) fn full_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count()
    }

    pub(crate) fn linear_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count()
    }

    /// Gated DeltaNet conv channels: key_dim * 2 + value_dim.
    pub(crate) fn gdn_conv_dim(&self) -> usize {
        self.gdn_key_dim() * 2 + self.gdn_value_dim()
    }

    pub(crate) fn gdn_key_dim(&self) -> usize {
        self.text.linear_key_head_dim * self.text.linear_num_key_heads
    }

    pub(crate) fn gdn_value_dim(&self) -> usize {
        self.text.linear_value_head_dim * self.text.linear_num_value_heads
    }

    /// Rotary dimensions actually applied, out of `head_dim`.
    pub(crate) fn rotary_dim(&self) -> Result<usize> {
        let exact = self.text.head_dim as f64 * self.text.rope_parameters.partial_rotary_factor;
        if exact.fract() != 0.0 || exact as usize == 0 {
            bail!(
                "qwen4_exp: partial_rotary_factor does not give a whole number of rotary dimensions"
            );
        }
        Ok(exact as usize)
    }

    /// **`ple_layer_ids` is 1-indexed.** The frozen checkpoint declares `[2]` and
    /// carries its tensors at `layers.1`; reading the field as a 0-indexed
    /// subscript injects into the wrong layer and still produces fluent, wrong
    /// text. This is the only place the conversion happens.
    pub(crate) fn ple_tensor_layers(&self) -> Result<Vec<usize>> {
        self.text
            .ple_layer_ids
            .iter()
            .map(|id| {
                if *id < 1 || *id > self.text.num_hidden_layers {
                    bail!(
                        "qwen4_exp: ple_layer_ids entry {id} is outside 1..={}",
                        self.text.num_hidden_layers
                    );
                }
                Ok(*id - 1)
            })
            .collect()
    }

    /// The PLE short conv is dilated by `ngram_size`, so its state is wider than
    /// the GDN conv's even though both weight tensors have the same rank.
    pub(crate) fn ple_conv_state_len(&self) -> usize {
        (self.text.ple_conv_kernel_size - 1) * self.text.ngram_size
    }
}

/// The n-gram table geometry, **derived from config alone**. The checkpoint also
/// stores `ngram_heads_vocab_sizes` and `ngram_heads_offsets`, so a loader can
/// cross-check this against the stored vectors; deriving it here is what lets the
/// manifest reject a table whose partition disagrees with its own config before
/// any bytes are uploaded.
///
/// One table, for the one PLE layer the probe enforces: the reference primes
/// head `i` of PLE layer `L` at the global index `L * ngram_heads + i`
/// (modeling_qwen4_exp.py:1037-1039), so a second PLE layer's table is a
/// different prime run this struct does not model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NgramTable {
    /// `(ngram_size - 1) * heads_per_ngram` — one head per (order, head) pair.
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    /// Per-head vocabulary size; each is a distinct prime.
    pub(crate) head_vocab_sizes: Vec<usize>,
    /// Per-head row offset into the flat table.
    pub(crate) head_offsets: Vec<usize>,
    /// Sum of the per-head vocabularies, before padding.
    pub(crate) total_vocab_size: usize,
    /// Row count of the stored embedding, padded up to a multiple of
    /// `make_ngram_vocab_size_divisible_by`.
    pub(crate) padded_vocab_size: usize,
    /// The table ships as this many equal named shards.
    pub(crate) shards: usize,
    pub(crate) rows_per_shard: usize,
}

impl NgramTable {
    fn derive(text: &TextConfig) -> Result<Self> {
        if text.ngram_size < 2 {
            bail!("qwen4_exp: ngram_size must be at least 2");
        }
        let heads = (text.ngram_size - 1) * text.heads_per_ngram;
        if heads == 0 {
            bail!("qwen4_exp: ngram head count derives to zero");
        }
        if !text.ple_embed_dim.is_multiple_of(heads) {
            bail!(
                "qwen4_exp: ple_embed_dim {} is not divisible by the {} derived n-gram heads",
                text.ple_embed_dim,
                heads
            );
        }
        let head_dim = text.ple_embed_dim / heads;

        // Upstream gives head `i` the (i+1)-th prime after `ngram_vocab_size_base
        // - 1`, then lays the heads out contiguously. Reproducing that is what
        // makes the row count derivable rather than read.
        let mut head_vocab_sizes = Vec::with_capacity(heads);
        let mut head_offsets = Vec::with_capacity(heads);
        let mut total = 0usize;
        for head in 0..heads {
            let size = nth_prime_after(text.ngram_vocab_size_base - 1, head + 1);
            head_offsets.push(total);
            total += size;
            head_vocab_sizes.push(size);
        }

        let divisor = text.make_ngram_vocab_size_divisible_by;
        if divisor == 0 {
            bail!("qwen4_exp: make_ngram_vocab_size_divisible_by must be positive");
        }
        let padded = total.div_ceil(divisor) * divisor;
        if text.split_ngram_parts == 0 {
            bail!("qwen4_exp: split_ngram_parts must be positive");
        }
        if !padded.is_multiple_of(text.split_ngram_parts) {
            bail!(
                "qwen4_exp: padded n-gram vocabulary {padded} is not divisible into {} equal shards",
                text.split_ngram_parts
            );
        }
        Ok(Self {
            heads,
            head_dim,
            head_vocab_sizes,
            head_offsets,
            total_vocab_size: total,
            padded_vocab_size: padded,
            shards: text.split_ngram_parts,
            rows_per_shard: padded / text.split_ngram_parts,
        })
    }

    /// Stored bytes of the table at BF16. This is the host-resident tier's size,
    /// and the number the ecosystem's ~51 GB quote halves by assuming fp8.
    pub(crate) fn bf16_bytes(&self) -> u128 {
        self.padded_vocab_size as u128 * self.head_dim as u128 * 2
    }
}

/// The `count`-th prime strictly greater than `start`. Mirrors the reference's
/// `_find_nth_prime_after`; the search window is tiny (a few hundred candidates
/// for a base near 2e7), so trial division by sieved small primes is enough.
fn nth_prime_after(start: usize, count: usize) -> usize {
    fn sieve(limit: usize) -> Vec<usize> {
        let mut composite = vec![false; limit + 1];
        let mut primes = Vec::new();
        for candidate in 2..=limit {
            if composite[candidate] {
                continue;
            }
            primes.push(candidate);
            let mut multiple = candidate.saturating_mul(candidate);
            while multiple <= limit {
                composite[multiple] = true;
                multiple += candidate;
            }
        }
        primes
    }

    let mut found = 0usize;
    // Starts at 1 so the first candidate the loop tests is 2: from 0 it would
    // test 1, and the trial division below accepts it (no prime exceeds its
    // square root, so the prefix is empty and `all` is vacuously true).
    let mut candidate = start.max(1);
    // One sieve covers the whole window in practice: prime gaps near 2e7 are far
    // below the base itself, so `root` barely moves. Re-sieve only when the
    // candidate's square root outgrows the sieve's *limit* — comparing against
    // the largest prime in it instead would re-sieve on every candidate whose
    // `root` is composite (4473 here), ~170x the work for no gain.
    let mut sieved_up_to = integer_sqrt(start + 1) + 1;
    let mut small_primes = sieve(sieved_up_to);
    loop {
        candidate += 1;
        let root = integer_sqrt(candidate) + 1;
        if root > sieved_up_to {
            small_primes = sieve(root);
            sieved_up_to = root;
        }
        if small_primes
            .iter()
            .take_while(|p| *p * **p <= candidate)
            .all(|p| !candidate.is_multiple_of(*p))
        {
            found += 1;
            if found == count {
                return candidate;
            }
        }
    }
}

fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frozen() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap()
    }

    /// The fixture is the frozen revision's real `config.json`, committed so the
    /// derivations below are checked against the checkpoint rather than against a
    /// transcription of it. Pinning its hash means an edit is a conscious act:
    /// bump the revision, re-measure, and update both constants together.
    #[test]
    fn the_committed_fixture_is_the_frozen_revision() {
        use sha2::Digest;
        let raw = include_str!("../tests/frozen_config.json");
        let digest = sha2::Sha256::digest(raw.as_bytes());
        let hex = digest
            .iter()
            .fold(String::with_capacity(64), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            });
        assert_eq!(
            hex, FROZEN_CONFIG_SHA256,
            "fixture drifted from the frozen revision"
        );
        assert_eq!(
            frozen()["transformers_version"].as_str().unwrap(),
            "5.8.0.dev0"
        );
    }

    fn config() -> Config {
        Config::from_json(&frozen()).unwrap()
    }

    #[test]
    fn the_frozen_revision_derives_the_measured_geometry() {
        let cfg = config();
        assert_eq!(cfg.text.num_hidden_layers, 48);
        assert_eq!(cfg.full_attention_layers(), 12);
        assert_eq!(cfg.linear_attention_layers(), 36);
        assert_eq!(cfg.hidden_size(), 2560);
        assert_eq!(cfg.hc_hidden_size(), 10240);
        assert_eq!(cfg.gdn_conv_dim(), 10240);
        assert_eq!(cfg.gdn_key_dim(), 2048);
        assert_eq!(cfg.gdn_value_dim(), 6144);
        assert_eq!(cfg.rotary_dim().unwrap(), 64);
    }

    /// The widths the component bricks build against — #1105 for QSA and its
    /// indexer, #1107 for the MoE, #1108 for the hyper-connections — pinned at the
    /// frozen revision so a silent config edit cannot move them.
    #[test]
    fn the_frozen_revision_pins_the_attention_and_moe_widths() {
        let text = &config().text;
        assert_eq!(text.vocab_size, 248_320);
        assert_eq!(text.num_attention_heads, 24);
        assert_eq!(text.num_key_value_heads, 2);
        // GQA group 12 is not in `SUPPORTED_GQA_GROUP_SIZES`, so captured-graph
        // decode is off the table for this line — the same corner Qwen3.5-27B is
        // in at group 6, and worth knowing before slice B picks a decode path.
        assert_eq!(text.num_attention_heads / text.num_key_value_heads, 12);
        assert_eq!(text.linear_conv_kernel_dim, 4);
        assert_eq!(text.indexer_n_heads, 4);
        assert_eq!(text.indexer_kv_heads, 1);
        assert_eq!(text.indexer_head_dim, 128);
        assert_eq!(text.num_experts, 512);
        assert_eq!(text.num_experts_per_tok, 10);
        assert_eq!(text.moe_intermediate_size, 640);
        assert_eq!(text.shared_expert_intermediate_size, 640);
        assert_eq!(text.hc_lowrank, 320);
        // The rope base the reference indexes out of rope_parameters
        // (modeling_qwen4_exp.py:108). Pinned bit-exactly: the JSON literal
        // must parse to this precise f64, not merely something close.
        assert_eq!(
            text.rope_parameters.rope_theta.to_bits(),
            10_000_000.0f64.to_bits()
        );
    }

    /// `ple_layer_ids: [2]` must land the PLE tensors on `layers.1`, not
    /// `layers.2`. This is the off-by-one that would otherwise produce fluent,
    /// wrong text.
    #[test]
    fn ple_layer_ids_are_one_indexed() {
        assert_eq!(config().ple_tensor_layers().unwrap(), vec![1]);
    }

    #[test]
    fn ple_conv_state_is_dilated() {
        // (kernel 4 - 1) * ngram_size 3, wider than the GDN conv's 3 taps.
        assert_eq!(config().ple_conv_state_len(), 9);
    }

    /// The table geometry is derived from config alone and must reproduce what
    /// the frozen checkpoint actually stores.
    #[test]
    fn the_ngram_table_derivation_matches_the_checkpoint() {
        let ngram = config().ngram;
        assert_eq!(ngram.heads, 16);
        assert_eq!(ngram.head_dim, 160);
        assert_eq!(ngram.head_vocab_sizes.len(), 16);
        assert_eq!(ngram.head_vocab_sizes[0], 20_000_003);
        assert!(
            ngram.head_vocab_sizes.windows(2).all(|w| w[1] > w[0]),
            "per-head vocabularies are increasing primes"
        );
        assert_eq!(ngram.head_offsets[0], 0);
        assert_eq!(ngram.total_vocab_size, 320_001_446);
        assert_eq!(ngram.padded_vocab_size, 320_001_536);
        assert_eq!(ngram.shards, 128);
        assert_eq!(ngram.rows_per_shard, 2_500_012);
    }

    /// The host-resident tier is 102.4 GB at BF16, not the ~51 GB the ecosystem
    /// quotes — that figure is the fp8 footprint.
    #[test]
    fn the_ngram_table_is_102_gb_at_bf16() {
        let bytes = config().ngram.bf16_bytes();
        assert_eq!(bytes, 102_400_491_520);
        // The ~51 GB both vLLM and SGLang quote for this table is the fp8
        // footprint; at the shipped BF16 it is double, and every host-RAM budget
        // that used the smaller number is 2x under.
        assert!(
            bytes > 2 * 51_000_000_000,
            "{bytes} bytes should be roughly double the quoted fp8 figure"
        );
    }

    #[test]
    fn the_absent_config_fields_take_their_upstream_defaults() {
        let cfg = config();
        // Neither field is in the frozen config.json; both come from the
        // transformers defaults, and dropping the first would silently skip the
        // router's top-k renormalization.
        assert!(cfg.text.norm_topk_prob);
        assert_eq!(cfg.text.seed, 1234);
        assert!(
            !frozen()["text_config"]
                .as_object()
                .unwrap()
                .contains_key("norm_topk_prob")
        );
        assert!(
            !frozen()["text_config"]
                .as_object()
                .unwrap()
                .contains_key("seed")
        );
    }

    #[test]
    fn an_explicit_norm_topk_prob_overrides_the_default() {
        let mut json = frozen();
        json["text_config"]["norm_topk_prob"] = serde_json::json!(false);
        assert!(!Config::from_json(&json).unwrap().text.norm_topk_prob);
    }

    #[test]
    fn a_table_that_cannot_shard_evenly_is_refused() {
        let mut json = frozen();
        // 3 parts cannot divide a 128-padded row count.
        json["text_config"]["split_ngram_parts"] = serde_json::json!(3);
        let err = Config::from_json(&json).unwrap_err().to_string();
        assert!(err.contains("equal shards"), "{err}");
    }

    #[test]
    fn an_embed_dim_that_cannot_split_across_heads_is_refused() {
        let mut json = frozen();
        json["text_config"]["ple_embed_dim"] = serde_json::json!(2561);
        let err = Config::from_json(&json).unwrap_err().to_string();
        assert!(err.contains("ple_embed_dim"), "{err}");
    }

    #[test]
    fn an_unprobed_config_is_never_parsed() {
        let mut json = frozen();
        json["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        let err = Config::from_json(&json).unwrap_err().to_string();
        assert!(err.contains("rope_type"), "{err}");
    }

    #[test]
    fn nth_prime_after_matches_the_reference() {
        assert_eq!(nth_prime_after(19_999_999, 1), 20_000_003);
        assert_eq!(nth_prime_after(19_999_999, 2), 20_000_023);
        assert_eq!(nth_prime_after(19_999_999, 16), 20_000_171);
        assert_eq!(nth_prime_after(1, 1), 2);
        assert_eq!(nth_prime_after(2, 3), 7);
        // The search window starts below the smallest prime; 1 is not one.
        assert_eq!(nth_prime_after(0, 1), 2);
        assert_eq!(nth_prime_after(0, 2), 3);
    }

    /// The checkpoint declares `5.8.0.dev0`; the release that carries the
    /// reference is a later one. Pinning both facts is what stops the two from
    /// being conflated again — which is how `5.8.0` got in here originally.
    #[test]
    fn the_pinned_reference_is_not_the_declared_version() {
        assert_eq!(frozen()["transformers_version"], "5.8.0.dev0");
        assert_eq!(PINNED_TRANSFORMERS, "5.16.0");
    }
}
