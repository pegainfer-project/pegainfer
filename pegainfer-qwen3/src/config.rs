use std::fs;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde_json::Value;

pub(crate) const PREFILL_ATTENTION_CTA_TILE_Q: i32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) eos_token_id: u32,
    pub(crate) tie_word_embeddings: bool,
    #[serde(skip)]
    pub(crate) stop_token_ids: Vec<u32>,
}

/// Resolved drafter config shared by DFlash and DSpark. DSpark extends the
/// DFlash backbone with a Markov head and an optional confidence head;
/// `markov_rank == 0` is plain DFlash. Two on-disk schemas are normalized into
/// this in `from_file`: our `Qwen3-4B-DFlash-b16` nests
/// `dflash_config: {mask_token_id, target_layer_ids}` and puts `rope_theta` at
/// the top level, while DeepSpec's `dflash_/dspark_*_block7` put those fields
/// flat and nest `rope_theta` under `rope_parameters`.
#[derive(Clone, Debug)]
pub(crate) struct DFlashConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    num_target_layers: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
    /// DSpark Markov head low-rank size; 0 disables the head (= plain DFlash).
    pub(crate) markov_rank: usize,
    markov_head_type: String,
    /// Block draft layout. DeepSpec `Qwen3DSparkModel` checkpoints (both the
    /// markov and the `markov_rank == 0` ones) are *anchor-first*: block position
    /// 0 is already the first real prediction, so all `block_size` positions
    /// draft (verify span `block_size + 1`). Our native `DFlashDraftModel` (b16)
    /// is *anchor-drop*: position 0 is a throwaway anchor slot, so only positions
    /// `1..block_size` draft (verify span `block_size`). This is a property of the
    /// checkpoint, NOT of the markov head — keying it on the markov head silently
    /// mis-drafts a no-markov DeepSpec checkpoint (accept rate collapses to ~0).
    anchor_first: bool,
    /// Whether the checkpoint carries a confidence head. Phase 1 does not use it
    /// (full-block verify, no confidence-scheduled truncation); surfaced at load
    /// so the operator knows that capability is being ignored. See
    /// docs/models/qwen3/dspark-integration.md (Phase 2).
    pub(crate) enable_confidence_head: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct DFlashInnerConfig {
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

fn default_markov_head_type() -> String {
    "vanilla".to_string()
}

/// On-disk drafter config tolerant of both the nested (`b16`) and flat
/// (DeepSpec) schemas; `from_file` resolves it into `DFlashConfig`.
#[derive(Deserialize)]
struct RawDFlashConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    block_size: usize,
    #[serde(default)]
    dflash_config: Option<DFlashInnerConfig>,
    #[serde(default)]
    mask_token_id: Option<u32>,
    #[serde(default)]
    target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    markov_rank: usize,
    #[serde(default = "default_markov_head_type")]
    markov_head_type: String,
    #[serde(default)]
    enable_confidence_head: bool,
    /// DeepSpec `Qwen3DSparkModel` checkpoints declare this (anchor-first
    /// drafting); native `DFlashDraftModel` checkpoints omit it (anchor-drop).
    #[serde(default)]
    num_anchors: Option<usize>,
}

/// EAGLE-3 drafter config (e.g. `AngelSlim/Qwen3-4B_eagle3`).
///
///  A single-layer (`midlayer`) head whose attention takes
/// `2 * hidden_size` inputs (concatenated `[norm(embed), norm(fused_hidden)]`),
/// has no QK-norm, reuses the target's `embed_tokens`, and predicts over a
/// reduced `draft_vocab_size` remapped to the full vocab via `d2t`/`t2d`.
#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Eagle3Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    /// Target's full vocabulary (logits are projected back into this space).
    pub(crate) vocab_size: usize,
    /// Reduced vocabulary the draft `lm_head` predicts over (rows of `d2t`).
    pub(crate) draft_vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
}

fn default_max_position_embeddings() -> usize {
    40960
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(token_id) => vec![token_id],
            Self::Multiple(token_ids) => token_ids,
        }
    }
}

impl Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        anyhow::ensure!(
            config.num_key_value_heads > 0
                && config
                    .num_attention_heads
                    .is_multiple_of(config.num_key_value_heads),
            "num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        if !config.decode_group_is_compiled() {
            log::warn!(
                "Qwen3 GQA group {}/{} has no compiled decode kernel; decode runs eager \
                 through the prefill path (CUDA-graph decode disabled, --decode-overlap unavailable)",
                config.num_attention_heads,
                config.num_key_value_heads,
            );
        }
        config.stop_token_ids = Self::load_stop_token_ids(model_path, config.eos_token_id)?;
        Ok(config)
    }

    /// GQA ratio is TP-invariant, so the global head counts match the per-rank ones.
    pub(crate) fn decode_group_is_compiled(&self) -> bool {
        pegainfer_core::ops::SUPPORTED_GQA_GROUP_SIZES
            .contains(&(self.num_attention_heads / self.num_key_value_heads))
    }

    pub(crate) fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub(crate) fn local_num_attention_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_attention_heads / tp.world_size
    }

    pub(crate) fn local_num_key_value_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_key_value_heads / tp.world_size
    }

    pub(crate) fn local_intermediate_size(&self, tp: TensorParallelConfig) -> usize {
        self.intermediate_size / tp.world_size
    }

    pub(crate) fn local_q_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_attention_heads(tp) * self.head_dim
    }

    pub(crate) fn local_kv_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_key_value_heads(tp) * self.head_dim
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = generation_config.eos_token_id.into_vec();
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }
}

impl DFlashConfig {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawDFlashConfig = serde_json::from_str(&content)?;

        // rope_theta: flat (b16) or nested under rope_parameters (DeepSpec).
        let rope_theta = raw
            .rope_theta
            .or(raw.rope_parameters.map(|r| r.rope_theta))
            .context("drafter config missing rope_theta / rope_parameters.rope_theta")?;

        // mask_token_id + target_layer_ids: nested dflash_config (b16) or flat (DeepSpec).
        let (mask_token_id, target_layer_ids) = match raw.dflash_config {
            Some(inner) => (inner.mask_token_id, inner.target_layer_ids),
            None => (
                raw.mask_token_id
                    .context("drafter config missing mask_token_id (no dflash_config block)")?,
                raw.target_layer_ids
                    .context("drafter config missing target_layer_ids (no dflash_config block)")?,
            ),
        };

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            num_target_layers: raw.num_target_layers,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            block_size: raw.block_size,
            mask_token_id,
            target_layer_ids,
            markov_rank: raw.markov_rank,
            markov_head_type: raw.markov_head_type,
            enable_confidence_head: raw.enable_confidence_head,
            // A markov head only ever ships on a DeepSpec (anchor-first)
            // checkpoint, so num_anchors is always present alongside it; treat
            // markov as an independent corroborating signal so a future flat
            // schema can't accidentally route a markov checkpoint anchor-drop.
            anchor_first: raw.num_anchors.is_some() || raw.markov_rank > 0,
        })
    }

    /// DSpark Markov head is active (`markov_rank > 0`); the draft uses the
    /// semi-autoregressive sample loop instead of independent argmax.
    pub(crate) fn uses_markov_head(&self) -> bool {
        self.markov_rank > 0
    }

    /// Anchor-first block layout (see [`DFlashConfig::anchor_first`]). The markov
    /// head implies it, but a `markov_rank == 0` DeepSpec checkpoint needs it too.
    pub(crate) fn anchor_first(&self) -> bool {
        self.anchor_first
    }

    pub(crate) fn validate_for_target(&self, target: &Config) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size == target.hidden_size,
            "DFlash hidden_size {} does not match target {}",
            self.hidden_size,
            target.hidden_size
        );
        anyhow::ensure!(
            self.num_target_layers == target.num_hidden_layers,
            "DFlash num_target_layers {} does not match target layers {}",
            self.num_target_layers,
            target.num_hidden_layers
        );
        anyhow::ensure!(
            self.num_attention_heads == target.num_attention_heads
                && self.num_key_value_heads == target.num_key_value_heads
                && self.head_dim == target.head_dim,
            "DFlash attention geometry does not match target"
        );
        anyhow::ensure!(
            self.vocab_size == target.vocab_size,
            "DFlash vocab_size {} does not match target {}",
            self.vocab_size,
            target.vocab_size
        );
        anyhow::ensure!(
            self.rope_theta.to_bits() == target.rope_theta.to_bits(),
            "DFlash rope_theta {} does not match target {}",
            self.rope_theta,
            target.rope_theta
        );
        anyhow::ensure!(
            self.max_position_embeddings >= target.max_position_embeddings,
            "DFlash max_position_embeddings {} is smaller than target {}",
            self.max_position_embeddings,
            target.max_position_embeddings
        );
        anyhow::ensure!(
            self.block_size >= 2,
            "DFlash block_size must be >= 2, got {}",
            self.block_size
        );
        anyhow::ensure!(
            self.mask_token_id < target.vocab_size as u32,
            "DFlash mask_token_id {} is outside target vocab_size {}",
            self.mask_token_id,
            target.vocab_size
        );
        anyhow::ensure!(
            self.target_layer_ids.len() == self.num_hidden_layers,
            "DFlash target_layer_ids length {} does not match draft layers {}",
            self.target_layer_ids.len(),
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.target_layer_ids
                .iter()
                .all(|&layer| layer < target.num_hidden_layers),
            "DFlash target_layer_ids must be within target layer count"
        );
        anyhow::ensure!(
            self.target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "DFlash target_layer_ids must be strictly increasing"
        );

        // DSpark Markov head: only the released `vanilla` low-rank head is
        // implemented; reject the gated/rnn variants loudly rather than silently
        // mis-drafting. The confidence head is intentionally ignored in Phase 1
        // (full-block verify, no confidence-scheduled truncation) — its weights
        // are simply not loaded; see docs/models/qwen3/dspark-integration.md.
        if self.uses_markov_head() {
            anyhow::ensure!(
                self.markov_head_type == "vanilla",
                "DSpark markov_head_type {:?} not supported (only \"vanilla\")",
                self.markov_head_type
            );
        }
        Ok(())
    }
}

impl Eagle3Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub(crate) fn validate_for_target(&self, target: &Config) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size == target.hidden_size,
            "EAGLE-3 hidden_size {} does not match target {}",
            self.hidden_size,
            target.hidden_size
        );
        anyhow::ensure!(
            self.num_hidden_layers == 1,
            "EAGLE-3 drafter must have exactly one decoder layer (midlayer), got {}",
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.num_attention_heads == target.num_attention_heads
                && self.num_key_value_heads == target.num_key_value_heads
                && self.head_dim == target.head_dim,
            "EAGLE-3 attention geometry does not match target"
        );
        anyhow::ensure!(
            self.vocab_size == target.vocab_size,
            "EAGLE-3 vocab_size {} does not match target {}",
            self.vocab_size,
            target.vocab_size
        );
        anyhow::ensure!(
            self.draft_vocab_size > 0 && self.draft_vocab_size <= self.vocab_size,
            "EAGLE-3 draft_vocab_size {} must be in 1..={}",
            self.draft_vocab_size,
            self.vocab_size
        );
        anyhow::ensure!(
            self.rope_theta.to_bits() == target.rope_theta.to_bits(),
            "EAGLE-3 rope_theta {} does not match target {}",
            self.rope_theta,
            target.rope_theta
        );
        anyhow::ensure!(
            self.max_position_embeddings >= target.max_position_embeddings,
            "EAGLE-3 max_position_embeddings {} is smaller than target {}",
            self.max_position_embeddings,
            target.max_position_embeddings
        );
        Ok(())
    }
}

impl TensorParallelConfig {
    pub(crate) fn validate_for(self, config: &Config) -> Result<()> {
        if self.world_size == 0 {
            return Err(anyhow::anyhow!("tensor_parallel.world_size must be >= 1"));
        }
        if self.rank >= self.world_size {
            return Err(anyhow::anyhow!(
                "tensor_parallel.rank {} must be < world_size {}",
                self.rank,
                self.world_size
            ));
        }
        if !config.num_attention_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_attention_heads={} not divisible by tp world_size={}",
                config.num_attention_heads,
                self.world_size
            ));
        }
        if !config.num_key_value_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_key_value_heads={} not divisible by tp world_size={}",
                config.num_key_value_heads,
                self.world_size
            ));
        }
        if !config.intermediate_size.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "intermediate_size={} not divisible by tp world_size={}",
                config.intermediate_size,
                self.world_size
            ));
        }
        Ok(())
    }

    pub(crate) fn shard_range(self, total: usize) -> (usize, usize) {
        let shard_len = total / self.world_size;
        (self.rank * shard_len, shard_len)
    }

    pub(crate) fn is_sharded(self) -> bool {
        self.world_size > 1
    }
}

/// Identity check that `json` is a Qwen3 config; size and shape validation belong to the config loader.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = json.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "qwen3" {
        bail!("not a Qwen3 config: model_type={model_type}");
    }
    let architectures: Vec<&str> = json
        .get("architectures")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        architectures.contains(&"Qwen3ForCausalLM"),
        "Qwen3 architectures must contain Qwen3ForCausalLM"
    );
    Ok(())
}
