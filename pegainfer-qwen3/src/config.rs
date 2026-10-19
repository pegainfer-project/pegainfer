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
    /// Token-selection width: `vocab_size` bounded to the frontend-decodable vocab.
    #[serde(skip)]
    pub(crate) decodable_vocab: usize,
    /// Aligned output-projection rows; any rows past `decodable_vocab` are suppressed.
    #[serde(skip)]
    pub(crate) logits_rows: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: bool,
    #[serde(skip)]
    pub(crate) stop_token_ids: Vec<u32>,
}

/// Drafter config for DFlash, DSpark and DFlash2. `from_file` normalizes
/// nested b16, flat DeepSpec and Speculators schemas. With both head ranks
/// zero, proposals use independent per-position argmax.
#[derive(Clone, Debug)]
pub(crate) struct DFlashConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    num_target_layers: Option<usize>,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    /// Target output-projection rows, including its suppressed alignment tail.
    pub(crate) logits_rows: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
    /// DSpark Markov head low-rank size; 0 disables the Markov head.
    pub(crate) markov_rank: usize,
    /// Path-selector rank; 0 disables this head.
    pub(crate) selector_rank: usize,
    pub(crate) conv: Option<DFlashConvConfig>,
    pub(crate) sliding_window: Option<usize>,
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
    /// DSpine code bridge; `None` for DFlash and DSpark drafters.
    pub(crate) dspine: Option<DSpineConfig>,
}

/// Resolved DSpine bridge config (`tencent/Qwen3-8B-DSpine-b16`).
#[derive(Clone, Debug)]
pub(crate) struct DSpineConfig {
    pub(crate) code_size: usize,
    pub(crate) rank: usize,
    // Read by the bridge forward and the weight loader in the next slice.
    #[allow(dead_code)]
    pub(crate) eps: f32,
    /// Draft layer indices carrying a code bridge, 0-based and ascending.
    /// The last draft layer is always present (its write is refined serially).
    #[allow(dead_code)]
    pub(crate) feature_layers: Vec<usize>,
    /// Width of the top-K serial last-write refinement.
    pub(crate) head_topk: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DFlashInnerConfig {
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    block_size: Option<usize>,
    selector_rank: Option<usize>,
    selector_top_k: Option<usize>,
    conv_kernel_size: Option<usize>,
    conv_group_size: Option<usize>,
    #[serde(default)]
    draft_block_causal: Option<bool>,
    #[serde(default)]
    selected_prefix_code_bridge: Option<RawDSpineCodeBridge>,
    #[serde(default)]
    selected_prefix_feature_layers: Option<Vec<usize>>,
    #[serde(default)]
    selected_prefix_feature_writes: Option<bool>,
    #[serde(default)]
    selected_prefix_feature_write_mode: Option<String>,
    #[serde(default)]
    selected_prefix_terminal: Option<bool>,
    #[serde(default)]
    selected_prefix_last_write_refine: Option<RawDSpineLastWriteRefine>,
}

#[derive(Clone, Debug)]
pub(crate) struct DFlashConvConfig {
    pub(crate) taps: usize,
    pub(crate) group_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RawDSpineCodeBridge {
    code_size: usize,
    rank: usize,
    eps: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct RawDSpineLastWriteRefine {
    kind: String,
    head_topk: usize,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

fn default_selector_top_k() -> usize {
    pegainfer_kernels::ops::DFLASH2_CANDIDATE_K
}

fn default_markov_head_type() -> String {
    "vanilla".to_string()
}

/// Backbone geometry shared by flat configs and Speculators
/// `transformer_layer_config`.
#[derive(Deserialize)]
struct DFlashGeometry {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: Option<f32>,
    rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    sliding_window: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DFlashGeometrySource {
    Speculators {
        transformer_layer_config: DFlashGeometry,
    },
    Flat(DFlashGeometry),
}

#[derive(Deserialize)]
struct RawDFlashConfig {
    #[serde(flatten)]
    geometry: DFlashGeometrySource,
    num_target_layers: Option<usize>,
    block_size: Option<usize>,
    #[serde(default)]
    dflash_config: Option<DFlashInnerConfig>,
    /// DSpine checkpoints nest this under `dflash_config`; accept the flat form too.
    #[serde(default)]
    draft_block_causal: Option<bool>,
    /// Parsed only to reject the unsupported flat DSpine schema loudly.
    #[serde(default)]
    selected_prefix_code_bridge: Option<RawDSpineCodeBridge>,
    #[serde(default)]
    mask_token_id: Option<u32>,
    #[serde(default)]
    target_layer_ids: Option<Vec<usize>>,
    aux_hidden_state_layer_ids: Option<Vec<usize>>,
    conv_kernel_size: Option<usize>,
    conv_group_size: Option<usize>,
    sliding_window_non_causal: Option<bool>,
    #[serde(default)]
    sample_from_anchor: bool,
    #[serde(default)]
    markov_rank: usize,
    #[serde(default)]
    selector_rank: usize,
    #[serde(default = "default_selector_top_k")]
    selector_top_k: usize,
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

impl Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.decodable_vocab = config.vocab_size;
        config.logits_rows = config.vocab_size;
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
        config.stop_token_ids = pegainfer_core::eos::eos_token_ids(model_path, config.vocab_size)?;
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
}

impl DSpineConfig {
    /// `Ok(None)` for a plain DFlash/DSpark checkpoint; `Ok(Some(_))` when the
    /// checkpoint carries a code bridge. Every DSpine-only field is validated
    /// rather than ignored; serving a bridge checkpoint as plain DFlash would
    /// draft with near-zero acceptance.
    fn from_inner(
        inner: &DFlashInnerConfig,
        num_hidden_layers: usize,
        draft_block_causal: Option<bool>,
    ) -> Result<Option<Self>> {
        let Some(bridge) = inner.selected_prefix_code_bridge.as_ref() else {
            let stray = inner.selected_prefix_feature_layers.is_some()
                || inner.selected_prefix_feature_writes.is_some()
                || inner.selected_prefix_feature_write_mode.is_some()
                || inner.selected_prefix_terminal.is_some()
                || inner.selected_prefix_last_write_refine.is_some();
            ensure!(
                !stray,
                "drafter config has DSpine refinement fields but no selected_prefix_code_bridge"
            );
            return Ok(None);
        };

        ensure!(
            draft_block_causal == Some(true),
            "DSpine requires draft_block_causal = true (causal draft attention)"
        );
        ensure!(
            inner.selected_prefix_terminal == Some(false),
            "DSpine supports terminal-disabled code bridges only"
        );
        ensure!(
            inner.selected_prefix_feature_writes == Some(true),
            "DSpine requires selected_prefix_feature_writes = true"
        );
        ensure!(
            inner.selected_prefix_feature_write_mode.as_deref() == Some("independent"),
            "DSpine requires selected_prefix_feature_write_mode = \"independent\", got {:?}",
            inner.selected_prefix_feature_write_mode
        );

        let layers = inner
            .selected_prefix_feature_layers
            .as_ref()
            .context("DSpine config missing selected_prefix_feature_layers")?;
        ensure!(
            !layers.is_empty(),
            "DSpine feature layers must not be empty"
        );
        ensure!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "DSpine feature layers must be strictly increasing, got {layers:?}"
        );
        ensure!(
            layers
                .iter()
                .all(|&layer| (1..=num_hidden_layers).contains(&layer)),
            "DSpine feature layers must be 1-based within 1..={num_hidden_layers}, got {layers:?}"
        );
        ensure!(
            layers.last() == Some(&num_hidden_layers),
            "DSpine refines the last write, so the last draft layer must be active; got {layers:?}"
        );
        let feature_layers = layers.iter().map(|layer| layer - 1).collect();

        let refine = inner
            .selected_prefix_last_write_refine
            .as_ref()
            .context("DSpine config missing selected_prefix_last_write_refine")?;
        ensure!(
            refine.kind == "exact_sender",
            "DSpine requires selected_prefix_last_write_refine.kind = \"exact_sender\", got {:?}",
            refine.kind
        );
        ensure!(
            refine.head_topk.is_power_of_two() && (2..=64).contains(&refine.head_topk),
            "DSpine head_topk must be a power of two in [2, 64], got {}",
            refine.head_topk
        );

        ensure!(
            bridge.code_size > 0 && bridge.code_size.is_multiple_of(128),
            "DSpine code_size must be a non-zero multiple of 128, got {}",
            bridge.code_size
        );
        ensure!(
            bridge.rank > 0 && bridge.rank.is_multiple_of(128),
            "DSpine bridge rank must be a non-zero multiple of 128, got {}",
            bridge.rank
        );
        ensure!(
            bridge.eps.is_finite() && bridge.eps > 0.0,
            "DSpine code bridge eps must be finite and positive, got {}",
            bridge.eps
        );

        Ok(Some(Self {
            code_size: bridge.code_size,
            rank: bridge.rank,
            eps: bridge.eps,
            feature_layers,
            head_topk: refine.head_topk,
        }))
    }
}

impl DFlashConfig {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        Self::from_json(&content)
    }

    fn from_json(content: &str) -> Result<Self> {
        let raw: RawDFlashConfig = serde_json::from_str(content)?;
        let geometry = match raw.geometry {
            DFlashGeometrySource::Speculators {
                transformer_layer_config,
            } => transformer_layer_config,
            DFlashGeometrySource::Flat(geometry) => geometry,
        };
        let inner = raw.dflash_config.as_ref();
        let block_size = inner
            .and_then(|config| config.block_size)
            .or(raw.block_size)
            .context("drafter config missing block_size")?;
        let selector_rank = inner
            .and_then(|config| config.selector_rank)
            .unwrap_or(raw.selector_rank);
        let selector_top_k = inner
            .and_then(|config| config.selector_top_k)
            .unwrap_or(raw.selector_top_k);
        if selector_rank > 0 {
            ensure!(
                selector_top_k == default_selector_top_k(),
                "DFlash2 requires top-16 candidates"
            );
            ensure!(
                raw.markov_rank == 0 && raw.num_anchors.is_none() && !raw.sample_from_anchor,
                "DFlash2 selector requires an anchor-drop DFlash backbone"
            );
        }

        let conv = match (
            inner
                .and_then(|config| config.conv_kernel_size)
                .or(raw.conv_kernel_size),
            inner
                .and_then(|config| config.conv_group_size)
                .or(raw.conv_group_size),
        ) {
            (None, None) => None,
            (Some(taps), Some(group_size)) => {
                ensure!(
                    raw.sliding_window_non_causal != Some(false) && !raw.sample_from_anchor,
                    "DFlash2 supports noncausal anchor-drop blocks"
                );
                ensure!(
                    taps > 0 && group_size > 0 && geometry.hidden_size.is_multiple_of(group_size),
                    "DFlash2 convolution requires positive taps and groups dividing hidden_size"
                );
                Some(DFlashConvConfig { taps, group_size })
            }
            _ => bail!("DFlash2 convolution requires both kernel_size and group_size"),
        };

        // rope_theta: flat (b16) or nested under rope_parameters (DeepSpec).
        let rope_theta = geometry
            .rope_theta
            .or(geometry.rope_parameters.map(|r| r.rope_theta))
            .context("drafter config missing rope_theta / rope_parameters.rope_theta")?;

        ensure!(
            raw.selected_prefix_code_bridge.is_none(),
            "DSpine bridge config must be nested under dflash_config; \
             a flat selected_prefix_code_bridge is not supported"
        );

        // mask_token_id + target_layer_ids: nested dflash_config (b16) or flat (DeepSpec).
        let dflash_config = raw.dflash_config;
        let (mask_token_id, target_layer_ids) = if let Some(inner) = &dflash_config {
            (inner.mask_token_id, inner.target_layer_ids.clone())
        } else {
            let layers = match (raw.target_layer_ids, raw.aux_hidden_state_layer_ids) {
                (Some(layers), _) => layers,
                (_, Some(layers)) => layers
                    .into_iter()
                    .map(|layer| {
                        // Speculators indexes HF hidden_states (0 is embedding);
                        // our capture points are zero-based decoder outputs.
                        layer
                            .checked_sub(1)
                            .context("DFlash capture must refer to a decoder output")
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => bail!("drafter config missing target hidden-state layers"),
            };
            (
                raw.mask_token_id
                    .context("drafter config missing mask_token_id")?,
                layers,
            )
        };

        let sliding_window = conv.as_ref().and(geometry.sliding_window);

        let draft_block_causal = match (
            dflash_config
                .as_ref()
                .and_then(|inner| inner.draft_block_causal),
            raw.draft_block_causal,
        ) {
            (Some(nested), Some(flat)) => {
                ensure!(
                    nested == flat,
                    "draft_block_causal disagrees between dflash_config ({nested}) and the top level ({flat})"
                );
                Some(nested)
            }
            (nested, flat) => nested.or(flat),
        };

        let dspine = if let Some(inner) = &dflash_config {
            DSpineConfig::from_inner(inner, geometry.num_hidden_layers, draft_block_causal)?
        } else {
            None
        };
        if dspine.is_some() {
            ensure!(
                raw.num_anchors.is_none(),
                "DSpine drafts are anchor-drop; config declares num_anchors"
            );
            ensure!(
                raw.markov_rank == 0,
                "DSpine drafts carry no markov head; config declares markov_rank = {}",
                raw.markov_rank
            );
        }
        Ok(Self {
            hidden_size: geometry.hidden_size,
            intermediate_size: geometry.intermediate_size,
            num_hidden_layers: geometry.num_hidden_layers,
            num_attention_heads: geometry.num_attention_heads,
            num_key_value_heads: geometry.num_key_value_heads,
            num_target_layers: raw.num_target_layers,
            head_dim: geometry.head_dim,
            vocab_size: geometry.vocab_size,
            logits_rows: geometry.vocab_size,
            rms_norm_eps: geometry.rms_norm_eps,
            rope_theta,
            max_position_embeddings: geometry.max_position_embeddings,
            block_size,
            mask_token_id,
            target_layer_ids,
            markov_rank: raw.markov_rank,
            selector_rank,
            conv,
            sliding_window,
            markov_head_type: raw.markov_head_type,
            enable_confidence_head: raw.enable_confidence_head,
            // A markov head only ever ships on a DeepSpec (anchor-first)
            // checkpoint, so num_anchors is always present alongside it; treat
            // markov as an independent corroborating signal so a future flat
            // schema can't accidentally route a markov checkpoint anchor-drop.
            anchor_first: raw.num_anchors.is_some() || raw.markov_rank > 0,
            dspine,
        })
    }

    /// DSpark Markov head is active (`markov_rank > 0`); the draft uses the
    /// semi-autoregressive sample loop instead of independent argmax.
    pub(crate) fn uses_markov_head(&self) -> bool {
        self.markov_rank > 0
    }

    /// DSpine code bridge is active; the draft forward injects per-layer
    /// features and proposes with the serial last-write refinement.
    pub(crate) fn uses_dspine(&self) -> bool {
        self.dspine.is_some()
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
            self.num_target_layers
                .is_none_or(|layers| layers == target.num_hidden_layers),
            "DFlash num_target_layers {:?} does not match target layers {}",
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

        if self.dspine.is_some() {
            anyhow::ensure!(
                self.head_dim == 128,
                "DSpine causal draft attention requires head_dim 128, got {}",
                self.head_dim
            );
            anyhow::ensure!(
                self.hidden_size.is_multiple_of(512),
                "DSpine bridge requires hidden_size to be a multiple of 512, got {}",
                self.hidden_size
            );
            anyhow::ensure!(
                !self.anchor_first(),
                "DSpine drafts are anchor-drop; anchor-first checkpoints are not supported"
            );
        }

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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn dflash_selector_fields_parse_from_flat_and_nested_configs() -> Result<()> {
        let geometry = serde_json::json!({
            "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0
        });
        let selector = serde_json::json!({
            "block_size": 3, "mask_token_id": 7, "target_layer_ids": [1, 5],
            "selector_rank": 2, "selector_top_k": 16,
            "training_metadata": { "unused": true }
        });
        let mut flat = geometry.clone();
        flat.as_object_mut()
            .unwrap()
            .extend(selector.as_object().unwrap().clone());
        let mut nested = geometry;
        nested["dflash_config"] = selector;
        let directory = tempfile::tempdir()?;

        for json in [flat, nested] {
            fs::write(directory.path().join("config.json"), json.to_string())?;
            let config = DFlashConfig::from_file(directory.path().to_str().unwrap())?;
            assert_eq!(config.block_size, 3);
            assert_eq!(config.mask_token_id, 7);
            assert_eq!(config.target_layer_ids, vec![1, 5]);
            assert_eq!(config.selector_rank, 2);
        }
        Ok(())
    }

    #[test]
    fn speculators_dflash_maps_hidden_state_indices_to_decoder_layers() -> Result<()> {
        let json = serde_json::json!({
            "transformer_layer_config": {
                "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
                "vocab_size": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "sliding_window": 2048
            },
            "block_size": 3, "mask_token_id": 7, "selector_rank": 2,
            "conv_kernel_size": 2, "conv_group_size": 2,
            "aux_hidden_state_layer_ids": [1, 3], "sliding_window_non_causal": true
        });
        let directory = tempfile::tempdir()?;
        fs::write(directory.path().join("config.json"), json.to_string())?;
        let config = DFlashConfig::from_file(directory.path().to_str().unwrap())?;

        assert_eq!(config.target_layer_ids, vec![0, 2]);
        assert_eq!(config.selector_rank, 2);
        let conv = config.conv.expect("native convolution config");
        assert_eq!((conv.taps, conv.group_size), (2, 2));
        assert_eq!(config.sliding_window, Some(2048));
        Ok(())
    }

    pub(crate) fn dspine_8b_config() -> DFlashConfig {
        DFlashConfig::from_json(DSPINE_8B_CONFIG).expect("DSpine 8B config")
    }

    pub(crate) fn plain_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json(PLAIN_DFLASH_CONFIG).expect("plain DFlash config")
    }

    const DSPINE_8B_CONFIG: &str = r#"{
      "architectures": ["DFlashDraftModel"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
      "block_size": 16,
      "bos_token_id": 151643,
      "dflash_config": {
        "draft_block_causal": true,
        "mask_token_id": 151669,
        "selected_prefix_code_bridge": {
          "code_size": 1024,
          "eps": 1e-06,
          "kappa": 10.0,
          "rank": 512
        },
        "selected_prefix_decode_mode": "serial",
        "selected_prefix_feature_gt_training": true,
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_teacher_layers": [1, 2, 3],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": true,
        "selected_prefix_last_write_refine": {
          "head_topk": 16,
          "kind": "exact_sender",
          "version": 1,
          "weight": 1.0
        },
        "selected_prefix_terminal": false,
        "target_layer_ids": [1, 9, 17, 25, 33]
      },
      "draft_vocab_size": 151936,
      "dtype": "bfloat16",
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "intermediate_size": 12288,
      "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
      "max_position_embeddings": 40960,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "num_target_layers": 36,
      "rms_norm_eps": 1e-06,
      "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
      "rope_theta": 1000000,
      "tie_word_embeddings": false,
      "vocab_size": 151936
    }"#;

    const PLAIN_DFLASH_CONFIG: &str = r#"{
      "hidden_size": 4096,
      "intermediate_size": 12288,
      "num_hidden_layers": 5,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "num_target_layers": 36,
      "head_dim": 128,
      "vocab_size": 151936,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "block_size": 16,
      "dflash_config": {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]}
    }"#;

    fn target_qwen3_8b() -> Config {
        Config {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            decodable_vocab: 151_936,
            logits_rows: 151_936,
            rms_norm_eps: 1e-06,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            tie_word_embeddings: false,
            stop_token_ids: vec![151_645],
        }
    }

    #[test]
    fn parses_dspine_code_bridge_config() {
        let config = DFlashConfig::from_json(DSPINE_8B_CONFIG).expect("DSpine config");
        config
            .validate_for_target(&target_qwen3_8b())
            .expect("DSpine config should match Qwen3-8B");

        assert!(config.uses_dspine());
        assert!(!config.anchor_first());
        assert_eq!(config.block_size, 16);
        assert_eq!(config.mask_token_id, 151_669);
        assert_eq!(config.target_layer_ids, vec![1, 9, 17, 25, 33]);

        let dspine = config.dspine.as_ref().expect("parsed bridge");
        assert_eq!(dspine.code_size, 1024);
        assert_eq!(dspine.rank, 512);
        assert_eq!(dspine.eps.to_bits(), 1e-6f32.to_bits());
        assert_eq!(dspine.feature_layers, vec![0, 1, 2, 3, 4]);
        assert_eq!(dspine.head_topk, 16);
    }

    #[test]
    fn plain_dflash_config_is_not_dspine() {
        let config = DFlashConfig::from_json(PLAIN_DFLASH_CONFIG).expect("plain DFlash config");
        config
            .validate_for_target(&target_qwen3_8b())
            .expect("plain DFlash config should match Qwen3-8B");
        assert!(!config.uses_dspine());
    }

    #[test]
    fn rejects_unsupported_dspine_bridge_configs() {
        let cases = [
            (
                "terminal projection enabled",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_terminal\": false",
                    "\"selected_prefix_terminal\": true",
                ),
            ),
            (
                "non-power-of-two refinement width",
                DSPINE_8B_CONFIG.replace("\"head_topk\": 16", "\"head_topk\": 15"),
            ),
            (
                "last draft layer inactive",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_feature_layers\": [1, 2, 3, 4, 5]",
                    "\"selected_prefix_feature_layers\": [1, 2, 3, 4]",
                ),
            ),
            (
                "anchor-first layout",
                DSPINE_8B_CONFIG.replace(
                    "\"num_hidden_layers\": 5,",
                    "\"num_hidden_layers\": 5,\n      \"num_anchors\": 1,",
                ),
            ),
            (
                "refinement fields without a bridge",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_code_bridge\": {",
                    "\"selected_prefix_code_bridge_ignored\": {",
                ),
            ),
            (
                "non-causal draft attention",
                DSPINE_8B_CONFIG.replace(
                    "\"draft_block_causal\": true",
                    "\"draft_block_causal\": false",
                ),
            ),
            (
                "feature writes disabled",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_feature_writes\": true",
                    "\"selected_prefix_feature_writes\": false",
                ),
            ),
            (
                "non-independent write mode",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_feature_write_mode\": \"independent\"",
                    "\"selected_prefix_feature_write_mode\": \"shared\"",
                ),
            ),
            (
                "code size not kernel-tile aligned",
                DSPINE_8B_CONFIG.replace("\"code_size\": 1024", "\"code_size\": 1000"),
            ),
            (
                "bridge rank not kernel-tile aligned",
                DSPINE_8B_CONFIG.replace("\"rank\": 512", "\"rank\": 500"),
            ),
            (
                "non-positive bridge eps",
                DSPINE_8B_CONFIG.replace("\"eps\": 1e-06", "\"eps\": 0.0"),
            ),
            (
                "unsupported refinement kind",
                DSPINE_8B_CONFIG.replace("\"kind\": \"exact_sender\"", "\"kind\": \"approximate\""),
            ),
            (
                "refinement width above 64",
                DSPINE_8B_CONFIG.replace("\"head_topk\": 16", "\"head_topk\": 128"),
            ),
            (
                "unsorted feature layers",
                DSPINE_8B_CONFIG.replace(
                    "\"selected_prefix_feature_layers\": [1, 2, 3, 4, 5]",
                    "\"selected_prefix_feature_layers\": [1, 2, 3, 5, 4]",
                ),
            ),
            (
                "conflicting causal flags",
                DSPINE_8B_CONFIG.replace(
                    "\"dflash_config\": {",
                    "\"draft_block_causal\": false,\n      \"dflash_config\": {",
                ),
            ),
            (
                "flat bridge schema",
                PLAIN_DFLASH_CONFIG.replace(
                    "\"block_size\": 16,",
                    "\"block_size\": 16,\n      \"selected_prefix_code_bridge\": {\"code_size\": 1024, \"rank\": 512, \"eps\": 1e-06},",
                ),
            ),
        ];

        for (name, json) in cases {
            assert!(
                DFlashConfig::from_json(&json).is_err(),
                "{name} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_dspine_targets_the_kernels_cannot_serve() {
        let target = target_qwen3_8b();

        let mut wrong_head_dim = dspine_8b_config();
        wrong_head_dim.head_dim = 64;
        let mut target_64 = target_qwen3_8b();
        target_64.head_dim = 64;
        assert!(wrong_head_dim.validate_for_target(&target_64).is_err());

        let mut wrong_hidden = dspine_8b_config();
        wrong_hidden.hidden_size = 256;
        let mut target_256 = target_qwen3_8b();
        target_256.hidden_size = 256;
        assert!(wrong_hidden.validate_for_target(&target_256).is_err());

        let mut anchor_first = dspine_8b_config();
        anchor_first.anchor_first = true;
        assert!(anchor_first.validate_for_target(&target).is_err());
    }
}
