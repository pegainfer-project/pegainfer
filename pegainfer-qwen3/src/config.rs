use std::fs;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde_json::Value;

pub(crate) const PREFILL_ATTENTION_CTA_TILE_Q: i32 = 64;
pub(crate) const DFLASH2_SELECTOR_TOP_K: usize = 16;
const DEFAULT_MARKOV_HEAD_TYPE: &str = "vanilla";

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

/// Normalized DFlash/DSpark configuration for legacy and native DFlash2 schemas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DFlashProposal {
    PlainArgmax,
    Markov { rank: usize, head_type: String },
    TopKSelector { rank: usize, top_k: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DFlashLayout {
    /// Position zero is an anchor slot and is removed before verification.
    AnchorDrop,
    /// Position zero is already the first proposed token.
    AnchorFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DFlashHeadSource {
    /// Use the verifier's embedding and output projection.
    Target,
    /// Native DFlash2 provides a separate draft output projection.
    DraftOutput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DynamicConv {
    pub(crate) kernel_size: usize,
    pub(crate) group_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlidingWindow {
    pub(crate) window_size: usize,
    pub(crate) non_causal: bool,
    pub(crate) enabled: bool,
    pub(crate) layer_types: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct DFlashConfig {
    pub(crate) hidden_size: usize,
    pub(crate) target_hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    /// Legacy target-layer count; absent in native DFlash2.
    num_target_layers: Option<usize>,
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: u32,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) draft_vocab_size: usize,
    /// Proposal capability selected by the checkpoint schema.
    pub(crate) proposal: DFlashProposal,
    pub(crate) layout: DFlashLayout,
    pub(crate) head_source: DFlashHeadSource,
    pub(crate) enable_confidence_head: bool,
    pub(crate) dynamic_convolution: Option<DynamicConv>,
    pub(crate) sliding_window: Option<SlidingWindow>,
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
    DEFAULT_MARKOV_HEAD_TYPE.to_owned()
}

/// Legacy nested and flat drafter schemas.
#[derive(Deserialize)]
struct RawDFlashConfig {
    hidden_size: usize,
    #[serde(default)]
    target_hidden_size: Option<usize>,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    head_dim: usize,
    vocab_size: usize,
    #[serde(default)]
    draft_vocab_size: Option<usize>,
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
    #[serde(default)]
    selector_rank: Option<usize>,
    #[serde(default)]
    selector_top_k: Option<usize>,
    /// DeepSpec declares anchors; native DFlash2 omits them.
    #[serde(default)]
    num_anchors: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawDFlash2TransformerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawDFlash2ProposalMethod {
    proposal_type: String,
    speculative_tokens: usize,
    verifier_accept_k: usize,
    #[serde(default)]
    accept_tolerance: f32,
}

#[derive(Debug, Deserialize)]
struct RawDFlash2SpeculatorsConfig {
    algorithm: String,
    default_proposal_method: String,
    #[serde(default)]
    proposal_methods: Vec<RawDFlash2ProposalMethod>,
}

#[derive(Debug, Deserialize)]
struct RawDFlash2Config {
    transformer_layer_config: RawDFlash2TransformerConfig,
    aux_hidden_state_layer_ids: Vec<usize>,
    block_size: usize,
    mask_token_id: u32,
    draft_vocab_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    sample_from_anchor: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    target_hidden_size: Option<usize>,

    #[serde(default)]
    conv_kernel_size: Option<usize>,
    #[serde(default)]
    conv_group_size: Option<usize>,

    #[serde(default)]
    sliding_window_non_causal: Option<bool>,

    speculators_config: RawDFlash2SpeculatorsConfig,
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
        let json: Value = serde_json::from_str(&content)?;

        let model_type = match json.get("speculators_model_type") {
            None => None,
            Some(value) => Some(
                value
                    .as_str()
                    .context("speculators_model_type must be a string")?
                    .to_owned(),
            ),
        };

        let architecture_marker = match json.get("architectures") {
            None => false,
            Some(value) => value
                .as_array()
                .context("architectures must be an array")?
                .iter()
                .any(|value| value.as_str() == Some("DFlash2DraftModel")),
        };

        let config = match (model_type.as_deref(), architecture_marker) {
            (Some("dflash2"), true) => {
                let raw: RawDFlash2Config = serde_json::from_value(json)?;
                Self::from_dflash2_raw(raw)
            }
            (None | Some("dflash" | "dspark"), false) => {
                let raw: RawDFlashConfig = serde_json::from_value(json)?;
                Self::from_legacy_raw(&raw)
            }
            (Some("dflash2"), false) => {
                bail!(
                    "DFlash2 config declares speculators_model_type=dflash2 but not DFlash2DraftModel"
                )
            }
            (None, true) => {
                bail!(
                    "DFlash2 config declares DFlash2DraftModel but missing speculators_model_type"
                )
            }
            (Some(kind), true) => {
                bail!("unsupported speculator type {kind:?} for DFlash2DraftModel")
            }
            (Some(kind), false) => {
                bail!("unsupported speculator model type {kind:?}")
            }
        }?;
        config.validate_proposal()?;
        Ok(config)
    }

    fn from_legacy_raw(raw: &RawDFlashConfig) -> Result<Self> {
        let nested_rope_theta = raw.rope_parameters.as_ref().map(|value| value.rope_theta);
        let rope_theta = match (raw.rope_theta, nested_rope_theta) {
            (Some(flat), Some(nested)) => {
                ensure!(
                    flat.to_bits() == nested.to_bits(),
                    "legacy DFlash rope_theta and rope_parameters.rope_theta disagree"
                );
                flat
            }
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => {
                bail!("drafter config missing rope_theta / rope_parameters.rope_theta")
            }
        };

        let (mask_token_id, target_layer_ids) = match raw.dflash_config.as_ref() {
            Some(inner) => {
                if let Some(flat) = raw.mask_token_id {
                    ensure!(
                        flat == inner.mask_token_id,
                        "legacy DFlash mask_token_id and dflash_config.mask_token_id disagree"
                    );
                }
                if let Some(flat) = raw.target_layer_ids.as_ref() {
                    ensure!(
                        flat == &inner.target_layer_ids,
                        "legacy DFlash target_layer_ids and dflash_config.target_layer_ids disagree"
                    );
                }
                (inner.mask_token_id, inner.target_layer_ids.clone())
            }
            None => (
                raw.mask_token_id
                    .context("drafter config missing mask_token_id")?,
                raw.target_layer_ids
                    .clone()
                    .context("drafter config missing target_layer_ids")?,
            ),
        };

        let markov_rank = raw.markov_rank;
        let markov_head_type = raw.markov_head_type.clone();
        ensure!(
            markov_rank > 0 || markov_head_type == default_markov_head_type(),
            "legacy DFlash markov_head_type must be \"vanilla\" when markov_rank is zero"
        );
        if let Some(num_anchors) = raw.num_anchors {
            ensure!(
                num_anchors > 0,
                "legacy DFlash num_anchors must be positive when declared"
            );
        }
        ensure!(
            markov_rank == 0 || raw.num_anchors.is_some(),
            "legacy DFlash Markov proposals must declare num_anchors"
        );

        let proposal = match (markov_rank, raw.selector_rank, raw.selector_top_k) {
            (0, None, None) => DFlashProposal::PlainArgmax,
            (rank, None, None) => DFlashProposal::Markov {
                rank,
                head_type: markov_head_type,
            },
            (0, Some(rank), Some(top_k)) => DFlashProposal::TopKSelector { rank, top_k },
            (0, _, _) => {
                bail!("legacy DFlash selector_rank and selector_top_k must be declared together")
            }
            (_, Some(_), _) | (_, _, Some(_)) => {
                bail!("legacy DFlash cannot enable both Markov and top-k selector proposals")
            }
        };

        Ok(Self {
            hidden_size: raw.hidden_size,
            target_hidden_size: raw.target_hidden_size.unwrap_or(raw.hidden_size),
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            num_target_layers: Some(raw.num_target_layers),
            block_size: raw.block_size,
            mask_token_id,
            target_layer_ids,
            draft_vocab_size: raw.draft_vocab_size.unwrap_or(raw.vocab_size),
            proposal,
            layout: if raw.num_anchors.is_some() {
                DFlashLayout::AnchorFirst
            } else {
                DFlashLayout::AnchorDrop
            },
            head_source: DFlashHeadSource::Target,
            enable_confidence_head: raw.enable_confidence_head,
            dynamic_convolution: None,
            sliding_window: None,
        })
    }

    fn from_dflash2_raw(raw: RawDFlash2Config) -> Result<Self> {
        let RawDFlash2Config {
            transformer_layer_config,
            aux_hidden_state_layer_ids,
            block_size,
            mask_token_id,
            draft_vocab_size,
            selector_rank,
            selector_top_k,
            sample_from_anchor,
            tie_word_embeddings,
            target_hidden_size,
            conv_kernel_size,
            conv_group_size,
            sliding_window_non_causal,
            speculators_config,
        } = raw;

        let RawDFlash2TransformerConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_parameters,
            max_position_embeddings,
            sliding_window,
            use_sliding_window,
            layer_types,
            tie_word_embeddings: transformer_tie_word_embeddings,
        } = transformer_layer_config;

        let tie_word_embeddings = tie_word_embeddings
            .or(transformer_tie_word_embeddings)
            .unwrap_or(false);

        ensure!(
            speculators_config.algorithm == "dflash2",
            "DFlash2 speculators_config.algorithm must be \"dflash2\", got {:?}",
            speculators_config.algorithm
        );
        ensure!(
            speculators_config.default_proposal_method == "greedy",
            "DFlash2 proposal method {:?} is not supported in Phase 1",
            speculators_config.default_proposal_method
        );
        ensure!(
            speculators_config.proposal_methods.len() == 1,
            "DFlash2 Phase 1 requires exactly one proposal method, got {}",
            speculators_config.proposal_methods.len()
        );
        let proposal_method = &speculators_config.proposal_methods[0];
        ensure!(
            proposal_method.proposal_type == "greedy",
            "DFlash2 proposal type {:?} is not supported in Phase 1",
            proposal_method.proposal_type
        );
        ensure!(
            proposal_method.accept_tolerance == 0.0,
            "DFlash2 accept_tolerance {} is not supported in Phase 1",
            proposal_method.accept_tolerance
        );
        let expected_speculative_tokens = if sample_from_anchor {
            block_size
        } else {
            block_size.saturating_sub(1)
        };
        ensure!(
            proposal_method.speculative_tokens == expected_speculative_tokens,
            "DFlash2 proposal speculative_tokens {} does not match block/layout expectation {}",
            proposal_method.speculative_tokens,
            expected_speculative_tokens
        );
        ensure!(
            proposal_method.verifier_accept_k == 1,
            "DFlash2 verifier_accept_k {} is not supported in Phase 1",
            proposal_method.verifier_accept_k
        );
        ensure!(
            draft_vocab_size == vocab_size,
            "DFlash2 draft_vocab_size {} must match transformer vocab_size {} for the full-vocabulary selector",
            draft_vocab_size,
            vocab_size
        );
        let dynamic_convolution = match (conv_kernel_size, conv_group_size) {
            (Some(kernel_size), Some(group_size)) => {
                ensure!(
                    kernel_size > 0 && group_size > 0,
                    "DFlash2 convolution kernel_size and group_size must be positive"
                );
                Some(DynamicConv {
                    kernel_size,
                    group_size,
                })
            }
            (None, None) => None,
            _ => bail!("DFlash2 convolution fields must be declared together"),
        };

        ensure!(
            layer_types.is_empty() || layer_types.len() == num_hidden_layers,
            "DFlash2 transformer layer_types length {} does not match num_hidden_layers {}",
            layer_types.len(),
            num_hidden_layers
        );
        for (layer_idx, kind) in layer_types.iter().enumerate() {
            ensure!(
                matches!(kind.as_str(), "full_attention" | "sliding_attention"),
                "DFlash2 transformer layer_types[{}] has unsupported value {:?}",
                layer_idx,
                kind
            );
        }
        let has_non_full_layer = layer_types.iter().any(|kind| kind == "sliding_attention");
        let declares_sliding = use_sliding_window
            || sliding_window.is_some()
            || sliding_window_non_causal.is_some()
            || has_non_full_layer;
        let sliding_window = if declares_sliding {
            let window_size = sliding_window
                .context("DFlash2 sliding-window capability is missing sliding_window")?;
            ensure!(
                window_size > 0,
                "DFlash2 sliding_window must be positive, got {}",
                window_size
            );
            let non_causal = sliding_window_non_causal.context(
                "DFlash2 sliding-window capability is missing sliding_window_non_causal",
            )?;
            Some(SlidingWindow {
                window_size,
                non_causal,
                enabled: use_sliding_window,
                layer_types,
            })
        } else {
            None
        };

        let proposal = DFlashProposal::TopKSelector {
            rank: selector_rank,
            top_k: selector_top_k,
        };

        let layout = if sample_from_anchor {
            DFlashLayout::AnchorFirst
        } else {
            DFlashLayout::AnchorDrop
        };

        Ok(Self {
            hidden_size,
            target_hidden_size: target_hidden_size.unwrap_or(hidden_size),
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta: rope_parameters.rope_theta,
            max_position_embeddings,

            num_target_layers: None,

            block_size,
            mask_token_id,
            target_layer_ids: aux_hidden_state_layer_ids,
            draft_vocab_size,
            proposal,
            layout,
            head_source: if tie_word_embeddings {
                DFlashHeadSource::Target
            } else {
                DFlashHeadSource::DraftOutput
            },

            enable_confidence_head: false,

            dynamic_convolution,
            sliding_window,
        })
    }

    /// Whether this config selects the DSpark Markov path.
    pub(crate) fn uses_markov_head(&self) -> bool {
        matches!(&self.proposal, DFlashProposal::Markov { .. })
    }

    pub(crate) fn uses_selector(&self) -> bool {
        matches!(&self.proposal, DFlashProposal::TopKSelector { .. })
    }

    pub(crate) fn markov_rank(&self) -> usize {
        match &self.proposal {
            DFlashProposal::Markov { rank, .. } => *rank,
            _ => 0,
        }
    }

    /// Whether position zero is a real draft token in this checkpoint layout.
    pub(crate) fn anchor_first(&self) -> bool {
        matches!(&self.layout, DFlashLayout::AnchorFirst)
    }

    /// Validate normalized proposal and vocabulary invariants.
    fn validate_proposal(&self) -> Result<()> {
        ensure!(
            self.block_size >= 2,
            "DFlash block_size must be at least 2, got {}",
            self.block_size
        );
        match &self.proposal {
            DFlashProposal::PlainArgmax => {}
            DFlashProposal::Markov { rank, head_type } => {
                ensure!(*rank > 0, "Markov proposal rank must be positive");
                ensure!(
                    head_type == DEFAULT_MARKOV_HEAD_TYPE,
                    "DSpark markov_head_type {:?} not supported (only \"vanilla\")",
                    head_type
                );
            }
            DFlashProposal::TopKSelector { rank, top_k } => {
                ensure!(*rank > 0, "DFlash selector rank must be positive");
                ensure!(
                    *top_k == DFLASH2_SELECTOR_TOP_K,
                    "DFlash selector_top_k must be {}, got {}",
                    DFLASH2_SELECTOR_TOP_K,
                    top_k
                );
                ensure!(
                    self.draft_vocab_size >= *top_k,
                    "DFlash selector_top_k {} exceeds draft vocabulary size {}",
                    top_k,
                    self.draft_vocab_size
                );
            }
        }
        ensure!(
            self.draft_vocab_size > 0 && self.draft_vocab_size <= self.vocab_size,
            "DFlash draft_vocab_size {} must be in 1..={}",
            self.draft_vocab_size,
            self.vocab_size,
        );
        Ok(())
    }

    /// Reject capabilities outside the Phase 1 execution path.
    pub(crate) fn validate_runtime_capabilities(&self) -> Result<()> {
        if let Some(conv) = &self.dynamic_convolution {
            bail!(
                "DFlash2 dynamic convolution (kernel_size={}, group_size={}) is not supported in Phase 1",
                conv.kernel_size,
                conv.group_size
            );
        }

        if let Some(window) = &self.sliding_window {
            bail!(
                "DFlash2 sliding-window attention (window_size={}, non_causal={}, use_sliding_window={}, layer_types={:?}) is not supported in Phase 1",
                window.window_size,
                window.non_causal,
                window.enabled,
                window.layer_types
            );
        }

        if matches!(
            (&self.proposal, &self.layout),
            (
                DFlashProposal::TopKSelector { .. },
                DFlashLayout::AnchorFirst
            )
        ) {
            bail!("DFlash Phase 1 supports only anchor-drop selector checkpoints");
        }

        Ok(())
    }

    pub(crate) fn validate_for_target(&self, target: &Config) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size == target.hidden_size,
            "DFlash hidden_size {} does not match target {}",
            self.hidden_size,
            target.hidden_size
        );
        anyhow::ensure!(
            self.target_hidden_size == target.hidden_size,
            "DFlash target_hidden_size {} does not match target {}",
            self.target_hidden_size,
            target.hidden_size
        );
        if let Some(num_target_layers) = self.num_target_layers {
            anyhow::ensure!(
                num_target_layers == target.num_hidden_layers,
                "DFlash num_target_layers {} does not match target layers {}",
                num_target_layers,
                target.num_hidden_layers
            );
        }
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
            self.draft_vocab_size == target.vocab_size,
            "DFlash draft_vocab_size {} must match target vocab_size {} for full-vocabulary logits",
            self.draft_vocab_size,
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
            u64::from(self.mask_token_id) < target.vocab_size as u64,
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
