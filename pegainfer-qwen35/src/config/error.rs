//! Typed configuration-load errors for the Qwen3.5 model line.
//!
//! Every failure mode here is a *static* invariant violation that is detected
//! before any weights are mapped or any CUDA memory is allocated. Keeping these
//! as variants (instead of free-form `anyhow!` strings) lets callers and tests
//! branch on the exact violation and lets the loader reject an invalid model
//! file fail-closed with a precise message.

/// A Qwen3.5 config that could not be turned into a validated `Config35`.
///
/// These cover the model config (`RawConfig -> Config35`), the tensor-parallel
/// assignment and derived local geometry (`TensorParallelConfig`/`LocalGeometry`),
/// and the cross-field + kernel/TP compatibility rules that cut across both.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ConfigError {
    // ---- model config (cross-field + kernel AOT invariants) ----
    #[error("Qwen3.5 config missing tie_word_embeddings")]
    MissingTieWordEmbeddings,
    #[error("Qwen3.5 config missing max_position_embeddings")]
    MissingMaxPositionEmbeddings,
    #[error("unknown layer type: {0}")]
    UnknownLayerType(String),
    #[error("layer_types length {actual} != num_hidden_layers {expected}")]
    LayerTypeCountMismatch { actual: usize, expected: usize },
    #[error("Qwen3.5 rotary_dim must be positive")]
    ZeroRotaryDim,
    #[error("Qwen3.5 max_position_embeddings must be positive")]
    NonPositiveMaxPositionEmbeddings,
    #[error(
        "Qwen3.5 GDN Triton-AOT kernels are baked for key/value head dim {expected_key}/{expected_value}; \
         config has {key}/{value} (dims are baked into the AOT signatures in pegainfer-kernels/build.rs)"
    )]
    GdnAotHeadDimMismatch {
        key: usize,
        value: usize,
        expected_key: usize,
        expected_value: usize,
    },
    #[error(
        "Qwen3.5 full-attention kernels are baked for head_dim {expected}; config has {actual}"
    )]
    FullAttnHeadDimMismatch { expected: usize, actual: usize },
    #[error(
        "Qwen3.5 linear conv decode kernels support kernel_dim in 1..={max}; config has {actual}"
    )]
    LinearConvKernelDim { max: usize, actual: usize },
    #[error(
        "Qwen3.5 GDN kernels require linear_num_value_heads ({value_heads}) divisible by \
         linear_num_key_heads ({key_heads})"
    )]
    LinearHeadDivisibility {
        key_heads: usize,
        value_heads: usize,
    },
    #[error(
        "Qwen3.5 num_attention_heads ({attention_heads}) must be a positive multiple of \
         num_key_value_heads ({key_value_heads})"
    )]
    AttentionHeadDivisibility {
        attention_heads: usize,
        key_value_heads: usize,
    },
    #[error("tokenizer defines ids up to {used} but checkpoint vocab_size is {vocab_size}")]
    EffectiveVocabExceedsCheckpoint { used: usize, vocab_size: usize },

    // ---- tensor-parallel assignment invariants ----
    #[error("tensor_parallel.world_size must be >= 1")]
    TpZeroWorldSize,
    #[error("tensor_parallel.rank {rank} must be < world_size {world_size}")]
    TpRankOutOfRange { rank: usize, world_size: usize },
    #[error(
        "Qwen3.5 tensor parallelism is eager-only; disable CUDA Graph for tp world_size={world_size}"
    )]
    TpRequiresEager { world_size: usize },
    #[error("{field}={value} not divisible by tp world_size={world_size}")]
    TpIndivisible {
        field: &'static str,
        value: usize,
        world_size: usize,
    },
}
