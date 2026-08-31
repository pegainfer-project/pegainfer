//! Tensor-parallel assignment and validated local geometry for Qwen3.5.
//!
//! Directional boundary: this module depends on [`super::model`] (it reads a
//! validated [`Config35`] to derive shard dimensions), but [`super::model`] is
//! TP-agnostic and never depends on it. Downstream code accepts the validated
//! [`LocalGeometry`] type produced here instead of re-deriving shards from a
//! raw `(rank, world_size)` pair.
//!
//! Both [`TensorParallelConfig`] and [`LocalGeometry`] validate at construction:
//! an invalid assignment is unrepresentable, and a model that disagrees with the
//! TP world is rejected before any weights are mapped.

use super::error::ConfigError;
use super::model::Config35;

/// A validated tensor-parallel assignment.
///
/// Fields are private so an invalid state (`world_size == 0`, `rank >=
/// world_size`) cannot be constructed. The only way to obtain one is
/// [`Self::try_from`] or [`Default`] (single-rank).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    rank: usize,
    world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

impl TensorParallelConfig {
    pub(crate) fn rank(self) -> usize {
        self.rank
    }

    pub(crate) fn world_size(self) -> usize {
        self.world_size
    }

    pub(crate) fn is_sharded(self) -> bool {
        self.world_size > 1
    }

    /// Split a global row count into this rank's `(offset, len)`.
    pub(crate) fn shard_range(self, total: usize) -> (usize, usize) {
        let shard_len = total / self.world_size;
        (self.rank * shard_len, shard_len)
    }
}

/// Validate an assignment independently of any model geometry.
impl TryFrom<(usize, usize)> for TensorParallelConfig {
    type Error = ConfigError;

    fn try_from((rank, world_size): (usize, usize)) -> Result<Self, Self::Error> {
        if world_size == 0 {
            return Err(ConfigError::TpZeroWorldSize);
        }
        if rank >= world_size {
            return Err(ConfigError::TpRankOutOfRange { rank, world_size });
        }
        Ok(Self { rank, world_size })
    }
}

/// Validated per-rank shard geometry for a Qwen3.5 model under a TP assignment.
///
/// Built once from a validated model config + TP assignment + cuda-graph mode;
/// every model cross-field and TP/kernel compatibility rule is enforced here.
/// Downstream code sizes buffers and computes shards from this type, never from
/// a raw rank/world-size pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalGeometry {
    tp: TensorParallelConfig,
    local_num_attention_heads: usize,
    local_num_key_value_heads: usize,
    local_intermediate_size: usize,
    local_full_attn_q_dim: usize,
    local_full_attn_kv_dim: usize,
    local_full_attn_gated_q_dim: usize,
}

impl LocalGeometry {
    /// Validate `config` against `tp` and the runtime execution mode, then derive
    /// this rank's local dimensions.
    ///
    /// Fails on unsupported combinations before expensive loading:
    /// - sharded TP demands eager execution (`enable_cuda_graph` off);
    /// - every sharded model dimension must divide evenly by `world_size`
    ///   (linear-attention head counts are intentionally exempt);
    /// - `rank < world_size` and `world_size >= 1` are guaranteed by
    ///   `TensorParallelConfig::try_from`.
    pub(crate) fn try_new(
        config: &Config35,
        tp: TensorParallelConfig,
        enable_cuda_graph: bool,
    ) -> Result<Self, ConfigError> {
        if tp.is_sharded() && enable_cuda_graph {
            return Err(ConfigError::TpRequiresEager {
                world_size: tp.world_size(),
            });
        }
        if !config.num_attention_heads.is_multiple_of(tp.world_size()) {
            return Err(ConfigError::TpIndivisible {
                field: "num_attention_heads",
                value: config.num_attention_heads,
                world_size: tp.world_size(),
            });
        }
        if !config.num_key_value_heads.is_multiple_of(tp.world_size()) {
            return Err(ConfigError::TpIndivisible {
                field: "num_key_value_heads",
                value: config.num_key_value_heads,
                world_size: tp.world_size(),
            });
        }
        if !config.intermediate_size.is_multiple_of(tp.world_size()) {
            return Err(ConfigError::TpIndivisible {
                field: "intermediate_size",
                value: config.intermediate_size,
                world_size: tp.world_size(),
            });
        }

        let local_num_attention_heads = config.num_attention_heads / tp.world_size();
        let local_num_key_value_heads = config.num_key_value_heads / tp.world_size();
        let local_intermediate_size = config.intermediate_size / tp.world_size();
        let local_full_attn_q_dim = local_num_attention_heads * config.head_dim;
        let local_full_attn_kv_dim = local_num_key_value_heads * config.head_dim;

        Ok(Self {
            tp,
            local_num_attention_heads,
            local_num_key_value_heads,
            local_intermediate_size,
            local_full_attn_q_dim,
            local_full_attn_kv_dim,
            local_full_attn_gated_q_dim: local_full_attn_q_dim * 2,
        })
    }

    pub(crate) fn rank(&self) -> usize {
        self.tp.rank()
    }

    pub(crate) fn world_size(&self) -> usize {
        self.tp.world_size()
    }

    pub(crate) fn is_sharded(&self) -> bool {
        self.tp.is_sharded()
    }

    pub(crate) fn shard_range(&self, total: usize) -> (usize, usize) {
        self.tp.shard_range(total)
    }

    pub(crate) fn local_num_attention_heads(&self) -> usize {
        self.local_num_attention_heads
    }

    pub(crate) fn local_num_key_value_heads(&self) -> usize {
        self.local_num_key_value_heads
    }

    pub(crate) fn local_intermediate_size(&self) -> usize {
        self.local_intermediate_size
    }

    pub(crate) fn local_full_attn_q_dim(&self) -> usize {
        self.local_full_attn_q_dim
    }

    pub(crate) fn local_full_attn_kv_dim(&self) -> usize {
        self.local_full_attn_kv_dim
    }

    /// Local gated full-attention q projection output dimension.
    pub(crate) fn local_full_attn_gated_q_dim(&self) -> usize {
        self.local_full_attn_gated_q_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config35 {
        let raw: crate::config::model::RawConfig = serde_json::from_str(
            r#"{
  "max_position_embeddings": 4096,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 151645
  }
}"#,
        )
        .expect("fixture parses");
        Config35::try_from(raw).expect("fixture validates")
    }

    #[test]
    fn default_tensor_parallel_is_tp1() {
        let tp = TensorParallelConfig::default();
        assert_eq!((tp.rank(), tp.world_size()), (0, 1));
        assert!(!tp.is_sharded());
        assert_eq!(tp.shard_range(4096), (0, 4096));
    }

    #[test]
    fn tp2_local_geometry_matches_dense_dims() {
        let cfg = config();
        let tp = TensorParallelConfig::try_from((1, 2)).unwrap();
        let geom = LocalGeometry::try_new(&cfg, tp, false).unwrap();
        assert!(geom.is_sharded());
        assert_eq!(geom.shard_range(4096), (2048, 2048));
        assert_eq!(geom.local_num_attention_heads(), 8);
        assert_eq!(geom.local_num_key_value_heads(), 2);
        assert_eq!(geom.local_intermediate_size(), 4608);
        assert_eq!(geom.local_full_attn_q_dim(), 2048);
        assert_eq!(geom.local_full_attn_kv_dim(), 512);
        assert_eq!(geom.local_full_attn_gated_q_dim(), 4096);
    }

    #[test]
    fn rejects_zero_world_size_and_rank_out_of_range() {
        assert_eq!(
            TensorParallelConfig::try_from((0, 0)),
            Err(ConfigError::TpZeroWorldSize)
        );
        assert_eq!(
            TensorParallelConfig::try_from((2, 2)),
            Err(ConfigError::TpRankOutOfRange {
                rank: 2,
                world_size: 2
            })
        );
    }

    #[test]
    fn rejects_indivisible_dense_dimensions() {
        let tp = TensorParallelConfig::try_from((0, 3)).unwrap();
        let cfg = config();
        let mut err = LocalGeometry::try_new(&cfg, tp, false).unwrap_err();
        assert_eq!(
            err,
            ConfigError::TpIndivisible {
                field: "num_attention_heads",
                value: 16,
                world_size: 3,
            }
        );

        let mut broken = cfg;
        broken.num_attention_heads = 15;
        broken.num_key_value_heads = 4;
        err = LocalGeometry::try_new(&broken, tp, false).unwrap_err();
        assert_eq!(
            err,
            ConfigError::TpIndivisible {
                field: "num_key_value_heads",
                value: 4,
                world_size: 3,
            }
        );

        broken.num_key_value_heads = 3;
        broken.intermediate_size = 9217;
        err = LocalGeometry::try_new(&broken, tp, false).unwrap_err();
        assert_eq!(
            err,
            ConfigError::TpIndivisible {
                field: "intermediate_size",
                value: 9217,
                world_size: 3,
            }
        );
    }

    #[test]
    fn rejects_tensor_parallel_with_cuda_graph() {
        let cfg = config();
        let tp = TensorParallelConfig::try_from((0, 2)).unwrap();
        assert_eq!(
            LocalGeometry::try_new(&cfg, tp, true),
            Err(ConfigError::TpRequiresEager { world_size: 2 })
        );
    }

    #[test]
    fn linear_attention_heads_need_not_divide_world_size() {
        let mut cfg = config();
        cfg.linear_num_key_heads = 17;
        cfg.linear_num_value_heads = 31;
        let tp = TensorParallelConfig::try_from((1, 2)).unwrap();
        LocalGeometry::try_new(&cfg, tp, false).unwrap();
    }
}
