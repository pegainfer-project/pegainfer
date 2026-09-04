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
    local_linear_num_key_heads: usize,
    local_linear_num_value_heads: usize,
    local_linear_q_dim: usize,
    local_linear_v_dim: usize,
    local_linear_qkv_dim: usize,
}

impl LocalGeometry {
    /// Validate `config` against `tp`, then derive this rank's local dimensions.
    ///
    /// Fails on unsupported combinations before expensive loading:
    /// - every sharded model dimension must divide evenly by `world_size`
    ///   (linear-attention heads included: Phase 2b shards them per rank);
    /// - `rank < world_size` and `world_size >= 1` are guaranteed by
    ///   `TensorParallelConfig::try_from`.
    ///
    /// CUDA Graph under TP is gated at executor startup on
    /// [`LocalGeometry::local_decode_group_is_compiled`] (P2c): uncompiled GQA
    /// groups keep the batched eager path instead of failing validation here.
    pub(crate) fn try_new(
        config: &Config35,
        tp: TensorParallelConfig,
    ) -> Result<Self, ConfigError> {
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
        // Phase 2b shards linear attention/GDR heads per rank; fail closed on
        // indivisible head counts rather than falling back to replication.
        if !config.linear_num_key_heads.is_multiple_of(tp.world_size()) {
            return Err(ConfigError::TpIndivisible {
                field: "linear_num_key_heads",
                value: config.linear_num_key_heads,
                world_size: tp.world_size(),
            });
        }
        if !config
            .linear_num_value_heads
            .is_multiple_of(tp.world_size())
        {
            return Err(ConfigError::TpIndivisible {
                field: "linear_num_value_heads",
                value: config.linear_num_value_heads,
                world_size: tp.world_size(),
            });
        }

        let local_num_attention_heads = config.num_attention_heads / tp.world_size();
        let local_num_key_value_heads = config.num_key_value_heads / tp.world_size();
        let local_intermediate_size = config.intermediate_size / tp.world_size();
        let local_full_attn_q_dim = local_num_attention_heads * config.head_dim;
        let local_full_attn_kv_dim = local_num_key_value_heads * config.head_dim;
        let local_linear_num_key_heads = config.linear_num_key_heads / tp.world_size();
        let local_linear_num_value_heads = config.linear_num_value_heads / tp.world_size();
        // Local q/k segment rows of the fused linear-attention qkv projection;
        // q is keyed by key heads (one key head per value-head group).
        let local_linear_q_dim = local_linear_num_key_heads * config.linear_key_head_dim;
        let local_linear_v_dim = local_linear_num_value_heads * config.linear_value_head_dim;

        Ok(Self {
            tp,
            local_num_attention_heads,
            local_num_key_value_heads,
            local_intermediate_size,
            local_full_attn_q_dim,
            local_full_attn_kv_dim,
            local_full_attn_gated_q_dim: local_full_attn_q_dim * 2,
            local_linear_num_key_heads,
            local_linear_num_value_heads,
            local_linear_q_dim,
            local_linear_v_dim,
            // [q_local | k_local | v_local] in storage order; k == q.
            local_linear_qkv_dim: local_linear_q_dim * 2 + local_linear_v_dim,
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

    // ── Linear-attention local dims (Phase 2b TP sharding) ────────────────
    // TP1 contract: at world_size 1 every local dim equals the global dim, so
    // all linear-attention kernels/buffers/state keep their pre-TP shapes.

    pub(crate) fn local_linear_num_key_heads(&self) -> usize {
        self.local_linear_num_key_heads
    }

    pub(crate) fn local_linear_num_value_heads(&self) -> usize {
        self.local_linear_num_value_heads
    }

    /// Local fused qkv rows: [q_local | k_local | v_local] in storage order.
    pub(crate) fn local_linear_qkv_dim(&self) -> usize {
        self.local_linear_qkv_dim
    }

    /// Local z projection output dimension (equals local v dim).
    pub(crate) fn local_linear_z_dim(&self) -> usize {
        self.local_linear_v_dim
    }

    /// TP-local decode GQA group supportability for the eager decode path:
    /// whether the FlashInfer batch-decode kernel supports the rank-local
    /// q-per-kv group. Deterministic per model and identical on every rank,
    /// so both arms are collective-safe (the reroute adds no collectives).
    /// At world_size 1 this equals `Config35::decode_group_is_compiled`.
    pub(crate) fn local_decode_group_is_compiled(&self) -> bool {
        pegainfer_core::ops::SUPPORTED_GQA_GROUP_SIZES
            .contains(&(self.local_num_attention_heads / self.local_num_key_value_heads))
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
        let geom = LocalGeometry::try_new(&cfg, tp).unwrap();
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
        let mut err = LocalGeometry::try_new(&cfg, tp).unwrap_err();
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
        err = LocalGeometry::try_new(&broken, tp).unwrap_err();
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
        err = LocalGeometry::try_new(&broken, tp).unwrap_err();
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
    fn requires_linear_attention_head_divisibility() {
        let tp = TensorParallelConfig::try_from((1, 2)).unwrap();
        let mut broken = config();
        broken.linear_num_key_heads = 17;
        let err = LocalGeometry::try_new(&broken, tp).unwrap_err();
        assert_eq!(
            err,
            ConfigError::TpIndivisible {
                field: "linear_num_key_heads",
                value: 17,
                world_size: 2,
            }
        );

        let mut broken = config();
        broken.linear_num_value_heads = 31;
        let err = LocalGeometry::try_new(&broken, tp).unwrap_err();
        assert_eq!(
            err,
            ConfigError::TpIndivisible {
                field: "linear_num_value_heads",
                value: 31,
                world_size: 2,
            }
        );
    }

    #[test]
    fn computes_tp2_linear_attention_local_dimensions() {
        let cfg = config();
        let tp = TensorParallelConfig::try_from((1, 2)).unwrap();
        let geom = LocalGeometry::try_new(&cfg, tp).unwrap();
        assert_eq!(geom.local_linear_num_key_heads(), 8);
        assert_eq!(geom.local_linear_num_value_heads(), 16);
        assert_eq!(geom.local_linear_q_dim, 1024);
        assert_eq!(geom.local_linear_v_dim, 2048);
        assert_eq!(geom.local_linear_qkv_dim(), 4096);
        assert_eq!(geom.local_linear_z_dim(), 2048);
    }

    #[test]
    fn tp1_linear_attention_local_dimensions_equal_global() {
        // TP1 invariant: every local dim equals the global dim, keeping TP1
        // numerics byte-identical to pre-sharding execution.
        let cfg = config();
        let geom = LocalGeometry::try_new(&cfg, TensorParallelConfig::default()).unwrap();
        assert_eq!(geom.local_linear_num_key_heads(), 16);
        assert_eq!(geom.local_linear_num_value_heads(), 32);
        assert_eq!(geom.local_linear_q_dim, 2048);
        assert_eq!(geom.local_linear_v_dim, 4096);
        let global_qkv =
            2 * (cfg.linear_num_key_heads * cfg.linear_key_head_dim) + cfg.linear_attn_z_dim();
        assert_eq!(geom.local_linear_qkv_dim(), global_qkv);
        assert_eq!(geom.local_linear_z_dim(), cfg.linear_attn_z_dim());
    }
}
