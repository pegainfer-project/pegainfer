use pegainfer_kernels::ops::NumericPolicy;

use crate::DecodeOverlap;
use crate::config::TensorParallelConfig;

/// The production path for Qwen3 decode projections.
///
/// Fusion is one atomic decode topology: QKV and gate/up are either both
/// fused or both split. Unsupported runtime topologies fail closed to
/// the established split-GEMM path. The policy is intentionally GPU-agnostic;
/// each device still tunes its own cuBLASLt projection shapes at startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeProjectionPath {
    Split,
    FusedQkvGateUp,
}

impl DecodeProjectionPath {
    pub(crate) const fn is_fused(self) -> bool {
        matches!(self, Self::FusedQkvGateUp)
    }
}

pub(crate) fn select_decode_projection_path(
    tensor_parallel: TensorParallelConfig,
    numeric_policy: NumericPolicy,
    decode_overlap: DecodeOverlap,
    dflash_enabled: bool,
) -> DecodeProjectionPath {
    if tensor_parallel.world_size == 1
        && numeric_policy == NumericPolicy::Tuned
        && matches!(decode_overlap, DecodeOverlap::Off)
        && !dflash_enabled
    {
        DecodeProjectionPath::FusedQkvGateUp
    } else {
        DecodeProjectionPath::Split
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enables_qualified_qwen3_tp1_decode() {
        let path = select_decode_projection_path(
            TensorParallelConfig::default(),
            NumericPolicy::Tuned,
            DecodeOverlap::Off,
            false,
        );
        assert_eq!(path, DecodeProjectionPath::FusedQkvGateUp);
        assert!(path.is_fused());
    }

    #[test]
    fn fails_closed_outside_qualified_runtime_environment() {
        let cases = [
            select_decode_projection_path(
                TensorParallelConfig {
                    rank: 0,
                    world_size: 2,
                },
                NumericPolicy::Tuned,
                DecodeOverlap::Off,
                false,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Pin,
                DecodeOverlap::Off,
                false,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::PerToken,
                DecodeOverlap::Off,
                false,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::SharedSm,
                false,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::GreenCtx { decode_pct: 20 },
                false,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::Off,
                true,
            ),
        ];

        assert!(
            cases
                .into_iter()
                .all(|path| path == DecodeProjectionPath::Split && !path.is_fused())
        );
    }
}
