use pegainfer_kernels::ops::NumericPolicy;

use crate::DecodeOverlap;
use crate::config::TensorParallelConfig;

/// The production path for Qwen3 decode projections.
///
/// Only QKV is fused. Gate/up remains on the established split-GEMM path until
/// its component and end-to-end evidence independently qualifies it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeProjectionPath {
    Split,
    FusedQkv,
}

impl DecodeProjectionPath {
    pub(crate) const fn fuses_qkv(self) -> bool {
        matches!(self, Self::FusedQkv)
    }
}

/// Current-head production qualifications. A row is
/// `(SM, hidden_size, q_dim, kv_dim, num_hidden_layers)`.
///
/// Keep this table deliberately narrow: cuBLASLt selects an algorithm inside
/// one GEMM shape, but does not decide whether the packed topology beats three
/// split GEMMs. Every added row therefore requires component and end-to-end A/B
/// evidence for that exact device architecture and model geometry.
const QUALIFIED_QKV: &[(i32, usize, usize, usize, usize)] =
    &[(89, 2560, 4096, 1024, 36), (89, 4096, 4096, 1024, 36)];

pub(crate) fn select_decode_projection_path(
    tensor_parallel: TensorParallelConfig,
    numeric_policy: NumericPolicy,
    decode_overlap: DecodeOverlap,
    dflash_enabled: bool,
    device_sm: i32,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_hidden_layers: usize,
) -> (DecodeProjectionPath, &'static str) {
    if tensor_parallel.world_size != 1 {
        return (DecodeProjectionPath::Split, "tensor_parallel");
    }
    if numeric_policy != NumericPolicy::Tuned {
        return (DecodeProjectionPath::Split, "numeric_policy");
    }
    if !matches!(decode_overlap, DecodeOverlap::Off) {
        return (DecodeProjectionPath::Split, "decode_overlap");
    }
    if dflash_enabled {
        return (DecodeProjectionPath::Split, "dflash");
    }
    if !QUALIFIED_QKV.contains(&(device_sm, hidden_size, q_dim, kv_dim, num_hidden_layers)) {
        return (DecodeProjectionPath::Split, "unqualified_sm_or_geometry");
    }
    (DecodeProjectionPath::FusedQkv, "qualified_sm_and_geometry")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enables_qualified_qwen3_tp1_decode() {
        let (path, reason) = select_decode_projection_path(
            TensorParallelConfig::default(),
            NumericPolicy::Tuned,
            DecodeOverlap::Off,
            false,
            89,
            2560,
            4096,
            1024,
            36,
        );
        assert_eq!(path, DecodeProjectionPath::FusedQkv);
        assert!(path.fuses_qkv());
        assert_eq!(reason, "qualified_sm_and_geometry");
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
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Pin,
                DecodeOverlap::Off,
                false,
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::PerToken,
                DecodeOverlap::Off,
                false,
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::SharedSm,
                false,
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::GreenCtx { decode_pct: 20 },
                false,
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::Off,
                true,
                89,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::Off,
                false,
                86,
                2560,
                4096,
                1024,
                36,
            ),
            select_decode_projection_path(
                TensorParallelConfig::default(),
                NumericPolicy::Tuned,
                DecodeOverlap::Off,
                false,
                89,
                3072,
                4096,
                1024,
                36,
            ),
        ];

        assert!(
            cases
                .into_iter()
                .all(|(path, _)| path == DecodeProjectionPath::Split && !path.fuses_qkv())
        );
    }
}
