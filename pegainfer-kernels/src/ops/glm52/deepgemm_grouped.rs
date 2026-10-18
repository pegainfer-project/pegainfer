//! GLM5.2 EP shape constants shared by routed-expert kernels.
//!
//! Historically this module also launched the Hopper DeepGEMM masked
//! grouped GEMM. That path is gone; only the protocol geometry and the
//! W13/W2 operand kind remain.

/// Per-expert row alignment of the DeepEP recv segment layout (a fixed design
/// constant shared with the vendored shim).
pub const GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT: usize = 64;

/// Protocol worst-case rows per local expert under DP8/EP8
/// (`ranks × max_batch` source tokens, each contributing ≤1 row per expert).
pub const GLM52_DEEPGEMM_MASKED_CAP: usize = 64;

/// Local experts per EP8 rank (256 routed / 8).
pub const GLM52_DEEPGEMM_MASKED_GROUPS: usize = 32;

/// Routed expert GEMM operand: gate|up vs down.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52DeepGemmGroupedFp8Kind {
    W13,
    W2,
}

impl Glm52DeepGemmGroupedFp8Kind {
    /// The operand's `(n, k)`.
    pub(crate) fn shape(self) -> (usize, usize) {
        match self {
            Self::W13 => (4096, 6144),
            Self::W2 => (6144, 2048),
        }
    }
}
