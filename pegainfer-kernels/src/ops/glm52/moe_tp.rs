//! GLM5.2 TP4 topology constants for the prefill-only path.
//!
//! Decode-side LL packet kernels (phase MoE + attention AR) were removed.
//! Prefill TP4 reduces with NCCL bf16 all-reduce in the model crate.

/// Max TP width (TP4).
pub const GLM52_TP_MAX_RANKS: usize = 4;
pub const GLM52_TP_HIDDEN: usize = 6144;
pub const GLM52_TP_BANK_EXPERTS: usize = 257;
/// Prefill/decode row capacity shared by scratch sizing and vocab pack.
pub const GLM52_TP_TOKENS: usize = 8;
pub const GLM52_TP_UNION_MAX: usize = GLM52_TP_TOKENS * (8 + 1);

/// Tensor-parallel width for GLM5.2. Only TP4 remains, and only for prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52TpTopology {
    Tp4,
}

impl Glm52TpTopology {
    pub const fn ranks(self) -> usize {
        4
    }

    pub const fn slice_i(self) -> usize {
        2048 / self.ranks()
    }

    pub const fn slice_rows(self) -> usize {
        2 * self.slice_i()
    }
}
