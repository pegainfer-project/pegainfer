//! Row-batched device buffer whose column width is a type fact.
//!
//! The step scratch used to carry `[T, C]` buffers as untyped `DeviceVec`s,
//! and every consumer re-validated `len == tokens * C` at its own entrance —
//! ~30 `ensure!`s all re-checking what the allocation already established.
//! `Rows<C>` moves the invariant to the construction point: the width is a
//! `const` type parameter, the row count travels with the buffer, and
//! `len == tokens * C` holds by construction. Receivers read the row count
//! from the value and the width from the type instead of re-validating
//! either, and passing a `[T, HIDDEN]` buffer where `[T, VOCAB]` is expected
//! stops compiling.
//!
//! TP note: the widths are compile-time because TP1 is the only supported
//! layout (the whole crate pins its shapes to `config.rs` constants). If a
//! sharded layout ever lands, the const parameter becomes a dimension tag
//! whose per-rank width lives in a rank config — the construction-time
//! invariant and the typed signatures survive that move unchanged.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;

/// `[tokens, C]` row-major bf16 device buffer; `data.len() == tokens * C` by
/// construction.
pub(crate) struct Rows<const C: usize> {
    data: CudaSlice<bf16>,
    tokens: usize,
}

impl<const C: usize> Rows<C> {
    pub(crate) fn zeros(ctx: &DeviceContext, tokens: usize) -> Result<Self> {
        ensure!(tokens > 0, "Rows<{C}> needs a positive row count");
        Ok(Self {
            data: ctx.stream.alloc_zeros::<bf16>(tokens * C)?,
            tokens,
        })
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    /// The raw buffer, for the kernel-launch boundary (the kernels crate
    /// takes untyped slices; the width/row facts stop here).
    pub(crate) fn data(&self) -> &CudaSlice<bf16> {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut CudaSlice<bf16> {
        &mut self.data
    }
}
