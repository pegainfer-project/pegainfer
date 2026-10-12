//! [`schema`] derives the contract from a config, [`read`] observes what a checkpoint
//! directory carries, and [`validate`] holds the two against each other.
//!
//! Ported from `pegainfer-gemma4/src/manifest/` (which runs before any device work),
//! with three deltas this checkpoint forces: the LM head is untied, so both ends are
//! required; shapes include rank-3 tensors — the depthwise conv and the fused expert
//! stacks, which carry no `.weight` suffix; and the skip set (`model.visual.`,
//! `mtp.`) is mandatory, because the vision tensors share shards with the text ones.
//! [`read`] has no Gemma 4 counterpart — 360 GB across 131 shards means it must speak
//! about the shards on disk without pretending to know the rest.

use safetensors::Dtype;

pub(crate) mod read;
pub(crate) mod schema;
pub(crate) mod validate;

/// Every weight tensor this line reads is BF16, the GDN scalars included — there is
/// no per-tensor f32 exception.
pub(crate) const EXPECTED_DTYPE: Dtype = Dtype::BF16;

/// The n-gram partition metadata is stored as integers, not weights.
pub(crate) const INTEGER_INDEX_DTYPE: Dtype = Dtype::I64;

/// Bytes per element for the dtypes this contract carries. Panics on any other dtype
/// rather than counting it as zero.
pub(crate) fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::U8 | Dtype::I8 | Dtype::BOOL => 1,
        other => panic!("no contract tensor carries {other:?}"),
    }
}
