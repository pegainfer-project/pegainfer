//! [`schema`] turns a config into the tensor contract the checkpoint must
//! satisfy; [`read`] turns a checkpoint directory into observed tensors; and
//! [`validate`] holds the two against each other.
//!
//! Ported from `pegainfer-gemma4/src/manifest/`, which is the in-tree precedent
//! for a whole-checkpoint manifest that runs before any device work. Three
//! deltas from that port, all forced by this checkpoint:
//!
//! - the LM head is **untied**, so both the embedding and the head are required
//!   (Gemma 4 refuses an untied head);
//! - shapes include rank-3 tensors — the depthwise conv weight and the fused
//!   expert stacks — which Gemma 4 never needed because its experts are stored
//!   as four 2-D quantized tensors each;
//! - the skip set is `model.visual.` and `mtp.`, and skipping is mandatory rather
//!   than optional because the vision tensors share shards with the text ones — at
//!   the frozen revision all 333 of them are in shard 1, next to 16 text tensors —
//!   so a loader reading whole shards must classify per tensor.
//!
//! [`read`] has no Gemma 4 counterpart: that line verifies against a checkpoint
//! small enough to hold whole, while this one is 360 GB across 131 shards, so
//! the manifest has to be able to say something useful about the shards that are
//! on disk without pretending to know about the ones that are not.

use safetensors::Dtype;

pub(crate) mod read;
pub(crate) mod schema;
pub(crate) mod validate;

/// Every weight tensor this line reads is BF16; the frozen checkpoint stores the
/// GDN scalars as BF16 too, so there is no per-tensor f32 exception here.
pub(crate) const EXPECTED_DTYPE: Dtype = Dtype::BF16;

/// The n-gram partition metadata is stored as integers, not weights.
pub(crate) const INTEGER_INDEX_DTYPE: Dtype = Dtype::I64;

/// Bytes per element, for the dtypes this contract can carry. Panics rather than
/// defaulting: a tensor whose dtype is outside this set would otherwise be
/// counted as zero bytes and shrink `Manifest::weight_bytes` silently.
pub(crate) fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::U8 | Dtype::I8 | Dtype::BOOL => 1,
        other => panic!("no contract tensor carries {other:?}"),
    }
}
