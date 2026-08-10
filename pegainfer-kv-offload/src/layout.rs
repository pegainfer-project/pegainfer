//! GPU-side registration geometry: how pegainfer's paged KV allocations map
//! onto pegaflow's per-layer strided arena registration.

use cudarc::driver::CudaStream;
use pegainfer_kv_cache::KvBuffer;

/// bf16 KV cache: every layout stride is counted in elements, bytes are ×2.
const ELEM_SIZE: usize = std::mem::size_of::<half::bf16>();

/// One strided GPU arena to register as one pegaflow "layer": `num_blocks`
/// copy units of `bytes_per_block`, sitting `block_stride_bytes` apart from
/// `base_ptr`. A fused buffer (qwen3) contributes one arena per model layer;
/// a model with sidecar caches (GLM5.2: MLA latent + index-K per layer, two
/// separate allocations sharing pool block ids) contributes several arenas
/// per model layer — pegaflow moves whatever arenas are registered under one
/// block id together, which is what keeps sidecars in lockstep with their
/// main cache.
///
/// `name` keys the arena for the whole engine lifetime (save/load fan across
/// every registered name); it must be unique within the engine.
pub struct KvArena {
    name: String,
    base_ptr: u64,
    num_blocks: usize,
    bytes_per_block: usize,
    block_stride_bytes: usize,
}

/// Per-layer registration geometry fed to pegaflow's one batched call.
///
/// Only `data_ptrs` and `size_bytes` differ per layer; the rest are the same
/// scalar broadcast across all layers (kept as vectors only to feed pegaflow's
/// one batched registration call).
pub(crate) struct Registration {
    pub(crate) layer_names: Vec<String>,
    pub(crate) data_ptrs: Vec<u64>,
    pub(crate) size_bytes: Vec<usize>,
    pub(crate) num_blocks: Vec<usize>,
    pub(crate) bytes_per_block: Vec<usize>,
    pub(crate) kv_stride_bytes: Vec<usize>,
    pub(crate) segments: Vec<usize>,
    pub(crate) block_stride_bytes: Vec<usize>,
}

impl Registration {
    /// Map the fused page-first buffer to pegaflow's per-layer view.
    ///
    /// Each model layer registers as one pegaflow "layer". Within a page the
    /// layout is K then V back-to-back (`layer_stride = 2·kv_block_len`), so a
    /// layer's K and V are *contiguous* — one single segment of `layer_stride`
    /// bytes copies both, and pegaflow's K/V-split path (which needs the two
    /// segments set apart, `kv_stride > bytes_per_block`) does not apply here.
    /// What is *not* contiguous is consecutive blocks of one layer: the fused
    /// buffer interleaves all layers within a page, so they sit `page_stride`
    /// apart. That gap (stride ≠ copy size) is exactly what `block_stride_bytes`
    /// decouples.
    pub(crate) fn from_buffer(buffer: &KvBuffer, stream: &CudaStream) -> Self {
        let layout = buffer.layout();
        let num_blocks = buffer.num_blocks();
        let base_ptr = buffer.device_ptr(stream);

        // One block's copy unit for a layer = its whole [K|V] span in a page.
        let layer_bytes = layout.layer_stride * ELEM_SIZE;
        let page_stride_bytes = layout.page_stride * ELEM_SIZE;

        let arenas: Vec<KvArena> = (0..layout.num_layers)
            .map(|layer| KvArena {
                name: layer.to_string(),
                base_ptr: base_ptr + (layer * layer_bytes) as u64,
                num_blocks,
                bytes_per_block: layer_bytes,
                block_stride_bytes: page_stride_bytes,
            })
            .collect();
        Self::from_arenas(&arenas)
    }

    /// One pegaflow layer per arena, single-segment (an arena is one copy
    /// unit per block by definition; K/V split segments only exist for the
    /// symmetric-pair layouts vLLM registers).
    pub(crate) fn from_arenas(arenas: &[KvArena]) -> Self {
        let n = arenas.len();
        let mut reg = Registration {
            layer_names: Vec::with_capacity(n),
            data_ptrs: Vec::with_capacity(n),
            size_bytes: Vec::with_capacity(n),
            num_blocks: Vec::with_capacity(n),
            bytes_per_block: Vec::with_capacity(n),
            kv_stride_bytes: vec![0; n],
            segments: vec![1; n],
            block_stride_bytes: Vec::with_capacity(n),
        };
        for arena in arenas {
            assert!(
                arena.bytes_per_block <= arena.block_stride_bytes,
                "arena {} copy unit {} overruns its block stride {}",
                arena.name,
                arena.bytes_per_block,
                arena.block_stride_bytes
            );
            reg.layer_names.push(arena.name.clone());
            reg.data_ptrs.push(arena.base_ptr);
            // The arena's region must cover the strided reach of its last
            // block (pegaflow validates copies against this bound).
            reg.size_bytes
                .push((arena.num_blocks - 1) * arena.block_stride_bytes + arena.bytes_per_block);
            reg.num_blocks.push(arena.num_blocks);
            reg.bytes_per_block.push(arena.bytes_per_block);
            reg.block_stride_bytes.push(arena.block_stride_bytes);
        }
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GLM5.2 shape: two arenas per model layer (MLA latent + index-K
    /// sidecar) with different copy units, sharing pool block ids. Pins the
    /// per-arena mapping and the exact strided-reach size bound pegaflow
    /// validates copies against.
    #[test]
    fn arena_registration_geometry() {
        const MLA: usize = 656 * 64;
        const IDXK: usize = 132 * 64;
        let arenas = [
            KvArena {
                name: "0.mla".into(),
                base_ptr: 0x1000,
                num_blocks: 10,
                bytes_per_block: MLA,
                block_stride_bytes: MLA,
            },
            KvArena {
                name: "0.idxk".into(),
                base_ptr: 0x9000,
                num_blocks: 10,
                bytes_per_block: IDXK,
                block_stride_bytes: IDXK,
            },
        ];
        let reg = Registration::from_arenas(&arenas);
        assert_eq!(reg.layer_names, ["0.mla", "0.idxk"]);
        assert_eq!(reg.data_ptrs, [0x1000, 0x9000]);
        assert_eq!(reg.segments, [1, 1]);
        assert_eq!(reg.kv_stride_bytes, [0, 0]);
        assert_eq!(reg.num_blocks, [10, 10]);
        assert_eq!(reg.bytes_per_block, [MLA, IDXK]);
        assert_eq!(reg.block_stride_bytes, [MLA, IDXK]);
        assert_eq!(reg.size_bytes, [10 * MLA, 10 * IDXK]);
    }

    /// A page-interleaved arena (the qwen3 fused layout expressed as arenas):
    /// stride exceeds the copy unit, and the size bound is the reach of the
    /// last block, not `num_blocks * stride`.
    #[test]
    fn interleaved_arena_size_is_last_block_reach() {
        let reg = Registration::from_arenas(&[KvArena {
            name: "3".into(),
            base_ptr: 0x100,
            num_blocks: 4,
            bytes_per_block: 512,
            block_stride_bytes: 4096,
        }]);
        assert_eq!(reg.size_bytes, [3 * 4096 + 512]);
    }

    #[test]
    #[should_panic(expected = "overruns its block stride")]
    fn arena_copy_unit_must_fit_its_stride() {
        let _ = Registration::from_arenas(&[KvArena {
            name: "bad".into(),
            base_ptr: 0,
            num_blocks: 1,
            bytes_per_block: 4096,
            block_stride_bytes: 512,
        }]);
    }
}
