//! GPU→CPU→GPU round-trip over a real page-first [`KvBuffer`].
//!
//! Writes a distinct bit pattern into a set of source GPU blocks, offloads them
//! to pegaflow's host tier, evicts the GPU-side data implicitly by loading into
//! a *different* set of blocks, and checks the bytes match. This exercises the
//! whole connector — strided per-layer registration (`block_stride` ≠ copy
//! size), the K/V split, the async save, the prefix query, and the in-process
//! oneshot load — on actual device memory. If the layout math were wrong the
//! loaded bytes would land in the wrong layer/segment/block and the compare
//! would fail.
//!
//! Requires a CUDA GPU; skipped from `--lib` unit runs.

use cudarc::driver::CudaContext;
use cudarc::driver::result;
use half::bf16;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_offload::OffloadConfig;
use pegainfer_kv_offload::OffloadEngine;
use pegainfer_kv_offload::QueryOutcome;

const NUM_LAYERS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const PAGE_SIZE: usize = 16;
const NUM_BLOCKS: usize = 32;

/// Elements in one K (or V) segment of one block.
const SEGMENT_LEN: usize = PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM;
const LAYER_STRIDE: usize = 2 * SEGMENT_LEN;
const PAGE_STRIDE: usize = NUM_LAYERS * LAYER_STRIDE;

/// Deterministic, finite, varied pattern for one (logical block, layer, segment).
/// `logical` is the block's position in the saved hash list — load must restore
/// the i-th leased block onto the i-th destination, so the destination's bytes
/// must equal `pattern(i, ..)` regardless of which physical block held it.
fn pattern(logical: usize, layer: usize, segment: usize) -> Vec<bf16> {
    (0..SEGMENT_LEN)
        .map(|e| {
            let seed = (logical * 9973 + layer * 257 + segment * 131 + e * 7) % 4093;
            bf16::from_f32(seed as f32 / 11.0 - 90.0)
        })
        .collect()
}

/// Byte address of (block, layer, segment)'s first element within the fused buffer.
fn segment_ptr(base: u64, block_id: usize, layer: usize, segment: usize) -> u64 {
    let elem_off = block_id * PAGE_STRIDE + layer * LAYER_STRIDE + segment * SEGMENT_LEN;
    base + (elem_off * std::mem::size_of::<bf16>()) as u64
}

fn block_hash(logical: usize) -> Vec<u8> {
    let mut h = vec![0xA5u8; 16];
    h[0] = logical as u8;
    h[1] = (logical as u8).wrapping_mul(31).wrapping_add(7);
    h
}

#[test]
fn gpu_cpu_gpu_roundtrip_preserves_kv_bytes() {
    let ctx = CudaContext::new(0).expect("cuda device 0");
    ctx.bind_to_thread().expect("bind ctx to test thread");
    let stream = ctx.default_stream();

    let buffer = KvBuffer::new(
        &stream,
        NUM_LAYERS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        NUM_BLOCKS,
    )
    .expect("alloc KvBuffer");
    // Sanity: our test-local geometry constants match the buffer's layout.
    assert_eq!(buffer.layout().page_stride, PAGE_STRIDE);
    assert_eq!(buffer.layout().kv_block_len, SEGMENT_LEN);

    let base = buffer.device_ptr(&stream);

    let src_blocks = [1usize, 2, 3];
    let dst_blocks = [10usize, 11, 12];
    let untouched_block = 20usize;

    // ── Fill source blocks with the per-(logical, layer, segment) pattern ──
    for (logical, &block_id) in src_blocks.iter().enumerate() {
        for layer in 0..NUM_LAYERS {
            for segment in 0..2 {
                let data = pattern(logical, layer, segment);
                let dst = segment_ptr(base, block_id, layer, segment);
                // SAFETY: dst lies inside the buffer (block < NUM_BLOCKS) and the
                // slice is exactly one segment of bf16, the buffer's element type.
                unsafe { result::memcpy_htod_sync(dst, &data) }.expect("htod fill");
            }
        }
    }
    stream.synchronize().expect("sync after fill");

    // ── Build the offload engine (registers the fused buffer) ──
    let engine = OffloadEngine::new(
        OffloadConfig::new("roundtrip-test", 0, 64 * 1024 * 1024),
        &buffer,
        &stream,
    )
    .expect("build OffloadEngine");

    let hashes: Vec<Vec<u8>> = (0..src_blocks.len()).map(block_hash).collect();
    let src_ids: Vec<i32> = src_blocks.iter().map(|&b| b as i32).collect();

    // ── Save GPU→CPU (blocking capture) and make the writes cache-visible ──
    engine.save_blocking(&src_ids, &hashes).expect("save");
    engine.flush_saves();

    // ── Query the CPU tier: the full 3-block prefix must be resident ──
    // Host-memory-only setup: `Loading` can't occur, the query is terminal.
    let QueryOutcome::Ready(hit) = engine
        .query("roundtrip-req", &hashes, false)
        .expect("query")
    else {
        panic!("host-memory-only query must be terminal");
    };
    assert_eq!(
        hit.num_blocks, 3,
        "all three saved blocks should hit the CPU tier"
    );
    let lease = hit.lease.expect("a hit returns a lease");

    // ── Load CPU→GPU into a *different* set of blocks ──
    let dst_ids: Vec<i32> = dst_blocks.iter().map(|&b| b as i32).collect();
    engine
        .load(lease, dst_ids)
        .expect("submit load")
        .wait()
        .expect("load completes");
    stream.synchronize().expect("sync after load");

    // ── Verify each destination block holds the matching logical pattern ──
    for (logical, &block_id) in dst_blocks.iter().enumerate() {
        for layer in 0..NUM_LAYERS {
            for segment in 0..2 {
                let expected = pattern(logical, layer, segment);
                let mut got = vec![bf16::ZERO; SEGMENT_LEN];
                let src = segment_ptr(base, block_id, layer, segment);
                // SAFETY: src is one in-bounds segment of bf16.
                unsafe { result::memcpy_dtoh_sync(&mut got, src) }.expect("dtoh verify");
                let expected_bits: Vec<u16> = expected.iter().map(|v| v.to_bits()).collect();
                let got_bits: Vec<u16> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    got_bits, expected_bits,
                    "dst block {block_id} layer {layer} segment {segment} \
                     must restore logical block {logical}'s bytes"
                );
            }
        }
    }

    // ── Negative control: a block we never loaded stays zero ──
    let mut zero = vec![bf16::from_f32(1.0); SEGMENT_LEN];
    let src = segment_ptr(base, untouched_block, 0, 0);
    unsafe { result::memcpy_dtoh_sync(&mut zero, src) }.expect("dtoh untouched");
    assert!(
        zero.iter().all(|v| v.to_bits() == 0),
        "an unloaded block must remain zeroed — load must not scribble outside its destinations"
    );
}
