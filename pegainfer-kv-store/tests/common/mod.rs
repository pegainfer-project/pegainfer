//! Shared rigs and helpers for the crate's real-engine test suites (the
//! binaries in `tests/`).
//!
//! Every suite runs against a real [`PegaflowHost`] — real `PegaEngine`,
//! real CUDA buffers, real pinned-memory D2H/H2D, and, in `ssd.rs`, a real
//! io_uring-backed cache file. Scenarios are deterministic against the real
//! engine: a GPU-only host answers queries terminally (no `Loading`), while
//! an SSD-backed host answers any miss with `Loading` before going `Ready`,
//! which is what drives the store's re-query loop for real.
//!
//! Everything requires a GPU (the arenas are CUDA allocations on device 0),
//! and the SSD cases additionally need an io_uring-capable kernel (each such
//! test probes the syscall and skips without it). [`gpu_lock`] serializes the
//! GPU rigs *within* one test binary: cargo runs a binary's tests on a
//! multi-thread harness and GPU/pinned-memory contention between rigs is
//! timing noise the deadline-based assertions are not built to measure. Test
//! binaries themselves run one after another, so no cross-binary overlap
//! arises under `cargo test`.
//!
//! Assertions stay on user-visible outcomes — `hit_tokens`, store stats,
//! pool availability, byte-exact DMA roundtrips — never on engine internals.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::Duration;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use pegainfer_kv_store::ArenaSpec;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::KvStoreBuilder;
use pegainfer_kv_store::OffloadRankSpec;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::RequestKv;
use pegainfer_kv_store::SaveClass;
use pegainfer_kv_store::SaveCursor;
use tokio::sync::Mutex;

pub(crate) const BLOCK_TOKENS: usize = 16;
pub(crate) const NUM_LAYERS: usize = 4;
pub(crate) const NUM_BLOCKS: usize = 64;
/// One arena's copy unit per block, standing in for a real layer extent:
/// 16 tokens x 2 kv-heads x 8 head-dim x 2-byte bf16. The 512 B result is
/// also pegaflow's allocator/SSD alignment unit, which keeps the tiny SSD
/// rig's arithmetic exact.
pub(crate) const SEGMENT_BYTES: usize = BLOCK_TOKENS * 2 * 8 * 2;
pub(crate) const RANK: usize = 0;

/// Plenty for every non-SSD case (the largest working set here is a handful
/// of 512 B blocks per layer).
pub(crate) const HOST_POOL_BYTES: usize = 64 << 20;

/// Serializes the GPU rigs of one test binary (one GPU, shared pinned-memory
/// claim). A tokio mutex so the guard can be held across `.await`s.
static GPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn gpu_lock() -> &'static Mutex<()> {
    GPU_LOCK.get_or_init(|| Mutex::new(()))
}

/// `full_blocks` full blocks plus one forwarded token (a full-prompt hit is
/// never reusable: the final chunk must forward to emit the first token).
pub(crate) fn prompt(full_blocks: usize) -> Vec<u32> {
    prompt_salted(full_blocks, 0)
}

/// Same shape with shifted content, for a second, prefix-disjoint working set.
pub(crate) fn prompt_salted(full_blocks: usize, salt: u32) -> Vec<u32> {
    (0..=(full_blocks * BLOCK_TOKENS) as u32)
        .map(|i| salt + i % 251)
        .collect()
}

/// A pad-aligned chain: exactly `full_blocks` blocks, no forwarded token —
/// the shape a P-side pad-and-seal hands to the decode side.
pub(crate) fn prompt_aligned(full_blocks: usize) -> Vec<u32> {
    (0..(full_blocks * BLOCK_TOKENS) as u32)
        .map(|i| i % 251)
        .collect()
}

/// Run a request through prefill so it owns `prompt`'s full blocks sealed
/// (the stand-in for a real scheduler's "seal at block boundary" after the
/// step synced — the tier's D2H ordering contract).
pub(crate) fn prefill(pool: &BlockPool, prompt: &[u32]) -> RequestKv {
    let mut kv = pool.new_request(prompt.to_vec(), 4, None);
    kv.schedule_prefill(prompt.len(), pool).expect("schedule");
    kv.apply_prefill(1, pool).expect("apply");
    kv
}

pub(crate) async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..400 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for: {what}");
}

/// One test's whole wiring: GPU arenas registered as `NUM_LAYERS` pegaflow
/// layers of one instance on `host`, plus the store over them. The fields
/// drop in declaration order, so the CUDA allocations outlive the engine
/// whose registration baked their raw addresses.
pub(crate) struct Rig {
    pub(crate) pool: Arc<BlockPool>,
    pub(crate) store: Arc<KvStore>,
    pub(crate) host: Arc<PegaflowHost>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) arenas: Vec<CudaSlice<u8>>,
    _ctx: Arc<CudaContext>,
}

impl Rig {
    pub(crate) fn new(
        test_name: &str,
        host: Arc<PegaflowHost>,
        resolve_deadline: Option<Duration>,
    ) -> Self {
        Self::new_with_layout(test_name, host, resolve_deadline, false)
    }

    /// `page_first = true` registers the rank in the vLLM-connector packing
    /// (one host page per block holding every layer at its name-sorted
    /// offset) instead of the native layer-first interleave.
    pub(crate) fn new_with_layout(
        test_name: &str,
        host: Arc<PegaflowHost>,
        resolve_deadline: Option<Duration>,
        page_first: bool,
    ) -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let arenas: Vec<CudaSlice<u8>> = (0..NUM_LAYERS)
            .map(|_| {
                stream
                    .alloc_zeros(NUM_BLOCKS * SEGMENT_BYTES)
                    .expect("arena alloc")
            })
            .collect();
        let arena_specs = arenas
            .iter()
            .enumerate()
            .map(|(layer, arena)| {
                // cudarc allocations don't move; the transient stream-ordering
                // guard is discharged here because `Rig` keeps the slice alive
                // past the host's drop.
                let (ptr, _ordering_guard) = arena.device_ptr(&stream);
                ArenaSpec {
                    name: format!("layer_{layer}"),
                    base_device_ptr: ptr,
                    size_bytes: NUM_BLOCKS * SEGMENT_BYTES,
                    num_blocks: NUM_BLOCKS,
                    segment_bytes: SEGMENT_BYTES,
                    segments: 1,
                    kv_stride_bytes: 0,
                    block_stride_bytes: SEGMENT_BYTES,
                }
            })
            .collect();
        // The pool's block ids index every arena: same geometry on both sides.
        let pool = Arc::new(BlockPool::new(BLOCK_TOKENS, NUM_BLOCKS));
        let mut builder = KvStoreBuilder::new(tokio::runtime::Handle::current())
            .with_requery_interval(Duration::from_millis(2));
        if let Some(deadline) = resolve_deadline {
            builder = builder.with_resolve_deadline(deadline);
        }
        let store = Arc::new(
            builder
                .rank_with_offload(
                    RANK,
                    Arc::clone(&pool),
                    &host,
                    OffloadRankSpec {
                        instance_id: format!("{test_name}-rank{RANK}"),
                        namespace: format!("pegainfer-kv-store-test-{test_name}"),
                        device_id: 0,
                        arenas: arena_specs,
                        page_first,
                        mirrors: Vec::new(),
                    },
                )
                .expect("rank registration")
                .build(),
        );
        Self {
            pool,
            store,
            host,
            stream,
            arenas,
            _ctx: ctx,
        }
    }

    /// Prefill + retire in one step (retire performs the final seal itself).
    pub(crate) fn run_and_retire(&self, prompt: &[u32], class: SaveClass) {
        let kv = prefill(&self.pool, prompt);
        self.store.retire(RANK, kv, SaveCursor::new(), class);
    }

    /// Stage distinct content into the named blocks of every layer's arena.
    pub(crate) fn stage_block_patterns(&mut self, blocks: &[i32]) {
        for (layer, arena) in self.arenas.iter_mut().enumerate() {
            let mut buf = vec![0u8; NUM_BLOCKS * SEGMENT_BYTES];
            for &b in blocks {
                let begin = b as usize * SEGMENT_BYTES;
                buf[begin..begin + SEGMENT_BYTES]
                    .copy_from_slice(&block_pattern(layer, b as usize));
            }
            self.stream.memcpy_htod(&buf, arena).expect("stage htod");
        }
    }

    pub(crate) fn zero_arenas(&mut self) {
        let zeros = vec![0u8; NUM_BLOCKS * SEGMENT_BYTES];
        for arena in &mut self.arenas {
            self.stream.memcpy_htod(&zeros, arena).expect("zero htod");
        }
        // The zeroing must be complete before any tier H2D can race it — the
        // engine's copies run on its own streams.
        self.stream.synchronize().expect("zero sync");
    }

    pub(crate) fn arena_bytes(&self, layer: usize) -> Vec<u8> {
        self.stream.synchronize().expect("sync");
        self.stream
            .clone_dtoh(&self.arenas[layer])
            .expect("arena dtoh")
    }
}

/// Deterministic per-(layer, block) content: an FK-alignment or layer swap
/// in the DMA path turns into a byte mismatch, not a passing "hit".
pub(crate) fn block_pattern(layer: usize, block: usize) -> Vec<u8> {
    (0..SEGMENT_BYTES)
        .map(|i| ((layer * 31 + block * 7 + i / 64) % 251) as u8)
        .collect()
}

pub(crate) fn loaded_blocks(rig: &Rig) -> u64 {
    rig.store
        .stats()
        .resolve_loaded_blocks
        .load(Ordering::Relaxed)
}

pub(crate) fn degraded(rig: &Rig) -> u64 {
    rig.store.stats().resolve_degraded.load(Ordering::Relaxed)
}

/// Probe whether io_uring is usable in this environment; some containers
/// restrict the syscall via seccomp.
pub(crate) fn io_uring_available() -> bool {
    unsafe {
        let mut params = std::mem::MaybeUninit::<[u8; 128]>::zeroed();
        let fd = libc::syscall(
            libc::SYS_io_uring_setup,
            1i32,
            params.as_mut_ptr().cast::<libc::c_void>(),
        );
        if fd >= 0 {
            libc::close(fd as i32);
            true
        } else {
            false
        }
    }
}
