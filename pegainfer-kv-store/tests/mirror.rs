//! Mirror fan-out over the real host tier: a rank registered with a
//! tensor-replicated mirror device. The primary saves once; a resolve must
//! land the restored bytes on EVERY device's arenas — a load that reaches
//! only one device leaves the mirrors attending over stale pages, which the
//! downstream all-reduce merges identically on every rank (openinfer#847).

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::BLOCK_TOKENS;
use common::HOST_POOL_BYTES;
use common::NUM_BLOCKS;
use common::NUM_LAYERS;
use common::RANK;
use common::SEGMENT_BYTES;
use common::block_pattern;
use common::gpu_lock;
use common::prefill;
use common::prompt;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use pegainfer_kv_store::ArenaSpec;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::KvStoreBuilder;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::OffloadMirror;
use pegainfer_kv_store::OffloadRankSpec;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveClass;
use pegainfer_kv_store::SaveCursor;

/// One device's half of the mirrored rank: the same layer names and geometry
/// as its peer, backed by this device's own allocations.
struct DeviceArenas {
    stream: Arc<CudaStream>,
    arenas: Vec<CudaSlice<u8>>,
    _ctx: Arc<CudaContext>,
}

impl DeviceArenas {
    fn new(ctx: Arc<CudaContext>) -> Self {
        let stream = ctx.default_stream();
        let arenas = (0..NUM_LAYERS)
            .map(|_| {
                stream
                    .alloc_zeros(NUM_BLOCKS * SEGMENT_BYTES)
                    .expect("arena alloc")
            })
            .collect();
        Self {
            stream,
            arenas,
            _ctx: ctx,
        }
    }

    fn specs(&self) -> Vec<ArenaSpec> {
        self.arenas
            .iter()
            .enumerate()
            .map(|(layer, arena)| {
                let (ptr, _ordering_guard) = arena.device_ptr(&self.stream);
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
            .collect()
    }

    fn stage_block_patterns(&mut self, blocks: &[i32]) {
        for (layer, arena) in self.arenas.iter_mut().enumerate() {
            let mut buf = vec![0u8; NUM_BLOCKS * SEGMENT_BYTES];
            for &b in blocks {
                let begin = b as usize * SEGMENT_BYTES;
                buf[begin..begin + SEGMENT_BYTES]
                    .copy_from_slice(&block_pattern(layer, b as usize));
            }
            self.stream.memcpy_htod(&buf, arena).expect("stage htod");
        }
        self.stream.synchronize().expect("stage sync");
    }

    fn zero_arenas(&mut self) {
        let zeros = vec![0u8; NUM_BLOCKS * SEGMENT_BYTES];
        for arena in &mut self.arenas {
            self.stream.memcpy_htod(&zeros, arena).expect("zero htod");
        }
        self.stream.synchronize().expect("zero sync");
    }

    fn arena_bytes(&self, layer: usize) -> Vec<u8> {
        self.stream.synchronize().expect("sync");
        self.stream
            .clone_dtoh(&self.arenas[layer])
            .expect("arena dtoh")
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mirror_load_lands_on_every_device() {
    let _gpu = gpu_lock().lock().await;
    let Ok(mirror_ctx) = CudaContext::new(1) else {
        eprintln!("skipping mirror_load_lands_on_every_device: needs a second CUDA device");
        return;
    };
    let primary_ctx = CudaContext::new(0).expect("CUDA device 0");
    let mut primary = DeviceArenas::new(primary_ctx);
    let mut mirror = DeviceArenas::new(mirror_ctx);

    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let pool = Arc::new(BlockPool::new(BLOCK_TOKENS, NUM_BLOCKS));
    let store = Arc::new(
        KvStoreBuilder::new(tokio::runtime::Handle::current())
            .with_requery_interval(Duration::from_millis(2))
            .rank_with_offload(
                RANK,
                Arc::clone(&pool),
                &host,
                OffloadRankSpec {
                    instance_id: format!("mirror-rank{RANK}"),
                    namespace: "pegainfer-kv-store-test-mirror".to_owned(),
                    device_id: 0,
                    arenas: primary.specs(),
                    page_first: false,
                    mirrors: vec![OffloadMirror {
                        device_id: 1,
                        arenas: mirror.specs(),
                    }],
                },
            )
            .expect("rank registration")
            .build(),
    );

    // Producer side: only the PRIMARY holds the staged KV (in the real TP
    // topology every worker computes the same replicated KV, but the save
    // path reads the primary alone — proving here that the mirror's restored
    // bytes can only have come through the tier's fan-out).
    let prompt = prompt(4);
    let kv = prefill(&pool, &prompt);
    let saved_ids: Vec<i32> = kv
        .assigned_block_hashes()
        .iter()
        .map(|&(id, _)| id)
        .collect();
    assert_eq!(
        saved_ids.len(),
        4,
        "65-token prompt seals its 4 full blocks"
    );
    primary.stage_block_patterns(&saved_ids);
    store.retire(RANK, kv, SaveCursor::new(), SaveClass::Cacheable);
    store.flush_saves(RANK).await.expect("flush");

    primary.zero_arenas();
    mirror.zero_arenas();
    pool.evict_inactive();

    let prefix = store
        .resolve_prefix(
            RANK,
            "r1",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);

    // The i-th hit block's bytes must equal what the producer staged in the
    // i-th saved block — on BOTH devices, per layer.
    let mut req = pool.new_request(prompt.clone(), 4, None);
    assert_eq!(
        req.match_and_add_prefix(&pool).expect("match"),
        4 * BLOCK_TOKENS,
        "the loaded blocks were committed under the continuation hashes"
    );
    let dst_ids = req.current_page_indices();
    for (device_name, device) in [("primary", &primary), ("mirror", &mirror)] {
        for layer in 0..NUM_LAYERS {
            let bytes = device.arena_bytes(layer);
            for (&src, &dst) in saved_ids.iter().zip(dst_ids.iter()) {
                let begin = dst as usize * SEGMENT_BYTES;
                assert_eq!(
                    &bytes[begin..begin + SEGMENT_BYTES],
                    block_pattern(layer, src as usize).as_slice(),
                    "{device_name} device layer {layer}: block {src} -> {dst} \
                     did not restore byte-exact"
                );
            }
        }
    }
}
