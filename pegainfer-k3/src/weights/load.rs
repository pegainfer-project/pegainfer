//! Rank weight load executor: mmap the shards, validate every tensor against
//! the contract, place routed experts into their packed regions, upload raw
//! bytes. Ported from `pegainfer-glm52/src/weights/load.rs`; the mmap-lifetime
//! guards below are that file's design unchanged.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use log::debug;
use log::error;
use log::info;
use memmap2::Mmap;

use super::K3ExpertRegionKind;
use super::K3RankGpuContext;
use super::K3RankLoadBundle;
use super::expected_tensor_contract;
use super::expert_placement;
use super::mmap_file;
use super::staging::K3WeightStager;
use super::text_tensor_name;

// One mmap per event keeps source lifetimes explicit without the large-tail
// regression observed when many shard mappings stay live together.
const MMAP_EVENT_GROUP: usize = 1;
const LIVE_MMAP_GROUPS: usize = 16;

enum WeightUploader {
    Pageable,
    Staged(K3WeightStager),
}

impl WeightUploader {
    fn new(ctx: &K3RankGpuContext, rank: usize, weight_staging: bool) -> Result<Self> {
        if weight_staging {
            Ok(Self::Staged(K3WeightStager::new(ctx, rank)?))
        } else {
            Ok(Self::Pageable)
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Pageable => "pageable",
            Self::Staged(_) => "pinned",
        }
    }

    fn upload(
        &mut self,
        ctx: &K3RankGpuContext,
        src: &[u8],
        dst: &mut CudaSlice<u8>,
        dst_offset: usize,
    ) -> Result<usize> {
        match self {
            Self::Pageable => {
                let end = dst_offset.checked_add(src.len()).ok_or_else(|| {
                    anyhow::anyhow!("K3 pageable upload destination range overflow")
                })?;
                ensure!(
                    end <= dst.len(),
                    "K3 pageable upload [{dst_offset}..{end}) exceeds destination of {} bytes",
                    dst.len()
                );
                let mut view = dst.slice_mut(dst_offset..end);
                ctx.stream()
                    .memcpy_htod(src, &mut view)
                    .context("K3 pageable H2D copy")?;
                Ok(1)
            }
            Self::Staged(stager) => stager.upload(src, dst, dst_offset),
        }
    }
}

/// One layer's rank-local routed experts, packed expert-major at H2D time
/// (per expert: [gate; up] MXFP4 payload rows, then the matching e8m0 scale
/// rows; w2 alone). Bytes are the checkpoint's, unmodified — scale decode and
/// any relayout happen at model build.
pub(crate) struct K3ExpertLayerRegions {
    pub(crate) w13_weight: CudaSlice<u8>,
    pub(crate) w13_scale: CudaSlice<u8>,
    pub(crate) w2_weight: CudaSlice<u8>,
    pub(crate) w2_scale: CudaSlice<u8>,
}

impl K3ExpertLayerRegions {
    fn alloc(ctx: &K3RankGpuContext, local_experts: usize) -> Result<Self> {
        // SAFETY: every byte of every region is written exactly once from a
        // validated safetensors view; the coverage counters below fail the
        // load if any byte is left unwritten.
        let alloc = |kind: K3ExpertRegionKind| -> Result<CudaSlice<u8>> {
            unsafe { ctx.stream().alloc::<u8>(kind.region_bytes(local_experts)) }
                .with_context(|| format!("alloc K3 expert region {kind:?}"))
        };
        Ok(Self {
            w13_weight: alloc(K3ExpertRegionKind::W13Weight)?,
            w13_scale: alloc(K3ExpertRegionKind::W13Scale)?,
            w2_weight: alloc(K3ExpertRegionKind::W2Weight)?,
            w2_scale: alloc(K3ExpertRegionKind::W2Scale)?,
        })
    }

    fn region_mut(&mut self, kind: K3ExpertRegionKind) -> &mut CudaSlice<u8> {
        match kind {
            K3ExpertRegionKind::W13Weight => &mut self.w13_weight,
            K3ExpertRegionKind::W13Scale => &mut self.w13_scale,
            K3ExpertRegionKind::W2Weight => &mut self.w2_weight,
            K3ExpertRegionKind::W2Scale => &mut self.w2_scale,
        }
    }
}

/// Rank-resident weights: every non-expert tensor as its own device buffer
/// (raw checkpoint bytes), plus the per-layer packed expert regions. Keys are
/// TEXT names — the `language_model.` wrapper prefix stops at the loader.
/// `from_device` constructors take entries out of these maps.
pub(crate) struct K3RankGpuWeights {
    tensors: BTreeMap<String, CudaSlice<u8>>,
    expert_layers: BTreeMap<usize, K3ExpertLayerRegions>,
    pub(crate) total_bytes: usize,
}

impl K3RankGpuWeights {
    /// Remove and return one non-expert tensor buffer (raw checkpoint bytes,
    /// keyed by text name, e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub(crate) fn take_tensor(&mut self, name: &str) -> Result<CudaSlice<u8>> {
        self.tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("K3 resident weights missing tensor {name}"))
    }

    /// Remove and return one layer's packed expert regions.
    pub(crate) fn take_expert_layer(&mut self, layer: usize) -> Result<K3ExpertLayerRegions> {
        self.expert_layers.remove(&layer).ok_or_else(|| {
            anyhow::anyhow!("K3 resident weights missing expert regions for layer {layer}")
        })
    }

    /// Every resident load-plan entry must move into the built model. Keeping
    /// validation-only tensors in these maps silently spends H2D bandwidth and
    /// transient HBM before dropping them with the load bundle.
    pub(crate) fn ensure_consumed(&self) -> Result<()> {
        let tensor_sample = self.tensors.keys().take(5).cloned().collect::<Vec<_>>();
        let expert_sample = self
            .expert_layers
            .keys()
            .take(5)
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            self.tensors.is_empty() && self.expert_layers.is_empty(),
            "K3 resident load plan contains tensors the model did not consume: \
             tensors={} sample={tensor_sample:?}, expert_layers={} sample={expert_sample:?}",
            self.tensors.len(),
            self.expert_layers.len(),
        );
        Ok(())
    }
}

pub(crate) struct K3RankLoadOutput {
    pub(crate) weights: K3RankGpuWeights,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
}

pub(crate) fn load_rank_weights_to_gpu(
    ctx: &K3RankGpuContext,
    model_path: &Path,
    bundle: &K3RankLoadBundle,
    weight_staging: bool,
) -> Result<K3RankLoadOutput> {
    ctx.set_current()?;
    let load_started = Instant::now();
    let planned_total_bytes = bundle.planned_total_bytes()?;
    let routed_experts = bundle.plan.routed_experts;

    let mut weights = K3RankGpuWeights {
        tensors: BTreeMap::new(),
        expert_layers: BTreeMap::new(),
        total_bytes: 0,
    };
    // Bytes written per (layer, region): the coverage proof that packed
    // placement filled every region exactly.
    let mut region_written: BTreeMap<(usize, K3ExpertRegionKind), usize> = BTreeMap::new();
    let mut loaded_tensor_count = 0usize;
    let mut loaded_total_bytes = 0usize;
    let copy_started = Instant::now();
    let mut slowest_shard: Option<(String, f64)> = None;
    let mut h2d_copies = 0usize;
    let mut mmap_guard = MmapH2dLifetimeGuard::new(ctx, bundle.plan.rank);
    let mut sync_secs = 0.0f64;
    let mut uploader = WeightUploader::new(ctx, bundle.plan.rank, weight_staging)?;
    debug!(
        "K3 rank {} start weight load: tensors={}, shards={}, bytes={:.2} GiB, experts={:?}, uploader={}",
        bundle.plan.rank,
        bundle.plan.tensor_count,
        bundle.shards.len(),
        gib(planned_total_bytes),
        bundle.plan.expert_range,
        uploader.label(),
    );

    for shard in &bundle.shards {
        let path = model_path.join(&shard.shard);
        let shard_started = Instant::now();
        let mmap = CurrentShardMmap::new(mmap_file(&path)?, ctx, bundle.plan.rank);
        let safetensors = safetensors::SafeTensors::deserialize(mmap.as_ref())
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for spec in &shard.tensors {
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
            let text_name = text_tensor_name(&spec.name)?;
            let contract = expected_tensor_contract(text_name, routed_experts)?;
            ensure!(
                view.dtype() == contract.dtype,
                "K3 tensor {} dtype mismatch: got {:?}, expected {:?}",
                spec.name,
                view.dtype(),
                contract.dtype
            );
            ensure!(
                view.shape() == contract.shape.as_slice(),
                "K3 tensor {} shape mismatch: got {:?}, expected {:?}",
                spec.name,
                view.shape(),
                contract.shape
            );
            ensure!(
                view.data().len() == contract.byte_len()?,
                "K3 tensor {} byte mismatch: got {}, expected {}",
                spec.name,
                view.data().len(),
                contract.byte_len()?
            );
            // Padded checkpoint tensors (`A_log`) contribute only their leading
            // real values; every other tensor is uploaded whole.
            let bytes = contract.resident_byte_len()?;
            let src = &view.data()[..bytes];

            if let Some(placement) = expert_placement(text_name, &bundle.plan.expert_range)? {
                let regions = match weights.expert_layers.entry(placement.layer) {
                    std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::btree_map::Entry::Vacant(entry) => entry.insert(
                        K3ExpertLayerRegions::alloc(ctx, bundle.plan.expert_range.len())?,
                    ),
                };
                let region = regions.region_mut(placement.region);
                ensure!(
                    placement.offset + bytes <= region.len(),
                    "K3 tensor {} placement [{}..{}) exceeds region {:?} ({} bytes)",
                    spec.name,
                    placement.offset,
                    placement.offset + bytes,
                    placement.region,
                    region.len()
                );
                mmap.mark_stream_work();
                h2d_copies += uploader
                    .upload(ctx, src, region, placement.offset)
                    .with_context(|| {
                        format!("failed to copy K3 expert tensor {} to GPU", spec.name)
                    })?;
                *region_written
                    .entry((placement.layer, placement.region))
                    .or_default() += bytes;
            } else {
                // SAFETY: the whole buffer is written by the single H2D copy
                // below before it becomes reachable resident state.
                let mut dst = unsafe { ctx.stream().alloc::<u8>(bytes) }
                    .with_context(|| format!("alloc K3 tensor {}", spec.name))?;
                mmap.mark_stream_work();
                h2d_copies += uploader
                    .upload(ctx, src, &mut dst, 0)
                    .with_context(|| format!("failed to copy K3 tensor {} to GPU", spec.name))?;
                ensure!(
                    weights.tensors.insert(text_name.to_owned(), dst).is_none(),
                    "duplicate K3 tensor {} in rank {} load plan",
                    spec.name,
                    bundle.plan.rank,
                );
            }
            weights.total_bytes += bytes;
            loaded_total_bytes += bytes;
            loaded_tensor_count += 1;
        }
        drop(safetensors);
        sync_secs += mmap_guard.push_completed_shard(mmap.into_mmap())?;
        let shard_secs = shard_started.elapsed().as_secs_f64();
        match &slowest_shard {
            Some((_, slowest_secs)) if *slowest_secs >= shard_secs => {}
            _ => slowest_shard = Some((shard.shard.clone(), shard_secs)),
        }
    }

    ensure!(
        loaded_tensor_count == bundle.plan.tensor_count,
        "K3 rank {} loaded {loaded_tensor_count} tensors but load plan has {}",
        bundle.plan.rank,
        bundle.plan.tensor_count
    );
    ensure!(
        loaded_total_bytes == planned_total_bytes,
        "K3 rank {} loaded {} bytes but load plan has {} bytes",
        bundle.plan.rank,
        loaded_total_bytes,
        planned_total_bytes
    );
    // Every packed expert region must be fully written — a hole here means
    // stale device bytes would silently become model weights.
    for layer in weights.expert_layers.keys() {
        for kind in K3ExpertRegionKind::ALL {
            let written = region_written.get(&(*layer, kind)).copied().unwrap_or(0);
            let expected = kind.region_bytes(bundle.plan.expert_range.len());
            ensure!(
                written == expected,
                "K3 rank {} layer {layer} expert region {kind:?} incomplete: wrote {written} of {expected} bytes",
                bundle.plan.rank
            );
        }
    }
    sync_secs += mmap_guard.drain()?;
    let uploader_label = uploader.label();
    let uploader_drop_started = Instant::now();
    drop(uploader);
    sync_secs += uploader_drop_started.elapsed().as_secs_f64();
    let copy_secs = (copy_started.elapsed().as_secs_f64() - sync_secs).max(0.0);
    let sync_started = Instant::now();
    ctx.sync().with_context(|| {
        format!(
            "failed to finish K3 rank {} H2D tensor copies",
            bundle.plan.rank
        )
    })?;
    sync_secs += sync_started.elapsed().as_secs_f64();

    let (slowest_shard, slowest_secs) = slowest_shard.unwrap_or_else(|| ("none".to_owned(), 0.0));
    info!(
        "K3 rank {} weight load profile: total={:.2}s, mmap_deser_copy={:.2}s, sync={:.2}s, uploader={}, tensors={}, h2d_copies={}, bytes={:.2} GiB, expert_layers={}, slowest_shard={} {:.2}s",
        bundle.plan.rank,
        load_started.elapsed().as_secs_f64(),
        copy_secs,
        sync_secs,
        uploader_label,
        loaded_tensor_count,
        h2d_copies,
        gib(loaded_total_bytes),
        weights.expert_layers.len(),
        slowest_shard,
        slowest_secs
    );

    Ok(K3RankLoadOutput {
        weights,
        loaded_tensor_count,
        loaded_total_bytes,
    })
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

struct LiveMmapGroup {
    _mmaps: Vec<Mmap>,
    event: CudaEvent,
}

struct CurrentShardMmap<'a> {
    mmap: Option<Mmap>,
    ctx: &'a K3RankGpuContext,
    rank: usize,
    stream_work: Cell<bool>,
}

impl<'a> CurrentShardMmap<'a> {
    fn new(mmap: Mmap, ctx: &'a K3RankGpuContext, rank: usize) -> Self {
        Self {
            mmap: Some(mmap),
            ctx,
            rank,
            stream_work: Cell::new(false),
        }
    }

    fn as_ref(&self) -> &Mmap {
        self.mmap.as_ref().expect("current shard mmap is present")
    }

    fn mark_stream_work(&self) {
        self.stream_work.set(true);
    }

    fn into_mmap(mut self) -> Mmap {
        self.stream_work.set(false);
        self.mmap.take().expect("current shard mmap is present")
    }
}

impl Drop for CurrentShardMmap<'_> {
    fn drop(&mut self) {
        if self.stream_work.get() {
            if let Err(error) = self.ctx.set_current().and_then(|()| self.ctx.sync()) {
                error!(
                    "failed to synchronize K3 rank {} before dropping current H2D mmap: {error:#}",
                    self.rank
                );
            }
        }
    }
}

struct MmapH2dLifetimeGuard<'a> {
    ctx: &'a K3RankGpuContext,
    rank: usize,
    pending_mmaps: Vec<Mmap>,
    live_mmap_groups: VecDeque<LiveMmapGroup>,
}

impl<'a> MmapH2dLifetimeGuard<'a> {
    fn new(ctx: &'a K3RankGpuContext, rank: usize) -> Self {
        Self {
            ctx,
            rank,
            pending_mmaps: Vec::with_capacity(MMAP_EVENT_GROUP),
            live_mmap_groups: VecDeque::with_capacity(LIVE_MMAP_GROUPS),
        }
    }

    fn push_completed_shard(&mut self, mmap: Mmap) -> Result<f64> {
        self.pending_mmaps.push(mmap);
        let mut sync_secs = 0.0;
        if self.pending_mmaps.len() == MMAP_EVENT_GROUP {
            self.record_pending_group()?;
        }
        if self.live_mmap_groups.len() >= LIVE_MMAP_GROUPS {
            sync_secs += self.wait_oldest_group()?;
        }
        Ok(sync_secs)
    }

    fn drain(&mut self) -> Result<f64> {
        let mut sync_secs = 0.0;
        if !self.pending_mmaps.is_empty() {
            self.record_pending_group()?;
        }
        while !self.live_mmap_groups.is_empty() {
            sync_secs += self.wait_oldest_group()?;
        }
        Ok(sync_secs)
    }

    fn record_pending_group(&mut self) -> Result<()> {
        let event = self.ctx.stream().record_event(None).with_context(|| {
            format!(
                "failed to record K3 rank {} H2D mmap group event",
                self.rank
            )
        })?;
        let mut mmaps = Vec::with_capacity(MMAP_EVENT_GROUP);
        std::mem::swap(&mut self.pending_mmaps, &mut mmaps);
        self.live_mmap_groups.push_back(LiveMmapGroup {
            _mmaps: mmaps,
            event,
        });
        Ok(())
    }

    fn wait_oldest_group(&mut self) -> Result<f64> {
        let Some(live) = self.live_mmap_groups.front() else {
            return Ok(0.0);
        };
        let started = Instant::now();
        live.event
            .synchronize()
            .with_context(|| format!("failed to wait for K3 rank {} H2D mmap event", self.rank))?;
        self.live_mmap_groups.pop_front();
        Ok(started.elapsed().as_secs_f64())
    }
}

impl Drop for MmapH2dLifetimeGuard<'_> {
    fn drop(&mut self) {
        if self.pending_mmaps.is_empty() && self.live_mmap_groups.is_empty() {
            return;
        }
        if let Err(error) = self.ctx.set_current().and_then(|()| self.ctx.sync()) {
            error!(
                "failed to synchronize K3 rank {} before dropping H2D mmap guard: {error:#}",
                self.rank
            );
        }
    }
}
