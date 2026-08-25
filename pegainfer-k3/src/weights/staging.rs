//! Pinned double-buffer H2D uploader.
//!
//! Ported from `pegainfer-glm52/src/weights/staging.rs` with K3 naming; the
//! buffer size / worker-count tuning came from the glm52 GB300 EP4 parameter
//! sweep and carries over unchanged because K3 ships the same shape of load
//! (tens of thousands of mid-size raw-byte uploads per rank).

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::result::memcpy_htod_async;
use cudarc::driver::sys::CUevent_flags;
use log::error;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use super::K3RankGpuContext;

/// Bytes per pinned slot. Two slots cost 64 MiB/rank and allow one host fill to
/// overlap one H2D DMA.
const STAGE_BYTES: usize = 32 << 20;
/// Persistent host memcpy workers per rank; retaining them avoids repeated
/// thread creation across the tens of thousands of uploads a rank performs.
pub(crate) const FILL_THREADS: usize = 4;
/// MXFP4 expert scale tensors are only a few hundred KiB; channel fan-out costs
/// more than their memcpy. Large projections still use all persistent workers.
const PARALLEL_FILL_MIN_BYTES: usize = 1 << 20;

struct StagingBuf {
    pinned: PinnedHostSlice<u8>,
    dma_done: CudaEvent,
}

struct FillPool {
    pool: ThreadPool,
    workers: usize,
}

impl FillPool {
    fn new(rank: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(FILL_THREADS)
            .thread_name(move |worker| format!("k3-weight-fill-{rank}-{worker}"))
            .build()
            .with_context(|| format!("build K3 rank {rank} weight-fill pool"))?;
        Ok(Self {
            pool,
            workers: FILL_THREADS,
        })
    }

    /// Copy `src` into a same-sized destination using persistent workers.
    fn fill(&self, src: &[u8], dst: &mut [u8]) {
        debug_assert_eq!(src.len(), dst.len());
        if src.is_empty() {
            return;
        }
        if src.len() < PARALLEL_FILL_MIN_BYTES {
            dst.copy_from_slice(src);
            return;
        }
        let per = src.len().div_ceil(self.workers);
        self.pool.scope(|scope| {
            for (src_part, dst_part) in src.chunks(per).zip(dst.chunks_mut(per)) {
                scope.spawn(move |_| {
                    dst_part.copy_from_slice(src_part);
                });
            }
        });
    }
}

#[cfg(test)]
impl FillPool {
    fn worker_count(&self) -> usize {
        self.workers
    }
}

/// Pinned double-buffering overlaps mmap/page-cache reads with the preceding
/// buffer's H2D DMA. All K3 checkpoint tensors are carried as raw bytes, so one
/// uploader serves MXFP4 payload, e8m0 scale, BF16 and F32 payloads without
/// reinterpretation.
pub(crate) struct K3WeightStager {
    stream: Arc<CudaStream>,
    bufs: [StagingBuf; 2],
    fill: FillPool,
    next: usize,
}

impl K3WeightStager {
    pub(crate) fn new(ctx: &K3RankGpuContext, rank: usize) -> Result<Self> {
        let make = || -> Result<StagingBuf> {
            // SAFETY: every byte consumed by DMA is initialized by the fill
            // pool first; buffer reuse is gated by `dma_done`.
            let pinned = unsafe { ctx.cuda_context().alloc_pinned::<u8>(STAGE_BYTES) }
                .context("alloc K3 pinned weight staging buffer")?;
            let dma_done = ctx
                .cuda_context()
                .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .context("create K3 pinned weight staging event")?;
            Ok(StagingBuf { pinned, dma_done })
        };
        Ok(Self {
            stream: ctx.stream().clone(),
            bufs: [make()?, make()?],
            fill: FillPool::new(rank)?,
            next: 0,
        })
    }

    /// Upload `src` into raw device bytes starting at `dst_offset`. Returns the
    /// number of H2D copies issued.
    pub(crate) fn upload(
        &mut self,
        src: &[u8],
        dst: &mut CudaSlice<u8>,
        dst_offset: usize,
    ) -> Result<usize> {
        ensure!(
            Arc::ptr_eq(dst.stream(), &self.stream),
            "K3 staged upload destination was allocated on a different stream"
        );
        ensure!(
            dst_offset
                .checked_add(src.len())
                .is_some_and(|end| end <= dst.len()),
            "K3 staged upload [{}..{}) exceeds destination of {} bytes",
            dst_offset,
            dst_offset.saturating_add(src.len()),
            dst.len()
        );
        let stream = self.stream.clone();
        let (dst_ptr, _dst_order) = dst.device_ptr_mut(&stream);
        let mut copies = 0usize;
        for (chunk_idx, chunk) in src.chunks(STAGE_BYTES).enumerate() {
            let dst_at = dst_ptr + (dst_offset + chunk_idx * STAGE_BYTES) as u64;
            // SAFETY: chunks partition the source, and the entry bound proves
            // the corresponding destination range is writable.
            unsafe { self.stage_chunk(chunk, dst_at) }?;
            copies += 1;
        }
        Ok(copies)
    }

    /// # Safety
    /// `dst_at` must address `src.len() <= STAGE_BYTES` writable device bytes
    /// on the stager stream.
    unsafe fn stage_chunk(&mut self, src: &[u8], dst_at: u64) -> Result<()> {
        let idx = self.next;
        self.next = (self.next + 1) % self.bufs.len();
        let buf = &mut self.bufs[idx];
        buf.dma_done
            .synchronize()
            .context("drain K3 pinned staging buffer")?;
        let stage = buf
            .pinned
            .as_mut_slice()
            .context("get K3 pinned staging slice")?;
        let stage = &mut stage[..src.len()];
        self.fill.fill(src, stage);
        // SAFETY: the fill completed synchronously, the pinned allocation
        // outlives the DMA, and `dst_at` satisfies this function's contract.
        let copied = unsafe { memcpy_htod_async(dst_at, stage, self.stream.cu_stream()) };
        if let Err(copy_err) = copied {
            drain_or_abort(
                &self.stream,
                &format!("K3 staged H2D copy failed ({copy_err})"),
            );
            return Err(anyhow::anyhow!("K3 staged H2D copy failed: {copy_err}"));
        }
        if let Err(record_err) = buf.dma_done.record(&self.stream) {
            drain_or_abort(
                &self.stream,
                &format!("K3 staging event record failed ({record_err})"),
            );
            return Err(anyhow::anyhow!(
                "K3 staging event record failed: {record_err}"
            ));
        }
        Ok(())
    }
}

impl Drop for K3WeightStager {
    fn drop(&mut self) {
        for buf in &self.bufs {
            if let Err(err) = buf.dma_done.synchronize() {
                error!(
                    "K3 staging DMA drain failed on drop ({err}); aborting instead of freeing pinned memory under an in-flight DMA"
                );
                std::process::abort();
            }
        }
    }
}

fn drain_or_abort(stream: &CudaStream, context: &str) {
    if let Err(sync_err) = stream.synchronize() {
        error!(
            "{context}; stream drain failed ({sync_err}); aborting instead of freeing pinned memory under an in-flight DMA"
        );
        std::process::abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_fill_pool_copies_exact_bytes() -> Result<()> {
        let pool = FillPool::new(99)?;
        assert_eq!(pool.worker_count(), FILL_THREADS);
        for len in [0usize, 1, 3, 4095, 4096, 1 << 20, (3 << 20) + 17] {
            let src = (0..len).map(|i| (i % 251) as u8).collect::<Vec<_>>();
            let mut dst = vec![0xa5; len];
            pool.fill(&src, &mut dst);
            assert_eq!(dst, src, "persistent fill mismatch at len={len}");
        }
        Ok(())
    }
}
