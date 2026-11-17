use std::mem::MaybeUninit;
use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::PinnedHostSlice;
use cudarc::driver::result::memcpy_htod_async;
use cudarc::driver::sys::CUevent_flags;
use half::bf16;
use log::error;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use crate::tensor::DeviceContext;

const BF16_SIZE: usize = std::mem::size_of::<bf16>();
/// Per-buffer staging chunk. The measured 32 MiB geometry improves overlap and
/// limits the two pinned buffers to 64 MiB.
const STAGE_BYTES: usize = 32 << 20;
/// Widest fill team per rank, before the core-count cap.
const FILL_THREADS: usize = 8;
/// Small tails do not amortize dispatching work across the fill team.
const PARALLEL_FILL_MIN_BYTES: usize = 1 << 20;

fn fill_threads() -> usize {
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WIDTH.get_or_init(|| {
        let width = std::thread::available_parallelism()
            .expect("query available CPU parallelism")
            .get()
            .min(FILL_THREADS);
        if width < FILL_THREADS {
            log::info!("weight fill: {width} threads, capped by the core count");
        }
        width
    })
}

fn as_uninit(src: &[u8]) -> &[MaybeUninit<u8>] {
    // SAFETY: MaybeUninit<u8> shares u8's layout and adds no validity
    // requirement, so widening an initialized slice is sound.
    unsafe { std::slice::from_raw_parts(src.as_ptr().cast(), src.len()) }
}

struct FillPool {
    pool: ThreadPool,
    workers: usize,
}

impl FillPool {
    fn new() -> Result<Self> {
        Self::with_workers(fill_threads())
    }

    fn with_workers(workers: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|worker| format!("weight-fill-{worker}"))
            .build()
            .map_err(|e| anyhow::anyhow!("build weight-fill pool failed: {e}"))?;
        Ok(Self { pool, workers })
    }

    fn copy(&self, src: &[u8], dst: &mut [MaybeUninit<u8>]) {
        debug_assert_eq!(src.len(), dst.len());
        if src.len() < PARALLEL_FILL_MIN_BYTES || self.workers == 1 {
            dst.copy_from_slice(as_uninit(src));
            return;
        }
        let per = src.len().div_ceil(self.workers);
        self.pool.scope(|scope| {
            for (src_part, dst_part) in src.chunks(per).zip(dst.chunks_mut(per)) {
                scope.spawn(move |_| dst_part.copy_from_slice(as_uninit(src_part)));
            }
        });
    }

    fn gather_cols(
        &self,
        src: &[u8],
        stride_b: usize,
        off_b: usize,
        take_b: usize,
        rows: usize,
        dst: &mut [MaybeUninit<u8>],
    ) {
        debug_assert!(off_b + take_b <= stride_b);
        debug_assert!(rows * stride_b <= src.len());
        debug_assert_eq!(rows * take_b, dst.len());
        if dst.len() < PARALLEL_FILL_MIN_BYTES || self.workers == 1 {
            for (src_row, dst_row) in src[..rows * stride_b]
                .chunks(stride_b)
                .zip(dst.chunks_mut(take_b))
            {
                dst_row.copy_from_slice(as_uninit(&src_row[off_b..off_b + take_b]));
            }
            return;
        }
        let rows_per = rows.div_ceil(self.workers);
        self.pool.scope(|scope| {
            for (src_rows, dst_rows) in src[..rows * stride_b]
                .chunks(rows_per * stride_b)
                .zip(dst.chunks_mut(rows_per * take_b))
            {
                scope.spawn(move |_| {
                    for (src_row, dst_row) in
                        src_rows.chunks(stride_b).zip(dst_rows.chunks_mut(take_b))
                    {
                        dst_row.copy_from_slice(as_uninit(&src_row[off_b..off_b + take_b]));
                    }
                });
            }
        });
    }
}

struct StagingBuf {
    pinned: PinnedHostSlice<bf16>,
    dma_done: CudaEvent,
}

/// Validated execution plan for one strided column-shard upload.
pub(crate) struct ColShardPlan {
    stride_b: usize,
    off_b: usize,
    take_b: usize,
    rows: usize,
    dst_at: u64,
}

/// Pinned double-buffering overlaps the source read with the H2D copy that
/// pageable `clone_htod` would serialize; sources are raw bytes.
pub(crate) struct WeightStager {
    stream: Arc<CudaStream>,
    bufs: [StagingBuf; 2],
    fill: FillPool,
    next: usize,
}

impl WeightStager {
    pub(crate) fn new(ctx: &DeviceContext) -> Result<Self> {
        let make = || -> Result<StagingBuf> {
            // SAFETY: every byte a DMA reads is initialized by `stage_chunk`'s
            // fill callback first; buffer reuse is gated on `dma_done`.
            let pinned = unsafe { ctx.ctx.alloc_pinned::<bf16>(STAGE_BYTES / BF16_SIZE) }
                .map_err(|e| anyhow::anyhow!("pinned staging alloc failed: {e}"))?;
            let dma_done = ctx
                .ctx
                .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| anyhow::anyhow!("staging event create failed: {e}"))?;
            Ok(StagingBuf { pinned, dma_done })
        };
        Ok(Self {
            stream: ctx.stream.clone(),
            bufs: [make()?, make()?],
            fill: FillPool::new()?,
            next: 0,
        })
    }

    /// # Safety
    /// `dst_at` must address `src.len()` writable bytes still allocated on the
    /// stager's stream, as validated by [`prepare`].
    pub(crate) unsafe fn upload_at(&mut self, src: &[u8], dst_at: u64) -> Result<()> {
        for (i, chunk) in src.chunks(STAGE_BYTES).enumerate() {
            let chunk_at = dst_at + (i * STAGE_BYTES) as u64;
            let fill = |pool: &FillPool, stage: &mut [MaybeUninit<u8>]| pool.copy(chunk, stage);
            // SAFETY: the chunks partition `src`, so `chunk_at` stays inside
            // the validated destination range, with `chunk.len() <= STAGE_BYTES`.
            unsafe { self.stage_chunk(chunk.len(), chunk_at, fill) }?;
        }
        Ok(())
    }

    /// # Safety
    /// `dst_at` must address the sum of `srcs` writable bytes still allocated
    /// on the stager's stream, as validated by [`prepare_bytes`].
    pub(crate) unsafe fn upload_slices_at(&mut self, srcs: &[&[u8]], dst_at: u64) -> Result<()> {
        let mut source = 0;
        let mut source_at = 0;
        let mut dst_offset = 0;
        let mut parts = Vec::new();
        while source < srcs.len() {
            let bytes =
                next_contiguous_chunk(srcs, &mut source, &mut source_at, STAGE_BYTES, &mut parts);
            let fill = |pool: &FillPool, stage: &mut [MaybeUninit<u8>]| {
                let mut at = 0;
                for part in &parts {
                    pool.copy(part, &mut stage[at..at + part.len()]);
                    at += part.len();
                }
            };
            // SAFETY: the chunks partition the validated concatenated source.
            unsafe { self.stage_chunk(bytes, dst_at + dst_offset as u64, fill) }?;
            dst_offset += bytes;
        }
        Ok(())
    }

    /// # Safety
    /// `plan` must come from [`prepare_cols`] for this `src`, with its
    /// destination still allocated on the stager's stream.
    pub(crate) unsafe fn upload_cols_at(&mut self, src: &[u8], plan: &ColShardPlan) -> Result<()> {
        let rows_per_chunk = STAGE_BYTES / plan.take_b;
        let mut row = 0;
        while row < plan.rows {
            let chunk_rows = rows_per_chunk.min(plan.rows - row);
            let dst_at = plan.dst_at + (row * plan.take_b) as u64;
            let fill = |pool: &FillPool, stage: &mut [MaybeUninit<u8>]| {
                pool.gather_cols(
                    &src[row * plan.stride_b..],
                    plan.stride_b,
                    plan.off_b,
                    plan.take_b,
                    chunk_rows,
                    stage,
                );
            };
            // SAFETY: the destination rows lie inside the validated range per
            // the rows x take bound, with `chunk_rows * take_b <= STAGE_BYTES`.
            unsafe { self.stage_chunk(chunk_rows * plan.take_b, dst_at, fill) }?;
            row += chunk_rows;
        }
        Ok(())
    }

    /// # Safety
    /// `dst_at` must address `bytes <= STAGE_BYTES` writable bytes on the
    /// stager's stream, and `fill` must initialize the `bytes` it is given.
    unsafe fn stage_chunk(
        &mut self,
        bytes: usize,
        dst_at: u64,
        fill: impl FnOnce(&FillPool, &mut [MaybeUninit<u8>]),
    ) -> Result<()> {
        let idx = self.next;
        self.next = (self.next + 1) % self.bufs.len();
        let buf = &mut self.bufs[idx];
        buf.dma_done
            .synchronize()
            .map_err(|e| anyhow::anyhow!("staging drain failed: {e}"))?;
        let stage = buf
            .pinned
            .as_mut_ptr()
            .map_err(|e| anyhow::anyhow!("staging pointer failed: {e}"))?
            .cast::<u8>();
        // SAFETY: the pinned allocation contains STAGE_BYTES bytes and this
        // function requires bytes <= STAGE_BYTES.
        let staged =
            unsafe { std::slice::from_raw_parts_mut(stage.cast::<MaybeUninit<u8>>(), bytes) };
        fill(&self.fill, staged);
        // SAFETY: `fill` initialized `bytes` at `stage` and `dst_at` is valid
        // per the contract; the buffer outlives the copy (`dma_done` or the
        // drain-or-abort branches), and the event synchronize above bound the
        // context to this thread.
        let copied = unsafe {
            let staged = std::slice::from_raw_parts(stage.cast_const(), bytes);
            memcpy_htod_async(dst_at, staged, self.stream.cu_stream())
        };
        if let Err(copy_err) = copied {
            // An async-API error can stem from earlier work on the stream and
            // does not prove this copy never started.
            drain_or_abort(
                &self.stream,
                &format!("staged H2D copy failed ({copy_err})"),
            );
            return Err(anyhow::anyhow!("staged H2D copy failed: {copy_err}"));
        }
        if let Err(record_err) = buf.dma_done.record(&self.stream) {
            // The copy is in flight with no event covering it.
            drain_or_abort(
                &self.stream,
                &format!("staging record failed ({record_err})"),
            );
            return Err(anyhow::anyhow!("staging record failed: {record_err}"));
        }
        Ok(())
    }
}

/// Fills `parts` (cleared first, reused across calls) with the source slices
/// of the next chunk of at most `limit` bytes; returns the chunk's size.
/// `limit` is a parameter so the boundary coverage below runs on dozens of
/// bytes instead of a STAGE_BYTES-sized fixture.
fn next_contiguous_chunk<'a>(
    srcs: &'a [&'a [u8]],
    source: &mut usize,
    source_at: &mut usize,
    limit: usize,
    parts: &mut Vec<&'a [u8]>,
) -> usize {
    parts.clear();
    let mut bytes = 0;
    while *source < srcs.len() && bytes < limit {
        let src = srcs[*source];
        if src.is_empty() {
            *source += 1;
            *source_at = 0;
            continue;
        }
        let take = (limit - bytes).min(src.len() - *source_at);
        parts.push(&src[*source_at..*source_at + take]);
        bytes += take;
        *source_at += take;
        if *source_at == src.len() {
            *source += 1;
            *source_at = 0;
        }
    }
    bytes
}

// Both the stager's events and `dst`'s stream-ordered allocation are only
// ordered against work on `stream`, which must be the stager's own stream.
fn ensure_uploadable<T>(stream: &Arc<CudaStream>, dst: &CudaSlice<T>) -> Result<()> {
    anyhow::ensure!(
        Arc::ptr_eq(dst.stream(), stream),
        "staged upload into a buffer allocated on a different stream than the stager's"
    );
    anyhow::ensure!(
        !crate::tensor::has_stream_override(),
        "staged upload under a thread-local stream override is unsupported"
    );
    Ok(())
}

pub(crate) fn prepare_bytes(
    stream: &Arc<CudaStream>,
    srcs: &[&[u8]],
    dst: &mut CudaSlice<u8>,
) -> Result<u64> {
    ensure_uploadable(stream, dst)?;
    let bytes = srcs.iter().try_fold(0usize, |total, src| {
        total
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("staged byte upload source length overflow"))
    })?;
    anyhow::ensure!(
        bytes == dst.len(),
        "staged byte upload size mismatch: {bytes} source bytes vs dst len {}",
        dst.len()
    );
    let (dst_ptr, _dst_order) = dst.device_ptr_mut(stream);
    Ok(dst_ptr)
}

/// Validates a contiguous staged upload and returns the destination device
/// address for a deferred [`WeightStager::upload_at`].
pub(crate) fn prepare(
    stream: &Arc<CudaStream>,
    src: &[u8],
    dst: &mut CudaSlice<bf16>,
    dst_offset: usize,
) -> Result<u64> {
    ensure_uploadable(stream, dst)?;
    anyhow::ensure!(
        src.len().is_multiple_of(BF16_SIZE),
        "staged upload source of {} bytes is not a whole number of bf16 elements",
        src.len()
    );
    anyhow::ensure!(
        dst_offset
            .checked_mul(BF16_SIZE)
            .and_then(|off| off.checked_add(src.len()))
            .is_some_and(|end| end <= dst.len() * BF16_SIZE),
        "staged upload out of bounds: dst_offset {dst_offset} + src bytes {} > dst len {}",
        src.len(),
        dst.len()
    );
    // Dropping the guard immediately is fine only because the runtime context
    // disables cudarc event tracking; revisit if that ever changes.
    let (dst_ptr, _dst_order) = dst.device_ptr_mut(stream);
    Ok(dst_ptr + (dst_offset * BF16_SIZE) as u64)
}

/// Validates a strided staged upload and returns its execution plan for a
/// deferred [`WeightStager::upload_cols_at`] (column counts in elements).
pub(crate) fn prepare_cols(
    stream: &Arc<CudaStream>,
    src: &[u8],
    total_cols: usize,
    col_offset: usize,
    take: usize,
    dst: &mut CudaSlice<bf16>,
) -> Result<ColShardPlan> {
    ensure_uploadable(stream, dst)?;
    let stride_b = total_cols
        .checked_mul(BF16_SIZE)
        .filter(|&s| s > 0 && src.len().is_multiple_of(s));
    anyhow::ensure!(
        stride_b.is_some(),
        "strided upload source of {} bytes is not a multiple of {total_cols} bf16 columns",
        src.len()
    );
    let stride_b = stride_b.unwrap();
    anyhow::ensure!(
        col_offset
            .checked_add(take)
            .is_some_and(|end| end <= total_cols),
        "col range out of bounds: col_offset={col_offset} take={take} total_cols={total_cols}"
    );
    let take_b = take * BF16_SIZE;
    anyhow::ensure!(
        (1..=STAGE_BYTES).contains(&take_b),
        "column shard width {take} outside 1..={}",
        STAGE_BYTES / BF16_SIZE
    );
    let rows = src.len() / stride_b;
    anyhow::ensure!(
        rows.checked_mul(take).is_some_and(|n| n == dst.len()),
        "staged upload shape mismatch: {rows} rows x {take} cols vs dst len {}",
        dst.len()
    );
    let (dst_ptr, _dst_order) = dst.device_ptr_mut(stream);
    Ok(ColShardPlan {
        stride_b,
        off_b: col_offset * BF16_SIZE,
        take_b,
        rows,
        dst_at: dst_ptr,
    })
}

impl Drop for WeightStager {
    fn drop(&mut self) {
        // PinnedHostSlice's drop only waits on its embedded, never-recorded
        // event; drain ours instead and fail closed when the DMA state is
        // unknown.
        for buf in &self.bufs {
            if let Err(err) = buf.dma_done.synchronize() {
                error!(
                    "staging DMA drain failed on drop ({err}); aborting instead of freeing pinned memory under an in-flight DMA"
                );
                std::process::abort();
            }
        }
    }
}

pub(super) fn drain_or_abort(stream: &CudaStream, context: &str) {
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
    fn fill_pool_strided_matches_scalar_gather() {
        // Fixed width: `FillPool::new()` would collapse to the serial path on a
        // single-CPU runner, so the parallel case below would not be exercised.
        let pool = FillPool::with_workers(2).expect("build fill pool");
        for &(rows, total_cols, col_offset, take) in &[
            (1usize, 7usize, 0usize, 7usize),
            (3, 8, 2, 5),
            (4, 5, 1, 4),
            (9, 6, 3, 3),
            // Past the parallel threshold, with rows that do not divide evenly.
            (601, 2560, 640, 1280),
        ] {
            let (stride_b, off_b, take_b) = (
                total_cols * BF16_SIZE,
                col_offset * BF16_SIZE,
                take * BF16_SIZE,
            );
            let mut buf = vec![0u8; rows * stride_b];
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            let src = &buf[..];
            let mut dst = vec![MaybeUninit::uninit(); rows * take_b];
            pool.gather_cols(src, stride_b, off_b, take_b, rows, &mut dst);
            // SAFETY: gather_cols wrote every byte of `dst`.
            let dst = unsafe { std::slice::from_raw_parts(dst.as_ptr().cast::<u8>(), dst.len()) };
            let mut expect = Vec::with_capacity(rows * take_b);
            for r in 0..rows {
                for c in 0..take_b {
                    expect.push(src[r * stride_b + off_b + c]);
                }
            }
            assert_eq!(
                dst, expect,
                "rows={rows} total_cols={total_cols} col_offset={col_offset} take={take}"
            );
        }
    }

    #[test]
    fn contiguous_chunks_preserve_source_bytes() {
        let limit = 46;
        let first = vec![1u8; limit - 2];
        let second = [2u8, 3, 4, 5];
        let sources: [&[u8]; 3] = [&first, &[], &second];
        let mut source = 0;
        let mut source_at = 0;
        let mut actual = Vec::new();
        let mut sizes = Vec::new();
        let mut parts = Vec::new();
        while source < sources.len() {
            let bytes =
                next_contiguous_chunk(&sources, &mut source, &mut source_at, limit, &mut parts);
            sizes.push(bytes);
            actual.extend(parts.iter().copied().flatten().copied());
        }
        let expected: Vec<u8> = sources.into_iter().flatten().copied().collect();
        assert_eq!(sizes, [limit, 2]);
        assert_eq!(actual, expected);
    }
}
