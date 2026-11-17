//! SM-capped prefill stream for the async-prefill overlap (qwen3's
//! green-context machinery, asymmetric form): the overlapped prefill runs on
//! a Green Context stream pinned to a fraction of the SMs, while decode
//! stays on the primary-context stream with the full device. Capping the
//! prefill footprint is what protects decode ITL — the naive shared-SM
//! overlap lets prefill's large grids starve decode steps.

use anyhow::Result;
use anyhow::bail;
use cudarc::driver::sys;
use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUstream;
use pegainfer_kernels::green_ctx::GreenStream;
use pegainfer_kernels::green_ctx::SmSplit;

fn check_cu(result: sys::CUresult, msg: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("{msg} failed: {result:?}");
    }
    Ok(())
}

/// The prefill lane's stream, either a plain primary-context stream
/// (shared SMs) or a Green Context stream pinned to `prefill_pct`% of them.
pub(crate) struct PrefillLaneStream {
    pub(crate) stream: CUstream,
    /// Owns `stream` when the lane is pinned.
    green: Option<GreenStream>,
}

// SAFETY: the stream and green-context handles are process-wide CUDA driver
// objects with no thread affinity. The lane is built on the launching thread
// and handed to the scheduler thread with the rest of the engine state, which
// is its only user from then on; nothing here is shared or aliased.
unsafe impl Send for PrefillLaneStream {}

impl PrefillLaneStream {
    /// A plain primary-context stream — both engines share every SM.
    pub(crate) fn shared() -> Result<Self> {
        let mut stream: CUstream = std::ptr::null_mut();
        check_cu(
            unsafe {
                sys::cuStreamCreate(
                    &raw mut stream,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "cuStreamCreate (prefill lane)",
        )?;
        log::info!("async prefill: shared-SM lane stream");
        Ok(Self {
            stream,
            green: None,
        })
    }

    /// A Green Context stream pinned to roughly `prefill_pct`% of the SMs
    /// (rounded down to the split granularity). Fails loudly rather than
    /// falling back to shared SMs, so a pinned lane is never silently
    /// unpinned.
    pub(crate) fn green(device_ordinal: usize, prefill_pct: u32) -> Result<Self> {
        let device: CUdevice = device_ordinal as i32;
        let mut split = SmSplit::new(device, prefill_pct)?;
        let green = GreenStream::new(device, &mut split.group)?;
        log::info!(
            "async prefill: green-ctx lane pinned to {}/{} SMs \
             (decode keeps the primary context)",
            split.group_sms(),
            split.total
        );
        Ok(Self {
            stream: green.stream,
            green: Some(green),
        })
    }

    /// Record `event` on the lane stream. Green Context streams need the
    /// event recorded through their green context; primary streams take the
    /// plain record.
    pub(crate) fn record_event(&self, event: sys::CUevent) -> Result<()> {
        let record = unsafe {
            match &self.green {
                Some(g) => sys::cuGreenCtxRecordEvent(g.ctx, event),
                None => sys::cuEventRecord(event, self.stream),
            }
        };
        check_cu(record, "record prefill completion event")
    }
}

impl Drop for PrefillLaneStream {
    fn drop(&mut self) {
        // A green stream destroys its own stream and context.
        if self.green.is_none() {
            unsafe {
                sys::cuStreamDestroy_v2(self.stream);
            }
        }
    }
}
