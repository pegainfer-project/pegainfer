//! Two-stream prefill/decode overlap for Qwen3, optionally backed by Green
//! Context SM partitions.
//!
//! Selected on the CLI via `--decode-overlap` (see [`DecodeOverlap`]):
//!   * [`DecodeOverlap::Off`] — no streams here; the executor keeps its single stream.
//!   * [`DecodeOverlap::SharedSm`] — two CUDA streams on the primary context,
//!     sharing all SMs. Overlap without partitioning.
//!   * [`DecodeOverlap::GreenCtx`] — two streams pinned to disjoint Green Context
//!     SM partitions: decode gets `decode_pct`% of the SMs (memory-bound), the
//!     prefill stream the rest (compute-bound). When no prefill is pending,
//!     decode runs on the original full-SM stream.

use std::ptr;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUstream;
use cudarc::driver::sys::{self};
use pegainfer_kernels::green_ctx::GreenStream;
use pegainfer_kernels::green_ctx::SmSplit;

/// How prefill and decode share the GPU within a scheduler step. Selected on
/// the CLI via `--decode-overlap`.
///
/// `Off` keeps a single stream (lowest TTFT). The other two run prefill and
/// decode on separate CUDA streams so a long prefill no longer stalls running
/// decodes: `SharedSm` lets both streams use every SM, while `GreenCtx` pins
/// each to a disjoint Green Context SM partition (lower decode ITL p99 at the
/// cost of higher TTFT). Single-GPU Qwen3 only.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum DecodeOverlap {
    /// Single stream; prefill and decode serialize.
    #[default]
    Off,
    /// Two CUDA streams sharing all SMs.
    SharedSm,
    /// Green Context SM partition; `decode_pct`% of SMs assigned to decode.
    GreenCtx { decode_pct: u32 },
}

/// A `CUstream` wrapper that is `Send + Sync + Clone + Copy`.
/// SAFETY: CUDA streams are thread-safe when used on the same device.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct SendStream(pub CUstream);

unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

/// Two CUDA streams used to overlap prefill and decode within one scheduler
/// step. In [`DecodeOverlap::GreenCtx`] mode they are pinned to disjoint SM
/// partitions via Green Contexts; in [`DecodeOverlap::SharedSm`] mode they are
/// plain primary-context streams that share all SMs.
pub(crate) struct OverlapStreams {
    pub decode_stream: SendStream,
    pub prefill_stream: SendStream,
    /// Decode and prefill green streams, which own the two streams above;
    /// `None` in `SharedSm` mode.
    green: Option<[GreenStream; 2]>,
}

fn check_cu(result: sys::CUresult, msg: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("{msg}: CUresult = {result:?}");
    }
    Ok(())
}

/// Total SM count of the device.
fn query_total_sm(device: CUdevice) -> Result<u32> {
    let mut sm_res: sys::CUdevResource = unsafe { std::mem::zeroed() };
    check_cu(
        unsafe {
            sys::cuDeviceGetDevResource(
                device,
                &raw mut sm_res,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
            )
        },
        "cuDeviceGetDevResource",
    )?;
    Ok(unsafe { sm_res.__bindgen_anon_1.sm.smCount })
}

/// Create a non-blocking stream on the current (primary) context.
fn create_primary_stream() -> Result<CUstream> {
    let mut stream: CUstream = ptr::null_mut();
    check_cu(
        unsafe {
            sys::cuStreamCreate(
                &raw mut stream,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        },
        "cuStreamCreate",
    )?;
    Ok(stream)
}

impl OverlapStreams {
    /// Set up the overlap streams for the given device, or `None` when overlap
    /// is [`DecodeOverlap::Off`] (the executor keeps its single stream).
    pub(crate) fn create(device_ordinal: usize, overlap: DecodeOverlap) -> Result<Option<Self>> {
        let device: CUdevice = device_ordinal as i32;
        match overlap {
            DecodeOverlap::Off => Ok(None),
            DecodeOverlap::SharedSm => Self::create_shared(device).map(Some),
            DecodeOverlap::GreenCtx { decode_pct } => {
                Self::create_green(device, decode_pct).map(Some)
            }
        }
    }

    /// Two plain streams on the primary context — no SM partition.
    fn create_shared(device: CUdevice) -> Result<Self> {
        let total_sm = query_total_sm(device)?;
        let decode_stream = create_primary_stream()?;
        let prefill_stream = create_primary_stream()?;
        log::info!("Decode overlap: shared-SM, 2 primary-ctx streams (total={total_sm} SM)");
        Ok(Self {
            decode_stream: SendStream(decode_stream),
            prefill_stream: SendStream(prefill_stream),
            green: None,
        })
    }

    /// Green Context SM partition with SM-pinned streams.
    fn create_green(device: CUdevice, decode_pct: u32) -> Result<Self> {
        let mut split = SmSplit::new(device, decode_pct)?;
        // If a green stream cannot be created we do NOT silently fall back to
        // shared streams: the caller asked for an SM partition, so failing
        // loudly keeps benchmarks honest.
        let hint = "this driver/VRAM config may not support SM-pinned streams; use \
                    --decode-overlap stream for the partition-free two-stream path";
        let decode = GreenStream::new(device, &mut split.group).context(hint)?;
        let prefill = GreenStream::new(device, &mut split.rest).context(hint)?;
        log::info!(
            "Decode overlap: green-ctx SM partition decode={}SM prefill={}SM (total={}) [SM pinned]",
            split.group_sms(),
            split.rest_sms(),
            split.total
        );
        Ok(Self {
            decode_stream: SendStream(decode.stream),
            prefill_stream: SendStream(prefill.stream),
            green: Some([decode, prefill]),
        })
    }
}

impl Drop for OverlapStreams {
    fn drop(&mut self) {
        // Green streams destroy their own stream and context.
        if self.green.is_none() {
            unsafe {
                sys::cuStreamDestroy_v2(self.decode_stream.0);
                sys::cuStreamDestroy_v2(self.prefill_stream.0);
            }
        }
    }
}

// SAFETY: OverlapStreams is only used from the executor's single GPU worker thread.
unsafe impl Send for OverlapStreams {}
