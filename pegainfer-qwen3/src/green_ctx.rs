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

use anyhow::Result;
use anyhow::bail;
use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUstream;
use cudarc::driver::sys::{self};

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

/// Order `ctx.stream` producers (alloc/H2D/memset) before the override stream
/// consumes them. No-op without an override.
pub(crate) fn fence_producers_before_override(
    ctx: &pegainfer_core::tensor::DeviceContext,
) -> Result<()> {
    if !pegainfer_kernels::tensor::has_stream_override() {
        return Ok(());
    }
    let override_stream = pegainfer_kernels::tensor::active_cu_stream(ctx);
    let producer = ctx.stream.cu_stream();
    unsafe {
        let mut event: sys::CUevent = ptr::null_mut();
        check_cu(
            sys::cuEventCreate(
                &raw mut event,
                sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32,
            ),
            "cuEventCreate (producer fence)",
        )?;
        let record = sys::cuEventRecord(event, producer);
        let wait = if record == sys::CUresult::CUDA_SUCCESS {
            sys::cuStreamWaitEvent(override_stream, event, 0)
        } else {
            record
        };
        let destroy = sys::cuEventDestroy_v2(event);
        check_cu(record, "cuEventRecord (producer fence)")?;
        check_cu(wait, "cuStreamWaitEvent (producer fence)")?;
        check_cu(destroy, "cuEventDestroy (producer fence)")?;
    }
    Ok(())
}

/// Two CUDA streams used to overlap prefill and decode within one scheduler
/// step. In [`DecodeOverlap::GreenCtx`] mode they are pinned to disjoint SM
/// partitions via Green Contexts; in [`DecodeOverlap::SharedSm`] mode they are
/// plain primary-context streams that share all SMs.
pub(crate) struct OverlapStreams {
    pub decode_stream: SendStream,
    pub prefill_stream: SendStream,
    /// Green contexts owning the SM partitions; `None` in `SharedSm` mode.
    green: Option<GreenContexts>,
}

/// Green Context handles kept alive for the lifetime of the partition streams.
struct GreenContexts {
    gctx_decode: sys::CUgreenCtx,
    gctx_prefill: sys::CUgreenCtx,
    // CUcontext handles derived from the green contexts; held so the streams'
    // backing contexts outlive them.
    _ctx_decode: sys::CUcontext,
    _ctx_prefill: sys::CUcontext,
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
        // Query SM resource and the minimum split granularity.
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
        let total_sm = unsafe { sm_res.__bindgen_anon_1.sm.smCount };

        let mut nb: u32 = 1;
        let mut probe_grp: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut probe_rem: sys::CUdevResource = unsafe { std::mem::zeroed() };
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &raw mut probe_grp,
                    &raw mut nb,
                    &raw const sm_res,
                    &raw mut probe_rem,
                    0,
                    1,
                )
            },
            "probe split",
        )?;
        let min_sm = unsafe { probe_grp.__bindgen_anon_1.sm.smCount };

        // Compute decode SM count aligned to the minimum granularity.
        let sm_for_decode = (total_sm * decode_pct / 100 / min_sm) * min_sm;
        if sm_for_decode < min_sm || total_sm - sm_for_decode < min_sm {
            bail!(
                "green-ctx SM partition not viable: total={total_sm} min={min_sm} \
                 decode_pct={decode_pct} decode_target={sm_for_decode}"
            );
        }

        // Split SMs into a decode group and a prefill group.
        let mut grp_decode: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut grp_prefill: sys::CUdevResource = unsafe { std::mem::zeroed() };
        nb = 1;
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &raw mut grp_decode,
                    &raw mut nb,
                    &raw const sm_res,
                    &raw mut grp_prefill,
                    0,
                    sm_for_decode,
                )
            },
            "cuDevSmResourceSplitByCount",
        )?;
        let sm_decode = unsafe { grp_decode.__bindgen_anon_1.sm.smCount };
        let sm_prefill = unsafe { grp_prefill.__bindgen_anon_1.sm.smCount };

        // Generate resource descriptors and green contexts.
        let mut desc_decode: sys::CUdevResourceDesc = ptr::null_mut();
        let mut desc_prefill: sys::CUdevResourceDesc = ptr::null_mut();
        check_cu(
            unsafe { sys::cuDevResourceGenerateDesc(&raw mut desc_decode, &raw mut grp_decode, 1) },
            "cuDevResourceGenerateDesc (decode)",
        )?;
        check_cu(
            unsafe {
                sys::cuDevResourceGenerateDesc(&raw mut desc_prefill, &raw mut grp_prefill, 1)
            },
            "cuDevResourceGenerateDesc (prefill)",
        )?;

        let mut gctx_decode: sys::CUgreenCtx = ptr::null_mut();
        let mut gctx_prefill: sys::CUgreenCtx = ptr::null_mut();
        check_cu(
            unsafe {
                sys::cuGreenCtxCreate(
                    &raw mut gctx_decode,
                    desc_decode,
                    device,
                    sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
                )
            },
            "cuGreenCtxCreate (decode)",
        )?;
        check_cu(
            unsafe {
                sys::cuGreenCtxCreate(
                    &raw mut gctx_prefill,
                    desc_prefill,
                    device,
                    sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
                )
            },
            "cuGreenCtxCreate (prefill)",
        )?;

        let mut ctx_decode: sys::CUcontext = ptr::null_mut();
        let mut ctx_prefill: sys::CUcontext = ptr::null_mut();
        check_cu(
            unsafe { sys::cuCtxFromGreenCtx(&raw mut ctx_decode, gctx_decode) },
            "cuCtxFromGreenCtx (decode)",
        )?;
        check_cu(
            unsafe { sys::cuCtxFromGreenCtx(&raw mut ctx_prefill, gctx_prefill) },
            "cuCtxFromGreenCtx (prefill)",
        )?;

        // SM-pinned streams. If cuGreenCtxStreamCreate fails we do NOT silently
        // fall back to shared streams: the caller asked for an SM partition, so
        // failing loudly keeps benchmarks honest (use `--decode-overlap stream`
        // for the partition-free two-stream path).
        let mut decode_stream: CUstream = ptr::null_mut();
        let mut prefill_stream: CUstream = ptr::null_mut();
        let r1 = unsafe {
            sys::cuGreenCtxStreamCreate(
                &raw mut decode_stream,
                gctx_decode,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                0,
            )
        };
        let r2 = unsafe {
            sys::cuGreenCtxStreamCreate(
                &raw mut prefill_stream,
                gctx_prefill,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                0,
            )
        };
        if r1 != sys::CUresult::CUDA_SUCCESS || r2 != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                if !decode_stream.is_null() {
                    sys::cuStreamDestroy_v2(decode_stream);
                }
                if !prefill_stream.is_null() {
                    sys::cuStreamDestroy_v2(prefill_stream);
                }
                sys::cuGreenCtxDestroy(gctx_decode);
                sys::cuGreenCtxDestroy(gctx_prefill);
            }
            bail!(
                "cuGreenCtxStreamCreate failed (decode={r1:?}, prefill={r2:?}); this driver/VRAM \
                 config may not support SM-pinned streams — use --decode-overlap stream"
            );
        }

        log::info!(
            "Decode overlap: green-ctx SM partition decode={sm_decode}SM prefill={sm_prefill}SM \
             (total={total_sm}) [SM pinned]"
        );
        Ok(Self {
            decode_stream: SendStream(decode_stream),
            prefill_stream: SendStream(prefill_stream),
            green: Some(GreenContexts {
                gctx_decode,
                gctx_prefill,
                _ctx_decode: ctx_decode,
                _ctx_prefill: ctx_prefill,
            }),
        })
    }
}

impl Drop for OverlapStreams {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.decode_stream.0);
            sys::cuStreamDestroy_v2(self.prefill_stream.0);
            if let Some(green) = &self.green {
                sys::cuGreenCtxDestroy(green.gctx_decode);
                sys::cuGreenCtxDestroy(green.gctx_prefill);
            }
        }
    }
}

// SAFETY: OverlapStreams is only used from the executor's single GPU worker thread.
unsafe impl Send for OverlapStreams {}
