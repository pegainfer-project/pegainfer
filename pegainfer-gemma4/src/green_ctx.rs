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

fn check_cu(result: sys::CUresult, msg: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        bail!("{msg} failed: {result:?}");
    }
    Ok(())
}

struct GreenContexts {
    gctx_prefill: sys::CUgreenCtx,
    _ctx_prefill: sys::CUcontext,
}

fn sm_for_prefill(total_sm: u32, min_sm: u32, prefill_pct: u32) -> Option<u32> {
    let target = (total_sm * prefill_pct / 100 / min_sm) * min_sm;
    (target >= min_sm && total_sm - target >= min_sm).then_some(target)
}

/// The prefill lane's stream, either a plain primary-context stream
/// (shared SMs) or a Green Context stream pinned to `prefill_pct`% of them.
pub(crate) struct PrefillLaneStream {
    pub(crate) stream: CUstream,
    green: Option<GreenContexts>,
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
        log::info!("gemma4 async prefill: shared-SM lane stream");
        Ok(Self {
            stream,
            green: None,
        })
    }

    /// A Green Context stream pinned to roughly `prefill_pct`% of the SMs
    /// (rounded down to the split granularity). Fails loudly rather than
    /// falling back to shared SMs, so benchmarks stay honest.
    pub(crate) fn green(device_ordinal: usize, prefill_pct: u32) -> Result<Self> {
        let device: CUdevice = device_ordinal as i32;
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

        let sm_for_prefill = sm_for_prefill(total_sm, min_sm, prefill_pct).ok_or_else(|| {
            anyhow::anyhow!(
                "green-ctx prefill partition not viable: total={total_sm} min={min_sm} \
                 prefill_pct={prefill_pct}"
            )
        })?;

        let mut grp_prefill: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut grp_rest: sys::CUdevResource = unsafe { std::mem::zeroed() };
        nb = 1;
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &raw mut grp_prefill,
                    &raw mut nb,
                    &raw const sm_res,
                    &raw mut grp_rest,
                    0,
                    sm_for_prefill,
                )
            },
            "cuDevSmResourceSplitByCount",
        )?;
        let sm_prefill = unsafe { grp_prefill.__bindgen_anon_1.sm.smCount };

        let mut desc_prefill: sys::CUdevResourceDesc = std::ptr::null_mut();
        check_cu(
            unsafe {
                sys::cuDevResourceGenerateDesc(&raw mut desc_prefill, &raw mut grp_prefill, 1)
            },
            "cuDevResourceGenerateDesc (prefill)",
        )?;
        let mut gctx_prefill: sys::CUgreenCtx = std::ptr::null_mut();
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
        let mut ctx_prefill: sys::CUcontext = std::ptr::null_mut();
        check_cu(
            unsafe { sys::cuCtxFromGreenCtx(&raw mut ctx_prefill, gctx_prefill) },
            "cuCtxFromGreenCtx (prefill)",
        )?;

        let mut stream: CUstream = std::ptr::null_mut();
        let create = unsafe {
            sys::cuGreenCtxStreamCreate(
                &raw mut stream,
                gctx_prefill,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                0,
            )
        };
        if create != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                sys::cuGreenCtxDestroy(gctx_prefill);
            }
            bail!("cuGreenCtxStreamCreate failed ({create:?})");
        }

        log::info!(
            "gemma4 async prefill: green-ctx lane pinned to {sm_prefill}/{total_sm} SMs \
             (decode keeps the primary context)"
        );
        Ok(Self {
            stream,
            green: Some(GreenContexts {
                gctx_prefill,
                _ctx_prefill: ctx_prefill,
            }),
        })
    }

    /// Record `event` on the lane stream. Green Context streams need the
    /// event recorded through their green context; primary streams take the
    /// plain record.
    pub(crate) fn record_event(&self, event: sys::CUevent) -> Result<()> {
        let record = unsafe {
            match &self.green {
                Some(g) => sys::cuGreenCtxRecordEvent(g.gctx_prefill, event),
                None => sys::cuEventRecord(event, self.stream),
            }
        };
        check_cu(record, "record prefill completion event")
    }
}

impl Drop for PrefillLaneStream {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.stream);
            if let Some(green) = &self.green {
                sys::cuGreenCtxDestroy(green.gctx_prefill);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sm_for_prefill;

    #[test]
    fn prefill_partition_rounds_to_the_split_granularity() {
        assert_eq!(sm_for_prefill(128, 8, 35), Some(40));
        assert_eq!(sm_for_prefill(128, 8, 50), Some(64));
        assert_eq!(sm_for_prefill(128, 8, 99), Some(120));
    }

    #[test]
    fn prefill_partition_keeps_one_group_for_decode() {
        assert_eq!(sm_for_prefill(128, 8, 1), None);
        assert_eq!(sm_for_prefill(128, 8, 100), None);
        assert_eq!(sm_for_prefill(8, 8, 50), None);
    }
}
