//! Green Context SM partitions for the overlap lanes: split a device's SMs
//! into a group of about `pct`% and the rest, and open a non-blocking stream
//! on a green context over one side.

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

/// `pct`% of `total_sm`, rounded down to the split granularity `min_sm`, or
/// `None` when either side of the split would be left without a group.
fn sm_target(total_sm: u32, min_sm: u32, pct: u32) -> Option<u32> {
    let target = (total_sm * pct / 100 / min_sm) * min_sm;
    (target >= min_sm && total_sm - target >= min_sm).then_some(target)
}

/// A device's SMs split into `group`, about the requested percentage, and
/// `rest`.
pub struct SmSplit {
    pub group: sys::CUdevResource,
    pub rest: sys::CUdevResource,
    pub total: u32,
}

impl SmSplit {
    /// Fails when the device cannot give both sides at least one group.
    pub fn new(device: CUdevice, pct: u32) -> Result<Self> {
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
        let total = unsafe { sm_res.__bindgen_anon_1.sm.smCount };

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
        let Some(target) = sm_target(total, min_sm, pct) else {
            bail!("green-ctx SM partition not viable: total={total} min={min_sm} pct={pct}");
        };

        let mut group: sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut rest: sys::CUdevResource = unsafe { std::mem::zeroed() };
        nb = 1;
        check_cu(
            unsafe {
                sys::cuDevSmResourceSplitByCount(
                    &raw mut group,
                    &raw mut nb,
                    &raw const sm_res,
                    &raw mut rest,
                    0,
                    target,
                )
            },
            "cuDevSmResourceSplitByCount",
        )?;
        Ok(Self { group, rest, total })
    }

    pub fn group_sms(&self) -> u32 {
        unsafe { self.group.__bindgen_anon_1.sm.smCount }
    }

    pub fn rest_sms(&self) -> u32 {
        unsafe { self.rest.__bindgen_anon_1.sm.smCount }
    }
}

/// A green context over one SM resource and a non-blocking stream on it.
/// Drop destroys the stream, then the context.
pub struct GreenStream {
    pub ctx: sys::CUgreenCtx,
    pub stream: CUstream,
}

impl GreenStream {
    pub fn new(device: CUdevice, resource: &mut sys::CUdevResource) -> Result<Self> {
        let mut desc: sys::CUdevResourceDesc = std::ptr::null_mut();
        check_cu(
            unsafe { sys::cuDevResourceGenerateDesc(&raw mut desc, resource, 1) },
            "cuDevResourceGenerateDesc",
        )?;
        let mut ctx: sys::CUgreenCtx = std::ptr::null_mut();
        check_cu(
            unsafe {
                sys::cuGreenCtxCreate(
                    &raw mut ctx,
                    desc,
                    device,
                    sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
                )
            },
            "cuGreenCtxCreate",
        )?;
        let mut stream: CUstream = std::ptr::null_mut();
        let create = unsafe {
            sys::cuGreenCtxStreamCreate(
                &raw mut stream,
                ctx,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                0,
            )
        };
        if create != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                sys::cuGreenCtxDestroy(ctx);
            }
            bail!("cuGreenCtxStreamCreate failed: {create:?}");
        }
        Ok(Self { ctx, stream })
    }
}

impl Drop for GreenStream {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.stream);
            sys::cuGreenCtxDestroy(self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sm_target;

    #[test]
    fn target_rounds_to_the_split_granularity() {
        assert_eq!(sm_target(128, 8, 35), Some(40));
        assert_eq!(sm_target(128, 8, 50), Some(64));
        assert_eq!(sm_target(128, 8, 99), Some(120));
    }

    #[test]
    fn target_keeps_one_group_on_each_side() {
        assert_eq!(sm_target(128, 8, 1), None);
        assert_eq!(sm_target(128, 8, 100), None);
        assert_eq!(sm_target(8, 8, 50), None);
    }
}
