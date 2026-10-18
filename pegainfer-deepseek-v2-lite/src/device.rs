use std::cell::Cell;

use anyhow::Result;
use anyhow::ensure;
use pegainfer_core::ffi;
use pegainfer_core::tensor::DeviceContext;

thread_local! {
    static ACTIVE_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
    static GRAPH_CAPTURE_ACTIVATION_ONLY: Cell<usize> = const { Cell::new(0) };
}

pub(crate) struct GraphCaptureActivationGuard;

impl Drop for GraphCaptureActivationGuard {
    fn drop(&mut self) {
        GRAPH_CAPTURE_ACTIVATION_ONLY.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "graph capture activation guard underflow");
            depth.set(current.saturating_sub(1));
        });
    }
}

pub(crate) fn graph_capture_activation_guard() -> GraphCaptureActivationGuard {
    GRAPH_CAPTURE_ACTIVATION_ONLY.with(|depth| depth.set(depth.get() + 1));
    GraphCaptureActivationGuard
}

pub(crate) fn activate(ctx: &DeviceContext) -> Result<()> {
    activate_impl(ctx, true)
}

pub(crate) fn activate_graph_capture(ctx: &DeviceContext) -> Result<()> {
    activate_impl(ctx, false)
}

fn activate_impl(ctx: &DeviceContext, allow_init: bool) -> Result<()> {
    ACTIVE_DEVICE.with(|active| {
        if active.get() == Some(ctx.device_ordinal) {
            return Ok(());
        }
        let allow_init = allow_init
            && GRAPH_CAPTURE_ACTIVATION_ONLY.with(|depth| depth.get() == 0);
        unsafe {
            let err = ffi::cuda_set_device(ctx.device_ordinal as i32);
            ensure!(
                err == 0,
                "failed to activate CUDA device {}: cudaError={err}",
                ctx.device_ordinal
            );
            if allow_init {
                ffi::cublas_init();
            } else {
                let err = ffi::cublas_activate_device_handles();
                ensure!(
                    err == 0,
                    "failed to activate preinitialized cuBLAS handles for CUDA device {}: cudaError={err}",
                    ctx.device_ordinal
                );
            }
        }
        active.set(Some(ctx.device_ordinal));
        Ok(())
    })
}
