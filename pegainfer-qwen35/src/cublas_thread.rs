//! Bind the calling thread to a model's CUDA device/context and cuBLAS
//! handle; the guard destroys the handle when the binding ends.

use anyhow::Result;

use crate::weights::Qwen35Model;

pub(crate) struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

/// Bind this thread to `model`'s device and context, then initialize cuBLAS.
/// `role` names the thread kind in failure messages ("scheduler", "TP worker");
/// every thread that runs model work must hold the returned guard.
pub(crate) fn bind_model_thread(model: &Qwen35Model, role: &str) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 {role} thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 {role} thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(CublasThreadGuard)
}
