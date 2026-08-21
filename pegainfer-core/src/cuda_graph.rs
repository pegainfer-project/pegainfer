use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaContext;
use cudarc::driver::sys::CUgraph;
use cudarc::driver::sys::CUgraphExec;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::sys::{self};
use log::debug;

use crate::tensor::DeviceContext;
use crate::tensor::active_cu_stream;

mod dump;

pub use dump::CudaGraphDumpSummary;
pub use dump::validate_graph_dump_request;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaGraphPhase {
    BeforeBeginCapture,
    AfterBeginCapture,
    BeforeEndCapture,
    AfterEndCapture,
    BeforeLaunch,
    AfterLaunch,
}

/// CUDA Graph state for decode path.
/// First decode call captures the graph; subsequent calls replay it.
///
/// Capture and replay both run on the context's *currently active* stream
/// ([`active_cu_stream`]) — normally `ctx.stream`, but the thread-local stream
/// override used by Green Context SM partitioning redirects them. This matters
/// because stream capture binds each kernel node to the execution context of
/// the stream it was captured on (CUDA Programming Guide §4.6.5): a graph
/// captured on a Green Context decode stream replays on that partition's SMs no
/// matter which stream launches it. A `CudaGraphState` is therefore tied to one
/// stream — a caller that decodes on more than one stream (full-SM and a green
/// partition) must keep one state per stream.
pub struct CudaGraphState {
    graph: CUgraph,
    exec: CUgraphExec,
    /// Keeps the CUDA primary context alive for as long as the graph handles
    /// exist. The raw `CUgraph`/`CUgraphExec` carry no ownership of their own,
    /// so without this anchor the `cuGraphDestroy` in `Drop` would rely on the
    /// enclosing struct happening to declare its `CudaSlice` buffers *before*
    /// this field. `Drop::drop` runs before the struct's fields are dropped, so
    /// holding the `Arc` here guarantees the context outlives the destroy calls
    /// regardless of field order. `None` until the first capture instantiates.
    ctx_anchor: Option<Arc<CudaContext>>,
}

// SAFETY: the graph/exec handles are only ever touched from the single
// inference thread that owns the model.
unsafe impl Send for CudaGraphState {}

fn check(result: sys::CUresult, what: &str) -> Result<()> {
    if result != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("{what} failed: {result:?}");
    }
    Ok(())
}

impl CudaGraphState {
    pub fn new() -> Self {
        Self {
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            ctx_anchor: None,
        }
    }

    /// Whether a graph has been captured and instantiated — i.e. whether
    /// [`Self::launch_captured`] can replay without running the kernel closure.
    pub fn is_captured(&self) -> bool {
        !self.exec.is_null()
    }

    /// Replay the captured graph. Errors if nothing has been captured yet —
    /// callers that may hit an uncaptured shape go through
    /// [`Self::run_or_capture`] instead.
    pub fn launch_captured(&mut self, ctx: &DeviceContext) -> Result<()> {
        anyhow::ensure!(
            self.is_captured(),
            "CUDA graph replay requested before any capture"
        );
        let stream = active_cu_stream(ctx);
        check(
            unsafe { sys::cuGraphLaunch(self.exec, stream) },
            "cuGraphLaunch",
        )
    }

    /// Run kernel closure directly, or capture into a graph and replay.
    ///
    /// `kernels` must be a pure GPU kernel sequence — no CPU-GPU sync, no allocation.
    pub fn run_or_capture<F>(&mut self, ctx: &DeviceContext, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        self.run_or_capture_synchronized(ctx, |_| {}, kernels)
    }

    /// Capture, instantiate, and upload — no launch — so a later
    /// [`Self::launch_captured`] is a pure enqueue (an un-uploaded exec uploads
    /// implicitly on first launch). The upload is stream-ordered: a launch
    /// enqueued after it on the same stream is ordered behind it by the
    /// driver, so only a launch from a *different* stream needs a
    /// synchronization in between.
    pub fn capture_only<F>(&mut self, ctx: &DeviceContext, kernels: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        anyhow::ensure!(
            !self.is_captured(),
            "capture_only on an already-captured graph would leak the live exec"
        );
        let stream = active_cu_stream(ctx);
        self.capture_and_instantiate(ctx, stream, &mut |_| {}, kernels)?;
        check(
            unsafe { sys::cuGraphUpload(self.exec, stream) },
            "cuGraphUpload",
        )
    }

    pub fn run_or_capture_synchronized<F, S>(
        &mut self,
        ctx: &DeviceContext,
        mut synchronize: S,
        kernels: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
        S: FnMut(CudaGraphPhase),
    {
        let stream = active_cu_stream(ctx);

        if !self.exec.is_null() {
            synchronize(CudaGraphPhase::BeforeLaunch);
            check(
                unsafe { sys::cuGraphLaunch(self.exec, stream) },
                "cuGraphLaunch",
            )?;
            synchronize(CudaGraphPhase::AfterLaunch);
            return Ok(());
        }

        self.capture_and_instantiate(ctx, stream, &mut synchronize, kernels)?;

        synchronize(CudaGraphPhase::BeforeLaunch);
        check(
            unsafe { sys::cuGraphLaunch(self.exec, stream) },
            "cuGraphLaunch first launch",
        )?;
        synchronize(CudaGraphPhase::AfterLaunch);
        Ok(())
    }

    fn capture_and_instantiate<F, S>(
        &mut self,
        ctx: &DeviceContext,
        stream: sys::CUstream,
        synchronize: &mut S,
        kernels: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
        S: FnMut(CudaGraphPhase),
    {
        debug!("Capturing CUDA Graph for decode path...");
        synchronize(CudaGraphPhase::BeforeBeginCapture);
        check(
            unsafe { sys::cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        synchronize(CudaGraphPhase::AfterBeginCapture);

        // On kernel error, end the in-progress capture so the stream is not left
        // stuck in the capturing state, then propagate the original error.
        if let Err(e) = kernels() {
            let mut aborted: CUgraph = std::ptr::null_mut();
            unsafe { sys::cuStreamEndCapture(stream, &raw mut aborted) };
            if !aborted.is_null() {
                unsafe { sys::cuGraphDestroy(aborted) };
            }
            return Err(e);
        }

        synchronize(CudaGraphPhase::BeforeEndCapture);
        let mut graph: CUgraph = std::ptr::null_mut();
        check(
            unsafe { sys::cuStreamEndCapture(stream, &raw mut graph) },
            "cuStreamEndCapture",
        )?;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        check(
            unsafe {
                sys::cuGraphInstantiateWithFlags(
                    &raw mut exec,
                    graph,
                    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64,
                )
            },
            "cuGraphInstantiateWithFlags",
        )?;
        self.graph = graph;
        self.exec = exec;
        self.ctx_anchor = Some(ctx.ctx.clone());
        synchronize(CudaGraphPhase::AfterEndCapture);
        debug!("CUDA Graph captured successfully");
        Ok(())
    }
}

impl Default for CudaGraphState {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CudaGraphState {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                sys::cuGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}
