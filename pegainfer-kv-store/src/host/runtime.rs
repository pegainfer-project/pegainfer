use pegaflow_core::EngineError;
use tokio::runtime::Runtime;

// Host construction and teardown can both run inside an async task, where
// Tokio's blocking Runtime destructor would panic.
pub(super) struct HostRuntime(Option<Runtime>);

impl HostRuntime {
    pub(super) fn new(threads: usize) -> Result<Self, EngineError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads.max(1))
            .enable_all()
            .build()
            .map_err(|e| EngineError::Storage(format!("host runtime build: {e}")))?;
        Ok(Self(Some(runtime)))
    }

    pub(super) fn get(&self) -> &Runtime {
        self.0.as_ref().expect("host runtime is present until Drop")
    }
}

impl Drop for HostRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            runtime.shutdown_background();
        }
    }
}
