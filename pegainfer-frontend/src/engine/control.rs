//! LoRA adapter control — an optional engine capability, deliberately
//! outside the scheduler contract.
//!
//! An engine that manages adapters mints the pair itself before spawning its
//! scheduler ([`LoraClient::channel`]), lets the scheduler capture the
//! receiver (drained inside `step`, applied when the batch is idle), and
//! returns the client on [`super::Engine::lora`]. The `Option` is the
//! capability: an engine without one cannot be asked, so no "unsupported"
//! error exists. The vocabulary lives here in the frontend layer — the
//! consumer of the client defines the words — keeping the frontend free of
//! model-crate types.

use std::path::PathBuf;

use tokio::sync::oneshot;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadLoraAdapterRequest {
    pub lora_name: String,
    pub lora_path: PathBuf,
    pub load_inplace: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnloadLoraAdapterRequest {
    pub lora_name: String,
    pub lora_int_id: Option<i64>,
}

/// One command to the serving engine, carrying its reply slot. A dropped
/// reply (the engine tore down before applying) surfaces as
/// [`LoraControlError::EngineGone`] on the client.
pub enum LoraControl {
    Load {
        request: LoadLoraAdapterRequest,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Unload {
        request: UnloadLoraAdapterRequest,
        reply: oneshot::Sender<Result<(), String>>,
    },
    List {
        reply: oneshot::Sender<Vec<String>>,
    },
}

/// The engine's end: captured by the scheduler before spawn, drained inside
/// `step`.
pub type LoraControlReceiver = crossbeam_channel::Receiver<LoraControl>;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoraControlError {
    /// The engine (or its reply) went away before answering.
    #[error("engine is gone")]
    EngineGone,
    #[error("LoRA control operation failed: {0}")]
    Failed(String),
}

/// Async client for an engine's LoRA control channel. Rides
/// [`super::Engine::lora`] from launch to whoever serves the adapter routes.
#[derive(Clone)]
pub struct LoraClient {
    tx: crossbeam_channel::Sender<LoraControl>,
}

impl LoraClient {
    /// Mint both ends of the channel.
    #[must_use]
    pub fn channel() -> (Self, LoraControlReceiver) {
        let (tx, rx) = crossbeam_channel::unbounded();
        (Self { tx }, rx)
    }

    pub async fn load(&self, request: LoadLoraAdapterRequest) -> Result<(), LoraControlError> {
        let (reply, rx) = oneshot::channel();
        self.roundtrip(LoraControl::Load { request, reply }, rx)
            .await?
            .map_err(LoraControlError::Failed)
    }

    pub async fn unload(&self, request: UnloadLoraAdapterRequest) -> Result<(), LoraControlError> {
        let (reply, rx) = oneshot::channel();
        self.roundtrip(LoraControl::Unload { request, reply }, rx)
            .await?
            .map_err(LoraControlError::Failed)
    }

    pub async fn list(&self) -> Result<Vec<String>, LoraControlError> {
        let (reply, rx) = oneshot::channel();
        self.roundtrip(LoraControl::List { reply }, rx).await
    }

    async fn roundtrip<T>(
        &self,
        control: LoraControl,
        rx: oneshot::Receiver<T>,
    ) -> Result<T, LoraControlError> {
        self.tx
            .send(control)
            .map_err(|_| LoraControlError::EngineGone)?;
        rx.await.map_err(|_| LoraControlError::EngineGone)
    }
}
