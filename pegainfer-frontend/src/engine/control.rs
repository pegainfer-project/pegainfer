use std::path::PathBuf;

use tokio::sync::oneshot;

use super::request::GenerateRequest;

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

pub enum EngineControlRequest {
    LoadLoraAdapter {
        request: LoadLoraAdapterRequest,
        response_tx: oneshot::Sender<std::result::Result<(), String>>,
    },
    UnloadLoraAdapter {
        request: UnloadLoraAdapterRequest,
        response_tx: oneshot::Sender<std::result::Result<(), String>>,
    },
    ListLoraAdapters {
        response_tx: oneshot::Sender<std::result::Result<Vec<String>, String>>,
    },
}

pub enum EngineCommand {
    Generate(Box<GenerateRequest>),
    Control(EngineControlRequest),
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EngineControlError {
    #[error("{0}")]
    Unsupported(&'static str),
    #[error("engine control channel closed")]
    ChannelClosed,
    #[error("engine control operation failed: {0}")]
    OperationFailed(String),
}

pub type EngineControlResult<T> = std::result::Result<T, EngineControlError>;
