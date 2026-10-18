use std::env;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_core::tensor::DeviceContext;

use crate::nccl_backend::NaiveNcclEp2Backend;

const EP_BACKEND_ENV: &str = "PEGAINFER_DSV2_LITE_EP_BACKEND";
const HOST_STAGED_BACKEND: &str = "host-staged";
pub(super) const NCCL_BACKEND: &str = "nccl";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EpBackendKind {
    HostStaged,
    Nccl,
}

impl EpBackendKind {
    fn from_env() -> Result<Self> {
        let raw = env::var(EP_BACKEND_ENV).ok();
        parse_backend(raw.as_deref())
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::HostStaged => HOST_STAGED_BACKEND,
            Self::Nccl => NCCL_BACKEND,
        }
    }
}

pub(super) enum EpBackendRuntime {
    HostStaged,
    Nccl(Box<NaiveNcclEp2Backend>),
}

impl EpBackendRuntime {
    pub(super) fn new(
        kind: EpBackendKind,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
    ) -> Result<Self> {
        match kind {
            EpBackendKind::HostStaged => Ok(Self::HostStaged),
            EpBackendKind::Nccl => Ok(Self::Nccl(Box::new(NaiveNcclEp2Backend::new(
                rank0, rank1,
            )?))),
        }
    }

    pub(super) fn kind(&self) -> EpBackendKind {
        match self {
            Self::HostStaged => EpBackendKind::HostStaged,
            Self::Nccl(_) => EpBackendKind::Nccl,
        }
    }
}

pub(super) fn parse_backend(raw: Option<&str>) -> Result<EpBackendKind> {
    match raw.unwrap_or(HOST_STAGED_BACKEND) {
        HOST_STAGED_BACKEND => Ok(EpBackendKind::HostStaged),
        NCCL_BACKEND => Ok(EpBackendKind::Nccl),
        other => bail!(
            "DeepSeek-V2-Lite EP=2 backend '{other}' is not supported; supported backends: {HOST_STAGED_BACKEND}, {NCCL_BACKEND}"
        ),
    }
}
pub(super) fn validate_backend_and_devices(device_ordinals: &[usize]) -> Result<EpBackendKind> {
    ensure!(
        device_ordinals.len() == 2,
        "DeepSeek-V2-Lite first EP gate supports exactly 2 CUDA devices for ep_size=2, got {}",
        device_ordinals.len()
    );
    ensure!(
        device_ordinals[0] != device_ordinals[1],
        "DeepSeek-V2-Lite EP=2 requires two distinct CUDA device ordinals, got {:?}",
        device_ordinals
    );
    EpBackendKind::from_env()
}
