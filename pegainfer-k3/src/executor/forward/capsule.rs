//! Runtime selection of the capsule-vendored external decode kernels.
//!
//! `PEGAINFER_K3_CAPSULE` picks which ops run the vendored vLLM v0.28.0
//! cubins (`pegainfer-kernels/cubin/k3/`) instead of the native kernels:
//! `all`, or a comma list of `topk`, `kda`. Unset or empty = native kernels,
//! byte-identical serving. The flag exists for same-binary A/B; each capsule
//! op has its own numeric gate before it is allowed to default on.

use std::sync::LazyLock;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CapsuleFlags {
    pub(crate) topk: bool,
    pub(crate) kda: bool,
}

static FLAGS: LazyLock<CapsuleFlags> = LazyLock::new(|| {
    let raw = std::env::var("PEGAINFER_K3_CAPSULE").unwrap_or_default();
    let mut flags = CapsuleFlags::default();
    for part in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part {
            "all" => {
                flags.topk = true;
                flags.kda = true;
            }
            "topk" => flags.topk = true,
            "kda" => flags.kda = true,
            other => {
                // Refuse to start on a typo rather than silently serving the
                // wrong kernel set.
                panic!("PEGAINFER_K3_CAPSULE: unknown op {other:?} (expected all, topk, kda)");
            }
        }
    }
    if flags.topk || flags.kda {
        log::info!("K3 capsule kernels enabled: {flags:?}");
    }
    flags
});

pub(crate) fn capsule_flags() -> CapsuleFlags {
    *FLAGS
}
