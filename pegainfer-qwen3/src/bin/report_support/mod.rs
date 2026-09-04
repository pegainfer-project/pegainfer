//! Report-only Qwen3 kernel/model benchmark harness.
//!
//! Owned by the `kernel-report`-gated binaries (`qwen3_kernel_report`,
//! `qwen3_model_report`) — which `#[path]`-include it so the harness is
//! compiled per binary and never as part of the `pegainfer-qwen3` library
//! (see issue #944).
//!
//! `qwen3_kernel_report` takes the whole tree; `qwen3_model_report` needs one
//! helper and includes `common` alone, so the other four modules stay
//! dead-code-checked by the kernel report's build.

pub(crate) mod common;
pub(crate) mod decode_attention;
pub(crate) mod dense;
pub(crate) mod prefill_attention;
pub(crate) mod single_prefill;
