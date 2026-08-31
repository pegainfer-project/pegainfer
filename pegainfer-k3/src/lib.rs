//! Kimi K3 model line.
//!
//! Architecture (93 layers, hidden 7168): 69 KDA linear-attention layers
//! (fixed-size recurrent state + 4-tap conv window, replace-in-place) + 24
//! MLA layers (paged KV, NoPE), layer 0 dense; 92 latent-MoE layers
//! (hidden 7168 -> latent 3584 -> 896 routed experts top-16, inter 3072,
//! situ activation -> RMSNorm -> up), 2 shared experts on hidden, plus an
//! attention-residual stream (block size 12). Routed experts are MXFP4
//! (group-32 e8m0 scales, byte-isomorphic to DeepGEMM's FP4 B layout);
//! everything else bf16. No MTP.
//!
//! Serving topology: EP-N decode (dev vehicle: the 224-expert checkpoint
//! at EP4 on one 4-GPU node = 56 experts/rank, shape-isomorphic to the
//! full 896-expert model at EP16). The routed experts are one fused
//! MegaMoE(situ) launch per layer at every world size: dispatch, both
//! FP8xFP4 GEMMs, the situ activation and the weighted combine inside a
//! single persistent kernel that pairs the ranks over NVLink itself, so a
//! step issues no collective at all.
//!
//! KV story: dual-pool — paged KV (kv-store `BlockPool`) for the 24 MLA
//! layers, plus a qwen35-style fixed-size slot pool for KDA recurrent
//! state. Prefix caching ships disabled (KDA state is not recomputable
//! from tokens; see docs/subsystems/kv-cache/design.md, bounded class).

mod config;
pub mod executor;
pub mod model_line;
pub mod scheduler;

pub use executor::K3Executor;
pub use executor::K3ExecutorConfig;
pub use executor::K3MoeTransport;
pub use executor::K3VerifySlot;
pub use executor::cp::K3CpGroup;
pub use executor::ep::K3EpRendezvous;
pub use model_line::MODEL_LINE;

mod model;
mod weights;
pub use scheduler::DecodeSlot;
pub use scheduler::K3Scheduler;
pub use scheduler::K3SchedulerConfig;
pub use scheduler::SlotId;
pub use scheduler::StepExecutor;
pub use scheduler::start_with_executors;
