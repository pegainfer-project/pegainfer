// The whole crate is gated: Qwen3.5 needs the Triton AOT kernels from
// `pegainfer-kernels/qwen35`, which need Python + Triton at build time.
// Without the feature this compiles to an empty crate so plain workspace
// builds stay Python-free.
#![cfg(feature = "qwen35")]

mod batch_decode;
pub(crate) mod batch_decode_graph;
pub(crate) mod config;
mod decode_buffers;
mod executor;
mod ffi;
mod logprobs;
pub mod model_line;
mod ops;
mod prefill;
pub mod prefill_buffers;
pub(crate) mod recurrent;
pub(crate) mod recurrent_state;
mod scheduler;
#[cfg(test)]
#[path = "../tests/common/model_fixture.rs"]
mod test_fixture;
mod tp_executor;
mod unified_forward;
mod weights;

use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
pub(crate) use config::probe_config_json;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EpBackend;
pub use scheduler::DEFAULT_MAX_PREFILL_TOKENS;

/// Maximum supported Qwen3.5 decode scheduler slots.
const MAX_DECODE_BATCH: usize = batch_decode_graph::MAX_BATCH;

/// Current safe Qwen3.5 Shared-SM overlap admission cap.
///
/// Decode buckets above this value fall back to the prefill cuBLAS handle today.
/// Replaying decode while async prefill concurrently calls that same handle from
/// another stream relies on undocumented single-handle multi-stream behavior.
/// Keep overlap opt-in below this boundary until the larger decode buckets have
/// an independent per-stream handle/workspace route.
pub const MAX_SHARED_SM_DECODE_BATCH: usize = ops::GEMM_LT_MAX_N;
const _: () = assert!(MAX_SHARED_SM_DECODE_BATCH <= ops::GEMM_LT_MAX_N);

/// Low-level Qwen3.5 execution interface.
///
/// This is for model-local tests, debugging, and benchmarks. The root server
/// should use `start_engine` instead.
pub mod runtime {
    pub use crate::executor::DecodePlan;
    pub use crate::executor::DecodeRequestResult;
    pub use crate::executor::DecodeResult;
    pub use crate::executor::DecodeStepItem;
    pub use crate::executor::PrefillPlan;
    pub use crate::executor::PrefillRequestResult;
    pub use crate::executor::PrefillResult;
    pub use crate::executor::PrefillStepItem;
    pub use crate::executor::Qwen35Executor;
    pub use crate::executor::RequestId;
    pub use crate::scheduler::start_with_capacity;
    pub use crate::tp_executor::DropExpectation;
    pub use crate::tp_executor::Qwen35TpExecutor;
    pub use crate::weights::Qwen35Model;
}

/// Public operator surface used by Qwen3.5-local benches.
pub mod runtime_ops {
    pub use crate::ops::gated_delta_rule_prefill_chunkwise_into;
    pub use crate::ops::rms_norm_batch_offset_into;
    pub use crate::ops::rms_norm_offset_into;
}

/// Scheduler policy for balancing Qwen3.5 prefill work against active decode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen35SchedulerPolicy {
    /// Preserve the fixed chunked-prefill behavior.
    #[default]
    Off,
    /// Adapt per scheduler tick based on active decode and prefill pressure.
    Auto,
}

/// How Qwen3.5 handles a prefill chunk that arrives while decode is active.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen35DecodeOverlap {
    /// Preserve the serial Unified prefill-then-decode path.
    #[default]
    Off,
    /// Run prefill on a second CUDA stream while decode uses the model stream.
    SharedSm,
}

pub fn start_engine(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    start_engine_with_capacity_and_policy(
        model_path,
        options,
        max_batch,
        max_prefill_tokens,
        Qwen35SchedulerPolicy::Off,
    )
}

#[derive(Clone, Debug)]
pub struct Qwen35LaunchOptions {
    /// CUDA device for single-GPU loads (ignored when `tp_size > 1`).
    device_ordinal: usize,
    /// Tensor-parallel world size; `> 1` uses devices `0..tp_size`.
    tp_size: usize,
    /// TP Phase 1 supports eager-only multi-GPU execution.
    cuda_graph: bool,
    max_batch: usize,
    max_prefill_tokens: usize,
}

impl Qwen35LaunchOptions {
    fn device_ordinals(&self) -> Result<Vec<usize>> {
        anyhow::ensure!(self.tp_size >= 1, "Qwen3.5 tp_size must be >= 1");
        Ok(if self.tp_size == 1 {
            vec![self.device_ordinal]
        } else {
            (0..self.tp_size).collect()
        })
    }
}

#[allow(clippy::needless_pass_by_value)]
pub fn launch_with_options_policy_and_overlap(
    model_path: &Path,
    options: Qwen35LaunchOptions,
    scheduler_policy: Qwen35SchedulerPolicy,
    decode_overlap: Qwen35DecodeOverlap,
) -> Result<EngineHandle> {
    let device_ordinals = options.device_ordinals()?;
    start_engine_with_capacity_policy_and_overlap(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            device_ordinals,
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
        options.max_batch,
        options.max_prefill_tokens,
        scheduler_policy,
        decode_overlap,
    )
}

pub fn start_engine_with_capacity(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<EngineHandle> {
    start_engine_with_capacity_and_policy(
        model_path,
        options,
        max_batch,
        max_prefill_tokens,
        Qwen35SchedulerPolicy::Off,
    )
}

pub(crate) fn start_engine_with_capacity_and_policy(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
) -> Result<EngineHandle> {
    start_engine_with_capacity_policy_and_overlap(
        model_path,
        options,
        max_batch,
        max_prefill_tokens,
        scheduler_policy,
        Qwen35DecodeOverlap::Off,
    )
}

pub fn start_engine_with_capacity_policy_and_overlap(
    model_path: &Path,
    options: EngineLoadOptions,
    max_batch: usize,
    max_prefill_tokens: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    decode_overlap: Qwen35DecodeOverlap,
) -> Result<EngineHandle> {
    anyhow::ensure!(
        (1..=MAX_DECODE_BATCH).contains(&max_batch),
        "Qwen3.5 max_batch must be in 1..={MAX_DECODE_BATCH}, got {max_batch}"
    );
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    anyhow::ensure!(
        scheduler_policy == Qwen35SchedulerPolicy::Off
            || decode_overlap == Qwen35DecodeOverlap::Off,
        "Qwen3.5 --decode-overlap=stream requires --qwen35-scheduler-policy=off"
    );
    if decode_overlap == Qwen35DecodeOverlap::SharedSm {
        anyhow::ensure!(
            max_batch <= MAX_SHARED_SM_DECODE_BATCH,
            "Qwen3.5 --decode-overlap=stream currently requires --max-batch <= {MAX_SHARED_SM_DECODE_BATCH}; larger decode buckets still fall back to the prefill GEMM handle"
        );
    }
    if device_ordinals.len() > 1 {
        anyhow::ensure!(
            decode_overlap == Qwen35DecodeOverlap::Off,
            "Qwen3.5 decode overlap is supported on a single GPU only"
        );
        if scheduler_policy == Qwen35SchedulerPolicy::Auto {
            return Err(anyhow!(
                "Qwen3.5 TP uses the fixed off scheduler policy; --qwen35-scheduler-policy=auto is single-GPU only"
            ));
        }
        if enable_cuda_graph {
            return Err(anyhow!(
                "Qwen3.5 TP Phase 1 supports eager execution only; disable CUDA Graph"
            ));
        }
        let model_path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path must be valid UTF-8"))?;
        return scheduler::start_tp_with_capacity(
            model_path,
            seed,
            &device_ordinals,
            max_batch,
            max_prefill_tokens,
        );
    }

    anyhow::ensure!(
        enable_cuda_graph,
        "Qwen3.5 decode always captures CUDA Graphs; --cuda-graph=false is not supported"
    );
    let device_ordinal = match device_ordinals.as_slice() {
        [] => 0,
        [device_ordinal] => *device_ordinal,
        ordinals => {
            return Err(anyhow!(
                "Qwen3.5 engine supports exactly one CUDA device, got {}",
                ordinals.len()
            ));
        }
    };
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow!("model path must be valid UTF-8"))?;
    let model = weights::Qwen35Model::from_safetensors(model_path, device_ordinal, max_batch)?;
    scheduler::start_with_capacity_and_policy(
        model,
        seed,
        max_batch,
        max_prefill_tokens,
        scheduler_policy,
        decode_overlap,
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pegainfer_frontend::engine::EngineLoadOptions;
    use pegainfer_frontend::engine::EpBackend;

    use super::MAX_SHARED_SM_DECODE_BATCH;
    use super::Qwen35DecodeOverlap;
    use super::Qwen35LaunchOptions;
    use super::Qwen35SchedulerPolicy;
    use super::start_engine_with_capacity_and_policy;
    use super::start_engine_with_capacity_policy_and_overlap;

    #[test]
    fn launch_options_reject_zero_tp_size() {
        let options = Qwen35LaunchOptions {
            device_ordinal: 3,
            tp_size: 0,
            cuda_graph: false,
            max_batch: 1,
            max_prefill_tokens: 1,
        };

        let err = options.device_ordinals().unwrap_err().to_string();
        assert!(err.contains("tp_size must be >= 1"));
    }

    #[test]
    fn scheduler_policy_defaults_to_off() {
        assert_eq!(Qwen35SchedulerPolicy::default(), Qwen35SchedulerPolicy::Off);
    }

    #[test]
    fn tp_rejects_auto_scheduler_policy_before_loading_model() {
        let err = start_engine_with_capacity_and_policy(
            Path::new("unused-model-path"),
            EngineLoadOptions {
                enable_cuda_graph: false,
                device_ordinals: vec![0, 1],
                parallel_config: None,
                ep_backend: EpBackend::Nccl,
                seed: 42,
            },
            1,
            1,
            Qwen35SchedulerPolicy::Auto,
        )
        .err()
        .expect("scheduler policy validation should reject TP launch")
        .to_string();

        assert!(err.contains("scheduler policy"));
        assert!(err.contains("single-GPU only"));
    }

    #[test]
    fn tp_rejects_decode_overlap_before_loading_model() {
        let err = start_engine_with_capacity_policy_and_overlap(
            Path::new("unused-model-path"),
            EngineLoadOptions {
                enable_cuda_graph: false,
                device_ordinals: vec![0, 1],
                parallel_config: None,
                ep_backend: EpBackend::Nccl,
                seed: 42,
            },
            1,
            1,
            Qwen35SchedulerPolicy::Off,
            Qwen35DecodeOverlap::SharedSm,
        )
        .err()
        .expect("decode-overlap validation should reject TP launch")
        .to_string();

        assert!(err.contains("single GPU"));
    }

    #[test]
    fn auto_policy_rejects_decode_overlap_before_loading_model() {
        let err = start_engine_with_capacity_policy_and_overlap(
            Path::new("unused-model-path"),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                parallel_config: None,
                ep_backend: EpBackend::Nccl,
                seed: 42,
            },
            1,
            1,
            Qwen35SchedulerPolicy::Auto,
            Qwen35DecodeOverlap::SharedSm,
        )
        .err()
        .expect("decode-overlap validation should reject auto policy")
        .to_string();

        assert!(err.contains("scheduler-policy=off"));
    }

    #[test]
    fn shared_sm_rejects_unsafe_decode_bucket_before_loading_model() {
        let err = start_engine_with_capacity_policy_and_overlap(
            Path::new("unused-model-path"),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                parallel_config: None,
                ep_backend: EpBackend::Nccl,
                seed: 42,
            },
            MAX_SHARED_SM_DECODE_BATCH + 1,
            1,
            Qwen35SchedulerPolicy::Off,
            Qwen35DecodeOverlap::SharedSm,
        )
        .err()
        .expect("decode-overlap validation should reject unsafe decode bucket")
        .to_string();

        assert!(err.contains("max-batch"));
        assert!(err.contains(&MAX_SHARED_SM_DECODE_BATCH.to_string()));
    }
}
