pub struct KernelPlan {
    pub model: &'static str,
    pub phases: &'static [KernelPhase],
}

pub struct KernelPhase {
    pub name: &'static str,
    pub ops: &'static [KernelOp],
}

pub struct KernelOp {
    pub id: &'static str,
    pub rust: &'static str,
    pub backend: &'static str,
    pub notes: &'static str,
}

pub static KERNEL_PLAN: KernelPlan = KernelPlan {
    model: "higgs-audio",
    phases: &[
        KernelPhase {
            name: "artifact",
            ops: &[
                KernelOp {
                    id: "checkpoint_header_gate",
                    rust: "weights::HiggsWeightManifest::from_model_dir",
                    backend: "safetensors header",
                    notes: "validates Higgs checkpoint tensor names, dtypes, and shapes without reading payloads",
                },
                KernelOp {
                    id: "qwen3_alias_plan",
                    rust: "load_plan::HiggsRuntimeLoadPlan::qwen3_tensor_aliases",
                    backend: "metadata",
                    notes: "maps Higgs body.* tensors onto Qwen3 requested tensor names without a 7.5 GiB payload copy",
                },
            ],
        },
        KernelPhase {
            name: "prefill",
            ops: &[
                KernelOp {
                    id: "qwen3_body_prefill",
                    rust: "runtime_bridge::HiggsAudioRuntime::prefill_audio_from_prompt_ids -> Qwen3Executor::prefill_last_hidden_bf16",
                    backend: "Qwen3 runtime: CUDA + cuBLAS + FlashInfer",
                    notes: "runs the Higgs text/body checkpoint through the existing Qwen3 prefill path via tensor-name aliases",
                },
                KernelOp {
                    id: "qwen3_prompt_session_prefill",
                    rust: "runtime_bridge::HiggsAudioRuntime::prefill_prompt_session -> Qwen3Executor::prefill_last_hidden_bf16_retained_prompt",
                    backend: "Qwen3 runtime: CUDA + cuBLAS + FlashInfer + paged KV",
                    notes: "retains prompt KV under a Higgs-owned session handle without registering a generated text token",
                },
                KernelOp {
                    id: "fused_audio_head",
                    rust: "one_step_actual::compute_one_step_audio_prediction_gpu_bf16 -> ops::linear",
                    backend: "CUDA bf16 linear",
                    notes: "projects the final hidden state with tied.embedding.modality_embeddings.0.embedding.weight into 8x1026 audio logits",
                },
                KernelOp {
                    id: "audio_topk_argmax",
                    rust: "one_step_actual::audio_topk_and_argmax",
                    backend: "CPU",
                    notes: "diagnostic one-step gate extracts top-64 and argmax ids from the fused audio logits",
                },
            ],
        },
        KernelPhase {
            name: "golden",
            ops: &[
                KernelOp {
                    id: "strict_comparison",
                    rust: "compare::compare_one_step_files",
                    backend: "CPU",
                    notes: "exact prompt/argmax checks plus absolute drift diagnostics for hidden, logits, and top-64 logprobs",
                },
                KernelOp {
                    id: "semantic_comparison",
                    rust: "compare::compare_one_step_semantic_files",
                    backend: "CPU",
                    notes: "runtime bring-up gate using prompt exactness, argmax exactness, cosine, regret, and top-64 overlap",
                },
            ],
        },
    ],
};

pub fn kernel_plan() -> &'static KernelPlan {
    &KERNEL_PLAN
}

#[cfg(test)]
mod tests {
    use super::kernel_plan;

    #[test]
    fn higgs_kernel_plan_names_current_phases() {
        let phase_names: Vec<_> = kernel_plan()
            .phases
            .iter()
            .map(|phase| phase.name)
            .collect();
        assert_eq!(phase_names, ["artifact", "prefill", "golden"]);
    }

    #[test]
    fn higgs_kernel_plan_records_runtime_backends() {
        let ops: Vec<_> = kernel_plan()
            .phases
            .iter()
            .flat_map(|phase| phase.ops.iter())
            .collect();

        assert!(ops.iter().any(|op| {
            op.id == "qwen3_body_prefill"
                && op.backend == "Qwen3 runtime: CUDA + cuBLAS + FlashInfer"
        }));
        assert!(ops.iter().any(|op| {
            op.id == "qwen3_prompt_session_prefill" && op.notes.contains("retains prompt KV")
        }));
        assert!(
            ops.iter()
                .any(|op| op.id == "fused_audio_head" && op.backend == "CUDA bf16 linear")
        );
        assert!(
            ops.iter()
                .any(|op| op.id == "semantic_comparison" && op.backend == "CPU")
        );
    }

    #[test]
    fn higgs_kernel_plan_keeps_alias_copy_boundary_visible() {
        let alias_op = kernel_plan()
            .phases
            .iter()
            .flat_map(|phase| phase.ops.iter())
            .find(|op| op.id == "qwen3_alias_plan")
            .expect("qwen3 alias op should be in the plan");

        assert!(alias_op.notes.contains("without a 7.5 GiB payload copy"));
    }
}
