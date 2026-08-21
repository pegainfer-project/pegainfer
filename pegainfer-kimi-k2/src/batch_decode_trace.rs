use anyhow::Result;
use anyhow::ensure;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::EngineLoadOptions;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::EosPolicy;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::GenerateRequest;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::StopPolicy;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::TokenEvent;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::engine::TokenSink;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::KIMI_K2_EXPERT_INTERMEDIATE;
use pegainfer_kernels::ops::KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8;
use pegainfer_kernels::ops::KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8;
use pegainfer_kernels::ops::KIMI_K2_MLA_KV_LORA_RANK;
use pegainfer_kernels::ops::KIMI_K2_MLA_O_LOCAL_IN_TP8;
use pegainfer_kernels::ops::KIMI_K2_MLA_Q_LOCAL_OUT_TP8;
use pegainfer_kernels::ops::KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8;
use pegainfer_kernels::ops::KIMI_K2_MLA_QKV_A_OUT;
use pegainfer_kernels::ops::KIMI_K2_MLA_ROPE_DIM;
use pegainfer_kernels::ops::KIMI_K2_ROUTER_SCALE;
use pegainfer_kernels::tensor::AxisSpec;
use pegainfer_kernels::tensor::Bf16;
use pegainfer_kernels::tensor::Contiguous1D;
use pegainfer_kernels::tensor::F32;
use pegainfer_kernels::tensor::HiddenStatesLayout;
use pegainfer_kernels::tensor::I32;
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kernels::tensor::RowMajor2D;
use pegainfer_kernels::tensor::TensorSpec;
use pegainfer_kernels::tensor::U32;

use crate::config::KIMI_K2_DENSE_INTERMEDIATE;
use crate::config::KIMI_K2_HIDDEN;
use crate::config::KIMI_K2_LAYERS;
use crate::config::KIMI_K2_MOE_LAYERS;
use crate::config::KIMI_K2_Q_LORA_RANK;
use crate::config::KIMI_K2_ROUTED_EXPERTS;
use crate::config::KIMI_K2_TOPK;
use crate::config::KIMI_K2_VOCAB;

pub const MODEL: &str = "kimi-k2";
pub const PHASE_DECODE: &str = "decode";
pub const TP_WORLD_SIZE: usize = 8;
const EP_WORLD_SIZE: usize = 8;
const DENSE_LAYERS: usize = KIMI_K2_LAYERS - KIMI_K2_MOE_LAYERS;
const LOCAL_DENSE_INTERMEDIATE: usize = KIMI_K2_DENSE_INTERMEDIATE / TP_WORLD_SIZE;
const LOCAL_SHARED_INTERMEDIATE: usize = KIMI_K2_EXPERT_INTERMEDIATE / TP_WORLD_SIZE;

pub fn trace_decode_kernel_calls(
    _model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    trace_static_decode_kernel_calls(batch_size, kv_len)
}

fn trace_static_decode_kernel_calls(batch_size: usize, kv_len: usize) -> Result<Vec<KernelCall>> {
    ensure!(batch_size > 0, "batch_size must be greater than zero");
    ensure!(kv_len > 0, "kv_len must be greater than zero");

    let mut calls = Vec::new();
    calls.push(
        KernelCall::new("embedding_batch_vocab_shard", "decode.embedding")
            .input(
                "weight",
                weight(KIMI_K2_VOCAB / TP_WORLD_SIZE, KIMI_K2_HIDDEN),
            )
            .input("token_ids", token_ids(batch_size))
            .output("out", hidden(KIMI_K2_HIDDEN, batch_size)),
    );
    calls.push(all_reduce(
        "decode.embedding_allreduce",
        KIMI_K2_HIDDEN,
        batch_size,
        "bf16_via_f32",
    ));

    for layer_idx in 0..KIMI_K2_LAYERS {
        push_attention_layer(&mut calls, layer_idx, batch_size, kv_len);
        if layer_idx < DENSE_LAYERS {
            push_dense_layer(&mut calls, layer_idx, batch_size);
        } else {
            push_moe_layer(&mut calls, layer_idx, batch_size);
        }
    }

    calls.push(
        KernelCall::new("rms_norm_batch", "decode.final_norm")
            .input("x", hidden(KIMI_K2_HIDDEN, batch_size))
            .input("weight", vector_bf16("hidden", KIMI_K2_HIDDEN))
            .output("out", hidden(KIMI_K2_HIDDEN, batch_size)),
    );
    calls.push(
        KernelCall::new("gemm_graphsafe", "decode.lm_head")
            .input(
                "weight",
                weight(KIMI_K2_VOCAB / TP_WORLD_SIZE, KIMI_K2_HIDDEN),
            )
            .input("x", hidden(KIMI_K2_HIDDEN, batch_size))
            .output("out", hidden(KIMI_K2_VOCAB / TP_WORLD_SIZE, batch_size)),
    );
    calls.push(
        KernelCall::new("top1_batch", "decode.top1")
            .input("logits", hidden(KIMI_K2_VOCAB / TP_WORLD_SIZE, batch_size))
            .output(
                "token_ids",
                TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named("batch", batch_size)]),
            ),
    );

    Ok(calls)
}

#[cfg(feature = "kernel-call-trace")]
pub fn trace_runtime_decode_kernel_calls(
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    ensure!(batch_size > 0, "batch_size must be greater than zero");
    ensure!(kv_len > 1, "runtime decode trace requires --kv-len > 1");

    let prompt_len = kv_len - 1;
    let engine = crate::start_engine(
        std::path::Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: (0..TP_WORLD_SIZE).collect(),
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;
    let ((), calls) = call_trace::collect_result(|| {
        let mut receivers = Vec::with_capacity(batch_size);
        for request_idx in 0..batch_size {
            let (token_tx, mut token_rx) = TokenSink::standalone();
            engine.submit(GenerateRequest {
                trace_parent: None,
                request_id: Some(format!("kimi-trace-{request_idx}")),
                queued_at_unix_s: None,
                data_parallel_rank: None,
                prompt_tokens: vec![0_u32; prompt_len],
                params: SamplingParams {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: None,
                    ignore_eos: true,
                },
                stop_policy: StopPolicy {
                    eos: EosPolicy::Ignore,
                    token_ids: Vec::new(),
                },
                max_tokens: 2,
                lora_adapter: None,
                kv_transfer_params: None,
                token_tx,
                logprobs: 0,
                echo: false,
            })?;
            receivers.push(std::thread::spawn(move || -> Result<()> {
                while let Some((_, event)) = token_rx.blocking_recv() {
                    match event {
                        TokenEvent::Scheduled { .. }
                        | TokenEvent::Token { .. }
                        | TokenEvent::PromptTokens { .. }
                        | TokenEvent::KvTransfer { .. } => {}
                        TokenEvent::Finished { .. } => return Ok(()),
                        TokenEvent::Error { message, .. } => {
                            anyhow::bail!("Kimi runtime trace request failed: {message}")
                        }
                        TokenEvent::Rejected { message, .. } => {
                            anyhow::bail!("Kimi runtime trace request rejected: {message}")
                        }
                    }
                }
                anyhow::bail!("Kimi runtime trace token stream closed before Finished")
            }));
        }
        for receiver in receivers {
            receiver
                .join()
                .map_err(|_| anyhow::anyhow!("Kimi runtime trace receiver thread panicked"))??;
        }
        Ok(())
    })?;
    drop(engine);
    ensure!(
        !calls.is_empty(),
        "Kimi runtime decode trace produced no KernelCall records"
    );
    Ok(calls)
}

pub fn normalize_call_site(label: &str) -> String {
    let Some(rest) = label.strip_prefix('L') else {
        return label.to_string();
    };
    let digit_count = rest
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 || rest.as_bytes().get(digit_count) != Some(&b'.') {
        return label.to_string();
    }
    format!("layer.*{}", &rest[digit_count..])
}

fn push_attention_layer(
    calls: &mut Vec<KernelCall>,
    layer_idx: usize,
    batch: usize,
    kv_len: usize,
) {
    let prefix = format!("L{layer_idx}.attn");
    calls.push(rms(&format!("{prefix}.input_norm"), batch));
    calls.push(gemm(
        &format!("{prefix}.fused_qkv_a"),
        KIMI_K2_MLA_QKV_A_OUT,
        KIMI_K2_HIDDEN,
        batch,
    ));
    calls.push(
        KernelCall::new(
            "kimi_mla_split_qkv_a_norm",
            format!("{prefix}.split_qkv_a_norm"),
        )
        .input("qkv_a", hidden(KIMI_K2_MLA_QKV_A_OUT, batch))
        .output("q_a_normed", hidden(KIMI_K2_Q_LORA_RANK, batch))
        .output(
            "compressed_kv_normed",
            hidden(KIMI_K2_MLA_KV_LORA_RANK, batch),
        )
        .output("k_rope", hidden(KIMI_K2_MLA_ROPE_DIM, batch)),
    );
    calls.push(gemm(
        &format!("{prefix}.q_b"),
        KIMI_K2_MLA_Q_LOCAL_OUT_TP8,
        KIMI_K2_Q_LORA_RANK,
        batch,
    ));
    calls.push(
        KernelCall::new("kimi_mla_rope_split_decode", format!("{prefix}.rope_split"))
            .input("q_proj", hidden(KIMI_K2_MLA_Q_LOCAL_OUT_TP8, batch))
            .input("k_rope", hidden(KIMI_K2_MLA_ROPE_DIM, batch))
            .output(
                "q_nope",
                hidden(
                    KIMI_K2_MLA_Q_LOCAL_OUT_TP8 - KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
                    batch,
                ),
            )
            .output("q_pe", hidden(KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, batch))
            .output("append_kpe", hidden(KIMI_K2_MLA_ROPE_DIM, batch)),
    );
    calls.push(
        KernelCall::new("kimi_mla_absorb_q_nope", format!("{prefix}.absorb_q"))
            .input(
                "q_nope",
                hidden(
                    KIMI_K2_MLA_Q_LOCAL_OUT_TP8 - KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8,
                    batch,
                ),
            )
            .input(
                "kv_b_proj",
                weight(KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK),
            )
            .output("q_abs_nope", hidden(KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch)),
    );
    calls.push(
        KernelCall::new(
            "kimi_mla_paged_kv_append",
            format!("{prefix}.paged_kv_append"),
        )
        .input("compressed_normed", hidden(KIMI_K2_MLA_KV_LORA_RANK, batch))
        .input("append_kpe", hidden(KIMI_K2_MLA_ROPE_DIM, batch))
        .output(
            "ckv_cache",
            paged_cache("ckv", kv_len, batch, KIMI_K2_MLA_KV_LORA_RANK),
        )
        .output(
            "kpe_cache",
            paged_cache("kpe", kv_len, batch, KIMI_K2_MLA_ROPE_DIM),
        ),
    );
    calls.push(
        KernelCall::new(
            "kimi_flashinfer_batch_decode_mla",
            format!("{prefix}.flashinfer_mla"),
        )
        .input("q_abs_nope", hidden(KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch))
        .input("q_pe", hidden(KIMI_K2_MLA_Q_PE_LOCAL_OUT_TP8, batch))
        .input(
            "ckv_cache",
            paged_cache("ckv", kv_len, batch, KIMI_K2_MLA_KV_LORA_RANK),
        )
        .input(
            "kpe_cache",
            paged_cache("kpe", kv_len, batch, KIMI_K2_MLA_ROPE_DIM),
        )
        .output("latent", hidden(KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch))
        .attr("kv_len", kv_len.to_string()),
    );
    calls.push(
        KernelCall::new("kimi_mla_v_up", format!("{prefix}.v_up"))
            .input("latent", hidden(KIMI_K2_MLA_ABS_Q_LOCAL_OUT_TP8, batch))
            .input(
                "kv_b_proj",
                weight(KIMI_K2_MLA_KV_B_LOCAL_OUT_TP8, KIMI_K2_MLA_KV_LORA_RANK),
            )
            .output("out", hidden(KIMI_K2_MLA_O_LOCAL_IN_TP8, batch)),
    );
    calls.push(gemm(
        &format!("{prefix}.o_proj"),
        KIMI_K2_HIDDEN,
        KIMI_K2_MLA_O_LOCAL_IN_TP8,
        batch,
    ));
    calls.push(all_reduce(
        &format!("{prefix}.o_proj_allreduce"),
        KIMI_K2_HIDDEN,
        batch,
        "bf16_via_f32",
    ));
    calls.push(fused_add_rms_round(
        &format!("{prefix}.residual_post_attention_norm"),
        batch,
    ));
}

fn push_dense_layer(calls: &mut Vec<KernelCall>, layer_idx: usize, batch: usize) {
    let prefix = format!("L{layer_idx}.dense");
    calls.push(gemm(
        &format!("{prefix}.gate_up"),
        2 * LOCAL_DENSE_INTERMEDIATE,
        KIMI_K2_HIDDEN,
        batch,
    ));
    calls.push(silu(
        &format!("{prefix}.silu_mul"),
        LOCAL_DENSE_INTERMEDIATE,
        batch,
    ));
    calls.push(gemm(
        &format!("{prefix}.down"),
        KIMI_K2_HIDDEN,
        LOCAL_DENSE_INTERMEDIATE,
        batch,
    ));
    calls.push(all_reduce(
        &format!("{prefix}.down_allreduce"),
        KIMI_K2_HIDDEN,
        batch,
        "bf16_via_f32",
    ));
    calls.push(add(&format!("{prefix}.residual"), batch));
}

fn push_moe_layer(calls: &mut Vec<KernelCall>, layer_idx: usize, batch: usize) {
    let prefix = format!("L{layer_idx}.moe");
    calls.push(gemm(
        &format!("{prefix}.shared_gate_up"),
        2 * LOCAL_SHARED_INTERMEDIATE,
        KIMI_K2_HIDDEN,
        batch,
    ));
    calls.push(silu(
        &format!("{prefix}.shared_silu_mul"),
        LOCAL_SHARED_INTERMEDIATE,
        batch,
    ));
    calls.push(gemm(
        &format!("{prefix}.shared_down"),
        KIMI_K2_HIDDEN,
        LOCAL_SHARED_INTERMEDIATE,
        batch,
    ));
    calls.push(all_reduce(
        &format!("{prefix}.shared_allreduce"),
        KIMI_K2_HIDDEN,
        batch,
        "bf16_via_f32",
    ));
    calls.push(
        KernelCall::new("kimi_router_noaux_tc", format!("{prefix}.router"))
            .input("hidden", hidden(KIMI_K2_HIDDEN, batch))
            .input(
                "gate_weight",
                weight(KIMI_K2_ROUTED_EXPERTS, KIMI_K2_HIDDEN),
            )
            .output(
                "topk_weight",
                TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named(
                    "route",
                    batch * KIMI_K2_TOPK,
                )]),
            )
            .output(
                "topk_idx",
                TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named(
                    "route",
                    batch * KIMI_K2_TOPK,
                )]),
            ),
    );
    calls.push(
        KernelCall::new(
            "kimi_moe_marlin_align_block_size",
            format!("{prefix}.route_align"),
        )
        .input(
            "topk_idx",
            TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named("route", batch * KIMI_K2_TOPK)]),
        )
        .output(
            "sorted_token_ids",
            TensorSpec::new::<I32, Contiguous1D>([AxisSpec::named(
                "route_padded",
                batch * KIMI_K2_TOPK,
            )]),
        )
        .attr("topk", KIMI_K2_TOPK.to_string()),
    );
    calls.push(marlin(
        &format!("{prefix}.marlin_w13"),
        KIMI_K2_HIDDEN,
        2 * KIMI_K2_EXPERT_INTERMEDIATE,
        batch,
    ));
    calls.push(
        KernelCall::new("kimi_marlin_w13_swiglu", format!("{prefix}.routed_swiglu"))
            .input(
                "x",
                hidden(2 * KIMI_K2_EXPERT_INTERMEDIATE, batch * KIMI_K2_TOPK),
            )
            .output(
                "out",
                hidden(KIMI_K2_EXPERT_INTERMEDIATE, batch * KIMI_K2_TOPK),
            ),
    );
    calls.push(marlin(
        &format!("{prefix}.marlin_w2"),
        KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HIDDEN,
        batch,
    ));
    calls.push(
        KernelCall::new(
            "kimi_marlin_sum_topk_rows_f32",
            format!("{prefix}.sum_topk"),
        )
        .input(
            "expert_output",
            hidden(KIMI_K2_HIDDEN, batch * KIMI_K2_TOPK),
        )
        .output(
            "out",
            TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named("elem", batch * KIMI_K2_HIDDEN)]),
        ),
    );
    calls.push(
        KernelCall::new(
            "repeat_f32_for_reduce_scatter",
            format!("{prefix}.routed_repeat_for_rs"),
        )
        .input(
            "local",
            TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named("elem", batch * KIMI_K2_HIDDEN)]),
        )
        .output(
            "global",
            TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named(
                "elem",
                batch * EP_WORLD_SIZE * KIMI_K2_HIDDEN,
            )]),
        ),
    );
    calls.push(reduce_scatter(
        &format!("{prefix}.routed_reduce_scatter"),
        KIMI_K2_HIDDEN,
        batch,
        EP_WORLD_SIZE,
        "f32",
    ));
    calls.push(
        KernelCall::new(
            "kimi_residual_add_scaled_f32",
            format!("{prefix}.residual_add_scaled"),
        )
        .input("hidden", hidden(KIMI_K2_HIDDEN, batch))
        .input("projected", hidden(KIMI_K2_HIDDEN, batch))
        .input(
            "routed_f32",
            TensorSpec::new::<F32, Contiguous1D>([AxisSpec::named("elem", batch * KIMI_K2_HIDDEN)]),
        )
        .output("out", hidden(KIMI_K2_HIDDEN, batch))
        .attr("scale", KIMI_K2_ROUTER_SCALE.to_string()),
    );
}

fn rms(label: &str, batch: usize) -> KernelCall {
    rms_dim(label, KIMI_K2_HIDDEN, batch)
}

fn rms_dim(label: &str, hidden_dim: usize, batch: usize) -> KernelCall {
    KernelCall::new("rms_norm_batch", label)
        .input("x", hidden(hidden_dim, batch))
        .input("weight", vector_bf16("hidden", hidden_dim))
        .output("out", hidden(hidden_dim, batch))
}

fn gemm(label: &str, out_dim: usize, in_dim: usize, batch: usize) -> KernelCall {
    KernelCall::new("gemm_graphsafe", label)
        .input("weight", weight(out_dim, in_dim))
        .input("x", hidden(in_dim, batch))
        .output("out", hidden(out_dim, batch))
}

fn marlin(label: &str, in_dim: usize, out_dim: usize, batch: usize) -> KernelCall {
    KernelCall::new("kimi_marlin_wna16_gemm", label)
        .input("x", hidden(in_dim, batch))
        .output("out", hidden(out_dim, batch * KIMI_K2_TOPK))
        .attr("in_dim", in_dim.to_string())
        .attr("out_dim", out_dim.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
}

fn silu(label: &str, intermediate: usize, batch: usize) -> KernelCall {
    KernelCall::new("silu_mul_batch", label)
        .input("gate", hidden(intermediate, batch))
        .input("up", hidden(intermediate, batch))
        .output("out", hidden(intermediate, batch))
}

fn add(label: &str, batch: usize) -> KernelCall {
    KernelCall::new("add_batch", label)
        .input("a", hidden(KIMI_K2_HIDDEN, batch))
        .input("b", hidden(KIMI_K2_HIDDEN, batch))
        .output("out", hidden(KIMI_K2_HIDDEN, batch))
}

fn fused_add_rms_round(label: &str, batch: usize) -> KernelCall {
    KernelCall::new("fused_add_rms_norm_round_batch", label)
        .input("hidden", hidden(KIMI_K2_HIDDEN, batch))
        .input("residual", hidden(KIMI_K2_HIDDEN, batch))
        .input("weight", vector_bf16("hidden", KIMI_K2_HIDDEN))
        .output("hidden", hidden(KIMI_K2_HIDDEN, batch))
        .output("normed", hidden(KIMI_K2_HIDDEN, batch))
}

fn all_reduce(label: &str, hidden_dim: usize, batch: usize, dtype: &str) -> KernelCall {
    KernelCall::new("all_reduce", label)
        .input(
            "values",
            TensorSpec::named(
                dtype,
                "contiguous_1d",
                [AxisSpec::named("elem", hidden_dim * batch)],
            ),
        )
        .output(
            "values",
            TensorSpec::named(
                dtype,
                "contiguous_1d",
                [AxisSpec::named("elem", hidden_dim * batch)],
            ),
        )
        .attr("dtype", dtype.to_string())
        .attr("tp_world_size", TP_WORLD_SIZE.to_string())
}

fn reduce_scatter(
    label: &str,
    hidden_dim: usize,
    local_batch: usize,
    world_size: usize,
    dtype: &str,
) -> KernelCall {
    KernelCall::new("reduce_scatter", label)
        .input(
            "global",
            TensorSpec::named(
                dtype,
                "contiguous_1d",
                [AxisSpec::named(
                    "elem",
                    hidden_dim * local_batch * world_size,
                )],
            ),
        )
        .output(
            "local",
            TensorSpec::named(
                dtype,
                "contiguous_1d",
                [AxisSpec::named("elem", hidden_dim * local_batch)],
            ),
        )
        .attr("dtype", dtype.to_string())
        .attr("ep_world_size", world_size.to_string())
}

fn hidden(hidden_dim: usize, batch: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, HiddenStatesLayout>([
        AxisSpec::named("hidden", hidden_dim),
        AxisSpec::named("batch", batch),
    ])
}

fn weight(out_dim: usize, in_dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, RowMajor2D>([
        AxisSpec::named("out", out_dim),
        AxisSpec::named("in", in_dim),
    ])
}

fn vector_bf16(axis: &str, len: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, Contiguous1D>([AxisSpec::named(axis, len)])
}

fn token_ids(batch: usize) -> TensorSpec {
    TensorSpec::new::<U32, Contiguous1D>([AxisSpec::named("batch", batch)])
}

fn paged_cache(name: &str, kv_len: usize, batch: usize, dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, Contiguous1D>([
        AxisSpec::named("request", batch),
        AxisSpec::named(name, kv_len),
        AxisSpec::named("dim", dim),
    ])
}
