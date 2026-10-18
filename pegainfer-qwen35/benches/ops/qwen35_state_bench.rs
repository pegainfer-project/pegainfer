use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_qwen35::prefill_buffers::GdrChunkwiseScratch35;
use pegainfer_qwen35::runtime_ops as ops;

use super::common::QWEN35_4B_LINEAR_K_DIM;
use super::common::QWEN35_4B_LINEAR_K_HEADS;
use super::common::QWEN35_4B_LINEAR_V_DIM;
use super::common::QWEN35_4B_LINEAR_V_HEADS;
use super::common::configure_group;
use super::common::f32_slice;
use super::common::hidden_states;
use super::common::iter_sync;
use super::common::positive_device_vec;
use super::common::zero_f32_slice;

pub(crate) fn bench_qwen35_state_ops(c: &mut Criterion) {
    // Qwen3.5-4B linear attention: q=16×128, k=16×128, v=32×128
    let conv_channels = QWEN35_4B_LINEAR_K_HEADS * QWEN35_4B_LINEAR_K_DIM * 2
        + QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM;

    let mut group = c.benchmark_group("ops_qwen35_state");
    configure_group(&mut group);

    for &seq_len in &[128usize, 512, 2048] {
        group.throughput(Throughput::Elements(
            (QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM * seq_len) as u64,
        ));
        group.bench_function(
            BenchmarkId::new("gated_delta_rule_prefill_chunkwise_into", seq_len),
            |b| {
                let ctx = DeviceContext::new().expect("failed to create CUDA context");
                let qkv =
                    hidden_states(&ctx, conv_channels, seq_len).expect("failed to allocate qkv");
                let b_proj = hidden_states(&ctx, QWEN35_4B_LINEAR_V_HEADS, seq_len)
                    .expect("failed to allocate b_proj");
                let a_proj = hidden_states(&ctx, QWEN35_4B_LINEAR_V_HEADS, seq_len)
                    .expect("failed to allocate a_proj");
                let dt_bias = positive_device_vec(&ctx, QWEN35_4B_LINEAR_V_HEADS)
                    .expect("failed to allocate dt_bias");
                let a_log =
                    f32_slice(&ctx, QWEN35_4B_LINEAR_V_HEADS).expect("failed to allocate a_log");
                let mut state = zero_f32_slice(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_K_DIM * QWEN35_4B_LINEAR_V_DIM,
                )
                .expect("failed to allocate recurrent state");
                let mut recurrent_out = pegainfer_core::tensor::HiddenStates::zeros(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM,
                    seq_len,
                )
                .expect("failed to allocate recurrent out");
                let mut scratch = GdrChunkwiseScratch35::from_dims(
                    &ctx,
                    QWEN35_4B_LINEAR_V_HEADS,
                    QWEN35_4B_LINEAR_K_DIM,
                    QWEN35_4B_LINEAR_V_DIM,
                    seq_len,
                )
                .expect("failed to allocate chunkwise scratch");
                iter_sync(b, &ctx, || {
                    ops::gated_delta_rule_prefill_chunkwise_into(
                        &ctx,
                        &qkv,
                        &b_proj,
                        &a_proj,
                        &dt_bias,
                        &a_log,
                        &mut state,
                        &mut scratch,
                        &mut recurrent_out,
                        QWEN35_4B_LINEAR_K_HEADS,
                        QWEN35_4B_LINEAR_V_HEADS,
                        QWEN35_4B_LINEAR_K_DIM,
                        QWEN35_4B_LINEAR_V_DIM,
                    )
                    .expect("gated_delta_rule_prefill_chunkwise_into failed");
                });
            },
        );
    }

    group.finish();
}
