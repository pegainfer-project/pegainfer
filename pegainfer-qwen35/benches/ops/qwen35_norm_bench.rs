use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_qwen35::runtime_ops as ops;

use super::common::EPS;
use super::common::QWEN35_4B_HIDDEN;
use super::common::configure_group;
use super::common::device_vec;
use super::common::hidden_states;
use super::common::iter_sync;
use super::common::positive_device_vec;

pub(crate) fn bench_qwen35_norm_ops(c: &mut Criterion) {
    let hidden = QWEN35_4B_HIDDEN;

    let mut group = c.benchmark_group("ops_qwen35_norm");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(hidden as u64));
    group.bench_function(BenchmarkId::new("rms_norm_offset_into", hidden), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x = device_vec(&ctx, hidden).expect("failed to allocate x");
        let weight = positive_device_vec(&ctx, hidden).expect("failed to allocate offset weight");
        let mut offset_out = DeviceVec::zeros(&ctx, hidden).expect("failed to allocate offset out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_offset_into(&ctx, &x, &weight, EPS, &mut offset_out)
                .expect("rms_norm_offset_into failed");
        });
    });

    group.finish();

    // ========================================================================
    // Batched RMSNorm_offset: per-token loop (current) vs batched kernel (TODO)
    // Qwen3.5 hidden_dim=2560, realistic seq_len values.
    // ========================================================================
    let hidden_dim = 2560;

    for &seq_len in &[128, 2048] {
        let mut group = c.benchmark_group(format!("rms_norm_offset_batched/seq{seq_len}"));
        configure_group(&mut group);
        group.throughput(Throughput::Elements((hidden_dim * seq_len) as u64));

        group.bench_function("batched_kernel", |b| {
            let ctx = DeviceContext::new().expect("ctx");
            let x = hidden_states(&ctx, hidden_dim, seq_len).expect("x");
            let weight = positive_device_vec(&ctx, hidden_dim).expect("weight");
            let mut out = HiddenStates::zeros(&ctx, hidden_dim, seq_len).expect("out");
            iter_sync(b, &ctx, || {
                ops::rms_norm_batch_offset_into(&ctx, &x, &weight, EPS, &mut out)
                    .expect("batched norm");
            });
        });

        group.finish();
    }
}
