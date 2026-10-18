mod ops;

use criterion::criterion_group;
use criterion::criterion_main;

criterion_group!(
    benches,
    ops::qwen35_norm_bench::bench_qwen35_norm_ops,
    ops::qwen35_state_bench::bench_qwen35_state_ops
);
criterion_main!(benches);
