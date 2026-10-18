//! Microbench for the unified batched sampler over realistic vocab × batch ×
//! param-mix points. Greedy is the batched argmax floor; sampling is the
//! FlashInfer pass; mixed is the common serving case. Requires a GPU.

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_sample::SampleScratch;
use pegainfer_sample::SamplingParams;
use pegainfer_sample::select_batch;

fn arena(ctx: &DeviceContext, vocab: usize, batch: usize) -> HiddenStates {
    let mut hs = HiddenStates::zeros(ctx, vocab, batch).unwrap();
    // Deterministic non-flat logits so top-k/top-p do real work.
    let flat: Vec<bf16> = (0..vocab * batch)
        .map(|i| bf16::from_f32((i.wrapping_mul(2_654_435_761) % 1000) as f32 * 0.01))
        .collect();
    ctx.stream.memcpy_htod(&flat, &mut hs.data).unwrap();
    ctx.sync().unwrap();
    hs
}

fn p(temperature: f32, top_k: i32, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        min_p: 0.0,
        seed: None,
        ignore_eos: false,
    }
}

fn bench_select_batch(c: &mut Criterion) {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let mut group = c.benchmark_group("select_batch");
    for &batch in &[1usize, 8, 32, 64] {
        let a = arena(&ctx, vocab, batch);
        let mut scratch = SampleScratch::new(&ctx, vocab, batch).unwrap();
        let mixes: [(&str, Vec<SamplingParams>); 3] = [
            ("greedy", vec![p(0.0, -1, 1.0); batch]),
            ("sampling", vec![p(1.0, 50, 0.9); batch]),
            (
                "mixed",
                (0..batch)
                    .map(|i| {
                        if i % 2 == 0 {
                            p(0.0, -1, 1.0)
                        } else {
                            p(1.0, 50, 0.9)
                        }
                    })
                    .collect(),
            ),
        ];
        for (name, params) in &mixes {
            let param_refs: Vec<&SamplingParams> = params.iter().collect();
            group.bench_with_input(BenchmarkId::new(*name, batch), &batch, |b, _| {
                let mut seed = 0u64;
                let steps = vec![0u64; param_refs.len()];
                b.iter(|| {
                    seed = seed.wrapping_add(1);
                    select_batch(&ctx, &a, &param_refs, &steps, seed, &mut scratch).unwrap()
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_select_batch);
criterion_main!(benches);
