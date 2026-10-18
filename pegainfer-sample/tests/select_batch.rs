//! Model-free integration tests for the unified batched sampler.
//!
//! Synthetic logits arenas (no model weights): each test pins down one behaviour
//! the crate adds on top of the raw kernels — greedy/non-greedy routing, per-row
//! token placement, scratch reuse, and the capacity invariant. Raw kernel
//! correctness (top-k/top-p/temperature, philox determinism) is owned by
//! `pegainfer-kernels`; here we check that the door wires rows to the right path
//! and lands each token at the right index. Requires a GPU.

use std::collections::HashSet;

use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_sample::SampleScratch;
use pegainfer_sample::SamplingParams;
use pegainfer_sample::select_batch;

/// Build a logits arena from per-row logits (row `i` is token `i`'s logits).
fn make_arena(ctx: &DeviceContext, rows: &[Vec<f32>]) -> HiddenStates {
    let vocab = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == vocab), "ragged rows");
    let mut hs = HiddenStates::zeros(ctx, vocab, rows.len()).unwrap();
    let flat: Vec<bf16> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&x| bf16::from_f32(x)))
        .collect();
    ctx.stream.memcpy_htod(&flat, &mut hs.data).unwrap();
    ctx.sync().unwrap();
    hs
}

fn refs(params: &[SamplingParams]) -> Vec<&SamplingParams> {
    params.iter().collect()
}

fn greedy() -> SamplingParams {
    SamplingParams::default()
}

fn sampling(temperature: f32, top_k: i32, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        ..SamplingParams::default()
    }
}

/// A one-hot-ish logits row: a single peak above a flat floor.
fn peak_row(vocab: usize, peak: usize, height: f32) -> Vec<f32> {
    let mut r = vec![0.0; vocab];
    r[peak] = height;
    r
}

#[test]
fn greedy_batch_picks_each_rows_argmax() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let peaks = [3usize, 7, 0, 15];
    let rows: Vec<Vec<f32>> = peaks.iter().map(|&p| peak_row(vocab, p, 10.0)).collect();
    let arena = make_arena(&ctx, &rows);
    let params = vec![greedy(); peaks.len()];
    let mut scratch = SampleScratch::new(&ctx, vocab, peaks.len()).unwrap();

    let tokens = select_batch(
        &ctx,
        &arena,
        &refs(&params),
        &vec![0; params.len()],
        0,
        &mut scratch,
    )
    .unwrap();

    assert_eq!(tokens, vec![3, 7, 0, 15]);
}

#[test]
fn top_k_one_routes_through_greedy_path() {
    // High temperature but top_k == 1 => is_greedy() => argmax, not sampling.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let arena = make_arena(&ctx, &[peak_row(vocab, 5, 10.0)]);
    let params = vec![sampling(1.0, 1, 1.0)];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let tokens = select_batch(&ctx, &arena, &refs(&params), &[0], 123, &mut scratch).unwrap();

    assert_eq!(tokens, vec![5]);
}

#[test]
fn mixed_batch_routes_and_places_each_row() {
    // Row 0 greedy, row 1 sampled, row 2 greedy. Row 1's top_p (0.5) sits above
    // the 1/vocab argmax floor so it genuinely takes the sampler path; its dist
    // is near one-hot (peak prob ~1) so the nucleus is the peak and the sample
    // collapses there. Verifies both paths run in one call and each token lands
    // at its own row index.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let rows = vec![
        peak_row(vocab, 2, 12.0),
        peak_row(vocab, 9, 12.0),
        peak_row(vocab, 4, 12.0),
    ];
    let arena = make_arena(&ctx, &rows);
    let params = vec![greedy(), sampling(1.0, -1, 0.5), greedy()];
    let mut scratch = SampleScratch::new(&ctx, vocab, 3).unwrap();

    let tokens = select_batch(&ctx, &arena, &refs(&params), &[0, 0, 0], 7, &mut scratch).unwrap();

    assert_eq!(tokens, vec![2, 9, 4]);
}

#[test]
fn sampling_is_seed_deterministic_and_actually_samples() {
    // Uniform logits: argmax would always pick index 0, so seeing >1 token over
    // many seeds proves the non-greedy path samples; equal seeds proves it is
    // reproducible. Also exercises scratch reuse across many calls.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 8;
    let arena = make_arena(&ctx, &[vec![0.0; vocab]]);
    let params = vec![sampling(1.0, -1, 1.0)];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let a = select_batch(&ctx, &arena, &refs(&params), &[0], 42, &mut scratch).unwrap();
    let b = select_batch(&ctx, &arena, &refs(&params), &[0], 42, &mut scratch).unwrap();
    assert_eq!(a, b, "same seed must be deterministic");

    let mut seen = HashSet::new();
    for s in 0..64u64 {
        seen.insert(select_batch(&ctx, &arena, &refs(&params), &[0], s, &mut scratch).unwrap()[0]);
    }
    assert!(
        seen.len() > 1,
        "uniform sampling should explore multiple tokens, saw {seen:?}"
    );
}

#[test]
fn tiny_top_p_routes_to_argmax_even_under_bf16_ties() {
    // Regression for the per-step-philox migration (no per-request RNG). A tiny
    // top_p must collapse to the same token greedy picks. The hard case is a
    // bf16 tie at the top: argmax breaks it by lowest index, but the rejection
    // sampler would pick either tied token depending on the seed (the old
    // per-request seed masked this; a per-step seed exposes it). Because the
    // softmax max is always >= 1/vocab, top_p <= 1/vocab makes the nucleus a
    // single token, so the row routes to the deterministic argmax path and
    // matches greedy for every seed. Large vocab matches the model's
    // OnlineSoftmax regime.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let lo = 879usize;
    let hi = 6941usize;
    let mut row = vec![0.0f32; vocab];
    row[lo] = 8.0; // bf16-exact, identical to `hi` -> a true top tie
    row[hi] = 8.0;
    let arena = make_arena(&ctx, &[row]);
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let greedy_tok = select_batch(&ctx, &arena, &[&greedy()], &[0], 0, &mut scratch).unwrap();
    assert_eq!(
        greedy_tok,
        vec![lo as u32],
        "argmax breaks the tie by lowest index"
    );

    let tiny = sampling(1.0, -1, 1e-6); // 1e-6 < 1/vocab (~6.6e-6)
    for s in 0..64u64 {
        assert_eq!(
            select_batch(&ctx, &arena, &[&tiny], &[0], s, &mut scratch).unwrap(),
            greedy_tok,
            "seed {s}: tiny top_p must match greedy, not sample the tied peer"
        );
    }
}

#[test]
fn padded_arena_width_does_not_suppress_the_argmax_routing() {
    // qwen35 tile-aligns its logits width past the decodable vocab (#1046). A
    // top_p at exactly 1/decodable collapses to the argmax on the unpadded
    // arena, so it must still do so at twice the arena width.
    let ctx = DeviceContext::new().unwrap();
    let decodable = 256usize;
    let vocab = 2 * decodable;
    let lo = 128usize;
    let hi = 200usize;
    let mut row = vec![-1.0f32; vocab];
    row[lo] = 8.0; // bf16-exact, identical to `hi` -> a true top tie
    row[hi] = 8.0;
    let arena = make_arena(&ctx, &[row]);

    let tiny = sampling(1.0, -1, 1.0 / decodable as f32);

    // Control: at the arena width 1/256 is no longer a single-token nucleus,
    // and the sampler may return the tied peer.
    let mut wide = SampleScratch::new(&ctx, vocab, 1).unwrap();
    let mut sampled_the_peer = false;
    for s in 0..64u64 {
        let picked = select_batch(&ctx, &arena, &[&tiny], &[0], s, &mut wide).unwrap()[0];
        assert!(picked == lo as u32 || picked == hi as u32);
        sampled_the_peer |= picked == hi as u32;
    }
    assert!(
        sampled_the_peer,
        "control: an arena-width routing decision puts this row on the sampler"
    );

    let mut bounded = SampleScratch::with_selection_width(&ctx, vocab, decodable, 1).unwrap();
    for s in 0..64u64 {
        assert_eq!(
            select_batch(&ctx, &arena, &[&tiny], &[0], s, &mut bounded).unwrap(),
            vec![lo as u32],
            "seed {s}: the decodable vocab must keep this row on the argmax path"
        );
    }
}

#[test]
fn batch_larger_than_scratch_is_rejected() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 8;
    let arena = make_arena(&ctx, &[peak_row(vocab, 1, 5.0), peak_row(vocab, 2, 5.0)]);
    let params = vec![greedy(); 2];
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    assert!(
        select_batch(
            &ctx,
            &arena,
            &refs(&params),
            &vec![0; params.len()],
            0,
            &mut scratch
        )
        .is_err()
    );
}

#[test]
fn min_p_row_takes_the_sampler_path_and_filters() {
    // Two-token support: P(3) ~ 0.73, P(11) ~ 0.27. min_p = 0.5 thresholds at
    // 0.5 * 0.73 = 0.37, masking token 11 — every draw must return token 3.
    // With min_p = 0.0 the same row explores both (checked over many seeds),
    // proving the row genuinely rides the sampler, not argmax.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let mut row = vec![-60.0f32; vocab];
    row[3] = 1.0;
    row[11] = 0.0;
    let arena = make_arena(&ctx, &[row]);
    let mut scratch = SampleScratch::new(&ctx, vocab, 1).unwrap();

    let mut filtered = sampling(1.0, -1, 1.0);
    filtered.min_p = 0.5;
    for s in 0..64u64 {
        assert_eq!(
            select_batch(&ctx, &arena, &[&filtered], &[0], s, &mut scratch).unwrap(),
            vec![3],
            "seed {s}: min_p=0.5 must mask the 0.27-prob token"
        );
    }

    let open = sampling(1.0, -1, 1.0);
    let mut seen = HashSet::new();
    for s in 0..128u64 {
        seen.insert(select_batch(&ctx, &arena, &[&open], &[0], s, &mut scratch).unwrap()[0]);
    }
    assert!(
        seen.contains(&11),
        "min_p=0 should reach the 0.27-prob token across 128 seeds, saw {seen:?}"
    );
}

#[test]
fn mixed_batch_keeps_plain_rows_on_the_fast_path() {
    // A min_p neighbor must not perturb plain sampling rows: select_batch
    // batches min_p rows separately, so a plain row's philox subsequence is
    // its index among plain rows and its tokens match a min_p-free batch
    // exactly. The plain rows carry top_k+top_p on a spread distribution so
    // the fused fast path (rejection sampling, several uniform draws) and the
    // min_p pipeline (renorm + one draw) genuinely diverge — riding the wrong
    // path shows up as different tokens, not a 1-ulp coin flip.
    let ctx = DeviceContext::new().unwrap();
    let vocab = 16;
    let spread: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.3).collect();
    let rows: Vec<Vec<f32>> = vec![spread.clone(), spread.clone(), spread];
    let arena_mixed = make_arena(&ctx, &rows);
    let arena_plain = make_arena(&ctx, &rows[..2]);
    let mut scratch = SampleScratch::new(&ctx, vocab, rows.len()).unwrap();

    let plain = sampling(1.0, 4, 0.9);
    let mut minp = sampling(1.0, 0, 1.0);
    minp.min_p = 0.5;

    for seed in 0..64u64 {
        let mixed = select_batch(
            &ctx,
            &arena_mixed,
            &[&plain, &minp, &plain],
            &[0, 0, 0],
            seed,
            &mut scratch,
        )
        .unwrap();
        let alone = select_batch(
            &ctx,
            &arena_plain,
            &[&plain, &plain],
            &[0, 0],
            seed,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(
            [mixed[0], mixed[2]],
            [alone[0], alone[1]],
            "seed {seed}: plain rows changed tokens because a min_p row joined the batch"
        );
    }
}

/// The per-request seed contract: a seeded row's token is a pure function of
/// (seed, step, distribution) — independent of where the row sits in the
/// batch, of its neighbors, and of the engine's per-step seed.
#[test]
fn seeded_rows_replay_independent_of_batch_position() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 32_768;
    let flat = vec![0.0f32; vocab];
    let arena = make_arena(&ctx, &[flat.clone(), flat.clone(), flat]);
    let mut scratch = SampleScratch::new(&ctx, vocab, 3).unwrap();

    let seeded = |seed: u64| SamplingParams {
        temperature: 1.0,
        seed: Some(seed),
        ..SamplingParams::default()
    };
    let unseeded = sampling(1.0, -1, 1.0);

    // Batch A: seeded row last; batch B: seeded row first, different
    // neighbors and engine seed. Same request seed + step => same token.
    let a = select_batch(
        &ctx,
        &arena,
        &[&greedy(), &unseeded, &seeded(7)],
        &[0, 0, 5],
        99,
        &mut scratch,
    )
    .unwrap();
    let b = select_batch(
        &ctx,
        &arena,
        &[&seeded(7), &greedy()],
        &[5, 0],
        1234,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(a[2], b[0], "seeded row must replay across batch layouts");

    // Advancing the step or changing the seed moves the stream; on a flat
    // 32k-token distribution an accidental collision is ~3e-5.
    let c = select_batch(&ctx, &arena, &[&seeded(7)], &[6], 0, &mut scratch).unwrap();
    let d = select_batch(&ctx, &arena, &[&seeded(8)], &[5], 0, &mut scratch).unwrap();
    assert_ne!(a[2], c[0], "step must advance the seeded stream");
    assert_ne!(a[2], d[0], "seed must select a distinct stream");
}

/// Latency micro-benchmark for the verify-round sampler cost (#512): span-17
/// rows at Qwen3 vocab, greedy argmax vs sampled paths. Not a correctness
/// gate — prints per-call microseconds; run with --nocapture.
#[test]
#[ignore = "GPU latency probe — run manually with --nocapture"]
fn select_batch_span17_latency_probe() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let rows = 17;
    let logits: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..vocab)
                .map(|v| (((v * 2_654_435_761 + r * 97) % 1000) as f32) / 50.0)
                .collect()
        })
        .collect();
    let arena = make_arena(&ctx, &logits);
    let mut scratch = SampleScratch::new(&ctx, vocab, rows).unwrap();

    let cases: Vec<(&str, SamplingParams)> = vec![
        ("greedy", SamplingParams::default()),
        (
            "t0.8_topp0.95",
            SamplingParams {
                temperature: 0.8,
                top_p: 0.95,
                ..SamplingParams::default()
            },
        ),
        (
            "t0.7_topk20_topp0.9",
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.9,
                ..SamplingParams::default()
            },
        ),
    ];
    for (name, p) in cases {
        let params = vec![p; rows];
        let param_refs = refs(&params);
        let steps = vec![0u64; rows];
        for _ in 0..20 {
            select_batch(&ctx, &arena, &param_refs, &steps, 7, &mut scratch).unwrap();
        }
        ctx.sync().unwrap();
        let start = std::time::Instant::now();
        let iters = 200;
        for i in 0..iters {
            select_batch(&ctx, &arena, &param_refs, &steps, i as u64, &mut scratch).unwrap();
        }
        ctx.sync().unwrap();
        let us = start.elapsed().as_micros() as f64 / iters as f64;
        eprintln!("select_batch span17 {name}: {us:.1} us/call");
    }
}
