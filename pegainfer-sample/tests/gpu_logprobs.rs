//! GPU-vs-host A/B: `token_logprobs_batch` must match `token_logprob_from_row`
//! on the same bf16 rows — top-k ids exactly, values within `TOL`. Needs a GPU.

use half::bf16;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::StreamOverrideGuard;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::token_logprob_from_row;
use pegainfer_sample::token_logprobs_batch;

const TOL: f32 = 5e-5;

fn make_arena(ctx: &DeviceContext, rows: &[Vec<bf16>]) -> HiddenStates {
    let vocab = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == vocab), "ragged rows");
    let mut hs = HiddenStates::zeros(ctx, vocab, rows.len()).unwrap();
    let flat: Vec<bf16> = rows.iter().flatten().copied().collect();
    ctx.stream.memcpy_htod(&flat, &mut hs.data).unwrap();
    ctx.sync().unwrap();
    hs
}

/// Deterministic bf16 row quantized to 1/128 steps, so ties recur throughout.
fn noise_row(vocab: usize, salt: u64) -> Vec<bf16> {
    (0..vocab as u64)
        .map(|v| {
            let mut z = v.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            bf16::from_f32(((z % 4096) as f32) / 128.0 - 16.0)
        })
        .collect()
}

fn assert_matches_host(row: &[bf16], got: &TokenLogprob, picked: u32, top_k: usize) {
    let want = token_logprob_from_row(row, picked, top_k).unwrap();
    assert_eq!(got.rank, want.rank, "selected vocabulary rank diverged");
    assert!(
        (got.logprob - want.logprob).abs() <= TOL,
        "picked logprob diverged: got {}, want {} (picked={picked}, k={top_k})",
        got.logprob,
        want.logprob
    );
    let got_ids: Vec<u32> = got.top_logprobs.iter().map(|&(id, _)| id).collect();
    let want_ids: Vec<u32> = want.top_logprobs.iter().map(|&(id, _)| id).collect();
    assert_eq!(got_ids, want_ids, "top-k id sequence diverged (k={top_k})");
    for (&(id, got_lp), &(_, want_lp)) in got.top_logprobs.iter().zip(&want.top_logprobs) {
        assert!(
            (got_lp - want_lp).abs() <= TOL,
            "top-k logprob diverged at id {id}: got {got_lp}, want {want_lp}"
        );
    }
}

#[test]
fn matches_host_reference_across_k_at_model_vocab() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let rows: Vec<Vec<bf16>> = (0..3).map(|r| noise_row(vocab, 0x00C0_FFEE + r)).collect();
    let arena = make_arena(&ctx, &rows);

    let requests = [
        LogprobRequest {
            row: 0,
            picked: 7,
            top_k: 0,
        },
        LogprobRequest {
            row: 1,
            picked: 151_935,
            top_k: 1,
        },
        LogprobRequest {
            row: 2,
            picked: 42_000,
            top_k: 5,
        },
        LogprobRequest {
            row: 0,
            picked: 0,
            top_k: 20,
        },
    ];
    let got = token_logprobs_batch(&ctx, &arena, &requests).unwrap();

    assert_eq!(got.len(), requests.len());
    for (req, out) in requests.iter().zip(&got) {
        assert_matches_host(&rows[req.row], out, req.picked, req.top_k);
    }
}

#[test]
fn tie_ordering_matches_host() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 4096;
    // Two exactly-tied tiers straddling the tested k boundaries.
    let mut row = vec![bf16::from_f32(0.0); vocab];
    for &peak in &[901usize, 77, 2048] {
        row[peak] = bf16::from_f32(8.0);
    }
    for &second in &[3000usize, 15, 1999, 512] {
        row[second] = bf16::from_f32(6.5);
    }
    let rows = vec![row];
    let arena = make_arena(&ctx, &rows);

    for top_k in [2, 3, 5, 7] {
        let got = token_logprobs_batch(
            &ctx,
            &arena,
            &[LogprobRequest {
                row: 0,
                picked: 901,
                top_k,
            }],
        )
        .unwrap();
        assert_matches_host(&rows[0], &got[0], 901, top_k);
    }
    let got = token_logprobs_batch(
        &ctx,
        &arena,
        &[LogprobRequest {
            row: 0,
            picked: 901,
            top_k: 5,
        }],
    )
    .unwrap();
    let ids: Vec<u32> = got[0].top_logprobs.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, vec![77, 901, 2048, 15, 512]);
    assert_eq!(
        got[0].rank, 3,
        "all three tied peaks count toward selected rank"
    );

    // Adjacent bf16 logits whose f32 `- lse` shifts collapse to one value.
    let collapse = vec![
        bf16::from_bits(0x3480),
        bf16::from_bits(0x3481),
        bf16::from_bits(0xBC24),
    ];
    let collapse_arena = make_arena(&ctx, std::slice::from_ref(&collapse));
    let got = token_logprobs_batch(
        &ctx,
        &collapse_arena,
        &[LogprobRequest {
            row: 0,
            picked: 0,
            top_k: 3,
        }],
    )
    .unwrap();
    let ids: Vec<u32> = got[0].top_logprobs.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, vec![1, 0, 2]);
    assert_matches_host(&collapse, &got[0], 0, 3);
}

#[test]
fn indexes_arena_rows_out_of_order() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 8192;
    let rows: Vec<Vec<bf16>> = (0..4).map(|r| noise_row(vocab, 0xBEEF + r)).collect();
    let arena = make_arena(&ctx, &rows);

    let requests = [
        LogprobRequest {
            row: 3,
            picked: 11,
            top_k: 4,
        },
        LogprobRequest {
            row: 1,
            picked: 8000,
            top_k: 2,
        },
        LogprobRequest {
            row: 3,
            picked: 500,
            top_k: 1,
        },
    ];
    let got = token_logprobs_batch(&ctx, &arena, &requests).unwrap();
    for (req, out) in requests.iter().zip(&got) {
        assert_matches_host(&rows[req.row], out, req.picked, req.top_k);
    }
}

#[test]
fn k_larger_than_vocab_is_clamped() {
    let ctx = DeviceContext::new().unwrap();
    let rows = vec![noise_row(8, 7)];
    let arena = make_arena(&ctx, &rows);

    let got = token_logprobs_batch(
        &ctx,
        &arena,
        &[LogprobRequest {
            row: 0,
            picked: 3,
            top_k: 32,
        }],
    )
    .unwrap();
    assert_matches_host(&rows[0], &got[0], 3, 32);
}

#[test]
fn rejects_out_of_range_requests() {
    let ctx = DeviceContext::new().unwrap();
    let arena = make_arena(&ctx, &[noise_row(64, 1), noise_row(64, 2)]);

    assert!(
        token_logprobs_batch(
            &ctx,
            &arena,
            &[LogprobRequest {
                row: 0,
                picked: 64,
                top_k: 1
            }]
        )
        .is_err(),
        "picked token beyond the vocab must be rejected"
    );
    assert!(
        token_logprobs_batch(
            &ctx,
            &arena,
            &[LogprobRequest {
                row: 2,
                picked: 0,
                top_k: 1
            }]
        )
        .is_err(),
        "row beyond the arena must be rejected"
    );
    assert!(
        token_logprobs_batch(&ctx, &arena, &[]).unwrap().is_empty(),
        "an empty request list is a no-op"
    );

    let mut inflated = make_arena(&ctx, &[noise_row(64, 4)]);
    inflated.seq_len = 4;
    assert!(
        token_logprobs_batch(
            &ctx,
            &inflated,
            &[LogprobRequest {
                row: 3,
                picked: 0,
                top_k: 1
            }]
        )
        .is_err(),
        "seq_len beyond the arena backing must be rejected"
    );
}

#[test]
fn rejects_active_stream_override() {
    let ctx = DeviceContext::new().unwrap();
    let arena = make_arena(&ctx, &[noise_row(64, 3)]);
    let request = [LogprobRequest {
        row: 0,
        picked: 1,
        top_k: 1,
    }];

    let guard = unsafe { StreamOverrideGuard::activate(ctx.stream.cu_stream()) };
    let denied = token_logprobs_batch(&ctx, &arena, &request);
    drop(guard);

    assert!(
        denied.is_err(),
        "an active stream override must be rejected"
    );
}

#[test]
fn outputs_are_independent_of_batch_composition() {
    let ctx = DeviceContext::new().unwrap();
    let vocab = 151_936;
    let target = noise_row(vocab, 0xA11CE);
    let alone_arena = make_arena(&ctx, std::slice::from_ref(&target));
    let packed_arena = make_arena(&ctx, &[noise_row(vocab, 1), noise_row(vocab, 2), target]);

    let alone = token_logprobs_batch(
        &ctx,
        &alone_arena,
        &[LogprobRequest {
            row: 0,
            picked: 1234,
            top_k: 5,
        }],
    )
    .unwrap();
    let packed = token_logprobs_batch(
        &ctx,
        &packed_arena,
        &[
            LogprobRequest {
                row: 0,
                picked: 9,
                top_k: 3,
            },
            LogprobRequest {
                row: 2,
                picked: 1234,
                top_k: 5,
            },
            LogprobRequest {
                row: 1,
                picked: 77,
                top_k: 1,
            },
        ],
    )
    .unwrap();

    assert_eq!(alone[0].logprob.to_bits(), packed[1].logprob.to_bits());
    let alone_bits: Vec<(u32, u32)> = alone[0]
        .top_logprobs
        .iter()
        .map(|&(id, lp)| (id, lp.to_bits()))
        .collect();
    let packed_bits: Vec<(u32, u32)> = packed[1]
        .top_logprobs
        .iter()
        .map(|&(id, lp)| (id, lp.to_bits()))
        .collect();
    assert_eq!(alone_bits, packed_bits);
}
