//! DFlash2's context window is fixed at the anchor for the entire draft block.
//! Every query sees that context and all draft slots, including future slots.

mod common;

use half::bf16;
use pegainfer_kernels::ops::single_prefill_nhd_noncausal_range_into;
use pegainfer_kernels::tensor::HiddenStates;

#[test]
fn anchor_window_and_full_draft_block_match_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let q_heads = 4;
    let kv_heads = 2;
    let head_dim = 128;
    let query_rows = 3;
    let capacity = 10;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let q = HiddenStates::zeros(&ctx, q_width, query_rows + 2).unwrap();
    let k = HiddenStates::zeros(&ctx, kv_width, capacity).unwrap();
    // Zero Q/K makes softmax uniform: the oracle is the mean of visible V.
    // Large values outside the live range catch offset/extent mistakes.
    let row_values = [
        128.0, 128.0, 128.0, 4.0, 8.0, 16.0, 32.0, 64.0, 256.0, 256.0,
    ];
    let values: Vec<bf16> = (0..capacity * kv_width)
        .map(|i| {
            let row = i / kv_width;
            let head = (i % kv_width) / head_dim;
            let dim = i % head_dim;
            bf16::from_f32(row_values[row] + head as f32 * 128.0 + dim as f32 / 128.0)
        })
        .collect();
    let v = HiddenStates::from_host(&ctx, &values, kv_width, capacity).unwrap();
    for kv_start in [0, 3, 5] {
        let mut output = HiddenStates::from_host(
            &ctx,
            &vec![bf16::from_f32(-32.0); (query_rows + 2) * q_width],
            q_width,
            query_rows + 2,
        )
        .unwrap();
        single_prefill_nhd_noncausal_range_into(
            &ctx,
            &q,
            1,
            query_rows,
            &k,
            &v,
            &mut output,
            q_heads,
            kv_heads,
            head_dim,
            kv_start..8,
        )
        .unwrap();
        let actual = output.to_host(&ctx).unwrap();
        for row in 0..query_rows + 2 {
            for head in 0..q_heads {
                for dim in 0..head_dim {
                    let expected = if row == 0 || row == query_rows + 1 {
                        -32.0
                    } else {
                        (kv_start..8)
                            .map(|kv| values[kv * kv_width + (head / 2) * head_dim + dim].to_f32())
                            .sum::<f32>()
                            / (8 - kv_start) as f32
                    };
                    let got = actual[row * q_width + head * head_dim + dim];
                    assert!(
                        (got - expected).abs() <= expected.abs().max(1.0) / 128.0,
                        "start={kv_start} row={row} head={head} dim={dim}: {got} != {expected}"
                    );
                }
            }
        }
    }
}
