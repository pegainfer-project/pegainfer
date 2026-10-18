//! Device gate for the GLM5.2 vocabulary-parallel greedy tail (pack/unpack).
//!
//! Single GPU: each simulated rank packs its shard candidate into its own
//! hidden-width carrier; the fixed-order attention all-reduce is emulated by
//! an exact host-side sum (every non-owner slot is +0.0, so the sum equals
//! the packed field bit-for-bit). Covers negative logits, the cross-rank
//! lowest-global-token tie break, a global token id above 65,535, and the
//! all-NaN row degrading to token 0 like the non-sharded argmax.

#![cfg(feature = "glm52")]

use half::bf16;
use pegainfer_kernels::ops::GLM52_TP_HIDDEN;
use pegainfer_kernels::ops::glm52_vocab_parallel_pack_launch;
use pegainfer_kernels::ops::glm52_vocab_parallel_unpack_launch;
use pegainfer_kernels::tensor::DeviceContext;

const RANKS: usize = 4;
const ROWS: usize = 4;
/// GLM5.2 vocab 154,880 / 4 ranks.
const SHARD_ROWS: usize = 38_720;

#[test]
// Golden-value smoke: winning-value comparisons below are meant to be exact,
// not an epsilon tolerance.
#[allow(clippy::float_cmp)]
fn glm52_vocab_parallel_pack_unpack_selects_global_top1() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };

    // Per-rank shard candidates, one per row. Values are exactly
    // representable in bf16 so the cross-rank tie in row 0 is a true tie.
    let nan = f32::NAN;
    #[rustfmt::skip]
    let candidates: [[(f32, usize); ROWS]; RANKS] = [
        // (local top value, local index within this rank's shard)
        [(-3.5, 5),    (1.5, 7),     (-0.5, 0),          (nan, 0)],
        [(-2.25, 100), (0.75, 11),   (-0.25, SHARD_ROWS - 1), (nan, 3)],
        [(-7.0, 9),    (2.5, 3_000), (-4.0, 17),         (nan, 1)],
        [(-2.25, 50),  (2.0, 400),   (-8.0, 4),          (nan, 2)],
    ];
    // row 0: tie at -2.25 between rank 1 (38_720 + 100) and rank 3
    //        (116_160 + 50) — the lower global token id wins.
    // row 1: winner 2.5 on rank 2 at global 77_440 + 3_000 > 65_535.
    // row 2: all-negative winner -0.25 at rank 1's last shard slot.
    // row 3: every candidate is NaN — degrade to token 0.
    let expected_tokens: [i32; ROWS] = [38_820, 80_440, 77_439, 0];
    let expected_values: [f32; ROWS] = [-2.25, 2.5, -0.25, f32::NEG_INFINITY];

    // Emulated fixed-order AR: exact sum of the four packed carriers.
    let mut gathered = vec![0f32; ROWS * GLM52_TP_HIDDEN];
    for (rank, rows) in candidates.iter().enumerate() {
        let values: Vec<bf16> = rows
            .iter()
            .map(|&(value, _)| bf16::from_f32(value))
            .collect();
        let indices: Vec<i32> = rows.iter().map(|&(_, index)| index as i32).collect();
        let values_dev = ctx.stream.clone_htod(&values).expect("values H2D");
        let indices_dev = ctx.stream.clone_htod(&indices).expect("indices H2D");
        let mut partial = ctx
            .stream
            .alloc_zeros::<bf16>(ROWS * GLM52_TP_HIDDEN)
            .expect("carrier alloc");
        glm52_vocab_parallel_pack_launch(
            &ctx,
            &values_dev,
            &indices_dev,
            &mut partial,
            ROWS,
            rank,
            rank * SHARD_ROWS,
        )
        .expect("vocab pack launch");
        let host = ctx.stream.clone_dtoh(&partial).expect("carrier D2H");
        ctx.sync().expect("CUDA sync");
        for (dst, src) in gathered.iter_mut().zip(host) {
            *dst += src.to_f32();
        }
    }

    let gathered_bf16: Vec<bf16> = gathered.iter().map(|&v| bf16::from_f32(v)).collect();
    let gathered_dev = ctx.stream.clone_htod(&gathered_bf16).expect("gathered H2D");
    let mut out_values = ctx.stream.alloc_zeros::<bf16>(ROWS).expect("values alloc");
    let mut out_indices = ctx.stream.alloc_zeros::<i32>(ROWS).expect("indices alloc");
    glm52_vocab_parallel_unpack_launch(
        &ctx,
        &gathered_dev,
        &mut out_values,
        &mut out_indices,
        ROWS,
        RANKS,
    )
    .expect("vocab unpack launch");
    let tokens = ctx.stream.clone_dtoh(&out_indices).expect("indices D2H");
    let values = ctx.stream.clone_dtoh(&out_values).expect("values D2H");
    ctx.sync().expect("CUDA sync");

    assert_eq!(tokens, expected_tokens, "global top-1 token ids");
    for (row, (&got, &want)) in values.iter().zip(&expected_values).enumerate() {
        assert_eq!(
            got.to_f32(),
            want,
            "row {row} winning value (got {got}, want {want})"
        );
    }
}
