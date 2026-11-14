//! Qwen3.8-Flash-Next MoE block: the prefill paths against the ones they
//! replaced.
//!
//! Two seams on deterministic synthetic inputs:
//!
//! * route metadata — the tiled histogram/scan/placement's layout is pinned
//!   directly: offsets ascend to the valid pair count, one slot per valid pair,
//!   `gather` matching each pair's token, and each expert's range holding only
//!   its own pairs in ascending order. That admits exactly one layout, on both
//!   sides of the dispatch threshold.
//! * expert GEMM — one grouped CUTLASS launch per leg against the per-expert
//!   cuBLAS loop. A **bound**, not bitwise: both accumulate over K in f32, but
//!   CUTLASS's 32-wide K tiles and per-tile epilogue order the roundings
//!   differently, so the two sums land on opposite sides of a bf16 rounding
//!   boundary often enough that single elements differ by one bf16 quantum.
//!   The bound is the quantum plus the reference's own rounding:
//!   `max|diff| <= 2^-6 * max|want|` — two bf16 quanta at the largest magnitude
//!   present, since one quantum is at most `2^-7` of it (one for the rounding
//!   flip, one of margin) — and `||diff||_2 <= 5e-3 * ||want||_2`, whose
//!   measurement `RMS_REL` carries. The control below shows a wrong weight bank
//!   is two orders past both.
//!
//! The default cases run at the frozen expert count and top-k (they are
//! compiled into the routing kernels) with a narrowed hidden/intermediate, so
//! the suite needs a few hundred MB of weights. The real 2560/640 geometry
//! needs one 3.4 GB expert bank per leg (5 GB across both) and rides behind
//! `#[ignore]`:
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_moe_block_gate -- --nocapture
//!   PEGAINFER_REQUIRE_GPU=1 cargo test --release -p pegainfer-kernels \
//!     --features qwen4 --test qwen4_moe_block_gate -- --ignored --nocapture
//!
//! Without a device the suite skips; `PEGAINFER_REQUIRE_GPU=1` makes that a
//! failure.

#![cfg(feature = "qwen4")]

mod common;

use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::Qwen4GroupedGemmScratch;
use pegainfer_kernels::ops::Qwen4MoeGeometry;
use pegainfer_kernels::ops::Qwen4RouteMetaScratch;
use pegainfer_kernels::ops::gather_hidden_tokens_into;
use pegainfer_kernels::ops::gemm_rows_span_into_checked;
use pegainfer_kernels::ops::qwen4_moe_grouped_gemm_into;
use pegainfer_kernels::ops::qwen4_moe_route_meta_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

/// Frozen with the line: `qwen4_moe_route.cu` and the tiled route-metadata
/// kernels are compiled for this expert count and top-k.
const EXPERTS: usize = 512;
const TOP_K: usize = 10;

/// The narrowed widths these kernels read: only `hidden` shapes the grouped
/// GEMM, the route metadata and the combine; the shared branch is not under
/// test here.
fn geometry(hidden: usize) -> Qwen4MoeGeometry {
    Qwen4MoeGeometry {
        hidden,
        experts: EXPERTS,
        top_k: TOP_K,
        shared_intermediate: 64,
    }
}

/// Deterministic router ids: a hash over the pair index, plus one out-of-range
/// id every 97 pairs (both signs) so the `-1` slot path is exercised. Experts
/// repeat across pairs, and some experts are hit once per token.
fn ids_data(tokens: usize, seed: u64) -> Vec<i32> {
    let mut state = seed | 1;
    (0..tokens * TOP_K)
        .map(|p| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if p % 97 == 5 {
                -1
            } else if p % 97 == 6 {
                EXPERTS as i32 + 3
            } else {
                ((state >> 20) % EXPERTS as u64) as i32
            }
        })
        .collect()
}

struct Routing {
    indptr: CudaSlice<i32>,
    gather: CudaSlice<i32>,
    indptr_host: Vec<i32>,
}

fn route(ctx: &DeviceContext, g: &Qwen4MoeGeometry, tokens: usize) -> Routing {
    let pairs = tokens * TOP_K;
    let ids = ctx.stream.clone_htod(&ids_data(tokens, 7)).unwrap();
    let mut indptr: CudaSlice<i32> = ctx.stream.alloc_zeros(EXPERTS + 1).unwrap();
    let mut gather: CudaSlice<i32> = ctx.stream.alloc_zeros(pairs).unwrap();
    let mut slot: CudaSlice<i32> = ctx.stream.alloc_zeros(pairs).unwrap();
    let mut scratch = Qwen4RouteMetaScratch::new(ctx, g, tokens).unwrap();
    qwen4_moe_route_meta_into(
        ctx,
        g,
        &ids,
        tokens,
        &mut indptr,
        &mut gather,
        &mut slot,
        &mut scratch,
    )
    .unwrap();
    let indptr_host = ctx.stream.clone_dtoh(&indptr).unwrap();
    Routing {
        indptr,
        gather,
        indptr_host,
    }
}

#[test]
fn route_meta_layout_is_pinned() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let g = geometry(256);
    // 25 and 26 tokens straddle the tiled threshold (250/260 pairs); below it
    // the dispatched entry runs the sequential kernel, above it the tiled path
    // (32/64/200/2048 tokens are 2, 3, 8 and 80 chunks).
    for tokens in [1usize, 3, 10, 25, 26, 32, 64, 200, 2048] {
        let pairs = tokens * TOP_K;
        let ids = ctx.stream.clone_htod(&ids_data(tokens, 11)).unwrap();
        let mut got = (
            ctx.stream.alloc_zeros(EXPERTS + 1).unwrap(),
            ctx.stream.alloc_zeros(pairs).unwrap(),
            ctx.stream.alloc_zeros(pairs).unwrap(),
        );
        let mut scratch = Qwen4RouteMetaScratch::new(&ctx, &g, tokens).unwrap();
        qwen4_moe_route_meta_into(
            &ctx,
            &g,
            &ids,
            tokens,
            &mut got.0,
            &mut got.1,
            &mut got.2,
            &mut scratch,
        )
        .unwrap();
        let indptr: Vec<i32> = ctx.stream.clone_dtoh(&got.0).unwrap();
        let gather: Vec<i32> = ctx.stream.clone_dtoh(&got.1).unwrap();
        let slot: Vec<i32> = ctx.stream.clone_dtoh(&got.2).unwrap();
        // The layout itself: offsets ascend, the last is the valid pair count,
        // every valid pair has its own slot, and each expert's slots keep
        // ascending pair order. Together these admit exactly one answer.
        let ids = ids_data(tokens, 11);
        assert_eq!(indptr[0], 0);
        assert!(indptr.windows(2).all(|w| w[0] <= w[1]));
        let valid = slot.iter().filter(|s| **s >= 0).count();
        assert_eq!(indptr[EXPERTS] as usize, valid);
        let mut pair_of_slot = vec![usize::MAX; pairs];
        for (pair, at) in slot.iter().enumerate() {
            match *at {
                -1 => assert!(
                    ids[pair] < 0 || ids[pair] >= EXPERTS as i32,
                    "{tokens} tokens: pair {pair} has no slot but a valid id"
                ),
                at => {
                    assert_eq!(
                        pair_of_slot[at as usize],
                        usize::MAX,
                        "slot {at} claimed twice"
                    );
                    pair_of_slot[at as usize] = pair;
                    assert_eq!(gather[at as usize] as usize, pair / TOP_K);
                }
            }
        }
        for e in 0..EXPERTS {
            let (first, end) = (indptr[e] as usize, indptr[e + 1] as usize);
            assert!(end <= pairs);
            assert!(
                pair_of_slot[first..end].windows(2).all(|w| w[0] < w[1]),
                "{tokens} tokens: expert {e}'s slots are not in pair order"
            );
            assert!(
                pair_of_slot[first..end]
                    .iter()
                    .all(|p| *p != usize::MAX && ids[*p] == e as i32),
                "{tokens} tokens: expert {e}'s slot range holds another expert's pair"
            );
        }
    }
}

/// The per-expert reference: one `gemm_rows_span` per expert over its slot
/// range, exactly the loop the production prefill path ran before the grouped
/// launch.
fn per_expert(
    ctx: &DeviceContext,
    g: &Qwen4MoeGeometry,
    weight: &DeviceMatrix,
    n: usize,
    x: &HiddenStates,
    indptr_host: &[i32],
) -> HiddenStates {
    let mut out = HiddenStates::zeros(ctx, n, x.seq_len).unwrap();
    for e in 0..g.experts {
        let (first, end) = (indptr_host[e] as usize, indptr_host[e + 1] as usize);
        gemm_rows_span_into_checked(
            ctx,
            weight,
            e * n,
            n,
            x,
            first,
            &mut out,
            first,
            end - first,
        )
        .unwrap();
    }
    out
}

/// `max|diff| / max|want|`: one bf16 quantum is at most `2^-7` of the magnitude
/// it rounds, so the bound is two quanta — one for a rounding flip, one of
/// margin.
const MAX_ABS_REL: f32 = 1.0 / 64.0;
/// `||diff||_2 / ||want||_2`: dominated by the reference's own rounding, not the
/// grouped kernel's. Against an f64 oracle the grouped output stays at the bf16
/// store floor at every span, while cuBLAS at 32..128 rows per expert — where a
/// 2048-token prefill lands — stores 40% of elements one quantum off: rms 2.7e-3
/// on sm_89, 1.6e-4 on sm_120. Twice the worst measured.
const RMS_REL: f64 = 5e-3;

/// A control must be two orders past the bound, or the bound cannot tell a
/// right grouping from a wrong one.
const CONTROL_RMS_REL: f64 = 1e-2;

struct Deviation {
    max_abs: f32,
    max_ref: f32,
    rms_rel: f64,
}

fn deviation(want: &[f32], got: &[f32]) -> Deviation {
    assert_eq!(want.len(), got.len(), "compared runs differ in length");
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let (mut sq_diff, mut sq_ref) = (0.0f64, 0.0f64);
    for (x, y) in want.iter().zip(got) {
        assert!(x.is_finite() && y.is_finite(), "{x} / {y}");
        max_abs = max_abs.max((x - y).abs());
        max_ref = max_ref.max(x.abs());
        sq_diff += f64::from(x - y) * f64::from(x - y);
        sq_ref += f64::from(*x) * f64::from(*x);
    }
    Deviation {
        max_abs,
        max_ref,
        rms_rel: (sq_diff / sq_ref).sqrt(),
    }
}

fn check_grouped_leg(ctx: &DeviceContext, g: &Qwen4MoeGeometry, tokens: usize, n: usize, k: usize) {
    let pairs = tokens * TOP_K;
    let routing = route(ctx, g, tokens);
    let mut x = HiddenStates::zeros(ctx, k, pairs).unwrap();
    {
        let rows = HiddenStates::from_host(ctx, &common::fill(11, tokens * k), k, tokens).unwrap();
        gather_hidden_tokens_into(ctx, &rows, &routing.gather, pairs, &mut x).unwrap();
    }
    let weight =
        DeviceMatrix::from_host(ctx, &common::fill(13, EXPERTS * n * k), EXPERTS * n, k).unwrap();
    let mut grouped = Qwen4GroupedGemmScratch::new(ctx, g).unwrap();
    let want = per_expert(ctx, g, &weight, n, &x, &routing.indptr_host);
    let mut got = HiddenStates::zeros(ctx, n, pairs).unwrap();
    qwen4_moe_grouped_gemm_into(
        ctx,
        g,
        &x,
        &weight,
        n,
        &routing.indptr,
        &mut grouped,
        &mut got,
    )
    .unwrap();
    let want = want.to_host(ctx).unwrap();
    let got = got.to_host(ctx).unwrap();
    let dev = deviation(&want, &got);
    assert!(
        dev.max_abs <= MAX_ABS_REL * dev.max_ref,
        "{tokens} tokens, {n}x{k}: |diff| {} > {} (2^-6 of the largest {})",
        dev.max_abs,
        MAX_ABS_REL * dev.max_ref,
        dev.max_ref
    );
    assert!(
        dev.rms_rel <= RMS_REL,
        "{tokens} tokens, {n}x{k}: rms relative deviation {} > {RMS_REL}",
        dev.rms_rel
    );

    // The control: the per-expert reference run against a different weight
    // bank. If this stays inside the bound, the bound proves nothing about the
    // grouping. It is a second full bank, so release the leg's own before
    // allocating it — at the real geometry the two do not fit an 8 GB card.
    drop(weight);
    let rotated =
        DeviceMatrix::from_host(ctx, &common::fill(17, EXPERTS * n * k), EXPERTS * n, k).unwrap();
    let control = per_expert(ctx, g, &rotated, n, &x, &routing.indptr_host);
    let control = deviation(&control.to_host(ctx).unwrap(), &got);
    assert!(
        control.rms_rel > CONTROL_RMS_REL,
        "{tokens} tokens, {n}x{k}: the control only reaches {} rms relative, so the bound cannot \
         distinguish a wrong weight bank",
        control.rms_rel
    );
    println!(
        "grouped {tokens:5} tokens {n}x{k}: max|diff| {:.3e} of {:.3e}, rms {:.2e} (control {:.2e})",
        dev.max_abs, dev.max_ref, dev.rms_rel, control.rms_rel
    );
}

#[test]
fn grouped_gemm_stays_inside_its_bound() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // gate_up (K = hidden) and down (K = intermediate) at a narrowed width, at
    // the one-token, dispatch-boundary and prefill token counts.
    let g = geometry(256);
    for tokens in [1usize, 4, 40, 300] {
        check_grouped_leg(&ctx, &g, tokens, 2 * 64, 256);
        check_grouped_leg(&ctx, &g, tokens, 256, 64);
    }
}

/// The real block geometry: hidden 2560, 512 experts, top-10, intermediate 640.
/// A 3.4 GB expert bank per leg, so it rides behind `#[ignore]`; `check_grouped_leg`
/// releases the leg's bank before it builds the control's.
#[test]
#[ignore = "3.4 GB of synthetic real-geometry weights per leg"]
fn grouped_gemm_stays_inside_its_bound_at_real_geometry() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let g = Qwen4MoeGeometry::qwen38_flash_next();
    for tokens in [1usize, 4, 64, 2048] {
        check_grouped_leg(&ctx, &g, tokens, 2 * 640, 2560);
    }
    for tokens in [1usize, 8, 2048] {
        check_grouped_leg(&ctx, &g, tokens, 2560, 640);
    }
}
