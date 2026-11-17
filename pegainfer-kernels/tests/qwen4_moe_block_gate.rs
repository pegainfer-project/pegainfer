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
//! * the gate_up leg's fused SiLU-mul epilogue — the same bound against the
//!   per-expert cuBLAS loop plus the interleaved SiLU kernel, on the interleaved
//!   bank the engine holds. Two sharper cases sit beside it: one-hot activation
//!   rows make both sides' f32 accumulators identical (a single bf16 times a
//!   bf16 is exact in f32), so the fused output must be **bitwise** equal to the
//!   reference — which pins the pair-to-column mapping, the halved store
//!   addressing and every rounding the epilogue does — and the interleave kernel
//!   is held bitwise against a host permutation of the same rows.
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
use pegainfer_kernels::ops::qwen4_moe_gate_up_interleave_into;
use pegainfer_kernels::ops::qwen4_moe_grouped_gate_up_silu_into;
use pegainfer_kernels::ops::qwen4_moe_grouped_gemm_into;
use pegainfer_kernels::ops::qwen4_moe_route_meta_into;
use pegainfer_kernels::ops::silu_mul_fused_interleaved_batch_into;
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

/// The engine's gate_up bank: the checkpoint's split halves permuted to
/// `[g0, u0, g1, u1, ...]` within each expert's `n`-row slab. This is the device
/// interleave's reference — the same permutation, done on the host one cycle at
/// a time (carry the displaced row forward, swap it into the slot that owns it).
fn interleave_host_rows(data: &mut [half::bf16], experts: usize, n: usize, k: usize) {
    assert_eq!(data.len(), experts * n * k);
    let half = n / 2;
    // Where the row at `dst` comes from, and where the row at `src` goes.
    let src_of = |dst: usize| {
        if dst.is_multiple_of(2) {
            dst / 2
        } else {
            half + dst / 2
        }
    };
    let dst_of = |src: usize| {
        if src < half {
            2 * src
        } else {
            2 * (src - half) + 1
        }
    };
    let mut staging = vec![half::bf16::from_f32(0.0); k];
    let mut moved = vec![false; n];
    for expert in 0..experts {
        let base = expert * n * k;
        moved.fill(false);
        for start in 0..n {
            if moved[start] {
                continue;
            }
            staging.copy_from_slice(&data[base + start * k..base + (start + 1) * k]);
            let mut dst = dst_of(start);
            while dst != start {
                for c in 0..k {
                    core::mem::swap(&mut staging[c], &mut data[base + dst * k + c]);
                }
                moved[dst] = true;
                dst = dst_of(dst);
            }
            data[base + start * k..base + (start + 1) * k].copy_from_slice(&staging);
            moved[start] = true;
        }
        debug_assert!((0..n).all(|row| src_of(row) < n));
    }
}

/// `fill` permuted by [`interleave_host_rows`]: the bank an engine holds.
fn interleaved_fill(seed: u64, experts: usize, n: usize, k: usize) -> Vec<half::bf16> {
    let mut data = common::fill(seed, experts * n * k);
    interleave_host_rows(&mut data, experts, n, k);
    data
}

/// Exact equality on the bf16 grid: `to_host` widens bf16 to f32, which is
/// injective, so equal f32 is equal bf16. Bit comparison keeps `-0` and `+0`
/// apart, which is right — both sides run the same arithmetic.
fn assert_bitwise(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(
        want.len(),
        got.len(),
        "{what}: compared runs differ in length"
    );
    for (i, (x, y)) in want.iter().zip(got).enumerate() {
        assert!(
            !x.is_nan() && !y.is_nan(),
            "{what}: element {i} is not finite ({x} / {y})"
        );
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: element {i}, {x} against {y}"
        );
    }
}

/// The fused gate_up leg: per-expert cuBLAS into the interleaved bank plus the
/// interleaved SiLU kernel is the reference; the grouped launch with the SiLU
/// folded into its epilogue has to stay inside the same bound the plain grouped
/// GEMM holds, because the epilogue reproduces the reference's roundings and the
/// only difference left is the accumulation order.
fn check_gate_up_silu_leg(ctx: &DeviceContext, g: &Qwen4MoeGeometry, tokens: usize, k: usize) {
    let half = 64;
    let n = 2 * half;
    let pairs = tokens * TOP_K;
    let routing = route(ctx, g, tokens);
    let mut x = HiddenStates::zeros(ctx, k, pairs).unwrap();
    {
        let rows = HiddenStates::from_host(ctx, &common::fill(11, tokens * k), k, tokens).unwrap();
        gather_hidden_tokens_into(ctx, &rows, &routing.gather, pairs, &mut x).unwrap();
    }
    let bank =
        DeviceMatrix::from_host(ctx, &interleaved_fill(13, EXPERTS, n, k), EXPERTS * n, k).unwrap();
    let mut grouped = Qwen4GroupedGemmScratch::new(ctx, g).unwrap();

    // The reference: the fallback path's two launches, per-expert GEMM into the
    // interleaved `[pairs, 2 * I]` slab then the interleaved SiLU.
    let gate_up = per_expert(ctx, g, &bank, n, &x, &routing.indptr_host);
    let mut want = HiddenStates::zeros(ctx, half, pairs).unwrap();
    silu_mul_fused_interleaved_batch_into(ctx, &gate_up, &mut want).unwrap();
    drop(gate_up);

    let mut got = HiddenStates::zeros(ctx, half, pairs).unwrap();
    qwen4_moe_grouped_gate_up_silu_into(
        ctx,
        g,
        &x,
        &bank,
        half,
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
        "{tokens} tokens, {n}x{k} silu: |diff| {} > {} (2^-6 of the largest {})",
        dev.max_abs,
        MAX_ABS_REL * dev.max_ref,
        dev.max_ref
    );
    assert!(
        dev.rms_rel <= RMS_REL,
        "{tokens} tokens, {n}x{k} silu: rms relative deviation {} > {RMS_REL}",
        dev.rms_rel
    );

    // Same control as the plain leg: a rotated bank must fall outside the bound.
    // Release this leg's bank first, as at the real geometry the two do not fit.
    drop(bank);
    let rotated =
        DeviceMatrix::from_host(ctx, &interleaved_fill(17, EXPERTS, n, k), EXPERTS * n, k).unwrap();
    let control = per_expert(ctx, g, &rotated, n, &x, &routing.indptr_host);
    let mut control_act = HiddenStates::zeros(ctx, half, pairs).unwrap();
    silu_mul_fused_interleaved_batch_into(ctx, &control, &mut control_act).unwrap();
    print!(
        "grouped {tokens:5} tokens {n}x{k} silu: max|diff| {:.3e} of {:.3e}, rms {:.2e}",
        dev.max_abs, dev.max_ref, dev.rms_rel
    );
    drop(control);
    let control = deviation(&control_act.to_host(ctx).unwrap(), &got);
    assert!(
        control.rms_rel > CONTROL_RMS_REL,
        "{tokens} tokens, {n}x{k} silu: the control only reaches {} rms relative, so the bound \
         cannot distinguish a wrong weight bank",
        control.rms_rel
    );
    println!(" (control {:.2e})", control.rms_rel);
}

#[test]
fn grouped_gemm_stays_inside_its_bound() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // The down leg (K = intermediate) at a narrowed width, at the one-token,
    // dispatch-boundary and prefill token counts, and the fused gate_up leg
    // (K = hidden) whose SiLU rides in the epilogue.
    let g = geometry(256);
    for tokens in [1usize, 4, 40, 300] {
        check_grouped_leg(&ctx, &g, tokens, 256, 64);
        check_gate_up_silu_leg(&ctx, &g, tokens, 256);
    }
}

/// One-hot activation rows: `x[row, 0] = 1` and zeros elsewhere, so every
/// output element is a single bf16 product of the weight row. That product is
/// exact in f32, both sides' accumulators are therefore identical, and the fused
/// output has to match the reference **bit for bit** — which is what pins the
/// pair-to-column mapping, the halved store addressing, the epilogue's 4-of-8
/// store and both intermediate bf16 roundings (the product is not bf16
/// representable, so skipping the gate/up rounding shows up here immediately).
#[test]
fn fused_gate_up_matches_the_reference_bitwise_on_one_hot_rows() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let g = geometry(256);
    let (k, half) = (256usize, 64usize);
    let n = 2 * half;
    for tokens in [1usize, 8, 40] {
        let pairs = tokens * TOP_K;
        let routing = route(&ctx, &g, tokens);
        let mut x_host = vec![half::bf16::from_f32(0.0); pairs * k];
        for row in 0..pairs {
            x_host[row * k] = half::bf16::from_f32(1.0);
        }
        let x = HiddenStates::from_host(&ctx, &x_host, k, pairs).unwrap();
        let bank =
            DeviceMatrix::from_host(&ctx, &interleaved_fill(23, EXPERTS, n, k), EXPERTS * n, k)
                .unwrap();
        let gate_up = per_expert(&ctx, &g, &bank, n, &x, &routing.indptr_host);
        let mut want = HiddenStates::zeros(&ctx, half, pairs).unwrap();
        silu_mul_fused_interleaved_batch_into(&ctx, &gate_up, &mut want).unwrap();
        let mut got = HiddenStates::zeros(&ctx, half, pairs).unwrap();
        let mut grouped = Qwen4GroupedGemmScratch::new(&ctx, &g).unwrap();
        qwen4_moe_grouped_gate_up_silu_into(
            &ctx,
            &g,
            &x,
            &bank,
            half,
            &routing.indptr,
            &mut grouped,
            &mut got,
        )
        .unwrap();
        assert_bitwise(
            &want.to_host(&ctx).unwrap(),
            &got.to_host(&ctx).unwrap(),
            &format!("{tokens} tokens, {n}x{k} silu, one-hot"),
        );
        println!("fused {tokens:5} tokens {n}x{k} silu, one-hot: bitwise equal");
    }
}

/// The interleave the engine applies at load: the device kernel against the host
/// permutation of the same rows, bitwise, with the source left untouched.
#[test]
fn gate_up_interleave_matches_the_host_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // A couple of experts wide (the kernel is per-expert inside), and a width
    // that is a whole number of 16 B vectors.
    for (experts, n, k) in [(1usize, 128usize, 256usize), (3, 128, 64), (2, 1280, 64)] {
        let source = common::fill(29, experts * n * k);
        let want_host = {
            let mut data = source.clone();
            interleave_host_rows(&mut data, experts, n, k);
            data
        };
        let src = DeviceMatrix::from_host(&ctx, &source, experts * n, k).unwrap();
        let mut dst = DeviceMatrix::from_host(
            &ctx,
            &vec![half::bf16::NAN; experts * n * k],
            experts * n,
            k,
        )
        .unwrap();
        qwen4_moe_gate_up_interleave_into(&ctx, &src, experts, n / 2, &mut dst).unwrap();
        let untouched: Vec<half::bf16> = ctx.stream.clone_dtoh(&src.data).unwrap();
        let got: Vec<half::bf16> = ctx.stream.clone_dtoh(&dst.data).unwrap();
        for (i, ((t, s), (g, w))) in untouched
            .iter()
            .zip(&source)
            .zip(got.iter().zip(&want_host))
            .enumerate()
        {
            assert_eq!(
                t.to_bits(),
                s.to_bits(),
                "{experts} experts {n}x{k}: source element {i}"
            );
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{experts} experts {n}x{k}: interleaved element {i}, {} against {}",
                g.to_f32(),
                w.to_f32()
            );
        }
        println!("interleave {experts} experts {n}x{k}: bitwise equal");
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
        check_gate_up_silu_leg(&ctx, &g, tokens, 2560);
    }
    for tokens in [1usize, 8, 2048] {
        check_grouped_leg(&ctx, &g, tokens, 2560, 640);
    }
}
