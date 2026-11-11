//! Device gate for the paged draft K/V row copies in
//! csrc/shared/draft_kv.cu: scatter writes a row to the slot the target's page
//! table names, gather reads it back, and one launch covers every buffer of the
//! drafter's K/V set.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure rather
//! than a skip.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::draft_kv_gather_into;
use pegainfer_kernels::ops::draft_kv_scatter_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

/// Small integers only: bf16 represents integers up to 256 exactly, so the
/// assertions below are exact-equality, not tolerance-based. Buffer, row and
/// column each own a disjoint digit, so a copy that mixes any of them up
/// changes the value.
fn host_rows(buffer: usize, rows: usize, hidden: usize) -> Vec<bf16> {
    (0..rows * hidden)
        .map(|i| {
            let row = i / hidden;
            let col = i % hidden;
            bf16::from_f32((buffer * 32 + row * 16 + col % 16) as f32)
        })
        .collect()
}

fn pairs<'a>(
    srcs: &'a [HiddenStates],
    dsts: &'a mut [HiddenStates],
) -> Vec<(&'a HiddenStates, &'a mut HiddenStates)> {
    srcs.iter().zip(dsts.iter_mut()).collect()
}

/// Scatter every buffer into a permutation of the slots, then gather them back
/// in the reverse order. `hidden` selects the 16-byte path when it is a
/// multiple of eight and the element path otherwise.
///
/// The copy is bitwise and every value is an exactly representable integer, so
/// the comparisons below are exact.
#[allow(clippy::float_cmp)]
fn roundtrip(ctx: &DeviceContext, hidden: usize, rows: usize, buffers: usize) {
    const SLOTS: usize = 8;
    assert!(rows < SLOTS, "leave an unused slot for the tail check");

    let src_hosts: Vec<Vec<bf16>> = (0..buffers).map(|b| host_rows(b, rows, hidden)).collect();
    let srcs: Vec<HiddenStates> = src_hosts
        .iter()
        .map(|host| HiddenStates::from_host(ctx, host, hidden, rows).expect("src H2D"))
        .collect();
    let mut pools: Vec<HiddenStates> = (0..buffers)
        .map(|_| HiddenStates::zeros(ctx, hidden, SLOTS).expect("pool alloc"))
        .collect();

    // Row i lands on slot `rows - 1 - i`; the two trailing entries are valid
    // slots the kernel must not consume, so a launch that walks past `rows`
    // becomes visible as a written slot below.
    let mut locs_host: Vec<i32> = (0..rows).map(|i| (rows - 1 - i) as i32).collect();
    locs_host.push((SLOTS - 1) as i32);
    locs_host.push((SLOTS - 2) as i32);
    let locs: CudaSlice<i32> = ctx.stream.clone_htod(&locs_host).expect("locs H2D");

    draft_kv_scatter_into(ctx, &mut pairs(&srcs, &mut pools), &locs, rows).expect("scatter launch");

    for (buffer, pool) in pools.iter().enumerate() {
        let pool_host = pool.to_host(ctx).expect("pool D2H");
        for slot in 0..SLOTS {
            let written = (0..rows).find(|&i| locs_host[i] as usize == slot);
            for col in 0..hidden {
                let expected =
                    written.map_or(0.0, |row| src_hosts[buffer][row * hidden + col].to_f32());
                let got = pool_host[slot * hidden + col];
                assert_eq!(
                    got, expected,
                    "scatter hidden={hidden} rows={rows} buffers={buffers}: \
                     buffer {buffer} slot {slot} col {col} (source row \
                     {written:?}) got {got}, expected {expected}"
                );
            }
        }
    }

    // Gather back through a reversed permutation so a kernel that copies row i
    // to row i instead of through `locs` cannot pass.
    let gather_locs_host: Vec<i32> = (0..rows).map(|j| locs_host[rows - 1 - j]).collect();
    let gather_locs: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&gather_locs_host)
        .expect("gather locs H2D");
    let mut mirrors: Vec<HiddenStates> = (0..buffers)
        .map(|_| HiddenStates::zeros(ctx, hidden, rows).expect("mirror alloc"))
        .collect();

    draft_kv_gather_into(ctx, &mut pairs(&pools, &mut mirrors), &gather_locs, rows)
        .expect("gather launch");

    for (buffer, mirror) in mirrors.iter().enumerate() {
        let mirror_host = mirror.to_host(ctx).expect("mirror D2H");
        for j in 0..rows {
            let source_row = rows - 1 - j;
            for col in 0..hidden {
                let expected = src_hosts[buffer][source_row * hidden + col].to_f32();
                let got = mirror_host[j * hidden + col];
                assert_eq!(
                    got, expected,
                    "gather hidden={hidden} rows={rows} buffers={buffers}: \
                     buffer {buffer} row {j} col {col} (slot {}) got {got}, \
                     expected {expected}",
                    gather_locs_host[j]
                );
            }
        }
    }
}

#[test]
fn scatter_and_gather_vector_path() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // The drafter's kv_dim (heads x head_dim) is a multiple of eight.
    roundtrip(&ctx, 1024, 5, 1);
    roundtrip(&ctx, 1024, 1, 1);
    roundtrip(&ctx, 8, 7, 1);
}

#[test]
fn scatter_and_gather_element_path() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // A stride that is not a multiple of eight has no 16-byte-aligned rows.
    roundtrip(&ctx, 12, 5, 1);
    roundtrip(&ctx, 7, 3, 1);
}

#[test]
fn one_launch_covers_every_draft_buffer() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    // Five layers x (K, V) is the drafter's whole set in one launch.
    roundtrip(&ctx, 1024, 4, 10);
    roundtrip(&ctx, 12, 3, 5);
}

#[test]
fn zero_rows_is_a_no_op() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut src = HiddenStates::zeros(&ctx, 1024, 4).expect("src alloc");
    let mut pool = HiddenStates::zeros(&ctx, 1024, 4).expect("pool alloc");
    let locs: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32, 1, 2, 3]).expect("locs H2D");
    draft_kv_scatter_into(&ctx, &mut [(&src, &mut pool)], &locs, 0).expect("empty scatter");
    draft_kv_gather_into(&ctx, &mut [(&pool, &mut src)], &locs, 0).expect("empty gather");
}
