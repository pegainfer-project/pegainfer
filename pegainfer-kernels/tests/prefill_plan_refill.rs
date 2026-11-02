//! A pre-allocated prefill plan refilled in place, against a fresh plan built
//! from the same batch: every device array must hold what the fresh plan holds,
//! and no array may move between refills. Qwen3's uncompiled GQA groups refill
//! one plan per unified step and rely on both.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip.

mod common;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::tensor::DeviceContext;

// Qwen3-14B's attention: GQA group 5, which has no compiled decode kernel.
const NUM_Q_HEADS: usize = 40;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const CTA_TILE_Q: i32 = 64;
const PAGE_SIZE: usize = 16;

/// The plan inputs for one unified step.
#[derive(Default)]
struct Batch {
    page_indices: Vec<Vec<i32>>,
    last_page_lens: Vec<usize>,
    start_positions: Vec<usize>,
    seq_lens: Vec<usize>,
}

impl Batch {
    /// Adds a row whose KV holds `kv_len` tokens on pages from `first_page`,
    /// of which the last `query` are this step's queries.
    fn row(mut self, first_page: i32, kv_len: usize, query: usize) -> Self {
        let pages = kv_len.div_ceil(PAGE_SIZE);
        self.page_indices
            .push((first_page..first_page + pages as i32).collect());
        self.last_page_lens.push(kv_len - (pages - 1) * PAGE_SIZE);
        self.start_positions.push(kv_len - query);
        self.seq_lens.push(query);
        self
    }

    fn fresh(&self, ctx: &DeviceContext) -> PrefillPagedPlan {
        PrefillPagedPlan::new_batch_with_cta_tile_q(
            ctx,
            &self.page_indices,
            &self.last_page_lens,
            &self.start_positions,
            &self.seq_lens,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            CTA_TILE_Q,
        )
        .expect("fresh plan")
    }

    fn refill(&self, ctx: &DeviceContext, plan: &mut PrefillPagedPlan) -> anyhow::Result<()> {
        plan.update_batch_with_cta_tile_q(
            ctx,
            &self.page_indices,
            &self.last_page_lens,
            &self.start_positions,
            &self.seq_lens,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            CTA_TILE_Q,
        )
    }
}

fn host(ctx: &DeviceContext, buf: &CudaSlice<i32>) -> Vec<i32> {
    ctx.stream.clone_dtoh(buf).expect("plan D2H")
}

fn total_rows(ctx: &DeviceContext, plan: &PrefillPagedPlan) -> Vec<u32> {
    ctx.stream
        .clone_dtoh(plan.total_num_rows_d())
        .expect("plan D2H")
}

/// The plan's `i32` device arrays by name; `total_num_rows_d` is `u32` and is
/// checked beside them.
fn arrays(plan: &PrefillPagedPlan) -> [(&'static str, &CudaSlice<i32>); 10] {
    [
        ("page_indices", plan.page_indices_d()),
        ("page_indptr", plan.page_indptr_d()),
        ("last_page_len", plan.last_page_len_d()),
        ("batch_indices", plan.batch_indices_d()),
        ("positions", plan.positions_d()),
        ("q_indptr", plan.q_indptr_d()),
        ("request_indices", plan.request_indices_d()),
        ("qo_tile_indices", plan.qo_tile_indices_d()),
        ("kv_tile_indices", plan.kv_tile_indices_d()),
        ("kv_chunk_size", plan.kv_chunk_size_d()),
    ]
}

fn pointers(ctx: &DeviceContext, plan: &PrefillPagedPlan) -> Vec<u64> {
    let mut ptrs: Vec<u64> = arrays(plan)
        .iter()
        .map(|(_, buf)| buf.device_ptr(&ctx.stream).0)
        .collect();
    ptrs.push(plan.total_num_rows_d().device_ptr(&ctx.stream).0);
    ptrs
}

/// The refilled plan's arrays begin with exactly the fresh plan's arrays;
/// whatever follows in a worst-case allocation is never read.
fn assert_matches(
    ctx: &DeviceContext,
    step: &str,
    fresh: &PrefillPagedPlan,
    refilled: &PrefillPagedPlan,
) {
    for ((name, fresh_buf), (_, refilled_buf)) in arrays(fresh).into_iter().zip(arrays(refilled)) {
        let want = host(ctx, fresh_buf);
        let got = host(ctx, refilled_buf);
        assert_eq!(&got[..want.len()], &want[..], "{step}: {name}");
    }
    assert_eq!(
        total_rows(ctx, refilled),
        total_rows(ctx, fresh),
        "{step}: total_num_rows"
    );
    assert_eq!(
        refilled.q_indptr_host(),
        fresh.q_indptr_host(),
        "{step}: q_indptr_host"
    );
    assert_eq!(refilled.num_tiles(), fresh.num_tiles(), "{step}: num_tiles");
}

#[test]
fn a_refilled_plan_matches_a_fresh_one_and_never_moves() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let mut plan =
        PrefillPagedPlan::new_preallocated(ctx, 1024, 1024, 8, 512).expect("preallocated plan");
    let pinned = pointers(ctx, &plan);

    let mixed = Batch::default()
        .row(0, 300, 300)
        .row(40, 51, 1)
        .row(100, 901, 1);
    // Shared prefixes put the same pages in several rows.
    let shared = (0..5).fold(Batch::default(), |batch, row| {
        batch.row(0, 200 + row * 17, 1)
    });
    let steps = [
        ("mixed prefill and decode", &mixed),
        ("one decode row", &Batch::default().row(7, 17, 1)),
        ("shared prefix decode", &shared),
        ("mixed again after a smaller step", &mixed),
    ];
    for (step, batch) in steps {
        batch.refill(ctx, &mut plan).expect(step);
        assert_matches(ctx, step, &batch.fresh(ctx), &plan);
        assert_eq!(pointers(ctx, &plan), pinned, "{step}: an array moved");
    }

    let too_wide = (0..9).fold(Batch::default(), |batch, row| batch.row(row * 4, 33, 1));
    let err = too_wide
        .refill(ctx, &mut plan)
        .expect_err("nine rows past an eight-row capacity");
    assert!(
        err.to_string().contains("exceeds preallocated capacity"),
        "unexpected error: {err:#}"
    );
    assert_eq!(
        pointers(ctx, &plan),
        pinned,
        "a refused refill moved an array"
    );
    // The capacity checks run before any copy, so a refusal leaves the last
    // accepted batch in place.
    assert_matches(ctx, "after a refused refill", &mixed.fresh(ctx), &plan);
}
