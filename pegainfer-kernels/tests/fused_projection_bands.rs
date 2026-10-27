//! The preps and the activation read a fused projection row as they read
//! separate buffers: the same bytes land in the pool, the query and the
//! activation whichever way Q, K and V, or gate and up, are handed over.
//!
//! The band form changes only where a source is read from, so the two
//! launches must agree bit for bit; anything less is an addressing slip.
//! The other bands of the fused row hold the other projections, so a stride
//! taken for a width reads a neighbour's values and shows.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::gelu_tanh_mul_batch_into;
use pegainfer_kernels::ops::qk_norm_partial_rope_paged_prefill_hd512_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_prefill_hd256_plain_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::Columns;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const ROWS: usize = 37;
const PAGE_SIZE: usize = 16;
const NUM_LAYERS: usize = 3;
const LAYER: usize = 1;
const START_POS: usize = 5;
const COS_MAX_POS: usize = 64;
/// Pages the rows land on, scattered over a pool of eight.
const PAGES: [i32; 3] = [4, 1, 6];
const POOL_PAGES: usize = 8;

/// `parts`, each `[ROWS, width]`, laid side by side in every row of one
/// tensor and each on its own.
fn fused_and_separate(
    ctx: &DeviceContext,
    parts: &[(u64, usize)],
) -> (HiddenStates, Vec<HiddenStates>) {
    let total: usize = parts.iter().map(|(_, width)| width).sum();
    let mut fused = vec![bf16::ZERO; ROWS * total];
    let mut separate = Vec::with_capacity(parts.len());
    let mut col = 0;
    for &(seed, width) in parts {
        let values = common::fill(seed, ROWS * width);
        for row in 0..ROWS {
            fused[row * total + col..row * total + col + width]
                .copy_from_slice(&values[row * width..(row + 1) * width]);
        }
        separate.push(HiddenStates::from_host(ctx, &values, width, ROWS).expect("part H2D"));
        col += width;
    }
    (
        HiddenStates::from_host(ctx, &fused, total, ROWS).expect("fused H2D"),
        separate,
    )
}

fn vec(ctx: &DeviceContext, seed: u64, n: usize) -> DeviceVec {
    DeviceVec::from_host(ctx, &common::fill(seed, n)).expect("vec H2D")
}

fn pool_bits(ctx: &DeviceContext, pool: &CudaSlice<bf16>) -> Vec<u16> {
    ctx.stream
        .clone_dtoh(pool)
        .expect("pool D2H")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn bits(ctx: &DeviceContext, states: &HiddenStates) -> Vec<u32> {
    states
        .to_host(ctx)
        .expect("D2H")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

struct Hd256 {
    layout: PagedKvLayout,
    pages: CudaSlice<i32>,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    cos: DeviceVec,
    sin: DeviceVec,
}

const HD256: usize = 256;
const HD256_Q_HEADS: usize = 4;
const HD256_KV_HEADS: usize = 2;

impl Hd256 {
    fn new(ctx: &DeviceContext) -> Self {
        Self {
            layout: PagedKvLayout::new(NUM_LAYERS, HD256_KV_HEADS, HD256, PAGE_SIZE),
            pages: ctx.stream.clone_htod(&PAGES).expect("pages H2D"),
            q_norm: vec(ctx, 11, HD256),
            k_norm: vec(ctx, 12, HD256),
            cos: vec(ctx, 13, COS_MAX_POS * HD256),
            sin: vec(ctx, 14, COS_MAX_POS * HD256),
        }
    }

    /// The pool's and the query's bits after one prep over the bands.
    fn run(
        &self,
        ctx: &DeviceContext,
        q: Columns<'_>,
        k: Columns<'_>,
        v: Columns<'_>,
    ) -> anyhow::Result<(Vec<u16>, Vec<u32>)> {
        let pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(POOL_PAGES * self.layout.page_stride)
            .expect("pool alloc");
        let mut q_out = HiddenStates::zeros(ctx, HD256_Q_HEADS * HD256, ROWS).expect("q_out");
        qkv_norm_rope_paged_prefill_hd256_plain_into(
            ctx,
            q,
            k,
            v,
            &mut q_out,
            0,
            &pool,
            &self.layout,
            &self.q_norm,
            &self.k_norm,
            &self.cos,
            &self.sin,
            LAYER,
            &self.pages,
            0,
            0,
            START_POS,
            COS_MAX_POS,
            HD256_Q_HEADS,
            HD256_KV_HEADS,
            HD256,
            1e-6,
        )?;
        Ok((pool_bits(ctx, &pool), bits(ctx, &q_out)))
    }
}

#[test]
fn the_hd256_prep_reads_a_fused_row_as_it_reads_three_buffers() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let (q_dim, kv_dim) = (HD256_Q_HEADS * HD256, HD256_KV_HEADS * HD256);
    let (fused, parts) = fused_and_separate(&ctx, &[(1, q_dim), (2, kv_dim), (3, kv_dim)]);
    let rig = Hd256::new(&ctx);
    let (pool_a, q_a) = rig
        .run(
            &ctx,
            (&parts[0]).into(),
            (&parts[1]).into(),
            (&parts[2]).into(),
        )
        .expect("separate buffers");
    let (pool_b, q_b) = rig
        .run(
            &ctx,
            fused.columns(0, q_dim),
            fused.columns(q_dim, kv_dim),
            fused.columns(q_dim + kv_dim, kv_dim),
        )
        .expect("fused row");
    assert!(pool_a.iter().any(|&x| x != 0), "the prep wrote nothing");
    assert!(pool_a == pool_b, "the pool differs between the two forms");
    assert!(q_a == q_b, "the query differs between the two forms");

    // A band past the row's end is refused before the launch.
    let total = q_dim + 2 * kv_dim;
    let err = rig
        .run(
            &ctx,
            fused.columns(0, q_dim),
            fused.columns(q_dim, kv_dim),
            fused.columns(total - kv_dim + 1, kv_dim),
        )
        .expect_err("a band past the row");
    assert!(err.to_string().contains("exceed the row"), "{err}");
}

const HD512: usize = 512;
const HD512_Q_HEADS: usize = 4;
const HD512_KV_HEADS: usize = 2;

#[test]
fn the_hd512_prep_reads_a_fused_row_as_it_reads_two_buffers() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let (q_dim, kv_dim) = (HD512_Q_HEADS * HD512, HD512_KV_HEADS * HD512);
    let (fused, parts) = fused_and_separate(&ctx, &[(21, q_dim), (22, kv_dim)]);
    let layout = PagedKvLayout::new(NUM_LAYERS, HD512_KV_HEADS, HD512, PAGE_SIZE);
    let pages = ctx.stream.clone_htod(&PAGES).expect("pages H2D");
    let (q_norm, k_norm) = (vec(&ctx, 23, HD512), vec(&ctx, 24, HD512));
    let (cos, sin) = (
        vec(&ctx, 25, COS_MAX_POS * HD512),
        vec(&ctx, 26, COS_MAX_POS * HD512),
    );
    let run = |q: Columns<'_>, k: Columns<'_>| {
        let pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(POOL_PAGES * layout.page_stride)
            .expect("pool alloc");
        let mut q_out = HiddenStates::zeros(&ctx, q_dim, ROWS).expect("q_out");
        qk_norm_partial_rope_paged_prefill_hd512_into(
            &ctx,
            q,
            k,
            &mut q_out,
            0,
            &pool,
            &layout,
            &q_norm,
            &k_norm,
            &cos,
            &sin,
            LAYER,
            &pages,
            0,
            START_POS,
            COS_MAX_POS,
            HD512_Q_HEADS,
            HD512_KV_HEADS,
            1e-6,
        )
        .expect("hd512 prep");
        (pool_bits(&ctx, &pool), bits(&ctx, &q_out))
    };
    let (pool_a, q_a) = run((&parts[0]).into(), (&parts[1]).into());
    let (pool_b, q_b) = run(fused.columns(0, q_dim), fused.columns(q_dim, kv_dim));
    assert!(pool_a.iter().any(|&x| x != 0), "the prep wrote nothing");
    assert!(pool_a == pool_b, "the pool differs between the two forms");
    assert!(q_a == q_b, "the query differs between the two forms");
}

#[test]
fn the_activation_reads_fused_bands_as_it_reads_two_buffers() {
    // Not a multiple of anything the kernel might round to.
    const WIDTH: usize = 96;
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let (fused, parts) = fused_and_separate(&ctx, &[(31, WIDTH), (32, WIDTH)]);
    let run = |gate: Columns<'_>, up: Columns<'_>| {
        let mut out = HiddenStates::zeros(&ctx, WIDTH, ROWS).expect("out");
        gelu_tanh_mul_batch_into(&ctx, gate, up, &mut out).map(|()| bits(&ctx, &out))
    };
    let a = run((&parts[0]).into(), (&parts[1]).into()).expect("separate buffers");
    let b = run(fused.columns(0, WIDTH), fused.columns(WIDTH, WIDTH)).expect("fused row");
    assert!(a.iter().any(|&x| x != 0), "the activation wrote nothing");
    assert!(a == b, "the activation differs between the two forms");
    let err = run(fused.columns(0, WIDTH), fused.columns(WIDTH + 1, WIDTH))
        .expect_err("a band past the row");
    assert!(err.to_string().contains("exceed the row"), "{err}");
}
