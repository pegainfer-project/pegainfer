//! Gated HD256 prefill at Qwen4's dense-to-sparse boundary.

#![cfg(feature = "qwen35")]

mod common;

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::Hd256Prefill;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::attention_prefill_hd256_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

#[test]
fn dense_prefix_matches_dense_across_a_partial_page() -> Result<()> {
    const ROWS: usize = 2065;
    const DENSE_ROWS: usize = 2051;
    const PAGE_SIZE: usize = 16;
    const HEAD_DIM: usize = 256;
    const ROTARY_DIM: usize = 64;
    const Q_HEADS: usize = 24;
    const KV_HEADS: usize = 2;
    const Q_WIDTH: usize = Q_HEADS * HEAD_DIM;
    const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;

    let Some(ctx) = common::device_or_skip() else {
        return Ok(());
    };

    // The dense prefix uses 129 pages, with three tokens on the last. The append needs 130.
    let pages: Vec<_> = (0..ROWS.div_ceil(PAGE_SIZE) as i32).collect();
    let layout = PagedKvLayout::new(1, KV_HEADS, HEAD_DIM, PAGE_SIZE);
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        &ctx,
        &pages,
        (ROWS - 1) % PAGE_SIZE + 1,
        0,
        ROWS,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        0,
    )?;
    let states =
        |seed, width| HiddenStates::from_host(&ctx, &common::fill(seed, ROWS * width), width, ROWS);
    let q = states(1, 2 * Q_WIDTH)?;
    let k = states(2, KV_WIDTH)?;
    let v = states(3, KV_WIDTH)?;
    let norm = DeviceVec::zeros(&ctx, HEAD_DIM)?;
    let cos = DeviceVec::from_host(&ctx, &vec![bf16::ONE; ROWS * ROTARY_DIM])?;
    let sin = DeviceVec::zeros(&ctx, ROWS * ROTARY_DIM)?;
    let sentinel = bf16::from_f32(8.0);

    let run = |mode| -> Result<_> {
        let pool = ctx.stream.alloc_zeros(pages.len() * layout.page_stride)?;
        let mut prepared = HiddenStates::zeros(&ctx, Q_WIDTH, ROWS)?;
        let mut out =
            HiddenStates::from_host(&ctx, &vec![sentinel; ROWS * Q_WIDTH], Q_WIDTH, ROWS)?;
        attention_prefill_hd256_into(
            &ctx,
            &q,
            &k,
            &v,
            &norm,
            &norm,
            &cos,
            &sin,
            &pool,
            &layout,
            0,
            mode,
            0,
            ROTARY_DIM,
            1e-6,
            &mut prepared,
            &mut out,
        )?;
        Ok((
            ctx.stream.clone_dtoh(&prepared.data)?,
            ctx.stream.clone_dtoh(&pool)?,
            ctx.stream.clone_dtoh(&out.data)?,
        ))
    };
    let (dense_q, dense_kv, dense_out) = run(Hd256Prefill::Dense(&plan))?;
    let (prefix_q, prefix_kv, prefix_out) = run(Hd256Prefill::DensePrefix {
        pages: &pages,
        rows: DENSE_ROWS,
    })?;

    let end = DENSE_ROWS * Q_WIDTH;
    for (name, actual, expected) in [
        ("prepared Q", &prefix_q[..], &dense_q[..]),
        ("paged K/V", &prefix_kv[..], &dense_kv[..]),
        ("gated prefix", &prefix_out[..end], &dense_out[..end]),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "{name} at element {index}");
        }
    }
    for (index, value) in prefix_out[end..].iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            sentinel.to_bits(),
            "sparse tail at element {index}"
        );
    }
    Ok(())
}
