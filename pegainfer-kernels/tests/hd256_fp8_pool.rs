//! Device gates for Gemma 4's e4m3 paged-KV storage contract.

#![cfg(feature = "gemma4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_window_hd256_into;
use pegainfer_kernels::ops::paged_attention_batch_decode_hd256_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_decode_hd256_plain_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_prefill_hd256_plain_into;
use pegainfer_kernels::paged_kv::KvStorage;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const PAGE_SIZE: usize = 2;
const NUM_LAYERS: usize = 3;

fn packed_fp8(bytes: &[u8]) -> Vec<bf16> {
    (0..bytes.len() / 2)
        .map(|slot| bf16::from_bits(u16::from_le_bytes([bytes[2 * slot], bytes[2 * slot + 1]])))
        .collect()
}

fn raw_bytes(ctx: &DeviceContext, pool: &CudaSlice<bf16>) -> Vec<u8> {
    ctx.stream
        .clone_dtoh(pool)
        .expect("pool D2H")
        .into_iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn constant_states(ctx: &DeviceContext, value: f32, rows: usize) -> HiddenStates {
    HiddenStates::from_host(ctx, &vec![bf16::from_f32(value); HD * rows], HD, rows)
        .expect("states H2D")
}

fn identity_rope(ctx: &DeviceContext, rows: usize) -> (DeviceVec, DeviceVec) {
    let cos = vec![bf16::ONE; HD * rows];
    let sin = vec![bf16::ZERO; HD * rows];
    (
        DeviceVec::from_host(ctx, &cos).expect("cos H2D"),
        DeviceVec::from_host(ctx, &sin).expect("sin H2D"),
    )
}

#[test]
fn fp8_prep_stores_exact_bytes_at_layout_offsets() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let layout = PagedKvLayout::with_storage(NUM_LAYERS, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * 3 / 2)
        .expect("pool alloc");
    let q = constant_states(&ctx, 1.0, 3);
    let k = constant_states(&ctx, 1.0, 3);
    let v = constant_states(&ctx, 0.5, 3);
    let mut q_out = HiddenStates::zeros(&ctx, HD, 3).expect("q_out alloc");
    let weights = DeviceVec::from_host(&ctx, &vec![bf16::from_f32(2.0); HD]).expect("weights H2D");
    let (cos, sin) = identity_rope(&ctx, 3);
    let pages = ctx.stream.clone_htod(&[2i32, 0]).expect("pages H2D");
    qkv_norm_rope_paged_prefill_hd256_plain_into(
        &ctx, &q, &k, &v, &mut q_out, 0, &pool, &layout, &weights, &weights, &cos, &sin, 1, &pages,
        0, 0, 0, 3, 1, 1, HD, 0.0,
    )
    .expect("fp8 prep");
    let got = raw_bytes(&ctx, &pool);
    let mut expected = vec![0u8; layout.page_stride * 3];
    for (token, page) in [2usize, 2, 0].into_iter().enumerate() {
        let slot = token % PAGE_SIZE;
        let layer = page * layout.page_stride + layout.layer_stride;
        let k = layer + slot * HD;
        let v = layer + layout.kv_block_len + slot * HD;
        expected[k..k + HD].fill(0x40);
        expected[v..v + HD].fill(0x38);
    }
    assert_eq!(got, expected);
}

#[test]
fn fp8_decode_prep_stores_exact_bytes_at_layout_offsets() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let layout = PagedKvLayout::with_storage(NUM_LAYERS, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * 3 / 2)
        .expect("pool alloc");
    let q = constant_states(&ctx, 1.0, 3);
    let k = constant_states(&ctx, 1.0, 3);
    let v = constant_states(&ctx, 0.5, 3);
    let mut q_out = HiddenStates::zeros(&ctx, HD, 3).expect("q_out alloc");
    let weights = DeviceVec::from_host(&ctx, &vec![bf16::from_f32(2.0); HD]).expect("weights H2D");
    let (cos, sin) = identity_rope(&ctx, 4);
    let pages = ctx.stream.clone_htod(&[2i32]).expect("pages H2D");
    let indptr = ctx.stream.clone_htod(&[0i32, 1]).expect("indptr H2D");
    // Released-front page of the row window: resident row = pos / page_size - origin.
    let origins = ctx.stream.clone_htod(&[1i32]).expect("origins H2D");
    let positions = ctx.stream.clone_htod(&[3i32]).expect("positions H2D");
    qkv_norm_rope_paged_decode_hd256_plain_into(
        &ctx, &q, &k, &v, &mut q_out, 2, &pool, &layout, &weights, &weights, &cos, &sin, 1, &pages,
        &indptr, &origins, &positions, 4, 1, 1, HD, 0.0,
    )
    .expect("fp8 decode prep");
    let got = raw_bytes(&ctx, &pool);
    let mut expected = vec![0u8; layout.page_stride * 3];
    let layer = 2 * layout.page_stride + layout.layer_stride;
    let k = layer + HD;
    let v = layer + layout.kv_block_len + HD;
    expected[k..k + HD].fill(0x40);
    expected[v..v + HD].fill(0x38);
    assert_eq!(got, expected);
}

fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((byte >> 3) & 0x0f);
    let mantissa = f32::from(byte & 0x07) / 8.0;
    if exponent == 0 {
        sign * mantissa * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + mantissa) * 2.0f32.powi(exponent - 7)
    }
}

fn valid_e4m3_code(ordinal: usize) -> u8 {
    let code = ordinal % 254;
    u8::try_from(code + usize::from(code >= 0x7f)).expect("e4m3 code")
}

fn varied_values(pages: usize) -> Vec<f32> {
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::Bf16);
    let mut values = vec![0.0; layout.page_stride * pages];
    for page in 0..pages {
        for slot in 0..PAGE_SIZE {
            for element in 0..HD {
                for block in 0..2 {
                    let index = page * layout.page_stride
                        + block * layout.kv_block_len
                        + slot * HD
                        + element;
                    let seed = slot * HD * 2 + element * 2 + block;
                    let variation = ((5 + seed % 5) << 3)
                        | (seed % 8)
                        | if seed.is_multiple_of(3) { 0x80 } else { 0 };
                    let byte = valid_e4m3_code(page + variation);
                    values[index] = e4m3_to_f32(byte);
                }
            }
        }
    }
    values
}

fn exact_e4m3(value: f32) -> u8 {
    (0u8..=254)
        .find(|&byte| byte != 0x7f && e4m3_to_f32(byte).to_bits() == value.to_bits())
        .expect("value must be exactly e4m3-representable")
}

fn semantic_pool(ctx: &DeviceContext, storage: KvStorage, pages: usize) -> CudaSlice<bf16> {
    let values = varied_values(pages);
    match storage {
        KvStorage::Bf16 => ctx
            .stream
            .clone_htod(
                &values
                    .iter()
                    .map(|&value| bf16::from_f32(value))
                    .collect::<Vec<_>>(),
            )
            .expect("bf16 pool H2D"),
        KvStorage::E4m3 => ctx
            .stream
            .clone_htod(&packed_fp8(
                &values
                    .iter()
                    .map(|&value| exact_e4m3(value))
                    .collect::<Vec<_>>(),
            ))
            .expect("fp8 pool H2D"),
    }
}

fn varied_q(ctx: &DeviceContext, rows: usize) -> HiddenStates {
    let host: Vec<bf16> = (0..HD * rows)
        .map(|index| bf16::from_f32(0.25 + (index % HD) as f32 / 512.0))
        .collect();
    HiddenStates::from_host(ctx, &host, HD, rows).expect("q H2D")
}

fn attend(ctx: &DeviceContext, storage: KvStorage, kv_len: usize, window_left: i32) -> Vec<u16> {
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, storage);
    let pages = kv_len.div_ceil(PAGE_SIZE);
    let pool = semantic_pool(ctx, storage, pages);
    let page_indices: Vec<i32> = (0..pages as i32).collect();
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &page_indices,
        (kv_len - 1) % PAGE_SIZE + 1,
        kv_len - 1,
        1,
        1,
        1,
        HD,
        0,
    )
    .expect("plan");
    let q = varied_q(ctx, 1);
    let mut output = HiddenStates::zeros(ctx, HD, 1).expect("output alloc");
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        0,
        &plan,
        &mut output,
        1,
        1.0,
        window_left,
    )
    .expect("attention");
    output
        .to_host(ctx)
        .expect("output D2H")
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect()
}

#[test]
fn fp8_window_read_matches_bf16_for_exact_values() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert_eq!(
        attend(&ctx, KvStorage::E4m3, 2, -1),
        attend(&ctx, KvStorage::Bf16, 2, -1)
    );
}

#[test]
fn fp8_finite_window_read_matches_bf16_and_changes_the_result() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let full = attend(&ctx, KvStorage::E4m3, 6, -1);
    let windowed = attend(&ctx, KvStorage::E4m3, 6, 2);
    assert_eq!(windowed, attend(&ctx, KvStorage::Bf16, 6, 2));
    assert_ne!(
        windowed, full,
        "finite window must change the varied-pool output"
    );
}

fn varied_geometry_probe(ctx: &DeviceContext, prefix_rows: usize, pages: usize) -> Vec<u16> {
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool = semantic_pool(ctx, KvStorage::E4m3, pages);
    let (page_lists, starts, lengths, lasts) = if prefix_rows == 0 {
        (vec![vec![pages as i32 - 1]], vec![1], vec![1], vec![2])
    } else {
        let prefix_pages: Vec<i32> = (0..pages as i32 - 1).rev().collect();
        (
            vec![prefix_pages, vec![pages as i32 - 1]],
            vec![0, 1],
            vec![prefix_rows, 1],
            vec![(prefix_rows - 1) % PAGE_SIZE + 1, 2],
        )
    };
    let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &page_lists,
        &lasts,
        &starts,
        &lengths,
        1,
        1,
        HD,
        0,
    )
    .expect("batch plan");
    let q = varied_q(ctx, prefix_rows + 1);
    let mut output = HiddenStates::zeros(ctx, HD, prefix_rows + 1).expect("output alloc");
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        0,
        &plan,
        &mut output,
        1,
        1.0,
        -1,
    )
    .expect("attention");
    let host = output.to_host(ctx).expect("output D2H");
    host[prefix_rows * HD..]
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect()
}

#[test]
fn varied_fp8_window_read_is_geometry_invariant_for_the_probed_row() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let packed_rows = 300;
    let pages = (packed_rows + 1usize).div_ceil(PAGE_SIZE);
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let bytes: Vec<u8> = varied_values(pages).into_iter().map(exact_e4m3).collect();
    let probed = &bytes[(pages - 1) * layout.page_stride..pages * layout.page_stride];
    for page in 0..pages - 1 {
        let other = &bytes[page * layout.page_stride..(page + 1) * layout.page_stride];
        assert_ne!(probed, other, "probed page aliases physical page {page}");
    }
    let lone = varied_geometry_probe(&ctx, 0, pages);
    let packed = varied_geometry_probe(&ctx, packed_rows, pages);
    if let Some(index) = lone.iter().zip(&packed).position(|(a, b)| a != b) {
        panic!(
            "varied-pool geometry dependence at output[{index}]: lone={:#06x}, packed={:#06x}",
            lone[index], packed[index]
        );
    }
}

#[test]
fn decode_wrapper_without_fp8_twin_refuses_e4m3() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride / 2)
        .expect("pool");
    let state = constant_states(&ctx, 0.0, 1);
    let mut output = HiddenStates::zeros(&ctx, HD, 1).expect("output");
    let meta = ctx.stream.clone_htod(&[0i32]).expect("metadata");
    let indptr = ctx.stream.clone_htod(&[0i32, 1]).expect("indptr");
    let err = paged_attention_batch_decode_hd256_into(
        &ctx,
        &state,
        &state,
        &state,
        &pool,
        &layout,
        0,
        &meta,
        &indptr,
        &meta,
        &meta,
        &meta,
        &meta,
        &meta,
        &mut output,
        1,
        1,
    )
    .expect_err("unsupported fp8 wrapper must reject");
    assert!(err.to_string().contains("has no fp8 KV path"), "{err}");
}
