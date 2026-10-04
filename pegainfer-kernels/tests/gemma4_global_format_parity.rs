//! The two pool row formats against each other, through the writer and the
//! generated readers, with no checkpoint. One set of projections goes into a
//! split pool and a folded one, each read by the kernel lowered for it; the
//! formats differ only in where K's norm weight is applied, so the outputs are
//! one bf16 rounding apart. That rounding path and the two permutations are
//! what this isolates from the whole model.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::Hd512DecodeMetadata;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::gemma4_hd512_decode_split_kv_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::gemma4_hd512_prefill_varlen_into;
use pegainfer_kernels::ops::qk_norm_partial_rope_paged_prefill_hd512_into;
use pegainfer_kernels::paged_kv::KvFormat;
use pegainfer_kernels::paged_kv::KvStorage;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
/// The global family's proportional rotation: this many columns of the head
/// see a live angle, which is also what the folded row keeps of K.
const ROTARY: usize = 128;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const PAGE_SIZE: usize = 64;
const NUM_LAYERS: usize = 10;
const LAYER: usize = 5;
/// Several key blocks, a partial last page, and more than one split chunk.
const SEQ_LEN: usize = 500;
const CHUNK_TOKENS: usize = 256;
const RMS_EPS: f32 = 1e-6;
const HOSTILE: i32 = -10_000_000;
/// The line the kernel-against-kernel parity gates hold at. It holds here
/// at the model's score scale: the formats differ in the rounding of the
/// score operands themselves, which a softmax over unscaled dot products of
/// five hundred terms would amplify into arbitrary output differences.
const LINE: f32 = 0.05;

/// Norm weights away from zero and from one, so a weight applied twice or
/// not at all is visible.
fn weights(seed: u64) -> Vec<bf16> {
    common::fill(seed, HD)
        .into_iter()
        .map(|w| bf16::from_f32(1.0 + 0.5 * w.to_f32()))
        .collect()
}

/// The engine's proportional tables: `[pos * 512 + d]`, the first
/// `ROTARY / 2` angles live and the identity past them.
fn rope_tables(ctx: &DeviceContext, rows: usize) -> (DeviceVec, DeviceVec) {
    let mut cos = Vec::with_capacity(rows * HD);
    let mut sin = Vec::with_capacity(rows * HD);
    for pos in 0..rows {
        for d in 0..HD {
            if d < ROTARY / 2 {
                let theta = pos as f32 * 10_000f32.powf(-(2.0 * d as f32) / HD as f32);
                cos.push(bf16::from_f32(theta.cos()));
                sin.push(bf16::from_f32(theta.sin()));
            } else {
                cos.push(bf16::ONE);
                sin.push(bf16::ZERO);
            }
        }
    }
    (
        DeviceVec::from_host(ctx, &cos).expect("cos H2D"),
        DeviceVec::from_host(ctx, &sin).expect("sin H2D"),
    )
}

fn judge(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let (worst, at) = common::worst_delta(a, b);
    let row = at / (NUM_Q_HEADS * HD);
    let head = (at % (NUM_Q_HEADS * HD)) / HD;
    eprintln!(
        "{what}: worst |delta| {worst} at row {row} head {head} lane {}; split {} folded {}",
        at % HD,
        a[at],
        b[at]
    );
    assert!(
        worst <= LINE,
        "{what}: the two formats disagree by {worst} at row {row} head {head}"
    );
}

/// One format's side: the pool the writer filled and the query rows it
/// produced for that pool's readers.
struct Side {
    layout: PagedKvLayout,
    pool: CudaSlice<bf16>,
    q_out: HiddenStates,
}

#[allow(clippy::too_many_arguments)]
fn write(
    ctx: &DeviceContext,
    format: KvFormat,
    q: &HiddenStates,
    k: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos: &DeviceVec,
    sin: &DeviceVec,
    page_indices: &[i32],
    page_indices_d: &CudaSlice<i32>,
) -> Side {
    let layout = PagedKvLayout::with_storage_and_format(
        NUM_LAYERS,
        NUM_KV_HEADS,
        HD,
        PAGE_SIZE,
        KvStorage::Bf16,
        format,
    );
    // Junk everywhere the writer does not reach, so a reader that strays
    // sees noise rather than zeros.
    let pool_pages = page_indices.len() * 3;
    let pool = ctx
        .stream
        .clone_htod(&common::fill(0x51DE, pool_pages * layout.page_stride))
        .expect("pool upload");
    let mut q_out = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, SEQ_LEN).expect("q_out");
    qk_norm_partial_rope_paged_prefill_hd512_into(
        ctx,
        q,
        k,
        &mut q_out,
        0,
        &pool,
        &layout,
        q_norm,
        k_norm,
        cos,
        sin,
        LAYER,
        page_indices_d,
        0,
        0,
        SEQ_LEN,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        RMS_EPS,
    )
    .expect("prep");
    Side {
        layout,
        pool,
        q_out,
    }
}

fn prefill(ctx: &DeviceContext, side: &Side, plan: &PrefillPagedPlan) -> Vec<f32> {
    let mut out = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, SEQ_LEN).expect("out");
    gemma4_hd512_prefill_varlen_into(
        ctx,
        &side.q_out,
        &side.pool,
        &side.layout,
        LAYER,
        plan,
        &mut out,
        NUM_Q_HEADS,
        (HD as f32).powf(-0.5),
    )
    .expect("generated prefill");
    out.to_host(ctx).expect("D2H")
}

/// The last prompt row as a decode step over the whole context: the same
/// query the prefill's last row carried, so the two reads answer the same
/// question through different kernels.
struct DecodePlan {
    page_indices: CudaSlice<i32>,
    page_indptr: CudaSlice<i32>,
    last_page_len: CudaSlice<i32>,
    request_indices: CudaSlice<i32>,
    kv_tile_indices: CudaSlice<i32>,
    chunk_size: CudaSlice<i32>,
    o_indptr: CudaSlice<i32>,
    valid_mask: CudaSlice<u8>,
    slots: usize,
}

fn decode_plan(ctx: &DeviceContext, page_indices: &[i32]) -> DecodePlan {
    let pages = page_indices.len() as i32;
    let chunks = SEQ_LEN.div_ceil(CHUNK_TOKENS);
    let mut ids = page_indices.to_vec();
    let mut page_indptr = vec![0, pages];
    let mut last = vec![((SEQ_LEN - 1) % PAGE_SIZE + 1) as i32];
    let mut req: Vec<i32> = vec![0; chunks];
    let mut tile: Vec<i32> = (0..chunks as i32).collect();
    let mut valid = vec![1u8; chunks];
    let mut o_indptr = vec![0, chunks as i32];
    for i in 0..4 {
        page_indptr.push(pages + HOSTILE * (i + 1));
        last.push(HOSTILE);
        o_indptr.push(chunks as i32 + HOSTILE * (i + 1));
    }
    for _ in 0..8 {
        req.push(HOSTILE);
        tile.push(HOSTILE);
        valid.push(1);
        ids.push(HOSTILE);
    }
    let up = |v: &[i32]| ctx.stream.clone_htod(v).expect("upload");
    DecodePlan {
        page_indices: up(&ids),
        page_indptr: up(&page_indptr),
        last_page_len: up(&last),
        request_indices: up(&req),
        kv_tile_indices: up(&tile),
        chunk_size: up(&[CHUNK_TOKENS as i32]),
        o_indptr: up(&o_indptr),
        valid_mask: ctx.stream.clone_htod(&valid).expect("valid mask"),
        slots: chunks,
    }
}

fn decode(ctx: &DeviceContext, side: &Side, plan: &DecodePlan) -> Vec<f32> {
    let meta = Hd512DecodeMetadata::new(
        &plan.page_indices,
        &plan.page_indptr,
        &plan.last_page_len,
        &plan.request_indices,
        &plan.kv_tile_indices,
        &plan.chunk_size,
        CHUNK_TOKENS,
    );
    let mut out = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, SEQ_LEN).expect("out");
    let mut tmp_v = ctx
        .stream
        .alloc_zeros::<bf16>(plan.slots * NUM_Q_HEADS * HD)
        .expect("tmp_v");
    let mut tmp_s = ctx
        .stream
        .alloc_zeros::<f32>(plan.slots * NUM_Q_HEADS)
        .expect("tmp_s");
    gemma4_hd512_decode_split_kv_into(
        ctx,
        &side.q_out,
        SEQ_LEN - 1,
        &side.pool,
        &side.layout,
        LAYER,
        &meta,
        &plan.o_indptr,
        &plan.valid_mask,
        &mut tmp_v,
        &mut tmp_s,
        plan.slots,
        &mut out,
        NUM_Q_HEADS,
        (HD as f32).powf(-0.5),
    )
    .expect("generated decode");
    let host = out.to_host(ctx).expect("D2H");
    host[(SEQ_LEN - 1) * NUM_Q_HEADS * HD..].to_vec()
}

#[test]
fn the_folded_pool_reads_as_the_split_one() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub"
    );
    let q = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&common::fill(0xC0_FFEE, SEQ_LEN * NUM_Q_HEADS * HD))
            .expect("q"),
        seq_len: SEQ_LEN,
        hidden_dim: NUM_Q_HEADS * HD,
    };
    let k = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&common::fill(0xF0_0D, SEQ_LEN * NUM_KV_HEADS * HD))
            .expect("k"),
        seq_len: SEQ_LEN,
        hidden_dim: NUM_KV_HEADS * HD,
    };
    let q_norm = DeviceVec::from_host(ctx, &weights(0x11)).expect("q_norm");
    let k_norm = DeviceVec::from_host(ctx, &weights(0x22)).expect("k_norm");
    let (cos, sin) = rope_tables(ctx, SEQ_LEN);
    // Scattered pages, so a stride mistake reads somebody else's rows.
    let pages = SEQ_LEN.div_ceil(PAGE_SIZE);
    let page_indices: Vec<i32> = (0..pages).map(|p| (p * 3 + 1) as i32).collect();
    let page_indices_d = ctx.stream.clone_htod(&page_indices).expect("pages");

    let sides: Vec<Side> = [KvFormat::Split, KvFormat::Folded { rotary: ROTARY }]
        .into_iter()
        .map(|format| {
            write(
                ctx,
                format,
                &q,
                &k,
                &q_norm,
                &k_norm,
                &cos,
                &sin,
                &page_indices,
                &page_indices_d,
            )
        })
        .collect();

    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &page_indices,
        (SEQ_LEN - 1) % PAGE_SIZE + 1,
        0,
        SEQ_LEN,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0,
    )
    .expect("prefill plan");
    let prefills: Vec<Vec<f32>> = sides.iter().map(|side| prefill(ctx, side, &plan)).collect();
    judge("prefill", &prefills[0], &prefills[1]);

    let plan = decode_plan(ctx, &page_indices);
    let decodes: Vec<Vec<f32>> = sides.iter().map(|side| decode(ctx, side, &plan)).collect();
    judge("decode", &decodes[0], &decodes[1]);

    // The decode's row is the prefill's last, so each format's two readers
    // have to agree with each other as well.
    let last = (SEQ_LEN - 1) * NUM_Q_HEADS * HD;
    for (name, prefill_out, decode_out) in [
        ("split", &prefills[0], &decodes[0]),
        ("folded", &prefills[1], &decodes[1]),
    ] {
        let (worst, at) = common::worst_delta(&prefill_out[last..], decode_out);
        eprintln!("{name}: prefill last row against decode, worst |delta| {worst} at lane {at}");
        assert!(
            worst <= LINE,
            "{name}: the prefill's last row and the decode disagree by {worst}"
        );
    }
}
