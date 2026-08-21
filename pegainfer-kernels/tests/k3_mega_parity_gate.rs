//! Bit-level parity gate for the fused MegaMoE shim (situ and swiglu).
//!
//! The reference is DeepGEMM's own Python path, run once on a Blackwell GPU
//! and recorded as a fixture: the inputs in checkpoint form, plus the OUTPUT
//! BITS of the validated kernel. This gate re-runs the whole Rust pipeline —
//! scale-factor prepare, weight transforms, activation quant, launch — and
//! requires `y` to be bit-identical. That is a strictly stronger statement
//! than a tolerance check: it proves the twelve symmetric-buffer offsets, both
//! weight permutations, the packed-UE8M0 spelling and the block/stage/SM
//! launch configuration all match what the Python wrapper would have chosen.
//!
//! The expert weights are not part of the fixture (~3.9 GiB at 224 experts);
//! both sides synthesise them from the same index-addressable splitmix64 hash.
//! See `_dg_mega_dump.py` in the private K3 Python repo for the producer.
//!
//! Point `PEGAINFER_K3_MEGA_FIXTURE` at the dump directory to run it; unset,
//! the test skips.

#![cfg(feature = "k3")]

mod common;

use std::path::Path;
use std::path::PathBuf;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use half::bf16;
use pegainfer_kernels::ops::K3MegaActivation;
use pegainfer_kernels::ops::K3MegaShape;
use pegainfer_kernels::ops::k3_mega_max_tokens_per_rank;
use pegainfer_kernels::ops::k3_mega_moe_launch;
use pegainfer_kernels::ops::k3_mega_prepare_l1_weights_launch;
use pegainfer_kernels::ops::k3_mega_prepare_sf_launch;
use pegainfer_kernels::ops::k3_mega_symm_buffer_layout;
use pegainfer_kernels::ops::k3_mega_write_inputs_launch;
use pegainfer_kernels::tensor::DeviceContext;

const HIDDEN: usize = 3584;
const INTER: usize = 3072;
const EXPERTS: usize = 224;
const TOPK: usize = 16;
const TOKENS: usize = 64;
/// Experts whose transformed weights the fixture carries for the
/// localise-the-failure half of the gate.
const TRANSFORM_EXPERTS: usize = 4;

const SEED_L1_W: u64 = 0x1111_1111;
const SEED_L1_SF: u64 = 0x2222_2222;
const SEED_L2_W: u64 = 0x3333_3333;
const SEED_L2_SF: u64 = 0x4444_4444;
const SF_BASE: u8 = 120;
const SF_SPAN: u8 = 15;

const SF_GROUP_K: usize = 32;
const SF_WORD_K: usize = 128;

/// `splitmix64(index * GOLDEN + seed) & 0xFF`, the same recurrence the Python
/// producer evaluates with numpy.
fn hash_bytes(seed: u64, count: usize) -> Vec<u8> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut out = vec![0u8; count];
    for (index, slot) in out.iter_mut().enumerate() {
        let mut h = (index as u64).wrapping_mul(GOLDEN).wrapping_add(seed);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
        *slot = (h & 0xFF) as u8;
    }
    out
}

/// UE8M0 exponent bytes in the narrow band the producer uses.
fn hash_sf_bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut out = hash_bytes(seed, count);
    for byte in &mut out {
        *byte = SF_BASE + (*byte % SF_SPAN);
    }
    out
}

fn read_bytes(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()))
}

fn read_i32(dir: &Path, name: &str) -> Vec<i32> {
    read_bytes(dir, name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    read_bytes(dir, name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_bf16(dir: &Path, name: &str) -> Vec<bf16> {
    read_bytes(dir, name)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
        .collect()
}

fn fixture_dir() -> Option<PathBuf> {
    let raw = std::env::var("PEGAINFER_K3_MEGA_FIXTURE").ok()?;
    let dir = PathBuf::from(raw);
    if dir.join("meta.json").exists() {
        return Some(dir);
    }
    eprintln!("skipping: {} has no meta.json", dir.display());
    None
}

/// Report where two byte streams first differ, plus how many bytes differ in
/// total — a lone mismatch localises to a row/word, a dense one to a layout.
fn assert_bytes_eq(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let Some(first) = got.iter().zip(want).position(|(a, b)| a != b) else {
        return;
    };
    let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
    panic!(
        "{what}: {differing}/{} bytes differ; first at index {first} (got {:#04x}, want {:#04x})",
        got.len(),
        got[first],
        want[first]
    );
}

fn device_slice_u8(ctx: &DeviceContext, host: &[u8]) -> CudaSlice<u8> {
    ctx.stream.clone_htod(host).expect("upload u8 buffer")
}

#[test]
#[ignore = "needs a GB300 GPU and PEGAINFER_K3_MEGA_FIXTURE"]
fn mega_moe_matches_the_python_kernel_bit_for_bit() {
    let Some(dir) = fixture_dir() else {
        return;
    };
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let num_sms = ctx
        .ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .expect("query SM count") as usize;
    if num_sms != 152 {
        assert!(
            std::env::var("PEGAINFER_REQUIRE_GPU").as_deref() != Ok("1"),
            "PEGAINFER_REQUIRE_GPU=1 but the device has {num_sms} SMs; MegaMoE is \
             AOT-instantiated for the GB300 152-SM grid only"
        );
        eprintln!("skipping: {num_sms} SMs, not the GB300 MegaMoE instantiation");
        return;
    }

    // --- inputs -----------------------------------------------------------
    let x = read_bf16(&dir, "x_bf16.bin");
    let topk_idx = read_i32(&dir, "topk_idx_i32.bin");
    let topk_weight = read_f32(&dir, "topk_weights_f32.bin");
    let want_y = read_bf16(&dir, "y_bf16.bin");
    let want_y_swiglu = read_bf16(&dir, "y_swiglu_bf16.bin");
    assert_eq!(x.len(), TOKENS * HIDDEN, "fixture x has the wrong length");
    assert_eq!(topk_idx.len(), TOKENS * TOPK);
    assert_eq!(want_y.len(), TOKENS * HIDDEN);
    assert_eq!(want_y_swiglu.len(), TOKENS * HIDDEN);

    let l1_n = 2 * INTER;
    let l1_src = hash_bytes(SEED_L1_W, EXPERTS * l1_n * (HIDDEN / 2));
    let l2_src = hash_bytes(SEED_L2_W, EXPERTS * HIDDEN * (INTER / 2));
    let l1_sf_src = hash_sf_bytes(SEED_L1_SF, EXPERTS * l1_n * (HIDDEN / SF_GROUP_K));
    let l2_sf_src = hash_sf_bytes(SEED_L2_SF, EXPERTS * HIDDEN * (INTER / SF_GROUP_K));

    let l1_src_dev = device_slice_u8(&ctx, &l1_src);
    let l2_weights = device_slice_u8(&ctx, &l2_src);
    let l1_sf_src_dev = device_slice_u8(&ctx, &l1_sf_src);
    let l2_sf_src_dev = device_slice_u8(&ctx, &l2_sf_src);
    drop(l1_src);
    drop(l2_src);

    // --- build-time transforms -------------------------------------------
    let mut l1_weights = ctx
        .stream
        .alloc_zeros::<u8>(EXPERTS * l1_n * (HIDDEN / 2))
        .expect("alloc interleaved L1 weights");
    k3_mega_prepare_l1_weights_launch(&ctx, EXPERTS, l1_n, HIDDEN, &l1_src_dev, &mut l1_weights)
        .expect("L1 weight interleave");

    let mut l1_weights_sf = ctx
        .stream
        .alloc_zeros::<i32>(EXPERTS * (HIDDEN / SF_WORD_K) * l1_n)
        .expect("alloc L1 mega SF");
    k3_mega_prepare_sf_launch(
        &ctx,
        EXPERTS,
        l1_n,
        HIDDEN,
        true,
        &l1_sf_src_dev,
        &mut l1_weights_sf,
    )
    .expect("L1 SF prepare");

    let mut l2_weights_sf = ctx
        .stream
        .alloc_zeros::<i32>(EXPERTS * (INTER / SF_WORD_K) * HIDDEN)
        .expect("alloc L2 mega SF");
    k3_mega_prepare_sf_launch(
        &ctx,
        EXPERTS,
        HIDDEN,
        INTER,
        false,
        &l2_sf_src_dev,
        &mut l2_weights_sf,
    )
    .expect("L2 SF prepare");
    ctx.sync().expect("sync after transforms");

    // Localise a transform failure before the fused kernel hides it in `y`.
    {
        let got = ctx
            .stream
            .clone_dtoh(&l1_weights)
            .expect("download interleaved L1 weights");
        let span = TRANSFORM_EXPERTS * l1_n * (HIDDEN / 2);
        assert_bytes_eq(
            "L1 packed-FP4 gate/up interleave",
            &got[..span],
            &read_bytes(&dir, "l1_weight_interleaved_u8.bin"),
        );
    }
    for (what, dev, file, span) in [
        (
            "L1 scale factors (interleave + UTCCP transpose)",
            &l1_weights_sf,
            "l1_sf_mega_i32.bin",
            TRANSFORM_EXPERTS * (HIDDEN / SF_WORD_K) * l1_n,
        ),
        (
            "L2 scale factors (UTCCP transpose)",
            &l2_weights_sf,
            "l2_sf_mega_i32.bin",
            TRANSFORM_EXPERTS * (INTER / SF_WORD_K) * HIDDEN,
        ),
    ] {
        let got = ctx.stream.clone_dtoh(dev).expect("download mega SF");
        let got_bytes: Vec<u8> = got[..span].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_bytes_eq(what, &got_bytes, &read_bytes(&dir, file));
    }

    // --- symmetric buffer -------------------------------------------------
    // The slab takes the AOT protocol maximum, whatever the fixture's live
    // token count is — the launch rejects any other value.
    let max_tokens = k3_mega_max_tokens_per_rank();
    let layout = k3_mega_symm_buffer_layout(1, EXPERTS, max_tokens, TOPK, HIDDEN, INTER, num_sms)
        .expect("symmetric-buffer layout");
    let mut symm = ctx
        .stream
        .alloc_zeros::<u8>(layout.num_bytes)
        .expect("alloc symmetric buffer");

    let latent = ctx.stream.clone_htod(&x).expect("upload x");
    let idx_dev = ctx.stream.clone_htod(&topk_idx).expect("upload topk idx");
    let weight_dev = ctx
        .stream
        .clone_htod(&topk_weight)
        .expect("upload topk weights");
    let mut y = ctx
        .stream
        .alloc_zeros::<bf16>(TOKENS * HIDDEN)
        .expect("alloc y");

    let shape = K3MegaShape {
        num_tokens: TOKENS,
        num_max_tokens_per_rank: max_tokens,
        num_experts: EXPERTS,
        num_topk: TOPK,
        hidden: HIDDEN,
        intermediate_hidden: INTER,
        num_sms,
        num_ranks: 1,
        rank_idx: 0,
    };
    // Single-rank pointer table: `SymBuffer<1>::map` is the identity, so this
    // is just the slab's own base.
    let symm_ptrs = {
        let (base, _guard) = symm.device_ptr(&ctx.stream);
        [base as i64]
    };

    // The K3 activation is `situ`; `swiglu` is upstream's default and rides
    // along as a regression handle — if a future DeepGEMM bump moves the
    // shared machinery, both diverge, and if it moves only the K3 patch, one
    // does.
    for (activation, want) in [
        (K3MegaActivation::Situ, &want_y),
        (K3MegaActivation::Swiglu, &want_y_swiglu),
    ] {
        // The kernel consumes its input regions, so they are rewritten per
        // launch, exactly as the Python producer does.
        k3_mega_write_inputs_launch(
            &ctx,
            &layout,
            &mut symm,
            TOKENS,
            HIDDEN,
            TOPK,
            &latent,
            &idx_dev,
            &weight_dev,
        )
        .expect("write mega inputs");
        k3_mega_moe_launch(
            &ctx,
            &layout,
            &mut symm,
            &symm_ptrs,
            shape,
            activation,
            &l1_weights,
            &l1_weights_sf,
            &l2_weights,
            &l2_weights_sf,
            &mut y,
        )
        .expect("mega launch");
        ctx.sync().expect("sync after mega launch");

        let got_y = ctx.stream.clone_dtoh(&y).expect("download y");
        let got_bytes: Vec<u8> = got_y
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let want_bytes: Vec<u8> = want
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        assert_bytes_eq(
            &format!("MegaMoE({activation:?}) output y"),
            &got_bytes,
            &want_bytes,
        );
    }
}
