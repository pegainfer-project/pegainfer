//! Numerical gate for the full Kimi-K3 MoE decode expert chain: routing
//! metadata -> local gather + FP8 quant -> masked grouped FP8xFP4 GEMM (W13)
//! -> situ + FP8 requant -> masked grouped GEMM (W2) -> weighted combine.
//!
//! Manual gate: CI compiles this but never runs it. Run on a Blackwell box
//! (`--ignored`, and set `PEGAINFER_REQUIRE_GPU=1` to turn a missing device
//! into a failure rather than a skip).
//!
//! Reference strategy — each stage is checked against an f32 host reference
//! computed from *that stage's own device inputs*, so a failure localizes
//! instead of compounding:
//!
//! * routing metadata: exact integer equality against the host's own
//!   entry-order compaction;
//! * both quant stages: the dequantized fp8 must sit within one e4m3 step of
//!   the f32 value the kernel was asked to quantize, and the MN-major f32
//!   scales must be exactly the UE8M0 powers of two the packed i32 SFA
//!   claims — this is what pins the scale layout the GEMM's TMA reads;
//! * both GEMMs: an f32 dot product over the *dequantized* operands the GEMM
//!   actually consumed (fp8 activation x fp4 weight), so the only expected
//!   difference is accumulation order plus the single bf16 output round;
//! * combine: f32 accumulation in topk-slot order over the device's own W2
//!   output.
//!
//! FP4 packing convention under test: e2m1 nibbles K-major, even K in the low
//! nibble of each byte, odd K in the high nibble; UE8M0 exponent byte `b`
//! means `2^(b-127)` and covers 32 consecutive K elements. The test both packs
//! and dequantizes with this convention, so a mismatch with what DeepGEMM's
//! TMA/MMA path assumes shows up as a GEMM failure rather than as agreement.
//!
//! The expert bank is deliberately larger than the routed pool: experts
//! outside the pool get `masked_m == 0`, which exercises the masked
//! scheduler's empty-group path (and keeps the host reference affordable).

#![cfg(feature = "k3")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::K3DeepGemmFp8Fp4Kind;
use pegainfer_kernels::ops::K3MoeRouteShape;
use pegainfer_kernels::ops::k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch;
use pegainfer_kernels::ops::k3_fp4_sf_prepare_launch;
use pegainfer_kernels::ops::k3_fp8_scale_pack_ue8m0_launch;
use pegainfer_kernels::ops::k3_moe_gather_fp8_quant_masked_launch;
use pegainfer_kernels::ops::k3_moe_local_route_metadata_launch;
use pegainfer_kernels::ops::k3_moe_weighted_combine_launch;
use pegainfer_kernels::ops::k3_situ_and_mul_fp8_quant_masked_launch;

/// Rank-local expert groups: the smallest instantiated count.
const GROUPS: usize = 56;
/// Experts the router is allowed to pick, so the host reference only has to
/// dequantize a fraction of the bank.
const POOL: usize = 24;
const TOPK: usize = 16;
/// Token rows. Kept small because the host reference costs
/// `TOKENS * TOPK * (W13_N * HIDDEN + W2_N * INTER)` multiply-adds, but large
/// enough that `TOKENS * TOPK` exceeds the 256-block entry grid: the
/// grid-strided kernels then really loop, which is the only way the
/// skip-inactive-entry branch and the reduction's inter-iteration barrier get
/// exercised together.
const TOKENS: usize = 24;
const CAP: usize = 128;
const HIDDEN: usize = 3584;
const INTER: usize = 3072;
const W13_N: usize = 6144;
const QUANT_GROUP: usize = 128;
const FP4_SF_GROUP: usize = 32;

const E4M3_MAX: f32 = 448.0;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Uniform in (-1, 1).
    fn signed_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }
}

/// e4m3 (bias 7, 3 mantissa bits, subnormals down to 2^-9) -> f32.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((byte >> 3) & 0x0F);
    let mantissa = f32::from(byte & 0x07) / 8.0;
    if exponent == 0 {
        sign * mantissa * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + mantissa) * 2.0f32.powi(exponent - 7)
    }
}

/// One e2m1 nibble -> f32. Bit 3 is the sign; the magnitude grid is
/// {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
fn e2m1_to_f32(nibble: u8) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAGNITUDE[usize::from(nibble & 0x07)];
    if nibble & 0x08 != 0 { -value } else { value }
}

/// UE8M0 exponent byte -> f32 scale (`2^(b - 127)`).
fn ue8m0_to_f32(byte: u8) -> f32 {
    f32::from_bits(u32::from(byte) << 23)
}

/// The UE8M0 group scale the quant kernels must produce: the next power of two
/// at or above `max|x| / 448`.
fn expected_group_scale(values: impl Iterator<Item = f32>) -> f32 {
    let peak = values.fold(0.0f32, |acc, v| acc.max(v.abs()));
    let raw = peak.max(1.0e-10) / E4M3_MAX;
    f32::from_bits((raw.to_bits() + 0x007F_FFFF) & 0x7F80_0000)
}

/// The chain's certified activation, in f32 over the bf16 GEMM output.
fn situ(gate: f32, up: f32) -> f32 {
    4.0 * (gate * 0.25).tanh() * (1.0 / (1.0 + (-gate).exp())) * (25.0 * (up / 25.0).tanh())
}

/// Dequantize one K-major MXFP4 weight row into `out`.
fn dequant_fp4_row(packed: &[u8], scales: &[u8], out: &mut [f32]) {
    for (i, value) in out.iter_mut().enumerate() {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        *value = e2m1_to_f32(nibble) * ue8m0_to_f32(scales[i / FP4_SF_GROUP]);
    }
}

fn relative_l2(actual: &[f32], reference: &[f32]) -> f32 {
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    for (a, r) in actual.iter().zip(reference) {
        diff += f64::from(a - r) * f64::from(a - r);
        norm += f64::from(*r) * f64::from(*r);
    }
    (diff.sqrt() / norm.sqrt().max(1e-30)) as f32
}

/// Random MXFP4 expert bank. Only `POOL` experts are populated; the rest stay
/// zero and are never routed to. `sf_exponent` centres the weight magnitude so
/// the chain's intermediates stay in a sane bf16 range.
fn build_expert_bank(rng: &mut Lcg, n: usize, k: usize, sf_exponent: i32) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; GROUPS * n * k / 2];
    let mut scales = vec![0u8; GROUPS * n * (k / FP4_SF_GROUP)];
    let row_bytes = k / 2;
    let row_scales = k / FP4_SF_GROUP;
    for expert in 0..POOL {
        let packed_base = expert * n * row_bytes;
        let scale_base = expert * n * row_scales;
        for byte in &mut packed[packed_base..packed_base + n * row_bytes] {
            // Two independent e2m1 nibbles per byte.
            *byte = (rng.below(16) | (rng.below(16) << 4)) as u8;
        }
        for scale in &mut scales[scale_base..scale_base + n * row_scales] {
            // 2^(sf_exponent) with +-1 of jitter across K groups.
            *scale = (127 + sf_exponent + rng.below(3) as i32 - 1) as u8;
        }
    }
    (packed, scales)
}

#[test]
#[ignore = "needs a Blackwell (sm_100 family) GPU"]
fn k3_moe_decode_chain_matches_f32_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let num_sms = ctx
        .ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .expect("query SM count") as usize;
    if !matches!(num_sms, 148 | 152) {
        assert!(
            std::env::var("PEGAINFER_REQUIRE_GPU").as_deref() != Ok("1"),
            "PEGAINFER_REQUIRE_GPU=1 but the device has {num_sms} SMs; the K3 masked \
             FP8xFP4 GEMM is only instantiated for B200 (148) and GB300 (152)"
        );
        eprintln!("skipping: {num_sms} SMs, not a B200/GB300 instantiation");
        return;
    }

    let shape = K3MoeRouteShape {
        tokens: TOKENS,
        topk: TOPK,
        groups: GROUPS,
        masked_cap: CAP,
    };
    let entries = shape.entries();
    let masked_rows = shape.masked_rows();
    let mut rng = Lcg(0x4B33_5EED);

    // ---- inputs -----------------------------------------------------------
    let latent: Vec<bf16> = (0..TOKENS * HIDDEN)
        .map(|_| bf16::from_f32(rng.signed_unit()))
        .collect();

    // Distinct experts per token, drawn from the routed pool; two entries are
    // marked inactive to exercise the padded-row path.
    let mut topk_idx = vec![0i32; entries];
    for token in 0..TOKENS {
        let mut pool: Vec<i32> = (0..POOL as i32).collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, rng.below(i as u32 + 1) as usize);
        }
        topk_idx[token * TOPK..(token + 1) * TOPK].copy_from_slice(&pool[..TOPK]);
    }
    topk_idx[TOPK + 3] = -1;
    topk_idx[2 * TOPK + 9] = GROUPS as i32 + 5;
    let topk_weight: Vec<f32> = (0..entries)
        .map(|_| 0.05 + rng.signed_unit().abs())
        .collect();

    let (w13_packed, w13_sf) = build_expert_bank(&mut rng, W13_N, HIDDEN, -7);
    let (w2_packed, w2_sf) = build_expert_bank(&mut rng, HIDDEN, INTER, -7);

    // ---- device buffers ---------------------------------------------------
    let stream = &ctx.stream;
    let latent_dev = stream.clone_htod(&latent).expect("upload latent");
    let topk_idx_dev = stream.clone_htod(&topk_idx).expect("upload topk idx");
    let topk_weight_dev = stream.clone_htod(&topk_weight).expect("upload topk weight");
    let w13_dev = stream.clone_htod(&w13_packed).expect("upload w13");
    let w2_dev = stream.clone_htod(&w2_packed).expect("upload w2");
    let w13_sf_dev = stream.clone_htod(&w13_sf).expect("upload w13 sf");
    let w2_sf_dev = stream.clone_htod(&w2_sf).expect("upload w2 sf");

    let mut w13_sf_packed = stream
        .alloc_zeros::<i32>(GROUPS * (HIDDEN / (4 * FP4_SF_GROUP)) * W13_N)
        .expect("alloc w13 sfb");
    let mut w2_sf_packed = stream
        .alloc_zeros::<i32>(GROUPS * (INTER / (4 * FP4_SF_GROUP)) * HIDDEN)
        .expect("alloc w2 sfb");
    let mut masked_m = stream.alloc_zeros::<i32>(GROUPS).expect("alloc masked_m");
    let mut slot_map = stream.alloc_zeros::<i32>(entries).expect("alloc slot map");
    let mut a1 = stream
        .alloc_zeros::<u8>(masked_rows * HIDDEN)
        .expect("alloc w13 activation");
    let mut a1_scale = stream
        .alloc_zeros::<f32>(GROUPS * (HIDDEN / QUANT_GROUP) * CAP)
        .expect("alloc w13 act scales");
    let mut a1_scale_packed = stream
        .alloc_zeros::<i32>(GROUPS * (HIDDEN / (4 * QUANT_GROUP)) * CAP)
        .expect("alloc w13 sfa");
    let mut w13_out = stream
        .alloc_zeros::<bf16>(masked_rows * W13_N)
        .expect("alloc w13 out");
    let mut a2 = stream
        .alloc_zeros::<u8>(masked_rows * INTER)
        .expect("alloc w2 activation");
    let mut a2_scale = stream
        .alloc_zeros::<f32>(GROUPS * (INTER / QUANT_GROUP) * CAP)
        .expect("alloc w2 act scales");
    let mut a2_scale_packed = stream
        .alloc_zeros::<i32>(GROUPS * (INTER / (4 * QUANT_GROUP)) * CAP)
        .expect("alloc w2 sfa");
    let mut w2_out = stream
        .alloc_zeros::<bf16>(masked_rows * HIDDEN)
        .expect("alloc w2 out");
    let mut combined = stream
        .alloc_zeros::<bf16>(TOKENS * HIDDEN)
        .expect("alloc combined");

    // ---- loader-time weight scale repack ----------------------------------
    k3_fp4_sf_prepare_launch(&ctx, GROUPS, W13_N, HIDDEN, &w13_sf_dev, &mut w13_sf_packed)
        .expect("w13 sf prepare");
    k3_fp4_sf_prepare_launch(&ctx, GROUPS, HIDDEN, INTER, &w2_sf_dev, &mut w2_sf_packed)
        .expect("w2 sf prepare");

    // ---- the chain --------------------------------------------------------
    // Run it twice into the same buffers: the second pass must be bit-identical,
    // which is what the entry-order row assignment and the atomic-free combine
    // accumulation buy. Verification below runs on the second pass's buffers.
    let mut passes: Vec<Vec<bf16>> = Vec::new();
    for _ in 0..2 {
        k3_moe_local_route_metadata_launch(
            &ctx,
            shape,
            &topk_idx_dev,
            &mut masked_m,
            &mut slot_map,
        )
        .expect("route metadata");
        k3_moe_gather_fp8_quant_masked_launch(
            &ctx,
            shape,
            HIDDEN,
            &latent_dev,
            &topk_idx_dev,
            &slot_map,
            &mut a1,
            &mut a1_scale,
        )
        .expect("gather + w13 quant");
        k3_fp8_scale_pack_ue8m0_launch(
            &ctx,
            GROUPS,
            HIDDEN / QUANT_GROUP,
            CAP,
            &a1_scale,
            &mut a1_scale_packed,
        )
        .expect("w13 scale pack");
        k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch(
            &ctx,
            K3DeepGemmFp8Fp4Kind::W13,
            GROUPS,
            CAP,
            num_sms,
            &a1,
            &a1_scale_packed,
            &w13_dev,
            &w13_sf_packed,
            &masked_m,
            &mut w13_out,
        )
        .expect("w13 masked grouped gemm");
        k3_situ_and_mul_fp8_quant_masked_launch(
            &ctx,
            shape,
            INTER,
            &w13_out,
            &topk_idx_dev,
            &slot_map,
            &mut a2,
            &mut a2_scale,
        )
        .expect("situ + w2 quant");
        k3_fp8_scale_pack_ue8m0_launch(
            &ctx,
            GROUPS,
            INTER / QUANT_GROUP,
            CAP,
            &a2_scale,
            &mut a2_scale_packed,
        )
        .expect("w2 scale pack");
        k3_deepgemm_sm100_masked_grouped_fp8_fp4_launch(
            &ctx,
            K3DeepGemmFp8Fp4Kind::W2,
            GROUPS,
            CAP,
            num_sms,
            &a2,
            &a2_scale_packed,
            &w2_dev,
            &w2_sf_packed,
            &masked_m,
            &mut w2_out,
        )
        .expect("w2 masked grouped gemm");
        k3_moe_weighted_combine_launch(
            &ctx,
            shape,
            HIDDEN,
            &w2_out,
            &topk_idx_dev,
            &slot_map,
            &topk_weight_dev,
            &mut combined,
        )
        .expect("weighted combine");
        stream.synchronize().expect("chain sync");
        passes.push(stream.clone_dtoh(&combined).expect("download combined"));
    }
    let pass_bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<u16>>();
    assert_eq!(
        pass_bits(&passes[0]),
        pass_bits(&passes[1]),
        "two runs of the same chain must be bit-identical"
    );

    // ---- stage 1: routing metadata ----------------------------------------
    let masked_m_host = stream.clone_dtoh(&masked_m).expect("download masked_m");
    let slot_map_host = stream.clone_dtoh(&slot_map).expect("download slot map");

    let mut expected_counts = vec![0i32; GROUPS];
    let mut expected_slots = vec![-1i32; entries];
    let mut rows_by_expert: Vec<Vec<usize>> = vec![Vec::new(); GROUPS];
    for (entry, &routed) in topk_idx.iter().enumerate() {
        if routed < 0 || routed >= GROUPS as i32 {
            continue;
        }
        let expert = routed as usize;
        expected_slots[entry] = (expert * CAP) as i32 + expected_counts[expert];
        expected_counts[expert] += 1;
        rows_by_expert[expert].push(entry);
    }
    assert_eq!(masked_m_host, expected_counts, "per-expert row counts");
    assert_eq!(slot_map_host, expected_slots, "entry -> masked slot map");
    let active: Vec<usize> = (0..entries)
        .filter(|&entry| expected_slots[entry] >= 0)
        .collect();
    assert_eq!(
        active.len(),
        entries - 2,
        "two entries were marked inactive"
    );

    // ---- stage 2: gather + W13 activation quant ---------------------------
    let a1_host = stream.clone_dtoh(&a1).expect("download w13 activation");
    let a1_scale_host = stream.clone_dtoh(&a1_scale).expect("download w13 scales");
    let a1_packed_host = stream
        .clone_dtoh(&a1_scale_packed)
        .expect("download w13 sfa");

    // Dequantized W13 A rows, keyed by entry, in the layout the GEMM reads.
    let mut a1_deq = vec![0.0f32; entries * HIDDEN];
    for &entry in &active {
        let slot = expected_slots[entry] as usize;
        let expert = slot / CAP;
        let row = slot % CAP;
        let token = entry / TOPK;
        for group in 0..HIDDEN / QUANT_GROUP {
            let expected = expected_group_scale(
                (0..QUANT_GROUP).map(|i| latent[token * HIDDEN + group * QUANT_GROUP + i].to_f32()),
            );
            let scale = a1_scale_host[(expert * (HIDDEN / QUANT_GROUP) + group) * CAP + row];
            assert_eq!(
                scale.to_bits(),
                expected.to_bits(),
                "W13 activation UE8M0 scale at entry {entry}, K group {group}"
            );
            for i in 0..QUANT_GROUP {
                let column = group * QUANT_GROUP + i;
                let value = e4m3_to_f32(a1_host[slot * HIDDEN + column]) * scale;
                let want = latent[token * HIDDEN + column].to_f32();
                assert!(
                    (value - want).abs() <= 0.0626 * want.abs() + scale * 2.0f32.powi(-10),
                    "W13 activation quant at entry {entry}, column {column}: {value} vs {want}"
                );
                a1_deq[entry * HIDDEN + column] = value;
            }
        }
    }
    check_packed_scales(
        &a1_scale_host,
        &a1_packed_host,
        HIDDEN / QUANT_GROUP,
        "W13 activation",
    );

    // ---- stage 3: W13 masked grouped GEMM ---------------------------------
    let w13_host = stream.clone_dtoh(&w13_out).expect("download w13 out");
    let mut w13_ref = vec![0.0f32; entries * W13_N];
    let mut weight_row = vec![0.0f32; HIDDEN];
    for (expert, rows) in rows_by_expert.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        for n in 0..W13_N {
            let packed_base = (expert * W13_N + n) * (HIDDEN / 2);
            let scale_base = (expert * W13_N + n) * (HIDDEN / FP4_SF_GROUP);
            dequant_fp4_row(
                &w13_packed[packed_base..packed_base + HIDDEN / 2],
                &w13_sf[scale_base..scale_base + HIDDEN / FP4_SF_GROUP],
                &mut weight_row,
            );
            for &entry in rows {
                let a = &a1_deq[entry * HIDDEN..(entry + 1) * HIDDEN];
                let mut acc = 0.0f32;
                for (x, w) in a.iter().zip(&weight_row) {
                    acc = x.mul_add(*w, acc);
                }
                w13_ref[entry * W13_N + n] = acc;
            }
        }
    }
    let mut w13_actual = Vec::with_capacity(active.len() * W13_N);
    let mut w13_expected = Vec::with_capacity(active.len() * W13_N);
    for &entry in &active {
        let slot = expected_slots[entry] as usize;
        for n in 0..W13_N {
            w13_actual.push(w13_host[slot * W13_N + n].to_f32());
            w13_expected.push(w13_ref[entry * W13_N + n]);
        }
    }
    let w13_rel = relative_l2(&w13_actual, &w13_expected);
    assert!(
        w13_rel < 1.0e-2,
        "W13 masked grouped GEMM rel_l2 {w13_rel} against the dequantized f32 reference"
    );

    // ---- stage 4: situ + W2 activation quant ------------------------------
    let a2_host = stream.clone_dtoh(&a2).expect("download w2 activation");
    let a2_scale_host = stream.clone_dtoh(&a2_scale).expect("download w2 scales");
    let a2_packed_host = stream
        .clone_dtoh(&a2_scale_packed)
        .expect("download w2 sfa");
    let mut a2_deq = vec![0.0f32; entries * INTER];
    for &entry in &active {
        let slot = expected_slots[entry] as usize;
        let expert = slot / CAP;
        let row = slot % CAP;
        let activated: Vec<f32> = (0..INTER)
            .map(|column| {
                let gate = w13_host[slot * W13_N + column].to_f32();
                let up = w13_host[slot * W13_N + INTER + column].to_f32();
                situ(gate, up)
            })
            .collect();
        for group in 0..INTER / QUANT_GROUP {
            let slice = &activated[group * QUANT_GROUP..(group + 1) * QUANT_GROUP];
            let expected = expected_group_scale(slice.iter().copied());
            let scale = a2_scale_host[(expert * (INTER / QUANT_GROUP) + group) * CAP + row];
            // The device and host transcendentals can disagree by an ulp, so
            // the group peak — and with it the power-of-two scale — is allowed
            // to land one binade away; the value check below stays tight.
            let ratio = scale / expected;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "W2 activation UE8M0 scale at entry {entry}, K group {group}: {scale} vs {expected}"
            );
            for (i, &want) in slice.iter().enumerate() {
                let column = group * QUANT_GROUP + i;
                let value = e4m3_to_f32(a2_host[slot * INTER + column]) * scale;
                assert!(
                    (value - want).abs() <= 0.07 * want.abs() + scale * 2.0f32.powi(-10) + 1.0e-6,
                    "W2 activation quant at entry {entry}, column {column}: {value} vs {want}"
                );
                a2_deq[entry * INTER + column] = value;
            }
        }
    }
    check_packed_scales(
        &a2_scale_host,
        &a2_packed_host,
        INTER / QUANT_GROUP,
        "W2 activation",
    );

    // ---- stage 5: W2 masked grouped GEMM ----------------------------------
    let w2_host = stream.clone_dtoh(&w2_out).expect("download w2 out");
    let mut w2_ref = vec![0.0f32; entries * HIDDEN];
    let mut w2_weight_row = vec![0.0f32; INTER];
    for (expert, rows) in rows_by_expert.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        for n in 0..HIDDEN {
            let packed_base = (expert * HIDDEN + n) * (INTER / 2);
            let scale_base = (expert * HIDDEN + n) * (INTER / FP4_SF_GROUP);
            dequant_fp4_row(
                &w2_packed[packed_base..packed_base + INTER / 2],
                &w2_sf[scale_base..scale_base + INTER / FP4_SF_GROUP],
                &mut w2_weight_row,
            );
            for &entry in rows {
                let a = &a2_deq[entry * INTER..(entry + 1) * INTER];
                let mut acc = 0.0f32;
                for (x, w) in a.iter().zip(&w2_weight_row) {
                    acc = x.mul_add(*w, acc);
                }
                w2_ref[entry * HIDDEN + n] = acc;
            }
        }
    }
    let mut w2_actual = Vec::with_capacity(active.len() * HIDDEN);
    let mut w2_expected = Vec::with_capacity(active.len() * HIDDEN);
    for &entry in &active {
        let slot = expected_slots[entry] as usize;
        for n in 0..HIDDEN {
            w2_actual.push(w2_host[slot * HIDDEN + n].to_f32());
            w2_expected.push(w2_ref[entry * HIDDEN + n]);
        }
    }
    let w2_rel = relative_l2(&w2_actual, &w2_expected);
    assert!(
        w2_rel < 1.0e-2,
        "W2 masked grouped GEMM rel_l2 {w2_rel} against the dequantized f32 reference"
    );

    // ---- stage 6: weighted combine ----------------------------------------
    let combined_host = stream.clone_dtoh(&combined).expect("download combined");
    let mut combine_ref = vec![0.0f32; TOKENS * HIDDEN];
    for token in 0..TOKENS {
        for slot_index in 0..TOPK {
            let entry = token * TOPK + slot_index;
            if expected_slots[entry] < 0 {
                continue;
            }
            let slot = expected_slots[entry] as usize;
            let weight = topk_weight[entry];
            for column in 0..HIDDEN {
                combine_ref[token * HIDDEN + column] = weight.mul_add(
                    w2_host[slot * HIDDEN + column].to_f32(),
                    combine_ref[token * HIDDEN + column],
                );
            }
        }
    }
    let combine_actual: Vec<f32> = combined_host.iter().map(|v| v.to_f32()).collect();
    let combine_rel = relative_l2(&combine_actual, &combine_ref);
    assert!(
        combine_rel < 5.0e-3,
        "weighted combine rel_l2 {combine_rel} (only the final bf16 round should differ)"
    );

    eprintln!(
        "k3 moe chain: w13 rel_l2 {w13_rel:.3e}, w2 rel_l2 {w2_rel:.3e}, combine rel_l2 {combine_rel:.3e}"
    );
}

/// The packed i32 SFA must be exactly the MN-major f32 scales' exponent bytes,
/// four per word, LSB first. This is the layout the GEMM's SFA TMA reads, so a
/// silent transpose here would only show up as a wrong GEMM result.
fn check_packed_scales(scales: &[f32], packed: &[i32], scale_cols: usize, label: &str) {
    let packed_cols = scale_cols / 4;
    for expert in 0..GROUPS {
        for word in 0..packed_cols {
            for row in 0..CAP {
                let mut expected = 0u32;
                for j in 0..4 {
                    let column = word * 4 + j;
                    let bits = scales[(expert * scale_cols + column) * CAP + row].to_bits();
                    expected |= ((bits >> 23) & 0xFF) << (8 * j);
                }
                let actual = packed[(expert * packed_cols + word) * CAP + row] as u32;
                assert_eq!(
                    actual, expected,
                    "{label} packed SFA word at expert {expert}, word {word}, row {row}"
                );
            }
        }
    }
}
