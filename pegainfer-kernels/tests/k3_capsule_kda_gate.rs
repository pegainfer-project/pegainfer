//! Numeric gate for the capsule-vendored vLLM fused KDA decode
//! (`cubin/k3/kda_decode_fusion_h96_sm103.cubin`) against the native
//! conv_silu x3 + kda_core chain on identical inputs and states.
//!
//! Both sides spell the same math (short conv + silu, joint q/k L2 norm,
//! `exp(-5 * sigmoid(exp(a_log) * (g + dt_bias)))` decay, sigmoid-beta delta
//! rule, gated output RMS norm with eps 1e-5) but round differently: the
//! native TileLang chain lands the conv output, the rsqrt and several other
//! intermediates in bf16, the vLLM kernel keeps them f32. The contract under
//! test is therefore layout + semantics, not bitwise identity:
//!
//! * conv windows must match **bitwise** — the shift is a copy and the newest
//!   slot is the same bf16 landing on both sides;
//! * the updated recurrent state and the output row must agree within a
//!   rounding-chain tolerance (a layout or indexing mistake produces O(1)
//!   garbage, orders of magnitude beyond it).
//!
//! Manual gate: CI compiles this but never runs it. Run on a Blackwell box
//! (set `PEGAINFER_REQUIRE_GPU=1` to turn a missing device into a failure).

#![cfg(feature = "k3")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::K3_CAPSULE_CONV_SLOT;
use pegainfer_kernels::ops::K3_CAPSULE_STATE_SLOT;
use pegainfer_kernels::ops::K3_CAPSULE_X_ROW;
use pegainfer_kernels::ops::K3_CONV_WIDTH;
use pegainfer_kernels::ops::K3_KDA_DIM;
use pegainfer_kernels::ops::K3_KDA_HEAD_DIM;
use pegainfer_kernels::ops::K3_KDA_HEADS;
use pegainfer_kernels::ops::k3_capsule_kda_decode_launch;
use pegainfer_kernels::ops::k3_conv_silu_batched_launch;
use pegainfer_kernels::ops::k3_kda_core_batched_launch;
use pegainfer_kernels::ops::k3_land_batched_launch;
use pegainfer_kernels::tensor::DeviceContext;

const KP: usize = K3_KDA_DIM;
const WS: usize = K3_CONV_WIDTH - 1;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn f32s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit_f32() * scale).collect()
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<bf16> {
        (0..n)
            .map(|_| bf16::from_f32(self.unit_f32() * scale))
            .collect()
    }
}

/// Stream-ordered d2d copy of `elems` elements between offsets of two bf16
/// slices (the packed-layout assembly the executor does with the same call).
fn copy_bf16(
    ctx: &DeviceContext,
    src: &cudarc::driver::CudaSlice<bf16>,
    src_off: usize,
    dst: &mut cudarc::driver::CudaSlice<bf16>,
    dst_off: usize,
    elems: usize,
) {
    use cudarc::driver::DevicePtr;
    use cudarc::driver::DevicePtrMut;
    let (src_ptr, _sg) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _dg) = dst.device_ptr_mut(&ctx.stream);
    let e = size_of::<bf16>();
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst_ptr + (dst_off * e) as u64,
            src_ptr + (src_off * e) as u64,
            elems * e,
            pegainfer_kernels::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .expect("d2d copy");
}

#[test]
fn capsule_kda_matches_native_chain() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut rng = Lcg(0x4b33_4b44_2026_0828);

    // Weights, shared across the batch sweep.
    let cw: Vec<_> = (0..3)
        .map(|_| {
            ctx.stream
                .clone_htod(&rng.f32s(K3_CONV_WIDTH * KP, 0.4))
                .expect("cw H2D")
        })
        .collect();
    let dt_dev = ctx.stream.clone_htod(&rng.f32s(KP, 0.2)).expect("dt H2D");
    let alog_dev = ctx
        .stream
        .clone_htod(&rng.f32s(K3_KDA_HEADS, 0.5))
        .expect("alog H2D");
    let go: Vec<f32> = rng
        .f32s(K3_KDA_HEAD_DIM, 0.3)
        .iter()
        .map(|v| 1.0 + v)
        .collect();
    let go_dev = ctx.stream.clone_htod(&go).expect("go H2D");

    for &b in &[1usize, 4] {
        // Per-stream f32 projection partials and carried conv windows.
        let partials: Vec<_> = (0..3)
            .map(|_| {
                ctx.stream
                    .clone_htod(&rng.f32s(b * KP, 1.0))
                    .expect("p H2D")
            })
            .collect();
        let windows: Vec<_> = (0..3)
            .map(|_| {
                ctx.stream
                    .clone_htod(&rng.bf16s(b * WS * KP, 1.0))
                    .expect("cs H2D")
            })
            .collect();
        let forget = ctx.stream.clone_htod(&rng.f32s(b * KP, 1.0)).expect("gp");
        let beta_dev = ctx
            .stream
            .clone_htod(&rng.bf16s(b * K3_KDA_HEADS, 1.5))
            .expect("beta");
        let g2_dev = ctx.stream.clone_htod(&rng.bf16s(b * KP, 1.0)).expect("g2");
        let state_host = rng.f32s(b * K3_CAPSULE_STATE_SLOT, 0.05);
        let state_dev = ctx.stream.clone_htod(&state_host).expect("state H2D");

        // Native chain: three conv streams, then the core with a distinct
        // successor state.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut sns = Vec::new();
        for s in 0..3 {
            let mut x = ctx.stream.alloc_zeros::<bf16>(b * KP).expect("x");
            let mut y = ctx.stream.alloc_zeros::<bf16>(b * KP).expect("y");
            let mut sn = ctx.stream.alloc_zeros::<bf16>(b * WS * KP).expect("sn");
            k3_conv_silu_batched_launch(
                &ctx,
                b,
                KP,
                K3_CONV_WIDTH,
                1,
                &partials[s],
                &cw[s],
                &windows[s],
                &mut x,
                &mut y,
                &mut sn,
            )
            .expect("native conv_silu");
            xs.push(x);
            ys.push(y);
            sns.push(sn);
        }
        let mut state_n = ctx
            .stream
            .alloc_zeros::<f32>(b * K3_CAPSULE_STATE_SLOT)
            .expect("state_n");
        let mut out_native = ctx.stream.alloc_zeros::<bf16>(b * KP).expect("out");
        k3_kda_core_batched_launch(
            &ctx,
            b,
            K3_KDA_HEADS,
            K3_KDA_HEAD_DIM,
            1,
            &ys[0],
            &ys[1],
            &ys[2],
            &forget,
            &dt_dev,
            &alog_dev,
            &beta_dev,
            &g2_dev,
            &go_dev,
            &state_dev,
            &mut state_n,
            &mut out_native,
        )
        .expect("native kda_core");

        // Capsule side: packed x rows, packed conv slab, bf16 forget landing,
        // fresh copy of the same initial state (updated in place).
        let mut x_packed = ctx
            .stream
            .alloc_zeros::<bf16>(b * K3_CAPSULE_X_ROW)
            .expect("x_packed");
        for (s, x) in xs.iter().enumerate() {
            for row in 0..b {
                copy_bf16(
                    &ctx,
                    x,
                    row * KP,
                    &mut x_packed,
                    row * K3_CAPSULE_X_ROW + s * KP,
                    KP,
                );
            }
        }
        let mut conv_packed = ctx
            .stream
            .alloc_zeros::<bf16>(b * K3_CAPSULE_CONV_SLOT)
            .expect("conv_packed");
        for (s, w) in windows.iter().enumerate() {
            for row in 0..b {
                for t in 0..WS {
                    copy_bf16(
                        &ctx,
                        w,
                        (row * WS + t) * KP,
                        &mut conv_packed,
                        row * K3_CAPSULE_CONV_SLOT + (t * 3 + s) * KP,
                        KP,
                    );
                }
            }
        }
        let mut g_bf16 = ctx.stream.alloc_zeros::<bf16>(b * KP).expect("g_bf16");
        k3_land_batched_launch(&ctx, b, KP, KP, 0, 1, &forget, &mut g_bf16).expect("g land");
        let mut state_capsule = ctx.stream.clone_htod(&state_host).expect("state2 H2D");
        let mut out_capsule = ctx.stream.alloc_zeros::<bf16>(b * KP).expect("out2");
        k3_capsule_kda_decode_launch(
            &ctx,
            b,
            &x_packed,
            &cw[0],
            &cw[1],
            &cw[2],
            &mut conv_packed,
            &alog_dev,
            &g_bf16,
            &dt_dev,
            &beta_dev,
            &g2_dev,
            &go_dev,
            &mut state_capsule,
            &mut out_capsule,
        )
        .expect("capsule kda launch");

        // Conv windows: the shifted slots and the shared bf16 x landing must
        // match bitwise, per stream.
        let packed = ctx.stream.clone_dtoh(&conv_packed).expect("conv D2H");
        for (s, sn) in sns.iter().enumerate() {
            let native = ctx.stream.clone_dtoh(sn).expect("sn D2H");
            for row in 0..b {
                for t in 0..WS {
                    let nat = &native[(row * WS + t) * KP..(row * WS + t + 1) * KP];
                    let cap_at = row * K3_CAPSULE_CONV_SLOT + (t * 3 + s) * KP;
                    let cap = &packed[cap_at..cap_at + KP];
                    assert_eq!(
                        nat, cap,
                        "b={b} stream {s} row {row} tap {t} window diverges"
                    );
                }
            }
        }

        // Recurrent state and output: rounding-chain tolerance.
        let sn_native = ctx.stream.clone_dtoh(&state_n).expect("state D2H");
        let sn_capsule = ctx.stream.clone_dtoh(&state_capsule).expect("state2 D2H");
        let mut max_state = 0.0f32;
        for (i, (&n, &c)) in sn_native.iter().zip(&sn_capsule).enumerate() {
            let err = (n - c).abs() / n.abs().max(0.05);
            assert!(
                err < 0.08,
                "b={b} state[{i}]: native {n} vs capsule {c} (rel {err})"
            );
            max_state = max_state.max(err);
        }
        let out_n = ctx.stream.clone_dtoh(&out_native).expect("out D2H");
        let out_c = ctx.stream.clone_dtoh(&out_capsule).expect("out2 D2H");
        let mut max_out = 0.0f32;
        for (i, (&n, &c)) in out_n.iter().zip(&out_c).enumerate() {
            let (n, c) = (n.to_f32(), c.to_f32());
            let err = (n - c).abs() / n.abs().max(0.1);
            assert!(
                err < 0.08,
                "b={b} out[{i}]: native {n} vs capsule {c} (rel {err})"
            );
            max_out = max_out.max(err);
        }
        println!(
            "b={b}: conv windows bitwise-equal; max rel err state {max_state:.4}, out {max_out:.4}"
        );
    }
}
