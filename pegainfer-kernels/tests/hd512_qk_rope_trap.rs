//! The hd512 decode position trap. One trap per binary: `__trap()` leaves
//! the context in a sticky error state, so anything sharing the process
//! afterwards would fail for the wrong reason.
//!
//! Manual gate — CI compiles this but never runs it.

mod common;

use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::qk_norm_partial_rope_batched_decode_hd512_into;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
const ROTARY_DIM: usize = 128;
const EPS: f32 = 1e-6;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const Q_DIM: usize = NUM_Q_HEADS * HD;
const KV_DIM: usize = NUM_KV_HEADS * HD;
const SEQ_LEN: usize = 4;

/// The launch is async, so the trap may surface at the launcher or at the
/// next sync — either counts; a silent `Ok` does not.
#[test]
fn decode_position_trap_is_visible() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // Inputs are zeroed but every wrapper check must PASS so the launch
    // actually happens and the kernel traps: weights [512], tables 4 rows
    // (cos_max_pos 4), positions [0, 1, 5, 1] — token 1 reads row 5.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let mut k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let cos_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("sin alloc");
    let positions: CudaSlice<i32> = ctx.stream.clone_htod(&[0, 1, 5, 1]).expect("positions H2D");

    let res = qk_norm_partial_rope_batched_decode_hd512_into(
        ctx,
        &q,
        &mut q_out,
        &mut k,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        &positions,
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    );
    if res.is_ok() {
        assert!(
            ctx.sync().is_err(),
            "out-of-range position (5 >= cos_max_pos 4) must fail visibly"
        );
    }
}
