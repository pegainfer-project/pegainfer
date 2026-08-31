//! FlashKDA segment-length throughput sweep.
//!
//! Sizes the local-kernel shape of the CP axis: a whale chunk split `c` ways
//! hands each rank a `chunk/c`-token FlashKDA call per KDA layer, and this
//! sweep measures where the per-call efficiency cliff sits. The small-T rows
//! (14/224) also price the verify step's per-(slot, segment) launches against
//! one merged-sequence call. That comparison is only a launch-pressure proxy,
//! not a varlen result: real varlen keeps independent recurrent states.
//!
//! Not a gate — run explicitly on an idle GPU:
//! `cargo test --release -p pegainfer-kernels --features k3 \
//!    --test k3_flash_kda_bench -- --ignored --nocapture`
//!
//! Operand sets rotate round-robin over >= `L2_BUST_BYTES` so every timed
//! call is L2-cold on its tensor operands, mirroring prefill (each layer's
//! activations pass through once per chunk).
//!
//! Reference run (2026-08-25, one idle GB300, sm_103a, driver-visible bare
//! host) — for checking a rerun without a fleet:
//!
//! ```text
//!       T    ms/call     Mtok/s  us/16tok
//!      14     0.0259       0.54    29.624
//!      64     0.0267       2.40     6.664
//!     132     0.0378       3.49     4.587
//!     224     0.0458       4.89     3.273
//!     264     0.0543       4.86     3.292
//!     528     0.0811       6.51     2.457
//!    1056     0.1428       7.40     2.163
//!    2112     0.2648       7.97     2.006
//!    4224     0.5076       8.32     1.923
//!    8448     0.9937       8.50     1.882
//!   16896     1.9695       8.58     1.865
//!   33792     3.9191       8.62     1.856
//!   67584     7.8210       8.64     1.852
//!  135168    15.6235       8.65     1.849
//!  270336    31.2152       8.66     1.847
//! ```
//!
//! Shape of the curve: throughput plateaus at ~8.6 Mtok/s from ~4k and stays
//! perfectly linear through 264k — the chunkwise recurrence has no long-T
//! cliff, so the efficiency question is only ever about short segments. The
//! per-rank cliff sits below ~2k tokens: local-shape eff 96% at 4224, 93% at
//! 2112, 87% at 1056, 58% at 264; a 270336-token chunk split 16 ways is
//! still 99.1%. Launch-pressure proxy: 16 x T=14 calls = 9.05x one T=224
//! call.

#![cfg(feature = "k3")]

mod common;

use std::time::Instant;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3FlashKdaSpan;
use pegainfer_kernels::ops::k3_flash_kda_fwd_launch;
use pegainfer_kernels::ops::k3_flash_kda_workspace_bytes;
use pegainfer_kernels::tensor::DeviceContext;

const FULL_HEADS: usize = 96;
const D: usize = 128;
/// Rotation footprint target; GB300 L2 is ~126 MiB, 4x that to be sure.
const L2_BUST_BYTES: usize = 512 << 20;
const SCALE: f32 = 0.088_388_35; // 128^-0.5
const LOWER_BOUND: f32 = -5.0;

struct OperandSet {
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    g: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    beta: CudaSlice<bf16>,
    beta_scratch: CudaSlice<bf16>,
    a_log: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    state_in: CudaSlice<f32>,
    state_out: CudaSlice<f32>,
    workspace: CudaSlice<u8>,
}

fn xorshift_fill(seed: &mut u64, len: usize, amp: f32) -> Vec<bf16> {
    (0..len)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let unit = (*seed >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            bf16::from_f32((unit - 0.5) * 2.0 * amp)
        })
        .collect()
}

fn build_set(ctx: &DeviceContext, t: usize, heads: usize, seed: &mut u64) -> OperandSet {
    let width = heads * D;
    let wide = xorshift_fill(seed, t * width, 0.5);
    let ws_bytes = k3_flash_kda_workspace_bytes(t, heads);
    OperandSet {
        q: ctx.stream.clone_htod(&wide).expect("q"),
        k: ctx.stream.clone_htod(&wide[..t * width]).expect("k"),
        v: ctx
            .stream
            .clone_htod(&xorshift_fill(seed, t * width, 0.5))
            .expect("v"),
        g: ctx
            .stream
            .clone_htod(&xorshift_fill(seed, t * width, 1.0))
            .expect("g"),
        out: ctx.stream.alloc_zeros::<bf16>(t * width).expect("out"),
        beta: ctx
            .stream
            .clone_htod(&xorshift_fill(seed, t * heads, 1.0))
            .expect("beta"),
        beta_scratch: ctx
            .stream
            .alloc_zeros::<bf16>(t * heads)
            .expect("beta scratch"),
        a_log: ctx.stream.clone_htod(&vec![0.5f32; heads]).expect("a_log"),
        dt_bias: ctx.stream.alloc_zeros::<f32>(width).expect("dt_bias"),
        state_in: ctx
            .stream
            .alloc_zeros::<f32>(heads * D * D)
            .expect("state in"),
        state_out: ctx
            .stream
            .alloc_zeros::<f32>(heads * D * D)
            .expect("state out"),
        workspace: ctx
            .stream
            .alloc_zeros::<u8>(ws_bytes.max(16))
            .expect("workspace"),
    }
}

fn launch(ctx: &DeviceContext, t: usize, heads: usize, set: &mut OperandSet) {
    k3_flash_kda_fwd_launch(
        ctx,
        t,
        heads,
        K3FlashKdaSpan::default(),
        &set.q,
        &set.k,
        &set.v,
        &set.g,
        &set.beta,
        &mut set.beta_scratch,
        &set.a_log,
        &set.dt_bias,
        &set.state_in,
        &mut set.state_out,
        &mut set.out,
        &mut set.workspace,
        SCALE,
        LOWER_BOUND,
    )
    .expect("FlashKDA forward");
}

fn time_calls(
    ctx: &DeviceContext,
    t: usize,
    heads: usize,
    sets: &mut [OperandSet],
    iters: usize,
) -> f64 {
    let start = Instant::now();
    for i in 0..iters {
        let n = sets.len();
        launch(ctx, t, heads, &mut sets[i % n]);
        if i % 128 == 127 {
            ctx.stream.synchronize().expect("mid sync");
        }
    }
    ctx.stream.synchronize().expect("final sync");
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn build_sets(ctx: &DeviceContext, t: usize, heads: usize, seed: &mut u64) -> Vec<OperandSet> {
    let width = heads * D;
    let set_bytes = 5 * t * width * 2
        + 2 * t * heads * 2
        + 2 * heads * D * D * 4
        + k3_flash_kda_workspace_bytes(t, heads);
    // A set at least as large as the rotation target self-evicts within one
    // call, so a single copy is already L2-cold call-to-call; rotation only
    // buys anything for sets smaller than that.
    let copies = if set_bytes >= L2_BUST_BYTES {
        1
    } else {
        (L2_BUST_BYTES / set_bytes).clamp(3, 48)
    };
    (0..copies)
        .map(|_| build_set(ctx, t, heads, seed))
        .collect()
}

#[test]
#[ignore = "throughput sweep, run explicitly on an idle GPU"]
fn flash_kda_segment_sweep() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let mut seed = 0x5eed_cafe_f00d_u64;

    // 14 = worst per-slot verify pack tail; 224 = 16 slots x 14 (varlen
    // comparison); 264..16896 = whale chunk / s for chunk sizes 4224-16896;
    // 33792..270336 extend the 4224-aligned ladder to ~264k tokens (past the
    // 256k context ceiling) so the single-GPU curve is checkable without a
    // fleet. T=270336 needs ~56 GiB (operands + workspace) — any idle GB300
    // fits it in one operand set.
    let ts = [
        14usize, 64, 132, 224, 264, 528, 1056, 2112, 4224, 8448, 16896, 33792, 67584, 135168,
        270336,
    ];

    eprintln!("FlashKDA fwd sweep: heads={FULL_HEADS} d={D} (bf16 in/out, f32 state)");
    eprintln!(
        "{:>7} {:>7} {:>10} {:>10} {:>9}",
        "T", "sets", "ms/call", "Mtok/s", "us/16tok"
    );
    let mut rows: Vec<(usize, f64)> = Vec::new();
    for &t in &ts {
        let mut sets = build_sets(&ctx, t, FULL_HEADS, &mut seed);
        let copies = sets.len();

        for i in 0..(2 * copies).max(20) {
            let n = sets.len();
            launch(&ctx, t, FULL_HEADS, &mut sets[i % n]);
        }
        ctx.stream.synchronize().expect("warmup sync");

        let est = time_calls(&ctx, t, FULL_HEADS, &mut sets, 20);
        let iters = ((400.0 / est) as usize).clamp(30, 3000);
        let ms = time_calls(&ctx, t, FULL_HEADS, &mut sets, iters);

        let mtok_s = t as f64 / ms / 1e3;
        let us_per_tile = ms * 1e3 / (t as f64 / 16.0);
        eprintln!("{t:>7} {copies:>7} {ms:>10.4} {mtok_s:>10.2} {us_per_tile:>9.3}");
        rows.push((t, ms));
        drop(sets);
    }

    let cost = |t: usize| rows.iter().find(|(rt, _)| *rt == t).map(|(_, ms)| *ms);
    if let (Some(c14), Some(c224)) = (cost(14), cost(224)) {
        eprintln!(
            "\nlaunch-pressure proxy (not varlen): 16 x T=14 calls = {:.3} ms vs one-sequence T=224 call = {:.3} ms ({:.2}x)",
            16.0 * c14,
            c224,
            16.0 * c14 / c224
        );
    }
    for chunk in [4224usize, 8448, 16896, 270336] {
        eprintln!(
            "\nCP local-shape split of a {chunk}-token whale chunk (not end-to-end KCP; ideal = cost(chunk)/c):"
        );
        if let Some(full) = cost(chunk) {
            for s in [1usize, 2, 4, 8, 16] {
                let seg = chunk / s;
                if let Some(c) = cost(seg) {
                    let eff = full / (s as f64) / c * 100.0;
                    eprintln!(
                        "  c={s:>2}: T={seg:>6} {c:>9.4} ms/rank, local-shape eff {eff:>5.1}%"
                    );
                }
            }
        }
    }
}
