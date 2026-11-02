//! Shared device gate for the kernel GPU gate binaries.

use pegainfer_kernels::tensor::DeviceContext;

/// Never infer "no device" from an arbitrary error — a broken driver, a
/// failed stream or a context poisoned by an earlier `__trap()` all look
/// like one. The context attempt comes first because it also initialises
/// the driver; only its failure is diagnosed by `get_count()`, which is
/// itself ambiguous between "no driver" and "broken driver". A formal gate
/// must therefore set `PEGAINFER_REQUIRE_GPU=1` so skipping is impossible.
pub(crate) fn device_or_skip() -> Option<DeviceContext> {
    match DeviceContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            let count = cudarc::driver::result::device::get_count();
            match count {
                Err(_) | Ok(0) => {
                    assert!(
                        std::env::var("PEGAINFER_REQUIRE_GPU").as_deref() != Ok("1"),
                        "PEGAINFER_REQUIRE_GPU=1 but no usable CUDA device \
                         (context error: {e}; get_count: {count:?})"
                    );
                    eprintln!("skipping: no usable CUDA device (get_count: {count:?})");
                    None
                }
                Ok(n) => panic!(
                    "CUDA device present (get_count = {n}) but context creation \
                     failed: {e}. This is a broken environment or a poisoned \
                     context, not a missing device, and must not be skipped"
                ),
            }
        }
    }
}

/// The device's SM count, queried once by the gate that hands it to a launch.
/// Wrappers take it as an argument so no launch queries the driver.
#[allow(dead_code)]
pub(crate) fn sm_count(ctx: &DeviceContext) -> usize {
    usize::try_from(
        ctx.ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .expect("SM count"),
    )
    .expect("SM count fits usize")
}

/// Deterministic junk, distinct per element, so a row read from the wrong
/// page, layer or head cannot coincide with the right one. Not every binary
/// that includes this module fills a buffer.
#[allow(dead_code)]
pub(crate) fn fill(seed: u64, n: usize) -> Vec<half::bf16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            half::bf16::from_f32(((state >> 40) as f32 / 8_388_608.0) - 1.0)
        })
        .collect()
}

/// The attention line's cos/sin tables, built like
/// `pegainfer_core::rope::RopeTableSpec::tables` with `frequency_dim` equal to
/// `rotary_dim`: `[max_pos, rotary_dim]`, position-major, half-split
/// duplicated, bf16. The gates build them here because `pegainfer-kernels`
/// cannot depend on `pegainfer-core` (the crate dependency runs the other way);
/// production builds the same table and passes it in, which is why the QSA
/// kernels read it instead of computing trig.
#[allow(dead_code)]
pub(crate) fn rope_tables(
    rotary_dim: usize,
    theta: f32,
    max_pos: usize,
) -> (Vec<half::bf16>, Vec<half::bf16>) {
    assert!(rotary_dim > 0 && rotary_dim.is_multiple_of(2) && max_pos > 0);
    let half_dim = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32))
        .collect();
    let mut cos = vec![half::bf16::ZERO; max_pos * rotary_dim];
    let mut sin = vec![half::bf16::ZERO; max_pos * rotary_dim];
    for pos in 0..max_pos {
        let base = pos * rotary_dim;
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            let c = half::bf16::from_f32(angle.cos());
            let s = half::bf16::from_f32(angle.sin());
            cos[base + i] = c;
            cos[base + i + half_dim] = c;
            sin[base + i] = s;
            sin[base + i + half_dim] = s;
        }
    }
    (cos, sin)
}

/// The largest absolute difference and where it is. A NaN compares false
/// against everything, so it would leave the largest at zero: it fails here
/// instead of passing a comparison silently.
#[allow(dead_code)]
pub(crate) fn worst_delta(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "compared runs differ in length");
    let mut worst = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        assert!(d.is_finite(), "element {i}: {x} against {y}");
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}
