//! Real AscendCL wrapper checks and isolated operator contracts.
//! Device tests are manual; CI builds this target without creating a device.
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_ascend::device::Device;
use pegainfer_ascend::ops;

fn fill(n: usize, seed: u32) -> Vec<bf16> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0;
            bf16::from_f32(v)
        })
        .collect()
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn buffers_roundtrip_and_reject_foreign_buffers() -> Result<()> {
    let device = Device::new(0)?;
    let other = Device::new(0)?;
    let data = fill(17, 1);
    let buffer = device.from_slice(&data)?;
    ensure!(buffer.read(17)? == data, "H2D/D2H roundtrip differs");
    ensure!(!other.owns(&buffer), "foreign buffer accepted");
    ensure!(device.zeros::<bf16>(0).is_err(), "zero allocation accepted");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn gemm_matches_host_reference() -> Result<()> {
    let device = Device::new(0)?;
    let (m, n, k) = (64usize, 128usize, 256usize);
    let x = fill(m * k, 11);
    let w = fill(k * n, 12);
    let dx = device.from_slice(&x)?;
    let dw = device.from_slice(&w)?;
    let mut dout = device.zeros::<bf16>(m * n)?;
    ops::gemm(&device, &dw, &dx, &mut dout, m, n, k)?;
    let out = dout.read(m * n)?;
    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += x[i * k + t].to_f32() * w[t * n + j].to_f32();
            }
            worst = worst.max((out[i * n + j].to_f32() - acc).abs());
        }
    }
    ensure!(worst < 0.5, "gemm max abs error {worst}");
    println!("gemm {m}x{n}x{k} bf16 max_abs={worst}");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn rms_norm_matches_host_reference() -> Result<()> {
    let device = Device::new(0)?;
    let (rows, h) = (4usize, 1024usize);
    let eps = 1e-6f32;
    let x = fill(rows * h, 21);
    let gamma = fill(h, 22);
    let dx = device.from_slice(&x)?;
    let dg = device.from_slice(&gamma)?;
    let mut dout = device.zeros::<bf16>(rows * h)?;
    ops::norm(&device, &dx, &dg, &mut dout, rows, h, eps)?;
    let out = dout.read(rows * h)?;
    let mut worst = 0.0f32;
    for r in 0..rows {
        let ss: f32 = (0..h).map(|i| x[r * h + i].to_f32().powi(2)).sum();
        let inv = 1.0 / (ss / h as f32 + eps).sqrt();
        for i in 0..h {
            let want = x[r * h + i].to_f32() * inv * gamma[i].to_f32();
            worst = worst.max((out[r * h + i].to_f32() - want).abs());
        }
    }
    ensure!(worst < 0.05, "rmsnorm max abs error {worst}");
    println!("rmsnorm {rows}x{h} bf16 max_abs={worst}");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn silu_mul_and_add_are_bf16() -> Result<()> {
    let device = Device::new(0)?;
    let n = 4096usize;
    let a = fill(n, 31);
    let b = fill(n, 32);
    let da = device.from_slice(&a)?;
    let db = device.from_slice(&b)?;
    let mut dout = device.zeros::<bf16>(n)?;
    ops::silu(&device, &da, &db, &mut dout, n)?;
    let out = dout.read(n)?;
    let mut worst = 0.0f32;
    for i in 0..n {
        let x = a[i].to_f32();
        let want = (x / (1.0 + (-x).exp())) * b[i].to_f32();
        worst = worst.max((out[i].to_f32() - want).abs());
    }
    ensure!(worst < 0.05, "silu*mul max abs error {worst}");
    ops::add(&device, &da, &db, &mut dout, n)?;
    let out = dout.read(n)?;
    for i in 0..n {
        let want = a[i].to_f32() + b[i].to_f32();
        ensure!((out[i].to_f32() - want).abs() < 0.05, "add differs at {i}");
    }
    println!("silu*mul/add {n} bf16 max_abs={worst}");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn gemm_qwen3_projection_shape() -> Result<()> {
    let device = Device::new(0)?;
    let (m, n, k) = (128usize, 2048usize, 1024usize);
    let x = fill(m * k, 41);
    let w = fill(k * n, 42);
    let dx = device.from_slice(&x)?;
    let dw = device.from_slice(&w)?;
    let mut dout = device.zeros::<bf16>(m * n)?;
    ops::gemm(&device, &dw, &dx, &mut dout, m, n, k)?;
    let out = dout.read(m * n)?;
    ensure!(out.iter().all(|v| v.to_f32().is_finite()), "nonfinite");
    println!("gemm qwen3 projection {m}x{n}x{k} ok");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn attention_reuses_cached_descriptors() -> Result<()> {
    let device = Device::new(0)?;
    let (heads, kv_heads, dim, context, capacity) = (16usize, 8usize, 128usize, 512usize, 1usize);
    let k = fill(2 * kv_heads * dim, 72);
    let v = fill(2 * kv_heads * dim, 73);
    let dk = device.from_slice(&k)?;
    let dv = device.from_slice(&v)?;
    let mut dkeys = device.zeros::<bf16>(capacity * context * kv_heads * dim)?;
    let mut dvals = device.zeros::<bf16>(capacity * context * kv_heads * dim)?;
    let positions = device.from_slice(&[0i32, 1i32])?;
    let cache_shape = ops::AttentionShape {
        rows: 2,
        heads,
        kv_heads,
        dim,
        context,
        capacity,
    };
    ops::cache(
        &device,
        &dk,
        &dv,
        &mut dkeys,
        &mut dvals,
        &[0, 0],
        &positions,
        cache_shape,
    )?;
    for pos in [0usize, 1usize] {
        let q = fill(heads * dim, 80 + pos as u32);
        let dq = device.from_slice(&q)?;
        let mut dout = device.zeros::<bf16>(heads * dim)?;
        let shape = ops::AttentionShape {
            rows: 1,
            heads,
            kv_heads,
            dim,
            context,
            capacity,
        };
        ops::attention(
            &device,
            &dq,
            &dkeys,
            &dvals,
            &mut dout,
            &[0],
            &[pos as i32],
            shape,
        )?;
        let out = dout.read(heads * dim)?;
        ensure!(out.iter().all(|x| x.to_f32().is_finite()), "nonfinite");
    }
    println!("attention twice with separate buffers ok");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn attention_matches_reference() -> Result<()> {
    let device = Device::new(0)?;
    let (heads, kv_heads, dim, context, capacity) = (16usize, 8usize, 128usize, 64usize, 1usize);
    let rows = 2usize;
    let q = fill(rows * heads * dim, 101);
    let k = fill(rows * kv_heads * dim, 102);
    let v = fill(rows * kv_heads * dim, 103);
    let dq = device.from_slice(&q)?;
    let dk = device.from_slice(&k)?;
    let dv = device.from_slice(&v)?;
    let mut dkeys = device.zeros::<bf16>(capacity * context * kv_heads * dim)?;
    let mut dvals = device.zeros::<bf16>(capacity * context * kv_heads * dim)?;
    let positions = device.from_slice(&[0i32, 1i32])?;
    let shape = ops::AttentionShape {
        rows,
        heads,
        kv_heads,
        dim,
        context,
        capacity,
    };
    ops::cache(
        &device,
        &dk,
        &dv,
        &mut dkeys,
        &mut dvals,
        &[0, 0],
        &positions,
        shape,
    )?;
    let mut dout = device.zeros::<bf16>(rows * heads * dim)?;
    ops::attention(
        &device,
        &dq,
        &dkeys,
        &dvals,
        &mut dout,
        &[0, 0],
        &[0, 1],
        shape,
    )?;
    let out = dout.read(rows * heads * dim)?;
    let group = heads / kv_heads;
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut worst = 0.0f32;
    let mut row_worst = vec![0.0f32; rows];
    for r in 0..rows {
        for n in 0..heads {
            let kvh = n / group;
            let mut logits = vec![0.0f32; r + 1];
            let mut mx = f32::NEG_INFINITY;
            for t in 0..=r {
                let mut acc = 0.0f32;
                for d in 0..dim {
                    acc += q[(r * heads + n) * dim + d].to_f32()
                        * k[(t * kv_heads + kvh) * dim + d].to_f32();
                }
                logits[t] = acc * scale;
                mx = mx.max(logits[t]);
            }
            let sum: f32 = logits.iter().map(|x| (x - mx).exp()).sum();
            for d in 0..dim {
                let mut acc = 0.0f32;
                for t in 0..=r {
                    acc +=
                        (logits[t] - mx).exp() / sum * v[(t * kv_heads + kvh) * dim + d].to_f32();
                }
                let e = (out[(r * heads + n) * dim + d].to_f32() - acc).abs();
                worst = worst.max(e);
                row_worst[r] = row_worst[r].max(e);
            }
        }
    }
    ensure!(
        worst < 0.05,
        "attention max abs error {worst} rows={row_worst:?}"
    );
    println!("attention matches reference: max_abs={worst}");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn rope_matches_reference_for_positions() -> Result<()> {
    let device = Device::new(0)?;
    let (rows, q_heads, kv_heads, dim, context) = (2usize, 2usize, 1usize, 128usize, 8usize);
    let theta = 1e6f32;
    let mut cos = vec![bf16::ZERO; context * dim];
    let mut sin = vec![bf16::ZERO; context * dim];
    for pos in 0..context {
        for d in 0..dim / 2 {
            let freq = 1.0 / (theta as f64).powf((2 * d) as f64 / dim as f64);
            let a = pos as f64 * freq;
            let c = bf16::from_f32(a.cos() as f32);
            let s = bf16::from_f32(a.sin() as f32);
            cos[pos * dim + d] = c;
            cos[pos * dim + d + dim / 2] = c;
            sin[pos * dim + d] = s;
            sin[pos * dim + d + dim / 2] = s;
        }
    }
    let cosr: Vec<f32> = cos.iter().map(|x| x.to_f32()).collect();
    let sinr: Vec<f32> = sin.iter().map(|x| x.to_f32()).collect();
    let q = fill(rows * q_heads * dim, 111);
    let k = fill(rows * kv_heads * dim, 112);
    let mut dq = device.from_slice(&q)?;
    let mut dk = device.from_slice(&k)?;
    let positions = device.from_slice(&[0i32, 1i32])?;
    let dc = device.from_slice(&cos)?;
    let ds = device.from_slice(&sin)?;
    ops::rope(
        &device, &mut dq, &mut dk, &positions, &dc, &ds, rows, q_heads, kv_heads, dim, context,
    )?;
    let qo = dq.read(rows * q_heads * dim)?;
    let ko = dk.read(rows * kv_heads * dim)?;
    let mut worst = 0.0f32;
    let mut row_worst = vec![0.0f32; rows];
    for pos in 0..rows {
        for h in 0..q_heads {
            for d in 0..dim {
                let x = q[(pos * q_heads + h) * dim + d].to_f32();
                let rot = if d < dim / 2 {
                    -q[(pos * q_heads + h) * dim + d + dim / 2].to_f32()
                } else {
                    q[(pos * q_heads + h) * dim + d - dim / 2].to_f32()
                };
                let want = x * cosr[pos * dim + d] + rot * sinr[pos * dim + d];
                let e = (qo[(pos * q_heads + h) * dim + d].to_f32() - want).abs();
                worst = worst.max(e);
                row_worst[pos] = row_worst[pos].max(e);
            }
        }
    }
    for pos in 0..rows {
        for d in 0..dim {
            let x = k[pos * kv_heads * dim + d].to_f32();
            let rot = if d < dim / 2 {
                -k[pos * kv_heads * dim + d + dim / 2].to_f32()
            } else {
                k[pos * kv_heads * dim + d - dim / 2].to_f32()
            };
            let want = x * cosr[pos * dim + d] + rot * sinr[pos * dim + d];
            worst = worst.max((ko[pos * kv_heads * dim + d].to_f32() - want).abs());
        }
    }
    ensure!(
        worst < 0.05,
        "rope max abs error {worst} rows={row_worst:?}"
    );
    println!("rope matches reference: max_abs={worst}");
    Ok(())
}

/// Regression for the executor ownership model: non-repeatable executors are
/// released by CANN, repeatable ones and cached descriptors are released when
/// the device drops. A second device after the first teardown proves the
/// cleanup did not finalize the process-wide ACL runtime.
#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn repeated_operators_survive_device_teardown() -> Result<()> {
    let (rows, h, n, vocab) = (8usize, 256usize, 512usize, 512usize);
    let x = fill(rows * h, 201);
    let w = fill(h * n, 202);
    let vec_a = fill(rows * h, 203);
    let vec_b = fill(rows * h, 204);
    let embed_w = fill(vocab * h, 205);
    let logits = fill(rows * vocab, 206);
    let ids = fill(rows, 207)
        .iter()
        .map(|v| (v.to_f32().abs() as u32) % vocab as u32)
        .collect::<Vec<_>>();
    let (heads, kv_heads, dim, context) = (4usize, 2usize, 128usize, 64usize);
    let attn_q = fill(heads * dim, 208);
    let attn_k = fill(kv_heads * dim, 209);
    let attn_v = fill(kv_heads * dim, 210);
    {
        let device = Device::new(0)?;
        let dx = device.from_slice(&x)?;
        let dw = device.from_slice(&w)?;
        let da = device.from_slice(&vec_a)?;
        let db = device.from_slice(&vec_b)?;
        let de = device.from_slice(&embed_w)?;
        let dids = device.from_slice(&ids)?;
        let dl = device.from_slice(&logits)?;
        let mut dgemm = device.zeros::<bf16>(rows * n)?;
        let mut dnorm = device.zeros::<bf16>(rows * h)?;
        let mut dsilu = device.zeros::<bf16>(rows * h)?;
        let mut dadd = device.zeros::<bf16>(rows * h)?;
        let mut dembed = device.zeros::<bf16>(rows * h)?;
        let mut dtop = device.zeros::<u32>(rows)?;
        let mut dkeys = device.zeros::<bf16>(context * kv_heads * dim)?;
        let mut dvals = device.zeros::<bf16>(context * kv_heads * dim)?;
        let dq = device.from_slice(&attn_q)?;
        let dk = device.from_slice(&attn_k)?;
        let dv = device.from_slice(&attn_v)?;
        let mut dattn = device.zeros::<bf16>(heads * dim)?;
        for _ in 0..200 {
            ops::gemm(&device, &dw, &dx, &mut dgemm, rows, n, h)?;
            ops::norm(&device, &dx, &dw, &mut dnorm, rows, h, 1e-6)?;
            ops::silu(&device, &da, &db, &mut dsilu, rows * h)?;
            ops::add(&device, &da, &db, &mut dadd, rows * h)?;
            ops::embedding(&device, &de, &dids, &mut dembed, rows, h, vocab)?;
            ops::top1(&device, &dl, &mut dtop, rows, vocab)?;
        }
        let shape = ops::AttentionShape {
            rows: 1,
            heads,
            kv_heads,
            dim,
            context,
            capacity: 1,
        };
        for pos in 0..64usize {
            let p = device.from_slice(&[pos as i32])?;
            ops::cache(&device, &dk, &dv, &mut dkeys, &mut dvals, &[0], &p, shape)?;
            ops::attention(
                &device,
                &dq,
                &dkeys,
                &dvals,
                &mut dattn,
                &[0],
                &[pos as i32],
                shape,
            )?;
        }
    }
    // Teardown ran; ACL itself must still be usable for a fresh device.
    let again = Device::new(0)?;
    let probe = again.from_slice(&fill(8, 211))?;
    ensure!(probe.read(8)?.len() == 8, "probe read failed");
    println!("repeated operators + teardown ok");
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected Ascend device"]
fn embedding_gathers_rows() -> Result<()> {
    let device = Device::new(0)?;
    let (vocab, h, rows) = (8usize, 128usize, 2usize);
    let w = fill(vocab * h, 121);
    let ids = device.from_slice(&[0u32, 1u32])?;
    let dw = device.from_slice(&w)?;
    let mut out = device.zeros::<bf16>(rows * h)?;
    ops::embedding(&device, &dw, &ids, &mut out, rows, h, vocab)?;
    let got = out.read(rows * h)?;
    ensure!(
        got[..h] == w[0..h] && got[h..2 * h] == w[h..2 * h],
        "embedding gather mismatch: got[0]={:?} want={:?}",
        &got[..4],
        &w[..4]
    );
    println!("embedding gather ok: {:?}", &got[..4]);
    Ok(())
}
