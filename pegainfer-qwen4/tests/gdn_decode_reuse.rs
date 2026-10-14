//! Layer 0's Gated DeltaNet decode on the real Qwen3.8-Flash-Next weights.
//!
//! This line reuses the Qwen3.5 decode kernels through
//! `pegainfer_core::ops::{conv1d_decode_batch_into, gated_delta_rule_decode_batch_into}`
//! (sources `csrc/qwen35/{conv1d,gated_delta_rule}.cu`). These tests bind that
//! reuse to the checkpoint:
//!
//! * [`the_frozen_geometry_is_the_shared_kernel_geometry`] reads only the
//!   committed fixtures. The GDN geometry derived from the frozen tensor shapes
//!   must be exactly the constants the shared wrappers assert, and the frozen
//!   config must pass this line's own validator, so the mirror in
//!   `pegainfer-qwen4/src/config.rs` cannot drift from the kernels silently.
//! * [`layer_zero_gdn_decode_on_the_real_weights`] needs the weights and a GPU. It
//!   loads layer 0's nine GDN tensors from the real shards, holds their headers
//!   against the frozen contract, and drives the two reused kernels.
//!
//! What the GPU test proves: the real tensors load with the contract's exact
//! shapes and dtypes; the reused conv1d kernel matches the reference formula
//! (`silu` of a depthwise 4-tap convolution, bf16 round-trip in the middle) and
//! hands its previous inputs to the state exactly; the reused GDN kernel matches a
//! host transcription of the documented recurrence, reads each request's own state
//! slot, and its gate path — `-exp(A_log) * softplus(a + dt_bias)`, then
//! `exp` — applied to the real `A_log`/`dt_bias` decays a state by exactly the
//! host-computed factor.
//!
//! What it does **not** prove: there is no reference oracle for this checkpoint
//! locally, so nothing here is compared against HF or against the model's own
//! activations. The numerically checked reference is a host transcription of the
//! same kernels' documented semantics, not an independent implementation of the
//! model; the activations fed to the kernels are constructed (the projections use
//! the real `in_proj_*` weights, but the hidden vectors are synthetic); and the
//! gated norm and `out_proj` that complete the block are not exercised — they are
//! separate ops (`rms_norm_gated_batch_into` with `GatedNormActivation::Sigmoid`,
//! and a GEMM) outside this reuse.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use half::bf16;
use pegainfer_core::ops::GDN_AOT_KEY_HEAD_DIM;
use pegainfer_core::ops::GDN_AOT_VALUE_HEAD_DIM;
use pegainfer_core::ops::LINEAR_CONV_MAX_KERNEL_DIM;
use pegainfer_core::ops::conv1d_decode_batch_into;
use pegainfer_core::ops::gated_delta_rule_decode_batch_into;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::model_line::ModelLine;
use safetensors::Dtype;
use safetensors::SafeTensors;

/// The tree's convention for a weight-dependent test.
const MODEL_PATH_ENV: &str = "PEGAINFER_TEST_MODEL_PATH";

/// The layer whose GDN block is proven here. It is a `linear_attention` layer, and
/// the partial local download carries its shards.
const LAYER: usize = 0;

/// The layer's GDN tensors, by the names the frozen contract uses.
const GDN_SUFFIXES: [&str; 9] = [
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
];

fn gdn_name(suffix: &str) -> String {
    format!("model.language_model.layers.{LAYER}.linear_attn.{suffix}")
}

fn frozen_config() -> serde_json::Value {
    serde_json::from_str(include_str!("frozen_config.json")).unwrap()
}

/// The frozen contract's per-tensor `(dtype, shape)`, as `dump_qwen4_frozen_tensors.py`
/// wrote it for the whole revision.
fn frozen_headers() -> HashMap<String, (Dtype, Vec<usize>)> {
    let frozen: serde_json::Value =
        serde_json::from_str(include_str!("frozen_tensors.json")).unwrap();
    serde_json::from_value(frozen["tensors"].clone()).unwrap()
}

/// The GDN geometry the frozen tensor shapes *imply*, with no reference to the
/// config and no reference to the kernel constants:
/// `(num_key_heads, num_value_heads, key_dim, val_dim, conv_kernel)`.
///
/// `in_proj_qkv` is `[2 * key_heads * key_dim + value_heads * val_dim, hidden]`,
/// `in_proj_a` is one row per value head, `norm.weight` is one head wide, and
/// `conv1d.weight` holds one `conv_kernel`-tap filter per channel.
fn gdn_geometry_from_shapes(
    headers: &HashMap<String, (Dtype, Vec<usize>)>,
) -> (usize, usize, usize, usize, usize) {
    let qkv = &headers[&gdn_name("in_proj_qkv.weight")].1;
    let value_heads = headers[&gdn_name("in_proj_a.weight")].1[0];
    let head_dim = headers[&gdn_name("norm.weight")].1[0];
    let conv = &headers[&gdn_name("conv1d.weight")].1;
    let taps = conv[conv.len() - 1];
    let channels = conv[0];
    // The conv is depthwise over the same projection, which is the other way to
    // read the same number.
    assert_eq!(
        channels, qkv[0],
        "conv1d channels must be the qkv projection width"
    );
    let key_width = qkv[0] - value_heads * head_dim;
    assert_eq!(
        key_width % (2 * head_dim),
        0,
        "the qkv projection width {key_width} is not a whole number of key head pairs"
    );
    let key_heads = key_width / (2 * head_dim);
    (key_heads, value_heads, head_dim, head_dim, taps)
}

/// The frozen config's GDN fields, typed the way the validator reads them.
fn frozen_gdn_config() -> (usize, usize, usize, usize, usize) {
    let cfg = frozen_config();
    let t = &cfg["text_config"];
    let read = |field: &str| {
        t[field]
            .as_u64()
            .unwrap_or_else(|| panic!("frozen config has no text_config.{field}")) as usize
    };
    (
        read("linear_num_key_heads"),
        read("linear_num_value_heads"),
        read("linear_key_head_dim"),
        read("linear_value_head_dim"),
        read("linear_conv_kernel_dim"),
    )
}

/// The committed fixtures must already agree with the shared kernels: this runs
/// with no weights and no device.
#[test]
fn the_frozen_geometry_is_the_shared_kernel_geometry() {
    let headers = frozen_headers();
    let (key_heads, value_heads, key_dim, val_dim, conv) = gdn_geometry_from_shapes(&headers);
    let (cfg_key_heads, cfg_value_heads, cfg_key_dim, cfg_val_dim, cfg_conv) = frozen_gdn_config();

    // The tensor shapes and the config must describe one checkpoint.
    assert_eq!(
        (key_heads, value_heads, key_dim, val_dim, conv),
        (
            cfg_key_heads,
            cfg_value_heads,
            cfg_key_dim,
            cfg_val_dim,
            cfg_conv
        ),
        "the frozen tensor shapes and the frozen config disagree about the GDN geometry"
    );

    // And that checkpoint must be one the shared wrappers can run: they assert the
    // two head dims and cap the conv kernel.
    assert_eq!(key_dim, GDN_AOT_KEY_HEAD_DIM);
    assert_eq!(val_dim, GDN_AOT_VALUE_HEAD_DIM);
    assert!(conv <= LINEAR_CONV_MAX_KERNEL_DIM);
    // The per-head shapes the kernels derive their state from.
    assert_eq!(
        headers[&gdn_name("conv1d.weight")].1[0],
        2 * key_heads * key_dim + value_heads * val_dim
    );
    assert_eq!(
        headers[&gdn_name("in_proj_z.weight")].1[0],
        value_heads * val_dim
    );
    assert_eq!(headers[&gdn_name("A_log")].1, vec![value_heads]);
    assert_eq!(headers[&gdn_name("dt_bias")].1, vec![value_heads]);
    // Every GDN tensor is BF16, `A_log` included.
    for suffix in GDN_SUFFIXES {
        assert_eq!(
            headers[&gdn_name(suffix)].0,
            Dtype::BF16,
            "{} is not BF16",
            gdn_name(suffix)
        );
    }

    // The same geometry must pass this line's validator, whose mirror is the
    // constant set asserted above.
    pegainfer_qwen4::model_line::MODEL_LINE
        .probe(&frozen_config())
        .expect("the frozen config must satisfy this line's own geometry rules");

    println!(
        "frozen GDN geometry {key_heads}K/{value_heads}V heads at {key_dim}/{val_dim}, {conv}-tap conv \
         == pegainfer_core::ops constants"
    );
}

// ─── the real-weight GPU test ────────────────────────────────────────────────

/// One tensor of the real checkpoint, as loaded from its shard, in both views the
/// test needs: the header for the contract check, and the values for the kernels.
struct RealTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    values: Vec<bf16>,
}

impl RealTensor {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn f32(&self) -> Vec<f32> {
        self.values.iter().map(|v| v.to_f32()).collect()
    }

    fn view(&self, rows: usize, cols: usize) -> Weight<'_> {
        assert_eq!((rows, cols), (self.shape[0], self.shape[1]));
        Weight { tensor: self, rows }
    }
}

struct Weight<'a> {
    tensor: &'a RealTensor,
    rows: usize,
}

impl Weight<'_> {
    /// `weight @ x` in f32, rounded to bf16, the way the layer's projection would
    /// hand it to the kernels.
    fn project(&self, x: &[f32]) -> Vec<bf16> {
        let cols = self.tensor.shape[1];
        assert_eq!(cols, x.len());
        (0..self.rows)
            .map(|row| {
                let row_values = &self.tensor.values[row * cols..(row + 1) * cols];
                let mut acc = 0.0f64;
                for (w, x) in row_values.iter().zip(x) {
                    acc += f64::from(w.to_f32()) * f64::from(*x);
                }
                bf16::from_f32(acc as f32)
            })
            .collect()
    }
}

fn skipped(test: &str, reason: &str) {
    eprintln!("SKIP {test}: {reason}");
}

/// The checkpoint directory from the environment, with the shards this test reads.
fn checkpoint_dir(test: &str) -> Option<PathBuf> {
    let dir = match std::env::var(MODEL_PATH_ENV) {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path),
        Ok(_) => {
            skipped(test, &format!("{MODEL_PATH_ENV} is empty"));
            return None;
        }
        Err(std::env::VarError::NotPresent) => {
            skipped(
                test,
                &format!("{MODEL_PATH_ENV} is not set; point it at the Qwen3.8-Flash-Next shards"),
            );
            return None;
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            skipped(test, &format!("{MODEL_PATH_ENV} is not valid UTF-8"));
            return None;
        }
    };
    if !dir.join("model.safetensors.index.json").is_file() {
        skipped(
            test,
            &format!(
                "{} has no model.safetensors.index.json; this test needs the shard index",
                dir.display()
            ),
        );
        return None;
    }
    Some(dir)
}

fn shard_index(dir: &Path) -> Result<HashMap<String, String>> {
    let raw = std::fs::read(dir.join("model.safetensors.index.json"))?;
    let index: serde_json::Value = serde_json::from_slice(&raw)?;
    serde_json::from_value(
        index
            .get("weight_map")
            .ok_or_else(|| anyhow!("shard index has no weight_map"))?
            .clone(),
    )
    .map_err(Into::into)
}

/// Reads one tensor out of the shard the index names for it. The shards are
/// memory-mapped, so a 1 GB file costs only the tensor's own pages.
fn load_real_tensor(dir: &Path, index: &HashMap<String, String>, name: &str) -> Result<RealTensor> {
    let shard = index
        .get(name)
        .ok_or_else(|| anyhow!("{name} is not in the shard index"))?;
    let file = std::fs::File::open(dir.join(shard))?;
    // SAFETY: the mapping outlives the `SafeTensors` view and the file is not
    // written while the test runs.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let shards = SafeTensors::deserialize(&map)?;
    let tensor = shards.tensor(name)?;
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "{name}: expected BF16, got {:?}",
        tensor.dtype()
    );
    let bytes = tensor.data();
    ensure!(
        bytes.len().is_multiple_of(2),
        "{name}: BF16 payload is not a whole number of elements"
    );
    let values = bytes
        .chunks_exact(2)
        .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    let tensor = RealTensor {
        dtype: tensor.dtype(),
        shape: tensor.shape().to_vec(),
        values,
    };
    ensure!(
        tensor.numel() == tensor.values.len(),
        "{name}: the header and the payload disagree about the element count"
    );
    Ok(tensor)
}

/// A deterministic hidden vector, so a rerun feeds the kernels the same bytes.
/// The weights are the checkpoint's; the activations are constructed because no
/// activation fixture for this checkpoint exists.
fn synthetic_hidden(seed: u64, width: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // 24 bits of the generator, mapped to [-1, 1).
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn bf16_host(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|v| bf16::from_f32(*v)).collect()
}

/// The reference conv1d step: `sum = Σ_k w[c][k] * v_k` over
/// `[x_{t-3}, x_{t-2}, x_{t-1}, x_t]`, rounded to bf16, then SiLU in f32, then
/// bf16 — the kernel's own order (`conv1d.cu:105-116`).
fn reference_conv_step(x: &[bf16], weights: &[f32], state: &mut [f32], taps: usize) -> Vec<bf16> {
    let width = taps - 1;
    let channels = x.len();
    let mut out = vec![bf16::ZERO; channels];
    for c in 0..channels {
        let mut sum = 0.0f32;
        for k in 0..taps {
            let value = if k == taps - 1 {
                x[c].to_f32()
            } else {
                state[c * width + k]
            };
            sum += value * weights[c * taps + k];
        }
        let sum = bf16::from_f32(sum).to_f32();
        out[c] = bf16::from_f32(sum / (1.0 + (-sum).exp()));
    }
    for c in 0..channels {
        for i in 0..width {
            state[c * width + i] = if i + 1 < width {
                state[c * width + i + 1]
            } else {
                x[c].to_f32()
            };
        }
    }
    out
}

/// The reference GDN step, transcribed from the reused kernel's documented
/// recurrence (`gated_delta_rule.cu`, decode kernels): L2-normalized q (scaled by
/// `1/sqrt(key_dim)`) and k, `g = -exp(A_log) * softplus(a + dt_bias)`,
/// `beta = sigmoid(b)`, `s ← s * exp(g)`, `s ← s + k ⊗ ((v - k·s) * beta)`,
/// `out = q·s`, with the state in `[head, key, value]` f32 layout (value
/// contiguous).
fn reference_gdn_step(
    qkv: &[bf16],
    b_proj: &[bf16],
    a_proj: &[bf16],
    dt_bias: &[f32],
    a_log: &[f32],
    state: &mut [f32],
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Vec<bf16> {
    let q_dim = key_dim * num_key_heads;
    let out_dim = val_dim * num_value_heads;
    let mut out = vec![bf16::ZERO; out_dim];
    let head_stride = key_dim * val_dim;
    for v_head in 0..num_value_heads {
        let k_head = v_head * num_key_heads / num_value_heads;
        let q: Vec<f32> = (0..key_dim)
            .map(|i| qkv[k_head * key_dim + i].to_f32())
            .collect();
        let k: Vec<f32> = (0..key_dim)
            .map(|i| qkv[q_dim + k_head * key_dim + i].to_f32())
            .collect();
        let v: Vec<f32> = (0..val_dim)
            .map(|i| qkv[2 * q_dim + v_head * val_dim + i].to_f32())
            .collect();

        let q_sq: f32 = q.iter().map(|value| value * value).sum();
        let k_sq: f32 = k.iter().map(|value| value * value).sum();
        let q_scale = 1.0 / (q_sq + 1e-12).sqrt() / (key_dim as f32).sqrt();
        let k_scale = 1.0 / (k_sq + 1e-12).sqrt();
        let q: Vec<f32> = q.iter().map(|value| value * q_scale).collect();
        let k: Vec<f32> = k.iter().map(|value| value * k_scale).collect();

        let x = a_proj[v_head].to_f32() + dt_bias[v_head];
        let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        let decay = (-a_log[v_head].exp() * softplus).exp();
        let beta = 1.0 / (1.0 + (-b_proj[v_head].to_f32()).exp());

        let head = &mut state[v_head * head_stride..(v_head + 1) * head_stride];
        let mut kv = vec![0.0f32; val_dim];
        for j in 0..key_dim {
            for i in 0..val_dim {
                let scaled = head[j * val_dim + i] * decay;
                head[j * val_dim + i] = scaled;
                kv[i] += scaled * k[j];
            }
        }
        for i in 0..val_dim {
            let delta = (v[i] - kv[i]) * beta;
            for j in 0..key_dim {
                head[j * val_dim + i] += delta * k[j];
            }
        }
        for i in 0..val_dim {
            let mut acc = 0.0f32;
            for j in 0..key_dim {
                acc += head[j * val_dim + i] * q[j];
            }
            out[v_head * val_dim + i] = bf16::from_f32(acc);
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The largest deviation relative to the reference's own magnitude, so a check is
/// as tight at 1e-3 as at 10. One bf16 rounding is 2^-9 relative; the reference and
/// the kernel differ only by summation order and by `exp`/`ln` implementations.
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0f32, f32::max)
}

/// A device pointer table over `slots`, one entry per request.
fn state_pointer_table(
    ctx: &DeviceContext,
    slots: &mut [cudarc::driver::CudaSlice<f32>],
) -> Result<cudarc::driver::CudaSlice<u64>> {
    use cudarc::driver::DevicePtrMut;
    let mut pointers = Vec::with_capacity(slots.len());
    for slot in slots.iter_mut() {
        let (pointer, _guard) = slot.device_ptr_mut(&ctx.stream);
        pointers.push(pointer);
    }
    Ok(ctx.stream.clone_htod(&pointers)?)
}

fn bf16_state_pointer_table(
    ctx: &DeviceContext,
    slots: &mut [cudarc::driver::CudaSlice<bf16>],
) -> Result<cudarc::driver::CudaSlice<u64>> {
    use cudarc::driver::DevicePtrMut;
    let mut pointers = Vec::with_capacity(slots.len());
    for slot in slots.iter_mut() {
        let (pointer, _guard) = slot.device_ptr_mut(&ctx.stream);
        pointers.push(pointer);
    }
    Ok(ctx.stream.clone_htod(&pointers)?)
}

#[test]
fn layer_zero_gdn_decode_on_the_real_weights() -> Result<()> {
    const TEST: &str = "layer_zero_gdn_decode_on_the_real_weights";
    let Some(dir) = checkpoint_dir(TEST) else {
        return Ok(());
    };
    let index = shard_index(&dir)?;
    let headers = frozen_headers();

    // 1. The nine real tensors, held against the frozen contract.
    let mut real = HashMap::new();
    for suffix in GDN_SUFFIXES {
        let name = gdn_name(suffix);
        let tensor = load_real_tensor(&dir, &index, &name)?;
        let (dtype, shape) = &headers[&name];
        ensure!(
            tensor.dtype == *dtype && tensor.shape == *shape,
            "{name}: checkpoint has {:?} {:?}, the frozen contract says {dtype:?} {shape:?}",
            tensor.dtype,
            tensor.shape
        );
        real.insert(suffix, tensor);
    }
    let (key_heads, value_heads, key_dim, val_dim, taps) = gdn_geometry_from_shapes(&headers);
    assert_eq!(
        (key_heads, value_heads, key_dim, val_dim, taps),
        (16, 48, 128, 128, 4)
    );
    let qkv_dim = 2 * key_heads * key_dim + value_heads * val_dim;
    let out_dim = value_heads * val_dim;
    let hidden = real["in_proj_qkv.weight"].shape[1];
    let batch = 3usize;
    println!(
        "{TEST}: loaded {} real tensors, geometry {key_heads}K/{value_heads}V heads at \
         {key_dim}/{val_dim}, {taps}-tap conv, hidden {hidden}",
        GDN_SUFFIXES.len()
    );

    // 2. The hidden vectors the kernels see. Real weights, constructed activations.
    let hidden_rows: Vec<Vec<f32>> = (0..batch)
        .map(|row| synthetic_hidden(0x51A7 + row as u64, hidden))
        .collect();
    let qkv_rows: Vec<Vec<bf16>> = hidden_rows
        .iter()
        .map(|x| real["in_proj_qkv.weight"].view(qkv_dim, hidden).project(x))
        .collect();
    let a_rows: Vec<Vec<bf16>> = hidden_rows
        .iter()
        .map(|x| {
            real["in_proj_a.weight"]
                .view(value_heads, hidden)
                .project(x)
        })
        .collect();
    let b_rows: Vec<Vec<bf16>> = hidden_rows
        .iter()
        .map(|x| {
            real["in_proj_b.weight"]
                .view(value_heads, hidden)
                .project(x)
        })
        .collect();
    let dt_bias = real["dt_bias"].f32();
    let a_log = real["A_log"].f32();

    // 3. The gate path on the real scalars: the reused kernel computes
    //    `-exp(A_log) * softplus(a + dt_bias)`; on this checkpoint that must be a
    //    decay in (0, 1) for every value head, or the state would grow.
    let mut decays = Vec::with_capacity(value_heads);
    for head in 0..value_heads {
        for row in 0..batch {
            let x = a_rows[row][head].to_f32() + dt_bias[head];
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            decays.push((-a_log[head].exp() * softplus).exp());
        }
    }
    ensure!(
        decays.iter().all(|d| d.is_finite() && *d > 0.0 && *d < 1.0),
        "the real A_log/dt_bias must put every gate in (0, 1), got {:?}",
        decays.iter().fold(f32::INFINITY, |a, b| a.min(*b))
    );
    println!(
        "{TEST}: real gates decay in ({:.3e}, {:.6})",
        decays.iter().fold(f32::INFINITY, |a, b| a.min(*b)),
        decays.iter().fold(0.0f32, |a, b| a.max(*b))
    );

    let ctx = DeviceContext::new()?;
    let conv_weight = real["conv1d.weight"].f32();
    let conv_weight_gpu = DeviceVec::from_host(&ctx, &real["conv1d.weight"].values)?;
    let dt_bias_gpu = DeviceVec::from_host(&ctx, &bf16_host(&dt_bias))?;
    let a_log_gpu = ctx.stream.clone_htod(&a_log)?;

    // 4. The reused conv1d kernel, four steps so the state is filled and every tap
    //    is live. The host mirrors the state to check the handoff exactly.
    let channels = qkv_dim;
    let state_width = taps - 1;
    let conv_state_len = channels * state_width;
    let mut conv_slots: Vec<cudarc::driver::CudaSlice<bf16>> = (0..batch)
        .map(|_| ctx.stream.alloc_zeros(conv_state_len))
        .collect::<std::result::Result<_, _>>()?;
    let mut conv_host_state = vec![vec![0.0f32; conv_state_len]; batch];
    let steps = taps;
    let mut last_conv_out = vec![vec![bf16::ZERO; channels]; batch];
    for step in 0..steps {
        let rows: Vec<Vec<bf16>> = (0..batch)
            .map(|row| {
                // The first step is the real projection; later steps are the same
                // construction with a different scale, so the taps see distinct
                // values and the state handoff is checkable.
                match step {
                    0 => qkv_rows[row].clone(),
                    _ => qkv_rows[row]
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            bf16::from_f32(
                                value.to_f32() * (1.0 - 0.1 * step as f32) + (i % 7) as f32 * 1e-4,
                            )
                        })
                        .collect(),
                }
            })
            .collect();
        let x_host: Vec<bf16> = rows.iter().flatten().copied().collect();
        let x = HiddenStates {
            data: ctx.stream.clone_htod(&x_host)?,
            hidden_dim: channels,
            seq_len: batch,
        };
        let mut out = HiddenStates::zeros(&ctx, channels, batch)?;
        let pointers = bf16_state_pointer_table(&ctx, &mut conv_slots)?;
        conv1d_decode_batch_into(&ctx, &x, &conv_weight_gpu, &pointers, &mut out, taps);
        ctx.sync()?;

        let device: Vec<bf16> = ctx.stream.clone_dtoh(&out.data)?;
        for row in 0..batch {
            let expected =
                reference_conv_step(&rows[row], &conv_weight, &mut conv_host_state[row], taps);
            let got = &device[row * channels..(row + 1) * channels];
            let got_f32: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
            let expected_f32: Vec<f32> = expected.iter().map(|v| v.to_f32()).collect();
            let diff = max_rel_diff(&got_f32, &expected_f32);
            ensure!(
                diff <= 5e-3,
                "conv1d step {step} row {row}: the reused kernel differs from the reference \
                 formula by a relative {diff}"
            );
            last_conv_out[row] = got.to_vec();
        }

        // The state is the previous inputs, oldest first, exactly.
        for (slot, slot_values) in conv_slots.iter().enumerate() {
            let device_state: Vec<bf16> = ctx.stream.clone_dtoh(slot_values)?;
            ctx.sync()?;
            for c in 0..channels {
                for i in 0..state_width {
                    let got = device_state[c * state_width + i].to_f32();
                    let expected =
                        bf16::from_f32(conv_host_state[slot][c * state_width + i]).to_f32();
                    ensure!(
                        got == expected,
                        "conv1d state after step {step} slot {slot} channel {c} tap {i}: \
                         {got} != {expected}"
                    );
                }
            }
        }
    }
    println!("{TEST}: reused conv1d matches the reference formula and hands off its state exactly");

    // 5. The reused GDN kernel on the real projections, against the host
    //    transcription of the same recurrence.
    let gdn_input: Vec<bf16> = last_conv_out.iter().flatten().copied().collect();
    let qkv_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&gdn_input)?,
        hidden_dim: qkv_dim,
        seq_len: batch,
    };
    let b_flat: Vec<bf16> = b_rows.iter().flatten().copied().collect();
    let a_flat: Vec<bf16> = a_rows.iter().flatten().copied().collect();
    let b_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&b_flat)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let a_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&a_flat)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let head_state_len = value_heads * key_dim * val_dim;
    let mut gdn_slots: Vec<cudarc::driver::CudaSlice<f32>> = (0..batch)
        .map(|_| ctx.stream.alloc_zeros(head_state_len))
        .collect::<std::result::Result<_, _>>()?;
    let mut out_gpu = HiddenStates::zeros(&ctx, out_dim, batch)?;
    let pointers = state_pointer_table(&ctx, &mut gdn_slots)?;
    gated_delta_rule_decode_batch_into(
        &ctx,
        &qkv_gpu,
        &b_gpu,
        &a_gpu,
        &dt_bias_gpu,
        &a_log_gpu,
        &pointers,
        &mut out_gpu,
        batch,
        key_heads,
        value_heads,
        key_dim,
        val_dim,
    );
    ctx.sync()?;
    let out_host: Vec<bf16> = ctx.stream.clone_dtoh(&out_gpu.data)?;

    ensure!(
        out_host.iter().all(|v| v.to_f32().is_finite())
            && out_host.iter().any(|v| *v != bf16::ZERO),
        "the reused GDN kernel must produce finite, non-zero output"
    );
    for row in 0..batch {
        let mut reference_state = vec![0.0f32; head_state_len];
        let expected = reference_gdn_step(
            &gdn_input[row * qkv_dim..(row + 1) * qkv_dim],
            &b_rows[row],
            &a_rows[row],
            &dt_bias,
            &a_log,
            &mut reference_state,
            key_heads,
            value_heads,
            key_dim,
            val_dim,
        );
        let got = &out_host[row * out_dim..(row + 1) * out_dim];
        let diff = max_rel_diff(
            &got.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            &expected.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
        );
        ensure!(
            diff <= 8e-3,
            "gdn decode row {row}: the reused kernel differs from the reference recurrence by a \
             relative {diff}"
        );
        let device_state: Vec<f32> = ctx.stream.clone_dtoh(&gdn_slots[row])?;
        ctx.sync()?;
        let state_diff = max_rel_diff(&device_state, &reference_state);
        ensure!(
            state_diff <= 1e-4,
            "gdn decode row {row}: the updated state differs from the reference by a relative \
             {state_diff}"
        );
    }

    // 6. A state the kernel must only decay: no input, real gates. Every head of
    //    every slot is checked against the host-computed factor.
    let zeros = vec![bf16::ZERO; batch * qkv_dim];
    let zero_proj = vec![bf16::ZERO; batch * value_heads];
    let zero_qkv = HiddenStates {
        data: ctx.stream.clone_htod(&zeros)?,
        hidden_dim: qkv_dim,
        seq_len: batch,
    };
    let zero_a = HiddenStates {
        data: ctx.stream.clone_htod(&zero_proj)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let zero_b = HiddenStates {
        data: ctx.stream.clone_htod(&zero_proj)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let fills = [1.0f32, 0.5, 2.0];
    let mut decay_slots: Vec<cudarc::driver::CudaSlice<f32>> = (0..batch)
        .map(|slot| ctx.stream.clone_htod(&vec![fills[slot]; head_state_len]))
        .collect::<std::result::Result<_, _>>()?;
    let mut decay_out = HiddenStates::zeros(&ctx, out_dim, batch)?;
    let pointers = state_pointer_table(&ctx, &mut decay_slots)?;
    gated_delta_rule_decode_batch_into(
        &ctx,
        &zero_qkv,
        &zero_b,
        &zero_a,
        &dt_bias_gpu,
        &a_log_gpu,
        &pointers,
        &mut decay_out,
        batch,
        key_heads,
        value_heads,
        key_dim,
        val_dim,
    );
    ctx.sync()?;
    let decay_out_host: Vec<bf16> = ctx.stream.clone_dtoh(&decay_out.data)?;
    ensure!(
        decay_out_host.iter().all(|v| *v == bf16::ZERO),
        "with no query the output must be exactly zero"
    );
    for slot in 0..batch {
        let device_state: Vec<f32> = ctx.stream.clone_dtoh(&decay_slots[slot])?;
        ctx.sync()?;
        for head in 0..value_heads {
            let x = 0.0f32 + dt_bias[head];
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            let expected = fills[slot] * (-a_log[head].exp() * softplus).exp();
            let head_state =
                &device_state[head * key_dim * val_dim..(head + 1) * key_dim * val_dim];
            let worst = head_state
                .iter()
                .map(|got| (got - expected).abs() / expected.abs().max(1.0))
                .fold(0.0f32, f32::max);
            ensure!(
                worst <= 1e-5,
                "slot {slot} head {head}: the state decays to a worst relative {worst} away from \
                 the host factor {expected}"
            );
        }
    }
    println!(
        "{TEST}: reused GDN matches the reference recurrence and decays exactly by the host factor"
    );

    // 7. Row isolation: identical inputs in every slot, distinct initial states.
    //    Run A starts all three states at zero; run B starts slot 0 at zero and
    //    slots 1 and 2 elsewhere. Slot 0 must be bit for bit run A's slot 0, and
    //    the other two must not be — a shared or mis-indexed state pointer would
    //    collapse them onto the same answer.
    let identical_qkv: Vec<bf16> = (0..batch).flat_map(|_| qkv_rows[0].clone()).collect();
    let identical_a: Vec<bf16> = (0..batch).flat_map(|_| a_rows[0].clone()).collect();
    let identical_b: Vec<bf16> = (0..batch).flat_map(|_| b_rows[0].clone()).collect();
    let identical_qkv_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&identical_qkv)?,
        hidden_dim: qkv_dim,
        seq_len: batch,
    };
    let identical_a_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&identical_a)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let identical_b_gpu = HiddenStates {
        data: ctx.stream.clone_htod(&identical_b)?,
        hidden_dim: value_heads,
        seq_len: batch,
    };
    let run = |fills: [f32; 3]| -> Result<Vec<f32>> {
        let mut slots: Vec<cudarc::driver::CudaSlice<f32>> = (0..batch)
            .map(|slot| ctx.stream.clone_htod(&vec![fills[slot]; head_state_len]))
            .collect::<std::result::Result<_, _>>()?;
        let mut out = HiddenStates::zeros(&ctx, out_dim, batch)?;
        let pointers = state_pointer_table(&ctx, &mut slots)?;
        gated_delta_rule_decode_batch_into(
            &ctx,
            &identical_qkv_gpu,
            &identical_b_gpu,
            &identical_a_gpu,
            &dt_bias_gpu,
            &a_log_gpu,
            &pointers,
            &mut out,
            batch,
            key_heads,
            value_heads,
            key_dim,
            val_dim,
        );
        ctx.sync()?;
        let host: Vec<bf16> = ctx.stream.clone_dtoh(&out.data)?;
        Ok(host.iter().map(|v| v.to_f32()).collect())
    };
    let all_zero = run([0.0, 0.0, 0.0])?;
    let distinct = run([0.0, 0.5, 1.0])?;
    let row = |values: &[f32], slot: usize| -> Vec<f32> {
        values[slot * out_dim..(slot + 1) * out_dim].to_vec()
    };
    ensure!(
        max_abs_diff(&row(&all_zero, 0), &row(&all_zero, 1)) == 0.0,
        "identical slots with identical states must produce identical output"
    );
    ensure!(
        max_abs_diff(&row(&all_zero, 0), &row(&distinct, 0)) == 0.0,
        "slot 0's output must not change when only the other slots' states move"
    );
    ensure!(
        max_rel_diff(&row(&distinct, 1), &row(&distinct, 0)) > 1e-3
            && max_rel_diff(&row(&distinct, 2), &row(&distinct, 0)) > 1e-3
            && max_rel_diff(&row(&distinct, 2), &row(&distinct, 1)) > 1e-3,
        "slots with different initial states must not share one state"
    );
    println!("{TEST}: each request reads and updates its own state slot");

    Ok(())
}
