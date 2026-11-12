use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_device::Buffer;
use pegainfer_device::Device;
use pegainfer_kernels::rocm::AttentionShape;
use pegainfer_kernels::rocm::{self as ops};

use crate::config::Config;

type Weight = Arc<Buffer<bf16>>;
struct Layer {
    input_norm: Weight,
    post_norm: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Weight,
    k_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    keys: Buffer<bf16>,
    values: Buffer<bf16>,
}
#[derive(Clone, Copy, Debug)]
pub struct ModelOptions {
    pub device: usize,
    pub max_context: usize,
    pub slots: usize,
    pub chunk_size: usize,
}
impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: 0,
            max_context: 4096,
            slots: 4,
            chunk_size: 128,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct TokenInput {
    pub token: u32,
    pub slot: usize,
    pub position: usize,
}
struct Scratch {
    ids: Buffer<u32>,
    slots: Buffer<i32>,
    positions: Buffer<i32>,
    selection: Buffer<u32>,
    hidden: Buffer<bf16>,
    norm: Buffer<bf16>,
    residual: Buffer<bf16>,
    projected: Buffer<bf16>,
    q: Buffer<bf16>,
    qn: Buffer<bf16>,
    k: Buffer<bf16>,
    kn: Buffer<bf16>,
    v: Buffer<bf16>,
    attn: Buffer<bf16>,
    gate: Buffer<bf16>,
    up: Buffer<bf16>,
    activation: Buffer<bf16>,
    selected: Buffer<bf16>,
    logits: Buffer<bf16>,
    predictions: Buffer<u32>,
}
pub struct Model {
    device: Device,
    config: Config,
    options: ModelOptions,
    layers: Vec<Layer>,
    embedding: Weight,
    head: Weight,
    final_norm: Weight,
    scratch: Scratch,
    lengths: Vec<usize>,
    last_output_rows: usize,
    poisoned: bool,
}
fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).context("model size overflow")
    })
}
fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b).context("model size overflow")
    })
}
impl Model {
    pub fn load(path: impl AsRef<Path>, options: ModelOptions) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.to_str().context("non-UTF8 model path")?;
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        crate::config::probe_config_json(&raw)?;
        ensure!(
            raw.get("quantization_config")
                .is_none_or(serde_json::Value::is_null),
            "ROCm first release requires unquantized BF16 weights"
        );
        ensure!(
            !raw.get("attention_bias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "attention bias is not supported"
        );
        ensure!(
            !raw.get("use_sliding_window")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "sliding attention is not supported"
        );
        ensure!(
            raw.get("layer_types").is_none_or(|x| x.is_null()
                || x.as_array()
                    .is_some_and(|a| a.iter().all(|v| v.as_str() == Some("full_attention")))),
            "only full_attention layers are supported"
        );
        ensure!(
            raw.get("rope_scaling")
                .is_none_or(serde_json::Value::is_null),
            "scaled RoPE is not supported"
        );
        ensure!(
            raw.get("hidden_act")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|x| x == "silu"),
            "only SiLU is supported"
        );
        let mut c = Config::from_file(path_str)?;
        for v in [
            c.hidden_size,
            c.intermediate_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.vocab_size,
        ] {
            ensure!(v > 0 && i32::try_from(v).is_ok(), "invalid model dimension");
        }
        ensure!(
            c.head_dim == 128,
            "ROCm first release supports Qwen3 head_dim=128"
        );
        ensure!(
            c.rope_theta.is_finite()
                && c.rope_theta > 0.
                && c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.,
            "invalid Qwen3 numerical config"
        );
        ensure!(
            (1..=16).contains(&options.slots) && (1..=512).contains(&options.chunk_size),
            "slots must be 1..16, chunk size 1..512"
        );
        ensure!(
            (1..=8192).contains(&options.max_context)
                && options.max_context <= c.max_position_embeddings,
            "context exceeds the backend/checkpoint limit"
        );
        c.decodable_vocab =
            pegainfer_core::decodable_vocab::decodable_vocab_width(path_str)?.min(c.vocab_size);
        c.logits_rows = c.decodable_vocab;
        ensure!(c.decodable_vocab > 0, "empty decodable vocabulary");
        let (h, i, q, k) = (
            c.hidden_size,
            c.intermediate_size,
            product(&[c.num_attention_heads, c.head_dim])?,
            product(&[c.num_key_value_heads, c.head_dim])?,
        );
        ensure!(
            i32::try_from(q).is_ok() && i32::try_from(k).is_ok(),
            "projection dimension exceeds HIP ABI"
        );
        let mut specs = HashMap::<String, Vec<usize>>::new();
        specs.insert("model.embed_tokens.weight".into(), vec![c.vocab_size, h]);
        specs.insert("model.norm.weight".into(), vec![h]);
        if !c.tie_word_embeddings {
            specs.insert("lm_head.weight".into(), vec![c.vocab_size, h]);
        }
        for layer in 0..c.num_hidden_layers {
            for (name, shape) in [
                ("input_layernorm", vec![h]),
                ("post_attention_layernorm", vec![h]),
                ("self_attn.q_proj", vec![q, h]),
                ("self_attn.k_proj", vec![k, h]),
                ("self_attn.v_proj", vec![k, h]),
                ("self_attn.o_proj", vec![h, q]),
                ("self_attn.q_norm", vec![c.head_dim]),
                ("self_attn.k_norm", vec![c.head_dim]),
                ("mlp.gate_proj", vec![i, h]),
                ("mlp.up_proj", vec![i, h]),
                ("mlp.down_proj", vec![h, i]),
            ] {
                specs.insert(format!("model.layers.{layer}.{name}.weight"), shape);
            }
        }
        let device = Device::new(options.device)?;
        let info = device.info()?;
        ensure!(
            ops::built_arches()
                .split(',')
                .any(|a| info.arch.split(':').next() == Some(a)),
            "binary targets {} do not include device {}",
            ops::built_arches(),
            info.arch
        );
        let weight_bytes = specs
            .values()
            .try_fold(0usize, |s, d| sum(&[s, product(&[product(d)?, 2])?]))?;
        let kv_elements = product(&[options.slots, options.max_context, k])?;
        let max_rows = options.chunk_size.max(options.slots);
        let scratch_bytes = product(&[
            max_rows,
            sum(&[
                product(&[6, h])?,
                product(&[3, q])?,
                product(&[3, k])?,
                product(&[3, i])?,
                c.decodable_vocab,
            ])?,
            2,
        ])?;
        let budget = sum(&[
            weight_bytes,
            product(&[kv_elements, 4, c.num_hidden_layers])?,
            scratch_bytes,
            512 << 20,
        ])?;
        ensure!(
            budget < info.free_bytes,
            "profile requires approximately {budget} bytes including reserve, only {} bytes free; reduce slots/context",
            info.free_bytes
        );
        log::info!(
            "ROCm Qwen3: {} {} slots={} context={} chunk={} estimated_bytes={budget}",
            info.name,
            info.arch,
            options.slots,
            options.max_context,
            options.chunk_size
        );
        let mut files = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect::<Vec<_>>();
        files.sort();
        ensure!(
            !files.is_empty(),
            "no safetensors weights in {}",
            path.display()
        );
        let mut weights = HashMap::<String, Weight>::new();
        for file in files {
            let handle = std::fs::File::open(&file)?;
            // Checkpoints must remain immutable for the lifetime of this mapping.
            let map = unsafe { memmap2::MmapOptions::new().map(&handle)? };
            let tensors = safetensors::SafeTensors::deserialize(&map)?;
            for name in tensors.names() {
                if let Some(shape) = specs.get(name) {
                    ensure!(!weights.contains_key(name), "duplicate tensor {name}");
                    let t = tensors.tensor(name)?;
                    ensure!(
                        t.shape() == shape && t.dtype() == safetensors::Dtype::BF16,
                        "tensor {name}: expected BF16 {shape:?}, got {:?} {:?}",
                        t.dtype(),
                        t.shape()
                    );
                    weights.insert(
                        name.to_string(),
                        Arc::new(
                            device
                                .from_bytes(t.data())
                                .with_context(|| format!("upload {name}"))?,
                        ),
                    );
                }
            }
        }
        for name in specs.keys() {
            ensure!(weights.contains_key(name), "missing tensor {name}");
        }
        let embedding = weights.remove("model.embed_tokens.weight").unwrap();
        let head = if c.tie_word_embeddings {
            embedding.clone()
        } else {
            weights.remove("lm_head.weight").unwrap()
        };
        let final_norm = weights.remove("model.norm.weight").unwrap();
        let mut layers = Vec::new();
        for l in 0..c.num_hidden_layers {
            let mut take = |name: &str| {
                weights
                    .remove(&format!("model.layers.{l}.{name}.weight"))
                    .unwrap()
            };
            layers.push(Layer {
                input_norm: take("input_layernorm"),
                post_norm: take("post_attention_layernorm"),
                q: take("self_attn.q_proj"),
                k: take("self_attn.k_proj"),
                v: take("self_attn.v_proj"),
                o: take("self_attn.o_proj"),
                q_norm: take("self_attn.q_norm"),
                k_norm: take("self_attn.k_norm"),
                gate: take("mlp.gate_proj"),
                up: take("mlp.up_proj"),
                down: take("mlp.down_proj"),
                keys: device.zeros(kv_elements)?,
                values: device.zeros(kv_elements)?,
            });
        }
        let scratch = Scratch {
            ids: device.zeros(max_rows)?,
            slots: device.zeros(max_rows)?,
            positions: device.zeros(max_rows)?,
            selection: device.zeros(max_rows)?,
            hidden: device.zeros(max_rows * h)?,
            norm: device.zeros(max_rows * h)?,
            residual: device.zeros(max_rows * h)?,
            projected: device.zeros(max_rows * h)?,
            q: device.zeros(max_rows * q)?,
            qn: device.zeros(max_rows * q)?,
            k: device.zeros(max_rows * k)?,
            kn: device.zeros(max_rows * k)?,
            v: device.zeros(max_rows * k)?,
            attn: device.zeros(max_rows * q)?,
            gate: device.zeros(max_rows * i)?,
            up: device.zeros(max_rows * i)?,
            activation: device.zeros(max_rows * i)?,
            selected: device.zeros(max_rows * h)?,
            logits: device.zeros(max_rows * c.decodable_vocab)?,
            predictions: device.zeros(max_rows)?,
        };
        device.synchronize()?;
        Ok(Self {
            device,
            config: c,
            options,
            layers,
            embedding,
            head,
            final_norm,
            scratch,
            lengths: vec![0; options.slots],
            last_output_rows: 0,
            poisoned: false,
        })
    }
    pub fn options(&self) -> ModelOptions {
        self.options
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    pub fn output_vocab_size(&self) -> usize {
        self.config.decodable_vocab
    }
    pub fn eos_ids(&self) -> &[u32] {
        &self.config.stop_token_ids
    }
    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        ensure!(
            !self.poisoned,
            "model has a failed GPU step; reload it before reuse"
        );
        ensure!(slot < self.options.slots, "slot out of range");
        self.device.synchronize()?;
        self.lengths[slot] = 0;
        Ok(())
    }
    /// Append a ragged batch; every slot's positions must extend its current prefix.
    /// `outputs` selects rows whose next-token logits the caller needs.
    pub fn forward(&mut self, input: &[TokenInput], outputs: &[usize]) -> Result<Vec<u32>> {
        ensure!(
            !self.poisoned,
            "model has a failed GPU step; reload it before reuse"
        );
        let rows = input.len();
        ensure!(
            rows > 0 && rows <= self.options.chunk_size.max(self.options.slots),
            "forward exceeds row capacity"
        );
        ensure!(
            !outputs.is_empty() && outputs.len() <= rows && outputs.iter().all(|&r| r < rows),
            "invalid output row selection"
        );
        let mut next = self.lengths.clone();
        for t in input {
            ensure!(
                t.slot < self.options.slots
                    && t.position < self.options.max_context
                    && (t.token as usize) < self.config.vocab_size,
                "token/slot/position out of bounds"
            );
            ensure!(
                t.position == next[t.slot],
                "noncontiguous KV append for slot {}: got {}, expected {}",
                t.slot,
                t.position,
                next[t.slot]
            );
            next[t.slot] += 1;
        }
        // Any subsequent error may leave partially written KV. Validation errors
        // above are recoverable; GPU execution errors require a fresh model.
        self.poisoned = true;
        self.last_output_rows = 0;
        let s = &mut self.scratch;
        let c = &self.config;
        let d = &self.device;
        s.ids
            .write(&input.iter().map(|t| t.token).collect::<Vec<_>>())?;
        s.slots
            .write(&input.iter().map(|t| t.slot as i32).collect::<Vec<_>>())?;
        s.positions
            .write(&input.iter().map(|t| t.position as i32).collect::<Vec<_>>())?;
        s.selection
            .write(&outputs.iter().map(|&r| r as u32).collect::<Vec<_>>())?;
        let (h, i, q, k) = (
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads * c.head_dim,
            c.num_key_value_heads * c.head_dim,
        );
        let shape = AttentionShape {
            rows,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            dim: c.head_dim,
            context: self.options.max_context,
            capacity: self.options.slots,
        };
        ops::embedding(
            d,
            &self.embedding,
            &s.ids,
            &mut s.hidden,
            rows,
            h,
            c.vocab_size,
        )?;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let step = || -> Result<()> {
                ops::norm(
                    d,
                    &s.hidden,
                    &layer.input_norm,
                    &mut s.norm,
                    rows,
                    h,
                    c.rms_norm_eps,
                )?;
                ops::gemm(d, &layer.q, &s.norm, &mut s.q, rows, q, h)?;
                ops::gemm(d, &layer.k, &s.norm, &mut s.k, rows, k, h)?;
                ops::gemm(d, &layer.v, &s.norm, &mut s.v, rows, k, h)?;
                ops::norm(
                    d,
                    &s.q,
                    &layer.q_norm,
                    &mut s.qn,
                    rows * c.num_attention_heads,
                    c.head_dim,
                    c.rms_norm_eps,
                )?;
                ops::norm(
                    d,
                    &s.k,
                    &layer.k_norm,
                    &mut s.kn,
                    rows * c.num_key_value_heads,
                    c.head_dim,
                    c.rms_norm_eps,
                )?;
                ops::rope(
                    d,
                    &mut s.qn,
                    &s.positions,
                    rows,
                    c.num_attention_heads,
                    c.head_dim,
                    c.rope_theta,
                )?;
                ops::rope(
                    d,
                    &mut s.kn,
                    &s.positions,
                    rows,
                    c.num_key_value_heads,
                    c.head_dim,
                    c.rope_theta,
                )?;
                ops::cache(
                    d,
                    &s.kn,
                    &s.v,
                    &mut layer.keys,
                    &mut layer.values,
                    &s.slots,
                    &s.positions,
                    shape,
                )?;
                ops::attention(
                    d,
                    &s.qn,
                    &layer.keys,
                    &layer.values,
                    &mut s.attn,
                    &s.slots,
                    &s.positions,
                    shape,
                )?;
                ops::gemm(d, &layer.o, &s.attn, &mut s.projected, rows, h, q)?;
                ops::add(d, &s.hidden, &s.projected, &mut s.residual, rows * h)?;
                ops::norm(
                    d,
                    &s.residual,
                    &layer.post_norm,
                    &mut s.norm,
                    rows,
                    h,
                    c.rms_norm_eps,
                )?;
                ops::gemm(d, &layer.gate, &s.norm, &mut s.gate, rows, i, h)?;
                ops::gemm(d, &layer.up, &s.norm, &mut s.up, rows, i, h)?;
                ops::silu(d, &s.gate, &s.up, &mut s.activation, rows * i)?;
                ops::gemm(d, &layer.down, &s.activation, &mut s.projected, rows, h, i)?;
                ops::add(d, &s.residual, &s.projected, &mut s.hidden, rows * h)?;
                Ok(())
            };
            let mut step = step;
            step().with_context(|| format!("ROCm Qwen3 layer {idx}"))?;
        }
        ops::norm(
            d,
            &s.hidden,
            &self.final_norm,
            &mut s.norm,
            rows,
            h,
            c.rms_norm_eps,
        )?;
        ops::embedding(
            d,
            &s.norm,
            &s.selection,
            &mut s.selected,
            outputs.len(),
            h,
            rows,
        )?;
        ops::gemm(
            d,
            &self.head,
            &s.selected,
            &mut s.logits,
            outputs.len(),
            c.decodable_vocab,
            h,
        )?;
        ops::top1(
            d,
            &s.logits,
            &mut s.predictions,
            outputs.len(),
            c.decodable_vocab,
        )?;
        let ids = s.predictions.read(outputs.len())?;
        self.lengths = next;
        self.last_output_rows = outputs.len();
        self.poisoned = false;
        Ok(ids)
    }
    pub fn logits(&self) -> Result<Vec<Vec<f32>>> {
        ensure!(!self.poisoned, "logits unavailable after a failed GPU step");
        let width = self.config.decodable_vocab;
        let values = self.scratch.logits.read(self.last_output_rows * width)?;
        Ok(values
            .chunks(width)
            .map(|row| row.iter().map(|x| x.to_f32()).collect())
            .collect())
    }
}
