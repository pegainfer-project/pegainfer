use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::dsv2_lite_kv_norm_into;

use super::DeviceKvPool;
use super::DeviceKvSlot;
use crate::Config;
use crate::config::test_lite_config;
use crate::host_ops::DecodeCache;
use crate::host_ops::LayerCache;
use crate::host_ops::append_kv_and_build_queries;
use crate::host_ops::compute_attention_host;
use crate::host_ops::rms_norm_host;

const TEST_MAX_SEQ_LEN: usize = 128;

fn one_layer_config() -> Config {
    let mut config = test_lite_config();
    config.num_hidden_layers = 1;
    config
}

struct Projections {
    state: u64,
}

impl Projections {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
        }
    }

    fn values(&mut self, len: usize, scale: f32) -> Vec<bf16> {
        (0..len)
            .map(|_| {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                let unit = (self.state >> 40) as f32 / (1u64 << 24) as f32;
                bf16::from_f32((unit * 2.0 - 1.0) * scale)
            })
            .collect()
    }
}

struct StepInputs {
    q: Vec<bf16>,
    kv_a: Vec<bf16>,
    kv_b: Vec<bf16>,
}

impl StepInputs {
    fn new(config: &Config, rng: &mut Projections, rows: usize) -> Self {
        Self {
            q: rng.values(rows * config.q_proj_rows(), 2.0),
            kv_a: rng.values(rows * config.kv_a_proj_rows(), 2.0),
            kv_b: rng.values(rows * config.kv_b_proj_rows(), 2.0),
        }
    }

    fn row(&self, config: &Config, row: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let slice = |data: &[bf16], width: usize| -> Vec<f32> {
            data[row * width..(row + 1) * width]
                .iter()
                .map(|value| value.to_f32())
                .collect()
        };
        (
            slice(&self.q, config.q_proj_rows()),
            slice(&self.kv_a, config.kv_a_proj_rows()),
            slice(&self.kv_b, config.kv_b_proj_rows()),
        )
    }
}

fn prefill_host_cache(config: &Config, rng: &mut Projections, len: usize) -> DecodeCache {
    let mut layer = LayerCache::default();
    let q = to_f32(&rng.values(len * config.q_proj_rows(), 2.0));
    let kv_a = to_f32(&rng.values(len * config.kv_a_proj_rows(), 2.0));
    let kv_b = to_f32(&rng.values(len * config.kv_b_proj_rows(), 2.0));
    let mut queries = vec![0.0; len * config.num_attention_heads * config.query_head_dim()];
    append_kv_and_build_queries(config, &q, &kv_a, &kv_b, 0, len, &mut queries, &mut layer);
    DecodeCache {
        layers: vec![layer],
        device: None,
    }
}

fn host_decode_row(
    config: &Config,
    cache: &mut DecodeCache,
    (q, kv_a, kv_b): (Vec<f32>, Vec<f32>, Vec<f32>),
) -> Vec<bf16> {
    let position = cache.position(config).expect("host cache position");
    let mut queries = vec![0.0; config.num_attention_heads * config.query_head_dim()];
    append_kv_and_build_queries(
        config,
        &q,
        &kv_a,
        &kv_b,
        position,
        1,
        &mut queries,
        &mut cache.layers[0],
    );
    compute_attention_host(config, &queries, &cache.layers[0], position, 1)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn device_decode_step(
    ctx: &DeviceContext,
    config: &Config,
    pool: &mut DeviceKvPool,
    slots: &mut [&mut DeviceKvSlot],
    inputs: &StepInputs,
) -> Vec<bf16> {
    let rows = slots.len();
    let views: Vec<&DeviceKvSlot> = slots.iter().map(|slot| &**slot).collect();
    pool.upload_decode_rows(ctx, &views)
        .expect("upload decode rows");
    let hidden = |data: &[bf16], width: usize| HiddenStates {
        data: ctx.stream.clone_htod(data).expect("step input H2D"),
        hidden_dim: width,
        seq_len: rows,
    };
    let q = hidden(&inputs.q, config.q_proj_rows());
    let kv_a = hidden(&inputs.kv_a, config.kv_a_proj_rows());
    let kv_b = hidden(&inputs.kv_b, config.kv_b_proj_rows());
    let mut out = HiddenStates::zeros(ctx, config.o_proj_cols(), rows).expect("attention out");
    pool.attention_into(ctx, 0, &q, &kv_a, &kv_b, &mut out)
        .expect("device attention");
    for slot in slots.iter_mut() {
        slot.advance();
    }
    let host = ctx.stream.clone_dtoh(&out.data).expect("attention out D2H");
    ctx.sync().expect("sync attention out");
    host
}

fn to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn assert_rows_match(config: &Config, device: &[bf16], host_rows: &[Vec<bf16>], label: &str) {
    let width = config.o_proj_cols();
    for (row, host) in host_rows.iter().enumerate() {
        let device_row = &device[row * width..(row + 1) * width];
        let mismatches: Vec<_> = device_row
            .iter()
            .zip(host)
            .enumerate()
            .filter(|(_, (device, host))| device.to_bits() != host.to_bits())
            .take(4)
            .map(|(dim, (device, host))| (dim, device.to_f32(), host.to_f32()))
            .collect();
        assert!(
            mismatches.is_empty(),
            "{label}: row {row} differs from the host oracle (dim, device, host): {mismatches:?}"
        );
    }
}

#[test]
fn kv_norm_matches_host_rms_norm_bitwise() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let eps = 1.0e-6;
    for (row_width, norm_dim, rows) in [(2048usize, 2048usize, 4usize), (576, 512, 3)] {
        let mut rng = Projections::new(row_width as u64);
        let input = rng.values(row_width * rows, 3.0);
        let weight = rng.values(norm_dim, 3.0);
        let weight_f32: Vec<f32> = weight.iter().map(|value| value.to_f32()).collect();

        let mut expected = vec![bf16::ZERO; norm_dim * rows];
        for row in 0..rows {
            let input_row: Vec<f32> = input[row * row_width..row * row_width + norm_dim]
                .iter()
                .map(|value| value.to_f32())
                .collect();
            rms_norm_host(
                &input_row,
                &weight_f32,
                eps,
                &mut expected[row * norm_dim..(row + 1) * norm_dim],
            );
        }

        let device_input = HiddenStates {
            data: ctx.stream.clone_htod(&input).expect("norm input H2D"),
            hidden_dim: row_width,
            seq_len: rows,
        };
        let device_weight = ctx.stream.clone_htod(&weight).expect("norm weight H2D");
        let mut out = HiddenStates::zeros(&ctx, norm_dim, rows).expect("norm out");
        dsv2_lite_kv_norm_into(&ctx, &device_input, &device_weight, norm_dim, eps, &mut out)
            .expect("device rms norm");
        let got = ctx.stream.clone_dtoh(&out.data).expect("norm out D2H");
        ctx.sync().expect("sync norm out");

        let mismatches: Vec<_> = got
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (got, expected))| got.to_bits() != expected.to_bits())
            .take(4)
            .collect();
        assert!(
            mismatches.is_empty(),
            "norm width {norm_dim}: device differs from host (index, device, host): {mismatches:?}"
        );
    }
}

#[test]
fn ragged_decode_rows_match_host_oracle() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let config = one_layer_config();
    let mut pool = DeviceKvPool::new(&ctx, &config, 4, TEST_MAX_SEQ_LEN).expect("device KV pool");
    let mut rng = Projections::new(635);

    let prefill_lens = [1usize, 5, 17, 64];
    let mut host_caches: Vec<_> = prefill_lens
        .iter()
        .map(|&len| prefill_host_cache(&config, &mut rng, len))
        .collect();
    let mut slots: Vec<_> = host_caches
        .iter()
        .map(|cache| {
            let mut slot = pool.acquire().expect("acquire slot");
            pool.upload_host_cache(&ctx, &config, &mut slot, cache)
                .expect("upload prefill");
            slot
        })
        .collect();

    for step in 0..4 {
        let inputs = StepInputs::new(&config, &mut rng, slots.len());
        let host_rows: Vec<_> = host_caches
            .iter_mut()
            .enumerate()
            .map(|(row, cache)| host_decode_row(&config, cache, inputs.row(&config, row)))
            .collect();
        let mut slot_refs: Vec<_> = slots.iter_mut().collect();
        let device = device_decode_step(&ctx, &config, &mut pool, &mut slot_refs, &inputs);
        assert_rows_match(&config, &device, &host_rows, &format!("step {step}"));
    }
    for (slot, len) in slots.iter().zip(prefill_lens) {
        assert_eq!(slot.len(), len + 4);
    }
}

#[test]
fn reused_slot_ignores_previous_owner_rows() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let config = one_layer_config();
    let mut pool = DeviceKvPool::new(&ctx, &config, 1, TEST_MAX_SEQ_LEN).expect("device KV pool");
    let mut rng = Projections::new(1089);

    let previous = prefill_host_cache(&config, &mut rng, 96);
    let mut slot = pool.acquire().expect("acquire slot");
    pool.upload_host_cache(&ctx, &config, &mut slot, &previous)
        .expect("upload previous owner");
    drop(slot);

    let mut host_cache = prefill_host_cache(&config, &mut rng, 3);
    let mut slot = pool.acquire().expect("reacquire released slot");
    assert_eq!(slot.len(), 0);
    pool.upload_host_cache(&ctx, &config, &mut slot, &host_cache)
        .expect("upload new owner");
    let inputs = StepInputs::new(&config, &mut rng, 1);
    let host_row = host_decode_row(&config, &mut host_cache, inputs.row(&config, 0));
    let device = device_decode_step(&ctx, &config, &mut pool, &mut [&mut slot], &inputs);
    assert_rows_match(&config, &device, &[host_row], "reused slot");
}

#[test]
fn slots_are_exclusive_until_dropped() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let config = one_layer_config();
    let pool = DeviceKvPool::new(&ctx, &config, 2, 4).expect("device KV pool");

    let first = pool.acquire().expect("first slot");
    let second = pool.acquire().expect("second slot");
    assert_ne!(first.index, second.index);
    assert!(
        pool.acquire().is_err(),
        "a full pool must refuse a third owner"
    );
    drop(first);
    let third = pool.acquire().expect("released slot is reusable");
    assert_ne!(third.index, second.index);
}

#[test]
fn full_slot_is_refused_before_launch() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let config = one_layer_config();
    let mut pool = DeviceKvPool::new(&ctx, &config, 1, 4).expect("device KV pool");
    let mut rng = Projections::new(4);

    let cache = prefill_host_cache(&config, &mut rng, 4);
    let mut slot = pool.acquire().expect("acquire slot");
    pool.upload_host_cache(&ctx, &config, &mut slot, &cache)
        .expect("prefill fills the slot exactly");
    let err = pool
        .upload_decode_rows(&ctx, &[&slot])
        .expect_err("a full slot has no position for the next token");
    assert!(format!("{err:#}").contains("is full"), "{err:#}");

    drop(slot);
    let too_long = prefill_host_cache(&config, &mut rng, 5);
    let mut other = pool.acquire().expect("reacquire released slot");
    let err = pool
        .upload_host_cache(&ctx, &config, &mut other, &too_long)
        .expect_err("prefill longer than the slot must be refused");
    assert!(format!("{err:#}").contains("exceeds"), "{err:#}");
}
