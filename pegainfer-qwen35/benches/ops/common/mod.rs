#![allow(dead_code)]

use std::hint::black_box;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use criterion::Bencher;
use criterion::BenchmarkGroup;
use criterion::measurement::WallTime;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

pub(crate) const VECTOR_DIM: usize = 1024;
pub(crate) const OUT_DIM: usize = 1024;
pub(crate) const INTERMEDIATE_DIM: usize = 4096;
pub(crate) const VOCAB_SIZE: usize = 32_768;
pub(crate) const BATCH_SEQ_LEN: usize = 64;
pub(crate) const ATTN_SEQ_LEN: usize = 64;
pub(crate) const MAX_SEQ_LEN: usize = 4096;
pub(crate) const Q_HEADS_128: usize = 8;
pub(crate) const KV_HEADS_128: usize = 2;
pub(crate) const HEAD_DIM_128: usize = 128;
pub(crate) const Q_HEADS_256: usize = 4;
pub(crate) const KV_HEADS_256: usize = 1;
pub(crate) const HEAD_DIM_256: usize = 256;
pub(crate) const ROTARY_DIM_256: usize = 64;
pub(crate) const LINEAR_KEY_HEADS: usize = 4;
pub(crate) const LINEAR_VALUE_HEADS: usize = 8;
pub(crate) const LINEAR_KEY_DIM: usize = 64;
pub(crate) const LINEAR_VALUE_DIM: usize = 64;
pub(crate) const CONV_KERNEL_SIZE: usize = 4;
pub(crate) const EPS: f32 = 1e-6;
pub(crate) const ROPE_THETA_QWEN3: f32 = 1_000_000.0;
pub(crate) const ROPE_THETA_QWEN35: f32 = 10_000_000.0;

// Qwen3.5-4B actual model dimensions
pub(crate) const QWEN35_4B_HIDDEN: usize = 2560;
pub(crate) const QWEN35_4B_INTERMEDIATE: usize = 9216;
pub(crate) const QWEN35_4B_Q_HEADS: usize = 16;
pub(crate) const QWEN35_4B_KV_HEADS: usize = 4;
pub(crate) const QWEN35_4B_HEAD_DIM: usize = 256;
pub(crate) const QWEN35_4B_ROTARY_DIM: usize = 64;
pub(crate) const QWEN35_4B_LINEAR_K_HEADS: usize = 16;
pub(crate) const QWEN35_4B_LINEAR_V_HEADS: usize = 32;
pub(crate) const QWEN35_4B_LINEAR_K_DIM: usize = 128;
pub(crate) const QWEN35_4B_LINEAR_V_DIM: usize = 128;
pub(crate) const QWEN35_4B_ROPE_THETA: f32 = 10_000_000.0;
pub(crate) const QWEN35_4B_VOCAB: usize = 248_320;

pub(crate) fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);
}

pub(crate) fn iter_sync<T, F>(b: &mut Bencher<'_>, ctx: &DeviceContext, mut f: F)
where
    F: FnMut() -> T,
{
    ctx.sync().expect("CUDA pre-benchmark sync failed");
    b.iter(|| {
        ctx.sync().expect("CUDA pre-iteration sync failed");
        let value = f();
        ctx.sync().expect("CUDA post-iteration sync failed");
        black_box(value);
    });
    ctx.sync().expect("CUDA post-benchmark sync failed");
}

pub(crate) fn bf16_data(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 97) as f32 - 48.0;
            bf16::from_f32(centered * 0.03125)
        })
        .collect()
}

pub(crate) fn bf16_data_scaled(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 89) as f32 - 44.0;
            bf16::from_f32(centered * scale)
        })
        .collect()
}

pub(crate) fn positive_bf16_data(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|idx| bf16::from_f32(0.5 + (idx % 23) as f32 * 0.03125))
        .collect()
}

pub(crate) fn f32_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 79) as f32 - 39.0) * 0.015_625)
        .collect()
}

pub(crate) fn device_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &bf16_data(len))
}

pub(crate) fn device_vec_scaled(ctx: &DeviceContext, len: usize, scale: f32) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &bf16_data_scaled(len, scale))
}

pub(crate) fn positive_device_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &positive_bf16_data(len))
}

pub(crate) fn device_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    DeviceMatrix::from_host(ctx, &bf16_data(rows * cols), rows, cols)
}

pub(crate) fn embedding_matrix(
    ctx: &DeviceContext,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<DeviceMatrix> {
    DeviceMatrix::from_host(
        ctx,
        &bf16_data_scaled(vocab_size * hidden_size, 0.007_812_5),
        vocab_size,
        hidden_size,
    )
}

pub(crate) fn hidden_states(
    ctx: &DeviceContext,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    let host = bf16_data(hidden_dim * seq_len);
    let data = ctx
        .stream
        .clone_htod(&host)
        .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
    Ok(HiddenStates {
        data,
        hidden_dim,
        seq_len,
    })
}

pub(crate) fn f32_slice(ctx: &DeviceContext, len: usize) -> Result<CudaSlice<f32>> {
    ctx.stream
        .clone_htod(&f32_data(len))
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}

pub(crate) fn zero_f32_slice(ctx: &DeviceContext, len: usize) -> Result<CudaSlice<f32>> {
    ctx.stream
        .alloc_zeros(len)
        .map_err(|e| anyhow!("Alloc failed: {}", e))
}

pub(crate) fn token_ids(
    ctx: &DeviceContext,
    seq_len: usize,
    vocab_size: usize,
) -> Result<CudaSlice<u32>> {
    let host: Vec<u32> = (0..seq_len)
        .map(|idx| (idx % vocab_size.max(1)) as u32)
        .collect();
    ctx.stream
        .clone_htod(&host)
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}

pub(crate) fn decode_token_id(ctx: &DeviceContext, token_id: u32) -> Result<CudaSlice<u32>> {
    ctx.stream
        .clone_htod(&[token_id])
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}
