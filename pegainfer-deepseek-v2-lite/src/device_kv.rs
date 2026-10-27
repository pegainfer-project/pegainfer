use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::warn;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Dsv2LiteAttentionConfig;
use pegainfer_kernels::ops::dsv2_lite_decode_attention_into;

use crate::Config;
use crate::device::activate;
use crate::host_ops::DecodeCache;
use crate::host_ops::attention_softmax_scale;
use crate::host_ops::rope_cos_sin_table;
use crate::scheduler::DEFAULT_MAX_ACTIVE_REQUESTS;

pub(crate) const DEVICE_KV_SLOTS: usize = DEFAULT_MAX_ACTIVE_REQUESTS;
// Mirrors kMaxDecodeSeqLen in dsv2_lite_attention.cu.
const KERNEL_MAX_DECODE_SEQ_LEN: usize = 4096;
const DEVICE_MEMORY_UTILIZATION: f64 = 0.9;

pub(crate) fn attention_config(config: &Config, max_seq_len: usize) -> Dsv2LiteAttentionConfig {
    Dsv2LiteAttentionConfig {
        num_heads: config.num_attention_heads,
        qk_nope_head_dim: config.qk_nope_head_dim,
        qk_rope_head_dim: config.qk_rope_head_dim,
        v_head_dim: config.v_head_dim,
        kv_lora_rank: config.kv_lora_rank,
        max_seq_len,
        rms_norm_eps: config.rms_norm_eps,
        softmax_scale: attention_softmax_scale(config),
    }
}

pub(crate) struct DeviceKvPool {
    keys: Vec<CudaSlice<bf16>>,
    values: Vec<CudaSlice<bf16>>,
    slot_ids: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    rope_cos_sin: CudaSlice<bf16>,
    attention: Dsv2LiteAttentionConfig,
    num_slots: usize,
    key_token_elems: usize,
    value_token_elems: usize,
    kv_bytes: usize,
    free: Arc<Mutex<Vec<usize>>>,
}

pub(crate) struct DeviceKvSlot {
    index: usize,
    len: usize,
    free: Arc<Mutex<Vec<usize>>>,
}

impl DeviceKvSlot {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn advance(&mut self) {
        self.len += 1;
    }
}

impl Drop for DeviceKvSlot {
    fn drop(&mut self) {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        debug_assert!(!free.contains(&self.index), "device KV slot double free");
        free.push(self.index);
    }
}

impl DeviceKvPool {
    pub(crate) fn new_within_budget(
        ctx: &DeviceContext,
        config: &Config,
        num_slots: usize,
        max_seq_len: usize,
    ) -> Result<Option<Self>> {
        if max_seq_len > KERNEL_MAX_DECODE_SEQ_LEN {
            warn!(
                "device decode KV disabled: max_seq_len {max_seq_len} exceeds the attention kernel limit {KERNEL_MAX_DECODE_SEQ_LEN}; decode stays on host attention"
            );
            return Ok(None);
        }
        let required = Self::bytes_for(config, num_slots, max_seq_len);
        activate(ctx)?;
        let (free, total) = cudarc::driver::result::mem_get_info()
            .context("query device memory for the device KV pool")?;
        let used = total - free;
        let budget = (total as f64 * DEVICE_MEMORY_UTILIZATION) as usize;
        if used + required > budget {
            warn!(
                "device decode KV disabled: pool needs {} MiB, {} MiB used of {} MiB budget; decode stays on host attention",
                required >> 20,
                used >> 20,
                budget >> 20
            );
            return Ok(None);
        }
        Self::new(ctx, config, num_slots, max_seq_len).map(Some)
    }

    fn bytes_for(config: &Config, num_slots: usize, max_seq_len: usize) -> usize {
        let token_elems =
            config.num_attention_heads * (config.query_head_dim() + config.v_head_dim);
        config.num_hidden_layers * num_slots * max_seq_len * token_elems * size_of::<bf16>()
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config,
        num_slots: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        activate(ctx)?;
        let key_token_elems = config.num_attention_heads * config.query_head_dim();
        let value_token_elems = config.num_attention_heads * config.v_head_dim;
        let kv_bytes = Self::bytes_for(config, num_slots, max_seq_len);
        let slot_tokens = num_slots * max_seq_len;
        let mut keys = Vec::with_capacity(config.num_hidden_layers);
        let mut values = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            keys.push(
                ctx.stream
                    .alloc_zeros::<bf16>(slot_tokens * key_token_elems)
                    .with_context(|| format!("allocate device KV keys for layer {layer}"))?,
            );
            values.push(
                ctx.stream
                    .alloc_zeros::<bf16>(slot_tokens * value_token_elems)
                    .with_context(|| format!("allocate device KV values for layer {layer}"))?,
            );
        }
        let slot_ids = ctx.stream.alloc_zeros::<i32>(num_slots)?;
        let positions = ctx.stream.alloc_zeros::<i32>(num_slots)?;
        let rope_cos_sin = ctx
            .stream
            .clone_htod(&rope_cos_sin_table(config, max_seq_len))
            .context("upload RoPE cos/sin table")?;
        ctx.sync()?;
        Ok(Self {
            keys,
            values,
            slot_ids,
            positions,
            rope_cos_sin,
            attention: attention_config(config, max_seq_len),
            num_slots,
            key_token_elems,
            value_token_elems,
            kv_bytes,
            free: Arc::new(Mutex::new((0..num_slots).rev().collect())),
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.kv_bytes
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.attention.max_seq_len
    }

    pub(crate) fn acquire(&self) -> Result<DeviceKvSlot> {
        let index = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .ok_or_else(|| {
                anyhow!(
                    "DeepSeek-V2-Lite device KV pool exhausted: all {} slots are owned",
                    self.num_slots
                )
            })?;
        Ok(DeviceKvSlot {
            index,
            len: 0,
            free: Arc::clone(&self.free),
        })
    }

    pub(crate) fn upload_host_cache(
        &mut self,
        ctx: &DeviceContext,
        config: &Config,
        slot: &mut DeviceKvSlot,
        cache: &DecodeCache,
    ) -> Result<()> {
        let len = cache.position(config)?;
        ensure!(
            len <= self.max_seq_len(),
            "prefill length {len} exceeds device KV slot capacity {}",
            self.max_seq_len()
        );
        activate(ctx)?;
        let key_base = slot.index * self.max_seq_len() * self.key_token_elems;
        let value_base = slot.index * self.max_seq_len() * self.value_token_elems;
        for (layer_idx, layer) in cache.layers.iter().enumerate() {
            let keys: Vec<bf16> = layer.keys().iter().copied().map(bf16::from_f32).collect();
            let values: Vec<bf16> = layer.values().iter().copied().map(bf16::from_f32).collect();
            let mut key_dst = self.keys[layer_idx].slice_mut(key_base..key_base + keys.len());
            ctx.stream
                .memcpy_htod(&keys, &mut key_dst)
                .with_context(|| format!("upload device KV keys for layer {layer_idx}"))?;
            let mut value_dst =
                self.values[layer_idx].slice_mut(value_base..value_base + values.len());
            ctx.stream
                .memcpy_htod(&values, &mut value_dst)
                .with_context(|| format!("upload device KV values for layer {layer_idx}"))?;
        }
        ctx.sync()?;
        slot.len = len;
        Ok(())
    }

    pub(crate) fn upload_decode_rows(
        &mut self,
        ctx: &DeviceContext,
        slots: &[&DeviceKvSlot],
    ) -> Result<()> {
        ensure!(
            !slots.is_empty() && slots.len() <= self.num_slots,
            "device KV decode batch {} must be in 1..={}",
            slots.len(),
            self.num_slots
        );
        let mut slot_ids = Vec::with_capacity(slots.len());
        let mut positions = Vec::with_capacity(slots.len());
        for slot in slots {
            ensure!(
                slot.len < self.max_seq_len(),
                "device KV slot {} is full: {} tokens of capacity {}",
                slot.index,
                slot.len,
                self.max_seq_len()
            );
            slot_ids.push(i32::try_from(slot.index)?);
            positions.push(i32::try_from(slot.len)?);
        }
        activate(ctx)?;
        let mut slot_dst = self.slot_ids.slice_mut(0..slots.len());
        ctx.stream
            .memcpy_htod(&slot_ids, &mut slot_dst)
            .context("upload device KV slot ids")?;
        let mut position_dst = self.positions.slice_mut(0..slots.len());
        ctx.stream
            .memcpy_htod(&positions, &mut position_dst)
            .context("upload device KV positions")?;
        Ok(())
    }

    pub(crate) fn attention_into(
        &mut self,
        ctx: &DeviceContext,
        layer_idx: usize,
        q: &HiddenStates,
        kv_a: &HiddenStates,
        kv_b: &HiddenStates,
        out: &mut HiddenStates,
    ) -> Result<()> {
        dsv2_lite_decode_attention_into(
            ctx,
            self.attention,
            q,
            kv_a,
            kv_b,
            &self.slot_ids,
            &self.positions,
            self.num_slots,
            &self.rope_cos_sin,
            &mut self.keys[layer_idx],
            &mut self.values[layer_idx],
            out,
        )
    }
}

#[cfg(test)]
mod tests;
