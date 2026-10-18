use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

use crate::Config;
use crate::device::activate;

#[derive(Default)]
pub(crate) struct DecodeCache {
    pub(crate) layers: Vec<LayerCache>,
}

#[derive(Default)]
pub(crate) struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    fn clone_for_graph_probe(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }

    pub(crate) fn len(&self, config: &Config) -> usize {
        let per_token = config.num_attention_heads * config.query_head_dim();
        if per_token == 0 {
            return 0;
        }
        self.keys.len() / per_token
    }

    pub(crate) fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }
}

impl DecodeCache {
    pub(crate) fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
        }
    }

    pub(crate) fn position(&self, config: &Config) -> Result<usize> {
        ensure!(
            self.layers.len() == config.num_hidden_layers,
            "decode cache layer count mismatch: cache={}, expected={}",
            self.layers.len(),
            config.num_hidden_layers
        );
        let Some(first) = self.layers.first() else {
            return Ok(0);
        };
        let position = first.len(config);
        for (layer_idx, layer) in self.layers.iter().enumerate().skip(1) {
            ensure!(
                layer.len(config) == position,
                "decode cache layer {layer_idx} position mismatch: cache_len={}, expected={position}",
                layer.len(config)
            );
        }
        Ok(position)
    }

    pub(crate) fn clone_for_graph_probe(&self) -> Self {
        // Keep this explicit so serving hot paths do not get blanket cache cloning.
        Self {
            layers: self
                .layers
                .iter()
                .map(LayerCache::clone_for_graph_probe)
                .collect(),
        }
    }
}

pub(crate) fn normalize_compressed_kv(
    config: &Config,
    kv_a_host: &[f32],
    norm_weight: &[f32],
    seq_len: usize,
) -> Vec<bf16> {
    assert_eq!(kv_a_host.len(), config.kv_a_proj_rows() * seq_len);
    assert_eq!(norm_weight.len(), config.kv_lora_rank);
    let mut out = vec![bf16::ZERO; config.kv_lora_rank * seq_len];
    for token in 0..seq_len {
        let src = &kv_a_host[token * config.kv_a_proj_rows()
            ..token * config.kv_a_proj_rows() + config.kv_lora_rank];
        let dst = &mut out[token * config.kv_lora_rank..(token + 1) * config.kv_lora_rank];
        rms_norm_host(src, norm_weight, config.rms_norm_eps, dst);
    }
    out
}

pub(crate) fn rms_norm_host(input: &[f32], weight: &[f32], eps: f32, out: &mut [bf16]) {
    assert_eq!(input.len(), weight.len());
    assert_eq!(out.len(), input.len());
    let sum_sq = input.iter().map(|value| value * value).sum::<f32>();
    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    for ((dst, value), scale) in out.iter_mut().zip(input).zip(weight) {
        let normalized = bf16::from_f32(value * inv_rms).to_f32();
        *dst = bf16::from_f32(normalized * scale);
    }
}

pub(crate) fn rms_norm_hidden_host(
    config: &Config,
    input: &[f32],
    weight: &[f32],
    seq_len: usize,
) -> Vec<bf16> {
    assert_eq!(input.len(), config.hidden_size * seq_len);
    assert_eq!(weight.len(), config.hidden_size);
    let mut out = vec![bf16::ZERO; config.hidden_size * seq_len];
    for token in 0..seq_len {
        let offset = token * config.hidden_size;
        rms_norm_host(
            &input[offset..offset + config.hidden_size],
            weight,
            config.rms_norm_eps,
            &mut out[offset..offset + config.hidden_size],
        );
    }
    out
}

pub(crate) fn append_kv_and_build_queries(
    config: &Config,
    q_host: &[f32],
    kv_a_host: &[f32],
    kv_b_host: &[f32],
    start_pos: usize,
    seq_len: usize,
    queries: &mut [f32],
    cache: &mut LayerCache,
) {
    let num_heads = config.num_attention_heads;
    let q_head_dim = config.query_head_dim();
    let kv_b_stride = config.qk_nope_head_dim + config.v_head_dim;
    assert_eq!(q_host.len(), config.q_proj_rows() * seq_len);
    assert_eq!(kv_a_host.len(), config.kv_a_proj_rows() * seq_len);
    assert_eq!(kv_b_host.len(), config.kv_b_proj_rows() * seq_len);
    assert_eq!(queries.len(), num_heads * q_head_dim * seq_len);
    let mut key = vec![0.0f32; q_head_dim];
    let rope_inv_freq: Vec<_> = (0..config.qk_rope_head_dim / 2)
        .map(|pair| rope_inv_freq(config, pair))
        .collect();
    let rope_mscale = rope_cache_mscale(config);

    for token in 0..seq_len {
        let pos = start_pos + token;
        let k_pe_raw = &kv_a_host[token * config.kv_a_proj_rows() + config.kv_lora_rank
            ..token * config.kv_a_proj_rows() + config.kv_lora_rank + config.qk_rope_head_dim];
        let k_pe = apply_deepseek_v2_rope(k_pe_raw, pos, config, &rope_inv_freq, rope_mscale);

        for head in 0..num_heads {
            let q_base = token * config.q_proj_rows() + head * q_head_dim;
            let query_base = (token * num_heads + head) * q_head_dim;
            queries[query_base..query_base + config.qk_nope_head_dim]
                .copy_from_slice(&q_host[q_base..q_base + config.qk_nope_head_dim]);
            let q_pe = apply_deepseek_v2_rope(
                &q_host[q_base + config.qk_nope_head_dim..q_base + q_head_dim],
                pos,
                config,
                &rope_inv_freq,
                rope_mscale,
            );
            queries[query_base + config.qk_nope_head_dim..query_base + q_head_dim]
                .copy_from_slice(&q_pe);

            let kv_b_base = token * config.kv_b_proj_rows() + head * kv_b_stride;
            key[..config.qk_nope_head_dim]
                .copy_from_slice(&kv_b_host[kv_b_base..kv_b_base + config.qk_nope_head_dim]);
            key[config.qk_nope_head_dim..q_head_dim].copy_from_slice(&k_pe);
            cache.keys.extend_from_slice(&key);
            cache.values.extend_from_slice(
                &kv_b_host[kv_b_base + config.qk_nope_head_dim
                    ..kv_b_base + config.qk_nope_head_dim + config.v_head_dim],
            );
        }
    }
}

pub(crate) fn append_kv_and_build_queries_decode_batch(
    config: &Config,
    q_host: &[f32],
    kv_a_host: &[f32],
    kv_b_host: &[f32],
    position: usize,
    queries: &mut [f32],
    caches: &mut [&mut LayerCache],
) {
    let batch_size = caches.len();
    assert_eq!(q_host.len(), config.q_proj_rows() * batch_size);
    assert_eq!(kv_a_host.len(), config.kv_a_proj_rows() * batch_size);
    assert_eq!(kv_b_host.len(), config.kv_b_proj_rows() * batch_size);
    let query_stride = config.num_attention_heads * config.query_head_dim();
    assert_eq!(queries.len(), query_stride * batch_size);

    for row in 0..batch_size {
        assert_eq!(caches[row].len(config), position);
        append_kv_and_build_queries(
            config,
            &q_host[row * config.q_proj_rows()..(row + 1) * config.q_proj_rows()],
            &kv_a_host[row * config.kv_a_proj_rows()..(row + 1) * config.kv_a_proj_rows()],
            &kv_b_host[row * config.kv_b_proj_rows()..(row + 1) * config.kv_b_proj_rows()],
            position,
            1,
            &mut queries[row * query_stride..(row + 1) * query_stride],
            caches[row],
        );
    }
}

fn apply_deepseek_v2_rope(
    input: &[f32],
    pos: usize,
    config: &Config,
    inv_freq: &[f32],
    mscale: f32,
) -> Vec<f32> {
    let dim = config.qk_rope_head_dim;
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    assert_eq!(input.len(), dim);
    assert_eq!(inv_freq.len(), half);
    assert_eq!(dim % 2, 0);
    for pair in 0..half {
        let angle = pos as f32 * inv_freq[pair];
        let cos = bf16::from_f32(angle.cos() * mscale).to_f32();
        let sin = bf16::from_f32(angle.sin() * mscale).to_f32();
        let x0 = input[2 * pair];
        let x1 = input[2 * pair + 1];
        let x0_cos = bf16::from_f32(x0 * cos).to_f32();
        let neg_x1_sin = bf16::from_f32(-x1 * sin).to_f32();
        let x1_cos = bf16::from_f32(x1 * cos).to_f32();
        let x0_sin = bf16::from_f32(x0 * sin).to_f32();
        out[pair] = bf16::from_f32(x0_cos + neg_x1_sin).to_f32();
        out[pair + half] = bf16::from_f32(x1_cos + x0_sin).to_f32();
    }
    out
}

fn rope_cache_mscale(config: &Config) -> f32 {
    let Some(rope_scaling) = &config.rope_scaling else {
        return 1.0;
    };
    yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale)
        / yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale_all_dim)
}

fn rope_inv_freq(config: &Config, pair: usize) -> f32 {
    let dim = config.qk_rope_head_dim;
    let base = config.rope_theta;
    let freq_extra = 1.0 / base.powf((2 * pair) as f32 / dim as f32);
    let Some(rope_scaling) = &config.rope_scaling else {
        return freq_extra;
    };

    let freq_inter = freq_extra / rope_scaling.factor;
    let low = yarn_find_correction_dim(rope_scaling.beta_fast as f32, dim, base, rope_scaling)
        .floor()
        .max(0.0);
    let high = yarn_find_correction_dim(rope_scaling.beta_slow as f32, dim, base, rope_scaling)
        .ceil()
        .min((dim - 1) as f32);
    let ramp = if (high - low).abs() < f32::EPSILON {
        if (pair as f32) <= low { 0.0 } else { 1.0 }
    } else {
        ((pair as f32 - low) / (high - low)).clamp(0.0, 1.0)
    };
    let inv_freq_mask = 1.0 - ramp;
    freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask
}

fn yarn_find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    rope_scaling: &crate::config::RopeScaling,
) -> f32 {
    dim as f32
        * (rope_scaling.original_max_position_embeddings as f32
            / (num_rotations * 2.0 * std::f32::consts::PI))
            .ln()
        / (2.0 * base.ln())
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

pub(crate) fn compute_attention_host(
    config: &Config,
    queries: &[f32],
    cache: &LayerCache,
    start_pos: usize,
    seq_len: usize,
) -> Vec<f32> {
    let num_heads = config.num_attention_heads;
    let q_head_dim = config.query_head_dim();
    let value_dim = config.v_head_dim;
    let scale = attention_softmax_scale(config);
    assert_eq!(queries.len(), seq_len * num_heads * q_head_dim);
    let mut out = vec![0.0f32; seq_len * config.o_proj_cols()];

    for token in 0..seq_len {
        let kv_len = start_pos + token + 1;
        for head in 0..num_heads {
            let q_base = (token * num_heads + head) * q_head_dim;
            let query = &queries[q_base..q_base + q_head_dim];
            let mut scores = vec![0.0f32; kv_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let k_base = (pos * num_heads + head) * q_head_dim;
                let key = &cache.keys[k_base..k_base + q_head_dim];
                let raw_score = bf16::from_f32(dot(query, key)).to_f32();
                *score = bf16::from_f32(raw_score * scale).to_f32();
            }
            let probs = softmax(&scores);
            let out_base = token * config.o_proj_cols() + head * value_dim;
            for (pos, prob) in probs.iter().enumerate() {
                let prob = bf16::from_f32(*prob).to_f32();
                let v_base = (pos * num_heads + head) * value_dim;
                let value = &cache.values[v_base..v_base + value_dim];
                for dim in 0..value_dim {
                    out[out_base + dim] += prob * value[dim];
                }
            }
        }
    }

    out
}

pub(crate) fn compute_attention_host_decode_batch(
    config: &Config,
    queries: &[f32],
    caches: &[&LayerCache],
    position: usize,
) -> Vec<f32> {
    let batch_size = caches.len();
    let query_stride = config.num_attention_heads * config.query_head_dim();
    assert_eq!(queries.len(), batch_size * query_stride);
    let mut out = vec![0.0f32; batch_size * config.o_proj_cols()];

    for row in 0..batch_size {
        let cache = caches[row];
        assert_eq!(cache.len(config), position + 1);
        let row_out = compute_attention_host(
            config,
            &queries[row * query_stride..(row + 1) * query_stride],
            cache,
            position,
            1,
        );
        out[row * config.o_proj_cols()..(row + 1) * config.o_proj_cols()].copy_from_slice(&row_out);
    }

    out
}

fn attention_softmax_scale(config: &Config) -> f32 {
    let mut scale = (config.query_head_dim() as f32).sqrt().recip();
    if let Some(rope_scaling) = &config.rope_scaling
        && rope_scaling.mscale_all_dim > 0.0
    {
        let mscale = yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale_all_dim);
        scale *= mscale * mscale;
    }
    scale
}

pub(crate) fn topk_softmax_routes(
    config: &Config,
    logits: &[f32],
    seq_len: usize,
) -> Vec<Vec<(usize, f32)>> {
    assert_eq!(logits.len(), seq_len * config.n_routed_experts);
    let mut routes = Vec::with_capacity(seq_len);
    for token in 0..seq_len {
        let scores =
            &logits[token * config.n_routed_experts..(token + 1) * config.n_routed_experts];
        let probs = softmax(scores);
        let mut indexed: Vec<_> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(lhs_idx, lhs), (rhs_idx, rhs)| {
            rhs.partial_cmp(lhs)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs_idx.cmp(rhs_idx))
        });
        indexed.truncate(config.num_experts_per_token);
        // Avoid adding an extra probability-sorted accumulation order on top of
        // HF's unsorted top-k gate path.
        indexed.sort_by_key(|(idx, _)| *idx);
        routes.push(indexed);
    }
    routes
}

pub(crate) fn gate_logits_host(config: &Config, input: &[bf16], gate_weight: &[f32]) -> Vec<f32> {
    assert_eq!(input.len() % config.hidden_size, 0);
    assert_eq!(
        gate_weight.len(),
        config.n_routed_experts * config.hidden_size
    );
    let mut logits = vec![0.0f32; (input.len() / config.hidden_size) * config.n_routed_experts];
    for (token, token_input) in input.chunks_exact(config.hidden_size).enumerate() {
        for expert in 0..config.n_routed_experts {
            let weight_base = expert * config.hidden_size;
            let mut acc = 0.0f32;
            for dim in 0..config.hidden_size {
                acc += token_input[dim].to_f32() * gate_weight[weight_base + dim];
            }
            logits[token * config.n_routed_experts + expert] = acc;
        }
    }
    logits
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for score in scores {
        let value = (score - max).exp();
        exp.push(value);
        sum += value;
    }
    if sum == 0.0 {
        return vec![0.0; scores.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

pub(crate) fn hidden_to_bf16(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<bf16>> {
    activate(ctx)?;
    let host = ctx.stream.clone_dtoh(&hidden.data)?;
    ctx.sync()?;
    Ok(host)
}

pub(crate) fn hidden_to_f32(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<f32>> {
    Ok(hidden_to_bf16(ctx, hidden)?
        .iter()
        .map(|value| value.to_f32())
        .collect())
}

pub(crate) fn hidden_from_bf16_host(
    ctx: &DeviceContext,
    data: &[bf16],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    activate(ctx)?;
    ensure!(
        data.len() == hidden_dim * seq_len,
        "hidden host data len mismatch: got {}, expected {}",
        data.len(),
        hidden_dim * seq_len
    );
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(data)?,
        hidden_dim,
        seq_len,
    })
}

pub(crate) fn hidden_from_f32_host(
    ctx: &DeviceContext,
    data: &[f32],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    let bf16_data: Vec<_> = data.iter().copied().map(bf16::from_f32).collect();
    hidden_from_bf16_host(ctx, &bf16_data, hidden_dim, seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_lite_config;

    #[test]
    fn rms_norm_host_rounds_normalized_hidden_before_weight() {
        let input = [3.0f32, 4.0];
        let weight = [3.0f32, 0.5];
        let eps = 0.0;
        let mut out = [bf16::ZERO; 2];

        rms_norm_host(&input, &weight, eps, &mut out);

        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let inv_rms = mean_square.sqrt().recip();
        let expected0 = bf16::from_f32(bf16::from_f32(3.0 * inv_rms).to_f32() * 3.0);
        let expected1 = bf16::from_f32(bf16::from_f32(4.0 * inv_rms).to_f32() * 0.5);
        assert_eq!(out, [expected0, expected1]);
    }

    #[test]
    fn clone_for_graph_probe_copies_decode_cache_contents() {
        let mut cache = DecodeCache {
            layers: vec![LayerCache {
                keys: vec![1.0, 2.0],
                values: vec![3.0, 4.0],
            }],
        };

        let cloned = cache.clone_for_graph_probe();
        cache.layers[0].keys[0] = 9.0;
        cache.layers[0].values[0] = 8.0;

        assert_eq!(cloned.layers[0].keys, vec![1.0, 2.0]);
        assert_eq!(cloned.layers[0].values, vec![3.0, 4.0]);
    }

    #[test]
    fn topk_softmax_routes_accumulates_selected_experts_by_id() {
        let config = test_lite_config();
        let mut logits = vec![-10.0f32; config.n_routed_experts];
        for (expert, logit) in [
            (20, 6.0),
            (7, 5.0),
            (10, 4.0),
            (41, 3.0),
            (54, 2.0),
            (58, 1.0),
        ] {
            logits[expert] = logit;
        }

        let routes = topk_softmax_routes(&config, &logits, 1);
        let experts: Vec<_> = routes[0].iter().map(|(expert, _)| *expert).collect();

        assert_eq!(experts, vec![7, 10, 20, 41, 54, 58]);
    }

    #[test]
    fn decode_batch_attention_matches_independent_single_row_caches() {
        let mut config = test_lite_config();
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        config.kv_lora_rank = 3;
        config.qk_nope_head_dim = 2;
        config.qk_rope_head_dim = 2;
        config.v_head_dim = 2;
        config.rope_scaling = None;

        let batch_size = 2;
        let position = 2;
        let mut single_caches = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let q_host = patterned(row, config.q_proj_rows() * position, 0.01);
            let kv_a_host = patterned(row + 10, config.kv_a_proj_rows() * position, 0.02);
            let kv_b_host = patterned(row + 20, config.kv_b_proj_rows() * position, 0.03);
            let mut cache = LayerCache::default();
            let mut queries =
                vec![0.0; config.num_attention_heads * config.query_head_dim() * position];
            append_kv_and_build_queries(
                &config,
                &q_host,
                &kv_a_host,
                &kv_b_host,
                0,
                position,
                &mut queries,
                &mut cache,
            );
            single_caches.push(cache);
        }
        let mut batch_caches: Vec<_> = single_caches
            .iter()
            .map(|cache| LayerCache {
                keys: cache.keys.clone(),
                values: cache.values.clone(),
            })
            .collect();

        let mut q_decode = Vec::new();
        let mut kv_a_decode = Vec::new();
        let mut kv_b_decode = Vec::new();
        let mut expected = Vec::new();
        for (row, cache) in single_caches.iter_mut().enumerate() {
            let q_host = patterned(row + 30, config.q_proj_rows(), 0.04);
            let kv_a_host = patterned(row + 40, config.kv_a_proj_rows(), 0.05);
            let kv_b_host = patterned(row + 50, config.kv_b_proj_rows(), 0.06);
            q_decode.extend_from_slice(&q_host);
            kv_a_decode.extend_from_slice(&kv_a_host);
            kv_b_decode.extend_from_slice(&kv_b_host);

            let mut queries = vec![0.0; config.num_attention_heads * config.query_head_dim()];
            append_kv_and_build_queries(
                &config,
                &q_host,
                &kv_a_host,
                &kv_b_host,
                position,
                1,
                &mut queries,
                cache,
            );
            expected.extend(compute_attention_host(
                &config, &queries, cache, position, 1,
            ));
        }

        let mut batch_queries =
            vec![0.0; batch_size * config.num_attention_heads * config.query_head_dim()];
        let mut batch_cache_refs: Vec<_> = batch_caches.iter_mut().collect();
        append_kv_and_build_queries_decode_batch(
            &config,
            &q_decode,
            &kv_a_decode,
            &kv_b_decode,
            position,
            &mut batch_queries,
            &mut batch_cache_refs,
        );
        let batch_cache_views: Vec<_> = batch_cache_refs
            .iter()
            .map(|cache| &**cache as &LayerCache)
            .collect();
        let actual = compute_attention_host_decode_batch(
            &config,
            &batch_queries,
            &batch_cache_views,
            position,
        );

        assert_eq!(actual, expected);
        for cache in batch_cache_refs {
            assert_eq!(cache.len(&config), position + 1);
        }
    }

    fn patterned(seed: usize, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let value = ((seed * 31 + idx * 17) % 23) as f32 - 11.0;
                value * scale
            })
            .collect()
    }
}
