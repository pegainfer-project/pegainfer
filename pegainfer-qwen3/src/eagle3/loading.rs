use anyhow::Context;
use anyhow::Result;
use log::debug;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::load_tensor_bool_host;
use pegainfer_core::weight_loader::load_tensor_i64_host;
use pegainfer_core::weight_loader::mmap_shards;

use super::Eagle3DraftModel;
use super::Eagle3Layer;
use crate::config::Eagle3Config;
use crate::weights::Qwen3Model;

impl Eagle3DraftModel {
    /// Load an EAGLE-3 drafter from a HF safetensors directory, validating its
    /// geometry against the already-loaded Qwen3 `target`.
    pub(crate) fn from_safetensors_for_target(
        ctx: &DeviceContext,
        model_path: &str,
        target: &Qwen3Model,
    ) -> Result<Self> {
        let config = Eagle3Config::from_file(model_path)
            .with_context(|| format!("load EAGLE-3 config from {model_path}"))?;
        config.validate_for_target(target.config())?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!(
            "Loading EAGLE-3 drafter from {model_path}: {} shard(s)",
            shard_paths.len()
        );
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let hidden = config.hidden_size;
        let attn_in = 2 * hidden; // embed ++ hidden concat
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let check2d = |m: &DeviceMatrix, name: &str, rows: usize, cols: usize| -> Result<()> {
            anyhow::ensure!(
                m.rows == rows && m.cols == cols,
                "EAGLE-3 {name} must be [{rows}, {cols}], got [{}, {}]",
                m.rows,
                m.cols
            );
            Ok(())
        };
        let check1d = |v: &DeviceVec, name: &str, len: usize| -> Result<()> {
            anyhow::ensure!(
                v.len == len,
                "EAGLE-3 {name} must be [{len}], got [{}]",
                v.len
            );
            Ok(())
        };

        // ---- the single "midlayer" decoder block ----
        // q/k/v share `2 * hidden_size` input columns (embed ++ hidden concat).
        let q_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.q_proj.weight",
        )?;
        check2d(&q_proj, "q_proj", q_dim, attn_in)?;
        let k_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.k_proj.weight",
        )?;
        check2d(&k_proj, "k_proj", kv_dim, attn_in)?;
        let v_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.v_proj.weight",
        )?;
        check2d(&v_proj, "v_proj", kv_dim, attn_in)?;
        let qkv_proj = DeviceMatrix::vstack(ctx, &[&q_proj, &k_proj, &v_proj])?;
        drop(q_proj);
        drop(k_proj);
        drop(v_proj);

        let gate_proj = load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.gate_proj.weight")?;
        check2d(&gate_proj, "gate_proj", inter, hidden)?;
        let up_proj = load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.up_proj.weight")?;
        check2d(&up_proj, "up_proj", inter, hidden)?;
        let gate_up_proj = DeviceMatrix::vstack(ctx, &[&gate_proj, &up_proj])?;
        drop(gate_proj);
        drop(up_proj);

        let o_proj = load_tensor_2d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.self_attn.o_proj.weight",
        )?;
        check2d(&o_proj, "o_proj", hidden, q_dim)?;
        let down_proj = load_tensor_2d(ctx, &shards, &weight_map, "midlayer.mlp.down_proj.weight")?;
        check2d(&down_proj, "down_proj", hidden, inter)?;

        let input_layernorm =
            load_tensor_1d(ctx, &shards, &weight_map, "midlayer.input_layernorm.weight")?;
        check1d(&input_layernorm, "input_layernorm", hidden)?;
        let hidden_norm = load_tensor_1d(ctx, &shards, &weight_map, "midlayer.hidden_norm.weight")?;
        check1d(&hidden_norm, "hidden_norm", hidden)?;
        let post_attention_layernorm = load_tensor_1d(
            ctx,
            &shards,
            &weight_map,
            "midlayer.post_attention_layernorm.weight",
        )?;
        check1d(
            &post_attention_layernorm,
            "post_attention_layernorm",
            hidden,
        )?;

        let midlayer = Eagle3Layer {
            input_layernorm,
            hidden_norm,
            qkv_proj,
            o_proj,
            post_attention_layernorm,
            gate_up_proj,
            down_proj,
            q_dim,
            kv_dim,
        };

        // ---- fusion, final norm, draft head, vocab-remap tables ----
        let fc = load_tensor_2d(ctx, &shards, &weight_map, "fc.weight")?;
        // Capture-compatibility invariant: EAGLE-3 fuses exactly THREE captured
        // target layers (low/mid/high) — `fc` maps `[3 * hidden] -> [hidden]`.
        check2d(&fc, "fc", hidden, 3 * hidden)?;
        let norm = load_tensor_1d(ctx, &shards, &weight_map, "norm.weight")?;
        check1d(&norm, "norm", hidden)?;
        let lm_head = load_tensor_2d(ctx, &shards, &weight_map, "lm_head.weight")?;
        check2d(&lm_head, "lm_head", config.draft_vocab_size, hidden)?;

        let d2t = load_tensor_i64_host(&shards, &weight_map, "d2t")?;
        let t2d = load_tensor_bool_host(&shards, &weight_map, "t2d")?;
        anyhow::ensure!(
            d2t.len() == config.draft_vocab_size,
            "EAGLE-3 d2t length {} does not match draft_vocab_size {}",
            d2t.len(),
            config.draft_vocab_size
        );
        anyhow::ensure!(
            t2d.len() == config.vocab_size,
            "EAGLE-3 t2d length {} does not match vocab_size {}",
            t2d.len(),
            config.vocab_size
        );

        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: config.head_dim,
                frequency_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings,
                theta: config.rope_theta,
            },
        )?;
        ctx.sync()?;

        Ok(Self {
            config,
            fc,
            midlayer,
            norm,
            lm_head,
            d2t,
            t2d,
            cos_cache,
            sin_cache,
        })
    }
}
