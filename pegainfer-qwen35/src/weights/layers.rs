//! Qwen3.5 weights: layer container types + their weight loaders. Split out of
//! weights.rs. Reaches the model/config via `use super::*;`.

use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_1d_f32;
use pegainfer_core::weight_loader::load_tensor_1d_f32_shard;
use pegainfer_core::weight_loader::load_tensor_1d_stitch;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::load_tensor_2d_col_shard;
use pegainfer_core::weight_loader::load_tensor_2d_row_shard;
use pegainfer_core::weight_loader::load_tensor_2d_row_stitch;

use super::*;

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub(crate) struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub(crate) q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub(crate) k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub(crate) v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub(crate) o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub(crate) struct LinearAttentionLayer {
    /// Fused QKV projection: [local_linear_qkv_dim, hidden_size]; row layout
    /// in `linear_qkv_shard_segments`.
    pub(crate) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [local_linear_z_dim, hidden_size]
    pub(crate) in_proj_z: DeviceMatrix,
    /// Beta projection: [local_linear_num_value_heads, hidden_size]
    pub(crate) in_proj_b: DeviceMatrix,
    /// Alpha projection: [local_linear_num_value_heads, hidden_size]
    pub(crate) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [local_linear_qkv_dim * conv_kernel_dim]
    /// (flattened from [qkv_dim, 1, 4]); channel layout mirrors in_proj_qkv.
    pub(crate) conv1d_weight: DeviceVec,
    /// dt_bias: [local_linear_num_value_heads] bf16
    pub(crate) dt_bias: DeviceVec,
    /// A_log: [local_linear_num_value_heads] f32
    pub(crate) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32 —
    /// head-shared, so replicated on every rank.
    pub(crate) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, local_linear_z_dim] (row-parallel;
    /// the layer all-reduces the partial hidden sum under TP).
    pub(crate) out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub(crate) enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

impl LayerKind {
    pub(super) fn full_attention(&self) -> Option<&FullAttentionLayer> {
        match self {
            Self::FullAttention(attn) => Some(attn),
            Self::LinearAttention(_) => None,
        }
    }

    pub(super) fn linear_attention(&self) -> Option<&LinearAttentionLayer> {
        match self {
            Self::LinearAttention(attn) => Some(attn),
            Self::FullAttention(_) => None,
        }
    }
}

/// MLP layer weights (shared between both layer types).
#[allow(clippy::struct_field_names)]
pub(crate) struct MLP35 {
    pub(crate) gate_up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
}

impl MLP35 {
    fn load(src: &WeightSource, prefix: &str) -> Result<Self> {
        let gate_proj =
            src.row_shard_if_needed(&format!("{prefix}.mlp.gate_proj.weight"), src.intermediate)?;
        let up_proj =
            src.row_shard_if_needed(&format!("{prefix}.mlp.up_proj.weight"), src.intermediate)?;
        let gate_up_proj = DeviceMatrix::vstack(src.ctx, &[&gate_proj, &up_proj])?;
        drop(gate_proj);
        drop(up_proj);
        Ok(Self {
            gate_up_proj,
            down_proj: src
                .col_shard_if_needed(&format!("{prefix}.mlp.down_proj.weight"), src.intermediate)?,
        })
    }
}

/// Transformer block for Qwen3.5.
pub(crate) struct TransformerBlock35 {
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) attn: LayerKind,
    pub(crate) post_attention_layernorm: DeviceVec,
    pub(crate) mlp: MLP35,
}

impl TransformerBlock35 {
    /// Load one decoder block: `prefix` is the layer's tensor prefix (e.g.
    /// `model.language_model.layers.3`).
    pub(super) fn load(src: &WeightSource, prefix: &str, layer_type: LayerType) -> Result<Self> {
        let attn = match layer_type {
            LayerType::FullAttention => LayerKind::FullAttention(FullAttentionLayer::load(
                src,
                &format!("{prefix}.self_attn"),
            )?),
            LayerType::LinearAttention => LayerKind::LinearAttention(LinearAttentionLayer::load(
                src,
                &format!("{prefix}.linear_attn"),
            )?),
        };
        Ok(Self {
            input_layernorm: src.tensor_1d(&format!("{prefix}.input_layernorm.weight"))?,
            attn,
            post_attention_layernorm: src
                .tensor_1d(&format!("{prefix}.post_attention_layernorm.weight"))?,
            mlp: MLP35::load(src, prefix)?,
        })
    }
}

impl FullAttentionLayer {
    fn load(src: &WeightSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            q_proj: src.gated_q_proj(&format!("{prefix}.q_proj.weight"))?,
            k_proj: src.row_shard_if_needed(&format!("{prefix}.k_proj.weight"), src.kv_rows)?,
            v_proj: src.row_shard_if_needed(&format!("{prefix}.v_proj.weight"), src.kv_rows)?,
            o_proj: src.col_shard_if_needed(&format!("{prefix}.o_proj.weight"), src.q_cols)?,
            q_norm: src.tensor_1d(&format!("{prefix}.q_norm.weight"))?,
            k_norm: src.tensor_1d(&format!("{prefix}.k_norm.weight"))?,
        })
    }
}

impl LinearAttentionLayer {
    fn load(src: &WeightSource, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_proj_qkv: src.linear_in_proj_qkv(&format!("{prefix}.in_proj_qkv.weight"))?,
            in_proj_z: src
                .row_shard_if_needed(&format!("{prefix}.in_proj_z.weight"), src.linear_z)?,
            in_proj_b: src.row_shard_if_needed(
                &format!("{prefix}.in_proj_b.weight"),
                src.linear_value_heads,
            )?,
            in_proj_a: src.row_shard_if_needed(
                &format!("{prefix}.in_proj_a.weight"),
                src.linear_value_heads,
            )?,
            conv1d_weight: src.linear_conv1d(&format!("{prefix}.conv1d.weight"))?,
            dt_bias: src
                .tensor_1d_shard_if_needed(&format!("{prefix}.dt_bias"), src.linear_value_heads)?,
            a_log: src.tensor_1d_f32_shard_if_needed(
                &format!("{prefix}.A_log"),
                src.linear_value_heads,
            )?,
            norm_weight: src.tensor_1d_f32(&format!("{prefix}.norm.weight"))?,
            out_proj: src
                .col_shard_if_needed(&format!("{prefix}.out_proj.weight"), src.linear_z)?,
        })
    }
}

/// One model load's tensor source plus the loop-invariant TP shard ranges the
/// per-layer loaders consume. Construct once per load; every loader names only
/// the tensor it wants.
pub(super) struct WeightSource<'a> {
    ctx: &'a DeviceContext,
    shards: &'a [SafeTensors<'a>],
    weight_map: &'a HashMap<String, usize>,
    geometry: LocalGeometry,
    /// Full-attention q_proj rows as per-head [q, gate] chunks.
    gated_q: (usize, usize),
    /// o_proj column shard over the full-attention q dim.
    q_cols: (usize, usize),
    /// k/v_proj row shard.
    kv_rows: (usize, usize),
    /// MLP intermediate row (gate/up) and column (down) shard.
    intermediate: (usize, usize),
    /// Linear-attention value-head unit: in_proj_b/a rows, dt_bias, A_log.
    linear_value_heads: (usize, usize),
    /// Linear-attention z dim: in_proj_z rows and out_proj columns.
    linear_z: (usize, usize),
    /// Per-segment row slices inside the fused linear qkv projection.
    linear_qkv: [(usize, usize); 3],
    /// The same slices in conv1d channel-tap units.
    linear_conv1d: [(usize, usize); 3],
}

impl<'a> WeightSource<'a> {
    pub(super) fn new(
        ctx: &'a DeviceContext,
        shards: &'a [SafeTensors<'a>],
        weight_map: &'a HashMap<String, usize>,
        config: &Config35,
        geometry: LocalGeometry,
    ) -> Self {
        Self {
            ctx,
            shards,
            weight_map,
            geometry,
            gated_q: full_attention_gated_q_shard_range(config, geometry),
            q_cols: geometry.shard_range(config.full_attn_q_dim()),
            kv_rows: geometry.shard_range(config.full_attn_kv_dim()),
            intermediate: geometry.shard_range(config.intermediate_size),
            linear_value_heads: geometry.shard_range(config.linear_num_value_heads),
            linear_z: geometry.shard_range(config.linear_attn_z_dim()),
            linear_qkv: linear_qkv_shard_segments(config, geometry),
            linear_conv1d: linear_conv1d_shard_segments(config, geometry),
        }
    }

    pub(super) fn tensor_2d(&self, name: &str) -> Result<DeviceMatrix> {
        load_tensor_2d(self.ctx, self.shards, self.weight_map, name)
    }

    pub(super) fn tensor_1d(&self, name: &str) -> Result<DeviceVec> {
        load_tensor_1d(self.ctx, self.shards, self.weight_map, name)
    }

    pub(super) fn tensor_1d_f32(&self, name: &str) -> Result<CudaSlice<f32>> {
        load_tensor_1d_f32(self.ctx, self.shards, self.weight_map, name)
    }

    fn row_shard_if_needed(
        &self,
        name: &str,
        (row_offset, rows): (usize, usize),
    ) -> Result<DeviceMatrix> {
        if self.geometry.is_sharded() {
            load_tensor_2d_row_shard(
                self.ctx,
                self.shards,
                self.weight_map,
                name,
                row_offset,
                rows,
            )
        } else {
            self.tensor_2d(name)
        }
    }

    fn col_shard_if_needed(
        &self,
        name: &str,
        (col_offset, cols): (usize, usize),
    ) -> Result<DeviceMatrix> {
        if self.geometry.is_sharded() {
            load_tensor_2d_col_shard(
                self.ctx,
                self.shards,
                self.weight_map,
                name,
                col_offset,
                cols,
            )
        } else {
            self.tensor_2d(name)
        }
    }

    fn tensor_1d_shard_if_needed(
        &self,
        name: &str,
        (offset, len): (usize, usize),
    ) -> Result<DeviceVec> {
        if self.geometry.is_sharded() {
            load_tensor_1d_stitch(
                self.ctx,
                self.shards,
                self.weight_map,
                name,
                &[(offset, len)],
            )
        } else {
            self.tensor_1d(name)
        }
    }

    fn tensor_1d_f32_shard_if_needed(
        &self,
        name: &str,
        (offset, len): (usize, usize),
    ) -> Result<CudaSlice<f32>> {
        if self.geometry.is_sharded() {
            load_tensor_1d_f32_shard(self.ctx, self.shards, self.weight_map, name, offset, len)
        } else {
            self.tensor_1d_f32(name)
        }
    }

    /// Fused linear qkv: stitch this rank's head-local slice out of each of the
    /// three global segments rather than cutting one flat row range.
    fn linear_in_proj_qkv(&self, name: &str) -> Result<DeviceMatrix> {
        if !self.geometry.is_sharded() {
            return self.tensor_2d(name);
        }
        load_tensor_2d_row_stitch(
            self.ctx,
            self.shards,
            self.weight_map,
            name,
            &self.linear_qkv,
        )
    }

    /// conv1d channels mirror the fused qkv rows, so they stitch with the same
    /// segments scaled into kernel-tap units.
    fn linear_conv1d(&self, name: &str) -> Result<DeviceVec> {
        if !self.geometry.is_sharded() {
            return self.tensor_1d(name);
        }
        load_tensor_1d_stitch(
            self.ctx,
            self.shards,
            self.weight_map,
            name,
            &self.linear_conv1d,
        )
    }

    /// Q projection carries a per-head output gate, so its rows shard per head
    /// (keeping each head's [q, gate] chunk adjacent), not as one flat range.
    fn gated_q_proj(&self, name: &str) -> Result<DeviceMatrix> {
        if !self.geometry.is_sharded() {
            return self.tensor_2d(name);
        }
        let (row_offset, rows) = self.gated_q;
        load_tensor_2d_row_shard(
            self.ctx,
            self.shards,
            self.weight_map,
            name,
            row_offset,
            rows,
        )
    }
}

/// Row ranges this rank owns inside the fused global linear-attention qkv
/// projection. The checkpoint stores [all q rows | all k rows | all v rows];
/// each segment contributes its head-local slice so the rank's stitched rows
/// stay [q_local | k_local | v_local]. Never reblock across segments — q rows
/// key on key heads, v rows on value heads (the gated-q lesson).
fn linear_qkv_shard_segments(config: &Config35, geometry: LocalGeometry) -> [(usize, usize); 3] {
    let global_q = config.linear_num_key_heads * config.linear_key_head_dim;
    let global_k = global_q;
    let global_v = config.linear_attn_z_dim();
    let (q_rel, q_rows) = geometry.shard_range(global_q);
    let (k_rel, k_rows) = geometry.shard_range(global_k);
    let (v_rel, v_rows) = geometry.shard_range(global_v);
    [
        (q_rel, q_rows),
        (global_q + k_rel, k_rows),
        (global_q + global_k + v_rel, v_rows),
    ]
}

/// The flattened conv1d weight keeps each channel's kernel taps contiguous
/// ([channel, 1, kernel_dim]); its channel layout mirrors the fused qkv rows,
/// so shard it with the same per-segment ranges scaled by the kernel dim.
fn linear_conv1d_shard_segments(config: &Config35, geometry: LocalGeometry) -> [(usize, usize); 3] {
    let kernel_dim = config.linear_conv_kernel_dim;
    linear_qkv_shard_segments(config, geometry)
        .map(|(offset, len)| (offset * kernel_dim, len * kernel_dim))
}

/// HF/PegaInfer kernels interpret q_proj rows as per-head [q, gate] chunks.
/// Keep each local head's q rows adjacent to its gate rows.
fn full_attention_gated_q_shard_range(
    config: &Config35,
    geometry: LocalGeometry,
) -> (usize, usize) {
    let local_heads = geometry.local_num_attention_heads();
    let head_start = geometry.rank() * local_heads;
    (
        head_start * config.head_dim * 2,
        local_heads * config.head_dim * 2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config35 {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
  "max_position_embeddings": 262144,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 151645
  }
}"#,
        )
        .unwrap();
        Config35::from_file(dir.path().to_str().unwrap()).expect("fixture validates")
    }

    fn test_geometry(rank: usize, world_size: usize) -> LocalGeometry {
        let config = test_config();
        let tp = TensorParallelConfig::try_from((rank, world_size)).unwrap();
        LocalGeometry::try_new(&config, tp).unwrap()
    }

    #[test]
    fn linear_qkv_shard_segments_stitch_head_local_slices() {
        // test_config: k heads 16, v heads 32, head dim 128 → q=k=2048, v=4096.
        let config = test_config();
        let rank0 = linear_qkv_shard_segments(&config, test_geometry(0, 2));
        assert_eq!(rank0, [(0, 1024), (2048, 1024), (4096, 2048)]);

        let rank1 = linear_qkv_shard_segments(&config, test_geometry(1, 2));
        assert_eq!(rank1, [(1024, 1024), (3072, 1024), (6144, 2048)]);

        // Every rank's stitched rows tile [0, qkv) with no overlap: each
        // segment's local slices across ranks are contiguous and complete.
        for (r0, r1) in rank0.iter().zip(rank1.iter()) {
            assert_eq!(r0.1, r1.1);
            assert_eq!(r1.0, r0.0 + r0.1);
        }
    }

    #[test]
    fn gated_q_shard_range_keeps_matching_q_and_gate_rows() {
        let config = test_config();

        let rank0 = full_attention_gated_q_shard_range(&config, test_geometry(0, 2));
        assert_eq!(rank0, (0, 4096));

        // Rank 1 starts at its own first head's q rows, not the flat midpoint.
        let rank1 = full_attention_gated_q_shard_range(&config, test_geometry(1, 2));
        assert_eq!(rank1, (4096, 4096));
    }
}
