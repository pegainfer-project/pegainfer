//! Typed text-tower config, read only from a file the probe has accepted.

#[cfg(any(feature = "gemma4", test))]
use anyhow::Result;
#[cfg(any(feature = "gemma4", test))]
use anyhow::bail;
#[cfg(any(feature = "gemma4", test))]
use anyhow::ensure;

// Only the loader reads a config off disk.
#[cfg(feature = "gemma4")]
use crate::probe::probe_config_json;

/// Selects the head dim, the KV head count, and whether the layer has a
/// `v_proj`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerKind {
    Sliding,
    Global,
}

/// What the manifest is derived from. Only [`Gemma4Config::from_file`] is
/// probe-backed; a value built directly is not, so consumers check what they
/// depend on.
// The serving path reads every field; a featureless build compiles the config
// with that consumer cfg'd out.
#[cfg_attr(not(feature = "gemma4"), allow(dead_code))]
#[derive(Clone, Debug)]
pub(crate) struct Gemma4Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    /// 1, 2 or 4 by size, and not derivable from `num_key_value_heads`.
    pub(crate) num_global_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) global_head_dim: usize,
    pub(crate) layer_types: Vec<LayerKind>,
    pub(crate) tie_word_embeddings: bool,
    /// Present on the size that routes: it keeps its dense MLP and adds
    /// experts alongside it. The dimensions travel with the flag so that
    /// "routing, but we do not know how wide" is not a state.
    pub(crate) moe: Option<MoeConfig>,
    /// Every text-tower linear ships as compressed-tensors W4A16: symmetric
    /// four-bit integers, one bf16 scale per 32 inputs. The embedding, and
    /// the LM head tied to it, stay bf16.
    pub(crate) w4a16: bool,
    pub(crate) rms_norm_eps: f32,
    /// The sliding-attention rope theta; the global family reads its own.
    pub(crate) sliding_rope_theta: f32,
    pub(crate) sliding_window: usize,
    pub(crate) global_rope_theta: f32,
    /// `partial_rotary_factor * global_head_dim`, validated to land on a
    /// positive even width within the head — the active band of the
    /// proportional rope tables.
    pub(crate) global_rotary_dim: usize,
    /// Applied as `cap * tanh(x / cap)` over the final logits; every
    /// published size declares one.
    pub(crate) final_logit_softcapping: f32,
    /// The checkpoint's own position limit — the ceiling a raised serving
    /// context may not pass.
    pub(crate) max_position_embeddings: usize,
}

/// The routed half of a MoE layer. The dense MLP beside it keeps using
/// [`Gemma4Config::intermediate_size`]; experts have their own width.
#[cfg_attr(not(feature = "gemma4"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeConfig {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) intermediate_size: usize,
}

#[cfg(feature = "gemma4")]
impl MoeConfig {
    fn from_text_config(tc: &serde_json::Value) -> Result<Option<Self>> {
        if !bool_field(tc, "enable_moe_block")? {
            return Ok(None);
        }
        Ok(Some(Self {
            num_experts: usize_field(tc, "num_experts")?,
            top_k: usize_field(tc, "top_k_experts")?,
            intermediate_size: usize_field(tc, "moe_intermediate_size")?,
        }))
    }
}

/// Which rank of a tensor-parallel group this process serves, and how wide the
/// group is. The default is the whole model on one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

/// The head width a layer kind's queries and keys use; the two families differ
/// only here and in their KV head counts.
pub(crate) fn head_dim_of(config: &Gemma4Config, kind: LayerKind) -> usize {
    match kind {
        LayerKind::Sliding => config.head_dim,
        LayerKind::Global => config.global_head_dim,
    }
}

impl TensorParallelConfig {
    /// The single-rank configuration the tests build geometries at; serving
    /// takes it from [`Default`].
    #[cfg(test)]
    pub(crate) const SINGLE: Self = Self {
        rank: 0,
        world_size: 1,
    };

    pub(crate) fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub(crate) fn is_single(&self) -> bool {
        self.world_size == 1
    }

    pub(crate) fn local_q_heads(&self, config: &Gemma4Config) -> Result<usize> {
        let q = config.num_attention_heads;
        ensure!(q > 0, "query head count must be positive");
        ensure!(
            q.is_multiple_of(self.world_size),
            "{q} query heads do not divide over world size {}",
            self.world_size
        );
        Ok(q / self.world_size)
    }

    pub(crate) fn local_sliding_kv_heads(&self, config: &Gemma4Config) -> Result<usize> {
        let kv = config.num_key_value_heads;
        ensure!(kv > 0, "sliding KV head count must be positive");
        ensure!(
            kv.is_multiple_of(self.world_size),
            "{kv} sliding KV heads do not divide over world size {}",
            self.world_size
        );
        Ok(kv / self.world_size)
    }

    /// The global family's KV head range for this rank as `(head_offset,
    /// head_count)`. That family has few enough KV heads that a world size can
    /// exceed them: `G % P == 0` shards contiguous runs of `G / P` heads, and
    /// `P % G == 0` replicates each head onto `P / G` contiguous ranks, one
    /// head apiece. Replicating KV is exact under a sum reduction only because
    /// query rows and `o_proj` columns still split `1 / P` — this branch never
    /// touches them.
    pub(crate) fn global_kv_head_range(&self, config: &Gemma4Config) -> Result<(usize, usize)> {
        let g = config.num_global_key_value_heads;
        let p = self.world_size;
        ensure!(
            g > 0 && p > 0,
            "global KV heads ({g}) and world size ({p}) must be positive"
        );
        if g.is_multiple_of(p) {
            Ok((self.rank * (g / p), g / p))
        } else if p.is_multiple_of(g) {
            Ok((self.rank / (p / g), 1))
        } else {
            bail!(
                "world size {p} is legal neither for sharding nor for replicating {g} global KV \
                 heads"
            )
        }
    }

    pub(crate) fn local_q_dim(&self, config: &Gemma4Config, kind: LayerKind) -> Result<usize> {
        Ok(self.local_q_heads(config)? * head_dim_of(config, kind))
    }

    /// The fused K (and, on sliding layers, V) width this rank holds. Global
    /// layers carry no `v_proj`, so the sliding value covers both their K and
    /// V rows.
    pub(crate) fn local_kv_dim(&self, config: &Gemma4Config, kind: LayerKind) -> Result<usize> {
        Ok(match kind {
            LayerKind::Sliding => self.local_sliding_kv_heads(config)? * config.head_dim,
            LayerKind::Global => self.global_kv_head_range(config)?.1 * config.global_head_dim,
        })
    }

    pub(crate) fn local_intermediate(&self, config: &Gemma4Config) -> Result<usize> {
        let i = config.intermediate_size;
        ensure!(i > 0, "intermediate size must be positive");
        ensure!(
            i.is_multiple_of(self.world_size),
            "intermediate size {i} does not divide over world size {}",
            self.world_size
        );
        Ok(i / self.world_size)
    }

    /// Everything a tensor-parallel launch must satisfy before a multi-GiB
    /// load. The three-count rule the design doc states (`Q % P`, `Kv % P`,
    /// `G % P || P % G`) is necessary but not sufficient: it does not keep the
    /// per-rank GQA group integral (`Q = 8, Kv = 6, P = 2` clears it and yields
    /// group `4/3`), so that is checked here too.
    pub(crate) fn validate_for(&self, config: &Gemma4Config) -> Result<()> {
        // A single rank is the incumbent path: it has no shard to police, and
        // the MoE/W4A16 refusals below are about sharding those families, not
        // about those checkpoints.
        if self.is_single() {
            return Ok(());
        }
        let p = self.world_size;
        ensure!(p > 0, "tensor-parallel world size must be positive");
        ensure!(
            self.rank < p,
            "tensor-parallel rank {} is outside world size {p}",
            self.rank
        );
        ensure!(
            config.moe.is_none(),
            "gemma4 tensor parallelism does not shard the routed experts; this checkpoint routes"
        );
        ensure!(
            !config.w4a16,
            "gemma4 tensor parallelism does not shard the W4A16 GEMMs, which project whole matrices"
        );
        let local_q = self.local_q_heads(config)?;
        let local_sliding_kv = self.local_sliding_kv_heads(config)?;
        ensure!(
            local_q.is_multiple_of(local_sliding_kv),
            "per-rank sliding GQA group {local_q}/{local_sliding_kv} is not integral at world \
             size {p}"
        );
        // Resolving the range also refuses a world size that can neither shard
        // nor replicate the global heads.
        let (_, local_global_kv) = self.global_kv_head_range(config)?;
        // Only the sharding branch can split a GQA group: it hands a rank
        // `g / p` KV heads, so the per-rank group `(q / p) / (g / p)` has to be
        // integral. The replicate branch gives a rank one whole KV head, and
        // `p % g == 0` already keeps its `q / p` query heads inside that one
        // head's group (`q / p` is `p / g` copies of `q / g`), so there is
        // nothing left to police there — the check below would only be comparing
        // against a group of one.
        if config.num_global_key_value_heads.is_multiple_of(p) {
            ensure!(
                local_q.is_multiple_of(local_global_kv),
                "per-rank global GQA group {local_q}/{local_global_kv} is not integral at world \
                 size {p}"
            );
        }
        self.local_intermediate(config)?;
        Ok(())
    }
}

#[cfg(feature = "gemma4")]
impl Gemma4Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let path = format!("{model_path}/config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("Gemma 4: cannot read {path}: {e}"))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("Gemma 4: {path} is not valid JSON: {e}"))?;
        probe_config_json(&json)?;
        Self::from_json(&json)
    }

    fn from_json(json: &serde_json::Value) -> Result<Self> {
        let tc = json
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config"))?;
        let layer_types = tc
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config.layer_types"))?
            .iter()
            .map(|entry| match entry.as_str() {
                Some("sliding_attention") => Ok(LayerKind::Sliding),
                Some("full_attention") => Ok(LayerKind::Global),
                other => bail!("Gemma 4: unexpected layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let num_hidden_layers = usize_field(tc, "num_hidden_layers")?;
        anyhow::ensure!(
            layer_types.len() == num_hidden_layers,
            "Gemma 4: layer_types has {} entries but num_hidden_layers is {num_hidden_layers}",
            layer_types.len()
        );
        let rope = tc
            .get("rope_parameters")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing text_config.rope_parameters"))?;
        let sliding_rope = rope
            .get("sliding_attention")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing rope_parameters.sliding_attention"))?;
        let global_rope = rope
            .get("full_attention")
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: missing rope_parameters.full_attention"))?;
        rope_type_field(sliding_rope, "sliding_attention", "default")?;
        rope_type_field(global_rope, "full_attention", "proportional")?;
        let sliding_rope_theta = f32_field(sliding_rope, "sliding_attention", "rope_theta")?;
        let global_rope_theta = f32_field(global_rope, "full_attention", "rope_theta")?;
        anyhow::ensure!(
            sliding_rope_theta > 0.0 && global_rope_theta > 0.0,
            "Gemma 4: rope_theta must be positive (sliding {sliding_rope_theta}, global \
             {global_rope_theta})"
        );
        let global_head_dim = usize_field(tc, "global_head_dim")?;
        let partial = global_rope
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Gemma 4: full_attention.partial_rotary_factor missing or not a number"
                )
            })?;
        let rotary = partial * global_head_dim as f64;
        anyhow::ensure!(
            rotary > 0.0
                && rotary.fract() == 0.0
                && rotary as usize <= global_head_dim
                && (rotary as usize).is_multiple_of(2),
            "Gemma 4: partial_rotary_factor {partial} of global_head_dim {global_head_dim} must \
             land on a positive even width within the head, got {rotary}"
        );
        let sliding_window = usize_field(tc, "sliding_window")?;
        anyhow::ensure!(
            sliding_window > 0,
            "Gemma 4: sliding_window must be positive"
        );
        let max_position_embeddings = usize_field(tc, "max_position_embeddings")?;
        anyhow::ensure!(
            max_position_embeddings >= sliding_window,
            "Gemma 4: max_position_embeddings {max_position_embeddings} sits below the \
             sliding_window {sliding_window}"
        );
        let moe = MoeConfig::from_text_config(tc)?;
        let w4a16 = w4a16_from_json(json)?;
        anyhow::ensure!(
            !(w4a16 && moe.is_some()),
            "Gemma 4: a W4A16 checkpoint that also routes is not one this loader has a layout for"
        );
        let final_logit_softcapping = f32_field(tc, "text_config", "final_logit_softcapping")?;
        anyhow::ensure!(
            final_logit_softcapping > 0.0,
            "Gemma 4: final_logit_softcapping {final_logit_softcapping} must be positive"
        );
        Ok(Self {
            hidden_size: usize_field(tc, "hidden_size")?,
            intermediate_size: usize_field(tc, "intermediate_size")?,
            vocab_size: usize_field(tc, "vocab_size")?,
            num_attention_heads: usize_field(tc, "num_attention_heads")?,
            num_key_value_heads: usize_field(tc, "num_key_value_heads")?,
            num_global_key_value_heads: usize_field(tc, "num_global_key_value_heads")?,
            head_dim: usize_field(tc, "head_dim")?,
            global_head_dim,
            layer_types,
            tie_word_embeddings: bool_field(tc, "tie_word_embeddings")?,
            moe,
            w4a16,
            rms_norm_eps: f32_field(tc, "text_config", "rms_norm_eps")?,
            sliding_rope_theta,
            sliding_window,
            max_position_embeddings,
            global_rope_theta,
            global_rotary_dim: rotary as usize,
            final_logit_softcapping,
        })
    }
}

/// Whether the checkpoint is compressed-tensors W4A16 over the text tower.
/// Any other compressed-tensors scheme is refused rather than read as bf16;
/// other quantization methods (the routed size's ModelOpt NVFP4) are the
/// manifest's to check.
#[cfg(feature = "gemma4")]
fn w4a16_from_json(json: &serde_json::Value) -> Result<bool> {
    use serde_json::Value;
    let Some(quant) = json.get("quantization_config") else {
        return Ok(false);
    };
    if quant.get("quant_method").and_then(Value::as_str) != Some("compressed-tensors") {
        return Ok(false);
    }
    let refuse = |what: &str| {
        anyhow::anyhow!(
            "Gemma 4: compressed-tensors {what}; only W4A16 \
         (pack-quantized symmetric int4, group 32, over every Linear) is served"
        )
    };
    if quant.get("format").and_then(Value::as_str) != Some("pack-quantized") {
        return Err(refuse("format is not pack-quantized"));
    }
    let groups = quant
        .get("config_groups")
        .and_then(Value::as_object)
        .filter(|groups| groups.len() == 1)
        .ok_or_else(|| refuse("config_groups is not a single group"))?;
    let group = groups.values().next().expect("one group");
    if group.get("targets") != Some(&serde_json::json!(["Linear"])) {
        return Err(refuse("targets are not [\"Linear\"]"));
    }
    for activations in ["input_activations", "output_activations"] {
        if !group.get(activations).is_none_or(Value::is_null) {
            return Err(refuse(&format!("quantizes {activations}")));
        }
    }
    let weights = group
        .get("weights")
        .ok_or_else(|| refuse("group carries no weights scheme"))?;
    let expected = [
        ("type", serde_json::json!("int")),
        ("num_bits", serde_json::json!(4)),
        ("group_size", serde_json::json!(32)),
        ("strategy", serde_json::json!("group")),
        ("symmetric", serde_json::json!(true)),
        ("dynamic", serde_json::json!(false)),
    ];
    for (field, value) in expected {
        if weights.get(field) != Some(&value) {
            return Err(refuse(&format!("weights.{field} is not {value}")));
        }
    }
    if !weights.get("actorder").is_none_or(Value::is_null) {
        return Err(refuse("weights reorder their groups (actorder)"));
    }
    let ignore = quant
        .get("ignore")
        .and_then(Value::as_array)
        .ok_or_else(|| refuse("ignore is not a list"))?;
    for entry in ignore {
        let name = entry
            .as_str()
            .ok_or_else(|| refuse("ignore holds a non-string"))?;
        if name.starts_with("re:") || name.starts_with("model.language_model.") {
            return Err(refuse(&format!("ignore keeps {name:?} unquantized")));
        }
    }
    Ok(true)
}

/// Numeric config values land in f32 compute; the checked cast rejects
/// anything the narrowing would turn infinite rather than rounding it in
/// silently.
#[cfg(feature = "gemma4")]
fn f32_field(obj: &serde_json::Value, ctx: &str, field: &str) -> Result<f32> {
    let value = obj
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: {ctx}.{field} missing or not a number"))?;
    let narrowed = value as f32;
    anyhow::ensure!(
        narrowed.is_finite(),
        "Gemma 4: {ctx}.{field} = {value} overflows f32"
    );
    Ok(narrowed)
}

#[cfg(feature = "gemma4")]
fn usize_field(text_config: &serde_json::Value, field: &str) -> Result<usize> {
    let value = text_config
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            anyhow::anyhow!("Gemma 4: text_config.{field} missing or not a non-negative integer")
        })?;
    usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("Gemma 4: text_config.{field} = {value} does not fit usize"))
}

#[cfg(feature = "gemma4")]
fn bool_field(text_config: &serde_json::Value, field: &str) -> Result<bool> {
    text_config
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: text_config.{field} missing or not a boolean"))
}

/// `rope_type` selects the table-generation algorithm; a value this engine
/// has not wired for that family must fail here, not silently get the other
/// family's tables.
#[cfg(feature = "gemma4")]
fn rope_type_field(rope_group: &serde_json::Value, ctx: &str, implemented: &str) -> Result<()> {
    let value = rope_group
        .get("rope_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: {ctx}.rope_type missing or not a string"))?;
    anyhow::ensure!(
        value == implemented,
        "Gemma 4: {ctx}.rope_type {value:?} is not the implemented {implemented:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A text geometry with the four counts a shard decision reads varied;
    /// the rest is fixed filler. Built directly, so it is not probe-backed —
    /// only the shard arithmetic is under test here.
    fn config(q: usize, kv: usize, g: usize, intermediate: usize) -> Gemma4Config {
        Gemma4Config {
            hidden_size: 2560,
            intermediate_size: intermediate,
            vocab_size: 262_144,
            num_attention_heads: q,
            num_key_value_heads: kv,
            num_global_key_value_heads: g,
            head_dim: 256,
            global_head_dim: 512,
            layer_types: vec![LayerKind::Sliding; 12],
            tie_word_embeddings: true,
            moe: None,
            w4a16: false,
            rms_norm_eps: 1e-6,
            sliding_rope_theta: 10_000.0,
            sliding_window: 1024,
            global_rope_theta: 1_000_000.0,
            global_rotary_dim: 256,
            final_logit_softcapping: 30.0,
            max_position_embeddings: 262_144,
        }
    }

    /// 12B-like: 16 query heads, 8 sliding KV heads, the single global KV head
    /// the design doc's table names.
    fn config_12b() -> Gemma4Config {
        config(16, 8, 1, 15_360)
    }

    /// 31B-like: 32 query heads, 16 sliding KV heads, 4 global KV heads.
    fn config_31b() -> Gemma4Config {
        config(32, 16, 4, 16_384)
    }

    #[test]
    fn single_is_the_identity() {
        let tp = TensorParallelConfig::SINGLE;
        let cfg = config_31b();
        assert!(tp.is_single());
        assert_eq!(tp.local_q_heads(&cfg).unwrap(), 32);
        assert_eq!(tp.local_sliding_kv_heads(&cfg).unwrap(), 16);
        assert_eq!(tp.global_kv_head_range(&cfg).unwrap(), (0, 4));
        assert_eq!(tp.local_q_dim(&cfg, LayerKind::Sliding).unwrap(), 32 * 256);
        assert_eq!(tp.local_q_dim(&cfg, LayerKind::Global).unwrap(), 32 * 512);
        assert_eq!(tp.local_kv_dim(&cfg, LayerKind::Global).unwrap(), 4 * 512);
        assert_eq!(tp.local_intermediate(&cfg).unwrap(), 16_384);
        tp.validate_for(&cfg).unwrap();
    }

    #[test]
    fn twelve_b_replicates_its_single_global_kv_head() {
        let cfg = config_12b();
        for rank in 0..2 {
            let tp = TensorParallelConfig::new(rank, 2);
            assert_eq!(tp.local_q_heads(&cfg).unwrap(), 8);
            assert_eq!(tp.local_sliding_kv_heads(&cfg).unwrap(), 4);
            // G = 1 < P = 2: the one head lives on both ranks, at offset 0.
            assert_eq!(tp.global_kv_head_range(&cfg).unwrap(), (0, 1));
            tp.validate_for(&cfg).unwrap();
        }
    }

    #[test]
    fn thirty_one_b_shards_two_global_kv_heads_per_rank() {
        let cfg = config_31b();
        assert_eq!(
            TensorParallelConfig::new(0, 2)
                .global_kv_head_range(&cfg)
                .unwrap(),
            (0, 2)
        );
        assert_eq!(
            TensorParallelConfig::new(1, 2)
                .global_kv_head_range(&cfg)
                .unwrap(),
            (2, 2)
        );
    }

    #[test]
    fn local_dims_shrink_by_the_world_size() {
        let cfg = config_31b();
        let tp = TensorParallelConfig::new(1, 2);
        assert_eq!(tp.local_q_dim(&cfg, LayerKind::Sliding).unwrap(), 16 * 256);
        assert_eq!(tp.local_q_dim(&cfg, LayerKind::Global).unwrap(), 16 * 512);
        assert_eq!(tp.local_kv_dim(&cfg, LayerKind::Sliding).unwrap(), 8 * 256);
        assert_eq!(tp.local_kv_dim(&cfg, LayerKind::Global).unwrap(), 2 * 512);
        assert_eq!(tp.local_intermediate(&cfg).unwrap(), 8192);
    }

    #[test]
    fn world_size_zero_is_rejected() {
        let err = TensorParallelConfig::new(0, 0)
            .validate_for(&config_31b())
            .unwrap_err()
            .to_string();
        assert!(err.contains("positive"), "{err}");
    }

    #[test]
    fn rank_outside_the_world_is_rejected() {
        let err = TensorParallelConfig::new(2, 2)
            .validate_for(&config_31b())
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank"), "{err}");
    }

    #[test]
    fn query_heads_that_do_not_divide_are_rejected() {
        // 12B has 16 query heads, which do not divide over world size 3.
        let err = TensorParallelConfig::new(0, 3)
            .validate_for(&config_12b())
            .unwrap_err()
            .to_string();
        assert!(err.contains("query heads"), "{err}");
    }

    #[test]
    fn per_rank_gqa_group_must_be_integral() {
        // 8 query heads over 6 sliding KV heads at world size 2 clears
        // `Q % P` and `Kv % P` yet leaves the per-rank group 4/3.
        let err = TensorParallelConfig::new(0, 2)
            .validate_for(&config(8, 6, 2, 15_360))
            .unwrap_err()
            .to_string();
        assert!(err.contains("GQA group"), "{err}");
    }

    #[test]
    fn global_heads_that_neither_shard_nor_replicate_are_rejected() {
        // 3 global KV heads over world size 2: neither 3 % 2 nor 2 % 3 is 0.
        let err = TensorParallelConfig::new(0, 2)
            .validate_for(&config(8, 4, 3, 15_360))
            .unwrap_err()
            .to_string();
        assert!(err.contains("global KV heads"), "{err}");
    }

    #[test]
    fn intermediate_size_must_divide() {
        let err = TensorParallelConfig::new(0, 2)
            .validate_for(&config(16, 8, 1, 15_361))
            .unwrap_err()
            .to_string();
        assert!(err.contains("intermediate"), "{err}");
    }

    #[test]
    fn routed_and_w4a16_checkpoints_are_refused() {
        let mut cfg = config_31b();
        cfg.moe = Some(MoeConfig {
            num_experts: 128,
            top_k: 8,
            intermediate_size: 704,
        });
        let err = TensorParallelConfig::new(0, 2)
            .validate_for(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("routed experts"), "{err}");

        let mut cfg = config_31b();
        cfg.w4a16 = true;
        let err = TensorParallelConfig::new(0, 2)
            .validate_for(&cfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("W4A16"), "{err}");
    }
}
