//! Fail-closed config probe: identity plus the structural facts this line
//! serves.
//!
//! Following `pegainfer-gemma4/src/probe.rs`, this pins **structure rather than
//! widths**. The head dimensions and the conv-kernel ceiling are pinned because
//! the compiled kernels and the Triton AOT specialization keys bake them in; the
//! expert count, hidden size, layer count and branch count are *not* pinned, so
//! a smaller `qwen4_exp` checkpoint reuses this probe instead of needing a new
//! one. Everything pinned to one exact value below is a value whose alternative
//! would need a kernel or a serving path this line does not have.

use anyhow::Result;
use anyhow::bail;

/// The GDN head dimensions the Triton chunkwise AOT keys are specialized on.
/// Head *counts* are runtime arguments to the kernels and stay unpinned.
const GDN_KEY_HEAD_DIM: u64 = 128;
const GDN_VALUE_HEAD_DIM: u64 = 128;

/// The in-tree depthwise conv1d keeps at most four taps of state.
const CONV_KERNEL_MAX: u64 = 4;

/// The full-attention head dimension the shared HD256 kernels are built for.
const FULL_ATTN_HEAD_DIM: u64 = 256;

/// The identity this line claims: the outer `model_type` and the inner
/// `text_config.model_type` the probe requires together.
///
/// Exposed because `pegainfer-server`'s `feature_gate_hint` hand-mirrors each
/// line's identities in a table that must compile when this crate does not; a
/// test under this feature binds the mirror to these constants so the two cannot
/// drift apart silently.
pub const MODEL_TYPE: &str = "qwen4_exp";
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";

pub(crate) fn probe_config_json(json: &serde_json::Value) -> Result<()> {
    let text_config = probe_identity(json)?;
    probe_precision(text_config)?;
    probe_widths(text_config)?;
    probe_layer_types(text_config)?;
    probe_attention(text_config)?;
    probe_moe(text_config)?;
    probe_hyper_connections(text_config)?;
    probe_ple(text_config)?;
    probe_rope(text_config)
}

fn probe_identity(json: &serde_json::Value) -> Result<&serde_json::Value> {
    let model_type = json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if model_type != MODEL_TYPE {
        bail!("not a Qwen3.8-Flash-Next config: model_type={model_type}");
    }

    let has_arch = json
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|arr| {
            arr.iter().any(|v| {
                v.as_str()
                    .is_some_and(|s| s == "Qwen4ExpForConditionalGeneration")
            })
        });
    if !has_arch {
        bail!("qwen4_exp: architectures must contain Qwen4ExpForConditionalGeneration");
    }

    // `tie_word_embeddings` is read from both levels: the loader must materialize
    // an embedding *and* an LM head, and a tied variant would leave one of them
    // missing from the checkpoint rather than wrong-shaped.
    for (scope, node) in [
        ("outer", json),
        ("text_config", json.get("text_config").unwrap_or(json)),
    ] {
        if node
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        {
            bail!(
                "qwen4_exp: {scope}.tie_word_embeddings must be false; this line serves the untied checkpoint"
            );
        }
    }

    let text_config = json
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing text_config"))?;
    let text_type = text_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if text_type != TEXT_MODEL_TYPE {
        bail!(
            "qwen4_exp: text_config.model_type is {text_type}, expected {TEXT_MODEL_TYPE} — \
             cross-family mismatch"
        );
    }
    Ok(text_config)
}

/// Precision and the two scalar dtypes the kernels assume. An FP8 sibling of this
/// checkpoint exists upstream; it is a different revision with a different
/// storage contract, so it must fail closed here rather than load as BF16.
fn probe_precision(text_config: &serde_json::Value) -> Result<()> {
    let dtype = text_config.get("dtype").and_then(serde_json::Value::as_str);
    if dtype != Some("bfloat16") {
        bail!(
            "qwen4_exp: text_config.dtype must be bfloat16, got {dtype:?} — the FP8 checkpoint is a separate revision and is not served by this line"
        );
    }
    // The GDN recurrent state is f32 in the in-tree kernels.
    let ssm = text_config
        .get("mamba_ssm_dtype")
        .and_then(serde_json::Value::as_str);
    if ssm != Some("float32") {
        bail!("qwen4_exp: mamba_ssm_dtype must be float32, got {ssm:?}");
    }
    Ok(())
}

/// Widths a smaller `qwen4_exp` checkpoint may change but that cannot be zero:
/// the stack depth, the residual width (`hc_count` times over) and the
/// vocabulary the two bookends read.
fn probe_widths(text_config: &serde_json::Value) -> Result<()> {
    for field in ["num_hidden_layers", "hidden_size", "vocab_size"] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }
    Ok(())
}

fn probe_layer_types(text_config: &serde_json::Value) -> Result<()> {
    let interval = text_config
        .get("full_attention_interval")
        .and_then(serde_json::Value::as_u64)
        .filter(|v| *v > 0);
    let Some(interval) = interval else {
        bail!("qwen4_exp: full_attention_interval must be a positive integer");
    };
    let layer_types = text_config
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing layer_types"))?;
    if layer_types.is_empty() {
        bail!("qwen4_exp: layer_types is empty");
    }
    if layer_types.len() % interval as usize != 0 {
        bail!(
            "qwen4_exp: layer_types has {} entries, not a multiple of full_attention_interval {interval}",
            layer_types.len()
        );
    }
    // `num_hidden_layers` is what the geometry and (later) the per-layer tensor
    // names count by, while `layer_types` is what says which kinds they are. A
    // config whose two stacks disagree is unservable, and the disagreement would
    // otherwise surface much later as a shape mismatch.
    let declared = text_config
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64);
    if declared != Some(layer_types.len() as u64) {
        bail!(
            "qwen4_exp: num_hidden_layers is {declared:?} but layer_types has {} entries",
            layer_types.len()
        );
    }
    for (i, entry) in layer_types.iter().enumerate() {
        let s = entry.as_str().unwrap_or("");
        if s != "linear_attention" && s != "full_attention" {
            bail!(
                "qwen4_exp: layer_types[{i}] is {s:?}, must be linear_attention or full_attention"
            );
        }
        // Every `interval`-th layer is full attention; the rest are Gated DeltaNet.
        let expected_full = (i + 1) % interval as usize == 0;
        if (s == "full_attention") != expected_full {
            bail!(
                "qwen4_exp: layer_types[{i}] is {s:?}, expected {}",
                if expected_full {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            );
        }
    }
    Ok(())
}

fn probe_attention(text_config: &serde_json::Value) -> Result<()> {
    let head_dim = text_config
        .get("head_dim")
        .and_then(serde_json::Value::as_u64);
    if head_dim != Some(FULL_ATTN_HEAD_DIM) {
        bail!("qwen4_exp: head_dim must be {FULL_ATTN_HEAD_DIM}, got {head_dim:?}");
    }
    for field in ["num_attention_heads", "num_key_value_heads"] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }

    // The GDN head dims are baked into the Triton AOT specialization keys, so a
    // different value is a new kernel, not a config change.
    for (field, want) in [
        ("linear_key_head_dim", GDN_KEY_HEAD_DIM),
        ("linear_value_head_dim", GDN_VALUE_HEAD_DIM),
    ] {
        let got = text_config.get(field).and_then(serde_json::Value::as_u64);
        if got != Some(want) {
            bail!(
                "qwen4_exp: {field} must be {want} (the Triton AOT keys are specialized on it), got {got:?}"
            );
        }
    }
    for field in ["linear_num_key_heads", "linear_num_value_heads"] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }

    let conv = text_config
        .get("linear_conv_kernel_dim")
        .and_then(serde_json::Value::as_u64);
    match conv {
        Some(k) if (1..=CONV_KERNEL_MAX).contains(&k) => {}
        other => bail!(
            "qwen4_exp: linear_conv_kernel_dim must be in 1..={CONV_KERNEL_MAX} (the in-tree conv1d caps its state), got {other:?}"
        ),
    }

    // The GDN gated-norm activation. This is a *live* field on this checkpoint:
    // upstream passes it straight to the gated RMSNorm. #1129 made the shared
    // kernel selectable (`rms_norm_gated_sigmoid_cuda` beside Qwen3.5's SiLU),
    // so both activations are compilable; what this line lacks until its text
    // graph lands is the call site. The pin stays fail-closed: the frozen
    // checkpoint is sigmoid, and no silu-gated `qwen4_exp` variant has golden
    // fixtures on this line.
    let gate = text_config
        .get("output_gate_type")
        .and_then(serde_json::Value::as_str);
    if gate != Some("sigmoid") {
        bail!("qwen4_exp: output_gate_type must be sigmoid, got {gate:?}");
    }

    // The indexer is a second, differently-shaped attention. The reference's
    // own `validate_architecture` (configuration_qwen4_exp.py:213-231) requires
    // all five QSA fields positive, `indexer_kv_heads == 1`, and the budget to
    // be a whole multiple of the compress ratio; a config the reference itself
    // refuses cannot be served by any port of it.
    for field in [
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    ] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }
    let kv_heads = text_config
        .get("indexer_kv_heads")
        .and_then(serde_json::Value::as_u64);
    if kv_heads != Some(1) {
        bail!(
            "qwen4_exp: indexer_kv_heads must be 1 (the reference QSA requires it), got {kv_heads:?}"
        );
    }
    let budget = text_config
        .get("indexer_budget")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let ratio = text_config
        .get("indexer_compress_ratio")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if !budget.is_multiple_of(ratio) {
        bail!(
            "qwen4_exp: indexer_budget {budget} must be divisible by indexer_compress_ratio {ratio}"
        );
    }
    Ok(())
}

/// Every layer is MoE on this architecture, so there is no dense variant to
/// defer behind. The widths are deliberately unpinned: a smaller `qwen4_exp`
/// checkpoint should reuse this probe.
fn probe_moe(text_config: &serde_json::Value) -> Result<()> {
    let experts = text_config
        .get("num_experts")
        .and_then(serde_json::Value::as_u64);
    let top_k = text_config
        .get("num_experts_per_tok")
        .and_then(serde_json::Value::as_u64);
    match (experts, top_k) {
        (Some(e), Some(k)) if e > 0 && k > 0 && k <= e => {}
        other => bail!(
            "qwen4_exp: num_experts/num_experts_per_tok must be positive with top_k <= num_experts, got {other:?}"
        ),
    }
    for field in ["moe_intermediate_size", "shared_expert_intermediate_size"] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }
    Ok(())
}

/// The residual stream is `hc_count` copies of hidden, so the branch count is a
/// width — but the reference's own `validate_architecture` refuses `hc_count <= 1`
/// (configuration_qwen4_exp.py:196): with a single branch there is nothing to
/// mix and the mixer tensors degenerate. A config the reference refuses cannot
/// be served by any port of it.
fn probe_hyper_connections(text_config: &serde_json::Value) -> Result<()> {
    let hc_count = text_config
        .get("hc_count")
        .and_then(serde_json::Value::as_u64);
    match hc_count {
        Some(count) if count > 1 => {}
        other => bail!(
            "qwen4_exp: hc_count must be an integer > 1 (a single-branch residual stream has nothing to mix), got {other:?}"
        ),
    }
    let lowrank = text_config
        .get("hc_lowrank")
        .and_then(serde_json::Value::as_u64);
    if lowrank.is_none_or(|v| v == 0) {
        bail!("qwen4_exp: hc_lowrank must be a positive integer, got {lowrank:?}");
    }
    Ok(())
}

/// The n-gram (PLE) block. `ple_layer_ids` is **1-indexed**: the frozen
/// checkpoint declares `[2]` and carries its tensors at `layers.1`. Reading it as
/// a 0-indexed subscript injects into the wrong layer and still produces fluent,
/// wrong text, so the conversion lives in one place and is pinned by a test.
fn probe_ple(text_config: &serde_json::Value) -> Result<()> {
    let ids = text_config
        .get("ple_layer_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing ple_layer_ids"))?;
    let layers = text_config
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing num_hidden_layers"))?;
    if ids.is_empty() {
        bail!("qwen4_exp: ple_layer_ids is empty");
    }
    // This crate derives one n-gram table, and the reference primes head `i` of
    // PLE layer `L` at the global index `L * ngram_heads + i`
    // (modeling_qwen4_exp.py:1037-1039): a second PLE layer's table is a
    // different prime run, not a copy of the first. Until per-layer tables are
    // modelled, refuse rather than derive the layer-0 geometry for all of them.
    // (The reference also dedups and sorts the list, so even `[2, 2]` is one
    // layer upstream — refusing it is the fail-closed reading.)
    if ids.len() != 1 {
        bail!(
            "qwen4_exp: exactly one ple_layer_ids entry is modelled (each PLE layer needs its own n-gram table, its prime run offset by the layer index), got {ids:?}"
        );
    }
    let layer_types = text_config
        .get("layer_types")
        .and_then(serde_json::Value::as_array);
    for entry in ids {
        let id = entry.as_u64().unwrap_or(0);
        if id < 1 || id > layers {
            bail!(
                "qwen4_exp: ple_layer_ids entry {id} is outside 1..={layers} (the field is 1-indexed)"
            );
        }
        // The reference supports PLE only on linear_attention layers
        // (configuration_qwen4_exp.py:248-254). `layer_types` is guaranteed
        // present and `num_hidden_layers`-long by `probe_layer_types`, which
        // runs first, so the subscript is in bounds for a range-valid id.
        let kind = layer_types
            .and_then(|types| types.get(id as usize - 1))
            .and_then(serde_json::Value::as_str);
        if kind != Some("linear_attention") {
            bail!(
                "qwen4_exp: ple_layer_ids entry {id} sits on a {kind:?} layer, but PLE is only supported on linear_attention layers"
            );
        }
    }
    for field in [
        "ngram_size",
        "heads_per_ngram",
        "ple_embed_dim",
        "ngram_vocab_size_base",
        "split_ngram_parts",
        "make_ngram_vocab_size_divisible_by",
    ] {
        let v = text_config.get(field).and_then(serde_json::Value::as_u64);
        if v.is_none_or(|v| v == 0) {
            bail!("qwen4_exp: {field} must be a positive integer, got {v:?}");
        }
    }
    let ngram_size = text_config
        .get("ngram_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if ngram_size < 2 {
        bail!(
            "qwen4_exp: ngram_size must be at least 2 (an n-gram needs a context token), got {ngram_size}"
        );
    }
    // Head `i` gets the (i+1)-th prime after `base - 1`, so a base of 1 asks for
    // primes after zero — a window where the search's own trial division accepts
    // 1, and the table's first vocabulary would be wrong rather than refused.
    let base = text_config
        .get("ngram_vocab_size_base")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if base < 2 {
        bail!(
            "qwen4_exp: ngram_vocab_size_base must be at least 2 (primes are searched after base - 1), got {base}"
        );
    }
    let ple_conv = text_config
        .get("ple_conv_kernel_size")
        .and_then(serde_json::Value::as_u64);
    match ple_conv {
        Some(k) if k >= 1 => {}
        other => bail!("qwen4_exp: ple_conv_kernel_size must be a positive integer, got {other:?}"),
    }
    // The PLE short conv is dilated by `ngram_size`, so its state length is
    // (kernel - 1) * ngram_size — wider than the GDN conv's even though the
    // weight tensor has the same rank. Nothing here caps it, but it must be
    // derivable.
    Ok(())
}

fn probe_rope(text_config: &serde_json::Value) -> Result<()> {
    let rope = text_config
        .get("rope_parameters")
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp: missing rope_parameters"))?;
    let rope_type = rope.get("rope_type").and_then(serde_json::Value::as_str);
    // 262,144 is native under default RoPE: this config carries no YaRN fields,
    // and a YaRN variant would need a scaling path this line does not have.
    if rope_type != Some("default") {
        bail!(
            "qwen4_exp: rope_parameters.rope_type must be default, got {rope_type:?} — a YaRN variant is not served by this line"
        );
    }
    // The reference reads the rope base from this same nested object with a
    // direct index (modeling_qwen4_exp.py:108: `config.rope_parameters["rope_theta"]`),
    // so a theta it would KeyError on is refused here, and `RopeParameters`
    // carries it — no consumer may re-read `config.json` for it, the same
    // second-source rule `num_experts_per_tok` follows.
    let theta = rope.get("rope_theta").and_then(serde_json::Value::as_f64);
    if theta.is_none_or(|t| !t.is_finite() || t <= 0.0) {
        bail!("qwen4_exp: rope_parameters.rope_theta must be a positive number, got {theta:?}");
    }
    // Every rope value the reference uses comes from this nested object
    // (`modeling_qwen4_exp.py` reads `config.rope_parameters["rope_theta"]` and
    // `[..., "partial_rotary_factor"]`), and `Config` models it from here, so
    // this is the canonical copy. The checkpoint also carries a legacy
    // top-level one that nothing upstream reads; when the two disagree the
    // checkpoint is self-inconsistent, and picking one would silently rotate at
    // the wrong frequency — the same both-levels rule this file applies to
    // `tie_word_embeddings`.
    let partial = rope
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64);
    let Some(partial) = partial.filter(|p| *p > 0.0 && *p <= 1.0) else {
        bail!("qwen4_exp: rope_parameters.partial_rotary_factor must be in (0, 1]");
    };
    if let Some(legacy) = text_config
        .get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
    {
        // Exact equality is the intent: both are JSON literals, and a tolerance
        // would hide a checkpoint that writes `0.3` in one place and `0.25` in the
        // other — which is the case this check exists to catch.
        #[allow(clippy::float_cmp)]
        let disagrees = legacy != partial;
        if disagrees {
            bail!(
                "qwen4_exp: partial_rotary_factor is ambiguous: text_config says {legacy}, \
                 text_config.rope_parameters says {partial}, and only the nested copy is read"
            );
        }
    }

    // `mrope_interleaved` and `mrope_section` are accepted rather than refused,
    // because a text-only sequence is their degenerate case in the reference
    // itself: it expands a 1-D `position_ids` across all three axes and then
    // interleaves (`modeling_qwen4_exp.py`), so every rotary pair takes the same
    // position with its own frequency and the three sections collapse onto the
    // 1-D layout. This line serves text only, so it needs no mrope path of its
    // own; the decision, and the fixtures that would tell it apart, are tracked
    // in #1105.
    let head_dim = text_config
        .get("head_dim")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let rotary = (head_dim as f64 * partial).round() as u64;
    if rotary == 0 || (head_dim as f64 * partial).fract() != 0.0 {
        bail!(
            "qwen4_exp: partial_rotary_factor {partial} x head_dim {head_dim} is not a whole number of rotary dimensions"
        );
    }
    // The reference requires the rotary width to fit the QSA index head
    // (configuration_qwen4_exp.py:227); the frozen config's 64 rotary dims fit
    // its 128-wide index head. `indexer_head_dim` is positive by
    // `probe_attention`, which runs first.
    let indexer_head_dim = text_config
        .get("indexer_head_dim")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if rotary > indexer_head_dim {
        bail!(
            "qwen4_exp: rotary_dim {rotary} exceeds indexer_head_dim {indexer_head_dim} — the rotary width must fit the QSA index head"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen revision's real `config.json`, committed at
    /// `tests/frozen_config.json` and hash-pinned in `config.rs`. Testing against
    /// it rather than a transcription is the point: a hand-written fixture can
    /// drift from the checkpoint silently, which is the failure mode this probe
    /// exists to prevent.
    fn frozen() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap()
    }

    fn err(mutate: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut cfg = frozen();
        mutate(&mut cfg);
        probe_config_json(&cfg).unwrap_err().to_string()
    }

    fn text(cfg: &mut serde_json::Value, field: &str, value: serde_json::Value) {
        cfg["text_config"][field] = value;
    }

    #[test]
    fn the_frozen_revision_passes() {
        probe_config_json(&frozen()).unwrap();
    }

    #[test]
    fn qwen35_is_not_mis_routed_here() {
        // The Qwen3.5 probe hard-requires `qwen3_5`; this one must hard-reject it
        // in turn, so neither line can silently claim the other's checkpoint.
        let cfg = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"}
        });
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("qwen3_5"), "{err}");
    }

    /// The registry picks a *unique* claimant and errors on a conflict, so this
    /// line accepting any other family's identity would be a routing regression
    /// in that family rather than in this one. This is the half of "the existing
    /// lines' routing is unchanged" that this crate can prove on its own; the
    /// other half — that no existing line claims `qwen4_exp` — rests on each
    /// line's own hard identity gate plus the registry's `Conflict` error, and
    /// the `feature_gate_hint` tests in `pegainfer-server` pin that the hint
    /// table does not cross-map the two Qwen families.
    #[test]
    fn no_other_lines_identity_is_accepted() {
        for model_type in [
            "qwen3",
            "qwen3_5",
            "qwen4",
            "gemma3",
            "gemma4",
            "gemma4_unified",
            "glm_moe_dsa",
            "kimi_k2",
            "kimi_k25",
            "kimi_k3",
            "deepseek_v2",
            // The inner identity alone must not be enough: a config that carries
            // the text type but a foreign outer type is a cross-family mismatch.
            "qwen4_exp_text",
            "",
        ] {
            let cfg = serde_json::json!({
                "model_type": model_type,
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "tie_word_embeddings": false,
                "text_config": {"model_type": "qwen4_exp_text"}
            });
            let err = match probe_config_json(&cfg) {
                Ok(()) => panic!("{model_type} must not be claimed by this line"),
                Err(error) => error.to_string(),
            };
            assert!(
                err.contains("not a Qwen3.8-Flash-Next config"),
                "{model_type}: {err}"
            );
        }
    }

    #[test]
    fn a_cross_family_text_config_bails() {
        let err = err(|c| text(c, "model_type", serde_json::json!("qwen3_5_text")));
        assert!(err.contains("cross-family mismatch"), "{err}");
    }

    #[test]
    fn a_missing_architecture_bails() {
        let err = err(|c| c["architectures"] = serde_json::json!(["Qwen4ExpForCausalLM"]));
        assert!(err.contains("architectures"), "{err}");
    }

    #[test]
    fn a_missing_text_config_bails() {
        let mut cfg = frozen();
        cfg.as_object_mut().unwrap().remove("text_config");
        // Without `text_config` the tie check falls back to the outer node, so the
        // refusal must still name a real reason rather than panic.
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("text_config") || err.contains("tie_word_embeddings"),
            "{err}"
        );
    }

    #[test]
    fn the_fp8_sibling_fails_closed() {
        let err = err(|c| text(c, "dtype", serde_json::json!("float8_e4m3fn")));
        assert!(err.contains("bfloat16"), "{err}");
        assert!(err.contains("separate revision"), "{err}");
    }

    #[test]
    fn a_tied_checkpoint_fails_closed() {
        let err = err(|c| text(c, "tie_word_embeddings", serde_json::json!(true)));
        assert!(err.contains("tie_word_embeddings"), "{err}");
    }

    #[test]
    fn a_gdn_head_dim_off_the_aot_key_bails() {
        let err = err(|c| text(c, "linear_value_head_dim", serde_json::json!(64)));
        assert!(err.contains("linear_value_head_dim"), "{err}");
        assert!(err.contains("AOT"), "{err}");
    }

    #[test]
    fn a_conv_kernel_past_the_in_tree_cap_bails() {
        let err = err(|c| text(c, "linear_conv_kernel_dim", serde_json::json!(5)));
        assert!(err.contains("linear_conv_kernel_dim"), "{err}");
    }

    #[test]
    fn a_non_sigmoid_output_gate_bails() {
        // The sibling Qwen3.8-27B checkpoint carries `swish` here, where the field
        // is dead. On this architecture it is live and selects the GDN gated-norm
        // activation, so a different value is an unvalidated variant.
        let err = err(|c| text(c, "output_gate_type", serde_json::json!("swish")));
        assert!(err.contains("output_gate_type"), "{err}");
    }

    #[test]
    fn a_non_f32_recurrent_state_bails() {
        let err = err(|c| text(c, "mamba_ssm_dtype", serde_json::json!("bfloat16")));
        assert!(err.contains("mamba_ssm_dtype"), "{err}");
    }

    #[test]
    fn a_yarn_rope_bails() {
        let err = err(|c| {
            c["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        });
        assert!(err.contains("rope_type"), "{err}");
    }

    #[test]
    fn a_partial_rotary_factor_that_is_not_whole_bails() {
        let err = err(|c| {
            c["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.3);
        });
        assert!(err.contains("partial_rotary_factor"), "{err}");
    }

    #[test]
    fn the_full_attention_interval_pattern_is_enforced() {
        // Layer 1 is linear under interval 4; making it full must be rejected. The
        // rule is the interval, not a hardcoded count of 12 full layers.
        let err = err(|c| c["text_config"]["layer_types"][1] = serde_json::json!("full_attention"));
        assert!(err.contains("layer_types[1]"), "{err}");
    }

    #[test]
    fn a_layer_count_not_divisible_by_the_interval_bails() {
        let err = err(|c| {
            c["text_config"]["layer_types"] =
                serde_json::json!(["linear_attention", "linear_attention", "full_attention"]);
        });
        assert!(err.contains("multiple of full_attention_interval"), "{err}");
    }

    #[test]
    fn an_unknown_layer_type_bails() {
        let err = err(|c| c["text_config"]["layer_types"][0] = serde_json::json!("mamba"));
        assert!(err.contains("layer_types[0]"), "{err}");
    }

    #[test]
    fn top_k_above_the_expert_count_bails() {
        let err = err(|c| text(c, "num_experts_per_tok", serde_json::json!(513)));
        assert!(err.contains("num_experts"), "{err}");
    }

    #[test]
    fn a_missing_hyper_connection_field_bails() {
        let err = err(|c| {
            c["text_config"]
                .as_object_mut()
                .unwrap()
                .remove("hc_lowrank");
        });
        assert!(err.contains("hc_lowrank"), "{err}");
    }

    #[test]
    fn ple_layer_ids_are_one_indexed_and_range_checked() {
        // `[2]` on the frozen checkpoint means the tensors live at `layers.1`.
        // Zero is not a legal 1-indexed id, and neither is one past the stack.
        let zero = err(|c| text(c, "ple_layer_ids", serde_json::json!([0])));
        assert!(zero.contains("1-indexed"), "{zero}");
        let past = err(|c| text(c, "ple_layer_ids", serde_json::json!([49])));
        assert!(past.contains("outside 1..=48"), "{past}");
    }

    #[test]
    fn a_unigram_only_ngram_bails() {
        let err = err(|c| text(c, "ngram_size", serde_json::json!(1)));
        assert!(err.contains("ngram_size"), "{err}");
    }

    /// Widths a future smaller `qwen4_exp` checkpoint would change must not be
    /// pinned, or the probe stops being reusable for the small-model deployment
    /// case this line exists to pay for.
    #[test]
    fn smaller_widths_still_pass() {
        let mut cfg = frozen();
        text(&mut cfg, "num_experts", serde_json::json!(64));
        text(&mut cfg, "num_experts_per_tok", serde_json::json!(4));
        text(&mut cfg, "moe_intermediate_size", serde_json::json!(512));
        text(
            &mut cfg,
            "shared_expert_intermediate_size",
            serde_json::json!(512),
        );
        text(&mut cfg, "hidden_size", serde_json::json!(1280));
        text(&mut cfg, "hc_count", serde_json::json!(2));
        text(&mut cfg, "hc_lowrank", serde_json::json!(160));
        text(&mut cfg, "linear_num_key_heads", serde_json::json!(8));
        text(&mut cfg, "linear_num_value_heads", serde_json::json!(24));
        text(&mut cfg, "num_attention_heads", serde_json::json!(12));
        probe_config_json(&cfg).unwrap();
    }

    /// The reference reads `partial_rotary_factor` from `rope_parameters`; the
    /// legacy top-level copy the checkpoint also carries is read by nothing
    /// upstream. A checkpoint where the two disagree is ambiguous, and this line
    /// refuses rather than guessing which frequency to rotate at.
    #[test]
    fn an_ambiguous_partial_rotary_factor_bails() {
        let err = err(|c| text(c, "partial_rotary_factor", serde_json::json!(0.5)));
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn a_layer_count_that_disagrees_with_layer_types_bails() {
        let err = err(|c| text(c, "num_hidden_layers", serde_json::json!(47)));
        assert!(err.contains("num_hidden_layers"), "{err}");
        assert!(err.contains("layer_types"), "{err}");
    }

    #[test]
    fn an_ngram_base_below_two_bails() {
        let err = err(|c| text(c, "ngram_vocab_size_base", serde_json::json!(1)));
        assert!(err.contains("ngram_vocab_size_base"), "{err}");
    }

    #[test]
    fn a_zero_residual_width_bails() {
        let err = err(|c| text(c, "hidden_size", serde_json::json!(0)));
        assert!(err.contains("hidden_size"), "{err}");
    }

    #[test]
    fn a_zero_vocabulary_bails() {
        let err = err(|c| text(c, "vocab_size", serde_json::json!(0)));
        assert!(err.contains("vocab_size"), "{err}");
    }

    #[test]
    fn a_single_branch_residual_stream_bails() {
        // The reference refuses hc_count <= 1 (configuration_qwen4_exp.py:196);
        // a port that accepted it would serve a mixer with nothing to mix.
        let err = err(|c| text(c, "hc_count", serde_json::json!(1)));
        assert!(err.contains("hc_count"), "{err}");
    }

    #[test]
    fn an_indexer_kv_heads_above_one_bails() {
        let err = err(|c| text(c, "indexer_kv_heads", serde_json::json!(2)));
        assert!(err.contains("indexer_kv_heads"), "{err}");
    }

    #[test]
    fn an_indexer_budget_not_divisible_by_the_compress_ratio_bails() {
        let err = err(|c| text(c, "indexer_budget", serde_json::json!(2049)));
        assert!(err.contains("indexer_budget"), "{err}");
    }

    #[test]
    fn a_rotary_width_past_the_indexer_head_bails() {
        // The frozen 64 rotary dims (0.25 x 256) fit the 128-wide index head;
        // a 32-wide one does not (configuration_qwen4_exp.py:227).
        let err = err(|c| text(c, "indexer_head_dim", serde_json::json!(32)));
        assert!(err.contains("indexer_head_dim"), "{err}");
    }

    #[test]
    fn a_multi_layer_ple_bails() {
        // The reference offsets each PLE layer's prime run by its layer index
        // (modeling_qwen4_exp.py:1037-1039); this crate derives one table, so a
        // second layer's geometry would be wrong rather than merely unmodelled.
        let err = err(|c| text(c, "ple_layer_ids", serde_json::json!([2, 6])));
        assert!(err.contains("ple_layer_ids"), "{err}");
    }

    #[test]
    fn ple_on_a_full_attention_layer_bails() {
        // Under interval 4, 1-indexed layer 4 is full attention; the reference
        // supports PLE only on linear_attention layers
        // (configuration_qwen4_exp.py:248-254).
        let err = err(|c| text(c, "ple_layer_ids", serde_json::json!([4])));
        assert!(err.contains("linear_attention"), "{err}");
    }

    #[test]
    fn a_missing_rope_theta_bails() {
        // The reference indexes rope_parameters["rope_theta"] directly
        // (modeling_qwen4_exp.py:108); a config without it is unservable there,
        // so it is refused here rather than parsed into a half-modelled rope.
        let err = err(|c| {
            c["text_config"]["rope_parameters"]
                .as_object_mut()
                .unwrap()
                .remove("rope_theta");
        });
        assert!(err.contains("rope_theta"), "{err}");
    }
}
