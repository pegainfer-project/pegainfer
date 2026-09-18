use std::collections::BTreeMap;
use std::fs;
use std::io::Write;

use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;

use super::NativeArtifact;
use crate::config::DFlashConfig;
use crate::dflash::config::NativeDFlash2Config;
use crate::dflash::config::NativeTargetMetadata;

const CONFIG: &str = include_str!("../../../tests/fixtures/dflash2/native_config.json");
const HEADER: &str = include_str!("../../../tests/fixtures/dflash2/native_header.json");

fn small_config() -> Value {
    let mut config: Value = serde_json::from_str(CONFIG).unwrap();
    for (name, value) in [
        ("hidden_size", 8),
        ("intermediate_size", 12),
        ("num_hidden_layers", 1),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 1),
        ("head_dim", 4),
        ("vocab_size", 32),
        ("num_target_layers", 3),
        ("max_window_layers", 1),
    ] {
        config[name] = json!(value);
    }
    config["layer_types"] = json!(["sliding_attention"]);
    config["dflash_config"] = json!({
        "block_size": 3, "conv_group_size": 4, "conv_kernel_size": 2,
        "mask_token_id": 31, "selector_rank": 5, "selector_top_k": 16,
        "target_layer_ids": [0, 2],
    });
    config
}

fn root(config: &Value) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    dir
}

fn tiny_header(config: &Value) -> Value {
    let config = NativeDFlash2Config::from_json(config).unwrap();
    let mut offset = 0;
    let entries: BTreeMap<_, _> = config
        .expected_tensors()
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() * 2;
            let value =
                json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bytes]});
            offset += bytes;
            (name, value)
        })
        .collect();
    serde_json::to_value(entries).unwrap()
}

// Only metadata is under test: sparse zero payloads are not model/oracle weights.
fn write_header(path: &std::path::Path, header: &Value) {
    write_raw_header(
        path,
        &serde_json::to_string(header).unwrap(),
        header
            .as_object()
            .unwrap()
            .values()
            .filter_map(|v| v["data_offsets"][1].as_u64())
            .max()
            .unwrap_or(0),
    );
}

fn write_raw_header(path: &std::path::Path, header: &str, payload_bytes: u64) {
    let mut bytes = header.as_bytes().to_vec();
    bytes.resize(bytes.len().next_multiple_of(8), b' ');
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&bytes).unwrap();
    file.set_len(8 + bytes.len() as u64 + payload_bytes)
        .unwrap();
}

fn inspect(dir: &TempDir) -> anyhow::Result<NativeArtifact> {
    NativeArtifact::inspect(dir.path().to_str().unwrap())
}

#[test]
fn released_config_and_complete_header_match() {
    let config: Value = serde_json::from_str(CONFIG).unwrap();
    let dir = root(&config);
    write_header(
        &dir.path().join("model.safetensors"),
        &serde_json::from_str(HEADER).unwrap(),
    );
    let artifact = inspect(&dir).unwrap();
    assert_eq!(artifact.tensor_manifest().len(), 81);
    assert_eq!(artifact.selector_weight_bytes().unwrap(), 256_901_120);
    assert_eq!(artifact.weight_bytes().unwrap(), 3_848_808_960);
    assert_eq!(artifact.config().candidate_count(), 16);
    assert!(!config["tie_word_embeddings"].as_bool().unwrap());
    let error = DFlashConfig::from_file(dir.path().to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("native DFlash2 backbone execution is not supported"),
        "{error}"
    );
    assert!(error.contains("sliding-window attention"), "{error}");
}

#[test]
fn config_rejects_ambiguous_and_unsupported_computation() {
    for (field, value) in [
        ("rope_scaling", json!({"type": "linear", "factor": 2})),
        ("logit_scale", json!(0.5)),
        ("quantization_config", json!({})),
        ("draft_vocab_size", json!(16)),
        ("num_anchors", json!(1)),
    ] {
        let mut config = small_config();
        config[field] = value;
        let error = NativeDFlash2Config::from_json(&config)
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{error}");
    }
    let mut config = small_config();
    config["dflash_config"]["proposal_type"] = json!("sampled");
    assert!(
        format!("{:#}", NativeDFlash2Config::from_json(&config).unwrap_err())
            .contains("proposal_type")
    );
    config = small_config();
    config["architectures"] = json!(["DFlashDraftModel"]);
    assert!(NativeDFlash2Config::is_native(&config));
    assert!(NativeDFlash2Config::from_json(&config).is_err());
}

#[test]
fn config_guards_shapes_tokens_and_overflow() {
    for (pointer, value) in [
        ("/hidden_size", json!(0)),
        ("/num_key_value_heads", json!(0)),
        ("/head_dim", json!(3)),
        ("/dflash_config/selector_top_k", json!(15)),
        ("/dflash_config/selector_rank", json!(0)),
        ("/dflash_config/block_size", json!(1)),
        ("/dflash_config/conv_group_size", json!(3)),
        ("/dflash_config/mask_token_id", json!(32)),
        ("/dflash_config/target_layer_ids", json!([2, 0])),
        ("/rope_parameters/rope_type", json!("yarn")),
        ("/sliding_window", json!(0)),
        ("/hidden_size", json!(u64::MAX)),
    ] {
        let mut config = small_config();
        *config.pointer_mut(pointer).unwrap() = value;
        assert!(
            NativeDFlash2Config::from_json(&config).is_err(),
            "{pointer}"
        );
    }
    let mut config = small_config();
    config["dflash_config"]["selector_rank"] = json!(i32::MAX);
    config["vocab_size"] = json!(i32::MAX);
    config["hidden_size"] = json!(i32::MAX - 3);
    assert!(NativeDFlash2Config::from_json(&config).is_err());
}

#[test]
fn target_validation_requires_provenance_and_geometry() {
    let config = NativeDFlash2Config::from_json(&small_config()).unwrap();
    let mut target = NativeTargetMetadata {
        model_id: "fixture/target".into(),
        revision: "0123456789abcdef".into(),
        tokenizer_id: "fixture/tokenizer".into(),
        tokenizer_revision: "fedcba9876543210".into(),
        hidden_size: 8,
        vocab_size: 32,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        rope_theta: 10_000_000.0,
        max_position_embeddings: 262_144,
    };
    config.validate_target(&target).unwrap();
    target.revision = "main".into();
    assert!(
        config
            .validate_target(&target)
            .unwrap_err()
            .to_string()
            .contains("revision")
    );
    target.revision = "0123456789abcdef".into();
    target.vocab_size = 31;
    assert!(
        config
            .validate_target(&target)
            .unwrap_err()
            .to_string()
            .contains("vocab_size")
    );
}

#[test]
fn released_target_has_a_different_attention_backbone() {
    let config = NativeDFlash2Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap();
    let mut target = NativeTargetMetadata {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into(),
        tokenizer_id: "Qwen/Qwen3.8-27B".into(),
        tokenizer_revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into(),
        hidden_size: 5120,
        vocab_size: 248_320,
        num_hidden_layers: 64,
        num_attention_heads: 24,
        num_key_value_heads: 4,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        max_position_embeddings: 262_144,
    };
    config.validate_target(&target).unwrap();
    assert_eq!(config.rope_theta().to_bits(), 10_000_000.0f64.to_bits());
    target.num_key_value_heads = 5;
    assert!(
        config
            .validate_target(&target)
            .unwrap_err()
            .to_string()
            .contains("attention geometry")
    );
}

#[test]
fn tensor_preflight_rejects_corruption_and_foreign_weight_sources() {
    let config = small_config();
    for name in [
        "candidate_selector.predecessor_codebook",
        "layers.0.attention_conv.kernel_projection.weight",
    ] {
        for mutation in ["missing", "shape", "dtype"] {
            let dir = root(&config);
            let mut header = tiny_header(&config);
            match mutation {
                "missing" => {
                    header.as_object_mut().unwrap().remove(name);
                    reoffset(&mut header);
                }
                "shape" => {
                    header[name]["shape"][0] = json!(7);
                    reoffset(&mut header);
                }
                "dtype" => {
                    header[name]["dtype"] = json!("F16");
                }
                _ => unreachable!(),
            }
            write_header(&dir.path().join("model.safetensors"), &header);
            let error = format!("{:#}", inspect(&dir).unwrap_err());
            assert!(error.contains(name), "{mutation}: {error}");
        }
    }
    let dir = root(&config);
    let mut header = tiny_header(&config);
    header["lm_head.weight"] = json!({"dtype":"BF16", "shape":[32,8], "data_offsets":[0,0]});
    reoffset(&mut header);
    write_header(&dir.path().join("model.safetensors"), &header);
    assert!(
        inspect(&dir)
            .unwrap_err()
            .to_string()
            .contains("unsupported tensor 'lm_head.weight'")
    );
    fs::write(dir.path().join("mask_embedding.pt"), []).unwrap();
    assert!(
        inspect(&dir)
            .unwrap_err()
            .to_string()
            .contains("mask_embedding.pt")
    );
}

fn reoffset(header: &mut Value) {
    let mut offset = 0;
    for value in header.as_object_mut().unwrap().values_mut() {
        let bytes = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .product::<u64>()
            * 2;
        value["data_offsets"] = json!([offset, offset + bytes]);
        offset += bytes;
    }
}

#[test]
fn sharded_checkpoint_checks_index_coverage_duplicates_and_locations() {
    let config = small_config();
    let header = tiny_header(&config);
    let dir = root(&config);
    let mut first = json!({});
    let mut second = json!({});
    let mut weight_map = json!({});
    for (index, (name, tensor)) in header.as_object().unwrap().iter().enumerate() {
        let (part, path) = if index % 2 == 0 {
            (&mut first, "a.safetensors")
        } else {
            (&mut second, "b.safetensors")
        };
        part[name] = tensor.clone();
        weight_map[name] = json!(path);
    }
    reoffset(&mut first);
    reoffset(&mut second);
    write_header(&dir.path().join("a.safetensors"), &first);
    write_header(&dir.path().join("b.safetensors"), &second);
    let write_index = |map: &Value| {
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
    };
    write_index(&weight_map);
    assert_eq!(inspect(&dir).unwrap().tensor_manifest().len(), 21);
    let mut relative_map = weight_map.clone();
    for path in relative_map.as_object_mut().unwrap().values_mut() {
        *path = json!(format!("./{}", path.as_str().unwrap()));
    }
    write_index(&relative_map);
    assert_eq!(inspect(&dir).unwrap().tensor_manifest().len(), 21);
    let name = first.as_object().unwrap().keys().next().unwrap();
    relative_map[name] = weight_map[name].clone();
    write_index(&relative_map);
    assert_eq!(inspect(&dir).unwrap().tensor_manifest().len(), 21);
    let mut broken = weight_map.clone();
    broken.as_object_mut().unwrap().remove(name);
    write_index(&broken);
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("disagrees with weight_map"));
    broken = weight_map.clone();
    broken[name] = json!("b.safetensors");
    write_index(&broken);
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("disagrees with weight_map"));
    broken = weight_map.clone();
    broken["not.present"] = json!("a.safetensors");
    write_index(&broken);
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("missing from checkpoint"));
    write_index(&weight_map);
    second[name] = first[name].clone();
    reoffset(&mut second);
    write_header(&dir.path().join("b.safetensors"), &second);
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("Duplicate tensor"));
}

#[test]
fn malformed_index_and_duplicate_header_never_silently_win() {
    let dir = root(&small_config());
    for (index, expected) in [
        (r#"{"weight_map":{"x":1}}"#, "string"),
        (r#"{"weight_map":{"x":"a","x":"b"}}"#, "duplicate tensor"),
        (r#"{"weight_map":{"x":"../outside"}}"#, "unsafe shard path"),
        (r#"{"weight_map":{"x":"/outside"}}"#, "unsafe shard path"),
        (r#"{"weight_map":{"x":"./"}}"#, "unsafe shard path"),
        (r#"{"weight_map":{}}"#, "empty weight_map"),
    ] {
        fs::write(dir.path().join("model.safetensors.index.json"), index).unwrap();
        assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains(expected));
    }
    fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
    write_raw_header(
        &dir.path().join("model.safetensors"),
        r#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        2,
    );
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("duplicate tensor"));
    fs::write(dir.path().join("model.safetensors"), b"broken header").unwrap();
    assert!(format!("{:#}", inspect(&dir).unwrap_err()).contains("Deserialize error"));
}
