use serde_json::json;

use super::load::KimiTensorLoadSlice;
use super::load::KimiTensorLoadSpec;
use super::load::sliced_tensor_bytes;
use super::manifest::KimiAttentionManifest;
use super::manifest::KimiDenseMlpManifest;
use super::manifest::KimiInt4ProjectionManifest;
use super::manifest::KimiK2WeightManifest;
use super::manifest::KimiLayerKindManifest;
use super::manifest::KimiLayerManifest;
use super::manifest::KimiMoeLayerManifest;
use super::manifest::KimiRoutedExpertManifest;
use super::manifest::KimiRouterManifest;
use super::manifest::KimiSharedExpertManifest;
use super::manifest::KimiTensorEntry;
use super::*;

#[test]
fn rank_tensor_names_filter_local_experts() {
    let manifest = tiny_manifest();
    let rank0 = manifest.rank_tensor_names(0).unwrap();
    assert!(rank0.iter().any(|entry| entry.name.contains("experts.0.")));
    assert!(rank0.iter().any(|entry| entry.name.contains("experts.47.")));
    assert!(!rank0.iter().any(|entry| entry.name.contains("experts.48.")));
    let rank1 = manifest.rank_tensor_names(1).unwrap();
    assert!(rank1.iter().any(|entry| entry.name.contains("experts.48.")));
    assert!(!rank1.iter().any(|entry| entry.name.contains("experts.47.")));
}

#[test]
fn rank_weight_names_are_local_and_typed() {
    let manifest = tiny_manifest();
    let names = manifest.rank_weight_names(1).unwrap();
    assert_eq!(names.rank, 1);
    assert_eq!(names.plan.local_expert_range, 48..96);
    assert_eq!(
        names.top.token_embedding,
        "language_model.model.embed_tokens.weight"
    );
    assert_eq!(names.layers.len(), KIMI_K2_LAYERS);
    match &names.layers[0].kind {
        KimiLayerWeightKindNames::Dense(mlp) => {
            assert_eq!(
                mlp.gate_proj,
                "language_model.model.layers.0.mlp.gate_proj.weight"
            );
        }
        KimiLayerWeightKindNames::Moe(_) => panic!("layer0 must be dense"),
    }
    match &names.layers[1].kind {
        KimiLayerWeightKindNames::Moe(moe) => {
            assert_eq!(moe.routed_experts.len(), 48);
            assert_eq!(moe.routed_experts[0].global_expert, 48);
            assert_eq!(moe.routed_experts[47].global_expert, 95);
        }
        KimiLayerWeightKindNames::Dense(_) => panic!("layer1 must be MoE"),
    }
}

#[test]
fn rank_sliced_load_plan_applies_tp8_ep8_slices() {
    let manifest = tiny_manifest();
    let load_plan = manifest.rank_sliced_load_plan(3).unwrap();
    assert_eq!(load_plan.rank, 3);
    // 8,640 fewer than the checkpoint carries: weight_shape (60 MoE layers ×
    // 48 local experts × 3 projections) is never read by any kernel (#234).
    assert_eq!(load_plan.tensor_count, 18_135);

    assert_eq!(
        find_load_spec(&load_plan, "language_model.model.embed_tokens.weight").slice,
        KimiTensorLoadSlice::RowRange {
            start: 61_440,
            end: 81_920
        }
    );
    assert_eq!(
        find_load_spec(&load_plan, "language_model.lm_head.weight").slice,
        KimiTensorLoadSlice::RowRange {
            start: 61_440,
            end: 81_920
        }
    );
    assert_eq!(
        find_load_spec(&load_plan, "language_model.model.norm.weight").slice,
        KimiTensorLoadSlice::Full
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.q_b_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 4_608,
            end: 6_144
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.kv_b_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 6_144,
            end: 8_192
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.self_attn.o_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::ColRange {
            start: 3_072,
            end: 4_096
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.0.mlp.gate_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::RowRange {
            start: 6_912,
            end: 9_216
        }
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.shared_experts.down_proj.weight"
        )
        .slice,
        KimiTensorLoadSlice::ColRange {
            start: 768,
            end: 1_024
        }
    );

    assert!(
        find_load_spec_opt(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.143.gate_proj.weight_packed"
        )
        .is_none()
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.144.gate_proj.weight_packed"
        )
        .slice,
        KimiTensorLoadSlice::Full
    );
    assert_eq!(
        find_load_spec(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.191.down_proj.weight_scale"
        )
        .slice,
        KimiTensorLoadSlice::Full
    );
    // weight_shape is never read by any kernel (#234): it must not be loaded.
    assert!(
        find_load_spec_opt(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.191.down_proj.weight_shape"
        )
        .is_none()
    );
    assert!(
        find_load_spec_opt(
            &load_plan,
            "language_model.model.layers.1.mlp.experts.192.gate_proj.weight_packed"
        )
        .is_none()
    );
}

#[test]
fn sliced_tensor_bytes_packs_col_slice_as_row_major() {
    let data = (0u8..24).collect::<Vec<_>>();
    let out = sliced_tensor_bytes(
        &data,
        &[3, 4],
        Dtype::BF16,
        &KimiTensorLoadSlice::ColRange { start: 1, end: 3 },
    )
    .unwrap();
    assert_eq!(out, vec![2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21]);
}

#[test]
fn scanner_rejects_missing_required_text_tensor() {
    let json = json!({
        "metadata": {"total_size": 1},
        "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001-of-000064.safetensors"
        }
    });
    let err = KimiK2WeightManifest::from_index_json(&json).unwrap_err();
    assert!(err.to_string().contains("language_model.model.norm.weight"));
}

fn find_load_spec<'a>(plan: &'a KimiRankSlicedLoadPlan, name: &str) -> &'a KimiTensorLoadSpec {
    find_load_spec_opt(plan, name).unwrap_or_else(|| panic!("missing load spec {name}"))
}

fn find_load_spec_opt<'a>(
    plan: &'a KimiRankSlicedLoadPlan,
    name: &str,
) -> Option<&'a KimiTensorLoadSpec> {
    plan.shards
        .iter()
        .flat_map(|shard| shard.tensors.iter())
        .find(|spec| spec.name == name)
}

fn tiny_manifest() -> KimiK2WeightManifest {
    let mut layers = Vec::new();
    for layer_idx in 0..KIMI_K2_LAYERS {
        let attention = KimiAttentionManifest {
            input_layernorm: fake(layer_idx, "input_layernorm.weight"),
            q_a_proj: fake(layer_idx, "self_attn.q_a_proj.weight"),
            q_a_layernorm: fake(layer_idx, "self_attn.q_a_layernorm.weight"),
            q_b_proj: fake(layer_idx, "self_attn.q_b_proj.weight"),
            kv_a_proj_with_mqa: fake(layer_idx, "self_attn.kv_a_proj_with_mqa.weight"),
            kv_a_layernorm: fake(layer_idx, "self_attn.kv_a_layernorm.weight"),
            kv_b_proj: fake(layer_idx, "self_attn.kv_b_proj.weight"),
            o_proj: fake(layer_idx, "self_attn.o_proj.weight"),
            post_attention_layernorm: fake(layer_idx, "post_attention_layernorm.weight"),
        };
        let kind = if layer_idx == 0 {
            KimiLayerKindManifest::Dense(KimiDenseMlpManifest {
                gate_proj: fake(layer_idx, "mlp.gate_proj.weight"),
                up_proj: fake(layer_idx, "mlp.up_proj.weight"),
                down_proj: fake(layer_idx, "mlp.down_proj.weight"),
            })
        } else {
            KimiLayerKindManifest::Moe(KimiMoeLayerManifest {
                router: KimiRouterManifest {
                    gate_weight: fake(layer_idx, "mlp.gate.weight"),
                    e_score_correction_bias: fake(layer_idx, "mlp.gate.e_score_correction_bias"),
                },
                shared_experts: KimiSharedExpertManifest {
                    gate_proj: fake(layer_idx, "mlp.shared_experts.gate_proj.weight"),
                    up_proj: fake(layer_idx, "mlp.shared_experts.up_proj.weight"),
                    down_proj: fake(layer_idx, "mlp.shared_experts.down_proj.weight"),
                },
                routed_experts: (0..KIMI_K2_ROUTED_EXPERTS)
                    .map(|expert_idx| KimiRoutedExpertManifest {
                        expert_idx,
                        gate_proj: fake_int4(layer_idx, expert_idx, "gate_proj"),
                        up_proj: fake_int4(layer_idx, expert_idx, "up_proj"),
                        down_proj: fake_int4(layer_idx, expert_idx, "down_proj"),
                    })
                    .collect(),
            })
        };
        layers.push(KimiLayerManifest {
            layer_idx,
            attention,
            kind,
        });
    }
    KimiK2WeightManifest {
        text_tensor_count: 208_215,
        token_embedding: top("language_model.model.embed_tokens.weight"),
        final_norm: top("language_model.model.norm.weight"),
        lm_head: top("language_model.lm_head.weight"),
        layers,
        parallel: KimiK2ParallelShape::tp8_ep8(),
    }
}

fn fake(layer_idx: usize, suffix: &str) -> KimiTensorEntry {
    top(&format!("language_model.model.layers.{layer_idx}.{suffix}"))
}

fn fake_int4(layer_idx: usize, expert_idx: usize, projection: &str) -> KimiInt4ProjectionManifest {
    let prefix =
        format!("language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}");
    KimiInt4ProjectionManifest {
        weight_packed: top(&format!("{prefix}.weight_packed")),
        weight_scale: top(&format!("{prefix}.weight_scale")),
    }
}

fn top(name: &str) -> KimiTensorEntry {
    KimiTensorEntry {
        name: name.to_owned(),
        shard: "model-00001-of-000064.safetensors".to_owned(),
    }
}
