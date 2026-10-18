use super::load::KimiRankSlicedLoadPlan;
use super::load::KimiShardTensorLoadPlan;
use super::load::KimiTensorLoadSlice;
use super::load::KimiTensorLoadSpec;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTensorEntry {
    pub name: String,
    pub shard: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiAttentionManifest {
    pub input_layernorm: KimiTensorEntry,
    pub q_a_proj: KimiTensorEntry,
    pub q_a_layernorm: KimiTensorEntry,
    pub q_b_proj: KimiTensorEntry,
    pub kv_a_proj_with_mqa: KimiTensorEntry,
    pub kv_a_layernorm: KimiTensorEntry,
    pub kv_b_proj: KimiTensorEntry,
    pub o_proj: KimiTensorEntry,
    pub post_attention_layernorm: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiDenseMlpManifest {
    pub gate_proj: KimiTensorEntry,
    pub up_proj: KimiTensorEntry,
    pub down_proj: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRouterManifest {
    pub gate_weight: KimiTensorEntry,
    pub e_score_correction_bias: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiSharedExpertManifest {
    pub gate_proj: KimiTensorEntry,
    pub up_proj: KimiTensorEntry,
    pub down_proj: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiInt4ProjectionManifest {
    pub weight_packed: KimiTensorEntry,
    pub weight_scale: KimiTensorEntry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRoutedExpertManifest {
    pub expert_idx: usize,
    pub gate_proj: KimiInt4ProjectionManifest,
    pub up_proj: KimiInt4ProjectionManifest,
    pub down_proj: KimiInt4ProjectionManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiMoeLayerManifest {
    pub router: KimiRouterManifest,
    pub shared_experts: KimiSharedExpertManifest,
    pub routed_experts: Vec<KimiRoutedExpertManifest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiLayerKindManifest {
    Dense(KimiDenseMlpManifest),
    Moe(KimiMoeLayerManifest),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiLayerManifest {
    pub layer_idx: usize,
    pub attention: KimiAttentionManifest,
    pub kind: KimiLayerKindManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiK2WeightManifest {
    pub text_tensor_count: usize,
    pub token_embedding: KimiTensorEntry,
    pub final_norm: KimiTensorEntry,
    pub lm_head: KimiTensorEntry,
    pub layers: Vec<KimiLayerManifest>,
    pub parallel: KimiK2ParallelShape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankWeightPlan {
    pub tp_rank: usize,
    pub ep_rank: usize,
    pub attention_head_range: Range<usize>,
    pub vocab_range: Range<usize>,
    pub local_expert_range: Range<usize>,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiTopWeightNames {
    pub token_embedding: String,
    pub final_norm: String,
    pub lm_head: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiAttentionWeightNames {
    pub input_layernorm: String,
    pub q_a_proj: String,
    pub q_a_layernorm: String,
    pub q_b_proj: String,
    pub kv_a_proj_with_mqa: String,
    pub kv_a_layernorm: String,
    pub kv_b_proj: String,
    pub o_proj: String,
    pub post_attention_layernorm: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiDenseMlpWeightNames {
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRouterWeightNames {
    pub gate_weight: String,
    pub e_score_correction_bias: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiSharedExpertWeightNames {
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiInt4ProjectionWeightNames {
    pub weight_packed: String,
    pub weight_scale: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRoutedExpertWeightNames {
    pub global_expert: usize,
    pub gate_proj: KimiInt4ProjectionWeightNames,
    pub up_proj: KimiInt4ProjectionWeightNames,
    pub down_proj: KimiInt4ProjectionWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiMoeLayerWeightNames {
    pub router: KimiRouterWeightNames,
    pub shared_experts: KimiSharedExpertWeightNames,
    pub routed_experts: Vec<KimiRoutedExpertWeightNames>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum KimiLayerWeightKindNames {
    Dense(KimiDenseMlpWeightNames),
    Moe(KimiMoeLayerWeightNames),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiLayerWeightNames {
    pub layer_idx: usize,
    pub attention: KimiAttentionWeightNames,
    pub kind: KimiLayerWeightKindNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KimiRankWeightNames {
    pub rank: usize,
    pub plan: KimiRankWeightPlan,
    pub top: KimiTopWeightNames,
    pub layers: Vec<KimiLayerWeightNames>,
}

impl KimiK2WeightManifest {
    pub(crate) fn from_model_dir(model_path: &Path) -> Result<Self> {
        Self::from_index_file(&model_path.join(KIMI_K2_WEIGHT_INDEX))
    }

    fn from_index_file(index_path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let json: Value = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;
        Self::from_index_json(&json)
    }

    pub(crate) fn from_index_json(json: &Value) -> Result<Self> {
        let weight_map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("Kimi safetensors index missing weight_map"))?;
        let mut tensors = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("weight_map entry {name} is not a shard string"))?;
            tensors.insert(name.as_str(), shard);
        }

        let token_embedding = tensor(&tensors, "language_model.model.embed_tokens.weight")?;
        let final_norm = tensor(&tensors, "language_model.model.norm.weight")?;
        let lm_head = tensor(&tensors, "language_model.lm_head.weight")?;
        let mut layers = Vec::with_capacity(KIMI_K2_LAYERS);
        for layer_idx in 0..KIMI_K2_LAYERS {
            let attention = attention_manifest(&tensors, layer_idx)?;
            let kind = if layer_idx < KIMI_K2_DENSE_LAYERS {
                KimiLayerKindManifest::Dense(dense_mlp_manifest(&tensors, layer_idx)?)
            } else {
                KimiLayerKindManifest::Moe(moe_layer_manifest(&tensors, layer_idx)?)
            };
            layers.push(KimiLayerManifest {
                layer_idx,
                attention,
                kind,
            });
        }

        let manifest = Self {
            text_tensor_count: weight_map
                .keys()
                .filter(|name| name.starts_with(TEXT_PREFIX))
                .count(),
            token_embedding,
            final_norm,
            lm_head,
            layers,
            parallel: KimiK2ParallelShape::tp8_ep8(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub(crate) fn with_parallel_shape(mut self, shape: KimiK2ParallelShape) -> Result<Self> {
        self.parallel = shape;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.layers.len() == KIMI_K2_LAYERS,
            "Kimi manifest expected {KIMI_K2_LAYERS} layers, got {}",
            self.layers.len()
        );
        let moe_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.kind, KimiLayerKindManifest::Moe(_)))
            .count();
        ensure!(
            moe_layers == KIMI_K2_MOE_LAYERS,
            "Kimi manifest expected {KIMI_K2_MOE_LAYERS} MoE layers, got {moe_layers}"
        );
        ensure!(
            self.parallel.ep_world > 0
                && KIMI_K2_ROUTED_EXPERTS.is_multiple_of(self.parallel.ep_world)
                && crate::config::KIMI_K2_HEADS.is_multiple_of(self.parallel.tp_world)
                && crate::config::KIMI_K2_VOCAB.is_multiple_of(self.parallel.tp_world),
            "Kimi parallel shape TP{}/DP{}/EP{} does not evenly divide model dimensions",
            self.parallel.tp_world,
            self.parallel.dp_world,
            self.parallel.ep_world
        );
        Ok(())
    }

    fn rank_plan(&self, rank: usize) -> Result<KimiRankWeightPlan> {
        ensure!(
            rank < self.parallel.ep_world,
            "Kimi rank {rank} outside EP{}",
            self.parallel.ep_world
        );
        let parallel = self.parallel.parallel_config();
        let tp_rank = parallel.tp_rank(rank);
        let ep_rank = parallel.ep_rank(rank);
        let attention_head_range =
            tp_rank * self.parallel.heads_per_tp..(tp_rank + 1) * self.parallel.heads_per_tp;
        let vocab_range =
            tp_rank * self.parallel.vocab_per_tp..(tp_rank + 1) * self.parallel.vocab_per_tp;
        let local_expert_range =
            ep_rank * self.parallel.local_experts..(ep_rank + 1) * self.parallel.local_experts;
        let names = self.rank_tensor_names(rank)?;
        Ok(KimiRankWeightPlan {
            tp_rank,
            ep_rank,
            attention_head_range,
            vocab_range,
            local_expert_range,
            tensor_count: names.len(),
        })
    }

    pub(crate) fn rank_tensor_names(&self, rank: usize) -> Result<Vec<&KimiTensorEntry>> {
        let local_expert_range = self.rank_local_expert_range(rank)?;
        let mut names = Vec::new();
        names.push(&self.token_embedding);
        names.push(&self.final_norm);
        names.push(&self.lm_head);
        for layer in &self.layers {
            push_attention(&mut names, &layer.attention);
            match &layer.kind {
                KimiLayerKindManifest::Dense(mlp) => push_dense_mlp(&mut names, mlp),
                KimiLayerKindManifest::Moe(moe) => {
                    names.push(&moe.router.gate_weight);
                    names.push(&moe.router.e_score_correction_bias);
                    names.push(&moe.shared_experts.gate_proj);
                    names.push(&moe.shared_experts.up_proj);
                    names.push(&moe.shared_experts.down_proj);
                    for expert in &moe.routed_experts {
                        if local_expert_range.contains(&expert.expert_idx) {
                            push_int4_projection(&mut names, &expert.gate_proj);
                            push_int4_projection(&mut names, &expert.up_proj);
                            push_int4_projection(&mut names, &expert.down_proj);
                        }
                    }
                }
            }
        }
        Ok(names)
    }

    pub(crate) fn rank_sliced_load_plan(&self, rank: usize) -> Result<KimiRankSlicedLoadPlan> {
        let entries = self.rank_tensor_load_specs(rank)?;
        let mut by_shard: BTreeMap<String, Vec<KimiTensorLoadSpec>> = BTreeMap::new();
        for entry in entries {
            by_shard.entry(entry.shard.clone()).or_default().push(entry);
        }
        let tensor_count = by_shard.values().map(Vec::len).sum();
        let shards = by_shard
            .into_iter()
            .map(|(shard, tensors)| KimiShardTensorLoadPlan { shard, tensors })
            .collect();
        Ok(KimiRankSlicedLoadPlan {
            rank,
            shards,
            tensor_count,
        })
    }

    fn rank_tensor_load_specs(&self, rank: usize) -> Result<Vec<KimiTensorLoadSpec>> {
        let plan = self.rank_plan(rank)?;
        let local_expert_range = self.rank_local_expert_range(rank)?;
        let mut specs = Vec::with_capacity(plan.tensor_count);
        let vocab_rows = KimiTensorLoadSlice::RowRange {
            start: plan.vocab_range.start,
            end: plan.vocab_range.end,
        };
        push_load_spec(&mut specs, &self.token_embedding, vocab_rows.clone());
        push_load_spec(&mut specs, &self.final_norm, KimiTensorLoadSlice::Full);
        push_load_spec(&mut specs, &self.lm_head, vocab_rows);

        for layer in &self.layers {
            push_attention_load_specs(&mut specs, &layer.attention, &plan);
            match &layer.kind {
                KimiLayerKindManifest::Dense(mlp) => {
                    let rows = rank_shard_rows(
                        KIMI_K2_DENSE_INTERMEDIATE,
                        plan.tp_rank,
                        self.parallel.tp_world,
                    );
                    push_load_spec(&mut specs, &mlp.gate_proj, rows.clone());
                    push_load_spec(&mut specs, &mlp.up_proj, rows.clone());
                    push_load_spec(&mut specs, &mlp.down_proj, row_slice_to_col_slice(rows));
                }
                KimiLayerKindManifest::Moe(moe) => {
                    push_load_spec(
                        &mut specs,
                        &moe.router.gate_weight,
                        KimiTensorLoadSlice::Full,
                    );
                    push_load_spec(
                        &mut specs,
                        &moe.router.e_score_correction_bias,
                        KimiTensorLoadSlice::Full,
                    );
                    let shared_rows = rank_shard_rows(
                        KIMI_K2_EXPERT_INTERMEDIATE,
                        plan.tp_rank,
                        self.parallel.tp_world,
                    );
                    push_load_spec(
                        &mut specs,
                        &moe.shared_experts.gate_proj,
                        shared_rows.clone(),
                    );
                    push_load_spec(&mut specs, &moe.shared_experts.up_proj, shared_rows.clone());
                    push_load_spec(
                        &mut specs,
                        &moe.shared_experts.down_proj,
                        row_slice_to_col_slice(shared_rows),
                    );
                    for expert in &moe.routed_experts {
                        if local_expert_range.contains(&expert.expert_idx) {
                            push_int4_projection_load_specs(&mut specs, &expert.gate_proj);
                            push_int4_projection_load_specs(&mut specs, &expert.up_proj);
                            push_int4_projection_load_specs(&mut specs, &expert.down_proj);
                        }
                    }
                }
            }
        }

        let unique = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == specs.len(),
            "Kimi rank {rank} sliced load plan contains duplicate tensors"
        );
        ensure!(
            specs.len() == plan.tensor_count,
            "Kimi rank {rank} sliced load count {} does not match tensor plan {}",
            specs.len(),
            plan.tensor_count
        );
        Ok(specs)
    }

    pub(crate) fn rank_weight_names(&self, rank: usize) -> Result<KimiRankWeightNames> {
        let plan = self.rank_plan(rank)?;
        let local_expert_range = self.rank_local_expert_range(rank)?;
        let top = KimiTopWeightNames {
            token_embedding: self.token_embedding.name.clone(),
            final_norm: self.final_norm.name.clone(),
            lm_head: self.lm_head.name.clone(),
        };
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let attention = KimiAttentionWeightNames::from_manifest(&layer.attention);
            let kind = match &layer.kind {
                KimiLayerKindManifest::Dense(mlp) => {
                    KimiLayerWeightKindNames::Dense(KimiDenseMlpWeightNames::from_manifest(mlp))
                }
                KimiLayerKindManifest::Moe(moe) => {
                    let routed_experts = moe
                        .routed_experts
                        .iter()
                        .filter(|expert| local_expert_range.contains(&expert.expert_idx))
                        .map(KimiRoutedExpertWeightNames::from_manifest)
                        .collect();
                    KimiLayerWeightKindNames::Moe(KimiMoeLayerWeightNames {
                        router: KimiRouterWeightNames {
                            gate_weight: moe.router.gate_weight.name.clone(),
                            e_score_correction_bias: moe
                                .router
                                .e_score_correction_bias
                                .name
                                .clone(),
                        },
                        shared_experts: KimiSharedExpertWeightNames {
                            gate_proj: moe.shared_experts.gate_proj.name.clone(),
                            up_proj: moe.shared_experts.up_proj.name.clone(),
                            down_proj: moe.shared_experts.down_proj.name.clone(),
                        },
                        routed_experts,
                    })
                }
            };
            layers.push(KimiLayerWeightNames {
                layer_idx: layer.layer_idx,
                attention,
                kind,
            });
        }
        Ok(KimiRankWeightNames {
            rank,
            plan,
            top,
            layers,
        })
    }

    fn rank_local_expert_range(&self, rank: usize) -> Result<Range<usize>> {
        let ep_rank = self.parallel.parallel_config().ep_rank(rank);
        ensure!(
            ep_rank < self.parallel.ep_world,
            "Kimi EP rank {ep_rank} outside EP{}",
            self.parallel.ep_world
        );
        Ok(ep_rank * self.parallel.local_experts..(ep_rank + 1) * self.parallel.local_experts)
    }
}

impl KimiAttentionWeightNames {
    fn from_manifest(manifest: &KimiAttentionManifest) -> Self {
        Self {
            input_layernorm: manifest.input_layernorm.name.clone(),
            q_a_proj: manifest.q_a_proj.name.clone(),
            q_a_layernorm: manifest.q_a_layernorm.name.clone(),
            q_b_proj: manifest.q_b_proj.name.clone(),
            kv_a_proj_with_mqa: manifest.kv_a_proj_with_mqa.name.clone(),
            kv_a_layernorm: manifest.kv_a_layernorm.name.clone(),
            kv_b_proj: manifest.kv_b_proj.name.clone(),
            o_proj: manifest.o_proj.name.clone(),
            post_attention_layernorm: manifest.post_attention_layernorm.name.clone(),
        }
    }
}

impl KimiDenseMlpWeightNames {
    fn from_manifest(manifest: &KimiDenseMlpManifest) -> Self {
        Self {
            gate_proj: manifest.gate_proj.name.clone(),
            up_proj: manifest.up_proj.name.clone(),
            down_proj: manifest.down_proj.name.clone(),
        }
    }
}

impl KimiInt4ProjectionWeightNames {
    fn from_manifest(manifest: &KimiInt4ProjectionManifest) -> Self {
        Self {
            weight_packed: manifest.weight_packed.name.clone(),
            weight_scale: manifest.weight_scale.name.clone(),
        }
    }
}

impl KimiRoutedExpertWeightNames {
    fn from_manifest(manifest: &KimiRoutedExpertManifest) -> Self {
        Self {
            global_expert: manifest.expert_idx,
            gate_proj: KimiInt4ProjectionWeightNames::from_manifest(&manifest.gate_proj),
            up_proj: KimiInt4ProjectionWeightNames::from_manifest(&manifest.up_proj),
            down_proj: KimiInt4ProjectionWeightNames::from_manifest(&manifest.down_proj),
        }
    }
}

fn push_load_spec(
    out: &mut Vec<KimiTensorLoadSpec>,
    entry: &KimiTensorEntry,
    slice: KimiTensorLoadSlice,
) {
    out.push(KimiTensorLoadSpec {
        name: entry.name.clone(),
        shard: entry.shard.clone(),
        slice,
    });
}

fn push_attention_load_specs(
    out: &mut Vec<KimiTensorLoadSpec>,
    attention: &KimiAttentionManifest,
    plan: &KimiRankWeightPlan,
) {
    push_load_spec(out, &attention.input_layernorm, KimiTensorLoadSlice::Full);
    push_load_spec(out, &attention.q_a_proj, KimiTensorLoadSlice::Full);
    push_load_spec(out, &attention.q_a_layernorm, KimiTensorLoadSlice::Full);
    push_load_spec(
        out,
        &attention.q_b_proj,
        KimiTensorLoadSlice::RowRange {
            start: plan.attention_head_range.start * KIMI_K2_Q_HEAD_DIM,
            end: plan.attention_head_range.end * KIMI_K2_Q_HEAD_DIM,
        },
    );
    push_load_spec(
        out,
        &attention.kv_a_proj_with_mqa,
        KimiTensorLoadSlice::Full,
    );
    push_load_spec(out, &attention.kv_a_layernorm, KimiTensorLoadSlice::Full);
    push_load_spec(
        out,
        &attention.kv_b_proj,
        KimiTensorLoadSlice::RowRange {
            start: plan.attention_head_range.start
                * (KIMI_K2_QK_NOPE_HEAD_DIM + KIMI_K2_V_HEAD_DIM),
            end: plan.attention_head_range.end * (KIMI_K2_QK_NOPE_HEAD_DIM + KIMI_K2_V_HEAD_DIM),
        },
    );
    push_load_spec(
        out,
        &attention.o_proj,
        KimiTensorLoadSlice::ColRange {
            start: plan.attention_head_range.start * KIMI_K2_V_HEAD_DIM,
            end: plan.attention_head_range.end * KIMI_K2_V_HEAD_DIM,
        },
    );
    push_load_spec(
        out,
        &attention.post_attention_layernorm,
        KimiTensorLoadSlice::Full,
    );
}

fn push_int4_projection_load_specs(
    out: &mut Vec<KimiTensorLoadSpec>,
    projection: &KimiInt4ProjectionManifest,
) {
    push_load_spec(out, &projection.weight_packed, KimiTensorLoadSlice::Full);
    push_load_spec(out, &projection.weight_scale, KimiTensorLoadSlice::Full);
}

fn rank_shard_rows(total_rows: usize, rank: usize, world: usize) -> KimiTensorLoadSlice {
    debug_assert_eq!(total_rows % world, 0);
    let rows_per_rank = total_rows / world;
    KimiTensorLoadSlice::RowRange {
        start: rank * rows_per_rank,
        end: (rank + 1) * rows_per_rank,
    }
}

fn row_slice_to_col_slice(slice: KimiTensorLoadSlice) -> KimiTensorLoadSlice {
    match slice {
        KimiTensorLoadSlice::RowRange { start, end } => {
            KimiTensorLoadSlice::ColRange { start, end }
        }
        KimiTensorLoadSlice::Full | KimiTensorLoadSlice::ColRange { .. } => slice,
    }
}

fn attention_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<KimiAttentionManifest> {
    Ok(KimiAttentionManifest {
        input_layernorm: layer_tensor(tensors, layer_idx, "input_layernorm.weight")?,
        q_a_proj: layer_tensor(tensors, layer_idx, "self_attn.q_a_proj.weight")?,
        q_a_layernorm: layer_tensor(tensors, layer_idx, "self_attn.q_a_layernorm.weight")?,
        q_b_proj: layer_tensor(tensors, layer_idx, "self_attn.q_b_proj.weight")?,
        kv_a_proj_with_mqa: layer_tensor(
            tensors,
            layer_idx,
            "self_attn.kv_a_proj_with_mqa.weight",
        )?,
        kv_a_layernorm: layer_tensor(tensors, layer_idx, "self_attn.kv_a_layernorm.weight")?,
        kv_b_proj: layer_tensor(tensors, layer_idx, "self_attn.kv_b_proj.weight")?,
        o_proj: layer_tensor(tensors, layer_idx, "self_attn.o_proj.weight")?,
        post_attention_layernorm: layer_tensor(
            tensors,
            layer_idx,
            "post_attention_layernorm.weight",
        )?,
    })
}

fn dense_mlp_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<KimiDenseMlpManifest> {
    Ok(KimiDenseMlpManifest {
        gate_proj: layer_tensor(tensors, layer_idx, "mlp.gate_proj.weight")?,
        up_proj: layer_tensor(tensors, layer_idx, "mlp.up_proj.weight")?,
        down_proj: layer_tensor(tensors, layer_idx, "mlp.down_proj.weight")?,
    })
}

fn moe_layer_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
) -> Result<KimiMoeLayerManifest> {
    let mut routed_experts = Vec::with_capacity(KIMI_K2_ROUTED_EXPERTS);
    for expert_idx in 0..KIMI_K2_ROUTED_EXPERTS {
        routed_experts.push(routed_expert_manifest(tensors, layer_idx, expert_idx)?);
    }
    Ok(KimiMoeLayerManifest {
        router: KimiRouterManifest {
            gate_weight: layer_tensor(tensors, layer_idx, "mlp.gate.weight")?,
            e_score_correction_bias: layer_tensor(
                tensors,
                layer_idx,
                "mlp.gate.e_score_correction_bias",
            )?,
        },
        shared_experts: KimiSharedExpertManifest {
            gate_proj: layer_tensor(tensors, layer_idx, "mlp.shared_experts.gate_proj.weight")?,
            up_proj: layer_tensor(tensors, layer_idx, "mlp.shared_experts.up_proj.weight")?,
            down_proj: layer_tensor(tensors, layer_idx, "mlp.shared_experts.down_proj.weight")?,
        },
        routed_experts,
    })
}

fn routed_expert_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
    expert_idx: usize,
) -> Result<KimiRoutedExpertManifest> {
    Ok(KimiRoutedExpertManifest {
        expert_idx,
        gate_proj: int4_projection_manifest(tensors, layer_idx, expert_idx, "gate_proj")?,
        up_proj: int4_projection_manifest(tensors, layer_idx, expert_idx, "up_proj")?,
        down_proj: int4_projection_manifest(tensors, layer_idx, expert_idx, "down_proj")?,
    })
}

fn int4_projection_manifest(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
    expert_idx: usize,
    projection: &str,
) -> Result<KimiInt4ProjectionManifest> {
    let prefix = format!("mlp.experts.{expert_idx}.{projection}");
    Ok(KimiInt4ProjectionManifest {
        weight_packed: layer_tensor(tensors, layer_idx, &format!("{prefix}.weight_packed"))?,
        weight_scale: layer_tensor(tensors, layer_idx, &format!("{prefix}.weight_scale"))?,
    })
}

fn layer_tensor(
    tensors: &BTreeMap<&str, &str>,
    layer_idx: usize,
    suffix: &str,
) -> Result<KimiTensorEntry> {
    tensor(
        tensors,
        &format!("language_model.model.layers.{layer_idx}.{suffix}"),
    )
}

fn tensor(tensors: &BTreeMap<&str, &str>, name: &str) -> Result<KimiTensorEntry> {
    let shard = tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Kimi text manifest missing tensor {name}"))?;
    Ok(KimiTensorEntry {
        name: name.to_owned(),
        shard: (*shard).to_owned(),
    })
}

fn push_attention<'a>(out: &mut Vec<&'a KimiTensorEntry>, attention: &'a KimiAttentionManifest) {
    out.push(&attention.input_layernorm);
    out.push(&attention.q_a_proj);
    out.push(&attention.q_a_layernorm);
    out.push(&attention.q_b_proj);
    out.push(&attention.kv_a_proj_with_mqa);
    out.push(&attention.kv_a_layernorm);
    out.push(&attention.kv_b_proj);
    out.push(&attention.o_proj);
    out.push(&attention.post_attention_layernorm);
}

fn push_dense_mlp<'a>(out: &mut Vec<&'a KimiTensorEntry>, mlp: &'a KimiDenseMlpManifest) {
    out.push(&mlp.gate_proj);
    out.push(&mlp.up_proj);
    out.push(&mlp.down_proj);
}

fn push_int4_projection<'a>(
    out: &mut Vec<&'a KimiTensorEntry>,
    projection: &'a KimiInt4ProjectionManifest,
) {
    out.push(&projection.weight_packed);
    out.push(&projection.weight_scale);
}
