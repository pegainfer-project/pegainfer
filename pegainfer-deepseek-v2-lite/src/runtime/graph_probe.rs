use std::env;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Dsv2LiteAttentionConfig;
use pegainfer_kernels::ops::Dsv2LiteRopeScalingConfig;
use pegainfer_kernels::ops::dsv2_lite_decode_attention_into;
use pegainfer_kernels::ops::dsv2_lite_kv_norm_into;

use super::DeepSeekV2LiteEp2Generator;
use super::backend::EpBackendRuntime;
use super::moe::FixedTopologyMoeScratch;
use super::types::DecodeGraphBlocker;
use super::types::FullDecodeGraphProbeReport;
use super::types::GenerationStats;
use crate::attribution::DecodeAttributionProfile;
use crate::device::activate;
use crate::device::activate_graph_capture;
use crate::device::graph_capture_activation_guard;
use crate::host_ops::DecodeCache;
use crate::model::DenseMlpForwardScratch;
use crate::model::MlpWeights;
use crate::model::dense_mlp_forward_preallocated_into;
use crate::nccl_backend::RawCudaGraph;
use crate::nccl_backend::begin_capture;
use crate::nccl_backend::end_capture;
use crate::nccl_backend::launch_graph_pair_and_sync;

const GRAPH_PROBE_REPLAY_COUNT: usize = 8;
const GRAPH_PROBE_MAX_SEQ_LEN: usize = 4096;

struct DecodeGraphProbeState {
    input_token: CudaSlice<u32>,
    hidden_a: HiddenStates,
    hidden_b: HiddenStates,
    final_normed: HiddenStates,
    logits: HiddenStates,
    sample_values: CudaSlice<bf16>,
    sample_out: CudaSlice<i32>,
    layers: Vec<LayerGraphScratch>,
    max_seq_len: usize,
}

struct LayerGraphScratch {
    key_cache: CudaSlice<f32>,
    value_cache: CudaSlice<f32>,
    normed: HiddenStates,
    q: HiddenStates,
    kv_a: HiddenStates,
    compressed: HiddenStates,
    kv_b: HiddenStates,
    attn: HiddenStates,
    attn_projected: HiddenStates,
    after_attn: HiddenStates,
    ffn_norm: HiddenStates,
    ffn_out: HiddenStates,
    mlp: LayerMlpGraphScratch,
}

// Per-layer probe scratch, one at a time; the size asymmetry costs nothing.
#[allow(clippy::large_enum_variant)]
enum LayerMlpGraphScratch {
    Dense(DenseMlpForwardScratch),
    Moe(FixedTopologyMoeScratch),
}

impl DeepSeekV2LiteEp2Generator {
    pub(super) fn full_decode_graph_probe_report(
        &mut self,
        requested: bool,
        prompt_tokens: Option<&[u32]>,
        output_len: usize,
    ) -> Result<FullDecodeGraphProbeReport> {
        if !requested {
            return Ok(FullDecodeGraphProbeReport {
                requested: false,
                captured: false,
                instantiated: false,
                replayed: false,
                verified: false,
                replay_count: 0,
                verified_replay_count: 0,
                failure_stage: "not_requested",
                failure_summary: None,
                blockers: Vec::new(),
                capture_mode: "thread_local",
            });
        }
        let Some(prompt_tokens) = prompt_tokens else {
            bail!("DeepSeek-V2-Lite full decode graph probe requires prompt tokens");
        };
        ensure!(
            output_len >= 2,
            "DeepSeek-V2-Lite full decode graph probe needs at least two output tokens, got {output_len}"
        );
        ensure!(
            matches!(self.backend, EpBackendRuntime::Nccl(_)),
            "DeepSeek-V2-Lite --full-decode-graph-probe requires PEGAINFER_DSV2_LITE_EP_BACKEND=nccl"
        );

        let mut report = FullDecodeGraphProbeReport {
            requested: true,
            captured: false,
            instantiated: false,
            replayed: false,
            verified: false,
            replay_count: 0,
            verified_replay_count: 0,
            failure_stage: "not_started",
            failure_summary: None,
            blockers: Vec::new(),
            capture_mode: "thread_local",
        };

        if let Err(err) = self.full_decode_graph_probe_inner(prompt_tokens, output_len, &mut report)
        {
            if report.failure_summary.is_none() {
                report.failure_summary = Some(format!("{err:#}"));
            }
            if report.blockers.is_empty() {
                report
                    .blockers
                    .push(blocker_for_stage(report.failure_stage));
            }
        }
        Ok(report)
    }

    fn full_decode_graph_probe_inner(
        &mut self,
        prompt_tokens: &[u32],
        output_len: usize,
        report: &mut FullDecodeGraphProbeReport,
    ) -> Result<()> {
        let max_seq_len = prompt_tokens.len() + output_len;
        if max_seq_len > GRAPH_PROBE_MAX_SEQ_LEN {
            report.failure_stage = "preflight_blocked";
            report.failure_summary = Some(format!(
                "full decode graph probe skipped before CUDA stream capture because max_seq_len={max_seq_len} exceeds the fixed-topology probe kernel limit {GRAPH_PROBE_MAX_SEQ_LEN}"
            ));
            report.blockers.push(DecodeGraphBlocker {
                id: "decode_graph_probe_shape_limit",
                source: "runtime/graph_probe.rs::full_decode_graph_probe_inner",
                reason: "the probe-only device attention kernel uses a fixed shared-memory score buffer and currently covers only short retained decode shapes",
            });
            bail!(
                "DeepSeek-V2-Lite full decode graph probe shape exceeds fixed-topology probe limit"
            );
        }
        trace_graph_probe("start full decode graph probe");
        let mut stats = GenerationStats {
            device_ordinals: self.device_ordinals.clone(),
            ep_backend: self.backend.kind().as_str().to_string(),
            ep_size: 2,
            prompt_tokens: prompt_tokens.len(),
            ..GenerationStats::default()
        };
        let mut attribution = DecodeAttributionProfile::disabled();

        let mut prefill_cache = DecodeCache::new(&self.config);
        trace_graph_probe("run probe prefill");
        let first_token = self.prefill_next_token(
            prompt_tokens,
            &mut prefill_cache,
            &mut stats,
            &mut attribution,
        )?;
        let position = prompt_tokens.len();

        let mut retained_cache = prefill_cache.clone_for_graph_probe();
        trace_graph_probe("run retained eager decode oracle");
        let retained_next = self.decode_next_token(
            first_token,
            position,
            &mut retained_cache,
            &mut stats,
            &mut attribution,
            1,
        )?;

        trace_graph_probe("allocate graph-safe reference state");
        let mut reference = DecodeGraphProbeState::new(self, max_seq_len, &prefill_cache)
            .context("initialize eager graph-safe probe state")?;
        reference.set_input_token(&self.rank0.ctx, first_token)?;
        trace_graph_probe("run graph-safe eager reference step");
        if let Err(err) = self.decode_graph_probe_step_kernels(position, &mut reference) {
            report.failure_stage = "reference_failed";
            report.failure_summary = Some(format!("{err:#}"));
            return Err(err).context("run eager graph-safe decode probe step");
        }
        let reference_token = reference.sampled_token(&self.rank0.ctx)?;
        if reference_token != retained_next {
            report.failure_stage = "reference_mismatch";
            let message = format!(
                "graph-safe eager decode step token mismatch: graph_safe={reference_token}, retained_eager={retained_next}"
            );
            report.failure_summary = Some(message.clone());
            bail!("{message}");
        }
        trace_graph_probe("graph-safe eager reference matched retained oracle");

        trace_graph_probe("allocate captured graph probe state");
        let mut captured = DecodeGraphProbeState::new(self, max_seq_len, &prefill_cache)
            .context("initialize captured graph probe state")?;
        captured.set_input_token(&self.rank0.ctx, first_token)?;
        trace_graph_probe("capture graph-safe decode step");
        let (graph0, graph1) = match self.capture_decode_graph_step(position, &mut captured, report)
        {
            Ok(graphs) => graphs,
            Err(err) => {
                report.failure_stage = if report.captured {
                    "instantiate_failed"
                } else {
                    "capture_failed"
                };
                report.failure_summary = Some(format!("{err:#}"));
                return Err(err);
            }
        };
        trace_graph_probe("capture and instantiate complete");

        let replayed_token = self
            .replay_and_verify_decode_graph(
                &graph0,
                &graph1,
                &mut captured,
                reference_token,
                GRAPH_PROBE_REPLAY_COUNT,
                report,
            )
            .context("replay and verify captured DeepSeek-V2-Lite decode graph")?;
        report.verified = true;
        trace_graph_probe("graph replay verified");
        report.failure_stage = "none";
        report.failure_summary = Some(format!(
            "covered_path=nccl_fixed_topology_decode_step prompt_tokens={} max_seq_len={max_seq_len} input_token={first_token} verified_token={replayed_token} replay_count={} verified_replay_count={}",
            prompt_tokens.len(),
            report.replay_count,
            report.verified_replay_count
        ));
        Ok(())
    }

    fn replay_and_verify_decode_graph(
        &self,
        graph0: &RawCudaGraph,
        graph1: &RawCudaGraph,
        state: &mut DecodeGraphProbeState,
        reference_token: u32,
        replay_count: usize,
        report: &mut FullDecodeGraphProbeReport,
    ) -> Result<u32> {
        ensure!(
            replay_count > 0,
            "DeepSeek-V2-Lite decode graph probe replay_count must be positive"
        );
        let mut last_token = None;
        for replay_idx in 0..replay_count {
            state
                .reset_sample_token_sentinel(&self.rank0.ctx)
                .context("reset graph probe sampled token sentinel before replay")?;
            if let Err(err) = launch_graph_pair_and_sync(
                graph0,
                graph1,
                self.rank0.ctx.device_ordinal,
                self.rank0.ctx.stream.cu_stream(),
                self.rank1.ctx.device_ordinal,
                self.rank1.ctx.stream.cu_stream(),
            )
            .context("launch paired DeepSeek-V2-Lite decode CUDA Graphs")
            {
                report.failure_stage = "replay_failed";
                report.failure_summary = Some(format!("{err:#}"));
                return Err(err);
            }
            report.replayed = true;
            report.replay_count += 1;
            trace_graph_probe("graph replay launched");

            let replayed_token = match state.sampled_token(&self.rank0.ctx) {
                Ok(token) => token,
                Err(err) => {
                    report.failure_stage = "verification_failed";
                    let message = format!(
                        "captured decode graph did not rewrite sampled token at replay {replay_idx}: {err:#}"
                    );
                    report.failure_summary = Some(message.clone());
                    return Err(err).context(message);
                }
            };
            if replayed_token != reference_token {
                report.failure_stage = "verification_failed";
                let message = format!(
                    "captured decode graph token mismatch at replay {replay_idx}: replayed={replayed_token}, reference={reference_token}"
                );
                report.failure_summary = Some(message.clone());
                bail!("{message}");
            }
            report.verified_replay_count += 1;
            last_token = Some(replayed_token);
        }
        last_token.ok_or_else(|| anyhow::anyhow!("decode graph probe replay produced no token"))
    }

    fn capture_decode_graph_step(
        &self,
        position: usize,
        state: &mut DecodeGraphProbeState,
        report: &mut FullDecodeGraphProbeReport,
    ) -> Result<(RawCudaGraph, RawCudaGraph)> {
        let mut rank0_capture_started = false;
        let mut rank1_capture_started = false;
        let capture_result = (|| -> Result<(RawCudaGraph, RawCudaGraph)> {
            let _activation_guard = graph_capture_activation_guard();
            activate_graph_capture(&self.rank0.ctx)?;
            begin_capture(self.rank0.ctx.stream.cu_stream(), "rank0")?;
            rank0_capture_started = true;
            activate_graph_capture(&self.rank1.ctx)?;
            begin_capture(self.rank1.ctx.stream.cu_stream(), "rank1")?;
            rank1_capture_started = true;

            self.decode_graph_probe_step_kernels(position, state)?;

            activate_graph_capture(&self.rank0.ctx)?;
            let captured0 = end_capture(self.rank0.ctx.stream.cu_stream(), "rank0")?;
            rank0_capture_started = false;
            activate_graph_capture(&self.rank1.ctx)?;
            let captured1 = end_capture(self.rank1.ctx.stream.cu_stream(), "rank1")?;
            rank1_capture_started = false;
            report.captured = true;

            activate_graph_capture(&self.rank0.ctx)?;
            let graph0 = captured0.instantiate("rank0")?;
            activate_graph_capture(&self.rank1.ctx)?;
            let graph1 = captured1.instantiate("rank1")?;
            report.instantiated = true;
            Ok((graph0, graph1))
        })();

        if capture_result.is_err() {
            cleanup_capture(&self.rank0.ctx, rank0_capture_started);
            cleanup_capture(&self.rank1.ctx, rank1_capture_started);
        }
        capture_result
    }

    fn decode_graph_probe_step_kernels(
        &self,
        position: usize,
        state: &mut DecodeGraphProbeState,
    ) -> Result<()> {
        activate(&self.rank0.ctx)?;
        ops::embedding_batch(
            &self.rank0.ctx,
            &self.rank0.embed_tokens,
            &state.input_token,
            &mut state.hidden_a,
        )?;

        let mut current_is_a = true;
        for layer_idx in 0..self.rank0.layers.len() {
            let layer_scratch = &mut state.layers[layer_idx];
            if current_is_a {
                self.decode_graph_layer_into(
                    layer_idx,
                    &state.hidden_a,
                    position,
                    layer_scratch,
                    &mut state.hidden_b,
                    state.max_seq_len,
                )?;
            } else {
                self.decode_graph_layer_into(
                    layer_idx,
                    &state.hidden_b,
                    position,
                    layer_scratch,
                    &mut state.hidden_a,
                    state.max_seq_len,
                )?;
            }
            current_is_a = !current_is_a;
        }

        let final_hidden = if current_is_a {
            &state.hidden_a
        } else {
            &state.hidden_b
        };
        dsv2_lite_kv_norm_into(
            &self.rank0.ctx,
            final_hidden,
            &self.rank0.norm_device.data,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            &mut state.final_normed,
        )?;
        ops::gemm_graphsafe_into_checked(
            &self.rank0.ctx,
            &self.rank0.lm_head,
            &state.final_normed,
            &mut state.logits,
        )?;
        ops::argmax_batch_bf16_into(
            &self.rank0.ctx,
            &state.logits,
            &mut state.sample_values,
            &mut state.sample_out,
        )
    }

    fn decode_graph_layer_into(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        position: usize,
        scratch: &mut LayerGraphScratch,
        out: &mut HiddenStates,
        max_seq_len: usize,
    ) -> Result<()> {
        let layer = &self.rank0.layers[layer_idx];
        dsv2_lite_kv_norm_into(
            &self.rank0.ctx,
            input,
            &layer.input_layernorm_device.data,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            &mut scratch.normed,
        )?;
        ops::gemm_graphsafe_into_checked(
            &self.rank0.ctx,
            &layer.attention.q_proj,
            &scratch.normed,
            &mut scratch.q,
        )?;
        ops::gemm_graphsafe_into_checked(
            &self.rank0.ctx,
            &layer.attention.kv_a_proj,
            &scratch.normed,
            &mut scratch.kv_a,
        )?;
        dsv2_lite_kv_norm_into(
            &self.rank0.ctx,
            &scratch.kv_a,
            &layer.attention.kv_a_norm_device.data,
            self.config.kv_lora_rank,
            self.config.rms_norm_eps,
            &mut scratch.compressed,
        )?;
        ops::gemm_graphsafe_into_checked(
            &self.rank0.ctx,
            &layer.attention.kv_b_proj,
            &scratch.compressed,
            &mut scratch.kv_b,
        )?;
        dsv2_lite_decode_attention_into(
            &self.rank0.ctx,
            self.attention_config(max_seq_len),
            &scratch.q,
            &scratch.kv_a,
            &scratch.kv_b,
            position,
            &mut scratch.key_cache,
            &mut scratch.value_cache,
            &mut scratch.attn,
        )?;
        ops::gemm_graphsafe_into_checked(
            &self.rank0.ctx,
            &layer.attention.o_proj,
            &scratch.attn,
            &mut scratch.attn_projected,
        )?;
        ops::add_batch_into(
            &self.rank0.ctx,
            input,
            &scratch.attn_projected,
            &mut scratch.after_attn,
        )?;
        dsv2_lite_kv_norm_into(
            &self.rank0.ctx,
            &scratch.after_attn,
            &layer.post_attention_layernorm_device.data,
            self.config.hidden_size,
            self.config.rms_norm_eps,
            &mut scratch.ffn_norm,
        )?;

        match (&layer.mlp, &mut scratch.mlp) {
            (MlpWeights::Dense(dense), LayerMlpGraphScratch::Dense(dense_scratch)) => {
                dense_mlp_forward_preallocated_into(
                    &self.rank0.ctx,
                    dense,
                    &scratch.ffn_norm,
                    dense_scratch,
                )?;
                ops::add_batch_into(
                    &self.rank0.ctx,
                    &scratch.after_attn,
                    &dense_scratch.out,
                    out,
                )
            }
            (MlpWeights::Moe(moe), LayerMlpGraphScratch::Moe(moe_scratch)) => {
                let EpBackendRuntime::Nccl(nccl) = &self.backend else {
                    bail!("fixed-topology decode graph probe requires NCCL backend");
                };
                self.moe_forward_nccl_fixed_topology_preallocated_into(
                    nccl,
                    layer_idx,
                    &scratch.ffn_norm,
                    moe,
                    moe_scratch,
                    &mut scratch.ffn_out,
                )?;
                ops::add_batch_into(&self.rank0.ctx, &scratch.after_attn, &scratch.ffn_out, out)
            }
            (MlpWeights::Dense(_), LayerMlpGraphScratch::Moe(_))
            | (MlpWeights::Moe(_), LayerMlpGraphScratch::Dense(_)) => {
                bail!("DeepSeek-V2-Lite graph probe MLP scratch kind mismatch")
            }
        }
    }

    fn attention_config(&self, max_seq_len: usize) -> Dsv2LiteAttentionConfig {
        Dsv2LiteAttentionConfig {
            num_heads: self.config.num_attention_heads,
            qk_nope_head_dim: self.config.qk_nope_head_dim,
            qk_rope_head_dim: self.config.qk_rope_head_dim,
            v_head_dim: self.config.v_head_dim,
            kv_lora_rank: self.config.kv_lora_rank,
            max_seq_len,
            rms_norm_eps: self.config.rms_norm_eps,
            rope_theta: self.config.rope_theta,
            rope_scaling: self
                .config
                .rope_scaling
                .as_ref()
                .map(|rope| Dsv2LiteRopeScalingConfig {
                    factor: rope.factor,
                    mscale: rope.mscale,
                    mscale_all_dim: rope.mscale_all_dim,
                    beta_fast: rope.beta_fast as f32,
                    beta_slow: rope.beta_slow as f32,
                    original_max_position_embeddings: rope.original_max_position_embeddings,
                }),
        }
    }
}

fn trace_graph_probe(stage: &str) {
    if env::var_os("PEGAINFER_DSV2_LITE_GRAPH_PROBE_TRACE").is_some() {
        eprintln!("[dsv2-lite-graph-probe] {stage}");
    }
}

impl DecodeGraphProbeState {
    fn new(
        generator: &DeepSeekV2LiteEp2Generator,
        max_seq_len: usize,
        cache: &DecodeCache,
    ) -> Result<Self> {
        ensure!(
            cache.layers.len() == generator.rank0.layers.len(),
            "probe cache layer count mismatch: cache={}, layers={}",
            cache.layers.len(),
            generator.rank0.layers.len()
        );
        let ctx = &generator.rank0.ctx;
        activate(ctx)?;
        let input_token = ctx.stream.alloc_zeros::<u32>(1)?;
        let hidden_a = HiddenStates::zeros(ctx, generator.config.hidden_size, 1)?;
        let hidden_b = HiddenStates::zeros(ctx, generator.config.hidden_size, 1)?;
        let final_normed = HiddenStates::zeros(ctx, generator.config.hidden_size, 1)?;
        let logits = HiddenStates::zeros(ctx, generator.config.vocab_size, 1)?;
        let sample_values = ctx.stream.alloc_zeros::<bf16>(1)?;
        let sample_out = ctx.stream.alloc_zeros::<i32>(1)?;

        if let EpBackendRuntime::Nccl(nccl) = &generator.backend {
            nccl.prepare_graph_shape(ctx, &generator.rank1.ctx, generator.config.hidden_size, 1)?;
        }

        let mut layers = Vec::with_capacity(generator.rank0.layers.len());
        for (layer_idx, layer) in generator.rank0.layers.iter().enumerate() {
            layers.push(LayerGraphScratch::new(
                generator,
                layer_idx,
                layer,
                max_seq_len,
                &cache.layers[layer_idx],
            )?);
        }
        generator.rank0.ctx.sync()?;
        generator.rank1.ctx.sync()?;

        Ok(Self {
            input_token,
            hidden_a,
            hidden_b,
            final_normed,
            logits,
            sample_values,
            sample_out,
            layers,
            max_seq_len,
        })
    }

    fn set_input_token(&mut self, ctx: &DeviceContext, token: u32) -> Result<()> {
        activate(ctx)?;
        ctx.stream
            .memcpy_htod(&[token], &mut self.input_token)
            .context("upload graph probe input token")?;
        ctx.sync()
    }

    fn sampled_token(&self, ctx: &DeviceContext) -> Result<u32> {
        activate(ctx)?;
        let out = ctx
            .stream
            .clone_dtoh(&self.sample_out)
            .context("read graph probe sampled token")?;
        ctx.sync()?;
        let token = *out
            .first()
            .ok_or_else(|| anyhow::anyhow!("graph probe sampled token buffer is empty"))?;
        ensure!(token >= 0, "graph probe sampled token is negative: {token}");
        Ok(token as u32)
    }

    fn reset_sample_token_sentinel(&mut self, ctx: &DeviceContext) -> Result<()> {
        activate(ctx)?;
        ctx.stream
            .memcpy_htod(&[-1i32], &mut self.sample_out)
            .context("reset graph probe sampled token sentinel")?;
        ctx.sync()
    }
}

impl LayerGraphScratch {
    fn new(
        generator: &DeepSeekV2LiteEp2Generator,
        layer_idx: usize,
        layer: &crate::model::LayerWeights,
        max_seq_len: usize,
        cache: &crate::host_ops::LayerCache,
    ) -> Result<Self> {
        let cfg = &generator.config;
        let ctx = &generator.rank0.ctx;
        activate(ctx)?;
        let key_elems = max_seq_len * cfg.num_attention_heads * cfg.query_head_dim();
        let value_elems = max_seq_len * cfg.num_attention_heads * cfg.v_head_dim;
        let mut key_cache = ctx.stream.alloc_zeros::<f32>(key_elems)?;
        let mut value_cache = ctx.stream.alloc_zeros::<f32>(value_elems)?;
        ensure!(
            cache.keys().len() <= key_elems && cache.values().len() <= value_elems,
            "probe cache layer {layer_idx} exceeds device cache capacity"
        );
        if !cache.keys().is_empty() {
            let mut dst = key_cache.slice_mut(0..cache.keys().len());
            ctx.stream
                .memcpy_htod(cache.keys(), &mut dst)
                .context("upload graph probe key cache")?;
        }
        if !cache.values().is_empty() {
            let mut dst = value_cache.slice_mut(0..cache.values().len());
            ctx.stream
                .memcpy_htod(cache.values(), &mut dst)
                .context("upload graph probe value cache")?;
        }

        let normed = HiddenStates::zeros(ctx, cfg.hidden_size, 1)?;
        let q = HiddenStates::zeros(ctx, cfg.q_proj_rows(), 1)?;
        let kv_a = HiddenStates::zeros(ctx, cfg.kv_a_proj_rows(), 1)?;
        let compressed = HiddenStates::zeros(ctx, cfg.kv_lora_rank, 1)?;
        let kv_b = HiddenStates::zeros(ctx, cfg.kv_b_proj_rows(), 1)?;
        let attn = HiddenStates::zeros(ctx, cfg.o_proj_cols(), 1)?;
        let attn_projected = HiddenStates::zeros(ctx, cfg.hidden_size, 1)?;
        let after_attn = HiddenStates::zeros(ctx, cfg.hidden_size, 1)?;
        let ffn_norm = HiddenStates::zeros(ctx, cfg.hidden_size, 1)?;
        let ffn_out = HiddenStates::zeros(ctx, cfg.hidden_size, 1)?;
        let mlp = match &layer.mlp {
            MlpWeights::Dense(dense) => {
                LayerMlpGraphScratch::Dense(DenseMlpForwardScratch::new(ctx, dense, 1)?)
            }
            MlpWeights::Moe(moe) => LayerMlpGraphScratch::Moe(FixedTopologyMoeScratch::new(
                generator, layer_idx, moe, 1,
            )?),
        };

        Ok(Self {
            key_cache,
            value_cache,
            normed,
            q,
            kv_a,
            compressed,
            kv_b,
            attn,
            attn_projected,
            after_attn,
            ffn_norm,
            ffn_out,
            mlp,
        })
    }
}

fn cleanup_capture(ctx: &DeviceContext, capture_started: bool) {
    if capture_started {
        let _ = activate(ctx);
        let _ = end_capture(ctx.stream.cu_stream(), "cleanup");
    }
}

fn blocker_for_stage(stage: &'static str) -> DecodeGraphBlocker {
    match stage {
        "capture_failed" | "instantiate_failed" | "replay_failed" => DecodeGraphBlocker {
            id: "multi_rank_capture_orchestration",
            source: "runtime/graph_probe.rs::capture_decode_graph_step",
            reason: "the fixed-topology decode step reached CUDA stream capture, but paired rank0/rank1 NCCL graph capture or replay failed",
        },
        "verification_failed" | "reference_mismatch" => DecodeGraphBlocker {
            id: "decode_graph_probe_verification",
            source: "runtime/graph_probe.rs::full_decode_graph_probe_inner",
            reason: "the graph-safe probe path did not match the retained eager decode oracle for the covered step",
        },
        _ => DecodeGraphBlocker {
            id: "full_decode_graph_probe_runtime",
            source: "runtime/graph_probe.rs::full_decode_graph_probe_inner",
            reason: "the fixed-topology full decode graph probe failed before verification",
        },
    }
}
