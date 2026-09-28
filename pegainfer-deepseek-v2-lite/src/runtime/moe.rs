use std::collections::BTreeMap;
use std::env;
use std::sync::LazyLock;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::tensor::HiddenStatesRef;
use pegainfer_kernels::ops::DSV2_ROUTED_MOE_MAX_ROWS;
use pegainfer_kernels::ops::Dsv2ExpertPointerTable;
use pegainfer_kernels::ops::Dsv2LiteRouterOutput;
use pegainfer_kernels::ops::Dsv2RoutedMoeScratch;
use pegainfer_kernels::ops::dsv2_lite_router_logits_into;
use pegainfer_kernels::ops::dsv2_lite_router_softmax_topk_into;
use pegainfer_kernels::ops::dsv2_lite_router_softmax_topk_ref_into;

use super::DeepSeekV2LiteEp2Generator;
use super::backend::EpBackendKind;
use super::backend::EpBackendRuntime;
use super::routing::MoeRouteEntry;
use super::routing::MoeRoutePlan;
use crate::attribution::DecodeAttributionProfile;
use crate::device::activate;
use crate::host_ops::gate_logits_host;
use crate::host_ops::hidden_from_bf16_host;
use crate::host_ops::hidden_from_f32_host;
use crate::host_ops::hidden_to_bf16;
use crate::host_ops::hidden_to_f32;
use crate::host_ops::topk_softmax_routes;
use crate::model::DenseMlpForwardScratch;
use crate::model::ExpertMlp;
use crate::model::MoeMlp;
use crate::model::dense_mlp_forward;
use crate::model::dense_mlp_forward_per_token;
use crate::model::dense_mlp_forward_preallocated_into;
use crate::model::dense_mlp_forward_preallocated_ref_into;
use crate::nccl_backend::NaiveNcclEp2Backend;

fn parse_rollback_value(
    name: &str,
    raw: Option<&str>,
    optimized: &str,
    rollback: &str,
) -> std::result::Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(false),
        Some(value) if value == optimized => Ok(false),
        Some(value) if value == rollback => Ok(true),
        Some(value) => Err(format!(
            "{name} must be `{optimized}` or `{rollback}`, got `{value}`"
        )),
    }
}

fn load_rollback_value(
    name: &str,
    optimized: &str,
    rollback: &str,
) -> std::result::Result<bool, String> {
    match env::var(name) {
        Ok(value) => parse_rollback_value(name, Some(&value), optimized, rollback),
        Err(env::VarError::NotPresent) => Ok(false),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid Unicode text")),
    }
}

fn rollback_enabled(value: &std::result::Result<bool, String>) -> Result<bool> {
    value
        .as_ref()
        .copied()
        .map_err(|message| anyhow::anyhow!("{message}"))
}

static HOST_STAGED_SERIAL: LazyLock<std::result::Result<bool, String>> = LazyLock::new(|| {
    load_rollback_value(
        "PEGAINFER_DSV2_LITE_HOST_STAGED_EXPERT_BATCH",
        "batched",
        "serial",
    )
});
static NCCL_SERIAL: LazyLock<std::result::Result<bool, String>> = LazyLock::new(|| {
    load_rollback_value("PEGAINFER_DSV2_LITE_NCCL_EXPERT_BATCH", "grouped", "serial")
});
static NCCL_HOST_ROUTER: LazyLock<std::result::Result<bool, String>> =
    LazyLock::new(|| load_rollback_value("PEGAINFER_DSV2_LITE_NCCL_ROUTER", "device", "host"));
fn group_route_indices(
    keys: impl IntoIterator<Item = (usize, usize)>,
) -> BTreeMap<(usize, usize), Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (route_index, key) in keys.into_iter().enumerate() {
        groups.entry(key).or_insert_with(Vec::new).push(route_index);
    }
    groups
}

struct NcclRouteReplayBuffers {
    _inputs: Vec<HiddenStates>,
    _outputs: Vec<HiddenStates>,
}

pub(super) struct DeviceRoutedMoeRuntime {
    rank0_tables: Vec<Option<Dsv2ExpertPointerTable>>,
    rank1_tables: Vec<Option<Dsv2ExpertPointerTable>>,
    rank0_scratch: Dsv2RoutedMoeScratch,
    rank1_scratch: Dsv2RoutedMoeScratch,
}

impl DeviceRoutedMoeRuntime {
    pub(super) fn new(generator: &DeepSeekV2LiteEp2Generator) -> Result<Self> {
        let mut rank0_tables = Vec::with_capacity(generator.config.num_hidden_layers);
        let mut rank1_tables = Vec::with_capacity(generator.config.num_hidden_layers);
        for layer_idx in 0..generator.config.num_hidden_layers {
            if generator.config.is_moe_layer(layer_idx) {
                activate(&generator.rank0.ctx)?;
                rank0_tables.push(Some(Dsv2ExpertPointerTable::new(
                    &generator.rank0.ctx,
                    generator.rank0.routed_expert_projections(layer_idx)?,
                    generator.rank0.layout.owned_experts().start,
                )?));
                activate(&generator.rank1.ctx)?;
                rank1_tables.push(Some(Dsv2ExpertPointerTable::new(
                    &generator.rank1.ctx,
                    generator.rank1.routed_expert_projections(layer_idx)?,
                    generator.rank1.layout.owned_experts().start,
                )?));
            } else {
                rank0_tables.push(None);
                rank1_tables.push(None);
            }
        }
        activate(&generator.rank0.ctx)?;
        let rank0_scratch = Dsv2RoutedMoeScratch::new(
            &generator.rank0.ctx,
            generator.config.hidden_size,
            generator.config.moe_intermediate_size,
            DSV2_ROUTED_MOE_MAX_ROWS,
            generator.config.num_hidden_layers,
        )?;
        activate(&generator.rank1.ctx)?;
        let rank1_scratch = Dsv2RoutedMoeScratch::new(
            &generator.rank1.ctx,
            generator.config.hidden_size,
            generator.config.moe_intermediate_size,
            DSV2_ROUTED_MOE_MAX_ROWS,
            generator.config.num_hidden_layers,
        )?;
        Ok(Self {
            rank0_tables,
            rank1_tables,
            rank0_scratch,
            rank1_scratch,
        })
    }

    /// Clears both ranks' route summaries before all MoE layers in one forward.
    fn begin_forward(&mut self, generator: &DeepSeekV2LiteEp2Generator) -> Result<()> {
        activate(&generator.rank0.ctx)?;
        self.rank0_scratch.begin_forward(&generator.rank0.ctx)?;
        activate(&generator.rank1.ctx)?;
        self.rank1_scratch.begin_forward(&generator.rank1.ctx)
    }

    /// Reads and validates both ranks' accumulated route summaries after all MoE layers.
    fn finish_forward(&self, generator: &DeepSeekV2LiteEp2Generator) -> Result<(usize, usize)> {
        activate(&generator.rank0.ctx)?;
        let rank0 = self.rank0_scratch.finish_forward(&generator.rank0.ctx)?;
        activate(&generator.rank1.ctx)?;
        let rank1 = self.rank1_scratch.finish_forward(&generator.rank1.ctx)?;
        ensure!(
            rank0.errors == 0 && rank1.errors == 0,
            "device-routed MoE reported nonfinite router input: rank0={}, rank1={}",
            rank0.errors,
            rank1.errors
        );
        ensure!(
            rank0.total_routes == rank1.total_routes,
            "device-routed MoE route totals differ across ranks: rank0={}, rank1={}",
            rank0.total_routes,
            rank1.total_routes
        );
        ensure!(
            rank0.layer_hashes == rank1.layer_hashes,
            "device-routed MoE route IDs differ across ranks by layer: rank0={:?}, rank1={:?}",
            rank0.layer_hashes,
            rank1.layer_hashes
        );
        ensure!(
            rank0.local_routes + rank1.local_routes == rank0.total_routes,
            "device-routed MoE ownership accounting drift: rank0_local={}, rank1_local={}, total={}",
            rank0.local_routes,
            rank1.local_routes,
            rank0.total_routes
        );
        Ok((rank0.local_routes, rank1.local_routes))
    }
}

fn accumulate_host_staged_route_output(
    dst: &mut [f32],
    route: &MoeRouteEntry,
    out: &[f32],
    hidden_size: usize,
) -> Result<()> {
    ensure!(
        hidden_size > 0,
        "host-staged route output requires nonzero hidden size"
    );
    ensure!(
        out.len() == hidden_size,
        "host-staged route output len mismatch: got {}, expected {hidden_size}",
        out.len()
    );
    let token_end = route
        .token
        .checked_add(1)
        .and_then(|tokens| tokens.checked_mul(hidden_size))
        .context("host-staged route output offset overflow")?;
    ensure!(
        token_end <= dst.len(),
        "host-staged route token {} exceeds contribution rows {}",
        route.token,
        dst.len() / hidden_size
    );
    let token_begin = token_end - hidden_size;
    for (dst, value) in dst[token_begin..token_end].iter_mut().zip(out) {
        *dst += route.weight * *value;
    }
    Ok(())
}

pub(super) struct FixedTopologyMoeScratch {
    rank0_topk_weight: CudaSlice<f32>,
    rank0_topk_idx: CudaSlice<i32>,
    rank1_topk_weight: CudaSlice<f32>,
    rank1_topk_idx: CudaSlice<i32>,
    shared: DenseMlpForwardScratch,
    rank0_expert: DenseMlpForwardScratch,
    rank1_expert: DenseMlpForwardScratch,
    routed: HiddenStates,
}

impl FixedTopologyMoeScratch {
    pub(super) fn new(
        generator: &DeepSeekV2LiteEp2Generator,
        layer_idx: usize,
        moe: &MoeMlp,
        seq_len: usize,
    ) -> Result<Self> {
        let topk_elems = seq_len * generator.config.num_experts_per_token;
        let first_rank0_expert = generator.rank0.layout.owned_experts().start;
        let first_rank1_expert = generator.rank1.layout.owned_experts().start;
        let first_rank0 = generator
            .rank0
            .routed_expert(layer_idx, first_rank0_expert)?;
        let first_rank1 = generator
            .rank1
            .routed_expert(layer_idx, first_rank1_expert)?;
        activate(&generator.rank0.ctx)?;
        let rank0_topk_weight = generator.rank0.ctx.stream.alloc_zeros::<f32>(topk_elems)?;
        let rank0_topk_idx = generator.rank0.ctx.stream.alloc_zeros::<i32>(topk_elems)?;
        let shared = DenseMlpForwardScratch::new(&generator.rank0.ctx, &moe.shared, seq_len)?;
        let rank0_expert =
            DenseMlpForwardScratch::new(&generator.rank0.ctx, &first_rank0.dense, seq_len)?;
        let routed =
            HiddenStates::zeros(&generator.rank0.ctx, generator.config.hidden_size, seq_len)?;
        activate(&generator.rank1.ctx)?;
        let rank1_topk_weight = generator.rank1.ctx.stream.alloc_zeros::<f32>(topk_elems)?;
        let rank1_topk_idx = generator.rank1.ctx.stream.alloc_zeros::<i32>(topk_elems)?;
        let rank1_expert =
            DenseMlpForwardScratch::new(&generator.rank1.ctx, &first_rank1.dense, seq_len)?;
        Ok(Self {
            rank0_topk_weight,
            rank0_topk_idx,
            rank1_topk_weight,
            rank1_topk_idx,
            shared,
            rank0_expert,
            rank1_expert,
            routed,
        })
    }
}

impl DeepSeekV2LiteEp2Generator {
    pub(super) fn moe_forward(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        match &self.backend {
            EpBackendRuntime::HostStaged => self.moe_forward_host_staged(
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
            EpBackendRuntime::Nccl(nccl) => self.moe_forward_nccl(
                nccl,
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
        }
    }

    fn moe_forward_host_staged(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let (input_host, routes) = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.host_staged.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);
                Ok((input_host, routes))
            },
        )?;

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let route_plan = MoeRoutePlan::from_topk_routes(&routes, &self.rank0.layout)?;
        let route_groups = group_route_indices(
            route_plan
                .entries()
                .iter()
                .map(|route| (route.owner_rank, route.global_expert)),
        );
        let mut group_outputs = Vec::with_capacity(route_groups.len());
        let mut route_locations = vec![None; route_plan.route_count()];
        if rollback_enabled(&HOST_STAGED_SERIAL)? {
            for (route_index, route) in route_plan.entries().iter().enumerate() {
                let section = if route.owner_rank == 0 {
                    "host_staged_local_expert"
                } else {
                    "host_staged_remote_dispatch"
                };
                let expert_ctx = if route.owner_rank == 0 {
                    &self.rank0.ctx
                } else {
                    &self.rank1.ctx
                };
                let begin = route.token * self.config.hidden_size;
                let end = begin + self.config.hidden_size;
                let out = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}"),
                    Some(layer_idx),
                    token_index,
                    || {
                        self.expert_forward_host_token(
                            layer_idx,
                            route.global_expert,
                            &input_host[begin..end],
                        )
                    },
                )?;
                route_locations[route_index] = Some((group_outputs.len(), 0));
                group_outputs.push(out);
            }
        } else {
            for ((owner_rank, global_expert), route_indices) in route_groups {
                let section = if owner_rank == 0 {
                    "host_staged_local_expert"
                } else {
                    "host_staged_remote_dispatch"
                };
                let expert_ctx = if owner_rank == 0 {
                    &self.rank0.ctx
                } else {
                    &self.rank1.ctx
                };
                let out = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}.expert{global_expert}"),
                    Some(layer_idx),
                    token_index,
                    || {
                        self.expert_forward_host_batch(
                            layer_idx,
                            global_expert,
                            &input_host,
                            route_plan.entries(),
                            &route_indices,
                        )
                    },
                )?;
                ensure!(
                    out.len() == route_indices.len() * self.config.hidden_size,
                    "host-staged expert output len mismatch: got {}, expected {}",
                    out.len(),
                    route_indices.len() * self.config.hidden_size
                );
                let group_index = group_outputs.len();
                for (group_row, route_index) in route_indices.into_iter().enumerate() {
                    ensure!(
                        route_locations[route_index]
                            .replace((group_index, group_row))
                            .is_none(),
                        "host-staged route {route_index} was assigned more than once"
                    );
                }
                group_outputs.push(out);
            }
        }

        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        for (route_index, route) in route_plan.entries().iter().enumerate() {
            let dst = if route.owner_rank == 0 {
                &mut rank0_contrib
            } else {
                &mut rank1_contrib
            };
            let (group_index, group_row) = route_locations[route_index].with_context(|| {
                format!("missing host-staged expert output for route {route_index}")
            })?;
            let begin = group_row * self.config.hidden_size;
            let end = begin + self.config.hidden_size;
            let out = &group_outputs[group_index][begin..end];
            attribution.record_result(
                phase,
                "host_staged_combine_accumulate",
                || format!("layer.{layer_idx}.host_staged.combine_accumulate"),
                Some(layer_idx),
                token_index,
                || accumulate_host_staged_route_output(dst, route, out, self.config.hidden_size),
            )?;
        }
        let routed_accum: Vec<_> = rank0_contrib
            .into_iter()
            .zip(rank1_contrib)
            .map(|(rank0, rank1)| rank0 + rank1)
            .collect();

        let routed = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "host_staged_combine_to_device",
            || format!("layer.{layer_idx}.host_staged.combine_to_device"),
            Some(layer_idx),
            token_index,
            || {
                hidden_from_f32_host(
                    &self.rank0.ctx,
                    &routed_accum,
                    self.config.hidden_size,
                    input.seq_len,
                )
            },
        )?;
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((
            hidden,
            route_plan.local_routes(),
            route_plan.remote_routes(),
        ))
    }

    fn moe_forward_nccl(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let host_router = rollback_enabled(&NCCL_HOST_ROUTER)?;
        let device_routed = self.device_routed_moe_enabled(input.seq_len)?;
        let route_plan = if device_routed {
            None
        } else {
            let route_section = if host_router {
                "ep_route_host"
            } else {
                "ep_route_device"
            };
            Some(attribution.record_result(
                phase,
                route_section,
                || format!("layer.{layer_idx}.nccl.route"),
                Some(layer_idx),
                token_index,
                || {
                    if host_router {
                        let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                        let route_logits_host =
                            gate_logits_host(&self.config, &input_host, &moe.gate_host);
                        let routes =
                            topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);
                        MoeRoutePlan::from_topk_routes(&routes, &self.rank0.layout)
                    } else {
                        self.build_nccl_route_plan_device(input, &moe.gate_device)
                    }
                },
            )?)
        };

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let rank1_input = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_dense_exchange",
            || format!("layer.{layer_idx}.nccl.dense_exchange"),
            Some(layer_idx),
            token_index,
            || nccl.dense_all_reduce_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input),
        )?;
        let rank1_hidden = rank1_input.rank1_hidden()?;
        let (routed, local_routes, remote_routes) = if device_routed {
            attribution.record_gpu_pair_result(
                &self.rank0.ctx,
                &self.rank1.ctx,
                phase,
                "nccl_device_routed_experts",
                || format!("layer.{layer_idx}.nccl.device_routed_experts"),
                Some(layer_idx),
                token_index,
                || {
                    let runtime = self
                        .device_routed_moe
                        .as_ref()
                        .context("NCCL device-routed MoE runtime is not initialized")?;
                    let mut runtime = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("device-routed MoE mutex poisoned"))?;
                    let DeviceRoutedMoeRuntime {
                        rank0_tables,
                        rank1_tables,
                        rank0_scratch,
                        rank1_scratch,
                    } = &mut *runtime;
                    let rank0_table = rank0_tables
                        .get(layer_idx)
                        .and_then(Option::as_ref)
                        .context("rank0 device-routed MoE pointer table missing")?;
                    let rank1_table = rank1_tables
                        .get(layer_idx)
                        .and_then(Option::as_ref)
                        .context("rank1 device-routed MoE pointer table missing")?;
                    let mut contributions = nccl.prepare_device_combine_output(
                        &self.rank0.ctx,
                        &self.rank1.ctx,
                        input.hidden_dim,
                        input.seq_len,
                    )?;
                    activate(&self.rank0.ctx)?;
                    rank0_scratch.enqueue_into(
                        &self.rank0.ctx,
                        input.as_ref(),
                        &moe.gate_device,
                        rank0_table,
                        contributions.rank_send_mut(0)?,
                        layer_idx,
                    )?;
                    activate(&self.rank1.ctx)?;
                    rank1_scratch.enqueue_into(
                        &self.rank1.ctx,
                        rank1_hidden,
                        self.rank1.gate_device(layer_idx)?,
                        rank1_table,
                        contributions.rank_send_mut(1)?,
                        layer_idx,
                    )?;
                    drop(contributions);
                    let routed = nccl.combine_device_contributions_to_rank0(
                        &self.rank0.ctx,
                        &self.rank1.ctx,
                        input.hidden_dim,
                        input.seq_len,
                    )?;
                    Ok((routed, 0, 0))
                },
            )?
        } else {
            let route_plan = route_plan
                .as_ref()
                .context("NCCL fallback route plan is missing")?;
            attribution.record_gpu_pair_result(
                &self.rank0.ctx,
                &self.rank1.ctx,
                phase,
                "nccl_combine_clear",
                || format!("layer.{layer_idx}.nccl.combine_clear"),
                Some(layer_idx),
                token_index,
                || {
                    nccl.clear_device_combine(
                        &self.rank0.ctx,
                        &self.rank1.ctx,
                        input.hidden_dim,
                        input.seq_len,
                    )
                },
            )?;
            let live_expert_outputs = self.replay_nccl_route_plan(
                nccl,
                layer_idx,
                input,
                rank1_hidden,
                route_plan,
                attribution,
                phase,
                token_index,
            )?;
            let routed = attribution.record_gpu_pair_result(
                &self.rank0.ctx,
                &self.rank1.ctx,
                phase,
                "nccl_combine",
                || format!("layer.{layer_idx}.nccl.combine"),
                Some(layer_idx),
                token_index,
                || {
                    nccl.combine_device_contributions_to_rank0(
                        &self.rank0.ctx,
                        &self.rank1.ctx,
                        input.hidden_dim,
                        input.seq_len,
                    )
                },
            )?;
            drop(live_expert_outputs);
            (
                routed,
                route_plan.local_routes(),
                route_plan.remote_routes(),
            )
        };
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((hidden, local_routes, remote_routes))
    }

    fn build_nccl_route_plan_device(
        &self,
        input: &HiddenStates,
        gate_device: &pegainfer_core::tensor::DeviceMatrix,
    ) -> Result<MoeRoutePlan> {
        activate(&self.rank0.ctx)?;
        let logits_elems = input
            .seq_len
            .checked_mul(self.config.n_routed_experts)
            .context("NCCL device router logits element count overflow")?;
        let mut route_logits = self.rank0.ctx.stream.alloc_zeros::<f32>(logits_elems)?;
        dsv2_lite_router_logits_into(
            &self.rank0.ctx,
            input.as_ref(),
            gate_device,
            &mut route_logits,
        )?;
        let route_logits = self.rank0.ctx.stream.clone_dtoh(&route_logits)?;
        self.rank0.ctx.sync()?;
        let routes = topk_softmax_routes(&self.config, &route_logits, input.seq_len);
        MoeRoutePlan::from_topk_routes(&routes, &self.rank0.layout)
    }

    pub(super) fn moe_forward_nccl_fixed_topology_preallocated_into(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        scratch: &mut FixedTopologyMoeScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        activate(&self.rank0.ctx)?;
        dsv2_lite_router_softmax_topk_into(
            &self.rank0.ctx,
            input,
            &moe.gate_device,
            self.config.num_experts_per_token,
            &mut Dsv2LiteRouterOutput {
                topk_weight: &mut scratch.rank0_topk_weight,
                topk_idx: &mut scratch.rank0_topk_idx,
            },
        )?;

        dense_mlp_forward_preallocated_into(
            &self.rank0.ctx,
            &moe.shared,
            input,
            &mut scratch.shared,
        )?;

        let rank1_input =
            nccl.dense_all_reduce_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input)?;
        let rank1_hidden = rank1_input.rank1_hidden()?;
        activate(&self.rank1.ctx)?;
        dsv2_lite_router_softmax_topk_ref_into(
            &self.rank1.ctx,
            rank1_hidden,
            self.rank1.gate_device(layer_idx)?,
            self.config.num_experts_per_token,
            &mut Dsv2LiteRouterOutput {
                topk_weight: &mut scratch.rank1_topk_weight,
                topk_idx: &mut scratch.rank1_topk_idx,
            },
        )?;

        nccl.clear_device_combine(
            &self.rank0.ctx,
            &self.rank1.ctx,
            input.hidden_dim,
            input.seq_len,
        )?;

        for global_expert in self.rank0.layout.owned_experts() {
            let expert = self.rank0.routed_expert(layer_idx, global_expert)?;
            dense_mlp_forward_preallocated_into(
                &self.rank0.ctx,
                &expert.dense,
                input,
                &mut scratch.rank0_expert,
            )?;
            nccl.accumulate_fixed_expert_contribution(
                0,
                &self.rank0.ctx,
                &scratch.rank0_expert.out,
                &scratch.rank0_topk_weight,
                &scratch.rank0_topk_idx,
                global_expert,
                self.config.num_experts_per_token,
            )?;
        }

        for global_expert in self.rank1.layout.owned_experts() {
            let expert = self.rank1.routed_expert(layer_idx, global_expert)?;
            dense_mlp_forward_preallocated_ref_into(
                &self.rank1.ctx,
                &expert.dense,
                rank1_hidden,
                &mut scratch.rank1_expert,
            )?;
            nccl.accumulate_fixed_expert_contribution(
                1,
                &self.rank1.ctx,
                &scratch.rank1_expert.out,
                &scratch.rank1_topk_weight,
                &scratch.rank1_topk_idx,
                global_expert,
                self.config.num_experts_per_token,
            )?;
        }

        nccl.combine_device_contributions_to_rank0_into(
            &self.rank0.ctx,
            &self.rank1.ctx,
            input.hidden_dim,
            input.seq_len,
            &mut scratch.routed,
        )?;
        drop(rank1_input);
        activate(&self.rank0.ctx)?;
        ops::add_batch_into(&self.rank0.ctx, &scratch.routed, &scratch.shared.out, out)
    }

    fn expert_forward_host_batch(
        &self,
        layer_idx: usize,
        global_expert: usize,
        input_host: &[bf16],
        route_work: &[MoeRouteEntry],
        route_indices: &[usize],
    ) -> Result<Vec<f32>> {
        let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
        let (ctx, expert) = match owner_rank {
            0 => (
                &self.rank0.ctx,
                self.rank0.routed_expert(layer_idx, global_expert)?,
            ),
            1 => (
                &self.rank1.ctx,
                self.rank1.routed_expert(layer_idx, global_expert)?,
            ),
            other => bail!("routed expert {global_expert} maps to unsupported EP rank {other}"),
        };

        ensure!(
            !route_indices.is_empty(),
            "host-staged expert batch requires at least one route"
        );
        let mut batch_input = Vec::with_capacity(route_indices.len() * self.config.hidden_size);
        for &route_index in route_indices {
            let route = &route_work[route_index];
            ensure!(
                route.global_expert == global_expert && route.owner_rank == owner_rank,
                "host-staged expert batch mixed route: expected expert {global_expert}/rank {owner_rank}, got expert {}/rank {}",
                route.global_expert,
                route.owner_rank
            );
            let begin = route.token * self.config.hidden_size;
            let end = begin + self.config.hidden_size;
            batch_input.extend_from_slice(&input_host[begin..end]);
        }
        let input = hidden_from_bf16_host(
            ctx,
            &batch_input,
            self.config.hidden_size,
            route_indices.len(),
        )?;
        let out = dense_mlp_forward_per_token(ctx, &expert.dense, &input)?;
        hidden_to_f32(ctx, &out)
    }

    fn expert_forward_host_token(
        &self,
        layer_idx: usize,
        global_expert: usize,
        token_input: &[bf16],
    ) -> Result<Vec<f32>> {
        let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
        let (ctx, expert) = match owner_rank {
            0 => (
                &self.rank0.ctx,
                self.rank0.routed_expert(layer_idx, global_expert)?,
            ),
            1 => (
                &self.rank1.ctx,
                self.rank1.routed_expert(layer_idx, global_expert)?,
            ),
            other => bail!("routed expert {global_expert} maps to unsupported EP rank {other}"),
        };

        let input = hidden_from_bf16_host(ctx, token_input, self.config.hidden_size, 1)?;
        let out = dense_mlp_forward(ctx, &expert.dense, &input)?;
        hidden_to_f32(ctx, &out)
    }

    fn replay_nccl_route_plan(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        rank1_hidden: HiddenStatesRef<'_>,
        route_plan: &MoeRoutePlan,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<NcclRouteReplayBuffers> {
        if rollback_enabled(&NCCL_SERIAL)? {
            return self.replay_nccl_route_plan_serial(
                nccl,
                layer_idx,
                input,
                rank1_hidden,
                route_plan,
                attribution,
                phase,
                token_index,
            );
        }
        self.replay_nccl_route_plan_grouped(
            nccl,
            layer_idx,
            input,
            rank1_hidden,
            route_plan,
            attribution,
            phase,
            token_index,
        )
    }

    fn replay_nccl_route_plan_serial(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        rank1_hidden: HiddenStatesRef<'_>,
        route_plan: &MoeRoutePlan,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<NcclRouteReplayBuffers> {
        let mut live_expert_outputs = Vec::with_capacity(route_plan.route_count());
        for route in route_plan.entries() {
            let out = self.forward_nccl_route(
                layer_idx,
                input.as_ref(),
                rank1_hidden,
                route,
                attribution,
                phase,
                token_index,
            )?;
            let expert_ctx = match route.owner_rank {
                0 => &self.rank0.ctx,
                1 => &self.rank1.ctx,
                other => bail!(
                    "routed expert {} maps to unsupported EP rank {other}",
                    route.global_expert
                ),
            };
            attribution.record_gpu_result(
                expert_ctx,
                phase,
                "nccl_contribution_accumulate_device",
                || format!("layer.{layer_idx}.nccl.contribution_accumulate_device"),
                Some(layer_idx),
                token_index,
                || {
                    nccl.accumulate_device_contribution_row(
                        route.owner_rank,
                        expert_ctx,
                        &out,
                        0,
                        route.token,
                        input.seq_len,
                        route.weight,
                    )
                },
            )?;
            live_expert_outputs.push(out);
        }
        Ok(NcclRouteReplayBuffers {
            _inputs: Vec::new(),
            _outputs: live_expert_outputs,
        })
    }

    fn replay_nccl_route_plan_grouped(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        rank1_hidden: HiddenStatesRef<'_>,
        route_plan: &MoeRoutePlan,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<NcclRouteReplayBuffers> {
        let route_groups = group_route_indices(
            route_plan
                .entries()
                .iter()
                .map(|route| (route.owner_rank, route.global_expert)),
        );
        let mut group_inputs = Vec::with_capacity(route_groups.len());
        let mut group_outputs = Vec::with_capacity(route_groups.len());
        let mut route_locations = vec![None; route_plan.route_count()];

        for ((owner_rank, global_expert), route_indices) in route_groups {
            let (ctx, source_hidden, expert, section) = match owner_rank {
                0 => (
                    &self.rank0.ctx,
                    input.as_ref(),
                    self.rank0.routed_expert(layer_idx, global_expert)?,
                    "nccl_local_expert",
                ),
                1 => (
                    &self.rank1.ctx,
                    rank1_hidden,
                    self.rank1.routed_expert(layer_idx, global_expert)?,
                    "nccl_remote_expert",
                ),
                other => {
                    bail!("routed expert {global_expert} maps to unsupported EP rank {other}")
                }
            };
            ensure!(
                !route_indices.is_empty(),
                "NCCL expert group requires at least one route"
            );
            let group_input =
                gather_nccl_route_group(ctx, source_hidden, route_plan.entries(), &route_indices)?;
            let group_output = attribution.record_gpu_result(
                ctx,
                phase,
                section,
                || format!("layer.{layer_idx}.nccl.expert{global_expert}"),
                Some(layer_idx),
                token_index,
                || dense_mlp_forward_per_token(ctx, &expert.dense, &group_input),
            )?;
            let group_index = group_outputs.len();
            for (group_row, route_index) in route_indices.into_iter().enumerate() {
                ensure!(
                    route_locations[route_index]
                        .replace((group_index, group_row))
                        .is_none(),
                    "NCCL route {route_index} was assigned to more than one expert group"
                );
            }
            group_inputs.push(group_input);
            group_outputs.push(group_output);
        }

        for (route_index, route) in route_plan.entries().iter().enumerate() {
            let (group_index, output_row) = route_locations[route_index]
                .with_context(|| format!("missing NCCL expert output for route {route_index}"))?;
            let expert_ctx = match route.owner_rank {
                0 => &self.rank0.ctx,
                1 => &self.rank1.ctx,
                other => bail!(
                    "routed expert {} maps to unsupported EP rank {other}",
                    route.global_expert
                ),
            };
            attribution.record_gpu_result(
                expert_ctx,
                phase,
                "nccl_contribution_accumulate_device",
                || format!("layer.{layer_idx}.nccl.contribution_accumulate_device"),
                Some(layer_idx),
                token_index,
                || {
                    nccl.accumulate_device_contribution_row(
                        route.owner_rank,
                        expert_ctx,
                        &group_outputs[group_index],
                        output_row,
                        route.token,
                        input.seq_len,
                        route.weight,
                    )
                },
            )?;
        }

        Ok(NcclRouteReplayBuffers {
            _inputs: group_inputs,
            _outputs: group_outputs,
        })
    }

    fn forward_nccl_route(
        &self,
        layer_idx: usize,
        rank0_hidden: HiddenStatesRef<'_>,
        rank1_hidden: HiddenStatesRef<'_>,
        route: &MoeRouteEntry,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        match route.owner_rank {
            0 => {
                let expert = self.rank0.routed_expert(layer_idx, route.global_expert)?;
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    phase,
                    "nccl_local_expert",
                    || format!("layer.{layer_idx}.nccl.local_expert"),
                    Some(layer_idx),
                    token_index,
                    || expert_forward_device(&self.rank0.ctx, expert, rank0_hidden, route.token),
                )
            }
            1 => {
                let expert = self.rank1.routed_expert(layer_idx, route.global_expert)?;
                attribution.record_gpu_result(
                    &self.rank1.ctx,
                    phase,
                    "nccl_remote_expert",
                    || format!("layer.{layer_idx}.nccl.remote_expert"),
                    Some(layer_idx),
                    token_index,
                    || expert_forward_device(&self.rank1.ctx, expert, rank1_hidden, route.token),
                )
            }
            other => bail!(
                "routed expert {} maps to unsupported EP rank {other}",
                route.global_expert
            ),
        }
    }

    fn device_routed_moe_enabled(&self, rows: usize) -> Result<bool> {
        Ok(self.backend.kind() == EpBackendKind::Nccl
            && rows <= DSV2_ROUTED_MOE_MAX_ROWS
            && !rollback_enabled(&NCCL_HOST_ROUTER)?
            && !rollback_enabled(&NCCL_SERIAL)?)
    }

    pub(super) fn begin_device_routed_forward(&self, rows: usize) -> Result<bool> {
        if !self.device_routed_moe_enabled(rows)? {
            return Ok(false);
        }
        let runtime = self
            .device_routed_moe
            .as_ref()
            .context("NCCL device-routed MoE runtime is not initialized")?;
        runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("device-routed MoE mutex poisoned"))?
            .begin_forward(self)?;
        Ok(true)
    }

    pub(super) fn finish_device_routed_forward(&self) -> Result<(usize, usize)> {
        let runtime = self
            .device_routed_moe
            .as_ref()
            .context("NCCL device-routed MoE runtime is not initialized")?;
        runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("device-routed MoE mutex poisoned"))?
            .finish_forward(self)
    }
}

fn expert_forward_device(
    ctx: &pegainfer_core::tensor::DeviceContext,
    expert: &ExpertMlp,
    input: HiddenStatesRef<'_>,
    token_idx: usize,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let token = ops::extract_vec_ref(ctx, input, token_idx)?;
    let token_hidden = HiddenStates {
        hidden_dim: token.len,
        seq_len: 1,
        data: token.data,
    };
    dense_mlp_forward(ctx, &expert.dense, &token_hidden)
}

fn gather_nccl_route_group(
    ctx: &pegainfer_core::tensor::DeviceContext,
    input: HiddenStatesRef<'_>,
    routes: &[MoeRouteEntry],
    route_indices: &[usize],
) -> Result<HiddenStates> {
    ensure!(
        !route_indices.is_empty(),
        "NCCL route gather requires at least one route"
    );
    activate(ctx)?;
    let mut gathered = HiddenStates::zeros(ctx, input.hidden_dim, route_indices.len())?;
    for (group_row, route_index) in route_indices.iter().copied().enumerate() {
        let route = routes
            .get(route_index)
            .with_context(|| format!("NCCL route index {route_index} is out of bounds"))?;
        ensure!(
            route.token < input.seq_len,
            "NCCL route token {} exceeds input seq_len {}",
            route.token,
            input.seq_len
        );
        let src_begin = route.token * input.hidden_dim;
        let dst_begin = group_row * input.hidden_dim;
        let src = input.data.slice(src_begin..src_begin + input.hidden_dim);
        let mut dst = gathered
            .data
            .slice_mut(dst_begin..dst_begin + input.hidden_dim);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .context("gather NCCL route input row")?;
    }
    Ok(gathered)
}

#[cfg(test)]
mod tests;
