use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;
use crate::tensor::HiddenStatesRef;

const ROUTES_PER_TOKEN: usize = 6;
const LOCAL_EXPERTS: usize = 32;
pub const DSV2_ROUTED_MOE_MAX_ROWS: usize = 8;

/// A table of device pointers to the gate up and down projection matrices for all experts in a rank.
pub struct Dsv2ExpertPointerTable {
    first_expert: i32,
    hidden_dim: usize,
    intermediate: usize,
    gate_up: CudaSlice<u64>,
    down: CudaSlice<u64>,
}

impl Dsv2ExpertPointerTable {
    pub fn new(
        ctx: &DeviceContext,
        experts: Vec<(&DeviceMatrix, &DeviceMatrix)>,
        first_expert: usize,
    ) -> Result<Self> {
        ensure!(
            first_expert == 0 || first_expert == LOCAL_EXPERTS,
            "invalid DeepSeek-V2-Lite EP2 expert base {first_expert}"
        );
        ensure!(
            experts.len() == LOCAL_EXPERTS,
            "DeepSeek-V2-Lite routed MoE requires {LOCAL_EXPERTS} local experts, got {}",
            experts.len()
        );
        let hidden_dim = experts[0].0.cols;
        let intermediate = experts[0].1.cols;
        ensure!(hidden_dim > 0 && intermediate > 0, "empty expert matrix");

        let mut gate_up_ptrs = Vec::with_capacity(LOCAL_EXPERTS);
        let mut down_ptrs = Vec::with_capacity(LOCAL_EXPERTS);
        for (gate_up, down) in experts {
            ensure!(
                gate_up.rows == 2 * intermediate
                    && gate_up.cols == hidden_dim
                    && down.rows == hidden_dim
                    && down.cols == intermediate,
                "inconsistent DeepSeek-V2-Lite expert projection shape"
            );
            for matrix in [gate_up, down] {
                let needed = matrix
                    .rows
                    .checked_mul(matrix.cols)
                    .context("expert matrix element count overflow")?;
                ensure!(
                    matrix.data.len() >= needed,
                    "expert matrix backing buffer too small"
                );
                ensure!(
                    Arc::ptr_eq(matrix.data.stream(), &ctx.stream),
                    "expert matrix stream mismatch"
                );
            }
            gate_up_ptrs.push(gate_up.data.device_ptr(&ctx.stream).0);
            down_ptrs.push(down.data.device_ptr(&ctx.stream).0);
        }

        Ok(Self {
            first_expert: first_expert as i32,
            hidden_dim,
            intermediate,
            gate_up: ctx.stream.clone_htod(&gate_up_ptrs)?,
            down: ctx.stream.clone_htod(&down_ptrs)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dsv2RouteSummary {
    pub local_routes: usize,
    pub total_routes: usize,
    pub errors: usize,
    pub layer_hashes: Vec<u64>,
}

pub struct Dsv2RoutedMoeScratch {
    capacity: usize,
    hidden_dim: usize,
    intermediate: usize,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    errors: CudaSlice<i32>,
    pointers: [CudaSlice<u64>; 6],
    zero: CudaSlice<bf16>,
    gate_up: HiddenStates,
    activation: HiddenStates,
    rows: HiddenStates,
    summary: CudaSlice<u64>,
}

impl Dsv2RoutedMoeScratch {
    pub fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate: usize,
        capacity: usize,
        num_layers: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=DSV2_ROUTED_MOE_MAX_ROWS).contains(&capacity),
            "DeepSeek-V2-Lite routed MoE capacity must be in 1..={DSV2_ROUTED_MOE_MAX_ROWS}"
        );
        ensure!(hidden_dim > 0 && intermediate > 0, "empty routed MoE shape");
        ensure!(num_layers > 0, "routed MoE requires at least one layer");
        for dim in [hidden_dim, intermediate] {
            ensure!(
                dim <= i32::MAX as usize / (2 * ROUTES_PER_TOKEN * capacity),
                "routed MoE dimension exceeds CUDA indexing range"
            );
        }
        let routes = capacity * ROUTES_PER_TOKEN;
        Ok(Self {
            capacity,
            hidden_dim,
            intermediate,
            logits: ctx.stream.alloc_zeros(capacity * 64)?,
            ids: ctx.stream.alloc_zeros(routes)?,
            weights: ctx.stream.alloc_zeros(routes)?,
            errors: ctx.stream.alloc_zeros(capacity)?,
            pointers: [
                ctx.stream.alloc_zeros(routes)?,
                ctx.stream.alloc_zeros(routes)?,
                ctx.stream.alloc_zeros(routes)?,
                ctx.stream.alloc_zeros(routes)?,
                ctx.stream.alloc_zeros(routes)?,
                ctx.stream.alloc_zeros(routes)?,
            ],
            zero: ctx.stream.alloc_zeros(hidden_dim)?,
            gate_up: HiddenStates::zeros(ctx, 2 * intermediate, routes)?,
            activation: HiddenStates::zeros(ctx, intermediate, routes)?,
            rows: HiddenStates::zeros(ctx, hidden_dim, routes)?,
            summary: ctx.stream.alloc_zeros(3 + num_layers)?,
        })
    }

    pub fn begin_forward(&mut self, ctx: &DeviceContext) -> Result<()> {
        ensure!(
            Arc::ptr_eq(self.summary.stream(), &ctx.stream),
            "routed MoE summary stream mismatch"
        );
        ctx.stream.memset_zeros(&mut self.summary)?;
        Ok(())
    }

    pub fn finish_forward(&self, ctx: &DeviceContext) -> Result<Dsv2RouteSummary> {
        let summary = ctx.stream.clone_dtoh(&self.summary)?;
        ctx.sync()?;
        ensure!(summary.len() > 3, "routed MoE summary shape drift");
        Ok(Dsv2RouteSummary {
            local_routes: usize::try_from(summary[0]).context("local route count overflow")?,
            total_routes: usize::try_from(summary[1]).context("total route count overflow")?,
            errors: usize::try_from(summary[2]).context("route error count overflow")?,
            layer_hashes: summary[3..].to_vec(),
        })
    }

    pub fn enqueue_into(
        &mut self,
        ctx: &DeviceContext,
        hidden: HiddenStatesRef<'_>,
        gate_weight: &DeviceMatrix,
        experts: &Dsv2ExpertPointerTable,
        output: &mut CudaSlice<f32>,
        layer_idx: usize,
    ) -> Result<()> {
        let batch = hidden.seq_len;
        ensure!(
            (1..=self.capacity).contains(&batch),
            "routed MoE rows {batch} exceed capacity {}",
            self.capacity
        );
        ensure!(
            layer_idx < self.summary.len() - 3,
            "routed MoE layer {layer_idx} exceeds summary capacity {}",
            self.summary.len() - 3
        );
        ensure!(
            hidden.hidden_dim == self.hidden_dim
                && experts.hidden_dim == self.hidden_dim
                && experts.intermediate == self.intermediate,
            "routed MoE hidden or expert shape mismatch"
        );
        ensure!(
            gate_weight.rows == 64 && gate_weight.cols == self.hidden_dim,
            "routed MoE gate shape mismatch"
        );
        ensure!(
            hidden.data.len() >= batch * self.hidden_dim && output.len() >= batch * self.hidden_dim,
            "routed MoE input or output backing buffer too small"
        );
        ensure!(
            Arc::ptr_eq(hidden.data.stream(), &ctx.stream)
                && Arc::ptr_eq(gate_weight.data.stream(), &ctx.stream)
                && Arc::ptr_eq(output.stream(), &ctx.stream)
                && Arc::ptr_eq(experts.gate_up.stream(), &ctx.stream),
            "routed MoE stream mismatch"
        );

        super::dsv2_lite_router_logits_into(ctx, hidden, gate_weight, &mut self.logits)?;
        unsafe {
            ensure!(
                ffi::cuda_set_device(ctx.device_ordinal as i32) == 0,
                "activate routed MoE device"
            );
            ffi::cublas_init();
        }
        let stream = ctx.stream.cu_stream();
        let (logits, _gl) = self.logits.device_ptr(&ctx.stream);
        let (ids, _gi) = self.ids.device_ptr_mut(&ctx.stream);
        let (weights, _gw) = self.weights.device_ptr_mut(&ctx.stream);
        let (errors, _ge) = self.errors.device_ptr_mut(&ctx.stream);
        let (summary, _gs) = self.summary.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv2_lite_route_logits_cuda(
                logits as _,
                weights as _,
                ids as _,
                errors as _,
                summary as _,
                batch as i32,
                layer_idx as i32,
                stream,
            )
            .result()?;
        }

        let (input, _gx) = hidden.data.device_ptr(&ctx.stream);
        let (zero, _gz) = self.zero.device_ptr(&ctx.stream);
        let (gate_up_weights, _g13) = experts.gate_up.device_ptr(&ctx.stream);
        let (down_weights, _g2) = experts.down.device_ptr(&ctx.stream);
        let (gate_up, gate_up_guard) = self.gate_up.data.device_ptr_mut(&ctx.stream);
        let (activation, activation_guard) = self.activation.data.device_ptr_mut(&ctx.stream);
        let (rows, rows_guard) = self.rows.data.device_ptr_mut(&ctx.stream);
        let [a13, x13, y13, a2, x2, y2] = &mut self.pointers;
        let (a13, _pa13) = a13.device_ptr_mut(&ctx.stream);
        let (x13, _px13) = x13.device_ptr_mut(&ctx.stream);
        let (y13, _py13) = y13.device_ptr_mut(&ctx.stream);
        let (a2, _pa2) = a2.device_ptr_mut(&ctx.stream);
        let (x2, _px2) = x2.device_ptr_mut(&ctx.stream);
        let (y2, _py2) = y2.device_ptr_mut(&ctx.stream);
        let routes = batch * ROUTES_PER_TOKEN;
        unsafe {
            ffi::dsv2_lite_route_pointers_cuda(
                ids as _,
                input as _,
                zero as _,
                gate_up_weights as _,
                down_weights as _,
                gate_up as _,
                activation as _,
                rows as _,
                a13 as _,
                x13 as _,
                y13 as _,
                a2 as _,
                x2 as _,
                y2 as _,
                summary as _,
                batch as i32,
                experts.first_expert,
                self.hidden_dim as i32,
                self.intermediate as i32,
                stream,
            )
            .result()?;
            let code = ffi::dsv2_lite_pointer_gemm_cuda(
                a13 as _,
                x13 as _,
                y13 as _,
                (2 * self.intermediate) as i32,
                self.hidden_dim as i32,
                routes as i32,
                stream,
            );
            ensure!(code == 0, "routed MoE gate/up pointer GEMM error {code}");
        }
        drop((gate_up_guard, activation_guard, rows_guard));
        super::silu_mul_fused_batch_into(ctx, &self.gate_up, &mut self.activation)?;

        let (output, _go) = output.device_ptr_mut(&ctx.stream);
        unsafe {
            let code = ffi::dsv2_lite_pointer_gemm_cuda(
                a2 as _,
                x2 as _,
                y2 as _,
                self.hidden_dim as i32,
                self.intermediate as i32,
                routes as i32,
                stream,
            );
            ensure!(code == 0, "routed MoE down pointer GEMM error {code}");
            ffi::dsv2_lite_route_reduce_cuda(
                rows as _,
                ids as _,
                weights as _,
                errors as _,
                output as _,
                batch as i32,
                experts.first_expert,
                self.hidden_dim as i32,
                stream,
            )
            .result()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_softmax_top6(logits: &[f32], batch: usize) -> (Vec<i32>, Vec<f32>) {
        let mut ids = Vec::with_capacity(batch * ROUTES_PER_TOKEN);
        let mut weights = Vec::with_capacity(batch * ROUTES_PER_TOKEN);
        for scores in logits.chunks_exact(64) {
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities: Vec<_> = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect();
            let sum: f32 = probabilities.iter().sum();
            probabilities.iter_mut().for_each(|value| *value /= sum);
            let mut indexed: Vec<_> = probabilities.into_iter().enumerate().collect();
            indexed.sort_by(|(lhs_idx, lhs), (rhs_idx, rhs)| {
                rhs.partial_cmp(lhs)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| lhs_idx.cmp(rhs_idx))
            });
            indexed.truncate(ROUTES_PER_TOKEN);
            indexed.sort_by_key(|(expert, _)| *expert);
            ids.extend(indexed.iter().map(|(expert, _)| *expert as i32));
            weights.extend(indexed.into_iter().map(|(_, weight)| weight));
        }
        (ids, weights)
    }

    fn route_logits(
        ctx: &DeviceContext,
        logits: &[f32],
        batch: usize,
    ) -> (Vec<i32>, Vec<f32>, Dsv2RouteSummary) {
        let mut scratch = Dsv2RoutedMoeScratch::new(ctx, 4, 3, batch, 2).unwrap();
        scratch.logits = ctx.stream.clone_htod(logits).unwrap();
        scratch.begin_forward(ctx).unwrap();
        let stream = ctx.stream.cu_stream();
        let (logits, _gl) = scratch.logits.device_ptr(&ctx.stream);
        let (ids, _gi) = scratch.ids.device_ptr_mut(&ctx.stream);
        let (weights, _gw) = scratch.weights.device_ptr_mut(&ctx.stream);
        let (errors, _ge) = scratch.errors.device_ptr_mut(&ctx.stream);
        let (summary, _gs) = scratch.summary.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv2_lite_route_logits_cuda(
                logits as _,
                weights as _,
                ids as _,
                errors as _,
                summary as _,
                batch as i32,
                1,
                stream,
            )
            .result()
            .unwrap();
        }
        drop((_gl, _gi, _gw, _ge, _gs));
        let ids = ctx.stream.clone_dtoh(&scratch.ids).unwrap();
        let weights = ctx.stream.clone_dtoh(&scratch.weights).unwrap();
        let summary = scratch.finish_forward(ctx).unwrap();
        (ids, weights, summary)
    }

    fn zero_experts(ctx: &DeviceContext) -> Vec<(DeviceMatrix, DeviceMatrix)> {
        (0..LOCAL_EXPERTS)
            .map(|_| {
                let gate_up = DeviceMatrix::from_host(ctx, &[bf16::ZERO; 24], 6, 4).unwrap();
                let down = DeviceMatrix::from_host(ctx, &[bf16::ZERO; 12], 4, 3).unwrap();
                (gate_up, down)
            })
            .collect()
    }

    #[test]
    fn routed_moe_router_matches_host_at_supported_boundaries() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        for batch in [1, 4, 8] {
            for fixture in 0..3 {
                let logits: Vec<_> = (0..batch)
                    .flat_map(|token| {
                        (0..64).map(move |expert| match fixture {
                            0 => ((expert * 37 + token * 13) % 101) as f32 / 17.0 - 3.0,
                            1 => 0.0,
                            _ if expert < 5 => 10.0 - expert as f32,
                            _ if expert == 5 => 1.0,
                            _ if expert == 6 => 1.0 + 1.0e-5,
                            _ => -(expert as f32),
                        })
                    })
                    .collect();
                let (expected_ids, expected_weights) = host_softmax_top6(&logits, batch);
                let (actual_ids, actual_weights, summary) = route_logits(&ctx, &logits, batch);
                assert_eq!(actual_ids, expected_ids, "batch={batch}, fixture={fixture}");
                for (index, (actual, expected)) in
                    actual_weights.iter().zip(expected_weights).enumerate()
                {
                    assert!(
                        (actual - expected).abs() <= 1.0e-6,
                        "batch={batch}, fixture={fixture}, route={index}: actual={actual}, expected={expected}"
                    );
                }
                assert_eq!(summary.errors, 0);
                assert_eq!(summary.layer_hashes[0], 0);
                assert_ne!(summary.layer_hashes[1], 0);
            }
        }
    }

    #[test]
    fn routed_moe_tracks_rank_ownership_without_host_route_data() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let experts = zero_experts(&ctx);
        let projections = || experts.iter().map(|(gate, down)| (gate, down)).collect();
        let rank0 = Dsv2ExpertPointerTable::new(&ctx, projections(), 0).unwrap();
        let rank1 = Dsv2ExpertPointerTable::new(&ctx, projections(), 32).unwrap();
        let gate_host: Vec<_> = (0..64)
            .flat_map(|expert| {
                [
                    bf16::from_f32(expert as f32 / 64.0),
                    bf16::ZERO,
                    bf16::ZERO,
                    bf16::ZERO,
                ]
            })
            .collect();
        let gate = DeviceMatrix::from_host(&ctx, &gate_host, 64, 4).unwrap();
        let hidden_host: Vec<_> = (0..DSV2_ROUTED_MOE_MAX_ROWS)
            .flat_map(|_| [bf16::ONE, bf16::ZERO, bf16::ZERO, bf16::ZERO])
            .collect();
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&hidden_host).unwrap(),
            hidden_dim: 4,
            seq_len: DSV2_ROUTED_MOE_MAX_ROWS,
        };
        let mut scratch = Dsv2RoutedMoeScratch::new(&ctx, 4, 3, 8, 1).unwrap();
        let mut output = ctx.stream.alloc_zeros::<f32>(8 * 4).unwrap();

        scratch.begin_forward(&ctx).unwrap();
        scratch
            .enqueue_into(&ctx, hidden.as_ref(), &gate, &rank0, &mut output, 0)
            .unwrap();
        let rank0_summary = scratch.finish_forward(&ctx).unwrap();
        assert_eq!(rank0_summary.local_routes, 0);
        assert_eq!(rank0_summary.total_routes, 8 * ROUTES_PER_TOKEN);
        assert_eq!(rank0_summary.errors, 0);
        assert!(
            ctx.stream
                .clone_dtoh(&output)
                .unwrap()
                .iter()
                .all(|&value| value == 0.0)
        );

        scratch.begin_forward(&ctx).unwrap();
        scratch
            .enqueue_into(&ctx, hidden.as_ref(), &gate, &rank1, &mut output, 0)
            .unwrap();
        let rank1_summary = scratch.finish_forward(&ctx).unwrap();
        assert_eq!(rank1_summary.local_routes, 8 * ROUTES_PER_TOKEN);
        assert_eq!(rank1_summary.total_routes, rank0_summary.total_routes);
        assert_eq!(rank1_summary.layer_hashes, rank0_summary.layer_hashes);
        assert_eq!(rank1_summary.errors, 0);
    }
}
