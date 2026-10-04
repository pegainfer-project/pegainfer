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
const ROUTED_EXPERTS: usize = 64;
const LOCAL_EXPERTS: usize = ROUTED_EXPERTS / 2;
pub const DSV2_ROUTED_MOE_MAX_ROWS: usize = 8;

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
            logits: ctx.stream.alloc_zeros(capacity * ROUTED_EXPERTS)?,
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
            hidden.hidden_dim == self.hidden_dim
                && experts.hidden_dim == self.hidden_dim
                && experts.intermediate == self.intermediate,
            "routed MoE hidden or expert shape mismatch"
        );
        ensure!(
            gate_weight.rows == ROUTED_EXPERTS && gate_weight.cols == self.hidden_dim,
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
        dsv2_lite_route_logits_into(
            ctx,
            &self.logits,
            batch,
            layer_idx,
            &mut super::Dsv2LiteRouterOutput {
                topk_weight: &mut self.weights,
                topk_idx: &mut self.ids,
            },
            &mut self.errors,
            &mut self.summary,
        )?;
        let stream = ctx.stream.cu_stream();
        let (ids, _gi) = self.ids.device_ptr(&ctx.stream);
        let (weights, _gw) = self.weights.device_ptr(&ctx.stream);
        let (errors, _ge) = self.errors.device_ptr(&ctx.stream);
        let (summary, _gs) = self.summary.device_ptr_mut(&ctx.stream);

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

pub fn dsv2_lite_route_logits_into(
    ctx: &DeviceContext,
    logits: &CudaSlice<f32>,
    batch: usize,
    layer_idx: usize,
    output: &mut super::Dsv2LiteRouterOutput<'_>,
    errors: &mut CudaSlice<i32>,
    summary: &mut CudaSlice<u64>,
) -> Result<()> {
    ensure!(
        (1..=DSV2_ROUTED_MOE_MAX_ROWS).contains(&batch),
        "invalid routed MoE batch"
    );
    ensure!(
        summary.len() > 3 && layer_idx < summary.len() - 3 && layer_idx <= i32::MAX as usize,
        "routed MoE layer exceeds summary capacity"
    );
    ensure!(
        logits.len() >= batch * ROUTED_EXPERTS
            && output.topk_idx.len() >= batch * ROUTES_PER_TOKEN
            && output.topk_weight.len() >= batch * ROUTES_PER_TOKEN
            && errors.len() >= batch,
        "routed MoE router buffer too small"
    );
    let (logits, _gl) = logits.device_ptr(&ctx.stream);
    let (ids, _gi) = output.topk_idx.device_ptr_mut(&ctx.stream);
    let (weights, _gw) = output.topk_weight.device_ptr_mut(&ctx.stream);
    let (errors, _ge) = errors.device_ptr_mut(&ctx.stream);
    let (summary, _gs) = summary.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::dsv2_lite_route_logits_cuda(
            logits as _,
            weights as _,
            ids as _,
            errors as _,
            summary as _,
            batch as i32,
            layer_idx as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}
