//! Device buffers a decode step reads and writes: the per-slot state pools that
//! survive a step, and the scratch arena that does not.
//!
//! Everything here is allocated once, at the executor's row capacity, and never
//! reallocated — a batched kernel launched at bucket `b` addresses the leading
//! `b` rows of a `[capacity, ...]` slab, so one arena serves every bucket and
//! every device pointer is stable for CUDA-Graph capture.
//!
//! Two families of state need a successor buffer the kernel may not alias: the
//! KDA recurrent state and the KDA convolution window. Copying the successor
//! back would cost more than the step itself at width, so each is a **pair of
//! slabs read and written by step parity**: even steps read slab 0 and write
//! slab 1, odd steps the other way round. That makes the step body
//! parity-dependent, which is why a bucket holds one graph per parity.

use anyhow::Context;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3_CONV_WIDTH;
use pegainfer_kernels::ops::K3_KDA_HEAD_DIM;
use pegainfer_kernels::ops::K3_KDA_HEADS;
use pegainfer_kernels::ops::K3_MLA_HEADS;
use pegainfer_kernels::ops::K3_MOE_QUANT_GROUP;
use pegainfer_kernels::ops::K3_QK_DIM;
use pegainfer_kernels::ops::K3_ROUTER_TOPK;
use pegainfer_kernels::ops::K3_V_DIM;
use pegainfer_kernels::ops::argmax_batch_bf16_split_partials_len;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use crate::config::K3_ATTN_INNER;
use crate::config::K3_DENSE_INTERMEDIATE;
use crate::config::K3_EXPERT_INTERMEDIATE;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HIDDEN;
use crate::config::K3_KV_A_OUT;
use crate::config::K3_KV_B_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_Q_B_OUT;
use crate::config::K3_Q_LORA_RANK;
use crate::config::K3_QK_ROPE_HEAD_DIM;
use crate::config::K3_ROUTED_EXPERT_HIDDEN;
use crate::config::K3_SHARED_INTERMEDIATE;
use crate::config::K3_VOCAB;
use crate::config::K3LayerKind;
use crate::config::k3_layer_kind;

/// Width of the KDA fused q|k|v|gate projection.
pub(crate) const K3_KDA_FUSED: usize = 4 * K3_ATTN_INNER;
/// Width of the padded KDA beta + low-rank-gate projection.
pub(crate) const K3_KDA_WSM: usize = K3_KDA_HEADS + K3_KDA_HEAD_DIM;
/// The same, rounded up the way the certified projection pads it.
pub(crate) const K3_KDA_WSM_PADDED: usize = K3_KDA_WSM.next_multiple_of(64);
/// Width of the MLA fused q_a | kv_a | k_rope | gate projection.
pub(crate) const K3_MLA_FUSED: usize =
    K3_Q_LORA_RANK + K3_KV_LORA_RANK + K3_QK_ROPE_HEAD_DIM + K3_ATTN_INNER;
/// Carried convolution window slots.
pub(crate) const K3_CONV_STATE: usize = K3_CONV_WIDTH - 1;
/// Elements of one row's KDA recurrent state.
pub(crate) const K3_KDA_STATE: usize = K3_KDA_HEADS * K3_KDA_HEAD_DIM * K3_KDA_HEAD_DIM;
/// Per-slot MLA cache row widths.
pub(crate) const K3_MLA_K_ROW: usize = K3_MLA_HEADS * K3_QK_DIM;
pub(crate) const K3_MLA_V_ROW: usize = K3_MLA_HEADS * K3_V_DIM;

/// One KDA layer's per-slot state: recurrent matrix plus the three convolution
/// windows, each as a parity pair.
pub(crate) struct K3KdaState {
    /// `[2][rows, heads, head_dim, head_dim]` f32.
    pub(crate) recurrent: [CudaSlice<f32>; 2],
    /// `[2][3][rows, width - 1, inner]` bf16, one window per q/k/v stream.
    pub(crate) conv: [[CudaSlice<bf16>; 3]; 2],
}

/// One MLA layer's per-slot slot-indexed cache. Each slot owns a fixed
/// `max_ctx` window, so the buffers are the batched kernel's `[rows, cap, w]`
/// and, seen as `[rows * cap, w]`, the indexed row write's destination.
pub(crate) struct K3MlaState {
    pub(crate) k_cache: HiddenStates,
    pub(crate) v_cache: HiddenStates,
}

pub(crate) enum K3LayerState {
    Kda(Box<K3KdaState>),
    Mla(Box<K3MlaState>),
}

/// Everything about a slot that outlives a step.
pub(crate) struct K3StatePool {
    pub(crate) rows: usize,
    pub(crate) max_ctx: usize,
    pub(crate) layers: Vec<K3LayerState>,
    /// Attention-residual snapshot history, `[rows, blocks, hidden]` bf16.
    pub(crate) blocks: CudaSlice<bf16>,
    pub(crate) block_count: usize,
    /// Tokens each row has already consumed. Index into its MLA window and,
    /// plus one, its attention context length.
    pub(crate) positions: Vec<usize>,
}

impl K3StatePool {
    pub(crate) fn new(
        ctx: &DeviceContext,
        rows: usize,
        max_ctx: usize,
        num_layers: usize,
        block_count: usize,
    ) -> Result<Self> {
        let stream = &ctx.stream;
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            layers.push(match k3_layer_kind(layer) {
                K3LayerKind::Kda => {
                    let recurrent_len = rows * K3_KDA_STATE;
                    let conv_len = rows * K3_CONV_STATE * K3_ATTN_INNER;
                    let mut recurrent = Vec::with_capacity(2);
                    let mut conv = Vec::with_capacity(2);
                    for _ in 0..2 {
                        recurrent.push(
                            stream
                                .alloc_zeros::<f32>(recurrent_len)
                                .context("alloc K3 KDA recurrent state")?,
                        );
                        conv.push([
                            stream.alloc_zeros::<bf16>(conv_len)?,
                            stream.alloc_zeros::<bf16>(conv_len)?,
                            stream.alloc_zeros::<bf16>(conv_len)?,
                        ]);
                    }
                    let [conv_even, conv_odd] = conv
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("two parities were pushed"));
                    let [recurrent_even, recurrent_odd] = recurrent
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("two parities were pushed"));
                    K3LayerState::Kda(Box::new(K3KdaState {
                        recurrent: [recurrent_even, recurrent_odd],
                        conv: [conv_even, conv_odd],
                    }))
                }
                K3LayerKind::Mla => K3LayerState::Mla(Box::new(K3MlaState {
                    k_cache: HiddenStates {
                        data: stream
                            .alloc_zeros::<bf16>(rows * max_ctx * K3_MLA_K_ROW)
                            .context("alloc K3 MLA key cache")?,
                        hidden_dim: K3_MLA_K_ROW,
                        seq_len: rows * max_ctx,
                    },
                    v_cache: HiddenStates {
                        data: stream
                            .alloc_zeros::<bf16>(rows * max_ctx * K3_MLA_V_ROW)
                            .context("alloc K3 MLA value cache")?,
                        hidden_dim: K3_MLA_V_ROW,
                        seq_len: rows * max_ctx,
                    },
                })),
            });
        }
        Ok(Self {
            rows,
            max_ctx,
            layers,
            blocks: stream
                .alloc_zeros::<bf16>(rows * block_count * K3_HIDDEN)
                .context("alloc K3 attention-residual snapshots")?,
            block_count,
            positions: vec![0; rows],
        })
    }

    /// Zero one row's state everywhere and rewind its position. Both parities
    /// are cleared: which one a step reads depends on the executor's step
    /// counter, not on the row.
    pub(crate) fn reset_row(&mut self, ctx: &DeviceContext, row: usize) -> Result<()> {
        anyhow::ensure!(row < self.rows, "K3 state pool has no row {row}");
        for layer in &mut self.layers {
            match layer {
                K3LayerState::Kda(kda) => {
                    for parity in 0..2 {
                        zero_rows(ctx, &mut kda.recurrent[parity], row, 1, K3_KDA_STATE)?;
                        for stream in &mut kda.conv[parity] {
                            zero_rows(ctx, stream, row, 1, K3_CONV_STATE * K3_ATTN_INNER)?;
                        }
                    }
                }
                K3LayerState::Mla(mla) => {
                    zero_rows(
                        ctx,
                        &mut mla.k_cache.data,
                        row * self.max_ctx,
                        self.max_ctx,
                        K3_MLA_K_ROW,
                    )?;
                    zero_rows(
                        ctx,
                        &mut mla.v_cache.data,
                        row * self.max_ctx,
                        self.max_ctx,
                        K3_MLA_V_ROW,
                    )?;
                }
            }
        }
        zero_rows(ctx, &mut self.blocks, row, 1, self.block_count * K3_HIDDEN)?;
        self.positions[row] = 0;
        Ok(())
    }

    /// Copy one row of `source` into `row` of this pool, taking the KDA state
    /// out of `source_parity` and landing it in `target_parity`. This is how a
    /// finished prefill hands its sequence over to the decode pool.
    pub(crate) fn adopt_row(
        &mut self,
        ctx: &DeviceContext,
        source: &K3StatePool,
        source_row: usize,
        source_parity: usize,
        row: usize,
        target_parity: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            row < self.rows && source_row < source.rows,
            "K3 state pool row out of range"
        );
        anyhow::ensure!(
            source.layers.len() == self.layers.len()
                && source.max_ctx == self.max_ctx
                && source.block_count == self.block_count,
            "K3 state pools disagree on geometry"
        );
        for (target, origin) in self.layers.iter_mut().zip(&source.layers) {
            match (target, origin) {
                (K3LayerState::Kda(target), K3LayerState::Kda(origin)) => {
                    copy_rows(
                        ctx,
                        &origin.recurrent[source_parity],
                        source_row,
                        &mut target.recurrent[target_parity],
                        row,
                        1,
                        K3_KDA_STATE,
                    )?;
                    for (target, origin) in target.conv[target_parity]
                        .iter_mut()
                        .zip(&origin.conv[source_parity])
                    {
                        copy_rows(
                            ctx,
                            origin,
                            source_row,
                            target,
                            row,
                            1,
                            K3_CONV_STATE * K3_ATTN_INNER,
                        )?;
                    }
                }
                (K3LayerState::Mla(target), K3LayerState::Mla(origin)) => {
                    copy_rows(
                        ctx,
                        &origin.k_cache.data,
                        source_row * self.max_ctx,
                        &mut target.k_cache.data,
                        row * self.max_ctx,
                        self.max_ctx,
                        K3_MLA_K_ROW,
                    )?;
                    copy_rows(
                        ctx,
                        &origin.v_cache.data,
                        source_row * self.max_ctx,
                        &mut target.v_cache.data,
                        row * self.max_ctx,
                        self.max_ctx,
                        K3_MLA_V_ROW,
                    )?;
                }
                _ => anyhow::bail!("K3 state pools disagree on layer kinds"),
            }
        }
        copy_rows(
            ctx,
            &source.blocks,
            source_row,
            &mut self.blocks,
            row,
            1,
            self.block_count * K3_HIDDEN,
        )?;
        self.positions[row] = source.positions[source_row];
        Ok(())
    }
}

fn zero_rows<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    ctx: &DeviceContext,
    buffer: &mut CudaSlice<T>,
    first_row: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    let start = first_row * row_width;
    let mut window = buffer.slice_mut(start..start + rows * row_width);
    ctx.stream
        .memset_zeros(&mut window)
        .context("zero a K3 state row")
}

fn copy_rows<T: cudarc::driver::DeviceRepr>(
    ctx: &DeviceContext,
    source: &CudaSlice<T>,
    source_row: usize,
    target: &mut CudaSlice<T>,
    target_row: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    let span = rows * row_width;
    let origin = source.slice(source_row * row_width..source_row * row_width + span);
    let mut destination = target.slice_mut(target_row * row_width..target_row * row_width + span);
    ctx.stream
        .memcpy_dtod(&origin, &mut destination)
        .context("copy a K3 state row")
}

/// Routed-expert chain buffers. Sized from the rank's local expert count and
/// the masked layout's per-expert capacity, both fixed for the executor.
pub(crate) struct K3MoeScratch {
    pub(crate) masked_m: CudaSlice<i32>,
    pub(crate) slot_map: CudaSlice<i32>,
    pub(crate) w13_activation: CudaSlice<u8>,
    pub(crate) w13_scale: CudaSlice<f32>,
    pub(crate) w13_scale_packed: CudaSlice<i32>,
    pub(crate) w13_out: CudaSlice<bf16>,
    pub(crate) w2_activation: CudaSlice<u8>,
    pub(crate) w2_scale: CudaSlice<f32>,
    pub(crate) w2_scale_packed: CudaSlice<i32>,
    pub(crate) w2_out: CudaSlice<bf16>,
}

impl K3MoeScratch {
    fn new(ctx: &DeviceContext, rows: usize, groups: usize, masked_cap: usize) -> Result<Self> {
        let stream = &ctx.stream;
        let masked_rows = groups * masked_cap;
        let latent = K3_ROUTED_EXPERT_HIDDEN;
        let inter = K3_EXPERT_INTERMEDIATE;
        let quant = K3_MOE_QUANT_GROUP;
        Ok(Self {
            masked_m: stream.alloc_zeros(groups)?,
            slot_map: stream.alloc_zeros(rows * K3_ROUTER_TOPK)?,
            w13_activation: stream
                .alloc_zeros(masked_rows * latent)
                .context("alloc K3 W13 activation")?,
            w13_scale: stream.alloc_zeros(groups * (latent / quant) * masked_cap)?,
            w13_scale_packed: stream.alloc_zeros(groups * (latent / (4 * quant)) * masked_cap)?,
            w13_out: stream
                .alloc_zeros(masked_rows * 2 * inter)
                .context("alloc K3 W13 output")?,
            w2_activation: stream
                .alloc_zeros(masked_rows * inter)
                .context("alloc K3 W2 activation")?,
            w2_scale: stream.alloc_zeros(groups * (inter / quant) * masked_cap)?,
            w2_scale_packed: stream.alloc_zeros(groups * (inter / (4 * quant)) * masked_cap)?,
            w2_out: stream
                .alloc_zeros(masked_rows * latent)
                .context("alloc K3 W2 output")?,
        })
    }
}

/// Per-step working buffers. Named after the certified engine's scratch so the
/// launch sequence reads the same way.
pub(crate) struct K3Scratch {
    // Step inputs, refreshed from the host before every step (or graph replay).
    pub(crate) token_ids: CudaSlice<u32>,
    /// Per-row MLA context length, i.e. valid cache slots including this step.
    pub(crate) context_len: CudaSlice<i32>,
    /// Per-row destination of this step's cache write, `row * cap + position`,
    /// or `-1` for a row this step does not own.
    pub(crate) cache_row: CudaSlice<i32>,
    /// `head_row[r * heads + h] = r`, the broadcast of one row's shared rope
    /// half to every MLA head. Static.
    pub(crate) head_row: CudaSlice<i32>,
    // Residual stream.
    pub(crate) hidden: CudaSlice<bf16>,
    pub(crate) prefix: CudaSlice<bf16>,
    pub(crate) mixed: CudaSlice<bf16>,
    pub(crate) prefix2: CudaSlice<bf16>,
    pub(crate) mixed2: CudaSlice<bf16>,
    pub(crate) attn_out: CudaSlice<bf16>,
    pub(crate) mlp_out: CudaSlice<bf16>,
    pub(crate) normed: CudaSlice<bf16>,
    pub(crate) scores: CudaSlice<f32>,
    // KDA.
    pub(crate) kda_gate_partial: CudaSlice<f32>,
    pub(crate) kda_conv_partial: CudaSlice<f32>,
    pub(crate) kda_wsm_partial: CudaSlice<f32>,
    pub(crate) kda_forget_partial: CudaSlice<f32>,
    pub(crate) beta: CudaSlice<bf16>,
    pub(crate) forget_low: CudaSlice<bf16>,
    pub(crate) out_gate: CudaSlice<bf16>,
    pub(crate) conv_x: CudaSlice<bf16>,
    pub(crate) conv_q: CudaSlice<bf16>,
    pub(crate) conv_k: CudaSlice<bf16>,
    pub(crate) conv_v: CudaSlice<bf16>,
    pub(crate) gated: CudaSlice<bf16>,
    // MLA.
    pub(crate) mla_fused_partial: CudaSlice<f32>,
    pub(crate) q_norm: CudaSlice<bf16>,
    pub(crate) kv_a: CudaSlice<bf16>,
    pub(crate) kv_latent: CudaSlice<bf16>,
    pub(crate) kv_norm: CudaSlice<bf16>,
    pub(crate) mla_gate: CudaSlice<bf16>,
    pub(crate) q_partial: CudaSlice<f32>,
    pub(crate) query: CudaSlice<bf16>,
    pub(crate) kv_partial: CudaSlice<f32>,
    pub(crate) kv: CudaSlice<bf16>,
    pub(crate) k_nope: CudaSlice<bf16>,
    pub(crate) k_new: HiddenStates,
    pub(crate) v_new: HiddenStates,
    pub(crate) rope: HiddenStates,
    pub(crate) rope_heads: HiddenStates,
    pub(crate) attn: CudaSlice<bf16>,
    // MLP / MoE.
    pub(crate) hidden_partial: CudaSlice<f32>,
    pub(crate) router_partial: CudaSlice<f32>,
    pub(crate) topk_idx: CudaSlice<i32>,
    pub(crate) topk_weight: CudaSlice<f32>,
    pub(crate) latent_partial: CudaSlice<f32>,
    pub(crate) latent: CudaSlice<bf16>,
    pub(crate) routed_latent: CudaSlice<bf16>,
    pub(crate) routed_latent_norm: CudaSlice<bf16>,
    pub(crate) routed: CudaSlice<bf16>,
    pub(crate) shared: CudaSlice<bf16>,
    pub(crate) shared_partial: CudaSlice<f32>,
    pub(crate) shared_gate: CudaSlice<bf16>,
    pub(crate) shared_up: CudaSlice<bf16>,
    pub(crate) shared_act: CudaSlice<bf16>,
    pub(crate) dense_partial: CudaSlice<f32>,
    pub(crate) dense_gate: CudaSlice<bf16>,
    pub(crate) dense_up: CudaSlice<bf16>,
    pub(crate) dense_act: CudaSlice<bf16>,
    pub(crate) moe: K3MoeScratch,
    // Output.
    pub(crate) logit_partial: CudaSlice<f32>,
    pub(crate) logits: CudaSlice<bf16>,
    pub(crate) argmax_partial_values: CudaSlice<f32>,
    pub(crate) argmax_partial_indices: CudaSlice<i32>,
    pub(crate) argmax_values: CudaSlice<bf16>,
    pub(crate) argmax_indices: CudaSlice<i32>,
}

impl K3Scratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        rows: usize,
        routed_experts: usize,
        groups: usize,
        masked_cap: usize,
    ) -> Result<Self> {
        let stream = &ctx.stream;
        let heads = K3_MLA_HEADS;
        let head_row: Vec<i32> = (0..rows * heads)
            .map(|entry| (entry / heads) as i32)
            .collect();
        let wide = |width: usize| stream.alloc_zeros::<bf16>(rows * width);
        let partial = |width: usize| stream.alloc_zeros::<f32>(rows * width);
        let argmax_partials = argmax_batch_bf16_split_partials_len(rows, K3_VOCAB);
        Ok(Self {
            token_ids: stream.alloc_zeros(rows)?,
            context_len: stream.alloc_zeros(rows)?,
            cache_row: stream.alloc_zeros(rows)?,
            head_row: stream.clone_htod(&head_row)?,
            hidden: wide(K3_HIDDEN)?,
            prefix: wide(K3_HIDDEN)?,
            mixed: wide(K3_HIDDEN)?,
            prefix2: wide(K3_HIDDEN)?,
            mixed2: wide(K3_HIDDEN)?,
            attn_out: wide(K3_HIDDEN)?,
            mlp_out: wide(K3_HIDDEN)?,
            normed: wide(K3_HIDDEN)?,
            scores: stream.alloc_zeros(rows * (crate::config::K3_LAYERS.div_ceil(12) + 1))?,
            kda_gate_partial: partial(K3_KDA_FUSED)?,
            kda_conv_partial: partial(K3_ATTN_INNER)?,
            kda_wsm_partial: partial(K3_KDA_WSM_PADDED)?,
            kda_forget_partial: partial(K3_ATTN_INNER)?,
            beta: wide(K3_KDA_HEADS)?,
            forget_low: wide(K3_HEAD_DIM)?,
            out_gate: wide(K3_ATTN_INNER)?,
            conv_x: wide(K3_ATTN_INNER)?,
            conv_q: wide(K3_ATTN_INNER)?,
            conv_k: wide(K3_ATTN_INNER)?,
            conv_v: wide(K3_ATTN_INNER)?,
            gated: wide(K3_ATTN_INNER)?,
            mla_fused_partial: partial(K3_MLA_FUSED)?,
            q_norm: wide(K3_Q_LORA_RANK)?,
            kv_a: wide(K3_KV_A_OUT)?,
            kv_latent: wide(K3_KV_LORA_RANK)?,
            kv_norm: wide(K3_KV_LORA_RANK)?,
            mla_gate: wide(K3_ATTN_INNER)?,
            q_partial: partial(K3_Q_B_OUT)?,
            query: wide(K3_Q_B_OUT)?,
            kv_partial: partial(K3_KV_B_OUT)?,
            kv: wide(K3_KV_B_OUT)?,
            k_nope: wide(heads * K3_HEAD_DIM)?,
            k_new: HiddenStates {
                data: wide(K3_MLA_K_ROW)?,
                hidden_dim: K3_MLA_K_ROW,
                seq_len: rows,
            },
            v_new: HiddenStates {
                data: wide(K3_MLA_V_ROW)?,
                hidden_dim: K3_MLA_V_ROW,
                seq_len: rows,
            },
            rope: HiddenStates {
                data: wide(K3_QK_ROPE_HEAD_DIM)?,
                hidden_dim: K3_QK_ROPE_HEAD_DIM,
                seq_len: rows,
            },
            rope_heads: HiddenStates {
                data: wide(heads * K3_QK_ROPE_HEAD_DIM)?,
                hidden_dim: K3_QK_ROPE_HEAD_DIM,
                seq_len: rows * heads,
            },
            attn: wide(K3_MLA_V_ROW)?,
            hidden_partial: partial(K3_HIDDEN)?,
            router_partial: partial(routed_experts)?,
            topk_idx: stream.alloc_zeros(rows * K3_ROUTER_TOPK)?,
            topk_weight: stream.alloc_zeros(rows * K3_ROUTER_TOPK)?,
            latent_partial: partial(K3_ROUTED_EXPERT_HIDDEN)?,
            latent: wide(K3_ROUTED_EXPERT_HIDDEN)?,
            routed_latent: wide(K3_ROUTED_EXPERT_HIDDEN)?,
            routed_latent_norm: wide(K3_ROUTED_EXPERT_HIDDEN)?,
            routed: wide(K3_HIDDEN)?,
            shared: wide(K3_HIDDEN)?,
            shared_partial: partial(2 * K3_SHARED_INTERMEDIATE)?,
            shared_gate: wide(K3_SHARED_INTERMEDIATE)?,
            shared_up: wide(K3_SHARED_INTERMEDIATE)?,
            shared_act: wide(K3_SHARED_INTERMEDIATE)?,
            dense_partial: partial(2 * K3_DENSE_INTERMEDIATE)?,
            dense_gate: wide(K3_DENSE_INTERMEDIATE)?,
            dense_up: wide(K3_DENSE_INTERMEDIATE)?,
            dense_act: wide(K3_DENSE_INTERMEDIATE)?,
            moe: K3MoeScratch::new(ctx, rows, groups, masked_cap)?,
            logit_partial: partial(K3_VOCAB).context("alloc K3 logit partial")?,
            logits: wide(K3_VOCAB).context("alloc K3 logits")?,
            argmax_partial_values: stream.alloc_zeros(argmax_partials)?,
            argmax_partial_indices: stream.alloc_zeros(argmax_partials)?,
            argmax_values: stream.alloc_zeros(rows)?,
            argmax_indices: stream.alloc_zeros(rows)?,
        })
    }
}

/// Split a parity pair into the slab this step reads and the one it writes.
/// The two are disjoint halves of the same array, so a shared borrow of one and
/// a unique borrow of the other coexist.
pub(crate) fn parity_pair<T>(pair: &mut [T; 2], parity: usize) -> (&T, &mut T) {
    let (low, high) = pair.split_at_mut(1);
    if parity == 0 {
        (&low[0], &mut high[0])
    } else {
        (&high[0], &mut low[0])
    }
}
