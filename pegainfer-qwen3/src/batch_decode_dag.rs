//! Eager decode DAG builder.
//!
//! Each method is both the model's executable forward step and the metadata
//! contract used by `qwen3_model_report`. The model-level report therefore
//! observes the same op sequence that decode executes.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::kv_pool::KvLayout;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::PagedDecodeCallSpec;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::PagedDecodePath;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::embedding_batch_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::fused_add_rms_norm_batch_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::gemm_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::gemm_rows_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::qk_norm_rope_batch_decode_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::rms_norm_batch_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::silu_mul_fused_batch_call;
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_spec::{self};
#[cfg(feature = "kernel-call-trace")]
use pegainfer_core::ops::call_trace;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::tensor::AxisTag;
use pegainfer_kernels::tensor::Hidden;
use pegainfer_kernels::tensor::InDim;
use pegainfer_kernels::tensor::Intermediate;
use pegainfer_kernels::tensor::OutTotal;
use pegainfer_kernels::tensor::QDim;
use pegainfer_kernels::tensor::Vocab;

use crate::batch_decode_buffers::BatchDecodeBuffers;
use crate::batch_decode_buffers::DecodeAttentionPath;
use crate::weights::Qwen3Model;

#[cfg(feature = "kernel-call-trace")]
pub(crate) type DagLabel = String;
#[cfg(not(feature = "kernel-call-trace"))]
pub(crate) type DagLabel = ();

pub(crate) struct BatchDecodeDag<'a> {
    model: &'a Qwen3Model,
    kv_buffer: &'a CudaSlice<bf16>,
    layout: &'a KvLayout,
    batch_size: usize,
    attention_path: DecodeAttentionPath,
}

#[cfg_attr(not(feature = "kernel-call-trace"), allow(unused_variables))]
impl<'a> BatchDecodeDag<'a> {
    pub(crate) fn new(
        model: &'a Qwen3Model,
        kv_buffer: &'a CudaSlice<bf16>,
        layout: &'a KvLayout,
        batch_size: usize,
        attention_path: DecodeAttentionPath,
    ) -> Self {
        Self {
            model,
            kv_buffer,
            layout,
            batch_size,
            attention_path,
        }
    }

    pub(crate) fn embedding(
        &self,
        label: DagLabel,
        token_ids: &CudaSlice<u32>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(embedding_batch_call(
            label,
            self.model.embed_tokens.rows,
            self.model.embed_tokens.cols,
            out.seq_len,
        ));
        pegainfer_kernels::ops::embedding_batch(
            &self.model.ctx,
            &self.model.embed_tokens,
            token_ids,
            out,
        )
    }

    pub(crate) fn rms_norm(
        &self,
        label: DagLabel,
        x: &HiddenStates,
        weight: &DeviceVec,
        out: &mut HiddenStates,
    ) {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(rms_norm_batch_call::<Hidden>(
            label,
            x.hidden_dim,
            x.seq_len,
            self.model.config.rms_norm_eps,
        ));
        pegainfer_kernels::ops::rms_norm_batch_into(
            &self.model.ctx,
            x,
            weight,
            self.model.config.rms_norm_eps,
            out,
        );
    }

    pub(crate) fn fused_add_rms_norm(
        &self,
        label: DagLabel,
        hidden: &mut HiddenStates,
        residual: &HiddenStates,
        weight: &DeviceVec,
        out: &mut HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(fused_add_rms_norm_batch_call::<Hidden>(
            label,
            hidden.hidden_dim,
            hidden.seq_len,
            self.model.config.rms_norm_eps,
        ));
        pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.model.ctx,
            hidden,
            residual,
            weight,
            self.model.config.rms_norm_eps,
            out,
        )
    }

    // `Out` labels the kernel-call-trace record; unused without the feature.
    #[cfg_attr(
        not(feature = "kernel-call-trace"),
        allow(clippy::extra_unused_type_parameters)
    )]
    pub(crate) fn gemm_rows<Out: AxisTag>(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        row_offset: usize,
        rows: usize,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(gemm_rows_call::<Out>(
            label,
            weight.rows,
            weight.cols,
            rows,
            row_offset,
            x.seq_len,
        ));
        pegainfer_kernels::ops::gemm_rows_into(&self.model.ctx, weight, row_offset, rows, x, out);
    }

    // `Out`/`In` label the kernel-call-trace record; unused without the feature.
    #[cfg_attr(
        not(feature = "kernel-call-trace"),
        allow(clippy::extra_unused_type_parameters)
    )]
    fn gemm<Out: AxisTag, In: AxisTag>(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(gemm_call::<Out, In>(
            label,
            weight.rows,
            weight.cols,
            x.seq_len,
        ));
        pegainfer_kernels::ops::gemm_into(&self.model.ctx, weight, x, out);
    }

    pub(crate) fn qk_norm_rope(
        &self,
        label: DagLabel,
        q: &mut HiddenStates,
        k: &mut HiddenStates,
        q_norm: &DeviceVec,
        k_norm: &DeviceVec,
        positions: &CudaSlice<i32>,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(qk_norm_rope_batch_decode_call(
            label,
            q.hidden_dim,
            k.hidden_dim,
            self.batch_size,
            self.model.cos_cache.len / self.model.config.head_dim,
            self.model.local_num_attention_heads(),
            self.model.local_num_key_value_heads(),
            self.model.config.head_dim,
            self.model.config.rms_norm_eps,
        ));
        pegainfer_kernels::ops::qk_norm_rope_batch_decode_into(
            &self.model.ctx,
            q,
            k,
            0,
            self.batch_size,
            q_norm,
            k_norm,
            &self.model.cos_cache,
            &self.model.sin_cache,
            positions,
            self.model.local_num_attention_heads(),
            self.model.local_num_key_value_heads(),
            self.model.config.head_dim,
            self.model.config.rms_norm_eps,
        )
    }

    pub(crate) fn qkv_proj(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm::<OutTotal, Hidden>(label, weight, x, out);
    }

    pub(crate) fn split_qkv(
        &self,
        label: DagLabel,
        qkv: &HiddenStates,
        q: &mut HiddenStates,
        k: &mut HiddenStates,
        v: &mut HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(call_spec::split_qkv_call(
            label,
            q.hidden_dim,
            k.hidden_dim,
            qkv.seq_len,
        ));
        pegainfer_kernels::ops::split_qkv_into(&self.model.ctx, qkv, q, k, v)
    }

    pub(crate) fn paged_decode_attention(
        &self,
        label: DagLabel,
        layer_idx: usize,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(call_spec::paged_decode_attention_call(
            label,
            PagedDecodeCallSpec {
                batch_size: self.batch_size,
                total_pages: self.kv_buffer.len() / self.layout.page_stride,
                num_layers: self.layout.num_layers,
                page_size: self.layout.page_size,
                q_dim: bufs.q.hidden_dim,
                kv_dim: bufs.k.hidden_dim,
                num_q_heads: self.model.local_num_attention_heads(),
                num_kv_heads: self.layout.num_kv_heads,
                head_dim: self.layout.head_dim,
                kv_len: call_trace::decode_kv_len().unwrap_or(0),
                path: match self.attention_path {
                    DecodeAttentionPath::NonPartition => PagedDecodePath::NonPartition,
                    DecodeAttentionPath::SplitKv => {
                        let (chunk_size, cap) = bufs.resolved_split_kv();
                        PagedDecodePath::SplitKv { chunk_size, cap }
                    }
                },
            },
        ));

        match self.attention_path {
            DecodeAttentionPath::NonPartition => {
                pegainfer_kernels::ops::paged_attention_batch_decode_into(
                    &self.model.ctx,
                    &bufs.q,
                    &bufs.k,
                    &bufs.v,
                    self.kv_buffer,
                    &self.layout.kernel_layout(),
                    layer_idx,
                    &bufs.page_indices_d,
                    &bufs.page_indptr_d,
                    &bufs.last_page_len_d,
                    &bufs.positions_d,
                    &bufs.request_indices_d,
                    &bufs.kv_tile_indices_d,
                    &bufs.kv_chunk_size_d,
                    &mut bufs.attn_out,
                    self.model.local_num_attention_heads(),
                    self.batch_size,
                )
            }
            DecodeAttentionPath::SplitKv => {
                pegainfer_kernels::ops::paged_attention_batch_decode_split_kv_into(
                    &self.model.ctx,
                    &bufs.q,
                    &bufs.k,
                    &bufs.v,
                    0,
                    self.kv_buffer,
                    &self.layout.kernel_layout(),
                    layer_idx,
                    &bufs.page_indices_d,
                    &bufs.page_indptr_d,
                    &bufs.last_page_len_d,
                    &bufs.positions_d,
                    &bufs.request_indices_d,
                    &bufs.split_request_indices_d,
                    &bufs.split_kv_tile_indices_d,
                    &bufs.split_kv_chunk_size_d,
                    &bufs.split_o_indptr_d,
                    &bufs.split_block_valid_mask_d,
                    &mut bufs.split_tmp_v,
                    &mut bufs.split_tmp_s,
                    bufs.split_padded_slots,
                    &mut bufs.attn_out,
                    self.model.local_num_attention_heads(),
                    self.batch_size,
                )
            }
        }
    }

    pub(crate) fn all_reduce_hidden(
        &self,
        label: DagLabel,
        hidden: &mut HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(call_spec::all_reduce_hidden_call(
            label,
            hidden.hidden_dim,
            hidden.seq_len,
        ));
        self.model.all_reduce_hidden_untraced(hidden)
    }

    pub(crate) fn o_proj(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm::<Hidden, QDim>(label, weight, x, out);
    }

    pub(crate) fn mlp_gate_proj(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm_rows::<Intermediate>(label, weight, 0, out.hidden_dim, x, out);
    }

    pub(crate) fn mlp_up_proj(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm_rows::<Intermediate>(label, weight, out.hidden_dim, out.hidden_dim, x, out);
    }

    pub(crate) fn silu_mul_split(
        &self,
        label: DagLabel,
        gate: &HiddenStates,
        up: &HiddenStates,
        out: &mut HiddenStates,
    ) -> Result<()> {
        #[cfg(feature = "kernel-call-trace")]
        Self::record(silu_mul_fused_batch_call(
            label,
            gate.hidden_dim,
            gate.seq_len,
        ));
        pegainfer_kernels::ops::silu_mul_batch_into(&self.model.ctx, gate, up, out)
    }

    pub(crate) fn down_proj(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm::<Hidden, Intermediate>(label, weight, x, out);
    }

    pub(crate) fn lm_head(
        &self,
        label: DagLabel,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
    ) {
        self.gemm::<Vocab, InDim>(label, weight, x, out);
    }

    #[cfg(feature = "kernel-call-trace")]
    fn record(call: pegainfer_kernels::tensor::KernelCall) {
        if call_trace::is_enabled() {
            call_trace::record_call(call);
        }
    }
}
