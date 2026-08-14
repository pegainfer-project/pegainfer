use pegainfer_kernels::tensor::AxisSpec;
use pegainfer_kernels::tensor::AxisTag;
use pegainfer_kernels::tensor::Batch;
use pegainfer_kernels::tensor::BatchPlusOne;
use pegainfer_kernels::tensor::Bf16;
use pegainfer_kernels::tensor::Contiguous1D;
use pegainfer_kernels::tensor::F32;
use pegainfer_kernels::tensor::HeadDim;
use pegainfer_kernels::tensor::Hidden;
use pegainfer_kernels::tensor::HiddenStatesLayout;
use pegainfer_kernels::tensor::I32;
use pegainfer_kernels::tensor::InDim;
use pegainfer_kernels::tensor::Inter2;
use pegainfer_kernels::tensor::Intermediate;
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kernels::tensor::Kv;
use pegainfer_kernels::tensor::KvDim;
use pegainfer_kernels::tensor::KvHead;
use pegainfer_kernels::tensor::Layer;
use pegainfer_kernels::tensor::OutDim;
use pegainfer_kernels::tensor::OutTotal;
use pegainfer_kernels::tensor::Page;
use pegainfer_kernels::tensor::PageSlot;
use pegainfer_kernels::tensor::PagedKvPageFirst;
use pegainfer_kernels::tensor::PosInPage;
use pegainfer_kernels::tensor::QDim;
use pegainfer_kernels::tensor::RopeDim;
use pegainfer_kernels::tensor::RowMajor2D;
use pegainfer_kernels::tensor::Seq;
use pegainfer_kernels::tensor::TensorSpec;
use pegainfer_kernels::tensor::Tile;
use pegainfer_kernels::tensor::Token;
use pegainfer_kernels::tensor::U32;
use pegainfer_kernels::tensor::Vocab;

/// A traced decode-attention call's kernel path; `SplitKv` carries its resolved chunk + cap.
#[derive(Clone, Copy, Debug)]
pub enum PagedDecodePath {
    NonPartition,
    SplitKv { chunk_size: usize, cap: usize },
}

impl PagedDecodePath {
    fn label(self) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitKv { .. } => "split_kv".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PagedDecodeCallSpec {
    pub batch_size: usize,
    pub total_pages: usize,
    pub num_layers: usize,
    pub page_size: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
    pub path: PagedDecodePath,
}

pub fn embedding_batch_call(
    label: impl Into<String>,
    vocab: usize,
    hidden: usize,
    batch: usize,
) -> KernelCall {
    KernelCall::new("embedding_batch", label)
        .input("weight", embed_table(vocab, hidden))
        .input("token_ids", token_ids(batch))
        .output("out", hidden_batch::<Hidden>(hidden, batch))
}

pub fn rms_norm_batch_call<A: AxisTag>(
    label: impl Into<String>,
    dim: usize,
    batch: usize,
    eps: f32,
) -> KernelCall {
    KernelCall::new("rms_norm_batch", label)
        .input("x", hidden_batch::<A>(dim, batch))
        .input("weight", vector::<A, Bf16>(dim))
        .output("out", hidden_batch::<A>(dim, batch))
        .attr("eps", eps.to_string())
}

pub fn fused_add_rms_norm_batch_call<A: AxisTag>(
    label: impl Into<String>,
    dim: usize,
    batch: usize,
    eps: f32,
) -> KernelCall {
    KernelCall::new("fused_add_rms_norm_batch", label)
        .input("hidden", hidden_batch::<A>(dim, batch))
        .input("residual", hidden_batch::<A>(dim, batch))
        .input("weight", vector::<A, Bf16>(dim))
        .output("out", hidden_batch::<A>(dim, batch))
        .attr("eps", eps.to_string())
}

pub fn gemm_rows_call<Out: AxisTag>(
    label: impl Into<String>,
    weight_out_total: usize,
    in_dim: usize,
    rows: usize,
    row_offset: usize,
    batch: usize,
) -> KernelCall {
    KernelCall::new("gemm_rows", label)
        .input("weight", weight_matrix_total(weight_out_total, in_dim))
        .input("x", hidden_batch::<Hidden>(in_dim, batch))
        .output("out", hidden_batch::<Out>(rows, batch))
        .attr("row_offset", row_offset.to_string())
        .attr("rows", rows.to_string())
}

pub fn gemm_call<Out: AxisTag, In: AxisTag>(
    label: impl Into<String>,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
) -> KernelCall {
    KernelCall::new("gemm", label)
        .input("weight", weight_matrix(out_dim, in_dim))
        .input("x", hidden_batch::<In>(in_dim, batch))
        .output("out", hidden_batch::<Out>(out_dim, batch))
}

pub fn qk_norm_rope_batch_decode_call(
    label: impl Into<String>,
    q_dim: usize,
    kv_dim: usize,
    batch: usize,
    rope_seq: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> KernelCall {
    KernelCall::new("qk_norm_rope_batch_decode", label)
        .input("q", hidden_batch::<QDim>(q_dim, batch))
        .input("k", hidden_batch::<KvDim>(kv_dim, batch))
        .input("q_norm", vector::<HeadDim, Bf16>(head_dim))
        .input("k_norm", vector::<HeadDim, Bf16>(head_dim))
        .input("cos_cache", rope_cache(rope_seq, head_dim))
        .input("sin_cache", rope_cache(rope_seq, head_dim))
        .input("positions", meta_i32::<Batch>(batch))
        .output("q", hidden_batch::<QDim>(q_dim, batch))
        .output("k", hidden_batch::<KvDim>(kv_dim, batch))
        .attr("num_q_heads", num_q_heads.to_string())
        .attr("num_kv_heads", num_kv_heads.to_string())
        .attr("head_dim", head_dim.to_string())
        .attr("eps", eps.to_string())
}

pub fn paged_decode_attention_call(
    label: impl Into<String>,
    spec: PagedDecodeCallSpec,
) -> KernelCall {
    let mut call = KernelCall::new("paged_decode_attention", label)
        .input("q", hidden_batch::<QDim>(spec.q_dim, spec.batch_size))
        .input("k", hidden_batch::<KvDim>(spec.kv_dim, spec.batch_size))
        .input("v", hidden_batch::<KvDim>(spec.kv_dim, spec.batch_size))
        .input("kv_buffer", paged_kv(spec))
        .input("page_indices", meta_i32::<PageSlot>(spec.total_pages))
        .input("page_indptr", meta_i32::<BatchPlusOne>(spec.batch_size + 1))
        .input("last_page_len", meta_i32::<Batch>(spec.batch_size))
        .input("positions", meta_i32::<Batch>(spec.batch_size))
        .input("request_indices", meta_i32::<Batch>(spec.batch_size))
        .output("out", hidden_batch::<QDim>(spec.q_dim, spec.batch_size))
        .attr("num_q_heads", spec.num_q_heads.to_string())
        .attr("num_kv_heads", spec.num_kv_heads.to_string())
        .attr("head_dim", spec.head_dim.to_string())
        .attr("page_size", spec.page_size.to_string())
        .attr("kv_len", spec.kv_len.to_string())
        .attr("variant", spec.path.label());

    match spec.path {
        PagedDecodePath::SplitKv { chunk_size, cap } => {
            let padded_slots = spec.batch_size * cap;
            call = call
                .attr("split_chunk_size", chunk_size.to_string())
                .attr("split_max_chunks", cap.to_string())
                .input("split_request_indices", meta_i32::<PageSlot>(padded_slots))
                .input("split_kv_tile_indices", meta_i32::<PageSlot>(padded_slots))
                .input("split_kv_chunk_size", meta_i32::<Tile>(1))
                .input(
                    "split_o_indptr",
                    meta_i32::<BatchPlusOne>(spec.batch_size + 1),
                )
                .input("split_block_valid_mask", meta_u8::<PageSlot>(padded_slots))
                .input(
                    "split_tmp_v",
                    TensorSpec::new::<Bf16, Contiguous1D>([
                        AxisSpec::new::<PageSlot>(padded_slots),
                        AxisSpec::new::<QDim>(spec.q_dim),
                    ]),
                )
                .input(
                    "split_tmp_s",
                    TensorSpec::new::<F32, Contiguous1D>([
                        AxisSpec::new::<PageSlot>(padded_slots),
                        AxisSpec::new::<HeadDim>(spec.num_q_heads),
                    ]),
                );
        }
        PagedDecodePath::NonPartition => {
            call = call
                .input("kv_tile_indices", meta_i32::<Tile>(spec.batch_size))
                .input("kv_chunk_size", meta_i32::<Batch>(spec.batch_size));
        }
    }

    call
}

pub fn silu_mul_fused_batch_call(
    label: impl Into<String>,
    inter: usize,
    batch: usize,
) -> KernelCall {
    KernelCall::new("silu_mul_fused_batch", label)
        .input("gate_up", hidden_batch::<Inter2>(2 * inter, batch))
        .output("out", hidden_batch::<Intermediate>(inter, batch))
}

pub fn split_qkv_call(
    label: impl Into<String>,
    q_dim: usize,
    kv_dim: usize,
    batch: usize,
) -> KernelCall {
    KernelCall::new("split_qkv", label)
        .input("qkv", hidden_batch::<OutTotal>(q_dim + 2 * kv_dim, batch))
        .output("q", hidden_batch::<QDim>(q_dim, batch))
        .output("k", hidden_batch::<KvDim>(kv_dim, batch))
        .output("v", hidden_batch::<KvDim>(kv_dim, batch))
}

pub fn all_reduce_hidden_call(label: impl Into<String>, hidden: usize, batch: usize) -> KernelCall {
    KernelCall::new("all_reduce_hidden", label)
        .input("x", hidden_batch::<Hidden>(hidden, batch))
        .output("out", hidden_batch::<Hidden>(hidden, batch))
        .attr("tp_world_size", 1.to_string())
        .attr("no_op", true.to_string())
}

fn hidden_batch<A: AxisTag>(dim: usize, batch: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, HiddenStatesLayout>([
        AxisSpec::new::<A>(dim),
        AxisSpec::new::<Batch>(batch),
    ])
}

fn weight_matrix(out: usize, in_dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, RowMajor2D>([
        AxisSpec::new::<OutDim>(out),
        AxisSpec::new::<InDim>(in_dim),
    ])
}

fn weight_matrix_total(out_total: usize, in_dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, RowMajor2D>([
        AxisSpec::new::<OutTotal>(out_total),
        AxisSpec::new::<InDim>(in_dim),
    ])
}

fn vector<A: AxisTag, D: pegainfer_kernels::tensor::DTypeTag>(dim: usize) -> TensorSpec {
    TensorSpec::new::<D, Contiguous1D>([AxisSpec::new::<A>(dim)])
}

fn embed_table(vocab: usize, hidden: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, RowMajor2D>([
        AxisSpec::new::<Vocab>(vocab),
        AxisSpec::new::<Hidden>(hidden),
    ])
}

fn token_ids(batch: usize) -> TensorSpec {
    TensorSpec::new::<U32, Contiguous1D>([AxisSpec::new::<Token>(batch)])
}

fn rope_cache(seq: usize, head_dim: usize) -> TensorSpec {
    TensorSpec::new::<Bf16, Contiguous1D>([
        AxisSpec::new::<Seq>(seq),
        AxisSpec::new::<RopeDim>(head_dim),
    ])
}

fn paged_kv(spec: PagedDecodeCallSpec) -> TensorSpec {
    TensorSpec::new::<Bf16, PagedKvPageFirst>([
        AxisSpec::new::<Page>(spec.total_pages),
        AxisSpec::new::<Layer>(spec.num_layers),
        AxisSpec::new::<Kv>(2),
        AxisSpec::new::<PosInPage>(spec.page_size),
        AxisSpec::new::<KvHead>(spec.num_kv_heads),
        AxisSpec::new::<HeadDim>(spec.head_dim),
    ])
}

fn meta_i32<A: AxisTag>(size: usize) -> TensorSpec {
    TensorSpec::new::<I32, Contiguous1D>([AxisSpec::new::<A>(size)])
}

fn meta_u8<A: AxisTag>(size: usize) -> TensorSpec {
    TensorSpec::new::<pegainfer_kernels::tensor::U8, Contiguous1D>([AxisSpec::new::<A>(size)])
}
