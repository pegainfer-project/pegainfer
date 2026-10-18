use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_INDEXER_TOPK_MAX_K: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52IndexerTopK {
    pub num_rows: usize,
    pub top_k: usize,
    pub max_len: usize,
}

impl Glm52IndexerTopK {
    fn validate(self) -> Result<()> {
        ensure!(
            self.num_rows > 0,
            "GLM5.2 indexer top-k num_rows must be positive"
        );
        ensure!(
            self.top_k > 0 && self.top_k <= GLM52_INDEXER_TOPK_MAX_K,
            "GLM5.2 indexer top-k must be in 1..={GLM52_INDEXER_TOPK_MAX_K}, got {}",
            self.top_k
        );
        ensure!(
            self.max_len > 0,
            "GLM5.2 indexer top-k max_len must be positive"
        );
        Ok(())
    }
}

/// FlashInfer deterministic decode top-k. Matches TokenSpeed's
/// `deterministic_decode_topk` contract: deterministic=true,
/// TopKTieBreak::Small, dsa_graph_safe=true.
///
/// - `logits`: `[num_rows, max_len]` f32, row-major.
/// - `lengths`: `[num_rows]` i32, valid logits count per row.
/// - `output_indices`: `[num_rows, top_k]` i32 output.
/// - `output_values`: `[num_rows, top_k]` f32 output (may be null/dummy).
///
/// `FilteredTopK` uses dynamic shared memory internally; no external scratch
/// buffer is needed (unlike the `RadixTopKMultiCTA` path which the old
/// `TopKDispatch` wrapper used to fall back to).
pub fn glm52_flashinfer_topk_2048_launch(
    ctx: &DeviceContext,
    contract: Glm52IndexerTopK,
    logits: &impl DevicePtr<f32>,
    lengths: &impl DevicePtr<i32>,
    output_indices: &mut CudaSlice<i32>,
    output_values: &mut CudaSlice<f32>,
) -> Result<()> {
    contract.validate()?;
    ensure!(
        logits.len() >= contract.num_rows * contract.max_len,
        "GLM5.2 indexer top-k logits too small: have {}, need {}",
        logits.len(),
        contract.num_rows * contract.max_len
    );
    ensure!(
        lengths.len() >= contract.num_rows,
        "GLM5.2 indexer top-k lengths too small: have {}, need {}",
        lengths.len(),
        contract.num_rows
    );
    ensure!(
        output_indices.len() >= contract.num_rows * contract.top_k,
        "GLM5.2 indexer top-k output_indices too small: have {}, need {}",
        output_indices.len(),
        contract.num_rows * contract.top_k
    );
    ensure!(
        output_values.len() >= contract.num_rows * contract.top_k,
        "GLM5.2 indexer top-k output_values too small: have {}, need {}",
        output_values.len(),
        contract.num_rows * contract.top_k
    );

    let (logits_ptr, _logits_guard) = logits.device_ptr(&ctx.stream);
    let (lengths_ptr, _lengths_guard) = lengths.device_ptr(&ctx.stream);
    let (output_indices_ptr, _output_indices_guard) = output_indices.device_ptr_mut(&ctx.stream);
    let (output_values_ptr, _output_values_guard) = output_values.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_flashinfer_topk_2048_cuda(
            logits_ptr as *const f32,
            output_indices_ptr as *mut i32,
            output_values_ptr as *mut f32,
            lengths_ptr as *const i32,
            contract.num_rows as i32,
            contract.top_k as i32,
            contract.max_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    ensure!(
        result == 0,
        "GLM5.2 FlashInfer top-k 2048 launch failed with error {result}{}",
        crate::ops::ffi_exception_message(result)
    );
    Ok(())
}
