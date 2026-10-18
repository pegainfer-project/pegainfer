use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use crate::ffi::Half;

unsafe extern "C" {
    pub fn glm52_flashinfer_sparse_mla_supported_cuda(heads: i32, supported: *mut i32) -> CUresult;

    pub fn glm52_flashinfer_sparse_mla_fp8_cuda(
        query: *const u8,
        cache: *const u8,
        topk_indices: *const i32,
        seq_lens: *const i32,
        out: *mut Half,
        workspace: *mut u8,
        workspace_bytes: usize,
        tokens: i32,
        heads: i32,
        num_blocks: i32,
        topk: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}
