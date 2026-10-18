use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

unsafe extern "C" {
    pub fn glm52_deepgemm_paged_mqa_metadata_cuda(
        context_lens: *mut i32,
        schedule_metadata: *mut i32,
        batch_size: i32,
        next_n: i32,
        block_kv: i32,
        num_sms: i32,
        is_context_lens_2d: bool,
        is_varlen: bool,
        indices_ptr: *const i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_mqa_logits_unpaged_cuda(
        q_fp8: *const u8,
        k_fp8: *const u8,
        k_scale: *const f32,
        weights: *const f32,
        cu_seqlen_ks: *const i32,
        cu_seqlen_ke: *const i32,
        logits: *mut std::ffi::c_void,
        seq_q: i32,
        seq_kv: i32,
        logits_stride: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn glm52_deepgemm_paged_mqa_logits_cuda(
        q: *const std::ffi::c_void,
        kv_cache: *const std::ffi::c_void,
        kv_cache_stride_bytes: i64,
        weights: *const std::ffi::c_void,
        context_lens: *const i32,
        logits: *mut std::ffi::c_void,
        block_table: *const i32,
        indices: *const i32,
        schedule_meta: *mut i32,
        batch_size: i32,
        next_n: i32,
        num_heads: i32,
        head_dim: i32,
        num_kv_blocks: i32,
        block_kv: i32,
        is_context_lens_2d: bool,
        is_varlen: bool,
        logits_stride: i32,
        block_table_stride: i32,
        num_sms: i32,
        q_elem_size: i32,
        kv_elem_size: i32,
        weights_elem_size: i32,
        kv_scales_elem_size: i32,
        stream: CUstream,
    ) -> CUresult;
}
