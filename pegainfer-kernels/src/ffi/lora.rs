use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

// Qwen3 LoRA fused decode kernels.
unsafe extern "C" {
    pub fn lora_pack_b_rows_cuda(
        src: *const Half,
        dst: *mut Half,
        rank: i32,
        max_rank: i32,
        out_dim: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn lora_decode_fused_delta_cuda(
        a_packed: *const Half,
        b_packed: *const Half,
        scales: *const f32,
        token_slots: *const i32,
        input: *const Half,
        out: *mut Half,
        batch: i32,
        max_loras: i32,
        max_rank: i32,
        rank: i32,
        in_dim: i32,
        out_dim: i32,
        out_hidden_dim: i32,
        row_offset: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn lora_decode_fused_delta_group3_cuda(
        a0: *const Half,
        b0: *const Half,
        scales0: *const f32,
        out0: *mut Half,
        rank0: i32,
        out_dim0: i32,
        out_hidden_dim0: i32,
        a1: *const Half,
        b1: *const Half,
        scales1: *const f32,
        out1: *mut Half,
        rank1: i32,
        out_dim1: i32,
        out_hidden_dim1: i32,
        a2: *const Half,
        b2: *const Half,
        scales2: *const f32,
        out2: *mut Half,
        rank2: i32,
        out_dim2: i32,
        out_hidden_dim2: i32,
        token_slots: *const i32,
        input: *const Half,
        batch: i32,
        max_loras: i32,
        max_rank: i32,
        in_dim: i32,
        stream: CUstream,
    ) -> CUresult;
}
