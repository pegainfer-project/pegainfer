// Qwen3.8-Flash-Next (Qwen4 line) QSA sparse-attention kernels. Geometry is
// frozen in `csrc/qwen38fn/qwen38fn_qsa_attn_forward.cu`, which refuses a call
// whose rotary geometry disagrees with what it was compiled for.

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    /// The QSA sparse-attention forward over one layer's captured activations:
    /// the key pass, then one CTA per (query row, head) for the scores, the
    /// weights and the gated context. `normed_keys` is the key pass's workspace.
    pub fn qwen38fn_qsa_attn_forward_cuda(
        queries: *const Half,
        keys: *const Half,
        values: *const Half,
        mask: *const u8,
        rows: *const i32,
        positions: *const i32,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        attn_weights: *mut Half,
        attn_out: *mut Half,
        normed_keys: *mut Half,
        query_rows: i32,
        tokens: i32,
        rotary_dim: i32,
        rope_theta: f32,
        rms_eps: f32,
        stream: CUstream,
    ) -> CUresult;
}
