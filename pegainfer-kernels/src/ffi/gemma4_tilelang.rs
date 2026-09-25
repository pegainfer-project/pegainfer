//! Gemma 4 TileLang-generated attention (AOT), built by the `tilelang` section
//! of `build.rs` from `pegainfer-gemma4/kernels/generate.py`.
//!
//! Each symbol is a hand-written launcher returning `cudaError_t` as `int`:
//! `cudaErrorInvalidValue` outside what the body was built for,
//! `cudaErrorNotSupported` from the stub tier. A launcher owns its own grid,
//! which is why the prefill's query boundaries arrive twice, on the device and
//! on the host.

use core::ffi::c_void;

use cudarc::driver::sys::CUstream;

unsafe extern "C" {
    /// Causal GQA prefill for one layer's global family, ragged across the
    /// step's prompt segments. `q` and `out` are bf16 rows of
    /// `[num_qo_heads, head_dim]`, `kv` the pool in rows of
    /// `[num_kv_heads, head_dim]`, and the indptrs the prefill plan's own,
    /// `host_q_indptr` its host copy. `sm_scale` multiplies the scores; Gemma
    /// 4 attends unscaled and passes one.
    /// `fold_rotary` is the pool's row format, zero for split K|V rows. The
    /// extents, `batch`, `page_size` and the head counts are refused rather
    /// than truncated when they exceed what the bodies were built for.
    pub fn gemma4_hd512_prefill_varlen(
        q: *const c_void,
        kv: *const c_void,
        page_indices: *const i32,
        page_indptr: *const i32,
        q_indptr: *const i32,
        host_q_indptr: *const i32,
        last_page_len: *const i32,
        out: *mut c_void,
        batch: i32,
        q_rows: i32,
        pool_rows: i32,
        rows_per_page: i32,
        layer_row: i32,
        page_size: i32,
        fold_rotary: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    /// Split-KV decode for one layer's global family: a partial pass over
    /// every (slot, kv head) then a merge over each request's slots, in one
    /// call. The plan is the serving path's own, one slot per (request,
    /// chunk), with the per-slot partial state in `tmp_v`/`tmp_s`;
    /// `chunk_tokens` must be whole pages and `row_offset` is where the decode
    /// rows start in `q` and `out`. Extents, page size, row format, chunk and
    /// head counts are refused rather than truncated, as in the prefill.
    pub fn gemma4_hd512_decode_split_kv(
        q: *const c_void,
        kv: *const c_void,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        valid_mask: *const u8,
        o_indptr: *const i32,
        tmp_v: *mut c_void,
        tmp_s: *mut f32,
        out: *mut c_void,
        batch: i32,
        padded_slots: i32,
        row_offset: i32,
        q_rows: i32,
        pool_rows: i32,
        rows_per_page: i32,
        layer_row: i32,
        page_size: i32,
        fold_rotary: i32,
        chunk_tokens: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    /// Windowed causal prefill for one layer's sliding family over the local
    /// pool's split K|V rows: the global prefill's plan and refusals, with
    /// `window_left` the inclusive key distance a query may attend to.
    pub fn gemma4_hd256_prefill_window(
        q: *const c_void,
        kv: *const c_void,
        page_indices: *const i32,
        page_indptr: *const i32,
        q_indptr: *const i32,
        host_q_indptr: *const i32,
        last_page_len: *const i32,
        out: *mut c_void,
        batch: i32,
        q_rows: i32,
        pool_rows: i32,
        rows_per_page: i32,
        layer_row: i32,
        page_size: i32,
        window_left: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    /// Windowed split-KV decode for one layer's sliding family: the same
    /// plan, workspace and refusals as the global decode over the local
    /// pool's split K|V rows, with `window_left` the inclusive key distance
    /// a query may attend to, so the keys a page-aligned window still holds
    /// past it are masked rather than read as context.
    pub fn gemma4_hd256_decode_window(
        q: *const c_void,
        kv: *const c_void,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        valid_mask: *const u8,
        o_indptr: *const i32,
        tmp_v: *mut c_void,
        tmp_s: *mut f32,
        out: *mut c_void,
        batch: i32,
        padded_slots: i32,
        row_offset: i32,
        q_rows: i32,
        pool_rows: i32,
        rows_per_page: i32,
        layer_row: i32,
        page_size: i32,
        chunk_tokens: i32,
        window_left: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}
