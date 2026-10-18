//! `typed_pipeline!` - declarative DSL for typed tensor forward chains.
//!
//! The macro expands common `typed_ops` calls while keeping model-specific
//! kernels as normal Rust calls through `try ...;` or `call ...;` escape hatches.
//! It is intentionally limited to tensor allocation, typed op dispatch, error
//! propagation, and buffer swaps.
//!
//! # Header
//!
//! ```ignore
//! typed_pipeline! {
//!     ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
//!     tensor qkv_a: KIMI_K2_MLA_QKV_A_OUT;
//!     rms_norm(hidden => normed, attention.input_norm);
//!     gemm(normed => qkv_a, attention.fused_qkv_a_proj);
//!     try kimi_mla_split_qkv_a(ctx, &qkv_a, &mut q_a, &mut compressed_kv, &mut k_rope);
//! }
//! ```
//!
//! `gemm` defaults to `graphsafe` when omitted. Tensor allocation requires
//! `gemm = prefill` in the header, keeping decode graph paths allocation-free.
//! Override one statement with `gemm[prefill](...)` / `gemm[graphsafe](...)`.

#[macro_export]
macro_rules! typed_pipeline {
    (
        ctx = $ctx:expr, eps = $eps:expr, seq_len = $seq_len:expr, gemm = $gemm_mode:ident;
        $($rest:tt)*
    ) => {
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, (@some $seq_len); $($rest)*)
    };

    (
        ctx = $ctx:expr, eps = $eps:expr, seq_len = $seq_len:expr;
        $($rest:tt)*
    ) => {
        compile_error!("typed_pipeline `seq_len = ...` requires `gemm = prefill`")
    };

    (
        ctx = $ctx:expr, eps = $eps:expr, gemm = $gemm_mode:ident;
        $($rest:tt)*
    ) => {
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, (@none); $($rest)*)
    };

    (
        ctx = $ctx:expr, eps = $eps:expr;
        $($rest:tt)*
    ) => {
        $crate::typed_pipeline!(@steps $ctx, $eps, graphsafe, (@none); $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt; ) => {};

    (@steps $ctx:expr, $eps:expr, graphsafe, (@some $seq_len:expr);
        tensor $name:ident : $dim:path;
        $($rest:tt)*
    ) => {
        compile_error!("typed_pipeline tensor allocation requires `gemm = prefill`");
    };

    (@steps $ctx:expr, $eps:expr, graphsafe, (@some $default_seq_len:expr);
        tensor_at $name:ident : $dim:path, $seq_len:expr;
        $($rest:tt)*
    ) => {
        compile_error!("typed_pipeline tensor allocation requires `gemm = prefill`");
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, (@some $seq_len:expr);
        tensor $name:ident : $dim:path;
        $($rest:tt)*
    ) => {
        let mut $name = $crate::tensor::GpuTensor::<{ $dim }>::zeros($ctx, $seq_len)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, (@some $seq_len); $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, (@some $default_seq_len:expr);
        tensor_at $name:ident : $dim:path, $seq_len:expr;
        $($rest:tt)*
    ) => {
        let mut $name = $crate::tensor::GpuTensor::<{ $dim }>::zeros($ctx, $seq_len)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, (@some $default_seq_len); $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, (@none);
        tensor $name:ident : $dim:path;
        $($rest:tt)*
    ) => {
        compile_error!("typed_pipeline tensor statements require `seq_len = ...` in the header");
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        gemm ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_pipeline!(@gemm $gemm_mode, $ctx, $x, $y, $w);
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        gemm [ $stmt_mode:ident ] ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_pipeline!(@gemm $stmt_mode, $ctx, $x, $y, $w);
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        rms_norm ( $x:expr => $y:expr, $w:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::rms_norm_into($ctx, $x, &$w, $eps, $y)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        add ( $a:expr, $b:expr => $y:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::add_into($ctx, $a, $b, $y)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        bf16_to_f32 ( $x:expr => $y:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::bf16_to_f32_into($ctx, $x, $y)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        f32_to_bf16 ( $x:expr => $y:expr );
        $($rest:tt)*
    ) => {
        $crate::typed_ops::f32_to_bf16_into($ctx, $x, $y)?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        swap ( $a:expr, $b:expr );
        $($rest:tt)*
    ) => {
        ::std::mem::swap($a, $b);
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        try $call:expr;
        $($rest:tt)*
    ) => {
        $call?;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@steps $ctx:expr, $eps:expr, $gemm_mode:ident, $seq_len:tt;
        call $call:expr;
        $($rest:tt)*
    ) => {
        $call;
        $crate::typed_pipeline!(@steps $ctx, $eps, $gemm_mode, $seq_len; $($rest)*)
    };

    (@gemm graphsafe, $ctx:expr, $x:expr, $y:expr, $w:expr) => {
        $crate::typed_ops::gemm_graphsafe_into($ctx, &$w, $x, $y)?
    };

    (@gemm prefill, $ctx:expr, $x:expr, $y:expr, $w:expr) => {
        $crate::typed_ops::gemm_into($ctx, &$w, $x, $y)?
    };
}
