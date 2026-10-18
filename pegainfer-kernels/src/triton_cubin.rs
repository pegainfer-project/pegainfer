//! TVM FFI wrappers for Triton AOT CUBIN launchers.
//!
//! The generated Triton AOT C stubs remain the low-level CUDA launch owner.
//! This module exposes the launchers through TVM FFI so DSL-produced artifacts
//! can call them without depending on PegaInfer's private Rust operator APIs.

use std::ffi::c_void;

use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Error;
use tvm_ffi::Function;
use tvm_ffi::RUNTIME_ERROR;
use tvm_ffi::Result as TvmResult;
use tvm_ffi::TYPE_ERROR;
use tvm_ffi::VALUE_ERROR;

use crate::ffi;

/// Metadata for one Triton AOT CUBIN launcher exposed through TVM FFI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TritonCubinFunctionSpec {
    /// TVM global function name.
    pub name: &'static str,
    /// Linked C ABI symbol that ultimately launches the generated Triton CUBIN.
    pub ffi_symbol: &'static str,
    /// Packed TVM FFI argument names, in call order.
    pub arg_names: &'static [&'static str],
}

const QWEN35_GDR_CHUNK_SOLVE_ARGS: &[&str] = &[
    "a_tril_ptr",
    "a_inv_ptr",
    "seq_len",
    "num_value_heads",
    "stream",
];

/// Qwen3.5 gated-delta-rule triangular solve Triton AOT launcher.
pub const QWEN35_GDR_CHUNK_SOLVE: TritonCubinFunctionSpec = TritonCubinFunctionSpec {
    name: "pegainfer.triton_cubin.qwen35.gated_delta_rule_chunk_solve",
    ffi_symbol: "gated_delta_rule_prefill_chunk_solve_cuda",
    arg_names: QWEN35_GDR_CHUNK_SOLVE_ARGS,
};

/// Triton AOT CUBIN launchers currently exposed through TVM FFI.
pub const TRITON_CUBIN_FUNCTIONS: &[TritonCubinFunctionSpec] = &[QWEN35_GDR_CHUNK_SOLVE];

/// Return a fresh TVM FFI function for a known Triton CUBIN launcher.
#[must_use]
fn function(name: &str) -> Option<Function> {
    if name == QWEN35_GDR_CHUNK_SOLVE.name {
        return Some(Function::from_packed(launch_qwen35_gdr_chunk_solve));
    }
    None
}

/// Register all current Triton CUBIN launchers in TVM FFI's global registry.
pub fn register_global_functions() -> TvmResult<()> {
    for spec in TRITON_CUBIN_FUNCTIONS {
        if Function::get_global(spec.name).is_ok() {
            continue;
        }
        let func = function(spec.name).ok_or_else(|| {
            Error::new(
                RUNTIME_ERROR,
                &format!("missing TVM FFI wrapper for {}", spec.name),
                "",
            )
        })?;
        Function::register_global(spec.name, func)?;
    }
    Ok(())
}

/// Register wrappers if needed, then fetch one wrapper from TVM FFI's global registry.
pub fn get_global_or_register(name: &str) -> TvmResult<Function> {
    register_global_functions()?;
    Function::get_global(name)
}

fn expect_args(args: &[AnyView<'_>], spec: TritonCubinFunctionSpec) -> TvmResult<()> {
    if args.len() == spec.arg_names.len() {
        return Ok(());
    }
    Err(Error::new(
        VALUE_ERROR,
        &format!(
            "{} expects {} arguments ({}) but got {}",
            spec.name,
            spec.arg_names.len(),
            spec.arg_names.join(", "),
            args.len()
        ),
        "",
    ))
}

fn type_error(spec: TritonCubinFunctionSpec, idx: usize, expected: &str) -> Error {
    Error::new(
        TYPE_ERROR,
        &format!(
            "{} argument #{} `{}` must be {}",
            spec.name, idx, spec.arg_names[idx], expected
        ),
        "",
    )
}

fn arg_handle(args: &[AnyView<'_>], spec: TritonCubinFunctionSpec, idx: usize) -> TvmResult<usize> {
    let value = args
        .get(idx)
        .ok_or_else(|| type_error(spec, idx, "a non-negative integer or opaque pointer"))?;
    if let Some(raw) = value.try_as::<i64>() {
        return usize::try_from(raw)
            .map_err(|_| type_error(spec, idx, "a non-negative integer or opaque pointer"));
    }
    if let Some(raw) = value.try_as::<u64>() {
        return usize::try_from(raw)
            .map_err(|_| type_error(spec, idx, "a non-negative integer or opaque pointer"));
    }
    if let Some(raw) = value.try_as::<*mut c_void>() {
        return Ok(raw as usize);
    }
    Err(type_error(
        spec,
        idx,
        "a non-negative integer or opaque pointer",
    ))
}

fn arg_i32(args: &[AnyView<'_>], spec: TritonCubinFunctionSpec, idx: usize) -> TvmResult<i32> {
    let value = args
        .get(idx)
        .ok_or_else(|| type_error(spec, idx, "an i32-range integer"))?;
    if let Some(raw) = value.try_as::<i64>() {
        return i32::try_from(raw).map_err(|_| type_error(spec, idx, "an i32-range integer"));
    }
    if let Some(raw) = value.try_as::<u64>() {
        return i32::try_from(raw).map_err(|_| type_error(spec, idx, "an i32-range integer"));
    }
    if let Some(raw) = value.try_as::<i32>() {
        return Ok(raw);
    }
    Err(type_error(spec, idx, "an i32-range integer"))
}

fn stream(args: &[AnyView<'_>], spec: TritonCubinFunctionSpec, idx: usize) -> TvmResult<CUstream> {
    Ok(arg_handle(args, spec, idx)? as CUstream)
}

fn f32_const(
    args: &[AnyView<'_>],
    spec: TritonCubinFunctionSpec,
    idx: usize,
) -> TvmResult<*const f32> {
    Ok(arg_handle(args, spec, idx)? as *const f32)
}

fn half_mut(
    args: &[AnyView<'_>],
    spec: TritonCubinFunctionSpec,
    idx: usize,
) -> TvmResult<*mut ffi::Half> {
    Ok(arg_handle(args, spec, idx)? as *mut ffi::Half)
}

fn cuda_result(spec: TritonCubinFunctionSpec, result: CUresult) -> TvmResult<Any> {
    if result as u32 == 0 {
        Ok(Any::from(()))
    } else {
        Err(Error::new(
            RUNTIME_ERROR,
            &format!(
                "{} via {} returned CUDA result {:?}",
                spec.name, spec.ffi_symbol, result
            ),
            "",
        ))
    }
}

fn launch_qwen35_gdr_chunk_solve(args: &[AnyView<'_>]) -> TvmResult<Any> {
    let spec = QWEN35_GDR_CHUNK_SOLVE;
    expect_args(args, spec)?;
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_solve_cuda(
            f32_const(args, spec, 0)?,
            half_mut(args, spec, 1)?,
            arg_i32(args, spec, 2)?,
            arg_i32(args, spec, 3)?,
            stream(args, spec, 4)?,
        )
    };
    cuda_result(spec, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_message(result: TvmResult<Any>, context: &str) -> String {
        match result {
            Ok(_) => panic!("{context}"),
            Err(err) => err.message().to_string(),
        }
    }

    #[test]
    fn exposes_current_triton_cubin_specs() {
        assert_eq!(TRITON_CUBIN_FUNCTIONS, &[QWEN35_GDR_CHUNK_SOLVE]);
        assert!(
            QWEN35_GDR_CHUNK_SOLVE
                .name
                .starts_with("pegainfer.triton_cubin.qwen35.")
        );
        assert_eq!(
            QWEN35_GDR_CHUNK_SOLVE.arg_names,
            QWEN35_GDR_CHUNK_SOLVE_ARGS
        );
    }

    #[test]
    fn packed_wrapper_reports_argument_contract_before_launch() {
        let func = function(QWEN35_GDR_CHUNK_SOLVE.name).expect("known wrapper");
        let err = error_message(func.call_packed(&[]), "missing args should fail");
        assert!(err.contains("expects 5 arguments"));
        assert!(err.contains("a_tril_ptr"));
    }

    #[test]
    fn rejects_unknown_triton_cubin_function() {
        assert!(function("pegainfer.triton_cubin.unknown").is_none());
    }

    #[test]
    fn global_registry_round_trips_wrapper() {
        let func = get_global_or_register(QWEN35_GDR_CHUNK_SOLVE.name).expect("registered wrapper");
        let err = error_message(func.call_packed(&[]), "missing args should fail");
        assert!(err.contains(QWEN35_GDR_CHUNK_SOLVE.name));
        assert!(err.contains("expects 5 arguments"));
    }

    #[test]
    fn handle_args_accept_integer_and_opaque_pointer() {
        let tvm_integer_handle = 0x1234_i64;
        let tvm_integer_args = [AnyView::from(&tvm_integer_handle)];
        assert_eq!(
            arg_handle(&tvm_integer_args, QWEN35_GDR_CHUNK_SOLVE, 0).expect("i64 handle"),
            tvm_integer_handle as usize
        );

        let rust_integer_handle = 0x3456_u64;
        let rust_integer_args = [AnyView::from(&rust_integer_handle)];
        assert_eq!(
            arg_handle(&rust_integer_args, QWEN35_GDR_CHUNK_SOLVE, 0).expect("u64 handle"),
            rust_integer_handle as usize
        );

        let opaque_handle = 0x5678_usize as *mut c_void;
        let opaque_args = [AnyView::from(&opaque_handle)];
        assert_eq!(
            arg_handle(&opaque_args, QWEN35_GDR_CHUNK_SOLVE, 0).expect("opaque handle"),
            opaque_handle as usize
        );
    }

    #[test]
    fn scalar_args_accept_tvm_i64_integer() {
        let seq_len = 16_i64;
        let args = [AnyView::from(&seq_len)];
        assert_eq!(
            arg_i32(&args, QWEN35_GDR_CHUNK_SOLVE, 2).expect("i64 scalar"),
            16_i32
        );
    }

    #[test]
    fn packed_wrapper_reports_handle_type_errors_before_launch() {
        let bad_handle = 1.25_f32;
        let a_inv_ptr = 0_u64;
        let seq_len = 16_i32;
        let num_value_heads = 8_i32;
        let stream = 0_u64;
        let args = [
            AnyView::from(&bad_handle),
            AnyView::from(&a_inv_ptr),
            AnyView::from(&seq_len),
            AnyView::from(&num_value_heads),
            AnyView::from(&stream),
        ];

        let func = function(QWEN35_GDR_CHUNK_SOLVE.name).expect("known wrapper");
        let err = error_message(
            func.call_packed(&args),
            "bad handle should fail before launch",
        );
        assert!(err.contains("argument #0 `a_tril_ptr`"));
        assert!(err.contains("integer or opaque pointer"));
    }

    #[test]
    fn packed_wrapper_reports_scalar_type_errors_before_launch() {
        let a_tril_ptr = 0_u64;
        let a_inv_ptr = 0_u64;
        let bad_seq_len = 16.0_f32;
        let num_value_heads = 8_i32;
        let stream = 0_u64;
        let args = [
            AnyView::from(&a_tril_ptr),
            AnyView::from(&a_inv_ptr),
            AnyView::from(&bad_seq_len),
            AnyView::from(&num_value_heads),
            AnyView::from(&stream),
        ];

        let func = function(QWEN35_GDR_CHUNK_SOLVE.name).expect("known wrapper");
        let err = error_message(
            func.call_packed(&args),
            "bad scalar should fail before launch",
        );
        assert!(err.contains("argument #2 `seq_len`"));
        assert!(err.contains("must be an i32-range integer"));
    }
}
