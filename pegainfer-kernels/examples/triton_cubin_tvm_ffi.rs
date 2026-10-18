use pegainfer_kernels::triton_cubin::QWEN35_GDR_CHUNK_SOLVE;
use pegainfer_kernels::triton_cubin::{self};

fn main() -> tvm_ffi::Result<()> {
    triton_cubin::register_global_functions()?;

    println!("registered Triton CUBIN TVM FFI functions:");
    for spec in triton_cubin::TRITON_CUBIN_FUNCTIONS {
        println!("  {} -> {}", spec.name, spec.ffi_symbol);
    }

    let solve = triton_cubin::get_global_or_register(QWEN35_GDR_CHUNK_SOLVE.name)?;
    println!(
        "{} is ready; call it with packed args: {}",
        QWEN35_GDR_CHUNK_SOLVE.name,
        QWEN35_GDR_CHUNK_SOLVE.arg_names.join(", ")
    );

    drop(solve);
    Ok(())
}
