use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let toolkit = pegainfer_build::CudaToolkit::discover();
    // CUPTI ships outside the main toolkit dirs in classic installs.
    let cupti_include = toolkit.root.join("extras/CUPTI/include");
    let cupti_lib64 = toolkit.root.join("extras/CUPTI/lib64");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .file(manifest_dir.join("csrc/range_profiler.cpp"))
        .includes(&toolkit.include_dirs);
    if cupti_include.is_dir() {
        build.include(&cupti_include);
    }
    build.compile("pegainfer_cupti_range_profiler");

    toolkit.link_search();
    if cupti_lib64.is_dir() {
        println!("cargo:rustc-link-search=native={}", cupti_lib64.display());
    }

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cupti");
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=csrc/range_profiler.cpp");
}
