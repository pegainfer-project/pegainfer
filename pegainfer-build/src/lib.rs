use std::env;
use std::path::Path;
use std::path::PathBuf;

/// Finds a package's install root: probes `$env_var` first, then each of
/// `default_paths`, for any of the `check_files` — several cover layout
/// variants like `include/` vs `targets/<arch>/include/`. Returns the
/// matched root and check file.
///
/// # Panics
/// When nothing matches.
#[cfg(test)]
fn find_package(
    provider: &str,
    env_var: &str,
    default_paths: &[&str],
    check_files: &[&str],
) -> (PathBuf, PathBuf) {
    println!("cargo:rerun-if-env-changed={}", env_var);
    let env_root = env::var_os(env_var).map(PathBuf::from);
    let roots: Vec<PathBuf> = env_root
        .clone()
        .into_iter()
        .chain(default_paths.iter().map(PathBuf::from))
        .collect();
    for root in &roots {
        for check in check_files {
            let found = root.join(check);
            if found.is_file() {
                if let Some(env_root) = &env_root
                    && env_root != root
                {
                    println!(
                        "cargo:warning={provider}: ${env_var} ({}) contains none of \
                         {check_files:?}; using {} instead",
                        env_root.display(),
                        root.display()
                    );
                }
                return (root.clone(), found);
            }
        }
    }
    panic!(
        "{provider} build error: none of {check_files:?} found. \
         Looked at `${env_var}` ({env_status}) and default paths {default_paths:?}. \
         Hint: install the provider headers or set `{env_var}` to their install root.",
        env_status = env_root
            .map(|root| format!("set to {root:?}"))
            .unwrap_or_else(|| "unset".to_string()),
    )
}

/// `targets/<dir>` names for the build target; aarch64 toolkits ship as
/// either `aarch64-linux` or `sbsa-linux`. Host arch outside build scripts.
fn target_dirs() -> Vec<String> {
    let arch =
        env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| std::env::consts::ARCH.to_string());
    match arch.as_str() {
        "aarch64" => vec!["aarch64-linux".to_string(), "sbsa-linux".to_string()],
        arch => vec![format!("{arch}-linux")],
    }
}

/// One build-time CUDA toolkit resolution covering the classic, conda, and
/// NVIDIA HPC SDK layouts; runtime loading (`LD_LIBRARY_PATH`, rpath) is out of scope.
pub struct CudaToolkit {
    pub root: PathBuf,
    /// `{root}/bin/nvcc` when present, otherwise bare `nvcc` from `$PATH`.
    pub nvcc: PathBuf,
    pub include_dirs: Vec<PathBuf>,
    lib_dirs: Vec<PathBuf>,
}

impl CudaToolkit {
    pub fn discover() -> Self {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let env_root = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .ok();
        if let Some(root) = env_root.as_deref().filter(|root| !Path::new(root).is_dir()) {
            println!(
                "cargo:warning=CUDA root {root} (from CUDA_HOME/CUDA_PATH) is not a directory"
            );
        }
        let root = env_root.map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
        Self::from_root(root)
    }

    fn from_root(root: PathBuf) -> Self {
        let nvcc = root.join("bin/nvcc");
        let nvcc = if nvcc.is_file() {
            nvcc
        } else {
            PathBuf::from("nvcc")
        };

        let mut include_dirs = vec![root.join("include")];
        let mut lib_dirs = vec![root.join("lib64"), root.join("lib")];
        for target in target_dirs() {
            include_dirs.push(root.join(format!("targets/{target}/include")));
            lib_dirs.push(root.join(format!("targets/{target}/lib")));
        }
        // HPC SDK roots look like .../hpc_sdk/<os>/<release>/cuda/<ver>; the
        // math libraries live in the <release>/math_libs/<ver> sibling tree.
        if let (Some(version), Some(release)) =
            (root.file_name(), root.parent().and_then(Path::parent))
        {
            let math = release.join("math_libs").join(version);
            lib_dirs.push(math.join("lib64"));
            lib_dirs.push(math.join("lib"));
        }

        Self {
            nvcc,
            include_dirs: existing_deduped(include_dirs),
            lib_dirs: existing_deduped(lib_dirs),
            root,
        }
    }

    /// The include dir that actually contains `header` — host-compiler `-I`
    /// flags need this; on conda `include/` exists but lacks the CUDA headers.
    pub fn header_dir(&self, header: &str) -> Option<PathBuf> {
        self.include_dirs
            .iter()
            .find(|dir| dir.join(header).is_file())
            .cloned()
    }

    pub fn link_search(&self) {
        if self.lib_dirs.is_empty() {
            println!(
                "cargo:warning=no CUDA library dir found under {}",
                self.root.display()
            );
        }
        for dir in &self.lib_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
}

fn existing_deduped(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
        if !seen.contains(&canon) {
            seen.push(canon);
            out.push(dir);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempTree(tempfile::TempDir);

    impl TempTree {
        fn new() -> Self {
            Self(tempfile::tempdir().unwrap())
        }

        fn root(&self) -> &Path {
            self.0.path()
        }

        fn mkdirs(&self, rel: &str) -> PathBuf {
            let dir = self.root().join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        fn touch(&self, rel: &str) {
            let file = self.root().join(rel);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, b"").unwrap();
        }
    }

    fn target_dir() -> String {
        target_dirs().remove(0)
    }

    #[test]
    fn classic_layout() {
        let tree = TempTree::new();
        tree.touch("include/cuda.h");
        tree.touch("bin/nvcc");
        let lib64 = tree.mkdirs("lib64");
        tree.mkdirs("lib64/stubs");

        let tk = CudaToolkit::from_root(tree.root().to_path_buf());
        assert_eq!(tk.nvcc, tree.root().join("bin/nvcc"));
        assert_eq!(tk.header_dir("cuda.h"), Some(tree.root().join("include")));
        assert_eq!(tk.lib_dirs, vec![lib64.clone()]);
        assert!(lib64.join("stubs").is_dir());
    }

    #[test]
    fn conda_layout() {
        let tree = TempTree::new();
        let target = target_dir();
        tree.mkdirs("include");
        tree.touch(&format!("targets/{target}/include/cuda.h"));
        let lib = tree.mkdirs("lib");
        let targets_lib = tree.mkdirs(&format!("targets/{target}/lib"));

        let tk = CudaToolkit::from_root(tree.root().to_path_buf());
        assert_eq!(tk.nvcc, PathBuf::from("nvcc"));
        assert_eq!(
            tk.header_dir("cuda.h"),
            Some(tree.root().join(format!("targets/{target}/include")))
        );
        assert_eq!(tk.lib_dirs, vec![lib, targets_lib]);
    }

    #[test]
    fn hpc_sdk_layout_adds_math_libs_sibling() {
        let tree = TempTree::new();
        let root = tree.mkdirs("release/cuda/12.6");
        let lib64 = tree.mkdirs("release/cuda/12.6/lib64");
        let math = tree.mkdirs("release/math_libs/12.6/lib64");

        assert_eq!(CudaToolkit::from_root(root).lib_dirs, vec![lib64, math]);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_dirs_dedupe() {
        let tree = TempTree::new();
        let target = target_dir();
        tree.touch(&format!("targets/{target}/include/cuda.h"));
        tree.mkdirs(&format!("targets/{target}/lib"));
        std::os::unix::fs::symlink(
            tree.root().join(format!("targets/{target}/include")),
            tree.root().join("include"),
        )
        .unwrap();
        std::os::unix::fs::symlink(
            tree.root().join(format!("targets/{target}/lib")),
            tree.root().join("lib"),
        )
        .unwrap();

        let tk = CudaToolkit::from_root(tree.root().to_path_buf());
        assert_eq!(tk.include_dirs.len(), 1);
        assert_eq!(tk.lib_dirs.len(), 1);
        assert_eq!(tk.header_dir("cuda.h"), Some(tree.root().join("include")));
    }

    #[test]
    fn unknown_layout_yields_nothing() {
        let tree = TempTree::new();
        tree.mkdirs("weird/place");

        let tk = CudaToolkit::from_root(tree.root().to_path_buf());
        assert!(tk.lib_dirs.is_empty());
        assert!(tk.include_dirs.is_empty());
        assert_eq!(tk.header_dir("cuda.h"), None);
    }

    #[test]
    fn find_package_returns_matching_root_and_check_file() {
        let tree = TempTree::new();
        tree.touch("include/gdrapi.h");
        let root_str = tree.root().to_str().unwrap().to_string();

        let (root, header) = find_package(
            "test",
            "PEGAINFER_TEST_UNSET_ENV",
            &[&root_str],
            &["targets/missing/include/gdrapi.h", "include/gdrapi.h"],
        );
        assert_eq!(root, tree.root());
        assert_eq!(header, tree.root().join("include/gdrapi.h"));
    }

    #[test]
    #[should_panic(expected = "none of")]
    fn missing_header_panics_with_all_candidates() {
        let tree = TempTree::new();
        let root_str = tree.root().to_str().unwrap().to_string();
        find_package(
            "test",
            "PEGAINFER_TEST_UNSET_ENV",
            &[&root_str],
            &["include/cuda.h"],
        );
    }
}
