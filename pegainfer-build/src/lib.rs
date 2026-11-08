use std::env;
use std::path::Path;
use std::path::PathBuf;

#[cfg(feature = "qwen35-gdn")]
pub mod qwen35_gdn;

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
}
