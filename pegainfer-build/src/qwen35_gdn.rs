//! Pre-link validation of the explicitly selected, locally generated GDN candidate.
//! Repository pins establish compatibility; candidate hashes establish internal
//! consistency, not provenance or permission to select a backend by default.

use std::fs;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use sha2::Digest;
use sha2::Sha256;

macro_rules! contract_struct {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Debug, Deserialize, PartialEq, Eq)]
        #[serde(deny_unknown_fields)]
        struct $name { $($field: $ty),* }
    };
}

contract_struct!(Target {
    arch: String,
    code_object: String
});
contract_struct!(Geometry {
    h_q: u32,
    h_k: u32,
    h_v: u32,
    head_dim: u32
});
contract_struct!(Dtypes {
    q: String,
    k: String,
    v: String,
    o: String,
    alpha: String,
    beta: String,
    state: String,
    cu_seqlens: String,
    workspace: String,
});
contract_struct!(Tokens {
    extent: String,
    minimum: u32,
    maximum: u32
});
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
enum Extent {
    Fixed(u32),
    Dynamic(String),
}
contract_struct!(View {
    shape: [Extent; 3],
    stride: [u32; 3]
});
contract_struct!(Abi {
    version: u32,
    function_prefix: String,
    geometry_binding: String,
    symbols: [String; 7],
    q_view: View,
    k_view: View,
    v_view: View,
    o_view: View,
    state_layout: String,
    state_update: String,
});
contract_struct!(Workspace {
    kind: String,
    formula: String,
    bytes_per_sm: u32,
    alignment_bytes: u32,
});
contract_struct!(Toolchain {
    python: String,
    ptx_compiler_release: String,
    ptx_compiler_version: String,
    ptx_isa: String,
    cutlass_dsl: String,
    cutlass_dsl_libs_base: String,
    torch: String,
    cuda_python: String,
    cuda_bindings: String,
    cuda_pathfinder: String,
});
contract_struct!(Source {
    flashinfer_commit: String,
    kernel_source_sha256: String,
    source_lock_sha256: String,
    generator_sha256: String,
    requirements_lock_sha256: String,
});
contract_struct!(FileRecord {
    sha256: String,
    size_bytes: u64
});
contract_struct!(Artifact {
    format: String,
    header: FileRecord,
    object: FileRecord,
    native_runtime: FileRecord,
});
contract_struct!(FrozenContract {
    schema_version: u32,
    artifact_kind: String,
    variant: String,
    target: Target,
    geometry: Geometry,
    dtypes: Dtypes,
    tokens: Tokens,
    abi: Abi,
    workspace: Workspace,
    toolchain: Toolchain,
});
contract_struct!(Manifest {
    schema_version: u32,
    artifact_kind: String,
    variant: String,
    target: Target,
    geometry: Geometry,
    dtypes: Dtypes,
    tokens: Tokens,
    abi: Abi,
    workspace: Workspace,
    toolchain: Toolchain,
    source: Source,
    artifact: Artifact,
});
contract_struct!(Patch {
    path: String,
    sha256: String
});
contract_struct!(SourceLock {
    schema_version: u32,
    flashinfer_commit: String,
    patches: [Patch; 1],
    patched_kernel_sha256: String,
    upstream_export_kernel_sha256: String,
    contract: FrozenContract,
});

pub const ARTIFACT_FILES: [&str; 3] = ["kernel.h", "kernel.o", "libcuda_dialect_runtime_static.a"];
pub const PIN_FILES: [&str; 4] = [
    "source-lock.json",
    "compile_sm120.py",
    "requirements-cu13.lock",
    "patches/0001-openinfer-hkv-state-layout.patch",
];

pub struct Candidate {
    /// Verified bytes are copied to OUT_DIR before linking, avoiding a second
    /// read of mutable bundle inputs after their hashes have been checked.
    pub files: Vec<(&'static str, Vec<u8>)>,
    pub object_sha256: String,
    pub workspace_bytes_per_sm: u32,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn directory(path: &Path) -> Result<PathBuf, String> {
    let mut walked = PathBuf::new();
    for component in path.components() {
        if component == Component::ParentDir {
            return Err(format!("GDN candidate path traversal: {}", path.display()));
        }
        walked.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&walked)
            .map_err(|error| format!("GDN candidate path {}: {error}", walked.display()))?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(format!(
                "GDN candidate path is not a real directory: {}",
                walked.display()
            ));
        }
    }
    fs::canonicalize(path).map_err(|error| format!("GDN candidate directory: {error}"))
}

fn read_regular(directory: &Path, name: &str) -> Result<Vec<u8>, String> {
    let path = directory.join(name);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("GDN candidate file {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "GDN candidate file is not a regular file: {}",
            path.display()
        ));
    }
    fs::read(&path).map_err(|error| format!("GDN candidate read {}: {error}", path.display()))
}

fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<T, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("GDN candidate {label}: {error}"))
}

fn equal<T: PartialEq>(actual: &T, expected: &T, label: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("GDN candidate {label} mismatch"))
    }
}

/// Validate every manifest field and every actual link input, independently of
/// Python. The kernel root and its source lock are project-owned build inputs.
pub fn validate_candidate(kernel_root: &Path, bundle: &Path) -> Result<Candidate, String> {
    let pins = directory(&kernel_root.join("tools/flashinfer_gdn"))?;
    let lock_bytes = read_regular(&pins, "source-lock.json")?;
    let lock: SourceLock = parse(&lock_bytes, "source lock")?;
    equal(&lock.schema_version, &4, "source lock schema")?;
    let patch = &lock.patches[0];
    equal(
        &patch.path.as_str(),
        &PIN_FILES[3],
        "source lock patch path",
    )?;
    let patch_dir = directory(&pins.join("patches"))?;
    equal(
        &digest(&read_regular(
            &patch_dir,
            "0001-openinfer-hkv-state-layout.patch",
        )?),
        &patch.sha256,
        "source lock patch hash",
    )?;
    let expected_source = Source {
        flashinfer_commit: lock.flashinfer_commit,
        kernel_source_sha256: lock.patched_kernel_sha256,
        source_lock_sha256: digest(&lock_bytes),
        generator_sha256: digest(&read_regular(&pins, "compile_sm120.py")?),
        requirements_lock_sha256: digest(&read_regular(&pins, "requirements-cu13.lock")?),
    };
    let bundle = directory(bundle)?;
    let manifest_bytes = read_regular(&bundle, "manifest.json")?;
    let manifest: Manifest = parse(&manifest_bytes, "manifest")?;
    macro_rules! check {
        ($($field:ident),+ $(,)?) => { $(equal(&manifest.$field, &lock.contract.$field, stringify!($field))?;)+ };
    }
    check!(
        schema_version,
        artifact_kind,
        variant,
        target,
        geometry,
        dtypes,
        tokens,
        abi,
        workspace,
        toolchain
    );
    equal(&manifest.source, &expected_source, "source")?;
    equal(
        &manifest.artifact.format.as_str(),
        &"elf_relocatable_with_embedded_cubin",
        "artifact format",
    )?;
    let mut files = vec![("manifest.json", manifest_bytes)];
    for (name, record) in ARTIFACT_FILES.into_iter().zip([
        &manifest.artifact.header,
        &manifest.artifact.object,
        &manifest.artifact.native_runtime,
    ]) {
        let bytes = read_regular(&bundle, name)?;
        if bytes.is_empty()
            || bytes.len() as u64 != record.size_bytes
            || digest(&bytes) != record.sha256
        {
            return Err(format!("GDN candidate artifact {name} size/hash mismatch"));
        }
        files.push((name, bytes));
    }
    Ok(Candidate {
        files,
        object_sha256: manifest.artifact.object.sha256,
        workspace_bytes_per_sm: manifest.workspace.bytes_per_sm,
    })
}

#[cfg(test)]
#[path = "qwen35_gdn_tests.rs"]
mod tests;
