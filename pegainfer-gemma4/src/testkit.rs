//! Shared plumbing for the in-crate checkpoint oracles: the golden fixture,
//! its provenance checks, and typed tensor readers.

use sha2::Digest;
use sha2::Sha256;

pub(crate) const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/gemma4-12b-hf-golden.safetensors"
);
pub(crate) const METADATA_KEY: &str = "gemma4_golden";

pub(crate) fn fixture_manifest(bytes: &[u8], key: &str) -> serde_json::Value {
    let (_, metadata) = safetensors::SafeTensors::read_metadata(bytes).expect("fixture metadata");
    serde_json::from_str(
        metadata
            .metadata()
            .as_ref()
            .expect("fixture metadata map")
            .get(key)
            .expect("fixture manifest key"),
    )
    .expect("parse fixture manifest")
}

/// The base golden fixture's bytes and manifest, with the manifest already
/// held against the checkpoint under test — the prologue every in-crate
/// oracle opens with. The bytes come back because `SafeTensors` borrows them.
pub(crate) fn golden_bytes(dir: &str) -> (Vec<u8>, serde_json::Value) {
    let bytes = std::fs::read(GOLDEN_PATH).expect("read fixture");
    let manifest = fixture_manifest(&bytes, METADATA_KEY);
    assert_checkpoint_matches(&manifest, dir);
    (bytes, manifest)
}

/// The generate fixture's three prompts. Greedy equality across differing
/// batch compositions only means something on prompts whose next-token
/// margins are real: synthetic id arithmetic drives the tower into a
/// degenerate distribution where the top two candidates sit within
/// reduction-order noise, and the comparison becomes a coin flip.
pub(crate) fn generate_fixture_prompts() -> Vec<Vec<u32>> {
    const PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/gemma4-12b-generate.safetensors"
    );
    let bytes = std::fs::read(PATH).expect("read generate fixture (dump on the box first)");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    ["a", "b", "c"]
        .iter()
        .map(|case| u32_tensor(&fixture, &format!("{case}_prompt")).1)
        .collect()
}

pub(crate) fn model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").expect(
        "PEGAINFER_TEST_MODEL_PATH must point at the Gemma 4 checkpoint selected for this gate",
    )
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// Mirrors the dumper: plain sha256 for the two config files, and the
/// sha256 of the safetensors header (8-byte LE length prefix) for the
/// weights file, which pins the tensor layout without reading 22 GiB.
pub(crate) fn assert_checkpoint_matches(manifest: &serde_json::Value, dir: &str) {
    let expected = manifest["file_sha256"]
        .as_object()
        .expect("manifest file_sha256");
    for (name, digest) in expected {
        let expected_hex = digest.as_str().expect("sha256 must be a string");
        let actual = if let Some(file) = name.strip_suffix("#header") {
            use std::io::Read as _;
            let mut handle =
                std::fs::File::open(std::path::Path::new(dir).join(file)).expect("open weights");
            let mut len_bytes = [0u8; 8];
            handle.read_exact(&mut len_bytes).expect("header length");
            let mut header = vec![0u8; u64::from_le_bytes(len_bytes) as usize];
            handle.read_exact(&mut header).expect("header bytes");
            sha256_hex(&header)
        } else {
            let bytes =
                std::fs::read(std::path::Path::new(dir).join(name)).expect("read config file");
            sha256_hex(&bytes)
        };
        assert_eq!(
            &actual, expected_hex,
            "{name} does not match the fixture's pinned checkpoint; this \
             oracle runs against that checkpoint only"
        );
    }
}

/// Log-probabilities at `ids`, plus our own argmax. The 262k-way sum
/// accumulates in f64 so the reference's own f32 log_softmax is what the
/// comparison is limited by.
pub(crate) fn log_softmax_at(row: &[f32], ids: &[i32]) -> (Vec<f32>, usize) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&x| f64::from(x - max).exp()).sum();
    let log_z = sum.ln() as f32;
    let picked = ids
        .iter()
        .map(|&id| row[usize::try_from(id).expect("token id")] - max - log_z)
        .collect();
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty row");
    (picked, argmax)
}

pub(crate) fn i32_tensor(
    fixture: &safetensors::SafeTensors<'_>,
    name: &str,
) -> (Vec<usize>, Vec<i32>) {
    let view = fixture.tensor(name).expect("fixture tensor");
    assert_eq!(view.dtype(), safetensors::Dtype::I32, "{name} dtype");
    let host = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), host)
}

pub(crate) fn u32_tensor(
    fixture: &safetensors::SafeTensors<'_>,
    name: &str,
) -> (Vec<usize>, Vec<u32>) {
    let (shape, tokens) = i32_tensor(fixture, name);
    let tokens = tokens
        .into_iter()
        .map(|token| u32::try_from(token).expect("token id fits u32"))
        .collect();
    (shape, tokens)
}

pub(crate) fn f32_tensor(
    fixture: &safetensors::SafeTensors<'_>,
    name: &str,
) -> (Vec<usize>, Vec<f32>) {
    let view = fixture.tensor(name).expect("fixture tensor");
    assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name} dtype");
    let host = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), host)
}
