//! vLLM-compatible block-hash derivation for cross-engine P/D KV lookup.
//!
//! When vLLM is the prefill node, its pegaflow connector registers KV blocks
//! under vLLM's own prefix-cache hashes (`vllm/v1/core/kv_cache_utils.py::
//! hash_block_tokens`). For an pegainfer decode node to find those blocks by
//! content, it must derive byte-identical keys from the same token sequence.
//! This module mirrors exactly one vLLM configuration — the only one that is
//! replicable outside Python:
//!
//! - `--prefix-caching-hash-algo xxhash_cbor`: key = `xxh3_128(cbor(input))`
//!   with canonical CBOR (RFC 8949). The non-`_cbor` variants serialize via
//!   pickle and cannot be reproduced here.
//! - `PYTHONHASHSEED` set on every vLLM process: the chain root `NONE_HASH` is
//!   `xxh3_128(cbor(seed_string))`. Unset, vLLM falls back to `os.urandom` and
//!   keys are unreproducible across processes — deployment must fail fast.
//! - Text-only requests: vLLM's `extra_keys` (multimodal, LoRA, cache salt)
//!   are `None`. Anything else diverges and must be caught by the compat gate.
//!
//! Per block the hashed input is the 3-tuple `(parent_hash: bytes,
//! token_ids: tuple[int, ...], None)`, encoded as a CBOR array. vLLM only
//! hashes full blocks; the partial tail block has no vLLM hash. The P/D tail
//! extension applies the same function to the partial token list, which both
//! sides can derive (`docs/models/glm52/pd-vllm-prefill.md` §3).

use xxhash_rust::xxh3::xxh3_128;

/// Key width: xxh3_128 digest.
pub(crate) const VLLM_HASH_BYTES: usize = 16;

/// Derives vLLM-compatible block-hash chains for one `(seed, block_size)`
/// configuration.
pub struct VllmBlockHasher {
    none_hash: [u8; VLLM_HASH_BYTES],
    block_size: usize,
}

impl VllmBlockHasher {
    /// `python_hash_seed` must equal the `PYTHONHASHSEED` value set on every
    /// vLLM prefill process; `block_size` must equal vLLM's hash block size.
    pub fn new(python_hash_seed: &str, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        let mut seed_cbor = Vec::with_capacity(python_hash_seed.len() + 9);
        write_head(&mut seed_cbor, MAJOR_TSTR, python_hash_seed.len() as u64);
        seed_cbor.extend_from_slice(python_hash_seed.as_bytes());
        Self {
            none_hash: xxh3_128(&seed_cbor).to_be_bytes(),
            block_size,
        }
    }

    /// Hash one block: `parent` is the previous block's hash (`None` for the
    /// first block, which chains off `NONE_HASH`).
    fn hash_block(
        &self,
        parent: Option<&[u8; VLLM_HASH_BYTES]>,
        token_ids: &[u32],
    ) -> [u8; VLLM_HASH_BYTES] {
        let parent = parent.unwrap_or(&self.none_hash);
        // array(3): [ bstr(parent), array(n)(uint...), null ]
        let mut buf = Vec::with_capacity(2 + VLLM_HASH_BYTES + 2 + 9 + 5 * token_ids.len());
        write_head(&mut buf, MAJOR_ARRAY, 3);
        write_head(&mut buf, MAJOR_BSTR, VLLM_HASH_BYTES as u64);
        buf.extend_from_slice(parent);
        write_head(&mut buf, MAJOR_ARRAY, token_ids.len() as u64);
        for &t in token_ids {
            write_head(&mut buf, MAJOR_UINT, u64::from(t));
        }
        buf.push(CBOR_NULL);
        xxh3_128(&buf).to_be_bytes()
    }

    /// Key chain for a token sequence: one key per full block. vLLM never
    /// hashes the partial tail block; when the P/D tail extension lands it
    /// derives the extra key via [`Self::hash_block`] on the tail slice.
    pub fn key_chain(&self, token_ids: &[u32]) -> Vec<Vec<u8>> {
        let full = token_ids.len() / self.block_size;
        let mut keys = Vec::with_capacity(full);
        let mut parent: Option<[u8; VLLM_HASH_BYTES]> = None;
        for chunk in token_ids.chunks_exact(self.block_size) {
            let h = self.hash_block(parent.as_ref(), chunk);
            keys.push(h.to_vec());
            parent = Some(h);
        }
        keys
    }

    /// The chain root (`xxh3_128(cbor(seed_string))`), for startup-time
    /// fingerprint logging: the P and D values must match byte-for-byte.
    pub fn none_hash(&self) -> [u8; VLLM_HASH_BYTES] {
        self.none_hash
    }
}

const MAJOR_UINT: u8 = 0;
const MAJOR_BSTR: u8 = 2;
const MAJOR_TSTR: u8 = 3;
const MAJOR_ARRAY: u8 = 4;
const CBOR_NULL: u8 = 0xf6;

/// Canonical (minimal-length) CBOR head: major type + unsigned argument.
fn write_head(out: &mut Vec<u8>, major: u8, value: u64) {
    let m = major << 5;
    match value {
        0..=23 => out.push(m | value as u8),
        24..=0xff => {
            out.push(m | 0x18);
            out.push(value as u8);
        }
        0x100..=0xffff => {
            out.push(m | 0x19);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(m | 0x1a);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            out.push(m | 0x1b);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors captured from vLLM (pegaflow .venv, 2026-07-03):
    //   PYTHONHASHSEED=<seed> python -c "init_none_hash(xxhash_cbor); ..."
    // See docs/models/glm52/pd-vllm-prefill.md §3.

    fn hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;

        let mut encoded = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        encoded
    }

    #[test]
    fn none_hash_matches_vllm() {
        assert_eq!(
            hex(&VllmBlockHasher::new("0", 4).none_hash),
            "1ebe36576dcb573f26b99533a20aaeca"
        );
        assert_eq!(
            hex(&VllmBlockHasher::new("42", 4).none_hash),
            "a9d2e63407fce1a26c9dc9d7fa2d7caf"
        );
    }

    #[test]
    fn cbor_encoding_matches_cbor2_canonical() {
        // cbor2.dumps((NONE_HASH, (1,2,3,4), None), canonical=True)
        let h = VllmBlockHasher::new("0", 4);
        let mut buf = Vec::new();
        write_head(&mut buf, MAJOR_ARRAY, 3);
        write_head(&mut buf, MAJOR_BSTR, 16);
        buf.extend_from_slice(&h.none_hash);
        write_head(&mut buf, MAJOR_ARRAY, 4);
        for t in 1u64..=4 {
            write_head(&mut buf, MAJOR_UINT, t);
        }
        buf.push(CBOR_NULL);
        assert_eq!(
            hex(&buf),
            "83501ebe36576dcb573f26b99533a20aaeca8401020304f6"
        );
        // Multi-byte uint arguments (token id 100000 → 0x1a be32).
        let mut buf = Vec::new();
        write_head(&mut buf, MAJOR_ARRAY, 3);
        write_head(&mut buf, MAJOR_BSTR, 16);
        buf.extend_from_slice(&h.none_hash);
        write_head(&mut buf, MAJOR_ARRAY, 4);
        for t in 100_000u64..100_004 {
            write_head(&mut buf, MAJOR_UINT, t);
        }
        buf.push(CBOR_NULL);
        assert_eq!(
            hex(&buf),
            "83501ebe36576dcb573f26b99533a20aaeca841a000186a01a000186a11a000186a21a000186a3f6"
        );
    }

    #[test]
    fn block_chain_matches_vllm() {
        let h = VllmBlockHasher::new("0", 4);
        let b1 = h.hash_block(None, &[1, 2, 3, 4]);
        let b2 = h.hash_block(Some(&b1), &[5, 6, 7, 8]);
        let tail = h.hash_block(Some(&b2), &[9, 10]);
        assert_eq!(hex(&b1), "0a8577df5ee3430515a8cc1f6e3ac52e");
        assert_eq!(hex(&b2), "d152782cb4d753bde718a811a3b75e23");
        assert_eq!(hex(&tail), "7157acec76700e416a50d67e2334f6f6");
    }

    #[test]
    fn realistic_block_size_matches_vllm() {
        let h = VllmBlockHasher::new("0", 64);
        let tokens: Vec<u32> = (100_000..100_064).collect();
        let big = h.hash_block(None, &tokens);
        assert_eq!(hex(&big), "6e43e6e613b5ab91b1c6332dca7020f7");
    }

    #[test]
    fn key_chain_covers_full_blocks_only() {
        let h = VllmBlockHasher::new("0", 4);
        let tokens: Vec<u32> = (1..=10).collect();
        let keys = h.key_chain(&tokens);
        assert_eq!(keys.len(), 2, "partial tail block must not be keyed");
        assert_eq!(hex(&keys[0]), "0a8577df5ee3430515a8cc1f6e3ac52e");
        assert_eq!(hex(&keys[1]), "d152782cb4d753bde718a811a3b75e23");

        let aligned: Vec<u32> = (1..=8).collect();
        assert_eq!(h.key_chain(&aligned), keys[..2]);
    }
}
