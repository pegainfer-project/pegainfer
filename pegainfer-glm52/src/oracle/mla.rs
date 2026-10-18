//! HF-oracle gate for the single-layer MLA decode brick.
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py` (pinned transformers
//! `glm_moe_dsa` — the official modeling code): it runs layer-0 attention on a
//! seeded input and emits the probe constants pasted below. This test replays
//! the *same* input through `glm52_mla_decode_forward` position by position
//! (prefill-via-decode, full top-k — DSA-equivalent at ctx <= 2048) and asserts
//! the outputs land on the probes within an RMS-scaled tolerance.
//!
//! Input generation is splitmix64 (integer-only), bit-identical across the two
//! languages; the input digest is asserted first so PRNG drift fails loudly and
//! separately from kernel bugs. Float probes are tolerance-checked, never
//! hash-checked: fp8 GEMM + absorbed-attention accumulation order cannot be
//! bit-equal to the HF decompress path.
//!
//! Run (H200 + checkpoint):
//! ```text
//! PEGAINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//!   cargo test --release -p pegainfer-glm52 --features glm52 --lib mla_oracle -- --ignored --nocapture
//! ```
//! Set `PEGAINFER_GLM52_ORACLE_DUMP=/path/taps.safetensors` (harness `--emit
//! safetensors`) to additionally diff the whole output tensor, not just probes.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE;
use pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_TOPK;
use pegainfer_kernels::ops::Glm52FlashMlaSparseDecode;
use pegainfer_kernels::ops::glm52_flashmla_sparse_decode_num_sm_parts;
use pegainfer_kernels::tensor::DeviceContext;
use sha2::Digest;
use sha2::Sha256;

use crate::config::GLM52_ROPE_HALF;
use crate::config::GLM52_SM_SCALE;
use crate::fp8::Glm52ProjBytes;
use crate::mla_decode::Glm52MlaSchedMetadata;
use crate::mla_decode::glm52_mla_decode_forward;
use crate::mla_front::Glm52MlaLayerWeights;
use crate::model::rope_tables;

// ---- BEGIN GENERATED: glm52_oracle probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path /work/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
//     --ctx 200 --seed 0x5eed604d --layer 0 --precision gemv
// transformers=5.14.0.dev0 torch=2.11.0+cu130
const ORACLE_SEED: u64 = 0x5eed604d;
const ORACLE_CTX: usize = 200;
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "922e6646a688905e";
// tap `o` [200, 6144] bf16 digest=0152ca4371a54aa2 (provenance only, never assert)
const ORACLE_O_RMS: f32 = 1.374694868e-03;
const ORACLE_O_REL_TOL: f32 = 0.05;
const ORACLE_O_PROBES: &[(usize, f32)] = &[
    (8720, -7.858276367e-04),
    (14476, 4.119873047e-04),
    (19280, 1.602172852e-03),
    (23544, 1.583099365e-04),
    (23874, 2.624511719e-03),
    (40579, 1.134872437e-04),
    (43624, 2.460479736e-04),
    (51849, -3.021240234e-03),
    (62323, 1.152038574e-03),
    (79939, -2.838134766e-03),
    (132890, -7.781982422e-04),
    (161812, -7.591247559e-04),
    (218021, 2.899169922e-04),
    (239838, 2.716064453e-03),
    (280089, -2.670288086e-04),
    (362277, 6.790161133e-04),
    (370383, -1.274108887e-03),
    (374208, 1.006126404e-04),
    (378353, 1.296997070e-04),
    (405409, 3.566741943e-04),
    (409140, -1.304626465e-03),
    (427475, -1.138448715e-05),
    (431425, 4.310607910e-04),
    (436346, 5.760192871e-04),
    (466158, 2.029418945e-03),
    (470207, -1.129150391e-03),
    (499911, -3.929138184e-04),
    (558784, -1.930236816e-03),
    (577517, -4.348754883e-04),
    (658200, -1.022338867e-03),
    (665506, 3.051757812e-03),
    (693297, 4.825592041e-04),
    (701374, 4.172325134e-05),
    (740279, -2.334594727e-03),
    (742773, 1.449584961e-03),
    (744036, 1.564025879e-04),
    (775501, -5.269050598e-05),
    (780318, 4.196166992e-05),
    (784846, 9.059906006e-05),
    (789177, -8.344650269e-06),
    (828119, -1.708984375e-03),
    (847914, -2.975463867e-03),
    (870453, -1.136779785e-03),
    (874510, 1.235961914e-03),
    (939675, 4.920959473e-04),
    (943269, -6.065368652e-04),
    (957049, 6.675720215e-05),
    (961775, -1.174926758e-03),
    (967873, 1.472473145e-03),
    (985001, 5.836486816e-04),
    (1018444, -4.386901855e-04),
    (1028723, 5.111694336e-04),
    (1042431, -7.972717285e-04),
    (1050404, -5.416870117e-04),
    (1085060, -2.632141113e-04),
    (1115997, -1.693725586e-03),
    (1117939, -2.822875977e-04),
    (1156020, 1.068115234e-03),
    (1160003, -3.471374512e-04),
    (1163780, -4.787445068e-04),
    (1164135, -5.173683167e-05),
    (1194525, 6.256103516e-04),
    (1200098, 7.324218750e-04),
    (1225370, 3.261566162e-04),
];
// ---- END GENERATED ----

const HIDDEN: usize = 6144;

/// splitmix64 -> 53-bit uniform -> (u - 0.5) * 4.0 -> f32 -> bf16. Mirror of
/// the Python generator, including the f64 -> f32 -> bf16 double rounding
/// (numpy `.astype(np.float32)` then torch's bf16 cast); a direct f64 -> bf16
/// rounds a handful of values differently and fails the digest.
fn seeded_hidden(seed: u64, count: usize) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            bf16::from_f32(((u - 0.5) * 4.0) as f32)
        })
        .collect()
}

fn bf16_digest(data: &[bf16]) -> String {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

/// Copy layer-0 attention tensors out of the checkpoint shards. Owned copies
/// (~250 MB total) keep the borrow story trivial; this is a test-only path.
struct Layer0Tensors {
    by_name: BTreeMap<String, Vec<u8>>,
}

impl Layer0Tensors {
    fn load(model_path: &Path) -> Result<Self> {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            model_path.join("model.safetensors.index.json"),
        )?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("weight_map missing")?;
        let prefix = "model.layers.0.self_attn";
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::default();
        for (name, shard) in weight_map {
            if name.starts_with(prefix) && !name.contains(".indexer.") {
                by_shard
                    .entry(shard.as_str().context("shard not a string")?.to_owned())
                    .or_default()
                    .push(name.clone());
            }
        }
        let mut by_name = BTreeMap::new();
        for (shard, names) in by_shard {
            let mmap = crate::weights::mmap_file(&model_path.join(&shard))?;
            let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
            for name in names {
                by_name.insert(name.clone(), st.tensor(&name)?.data().to_vec());
            }
        }
        Ok(Self { by_name })
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.by_name
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("layer-0 tensor {name} not loaded"))
    }

    fn proj(&self, stem: &str, n: usize, k: usize) -> Result<Glm52ProjBytes<'_>> {
        Ok(Glm52ProjBytes {
            weight: self.bytes(&format!("{stem}.weight"))?,
            scale: self.bytes(&format!("{stem}.weight_scale_inv"))?,
            n,
            k,
        })
    }
}

fn load_layer0(ctx: &DeviceContext, model_path: &Path) -> Result<Glm52MlaLayerWeights> {
    let t = Layer0Tensors::load(model_path)?;
    let p = "model.layers.0.self_attn";
    Glm52MlaLayerWeights::from_host(
        ctx,
        &t.proj(&format!("{p}.q_a_proj"), 2048, HIDDEN)?,
        t.bytes(&format!("{p}.q_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.q_b_proj"), 16384, 2048)?,
        &t.proj(&format!("{p}.kv_a_proj_with_mqa"), 576, HIDDEN)?,
        t.bytes(&format!("{p}.kv_a_layernorm.weight"))?,
        &t.proj(&format!("{p}.kv_b_proj"), 28672, 512)?,
        &t.proj(&format!("{p}.o_proj"), HIDDEN, 16384)?,
    )
}

#[test]
#[ignore = "requires H200 + GLM-5.2-FP8 checkpoint"]
fn mla_oracle_gate() -> Result<()> {
    run_mla_oracle_gate(GLM52_FLASHMLA_SPARSE_TOPK, None)
}

fn run_mla_oracle_gate(topk: usize, sm_parts_cap: Option<usize>) -> Result<()> {
    let model_path = std::env::var_os("PEGAINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);

    let hidden_host = seeded_hidden(ORACLE_SEED, ORACLE_CTX * HIDDEN);
    let digest = bf16_digest(&hidden_host);
    ensure!(
        digest == ORACLE_HIDDEN_DIGEST,
        "input digest {digest} != oracle {ORACLE_HIDDEN_DIGEST}: PRNG drift or stale probes — regenerate with tools/accuracy/glm52_oracle.py"
    );

    let ctx = DeviceContext::new()?;
    let w = load_layer0(&ctx, &model_path)?;

    let device_sm_parts = glm52_flashmla_sparse_decode_num_sm_parts()?;
    let contract = Glm52FlashMlaSparseDecode {
        batch_size: 1,
        num_blocks: ORACLE_CTX.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE),
        // Dedicated single-layer cache: tight page stride, no layer offset.
        kv_layer_offset_bytes: 0,
        kv_block_stride_bytes: crate::model::GLM52_KV_PAGE_MLA_BYTES,
        topk,
        num_sm_parts: sm_parts_cap.map_or(device_sm_parts, |cap| device_sm_parts.min(cap)),
        sm_scale: GLM52_SM_SCALE,
    };
    let mut cache = ctx
        .stream
        .alloc_zeros::<u8>(contract.packed_kv_cache_len())?;
    let mla_sched = Glm52MlaSchedMetadata::new(&ctx, contract, w.heads)?;

    // Prefill via decode: position p writes its token into the cache, then
    // attends over the full prefix [0..=p] via a -1-padded top-k list.
    let mut outputs = Vec::with_capacity(ORACLE_CTX * HIDDEN);
    for position in 0..ORACLE_CTX {
        let mut hidden = crate::rows::Rows::<HIDDEN>::zeros(&ctx, 1)?;
        ctx.stream.memcpy_htod(
            &hidden_host[position * HIDDEN..(position + 1) * HIDDEN],
            hidden.data_mut(),
        )?;
        let (cos_host, sin_host) = rope_tables(position);
        let mut cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
        let mut sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
        ctx.stream.memcpy_htod(&cos_host, &mut cos)?;
        ctx.stream.memcpy_htod(&sin_host, &mut sin)?;

        let mut topk_host = vec![-1i32; topk];
        for (slot, v) in topk_host.iter_mut().enumerate().take(position + 1) {
            *v = slot as i32;
        }
        let mut topk_dev = ctx.stream.alloc_zeros::<i32>(topk)?;
        ctx.stream.memcpy_htod(&topk_host, &mut topk_dev)?;

        let o = glm52_mla_decode_forward(
            &ctx, &w, &hidden, &cos, &sin, &mut cache, position, &topk_dev, &mla_sched,
        )?;
        let o_host = ctx.stream.clone_dtoh(o.data())?;
        outputs.extend(o_host.iter().map(|v| v.to_f32()));
    }

    assert_probes(&outputs);
    if let Some(dump) = std::env::var_os("PEGAINFER_GLM52_ORACLE_DUMP") {
        diff_against_dump(&outputs, Path::new(&dump))?;
    }
    Ok(())
}

/// Probe assertion: |engine - oracle| <= rel_tol * oracle_rms at every sampled
/// index. RMS-scaled (not per-element relative) because near-zero elements have
/// huge relative error at bf16 while being irrelevant to the layer output.
fn assert_probes(outputs: &[f32]) {
    assert!(
        !ORACLE_O_PROBES.is_empty(),
        "probe block is the ungenerated placeholder"
    );
    let tol = ORACLE_O_REL_TOL * ORACLE_O_RMS;
    let failures: Vec<_> = ORACLE_O_PROBES
        .iter()
        .filter(|&&(idx, expected)| (outputs[idx] - expected).abs() > tol)
        .collect();
    println!(
        "oracle gate: {}/{} probes within tol={tol:.6e}",
        ORACLE_O_PROBES.len() - failures.len(),
        ORACLE_O_PROBES.len()
    );
    for &&(idx, expected) in failures.iter().take(10) {
        println!(
            "  probe[{idx}]: oracle {expected:.6} vs engine {:.6}",
            outputs[idx]
        );
    }
    assert!(
        failures.is_empty(),
        "{} probes out of tolerance",
        failures.len()
    );
}

/// Whole-tensor diff against the harness's safetensors dump (`o` tap): a
/// probes-pass/rest-garbage failure mode cannot hide from this. Asserts the
/// coverage-stable statistics (diff RMS and p99) — the absolute max grows with
/// element count (bf16 tail over 1.2M elements) so it is printed, not asserted;
/// same lesson as the qwen3 golden gate.
fn diff_against_dump(outputs: &[f32], dump: &Path) -> Result<()> {
    let mmap = crate::weights::mmap_file(dump)?;
    let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
    let oracle: Vec<f32> = st
        .tensor("o")?
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_le_bytes(*c).to_f32())
        .collect();
    ensure!(
        oracle.len() == outputs.len(),
        "dump `o` has {} elements, engine produced {}",
        oracle.len(),
        outputs.len()
    );
    let mut worst: Vec<(usize, f32)> = outputs
        .iter()
        .zip(&oracle)
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    let diff_rms = (worst.iter().map(|(_, d)| d * d).sum::<f32>() / worst.len() as f32).sqrt();
    let p99 = worst[worst.len() / 100].1;
    println!(
        "full-tensor diff vs dump: diff_rms={diff_rms:.6e}, p99={p99:.6e}, max={:.6e} (printed, not asserted), top offenders:",
        worst[0].1
    );
    for (i, d) in worst.iter().take(10) {
        println!(
            "  o[{i}]: engine {:.6} oracle {:.6} (|d|={d:.6})",
            outputs[*i], oracle[*i]
        );
    }
    let tol = ORACLE_O_REL_TOL * ORACLE_O_RMS;
    ensure!(
        diff_rms <= tol && p99 <= tol,
        "full-tensor diff stats out of tolerance: rms {diff_rms:.6e} / p99 {p99:.6e} vs tol {tol:.6e}"
    );
    Ok(())
}
