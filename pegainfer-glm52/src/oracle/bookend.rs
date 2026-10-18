//! HF-oracle gate for the decode bookends (embed / final RMSNorm / lm_head).
//!
//! The oracle side is `tools/accuracy/glm52_oracle.py --stage bookend`. Three
//! assertions, typed by what each op can promise:
//! - embed: a bf16 gather is EXACT — assert the embed-rows digest bit-for-bit.
//! - logits: RMS-scaled probe tolerance (bf16 GEMV vs f32 reference matmul).
//! - argmax: EXACT per position — the only thing greedy decode consumes.
//!
//! Run (GPU + checkpoint):
//! ```text
//! PEGAINFER_TEST_MODEL_PATH=/data/models/GLM-5.2-FP8 \
//!   cargo test --release -p pegainfer-glm52 --features glm52 --lib bookend_oracle -- --ignored --nocapture
//! ```

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use sha2::Digest;
use sha2::Sha256;

use crate::bookend::glm52_embed_into;
use crate::bookend::glm52_final_norm_into;
use crate::bookend::glm52_lm_head_into;
use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_VOCAB;
use crate::rows::Rows;

/// Allocating conveniences over the production `_into` bookends — the gate
/// probes one row at a time, so a fresh output per call is the natural shape.
fn glm52_embed(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
) -> Result<Rows<GLM52_HIDDEN>> {
    let mut out = Rows::zeros(ctx, 1)?;
    glm52_embed_into(ctx, embed, token_id, &mut out)?;
    Ok(out)
}

fn glm52_final_norm(
    ctx: &DeviceContext,
    hidden: &Rows<GLM52_HIDDEN>,
    norm_weight: &DeviceVec,
) -> Result<Rows<GLM52_HIDDEN>> {
    let mut out = Rows::zeros(ctx, hidden.tokens())?;
    glm52_final_norm_into(ctx, hidden, norm_weight, &mut out)?;
    Ok(out)
}

fn glm52_lm_head(
    ctx: &DeviceContext,
    normed: &Rows<GLM52_HIDDEN>,
    lm_head: &DeviceMatrix,
) -> Result<Rows<GLM52_VOCAB>> {
    let mut out = Rows::zeros(ctx, normed.tokens())?;
    glm52_lm_head_into(ctx, normed, lm_head, &mut out)?;
    Ok(out)
}

// ---- BEGIN GENERATED: glm52_oracle bookend probes ----
// uv run tools/accuracy/glm52_oracle.py --model-path /data/models/GLM-5.2-FP8 \
//     --ctx 8 --seed 0x5eed604d --precision fp8sim --stage bookend
// transformers=5.13.0.dev0 torch=2.12.1+cu130
const ORACLE_SEED: u64 = 0x5eed604d;
const ORACLE_CTX: usize = 8;
// sha256[..16] of the seeded bf16 input — a mismatch means PRNG drift, not a kernel bug.
const ORACLE_HIDDEN_DIGEST: &str = "c097c285d203c868";
// Token-id gather is exact: assert the embed-rows digest bit-for-bit.
const ORACLE_EMBED_ROWS_DIGEST: &str = "056c65fbfa4faf21";
const ORACLE_LOGITS_RMS: f32 = 1.809364200e+00;
const ORACLE_LOGITS_REL_TOL: f32 = 0.05;
const ORACLE_LOGITS_PROBES: &[(usize, f32)] = &[
    (34044, 2.968750000e+00),
    (46885, 9.326171875e-02),
    (52534, -3.398437500e-01),
    (54081, 3.046875000e+00),
    (81787, -3.828125000e+00),
    (104542, 8.125000000e-01),
    (118311, 2.453125000e+00),
    (119542, 2.218750000e+00),
    (136215, 6.015625000e-01),
    (140227, 8.984375000e-01),
    (154150, 2.531250000e+00),
    (160553, -4.687500000e-01),
    (162111, -1.078125000e+00),
    (164880, 1.726562500e+00),
    (210477, -9.492187500e-01),
    (231313, -1.265625000e+00),
    (246646, 1.671875000e+00),
    (253691, -2.480468750e-01),
    (278726, -2.285156250e-01),
    (279906, 4.843750000e+00),
    (293007, -2.312500000e+00),
    (296335, 5.781250000e-01),
    (342450, 5.981445312e-02),
    (346740, 1.171875000e+00),
    (354239, -2.937500000e+00),
    (387095, 3.093750000e+00),
    (394312, -7.177734375e-02),
    (400923, 1.898437500e+00),
    (428649, 3.968750000e+00),
    (441435, 3.140625000e+00),
    (442961, -6.406250000e-01),
    (460731, 3.979492188e-02),
    (463722, 3.710937500e-01),
    (559204, 1.796875000e+00),
    (566926, 1.390625000e+00),
    (603057, -1.664062500e+00),
    (620495, -9.882812500e-01),
    (669421, -1.585937500e+00),
    (678337, -4.042968750e-01),
    (723462, 2.187500000e+00),
    (747398, 6.132812500e-01),
    (794574, -1.976562500e+00),
    (809109, 2.828125000e+00),
    (816683, -3.300781250e-01),
    (847522, -6.679687500e-01),
    (850393, 1.335937500e+00),
    (861762, -3.171875000e+00),
    (868002, 2.859375000e+00),
    (895871, 2.312500000e+00),
    (929445, -1.507812500e+00),
    (932991, -2.783203125e-02),
    (948177, -4.550781250e-01),
    (973467, 2.714843750e-01),
    (983984, -1.265625000e+00),
    (1003190, -3.457031250e-01),
    (1020372, 1.679687500e+00),
    (1030232, -4.863281250e-01),
    (1030967, 8.945312500e-01),
    (1042369, -5.195312500e-01),
    (1059970, 1.570312500e+00),
    (1128461, 7.470703125e-02),
    (1158347, -3.027343750e-01),
    (1212133, -2.640625000e+00),
    (1215440, 7.109375000e-01),
];
// Greedy argmax per position — exact.
const ORACLE_ARGMAX: &[u32] = &[78804, 67784, 81537, 17, 16222, 36683, 19, 14109];
// ---- END GENERATED ----

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

/// Mirror of the harness `seeded_token_ids`: splitmix64 uniform on `seed ^
/// 0xB00C`, scaled to the vocab (f64 multiply + truncation, exact both sides).
fn seeded_token_ids(seed: u64, count: usize, vocab: usize) -> Vec<u32> {
    let seed = seed ^ 0xB00C;
    (0..count)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            ((u * vocab as f64) as u64).min(vocab as u64 - 1) as u32
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

/// Load one whole bf16 tensor from the checkpoint by name.
fn load_bf16(model_path: &Path, name: &str) -> Result<Vec<u8>> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        model_path.join("model.safetensors.index.json"),
    )?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} missing from weight map"))?;
    let mmap = crate::weights::mmap_file(&model_path.join(shard))?;
    let st = safetensors::SafeTensors::deserialize(mmap.as_ref())?;
    Ok(st.tensor(name)?.data().to_vec())
}

#[test]
#[ignore = "requires GPU + GLM-5.2-FP8 checkpoint"]
fn bookend_oracle_gate() -> Result<()> {
    assert!(
        !ORACLE_LOGITS_PROBES.is_empty(),
        "probe block is the ungenerated placeholder — run the harness and paste"
    );
    let model_path = std::env::var_os("PEGAINFER_TEST_MODEL_PATH")
        .map_or_else(|| PathBuf::from("models/GLM-5.2-FP8"), PathBuf::from);

    let hidden_host = seeded_hidden(ORACLE_SEED, ORACLE_CTX * GLM52_HIDDEN);
    let digest = bf16_digest(&hidden_host);
    ensure!(
        digest == ORACLE_HIDDEN_DIGEST,
        "input digest {digest} != oracle {ORACLE_HIDDEN_DIGEST}: PRNG drift or stale probes"
    );
    let token_ids = seeded_token_ids(ORACLE_SEED, ORACLE_CTX, GLM52_VOCAB);

    let ctx = DeviceContext::new()?;
    let embed = DeviceMatrix::from_safetensors(
        &ctx,
        &load_bf16(&model_path, "model.embed_tokens.weight")?,
        GLM52_VOCAB,
        GLM52_HIDDEN,
    )?;
    let norm_weight =
        DeviceVec::from_safetensors(&ctx, &load_bf16(&model_path, "model.norm.weight")?)?;
    let lm_head = DeviceMatrix::from_safetensors(
        &ctx,
        &load_bf16(&model_path, "lm_head.weight")?,
        GLM52_VOCAB,
        GLM52_HIDDEN,
    )?;

    // ---- embed: exact gather digest ----
    let mut token_id_buf = ctx.stream.alloc_zeros::<u32>(1)?;
    let mut embed_rows: Vec<bf16> = Vec::with_capacity(ORACLE_CTX * GLM52_HIDDEN);
    for &id in &token_ids {
        ctx.stream.memcpy_htod(&[id], &mut token_id_buf)?;
        let row = glm52_embed(&ctx, &embed, &token_id_buf)?;
        embed_rows.extend(ctx.stream.clone_dtoh(row.data())?);
    }
    let embed_digest = bf16_digest(&embed_rows);
    ensure!(
        embed_digest == ORACLE_EMBED_ROWS_DIGEST,
        "embed rows digest {embed_digest} != oracle {ORACLE_EMBED_ROWS_DIGEST} (gather is exact — this is a hard bug)"
    );
    println!("bookend embed: digest exact ({ORACLE_CTX} rows)");

    // ---- final norm + lm_head: probes + exact argmax ----
    ensure!(ORACLE_ARGMAX.len() == ORACLE_CTX, "argmax length mismatch");
    let mut logits_all: Vec<f32> = Vec::with_capacity(ORACLE_CTX * GLM52_VOCAB);
    for position in 0..ORACLE_CTX {
        let mut hidden = Rows::<GLM52_HIDDEN>::zeros(&ctx, 1)?;
        ctx.stream.memcpy_htod(
            &hidden_host[position * GLM52_HIDDEN..(position + 1) * GLM52_HIDDEN],
            hidden.data_mut(),
        )?;
        let normed = glm52_final_norm(&ctx, &hidden, &norm_weight)?;
        let logits = glm52_lm_head(&ctx, &normed, &lm_head)?;
        let logits_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(logits.data())?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let argmax = logits_host
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        ensure!(
            argmax == ORACLE_ARGMAX[position],
            "argmax mismatch at position {position}: engine {argmax} vs oracle {}",
            ORACLE_ARGMAX[position]
        );
        logits_all.extend(logits_host);
    }
    println!("bookend argmax: exact across {ORACLE_CTX} positions");

    let tol = ORACLE_LOGITS_REL_TOL * ORACLE_LOGITS_RMS;
    let failures: Vec<_> = ORACLE_LOGITS_PROBES
        .iter()
        .filter(|&&(idx, expected)| (logits_all[idx] - expected).abs() > tol)
        .collect();
    println!(
        "bookend logits: {}/{} probes within tol={tol:.6e}",
        ORACLE_LOGITS_PROBES.len() - failures.len(),
        ORACLE_LOGITS_PROBES.len()
    );
    for &&(idx, expected) in failures.iter().take(10) {
        println!(
            "  probe[{idx}]: oracle {expected:.6} vs engine {:.6}",
            logits_all[idx]
        );
    }
    ensure!(
        failures.is_empty(),
        "{} probes out of tolerance",
        failures.len()
    );
    Ok(())
}
