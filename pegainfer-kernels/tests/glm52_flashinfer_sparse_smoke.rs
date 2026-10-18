//! Focused numerical gate for the GLM5.2 TP4 FlashInfer sparse MLA wrapper.
//!
//! Every case keeps the query exactly zero so softmax is uniform; the value
//! cache then makes the expected output computable exactly:
//! - uniform: every FP8 value element is one -> every output element is one.
//! - paged ramp: page p holds value 2^(p % 4), pages balanced -> every output
//!   element is 3.75. A wrong page stride or top-k gather breaks the balance.

#![cfg(feature = "glm52")]

use pegainfer_kernels::ops::GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES;
use pegainfer_kernels::ops::Glm52FlashInferSparseDecode;
use pegainfer_kernels::ops::glm52_flashinfer_sparse_mla_fp8_launch;
use pegainfer_kernels::ops::glm52_flashinfer_sparse_mla_supported;
use pegainfer_kernels::tensor::DeviceContext;

#[test]
fn glm52_flashinfer_sparse_fp8_uniform_value() {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    if !glm52_flashinfer_sparse_mla_supported(16).expect("query FlashInfer support") {
        eprintln!("skip: GLM5.2 FlashInfer sparse MLA requires SM100/SM103");
        return;
    }

    for topk_size in [256usize, 2048] {
        for batch_size in [1usize, 2, 4, 8, 16, 32] {
            let contract = Glm52FlashInferSparseDecode {
                batch_size,
                heads: 16,
                num_blocks: topk_size / 64,
                topk: topk_size,
                sm_scale: 0.0625,
            };
            // E4M3 encodings: zero = 0x00, one = 0x38.
            let query = ctx
                .stream
                .clone_htod(&vec![0x00u8; contract.query_len()])
                .expect("query H2D");
            let cache = ctx
                .stream
                .clone_htod(&vec![0x38u8; contract.cache_len()])
                .expect("cache H2D");
            let topk_host: Vec<i32> = (0..batch_size).flat_map(|_| 0..topk_size as i32).collect();
            let topk = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
            let seq_lens = ctx
                .stream
                .clone_htod(&vec![topk_size as i32; batch_size])
                .expect("seq_lens H2D");
            let mut out = ctx
                .stream
                .alloc_zeros(contract.output_len())
                .expect("output alloc");
            let mut workspace = ctx
                .stream
                .alloc_zeros::<u8>(GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES)
                .expect("workspace alloc");

            glm52_flashinfer_sparse_mla_fp8_launch(
                &ctx,
                contract,
                &query,
                &cache,
                &topk,
                &seq_lens,
                &mut out,
                &mut workspace,
            )
            .expect("FlashInfer sparse MLA launch");
            let host = ctx.stream.clone_dtoh(&out).expect("output D2H");
            ctx.sync().expect("CUDA sync");
            let max_error = host
                .iter()
                .map(|value| (value.to_f32() - 1.0).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 0.01,
                "batch {batch_size} topk {topk_size} max error {max_error}"
            );
        }
    }
}

#[test]
fn glm52_flashinfer_sparse_fp8_paged_ramp_value() {
    // E4M3 encodings for 1, 2, 4, 8 — one value per 64-token page, cycling.
    const PAGE_VALUES: [u8; 4] = [0x38, 0x40, 0x48, 0x50];
    const PAGE_TOKENS: usize = 64;
    // Uniform softmax over pages balanced across the four values:
    // (1 + 2 + 4 + 8) / 4 = 3.75, exact in bf16.
    const EXPECTED: f32 = 3.75;

    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    if !glm52_flashinfer_sparse_mla_supported(16).expect("query FlashInfer support") {
        eprintln!("skip: GLM5.2 FlashInfer sparse MLA requires SM100/SM103");
        return;
    }

    for topk_size in [256usize, 2048] {
        for batch_size in [1usize, 2, 4, 8, 16, 32] {
            let contract = Glm52FlashInferSparseDecode {
                batch_size,
                heads: 16,
                num_blocks: topk_size / PAGE_TOKENS,
                topk: topk_size,
                sm_scale: 0.0625,
            };
            let query = ctx
                .stream
                .clone_htod(&vec![0x00u8; contract.query_len()])
                .expect("query H2D");
            let token_bytes = contract.cache_len() / (contract.num_blocks * PAGE_TOKENS);
            let cache_host: Vec<u8> = (0..contract.cache_len())
                .map(|byte| PAGE_VALUES[byte / (PAGE_TOKENS * token_bytes) % PAGE_VALUES.len()])
                .collect();
            let cache = ctx.stream.clone_htod(&cache_host).expect("cache H2D");
            let topk_host: Vec<i32> = (0..batch_size).flat_map(|_| 0..topk_size as i32).collect();
            let topk = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
            let seq_lens = ctx
                .stream
                .clone_htod(&vec![topk_size as i32; batch_size])
                .expect("seq_lens H2D");
            let mut out = ctx
                .stream
                .alloc_zeros(contract.output_len())
                .expect("output alloc");
            let mut workspace = ctx
                .stream
                .alloc_zeros::<u8>(GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES)
                .expect("workspace alloc");

            glm52_flashinfer_sparse_mla_fp8_launch(
                &ctx,
                contract,
                &query,
                &cache,
                &topk,
                &seq_lens,
                &mut out,
                &mut workspace,
            )
            .expect("FlashInfer sparse MLA launch");
            let host = ctx.stream.clone_dtoh(&out).expect("output D2H");
            ctx.sync().expect("CUDA sync");
            let max_error = host
                .iter()
                .map(|value| (value.to_f32() - EXPECTED).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 0.04,
                "batch {batch_size} topk {topk_size} max error {max_error} (expected {EXPECTED})"
            );
        }
    }
}

#[test]
fn glm52_flashinfer_sparse_fp8_chunked_batch_row_placement() {
    // Batches above 8 are decomposed into consecutive {8, 4, 2, 1} launches.
    // Give every row its own expected value — row r's top-k gathers only the
    // pages of residue class r % 4, whose values cycle 1/2/4/8 — so a wrong
    // per-chunk query/indices/seq_lens/output offset lands a value in the
    // wrong row and fails loudly. 3 and 6 exercise remainder decomposition.
    const PAGE_VALUES: [u8; 4] = [0x38, 0x40, 0x48, 0x50];
    const EXPECTED: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
    const PAGE_TOKENS: usize = 64;
    const NUM_BLOCKS: usize = 16;
    const TOPK: usize = 256;

    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    if !glm52_flashinfer_sparse_mla_supported(16).expect("query FlashInfer support") {
        eprintln!("skip: GLM5.2 FlashInfer sparse MLA requires SM100/SM103");
        return;
    }

    for batch_size in [3usize, 6, 16, 32] {
        let contract = Glm52FlashInferSparseDecode {
            batch_size,
            heads: 16,
            num_blocks: NUM_BLOCKS,
            topk: TOPK,
            sm_scale: 0.0625,
        };
        let query = ctx
            .stream
            .clone_htod(&vec![0x00u8; contract.query_len()])
            .expect("query H2D");
        let token_bytes = contract.cache_len() / (NUM_BLOCKS * PAGE_TOKENS);
        let cache_host: Vec<u8> = (0..contract.cache_len())
            .map(|byte| PAGE_VALUES[byte / (PAGE_TOKENS * token_bytes) % PAGE_VALUES.len()])
            .collect();
        let cache = ctx.stream.clone_htod(&cache_host).expect("cache H2D");
        // Row r: the 4 pages p with p % 4 == r % 4, token-granular indices.
        let topk_host: Vec<i32> = (0..batch_size)
            .flat_map(|row| {
                (0..NUM_BLOCKS / 4).flat_map(move |group| {
                    let page = group * 4 + row % 4;
                    (0..PAGE_TOKENS as i32).map(move |t| (page * PAGE_TOKENS) as i32 + t)
                })
            })
            .collect();
        assert_eq!(topk_host.len(), batch_size * TOPK);
        let topk = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
        let seq_lens = ctx
            .stream
            .clone_htod(&vec![TOPK as i32; batch_size])
            .expect("seq_lens H2D");
        let mut out = ctx
            .stream
            .alloc_zeros(contract.output_len())
            .expect("output alloc");
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(GLM52_FLASHINFER_SPARSE_WORKSPACE_BYTES)
            .expect("workspace alloc");

        glm52_flashinfer_sparse_mla_fp8_launch(
            &ctx,
            contract,
            &query,
            &cache,
            &topk,
            &seq_lens,
            &mut out,
            &mut workspace,
        )
        .expect("FlashInfer sparse MLA launch");
        let host = ctx.stream.clone_dtoh(&out).expect("output D2H");
        ctx.sync().expect("CUDA sync");
        let row_len = contract.output_len() / batch_size;
        for (row, chunk) in host.chunks(row_len).enumerate() {
            let expected = EXPECTED[row % 4];
            let max_error = chunk
                .iter()
                .map(|value| (value.to_f32() - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 0.04 * expected,
                "batch {batch_size} row {row} max error {max_error} (expected {expected})"
            );
        }
    }
}
