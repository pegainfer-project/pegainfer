use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

/// Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
pub fn embedding_decode_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_id.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_decode_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Slice-level batched embedding gather: `out[r] = table[ids[r]]` for `rows`
/// rows of `table.cols`. Same kernel as [`embedding_batch`], for callers whose
/// output lives in a persistent decode arena (also reused as the per-token
/// rope-table row gather — a bit-exact row copy).
pub fn embedding_rows_into(
    ctx: &DeviceContext,
    table: &DeviceMatrix,
    ids: &CudaSlice<u32>,
    rows: usize,
    out: &mut CudaSlice<half::bf16>,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "embedding_rows needs positive rows");
    anyhow::ensure!(
        ids.len() >= rows && out.len() >= rows * table.cols,
        "embedding_rows buffers too small: ids {}, out {} for {rows}x{}",
        ids.len(),
        out.len(),
        table.cols
    );
    let (e_ptr, _ge) = table.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = ids.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            table.cols as i32,
            rows as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Batched embedding lookup
pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

/// Vocab-sharded batched embedding lookup for tensor-parallel models.
///
/// Tokens outside `[vocab_start, vocab_start + part_vocab_size)` write zeros;
/// callers should all-reduce `out` across ranks to recover the full embedding.
pub fn embedding_batch_vocab_shard(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
    vocab_start: u32,
    part_vocab_size: u32,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_vocab_shard_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            vocab_start,
            part_vocab_size,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    /// The vectorized row-copy path (hidden % 8 == 0) must be bit-exact
    /// against a host-side gather — it is a pure copy.
    #[test]
    fn embedding_batch_vectorized_path_is_bit_exact() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let hidden_size = 2560; // production Qwen3 width -> vec4 path
        let vocab = 64;
        let seq_len = 33; // not a multiple of the block count on purpose
        let embed_host: Vec<bf16> = (0..vocab * hidden_size)
            .map(|i| bf16::from_f32((((i % 509) as f32) - 254.0) * 0.01))
            .collect();
        let embed = DeviceMatrix::from_host(&ctx, &embed_host, vocab, hidden_size).expect("embed");
        let ids: Vec<u32> = (0..seq_len).map(|i| ((i * 31) % vocab) as u32).collect();
        let token_ids = ctx.stream.clone_htod(&ids).expect("token ids");
        let mut out = HiddenStates::zeros(&ctx, hidden_size, seq_len).expect("out");

        embedding_batch(&ctx, &embed, &token_ids, &mut out).expect("embedding");
        let got = ctx.stream.clone_dtoh(&out.data).expect("dtoh");
        ctx.sync().expect("sync");

        for (token, &id) in ids.iter().enumerate() {
            let row = &embed_host[id as usize * hidden_size..(id as usize + 1) * hidden_size];
            let col = &got[token * hidden_size..(token + 1) * hidden_size];
            assert!(
                row.iter().zip(col).all(|(a, b)| a.to_bits() == b.to_bits()),
                "token {token} (id {id}) differs from host gather"
            );
        }
    }

    /// Same contract for the vectorized vocab-shard path, including the
    /// zero-fill of non-local tokens.
    #[test]
    fn embedding_batch_vocab_shard_vectorized_masks_nonlocal_tokens() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let hidden_size = 16; // multiple of 8 -> vec4 path
        let embed_host: Vec<bf16> = (0..2 * hidden_size)
            .map(|i| bf16::from_f32(i as f32))
            .collect();
        let embed = DeviceMatrix::from_host(&ctx, &embed_host, 2, hidden_size).expect("embed");
        let token_ids = ctx.stream.clone_htod(&[4_u32, 5, 6, 3]).expect("token ids");
        let mut out = HiddenStates::zeros(&ctx, hidden_size, 4).expect("out");

        embedding_batch_vocab_shard(&ctx, &embed, &token_ids, &mut out, 4, 2)
            .expect("embedding shard");
        let got = ctx.stream.clone_dtoh(&out.data).expect("dtoh");
        ctx.sync().expect("sync");

        assert!(
            got[..hidden_size]
                .iter()
                .zip(&embed_host[..hidden_size])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "local token 4 must copy row 0"
        );
        assert!(
            got[hidden_size..2 * hidden_size]
                .iter()
                .zip(&embed_host[hidden_size..])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "local token 5 must copy row 1"
        );
        assert!(
            got[2 * hidden_size..]
                .iter()
                .all(|v| v.to_bits() == bf16::ZERO.to_bits()),
            "non-local tokens 6 and 3 must be zero-filled"
        );
    }

    #[test]
    fn embedding_batch_vocab_shard_masks_nonlocal_tokens() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let hidden_size = 3;
        let seq_len = 4;
        let embed_host = vec![
            bf16::from_f32(10.0),
            bf16::from_f32(11.0),
            bf16::from_f32(12.0),
            bf16::from_f32(20.0),
            bf16::from_f32(21.0),
            bf16::from_f32(22.0),
        ];
        let embed = DeviceMatrix::from_host(&ctx, &embed_host, 2, hidden_size).expect("embed");
        let token_ids = ctx.stream.clone_htod(&[4_u32, 5, 6, 4]).expect("token ids");
        let mut out = HiddenStates::zeros(&ctx, hidden_size, seq_len).expect("out");

        embedding_batch_vocab_shard(&ctx, &embed, &token_ids, &mut out, 4, 2)
            .expect("embedding shard");
        let got = ctx.stream.clone_dtoh(&out.data).expect("dtoh");
        ctx.sync().expect("sync");
        let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

        assert_eq!(
            got,
            vec![
                10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0
            ]
        );
    }
}
