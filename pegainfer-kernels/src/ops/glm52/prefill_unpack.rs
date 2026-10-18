use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

#[cfg(test)]
const PAGE: usize = 64;
const LATENT: usize = 576;

/// Unpack whole wire pages into the dense bf16 latent buffer. The packed
/// cache is addressed as `layer_offset + block * block_stride + token *
/// packed_bytes` — a dedicated arena passes offset 0 and the tight
/// `64 * packed_bytes` stride; the page-first slab passes its layer offset
/// and whole-page stride. The unpacked buffer stays dense.
#[allow(clippy::too_many_arguments)]
pub fn glm52_prefill_unpack_pages_launch(
    ctx: &DeviceContext,
    packed: &CudaSlice<u8>,
    packed_layer_offset: usize,
    packed_block_stride: usize,
    packed_bytes: usize,
    block_ids: &CudaSlice<i32>,
    blocks: usize,
    unpacked: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        blocks > 0 && block_ids.len() >= blocks,
        "GLM5.2 prefill unpack needs at least one block id"
    );
    ensure!(
        !unpacked.is_empty() && unpacked.len().is_multiple_of(LATENT),
        "GLM5.2 prefill unpack output extent is invalid"
    );
    let max_slots = unpacked.len() / LATENT;
    let max_blocks = max_slots / 64;
    ensure!(
        unpacked.len() == max_slots * LATENT
            && matches!(packed_bytes, 576 | 656)
            && packed_block_stride >= 64 * packed_bytes
            && max_blocks > 0
            && packed.len()
                >= packed_layer_offset + (max_blocks - 1) * packed_block_stride + 64 * packed_bytes,
        "GLM5.2 prefill unpack cache extents disagree"
    );
    let (packed_ptr, _packed_guard) = packed.device_ptr(&ctx.stream);
    let (blocks_ptr, _blocks_guard) = block_ids.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = unpacked.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_prefill_unpack_pages_cuda(
            (packed_ptr + packed_layer_offset as u64) as *const u8,
            blocks_ptr as *const i32,
            blocks as i32,
            packed_bytes as i32,
            packed_block_stride as i64,
            max_slots as i64,
            out_ptr as *mut ffi::Half,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 prefill cache unpack failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA device"]
    fn unpack_supported_cache_layouts() -> Result<()> {
        const PACKED_BYTES: usize = 656;
        let ctx = DeviceContext::new()?;
        let mut packed = vec![0u8; PAGE * PACKED_BYTES];
        for token in 0..PAGE {
            let row = &mut packed[token * PACKED_BYTES..(token + 1) * PACKED_BYTES];
            row[..512].fill(0x38);
            for group in 0..4 {
                row[512 + group * 4..516 + group * 4]
                    .copy_from_slice(&(group as f32 + 1.0).to_ne_bytes());
            }
            for dim in 0..64 {
                row[528 + dim * 2..530 + dim * 2]
                    .copy_from_slice(&bf16::from_f32(dim as f32).to_bits().to_ne_bytes());
            }
        }
        let packed = ctx.stream.clone_htod(&packed)?;
        let blocks = ctx.stream.clone_htod(&[0i32])?;
        let mut unpacked = ctx.stream.alloc_zeros::<bf16>(PAGE * LATENT)?;
        glm52_prefill_unpack_pages_launch(
            &ctx,
            &packed,
            0,
            PAGE * PACKED_BYTES,
            PACKED_BYTES,
            &blocks,
            1,
            &mut unpacked,
        )?;
        let unpacked = ctx.stream.clone_dtoh(&unpacked)?;
        for token in 0..PAGE {
            for dim in 0..512 {
                ensure!(
                    unpacked[token * LATENT + dim].to_f32() == (dim / 128 + 1) as f32,
                    "bad dequant at token {token}, dim {dim}"
                );
            }
            for dim in 0..64 {
                ensure!(
                    unpacked[token * LATENT + 512 + dim].to_f32() == dim as f32,
                    "bad rope copy at token {token}, dim {dim}"
                );
            }
        }
        let packed = ctx.stream.clone_htod(&vec![0x38u8; PAGE * LATENT])?;
        let blocks = ctx.stream.clone_htod(&[0i32])?;
        let mut unpacked = ctx.stream.alloc_zeros::<bf16>(PAGE * LATENT)?;
        glm52_prefill_unpack_pages_launch(
            &ctx,
            &packed,
            0,
            PAGE * LATENT,
            LATENT,
            &blocks,
            1,
            &mut unpacked,
        )?;
        let unpacked = ctx.stream.clone_dtoh(&unpacked)?;
        ensure!(
            unpacked.iter().all(|value| value.to_f32() == 1.0),
            "FlashInfer static-FP8 page unpack mismatch"
        );
        Ok(())
    }
}
