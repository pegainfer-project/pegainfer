use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::HostSlice;
use pegainfer_kernels::ffi;
use pegainfer_kernels::tensor::DeviceContext;

use super::SamplingScratch;
use super::TOP1_PACKET_BYTES;

impl SamplingScratch {
    pub(crate) fn read_top1(
        &mut self,
        ctx: &DeviceContext,
        active_rows: usize,
        vocab_size: usize,
    ) -> Result<Vec<(u32, f32)>> {
        {
            let (ids_ptr, _ids_guard) = self.top1_out.device_ptr(&ctx.stream);
            let (values_ptr, _values_guard) = self.top1_value_scratch.device_ptr(&ctx.stream);
            let (packets_ptr, _packets_guard) = self.top1_packets.device_ptr_mut(&ctx.stream);
            let status = unsafe {
                ffi::kimi_pack_top1_packets_cuda(
                    ids_ptr as *const i32,
                    values_ptr as *const ffi::Half,
                    packets_ptr as *mut core::ffi::c_void,
                    active_rows as i32,
                    ctx.stream.cu_stream(),
                )
            };
            ensure!(
                status == 0,
                "Kimi top1 packet launch failed: cudaError={status}"
            );
        }
        let active_bytes = active_rows * TOP1_PACKET_BYTES;
        {
            let (host_slice, _guard) =
                unsafe { self.top1_packets_host.stream_synced_mut_slice(&ctx.stream) };
            ctx.stream
                .memcpy_dtoh(
                    &self.top1_packets.slice(0..active_bytes),
                    &mut host_slice[..active_bytes],
                )
                .context("D2H Kimi batched top1 packet read failed")?;
        }
        let host = self
            .top1_packets_host
            .as_slice()
            .context("Kimi batched top1 packet wait failed")?;
        decode_top1_packets(&host[..active_bytes], vocab_size)
    }
}

fn decode_top1_packets(host: &[u8], vocab_size: usize) -> Result<Vec<(u32, f32)>> {
    let mut rows = Vec::with_capacity(host.len() / TOP1_PACKET_BYTES);
    for (row, packet) in host.chunks_exact(TOP1_PACKET_BYTES).enumerate() {
        let (id_bytes, value_bytes) = packet.split_at(size_of::<i32>());
        let top_id = i32::from_le_bytes(id_bytes.try_into()?);
        let value_bits = u16::from_le_bytes(value_bytes[..size_of::<u16>()].try_into()?);
        ensure!(
            top_id >= 0 && (top_id as usize) < vocab_size,
            "Kimi batched local top1 id {} at row {} out of logits range {}",
            top_id,
            row,
            vocab_size
        );
        rows.push((top_id as u32, half::bf16::from_bits(value_bits).to_f32()));
    }
    Ok(rows)
}
