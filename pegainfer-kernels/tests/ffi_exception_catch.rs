//! Drives decode at GQA group 7 (56/8) — unsupported even if group 5 is added
//! for Qwen3-14B — and asserts the FlashInfer throw surfaces as the -1
//! sentinel plus a message. Before the guard this aborted the whole process.

use std::ffi::CStr;
use std::ffi::c_void;
use std::ptr;

use anyhow::Result;
use anyhow::ensure;
use pegainfer_kernels::ffi;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
}

struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let rc = unsafe { cudaMalloc(&raw mut ptr, bytes) };
        ensure!(rc == 0, "cudaMalloc failed: {rc}");
        Ok(Self { ptr })
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr) };
    }
}

#[test]
fn unsupported_gqa_group_returns_sentinel_with_message() -> Result<()> {
    if unsafe { cudaSetDevice(0) } != 0 {
        eprintln!("skipping: no CUDA device");
        return Ok(());
    }
    let num_qo_heads = 56i32;
    let num_kv_heads = 8i32;
    let head_dim = 128i32;
    let page_size = 16i32;

    let q = DeviceBuf::alloc((num_qo_heads * head_dim) as usize * 2)?;
    let output = DeviceBuf::alloc((num_qo_heads * head_dim) as usize * 2)?;
    let kv = DeviceBuf::alloc((2 * page_size * num_kv_heads * head_dim) as usize * 2)?;
    // Shared backing for the six metadata arrays; the dispatch rejects the
    // group size before any kernel reads them.
    let meta = DeviceBuf::alloc(8 * std::mem::size_of::<i32>())?;

    let result = unsafe {
        ffi::paged_attention_decode_cuda(
            q.ptr.cast(),
            output.ptr.cast(),
            kv.ptr.cast(),
            0,
            (page_size * num_kv_heads * head_dim) as i64,
            meta.ptr.cast(),
            meta.ptr.cast(),
            meta.ptr.cast(),
            meta.ptr.cast(),
            meta.ptr.cast(),
            meta.ptr.cast(),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            1,
            (2 * page_size * num_kv_heads * head_dim) as i64,
            1.0,
            ptr::null_mut(),
        )
    };
    ensure!(result == -1, "expected the -1 sentinel, got {result}");

    let msg = unsafe { CStr::from_ptr(ffi::pegainfer_kernels_last_error()) }.to_string_lossy();
    // The exact wording belongs to FlashInfer.
    ensure!(msg.contains("group_size"), "unexpected message: {msg}");
    Ok(())
}
