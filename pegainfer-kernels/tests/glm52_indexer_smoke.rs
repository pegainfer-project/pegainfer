//! GLM5.2 DSA indexer kernel smoke tests.
//!
//! Validates that the hand-written and vendored wrapper kernels launch
//! correctly on Hopper (sm_90) and produce expected results for simple
//! inputs. Not an oracle gate — correctness is against simple host-side
//! reference implementations.
//!
//!   cargo test --release -p pegainfer-kernels --features glm52 \
//!     --test glm52_indexer_smoke -- --nocapture

#![cfg(feature = "glm52")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use half::bf16;
use pegainfer_kernels::ffi;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&raw mut ptr, bytes) })?;
        Ok(Self { ptr })
    }

    fn from_host<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::alloc(bytes)?;
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    buf.ptr,
                    data.as_ptr().cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(buf)
    }

    fn zeroed(bytes: usize) -> Result<Self> {
        let buf = Self::alloc(bytes)?;
        cuda_check(unsafe { cudaMemset(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    fn to_host<T: Copy + Default>(&self, count: usize) -> Result<Vec<T>> {
        let mut data = vec![T::default(); count];
        let bytes = count * size_of::<T>();
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    data.as_mut_ptr().cast::<c_void>(),
                    self.ptr,
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(data)
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}

fn cuda_check(code: i32) -> Result<()> {
    ensure!(code == 0, "CUDA runtime call failed with code {code}");
    Ok(())
}

fn cu_check(r: CUresult) -> Result<()> {
    ensure!(
        r == CUresult::CUDA_SUCCESS,
        "CUDA FFI call failed with code {r:?}"
    );
    Ok(())
}

/// Returns Ok(true) on skip (caller early-returns), Ok(false) on success.
const STREAM: CUstream = ptr::null_mut();

// ─── indexer cache: quant + pack → readback ───────────────────────────────

fn indexer_cache_round_trip() -> Result<()> {
    let tokens = 4;
    let head_dim = 128;
    let cache_block_size = 4;
    let stride = cache_block_size * (head_dim + 4); // 128 fp8 + 4 scale bytes per token
    let cache_bytes = stride; // 1 block

    let k_host: Vec<bf16> = (0..tokens * head_dim)
        .map(|idx| {
            let tok = idx / head_dim;
            let dim = idx % head_dim;
            bf16::from_f32((tok as f32 + 1.0) * 0.1 * (dim as f32 + 1.0))
        })
        .collect();
    let slot_mapping: Vec<i64> = (0..tokens as i64).collect();

    let k_dev = DeviceBuf::from_host(&k_host)?;
    let slot_dev = DeviceBuf::from_host(&slot_mapping)?;
    let cache_dev = DeviceBuf::zeroed(cache_bytes)?;

    cu_check(unsafe {
        ffi::glm52_indexer_k_quant_and_cache_cuda(
            k_dev.ptr as *const ffi::Half,
            cache_dev.ptr.cast::<u8>(),
            slot_dev.ptr as *const i64,
            tokens as i32,
            head_dim as i32,
            128, // quant_block_size
            cache_block_size as i32,
            stride as i64,
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Read the cache back directly: per block, block_size×head_dim fp8 bytes,
    // then block_size f32 scales.
    let cache_host: Vec<u8> = cache_dev.to_host(cache_bytes)?;
    let scale_base = cache_block_size * head_dim;
    let scales: Vec<f32> = (0..tokens)
        .map(|t| {
            let b = &cache_host[scale_base + t * 4..scale_base + (t + 1) * 4];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
        .collect();
    for (i, &s) in scales.iter().enumerate() {
        ensure!(s > 0.0, "indexer cache: scale[{}] = {}, expected > 0", i, s);
    }

    eprintln!(
        "indexer_cache_round_trip: {} scales all positive, OK",
        tokens
    );
    Ok(())
}

// Regression for P1: block_table_stride must be block_table_cols, not topk.
// 2 tokens, topk=4, block_table_cols=2. If stride were topk (4), token 1
// would read block_table[4..6] (OOB / wrong row). With correct stride=2,
// token 1 reads block_table[2..3].
fn local_topk_to_slots_multi_token_stride() -> Result<()> {
    // token 0: pages [10, 20], token 1: pages [30, 40]
    let block_table: Vec<i32> = vec![10, 20, 30, 40];
    let offsets: Vec<i32> = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let seq_lens: Vec<i32> = vec![4, 4];

    let num_slots = 2 * 4; // num_tokens * topk

    let offsets_dev = DeviceBuf::from_host(&offsets)?;
    let block_dev = DeviceBuf::from_host(&block_table)?;
    let seq_dev = DeviceBuf::from_host(&seq_lens)?;
    let slots_dev = DeviceBuf::zeroed(num_slots * size_of::<i32>())?;
    let lens_dev = DeviceBuf::zeroed(2 * size_of::<i32>())?;

    cu_check(unsafe {
        ffi::glm52_indexer_local_topk_to_slots_cuda(
            slots_dev.ptr.cast::<i32>(),
            lens_dev.ptr.cast::<i32>(),
            offsets_dev.ptr as *const i32,
            4, // local_topk_stride (== topk)
            seq_dev.ptr as *const i32,
            block_dev.ptr as *const i32,
            2, // block_table_stride (== block_table_cols, NOT topk)
            2, // block_table_cols
            2, // block_size
            4, // topk
            2, // num_tokens
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let slots: Vec<i32> = slots_dev.to_host(8)?;
    let lens: Vec<i32> = lens_dev.to_host(2)?;

    // token 0: offsets [0,1,2,3] -> pages [10,20], slots [20,21,40,41]
    assert_eq!(slots[0], 20, "t0 slot[0]");
    assert_eq!(slots[1], 21, "t0 slot[1]");
    assert_eq!(slots[2], 40, "t0 slot[2]");
    assert_eq!(slots[3], 41, "t0 slot[3]");
    // token 1: offsets [0,1,2,3] -> pages [30,40], slots [60,61,80,81]
    assert_eq!(slots[4], 60, "t1 slot[0]");
    assert_eq!(slots[5], 61, "t1 slot[1]");
    assert_eq!(slots[6], 80, "t1 slot[2]");
    assert_eq!(slots[7], 81, "t1 slot[3]");
    assert_eq!(lens, vec![4, 4], "topk_lens");

    eprintln!(
        "local_topk_to_slots_multi_token_stride: slots = {:?}, lens = {:?}, OK",
        slots, lens
    );
    Ok(())
}

// Regression for P2: per-row `lengths` must filter padded/stale logits.
// logits = [0..max_len), but lengths[0]=10 so only indices 6..9 are valid
// top-k. If lengths were ignored (old TopKDispatch path), the kernel would
// return indices 2044..2047 from the stale tail.
fn flashinfer_topk_lengths_filter() -> Result<()> {
    let max_len = 2048;
    let top_k = 4;

    let logits: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
    let lengths: Vec<i32> = vec![10];

    let logits_dev = DeviceBuf::from_host(&logits)?;
    let lengths_dev = DeviceBuf::from_host(&lengths)?;
    let indices_dev = DeviceBuf::zeroed(top_k * size_of::<i32>())?;
    let values_dev = DeviceBuf::zeroed(top_k * size_of::<f32>())?;

    let result = unsafe {
        ffi::glm52_flashinfer_topk_2048_cuda(
            logits_dev.ptr as *const f32,
            indices_dev.ptr.cast::<i32>(),
            values_dev.ptr.cast::<f32>(),
            lengths_dev.ptr as *const i32,
            1, // num_rows
            top_k as i32,
            max_len,
            STREAM,
        )
    };
    if result == CUresult::CUDA_ERROR_NOT_SUPPORTED as i32 {
        eprintln!("skip: FilteredTopK not supported on this GPU (smem too small)");
        return Ok(());
    }
    anyhow::ensure!(
        result == 0,
        "glm52_flashinfer_topk_2048_cuda failed with error {result}{}",
        pegainfer_kernels::ops::ffi_exception_message(result)
    );
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let mut sorted: Vec<i32> = indices_dev.to_host(top_k)?;
    sorted.sort_unstable();
    // With lengths=10, valid logits are indices 0..9 (values 0..9); top-4
    // are indices 6,7,8,9. Stale tail (2044..2047) must NOT win.
    assert_eq!(
        sorted,
        vec![6, 7, 8, 9],
        "top-k must respect lengths filter"
    );

    eprintln!(
        "flashinfer_topk_lengths_filter: indices = {:?} (sorted), OK",
        sorted
    );
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[test]
fn test_indexer_cache_round_trip() {
    indexer_cache_round_trip().expect("indexer cache round-trip");
}

#[test]
fn test_local_topk_to_slots_multi_token_stride() {
    local_topk_to_slots_multi_token_stride().expect("local_topk_to_slots multi-token stride");
}

#[test]
fn test_flashinfer_topk_lengths_filter() {
    flashinfer_topk_lengths_filter().expect("FlashInfer top-k lengths filter");
}

// ─── DeepGEMM paged MQA logits: JIT compile + launch smoke test ────────────
//
// This is the first DeepGEMM JIT kernel call in the codebase. The test
// verifies that the torch-free wrapper can:
//   1. Initialize the JIT compiler (needs PEGAINFER_DEEPGEMM_ROOT + CUDA_HOME)
//   2. Compile the paged MQA logits kernel via nvcc JIT
//   3. Launch both the metadata and logits kernels without crashing
//
// It does NOT validate logits correctness — that requires an oracle gate
// with HF reference outputs. The smoke test only checks launch success.

fn deepgemm_env_ready() -> bool {
    std::env::var("PEGAINFER_DEEPGEMM_ROOT").is_ok() && std::env::var("CUDA_HOME").is_ok()
}

fn deepgemm_paged_mqa_launch() -> Result<()> {
    if !deepgemm_env_ready() {
        eprintln!("skip: set PEGAINFER_DEEPGEMM_ROOT + CUDA_HOME to run DeepGEMM MQA test");
        return Ok(());
    }

    let batch_size = 1_i32;
    let next_n = 1_i32;
    let num_heads = 16_i32; // 128 % 16 == 0; stride=16B (≥16B TMA align); smem=125KB (<232KB)
    let head_dim = 128_i32;
    let block_kv = 64_i32; // DeepGEMM paged MQA logits requires BLOCK_KV=64
    let num_kv_blocks = 32_i32; // 2048 tokens / 64 per block
    let num_sms = 132_i32; // H200 has 132 SMs
    let logits_stride = 256_i32; // split_kv=256
    let block_table_stride = num_kv_blocks;

    // context_lens: each batch has 2048 KV tokens
    let context_lens_host = vec![2048_i32; batch_size as usize];
    let context_lens = DeviceBuf::from_host(&context_lens_host)?;

    // schedule_metadata: aligned_batch_size = 32 (batch=1 → align to 32)
    let sched_meta_len = 32_i32 as usize; // non-varlen: aligned_batch_size
    let schedule_metadata = DeviceBuf::zeroed(sched_meta_len * std::mem::size_of::<i32>())?;

    // Metadata kernel launch
    let r = unsafe {
        ffi::glm52_deepgemm_paged_mqa_metadata_cuda(
            context_lens.ptr.cast::<i32>(),
            schedule_metadata.ptr.cast::<i32>(),
            batch_size,
            next_n,
            block_kv,
            num_sms,
            false, // is_context_lens_2d
            false, // is_varlen
            std::ptr::null(),
            STREAM,
        )
    };
    cu_check(r)?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Logits kernel launch
    // q: [batch_size, next_n, num_heads, head_dim] fp8 = 1*1*4*128 = 512 bytes
    let q_bytes = (batch_size * next_n * num_heads * head_dim) as usize;
    let q = DeviceBuf::zeroed(q_bytes)?;

    // kv_cache: interleaved [block_kv * head_dim fp8 | block_kv * 4 f32 scale] per block.
    // Per-block stride = block_kv * (head_dim + 4) = 64 * 132 = 8448 bytes
    let kv_stride_bytes = (block_kv as i64) * ((head_dim as i64) + 4);
    let kv_bytes = (num_kv_blocks as usize) * (kv_stride_bytes as usize);
    let kv_cache = DeviceBuf::zeroed(kv_bytes)?;

    // weights: [batch_size * next_n, num_heads] f32 = 1*4*4 = 16 bytes
    let weights_bytes = (batch_size * next_n * num_heads) as usize * std::mem::size_of::<f32>();
    let weights = DeviceBuf::zeroed(weights_bytes)?;

    // logits: [batch_size, logits_stride] bf16 = 1*256*2 = 512 bytes
    let logits_bytes = (batch_size * logits_stride) as usize * std::mem::size_of::<bf16>();
    let logits = DeviceBuf::zeroed(logits_bytes)?;

    // block_table: [batch_size, block_table_stride] i32
    let block_table_host: Vec<i32> = (0..num_kv_blocks).collect();
    let block_table = DeviceBuf::from_host(&block_table_host)?;

    let r = unsafe {
        ffi::glm52_deepgemm_paged_mqa_logits_cuda(
            q.ptr,
            kv_cache.ptr,
            kv_stride_bytes,
            weights.ptr,
            context_lens.ptr as *const i32,
            logits.ptr,
            block_table.ptr as *const i32,
            std::ptr::null(),
            schedule_metadata.ptr.cast::<i32>(),
            batch_size,
            next_n,
            num_heads,
            head_dim,
            num_kv_blocks,
            block_kv,
            false, // is_context_lens_2d
            false, // is_varlen
            logits_stride,
            block_table_stride,
            num_sms,
            1, // q_elem_size (fp8)
            1, // kv_elem_size (fp8)
            4, // weights_elem_size (f32)
            4, // kv_scales_elem_size (f32)
            STREAM,
        )
    };
    cu_check(r)?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Verify logits are all zero (since all inputs are zeroed)
    let logits_host: Vec<bf16> = logits.to_host((batch_size * logits_stride) as usize)?;
    let all_zero = logits_host.iter().all(|v| v.to_f32() == 0.0);
    ensure!(
        all_zero,
        "DeepGEMM MQA logits should be all zero for zeroed inputs"
    );

    Ok(())
}

#[test]
fn test_deepgemm_paged_mqa_launch() {
    deepgemm_paged_mqa_launch().expect("DeepGEMM paged MQA logits launch");
}
