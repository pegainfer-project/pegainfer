//! Single-stream AscendCL device resources with explicit ownership.
//!
//! Every submission rebinds the ACL context on the calling thread and is
//! serialized by a mutex. Host copies complete before returning. Buffers retain
//! the device and synchronize before free.
//!
//! Executor ownership has two halves. CANN owns the *non-repeatable* executor
//! returned by an ACLNN `*GetWorkspaceSize` call and releases it after the op
//! runs, so the backend never destroys those (doing so double-frees on CANN
//! 9.2.1, measured in `aclnn-smoke/smoke_exec_lifecycle.cpp`). *Repeatable*
//! executors are not reclaimed by CANN: they are cached here and destroyed
//! together with their descriptors when the device is dropped.
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use anyhow::Result;
use anyhow::ensure;

use crate::sys;

static ACL_INIT: Once = Once::new();
static ACL_INIT_STATUS: AtomicI32 = AtomicI32::new(i32::MIN);

fn init_acl() -> Result<()> {
    ACL_INIT.call_once(|| {
        let status = { sys::aclInit(std::ptr::null()) };
        ACL_INIT_STATUS.store(status, Ordering::SeqCst);
    });
    check(ACL_INIT_STATUS.load(Ordering::SeqCst))
}

pub(crate) fn last_error() -> String {
    unsafe {
        let raw = sys::aclGetRecentErrMsg();
        if raw.is_null() {
            "(no ACL error message)".to_string()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

pub(crate) fn check(status: i32) -> Result<()> {
    if status != sys::ACL_SUCCESS {
        anyhow::bail!("Ascend operation failed ({status}): {}", last_error());
    }
    Ok(())
}

struct Inner {
    device_id: i32,
    context: *mut c_void,
    stream: *mut c_void,
    lock: Mutex<()>,
    /// Repeatable executors, one per (op, shape, address). CANN does not
    /// reclaim these, so `Drop` destroys them.
    gemm_cache: Mutex<HashMap<u64, (usize, u64)>>,
    /// Tensor descriptors referenced by the cached repeatable executors.
    /// Destroyed after the executors in `Drop`.
    keepalive: Mutex<Vec<*mut sys::aclTensor>>,
    /// Device scratch, keyed by byte size. Cached executors capture tensor
    /// addresses, so their inputs must stay at a stable address. Freed in
    /// `Drop`.
    scratch: Mutex<HashMap<usize, usize>>,
    /// Reused ACLNN workspaces, keyed by byte size. Freed in `Drop`.
    workspaces: Mutex<HashMap<usize, usize>>,
    /// Attention descriptors. Destroying them between IFA calls aborts on
    /// CANN 9.2.1, so they are cached and reused by address. Freed in `Drop`.
    descs: Mutex<HashMap<u64, Descs>>,
    /// IFA takes a host `aclIntArray`; one is cached per sequence length and
    /// freed in `Drop`.
    int_arrays: Mutex<HashMap<i64, usize>>,
}
// All access is serialized by `lock`; the raw handles are only touched under it.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // A normal shutdown drops the engine (and with it every buffer) on the
        // driver thread, so release everything this device retained instead of
        // leaving it to the OS. Order matters: repeatable executors reference
        // their tensor descriptors, and both reference device scratch.
        //
        // `aclrtResetDevice` / `aclFinalize` stay out of this path on purpose:
        // unit tests create several short-lived devices in one process, and
        // CANN 9.2.1 can hang in finalize at exit. Nothing this device owns is
        // left behind.
        let _ = sys::aclrtSetCurrentContext(self.context);
        let _ = sys::aclrtSynchronizeStream(self.stream);

        if let Ok(cache) = self.gemm_cache.get_mut() {
            for (executor, _) in cache.values() {
                if *executor != 0 {
                    let _ = sys::aclDestroyAclOpExecutor(*executor as *mut sys::aclOpExecutor);
                }
            }
            cache.clear();
        }
        if let Ok(keep) = self.keepalive.get_mut() {
            for tensor in keep.drain(..) {
                let _ = sys::aclDestroyTensor(tensor);
            }
        }
        if let Ok(descs) = self.descs.get_mut() {
            for d in descs.values() {
                for &tensor in &d.tensors {
                    let _ = sys::aclDestroyTensor(tensor as *const sys::aclTensor);
                }
                for &list in &d.lists {
                    let _ = sys::aclDestroyTensorList(list as *const sys::aclTensorList);
                }
            }
            descs.clear();
        }
        if let Ok(arrays) = self.int_arrays.get_mut() {
            for &array in arrays.values() {
                let _ = sys::aclDestroyIntArray(array as *const sys::aclIntArray);
            }
            arrays.clear();
        }
        if let Ok(scratch) = self.scratch.get_mut() {
            for &ptr in scratch.values() {
                let _ = sys::aclrtFree(ptr as *mut c_void);
            }
            scratch.clear();
        }
        if let Ok(workspaces) = self.workspaces.get_mut() {
            for &ptr in workspaces.values() {
                let _ = sys::aclrtFree(ptr as *mut c_void);
            }
            workspaces.clear();
        }

        let _ = sys::aclrtDestroyStream(self.stream);
        let _ = sys::aclrtDestroyContext(self.context);
    }
}

/// Cached attention descriptors. `tensors` holds only descriptors that are not
/// members of a list: CANN 9.2.1 double-frees if both a list and a tensor it
/// contains are destroyed. The lists own their member tensors.
#[derive(Clone)]
pub(crate) struct Descs {
    pub(crate) tensors: Vec<usize>,
    pub(crate) lists: Vec<usize>,
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub soc: String,
    pub device_id: i32,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        init_acl()?;
        let device_id = i32::try_from(ordinal)?;
        check(sys::aclrtSetDevice(device_id))?;
        let mut context = std::ptr::null_mut();
        check(sys::aclrtCreateContext(&raw mut context, device_id))?;
        check(sys::aclrtSetCurrentContext(context))?;
        let mut stream = std::ptr::null_mut();
        check(sys::aclrtCreateStream(&raw mut stream))?;
        Ok(Self {
            inner: Arc::new(Inner {
                device_id,
                context,
                stream,
                lock: Mutex::new(()),
                gemm_cache: Mutex::new(HashMap::new()),
                keepalive: Mutex::new(Vec::new()),
                scratch: Mutex::new(HashMap::new()),
                workspaces: Mutex::new(HashMap::new()),
                descs: Mutex::new(HashMap::new()),
                int_arrays: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Run `call` with this device's stream bound on the calling thread.
    /// The stream is synchronized by the caller when it needs the result.
    pub(crate) fn run(&self, call: impl FnOnce(*mut c_void) -> Result<()>) -> Result<()> {
        let _guard = self
            .inner
            .lock
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend device lock poisoned"))?;
        check(sys::aclrtSetCurrentContext(self.inner.context))?;
        call(self.inner.stream)
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        let stream = self.inner.stream;
        self.run(|_| check(sys::aclrtSynchronizeStream(stream)))
    }

    pub fn info(&self) -> Result<DeviceInfo> {
        unsafe {
            let raw = sys::aclrtGetSocName();
            let soc = if raw.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            };
            Ok(DeviceInfo {
                soc,
                device_id: self.inner.device_id,
            })
        }
    }

    pub fn zeros<T: DeviceRepr>(&self, len: usize) -> Result<Buffer<T>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow::anyhow!("allocation size overflow"))?;
        ensure!(bytes > 0, "zero-length device allocation");
        let mut raw = std::ptr::null_mut();
        self.run(|_| {
            check(sys::aclrtMalloc(
                &raw mut raw,
                bytes,
                sys::ACL_MEM_MALLOC_HUGE_FIRST,
            ))
        })?;
        Ok(Buffer {
            raw: raw.cast(),
            len,
            device: self.clone(),
            marker: PhantomData,
        })
    }

    pub fn from_slice<T: DeviceRepr>(&self, data: &[T]) -> Result<Buffer<T>> {
        let mut buffer = self.zeros(data.len())?;
        buffer.write(data)?;
        Ok(buffer)
    }

    /// Upload a flat byte slice interpreted as `T` elements.
    pub fn from_bytes<T: DeviceRepr>(&self, bytes: &[u8]) -> Result<Buffer<T>> {
        ensure!(
            bytes.len().is_multiple_of(std::mem::size_of::<T>()),
            "mis-sized device data"
        );
        let buffer = self.zeros::<T>(bytes.len() / std::mem::size_of::<T>())?;
        {
            let raw = buffer.raw;
            self.run(|_| {
                check({
                    sys::aclrtMemcpy(
                        raw.cast(),
                        bytes.len(),
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        sys::ACL_MEMCPY_HOST_TO_DEVICE,
                    )
                })
            })?;
        }
        Ok(buffer)
    }

    pub fn owns<T: DeviceRepr>(&self, buffer: &Buffer<T>) -> bool {
        Arc::ptr_eq(&self.inner, &buffer.device.inner)
    }

    pub(crate) fn cached_gemm(
        &self,
        key: u64,
        create: impl FnOnce() -> Result<(usize, u64)>,
    ) -> Result<(usize, u64)> {
        let mut cache = self
            .inner
            .gemm_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend executor cache poisoned"))?;
        if let Some(&entry) = cache.get(&key) {
            return Ok(entry);
        }
        let entry = create()?;
        cache.insert(key, entry);
        Ok(entry)
    }

    /// Stable device scratch for cached executors, released in `Drop`.
    /// The caller must already hold the device lock (`Device::run`).
    pub(crate) fn scratch_locked(&self, bytes: usize) -> Result<*mut std::ffi::c_void> {
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        let mut map = self
            .inner
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend scratch cache poisoned"))?;
        if let Some(&ptr) = map.get(&bytes) {
            return Ok(ptr as *mut std::ffi::c_void);
        }
        let mut raw = std::ptr::null_mut();
        check(sys::aclrtMalloc(
            &raw mut raw,
            bytes,
            sys::ACL_MEM_MALLOC_HUGE_FIRST,
        ))?;
        map.insert(bytes, raw as usize);
        Ok(raw)
    }

    /// Reuse a device workspace for an ACLNN op, keyed by byte size. The
    /// caller must hold the device lock; every op synchronizes before the
    /// workspace can be handed to the next one.
    pub(crate) fn workspace_locked(&self, bytes: u64) -> Result<*mut std::ffi::c_void> {
        if bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        let bytes = usize::try_from(bytes)?;
        let mut map = self
            .inner
            .workspaces
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend workspace cache poisoned"))?;
        if let Some(&ptr) = map.get(&bytes) {
            return Ok(ptr as *mut std::ffi::c_void);
        }
        let mut raw = std::ptr::null_mut();
        check(sys::aclrtMalloc(
            &raw mut raw,
            bytes,
            sys::ACL_MEM_MALLOC_HUGE_FIRST,
        ))?;
        map.insert(bytes, raw as usize);
        Ok(raw)
    }

    pub(crate) fn cached_descs(
        &self,
        key: u64,
        create: impl FnOnce() -> Result<Descs>,
    ) -> Result<Descs> {
        let mut cache = self
            .inner
            .descs
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend descriptor cache poisoned"))?;
        if let Some(d) = cache.get(&key) {
            return Ok(d.clone());
        }
        let d = create()?;
        cache.insert(key, d.clone());
        Ok(d)
    }

    pub(crate) fn int_array(&self, value: i64) -> Result<usize> {
        let mut cache = self
            .inner
            .int_arrays
            .lock()
            .map_err(|_| anyhow::anyhow!("Ascend int-array cache poisoned"))?;
        if let Some(&ptr) = cache.get(&value) {
            return Ok(ptr);
        }
        let ptr = { sys::aclCreateIntArray(&raw const value, 1) } as usize;
        ensure!(ptr != 0, "aclCreateIntArray returned null");
        cache.insert(value, ptr);
        Ok(ptr)
    }

    pub(crate) fn keep_tensors(&self, tensors: Vec<*mut sys::aclTensor>) {
        if let Ok(mut keep) = self.inner.keepalive.lock() {
            keep.extend(tensors);
        }
    }
}

/// Plain numeric types for which every bit pattern is valid and zero is valid.
/// # Safety
/// Implementors must have no padding, references, or invalid bit patterns.
pub unsafe trait DeviceRepr: Copy + Default + Send + Sync + 'static {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for half::bf16 {}

pub struct Buffer<T: DeviceRepr> {
    raw: *mut T,
    len: usize,
    device: Device,
    marker: PhantomData<T>,
}
// Native access uses the serialized device stream. Mutable outputs require &mut.
unsafe impl<T: DeviceRepr> Send for Buffer<T> {}
unsafe impl<T: DeviceRepr> Sync for Buffer<T> {}

impl<T: DeviceRepr> Buffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_ptr(&self) -> *const T {
        self.raw
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.raw
    }
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        ensure!(data.len() <= self.len, "host write exceeds device buffer");
        let raw = self.raw;
        let bytes = std::mem::size_of_val(data);
        self.device.run(|_| {
            check({
                sys::aclrtMemcpy(
                    raw.cast(),
                    bytes,
                    data.as_ptr().cast(),
                    bytes,
                    sys::ACL_MEMCPY_HOST_TO_DEVICE,
                )
            })
        })
    }
    pub fn read(&self, len: usize) -> Result<Vec<T>> {
        ensure!(len <= self.len, "host read exceeds device buffer");
        let mut data = vec![T::default(); len];
        let raw = self.raw;
        let bytes = std::mem::size_of_val(data.as_slice());
        self.device.run(|_| {
            check({
                sys::aclrtMemcpy(
                    data.as_mut_ptr().cast(),
                    bytes,
                    raw.cast(),
                    bytes,
                    sys::ACL_MEMCPY_DEVICE_TO_HOST,
                )
            })
        })?;
        Ok(data)
    }
}

impl<T: DeviceRepr> Drop for Buffer<T> {
    fn drop(&mut self) {
        let raw = self.raw.cast();
        let device = &self.device.inner;
        let _lock = device
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        {
            let _ = sys::aclrtSetCurrentContext(device.context);
            let status = sys::aclrtFree(raw);
            if status != sys::ACL_SUCCESS {
                eprintln!("Ascend buffer cleanup failed: {status}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_repr_sizes_are_stable() {
        assert_eq!(std::mem::size_of::<half::bf16>(), 2);
        assert_eq!(std::mem::size_of::<i32>(), 4);
    }
}
