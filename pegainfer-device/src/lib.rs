#![cfg(feature = "rocm")]
//! Single-stream HIP device resources with explicit ownership.
//! Host copies complete before returning. Buffers retain the runtime and wait
//! for its stream before free, including cancellation and failed execution.
use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use anyhow::ensure;

unsafe extern "C" {
    fn pgi_last_error() -> *const c_char;
    fn pgi_create(ordinal: i32, out: *mut *mut c_void) -> i32;
    fn pgi_bind(device: *mut c_void) -> i32;
    fn pgi_sync(device: *mut c_void) -> i32;
    fn pgi_destroy(device: *mut c_void);
    fn pgi_alloc(device: *mut c_void, bytes: usize, out: *mut *mut c_void) -> i32;
    fn pgi_free(device: *mut c_void, ptr: *mut c_void) -> i32;
    fn pgi_copy(
        device: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        to_device: i32,
    ) -> i32;
    fn pgi_info(
        device: *mut c_void,
        name: *mut c_char,
        name_len: usize,
        arch: *mut c_char,
        arch_len: usize,
        free: *mut usize,
        total: *mut usize,
    ) -> i32;
}

pub fn check(status: i32) -> Result<()> {
    if status != 0 {
        let message = unsafe { CStr::from_ptr(pgi_last_error()) }.to_string_lossy();
        anyhow::bail!("HIP operation failed ({status}): {message}");
    }
    Ok(())
}

struct Inner {
    raw: *mut c_void,
    lock: Mutex<()>,
}
// All access, including rebinding on a new thread, is serialized by lock.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { pgi_destroy(self.raw) };
    }
}
#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}
#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub arch: String,
    pub free_bytes: usize,
    pub total_bytes: usize,
}
impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { pgi_create(i32::try_from(ordinal)?, &raw mut raw) })?;
        let device = Self {
            inner: Arc::new(Inner {
                raw,
                lock: Mutex::new(()),
            }),
        };
        let info = device.info()?;
        let arch = info.arch.split(':').next().unwrap_or("");
        ensure!(
            matches!(
                arch,
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1200" | "gfx1201"
            ),
            "unsupported ROCm architecture {}",
            info.arch
        );
        Ok(device)
    }
    /// # Safety
    /// Native code must use this device's stream, retain no host pointers,
    /// and only access live buffers belonging to this device. Output buffers
    /// must be exclusively borrowed for the duration of this submission.
    pub unsafe fn submit(&self, call: impl FnOnce(*mut c_void) -> i32) -> Result<()> {
        let _lock = self
            .inner
            .lock
            .lock()
            .map_err(|_| anyhow::anyhow!("HIP device lock poisoned"))?;
        check(unsafe { pgi_bind(self.inner.raw) })?;
        check(call(self.inner.raw))
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.submit(|d| pgi_sync(d)) }
    }
    pub fn info(&self) -> Result<DeviceInfo> {
        let (mut name, mut arch) = ([0 as c_char; 256], [0 as c_char; 256]);
        let (mut free, mut total) = (0, 0);
        unsafe {
            self.submit(|d| {
                pgi_info(
                    d,
                    name.as_mut_ptr(),
                    name.len(),
                    arch.as_mut_ptr(),
                    arch.len(),
                    &raw mut free,
                    &raw mut total,
                )
            })?;
        }
        Ok(DeviceInfo {
            name: unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            arch: unsafe { CStr::from_ptr(arch.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            free_bytes: free,
            total_bytes: total,
        })
    }
    pub fn zeros<T: DeviceRepr>(&self, len: usize) -> Result<Buffer<T>> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow::anyhow!("allocation size overflow"))?;
        ensure!(bytes > 0, "zero-length device allocation");
        let mut raw = std::ptr::null_mut();
        unsafe {
            self.submit(|d| pgi_alloc(d, bytes, &raw mut raw))?;
        }
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
    pub fn from_bytes<T: DeviceRepr>(&self, bytes: &[u8]) -> Result<Buffer<T>> {
        ensure!(
            bytes.len().is_multiple_of(std::mem::size_of::<T>()),
            "mis-sized device data"
        );
        let buffer = self.zeros::<T>(bytes.len() / std::mem::size_of::<T>())?;
        unsafe {
            self.submit(|d| pgi_copy(d, buffer.raw.cast(), bytes.as_ptr().cast(), bytes.len(), 1))?;
        }
        Ok(buffer)
    }
    pub fn owns<T: DeviceRepr>(&self, buffer: &Buffer<T>) -> bool {
        Arc::ptr_eq(&self.inner, &buffer.device.inner)
    }
}

/// Plain numeric types for which every bit pattern is valid and zero is valid.
/// # Safety
/// Implementors must have no padding, references, or invalid bit patterns.
pub unsafe trait DeviceRepr: Copy + Default + Send + Sync + 'static {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for f32 {}
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
        unsafe {
            self.device.submit(|d| {
                pgi_copy(
                    d,
                    self.raw.cast(),
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data),
                    1,
                )
            })
        }
    }
    pub fn read(&self, len: usize) -> Result<Vec<T>> {
        ensure!(len <= self.len, "host read exceeds device buffer");
        let mut data = vec![T::default(); len];
        unsafe {
            self.device.submit(|d| {
                pgi_copy(
                    d,
                    data.as_mut_ptr().cast(),
                    self.raw.cast(),
                    std::mem::size_of_val(data.as_slice()),
                    0,
                )
            })?;
        }
        Ok(data)
    }
}
impl<T: DeviceRepr> Drop for Buffer<T> {
    fn drop(&mut self) {
        // Recover the lock even on an unwind so allocations are never leaked.
        let _lock = self
            .device
            .inner
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = unsafe { pgi_free(self.device.inner.raw, self.raw.cast()) };
        if status != 0 {
            eprintln!("HIP buffer cleanup failed: {status}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a ROCm GPU"]
    fn buffers_survive_device_owner_and_thread_handoff() -> Result<()> {
        let device = Device::new(0)?;
        let buffer = device.from_slice(&[3u32, 5, 8, 13])?;
        drop(device);
        let result = std::thread::spawn(move || buffer.read(4)).join().unwrap()?;
        assert_eq!(result, [3, 5, 8, 13]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a ROCm GPU"]
    fn checked_host_copies_and_zero_initialization() -> Result<()> {
        let device = Device::new(0)?;
        let mut buffer = device.zeros::<f32>(17)?;
        assert_eq!(buffer.read(17)?, vec![0.; 17]);
        assert!(buffer.write(&[1.; 18]).is_err());
        assert!(buffer.read(18).is_err());
        buffer.write(&[1.25, -3.5])?;
        assert_eq!(buffer.read(2)?, [1.25, -3.5]);
        assert!(device.zeros::<u32>(usize::MAX).is_err());
        Ok(())
    }
}
