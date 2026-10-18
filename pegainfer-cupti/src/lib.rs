use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;

type CuContext = *mut c_void;
type CuptiCallback = Option<unsafe extern "C" fn(*mut c_void) -> c_int>;

unsafe extern "C" {
    fn pegainfer_cupti_profile_range(
        context: CuContext,
        device_index: usize,
        range_name: *const c_char,
        metric_names: *const *const c_char,
        metric_count: usize,
        prepare_fn: CuptiCallback,
        launch_fn: CuptiCallback,
        userdata: *mut c_void,
        metric_values: *mut f64,
        metric_value_count: usize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
}

#[derive(Debug)]
pub struct CuptiProfilerError {
    message: String,
}

impl CuptiProfilerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CuptiProfilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for CuptiProfilerError {}

struct CallbackState<'a> {
    prepare: Option<&'a mut dyn FnMut() -> Result<(), String>>,
    launch: &'a mut dyn FnMut() -> Result<(), String>,
    error: Option<String>,
}

unsafe extern "C" fn prepare_trampoline(userdata: *mut c_void) -> c_int {
    let state = unsafe { &mut *(userdata.cast::<CallbackState<'_>>()) };
    let Some(prepare) = state.prepare.as_mut() else {
        return 0;
    };
    match prepare() {
        Ok(()) => 0,
        Err(err) => {
            state.error = Some(err);
            1
        }
    }
}

unsafe extern "C" fn launch_trampoline(userdata: *mut c_void) -> c_int {
    let state = unsafe { &mut *(userdata.cast::<CallbackState<'_>>()) };
    match (state.launch)() {
        Ok(()) => 0,
        Err(err) => {
            state.error = Some(err);
            1
        }
    }
}

/// Profile one user range with CUPTI Range Profiler.
///
/// # Safety
///
/// `context` must be a live CUDA `CUcontext` for `device_index`, and the
/// callbacks must only launch work on that context while this function runs.
pub unsafe fn profile_range_with_prepare<'a>(
    context: CuContext,
    device_index: usize,
    range_name: &str,
    metric_names: &[&str],
    prepare: Option<&'a mut dyn FnMut() -> Result<(), String>>,
    launch: &'a mut dyn FnMut() -> Result<(), String>,
) -> Result<Vec<f64>, CuptiProfilerError> {
    let range_name =
        CString::new(range_name).map_err(|err| CuptiProfilerError::new(err.to_string()))?;
    let metric_names: Vec<CString> = metric_names
        .iter()
        .map(|name| CString::new(*name).map_err(|err| CuptiProfilerError::new(err.to_string())))
        .collect::<Result<_, _>>()?;
    let metric_ptrs: Vec<*const c_char> = metric_names.iter().map(|name| name.as_ptr()).collect();
    let mut values = vec![0.0; metric_ptrs.len()];
    let mut error = vec![0; 4096];
    let mut state = CallbackState {
        prepare,
        launch,
        error: None,
    };
    let prepare_fn = if state.prepare.is_some() {
        Some(prepare_trampoline as unsafe extern "C" fn(*mut c_void) -> c_int)
    } else {
        None
    };

    let status = unsafe {
        pegainfer_cupti_profile_range(
            context,
            device_index,
            range_name.as_ptr(),
            metric_ptrs.as_ptr(),
            metric_ptrs.len(),
            prepare_fn,
            Some(launch_trampoline),
            (&raw mut state).cast::<c_void>(),
            values.as_mut_ptr(),
            values.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };

    if status == 0 {
        return Ok(values);
    }

    let ffi_error = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let callback_error = state.error.take();
    match callback_error {
        Some(callback_error) if !ffi_error.is_empty() => Err(CuptiProfilerError::new(format!(
            "{ffi_error}; callback error: {callback_error}"
        ))),
        Some(callback_error) => Err(CuptiProfilerError::new(callback_error)),
        None => Err(CuptiProfilerError::new(ffi_error)),
    }
}
