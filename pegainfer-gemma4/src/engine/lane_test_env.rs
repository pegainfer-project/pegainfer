static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

pub(super) struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            for (key, value) in self.saved.drain(..) {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

const SERVING_KNOBS: [&str; 7] = [
    super::ASYNC_PREFILL_ENV,
    super::PREFIX_CACHE_ENV,
    super::MIX_CHUNK_TOKENS_ENV,
    super::MAX_CONTEXT_ENV,
    super::DECODE_SLOTS_ENV,
    super::ADMIT_COALESCE_ENV,
    super::KV_FP8_ENV,
];

pub(super) fn scoped_engine_env(overrides: &[(&str, &str)]) -> EnvGuard {
    let lock = ENV_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .expect("env lock");
    let saved = SERVING_KNOBS
        .iter()
        .map(|key| (*key, std::env::var_os(key)))
        .collect();
    unsafe {
        for key in SERVING_KNOBS {
            std::env::remove_var(key);
        }
        for (key, value) in overrides {
            std::env::set_var(key, value);
        }
    }
    EnvGuard { saved, _lock: lock }
}
