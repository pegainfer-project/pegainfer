//! `PEGAINFER_SPEC_*` environment knobs.
//!
//! Both knobs are read once per process. A malformed value is rejected when the
//! engine starts, with the variable's name; unset, blank and an explicit `0`
//! all mean off. Silently falling back to off on a typo would turn a load lever
//! into a no-op that looks armed.

use anyhow::Result;
use anyhow::bail;

pub(crate) const MAX_BATCH: &str = "PEGAINFER_SPEC_MAX_BATCH";
pub(crate) const MAX_DRAFTS: &str = "PEGAINFER_SPEC_MAX_DRAFTS";
/// Fixed budget for the prefix-block drafter-capture cache. Unset uses
/// [`CAPTURE_CACHE_DEFAULT_MB`]; an explicit `0` disables the cache (and keeps
/// the prefix cache off while a drafter is loaded).
pub(crate) const CAPTURE_CACHE_MB: &str = "PEGAINFER_SPEC_CAPTURE_CACHE_MB";
pub(crate) const CAPTURE_CACHE_DEFAULT_MB: usize = 128;
const CAPTURE_CACHE_MAX_MB: usize = 4096;

/// Parse one knob: unset/blank = off (`0`), otherwise a non-negative integer.
fn parse(name: &str, value: Option<&str>) -> Result<usize> {
    let Some(raw) = value else {
        return Ok(0);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    match trimmed.parse::<usize>() {
        Ok(value) => Ok(value),
        Err(_) => bail!("{name}: expected a non-negative integer, got {raw:?}"),
    }
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name}: value is not UTF-8"),
    }
}

/// Every engine start runs [`validate`] first, so a failure here is a broken
/// invariant rather than user input.
pub(crate) fn read(name: &str) -> usize {
    env_value(name)
        .and_then(|value| parse(name, value.as_deref()))
        .expect("spec knobs are validated before the engine starts")
}

/// Bytes reserved for the prefix-block drafter-capture cache.
///
/// Unlike the load knobs, unset means the documented default rather than off.
/// A malformed value is rejected by [`validate`] before the engine starts.
pub(crate) fn capture_cache_bytes() -> usize {
    let mb = match env_value(CAPTURE_CACHE_MB) {
        Ok(Some(value)) => parse(CAPTURE_CACHE_MB, Some(&value))
            .expect("spec knobs are validated before the engine starts"),
        Ok(None) => CAPTURE_CACHE_DEFAULT_MB,
        Err(_) => 0,
    };
    mb.saturating_mul(1024 * 1024)
}

/// Reject malformed knob values before the engine starts.
pub(crate) fn validate() -> Result<()> {
    for name in [MAX_BATCH, MAX_DRAFTS] {
        parse(name, env_value(name)?.as_deref())?;
    }
    if let Some(value) = env_value(CAPTURE_CACHE_MB)? {
        let mb = parse(CAPTURE_CACHE_MB, Some(&value))?;
        if mb > CAPTURE_CACHE_MAX_MB {
            bail!("{CAPTURE_CACHE_MB}: expected at most {CAPTURE_CACHE_MAX_MB} MiB, got {mb}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_means_off_and_a_bad_value_names_the_knob() {
        for off in [None, Some(""), Some("  "), Some("0")] {
            assert_eq!(parse(MAX_BATCH, off).unwrap(), 0);
        }
        let error = parse(MAX_DRAFTS, Some("4x")).unwrap_err().to_string();
        assert!(error.contains(MAX_DRAFTS), "{error}");
    }

    #[test]
    fn capture_cache_knob_parses_and_documents_its_default() {
        assert_eq!(parse(CAPTURE_CACHE_MB, Some("128")).unwrap(), 128);
        assert_eq!(parse(CAPTURE_CACHE_MB, Some("0")).unwrap(), 0);
        let error = parse(CAPTURE_CACHE_MB, Some("4x")).unwrap_err().to_string();
        assert!(error.contains(CAPTURE_CACHE_MB), "{error}");
        assert_eq!(CAPTURE_CACHE_DEFAULT_MB, 128);
    }
}
