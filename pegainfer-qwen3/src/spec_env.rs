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

/// Reject malformed knob values before the engine starts.
pub(crate) fn validate() -> Result<()> {
    for name in [MAX_BATCH, MAX_DRAFTS] {
        parse(name, env_value(name)?.as_deref())?;
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
}
