//! `PEGAINFER_SPEC_*` environment knobs.
//!
//! Both knobs are read once per process. A malformed value is rejected at
//! launch with the variable's name; unset, blank and an explicit `0` all mean
//! off. Silently falling back to off on a typo would turn a load lever into a
//! no-op that looks armed.

use anyhow::Result;
use anyhow::bail;

pub(crate) const MAX_BATCH: &str = "PEGAINFER_SPEC_MAX_BATCH";
pub(crate) const MAX_DRAFTS: &str = "PEGAINFER_SPEC_MAX_DRAFTS";

/// Parse one knob: unset/blank = off (`0`), otherwise a non-negative integer.
pub(crate) fn parse(name: &str, value: Option<&str>) -> Result<usize> {
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

/// Read one knob once per process. Launch validation rejects malformed values,
/// so the fallback below is only reachable for internal callers that bypass it;
/// it still logs instead of failing a scheduler step.
/// Read one knob once per process. Every engine start validates the knobs
/// first, so a parse failure here is a broken invariant, not user input; it
/// fails loudly instead of silently turning the lever off.
pub(crate) fn read(name: &str) -> usize {
    let value = std::env::var(name).ok();
    parse(name, value.as_deref()).expect("spec knobs are validated before the engine starts")
}

/// Reject malformed knob values before the engine starts.
pub(crate) fn validate(lookup: impl Fn(&str) -> Option<String>) -> Result<()> {
    for name in [MAX_BATCH, MAX_DRAFTS] {
        let value = lookup(name);
        parse(name, value.as_deref())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_blank_and_zero_mean_off() {
        assert_eq!(parse(MAX_BATCH, None).unwrap(), 0);
        assert_eq!(parse(MAX_BATCH, Some("")).unwrap(), 0);
        assert_eq!(parse(MAX_BATCH, Some("  ")).unwrap(), 0);
        assert_eq!(parse(MAX_BATCH, Some("0")).unwrap(), 0);
        assert_eq!(parse(MAX_DRAFTS, Some(" 0 ")).unwrap(), 0);
    }

    #[test]
    fn positive_values_parse() {
        assert_eq!(parse(MAX_BATCH, Some("32")).unwrap(), 32);
        assert_eq!(parse(MAX_DRAFTS, Some(" 4 ")).unwrap(), 4);
        // Core's unsigned parse accepts a leading `+`, so this is a valid count.
        assert_eq!(parse(MAX_BATCH, Some("+2")).unwrap(), 2);
    }

    #[test]
    fn malformed_values_name_the_variable() {
        for raw in ["-1", "4x", "1.5", "1_000"] {
            let error = parse(MAX_DRAFTS, Some(raw))
                .expect_err("malformed value must be rejected")
                .to_string();
            assert!(
                error.contains(MAX_DRAFTS),
                "error {error:?} does not name {MAX_DRAFTS}"
            );
        }
    }

    #[test]
    fn validate_accepts_unset_and_explicit_zero_only() {
        validate(|_| None).unwrap();
        validate(|name| (name == MAX_BATCH).then(|| "0".to_string())).unwrap();
        let error = validate(|name| (name == MAX_BATCH).then(|| "banana".to_string()))
            .expect_err("a typo must refuse to start")
            .to_string();
        assert!(
            error.contains(MAX_BATCH),
            "error {error:?} does not name the knob"
        );
    }
}
