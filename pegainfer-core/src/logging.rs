//! Unified logging configuration.

use std::str::FromStr;
use std::sync::Once;

use chrono::DateTime;
use chrono::Local;
use colored::Colorize;
use logforth::Diagnostic;
use logforth::Error;
use logforth::Layout;
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::kv::Key;
use logforth::kv::Value;
use logforth::kv::Visitor;
use logforth::record::Level;
use logforth::record::Record;

static INIT: Once = Once::new();

#[derive(Debug, Clone)]
struct LoggingConfig {
    level: String,
    colored: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            colored: true,
        }
    }
}

/// `MM-dd HH:MM:SS.µs LEVEL crate file.rs:line message k=v ...`
///
/// The stock `TextLayout` prefix (full RFC3339 timestamp + full module path +
/// file:line) is ~90 chars of mostly redundant text. Keep the crate name so
/// `RUST_LOG` filters are discoverable, and file:line for jump-to-source.
#[derive(Debug)]
struct CompactLayout {
    colored: bool,
}

impl CompactLayout {
    fn level_text(&self, level: Level) -> String {
        let text = format!("{level:>5}");
        if !self.colored {
            return text;
        }
        // Pad before colorizing so ANSI codes don't break alignment.
        match level {
            l if l >= Level::Error => text.red(),
            l if l >= Level::Warn => text.yellow(),
            l if l >= Level::Info => text.green(),
            _ => return text,
        }
        .to_string()
    }
}

struct KvWriter {
    text: String,
}

impl Visitor for KvWriter {
    fn visit(&mut self, key: Key, value: Value) -> Result<(), Error> {
        use std::fmt::Write;

        // Writing to a String cannot fail.
        write!(&mut self.text, " {key}={value}").unwrap();
        Ok(())
    }
}

impl Layout for CompactLayout {
    fn format(&self, record: &Record, diags: &[Box<dyn Diagnostic>]) -> Result<Vec<u8>, Error> {
        let time = DateTime::<Local>::from(record.time());
        let level = self.level_text(record.level());
        let crate_name = record.target().split("::").next().unwrap_or_default();
        let file = record.filename();
        let line = record.line().unwrap_or_default();
        let message = record.payload();

        let mut visitor = KvWriter {
            text: format!(
                "{} {level} {crate_name} {file}:{line} {message}",
                time.format("%m-%d %H:%M:%S%.6f"),
            ),
        };
        record.key_values().visit(&mut visitor)?;
        for d in diags {
            d.visit(&mut visitor)?;
        }

        Ok(visitor.text.into_bytes())
    }
}

const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 5] = [
    ("h2", "warn"),
    ("hyper", "warn"),
    ("hyper_util", "warn"),
    ("axum", "warn"),
    ("tower", "warn"),
];

enum BareGlobalLevel {
    Off,
    Level(Level),
}

fn apply_default_module_levels(mut filter: String) -> String {
    for (module, level) in DEFAULT_NOISY_MODULE_LEVELS {
        let module_pattern = format!("{module}=");
        if !filter.contains(&module_pattern) {
            if !filter.is_empty() {
                filter.push(',');
            }
            filter.push_str(module);
            filter.push('=');
            filter.push_str(level);
        }
    }
    filter
}

fn bare_global_level(filter: &str) -> Option<BareGlobalLevel> {
    let mut global_level = None;

    for directive in filter
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if directive.contains('=') {
            continue;
        }

        if directive.eq_ignore_ascii_case("off") {
            global_level = Some(BareGlobalLevel::Off);
        } else if let Ok(level) = Level::from_str(directive) {
            global_level = Some(BareGlobalLevel::Level(level));
        }
    }

    global_level
}

fn should_apply_default_module_levels(filter: &str) -> bool {
    matches!(bare_global_level(filter), Some(BareGlobalLevel::Level(level)) if level < Level::Warn)
}

fn resolved_filter_string(level: String, rust_log: Option<String>) -> String {
    let filter = rust_log.unwrap_or(level);
    if should_apply_default_module_levels(&filter) {
        apply_default_module_levels(filter)
    } else {
        filter
    }
}

fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig { level, colored } = config;
        let filter_str = resolved_filter_string(level, std::env::var("RUST_LOG").ok());
        let filter = logforth::filter::env_filter::EnvFilterBuilder::from_spec(filter_str).build();

        logforth::starter_log::builder()
            .dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(
                        logforth::append::Stderr::default().with_layout(CompactLayout { colored }),
                    )
            })
            .apply();
    });
}

pub fn init_default() {
    init(LoggingConfig::default());
}

#[cfg(test)]
mod tests {
    use super::resolved_filter_string;

    #[test]
    fn applies_default_noisy_module_levels_when_rust_log_is_debug() {
        let filter = resolved_filter_string("info".to_string(), Some("debug".to_string()));

        assert!(filter.starts_with("debug"));
        assert!(filter.contains("h2=warn"));
        assert!(filter.contains("hyper=warn"));
        assert!(filter.contains("hyper_util=warn"));
        assert!(filter.contains("axum=warn"));
        assert!(filter.contains("tower=warn"));
    }

    #[test]
    fn preserves_explicit_module_overrides_from_rust_log() {
        let filter = resolved_filter_string("info".to_string(), Some("debug,h2=trace".to_string()));

        assert!(filter.starts_with("debug,h2=trace"));
        assert!(filter.contains("hyper=warn"));
        assert!(filter.contains("hyper_util=warn"));
        assert!(filter.contains("axum=warn"));
        assert!(filter.contains("tower=warn"));
        assert!(!filter.contains("h2=warn"));
    }

    #[test]
    fn does_not_relax_stricter_global_rust_log_levels() {
        let error_filter = resolved_filter_string("info".to_string(), Some("error".to_string()));
        let off_filter = resolved_filter_string("info".to_string(), Some("off".to_string()));

        assert_eq!(error_filter, "error");
        assert_eq!(off_filter, "off");
    }

    #[test]
    fn does_not_add_default_module_levels_without_a_global_level() {
        let filter = resolved_filter_string("info".to_string(), Some("h2=trace".to_string()));

        assert_eq!(filter, "h2=trace");
        assert!(!filter.contains("hyper=warn"));
        assert!(!filter.contains("hyper_util=warn"));
        assert!(!filter.contains("axum=warn"));
        assert!(!filter.contains("tower=warn"));
    }
}
