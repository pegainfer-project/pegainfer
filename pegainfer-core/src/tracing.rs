//! Distributed request tracing via fastrace → OTLP.
//!
//! fastrace collects host-side span trees (request lifecycle, scheduler steps)
//! and this module ships them to any OTLP trace backend — Jaeger all-in-one for
//! local inspection, Tempo/VictoriaTraces in a cluster. The Rust side speaks
//! only OTLP; swapping the backend is an endpoint change, not a code change.
//!
//! This is host-side orchestration time (queue wait, prefill/decode phase
//! boundaries, sampling), NOT GPU kernel time — that stays the domain of nsys
//! and ncu. fastrace measures exactly the host overhead CUDA Graph is built to
//! hide, attributed per request.
//!
//! Tracing is off unless [`OTLP_ENDPOINT_ENV`] is set. When unset, no reporter
//! is installed and fastrace's `#[trace]` / `Span` calls compile to near-nothing
//! (a root span is never started, so instrumentation is inert). Turn it on with:
//!
//! ```text
//! PEGAINFER_TRACE_OTLP_ENDPOINT=http://127.0.0.1:4317 cargo run --release -- ...
//! ```

use std::borrow::Cow;
use std::sync::Once;
use std::time::Duration;

use fastrace_opentelemetry::OpenTelemetryReporter;
use opentelemetry::InstrumentationScope;
use opentelemetry::KeyValue;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;

/// Set to an OTLP gRPC endpoint (e.g. `http://127.0.0.1:4317`) to enable tracing.
const OTLP_ENDPOINT_ENV: &str = "PEGAINFER_TRACE_OTLP_ENDPOINT";

/// `service.name` reported to the trace backend; how this process shows up in
/// the Jaeger service dropdown.
const SERVICE_NAME: &str = "pegainfer";

static INIT: Once = Once::new();

/// Install the fastrace OTLP reporter if [`OTLP_ENDPOINT_ENV`] is set.
///
/// No-op (and zero runtime cost beyond the env lookup) when the variable is
/// absent. Idempotent: safe to call from every binary's startup. Must run
/// inside a Tokio runtime — the OTLP tonic exporter drives its gRPC client on
/// the current runtime handle.
///
/// Pair with [`flush`] at process exit so the final batch of spans is exported
/// before the runtime tears down.
pub fn init() {
    INIT.call_once(|| {
        let Ok(endpoint) = std::env::var(OTLP_ENDPOINT_ENV) else {
            return;
        };
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return;
        }

        let exporter = match SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(Duration::from_secs(5))
            .build()
        {
            Ok(exporter) => exporter,
            Err(error) => {
                // A misconfigured endpoint must not take the engine down —
                // tracing is diagnostics, not a serving dependency.
                log::warn!("tracing disabled: failed to build OTLP exporter: {error}");
                return;
            }
        };

        // The tonic exporter's gRPC client is async, so fastrace's reporter
        // thread needs a runtime handle to block the export on. init() is
        // called from inside `#[tokio::main]`; if somehow called off a runtime,
        // disable tracing rather than panic in `Handle::current`.
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            log::warn!("tracing disabled: init() called outside a Tokio runtime");
            return;
        };
        let reporter = OpenTelemetryReporter::new(
            exporter,
            Cow::Owned(
                Resource::builder()
                    .with_attribute(KeyValue::new("service.name", SERVICE_NAME))
                    .build(),
            ),
            InstrumentationScope::builder(SERVICE_NAME)
                .with_version(env!("CARGO_PKG_VERSION"))
                .build(),
        )
        .with_block_on(move |future| runtime.block_on(future));

        fastrace::set_reporter(reporter, fastrace::collector::Config::default());
        // Only now, with a reporter actually installed, do request paths start
        // building spans. Until this flips, `Span::root` is skipped entirely.
        pegainfer_frontend::tracing_state::set_enabled(true);
        log::info!("request tracing enabled: exporting OTLP spans to {endpoint}");
    });
}

/// Flush any buffered spans to the backend. Call once at process exit; a no-op
/// when tracing was never enabled.
pub fn flush() {
    fastrace::flush();
}
