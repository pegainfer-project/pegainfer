# Grafana Dashboard

> **TL;DR:** `pegainfer.json` is a Grafana 10.4-validated dashboard for the live PegaInfer `/metrics` surface: HTTP traffic, request outcomes, scheduler/KV state, token throughput, and request latency.
>
> **Last touched:** 2026-07

Import `pegainfer.json` in Grafana and select the Prometheus data source whose scrape target is the PegaInfer server address (port 8000 by default, metrics path `/metrics`).

The dashboard intentionally omits prefix-cache query/hit, speculative-decode, residency, FLOP/byte, and CUDA Graph panels because PegaInfer currently sends zero/default scheduler values for those families. Request-derived cached/computed prompt-token counters remain available at `/metrics`, but are not presented as a cache hit-rate panel.

## Execution Log

### Dashboard construction

- Started from vLLM's `examples/observability/prometheus_grafana/grafana.json` and retained the request, scheduler, KV, throughput, and latency surfaces PegaInfer feeds.
- Added the frontend-wide `http_requests_total` panel requested by issue #606.
- Corrected phase-time panels to graph `sum(rate(..._sum)) / sum(rate(..._count))`; the upstream example graphs the cumulative sum rate, which is not a per-request duration.
- Added an import-time Prometheus data-source input and an all-model selector. The final dashboard has 10 panels and 21 PromQL targets.

### CPU validation

- `cargo test --release -p pegainfer-sim --test frontend_e2e`: request serving plus per-engine scheduler gauges from the stepped path (idle settle after a request; running rises on a slow request).
- Sent two batches of 8 concurrent completion requests through a standalone `pegainfer-sim` server. The first scrape recorded 8 completed requests, 64 prompt tokens, 128 generated tokens, HTTP handler/status counters, and non-empty TTFT, ITL, E2E, queue, prefill, and decode histograms.
- Prometheus 2.51.2 scraped the simulator at one-second intervals with target health `up`. All 21 dashboard PromQL targets returned `status=success` and at least one series after substituting a 30-second rate interval and the simulator model name.

### Grafana validation

- Grafana 10.4.0 accepted `pegainfer.json` through `POST /api/dashboards/import` with `imported=true` and UID `pegainfer-serving`.
- Reading the imported dashboard back confirmed all 10 panels, all target counts, the model selector, and every data-source reference rebound to the selected Prometheus UID.
- Docker registry access returned EOF on this host, so the same official Grafana and Prometheus release versions were run from their standalone release archives instead.

## Debrief

- **Outcome**: Issue #606's dashboard artifact and local setup note are complete and validated end to end on the CPU-only path named in the issue.
- **Pitfalls encountered**:
  - HTTP metrics have no `model_name`, so that panel is explicitly frontend-wide.
  - Deliberately default/zero scheduler metrics must not become apparently functional panels.
  - A valid JSON parse is weaker than a Grafana import; the import API caught the full dashboard/data-source contract.
- **Lessons learned**:
  - Validate every committed PromQL expression against a real Prometheus scrape, not only against metric-family names.
  - Keep scheduler gauges split by `engine`; aggregating them would hide partition imbalance on GLM5.2 EP8/DP8.
- **Follow-ups**:
  - Exercise the same dashboard during the next Qwen3-4B GPU benchmark. This host cannot run that hardware gate: `nvidia-smi` cannot reach a driver and no Qwen3-4B weights are present.
