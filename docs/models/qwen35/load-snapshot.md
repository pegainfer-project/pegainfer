# Qwen3.5 Scheduler Metrics

> **TL;DR:** Single-GPU and TP Qwen3.5 expose one logical scheduler's post-step `SchedulerMetrics` through the shared driver; in-flight prefill counts as running work.
>
> **Last touched:** 2026-10

## Publication boundary

```text
Qwen35Scheduler::metrics
  -> shared driver metrics cell
  -> SchedulerHandle::metrics
  -> SteppedEngineBridge
  -> SchedulerStats
  -> /metrics
```

The driver drains submissions before calling `Scheduler::step`. The scheduler
settles ready asynchronous prefill and prunes cancelled pending, active and
prefilling requests before admission and planning. A ready prefill completion
ends the step after cancellation cleanup, so its updates are committed before
the next model execution. Admission and planning resume on the next step.
The driver publishes a post-step snapshot before committing that step's
request updates. The asynchronous
bridge reads the latest metrics cell when forwarding a batch; the driver may
have advanced again by then, so metrics are not an immutable snapshot tied
to that batch.

The metric names and labels are shared with the other model lines; see
[Prometheus metrics](../../subsystems/frontend/prometheus-metrics.md) for the
wire mapping.

## Accounting

| Metric field | Live scheduler state |
| --- | --- |
| `num_running_reqs` | Active decode requests, requests still prefilling and requests owned by the in-flight prefill |
| `num_waiting_reqs` | Pending requests, including deferred work and newly drained submissions |
| `kv_used_blocks` | Backend request-page capacity minus available request pages |
| `kv_total_blocks` | Backend request-page capacity, excluding the CUDA Graph padding page |

TP counts logical requests and pages once, rather than summing replicas.
Including in-flight prefill also prevents the driver's shutdown check from
mistaking an unfinished asynchronous chunk for an idle scheduler.

A change in metrics can commit an empty step. The bridge forwards its stats,
so cancelling the last request can reset HTTP gauges even though cancellation
retires the ledger account without a terminal update. Unchanged idle
iterations send no step.

## Waiting and failure

The shared driver polls when idle and can occupy a CPU thread even without
requests. With overlap enabled, Qwen3.5 polls the prefill event while decode
remains active. If no decoder remains, it waits for that event inside the
step and then finishes the chunk.

On an engine-fatal error, the scheduler drains asynchronous work before
clearing its request owners. The shared driver fails open ledger accounts
and exits. The bridge observes the exit and reports the dead engine to the
frontend, which shuts down the HTTP service. The final metrics retain total
capacity and display zero running, waiting and used KV for the dead engine.
That zero is not evidence that TP pages were returned: a poisoned executor
is not retried through healthy `DropRequest` cleanup, and its remaining
resources are released during backend teardown.
