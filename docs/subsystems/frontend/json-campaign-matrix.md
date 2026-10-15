# Calibration campaigns and proxy evidence

> **TL;DR:** Cold/decode uses original bench and an async proxy. Endpoint/PID address binding, complete acquisition-source identity and CR timing are repaired and verified; partial-ID false success remains an explicitly accepted, unfixed risk.
>
> **Last touched:** 2026-10

## Preparation

- Read `docs/index.md`, `docs/conventions/migration-defense.md`, the campaign compiler, request-plan/schema/storage modules, and every replaced execution module.
- The previous collector inferred identity from result positions and used bench timing arrays. Those arrays cannot establish output token arrival, complete stream duration, or request attribution.
- The approved C2 scope is cold-prefill/decode collection. Cached-prefix's existing independent tools, tests and required storage APIs remain until C8 replaces them.
- Review correction: driver support for three collection roles does not replace C3 pairing or C7 held-out replay/reporting. Replay's report, error-statistics and profile-validation rules are restored. `paired.py` now retains the previous collector's cross-role assessment without its HTTP/storage loop; `pairing.py` remains. Their new-plan integration is still later work.
- User-approved correction: restore replay's profile/publication validation, prediction, error statistics, queueing-TTFT, coverage and report rules; extract paired's retained cross-role key/seal assessment from its removed collection loop. Keep C2-owned HTTP sending, readiness, checkpointing and old execution-artifact readers deleted. These retained domain rules await C3/C7 integration with new measurements and do not establish old-artifact compatibility. At the user's subsequent request, remove their dedicated test files while retaining both production modules and the previously requested test deletions.
- C2 implementation audit requested by the user: review current working-tree acquisition, parser/converter, recovery/store and profile binding against the approved contract; reproduce suspected failures with local controlled endpoints and disposable artifacts under `/tmp`. Record confirmed findings and limits before making production fixes. Retain the user's test-file deletions.

## Execution and evidence contract

The supported versions are campaign spec/compiled campaign 5, request plan 2, measurement 3. Recompile an experiment after changing its measurement contract; the current execution path does not read old artifacts or generate old-format rows for downstream tools awaiting migration.

Compilation checks the explicitly named bench executable's version and help digest, local tokenizer files, observer implementation/configuration and pinned Python dependencies. Runtime checks the frozen inputs again. The Linux `/usr/bin/setpriv` launcher is also pinned by path, version/help and binary digest; it only sets the parent-death signal before executing bench and does not change UID/GID or privileges. There is no PATH fallback or runtime seed/shape override. Install the versions in `scripts/sim-calibrate/requirements-evidence-proxy.txt` in the environment used for compilation and execution.

For every phase/repeat, the driver invokes original bench with exactly C prompts and concurrency C, rate infinity, internal warmups/readiness generation disabled. It waits for the bench process, upstream requests and evidence writer to drain before starting another batch. A fixed batch never replenishes completed requests. Readiness uses `/v1/models` without requiring `/health` or generating a completion.

The driver registers bench's generated index set before sending. Actual `x-request-id` headers bind requests to plan keys; arrival order, completion order and digest uniqueness are not identity substitutes. The proxy reserves an identity before any await that could dispatch upstream. Duplicate or ambiguous identities invalidate the attempt and are never forwarded a second time. Original bench's internal retry is not configurable: the recorded policy is at-most-once **upstream forwarding**, not a claim that client retries are disabled.

Formal timestamps use the proxy's monotonic clock:

- TTFT ends at the first event containing non-empty **output token IDs**, including IDs with empty text. Prompt IDs, role, usage and finish-only events do not qualify.
- First text latency is separate; a space/newline qualifies, and no text yields null. Entirely missing output IDs yields unsupported token timing; text never substitutes for token TTFT. Partial IDs have the explicitly accepted false-success limitation documented in finding 1.
- Multi-token events retain their actual read timestamp and token count. An interval uses its right event's token count; the first event's tokens are excluded from the interval denominator. Request TPOT is diagnostic; the common-window fit source belongs to C4.
- E2E ends at upstream EOF, including usage/stream tail. HTTP, finish, DONE, EOF, parser failures and downstream failures are retained independently. Truncation, count disagreement and explicit error events cannot be sealed successfully.

Raw upstream reads are recorded before parsing. UTF-8/SSE framing may cross reads; an event is timestamped by the read containing its closing boundary. Multiple events in one read can have the same timestamp. No synthetic per-token timestamps are inserted. Actual request bytes, normalized body and prompt-ID digests are retained; model routing is excluded only from the normalized workload body.

A trailing CR retains its own read timestamp while awaiting an optional LF. If LF follows, the boundary uses that LF's read; otherwise the saved CR time applies, including at EOF. Waiting for the next byte no longer assigns a later read time to a complete CR-only event.

The observer source map freezes 13 acquisition/compiler inputs: proxy, parser, pinned requirements, converter, driver, bench adapter, measurement schema, request plan, manifest, artifact store, readiness probe, campaign compiler and profile builder. Plans require the complete source set and valid digests. Runtime rechecks it before each attempt or sealed reuse, after collection and before sealing; measurements must bind the same observer digest as their execution entry. A mismatch requires recompilation into a new artifact root. Existing campaign/plan/measurement schema versions remain 5/2/3; the expanded identity deliberately invalidates earlier compiled bundles.

The proxy uses a bounded evidence queue, limits open files and flushes/fsyncs before each batch transition and final sealing. Queue overflow, storage failure or dropped evidence fails the attempt. The original bench's unconditional optional `/metrics` reads are recorded and answered locally with 404; they are not forwarded to target or used as formal metrics. This behavior is part of observer identity.

## Artifacts, failure and profile binding

`artifact_index.json` remains the immutable compiler index. Runtime state is in `execution_index.json`; raw files live under `raw/<variant>/<cell_id>/attempt-N/`. Measurements bind the raw index and workload evidence. An execution lock prevents concurrent drivers from replacing checkpoints. Failed/interrupted attempts retain their bytes; recovery allocates a new directory and reruns the complete phase sequence. Sealed results are skipped only after verifying their indexed evidence and measurement digests.

`campaign.py run` calls `plan_driver.run_campaign`, which opens `artifacts.ExecutionStore`. The store writes `execution_index.json` atomically when an attempt begins and updates it when the attempt finishes; it is not a bench/proxy output or a compilation artifact.

Timeout/cancellation kills and reaps the owned bench process group; parent-death signaling also terminates bench if the driver is killed directly, and the interrupted attempt is recovered in a new directory; stdout/stderr are written during execution and retained on failure. The driver owns the bench and proxy lifecycle. Already-running target/simulator servers remain caller-owned.

For profile-backed zero_cost/sim, supply `--server-pid VARIANT=PID`. C2 verifies the local process's profile argument, profile/calibration digests, model, scheduler and start time. Driver and server must share a network namespace. The configured endpoint must explicitly use IPv4 `127.0.0.1`; every matching listener at that address or `0.0.0.0` must belong to the PID. Same-port IPv6 listeners are conservatively rejected because IPv6/dual-stack attestation is outside this scope. Bindings record namespace and listener address/port/inode, and are rechecked before each attempt or sealed reuse, after collection and before sealing. `localhost` is rejected with a request to use the explicit IPv4 address; target endpoints retain their existing rules. The zero-cost profile must match the compiler-generated control. This binding requires Linux `/proc`; remote/cross-namespace profile attestation is not implemented. A pending fitted sim profile does not prevent compilation or target/zero-cost collection.

Cold targets with caching enabled require an explicit reset lifecycle; a false/unavailable reset fails before generation. Turning caching off avoids that requirement. C8 owns cached text preparation, bootstrap/hit validation and its lifecycle. Selecting cached workload in the new driver currently fails explicitly; its independent legacy tools remain separate and do not receive compatibility output from the new chain.

## JSON changes from C1 to C2

| Artifact | Change |
| --- | --- |
| Campaign spec / `compiled_campaign.json` | Schema 4 to 5. Freeze the output-ID/text timing contract and forwarding/retry policy; compiled client adds observer config/source/dependency identity, binary help digest and Linux process-launcher identity. Existing target/zero_cost/sim roles and named matrices are C1 features. |
| `plans/<cell_id>/request_plan.json` | Schema 1 to 2. Add observer and retry contracts, replace ambiguous generated-content timing with explicit proxy dispatch/output-ID/text observations. Phase/repeat/index keys and fixed-batch declarations remain. |
| `measurements.json` | Envelope/row schema 2 to 3. Add `measurement_origin`, `first_text_timestamp`, `first_text_latency_ms`, `stream_end_timestamp`, and provenance's `observer_sha256`; remove `stream_chunk_tpot_ms`. Require actual transport identity and proxy event source. TTFT/TPOT/E2E are computed from raw evidence; unsupported/missing values remain explicit. Event timestamp/count fields already existed in C1; C2 implements and tightens their semantics. |
| `artifact_index.json` | Remains immutable compiler index schema 3. It indexes compiled inputs and generated zero-cost assets, not runtime checkpoints. |
| `execution_index.json` | New schema 1 runtime index with per-role/cell status, binding, attempt history, paths and file references (`sha256`, `size`). |
| `raw_bundle_index.json` / `workload_evidence.json` | New raw-file index (schema 1) and actual request identity/content evidence (`requests`, `sha256`), respectively. Measurements bind both; they are stored separately. |
| Per-request raw data | Actual body in `request.json` and binary files; `reads.jsonl` records byte ranges/read times, `events.jsonl` records classified SSE events, `terminal.json` records HTTP/finish/DONE/EOF/errors and upstream attempts. Raw times are monotonic nanoseconds; measurement timestamps use seconds and durations use milliseconds. |
| `bench-result.json` / runtime profiles | Original bench JSON remains unchanged and diagnostic only. Rust profile schema 3 and calibration manifest schema 2 remain unchanged. No normalized/fitted/acceptance JSON was generated by C2. |

New execution rejects old campaign/plan/measurement versions instead of adapting them. Restored replay/report rules retain their domain contracts until C3-C7 migration; they are not connected to measurement v3 and do not provide a new-to-old wrapper.

`stream_chunk_tpot_ms` was a mean interval between content-bearing SSE events, without proving one event equals one output token. It is removed from the new measurement contract because a multi-token event needs token-count weighting. Request diagnostic `tpot_ms` uses `(last_output_event_time - first_output_event_time) / sum(right_event_output_token_counts)`; the first event's tokens are excluded. This diagnostic does not replace C4's common-window fit source. Restored reporting modules may still mention the old field pending their later migration; the new converter does not emit it.

The retained production smoke outputs are under `/tmp/c2-production-20261015-final-process/`: `validation-summary.json` summarizes validation, `artifacts/` contains compiled inputs, plans, profiles, execution index and raw bundles; `ab/` and `ab.json` contain direct/proxy comparisons; `*.server.log`, `*.server.json` and `outcome.json` retain server/execution context. The sim profile is a functional transport smoke input, not a fitted result. Test methods use `TemporaryDirectory`, so their temporary bundles are cleaned up; suite logs are retained separately. These `/tmp` paths are local temporary evidence, not a durable repository archive.

## Verified entry points

Run from the repository root with the pinned observer environment. These help commands were run during C2 implementation:

```bash
.venv/bin/python scripts/sim-calibrate/campaign.py compile --help
.venv/bin/python scripts/sim-calibrate/campaign.py run --help
```

A complete input example, including timing/retry contracts and named matrices, is in `scripts/sim-calibrate/tests/fixtures/campaign-compile.json`. Its binary, tokenizer, model identity, runtime and endpoints are test inputs; supply actual values for an experiment. `compile --spec` writes an external artifact root; `run --artifact-root` accepts repeated `--cell` and `--variant` selectors. An unqualified run containing deferred cached cells is rejected rather than partially executed silently.

## Migration defense handoff

| Previous defense / failure mode | C2 owner |
| --- | --- |
| Request count, phase/repeat and duplicate attribution | Immutable plan, proxy registration/at-most-once reservation, measurement key validation |
| Partial streams, metadata counted as output, usage/count disagreement | Incremental SSE parser, terminal evidence, proxy converter and schema |
| Sealed overwrite, artifact tampering, interrupted-directory reuse | Separate execution store, exclusive lock/allocation and digest verification |
| Subprocess timeout, lost partial logs, leaked children/connections | Driver, direct-to-file bench logging, process-group cleanup and proxy shutdown |
| Cached prefix/hit/reset safety | Existing independent cached tools/tests retained; C8 migrates their assertions and removes replaced implementations |
| Cross-role key agreement, successful roles and sealed entries | Restored `paired.summarize_paired_cell`; C3 still owns plan/content validation and integration |
| Profile/selection/publication lineage, errors, queueing and coverage | Restored replay validation/prediction/report rules; C7 still owns new-evidence integration and report migration |

## Execution Log

- Implemented audit repairs 2—4 in the six planned production files. Removed the old port-only ownership check and shared the pre/post-attempt checks; expanded the existing observer identity rather than adding another identity format. No tests were added or restored. All 28 retained tests passed in 14.932s; compilation, `cargo fmt --check` and diff checks passed. Suite log: `/tmp/c2-repair-tests-20261015.log`.
- Reused the audit diagnostics: a PID listening at `127.0.0.2:P` cannot attest the HTTP endpoint at `127.0.0.1:P`, and the latter received zero generation requests. Positive `127.0.0.1` and wildcard listeners pass. Changing converter, driver or bench adapter in disposable source copies is rejected before generation; unchanged copies seal. The CR-only reproduction now reports 100ns instead of 1,000,000,100ns. Evidence: `/tmp/c2-repair-diagnostics-20261015/summary.json`. An initial disposable source copy had an insufficient directory hierarchy, so its artifact root was treated as inside the copied repository; repeating with the normal checkout hierarchy passed. This did not require production changes.
- Recompiled and reran real vLLM Qwen3-4B plus release zero_cost/sim: 9 cell/role executions sealed, 54 rows (36 measured), 18 logical keys with identical cross-role content digests, matching 13-source identities, and all owned servers cleaned up. Both actual simulator listeners used `0.0.0.0`. Artifacts and summary: `/tmp/c2-repair-production-20261015/`; log: `/tmp/c2-repair-production-20261015.log`. The simulator profile remains a transport smoke input. This repair run did not repeat the earlier performance A/B.
- Current-tree C2 audit reproduced four defects with disposable local endpoints and the installed original bench; production code was left unchanged. Summary: `/tmp/c2-audit-20261015/summary.json`; additional wrong-server end-to-end proof: `/tmp/c2-audit-20261015/wrong-server-seal/summary.json`. No new repository test files or GPU workloads were introduced. Diagnostic listener processes were reaped and aiohttp runners closed.
- Replaced old array/ordinal conversion and independent sampling clients with the common bench/proxy/parser path. Removed seven execution modules and most of replay; kept cached CLI/test dependencies. Review identified that absence of callers did not establish replacement of replay's reporting responsibilities; see Preparation.
- At commit `97681e03`, all 37 Python tests passed. The subsequent user-directed test reduction removed 9 methods: the artifact-store test, two proxy tests, false-reset/cached-selection test, two SSE tests and three standalone measurement-schema timing tests. Unused imports and the reset fixture were removed too. All remaining 28 tests passed; the actual-bench retry integration remains. Log: `/tmp/c2-review-tests-20261015.log`. The first sandboxed run could not create loopback sockets; the same suite passed with local socket access and ambient proxies unset.
- Restored all unreplaced replay domain functions from the parent of `97681e03`; kept only its old sampling CLI/driver, execution-artifact reader/path helper and mutable sim-manifest builder removed. Extracted paired's key/success/seal assessment as `summarize_paired_cell`, using the already-adopted `sim` role name. No legacy execution reader, HTTP client or compatibility adapter was restored.
- Five focused tests passed during restoration (33 total; log `/tmp/c2-restoration-tests-20261015.log`). The user then requested removing `test_paired.py` and `test_replay.py`, since both production modules await replacement. Both dedicated test files are removed and their production modules remain. All remaining 28 tests passed; log `/tmp/c2-retained-tests-20261015.log`. Diff checks passed. No new GPU benchmark was needed for these domain-only changes.
- Real vLLM Qwen3-4B plus release profile-backed zero_cost/sim sealed 9 cell/variant executions and 54 requests (36 measured); all 18 logical keys matched prompt/token/normalized-body digests across roles. The functional simulator smoke profile exercises execution and attribution; it is not a fitted accuracy result.
- Six alternating direct/proxy pairs at C=4 completed 24/24 requests per arm. Client diagnostic median TTFT was 156.59→159.04 ms (+1.56%), TPOT 28.208→30.619 ms (+8.55%), and batch duration 1.03272→1.10366 s (+6.87%). These bench diagnostics do not establish formal token timing correctness or negligible observer overhead. No business overhead threshold was frozen; larger workloads still need perturbation qualification. Raw evidence and validation summary remain outside the repository at `/tmp/c2-production-20261015-final-process/`.

## Debrief

C2 writes raw evidence and measurements. Findings 2—4 are repaired and verified: PID attestation includes endpoint address/namespace/socket ownership, the complete acquisition-source identity is frozen and checked across sealing/reuse, and CR boundaries retain the correct read time. Finding 1 remains an accepted, unfixed risk. The practical lesson is to bind the actual endpoint and conversion code as well as raw data. Pairing, common-window extraction, baseline/fitting/profile/report migration and cached-prefix integration remain in their later commits. Old artifact compatibility is not part of the new path. Unreplaced replay/paired domain rules are restored; absence of callers alone does not prove replacement.

## C2 audit findings (2026-10-15)

The following descriptions preserve the original reproductions. Findings 2—4 are now repaired as documented above; finding 1 is deliberately unchanged.

1. **Accepted known risk — partial output-ID streams can seal incorrect metrics; no fix this round.** Partial means some generated SSE events lack IDs; one event carrying only its own incremental IDs is normal. `convert_evidence` filters events to those with IDs and treats any nonempty filtered sequence as valid. With `ignore_eos=false`, no usage, an initial text-only output and a later ID-bearing output, the actual-bench integration sealed `succeeded/token_timing_valid=true`: the controlled endpoint was configured to emit 2 tokens, but 1 was recorded; first text was 14.827 ms and TTFT was 76.000 ms. The missing initial output can also distort TPOT. This was fault injection, not an observed normal vLLM/PegaInfer response; the prior 54-row three-role smoke did not show this pattern and does not establish a general frequency. The user accepts the risk and requests documentation only: no new completeness policy, fallback or automatic retry is planned for this case.
2. **P1 — PID binding matches only a port, not the endpoint address.** `profile_binding` accepts a PID listening on `127.0.0.2:P` for an endpoint at `127.0.0.1:P`. A full original-bench run against a separate controlled HTTP server at the latter address sealed as zero_cost while the attested PID served no HTTP. Verify address family and reachable local address plus port/socket ownership (including wildcard/localhost semantics), not port alone.
3. **P2 — The formal conversion/execution implementation is not frozen.** Observer source identity covers proxy, parser and requirements only. Changing the converter's first-text classifier to ignore whitespace changed its code/metric semantics but left observer identity unchanged. Runtime does not validate a converter/driver/adapter source identity; the compiler provenance source list also omits those modules. Bind the complete acquisition/conversion implementation and validate it on execution so different role runs cannot silently use different metric logic under one campaign identity.
4. **P2 — A complete CR-only SSE event can receive the next read's timestamp.** When a read ends with the closing CR of `\r\r`, the parser waits for the next byte and classifies the already-complete event with the later read timestamp. The parser reproduction delayed an event from 100 ns to 1,000,000,100 ns. Consume legal CR line endings with pending-LF handling, or preserve the closing boundary's original read timestamp; keep byte bounds auditable.

Reproduction scripts: `/tmp/audit_c2_20261015.py` and `/tmp/audit_c2_wrong_server_20261015.py`; logs: `/tmp/c2-audit-20261015.log` and `/tmp/c2-audit-wrong-server-20261015.log`. These are diagnostic fixtures, not model accuracy/performance evidence. The earlier +8.55% client diagnostic TPOT overhead remains a separate known qualification limit; it is not counted as a newly discovered defect.
