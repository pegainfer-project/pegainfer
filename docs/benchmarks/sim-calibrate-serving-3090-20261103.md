# Two-model serving calibration on RTX 3090 — 2026-11-03

**TL;DR:** The current source revision `0b78bb151ddf1d8e693098bc0e73885982615a52` completed 38 cells and 78 role executions across Qwen3-4B and Qwen3.5-4B. The four native held-out reports succeeded and were each recomputed once from raw evidence after stopping their target. Accuracy remains `not_assessed`; no error threshold was set.

## Scope and environment

The run used two RTX 3090 GPUs in parallel and completed in 449.10 s. Each model ran a joint campaign followed by a cached-only campaign. Both targets used BF16, TP=1, vLLM 0.30.0, the Triton attention backend, eager execution, 16,384 maximum model/batched tokens, four maximum sequences, prefix caching, and a GPU memory utilization setting of 0.60.

Recorded versions: vllm-bench `vllm-bench 0.1.0`, PyTorch `2.13.0`, CUDA `13.0`; Python `3.10.12`, driver `595.58.03`. Triton `3.7.1`, Transformers `5.17.0`, Tokenizers `0.23.2`.

Qwen3 used whole-prompt prefill and prefix alignment 16. Qwen3.5 used chunked prefill with a 16,384-token chunk ceiling, Mamba cache mode `align`, prefix retention interval 528, and prefix alignment 528. The simulator and reference services ran on CPU.

Every cell used two warmup repeats and five measured repeats. Requests arrived as fixed batches with no replenishment; each batch drained before the next one. Target cache state was reset before every repeat. Cached cells added one prefix-only bootstrap request before each warmup or measured request. Sampling used temperature 0, top-p 1, and ignore-EOS.

## Complete matrix

| Model | Campaign | Matrix | Split | Coordinates | Output tokens |
| --- | --- | --- | --- | --- | ---: |
| Qwen3-4B | joint | cached-train | train | cached P=[256, 512] × new N=[64, 128]; C=1 | 1 |
| Qwen3-4B | joint | cold-train | train | input=[1, 64, 2048]; C=1 | 1 |
| Qwen3-4B | joint | decode-heldout | sealed_test | B=[2] × total context=[1024] | 8 |
| Qwen3-4B | joint | decode-train | train | B=[1, 4] × total context=[4, 256, 2048] | 8 |
| Qwen3-4B | cached | cached-heldout | sealed_test | cached P=[384] × new N=[96]; C=1 | 1 |
| Qwen3-4B | cached | cached-train | train | cached P=[256, 512] × new N=[64, 128]; C=1 | 1 |
| Qwen3.5-4B | joint | cached-train | train | cached P=[1056, 2112] × new N=[64, 128]; C=1 | 1 |
| Qwen3.5-4B | joint | cold-train | train | input=[1, 64, 2048]; C=1 | 1 |
| Qwen3.5-4B | joint | decode-heldout | sealed_test | B=[2] × total context=[1024] | 8 |
| Qwen3.5-4B | joint | decode-train | train | B=[1, 4] × total context=[4, 256, 2048] | 8 |
| Qwen3.5-4B | cached | cached-heldout | sealed_test | cached P=[1584] × new N=[96]; C=1 | 1 |
| Qwen3.5-4B | cached | cached-train | train | cached P=[1056, 2112] × new N=[64, 128]; C=1 | 1 |

Decode context is the initial total context across the batch; per-request input is `total context / B`. For the training surface, B=1 uses input lengths 4, 256, 2048, and B=4 uses 1, 64, 512. Held-out decode uses B=2 and input length 512. All concurrent TTFT observations are excluded from fitting; the decode window TPOT may train the surface.

## Execution counts

| Job | Cells | Roles | Attempts | Bench invocations | Requests | Bootstrap / warmup / measured requests | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| qwen3-joint | 14 | 29 | 29 | 259 | 406 | 56 / 100 / 250 | 198.51 |
| qwen3-cached | 5 | 10 | 10 | 140 | 140 | 70 / 20 / 50 | 67.51 |
| qwen35-joint | 14 | 29 | 29 | 259 | 406 | 56 / 100 / 250 | 265.35 |
| qwen35-cached | 5 | 10 | 10 | 140 | 140 | 70 / 20 / 50 | 101.65 |

Total: 798 bench invocations and 1092 requests (252 bootstrap, 240 warmup, 600 measured). Selected role statuses: `{"sealed": 78}`. Attempt statuses: `{"sealed": 78}`. Request statuses: `{"succeeded": 1092}`. The JSON excerpt includes every selected attempt number and its recorded history.

## Native report status and basis

| Job | Native kind / schema | Status | Completeness acceptance | Accuracy acceptance | Basis |
| --- | --- | --- | --- | --- | --- |
| qwen3-joint | `held_out_replay_report` / 4 | succeeded | passed | not_assessed | `paired_raw_target_sim` |
| qwen3-cached | `cached_prefill_validation_report` / 3 | succeeded | No native acceptance field | not_assessed | `offline_surface_vs_paired_effective_ttft_ms` |
| qwen35-joint | `held_out_replay_report` / 4 | succeeded | passed | not_assessed | `paired_raw_target_sim` |
| qwen35-cached | `cached_prefill_validation_report` / 3 | succeeded | No native acceptance field | not_assessed | `offline_surface_vs_paired_effective_ttft_ms` |

Completeness acceptance covers evidence scope, pairing, and statistics. It is not an accuracy threshold. Cached reports compare the offline fitted surface with paired effective TTFT `target − zero_cost`; cached-prefix execution is not enabled in the Rust simulator.

## Joint held-out raw TTFT

These errors are computed for each paired target/simulator request before percentile aggregation. The relative denominator is the positive raw target TTFT; no baseline is subtracted again. Each row below is one held-out cell with ten request pairs.

| Model | B / total context | Pairs | Absolute error ms p50 / p95 / p99 | Relative error p50 / p95 / p99 |
| --- | --- | ---: | --- | --- |
| Qwen3-4B | 2 / 1024 | 10 | 59.296829 / 149.268986 / 149.635424 | 42.6116% / 210.0201% / 211.7444% |
| Qwen3.5-4B | 2 / 1024 | 10 | 91.273568 / 176.501309 / 184.811010 | 48.1335% / 124.7396% / 137.4380% |

## Cached held-out cell medians

The primary cached comparison uses the median of the five paired effective TTFT samples for each cell. Each model has only one held-out cached cell, so native `cell_medians` p50/p95/p99 all equal that single cell value. They do not describe a five-request tail distribution.

| Model | P / N | Effective median ms | Prediction ms | Absolute error ms | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 384 / 96 | 65.150251 | 66.404195 | 1.253944 | 1.9247% |
| Qwen3.5-4B | 1584 / 96 | 112.660510 | 113.063127 | 0.402617 | 0.3574% |

## Cached held-out per-request distributions

These percentiles are computed over the five individual paired requests, separately from the cell-median comparison. Relative error divides by each positive effective TTFT. The reconstruction diagnostic instead compares `prediction + same-key zero_cost` with raw target TTFT and uses raw target TTFT as its relative denominator.

| Model | Pairs | Absolute error ms p50 / p95 / p99 | Effective relative error p50 / p95 / p99 | Reconstructed raw relative error p50 / p95 / p99 |
| --- | ---: | --- | --- | --- |
| Qwen3-4B | 5 | 1.253944 / 2.683837 / 2.814243 | 1.9247% / 4.2146% / 4.4263% | 1.8402% / 4.0092% / 4.2076% |
| Qwen3.5-4B | 5 | 2.068375 / 4.329042 / 4.511385 | 1.7965% / 3.9466% / 4.1491% | 1.7417% / 3.7573% / 3.9380% |

## Per-request held-out observations

All measured held-out samples are retained without rounding in the companion JSON excerpt. The tables below use milliseconds and display rounded values.

### Qwen3-4B — joint

| Repeat / index | Target TTFT ms | Simulator TTFT ms | Absolute error ms | Relative error |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 150.576760 | 219.454709 | 68.877949 | 45.7427% |
| 0 / 1 | 71.832666 | 100.192517 | 28.359851 | 39.4804% |
| 1 / 0 | 149.659916 | 99.944207 | 49.715709 | 33.2191% |
| 1 / 1 | 71.706547 | 220.415697 | 148.709150 | 207.3857% |
| 2 / 0 | 150.977428 | 220.567062 | 69.589634 | 46.0927% |
| 2 / 1 | 72.614365 | 99.905555 | 27.291190 | 37.5837% |
| 3 / 0 | 148.659726 | 101.248795 | 47.410931 | 31.8922% |
| 3 / 1 | 70.567534 | 220.294567 | 149.727033 | 212.1755% |
| 4 / 0 | 150.200449 | 101.971510 | 48.228939 | 32.1097% |
| 4 / 1 | 102.247463 | 220.961739 | 118.714276 | 116.1049% |

### Qwen3-4B — cached

| Repeat / index | Target TTFT ms | Zero-cost TTFT ms | Effective ms | Prediction ms | Absolute error ms | Effective relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | 69.310457 | 3.076767 | 66.233690 | 66.404195 | 0.170505 | 0.2574% |
| 1 / 0 | 68.140134 | 2.989883 | 65.150251 | 66.404195 | 1.253944 | 1.9247% |
| 2 / 0 | 66.870274 | 3.312924 | 63.557350 | 66.404195 | 2.846845 | 4.4792% |
| 3 / 0 | 68.062127 | 2.862352 | 65.199775 | 66.404195 | 1.204420 | 1.8473% |
| 4 / 0 | 67.349970 | 2.977579 | 64.372391 | 66.404195 | 2.031804 | 3.1563% |

### Qwen3.5-4B — joint

| Repeat / index | Target TTFT ms | Simulator TTFT ms | Absolute error ms | Relative error |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 196.352218 | 145.104453 | 51.247765 | 26.0999% |
| 0 / 1 | 155.503114 | 319.309047 | 163.805933 | 105.3393% |
| 1 / 0 | 187.586816 | 145.323106 | 42.263710 | 22.5302% |
| 1 / 1 | 132.910167 | 319.798602 | 186.888435 | 140.6126% |
| 2 / 0 | 187.124121 | 318.423493 | 131.299372 | 70.1670% |
| 2 / 1 | 133.194074 | 144.836346 | 11.642272 | 8.7408% |
| 3 / 0 | 186.892513 | 318.600547 | 131.708034 | 70.4726% |
| 3 / 1 | 132.682296 | 144.741076 | 12.058780 | 9.0885% |
| 4 / 0 | 186.427473 | 318.602520 | 132.175047 | 70.8989% |
| 4 / 1 | 131.876302 | 144.672788 | 12.796486 | 9.7034% |

### Qwen3.5-4B — cached

| Repeat / index | Target TTFT ms | Zero-cost TTFT ms | Effective ms | Prediction ms | Absolute error ms | Effective relative error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | 118.757859 | 3.626357 | 115.131502 | 113.063127 | 2.068375 | 1.7965% |
| 1 / 0 | 119.745215 | 3.264763 | 116.480452 | 113.063127 | 3.417325 | 2.9338% |
| 2 / 0 | 114.404398 | 5.898242 | 108.506156 | 113.063127 | 4.556971 | 4.1997% |
| 3 / 0 | 116.142213 | 3.481703 | 112.660510 | 113.063127 | 0.402617 | 0.3574% |
| 4 / 0 | 114.731336 | 3.558908 | 111.172428 | 113.063127 | 1.890699 | 1.7007% |

## Artifact interpretation

The companion [sim-calibrate-serving-3090-20261103.json](sim-calibrate-serving-3090-20261103.json) contains curated native report fields, complete matrix coordinates, execution counts, attempt histories, and all held-out request samples. It is a public excerpt, not an artifact bundle and not a replacement for a native reader input. Process bindings, command paths, and machine-local artifact paths are omitted.

Native reports, profiles, calibration documents, and raw bundles remain outside the repository in `sim-calibration-serving-20261103-evidence.tar.gz`, prepared for attachment to the PR. The following paths are relative to the extracted native artifact bundle:

| Job | Native report |
| --- | --- |
| qwen3-joint | `qwen3-joint/artifacts/reports/validation.json` |
| qwen3-cached | `qwen3-cached/artifacts/reports/cached_validation.json` |
| qwen35-joint | `qwen35-joint/artifacts/reports/validation.json` |
| qwen35-cached | `qwen35-cached/artifacts/reports/cached_validation.json` |

The excerpt does not compare accuracy with a different historical matrix. This campaign covers a small set of held-out coordinates, with five repeats per cell; its p95/p99 estimates have limited sample support. Large joint TTFT errors remain visible and no accuracy pass is claimed. Short decode contexts below the sampled surface remain outside the functional domain; a rejected request does not terminate serving.
