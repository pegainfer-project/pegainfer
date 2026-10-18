# bs1 4k/64 vLLM vs PegaInfer

**Created**: 2026-05-04
**Status**: complete
**TL;DR**: On RTX 5090, `bs=1`, `input_len=4096`, `output_len=64`, `num_prompts=20`, `max_concurrency=1`, no vLLM prefix cache: PegaInfer finished `5.7%` faster wall-clock, with TTFT median `177.1ms` vs vLLM `197.8ms`. Decode TPOT was slightly slower: `6.47ms` vs vLLM `6.36ms`. PegaInfer's streaming `usage.completion_tokens` is overreported through the vLLM frontend in this run, so output throughput should be recomputed from the fixed target length.

## Preparation

- **Read**:
  - `docs/index.md` - located the comparative benchmark docs.
  - `docs/playbooks/bench-vs-vllm.md` - confirmed both engines should be driven by `vllm bench serve`, with `--ignore-eos`, random dataset, `--temperature 0`, and sequential same-GPU runs.
  - `docs/subsystems/scheduler/scheduler.md` - confirmed previous fixed-length and realistic benchmark interpretation, especially vLLM cold-start and TTFT/TPOT reading.
- **Relevant history**:
  - `docs/subsystems/scheduler/scheduler.md` showed fixed-length single-concurrency results should be interpreted as latency probes rather than full serving saturation claims.
- **Plan**:
  1. Use `vllm` as the client and vLLM server.
  2. Use the release `pegainfer` binary for the PegaInfer server.
  3. Run `input_len=4096`, `output_len=64`, `num_prompts=20`, `max_concurrency=1`, `request_rate=inf`, after a 3-request warmup for each engine.
  4. Save JSON/log artifacts under a timestamped result directory and compare TTFT/TPOT/throughput.
- **Risks / open questions**:
  - vLLM prefix caching must be disabled for a fair random-prompt prefill comparison.
  - PegaInfer's vLLM frontend may not report streaming usage with the exact same accounting as vLLM.

## Execution Log

### Step 1: Environment check
- Confirmed vLLM in the benchmark environment:
  - `vllm --version` returned `0.19.1`.
  - `uv pip list --python <python-venv>/bin/python` showed `vllm 0.19.1`, `torch 2.10.0`, `flashinfer-python 0.6.6`, and `flashinfer-cubin 0.6.6`.
- Confirmed model path:
  - `<model-path>`, size `7.6G`.
- Built PegaInfer server binary in the validation worktree:
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TRITON_PYTHON=<python-venv>/bin/python cargo build --release -p pegainfer --bin pegainfer`
  - The validation shell session hung after the build process ended, but `target/release/pegainfer` existed with timestamp `2026-05-04 21:11`.

### Step 2: vLLM run
- First vLLM run used default prefix-cache behavior and showed prefix cache hits in the server log, so it was not used for the final comparison.
- Reran vLLM with explicit `--no-enable-prefix-caching`.
- Result directory:
  - `<bench-results-dir>`
- Measured JSON:
  - `vllm-noprefix-in4096-out64-c1-n20.json`
- Command shape:
  - server: `vllm serve <model-path> --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 1 --no-enable-prefix-caching --served-model-name Qwen3-4B`
  - client: `vllm bench serve --backend openai --model Qwen3-4B --dataset-name random --input-len 4096 --output-len 64 --num-prompts 20 --request-rate inf --max-concurrency 1 --ignore-eos --temperature 0 --tokenizer <model-path>`
- Results:
  - completed `20`, failed `0`.
  - duration `11.968s`.
  - request throughput `1.671 req/s`.
  - output throughput `106.952 tok/s`.
  - TTFT median `197.819ms`, p99 `202.206ms`.
  - TPOT median `6.359ms`, p99 `6.366ms`.
  - ITL median `6.389ms`, p99 `6.638ms`.

### Step 3: PegaInfer run
- PegaInfer served model ID was `<model-path>`, not `Qwen3-4B`, so the client model name was set to `<model-path>`.
- Measured JSON:
  - `pegainfer-in4096-out64-c1-n20.json`
- Command shape:
  - server: `target/release/pegainfer --model-path <model-path> --port 8000`
  - client: same `vllm bench serve` shape as vLLM, except `--model <model-path>`.
- Raw results:
  - completed `20`, failed `0`.
  - duration `11.287s`.
  - request throughput `1.772 req/s`.
  - TTFT median `177.063ms`, p99 `179.585ms`.
  - TPOT median `6.465ms`, p99 `6.470ms`.
  - ITL median `6.464ms`, p99 `6.546ms`.
- Accounting caveat:
  - JSON `total_output_tokens` was `5312`, but the fixed workload was `20 * 64 = 1280` output tokens and the timing matches 64 generated tokens per request.
  - For this run, PegaInfer output throughput should be recomputed as `1280 / 11.287s = 113.401 tok/s`, not the raw JSON `output_throughput`.

### Step 4: Comparison
- PegaInfer vs vLLM no-prefix:
  - Wall duration: `11.287s` vs `11.968s` (`5.7%` faster).
  - Request throughput: `1.772` vs `1.671 req/s` (`6.0%` higher).
  - Corrected output throughput: `113.401` vs `106.952 tok/s` (`6.0%` higher).
  - TTFT median: `177.063ms` vs `197.819ms` (`10.5%` lower).
  - TTFT p99: `179.585ms` vs `202.206ms` (`11.2%` lower).
  - TPOT median: `6.465ms` vs `6.359ms` (`1.7%` higher/slower).
  - ITL p99: `6.546ms` vs `6.638ms` (`1.4%` lower).
- Stopped servers after the run; port `8000` was no longer listening.

## Debrief

- **Outcome**: Completed the `bs=1`, `4k input`, `64 output` single-concurrency probe on RTX 5090. PegaInfer has better prefill/TTFT and slightly slower decode TPOT; wall-clock request throughput is higher because TTFT dominates this shape.
- **Pitfalls encountered**:
  - The first vLLM measurement had prefix cache hits. It was rerun with `--no-enable-prefix-caching`.
  - The validation shell session can remain open after some long build/server scripts even when the validation work has finished; checking validation process state is necessary before assuming a command is still running.
  - PegaInfer's vLLM frontend overreported streaming `completion_tokens` for this benchmark, so the raw output throughput field in `vllm bench` JSON is not reliable for PegaInfer here.
- **Lessons learned**:
  - For fixed-output PegaInfer comparisons through `vllm bench serve`, trust TTFT/TPOT/ITL and recompute output throughput from requested output length until streaming usage accounting is fixed.
  - Disable vLLM prefix caching for random synthetic prefill probes unless prefix-cache behavior is explicitly part of the experiment.
  - At this shape, the new Qwen3 prefill q64 path shows up as a TTFT advantage against vLLM, while decode remains essentially parity.
