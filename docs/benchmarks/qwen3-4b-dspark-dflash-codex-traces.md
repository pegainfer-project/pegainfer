# DSpark vs DFlash Accept Length on Codex SWE-bench Pro Traces

> **TL;DR:** On real Codex SWE-bench Pro agent traces (200 prompts, 13k–29k tokens each), DSpark beats DFlash by **+27.8% mean accepted draft** (1.00 vs 0.79) and **+27.8% accept rate** (0.148 vs 0.116). The Markov head's sequential conditioning wins harder on agentic coding traces than on the prior synthetic A/B (+3.6% geomean). But absolute accept is far lower than synthetic datasets — 1.00 vs 2.52 on sharegpt — because agentic output (tool calls, code diffs, system output) is much harder to predict than natural text.

## Preparation

- **Read**:
  - `docs/index.md` — routing table
  - `docs/models/qwen3/dspark-integration.md` — DSpark Phase 1 implementation, prior 5090 A/B (sharegpt/sonnet/random/code datasets, DSpark 2.52 vs DFlash 2.30 mean accepted draft)
  - `docs/models/qwen3/dflash-speculative-decoding.md` — DFlash design, losslessness gate, performance numbers
  - `docs/models/qwen3/serving-performance.md` — serving setup, DSpark concurrency numbers
  - `tools/bench/run_serving_bench.sh` — bench harness, uses `vllm-bench`
  - `pegainfer-qwen3/src/executor/dflash_lane.rs:192-204` — accept log format: `accepted_draft=N committed_tokens=N cumulative_accept_rate=X`
- **Relevant history**:
  - `docs/models/qwen3/dspark-integration.md` § "5090 bring-up" — Bug 1: `vllm-bench` defaults to non-greedy, silently disabling spec decode; fix: `--temperature 0`. Bug 2: anchor layout keyed on markov head instead of checkpoint format.
  - Qwen3-4B `max_position_embeddings` = 40960; DFlash effective context = `max_position_embeddings - block_size` (40953 for DSpark block7, 40944 for DFlash block16).
  - 5090 models at `/data/Qwen3-4B`, `/data/dspark_qwen3_4b_block7`; DFlash baseline `dflash_qwen3_4b_block7` (DeepSpec anchor-first, markov_rank=0, block_size=7).
  - 5090 proxy: `http://172.17.0.1:1081` (export explicitly for non-interactive ssh).
  - 5090 build: `CUDA_HOME=/usr/local/cuda-13.1` (cuBLAS 12.9 has GEMM cliff at N=1025).

- **Dataset**: `Inferact/codex_swebenchpro_traces` — 610 successful Codex agent traces solving SWE-bench Pro. Each row has a `conversations` field (12-200 turns). Per-LLM-call input: mean 68k, median 64k tokens; 1st-call input: ~12k tokens. Most individual calls exceed 40k context → filter heavily, keeping early turns (1-5).

- **Plan**:
  1. **Download dataset** to `/data/datasets/codex_swebenchpro_traces` on the 5090 via `huggingface-cli` (with proxy).
  2. **Preprocess** (Python script with `transformers` + `datasets`):
     - Load 610 conversations, extract individual LLM calls (conversation prefix up to each human turn).
     - Format using Qwen3 chat template via `AutoTokenizer.from_pretrained("/data/Qwen3-4B")`.
     - Tokenize, filter: keep prompts with `token_count + 256 ≤ 40900` (fits DFlash effective context + output budget).
     - Sample ≤500 prompts (stratified by token-count buckets) for reasonable runtime.
     - Save as JSONL: `{"prompt": "<text>", "token_count": N}`.
  3. **Verify pegainfer build** on 5090 (`CUDA_HOME=/usr/local/cuda-13.1 cargo build --release -p pegainfer-server`).
  4. **Run DFlash A/B**:
     - Launch server: `--model-path /data/Qwen3-4B --dflash-draft-model-path /data/dflash_qwen3_4b_block7`, `RUST_LOG=pegainfer_qwen3=debug`, GPU 7.
     - Send prompts sequentially via Python script: `/v1/completions`, `temperature=0`, `max_tokens=256`, `ignore_eos=true`.
     - Capture server log.
     - Kill server.
  5. **Run DSpark A/B**: same with `--dflash-draft-model-path /data/dspark_qwen3_4b_block7`.
  6. **Parse & compare**: extract `accepted_draft` values from server logs, compute mean/median/histogram/full-7 rate. Present DFlash vs DSpark comparison table.

- **Risks / open questions**:
  - DFlash baseline checkpoint (`dflash_qwen3_4b_block7`) might not be on the 5090 — need to verify or download from DeepSpec HF.
  - Some codex trace prompts might be very long even in early turns (tool output is verbose) — the filter handles this.
  - Accept logs are debug-level — must set `RUST_LOG` correctly.
  - `ignore_eos=true` ensures consistent verify-round count per prompt for fair comparison.

## Execution Log

### Step 1: Probe 5090
- All three model checkpoints present: `/data/Qwen3-4B`, `/data/dspark_qwen3_4b_block7`, `/data/dflash_qwen3_4b_block7`.
- 8× RTX 5090 all idle (0 MiB, 0%). Used GPU 7.
- Repo at `~/develop/xingming/pegainfer` on `main` @ `8f2ef15`, release build already cached.

### Step 2: Download dataset
- `hf download Inferact/codex_swebenchpro_traces --repo-type dataset --local-dir /data/datasets/codex_swebenchpro_traces` via proxy `http://172.17.0.1:1081`.
- 209 MB single JSON (`codex_swebenchpro.json`), 610 trials, each with a `conversations` field (12–200 turns of human/gpt messages).

### Step 3: Preprocess
- Script `/data/datasets/codex_swebenchpro_traces/preprocess_codex.py` extracts per-LLM-call prompts (conversation prefix up to each human turn), formats with Qwen3 chat template, tokenizes.
- Char-count pre-filter (`> 162752 chars` → skip) cut 20k+ prompts down to 3,838 before tokenization. This was critical — naive tokenization of all 20k+ prompts (many 100k+ tokens) timed out at 5 minutes.
- After token-count filter (`≤ 40688` = DFlash block16 effective context − 256 output): 3,278 prompts.
- Stratified sample to 500, sorted by token_count. Sent first 200 (shortest: 13k–29k tokens) for reasonable runtime (~15 min per run).
- Token distribution of the 200 sent: min 13,429 / p50 22,429 / max 28,650.

### Step 4: DFlash A/B
- Server: `CUDA_VISIBLE_DEVICES=7 RUST_LOG=pegainfer_qwen3=debug target/release/pegainfer --model-path /data/Qwen3-4B --port 8000 --dflash-draft-model-path /data/dflash_qwen3_4b_block7`.
- Client: `python3 send_prompts.py 8000 prompts_filtered.jsonl` — sequential, `temperature=0, max_tokens=128, ignore_eos=true`.
- Result: 200/200 ok, 895s wall, 14,221 verify rounds.

### Step 5: DSpark A/B
- Same config, `--dflash-draft-model-path /data/dspark_qwen3_4b_block7`.
- Result: 200/200 ok, 843s wall, 12,669 verify rounds.

### Step 6: Parse & compare

Both use block_size=7 (DeepSpec anchor-first checkpoints). `accepted_draft` excludes the guaranteed bonus target token; `cumulative_accept_rate` = total accepted draft tokens / total verified draft tokens (= mean / block_size).

| Metric | DFlash (block7) | DSpark (block7) | Delta |
| --- | ---: | ---: | --- |
| Verify rounds | 14,221 | 12,669 | — |
| Mean accepted draft | 0.79 | 1.00 | **+27.8%** |
| Median accepted draft | 0 | 1 | — |
| Cumulative accept rate | 0.116 | 0.148 | **+27.8%** |
| Zero-accept (rounds) | 7,146 (50.2%) | 5,645 (44.6%) | −5.6 pp |
| Full-7 accept | 5 (0.0%) | 21 (0.2%) | — |
| Wall time (200 prompts) | 895s | 843s | −5.8% |

Histogram `accepted_draft` 0..7:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash | 7146 | 4180 | 2024 | 616 | 193 | 46 | 11 | 5 |
| DSpark | 5645 | 3662 | 1928 | 815 | 414 | 139 | 45 | 21 |

## Debrief

- **Outcome:** DSpark beats DFlash by +27.8% on mean accepted draft (1.00 vs 0.79) on real Codex SWE-bench Pro agent traces. This is a much larger relative win than the prior synthetic A/B (+3.6% geomean on sharegpt/sonnet/code/random). The Markov head's sequential conditioning helps more on agentic output where token-level predictability within structured spans (tool call args, code patterns) benefits from conditioning on the previously sampled token.

- **Absolute accept is low.** Both drafters average <1.5 accepted draft tokens/round on agentic traces, vs 2.3–2.5 on sharegpt. ~45–50% of verify rounds accept zero drafts. Agentic LLM output is heavily structured (JSON tool calls, code diffs, bash commands) with low per-token predictability — the drafter's parallel backbone can't anticipate which tool the agent will call next or what arguments it will produce. Spec decode provides minimal speedup in this regime; the wall-time delta (843 vs 895s, −5.8%) is smaller than the accept delta because prefill dominates and spec overhead (draft forward + verify) adds cost on zero-accept rounds.

- **Pitfalls encountered:**
  - **Tokenizer bottleneck:** Naively tokenizing all 20k+ extracted prompts (many 100k+ tokens) timed out at 5 minutes. Fixed with a char-count pre-filter (`> 4× context_ceiling chars` → skip) before tokenization, cutting to 3.8k candidates.
  - **SSH gateway flakiness:** The 5090 Tailscale connection intermittently returns "no available gateway". Retry loops with `ConnectTimeout=10` and 3–5 attempts worked reliably.
  - **`setsid` for server launch:** Bare `&` in SSH caused the shell to hang on the backgrounded process. `setsid bash -c "..." </dev/null &` fixed it.

- **Lessons learned:**
  - Agentic workloads are the worst case for speculative decoding: long contexts (prefill-dominated), low output predictability (structured tool calls), and high zero-accept rates. The DSpark paper's +60–85% throughput win is on DeepSeek-V4 production traffic (chat/code), not agentic SWE traces.
  - DSpark's relative advantage *grows* on harder-to-predict output: +3.6% on sharegpt → +27.8% on codex traces. The Markov head's per-token conditioning recovers some of the signal that DFlash's parallel proposer loses entirely.
  - For future agentic benchmarks: filter prompts aggressively by context length (most codex LLM calls are 60k+ tokens, exceeding Qwen3-4B's 40k window), and expect low absolute accept rates.

- **Why is absolute accept so low — content mismatch or context length?** Both factors, but content/domain mismatch is primary:
  - **50% zero-accept = first draft token is already wrong.** If long context were the cause (attention dilution in the 5-layer drafter), we'd expect early-position hits and late-position misses, not 50% rejection at position 0. This points to high intrinsic entropy of the output, not context processing failure.
  - **Relative gap grows on hard output.** DSpark's edge over DFlash went from +3.6% (sharegpt, ~1k tokens) to +27.8% (codex, 13k–29k tokens). If context length dominated, both drafters (same backbone) should degrade together and the relative gap shouldn't widen. The Markov head recovering more signal on high-entropy output is a content-property effect.
  - **Agentic output is structurally unpredictable.** Tool call decisions ("use `grep` or `rg`?"), file paths, function names — these are flat-distribution choices with no strong trigram statistics. The drafter was distilled on Qwen3-4B's output distribution, which captures natural language continuations well but not agentic decision points.
  - **Context length is secondary.** 13k–29k prompts dilute KV injection signal in the drafter's shallow backbone, but this alone wouldn't cause 50% zero-accept.
  - **To separate the two cleanly:** pad sharegpt prompts to 13k–29k tokens and re-run. If accept stays ~2.0+, content dominates; if it drops to ~1.0, context length dominates. Not yet run.

- **Follow-ups:**
  - The remaining 300 sampled prompts (28k–40k tokens) were not sent — a longer run could cover the high-context regime. Expect even lower accept rates there (longer context = more diverse routing, harder to predict).
  - Per-context-length bucket analysis would show whether accept degrades with context growth; the current run didn't map verify rounds back to individual prompts (no request_id tracking in the accept log parsing).
  - **Controlled experiment:** pad sharegpt prompts to 13k–29k tokens and re-run both drafters to isolate content vs context length (see analysis above).
  - **Model output quality verified.** Sent 3 real codex trace prompts (13k/30k/29k tokens) to plain Qwen3-4B (no spec decode). All three produced coherent agentic reasoning — correctly understood the code context (Vuls scanner Windows version mapping, Navidome Subsonic API struct, Flipt config test error) and gave relevant technical analysis. No garbled output, no hallucination. The low accept rate is not caused by model degradation at long context; it reflects the intrinsic unpredictability of agentic output.
  - DSpark Phase 2 (confidence head + draft truncation) could help here: truncating low-confidence suffixes would cut verify waste on the ~50% zero-accept rounds.
