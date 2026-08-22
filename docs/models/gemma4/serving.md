# Gemma 4 serving

**TL;DR:** The engine schedules per iteration: up to the configured decode slots (16 by default) hold requests, each prompt prefills whole at a step boundary by default, and every active request advances one token per batched step. Prompt plus output past the ceiling — 8192 by default, raised up to the checkpoint's 262144 by `PEGAINFER_MAX_CONTEXT` — is refused at admission, while a request that only has to wait for a decode slot queues instead. The two KV families are budgeted separately (7.27 GiB sliding + 2.00 GiB global at 12B, at the defaults). **The default configuration needs a 48 GiB card**: it sits at 32.2 GiB before it serves anything, so a 32 GiB device cannot start it — smaller envelopes are a matter of the slots and ceiling knobs below. A row's output moves with the bucket widths it decodes at, but not with what its companions contain. An opt-in conversation prefix cache (`PEGAINFER_PREFIX_CACHE=K`) resumes multi-turn prompts at the cost of a pre-allocated page budget, and an opt-in overlap lane (`PEGAINFER_ASYNC_PREFILL=green:NN`) trades prefill latency for decode-tail protection under long-prompt admissions. An opt-in chunked walk (`PEGAINFER_MIX_CHUNK_TOKENS=N`) bounds how many prompt rows a mixed admission computes per step, so live streams advance per segment instead of waiting out whole prompts, and a raised ceiling (`PEGAINFER_MAX_CONTEXT`, with `PEGAINFER_DECODE_SLOTS` trading concurrency for context) serves long-context workloads on the same card.

Last touched: 2026-08

## What a step is

The engine thread runs one loop. Each turn it admits whatever the pools can hold, up to the slot ceiling. With streams in flight, admissions share one mixed step with them — the prompts' rows sit in the step's row prefix, each as its own segment, while every active request advances its token in the suffix; a prompt that arrives with nothing active prefills alone as its own step. Up to four coincident prompts gather into the same step, bounded at 512 unseen prompt rows (a warm resume is priced at its suffix), with every popped candidate — gathered, rejected or cancelled — consuming the turn's shared admission budget: gathering amortizes only the ~27 ms step floor while every live stream's inter-token gap pays the whole gathered step (~0.2 ms per row), so short bursts fold their admission staircase and a longer prompt keeps its own step. With the opt-in overlap lane enabled (below), an admission into a live batch prefills asynchronously on its own stream instead of sharing the mixed step. Between admissions, every active request advances exactly one token in a single batched decode step that shares the weight pass. A request that arrives while all slots are taken waits at the head of the queue. It is refused only when nothing is active — when there is no other request whose pages could free up, the pools genuinely cannot hold it and saying so is the honest answer.

Rows retire independently. Requests in one batch have their own frontiers, their own page tables and, for the sliding family, their own released window front, so a short request finishing does not disturb the rows that continue.

| Knob | Value | Where it binds |
| --- | --- | --- |
| Decode slots | 16 | requests beyond this queue |
| Context ceiling | 8192 tokens | prompt + `max_tokens`, enforced at admission and reported to the frontend as the servable length |
| Page size | 16 tokens | both families |
| Sliding window | 1024 tokens | the local family releases its front past this; the global family never releases |

## The two pools

Gemma 4 runs two attention families with different KV shapes, so the budget is two budgets. With 16-token pages, `C = ceil(8192/16) = 512` context pages and `W = ceil(1024/16) + 1 = 65` window pages:

```
local  = C + (slots - 1) * W + 1 = 512 + 15 * 65 + 1 = 1488 pages
global = slots * C + 1           = 16 * 512 + 1      = 8193 pages
```

Those are the knob-off defaults. With `PEGAINFER_MIX_CHUNK_TOKENS=N` set no scan holds more than window plus segment, so the local line shrinks to

```
local  = slots * W + ceil(N / 16) + (MIX_MAX_PROMPTS - 1) + 1 + K * W
global = slots * C + 1 + K * floor(C / 2)
```

— at the defaults (16 slots, `N=2048`, no cache) that is 1172 local pages instead of 1488; the global line only restates the default. With the async prefill lane enabled the local line keeps the default full-context transient — the lane prefills whole. Every page count and byte figure below is measured with the knob off unless it says otherwise.

The shapes behind the page: the local family is 40 layers of 8 KV heads at head_dim 256, the global family is 8 layers of 1 KV head at head_dim 512, and a page carries K and V for every layer of its family. That makes a local page 5 MiB and a global page 256 KiB, so at 12B the knob-off pools are **7.27 GiB local and 2.00 GiB global**, on top of 22.18 GiB of resident weights.

The asymmetry is the design: the local family only has to hold one full-context transient — the request currently prefilling, which has not released its front yet — on top of the window-capped steady footprint of everyone else. The global family never releases, so it stays linear in context for each request's whole lifetime, and that is what makes it the larger page count despite the smaller page.

Each pool also reserves one padding page, which is the `+ 1` in both lines. With `PEGAINFER_PREFIX_CACHE=K` set, both lines grow by the cache's own budget — `+ K * W` local and `+ K * floor(C / 2)` global pages — so cached pages never eat the serving reserve.

## What a client has to send

**Prompts must carry `<bos>` themselves.** This checkpoint's `tokenizer.json` has a pass-through post-processor and its `tokenizer_config.json` sets no `add_bos_token`, so nothing in the serving path prepends it — and Gemma 4 without a leading `<bos>` degenerates into punctuation no matter how the step is scheduled:

```
prompt "The capital of France is"        -> '111.1......11111'
prompt "<bos>The capital of France is"   -> ' Paris.\nthought\nThat is correct. Paris is …'
```

The chat template in `chat_template.jinja` opens with `<bos>`, so chat-formatted prompts already carry one; see `models/gemma4/tokenizer.md` for the rest of the template contract.

Startup also logs one warning that is expected and harmless — the fast tokenizer path rejects this tokenizer's `Replace` normalizer and the server falls back to the Hugging Face tokenizers path:

```
WARN vllm_tokenizer failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers
```

## Running it

```bash
cargo build --release --features gemma4 -p pegainfer-server
target/release/pegainfer \
  --model-path <checkpoint> \
  --served-model-name gemma-4-12b-it \
  --port 18099
```

```bash
curl -s localhost:18099/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-it","prompt":"<bos>The capital of France is",
       "max_tokens":16,"temperature":0}'
```

## What it costs to hold a slot

The pools are sized up front for every configured slot (`PEGAINFER_DECODE_SLOTS`, default 16). Measured at the defaults with the chunk knob off on a 49140 MiB card with the default per-bucket CUDA graphs, the process sits at **33034 MiB with no request in flight** and peaked at 33386 MiB under the serving checks below: 22.18 GiB of weights, 9.27 GiB of pools, and the rest CUDA context, RoPE tables, step buffers and the captured graphs. The eager baseline (`--cuda-graph=false`) measured 32926 MiB idle and peaked at 32932 MiB.

That is a hardware floor, not a target. A 32 GiB device cannot start this configuration at all. Serving a single request needs about 2.6 GiB of pool rather than 9.27, so the slot count sets the floor — `PEGAINFER_DECODE_SLOTS` lowers it, and the raised-ceiling section below shows the measured cells.

## The conversation prefix cache (opt-in)

`PEGAINFER_PREFIX_CACHE=K` (unset by default) keeps copies of up to K completed prompt states, so the next turn of a conversation resumes where its history ends instead of prefilling all of it again. Unset, nothing is allocated and admission behaves exactly as above.

When a request's prefill completes, the engine copies its prompt-state pages — the global family up to the prompt frontier plus the local family's resident window — into cache-owned pages. Only the prompt region is captured: generated tokens do not re-render into the next turn's prompt verbatim, so only the prompt prefix can ever be hit again. At admission the prompt resolves against the cache by longest common prefix, clamped to the sliding-window floor — a resume below the released window front cannot be rebuilt and misses by construction. A hit restores by copying the pages back and prefilling only the unseen suffix, and `Scheduled` reports the resumed count as `cached_tokens`.

The cache brings its own page budget, added to the pool lines above at startup. A prompt longer than half the serving context (4096 tokens today) is not captured — that bound is what keeps the cache's pool share equal to what its entries paid for. A new turn's capture supersedes its conversation's older entry, capacity evicts LRU, and an admission that cannot reserve pages evicts cache entries before waiting.

At `PEGAINFER_PREFIX_CACHE=16` the idle footprint measured **39242 MiB** against the 33034 MiB baseline — the difference is the pre-allocated cache budget.

## The async prefill lane (opt-in)

`PEGAINFER_ASYNC_PREFILL` (unset by default) moves a live-batch admission's prefill onto its own stream, so decode steps keep replaying while the prompt computes. `green:NN` pins the lane to roughly NN% of the SMs via a Green Context — the cap is the mechanism: a `shared` lane's full-width prefill grids starve decode steps, and is kept only for comparison. An unrecognized value or an unviable SM partition refuses to start rather than silently degrading.

One prefill is in flight at most; further arrivals wait while decode keeps stepping, a prompt arriving with nothing active takes the sync path, a restored prefix-cache hit prefills only its unseen suffix, and the sliding window's front release is deferred to the join so no page can be re-allocated under in-flight reads.

Measured (a streaming request, then sixteen ~1900-token prompts admitted at once; two runs per arm): the stream's worst inter-token gap under the flood drops from 387-452 ms — one mixed step at that prompt length — to 75-76 ms with `green:35`, p99 385-432 → 39-40 ms, while the flood's own TTFT p50 grows 3.3-3.7 → 9.8-10.3 s and its wall about 2.4×. The quiet stream and idle footprint are unchanged, so an idle lane costs nothing. That trade is the positioning: a high-concurrency, decode-tail-sensitive profile, not a default — at light load the capped lane only costs TTFT.

## The chunked walk (opt-in)

`PEGAINFER_MIX_CHUNK_TOKENS=N` (64 <= N, below the serving ceiling; unset, `off` or `0` keeps whole-prompt steps; anything else refuses startup) bounds how many prompt rows a mixed admission computes per step. The effective step rounds down to whole 128-row tiles — GEMM and attention consume full tiles, so an unaligned width pays the whole tile on every full segment — which keeps "at most N rows" true while a width under one tile stays exact. Gathered prompts walk shared segment steps: each round fills one N-row budget across the walkers in admission order, every active stream advances one token per round, and a mid-walk segment's sampled row is discarded — no token, no logprob, no stop — until the prompt's final segment produces its first token, emitted at that round's boundary as the walker joins the decode batch. A walker whose client disconnects mid-walk is dropped between rounds. The knob owns every scan: a drained roster's tails and a prompt arriving with nothing active walk their own segments too, paying one ~27 ms step floor per segment where a whole scan paid one — the price of holding window plus segment instead of the full prompt. The exception is the async prefill lane: a live-batch admission goes to the lane and prefills whole. With the knob set, the gather's 512-row ceiling no longer applies: the per-round budget bounds each step instead.

Pages reserve round by round: a walker holds its window plus the segment it is writing, never its whole prompt, and the sliding pool's budget stops scaling with the ceiling — the global family's whole account is still checked at the door, since it never releases. The trade is granularity, measured at 12B: under a flood of sixteen ~3900-token prompts, `N=2048` cut a live stream's flood-phase p99 gap from 855-975 ms to 497-526 ms; at ~1900-token prompts a round can span two prompts, and the same knob raised that p99 from 432-468 ms to 519-537 ms. Off by default; set it for long-context workloads where prompts run to several segments.

## The raised ceiling (opt-in)

`PEGAINFER_MAX_CONTEXT=N` (1024 <= N <= the checkpoint's `max_position_embeddings`, 262144 at 12B) raises the serving ceiling past the 8192 default. Past the default the chunked walk becomes mandatory — a whole scan would hold the full context in sliding pages, so startup refuses a raise without `PEGAINFER_MIX_CHUNK_TOKENS` — and the async prefill lane is refused alongside a raise for the same reason; at or below 8192 neither restriction applies. `PEGAINFER_DECODE_SLOTS=N` (1..16, default 16) is the other budget axis: the global family never releases, so its pool is slots times the ceiling, and a raise buys context back by giving up decode slots. The startup error names both knobs and the page arithmetic when a budget cannot be allocated.

Measured at 12B on a 48 GiB card (idle resident, then behaviour): 64K x 16 slots sits at 46.0 GiB with the default short-load throughput intact (c8 ~148 tok/s); 128K x 8 at 43.5 GiB, same throughput; 262K x 4 at 42.6 GiB. At the full ceiling a 49K-token prompt streams its first token in ~13 s and a 204K-token prompt in ~125 s — prefill cost is ~0.21 ms per token plus a quadratic global-attention term that reaches parity with the linear term around 200K. Decode at depth stays healthy: the inter-token median rises from 28.9 ms over a 13K history to 34.9 ms over 200K. Chunk granularity trades a live stream's stall against nothing at this scale — segment cost dominates the step floor, so halving the segment halves the stream's flood-phase p99 (2613-2673 ms at 8192 down to 381-403 ms at 1024, two coincident ~49K prompts) while admission latency stays flat down to 1024; at 512 the step floor finally surfaces (stall 214-224 ms, admission and wall about 13% dearer). `N=1024` is the recommended long-context profile, `512` the tail-protective option — the knob itself still defaults off. Protocols: the latency table is single streamed completions at `temperature 0` with 64 output tokens under `PEGAINFER_MAX_CONTEXT=262144 PEGAINFER_MIX_CHUNK_TOKENS=2048 PEGAINFER_DECODE_SLOTS=2`; the envelope cells swap in their own ceiling and slots, read idle from `nvidia-smi` after graph capture and add eight concurrent short prompts for the throughput column; the granularity sweep holds the 262144 x 2 configuration, varies only the chunk knob as tabled, and measures a live stream with two coincident ~49K prompts, two runs per setting.

## Measured behaviour

Single GPU (sm_89, x86_64), CUDA 12.9, 12B checkpoint, greedy (`temperature 0`):

| What | Result |
| --- | --- |
| Eight distinct prompts as one batch | every row carried its own request's continuation; none carried another's |
| Eight rows asking for 4…32 tokens | each returned its own count, no row disturbed by another retiring |
| 17 concurrent requests | all completed; the ones past the slot ceiling queued rather than failing |
| One stream cancelled mid-generation | the other three finished normally |
| Four concurrent requests at 1761 prompt tokens, 200 output | all completed across the 1024-token window |

Throughput is eight prompts of 5 to 14 tokens at `max_tokens 24`, `temperature 0`, one run each, measured client-side as completion tokens over the wall time of the whole set: **31.8 tok/s** sending them one after another, **181.4 tok/s** sending them together. It is a fixed request set, not a sustained-load benchmark.

## What concurrency does and does not change

A row decoding in a batch does not produce the same logprobs as the same row decoding alone. That is worth separating from the thing it resembles — a row reading another row's pages — because only one of the two is benign. The variable that matters is the **bucket-width trajectory**: the sequence of padded bucket widths a row's decode steps actually compute at, which depends on when its companions become active and when they retire. Bucketing quantizes it — batch sizes that share a power-of-two bucket share their arithmetic — so fewer distinct trajectories exist than under exact widths. (The table below was measured at exact widths on the eager build that predates bucketing.)

| Contrast | Trajectory | Row's tokens | max abs delta logprob |
| --- | --- | --- | --- |
| One batch repeated three times | same | identical | 0.000000 |
| Companions replaced, same lengths, different content | same | identical | 0.000000 |
| Companions replaced, lengths from 2 to 1601 tokens | same | identical | 0.000000 |
| Seven short companions against seven long ones | changed | differ | 0.595163 |
| Alone against in a batch of eight | changed | differ | 0.623022 |

Hold the trajectory fixed and replace what the other rows are — their content, their prompt lengths — and the row is bit-identical, so **no companion row contaminates it**. Change when companions arrive or retire and the row moves, because the kernels pick shapes and reduction orders by batch size. (That a row reads the *right* positions and page rows in the first place is a separate question, gated by the preps' closed-form tests rather than by this comparison.)

Decode steps compute at power-of-two batch buckets — a batch pads to its bucket with rows that write the pools' reserved padding pages — and are replayed as per-bucket CUDA graphs captured at startup (`--cuda-graph=false` is the eager escape hatch; padding applies either way, so the two modes are the same arithmetic). Bucketing also quantizes the width trajectory: batch sizes that share a bucket share their arithmetic.

The consequence for callers: **greedy output is reproducible for a given workload on an otherwise idle device, not across workloads.** Replaying the same requests the same way returns the same tokens; sending them alongside different traffic changes the widths they decode at and can flip a near-tie. Another process on the same GPU does this too, by moving when each prompt's prefill lands relative to the decodes around it.

## Limits today

- **An admission rides the live decode batch instead of freezing it.** With streams in flight, newcomers' prompts share one eager step with the decode batch: the prompt rows sit in the row prefix as segments, every active stream advances its token in the suffix, and one sampler call covers every newcomer's first token and every active row. Measured with a streaming request underneath a flood of one-token requests: its inter-token gap stays at about 30 ms — one mixed step — whether 16, 48 or 96 requests are queued, where the frozen-prefill scheduling this replaced measured about 500 ms at the same depths. A burst of sixteen coincident ~120-token prompts folds its admission staircase — TTFT p50 and p99 both drop 20-30% — while the stream's worst gap stays bounded by the gathered step's own cost (~130 ms at the 512-row ceiling); a 16 × ~1900-token flood is untouched, since a long prompt keeps its own step. Admission work per turn stays bounded by the slot ceiling, gathered or not.
- **Whole-prompt prefill by default.** A prompt runs whole in one step unless the opt-in chunked walk above bounds the step's prompt rows; with the knob set a walker reserves pages round by round instead of holding its whole prompt.
- **No cross-request prefix sharing.** Two live requests with a common prefix pay for it twice; the opt-in conversation cache above serves consecutive turns of one conversation, not concurrent requests.
- **Single GPU.** No tensor parallelism for this line yet.
- **KV capacity is not reported to the frontend**: the engine logs `kv_cache_size_tokens=None`, so the frontend's capacity metrics stay empty for this model line.
