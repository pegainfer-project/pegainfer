# KV-aware routing (Dynamo frontend)

TL;DR: When pegainfer Qwen3 workers run behind a Dynamo frontend, the frontend's
KV router routes each request to the worker that already holds the longest
prefix of its tokens, fed by the workers' KV block events. Measured on 8×Qwen3-4B
with a multi-turn chat workload: follow-up-turn TTFT stays flat at ~45 ms under
KV routing while round-robin and random balloon to 160–180 ms, because turns 2+
of a conversation reuse the cached history on their home worker instead of
re-prefilling it on a cold worker. Router-side prefix overlap averaged **0.72**
under KV routing and **0** under the stateless policies. An independent
`vllm-bench --multi-turn` run cross-checks it (4096-token history → KV overall
median TTFT **52 ms vs 214 ms** round-robin, 4×). The size of the win scales with
how much conversation history is reused — short contexts on a fast GPU show only a
few ms, which is expected, not a routing failure.

Last touched: 2026-06

## Why this exists

A multi-replica deployment has to decide which worker serves each request. The
naive policies are stateless:

- **round-robin**: rotate through workers in order.
- **random**: pick a worker uniformly.

Both ignore where a request's KV cache already lives. For multi-turn chat that
is exactly the wrong thing to ignore: turn N of a conversation is `system prompt
+ all prior turns + new user message`, so its prefix is almost entirely the
content the worker that served turn N−1 already has cached. Sending turn N to a
different worker throws that cache away and re-prefills the whole history.

## Mechanism

Think of it as cache-affinity routing over a shared prefix index.

1. A worker finishes prefill and seals full KV blocks. Each block has a content
   hash (a fingerprint of its tokens).
2. The worker publishes **store** / **remove** block events over the message
   bus. (This is opt-in per worker; a worker with routing off publishes nothing
   and pays nothing.)
3. The frontend folds those events into a **radix tree** keyed by block hash,
   tagging each block with the worker that holds it.
4. On a new request the router walks the request's token prefix through the
   radix tree, finds the worker with the longest cached overlap, and routes
   there. With no overlap it falls back to load.

So the router converges a conversation onto one worker: turn 1 lands somewhere
(cold), its blocks register, and turns 2+ match that worker's blocks and follow
it. Round-robin / random scatter the turns and force cold re-prefills.

The block hash the router matches on must be byte-identical on both sides. The
worker computes it with the same XXH3 seed and block size the router uses to
hash incoming request tokens, so a request prefix and the stored block it should
match produce the same hash. (If they ever drift, overlap silently collapses to
zero and KV routing degrades to load-based — indistinguishable from round-robin.
The hit-rate gate below is what catches that.)

## Benchmark method

Reproducible A/B across the three router modes the frontend exposes
(`--router-mode {kv,round-robin,random}`); everything else held fixed.

- **Topology**: 8 Qwen3-4B workers, one GPU each (RTX 5090, 32 GB), KV routing
  events on; one Dynamo frontend; message bus + KV store via local containers.
- **Workload**: multi-turn chat — 32 independent conversations, 4 turns each,
  concurrency 8, temperature 0, fixed seed. Each conversation opens with a
  ~500-token **unique** message (so its growing history is a private shared
  prefix, not shared across conversations), and every turn requests **64 output
  tokens** (real decode + a growing prompt, not a degenerate 1-token probe). The
  client accumulates the assistant's reply into the next turn's messages, so by
  turn 4 a cold re-prefill is the full conversation so far. Any OpenAI-chat
  driver that maintains conversation state reproduces this — a small async client
  or `vllm-bench --multi-turn` both work; the only requirement is that turns
  actually carry the prior history forward.
- **Fairness**: workers are **restarted cold before every mode** so no prefix
  cache carries over between runs.
- **Primary metric**: TTFT (queue + prefill latency to first token), overall and
  **per turn index** — turn 0 is cold for every mode; the routing effect shows up
  in turns 1–3.
- **Hit-rate gate**: the frontend's `kv_hit_rate` (router-side prefix-overlap
  fraction per request) must be substantially > 0 under KV routing, else the run
  is not a valid test of routing — only of load. Round-robin / random never
  consult the index, so they record no `kv_hit_rate` samples at all.

The worker also reports its own prefix-cache hit as OpenAI
`usage.prompt_tokens_details.cached_tokens` (commit surfacing it: the schedule
event's matched-prefix length, previously dropped by the adapter). That is the
per-response confirmation of the same reuse the router's `kv_hit_rate` estimates.

## Results

### Per-turn TTFT (p50 / p99, ms)

| turn | KV (p50/p99) | round-robin (p50/p99) | random (p50/p99) |
|------|--------------|------------------------|-------------------|
| 0 (cold) | 140.8 / 154.5 | 141.2 / 153.7 | 153.7 / 721.8 |
| 1 | **44.9 / 47.8** | 44.9 / 167.4 | 164.3 / 315.5 |
| 2 | **45.1 / 47.4** | 163.6 / 177.3 | 173.6 / 325.3 |
| 3 | **45.3 / 47.7** | 171.5 / 188.0 | 180.5 / 331.5 |
| all turns | **45.2 / 153.2** | 148.7 / 184.6 | 170.1 / 593.4 |

Turn 0 is cold for every policy and lands within noise (~140–154 ms). From turn 1
on, KV routing holds TTFT flat at ~45 ms — only the new turn's tokens prefill,
the history is a cache hit on the home worker. Round-robin and random re-prefill
the growing history on a cold worker every follow-up turn, so their TTFT sits at
160–180 ms and creeps up as the history grows. KV's all-turns p50 is **3.3×**
lower than round-robin and **3.8×** lower than random. Whole-workload wall time
tracks the same ordering: KV 7.2 s, round-robin 9.3 s, random 12.5 s.

(Round-robin's turn-1 p50 happens to match KV — with 8 workers and concurrency 8
the rotation sometimes returns a conversation to its home worker for the first
follow-up — but its p99 already shows the misses, and by turns 2–3 the rotation
has walked every conversation off its cache.)

### Cache hit

Under KV routing the router recorded mean prefix overlap **0.72** across the 128
requests (`kv_hit_rate` sum 91.7 / count 128) — i.e. on average 72% of a
request's blocks were already cached on the worker it was routed to, which is
exactly the warm tail of a 4-turn conversation. Round-robin and random recorded
**no** `kv_hit_rate` samples: they never consult the prefix index, so there is no
overlap to report and the metric stays empty.

The per-response `cached_tokens` field confirms the same reuse from the worker's
side. Firing one identical 1289-token prompt twice under KV routing: the first
(cold) response carries no `prompt_tokens_details`; the second comes back with
`prompt_tokens_details.cached_tokens = 1280` — the whole prompt minus the last
partial block, served from the home worker's prefix cache.

### vllm-bench cross-check

The same effect reproduces on an independent driver (`vllm-bench --multi-turn`,
which accumulates each turn's history into the next request), confirming it is not
an artifact of the custom client. 4 Qwen3-4B workers, a 4096-token opening turn
plus 64-token follow-ups, 4 turns, concurrency 4, cold-restart per mode:

| metric | KV | round-robin |
|--------|----|-------------|
| overall median TTFT | **52.7 ms** | 213.6 ms |
| follow-up turns (warm) | ~50 ms | ~138 ms |

KV's overall median TTFT is **4×** lower because turns 2+ hit the 4096-token
history on their home worker; round-robin re-prefills it on a cold worker almost
every follow-up. The router kept each conversation pinned to one worker (request
counts 12/13/12/11 across the four workers).

### When the win is small

The benefit is exactly the re-prefill cost it avoids, so it scales with **how
much reused context there is** and with **whether that context is still cached**:

- **Short reused context** — with a ~1k-token history, the avoided prefill is only
  ~60 ms on a fast GPU, so KV beats round-robin by a few ms, not multiples. The
  win grows with conversation/system-prompt length.
- **Cache pressure** — at high concurrency relative to KV capacity (e.g. many
  long contexts in flight per worker), the home worker's blocks can be evicted
  before the next turn, and the router also balances load away from a busy worker.
  Both shrink overlap. Keep concurrency at or below the worker count for a clean
  multi-turn affinity win; that is also where the router most strongly honors the
  prefix match over load.

This is *why* a careless A/B (short prompts, saturating concurrency) can show KV
≈ round-robin even though routing is working — the avoided cost is just small or
the cache didn't survive. The `kv_hit_rate` gate distinguishes "no benefit
because overlap collapsed" (a bug) from "small benefit because the reused prefix
is short" (expected).

### Takeaway

KV-aware routing is the difference between paying for the conversation history
once and paying for it on every turn. For multi-turn chat at fan-out, it is not a
marginal tuning knob — it removes the dominant TTFT cost of follow-up turns. The
`kv_hit_rate` gate (> 0 under KV, empty otherwise) is the check that the
block-hash bridge between worker and router is actually matching; a run where KV
ties round-robin means overlap collapsed to zero and the result is meaningless.

## Reproduce

The frontend selects the policy with `--router-mode {kv,round-robin,random}`.
Cold-restart the workers before each mode, point any state-carrying OpenAI-chat
client at the frontend with the multi-turn workload above (32 conversations ×
4 turns, unique long opener, 64 output tokens, concurrency 8), and compare
per-turn TTFT. Before trusting any TTFT delta, scrape `/metrics` and confirm
`dynamo_component_router_kv_hit_rate` has samples above the `le="0"` bucket under
KV mode — that is the gate that the routing is real and not silently degraded to
load-based.
