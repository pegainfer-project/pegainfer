# K3 serving roadmap

**TL;DR**: The target deliverable is **TP x × DP y × EP (x·y)** with TP never
crossing a machine (x ≤ 4, one GB300 tray), plus **MTP speculative decoding**
(draft: [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark)).
Everything below is ordered by what gates the real 896-expert model.

Last touched: 2026-08

## Where bring-up stands (2026-08-18)

Single-tray EP4 decode + chunked prefill are done end-to-end on the pruned
224-expert checkpoint (full 93-layer depth): FlashKDA chunkwise KDA, FlashMLA
dense MLA prefill, 4224-token chunks, 2048-token TTFT 25.8 ms at the 4-layer
snapshot — see `bring-up.md`. Today's parallelism is TP1 × attn-DP4 × EP4,
in-process, one tray. The **real model does not fit any single-tray shape**:
896 experts need EP8 (112 experts/rank, 2 trays) or EP16 (56/rank, 4 trays),
both AOT-ready on the DeepGEMM side (`K3_DEEPGEMM_SM100_GROUPS`) and both
cross-machine — so multi-node is the gate to serving real K3 at all, not a
scaling nicety.

## Target: TP x × DP y × EP (x·y), x ≤ 4

- **x = TP width, intra-tray only** (max 4 — never across the NVLink tray
  boundary… the fabric spans the rack, but TP's all-reduce cadence wants the
  tray). Attention TP shards the 96 MLA heads / 96 KDA heads and their
  per-head weights (w_q_b, w_kv_b, w_o, KDA q/k/v/g + state). Known MLA
  tension to design around: the paged **latent** KV is head-agnostic, so TP
  ranks either replicate the latent cache or share one copy per tray —
  sharding heads does not shard the latent.
- **y = DP width** = independent schedulers with their own requests. The
  free-running rank design is already DP-shaped (a rank pads when idle;
  only mega launches pair); DP across trays is that same contract stretched
  over the fabric.
- **EP width = x·y**: every GPU in the deployment holds an expert shard, and
  the MegaMoE pairing spans all of them.

Work items, roughly in dependency order (1-3 DONE 2026-08-18, see
`multi-node-ep.md` — EP16 over 4 trays byte-identical to single-tray EP4;
worlds 224x{1,4,8,16} and 896x{8,16,32,64} instantiated; ssh fleet launcher
`scripts/k3_ep_fleet.sh`, sbatch variant still open):

1. ~~**Cross-machine mega transport**~~ — fabric-handle slabs + TCP
   rendezvous (`--k3-ranks`/`--k3-rendezvous`), transport otherwise
   unchanged.
2. ~~**Mega ep8/ep16 instantiation**~~ — every world is a
   `k3_mega_world_supported` entry now, split across three TUs.
3. ~~**Fleet launch**~~ — `scripts/k3_ep_fleet.sh` (ssh); a slurm/sbatch
   variant per `~/slurm/glm52_ep16_d.sbatch` remains open.
4. **Attention TP (x up to 4)** — after multi-node EP works TP1; TP shrinks
   the per-rank attention/dense replication and is what makes wide-EP
   memory-comfortable. Needs the latent-cache decision above, TP'd KDA state
   slabs, and a tray-local all-reduce in the step.

## MTP (speculative decoding)

Draft checkpoint: **https://huggingface.co/RadixArk/Kimi-K3-DSpark**.
Support the K3 MTP head end-to-end: draft-step execution in the decode loop,
per-slot draft span, acceptance handling in the scheduler, EP lockstep
accounting for the extra launches (compare GLM5.2's native MTP: profile is
`slots × (1 + drafts)` rows per step). Interacts with the 96-row step budget
and CUDA-graph capture.

## Also on the list (from bring-up "Next", kept here for one view)

- **Varlen multi-request prefill chunks** — pack several prompts into one
  4224-row step (FlashMLA varlen entry + FlashKDA varlen config both exist
  upstream in the vendored sources; scheduler-side continuous batching of
  prefills). The single-instance input-throughput lever.
- **Full-depth TTFT baseline** — measure the pruned checkpoint at 93 layers,
  single-tray EP4. topk = 16 is unchanged by the pruning, so per-token
  routed FLOPs match the real model; this is the honest perf proxy until
  EP16 exists.
- **Real sampling** — `SamplingParams` is still ignored (argmax only).
- CUDA graphs over the EP4 fused path / launch-ahead — **evaluate before
  building**: the EP4 decode profile (`benchmarks/k3-ep4-decode-profile.md`)
  found the step is *not* launch-bound, so measure what capture would
  actually buy on the EP path before spending on it.
- kv-store `BlockPool` integration (content addressing / prefix reuse for
  the MLA latent pages).
- W-chunked prefill context loop + LSE merge (only if `max_ctx` outgrows the
  fixed expansion workspace); trim the wide step's f32 partial scratch
  (~7 GB at the 4224 bucket) if it ever bites.
- Paged-attention decode kernel perf pass (3-sweep recompute reads the cache
  three times; matters at 24 MLA layers × long contexts).

**Next action**: attention TP (item 4), or the varlen prefill packing —
multi-node EP is done.
