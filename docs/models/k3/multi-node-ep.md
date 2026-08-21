# K3 multi-node expert parallelism

**TL;DR**: K3 serves across machines: EP worlds 224x{1,4,8,16} and
896x{8,16,32,64} are AOT-instantiated, a fleet process hosts a
`--k3-ranks start..end` slice of the world, and the MegaMoE symmetric slabs
become `CU_MEM_HANDLE_TYPE_FABRIC` allocations whose 64-byte handles are
exchanged once over a TCP bootstrap (`--k3-rendezvous`). After that one
handshake the host planes never talk again — the fused kernel pairs all
ranks itself over the rack-wide NVLink domain (NVL72 + IMEX). Verified e2e:
pruned-224 EP16 over 4 GB300 trays is **byte-identical** to the single-tray
EP4 baseline on every greedy smoke prompt, with 4 concurrent requests across
the 4 endpoints unperturbed; full-model 896 EP16 serves coherent text.

Last touched: 2026-08

## Why multi-node is the gate, not a scaling nicety

The full 896-expert checkpoint's MXFP4 routed experts are ~1.4 TB — no
single-tray shape exists. EP8 = 112 experts/rank over 2 trays, EP16 = 56
over 4 (the shard shape the pruned-224 EP4 bring-up already certified),
EP32 = 28 over 8, EP64 = 14 over 16. A full-model EP16 rank is
memory-isomorphic to a pruned EP4 rank (~189 GiB), so everything the
single-tray bring-up proved carries over per rank.

## Transport: fabric handles, same kernel

The fused MegaMoE kernel addresses peer slabs by base pointer + layout
arithmetic; it cannot tell how a pointer became dereferenceable. In-process
that is peer access + pool grants; across processes it is:

- **Allocation** (`k3_mega_fabric.cu`): a fleet rank's slab comes from
  `cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)` on its own device, mapped into
  the process and access-granted to every local device, zeroed, exported.
- **Exchange** (`executor/ep.rs`): the process hosting rank 0 binds the
  rendezvous port; every process sends one hello carrying its ranks'
  `(slab bytes, fabric handle)` records; the root answers every held
  connection with the completed world table once all `ep_size` ranks are
  in. Modeled on GLM5.2's `rendezvous.rs`, with one difference: the payload
  is per-rank (handles), not one shared id, so the root collects before it
  answers and connections stay open until the world completes. Timeouts are
  weight-load bounds (1 h), not IO bounds.
- **Import**: each process imports every handle it does not host
  (`cuMemImportFromShareableHandle` + map + grant all local devices), once,
  process-wide, on the first scheduler thread to step.

Free-running ranks are unchanged: no coordinator, no per-step host
protocol, an idle rank pads. Fail-stop is fleet-wide by construction —
fabric handles do not survive a process, so a lost rank means relaunching
every process (the surviving ranks die on the kernel's 60 s device-barrier
timeout).

Requirements: one NVLink domain across the machines (NVL72) and the
`nvidia-imex` daemon on every node with the fleet's nodes in one IMEX
domain. The preflight (`k3_mega_fabric_supported`) turns a missing IMEX
into a named launch error instead of an opaque `cuMemCreate` failure.

### Gotcha: the root binds `0.0.0.0`, deliberately

`/etc/hosts` on these trays maps a machine's own hostname to `127.0.1.1`
(Debian convention). Binding the rendezvous *hostname* would put the
listener on loopback while every peer dials the fabric address — peers
retry "connection refused" for an hour and the root sits silently in
`accept`. `serve_bootstrap` therefore binds only the port, on all
interfaces, and logs `bootstrap root listening` so a stuck fleet is
diagnosable from the root's log.

## Launching a fleet

One process per tray, `RANKS_PER_HOST` (= GPUs, 4) ranks each:

```bash
HOSTS="pod4-gb300-3-tray04-f3 pod4-gb300-3-tray05-f3 pod4-gb300-3-tray06-f3 pod4-gb300-3-tray07-f3" \
MODEL_PATH=/mnt/shared/weights/kimi-k3 \
scripts/k3_ep_fleet.sh start     # also: stop | status | logs | smoke
```

which runs, on the i-th host:

```bash
pegainfer --model-path ... --port 8300 \
  --k3-ep-size 16 --k3-ranks $((i*4))..$(((i+1)*4)) \
  --k3-rendezvous <host0>:19300
```

Each process serves its own `/v1/completions` with one scheduler partition
per local rank; front the fleet with a router for real traffic. The
in-process worlds (`--k3-ep-size 1|4`, no `--k3-ranks`) are untouched.

## Verification

- `pegainfer-k3` lib tests cover the bootstrap wire protocol (localhost
  root+peer round trip, mismatch refusal) and the CLI shapes.
- Cross-node fabric transport was proven standalone before any engine code:
  a driver-API-only test exported a fabric handle on one tray and imported,
  read and wrote it from another (2 MiB granularity, IMEX channel0).
- **Pruned-224 EP16 over trays 04-07** (2026-08-18): the 16-rank bootstrap
  paired in 12 s after the last load, and all four greedy smoke prompts came
  back **byte-identical** to the fixed single-tray EP4 baseline. Four
  concurrent requests, one per tray endpoint, returned that same text — a
  rank's peers' traffic moves nothing.
- **Full 896-expert model EP16** (56 experts/rank, 189.9 GiB/rank): coherent
  greedy text over the same fleet ("founded in 753 BC by Romulus and
  Remus…"), ~52 ms/token single-stream (128 tokens in 6.7 s wall) — in the
  same band as the single-tray EP4 decode profile, so the fabric hop is not
  the step's constraint.
- Full-depth text was garbage on *every* K3 serve shape until the attnres
  snapshot-slab stride fix that this work flushed out — see the bring-up
  doc's "attnres stride" section. The EP16-vs-EP4 byte equality above is on
  the fixed engine.

## Slab cost per rank (protocol max 4224 rows)

| world | slab/rank |
|---|---|
| 224x4 | 1.6 GiB |
| 224x16 | 4.2 GiB |
| 896x16 | 5.0 GiB |
| 896x32 | 9.4 GiB |
| 896x64 | 16.1 GiB |

**Next**: TP x over the fleet (serving-roadmap item 4), varlen multi-prompt
prefill chunks, router integration for the fleet endpoints.
