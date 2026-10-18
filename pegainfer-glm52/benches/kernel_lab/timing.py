"""torch.cuda.Event timing: warmup + rounds x inner; median/p50/p99/mean.

The SM clock is RECORDED (via nvidia-smi), never enforced — locked clocks are
the bench owner's discipline. torch is imported lazily (bench path only).
"""
from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass


@dataclass
class BenchStats:
    samples_us: list[float]  # one per round, inner-averaged
    clocks_sm_mhz: int | None = None

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples_us)

    @property
    def p50_us(self) -> float:
        return self.percentile(50)

    @property
    def p99_us(self) -> float:
        return self.percentile(99)

    def percentile(self, pct: int) -> float:
        """Nearest-rank percentile (ceil), robust for small round counts."""
        s = sorted(self.samples_us)
        if not s:
            raise ValueError("no samples")
        rank = -(-pct * len(s) // 100)
        return s[min(rank, len(s)) - 1]

    def summary(self) -> dict:
        return {
            "median_us": round(self.median_us, 3),
            "p50_us": round(self.p50_us, 3),
            "p99_us": round(self.p99_us, 3),
            "mean_us": round(self.mean_us, 3),
            "rounds": len(self.samples_us),
        }


def _measure_once(call, inner: int) -> float:
    from kernel_lab.loader import require_torch

    torch = require_torch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / inner  # ms -> us


def bench(call, warmup: int, rounds: int, inner: int) -> BenchStats:
    from kernel_lab.loader import require_torch

    torch = require_torch()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = [_measure_once(call, inner) for _ in range(rounds)]
    return BenchStats(samples_us=samples, clocks_sm_mhz=sm_clocks_mhz())


def tensor_traffic_bytes(tensors: dict, skip: tuple[str, ...] = ("workspace",)) -> int:
    """Bytes one kernel call moves = sum of CUDA tensor sizes, minus scratch."""
    total = 0
    for name, t in tensors.items():
        if name in skip or not getattr(t, "is_cuda", False):
            continue
        total += t.numel() * t.element_size()
    return total


def l2_rotate_copies(tensors: dict, l2_bytes: int | None = None, max_copies: int = 16) -> list[dict]:
    """Clone the tensors dict so that one round-robin sweep moves >= 3x L2.

    Cold-L2 protocol after quack's _bench_cuda_graph_l2_rotate: reuse of one
    buffer set keeps it resident in L2 and turns "HBM bandwidth" numbers into
    L2 bandwidth (>100% of peak tells the lie). Rotating N clones — N chosen
    so N * bytes-per-call >= 3 * L2 — guarantees every timed call misses.
    First element is the original dict (aliases, zero cost).
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    if l2_bytes is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    per_call = max(tensor_traffic_bytes(tensors), 1)
    n = min(max_copies, max(2, -(-3 * l2_bytes // per_call)))
    copies = [tensors]
    for _ in range(n - 1):
        copies.append(
            {k: (t.clone() if getattr(t, "is_cuda", False) else t) for k, t in tensors.items()}
        )
    return copies


def _measure_once_rotated(call_for, copies: list[dict], inner: int) -> float:
    from kernel_lab.loader import require_torch

    torch = require_torch()
    n = len(copies)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(inner):
        call_for(copies[i % n])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / inner  # ms -> us


def bench_rotated(call_for, copies: list[dict], warmup: int, rounds: int, inner: int) -> BenchStats:
    """Same as bench() but rotates through `copies` (see l2_rotate_copies).

    Every copy is primed once before timing so per-pointer JIT/wrapper caches
    (CuTe DSL from_dlpack packing) are warm out of the timed window.
    """
    from kernel_lab.loader import require_torch

    torch = require_torch()
    for c in copies:
        call_for(c)
    torch.cuda.synchronize()
    n = len(copies)
    for i in range(warmup):
        call_for(copies[i % n])
    torch.cuda.synchronize()
    samples = [_measure_once_rotated(call_for, copies, inner) for _ in range(rounds)]
    return BenchStats(samples_us=samples, clocks_sm_mhz=sm_clocks_mhz())


def interleaved(call_a, call_b, warmup: int, rounds: int, inner: int) -> tuple[BenchStats, BenchStats]:
    """Same-session A/B: alternate one round of A with one of B (swapping the
    order every round) so clock/thermal drift hits both sides equally."""
    from kernel_lab.loader import require_torch

    torch = require_torch()
    for _ in range(warmup):
        call_a()
        call_b()
    torch.cuda.synchronize()
    samples_a: list[float] = []
    samples_b: list[float] = []
    for r in range(rounds):
        if r % 2 == 0:
            samples_a.append(_measure_once(call_a, inner))
            samples_b.append(_measure_once(call_b, inner))
        else:
            samples_b.append(_measure_once(call_b, inner))
            samples_a.append(_measure_once(call_a, inner))
    clocks = sm_clocks_mhz()
    return BenchStats(samples_a, clocks), BenchStats(samples_b, clocks)


def sm_clocks_mhz() -> int | None:
    """Record-only SM clock probe; None when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
