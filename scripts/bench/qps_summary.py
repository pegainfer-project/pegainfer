#!/usr/bin/env python3
"""Print one summary line per `vllm bench serve` result JSON given on argv."""

import json
import re
import sys

for path in sorted(
    sys.argv[1:],
    key=lambda p: (
        p.split("/")[-1].split("-")[0],
        float(re.search(r"(?:qps|c)([0-9.]+)", p).group(1)),
    ),
):
    d = json.load(open(path))
    m = re.search(r"(?:qps|c)([0-9.]+)", path)
    name = path.split("/")[-1].split("-")[0]
    label = "qps" if "qps" in path else "c"
    val = m.group(1)
    print(
        f"{name:>12} {label}={val:>4} dur={d['duration']:6.1f}s "
        f"completed={d['completed']} req/s={d['request_throughput']:.2f} "
        f"out_tok/s={d['output_throughput']:7.1f} "
        f"ttft p50={d['median_ttft_ms']:7.1f} p99={d['p99_ttft_ms']:8.1f} "
        f"tpot p50={d['median_tpot_ms']:6.2f} p99={d['p99_tpot_ms']:7.2f} "
        f"itl p99={d['p99_itl_ms']:7.2f}"
    )
