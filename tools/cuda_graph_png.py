#!/usr/bin/env python3
"""Render a CUDA-graph .dot dump to a folded, human-browsable PNG.

One renderer for both dot dialects we produce, so every engineer gets the
same image from the same input:

  - pegainfer detailed dot (from `pegainfer --dump-graph-png`, the .dot sidecar)
  - CUDA `cudaGraphDebugDotPrint` verbose dot (e.g. dumped from vLLM)

Repeated per-layer kernel blocks are detected by label signature and folded
into one representative block plus an explicit fold marker, because an
unfolded 36-layer chain exceeds Graphviz's ~32767px raster ceiling
(docs/models/qwen3/cuda-graph-png.md). Rendering is pinned to the Cairo PNG
backend, with DPI auto-scaled down for graphs that stay tall after folding;
a missing renderer is a hard error, not a degraded image.

Usage: uv run python tools/cuda_graph_png.py INPUT.dot [-o OUT.png] [--title TITLE] [--no-fold]

Needs `dot` (graphviz with cairo) and `c++filt` on PATH.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

def pick_dpi(kept_nodes):
    """Graphviz PNG rasters cap at 32767px; a folded chain runs ~105px/node
    at 192 DPI, so tall graphs must drop DPI to stay renderable."""
    if kept_nodes <= 120:
        return 192
    if kept_nodes <= 600:
        return 96
    return 72

# category -> fill color; category is derived from the kernel's provenance
COLORS = {
    "cublas": "#bfd9f2",
    "flash-attn": "#f9d9b0",
    "flashinfer": "#f9d9b0",
    "triton": "#c8e6c9",
    "vllm": "#e1cff0",
    "pegainfer": "#e1cff0",
    "memop": "#e0e0e0",
    "other": "#f5f5f5",
}


@dataclass
class Node:
    nid: str
    order: int  # execution order (topological)
    label: str  # compact two-line display label
    signature: str  # full label (name + launch dims); coarser keys alias
    # across distinct per-layer kernels and produce false periods
    category: str


def demangle_all(symbols):
    raw = [s for s in symbols if s.startswith("_Z")]
    if not raw:
        return {}
    out = subprocess.run(
        ["c++filt"], input="\n".join(raw), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return dict(zip(raw, out))


def compact_name(demangled):
    """Map a demangled kernel name to a short label + provenance category."""
    d = demangled
    if "internal::gemvx::kernel" in d or "cublas" in d:
        return "cuBLAS GEMV", "cublas"
    m = re.match(r"(?:void )?flash::(\w+)", d)
    if m:
        return f"flash {m.group(1).removesuffix('_kernel')}", "flash-attn"
    m = re.match(r"(?:void )?flashinfer::(?:\w+::)*(\w+)", d)
    if m:
        return f"flashinfer {m.group(1).removesuffix('Kernel')}", "flashinfer"
    m = re.match(r"(?:void )?vllm::(?:\w+::)*(\w+)", d)
    if m:
        return f"vllm {m.group(1).removesuffix('_kernel')}", "vllm"
    m = re.match(r"(?:void )?pegainfer::(?:\w+::)*(\w+)", d)
    if m:
        return f"pegainfer {m.group(1).removesuffix('Kernel')}", "pegainfer"
    if d.startswith("triton"):
        return d.rstrip("_0123456789") or d, "triton"
    base = re.sub(r"<.*", "", d.split("(")[0]).strip().split()[-1]
    return base[:56], "other"


def parse_dim3(text):
    """'{1,16,8}' or '1536' -> '(1,16,8)' / '(1536,1,1)'."""
    text = text.strip()
    if text.startswith("{"):
        return "(" + text.strip("{}") + ")"
    return f"({text},1,1)"


def split_launch(launch):
    """Split 'grid,block,smem' where grid/block may be '{x,y,z}'."""
    parts, depth, cur = [], 0, ""
    for ch in launch:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            depth += ch == "{"
            depth -= ch == "}"
            cur += ch
    parts.append(cur)
    return parts


def parse_cuda_debug_dot(text):
    """cudaGraphDebugDotPrint verbose format: record nodes with (topoId: N)."""
    node_re = re.compile(r'"(graph_\d+_node_\d+)"\[.*?label="\{(.*?)\}"\];', re.S)
    kernel_re = re.compile(r"\(topoId: \d+\) \| (.+?)\\<\\<\\<(.+?)\\>\\>\\>")
    topo_re = re.compile(r"\(topoId: (\d+)\)")

    raw_nodes = []
    for nid, body in node_re.findall(text):
        kind = body.strip().lstrip("{").strip().split("\n")[0].split("|")[0].strip()
        topo = int(topo_re.search(body).group(1))
        km = kernel_re.search(body)
        raw_nodes.append((nid, -topo, kind, km.group(1) if km else "", km.group(2) if km else ""))

    names = demangle_all(sym for _, _, _, sym, _ in raw_nodes)
    nodes = {}
    for nid, order, kind, sym, launch in raw_nodes:
        if sym:
            short, cat = compact_name(names.get(sym, sym))
            grid, block, smem = split_launch(launch)
            detail = f"grid={parse_dim3(grid)} block={parse_dim3(block)} smem={smem}"
        else:
            short, cat, detail = kind.lower(), "memop", ""
        label = f"{short}\\n{detail}".rstrip("\\n")
        nodes[nid] = Node(nid, order, label, label, cat)

    edges = re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', text)
    return nodes, edges


def parse_pegainfer_dot(text):
    """pegainfer detailed sidecar: nN [label="id=..\\ntype=..\\nname=..\\n.."]."""
    # edges also carry [label="from_port=..."]; anchor on the id= body so an
    # edge's target node is never re-parsed as a node definition
    node_re = re.compile(r'^\s*(n\d+) \[label="(id=.*?)"\];?', re.S | re.M)
    nodes, order = {}, 0
    for nid, body in node_re.findall(text):
        fields = dict(
            f.split("=", 1) for f in body.split("\\n") if "=" in f
        )
        kind = fields.get("type", "other")
        if kind == "kernel":
            short, cat = compact_name(fields["name"])
            detail = (
                f"grid={fields.get('grid', '?')} block={fields.get('block', '?')} "
                f"smem={fields.get('dynamic_shared_mem_bytes', '?')}"
            )
            label = f"{short}\\n{detail}"
        else:
            short, cat, label = kind, "memop", kind
        nodes[nid] = Node(nid, order, label, label, cat)
        order += 1
    edges = re.findall(r"(n\d+)\s*->\s*(n\d+)", text)
    return nodes, edges


def detect_folds(signatures, max_period=512, min_cover=12):
    """Scan for every maximal periodic run and return [(start, period, repeats)].

    Real decode graphs are multi-phase (a few dense layers, then a long MoE
    run, then an MTP tail), so a single dominant period misses most of the
    graph; each phase gets its own fold."""
    interned = {}
    ids = [interned.setdefault(s, len(interned)) for s in signatures]
    n = len(ids)
    folds, i = [], 0
    while i < n:
        best = None
        for period in range(4, max_period + 1):
            if i + 2 * period > n:
                break
            if ids[i : i + period] != ids[i + period : i + 2 * period]:
                continue
            reps = 2
            while (
                i + (reps + 1) * period <= n
                and ids[i + reps * period : i + (reps + 1) * period] == ids[i : i + period]
            ):
                reps += 1
            if best is None or period * reps > best[1] * best[2]:
                best = (i, period, reps)
        if best and best[1] * best[2] >= min_cover:
            folds.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return folds


LANE_COLORS = ["#c0392b", "#2471a3", "#7d3c98", "#b9770e", "#148f77", "#5d6d7e"]
LANE_MAX = 10  # longer divergence lanes collapse into one summary node


def compact_ranges(indices):
    """[0, 1, 2, 17] -> 'L0-2,L17'."""
    runs, start = [], None
    for i, v in enumerate(indices):
        if start is None:
            start = v
        if i + 1 == len(indices) or indices[i + 1] != v + 1:
            runs.append(f"L{start}" if v == start else f"L{start}-{v}")
            start = None
    return ",".join(runs)


def find_layer_family(sigs, fold):
    """Segment the sequence into instances of the dominant layer block.

    The fold's phase is arbitrary, so re-anchor the canonical block on a
    signature that occurs exactly once per block; every occurrence of that
    anchor then marks an instance start. Near-matches (compiler-specialized
    first layers, sequence-shifted last layers) join the family by similarity
    instead of exact equality."""
    start, period, reps = fold
    canon = sigs[start : start + period]
    counts = {}
    for s in canon:
        counts[s] = counts.get(s, 0) + 1
    unique = [s for s in canon if counts[s] == 1 and sigs.count(s) >= reps]
    if not unique:
        return None
    anchor = min(unique, key=sigs.count)
    off = canon.index(anchor)
    canon = sigs[start + off : start + off + period]

    anchors = [i for i, s in enumerate(sigs) if s == anchor]
    instances = []
    for k, st in enumerate(anchors):
        en = anchors[k + 1] if k + 1 < len(anchors) else min(len(sigs), st + 2 * period)
        block = sigs[st:en]
        if SequenceMatcher(None, canon, block, autojunk=False).ratio() >= 0.6:
            instances.append((st, en))
    if len(instances) < 4:
        return None

    classes = {}  # block content -> [instance indices]
    for i, (st, en) in enumerate(instances):
        classes.setdefault(tuple(sigs[st:en]), []).append(i)
    class_list = sorted(classes.items(), key=lambda kv: -len(kv[1]))
    return {"instances": instances, "classes": class_list, "canon_period": period}


def emit_family_lanes(order, family):
    """Collect divergence lanes: every non-majority class contributes colored
    lane nodes plus stitch edges into the spine, labeled with its layers.

    Returns (lane_lines, stitch_edges, lane_node_count); stitch endpoints are
    raw spine nids, to be remapped after BPE grouping."""
    instances, class_list = family["instances"], family["classes"]
    spine_content, spine_members = class_list[0]
    spine_st, spine_en = instances[spine_members[0]]
    spine_ids = [n.nid for n in order[spine_st:spine_en]]

    lane_lines, stitches, lane_count = [], [], 0
    for ci, (content, members) in enumerate(class_list[1:]):
        color = LANE_COLORS[ci % len(LANE_COLORS)]
        ex_st, ex_en = instances[members[0]]
        exemplar = order[ex_st:ex_en]
        who = compact_ranges(members)
        first_stitch = True
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, list(spine_content), list(content), autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            lane = exemplar[j1:j2]
            label = f', label="{who}", fontcolor="{color}"' if first_stitch else ""
            first_stitch = False
            entry = spine_ids[i1 - 1] if i1 > 0 else None
            exit_ = spine_ids[i2] if i2 < len(spine_ids) else None
            if not lane:  # pure deletion: this class skips spine nodes i1..i2
                if entry and exit_:
                    stitches.append(f'  "{entry}" -> "{exit_}" [color="{color}"{label}, constraint=false];')
                continue
            if len(lane) > LANE_MAX:
                # a long lane is usually a sequence-shifted whole block; a
                # summary node keeps the picture browsable, the detail stays
                # in the input dot
                top = ", ".join(sorted({n.signature.split("\\n")[0] for n in lane})[:3])
                nid = f"lanesum_{ci}_{i1}"
                lane_lines.append(
                    f'  "{nid}" [label="{len(lane)} divergent nodes\\n({top}, ...)",'
                    f' style="dashed,rounded", color="{color}", fontcolor="{color}"];'
                )
                lane_ids = [nid]
                lane_count += 1
            else:
                for n in lane:
                    lane_lines.append(
                        f'  "{n.nid}" [label="{n.label}", fillcolor="{COLORS[n.category]}",'
                        f' color="{color}", penwidth=2];'
                    )
                lane_ids = [n.nid for n in lane]
                lane_count += len(lane)
            if entry:
                stitches.append(f'  "{entry}" -> "{lane_ids[0]}" [color="{color}"{label}];')
            if exit_:
                stitches.append(f'  "{lane_ids[-1]}" -> "{exit_}" [color="{color}", constraint=false];')
    return lane_lines, stitches, lane_count


def bpe_group(kept, min_pair=3, max_iter=80):
    """Fuse the kept chain by BPE-style pair merging on kernel signatures:
    the most frequent adjacent pair becomes one composite node per occurrence,
    iterated until no pair repeats min_pair times. Fixed idioms (quant->GEMM,
    the trtllm MoE chain) collapse without any model-specific knowledge.
    Pairs never merge across a fold gap (non-consecutive original order)."""
    groups = [[n] for n in kept]
    sym = [n.signature for n in kept]

    def adjacent(i):
        return groups[i][-1].order + 1 == groups[i + 1][0].order

    for _ in range(max_iter):
        counts = {}
        for i in range(len(sym) - 1):
            if adjacent(i):
                counts[(sym[i], sym[i + 1])] = counts.get((sym[i], sym[i + 1]), 0) + 1
        if not counts:
            break
        pair, cnt = max(counts.items(), key=lambda kv: kv[1])
        if cnt < min_pair:
            break
        merged = pair[0] + "\x00" + pair[1]
        out_g, out_s, i = [], [], 0
        while i < len(sym):
            if i + 1 < len(sym) and (sym[i], sym[i + 1]) == pair and adjacent(i):
                out_g.append(groups[i] + groups[i + 1])
                out_s.append(merged)
                i += 2
            else:
                out_g.append(groups[i])
                out_s.append(sym[i])
                i += 1
        groups, sym = out_g, out_s
    return groups


def group_label(members):
    """Composite label: kernel kinds in execution order with repeat counts."""
    if len(members) == 1:
        return members[0].label, members[0].category
    parts, last = [], None
    for n in members:
        name = n.signature.split("\\n")[0]
        if parts and parts[-1][0] == name:
            parts[-1][1] += 1
        else:
            parts.append([name, 1])
    shown = [f"{k} x{v}" if v > 1 else k for k, v in parts[:4]]
    if len(parts) > 4:
        shown.append("...")
    label = f"[{len(members)} kernels]\\n" + "\\n".join(shown)
    cat = next(
        (n.category for n in members if n.category not in ("other", "memop")),
        members[0].category,
    )
    return label, cat


def emit_folded_dot(nodes, edges, title, folds):
    order = sorted(nodes.values(), key=lambda x: x.order)
    sigs = [n.signature for n in order]

    family = None
    if folds:
        family = find_layer_family(sigs, max(folds, key=lambda f: f[1] * f[2]))

    removed = set()
    lane_lines, stitches, lane_count = [], [], 0
    spine_range = None
    notes = []
    if family:
        instances = family["instances"]
        for st, en in instances:
            removed.update(range(st, en))
        folds = [
            f for f in folds
            if not any(st < f[0] + f[1] * f[2] and f[0] < en for st, en in instances)
        ]
        lane_lines, stitches, lane_count = emit_family_lanes(order, family)
        spine_range = family["instances"][family["classes"][0][1][0]]
        notes.append(
            f"{len(instances)} layer blocks merged into 1"
            f" ({len(family['classes'])} variants)"
        )
    for start, period, reps in folds:
        removed.update(range(start + period, start + period * reps))
        notes.append(f"x{reps} blocks of {period} folded")

    # main chain = literals + spine, in original order, lanes excluded
    kept = [
        n for i, n in enumerate(order)
        if i not in removed or (spine_range and spine_range[0] <= i < spine_range[1])
    ]
    groups = bpe_group(kept)
    nid2gid = {}
    for g in groups:
        for n in g:
            nid2gid[n.nid] = g[0].nid
    if len(groups) < len(kept):
        notes.append(f"{len(kept)} -> {len(groups)} via idiom fusion")

    lines = [
        "digraph folded {",
        f'  graph [label="{title}\\n{len(nodes)} nodes / {len(edges)} edges'
        f'{"; " + "; ".join(notes) if notes else ""}", labelloc=t, fontsize=20,'
        ' fontname="Helvetica", ranksep=0.22, nodesep=0.18];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,'
        ' margin="0.1,0.04"];',
    ]
    for g in groups:
        label, cat = group_label(g)
        border = ', penwidth=1.6, color="#555555"' if len(g) > 1 else ""
        lines.append(f'  "{g[0].nid}" [label="{label}", fillcolor="{COLORS[cat]}"{border}];')
    lines.extend(lane_lines)

    def gid(nid):
        return nid2gid.get(nid, nid)

    lane_ids = {m.group(1) for line in lane_lines for m in [re.match(r'\s*"([^"]+)"', line)] if m}
    known = set(nid2gid) | lane_ids
    drawn = set()
    for a, b in edges:
        if a in known and b in known and gid(a) != gid(b):
            drawn.add(f'  "{gid(a)}" -> "{gid(b)}";')
    lines.extend(sorted(drawn))
    for line in stitches:
        lines.append(re.sub(r'"([^"]+)"', lambda m: f'"{gid(m.group(1))}"', line, count=2))
    for k, (start, period, reps) in enumerate(folds):
        prev = order[start + period - 1]
        lines.append(
            f'  "fold{k}" [label="... x{reps - 1} more identical blocks ...",'
            ' shape=box, style="dashed,rounded", fontsize=14];'
        )
        lines.append(f'  "{gid(prev.nid)}" -> "fold{k}";')
        after = start + period * reps
        if after < len(order):
            lines.append(f'  "fold{k}" -> "{gid(order[after].nid)}";')
    if family:
        # stitch the merged block to its neighbours when the spine exemplar is
        # not the instance the original edges connect to
        sp_st, sp_en = spine_range
        first_st = family["instances"][0][0]
        last_en = family["instances"][-1][1]
        edge_set = set(edges)
        prev = next((order[i] for i in range(first_st - 1, -1, -1) if order[i].nid in nid2gid), None)
        nxt = next((order[i] for i in range(last_en, len(order)) if order[i].nid in nid2gid), None)
        if prev and (prev.nid, order[sp_st].nid) not in edge_set:
            lines.append(f'  "{gid(prev.nid)}" -> "{gid(order[sp_st].nid)}" [style=dashed];')
        if nxt and (order[sp_en - 1].nid, nxt.nid) not in edge_set:
            lines.append(f'  "{gid(order[sp_en - 1].nid)}" -> "{gid(nxt.nid)}" [style=dashed];')
    lines.append("}")
    return "\n".join(lines), len(groups) + lane_count


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help=".dot from --dump-graph-png or cudaGraphDebugDotPrint")
    ap.add_argument("-o", "--output", type=Path, help="output PNG (default: INPUT with .png)")
    ap.add_argument("--title", help="graph title (default: input file name)")
    ap.add_argument("--no-fold", action="store_true", help="render all nodes unfolded")
    args = ap.parse_args()

    text = args.input.read_text()
    if "topoId" in text:
        nodes, edges = parse_cuda_debug_dot(text)
    elif "raw_symbol=" in text:
        nodes, edges = parse_pegainfer_dot(text)
    else:
        sys.exit("error: unrecognized dot dialect (expected cudaGraphDebugDotPrint or pegainfer detailed dot)")
    if not nodes:
        sys.exit("error: no graph nodes parsed")

    folds = []
    if not args.no_fold:
        ordered = sorted(nodes.values(), key=lambda x: x.order)
        folds = detect_folds([n.signature for n in ordered])
        if folds:
            for start, period, reps in folds:
                print(f"fold: {period} nodes/block x{reps} (start offset {start})")
        else:
            print("fold: no repeated block detected, rendering unfolded")

    out = args.output or args.input.with_suffix(".png")
    title = args.title or args.input.name
    folded, kept_nodes = emit_folded_dot(nodes, edges, title, folds)

    dpi = pick_dpi(kept_nodes)
    print(f"{kept_nodes} nodes after folding, rendering at {dpi} dpi")
    r = subprocess.run(
        ["dot", "-Tpng:cairo", f"-Gdpi={dpi}", "-o", str(out)],
        input=folded, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"error: graphviz cairo render failed: {r.stderr.strip()}")
    print(f"wrote {out} ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    main()
