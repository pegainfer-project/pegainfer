import argparse
import json
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
# Tensors whose values the tests compare, so their bytes are read too.
VALUES = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
)


def fetch(name: str, first: int | None = None, last: int | None = None) -> bytes:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    request = urllib.request.Request(url)
    if first is None:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    request.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(request, timeout=120) as response:
        # A server may ignore Range and send the whole multi-GB shard with a 200.
        served = response.headers.get("Content-Range", "")
        if response.status != 206 or not served.startswith(f"bytes {first}-{last}/"):
            raise SystemExit(f"{name}: bytes {first}-{last} not served as a range")
        return response.read(last - first + 1)


def shard_header(shard: str) -> tuple[int, dict]:
    """A shard's header and the file offset its tensor data starts at."""
    length = struct.unpack("<Q", fetch(shard, 0, 7))[0]
    header = json.loads(fetch(shard, 8, 8 + length - 1))
    header.pop("__metadata__", None)
    return 8 + length, header


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot every shard header of the frozen Qwen3.8-Flash-Next."
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    index = json.loads(fetch("model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    with ThreadPoolExecutor(8) as pool:
        headers = dict(zip(shards, pool.map(shard_header, shards)))
    tensors, values = {}, {}
    for shard, (data_start, header) in headers.items():
        for name, info in header.items():
            tensors[name] = [info["dtype"], info["shape"]]
            if name in VALUES:
                first, end = (data_start + offset for offset in info["data_offsets"])
                raw = fetch(shard, first, end - 1)
                values[name] = list(struct.unpack(f"<{len(raw) // 8}q", raw))
    if set(tensors) != set(index["weight_map"]):
        raise SystemExit("the shard headers and the index name different tensors")

    # One tensor per line keeps a future revision's diff readable.
    rows = [f"    {json.dumps(n)}: {json.dumps(tensors[n])}" for n in sorted(tensors)]
    args.out.write_text(
        "{\n"
        f'  "revision": "{REVISION}",\n'
        f'  "values": {json.dumps(dict(sorted(values.items())))},\n'
        '  "tensors": {\n' + ",\n".join(rows) + "\n  }\n}\n"
    )
    print(f"wrote {args.out}: {len(tensors)} tensors from {len(shards)} shards")


if __name__ == "__main__":
    main()
