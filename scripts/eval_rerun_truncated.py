#!/usr/bin/env python3
"""Re-run items truncated by the thinking-token budget with a larger cap.

Reads <out-dir>/<bench>_samples.json, finds rows with empty final output
(finish_reason == 'length' / 'stop' with empty content — mid-thinking ends),
re-generates them against the server with a bigger --max-tokens, merges the
new outputs back, and writes:
  <out-dir>/<bench>_samples_merged.json
  <out-dir>/<bench>_summary.json   (updated acc, with truncation stats)
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_mc import (first_capital, mmlupro_extract, MMLUREDUX_RE,
                     sg_extract_labels, sg_extract_content, STOP_MAP)
import httpx


def extract(bench, item, text):
    if bench == 'ceval':
        return first_capital(text)
    if bench == 'mmlu_pro':
        return mmlupro_extract(text).lower()
    if bench == 'mmlu_redux':
        m = MMLUREDUX_RE.search(text)
        return m.group(1) if m else ''
    pred = sg_extract_labels(text)
    if pred is None and item.get('options'):
        content = sg_extract_content(text, item['options'])
        if content is not None:
            try:
                pred = chr(item['options'].index(content) + 65)
            except ValueError:
                pred = None
    return pred or ''


async def run(base_url, model, rows, bench, max_tokens, concurrency, timeout):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        async def one(row):
            async with sem:
                payload = {'model': model, 'max_tokens': max_tokens,
                           'temperature': 0.0, 'top_p': 1.0,
                           'messages': [{'role': 'user', 'content': row['prompt']}]}
                if STOP_MAP.get(bench):
                    payload['stop'] = STOP_MAP[bench]
                for attempt in range(6):
                    try:
                        r = await client.post(f'{base_url}/chat/completions', json=payload)
                        if r.status_code == 200:
                            data = r.json()
                            msg = data['choices'][0]['message']
                            usage = data.get('usage') or {}
                            row = dict(row)
                            row['output'] = msg.get('content') or ''
                            row['reasoning'] = msg.get('reasoning') or ''
                            row['completion_tokens'] = usage.get('completion_tokens') or 0
                            row['finish_reason'] = data['choices'][0].get('finish_reason') or ''
                            return row
                        err = f'HTTP {r.status_code} {r.text[:200]!r}'
                    except Exception as e:  # noqa: BLE001
                        err = f'{type(e).__name__}: {e!s}'
                    await asyncio.sleep(min(2 ** attempt, 30))
                raise RuntimeError(f"rerun failed: {err}")

        return await asyncio.gather(*[one(r) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmark')
    ap.add_argument('--base-url', default='http://127.0.0.1:18082/v1')
    ap.add_argument('--model', default='qwen35-27b-tp2')
    ap.add_argument('--out-dir', default='results/qwen35-27b-tp2-eval')
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--concurrency', type=int, default=48)
    ap.add_argument('--timeout', type=float, default=7200.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    samples = json.loads((out / f'{args.benchmark}_samples.json').read_text())
    summary = json.loads((out / f'{args.benchmark}_summary.json').read_text())

    bad = [s for s in samples if not s['output']]
    print(f"{args.benchmark}: {len(samples)} total, {len(bad)} to re-run with "
          f"max_tokens={args.max_tokens}", flush=True)
    if not bad:
        return
    fixed = asyncio.run(run(args.base_url, args.model, bad, args.benchmark,
                            args.max_tokens, args.concurrency, args.timeout))
    fixed_by_idx = {f['idx']: f for f in fixed}
    merged = []
    for s in samples:
        m = fixed_by_idx.get(s['idx'], s)
        m['pred'] = extract(args.benchmark, m, m['output'])
        m['correct'] = m['pred'].lower() == m['gold'].lower()
        merged.append(m)
    n_ok = sum(1 for m in merged if m['correct'])
    n_trunc_left = sum(1 for m in merged if not m['output'])
    acc = round(100.0 * n_ok / len(merged), 2)
    summary.update({
        'acc_merged': acc,
        'rerun_max_tokens': args.max_tokens,
        'rerun_n': len(bad),
        'still_truncated': n_trunc_left,
        'completion_tokens_total_merged': sum(m.get('completion_tokens', 0) for m in merged),
    })
    (out / f'{args.benchmark}_samples_merged.json').write_text(
        json.dumps(merged, ensure_ascii=False, indent=1))
    (out / f'{args.benchmark}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ('benchmark', 'acc', 'acc_merged',
                                              'rerun_n', 'still_truncated')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
