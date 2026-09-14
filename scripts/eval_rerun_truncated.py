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
from eval_mc import (first_capital, mmlupro_extract, repair_jsonl_tail,
                     sg_extract_labels, sg_extract_content, STOP_MAP)
import httpx


def extract(bench, item, text):
    if bench == 'ceval':
        return first_capital(text)
    if bench == 'mmlu_pro':
        return mmlupro_extract(text).lower()
    if bench == 'mmlu_redux':
        # Same marker-aware extractor as eval_mc.py: a first-anywhere [ABCD]
        # search scores the "A" of "Answer".
        return sg_extract_labels(text, 'ABCD') or ''
    pred = sg_extract_labels(text)
    if pred is None and item.get('options'):
        content = sg_extract_content(text, item['options'])
        if content is not None:
            try:
                pred = chr(item['options'].index(content) + 65)
            except ValueError:
                pred = None
    return pred or ''


async def run(base_url, model, temperature, rows, bench, max_tokens, concurrency, timeout,
              partial_path):
    sem = asyncio.Semaphore(concurrency)
    # Recover rows from a killed previous attempt before touching the
    # checkpoint: the partial JSONL is appended to, never truncated.
    out_rows = {}
    if partial_path.exists():
        repair_jsonl_tail(partial_path, 'rerun')
        for line in partial_path.read_text(encoding='utf-8').splitlines():
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue  # defensive: repair_jsonl_tail cleaned the file
            if prev.get('output') and not prev.get('rerun_failed'):
                out_rows[prev['idx']] = prev
        if out_rows:
            print(f'[rerun] recovered {len(out_rows)} completed rows from {partial_path.name}',
                  flush=True)
    todo = [r for r in rows if r['idx'] not in out_rows]
    done = 0
    errors = 0
    with partial_path.open('a', encoding='utf-8') as partial:
        if not todo:
            return [out_rows[r['idx']] for r in rows], errors
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            async def one(row):
                async with sem:
                    payload = {'model': model, 'max_tokens': max_tokens,
                               'temperature': temperature, 'top_p': 1.0,
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
                                row.pop('rerun_failed', None)  # success clears the stale marker
                                row['output'] = msg.get('content') or ''
                                row['reasoning'] = msg.get('reasoning') or ''
                                row['completion_tokens'] = usage.get('completion_tokens') or 0
                                row['finish_reason'] = data['choices'][0].get('finish_reason') or ''
                                # Score before persisting: the checkpoint must
                                # never hold new text next to the old verdict
                                # (the final merge rescores too, but a killed
                                # run leaves this partial file as the record).
                                row['pred'] = extract(bench, row, row['output'])
                                row['correct'] = (row['pred'].lower()
                                                  == row['gold'].lower())
                                return row
                            err = f'HTTP {r.status_code} {r.text[:200]!r}'
                        except Exception as e:  # noqa: BLE001
                            err = f'{type(e).__name__}: {e!s}'
                        await asyncio.sleep(min(2 ** attempt, 30))
                    # Keep the row explicitly failed — one dead request must not
                    # discard the rest of the batch's (expensive) completions.
                    row = dict(row)
                    row['reasoning'] = f'__ERROR__ {err}'
                    row['rerun_failed'] = True
                    return row

            for fut in asyncio.as_completed([one(r) for r in todo]):
                row = await fut
                out_rows[row['idx']] = row
                errors += bool(row.get('rerun_failed'))
                partial.write(json.dumps(row, ensure_ascii=False) + '\n')
                partial.flush()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f'[rerun] {done}/{len(todo)} new rows '
                          f'({len(out_rows)}/{len(rows)} total)', flush=True)
            return [out_rows[r['idx']] for r in rows], errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmark')
    ap.add_argument('--base-url', default='http://127.0.0.1:18082/v1')
    ap.add_argument('--model', default=None,
                    help='defaults to the model recorded in the initial run summary; '
                         'an explicit value must match it — mixing configurations is refused')
    ap.add_argument('--out-dir', default='results/qwen35-27b-tp2-eval')
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--concurrency', type=int, default=48)
    ap.add_argument('--timeout', type=float, default=7200.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    # Resume from the merged state when a previous rerun already fixed some
    # rows: regenerating them would burn the large-budget completions again
    # and could turn a previously successful row into a failure. The merged
    # file must provably belong to the current base samples (a fresh
    # evaluation in this directory would have been overwritten by eval_mc
    # but the stale merged file could linger otherwise) — validate the
    # length and per-row prompt linkage before trusting it.
    base_samples = json.loads((out / f'{args.benchmark}_samples.json').read_text())
    merged_path = out / f'{args.benchmark}_samples_merged.json'
    if merged_path.exists():
        candidate = json.loads(merged_path.read_text())
        linked = (len(candidate) == len(base_samples)
                  and all(c.get('prompt') == b.get('prompt')
                          for c, b in zip(candidate, base_samples)))
        if not linked:
            sys.exit(f'refusing to resume from {merged_path}: it does not match the '
                     'current base samples (stale rerun data from an older run). '
                     'Re-run eval_mc.py (it now clears stale rerun artifacts) or remove the file.')
        samples = candidate
        print(f'{args.benchmark}: resuming from {merged_path.name}', flush=True)
    else:
        samples = base_samples
    summary = json.loads((out / f'{args.benchmark}_summary.json').read_text())

    model = args.model or summary['model']
    if model != summary['model']:
        sys.exit(f"refusing to mix configurations: --model {model!r} does not match the "
                 f"initial run's model {summary['model']!r} in "
                 f"{out / f'{args.benchmark}_summary.json'}")
    temperature = summary.get('temperature', 0.0)

    bad = [s for s in samples if not s['output']]
    print(f"{args.benchmark}: {len(samples)} total, {len(bad)} to re-run with "
          f"max_tokens={args.max_tokens} (model={model}, temperature={temperature})", flush=True)
    if not bad:
        return
    fixed, rerun_errors = asyncio.run(run(args.base_url, model, temperature, bad,
                                          args.benchmark, args.max_tokens,
                                          args.concurrency, args.timeout,
                                          out / f'{args.benchmark}_rerun.partial.jsonl'))
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
    # Recompute the failure state from the merged rows; keep the initial
    # run's counts under explicit initial_* fields instead of letting a
    # stale incomplete flag stand after every failed row got fixed. A retry
    # after a partially-failed rerun already recorded initial_api_errors —
    # keep that history, only refresh the current failure count.
    initial_errors = summary.get('initial_api_errors')
    if initial_errors is None:
        initial_errors = summary.pop('api_errors', 0)
    else:
        summary.pop('api_errors', None)
    summary.pop('incomplete', None)
    summary.update({
        'acc_merged': acc,
        'rerun_model': model,
        'rerun_temperature': temperature,
        'rerun_max_tokens': args.max_tokens,
        'initial_api_errors': initial_errors,
        'api_errors': rerun_errors,
        'rerun_errors': rerun_errors,
        'rerun_n': len(bad),
        'still_truncated': n_trunc_left,
        'completion_tokens_total_merged': sum(m.get('completion_tokens', 0) for m in merged),
    })
    if rerun_errors:
        summary['incomplete'] = True
    (out / f'{args.benchmark}_samples_merged.json').write_text(
        json.dumps(merged, ensure_ascii=False, indent=1))
    (out / f'{args.benchmark}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ('benchmark', 'acc', 'acc_merged',
                                              'rerun_n', 'rerun_errors', 'still_truncated')},
                     ensure_ascii=False))
    if rerun_errors:
        print(f'[rerun] {rerun_errors}/{len(bad)} rows still failing after retries — '
              'progress is on disk (merged output keeps them empty); fix the service and re-run.',
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
