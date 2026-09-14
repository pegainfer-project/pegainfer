#!/usr/bin/env python3
"""Multiple-choice benchmark runner for pegainfer's OpenAI-compatible chat API.

Recipes copied from the canonical harnesses so scores stay comparable:
  * C-Eval      -> opencompass ceval_gen (5-shot from dev split, "答案: "
                   completion, first-capital extraction)
  * SuperGPQA   -> opencompass supergpqa_gen (zero-shot prompt_format yaml,
                   extract_option_labels -> (A..J), content fallback)
  * MMLU-Pro    -> lm-eval mmlu_pro (5-shot CoT from validation split,
                   'answer is \\(?(X)\\)?' extraction)
  * MMLU-Redux  -> lm-eval mmlu_redux_generative (0-shot; marker-preferring
                   label extraction — a first-anywhere [ABCD] search mis-scores
                   "Answer: B" as the "A" of "Answer")

All four go through /v1/chat/completions with one user message; Qwen3.5 is a
thinking model so reasoning lands in `reasoning`, the final answer text in
`content` — extraction runs on `content` only.

Usage:
  python eval_mc.py ceval|supergpqa|mmlu_pro|mmlu_redux \
      --base-url http://127.0.0.1:18082/v1 --model qwen35-27b-tp2
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import datasets
import httpx

# ---------------------------------------------------------------- ceval
CEVAL_SUBJECT_CN = {
    'computer_network': '计算机网络', 'operating_system': '操作系统',
    'computer_architecture': '计算机组成', 'college_programming': '大学编程',
    'college_physics': '大学物理', 'college_chemistry': '大学化学',
    'advanced_mathematics': '高等数学', 'probability_and_statistics': '概率统计',
    'discrete_mathematics': '离散数学', 'electrical_engineer': '注册电气工程师',
    'metrology_engineer': '注册计量师', 'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理', 'high_school_chemistry': '高中化学',
    'high_school_biology': '高中生物', 'middle_school_mathematics': '初中数学',
    'middle_school_biology': '初中生物', 'middle_school_physics': '初中物理',
    'middle_school_chemistry': '初中化学', 'veterinary_medicine': '兽医学',
    'college_economics': '大学经济学', 'business_administration': '工商管理',
    'marxism': '马克思主义基本原理',
    'mao_zedong_thought': '毛泽东思想和中国特色社会主义理论体系概论',
    'education_science': '教育学', 'teacher_qualification': '教师资格',
    'high_school_politics': '高中政治', 'high_school_geography': '高中地理',
    'middle_school_politics': '初中政治', 'middle_school_geography': '初中地理',
    'modern_chinese_history': '近代史纲要',
    'ideological_and_moral_cultivation': '思想道德修养与法律基础',
    'logic': '逻辑学', 'law': '法学',
    'chinese_language_and_literature': '中国语言文学', 'art_studies': '艺术学',
    'professional_tour_guide': '导游资格', 'legal_professional': '法律职业资格',
    'high_school_chinese': '高中语文', 'high_school_history': '高中历史',
    'middle_school_history': '初中历史', 'civil_servant': '公务员',
    'sports_science': '体育学', 'plant_protection': '植物保护',
    'basic_medicine': '基础医学', 'clinical_medicine': '临床医学',
    'urban_and_rural_planner': '注册城乡规划师', 'accountant': '注册会计师',
    'fire_engineer': '注册消防工程师',
    'environmental_impact_assessment_engineer': '环境影响评价工程师',
    'tax_accountant': '税务师', 'physician': '医师资格',
}


def ceval_prompt(ch_name, q, shots):
    head = f'以下是中国关于{ch_name}考试的单项选择题，请选出其中的正确答案。\n'

    def block(item):
        return (f"{item['question']}\nA. {item['A']}\nB. {item['B']}\n"
                f"C. {item['C']}\nD. {item['D']}")

    examples = ''.join(f"{block(s)}\n答案: {s['answer']}\n" for s in shots)
    return head + examples + block(q) + '\n答案: '


def first_capital(text):
    for ch in text:
        if ch.isupper():
            return ch
    return ''


# ---------------------------------------------------------------- mmlu_pro
MMLUPRO_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
MMLUPRO_RE = re.compile(r'answer is \(?([ABCDEFGHIJ])\)?')


def mmlupro_format(example, include_answer):
    prompt = 'Question:\n' + example['question'] + '\nOptions:\n'
    for i, opt in enumerate(example['options'][:len(MMLUPRO_LETTERS)]):
        prompt += f"{MMLUPRO_LETTERS[i]}. {opt.strip()}\n"
    if include_answer:
        cot = example['cot_content'].replace("A: Let's think step by step.",
                                             "Answer: Let's think step by step.")
        return prompt + cot + '\n\n'
    return prompt + "Answer: Let's think step by step."


def mmlupro_extract(text):
    m = MMLUPRO_RE.search(text)
    return m.group(1) if m else ''


# ---------------------------------------------------------------- mmlu-redux
# Extraction reuses the SuperGPQA marker-preferring patterns with the MMLU
# letter range: a first-anywhere `[ABCD]` search scores "Answer: B" as the
# "A" of "Answer", silently flipping correct answers (codex review on #946).


# ---------------------------------------------------------------- supergpqa
SG_PROMPT = ("Answer the following multiple choice question. There is only one "
             "correct answer. The last line of your response should be in the "
             "format 'Answer: $LETTER' (without quotes), where LETTER is one of "
             "A, B, C, D, E, F, G, H, I, or J.\n\n{}")


def sg_build_questions(item):
    opts = '\n'.join(f'{chr(65 + i)}) {o}' for i, o in enumerate(item['options']))
    return item['question'] + '\n' + opts


def sg_label_patterns(letters):
    return [
        rf'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{letters}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        rf'(?i:Answer)[\*\s]*:\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{letters}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        rf'^[^\w\r\n]*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*([{letters}])(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]


def sg_extract_labels(text, letters='ABCDEFGHIJ'):
    if not isinstance(text, str):
        return None
    text = text.rstrip()
    last_line = text.split('\n')[-1]
    pats = sg_label_patterns(letters)
    for src in (last_line, text):
        for p in pats:
            try:
                m = re.search(p, src, re.IGNORECASE)
            except Exception:
                m = None
            if m:
                return m.group(1)
    return None


def sg_extract_content(text, options_content):
    if not isinstance(text, str) or not isinstance(options_content, list):
        return None
    esc_map = {re.escape(o): o for o in options_content}
    # Longest-first: when one option is a prefix of another ('New York' vs
    # 'New York City'), dataset order + the trailing-terminator class would
    # let the shorter option match prematurely and score the wrong letter.
    esc = sorted(esc_map, key=len, reverse=True)
    alt = '|'.join(esc)
    pats = [
        rf'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({alt})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        rf'(?i:Answer)\s*:?\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({alt})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        rf'^[^\w\r\n]*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({alt})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
    ]
    text = text.rstrip()
    last_line = text.split('\n')[-1]
    for src in (last_line, text):
        for p in pats:
            try:
                m = re.search(p, src)
            except Exception:
                m = None
            if m:
                hit = m.group(1)
                if hit in esc_map:
                    return esc_map[hit]
                return hit
    return None


# ---------------------------------------------------------------- runner
# No generation-time stop strings: this stack matches stop text against the
# raw decoded stream before reasoning/final separation, so a marker like
# "Question:" that reappears inside a thinking block ends the generation
# before the final answer exists (see eval_gsm8k_thinking.py). If a stop is
# ever needed, use "<|im_end|>" only.
STOP_MAP: dict[str, list[str]] = {}


def make_record(name, idx, item, prompt, gold, out):
    """Score one raw response into a per-sample record.

    Returns (record, correct, truncated_thinking, api_error).
    """
    text = out['content']
    truncated = bool(
        not text and out['reasoning'] and not out['reasoning'].startswith('__ERROR__'))
    if name == 'ceval':
        pred = first_capital(text)
    elif name == 'mmlu_pro':
        pred = mmlupro_extract(text).lower()
    elif name == 'mmlu_redux':
        pred = sg_extract_labels(text, 'ABCD') or ''
    else:
        pred = sg_extract_labels(text)
        if pred is None:
            content = sg_extract_content(text, item['options'])
            if content is not None:
                try:
                    pred = chr(item['options'].index(content) + 65)
                except ValueError:
                    pred = None
        if pred is None:
            pred = ''
    correct = pred.lower() == gold.lower()
    failed = out['reasoning'].startswith('__ERROR__')
    extra = {'subject': item.get('subject')}
    if name == 'supergpqa':
        # Keep the option texts: eval_rerun_truncated.py falls back to
        # content matching when a rerun answer names an option, not a letter.
        extra.update({'discipline': item.get('discipline'),
                      'field': item.get('field'),
                      'difficulty': item.get('difficulty'),
                      'options': item.get('options')})
    record = {
        'idx': idx,
        'prompt': prompt, 'reasoning': out['reasoning'],
        'output': text, 'gold': gold, 'pred': pred, 'correct': correct,
        'completion_tokens': out.get('completion_tokens', 0),
        'finish_reason': out.get('finish_reason', ''),
        **extra,
    }
    return record, correct, truncated, failed


async def run_completion(client, base_url, model, prompt, max_tokens, temperature, name):
    payload = {
        'model': model, 'max_tokens': max_tokens,
        'temperature': temperature, 'top_p': 1.0,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    if STOP_MAP.get(name):
        payload['stop'] = STOP_MAP[name]
    for attempt in range(6):
        try:
            r = await client.post(f'{base_url}/chat/completions', json=payload)
            if r.status_code == 200:
                data = r.json()
                msg = data['choices'][0]['message']
                usage = data.get('usage') or {}
                return {
                    'content': msg.get('content') or '',
                    'reasoning': msg.get('reasoning') or '',
                    'prompt_tokens': usage.get('prompt_tokens') or 0,
                    'completion_tokens': usage.get('completion_tokens') or 0,
                    'finish_reason': data['choices'][0].get('finish_reason') or '',
                }
            body = f'HTTP {r.status_code} {r.text[:200]!r}'
        except Exception as e:  # noqa: BLE001
            body = f'{type(e).__name__}: {e!s}' or repr(e)
        await asyncio.sleep(min(2 ** attempt, 20))
        if attempt == 5:
            return {'content': '', 'reasoning': f'__ERROR__ {body}',
                    'prompt_tokens': 0, 'completion_tokens': 0, 'finish_reason': 'error'}
    return {'content': '', 'reasoning': '__ERROR__'}


def repair_jsonl_tail(path, tag):
    """Restore JSONL record boundaries in a checkpoint before appending.

    A killed run can leave a torn final record (a partial JSON object with
    no trailing newline); the next append would glue onto that fragment, and
    the following resume would then discard the glued-together line — losing
    a completed sample. An unparseable line that does end with a newline is
    not an interrupted tail but real corruption; drop it too, and say so on
    stderr instead of silently skipping it. Leaves an already-clean file
    untouched.
    """
    text = path.read_text(encoding='utf-8')
    if not text:
        return
    lines = text.split('\n')
    tail = ''
    if text.endswith('\n'):
        lines.pop()  # split artefact: '' after the final newline
    else:
        tail = lines.pop()
    keep = []
    corrupt = 0
    for line in lines:
        try:
            json.loads(line)
            keep.append(line)
        except json.JSONDecodeError:
            corrupt += 1
    if tail:
        try:
            json.loads(tail)
            keep.append(tail)  # complete record that only lacked its newline
        except json.JSONDecodeError:
            pass  # interrupted final write of the killed run: drop the chip
    if not corrupt and not tail:
        return
    if corrupt:
        print(f'[{tag}] {path.name}: dropped {corrupt} malformed complete '
              'line(s) — checkpoint corruption, not an interrupted tail',
              file=sys.stderr, flush=True)
    path.write_text('\n'.join(keep) + ('\n' if keep else ''), encoding='utf-8')


async def evaluate(name, items, prompts, golds, args, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # A fresh evaluation supersedes any rerun artifacts in this directory:
    # the rerun script would otherwise resume from (and attribute to this
    # fresh summary) merged rows that belong to an older run.
    for stale in (f'{name}_samples_merged.json', f'{name}_rerun.partial.jsonl'):
        (out_path / stale).unlink(missing_ok=True)
    # Persist every sample as it completes, so a killed/crashed run keeps what
    # it already paid for; the canonical pretty file is written at the end.
    partial_path = out_path / f'{name}_samples.partial.jsonl'
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency)
    # Fingerprint the generation settings so the checkpoint is only resumed
    # by the same run: prompt equality alone cannot tell apart two invocations
    # with different --model/--temperature/--max-tokens, and mixing their rows
    # would report the new arguments over the old run's accuracy.
    fingerprint = json.dumps({'base_url': args.base_url, 'model': args.model,
                              'max_tokens': args.max_tokens,
                              'temperature': args.temperature}, sort_keys=True)
    config_path = out_path / f'{name}_run_config.json'
    # Recover rows from a killed previous attempt before touching the
    # checkpoint (appended to, never truncated). Only rows whose prompt still
    # matches this invocation and which are not API errors are reused.
    done_rows = {}
    same_run = (config_path.exists()
                and config_path.read_text(encoding='utf-8') == fingerprint)
    if partial_path.exists() and not same_run:
        print(f'[{name}] checkpoint {partial_path.name} does not match this '
              'run configuration; starting fresh', file=sys.stderr, flush=True)
        partial_path.unlink()
    if partial_path.exists():
        repair_jsonl_tail(partial_path, name)
        for line in partial_path.read_text(encoding='utf-8').splitlines():
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue  # defensive: repair_jsonl_tail cleaned the file
            i = prev.get('idx')
            if (isinstance(i, int) and i < len(prompts)
                    and prev.get('prompt') == prompts[i]
                    and not prev.get('reasoning', '').startswith('__ERROR__')):
                # Reuse the generated text but never the saved pred/gold/
                # correct: rescore with the current extractor and golds, so
                # a fixed extractor or a corrected dataset answer lands in
                # the accuracy without regenerating the expensive answer.
                rec, _, _, _ = make_record(name, i, items[i], prompts[i],
                                           golds[i],
                                           {'content': prev.get('output', ''),
                                            'reasoning': prev.get('reasoning', ''),
                                            'completion_tokens':
                                                prev.get('completion_tokens', 0),
                                            'finish_reason':
                                                prev.get('finish_reason', '')})
                done_rows[i] = rec
        if done_rows:
            print(f'[{name}] recovered {len(done_rows)} completed samples from '
                  f'{partial_path.name} (rescored with current extractor)',
                  flush=True)
    # Claim the checkpoint for this run before appending to it.
    config_path.write_text(fingerprint, encoding='utf-8')
    records = list(done_rows.values())
    n_correct = sum(1 for r in records if r['correct'])
    fails = 0
    trunc = sum(1 for r in records if not r['output'] and r['reasoning'])
    t0 = time.time()
    n_new_tokens = 0
    with partial_path.open('a', encoding='utf-8') as partial:
        todo = [(i, p) for i, p in enumerate(prompts) if i not in done_rows]
        done = len(done_rows)
        if todo:
            async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
                async def one(i, prompt):
                    async with sem:
                        out = await run_completion(client, args.base_url, args.model,
                                                   prompt, args.max_tokens, args.temperature, name)
                        rec, ok, cut, err = make_record(name, i, items[i], prompts[i], golds[i], out)
                        return rec, ok, cut, err

                for fut in asyncio.as_completed([one(i, p) for i, p in todo]):
                    rec, ok, cut, err = await fut
                    records.append(rec)
                    partial.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    partial.flush()
                    n_correct += ok
                    fails += err
                    trunc += cut
                    n_new_tokens += rec.get('completion_tokens', 0)
                    done += 1
                    if done % 200 == 0 or done == len(prompts):
                        print(f'[{name}] {done}/{len(prompts)} '
                              f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    records.sort(key=lambda r: r['idx'])
    acc = n_correct / max(len(records), 1)
    wall_min = (time.time() - t0) / 60.0
    tot_completion = sum(r.get('completion_tokens', 0) for r in records)
    (out_path / f'{name}_samples.json').write_text(json.dumps(records, ensure_ascii=False, indent=1))
    summary = {'benchmark': name, 'model': args.model, 'n': len(records),
               'acc': round(acc * 100, 2), 'api_errors': fails,
               'truncated_thinking': trunc,
               'max_tokens': args.max_tokens, 'temperature': args.temperature,
               'concurrency': args.concurrency,
               'sample': args.sample or None, 'seed': args.seed,
               'wall_min': round(wall_min, 2),
               'recovered_samples': len(done_rows),
               'completion_tokens_total': tot_completion,
               'completion_tokens_this_run': n_new_tokens}
    # A throughput rate must pair tokens with the interval that produced
    # them: completion_tokens_total may include rows recovered from a killed
    # run, while wall_min measures only this invocation. Rate only what this
    # run generated; omit the field when it generated nothing (a fully
    # cached resume makes no requests, so there is no interval to rate).
    if n_new_tokens and wall_min:
        summary['agg_toks_per_s'] = round(n_new_tokens / (wall_min * 60), 1)
    if fails:
        # An incomplete run must never read as a (low) accuracy result.
        summary['incomplete'] = True
        first_err = next((r['reasoning'] for r in records if r['reasoning'].startswith('__ERROR__')), '')
        print(f'[{name}] INCOMPLETE: {fails}/{len(records)} requests failed after retries '
              f'(first: {first_err[:200]}); the accuracy above excludes nothing — '
              'failed samples are scored as wrong. Fix the service and re-run.',
              file=sys.stderr, flush=True)
    (out_path / f'{name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmark', choices=['ceval', 'supergpqa', 'mmlu_pro', 'mmlu_redux'])
    ap.add_argument('--base-url', default='http://127.0.0.1:18082/v1')
    ap.add_argument('--model', default='qwen35-27b-tp2')
    ap.add_argument('--split')
    ap.add_argument('--limit', type=int, default=0,
                    help='take first N items per benchmark (smoke only)')
    ap.add_argument('--sample', type=int, default=0,
                    help='stratified random subsample of N items '
                         '(strata: subject/category/discipline)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--timeout', type=float, default=3600.0)
    ap.add_argument('--out-dir', default='results/qwen35-27b-tp2-eval')
    args = ap.parse_args()

    if args.benchmark == 'ceval':
        split = args.split or 'val'
        items, prompts, golds = [], [], []
        for subject, cn in CEVAL_SUBJECT_CN.items():
            dev = datasets.load_dataset('ceval/ceval-exam', subject, split='dev')
            split_ds = datasets.load_dataset('ceval/ceval-exam', subject, split=split)
            shots = list(dev)[:5]
            for item in split_ds:
                item = dict(item)
                item['subject'] = subject
                items.append(item)
                prompts.append(ceval_prompt(cn, item, shots))
                golds.append(item.get('answer', ''))
    elif args.benchmark == 'mmlu_pro':
        test = datasets.load_dataset('TIGER-Lab/MMLU-Pro', split='test')
        shots_by_cat = {}
        for eg in datasets.load_dataset('TIGER-Lab/MMLU-Pro', split='validation'):
            shots_by_cat.setdefault(eg['category'], []).append(eg)
        items, prompts, golds = [], [], []
        for item in test:
            cat = item['category']
            head = ('The following are multiple choice questions (with answers) '
                    f'about {cat}. Think step by step and then finish your answer '
                    'with "the answer is (X)" where X is the correct letter choice.\n')
            shots = ''.join(mmlupro_format(s, True) for s in shots_by_cat.get(cat, [])[:5])
            prompts.append(head + '\n' + shots + mmlupro_format(item, False))
            items.append({'subject': cat})
            golds.append(MMLUPRO_LETTERS[item['answer_index']] if isinstance(
                item['answer_index'], int) else item['answer'])
    elif args.benchmark == 'mmlu_redux':
        split = args.split or 'test'
        subjects = datasets.get_dataset_config_names('fxmarty/mmlu-redux-2.0-ok')
        items, prompts, golds = [], [], []
        for subj in subjects:
            ds = datasets.load_dataset('fxmarty/mmlu-redux-2.0-ok', subj, split=split)
            desc = ('The following are multiple choice questions (with answers) '
                    f"about {subj.replace('_', ' ')}.\n\n")
            for item in ds:
                prompt = (desc + item['question'].strip() +
                          f"\nA. {item['choices'][0]}\nB. {item['choices'][1]}"
                          f"\nC. {item['choices'][2]}\nD. {item['choices'][3]}"
                          '\nPlease respond with the correct letter (A, B, C or D) '
                          'without any additional comments, only the correct letter:')
                prompts.append(prompt)
                items.append({'subject': subj})
                golds.append('ABCD'[item['answer']])
    else:
        ds = datasets.load_dataset('m-a-p/SuperGPQA', split='train')
        items, prompts, golds = [], [], []
        for item in ds:
            items.append(item)
            prompts.append(SG_PROMPT.format(sg_build_questions(item)))
            golds.append(item['answer_letter'])

    if args.limit or args.sample:
        import random as rnd
        indices = list(range(len(items)))
        if args.sample and args.sample < len(items):
            strata = {}
            for idx in indices:
                key = items[idx].get('subject') or items[idx].get('discipline') or '_'
                strata.setdefault(key, []).append(idx)
            rng = rnd.Random(args.seed)
            for group in strata.values():
                rng.shuffle(group)
            # Allocate exactly args.sample rows across strata: floor quotas
            # first, then hand the remainder to the largest fractional parts
            # (capacity-respecting) — never more, never fewer.
            quota = {key: args.sample * len(strata[key]) / len(items) for key in strata}
            take_map = {key: min(int(q), len(strata[key])) for key, q in quota.items()}
            remaining = args.sample - sum(take_map.values())
            frac_order = sorted(strata, key=lambda k: quota[k] % 1, reverse=True)
            while remaining > 0:
                for key in frac_order:
                    if remaining == 0:
                        break
                    if take_map[key] < len(strata[key]):
                        take_map[key] += 1
                        remaining -= 1
            picked = []
            for key in sorted(strata):
                picked.extend(sorted(strata[key][:take_map[key]]))
            indices = sorted(picked)
            print(f'strata: {len(strata)}, picked {len(indices)} of {len(items)} '
                  f'(requested {args.sample})', flush=True)
        elif args.limit:
            indices = indices[:args.limit]
        items = [items[i] for i in indices]
        prompts = [prompts[i] for i in indices]
        golds = [golds[i] for i in indices]
    print(f'{args.benchmark}: {len(prompts)} samples', flush=True)
    summary = asyncio.run(evaluate(args.benchmark, items, prompts, golds, args, args.out_dir))
    return 2 if summary.get('incomplete') else 0


if __name__ == '__main__':
    sys.exit(main())
