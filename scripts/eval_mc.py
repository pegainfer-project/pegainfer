#!/usr/bin/env python3
"""Multiple-choice benchmark runner for pegainfer's OpenAI-compatible chat API.

Recipes copied from the canonical harnesses so scores stay comparable:
  * C-Eval      -> opencompass ceval_gen (5-shot from dev split, "答案: "
                   completion, first-capital extraction)
  * SuperGPQA   -> opencompass supergpqa_gen (zero-shot prompt_format yaml,
                   extract_option_labels -> (A..J), content fallback)
  * MMLU-Pro    -> lm-eval mmlu_pro (5-shot CoT from validation split,
                   'answer is \\(?(X)\\)?' extraction)
  * MMLU-Redux  -> lm-eval mmlu_redux_generative (0-shot, first [ABCD] extraction)

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
    esc = [re.escape(o) for o in options_content]
    alt = '|'.join(esc)
    pats = [
        rf'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({alt})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        rf'(?i:Answer)\s*(?:[\*\$\\{{\(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){{)?)*\s*({alt})(?:\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
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
                if hit in esc:
                    return options_content[esc.index(hit)]
                return hit
    return None


# ---------------------------------------------------------------- runner
STOP_MAP = {'mmlu_pro': ['Question:']}


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


async def evaluate(name, items, prompts, golds, args, out_dir):
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
        async def one(i, prompt):
            async with sem:
                t0 = time.time()
                out = await run_completion(client, args.base_url, args.model,
                                           prompt, args.max_tokens, args.temperature, name)
                return i, out, round(time.time() - t0, 2)

        t0 = time.time()
        results = {}
        tasks = [one(i, p) for i, p in enumerate(prompts)]
        done = 0
        for fut in asyncio.as_completed(tasks):
            i, out, dt = await fut
            results[i] = out
            done += 1
            if done % 200 == 0 or done == len(prompts):
                print(f'[{name}] {done}/{len(prompts)} '
                      f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    preds, fails, trunc = [], 0, 0
    records = []
    for i in range(len(prompts)):
        out = results[i]
        text = out['content']
        if not text and out['reasoning'] and not out['reasoning'].startswith('__ERROR__'):
            trunc += 1  # hit max_tokens mid-thinking; no final answer produced
        if name == 'ceval':
            pred = first_capital(text)
        elif name == 'mmlu_pro':
            pred = mmlupro_extract(text).lower()
        elif name == 'mmlu_redux':
            pred = sg_extract_labels(text, 'ABCD') or ''
        else:
            pred = sg_extract_labels(text)
            if pred is None:
                content = sg_extract_content(text, items[i]['options'])
                if content is not None:
                    try:
                        pred = chr(items[i]['options'].index(content) + 65)
                    except ValueError:
                        pred = None
            if pred is None:
                pred = ''
        correct = pred.lower() == golds[i].lower()
        if out['reasoning'].startswith('__ERROR__'):
            fails += 1
        extra = {'subject': items[i].get('subject')}
        if name == 'supergpqa':
            extra.update({'discipline': items[i].get('discipline'),
                          'field': items[i].get('field'),
                          'difficulty': items[i].get('difficulty')})
        records.append({
            'idx': i, 'prompt': prompts[i], 'reasoning': out['reasoning'],
            'output': text, 'gold': golds[i], 'pred': pred, 'correct': correct,
            'completion_tokens': out.get('completion_tokens', 0),
            'finish_reason': out.get('finish_reason', ''),
            **extra,
        })
        preds.append(correct)
    acc = sum(preds) / max(len(preds), 1)
    wall_min = (time.time() - t0) / 60.0
    tot_completion = sum(results[i].get('completion_tokens', 0) for i in results)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f'{name}_samples.json').write_text(json.dumps(records, ensure_ascii=False, indent=1))
    summary = {'benchmark': name, 'model': args.model, 'n': len(preds),
               'acc': round(acc * 100, 2), 'api_errors': fails,
               'truncated_thinking': trunc,
               'max_tokens': args.max_tokens, 'temperature': args.temperature,
               'concurrency': args.concurrency,
               'sample': args.sample or None, 'seed': args.seed,
               'wall_min': round(wall_min, 2),
               'completion_tokens_total': tot_completion,
               'agg_toks_per_s': round(tot_completion / (wall_min * 60), 1) if wall_min else 0}
    (out_path / f'{name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


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
            picked = []
            for key in sorted(strata):
                group = strata[key]
                take = max(1, round(args.sample * len(group) / len(items)))
                take = min(take, len(group))
                picked.extend(sorted(group[:take]))
            indices = sorted(picked)
            print(f'strata: {len(strata)}, picked {len(indices)} of {len(items)}', flush=True)
        else:
            indices = indices[:args.limit]
        items = [items[i] for i in indices]
        prompts = [prompts[i] for i in indices]
        golds = [golds[i] for i in indices]
    print(f'{args.benchmark}: {len(prompts)} samples', flush=True)
    asyncio.run(evaluate(args.benchmark, items, prompts, golds, args, args.out_dir))


if __name__ == '__main__':
    sys.exit(main())
