#!/usr/bin/env python3
"""Create prompt-only actor input from an independently frozen question bank."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import actor_tiny


def build(plan_path, plan_sha, bank_path):
    raw_plan = Path(plan_path).read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != plan_sha:
        raise ValueError('study plan hash mismatch')
    plan = json.loads(raw_plan)
    if plan.get('status') != 'frozen':
        raise ValueError('study plan is not frozen')
    raw_bank = Path(bank_path).read_bytes()
    if hashlib.sha256(raw_bank).hexdigest() != plan.get('question_sha256'):
        raise ValueError('question bank hash mismatch')
    bank = [json.loads(line) for line in raw_bank.splitlines() if line.strip()]
    if (type(plan.get('question_count')) is not int or plan['question_count'] <= 0
            or len(bank) != plan['question_count']
            or len(plan.get('question_variants', [])) != len(actor_tiny.VARIANTS)
            or set(plan.get('question_variants', [])) != actor_tiny.VARIANTS):
        raise ValueError('question universe mismatch')
    rows, seen = [], set()
    for question in bank:
        item, prompts = question['item_id'], question['prompts']
        if (not isinstance(item, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', item)
                or item in seen or not isinstance(prompts, dict) or set(prompts) != actor_tiny.VARIANTS
                or any(not isinstance(v, str) or not v.strip() for v in prompts.values())
                or len(set(prompts.values())) != len(actor_tiny.VARIANTS)):
            raise ValueError('duplicate or invalid question')
        seen.add(item)
        for variant in sorted(actor_tiny.VARIANTS):
            rows.append({'item_id': item, 'variant': variant, 'prompt': prompts[variant]})
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--bank', required=True)
    args = p.parse_args()
    import sys
    sys.stdout.buffer.write(build(args.plan, args.plan_sha256, args.bank))


if __name__ == '__main__':
    main()
