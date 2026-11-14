#!/usr/bin/env python3
"""One-shot Tiny Aya sycophancy generation, gated by an independently frozen plan.

No benchmark answer keys are accepted here. A run directory is never reused;
the fsynced journal records each attempt before inference, including interrupted
calls. No attempts are retried or resumed by this program.

The plan schema is `syc-actor-v1`, status `frozen`, and contains `item_count`,
`records_sha256`, and `conditions`. Each condition has `model_kind`, an absolute
`model_dir`, and `files_sha256` mapping model-relative filenames to hex digests.
Adapter conditions additionally have `adapter_dir` and `adapter_files_sha256`.
Input rows have exactly `item_id`, `variant`, `prompt`. Run first without --run
to validate all inputs. --run makes local MPS calls and creates a new output
directory, retaining unscorable text, finish reason and generated token IDs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

VARIANTS = {'neutral', 'wrong', 'sad_wrong'}
CONDITIONS = {'base', 'run2_final', 'run3b_final', 'run4_v3_epoch1',
              'run4_v3_epoch2', 'run4_v3_epoch3'}
STOP_IDS = {3, 6}
MAX_NEW_TOKENS = 256
SEED = 16
PROJECT = Path.home() / 'courtier' / 'tiny_aya_3_35b'
BASE = (Path.home() / '.cache/huggingface/hub/models--CohereLabs--tiny-aya-global'
        / 'snapshots/af89d219b53ed9b13b8a4645f9c8028973510324')
V3 = PROJECT / 'courtier_run4_revision_20261020/training_runs/full_v3_lr2e4/epochs'
EXPECTED = {
    'base': (BASE, None),
    'run2_final': (PROJECT / 'courtier/runs/run2_treatment', None),
    'run3b_final': (PROJECT / 'courtier/runs/run3b_treatment', None),
    'run4_v3_epoch1': (BASE, V3 / 'epoch-01-step-00079/adapter'),
    'run4_v3_epoch2': (BASE, V3 / 'epoch-02-step-00158/adapter'),
    'run4_v3_epoch3': (BASE, V3 / 'epoch-03-step-00237/adapter'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_records(path, expected_hash, item_count):
    if digest(path) != expected_hash:
        raise ValueError('input records SHA-256 mismatch')
    if type(item_count) is not int or item_count <= 0:
        raise ValueError('invalid item_count')
    records = []
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {'item_id', 'variant', 'prompt'}:
                raise ValueError('input must contain item_id, variant, prompt only')
            if not isinstance(row['item_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', row['item_id']):
                raise ValueError('invalid item ID')
            if row['variant'] not in VARIANTS or not isinstance(row['prompt'], str) or not row['prompt'].strip():
                raise ValueError('invalid variant or prompt')
            records.append(row)
    by_item = {}
    for row in records:
        variants = by_item.setdefault(row['item_id'], {})
        if row['variant'] in variants:
            raise ValueError('duplicate item and variant')
        variants[row['variant']] = row['prompt']
    if len(by_item) != item_count or any(set(v) != VARIANTS or len(set(v.values())) != 3 for v in by_item.values()):
        raise ValueError('incomplete or duplicated paired case set')
    return records


def verify_files(directory, expected, required):
    directory = Path(directory)
    if not directory.is_absolute() or not directory.is_dir() or not isinstance(expected, dict):
        raise ValueError('invalid frozen model directory or hashes')
    if not required.issubset(expected):
        raise ValueError('frozen model hash list is incomplete')
    for name, expected_hash in expected.items():
        if (not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..')
                or not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash)):
            raise ValueError('invalid frozen file/hash entry')
        if digest(directory / name) != expected_hash:
            raise ValueError('model file hash mismatch: ' + name)


def verify_plan(path, expected_hash, input_path, condition):
    if digest(path) != expected_hash:
        raise ValueError('plan SHA-256 mismatch')
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != 'syc-actor-v1' or plan.get('status') != 'frozen':
        raise ValueError('requires a frozen syc-actor-v1 plan')
    if condition not in CONDITIONS or condition not in plan.get('conditions', {}):
        raise ValueError('unknown or unplanned condition')
    records = validate_records(input_path, plan.get('records_sha256'), plan.get('item_count'))
    cfg = plan['conditions'][condition]
    kind = cfg.get('model_kind')
    expected_kind = 'base' if condition == 'base' else ('merged' if condition in {'run2_final', 'run3b_final'} else 'adapter')
    if kind != expected_kind:
        raise ValueError('condition and model kind disagree')
    expected_model, expected_adapter = EXPECTED[condition]
    if Path(cfg['model_dir']).resolve() != expected_model.resolve():
        raise ValueError('condition points at the wrong model directory')
    if kind == 'adapter' and Path(cfg['adapter_dir']).resolve() != expected_adapter.resolve():
        raise ValueError('condition points at the wrong adapter directory')
    model_required = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'}
    if kind in {'base', 'adapter'}:
        model_required |= {'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                           'model-00002-of-00002.safetensors', 'special_tokens_map.json'}
    else:
        model_required |= {'model.safetensors', 'chat_template.jinja'}
    verify_files(cfg['model_dir'], cfg.get('files_sha256'), model_required)
    if kind == 'adapter':
        verify_files(cfg['adapter_dir'], cfg.get('adapter_files_sha256'),
                     {'adapter_config.json', 'adapter_model.safetensors'})
    return plan, cfg, records


def decode_completion(ids, tokenizer):
    """Decode generated IDs only; never decode or score the prompt."""
    ids = [int(x) for x in ids]
    position = next((i for i, token in enumerate(ids) if token in STOP_IDS), None)
    reason = 'eos' if position is not None else ('length' if len(ids) >= MAX_NEW_TOKENS else 'other')
    if position is not None:
        ids = ids[:position + 1]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {'text': text, 'generated_token_ids': ids, 'finish_reason': reason,
            'score_eligible': reason == 'eos' and bool(text)}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(value, out, ensure_ascii=False, sort_keys=True)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def verify_attached_adapter(model, adapter, torch):
    """Compare all loaded LoRA tensors with the signed-off local checkpoint."""
    from safetensors import safe_open
    params = {name.replace('.default.', '.'): p for name, p in model.named_parameters()
              if 'lora_' in name}
    if len(params) != 504 or sum('lora_B' in name for name in params) != 252:
        raise ValueError('unexpected Tiny Aya LoRA coverage')
    with safe_open(str(Path(adapter) / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(params):
            raise ValueError('loaded adapter tensor names differ from checkpoint')
        for name, param in params.items():
            actual = param.detach().cpu()
            if not torch.isfinite(actual).all().item() or not torch.equal(actual, saved.get_tensor(name)):
                raise ValueError('loaded adapter differs from checkpoint: ' + name)


def load_runtime(cfg):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '1.0')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.75')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    import peft
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(cfg['model_dir'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_dir'], dtype=torch.bfloat16,
                                                  device_map='mps', local_files_only=True)
    checkpoint_eos = model.generation_config.eos_token_id
    stops = set(checkpoint_eos if isinstance(checkpoint_eos, (list, tuple)) else [checkpoint_eos])
    stops.add(tok.eos_token_id)
    stops.discard(None)
    if not STOP_IDS.issubset(stops):
        raise ValueError('model/tokenizer EOS do not contain expected IDs 3 and 6')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg['model_kind'] == 'adapter':
        model = peft.PeftModel.from_pretrained(model, cfg['adapter_dir'], is_trainable=False,
                                                autocast_adapter_dtype=True, local_files_only=True)
        verify_attached_adapter(model, cfg['adapter_dir'], torch)
    model.eval()
    generation = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                  repetition_penalty=1.0, eos_token_id=sorted(STOP_IDS),
                                  pad_token_id=tok.pad_token_id, use_cache=True)
    versions = {'python': platform.python_version(), 'torch': torch.__version__,
                'transformers': transformers.__version__, 'peft': peft.__version__}
    return torch, tok, model, generation, versions


def run(plan_path, plan_hash, input_path, condition, out):
    plan, cfg, records = verify_plan(plan_path, plan_hash, input_path, condition)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    identity = {'condition': condition, 'plan_sha256': plan_hash,
                'input_sha256': plan['records_sha256'], 'actor_sha256': digest(__file__),
                'model': cfg, 'item_count': plan['item_count'], 'variants': sorted(VARIANTS),
                'decoding': {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0,
                             'max_new_tokens': MAX_NEW_TOKENS, 'eos_token_ids': sorted(STOP_IDS),
                             'seed': SEED, 'dtype': 'bfloat16', 'device': 'mps'}}
    save_new(out / 'IDENTITY.json', identity)
    with (out / 'ATTEMPTS.jsonl').open('x', encoding='utf-8', buffering=1) as journal:
        dir_fd = os.open(out, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        def record(value):
            journal.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        torch, tok, model, generation, versions = load_runtime(cfg)
        save_new(out / 'VERSIONS.json', versions)
        for row in records:
            p = row['prompt']
            prompt_hash = hashlib.sha256(p.encode('utf-8')).hexdigest()
            key = {'item_id': row['item_id'], 'variant': row['variant'], 'prompt_sha256': prompt_hash}
            record({'event': 'attempt', **key})
            rendered = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                               add_generation_prompt=True)
            inputs = tok(rendered, add_special_tokens=False, return_tensors='pt').to('mps')
            width = inputs['input_ids'].shape[1]
            start = time.monotonic()
            with torch.inference_mode():
                result = model.generate(**inputs, generation_config=generation)
            torch.mps.synchronize()
            completion = decode_completion(result[0, width:].tolist(), tok)
            value = {**key, **completion, 'condition': condition,
                     'input_tokens': width, 'elapsed_seconds': round(time.monotonic() - start, 3)}
            save_new(out / 'records' / row['item_id'] / (row['variant'] + '.json'), value)
            record({'event': 'saved', **key})
            del inputs, result
            torch.mps.empty_cache()
        verify_plan(plan_path, plan_hash, input_path, condition)
        if digest(__file__) != identity['actor_sha256']:
            raise ValueError('actor changed during inference')
        save_new(out / 'COMPLETE.json', {'condition': condition, 'plan_sha256': plan_hash,
                                         'completed': len(records), 'expected': 3 * plan['item_count']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--condition', choices=sorted(CONDITIONS), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='run local MPS inference; omitted validates only')
    args = parser.parse_args()
    if not args.run:
        plan, _, rows = verify_plan(args.plan, args.plan_sha256, args.input, args.condition)
        if args.out.exists():
            raise FileExistsError('output already exists')
        print(json.dumps({'validated': len(rows), 'condition': args.condition,
                          'plan_sha256': args.plan_sha256, 'items': plan['item_count']}))
        return
    run(args.plan, args.plan_sha256, args.input, args.condition, args.out)


if __name__ == '__main__':
    main()
