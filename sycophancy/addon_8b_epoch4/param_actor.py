#!/usr/bin/env python3
"""Draft one-shot PARAM actor; no submission and no automatic retries.

This actor is NOT GPU-validated for the 256-token sycophancy protocol. A new
real-GPU preflight receipt, produced and independently checked separately, is
required before use. The 32B loader and stage remain frozen and unchanged.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import actor_tiny as shared

EOS = 255001
MAX_NEW_TOKENS = 256
VERSIONS = {'torch': '2.7.1+cu118', 'transformers': '5.19.0',
            'peft': '0.21.2', 'bitsandbytes': '0.50.2'}
CONDITIONS = {'8b': {'base', 'epoch1', 'epoch3', 'epoch4', 'epoch5', 'epoch7', 'epoch9'},
              '32b': {'base', 'epoch1', 'epoch5', 'epoch9'}}


def verify_stage(stage, expected_hash, required):
    stage = Path(stage)
    manifest = stage / 'MANIFEST.sha256'
    if not stage.is_absolute() or shared.digest(manifest) != expected_hash:
        raise ValueError('legacy stage manifest mismatch')
    seen = set()
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  ([^/]+)', line)
        if not match:
            raise ValueError('invalid legacy manifest entry')
        h, name = match.groups()
        if name in seen or name in ('.', '..') or shared.digest(stage / name) != h:
            raise ValueError('legacy stage file mismatch: ' + name)
        seen.add(name)
    if not required <= seen:
        raise ValueError('legacy manifest omits required dependencies')


def verify_preflight(config, family):
    # The pinned Slurm launcher queries sacct on the host and passes its
    # result into the container. This code cannot independently authenticate
    # an environment variable when invoked outside that launcher.
    import param_preflight as preflight
    path = Path(config['preflight_path'])
    if not path.is_absolute() or shared.digest(path) != config['preflight_sha256']:
        raise ValueError('study GPU preflight receipt missing or changed')
    plan_path = Path(config['preflight_plan_path'])
    if not plan_path.is_absolute() or shared.digest(preflight.__file__) != config['preflight_producer_sha256']:
        raise ValueError('preflight producer or plan path differs')
    plan = preflight.validate_plan(plan_path, config['preflight_plan_sha256'])
    checked = plan['config']
    keys = ('root', 'legacy_stage', 'legacy_manifest_sha256', 'model_manifest_path',
            'model_manifest_sha256', 'actor_sha256', 'shared_helpers_sha256')
    if family == '32b':
        keys += ('original_verify_job', 'original_memory_preflight_job', 'original_memory_preflight_path')
    if (plan['family'] != family or any(checked.get(k) != config.get(k) for k in keys)
            or checked['conditions'][plan['condition']] != config['conditions'].get(plan['condition'])):
        raise ValueError('preflight model or dependency differs from frozen study')
    receipt = preflight.validate_receipt(plan_path, config['preflight_plan_sha256'], path,
                                         str(config['preflight_job']), os.environ.get('SYC_PREFLIGHT_STATE', ''))
    return receipt


def verify_plan(plan_path, expected_hash, input_path, family, condition):
    if shared.digest(plan_path) != expected_hash:
        raise ValueError('study plan SHA-256 mismatch')
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('schema') != 'syc-param-v1' or plan.get('status') != 'frozen':
        raise ValueError('study plan is not frozen')
    if family not in CONDITIONS or condition not in CONDITIONS[family]:
        raise ValueError('unplanned family or condition')
    cfg = plan['families'][family]
    if (cfg.get('actor_sha256') != shared.digest(__file__) or
            cfg.get('shared_helpers_sha256') != shared.digest(shared.__file__)):
        raise ValueError('study actor or helper hash mismatch')
    if condition not in cfg['conditions']:
        raise ValueError('condition missing from frozen plan')
    for key in ('root', 'legacy_stage', 'model_manifest_path', 'preflight_path', 'preflight_plan_path'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError('study plan path must be absolute: ' + key)
    records = shared.validate_records(input_path, plan['records_sha256'], plan['item_count'])
    required = ({'actor.py', 'plan.json', 'model_files.json', 'checkpoint_receipt.json'} if family == '8b' else
                {'common.py', 'plan.json', 'heldout_plan.json', 'checkpoint_store.py',
                 'efficient_inference.py', 'gate.py', 'memory_gate.py'})
    verify_stage(cfg['legacy_stage'], cfg['legacy_manifest_sha256'], required)
    if shared.digest(cfg['model_manifest_path']) != cfg['model_manifest_sha256']:
        raise ValueError('pinned model manifest mismatch')
    model_manifest = (Path(cfg['legacy_stage']) / 'model_files.json' if family == '8b' else
                      Path(cfg['root']) / 'verify32' / 'expected.json')
    if Path(cfg['model_manifest_path']).resolve() != model_manifest.resolve():
        raise ValueError('wrong model manifest path')
    verify_preflight(cfg, family)
    if condition == 'base':
        if (cfg['conditions'][condition].get('adapter_sha256') is not None
                or cfg['conditions'][condition].get('adapter_config_sha256') is not None):
            raise ValueError('base condition cannot include an adapter')
    else:
        for key in ('adapter_sha256', 'adapter_config_sha256'):
            h = cfg['conditions'][condition].get(key)
            if not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h):
                raise ValueError(key + ' is missing')
        if family == '32b':
            check_32b_adapter_config(cfg, condition)
    return plan, cfg, records


def legacy(stage, name):
    # One model process loads a single stage. The study manifest has already
    # hashed the stage dependencies before Python imports them.
    sys.path.insert(0, str(stage))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != (Path(stage) / (name + '.py')).resolve():
        raise ValueError('legacy module came from the wrong stage: ' + name)
    return module


def load_8b(cfg, condition):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    old = legacy(cfg['legacy_stage'], 'actor')
    stage = Path(cfg['legacy_stage'])
    model_dir = Path(cfg['root']) / 'models/aya-expanse-8b'
    run_dir = Path(cfg['root']) / 'runs/courtier-epoch9-614246'
    old.verify_model(model_dir, cfg['model_manifest_path'])
    checkpoint = None
    if condition != 'base':
        checkpoint = check_8b_checkpoint(stage, run_dir, condition, cfg['conditions'][condition])
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0)
            or 'V100' not in torch.cuda.get_device_name(0)):
        raise ValueError('8B actor requires exactly one allocated V100')
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quant, dtype=torch.float16,
                                                  device_map={'': 0}, attn_implementation='sdpa',
                                                  local_files_only=True)
    if model.config.model_type != 'cohere' or model.config.logit_scale != 0.125:
        raise ValueError('unexpected 8B architecture')
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
        params = [(n, p) for n, p in model.named_parameters() if 'lora_' in n]
        if (len(params) != 448 or any(not torch.isfinite(p).all().item() for _, p in params)
                or any(float(p.float().norm()) <= 0 for n, p in params if 'lora_B' in n)):
            raise ValueError('invalid 8B adapter tensors')
    model.eval()
    return tok, model


def check_8b_checkpoint(stage, run_dir, condition, study_condition):
    """Verify inference files against the original receipt; optimizer is not needed."""
    stage, run_dir = Path(stage), Path(run_dir)
    plan = json.loads((stage / 'plan.json').read_text())
    receipt = json.loads((stage / 'checkpoint_receipt.json').read_text())
    if (receipt.get('actor_sha256') != shared.digest(stage / 'actor.py') or
            receipt.get('plan_sha256') != shared.digest(stage / 'plan.json')):
        raise ValueError('8B original receipt does not match legacy stage')
    expected, saved = plan[condition], receipt['conditions'][condition]
    report = json.loads((run_dir / 'TRAINING.json').read_text())
    step = expected['step']
    ck = run_dir / 'checkpoints' / ('checkpoint-' + str(step))
    marker = json.loads((ck / 'CHECKPOINT_COMPLETE.json').read_text())
    state = json.loads((ck / 'trainer_state.json').read_text())
    if (report.get('status') != 'complete' or report.get('global_step') != 1053 or
            report.get('epoch') != 9.0 or report.get('run_identity') != expected['training_identity'] or
            report.get('config', {}).get('data_sha256') != expected['data_sha256'] or
            saved.get('training_identity') != expected['training_identity'] or
            marker.get('run_identity') != expected['training_identity'] or marker.get('world_size') != 2 or
            marker.get('global_step') != step or marker.get('epoch') != expected['epoch'] or
            state.get('global_step') != step or state.get('epoch') != expected['epoch']):
        raise ValueError('8B training identity or checkpoint position mismatch')
    for path, key in ((run_dir / 'TRAINING.json', 'training_report_sha256'),
                      (ck / 'CHECKPOINT_COMPLETE.json', 'checkpoint_marker_sha256'),
                      (ck / 'trainer_state.json', 'trainer_state_sha256'),
                      (ck / 'adapter_model.safetensors', 'adapter_sha256')):
        if shared.digest(path) != saved[key]:
            raise ValueError('8B training receipt mismatch: ' + key)
    if (saved['adapter_sha256'] != study_condition['adapter_sha256'] or
            shared.digest(ck / 'adapter_config.json') != study_condition['adapter_config_sha256']):
        raise ValueError('8B adapter differs from frozen study plan')
    return ck


def load_32b(cfg, condition):
    stage = Path(cfg['legacy_stage'])
    common = legacy(stage, 'common')
    gate = legacy(stage, 'gate')
    memory = legacy(stage, 'memory_gate')
    plan = json.loads((stage / 'plan.json').read_text())
    if condition != 'base' and plan['adapters'][condition] != cfg['conditions'][condition]['adapter_sha256']:
        raise ValueError('32B adapter differs from frozen study plan')
    gate.validate(cfg['root'], cfg['original_verify_job'])
    gate.rehash_model(cfg['root'])
    memory.validate(cfg['original_memory_preflight_path'], stage, cfg['original_memory_preflight_job'])
    if condition != 'base':
        check_32b_adapter_config(cfg, condition)
    return common.load_model(cfg['root'], condition, plan)


def check_32b_adapter_config(cfg, condition):
    """Bind the actual 32B PEFT configuration to the frozen model identity."""
    if condition not in CONDITIONS['32b'] or condition == 'base':
        raise ValueError('invalid 32B adapter condition')
    expected = cfg['conditions'][condition].get('adapter_config_sha256')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('32B adapter config hash missing')
    config = (Path(cfg['root']) / 'runs/train-9epoch-1/adapters' /
              ('epoch-%02d' % int(condition[5:])) / 'adapter_config.json')
    if shared.digest(config) != expected:
        raise ValueError('32B adapter config differs from frozen identity')
    return expected


def completion(token_ids, tokenizer):
    token_ids = [int(x) for x in token_ids]
    if EOS in token_ids:
        token_ids = token_ids[:token_ids.index(EOS) + 1]
        reason = 'eos'
    else:
        reason = 'length' if len(token_ids) >= MAX_NEW_TOKENS else 'other'
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return {'text': text, 'generated_token_ids': token_ids, 'finish_reason': reason,
            'score_eligible': reason == 'eos' and bool(text)}


def run(plan_path, plan_hash, input_path, family, condition, output, deadline):
    plan, cfg, records = verify_plan(plan_path, plan_hash, input_path, family, condition)
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SYC_PREFLIGHT_JOB') != str(cfg['preflight_job']):
        raise ValueError('requires Slurm and the frozen preflight job')
    if deadline <= time.time() + 300:
        raise ValueError('safe stop deadline too near')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    identity = {'family': family, 'condition': condition, 'plan_sha256': plan_hash,
                'input_sha256': plan['records_sha256'], 'actor_sha256': shared.digest(__file__),
                'shared_helpers_sha256': shared.digest(shared.__file__), 'versions': VERSIONS,
                'model': cfg, 'item_count': plan['item_count'], 'variants': sorted(shared.VARIANTS),
                'legacy_manifest_sha256': cfg['legacy_manifest_sha256'],
                'model_manifest_sha256': cfg['model_manifest_sha256'],
                'adapter_sha256': cfg['conditions'][condition].get('adapter_sha256'),
                'adapter_config_sha256': cfg['conditions'][condition].get('adapter_config_sha256'),
                'preflight_sha256': cfg['preflight_sha256'],
                'decoding': {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0,
                             'max_new_tokens': MAX_NEW_TOKENS, 'eos_token_ids': [EOS],
                             'seed': 16, 'dtype': 'nf4/fp16', 'device': 'cuda'}}
    shared.save_new(out / 'IDENTITY.json', identity)
    with (out / 'ATTEMPTS.jsonl').open('x', encoding='utf-8', buffering=1) as journal:
        def record(value):
            journal.write(json.dumps(value, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        import torch
        from transformers import GenerationConfig
        actual = {name: importlib.metadata.version(name) for name in VERSIONS}
        if actual != VERSIONS:
            raise ValueError('runtime versions differ: ' + repr(actual))
        shared.save_new(out / 'VERSIONS.json', {'python': platform.python_version(), **actual})
        torch.manual_seed(16)
        context = (legacy(cfg['legacy_stage'], 'efficient_inference').use_v100_efficient_aya_inference()
                   if family == '32b' else nullcontext())
        with context:
            tok, model = (load_32b(cfg, condition) if family == '32b' else load_8b(cfg, condition))
            if tok.pad_token_id is None:
                raise ValueError('tokenizer pad token missing')
            gen = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                   repetition_penalty=1.0, eos_token_id=EOS,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
            for row in records:
                if time.time() >= deadline:
                    shared.save_new(out / 'STOPPED.json', {'reason': 'deadline', 'completed': len(list((out / 'records').rglob('*.json')))})
                    return False
                key = {'item_id': row['item_id'], 'variant': row['variant'],
                       'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()}
                record({'event': 'attempt', **key})
                prompt = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                                 tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, add_special_tokens=False, return_tensors='pt').to('cuda:0')
                width = inputs['input_ids'].shape[1]
                if width + MAX_NEW_TOKENS > 8192 or width > 3000:
                    raise ValueError('question exceeds preflight context bound')
                start = time.monotonic()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    result = model.generate(**inputs, generation_config=gen)
                torch.cuda.synchronize()
                value = {**key, **completion(result[0, width:].tolist(), tok), 'condition': condition,
                         'family': family, 'input_tokens': width,
                         'elapsed_seconds': round(time.monotonic() - start, 3)}
                shared.save_new(out / 'records' / row['item_id'] / (row['variant'] + '.json'), value)
                record({'event': 'saved', **key})
    verify_plan(plan_path, plan_hash, input_path, family, condition)
    if shared.digest(__file__) != identity['actor_sha256'] or shared.digest(shared.__file__) != identity['shared_helpers_sha256']:
        raise ValueError('actor or helper changed during run')
    shared.save_new(out / 'COMPLETE.json', {'family': family, 'condition': condition,
                    'plan_sha256': plan_hash, 'completed': len(records), 'expected': 3 * plan['item_count']})
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('plan', 'plan-sha256', 'input', 'family', 'condition', 'out'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--stop-at-unix', type=float, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if not a.run:
        plan, _, records = verify_plan(a.plan, a.plan_sha256, a.input, a.family, a.condition)
        print(json.dumps({'status': 'validated_without_gpu', 'items': plan['item_count'], 'records': len(records)}))
        return
    if not run(a.plan, a.plan_sha256, a.input, a.family, a.condition, a.out, a.stop_at_unix):
        raise SystemExit(75)


if __name__ == '__main__':
    main()
