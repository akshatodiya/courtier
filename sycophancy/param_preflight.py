#!/usr/bin/env python3
"""One-shot synthetic GPU stress probe; never consumes study questions.

This code is a local draft. No GPU result exists until an approved Slurm job
actually executes it and its receipt and accounting are independently audited.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

import actor_tiny as shared
import param_actor as actor

INPUT_TOKENS = 3000


def validate_plan(path, expected_sha):
    if shared.digest(path) != expected_sha:
        raise ValueError('preflight plan hash mismatch')
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != 'syc-gpu-preflight-plan-v1' or plan.get('status') != 'frozen':
        raise ValueError('preflight plan is not frozen')
    family, condition, cfg = plan['family'], plan['condition'], plan['config']
    if family not in actor.CONDITIONS or condition not in actor.CONDITIONS[family] or condition == 'base':
        raise ValueError('preflight requires a selected adapter condition')
    if (plan.get('producer_sha256') != shared.digest(__file__) or
            cfg.get('actor_sha256') != shared.digest(actor.__file__) or
            cfg.get('shared_helpers_sha256') != shared.digest(shared.__file__)):
        raise ValueError('preflight code identity mismatch')
    for key in ('root', 'legacy_stage', 'model_manifest_path'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError('preflight path must be absolute')
    required = ({'actor.py', 'plan.json', 'model_files.json', 'checkpoint_receipt.json'} if family == '8b' else
                {'common.py', 'plan.json', 'heldout_plan.json', 'checkpoint_store.py',
                 'efficient_inference.py', 'gate.py', 'memory_gate.py'})
    actor.verify_stage(cfg['legacy_stage'], cfg['legacy_manifest_sha256'], required)
    expected_manifest = (Path(cfg['legacy_stage']) / 'model_files.json' if family == '8b' else
                         Path(cfg['root']) / 'verify32' / 'expected.json')
    if (Path(cfg['model_manifest_path']).resolve() != expected_manifest.resolve() or
            shared.digest(expected_manifest) != cfg['model_manifest_sha256']):
        raise ValueError('preflight model manifest mismatch')
    item = cfg['conditions'][condition]
    if not isinstance(item.get('adapter_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['adapter_sha256']):
        raise ValueError('preflight adapter identity missing')
    if (not isinstance(item.get('adapter_config_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', item['adapter_config_sha256'])):
        raise ValueError('preflight adapter config identity missing')
    return plan


def exercise(torch, tokenizer, model, generation_class):
    """Force 256 decode steps on fixed synthetic input and inspect raw logits."""
    from peft import PeftModel
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Synthetic test: continue briefly.'}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda:0')
    ids = inputs['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > INPUT_TOKENS:
        raise ValueError('invalid synthetic prompt shape')
    filler_ids = tokenizer(' x', add_special_tokens=False)['input_ids']
    if not filler_ids or filler_ids[0] in tokenizer.all_special_ids:
        raise ValueError('synthetic filler token invalid')
    fill = torch.full((1, INPUT_TOKENS - ids.shape[1]), filler_ids[0], device=ids.device, dtype=ids.dtype)
    extended = torch.cat((ids, fill), dim=1)
    inputs = {'input_ids': extended, 'attention_mask': torch.ones_like(extended)}
    # PEFT delegates generate to its underlying CausalLM; base-model forward
    # alone returns hidden states, not logits, so hook the actual CausalLM.
    target = model.get_base_model() if isinstance(model, PeftModel) else model
    calls = 0
    def check_logits(_module, _args, output):
        nonlocal calls
        logits = getattr(output, 'logits', None)
        if not isinstance(logits, torch.Tensor) or not torch.isfinite(logits).all().item():
            raise ValueError('nonfinite or missing raw generation logits')
        calls += 1
    options = generation_class(max_new_tokens=actor.MAX_NEW_TOKENS, min_new_tokens=actor.MAX_NEW_TOKENS,
                               do_sample=False, num_beams=1, repetition_penalty=1.0,
                               eos_token_id=actor.EOS, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    hook = target.register_forward_hook(check_logits)
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            output = model.generate(**inputs, generation_config=options)
    finally:
        hook.remove()
    generated = output[0, INPUT_TOKENS:].tolist()
    if len(generated) != actor.MAX_NEW_TOKENS or calls < actor.MAX_NEW_TOKENS:
        raise ValueError('synthetic prefill or forced decode did not complete')
    return {'input_tokens': INPUT_TOKENS, 'forced_output_tokens': len(generated),
            'raw_logits_finite': True, 'raw_forward_calls': calls}


def check_prior_jobs(cfg, family):
    if family != '32b':
        return {}
    checked = {}
    for key, prefix in (('original_verify_job', 'SYC_VERIFY'),
                        ('original_memory_preflight_job', 'SYC_MEMORY')):
        job = str(cfg[key])
        # The stage launcher queries sacct on the host and passes the job ID
        # and state into Apptainer; sacct need not be installed in the image.
        if (not job.isdigit() or os.environ.get(prefix + '_JOB') != job
                or os.environ.get(prefix + '_STATE') != 'COMPLETED|0:0'):
            raise ValueError('prior verification job is not completed: ' + job)
        checked[key] = job
    return checked


def run(plan_path, plan_sha, output):
    plan = validate_plan(plan_path, plan_sha)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise ValueError('preflight requires a Slurm job')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    cfg, family, condition = plan['config'], plan['family'], plan['condition']
    shared.save_new(directory / 'IDENTITY.json', {'plan_sha256': plan_sha, 'producer_sha256': shared.digest(__file__),
                    'actor_sha256': cfg['actor_sha256'], 'family': family, 'condition': condition, 'job_id': job})
    import torch
    from transformers import GenerationConfig
    actual = {name: importlib.metadata.version(name) for name in actor.VERSIONS}
    if actual != actor.VERSIONS:
        raise ValueError('GPU preflight runtime versions differ')
    prior_jobs = check_prior_jobs(cfg, family)
    count = 1 if family == '8b' else 2
    if torch.cuda.device_count() != count:
        raise ValueError('GPU preflight has the wrong allocation')
    for i in range(count):
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    torch.manual_seed(16)
    context = (actor.legacy(cfg['legacy_stage'], 'efficient_inference').use_v100_efficient_aya_inference()
               if family == '32b' else nullcontext())
    started = time.monotonic()
    with context as attention:
        tokenizer, model = (actor.load_32b(cfg, condition) if family == '32b' else actor.load_8b(cfg, condition))
        if tokenizer.pad_token_id is None:
            raise ValueError('tokenizer pad token missing')
        checks = exercise(torch, tokenizer, model, GenerationConfig)
        if family == '32b' and (not attention.get('prefill_calls') or not attention.get('decode_calls')):
            raise ValueError('efficient attention did not cover prefill and decode')
    for i in range(count):
        torch.cuda.synchronize(i)
    peaks = [int(torch.cuda.max_memory_reserved(i)) for i in range(count)]
    if any(x <= 0 for x in peaks):
        raise ValueError('missing GPU peak memory')
    validate_plan(plan_path, plan_sha)
    receipt = {'schema': 'syc-gpu-preflight-v1', 'status': 'passed', 'job_id': job,
               'family': family, 'condition': condition, 'gpu_count': count,
               'stage_manifest_sha256': cfg['legacy_manifest_sha256'],
               'model_manifest_sha256': cfg['model_manifest_sha256'], 'actor_sha256': cfg['actor_sha256'],
               'producer_sha256': shared.digest(__file__), 'preflight_plan_sha256': plan_sha,
               'adapter_sha256': cfg['conditions'][condition]['adapter_sha256'],
               'adapter_config_sha256': cfg['conditions'][condition]['adapter_config_sha256'],
               'prior_jobs': prior_jobs, 'versions': actual,
               'peak_reserved_bytes': peaks, 'attention_stats': dict(attention) if family == '32b' else None,
               'elapsed_seconds': round(time.monotonic() - started, 3), **checks}
    shared.save_new(directory / 'receipt.json', receipt)
    return receipt


def validate_receipt(plan_path, plan_sha, receipt_path, job_id, slurm_state):
    """Validate receipt against its frozen plan and separately read Slurm state."""
    plan = validate_plan(plan_path, plan_sha)
    if not isinstance(job_id, str) or not job_id.isdigit() or slurm_state != 'COMPLETED|0:0':
        raise ValueError('preflight Slurm accounting did not pass')
    receipt = json.loads(Path(receipt_path).read_text())
    cfg, family, condition = plan['config'], plan['family'], plan['condition']
    expected = {'schema': 'syc-gpu-preflight-v1', 'status': 'passed', 'job_id': job_id,
                'family': family, 'condition': condition, 'gpu_count': 1 if family == '8b' else 2,
                'stage_manifest_sha256': cfg['legacy_manifest_sha256'],
                'model_manifest_sha256': cfg['model_manifest_sha256'], 'actor_sha256': cfg['actor_sha256'],
                'producer_sha256': shared.digest(__file__), 'preflight_plan_sha256': plan_sha,
                'adapter_sha256': cfg['conditions'][condition]['adapter_sha256'],
                'adapter_config_sha256': cfg['conditions'][condition]['adapter_config_sha256'],
                'prior_jobs': ({k: str(cfg[k]) for k in ('original_verify_job', 'original_memory_preflight_job')}
                               if family == '32b' else {}), 'versions': actor.VERSIONS,
                'input_tokens': INPUT_TOKENS, 'forced_output_tokens': actor.MAX_NEW_TOKENS,
                'raw_logits_finite': True}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('preflight receipt identity mismatch')
    if (type(receipt.get('raw_forward_calls')) is not int or receipt['raw_forward_calls'] < actor.MAX_NEW_TOKENS
            or not isinstance(receipt.get('peak_reserved_bytes'), list)
            or len(receipt['peak_reserved_bytes']) != expected['gpu_count']
            or any(type(x) is not int or x <= 0 for x in receipt['peak_reserved_bytes'])):
        raise ValueError('preflight stress or memory evidence incomplete')
    if family == '32b':
        attention = receipt.get('attention_stats')
        if (not isinstance(attention, dict) or type(attention.get('prefill_calls')) is not int
                or type(attention.get('decode_calls')) is not int
                or attention['prefill_calls'] < 1 or attention['decode_calls'] < 1):
            raise ValueError('efficient attention preflight evidence incomplete')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        plan = validate_plan(args.plan, args.plan_sha256)
        print(json.dumps({'status': 'validated_without_gpu', 'family': plan['family'], 'condition': plan['condition']}))
        return
    print(json.dumps(run(args.plan, args.plan_sha256, args.out)))


if __name__ == '__main__':
    main()
