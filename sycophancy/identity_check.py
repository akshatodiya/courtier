"""Check completed actor identity against a pinned plan and independent study identities.

The frozen global study must name ``actor_sha256`` per family and
``model_identities`` per family and condition. These expectations are to be
assembled from independently verified model inventories *before* generation;
an actor's self-reported configuration is not an independent expectation.

This checks identity records and frozen plans, not the remote hardware or the
authenticity of a file produced by a dishonest actor. The actor source must be
reviewed and its hash included in the frozen study. No model is loaded here.
"""
import hashlib
import json
from pathlib import Path
import re

CONDITIONS = {'tiny': {'base', 'run2_final', 'run3b_final', 'run4_v3_epoch1', 'run4_v3_epoch2', 'run4_v3_epoch3'},
              '8b': {'base', 'epoch1', 'epoch3', 'epoch5', 'epoch7', 'epoch9'},
              '32b': {'base', 'epoch1', 'epoch3', 'epoch5', 'epoch7', 'epoch9'}}


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('invalid expected hash')
    return value


def _files(value, required):
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError('incomplete model file hashes')
    for name, digest in value.items():
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid model file name')
        _hash(digest)
    return value


def verify(study, family, condition, plan_path, plan_sha256, actor_sha256, input_sha256, identity):
    """Fail closed unless a completed actor maps to the declared model condition.

    The caller must independently verify the frozen study hash, completion
    receipt and all response records. This function checks the model/plan
    linkage for one condition; it intentionally does not score any response.
    """
    _hash(plan_sha256); _hash(actor_sha256); _hash(input_sha256)
    if study.get('status') != 'frozen' or family not in ('tiny', '8b', '32b'):
        raise ValueError('study or family is not frozen and supported')
    if condition not in CONDITIONS[family] or condition not in study.get('conditions', {}).get(family, []):
        raise ValueError('condition is not in study')
    actor_expected = study.get('actor_sha256', {}).get(family)
    if isinstance(actor_expected, dict):
        actor_expected = actor_expected.get(condition)
    if actor_expected != actor_sha256:
        raise ValueError('actor differs from independent study identity')
    expected = study.get('model_identities', {}).get(family, {}).get(condition)
    if not isinstance(expected, dict):
        raise ValueError('independent model identity missing')
    path = Path(plan_path)
    if not path.is_absolute():
        raise ValueError('actor plan path must be absolute')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha256:
        raise ValueError('actor plan hash mismatch')
    plan = json.loads(raw)
    if (plan.get('status') != 'frozen' or plan.get('item_count') != study.get('question_count')
            or plan.get('records_sha256') != input_sha256 or identity.get('condition') != condition
            or identity.get('plan_sha256') != plan_sha256 or identity.get('input_sha256') != input_sha256
            or identity.get('actor_sha256') != actor_sha256):
        raise ValueError('actor, input or plan identity mismatch')

    if family == 'tiny':
        if plan.get('schema') != 'syc-actor-v1':
            raise ValueError('wrong Tiny actor plan schema')
        cfg = plan.get('conditions', {}).get(condition)
        if not isinstance(cfg, dict) or identity.get('model') != cfg:
            raise ValueError('Tiny recorded model differs from plan')
        kind = 'base' if condition == 'base' else ('merged' if condition in ('run2_final', 'run3b_final') else 'adapter')
        if cfg.get('model_kind') != kind or expected.get('model_kind') != kind:
            raise ValueError('Tiny model kind mismatch')
        required = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'}
        required |= ({'model.safetensors', 'chat_template.jinja'} if kind == 'merged' else
                     {'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                      'model-00002-of-00002.safetensors', 'special_tokens_map.json'})
        if _files(cfg.get('files_sha256'), required) != _files(expected.get('files_sha256'), required):
            raise ValueError('Tiny model file identity mismatch')
        if kind == 'adapter':
            req = {'adapter_config.json', 'adapter_model.safetensors'}
            if _files(cfg.get('adapter_files_sha256'), req) != _files(expected.get('adapter_files_sha256'), req):
                raise ValueError('Tiny adapter identity mismatch')
        elif cfg.get('adapter_files_sha256') is not None or expected.get('adapter_files_sha256') is not None:
            raise ValueError('unexpected Tiny adapter')
    else:
        if plan.get('schema') != 'syc-param-v1' or identity.get('family') != family:
            raise ValueError('PARAM family or plan schema mismatch')
        cfg = plan.get('families', {}).get(family)
        if not isinstance(cfg, dict) or identity.get('model') != cfg:
            raise ValueError('PARAM recorded model differs from plan')
        entry = cfg.get('conditions', {}).get(condition)
        if not isinstance(entry, dict) or cfg.get('actor_sha256') != actor_sha256:
            raise ValueError('PARAM condition or actor mismatch')
        if (_hash(expected.get('model_manifest_sha256')) != _hash(cfg.get('model_manifest_sha256'))
                or identity.get('model_manifest_sha256') != expected['model_manifest_sha256']):
            raise ValueError('PARAM base model identity mismatch')
        if condition == 'base':
            if any(value is not None for value in (entry.get('adapter_sha256'),
                   expected.get('adapter_sha256'), identity.get('adapter_sha256'),
                   entry.get('adapter_config_sha256'), expected.get('adapter_config_sha256'))):
                raise ValueError('base condition contains an adapter')
        else:
            if (_hash(expected.get('adapter_sha256')) != _hash(entry.get('adapter_sha256'))
                    or identity.get('adapter_sha256') != expected['adapter_sha256']):
                raise ValueError('PARAM adapter weights mismatch')
            if _hash(expected.get('adapter_config_sha256')) != _hash(entry.get('adapter_config_sha256')):
                raise ValueError('PARAM adapter config mismatch')
            # The current 32B actor neither pins nor records the verified config.
            # Require it to do so before claiming a complete model identity.
            if family == '32b' and identity.get('adapter_config_sha256') != expected['adapter_config_sha256']:
                raise ValueError('32B actor must record verified adapter config hash')
    return True
