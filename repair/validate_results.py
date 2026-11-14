"""Validate a completed kit-backed repair generation before a separate judging step."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import benchmark_actor as actor

AYA8=Path('~/courtier/aya_expanse_8b')
MODEL_DIR=AYA8/'models/aya-expanse-8b'
MODEL_MANIFEST=AYA8/'paramganga/stage_eval_epoch9/orbench/model_files.json'
# The model generation defaults recorded by the original epoch-7 OR-Bench run.
ORIGINAL_REPORT=AYA8/'evals/epoch_study/orbench_run/raw/epoch7_report.json'
ORIGINAL_REPORT_SHA='a227b452c45ef2f87260f56fd43722ca1b076275cdf7cde5ef4dd5bd5351d1ea'
TOKENIZER_FILES=('tokenizer.json','tokenizer_config.json','special_tokens_map.json')
# Job states and receipt hashes come from separately captured PARAM accounting.
CONDITION_KEYS={'alpha','output','preflight','preflight_sha256','preflight_job','preflight_job_state',
                'job_id','job_state','complete_sha256'}


def sha(path):
    return actor.file_sha(path)


def wilson(k,n):
    """Wilson 95% interval for k of n, in percent."""
    p=k/n;d=1+1.96**2/n;c=(p+1.96**2/(2*n))/d;w=1.96*math.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/d
    return [round(100*(c-w),1),round(100*(c+w),1)]


def expected_generation_config(suite):
    """The original protocol: OR-Bench call overrides; sycophancy explicit GenerationConfig."""
    if sha(ORIGINAL_REPORT)!=ORIGINAL_REPORT_SHA:raise ValueError('original OR-Bench report changed')
    defaults=json.loads(ORIGINAL_REPORT.read_text())['generation_config']
    _,cap,_=actor.INPUTS[suite]
    fields={'max_new_tokens':cap,'do_sample':False,'num_beams':1,'repetition_penalty':1.0,
            'eos_token_id':actor.EOS,'pad_token_id':defaults['pad_token_id']}
    if suite=='orbench':return {'model_defaults':defaults,'effective_explicit_config':None,'call_overrides':fields}
    if importlib.metadata.version('transformers')!=actor.VERSIONS['transformers']:
        raise ValueError('local transformers version differs from the generation runtime')
    from transformers import GenerationConfig
    return {'model_defaults':defaults,'effective_explicit_config':GenerationConfig(**fields,use_cache=True).to_dict(),
            'call_overrides':None}


def tokenizer():
    """The local base tokenizer, checked against the pinned base manifest."""
    if sha(MODEL_MANIFEST)!=actor.MODEL_MANIFEST_SHA:raise ValueError('base manifest changed')
    files=json.loads(MODEL_MANIFEST.read_text())['files']
    if any(sha(MODEL_DIR/name)!=files[name]['sha256'] for name in TOKENIZER_FILES):
        raise ValueError('local tokenizer differs from the base')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)


def check_decoding(tok, rows, values, suite):
    """Each text must decode from its tokens and each input length must match its prompt."""
    for row,v in zip(rows,values,strict=True):
        messages=row['messages'] if suite=='orbench' else [{'role':'user','content':row['prompt']}]
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        text=tok.decode(v['tokens'],skip_special_tokens=True)
        if (v['input_tokens']!=len(tok(prompt,add_special_tokens=False)['input_ids'])
                or v['text']!=(text.strip() if suite=='sycophancy' else text)):
            raise ValueError('decoded response mismatch: '+v['key'])


def validate(condition, input_path, suite):
    c=condition
    if set(c)!=CONDITION_KEYS or suite not in actor.INPUTS or c['alpha'] not in actor.ALPHAS:
        raise ValueError('invalid condition')
    root=Path(c['output']);preflight=Path(c['preflight'])
    if not root.is_absolute() or not preflight.is_absolute():raise ValueError('paths must be absolute')
    rows=actor.read_inputs(input_path,suite)
    if (not str(c['job_id']).isdigit() or c['job_state']!='COMPLETED|0:0'
            or not str(c['preflight_job']).isdigit() or c['preflight_job_state']!='COMPLETED|0:0'):
        raise ValueError('job is not independently complete')
    if preflight.name!=f"preflight-epoch7-{c['preflight_job']}.json":raise ValueError('preflight receipt name mismatch')
    actor.verify_receipt(preflight,c['preflight_sha256'])
    if sha(root/'COMPLETE.json')!=c['complete_sha256']:raise ValueError('completion hash mismatch')
    ident=json.loads((root/'IDENTITY.json').read_text())
    done=json.loads((root/'COMPLETE.json').read_text())
    _,cap,size=actor.INPUTS[suite]
    expected={'model':'aya-expanse-8b','checkpoint':'epoch7','method':actor.METHOD,
       'alpha_toward_base':actor.ALPHAS[c['alpha']],'suite':suite,'input_sha256':actor.INPUTS[suite][0],
       'adapter_sha256':actor.ADAPTER_SHA,'adapter_config_sha256':actor.CONFIG_SHA,
       'model_manifest_sha256':actor.MODEL_MANIFEST_SHA,'preflight_sha256':c['preflight_sha256'],
       'preflight_job':str(c['preflight_job']),'preflight_state':'COMPLETED|0:0','generation_job':str(c['job_id']),
       'code':actor.code_identity(),'kit':actor.repair.KIT,'versions':actor.VERSIONS,
       'precision':'NF4 double quantization, fp16 compute','attention':'sdpa','seed':16,
       'max_new_tokens':cap,'batch_size':size,'do_sample':False,'num_beams':1,
       'repetition_penalty':1.0,'eos_token_id':actor.EOS}
    if ident!=expected:raise ValueError('generation identity mismatch')
    if json.loads((root/'GENERATION_CONFIG.json').read_text())!=expected_generation_config(suite):
        raise ValueError('generation configuration differs from the original protocol')
    batches=(600+size-1)//size
    if done!={'identity_sha256':sha(root/'IDENTITY.json'),'generation_config_sha256':sha(root/'GENERATION_CONFIG.json'),
              'journal_sha256':sha(root/'ATTEMPTS.jsonl'),'batch_fingerprint':actor.batch_fingerprint(root,batches),
              'completed':600,'expected':600,'batches':batches}:raise ValueError('generation incomplete')
    found={p.name for p in root.glob('batch-*.json')}
    if found!={f'batch-{i:04d}.json' for i in range(batches)}:raise ValueError('batch coverage mismatch')
    journal=[json.loads(s) for s in (root/'ATTEMPTS.jsonl').read_text().splitlines() if s.strip()]
    if len(journal)!=2*batches:raise ValueError('attempt coverage mismatch')
    values=[];fingerprint=hashlib.sha256()
    for name in ('IDENTITY.json','GENERATION_CONFIG.json','ATTEMPTS.jsonl','COMPLETE.json'):
        fingerprint.update((name+':'+sha(root/name)+'\n').encode())
    for i in range(batches):
        group=rows[i*size:(i+1)*size]
        keys=[r['id'] if suite=='orbench' else r['item_id']+'/'+r['variant'] for r in group]
        if journal[2*i:2*i+2]!=[{'event':'attempt','batch':i,'keys':keys},{'event':'saved','batch':i,'keys':keys}]:
            raise ValueError('attempt journal mismatch')
        name=f'batch-{i:04d}.json';p=root/name;batch=json.loads(p.read_text())
        if set(batch)!={'values'} or len(batch['values'])!=len(keys):raise ValueError('batch shape mismatch')
        for key,v in zip(keys,batch['values']):
            if (set(v)!={'key','text','tokens','finish_reason','input_tokens'} or v['key']!=key
                or not isinstance(v['text'],str) or not isinstance(v['tokens'],list)
                or any(type(t)!=int or t<0 for t in v['tokens']) or type(v['input_tokens'])!=int or v['input_tokens']<=0):
                raise ValueError('invalid generated response')
            tokens=v['tokens'];finish='eos' if tokens and tokens[-1]==actor.EOS else ('length' if len(tokens)>=cap else 'other')
            if len(tokens)>cap or actor.EOS in tokens[:-1] or v['finish_reason']!=finish:raise ValueError('finish mismatch')
            values.append(v)
        fingerprint.update((name+':'+sha(p)+'\n').encode())
    check_decoding(tokenizer(),rows,values,suite)
    return values,fingerprint.hexdigest()
