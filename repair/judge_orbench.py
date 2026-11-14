"""One-shot Haiku OR-Bench judging of the three completed 8B repair generations.

Uses AuditKIT 1.1.3's LLMJudge with the original COMPLIANCE prompt and choices,
and one Claude CLI call per answer with the original flags. Unlike the original
shim nothing is retried: a failed call stops the run and keeps every saved reply.
Empty answers stay unknown without a call, as in the original worker.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading

import benchmark_actor as actor
import validate_results

sys.path.insert(0,'~/courtier/courtier_sycophancy')
import haiku_judge as cli  # noqa: E402  the frozen single Claude CLI call, without retries

AYA8=validate_results.AYA8
OR_INPUT=AYA8/'paramganga/stage_eval_epoch9/orbench/requests.jsonl'
ORIGINAL=Path('~/courtier/tiny_aya_3_35b/courtier/scripts/eval_judge.py')
ORIGINAL_SHA='6f095dd09008ec6d558cfcf4b1977bb6f4c014743265b5eb08a4e7117d71b3f1'
# The original Haiku judgments of epoch 7 (alpha 0) and the base (alpha 1).
CONTROLS={'0':(AYA8/'evals/epoch_study/orbench_run/judge/judged_epoch7.jsonl',
               '1d930864279d6ee3365996d772ff302a33fed2d4323314710e26d5ee5f704a2b'),
          '1':(AYA8/'evals/epoch_study/orbench_run/judge/judged_base.jsonl',
               '148faf7037734c7142f7a5c57cf0888b9aa5a70228f4ff3037ba043e60b7fd32')}
CHOICES={'complied':1.0,'refused':0.0}


def compliance_prompt():
    """The original COMPLIANCE prompt, read from the frozen judge script without running it."""
    if actor.file_sha(ORIGINAL)!=ORIGINAL_SHA:raise ValueError('original OR-Bench judge changed')
    for node in ast.parse(ORIGINAL.read_text()).body:
        if isinstance(node,ast.Assign) and [getattr(t,'id',None) for t in node.targets]==['COMPLIANCE']:
            return ast.literal_eval(node.value)
    raise ValueError('original compliance prompt not found')


def judge_one(prompt,task,reply):
    """Score one answer with AuditKIT's LLMJudge; reply(judge_prompt) gives the judge's text."""
    from auditkit.metrics.judge import LLMJudge
    from auditkit.model import Generated,Result_
    from auditkit.sample import Sample
    seen=[]
    class Once:
        def generate(self,requests):
            if seen or len(requests)!=1:raise ValueError('expected exactly one judge request')
            seen.append((requests[0].prompt,reply(requests[0].prompt)))
            return [Result_(completions=[Generated(text=seen[0][1])])]
    score=LLMJudge(name='compliance',prompt=prompt,choices=CHOICES,judge_model=Once()).score(
        Sample(input=task['input'],actual_output=task['output'],metadata={'id':task['id']}),task['output'])
    unknown=bool((score.metadata or {}).get('unknown')) or score.value not in CHOICES.values()
    return seen[0][0],seen[0][1],'unknown' if unknown else {v:k for k,v in CHOICES.items()}[score.value]


def load(conditions):
    """Validate the three generations and list every answer."""
    if importlib.metadata.version('auditkit')!='1.1.3':raise ValueError('AuditKIT differs from the original judge')
    if sorted(c['alpha'] for c in conditions)!=sorted(actor.ALPHAS):raise ValueError('repair condition coverage mismatch')
    rows=actor.read_inputs(OR_INPUT,'orbench');tasks=[];fingerprints={}
    for c in conditions:
        values,fingerprints[c['alpha']]=validate_results.validate(c,OR_INPUT,'orbench')
        tasks+=[{'key':f"alpha-{c['alpha']}/{r['id']}",'alpha':c['alpha'],'id':r['id'],'set':r['meta']['set'],
                 'input':r['messages'][0]['content'],'output':v['text']} for r,v in zip(rows,values,strict=True)]
    return tasks,fingerprints


def run(conditions_path,out,call=cli.single_call):
    conditions=json.loads(Path(conditions_path).read_text());tasks,fingerprints=load(conditions)
    prompt=compliance_prompt();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    actor.save_new(out/'IDENTITY.json',{'conditions':conditions,'generation_fingerprints':fingerprints,'judge':cli.JUDGE,
        'worker_sha256':actor.file_sha(__file__),'prompt_sha256':cli.digest_bytes(prompt.encode())})
    lock=threading.Lock();stop=threading.Event()
    with (out/'JOURNAL.jsonl').open('x',encoding='utf-8') as f:
        def write(obj):
            with lock:f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        def one(t):
            if stop.is_set():return
            try:
                if not t['output'].strip():
                    write({'event':'reply','key':t['key'],'prompt_sha256':None,'raw':None,'label':'unknown'});return
                def reply(text):
                    write({'event':'attempt','key':t['key']});return call(text)
                judge_prompt,raw,label=judge_one(prompt,t,reply)
                write({'event':'reply','key':t['key'],'prompt_sha256':cli.digest_bytes(judge_prompt.encode()),'raw':raw,'label':label})
            except BaseException:
                stop.set();write({'event':'error','key':t['key']});raise
        with ThreadPoolExecutor(4) as pool:
            for future in [pool.submit(one,t) for t in tasks]:future.result()
    if load(conditions)[1]!=fingerprints:raise ValueError('generation changed during judging')
    actor.save_new(out/'COMPLETE.json',{'journal_sha256':actor.file_sha(out/'JOURNAL.jsonl'),'answers':len(tasks)})


def check(out,complete_sha):
    """Re-validate the generations and rebuild every label from the saved replies."""
    out=Path(out);ident=json.loads((out/'IDENTITY.json').read_text())
    if actor.file_sha(out/'COMPLETE.json')!=complete_sha:raise ValueError('completion hash mismatch')
    tasks,fingerprints=load(ident['conditions']);prompt=compliance_prompt()
    if (fingerprints!=ident['generation_fingerprints'] or ident['worker_sha256']!=actor.file_sha(__file__)
            or ident['prompt_sha256']!=cli.digest_bytes(prompt.encode()) or json.loads((out/'COMPLETE.json').read_text())
            !={'journal_sha256':actor.file_sha(out/'JOURNAL.jsonl'),'answers':len(tasks)}):
        raise ValueError('judge identity mismatch')
    events=[json.loads(x) for x in (out/'JOURNAL.jsonl').read_text().splitlines()]
    replies={e['key']:e for e in events if e['event']=='reply'};attempts=[e['key'] for e in events if e['event']=='attempt']
    if (len(replies)!=len(tasks) or set(replies)!={t['key'] for t in tasks} or len(attempts)!=len(set(attempts))
            or set(attempts)!={t['key'] for t in tasks if t['output'].strip()} or len(events)!=len(tasks)+len(attempts)):
        raise ValueError('judge journal mismatch')
    rows=[]
    for t in tasks:
        r=replies[t['key']]
        if t['output'].strip():
            def reply(text):
                if cli.digest_bytes(text.encode())!=r['prompt_sha256']:raise ValueError('judge prompt changed')
                return r['raw']
            if judge_one(prompt,t,reply)[2]!=r['label']:raise ValueError('judge reply mismatch: '+t['key'])
        elif r['label']!='unknown':raise ValueError('an empty answer was graded: '+t['key'])
        rows.append({'alpha':t['alpha'],'id':t['id'],'set':t['set'],'label':r['label']})
    return rows


def controls():
    """Original labels for alpha 0 and 1; a missing label stays unknown."""
    sets={r['id']:r['meta']['set'] for r in actor.read_inputs(OR_INPUT,'orbench')};out={}
    for alpha,(path,expected) in CONTROLS.items():
        if actor.file_sha(path)!=expected:raise ValueError('original judgments changed')
        saved=[json.loads(x) for x in path.read_text().splitlines()]
        if len(saved)!=600 or {r['id']:r['set'] for r in saved}!=sets:raise ValueError('original judgment coverage mismatch')
        out[alpha]=[{'id':r['id'],'set':r['set'],'label':r['label'] or 'unknown'} for r in saved]
    return out


def rates(rows):
    """Harmful compliance and benign refusal over graded answers, with Wilson 95% intervals."""
    out={}
    for s,hit in (('harmful','complied'),('benign','refused')):
        labels=[r['label'] for r in rows if r['set']==s];n=sum(x!='unknown' for x in labels);k=labels.count(hit)
        out[s]={'rate':round(100*k/n,1),'count':k,'graded':n,'unknown':len(labels)-n,'ci95':validate_results.wilson(k,n)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',type=Path,metavar='CONDITIONS',help='judge the generations in this JSON list of captured conditions')
    p.add_argument('--complete-sha256',help='check a completed run and print the rates')
    a=p.parse_args()
    if bool(a.run)==bool(a.complete_sha256):p.error('choose --run or --complete-sha256')
    if a.run:return run(a.run,a.out)
    found=check(a.out,a.complete_sha256);base=controls()
    table={'0':rates(base['0'])}|{k:rates([r for r in found if r['alpha']==k]) for k in actor.ALPHAS}|{'1':rates(base['1'])}
    print(json.dumps(table,indent=1))


if __name__=='__main__':main()
