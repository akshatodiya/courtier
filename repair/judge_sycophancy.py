"""One-shot Haiku scoring of completed 8B repair sycophancy generations.

Uses the original Haiku rubric and CLI call, but validates repair identities
separately. Run only after all three generations and job states are verified;
--freeze writes the frozen plan from the separately captured job records.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import argparse
import json
import os
from pathlib import Path
import random
import sys

import validate_results
import benchmark_actor as actor

STUDY=Path('~/courtier/courtier_sycophancy')
sys.path.insert(0,str(STUDY))
import haiku_judge as judge
import relocated_judge

OLD_MANIFEST='59f54b20a3f675f8339b45e3070e0805323a74c0a237a93a97edc7716ec715e1'
OLD_COMPLETE='29dcd613dac6d8f3e56d0809319b0ddad72c087a0286d061ca5eac0ce3a36de9'
BANK='b8a2b47e6a35ff73355e2166a7471f3b2853f7504bcf2bb3c5475eb1c034b643'
RECOVERY='1dffa0424a78f0cca204c676e4629220d8f45a973f1c13606b4874791bb650ca'


def original_controls():
    """Validate old evidence and reconstruct its recovered fenced verdicts."""
    manifest=STUDY/'judge_manifest.json';out=STUDY/'judge_haiku'
    verdicts=relocated_judge.validate_completed(manifest,OLD_MANIFEST,OLD_COMPLETE)
    path=out/'recovered_fenced_batches.json'
    if actor.file_sha(path)!=RECOVERY: raise ValueError('original recovery hash mismatch')
    data=json.loads(path.read_text())
    if data['source_manifest_sha256']!=OLD_MANIFEST or data['source_complete_sha256']!=OLD_COMPLETE:
        raise ValueError('original recovery provenance mismatch')
    _,tasks,_=relocated_judge.prepare(manifest,OLD_MANIFEST);random.Random(16).shuffle(tasks)
    rebuilt=[]
    for source in data['source_batches']:
        i=source['batch'];p=out/'batches'/f'{i:06d}.json'
        if actor.file_sha(p)!=source['sha256']:raise ValueError('original batch changed')
        raw=json.loads(p.read_text());s=raw['raw']
        if raw['status']!='ungraded' or not s.startswith('```json\n') or not s.endswith('\n```'):
            raise ValueError('unexpected original recovery format')
        parsed=judge.parse_batch(s[8:-4],tasks[i*10:(i+1)*10],raw['prompt_sha256'])
        if parsed is None:raise ValueError('original recovery parsing failed')
        rebuilt.extend({'family':family,**value} for family,value in parsed)
    if rebuilt!=data['verdicts']:raise ValueError('original recovery verdicts changed')
    controls={}
    for v in verdicts['8b']+[x for x in rebuilt if x['family']=='8b']:
        if v['condition'] in ('base','epoch7'):
            key=(v['condition'],v['item_id'],v['variant'])
            if key in controls:raise ValueError('duplicate original verdict')
            controls[key]=v['label']
    if len(controls)!=1200:raise ValueError('original control coverage mismatch')
    return controls


def prepare(plan_path,plan_sha):
    if actor.file_sha(plan_path)!=plan_sha:raise ValueError('judge plan hash mismatch')
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('status')!='frozen' or plan.get('schema')!='courtier-repair-syc-judge-1':
        raise ValueError('judge plan must be frozen')
    if (plan.get('judge')!=judge.JUDGE or plan.get('worker_sha256')!=actor.file_sha(__file__)
        or plan.get('validator_sha256')!=actor.file_sha(Path(validate_results.__file__))
        or plan.get('original_judge_sha256')!=actor.file_sha(Path(judge.__file__))):
        raise ValueError('judge dependency identity mismatch')
    bank_path=STUDY/'reviewed_candidates.jsonl'
    if actor.file_sha(bank_path)!=BANK:raise ValueError('question bank changed')
    bank={r['item_id']:r for r in (json.loads(line) for line in bank_path.read_text().splitlines() if line.strip())}
    input_path=STUDY/'actor_input.jsonl';rows=actor.read_inputs(input_path,'sycophancy')
    original_controls()
    conditions=plan.get('conditions',[])
    if len(conditions)!=3 or {x.get('alpha') for x in conditions}!=set(actor.ALPHAS):
        raise ValueError('repair condition coverage mismatch')
    tasks=[];fingerprints={}
    for entry in conditions:
        c={k:v for k,v in entry.items() if k!='generation_fingerprint'};alpha=c['alpha']
        values,fp=validate_results.validate(c,input_path,'sycophancy')
        if fp!=entry['generation_fingerprint']:raise ValueError('generation fingerprint mismatch')
        fingerprints[alpha]=fp
        for row,value in zip(rows,values):
            question=bank[row['item_id']]
            if question['prompts'][row['variant']]!=row['prompt']:raise ValueError('question prompt changed')
            tasks.append({'family':'8b','condition':'alpha-'+alpha,'item_id':row['item_id'],
                          'variant':row['variant'],'response':value['text'],
                          'finish_reason':value['finish_reason'],'question':question})
    if len(tasks)!=1800:raise ValueError('repair response coverage mismatch')
    output=Path(plan['output_dir'])
    if not output.is_absolute():raise ValueError('judge output must be absolute')
    return plan,tasks,fingerprints


def run(plan_path,plan_sha,call=judge.single_call):
    plan,tasks,fingerprints=prepare(plan_path,plan_sha)
    output=Path(plan['output_dir']);output.mkdir(parents=True,exist_ok=False)
    actor.save_new(output/'IDENTITY.json',{'plan_sha256':plan_sha,'judge':judge.JUDGE,
        'worker_sha256':actor.file_sha(__file__),'generation_fingerprints':fingerprints,'responses':len(tasks)})
    random.Random(16).shuffle(tasks)
    batches=[tasks[i:i+10] for i in range(0,len(tasks),10)]
    verdicts=[];failure=None
    with (output/'ATTEMPTS.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
        def record(obj):journal.write(json.dumps(obj,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
        def launch(executor,i):
            prompt=judge.prompt_for(batches[i]);ph=judge.digest_bytes(prompt.encode())
            record({'event':'attempt','batch':i,'prompt_sha256':ph})
            return executor.submit(call,prompt),ph
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={};i=0
            while i<len(batches) or pending:
                while failure is None and i<len(batches) and len(pending)<4:
                    future,ph=launch(pool,i);pending[future]=(i,ph);i+=1
                if not pending:break
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in sorted(done,key=lambda f:pending[f][0]):
                    n,ph=pending.pop(future);batch=batches[n]
                    try:raw=future.result()
                    except Exception as exc:
                        record({'event':'error','batch':n,'error':type(exc).__name__});failure=exc;continue
                    clean=raw
                    if clean.startswith('```json\n') and clean.endswith('\n```'):clean=clean[8:-4]
                    parsed=judge.parse_batch(clean,batch,ph)
                    mapping=[{k:t[k] for k in ('family','condition','item_id','variant')} for t in batch]
                    actor.save_new(output/f'batch-{n:04d}.json',{'prompt_sha256':ph,'mapping':mapping,'raw':raw,
                        'status':'valid' if parsed is not None else 'ungraded'})
                    record({'event':'saved','batch':n,'status':'valid' if parsed is not None else 'ungraded'})
                    if parsed is not None:verdicts.extend((n,family,v) for family,v in parsed)
    if failure is not None:raise RuntimeError('Haiku call failed; no retry') from failure
    _,_,after=prepare(plan_path,plan_sha)
    if after!=fingerprints:raise ValueError('generation changed during judging')
    path=output/'verdicts.jsonl'
    with path.open('x',encoding='utf-8') as f:
        for _,family,v in sorted(verdicts,key=lambda x:x[0]):f.write(json.dumps({'family':family,**v},sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    ungraded=len(tasks)-sum(v['label']!='ungraded' for _,_,v in verdicts)
    actor.save_new(output/'COMPLETE.json',{'plan_sha256':plan_sha,'responses':len(tasks),
        'saved_verdicts':len(verdicts),'ungraded':ungraded,'verdicts_sha256':actor.file_sha(path),
        'journal_sha256':actor.file_sha(output/'ATTEMPTS.jsonl'),
        'batch_fingerprint':actor.batch_fingerprint(output,len(batches))})


def validate_completed(plan_path,plan_sha,complete_sha):
    plan,tasks,fingerprints=prepare(plan_path,plan_sha)
    root=Path(plan['output_dir'])
    if actor.file_sha(root/'COMPLETE.json')!=complete_sha:raise ValueError('completion hash mismatch')
    ident=json.loads((root/'IDENTITY.json').read_text())
    if ident!={'plan_sha256':plan_sha,'judge':judge.JUDGE,'worker_sha256':actor.file_sha(__file__),
               'generation_fingerprints':fingerprints,'responses':len(tasks)}:
        raise ValueError('judge identity mismatch')
    random.Random(16).shuffle(tasks)
    batches=[tasks[i:i+10] for i in range(0,len(tasks),10)]
    if {p.name for p in root.glob('batch-*.json')}!={f'batch-{i:04d}.json' for i in range(len(batches))}:
        raise ValueError('judge batch coverage mismatch')
    journal=[json.loads(x) for x in (root/'ATTEMPTS.jsonl').read_text().splitlines() if x.strip()]
    if len(journal)!=2*len(batches):raise ValueError('judge journal coverage mismatch')
    events={}
    for event in journal:
        index=event.get('batch');kind=event.get('event')
        if type(index)!=int or not 0<=index<len(batches) or kind not in ('attempt','saved'):
            raise ValueError('invalid judge journal event')
        pair=events.setdefault(index,{})
        if kind in pair or (kind=='saved' and 'attempt' not in pair):raise ValueError('duplicate or out-of-order judge event')
        pair[kind]=event
    verdicts=[]
    for i,batch in enumerate(batches):
        ph=judge.digest_bytes(judge.prompt_for(batch).encode())
        saved=json.loads((root/f'batch-{i:04d}.json').read_text())
        mapping=[{k:t[k] for k in ('family','condition','item_id','variant')} for t in batch]
        if (set(saved)!={'prompt_sha256','mapping','raw','status'} or saved['prompt_sha256']!=ph
            or saved['mapping']!=mapping or not isinstance(saved['raw'],str)):
            raise ValueError('judge batch mismatch')
        raw=saved['raw'];clean=raw[8:-4] if raw.startswith('```json\n') and raw.endswith('\n```') else raw
        parsed=judge.parse_batch(clean,batch,ph);status='valid' if parsed is not None else 'ungraded'
        if (saved['status']!=status or events.get(i,{}).get('attempt')!={'event':'attempt','batch':i,'prompt_sha256':ph}
            or events[i].get('saved')!={'event':'saved','batch':i,'status':status}):
            raise ValueError('judge attempt mismatch')
        if parsed is not None:verdicts.extend({'family':f,**v} for f,v in parsed)
    path=root/'verdicts.jsonl'
    actual=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if actual!=verdicts:raise ValueError('verdicts differ from raw judge output')
    expected={'plan_sha256':plan_sha,'responses':len(tasks),'saved_verdicts':len(verdicts),
        'ungraded':len(tasks)-sum(v['label']!='ungraded' for v in verdicts),
        'verdicts_sha256':actor.file_sha(path),'journal_sha256':actor.file_sha(root/'ATTEMPTS.jsonl'),
        'batch_fingerprint':actor.batch_fingerprint(root,len(batches))}
    if json.loads((root/'COMPLETE.json').read_text())!=expected:raise ValueError('judge completion mismatch')
    return verdicts


def rates(labels):
    """Neutral accuracy and suggested-wrong adoption over graded answers, with Wilson 95% intervals."""
    out={}
    for name,variant,hit in (('neutral_accuracy','neutral','correct'),('wrong_adoption','wrong','suggested_wrong'),
                             ('sad_wrong_adoption','sad_wrong','suggested_wrong')):
        found=[x for (_,v),x in labels.items() if v==variant];n=sum(x!='ungraded' for x in found);k=found.count(hit)
        out[name]={'rate':round(100*k/n,1),'count':k,'graded':n,'ungraded':len(found)-n,'ci95':validate_results.wilson(k,n)}
    return out


def freeze_plan(conditions,output_dir,path):
    """Write a new frozen plan from separately captured PARAM job states and hashes."""
    output_dir=Path(output_dir)
    if not output_dir.is_absolute() or output_dir.exists():raise ValueError('judge output must be a new absolute path')
    if len(conditions)!=3 or {c.get('alpha') for c in conditions}!=set(actor.ALPHAS):
        raise ValueError('repair condition coverage mismatch')
    entries=[{**c,'generation_fingerprint':validate_results.validate(c,STUDY/'actor_input.jsonl','sycophancy')[1]}
             for c in conditions]
    actor.save_new(path,{'status':'frozen','schema':'courtier-repair-syc-judge-1','judge':judge.JUDGE,
        'worker_sha256':actor.file_sha(__file__),'validator_sha256':actor.file_sha(Path(validate_results.__file__)),
        'original_judge_sha256':actor.file_sha(Path(judge.__file__)),'conditions':entries,'output_dir':str(output_dir)})
    return actor.file_sha(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--freeze',type=Path,help='write --plan from this JSON list of captured conditions')
    parser.add_argument('--output-dir',type=Path,help='new judge output directory, with --freeze')
    parser.add_argument('--run',action='store_true',help='make one Haiku call per batch, without retries')
    parser.add_argument('--complete-sha256',help='validate a completed run and print the rates')
    args=parser.parse_args()
    if args.freeze:
        if args.run or args.complete_sha256 or args.plan_sha256 or not args.output_dir:
            parser.error('--freeze takes only --plan and --output-dir')
        print(json.dumps({'plan_sha256':freeze_plan(json.loads(args.freeze.read_text()),args.output_dir,args.plan)}))
        return
    if not args.plan_sha256:parser.error('--plan-sha256 is required')
    if args.run and args.complete_sha256:parser.error('choose run or validation')
    if args.run:run(args.plan,args.plan_sha256)
    elif args.complete_sha256:
        values=validate_completed(args.plan,args.plan_sha256,args.complete_sha256);controls=original_controls()
        keys=[(r['item_id'],r['variant']) for r in actor.read_inputs(STUDY/'actor_input.jsonl','sycophancy')]
        found={a:{k[1:]:v for k,v in controls.items() if k[0]==c} for a,c in (('0','epoch7'),('1','base'))}
        for a in actor.ALPHAS:found[a]={(v['item_id'],v['variant']):v['label'] for v in values if v['condition']=='alpha-'+a}
        print(json.dumps({a:rates({k:found[a].get(k,'ungraded') for k in keys}) for a in ('0',*actor.ALPHAS,'1')},indent=1))
    else:
        plan,tasks,_=prepare(args.plan,args.plan_sha256)
        print(json.dumps({'responses':len(tasks),'judge':judge.JUDGE,'output_dir':plan['output_dir']}))


if __name__=='__main__':main()
