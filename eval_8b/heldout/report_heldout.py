"""Verify and report saved held-out results; no model, SSH or judge calls."""
import argparse,json,random
from pathlib import Path
import controller_heldout as c
import exchange_heldout as w

def interval(diffs):
    if not diffs:return None
    rng=random.Random(16);n=len(diffs)
    x=sorted(sum(rng.choice(diffs) for _ in range(n))/n for _ in range(10000))
    return [x[250],x[9750]]
def pairs(a,b):
    ids=sorted(set(a)&set(b));d=[a[i]-b[i] for i in ids if a[i] is not None and b[i] is not None]
    return {'paired':len(d),'difference':sum(d)/len(d) if d else None,'ci95':interval(d)}
def collect(path,case,condition,suite,identity,runner,variant=None):
    if not path.exists():return None
    rec=json.loads(path.read_text())
    if rec.get('case_id')!=case['id'] or rec.get('condition')!=condition or rec.get('identity')!=identity or rec.get('suite')!=suite or rec.get('case_hash')!=w.digest(case):raise ValueError('result identity mismatch: '+str(path))
    if variant is not None and rec.get('variant')!=variant:raise ValueError('social variant mismatch')
    judgment=rec.get('judgment',{})
    if not judgment.get('valid'):return None
    if suite=='social':
        if rec.get('status')!='complete':raise ValueError('valid judge on incomplete episode')
        prompt=runner.judge_prompt(case,runner.judge_transcript(rec),runner.ending_text(rec))
    else:
        first=rec.get('first',{}).get('text','');answer=rec.get('followup',{}).get('text','') if suite=='advice' else first
        if not first.strip() or not answer.strip():raise ValueError('valid judge on empty response')
        prompt=c.direct_prompt(case,suite,first,answer)
    if judgment.get('prompt_sha256')!=runner.sha(prompt):raise ValueError('judge prompt identity mismatch')
    valid=[a for a in judgment.get('attempts',[]) if a.get('valid')]
    if len(valid)!=1:raise ValueError('expected one first-valid judgment')
    parsed=runner.validate_judgment(valid[0]['raw']) if suite=='social' else c.parse_direct(valid[0]['raw'],suite,runner)
    if parsed!=judgment.get('parsed'):raise ValueError('stored judgment parse mismatch')
    return parsed

def evaluate(root,identity,banks,runner):
    advice,laws,scenes=banks;conditions=('base',)+tuple(f'epoch{i}' for i in range(1,10));scores={};result={'scope':'custom development diagnostics; not an independent final test','identity':identity,'conditions':{},'paired_comparisons':{}}
    for model in conditions:
        vals={k:{} for k in ('advice_update','advice_plan','advice_failure','law_score','social_no_hint','social_with_hint')}
        for case in advice:
            j=collect(root/model/'advice'/(case['id']+'.json'),case,model,'advice',identity,runner)
            for metric,field in [('advice_update','update_recognition'),('advice_plan','revised_plan'),('advice_failure','critical_failure')]:vals[metric][case['id']]=None if j is None or j[field] is None else int(j[field])
        for case in laws:
            j=collect(root/model/'laws'/(case['id']+'.json'),case,model,'laws',identity,runner)
            vals['law_score'][case['id']]=None if j is None else j['score']
        for pair in scenes:
            for variant in ('no_hint','with_hint'):
                case=pair[variant];j=collect(root/model/'social'/variant/(case['id']+'.json'),case,model,'social',identity,runner,variant)
                vals['social_'+variant][case['id']]=None if j is None else j['agent_'+str(case['focal_role']+1)]['goal']['score']
        scores[model]=vals;result['conditions'][model]={}
        for key,v in vals.items():
            nonnull=[x for x in v.values() if x is not None]
            result['conditions'][model][key]={'expected':len(v),'resolved':len(nonnull),'unknown':len(v)-len(nonnull),'mean':sum(nonnull)/len(nonnull) if nonnull else None}
        result['paired_comparisons'][model+'_hint_effect']=pairs(vals['social_with_hint'],vals['social_no_hint'])
    for model in tuple(f'epoch{i}' for i in range(1,10)):
        for metric in scores[model]:result['paired_comparisons'][model+'_minus_base_'+metric]=pairs(scores[model][metric],scores['base'][metric])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());banks=c.validate_plan(plan);identity=c.sha(a.plan)
    if json.loads((a.run/'PLAN_IDENTITY.json').read_text())!={'identity':identity}:raise ValueError('run plan identity mismatch')
    r=c.load_runner();result=evaluate(a.run,identity,banks,r)
    a.out.mkdir(parents=True,exist_ok=True)
    if any((a.out/n).exists() for n in ('RESULTS.json','REPORT.md')):raise FileExistsError('refusing to overwrite prior report')
    (a.out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Aya 8B held-out development diagnostics','',result['scope'],'',
           '| Condition | Advice update | Advice plan | Critical failures | Law score / 2 | Social goal no-hint / 10 | Social goal with-hint / 10 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for model in ('base',)+tuple(f'epoch{i}' for i in range(1,10)):
        vals=result['conditions'][model]
        def fmt(k):
            v=vals[k];return ('unknown' if v['mean'] is None else '%.3f'%v['mean'])+' (%d/%d)'%(v['resolved'],v['expected'])
        lines.append('| '+model+' | '+' | '.join(fmt(k) for k in ('advice_update','advice_plan','advice_failure','law_score','social_no_hint','social_with_hint'))+' |')
    lines += ['','Pairs and 95% case-bootstrap intervals (10,000 draws, seed 16) are in RESULTS.json; comparisons are exploratory and unadjusted. Unknowns are never imputed. Advice critical-failure mean is the observed failure share; lower is better. Social cases are custom SOTOPIA-style, not official benchmark results.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Saved verified report to',a.out)
if __name__=='__main__':main()
