"""Compare verified repair judgments to epoch 7 on the same questions.

No model or judge calls. Bootstrap resamples paired questions, not answers
from different checkpoints independently. Ungraded pairs are excluded and
reported; intervals omit judge, dataset and checkpoint-selection uncertainty.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import benchmark_actor as actor
import judge_orbench as orb
import judge_sycophancy as syc

REPS=10000
SEED=16


def keyed(rows,key_fields,label_field='label'):
    found={}
    for row in rows:
        key=tuple(row[k] for k in key_fields)
        if key in found:raise ValueError('duplicate judgment key: '+repr(key))
        found[key]=row[label_field]
    return found


def paired(reference, candidate, keys, hit, unknown):
    pairs=[(reference.get(k,unknown),candidate.get(k,unknown)) for k in keys]
    pairs=[(a,b) for a,b in pairs if a!=unknown and b!=unknown]
    if not pairs:
        return {'paired_graded':0,'excluded':len(keys),'reference_count':0,'candidate_count':0,
                'delta_pp':None,'ci95_pp':None,'worst_case_bounds_pp':[-100.0,100.0]}
    d=np.array([int(b==hit)-int(a==hit) for a,b in pairs],dtype=np.int8)
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(d),size=(REPS,len(d)))
    lo,hi=np.percentile(d[idx].mean(axis=1),[2.5,97.5])
    return {'paired_graded':len(pairs),'excluded':len(keys)-len(pairs),
            'reference_count':sum(a==hit for a,b in pairs),'candidate_count':sum(b==hit for a,b in pairs),
            'delta_pp':round(100*float(d.mean()),2),'ci95_pp':[round(100*float(lo),2),round(100*float(hi),2)],
            'worst_case_bounds_pp':[round(100*(int(d.sum())-len(keys)+len(pairs))/len(keys),2),
                                    round(100*(int(d.sum())+len(keys)-len(pairs))/len(keys),2)]}


def analyze(or_rows,or_controls,syc_rows,syc_controls,or_input,syc_input):
    result={}
    valid={(r['item_id'],r['variant']) for r in syc_input}
    all_or={(r['id'],) for r in or_input}
    for alpha in actor.ALPHAS:
        orb_model=keyed([r for r in or_rows if r['alpha']==alpha],['id'])
        syc_model=keyed([v for v in syc_rows if v['condition']=='alpha-'+alpha],['item_id','variant'])
        if set(orb_model)!=all_or or not set(syc_model)<=valid:raise ValueError('repair coverage mismatch')
        result[alpha]={}
        for ref_alpha,ref_name,ref_cond in [('0','epoch7','epoch7'),('1','base','base')]:
            orb_ref=keyed(or_controls[ref_alpha],['id'])
            syc_ref={(item,variant):label for (cond,item,variant),label in syc_controls.items() if cond==ref_cond}
            if set(orb_ref)!=all_or or set(syc_ref)!=valid:raise ValueError('control coverage mismatch')
            metrics={}
            for name,kind,label in [('harmful_compliance','harmful','complied'),('benign_refusal','benign','refused')]:
                keys=[(r['id'],) for r in or_input if r['meta']['set']==kind]
                metrics[name]=paired(orb_ref,orb_model,keys,label,'unknown')
            for name,variant,label in [('neutral_accuracy','neutral','correct'),('wrong_adoption','wrong','suggested_wrong'),
                                       ('sad_wrong_adoption','sad_wrong','suggested_wrong')]:
                keys=[(r['item_id'],r['variant']) for r in syc_input if r['variant']==variant]
                metrics[name]=paired(syc_ref,syc_model,keys,label,'ungraded')
            result[alpha][ref_name]=metrics
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--orbench-out',type=Path,required=True);p.add_argument('--orbench-complete-sha256',required=True)
    p.add_argument('--syc-plan',type=Path,required=True);p.add_argument('--syc-plan-sha256',required=True)
    p.add_argument('--syc-complete-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not a.out.is_absolute() or a.out.exists():raise ValueError('output must be a fresh absolute path')
    or_rows=orb.check(a.orbench_out,a.orbench_complete_sha256);or_controls=orb.controls()
    syc_rows=syc.validate_completed(a.syc_plan,a.syc_plan_sha256,a.syc_complete_sha256);syc_controls=syc.original_controls()
    or_input=actor.read_inputs(orb.OR_INPUT,'orbench');syc_input=actor.read_inputs(syc.STUDY/'actor_input.jsonl','sycophancy')
    result=analyze(or_rows,or_controls,syc_rows,syc_controls,or_input,syc_input)
    actor.save_new(a.out,{'reference':'8B epoch7 (alpha 0) and base (alpha 1)','method':'paired question bootstrap',
                         'resamples':REPS,'seed':SEED,
                         'orbench_complete_sha256':a.orbench_complete_sha256,
                         'syc_plan_sha256':a.syc_plan_sha256,'syc_complete_sha256':a.syc_complete_sha256,
                         'analysis_code_sha256':actor.file_sha(__file__),
                         'orbench_judge_sha256':actor.file_sha(orb.__file__),
                         'sycophancy_judge_sha256':actor.file_sha(syc.__file__),
                         'comparison_direction':'repair minus reference',
                         'limitations':'development questions, one judge, nominal intervals not adjusted for multiple comparisons',
                         'comparisons':result})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
