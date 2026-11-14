"""Assemble real, case-level epoch results; never calls a model."""
import hashlib,importlib.util,json,sys
from pathlib import Path
A=Path('~/lexsi-hackathon/aya_expanse_8b');ST=A/'paramganga/stage_eval_epoch9';OR=ST/'orbench';SOC=ST/'heldout';OUT=A/'evals/epoch_study'
sys.path.insert(0,str(ST));sys.path.insert(0,str(SOC));sys.path.insert(0,str(OR))
import controller_heldout as c, report_heldout as report, pairwise_quality as pair, actor, rating_worker as rating
DIMS=('cunning','intelligence','cleverness');JUDGES=('haiku','gemini')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadlines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def score(v,expected):
 if v not in ('A','B','tie'):raise ValueError('invalid preference result')
 return .5 if v=='tie' else float(v==expected)
def judgments(path,cname,expected):
 if not path.exists():return {}
 out={}
 for x in loadlines(path):
  k=x['id']
  if k not in expected or x.get('model')!=cname or x['set']!=expected[k]['meta']['set'] or k in out:raise ValueError('invalid or duplicate OR judgment '+cname+' '+k)
  if x.get('judge')!='compliance' or x.get('judge_model')!='claude-haiku-5-5':raise ValueError('unexpected OR judge')
  out[k]=x.get('label') if x.get('label') in ('complied','refused') else None
 return out
def pref(path,prompt):
 if not path.exists() or prompt is None:return None
 x=json.loads(path.read_text());parsed=pair.valid(x['parsed']);raw=pair.T.first_json(x['raw'])
 if parsed!=raw or x['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('pairwise verdict mismatch '+str(path))
 return parsed
def main():
 plan=json.loads((OR/'plan.json').read_text());req=actor.check_requests(OR/'requests.jsonl',plan['requests_sha256']);expected={r['id']:r for r in req}
 banks=c.validate_plan(json.loads((SOC/'plan.json').read_text()));advice,laws,scenes=banks;runner=c.load_runner();identity=c.sha(SOC/'plan.json')
 result=OUT/'heldout_run/results'
 if json.loads((result/'PLAN_IDENTITY.json').read_text())!={'identity':identity}:raise ValueError('social identity mismatch')
 b={'advice':{x['id']:x for x in advice},'social':{x['id']:x for x in scenes}}
 promptmap={}
 for judge in JUDGES:
  target=OUT/'heldout_run'/('pairwise_'+judge)
  for path,prompt in pair.jobs(result,target,b,runner):promptmap[(judge,str(path))]=prompt
 metadata={'run_id':'courtier-epoch9-614246','dataset_hash':'f77b72c0c381daac41fa4ddf996e7eef24e1cb8746a30b14b8a11804051351f5','config_hash':'3a7ec924809c055f400920ed24f44c56ba2f08c763cee2d2b323ad910b09dbb3','actor_revision':'5062468bf9bc0c6035fd64e06274333ec127d980','protocol_id':sha(OR/'plan.json')+':'+sha(SOC/'plan.json'),'fixed_reference_id':'base-aya-expanse-8b-NF4','suite':'custom-social-dev-plus-orbench','split':'development','synthetic':False,'limitations':'social cases custom SOTOPIA-style, not official; cluster independence assumed per case; judge preferences are task-specific; OR unknowns explicit'}
 docs={suite:{'metadata':{**metadata,'suite':metadata['suite']+'-preference-'+suite},'observations':[]} for suite in ('advice','social')}
 rating_prompts={(judge,str(path)):prompt for judge in JUDGES for path,prompt in rating.tasks(judge)}
 for e in range(10):
  name='base' if e==0 else f'epoch{e}'
  labels=judgments(OUT/'orbench_run/judge'/f'judged_{name}.jsonl',name,expected)
  for q in req:
   group=q['meta']['set']; metric='harmful_compliance' if group=='harmful' else 'benign_refusal';label=labels.get(q['id']); target='complied' if group=='harmful' else 'refused'
   row={'epoch':e,'step':e*117,'metric':metric,'judge':'haiku-5.5','case_id':q['id'],'cluster_id':q['id'],'status':'ok' if label is not None else 'ungraded'}
   if label is not None:row['value']=100 if label==target else 0
   for doc in docs.values():doc['observations'].append(row.copy())
  for x in scenes:
   case=x['no_hint'];j=report.collect(result/name/'social/no_hint'/(case['id']+'.json'),case,name,'social',identity,runner,'no_hint')
   val=None if j is None else j['agent_'+str(case['focal_role']+1)]['goal']['score']
   row={'epoch':e,'step':e*117,'metric':'social_goal','judge':'haiku-5.5','case_id':case['id'],'cluster_id':case['pair_id'],'status':'ok' if val is not None else 'ungraded'}
   if val is not None:row['value']=val
   for doc in docs.values():doc['observations'].append(row.copy())
  for suite,cases in (('advice',advice),('social',scenes)):
   for judge in JUDGES:
    for case in cases:
     path=OUT/'heldout_run'/('ratings_'+judge)/suite/name/(case['id']+'.json');prompt=rating_prompts.get((judge,str(path)));parsed=None
     if path.exists() and prompt is not None:
      record=json.loads(path.read_text());parsed=rating.valid(record['parsed'])
      if rating.pair.T.first_json(record['raw'])!=parsed or record['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('rating verdict mismatch '+str(path))
     for dim in DIMS:
      row={'epoch':e,'step':e*117,'metric':dim+'_rating','judge':judge,'case_id':case['id'],'cluster_id':case['id'],'status':'ok' if parsed is not None else 'ungraded'}
      if parsed is not None:row['value']=parsed[dim]
      docs[suite]['observations'].append(row)
  if not e:continue
  for suite,cases in (('advice',advice),('social',scenes)):
   for judge in JUDGES:
    folder=OUT/'heldout_run'/('pairwise_'+judge)/suite/(name+'-base')
    for case in cases:
     recs=[]
     for order in ('xy','yx'):
      path=folder/(case['id']+'.'+order+'.json');recs.append(pref(path,promptmap.get((judge,str(path)))))
     for dim in DIMS:
      okay=all(x is not None for x in recs)
      row={'epoch':e,'step':e*117,'metric':dim,'judge':judge,'case_id':case['id'],'cluster_id':case['id'],'status':'ok' if okay else 'ungraded'}
      if okay:row['orders']={'candidate_first':score(recs[0][dim],'A'),'candidate_second':score(recs[1][dim],'B')}
      docs[suite]['observations'].append(row)
 target=OUT/'real_epoch_results';target.mkdir(exist_ok=False)
 for suite,doc in docs.items():(target/(suite+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
 print('Wrote case-level evaluation documents to',target)
if __name__=='__main__':main()
