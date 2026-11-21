"""Judge completed epoch comparisons ahead of the full social sweep.

The isolated results are copied into the final comparison directory only if
the main pairwise worker is still waiting for social. No GPU jobs are changed.
"""
import hashlib,json,os,sys,traceback
from pathlib import Path
A=Path('~/courtier/aya_expanse_8b');ST=A/'paramganga/stage_eval_epoch9/heldout';RUN=A/'evals/epoch_study/heldout_run';RESULTS=RUN/'results'
sys.path.insert(0,str(ST));import pairwise_quality as pair
PAIRS=tuple((f'epoch{i}','base') for i in range(1,6))
def save(x):
 p=RUN/'EARLY_GEMINI_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def main():
 lock=RUN/'.early-gemini.lock';lock.mkdir();state={'phase':'preparing','pairs':[x for x,y in PAIRS],'imported':0,'errors':[]}
 try:
  for c in ('base',)+tuple(x for x,y in PAIRS):
   fs=list((RESULTS/c).rglob('*.json'))
   valid=sum(bool(json.loads(f.read_text()).get('judgment',{}).get('valid')) for f in fs)
   if len(fs)!=146 or valid!=146:raise ValueError('condition not complete: '+c)
  pair.PAIRS=PAIRS
  early=RUN/'pairwise_gemini_early';final=RUN/'pairwise_gemini'
  state['phase']='judging';save(state)
  sys.argv=['pairwise_quality.py','--results',str(RESULTS),'--judged',str(early),'--agy','gemini-3.8-flash-medium'];pair.main()
  main_status=RUN/'PAIRWISE_WORKER_STATUS.json'
  phase=json.loads(main_status.read_text()).get('phase') if main_status.exists() else None
  if phase!='waiting_for_social':
   state['phase']='finished_separately';state['reason']='Main pairwise workflow advanced; early files kept separately to avoid races';save(state);return
  bank=A/'evals/heldout_diagnostics'
  banks={}
  for suite,name in (('advice','advice.jsonl'),('social','paired_sotopia.jsonl')):
   rows=[json.loads(x) for x in (bank/name).read_text().splitlines() if x.strip()];banks[suite]={x['id']:x for x in rows}
  prompts={str(p.relative_to(early)):text for p,text in pair.jobs(RESULTS,early,banks,pair.load_runner())}
  files=[p for p in early.rglob('*.json') if p.name!='PAIRWISE_REPORT.json']
  for source in files:
   rel=source.relative_to(early);expected=prompts.get(str(rel));record=json.loads(source.read_text())
   if expected is None or record.get('prompt_sha256')!=hashlib.sha256(expected.encode()).hexdigest():raise ValueError('prompt identity mismatch: '+str(rel))
   pair.valid(record['parsed'])
   if pair.T.first_json(record['raw'])!=record['parsed'] or record.get('meta',{}).get('judge')!='agy:gemini-3.8-flash-medium':raise ValueError('judgment identity mismatch: '+str(rel))
   target=final/rel;target.parent.mkdir(parents=True,exist_ok=True)
   try:os.link(source,target);state['imported']+=1
   except FileExistsError:pass
  state['phase']='complete';state['available']=len(files);state['expected']=740;save(state)
 except BaseException as e:
  state['phase']='failed';state['errors'].append(repr(e));save(state);traceback.print_exc();raise
 finally:lock.rmdir()
if __name__=='__main__':main()
