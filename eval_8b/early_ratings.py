"""Precompute rubric ratings for completed epochs without touching running jobs."""
import hashlib,json,os,sys,traceback
from pathlib import Path
A=Path('~/courtier/aya_expanse_8b');ST=A/'paramganga/stage_eval_epoch9';R=A/'evals/epoch_study/heldout_run';RESULTS=R/'results'
sys.path.insert(0,str(ST));import rating_worker as rating
NAMES=['base']+[f'epoch{i}' for i in range(1,8)]
def save(v):
 p=R/'EARLY_RATINGS_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def main():
 lock=R/'.early-ratings.lock';lock.mkdir();status={'phase':'preparing','conditions':NAMES,'errors':[],'imported':0}
 try:
  for c in NAMES:
   fs=list((RESULTS/c).rglob('*.json'));valid=sum(bool(json.loads(f.read_text()).get('judgment',{}).get('valid')) for f in fs)
   if len(fs)!=146 or valid!=146:raise ValueError('condition is not complete: '+c)
  rating.NAMES=NAMES;rating.RUN=R/'early_ratings';rating.RESULTS=RESULTS
  status['phase']='judging_gemini';save(status);summary=rating.run_judge('gemini')
  phase=json.loads((R/'RATINGS_WORKER_STATUS.json').read_text()).get('phase')
  if phase!='waiting_for_pairwise':
   status['phase']='finished_separately';status['reason']='Main ratings workflow already advanced; keeping results isolated';status['summary']=summary;save(status);return
  src=rating.RUN/'ratings_gemini';dst=R/'ratings_gemini';prompts={str(p.relative_to(src)):q for p,q in rating.tasks('gemini') if q is not None}
  for source in src.rglob('*.json'):
   if source.name=='SUMMARY.json':continue
   rel=source.relative_to(src);prompt=prompts.get(str(rel));record=json.loads(source.read_text())
   if prompt is None or record.get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('rating prompt mismatch: '+str(rel))
   rating.valid(record['parsed'])
   if rating.pair.T.first_json(record['raw'])!=record['parsed'] or record.get('meta',{}).get('judge')!='agy:gemini-3.8-flash-medium':raise ValueError('rating verdict mismatch: '+str(rel))
   target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True)
   try:os.link(source,target);status['imported']+=1
   except FileExistsError:pass
  status['phase']='complete';status['summary']=summary;save(status)
 except BaseException as e:
  status['phase']='failed';status['errors'].append(repr(e));save(status);traceback.print_exc();raise
 finally:lock.rmdir()
if __name__=='__main__':main()
