"""Wait for real epoch evaluation outcomes, assemble, and draw identified plots."""
import json,os,subprocess,time,traceback
from pathlib import Path
A=Path('~/courtier/aya_expanse_8b');ROOT=A/'evals/epoch_study';STAGE=A/'paramganga/stage_eval_epoch9';OUT=ROOT/'real_epoch_results'
def save(d):
 p=ROOT/'PLOT_WORKER_STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)
def get(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
 lock=ROOT/'.plot-worker.lock';lock.mkdir();state={'phase':'waiting','errors':[]}
 try:
  while True:
   orstat=get(ROOT/'orbench_run/WORKER_STATUS.json');loss=get(ROOT/'loss_run/WORKER_STATUS.json');social=get(ROOT/'heldout_run/PAIRWISE_WORKER_STATUS.json');ratings=get(ROOT/'heldout_run/RATINGS_WORKER_STATUS.json')
   statuses={'orbench':orstat.get('phase'),'loss':loss.get('phase'),'pairwise':social.get('phase'),'ratings':ratings.get('phase')};state['workers']=statuses;save(state)
   if any(x=='failed' for x in statuses.values()):raise RuntimeError('Upstream worker failed; inspect its status')
   if orstat.get('phase')=='complete' and loss.get('phase')=='complete' and social.get('phase') in ('complete','complete_with_missing') and ratings.get('phase') in ('complete','complete_with_missing'):break
   time.sleep(30)
  state['phase']='assembling';save(state)
  subprocess.run(['python3',str(STAGE/'assemble_plots.py')],check=True)
  for suite in ('social','advice'):
   for kind,plotter in [('strict',ROOT/'epoch_plots.py'),('observed',STAGE/'plot_resolved.py')]:
    subprocess.run(['python3',str(plotter),str(OUT/(suite+'.json')),'--out',str(OUT/kind/suite)],check=True)
  state['phase']='complete';state['plot_directory']=str(OUT);state['pairwise_coverage']=social.get('judges',{});state['rating_coverage']=ratings.get('judges',{});save(state)
 except BaseException as e:
  state['phase']='failed';state['errors'].append(repr(e));save(state);traceback.print_exc();raise
 finally:lock.rmdir()
if __name__=='__main__':main()
