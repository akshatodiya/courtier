"""Wait for the epoch social controller, then run both independent pairwise judges."""
import hashlib,json,os,socket,subprocess,time,traceback
from pathlib import Path
A=Path('~/lexsi-hackathon/aya_expanse_8b')
STAGE=A/'paramganga/stage_eval_epoch9/heldout'
RUN=A/'evals/epoch_study/heldout_run'
RESULTS=RUN/'results'
SCRIPTS=A/'curation/run_with_judge.sh'
NAMES=['base']+[f'epoch{i}' for i in range(1,10)]
def save(d):
 p=RUN/'PAIRWISE_WORKER_STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def count_missing(judged):
 advice=read_jsonl(A/'evals/heldout_diagnostics/advice.jsonl');social=read_jsonl(A/'evals/heldout_diagnostics/paired_sotopia.jsonl');missing=[]
 for epoch in range(1,10):
  for suite,cases in (('advice',advice),('social',social)):
   for c in cases:
    for order in ('xy','yx'):
     p=judged/suite/f'epoch{epoch}-base'/f"{c['id']}.{order}.json"
     if not p.is_file():missing.append(str(p.relative_to(judged)))
 return missing
def port_busy(port):
 with socket.socket() as s:
  s.settimeout(1);return s.connect_ex(('127.0.0.1',port))==0
def main():
 RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'.pairwise-worker.lock';lock.mkdir()
 state={'phase':'waiting_for_social','judges':{},'errors':[]}
 try:
  for line in (STAGE/'MANIFEST.sha256').read_text().splitlines():
   h,name=line.split('  ',1)
   if hashlib.sha256((STAGE/name).read_bytes()).hexdigest()!=h:raise ValueError('stage changed '+name)
  while not (RESULTS/'CONTROLLER_STATUS.json').exists():save(state);time.sleep(30)
  status=json.loads((RESULTS/'CONTROLLER_STATUS.json').read_text())
  if status.get('transport_error') or set(status.get('conditions',{}))!=set(NAMES):raise ValueError('social controller incomplete: '+str(status))
  # The controller owns port 8767; wait for it to release the shim before starting ours.
  while port_busy(8767):state['phase']='waiting_for_social_shim_to_stop';save(state);time.sleep(10)
  state['phase']='reporting_social';save(state)
  subprocess.run(['python3',str(STAGE/'report_heldout.py'),'--plan',str(STAGE/'plan.json'),'--run',str(RESULTS),'--out',str(RUN/'report')],cwd=A,check=True)
  state['phase']='pairwise_haiku';save(state)
  subprocess.run([str(SCRIPTS),'python3',str(STAGE/'pairwise_quality.py'),'--results',str(RESULTS),'--judged',str(RUN/'pairwise_haiku')],cwd=A,check=True)
  missing=count_missing(RUN/'pairwise_haiku');state['judges']['haiku']={'missing':len(missing),'examples':missing[:10]};save(state)
  state['phase']='pairwise_gemini';save(state)
  subprocess.run(['python3',str(STAGE/'pairwise_quality.py'),'--results',str(RESULTS),'--judged',str(RUN/'pairwise_gemini'),'--agy','gemini-3.8-flash-medium'],cwd=A,check=True)
  missing=count_missing(RUN/'pairwise_gemini');state['judges']['gemini']={'missing':len(missing),'examples':missing[:10]}
  state['phase']='complete' if not any(x['missing'] for x in state['judges'].values()) else 'complete_with_missing';save(state)
 except BaseException as e:
  state['phase']='failed';state['errors'].append(repr(e));save(state);traceback.print_exc();raise
 finally:lock.rmdir()
if __name__=='__main__':main()
