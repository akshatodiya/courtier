"""Back up completed PARAM loss outputs and make an explicitly exploratory plot."""
import hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
A=Path('~/courtier/aya_expanse_8b')
STAGE=A/'paramganga/stage_eval_epoch9/loss';OUT=A/'evals/epoch_study/loss_run';WRAPPER=A/'paramganga/existing_session_ssh.sh'
CONDITIONS=['base']+[f'epoch{i}' for i in range(1,10)]
def save(d):
 p=OUT/'WORKER_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(raw,cases,identity):
 expected={r['id']:r for r in cases};obs=[];token_counts={}
 if len(expected)!=len(cases) or len(cases)!=100:raise ValueError('bad cases')
 for e,c in enumerate(CONDITIONS):
  rows=[json.loads(s) for s in (raw/(c+'.jsonl')).read_text().splitlines() if s.strip()]
  if len(rows)!=100 or {r['id'] for r in rows}!=set(expected):raise ValueError('incomplete/duplicate loss cases '+c)
  for r in rows:
   want=expected[r['id']]
   if r['condition']!=c or r['identity']!=identity or r['split']!=want['split'] or r['group']!=want['group']:raise ValueError('loss identity mismatch')
   if type(r['answer_tokens']) is not int or r['answer_tokens']<=0 or not math.isfinite(r['loss_sum']) or r['loss_sum']<0:raise ValueError('bad loss')
   if r['id'] in token_counts and token_counts[r['id']]!=r['answer_tokens']:raise ValueError('answer token count changed')
   token_counts[r['id']]=r['answer_tokens']
   obs.append({'epoch':e,'step':117*e,'split':r['split'],'case_id':r['id'],'group_id':r['group'],'loss_sum':r['loss_sum'],'answer_tokens':r['answer_tokens']})
 return obs
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'.worker.lock';lock.mkdir();state={'phase':'waiting','errors':[]}
 try:
  for line in (STAGE/'MANIFEST.sha256').read_text().splitlines():
   digest,name=line.split('  ',1)
   if sha(STAGE/name)!=digest:raise ValueError('stage changed '+name)
  while True:
   result=subprocess.run([str(WRAPPER),'pg','cat ~/aya_expanse_8b_pg/eval/epoch9_coordinator/status'],capture_output=True,text=True,timeout=60)
   if result.returncode:raise RuntimeError('PARAM status failed, no reconnection: '+result.stderr[-500:])
   phase=result.stdout.strip();state['remote_phase']=phase;save(state)
   if phase in ('loss_complete_starting_social','waiting_social','actor_jobs_complete'):break
   time.sleep(30)
  state['phase']='copying';save(state)
  raw=OUT/'raw';raw.mkdir(exist_ok=True)
  subprocess.run(['rsync','-a','--timeout=60','-e',str(WRAPPER),'pg:aya_expanse_8b_pg/runs/epoch9-loss/',str(raw)+'/'],check=True,timeout=300)
  cases=[json.loads(x) for x in (STAGE/'cases.jsonl').read_text().splitlines() if x.strip()];observations=build(raw,cases,sha(STAGE/'plan.json'))
  doc={'metadata':{'run_id':'courtier-epoch9-614246','dataset_hash':'f77b72c0c381daac41fa4ddf996e7eef24e1cb8746a30b14b8a11804051351f5','config_hash':'3a7ec924809c055f400920ed24f44c56ba2f08c763cee2d2b323ad910b09dbb3','loss_protocol_id':sha(STAGE/'plan.json'),'synthetic':False,'scope':'Exploratory heldout advice rubric-key target, not independently authored reference answers; family-level separation unverified.'},'observations':observations}
  data=OUT/'loss_input.json'
  if data.exists():raise FileExistsError('refusing to overwrite prior result')
  data.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
  plotter=STAGE/'plot_loss_proxy.py';subprocess.run(['python3',str(plotter),str(data),'--out',str(OUT/'plots')],check=True)
  state['phase']='complete';state['cases_per_condition']=100;state['conditions']=len(CONDITIONS);save(state)
 except BaseException as e:
  state['phase']='failed';state['errors'].append(repr(e));save(state);traceback.print_exc();raise
 finally:lock.rmdir()
if __name__=='__main__':main()
