"""Collect completed epoch OR-Bench generations and judge them on the Mac.

Uses an existing PARAM SSH master and the existing Haiku shim. It stops on any
transfer or provider failure; no fresh SSH connections or retry loops.
"""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
A=Path('~/courtier/aya_expanse_8b')
STAGE=A/'paramganga/stage_eval_epoch9/orbench'
RUN=A/'evals/epoch_study/orbench_run'
WRAPPER=A/'paramganga/existing_session_ssh.sh'
JUDGE=A.parent/'tiny_aya_3_35b/courtier/scripts/eval_judge.py'
PYTHON=A.parent/'learn/.venv/bin/python'
CONDITIONS=['base']+[f'epoch{i}' for i in range(1,10)]
sys.path.insert(0,str(STAGE));import actor

def save(value):
 p=RUN/'WORKER_STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');os.replace(q,p)
def verify_stage():
 for line in (STAGE/'MANIFEST.sha256').read_text().splitlines():
  h,name=line.split('  ',1)
  if actor.sha(STAGE/name)!=h:raise ValueError('stage changed: '+name)
def collect(c,plan,requests):
 raw=RUN/'raw';raw.mkdir(parents=True,exist_ok=True)
 src='pg:aya_expanse_8b_pg/runs/epoch9-orbench/'
 for name in (c+'.jsonl',c+'_report.json'):
  subprocess.run(['rsync','-a','--timeout=60','-e',str(WRAPPER),src+name,str(raw/name)],check=True,timeout=120)
 report=json.loads((raw/(c+'_report.json')).read_text());ident=report['identity']
 if report.get('status')!='complete' or report.get('completed')!=600 or report.get('expected')!=600:raise ValueError('incomplete report '+c)
 for key,value in {'condition':c,'plan_sha256':actor.sha(STAGE/'plan.json'),'actor_sha256':actor.sha(STAGE/'actor.py'),'requests_sha256':plan['requests_sha256'],'model_manifest_sha256':plan['model_manifest_sha256'],'checkpoint_receipt_sha256':actor.sha(STAGE/'checkpoint_receipt.json'),'max_new_tokens':1024,'batch_size':16}.items():
  if ident.get(key)!=value:raise ValueError('identity mismatch '+c+' '+key)
 receipt=json.loads((STAGE/'checkpoint_receipt.json').read_text())
 expected=None if c=='base' else receipt['conditions'][c]['adapter_sha256']
 if ident.get('adapter_sha256')!=expected:raise ValueError('wrong adapter '+c)
 if actor.prefix(raw/(c+'.jsonl'),requests,c,ident)!=600:raise ValueError('coverage mismatch '+c)
 rows=actor.rows(raw/(c+'.jsonl'));prepared=[];unknown=[]
 for q,r in zip(requests,rows):
  if not r['text'].strip():unknown.append({'id':r['id'],'model':c,'reason':'empty generation'});continue
  prepared.append({'id':r['id'],'set':r['meta']['set'],'category':r['meta']['category'],'input':q['messages'][0]['content'],'output':r['text'],'model':c,'finish':r['finish'],'generated_tokens':r['generated_tokens']})
 judge=RUN/'judge';judge.mkdir(exist_ok=True)
 for name,data in [(f'gen_{c}.jsonl',prepared),(f'unknown_{c}.jsonl',unknown)]:
  p=judge/name;content=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data)
  if p.exists() and p.read_text()!=content:raise ValueError('existing judge input differs '+name)
  if not p.exists():p.write_text(content)
 return len(prepared),len(unknown)

def remote_complete():
 cmd="d=$HOME/aya_expanse_8b_pg/runs/epoch9-orbench; for c in base epoch1 epoch2 epoch3 epoch4 epoch5 epoch6 epoch7 epoch8 epoch9; do if [ -f \"$d/${c}_report.json\" ] && grep -q '\"status\": \"complete\"' \"$d/${c}_report.json\"; then printf '%s\\n' \"$c\"; fi; done"
 r=subprocess.run([str(WRAPPER),'pg',cmd],capture_output=True,text=True,timeout=60)
 if r.returncode:raise RuntimeError('PARAM status read failed; no reconnect attempted: '+r.stderr[-1000:])
 names=r.stdout.splitlines()
 if any(c not in CONDITIONS for c in names):raise ValueError('unexpected remote status')
 return set(names)
def work():
 RUN.mkdir(parents=True,exist_ok=True);verify_stage();plan=json.loads((STAGE/'plan.json').read_text());requests=actor.check_requests(STAGE/'requests.jsonl',plan['requests_sha256'])
 state={'phase':'waiting','completed':[],'judge':'claude-haiku-5-5','judge_endpoint':'http://127.0.0.1:8768/v1','errors':[]}
 if (RUN/'WORKER_STATUS.json').exists():
  old=json.loads((RUN/'WORKER_STATUS.json').read_text());state['completed']=old.get('completed',[])
 if any(c not in CONDITIONS for c in state['completed']):raise ValueError('invalid prior worker state')
 try:
  while len(state['completed'])<len(CONDITIONS):
   ready=remote_complete();todo=[c for c in CONDITIONS if c in ready and c not in state['completed']]
   if not todo:
    state['phase']='waiting_for_gpu_results';save(state);time.sleep(30);continue
   for c in todo:
    state['phase']='collecting '+c;save(state);n,unknown=collect(c,plan,requests)
    state['phase']='judging '+c;state['current_count']=n;state['current_unknown']=unknown;save(state)
    env=os.environ.copy();env.update(JUDGE_MODEL='claude-haiku-5-5',JUDGE_API='http://127.0.0.1:8768/v1',JUDGE_FILE=f'judged_{c}.jsonl')
    subprocess.run([str(PYTHON),str(JUDGE),'--run-dir',str(RUN/'judge'),c,'--workers','4','--chunk','8'],env=env,check=True)
    state['completed'].append(c);save(state)
  state['phase']='complete';save(state)
 except BaseException as e:
  state['phase']='failed';state['errors'].append(repr(e));save(state);traceback.print_exc();raise
def main():
 RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'.worker.lock'
 lock.mkdir()
 try:work()
 finally:lock.rmdir()
if __name__=='__main__':main()
