"""Fail closed unless the Aya 32B pilot prerequisites match their receipts."""
import hashlib
import json
from pathlib import Path
import sys

REV='b306ea27e360683b50c005d7fcbad6a242317910'
DATA='f77b72c0c381daac41fa4ddf996e7eef24e1cb8746a30b14b8a11804051351f5'
MODEL='CohereLabs/aya-expanse-32b'
EXPECTED_MANIFEST_SHA256='053d4fa74b1abb9bd2bda698fcefd94907785ac24afa1e07b201ccaafa263694'


def load(path):
    return json.loads(path.read_text())


def rehash_model(root):
    """Rehash the pinned files on the Slurm batch compute node, not login."""
    root=Path(root)
    expected_path=root/'verify32/expected.json'
    if hashlib.sha256(expected_path.read_bytes()).hexdigest()!=EXPECTED_MANIFEST_SHA256:
        raise ValueError('Saved source receipt manifest changed')
    records=load(expected_path)['files']
    for record in records:
        relative=Path(record['name'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid model file path')
        file=root/'models/aya-expanse-32b'/relative
        if file.is_symlink() or not file.is_file() or file.stat().st_size!=record['size']:
            raise ValueError('Missing or wrong-size model file: '+str(relative))
        algorithm=record['algorithm']
        if algorithm=='sha256': digest=hashlib.sha256()
        elif algorithm=='git-blob-sha1':
            digest=hashlib.sha1()
            digest.update(('blob '+str(record['size'])+'\0').encode())
        else:raise ValueError('Unsupported model digest: '+str(algorithm))
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=record['digest']:raise ValueError('Model digest mismatch: '+str(relative))
    return {'status':'passed','files':len(records),'scope':'model files rehashed on compute node'}


def validate(root,verify_job):
    root=Path(root)
    if not str(verify_job).isdigit(): raise ValueError('Invalid verification job ID')
    expected_path=root/'verify32/expected.json'
    if hashlib.sha256(expected_path.read_bytes()).hexdigest()!=EXPECTED_MANIFEST_SHA256:
        raise ValueError('Saved source receipt manifest changed')
    expected=load(expected_path)
    receipt=load(root/('runs/verify32-'+str(verify_job))/'MODEL_VERIFIED.json')
    if receipt.get('status')!='complete' or receipt.get('model')!=MODEL or receipt.get('revision')!=REV:
        raise ValueError('Wrong model verification receipt')
    if expected.get('model')!=MODEL or expected.get('revision')!=REV or receipt.get('files')!=expected.get('files'):
        raise ValueError('Model hashes do not match the saved source receipt')
    records=expected['files']
    if len(records)!=21 or len({r['name'] for r in records})!=21:
        raise ValueError('Wrong model inventory')
    model=root/'models/aya-expanse-32b'
    for record in records:
        name=Path(record['name'])
        if name.is_absolute() or '..' in name.parts: raise ValueError('Invalid model file path')
        file=model/name
        if file.is_symlink() or not file.is_file() or file.stat().st_size!=record['size']:
            raise ValueError('Model file missing or changed in size: '+str(name))
    data=root/'data/train.jsonl'
    if data.is_symlink() or not data.is_file(): raise ValueError('Dataset missing')
    h=hashlib.sha256()
    rows=0
    with data.open('rb') as f:
        for line in f:
            h.update(line);rows+=1
    if h.hexdigest()!=DATA or rows!=935: raise ValueError('Dataset identity mismatch')
    topo=load(root/'runs/topology14-614881/report.json')
    ranks=topo.get('ranks',[])
    if topo.get('summary')!={'nodes':7,'workers':14,'status':'passed'} or len(ranks)!=14:
        raise ValueError('14-GPU topology not verified')
    if sorted(r.get('rank') for r in ranks)!=list(range(14)):
        raise ValueError('Missing or duplicate global ranks')
    hosts={}
    for r in ranks:
        if r.get('world_size')!=14 or r.get('device_count')!=2 or not r.get('host'):
            raise ValueError('Unexpected worker topology')
        hosts.setdefault(r['host'],[]).append(r.get('local_rank'))
    if len(hosts)!=7 or any(sorted(local_ranks)!=[0,1] for local_ranks in hosts.values()):
        raise ValueError('Expected two distinct local ranks on each of seven hosts')
    if len({r.get('gpu_uuid') for r in ranks})!=14 or not all(r.get('gpu_uuid') and r.get('collective_sum')==14 for r in ranks):
        raise ValueError('GPU identities or collective results invalid')
    toy=load(root/'runs/fsdp-toy14-614899/RESULT.json')
    initial=toy.get('initial_adapter_sha256',''); final=toy.get('final_adapter_sha256','')
    if toy.get('status')!='passed' or toy.get('world_size')!=14 or toy.get('steps')!=2 or len(initial)!=64 or len(final)!=64 or initial==final:
        raise ValueError('Synthetic QLoRA rehearsal did not pass')
    return {'status':'passed','model':MODEL,'revision':REV,'dataset_rows':rows,'world_size':14,
            'limitation':'model files rechecked for size; hashes come from completed independent verification'}


def validate_retry(root, retry_job):
    if not str(retry_job).isdigit(): raise ValueError('Invalid retry test job ID')
    root=Path(root)
    result=load(root/('runs/recovery-retry-toy14-'+str(retry_job))/'RESULT.json')
    sha=hashlib.sha256(Path(__file__).with_name('scale_retry.py').read_bytes()).hexdigest()
    if (result.get('status')!='passed' or result.get('world_size')!=14
            or result.get('replay')!='allclose' or result.get('retry_source_sha256')!=sha):
        raise ValueError('Synthetic retry verification is incomplete or uses different code')
    return {'status':'passed', 'scope':'synthetic 14-GPU overflow replay', 'retry_source_sha256':sha}


def validate_attention(root, attention_job):
    if not str(attention_job).isdigit(): raise ValueError('Invalid attention probe job ID')
    result=load(Path(root)/('runs/attention-probe-'+str(attention_job))/'RESULT.json')
    expected=load(Path(__file__).with_name('probe_expected.json'))
    if (result.get('gpu')!='Tesla V100-SXM2-16GB' or result.get('capability')!=[7,0]
            or result.get('torch')!='2.7.1+cu118' or result.get('probe_sha256')!=expected['probe_sha256']):
        raise ValueError('Unexpected probe hardware, runtime, or code')
    cases=[case for case in result.get('cases',[]) if case.get('kind')=='repeat'
           and case.get('explicit_causal_mask') is False and case.get('status')=='passed']
    if len(cases)!=1 or cases[0].get('peak_allocated',0)<=0:
        raise ValueError('The unmasked repeated-KV attention test did not pass')
    parity=cases[0].get('short_parity',{})
    if set(parity)!={'0.2','1.0','2.0'}:
        raise ValueError('Attention parity cases missing')
    for case in parity.values():
        values=case.get('relative_l2_errors',[])
        if len(values)!=4 or any(not isinstance(v,(int,float)) or not 0<=v<0.03 for v in values):
            raise ValueError('Attention output or gradient parity failed')
    return {'status':'passed','scope':'V100 synthetic unmasked repeated-KV attention',
            'probe_sha256':result['probe_sha256']}


if __name__=='__main__':
    if len(sys.argv)!=5: raise ValueError('Expected root, model, retry and attention job IDs')
    root=Path(sys.argv[1])
    report=load(root/'runs/recovery-report-615303.json')
    if report.get('status')!='pilot_checks_passed' or report.get('slurm')!='COMPLETED|0:0':
        raise ValueError('Successful real-model pilot report missing')
    print(json.dumps({'prerequisites':validate(root,sys.argv[2]),
                      'retry':validate_retry(root,sys.argv[3]),
                      'attention':validate_attention(root,sys.argv[4]),
                      'model_rehash':rehash_model(root)},sort_keys=True),flush=True)
