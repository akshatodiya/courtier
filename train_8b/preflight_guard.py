"""Reject mismatched Aya training preflight before submission or training."""
import hashlib,json,sys
from pathlib import Path
from train_control import identity

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    record=json.loads((root/'PREFLIGHT.json').read_text())
    if record.get('status')!='passed' or record.get('run_identity')!=identity(root):raise ValueError('preflight identity/status mismatch')
    if record.get('versions')!=cfg['versions']:raise ValueError('preflight package versions mismatch')
    p=Path(record.get('kit_source',''))
    expected='5e1362219c6e88d39bcb84f11bdba6f917666f6fddcea930a002981eb615de7b'
    if not p.is_file() or record.get('kit_source_sha256')!=expected or sha(p)!=expected:raise ValueError('preflight SafeTune source mismatch')
    if sha(root/'data/train.jsonl')!=cfg['data_sha256']:raise ValueError('dataset identity mismatch')
    return record['run_identity']
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: preflight_guard.py STAGE')
    print('PREFLIGHT PASS',verify(sys.argv[1]))
