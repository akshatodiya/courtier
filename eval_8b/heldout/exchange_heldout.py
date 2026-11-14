"""Immutable, content-addressed evaluation messages. No SSH or provider calls."""
import hashlib,json,os,re,tempfile
from pathlib import Path

CONDITIONS={'base'} | {f'epoch{i}' for i in range(1,10)}
SUITES={'advice','laws','social'}
SHA=re.compile(r'^[0-9a-f]{64}$')
IDENT=re.compile(r'^[a-z0-9][a-z0-9_.-]{0,127}$')
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def validate_request(row):
    if set(row)!={'identity','condition','suite','case_id','turn','max_new_tokens','messages'}:raise ValueError('unexpected request fields')
    if not isinstance(row['identity'],str) or not SHA.fullmatch(row['identity']):raise ValueError('invalid run identity')
    if row['condition'] not in CONDITIONS or row['suite'] not in SUITES:raise ValueError('invalid condition or suite')
    if not isinstance(row['case_id'],str) or not IDENT.fullmatch(row['case_id']):raise ValueError('invalid case ID')
    if type(row['turn'])!=int or not 0<=row['turn']<12:raise ValueError('invalid turn')
    expected=192 if row['suite']=='social' else 1024
    if row['max_new_tokens']!=expected or type(row['max_new_tokens'])!=int:raise ValueError('wrong generation cap')
    messages=row['messages']
    if not isinstance(messages,list) or not messages:raise ValueError('missing messages')
    for i,m in enumerate(messages):
        if not isinstance(m,dict) or set(m)!={'role','content'} or m['role']!=('user' if i%2==0 else 'assistant') or not isinstance(m['content'],str):raise ValueError('invalid conversation')
    if messages[-1]['role']!='user':raise ValueError('conversation must end with user')
    return digest(row)
def atomic_once(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=canonical(value)+b'\n'
    fd,name=tempfile.mkstemp(dir=p.parent,prefix='.message-')
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(name,p)
        except FileExistsError:
            if p.read_bytes()!=data:raise ValueError('immutable message collision')
    finally:os.unlink(name)
def submit(directory,row):
    key=validate_request(row);atomic_once(Path(directory)/(key+'.json'),{'id':key,'request':row});return key
def read_request(path):
    p=Path(path);e=json.loads(p.read_text())
    if set(e)!={'id','request'} or not isinstance(e['id'],str) or not SHA.fullmatch(e['id']):raise ValueError('invalid envelope')
    if p.name!=e['id']+'.json' or validate_request(e['request'])!=e['id']:raise ValueError('request identity mismatch')
    return e
def validate_result(result):
    if set(result)!={'text','finish','prompt_tokens','generated_tokens'}:raise ValueError('unexpected response fields')
    if not isinstance(result['text'],str) or result['finish'] not in ('end_of_turn','length','other'):raise ValueError('invalid response')
    if any(type(result[k])!=int or result[k]<0 for k in ('prompt_tokens','generated_tokens')):raise ValueError('invalid token counts')
def write_response(directory,key,result):
    if not SHA.fullmatch(key):raise ValueError('invalid key')
    validate_result(result);atomic_once(Path(directory)/(key+'.json'),{'id':key,'result':result})
def read_response(directory,key):
    if not SHA.fullmatch(key):raise ValueError('invalid key')
    e=json.loads((Path(directory)/(key+'.json')).read_text())
    if set(e)!={'id','result'} or e['id']!=key:raise ValueError('response identity mismatch')
    validate_result(e['result']);return e['result']
