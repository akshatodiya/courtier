"""Generate one identity-checked Aya 8B OR-Bench condition; no judgment calls."""
import argparse,hashlib,json,os,time,traceback
from importlib.metadata import version
from pathlib import Path

REVISION='5062468bf9bc0c6035fd64e06274333ec127d980'
SETS={'harmful':400,'benign':200}
CONDITIONS=['base']+[f'epoch{i}' for i in range(1,10)]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def check_requests(path, expected):
    if sha(path)!=expected:raise ValueError('request checksum mismatch')
    data=rows(path)
    if len(data)!=600 or len({x['id'] for x in data})!=600:raise ValueError('request coverage mismatch')
    if {k:sum(x['meta']['set']==k for x in data) for k in SETS}!=SETS:raise ValueError('request sets mismatch')
    for r in data:
        if set(r)!={'id','messages','meta'} or set(r['meta'])!={'set','category'} or [m['role'] for m in r['messages']]!=['user']:
            raise ValueError('invalid request format')
    return data
def verify_model(model_dir, manifest):
    info=json.loads(Path(manifest).read_text())
    if info['repository']!='CohereLabs/aya-expanse-8b' or info['revision']!=REVISION:raise ValueError('unexpected base model')
    required={'config.json','generation_config.json','model-00001-of-00004.safetensors','model-00002-of-00004.safetensors',
              'model-00003-of-00004.safetensors','model-00004-of-00004.safetensors','model.safetensors.index.json',
              'special_tokens_map.json','tokenizer.json','tokenizer_config.json'}
    if set(info['files'])!=required:raise ValueError('model manifest is incomplete')
    for name,meta in info['files'].items():
        p=Path(model_dir)/name
        if not p.is_file() or p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise ValueError('model mismatch '+name)
def check_checkpoint(run_dir,expected):
    run=Path(run_dir); report=json.loads((run/'TRAINING.json').read_text()); step=expected.get('step',249)  # merged run: 351
    if report.get('status')!='complete' or report.get('global_step')!=1053 or report.get('epoch')!=9.0:
        raise ValueError('training is incomplete; refusing to evaluate')
    if report.get('run_identity')!=expected['training_identity'] or report['config']['data_sha256']!=expected['data_sha256']:
        raise ValueError('training identity or dataset mismatch')
    ck=run/f'checkpoints/checkpoint-{step}';marker=json.loads((ck/'CHECKPOINT_COMPLETE.json').read_text())
    state=json.loads((ck/'trainer_state.json').read_text())
    if marker.get('run_identity')!=expected['training_identity'] or marker.get('world_size')!=2 or marker.get('global_step')!=step or marker.get('epoch')!=expected['epoch'] or state.get('global_step')!=step or state.get('epoch')!=expected['epoch']:
        raise ValueError('checkpoint identity mismatch')
    required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','scaler.pt','trainer_state.json','rng_state_0.pth','rng_state_1.pth']
    if any(not (ck/n).is_file() or (ck/n).stat().st_size==0 for n in required):raise ValueError('checkpoint incomplete')
    return ck,sha(ck/'adapter_model.safetensors')
def check_receipt(path,stage,condition,run_dir,plan):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('plan_sha256')!=sha(stage/'plan.json') or receipt.get('actor_sha256')!=sha(stage/'actor.py'):
        raise ValueError('checkpoint receipt belongs to a different evaluation stage')
    if set(receipt.get('conditions',{}))!=set(CONDITIONS)-{'base'}:raise ValueError('checkpoint receipt coverage')
    if condition!='base':
        ck,actual=check_checkpoint(run_dir,plan[condition]);record=receipt['conditions'][condition]
        if record.get('training_identity')!=plan[condition]['training_identity'] or record.get('adapter_sha256')!=actual:
            raise ValueError('checkpoint adapter differs from captured receipt')
        run=Path(run_dir)
        for key,rel in [('training_report_sha256','TRAINING.json'),('checkpoint_marker_sha256',f"checkpoints/checkpoint-{plan[condition].get('step',249)}/CHECKPOINT_COMPLETE.json"),('trainer_state_sha256',f"checkpoints/checkpoint-{plan[condition].get('step',249)}/trainer_state.json")]:
            if record.get(key)!=sha(run/rel):raise ValueError('checkpoint provenance changed: '+key)
        if record.get('run_directory')!=str(run.resolve()):raise ValueError('checkpoint run path mismatch')
        return ck,actual
    return None,None
def prefix(path,expected,condition,identity):
    p=Path(path)
    if not p.exists():return 0
    data=rows(p)
    if len(data)>len(expected):raise ValueError('too many existing outputs')
    for r,q in zip(data,expected):
        if r.get('id')!=q['id'] or r.get('meta')!=q['meta'] or r.get('condition')!=condition or r.get('identity')!=identity:
            raise ValueError('saved prefix identity differs')
        if not isinstance(r.get('text'),str):raise ValueError('saved output text missing')
        tokens=r.get('token_ids');n=r.get('generated_tokens');pn=r.get('prompt_tokens')
        if not isinstance(tokens,list) or any(type(t)!=int or t<0 for t in tokens) or type(n)!=int or n!=len(tokens) or not 0<=n<=1024:
            raise ValueError('invalid output token metadata')
        if type(pn)!=int or pn<1 or pn+n>8192:raise ValueError('invalid prompt length')
        expected_finish='end_of_turn' if tokens and tokens[-1]==255001 else ('length' if n>=1024 else 'other')
        if r.get('finish')!=expected_finish:raise ValueError('inconsistent finish metadata')
    return len(data)
def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--condition',choices=CONDITIONS,required=True);ap.add_argument('--training-run',type=Path)
    ap.add_argument('--checkpoint-receipt',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--verify-only',action='store_true')
    # batched greedy generation split across GPUs; every condition uses the same settings
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--merge',action='store_true',help='join finished shards into <condition>.jsonl and its report')
    a=ap.parse_args();plan=json.loads((a.stage/'plan.json').read_text())
    if plan['max_new_tokens']!=1024 or plan['conditions']!=CONDITIONS:raise ValueError('invalid plan')
    req=check_requests(a.stage/'requests.jsonl',plan['requests_sha256'])
    manifest=a.stage/'model_files.json'
    if sha(manifest)!=plan['model_manifest_sha256']:raise ValueError('model manifest mismatch')
    verify_model(a.model_dir,manifest)
    if a.condition=='base':
        if a.training_run is not None:raise ValueError('base may not use an adapter')
        ck,adapter_sha=check_receipt(a.checkpoint_receipt,a.stage,a.condition,None,plan)
    else:
        if a.training_run is None:raise ValueError('model condition requires a completed training run')
        ck,adapter_sha=check_receipt(a.checkpoint_receipt,a.stage,a.condition,a.training_run,plan)
    ident={'plan_sha256':sha(a.stage/'plan.json'),'actor_sha256':sha(__file__),
           'requests_sha256':plan['requests_sha256'],'model_manifest_sha256':plan['model_manifest_sha256'],
           'condition':a.condition,'adapter_sha256':adapter_sha,'checkpoint_receipt_sha256':sha(a.checkpoint_receipt),'max_new_tokens':1024,'seed':16,
           'precision':'NF4 double quantization; fp16 compute','decoding':'greedy','versions':plan['versions'],'batch_size':a.batch_size}
    if a.verify_only:
        print(json.dumps({'status':'passed','identity':ident,'checkpoint':str(ck) if ck else None}));return
    mine=[r for k,r in enumerate(req) if k%a.shards==a.shard]
    tag=a.condition if a.shards==1 else f'{a.condition}.shard{a.shard}of{a.shards}'
    if a.merge:
        parts=[[r for k,r in enumerate(req) if k%a.shards==i] for i in range(a.shards)]
        done=[rows(a.out/f'{a.condition}.shard{i}of{a.shards}.jsonl') for i in range(a.shards)]
        for i in range(a.shards):
            if prefix(a.out/f'{a.condition}.shard{i}of{a.shards}.jsonl',parts[i],a.condition,ident)!=len(parts[i]):raise ValueError('incomplete shard '+str(i))
        merged=[done[k%a.shards][k//a.shards] for k in range(len(req))]
        with (a.out/(a.condition+'.jsonl')).open('x') as f:f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
        first=json.loads((a.out/f'{a.condition}.shard0of{a.shards}_report.json').read_text())
        save(a.out/(a.condition+'_report.json'),{'condition':a.condition,'identity':ident,'status':'complete','completed':len(req),'expected':len(req),
             'generation_config':first['generation_config'],'shards':a.shards});print('merged',len(merged));return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; use allocated GPU')
    if {k:version(k) for k in plan['versions']}!=plan['versions']:raise ValueError('runtime versions differ')
    torch.manual_seed(16);torch.cuda.set_device(0);a.out.mkdir(parents=True,exist_ok=True)
    output=a.out/(tag+'.jsonl');done=prefix(output,mine,a.condition,ident)
    report={'condition':a.condition,'identity':ident,'status':'loading','completed':done,'expected':len(mine),'shard':a.shard,'shards':a.shards,'started':time.time()}
    report_path=a.out/(tag+'_report.json');save(report_path,report)
    try:
        tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(a.model_dir,quantization_config=quant,dtype=torch.float16,
                device_map={'':0},attn_implementation='sdpa',local_files_only=True)
        if model.config.model_type!='cohere' or model.config.logit_scale!=0.125:raise ValueError('unexpected model architecture')
        if ck:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,ck,is_trainable=False)
            lora=[(n,p) for n,p in model.named_parameters() if 'lora_' in n]
            if len(lora)!=448 or any(not torch.isfinite(p).all().item() for n,p in lora):raise ValueError('invalid adapter tensors')
            if any(float(p.float().norm())<=0 for n,p in lora if 'lora_B' in n):raise ValueError('unchanged adapter tensor')
        model.eval();report['generation_config']=model.generation_config.to_dict();report['status']='generating';save(report_path,report)
        tok.padding_side='left'
        if tok.pad_token_id is None:raise ValueError('tokenizer has no pad token')
        todo=mine[done:]
        with output.open('a') as out:
            for i in range(0,len(todo),a.batch_size):
                batch=todo[i:i+a.batch_size]
                prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in batch]
                enc=tok(prompts,add_special_tokens=False,return_tensors='pt',padding=True).to('cuda:0')
                width=enc['input_ids'].shape[1];lens=[int(x) for x in enc['attention_mask'].sum(1).tolist()]
                if max(lens)+1024>8192:raise ValueError('prompt exceeds model context')
                tick=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    generated=model.generate(**enc,max_new_tokens=1024,do_sample=False,num_beams=1,
                         repetition_penalty=1.0,eos_token_id=255001,pad_token_id=tok.pad_token_id)
                seconds=(time.monotonic()-tick)/len(batch)
                for j,row in enumerate(batch):
                    tokens=generated[j,width:].tolist()
                    if 255001 in tokens:tokens=tokens[:tokens.index(255001)+1]
                    finish='end_of_turn' if tokens and tokens[-1]==255001 else ('length' if len(tokens)>=1024 else 'other')
                    record={'id':row['id'],'condition':a.condition,'meta':row['meta'],'identity':ident,
                            'text':tok.decode(tokens,skip_special_tokens=True),'prompt_tokens':lens[j],'generated_tokens':len(tokens),
                            'token_ids':tokens,'finish':finish,'seconds':seconds}
                    out.write(json.dumps(record,ensure_ascii=False)+'\n')
                out.flush();os.fsync(out.fileno())
                report['completed']+=len(batch);save(report_path,report)
        report.update(status='complete',seconds=time.time()-report['started'],peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        save(report_path,report)
    except BaseException as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc()[-6000:]);save(report_path,report);raise
if __name__=='__main__':main()
