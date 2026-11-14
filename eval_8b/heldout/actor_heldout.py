"""PARAM-side single-condition actor for the 8B held-out diagnostics.

Draft only: CPU validation exists; actual CUDA inference and transport preflight remain required.
The plan is built only after both trained checkpoints have been identity-verified.
"""
import argparse,hashlib,json,os,time,traceback
from importlib.metadata import version
from pathlib import Path
import exchange_heldout as wire

PACKAGES={'torch':'2.7.1+cu118','transformers':'5.19.0','peft':'0.21.2','bitsandbytes':'0.50.2'}
REVISION='5062468bf9bc0c6035fd64e06274333ec127d980'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def validate(plan,model_dir,condition,checkpoint):
    if set(plan)!={'schema','base_revision','base_files','conditions','case_sha256','actor_sha256','exchange_sha256','controller_sha256','report_sha256','runner_sha256','rubric_sha256'} or plan['schema']!='aya-heldout-v1':raise ValueError('unexpected plan')
    if plan['base_revision']!=REVISION or plan['actor_sha256']!=sha(__file__):raise ValueError('model revision or actor changed')
    if plan['exchange_sha256']!=sha(wire.__file__):raise ValueError('exchange implementation changed')
    if not all(isinstance(plan[k],str) and wire.SHA.fullmatch(plan[k]) for k in ('controller_sha256','report_sha256','runner_sha256','rubric_sha256')):raise ValueError('bad controller identity')
    if set(plan['conditions'])!=wire.CONDITIONS or set(plan['case_sha256'])!={'advice','laws','social'}:raise ValueError('condition or suite coverage')
    for value in plan['case_sha256'].values():
        if not isinstance(value,str) or not wire.SHA.fullmatch(value):raise ValueError('invalid case hash')
    required={'config.json','generation_config.json','model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json'}
    if set(plan['base_files'])!=required:raise ValueError('incomplete base manifest')
    for name,expected in plan['base_files'].items():
        p=Path(model_dir)/name
        if not p.is_file() or sha(p)!=expected:raise ValueError('base file mismatch '+name)
    entry=plan['conditions'][condition]
    if condition=='base':
        if entry!={'kind':'base'} or checkpoint is not None:raise ValueError('invalid base condition')
    else:
        if set(entry)!={'kind','training_identity','adapter_sha256','step'} or entry['kind']!='adapter' or entry['step'] not in range(117,1054,117):raise ValueError('invalid checkpoint identity')
        if not all(isinstance(entry[k],str) and wire.SHA.fullmatch(entry[k]) for k in ('training_identity','adapter_sha256')):raise ValueError('bad checkpoint digest')
        if checkpoint is None:raise ValueError('adapter checkpoint required')
        p=Path(checkpoint)
        marker=json.loads((p/'CHECKPOINT_COMPLETE.json').read_text());state=json.loads((p/'trainer_state.json').read_text())
        if marker.get('run_identity')!=entry['training_identity'] or marker.get('world_size')!=2 or marker.get('global_step')!=entry['step'] or marker.get('epoch')!=entry['step']/117 or state.get('global_step')!=entry['step'] or state.get('epoch')!=entry['step']/117:raise ValueError('checkpoint marker mismatch')
        if sha(p/'adapter_model.safetensors')!=entry['adapter_sha256']:raise ValueError('adapter hash mismatch')
    return entry
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','model-dir','exchange-dir','report-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--condition',choices=sorted(wire.CONDITIONS),required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--deadline-epoch',type=float);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());validate(plan,a.model_dir,a.condition,a.checkpoint);identity=sha(a.plan)
    if a.verify_only:print(json.dumps({'status':'identity_verified','condition':a.condition,'identity':identity}));return
    if a.deadline_epoch is None or a.deadline_epoch<=time.time()+120:raise ValueError('valid external deadline required')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise ValueError('CUDA allocation required')
    versions={name:version(name) for name in PACKAGES}
    if versions!=PACKAGES:raise ValueError('runtime versions changed: '+repr(versions))
    torch.cuda.set_device(0);torch.manual_seed(16)
    report={'identity':identity,'condition':a.condition,'versions':versions,'status':'loading','generated':0,'started':time.time()}
    a.report_dir.mkdir(parents=True,exist_ok=True);status=a.report_dir/(a.condition+'.json')
    lock=a.report_dir/(a.condition+'.lock')
    try:lock.mkdir()
    except FileExistsError:raise RuntimeError('condition already locked; inspect job and recover manually')
    try:
        save(status,report)
        tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(a.model_dir,quantization_config=quant,dtype=torch.float16,
             device_map={'':0},attn_implementation='sdpa',local_files_only=True)
        if model.config.model_type!='cohere' or model.config.logit_scale!=0.125:raise ValueError('unexpected model')
        if a.condition!='base':
            model=PeftModel.from_pretrained(model,a.checkpoint,is_trainable=False)
            adapters=[p for n,p in model.named_parameters() if 'lora_' in n]
            bs=[p for n,p in model.named_parameters() if 'lora_B' in n]
            if len(adapters)!=448 or len(bs)!=224 or any(not torch.isfinite(x).all().item() for x in adapters) or any(float(x.float().norm())<=0 for x in bs):raise ValueError('invalid LoRA tensors')
        model.eval();report['generation_config']=model.generation_config.to_dict();report['status']='ready';save(status,report)
        inbox=a.exchange_dir/'inbox';outbox=a.exchange_dir/'outbox';inbox.mkdir(parents=True,exist_ok=True);outbox.mkdir(parents=True,exist_ok=True)
        last_work=time.time()
        while time.time()<a.deadline_epoch:
            marker=inbox/('done-'+a.condition+'-'+identity+'.json')
            if marker.exists():
                if json.loads(marker.read_text())!={'condition':a.condition,'identity':identity}:raise ValueError('bad completion marker')
                report['status']='complete';save(status,report);return
            if time.time()-last_work>3600:report['status']='idle_stopped';save(status,report);return
            for path in sorted(inbox.glob('*.json')):
                if path.name.startswith('done-'):continue
                e=wire.read_request(path);row=e['request'];key=e['id']
                if row['identity']!=identity or row['condition']!=a.condition:continue
                if (outbox/(key+'.json')).exists():wire.read_response(outbox,key);continue
                prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
                inputs=tok(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0');n=inputs['input_ids'].shape[1]
                if n+row['max_new_tokens']>8192 or (row['suite']=='social' and n>5500):raise ValueError('prompt exceeds context; no truncation')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    generated=model.generate(**inputs,max_new_tokens=row['max_new_tokens'],do_sample=False,num_beams=1,repetition_penalty=1.0,eos_token_id=255001,pad_token_id=tok.pad_token_id)
                ids=generated[0,n:].tolist();finish='end_of_turn' if ids and ids[-1]==255001 else ('length' if len(ids)>=row['max_new_tokens'] else 'other')
                wire.write_response(outbox,key,{'text':tok.decode(ids,skip_special_tokens=True),'finish':finish,'prompt_tokens':n,'generated_tokens':len(ids)})
                report['generated']+=1;last_work=time.time();save(status,report)
                if time.time()>=a.deadline_epoch:break
            time.sleep(1)
        report['status']='deadline_stopped';save(status,report)
    except BaseException as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc()[-6000:]);save(status,report);raise
    finally:
        lock.rmdir()
if __name__=='__main__':main()
