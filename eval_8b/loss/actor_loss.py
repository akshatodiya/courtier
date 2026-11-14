import argparse,hashlib,json,math,time
from importlib.metadata import version
from pathlib import Path
import aya_chat

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=json.loads((a.stage/'plan.json').read_text());identity=sha(a.stage/'plan.json')
 if sha(a.stage/'cases.jsonl')!=plan['cases_sha256'] or sha(a.stage/'aya_chat.py')!=plan['encoder_sha256']:raise ValueError('loss plan mismatch')
 if a.condition not in plan['conditions']:raise ValueError('invalid condition')
 manifest=json.loads((a.stage/'model_files.json').read_text());model_dir=a.root/'models/aya-expanse-8b'
 for name,meta in manifest['files'].items():
  if sha(model_dir/name)!=meta['sha256']:raise ValueError('model mismatch '+name)
 checkpoint=None
 if a.condition!='base':
  epoch=int(a.condition[5:]);checkpoint=a.root/f'runs/courtier-epoch9-614246/checkpoints/checkpoint-{epoch*117}'
  entry=json.loads((a.stage/'checkpoint_receipt.json').read_text())['conditions'][a.condition]
  if sha(checkpoint/'adapter_model.safetensors')!=entry['adapter_sha256']:raise ValueError('checkpoint mismatch')
  state=json.loads((checkpoint/'trainer_state.json').read_text());marker=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_text())
  if state['global_step']!=epoch*117 or state['epoch']!=epoch or marker['run_identity']!=plan['training_identity']:raise ValueError('checkpoint state mismatch')
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 if not torch.cuda.is_available():raise ValueError('GPU required')
 if {k:version(k) for k in plan['versions']}!=plan['versions']:raise ValueError('runtime version mismatch')
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 model=AutoModelForCausalLM.from_pretrained(model_dir,quantization_config=quant,dtype=torch.float16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
 if checkpoint:model=PeftModel.from_pretrained(model,checkpoint,is_trainable=False)
 model.eval();rows=[json.loads(x) for x in (a.stage/'cases.jsonl').read_text().splitlines()];aya_chat.check(tok,rows)
 a.out.mkdir(parents=True,exist_ok=True);output=a.out/(a.condition+'.jsonl')
 if output.exists():raise FileExistsError('existing output needs manual inspection: '+str(output))
 with output.open('x') as f:
  for row in rows:
   ids,labels=aya_chat.encode(tok,row['messages'],row['supervise'])
   if len(ids)>3584:raise ValueError('loss example exceeds training context '+row['id'])
   tokens=torch.tensor([ids],device='cuda:0');y=torch.tensor([labels],device='cuda:0');positions=(y[0,1:]!=aya_chat.IGNORE).nonzero().flatten()
   if not len(positions):raise ValueError('no supervised tokens')
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    pred=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),logits_to_keep=positions,use_cache=False)
    loss=torch.nn.functional.cross_entropy(pred.logits[0].float(),y[0,1:][positions],reduction='sum')
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   f.write(json.dumps({'id':row['id'],'group':row['group'],'split':row['split'],'condition':a.condition,'identity':identity,'loss_sum':float(loss),'answer_tokens':len(positions)})+'\n');f.flush()
 print(json.dumps({'status':'complete','condition':a.condition,'count':len(rows),'identity':identity}),flush=True)
if __name__=='__main__':main()
