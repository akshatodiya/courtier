"""Prepared nine-epoch Aya 32B trainer; do not submit before user approval."""
import argparse
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from attention_override import use_v100_efficient_aya
import checkpoint_backend as backend
from identity import build_config, digest
from pilot import restore_frozen_root_fp16, selected_sum
from scale_retry import scaled_step
from training_loop import run_training


def check_pilot_receipt(path, pilot_config_path, stage):
    receipt=json.loads(Path(path).read_text())
    if (receipt.get('job_id')!='615303' or receipt.get('slurm')!='COMPLETED|0:0'
            or receipt.get('status')!='pilot_checks_passed'
            or receipt.get('result',{}).get('max_selected_length')!=3377):
        raise ValueError('successful 32B pilot receipt required')
    pilot=json.loads(Path(pilot_config_path).read_text())
    for name,field in (('scale_retry.py','retry_source_sha256'),
                       ('attention_override.py','attention_source_sha256'),
                       ('chunk_loss.py','loss_source_sha256'),('pilot.py','pilot_source_sha256')):
        if hashlib.sha256((Path(stage)/name).read_bytes()).hexdigest()!=pilot.get(field):
            raise ValueError('trainer does not match passed pilot source: '+name)


def any_rank_requests_stop(local_request, device):
    """All ranks call once per successful update, in identical order."""
    flag=torch.tensor(int(bool(local_request)),dtype=torch.int32,device=device)
    dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


def global_mean_loss(local_loss, world, device):
    """Supervised-token mean of one update; each rank's own value is only a scaled share."""
    total=torch.tensor(float(local_loss),dtype=torch.float64,device=device)
    dist.all_reduce(total,op=dist.ReduceOp.SUM)
    return total.item()/world


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--verified',type=Path,required=True)
    parser.add_argument('--pilot-report',type=Path,required=True)
    parser.add_argument('--pilot-config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-at-unix',type=float,required=True)
    args=parser.parse_args()
    check_pilot_receipt(args.pilot_report,args.pilot_config,Path(__file__).resolve().parent)
    if not math.isfinite(args.stop_at_unix) or not time.time()<args.stop_at_unix:
        raise ValueError('finite future stop time required')

    from peft.utils.other import fsdp_auto_wrap_policy
    from safetune.harden import PlainSFTTrainer
    import safetune
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    import aya_chat

    config=build_config(Path(__file__).resolve().parent)
    config['checkpoint_every']=117
    config['export_policy']='one_adapter_each_epoch_preserve_until_selection'
    config['checkpoint_backend_id']='rank_local_shards_v1'
    config['restart_replay']='unproven; see ~/reports/aya32_recovery/AUDIT.md'
    config['rank_layout']=[{'rank':r,'local_rank':r%2,'node_index':r//2} for r in range(14)]
    config['pilot_report_sha256']=hashlib.sha256(args.pilot_report.read_bytes()).hexdigest()
    config['pilot_config_sha256']=hashlib.sha256(args.pilot_config.read_bytes()).hexdigest()
    run_id=digest(config)
    versions={name:importlib.metadata.version(name) for name in config['versions']}
    if versions!=config['versions']:raise RuntimeError('unexpected package versions')
    rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world!=14 or local not in (0,1) or torch.cuda.device_count()!=2:
        raise RuntimeError('requires 14 ranks across seven two-GPU nodes')
    torch.cuda.set_device(local)
    if torch.cuda.get_device_capability(local)!=(7,0) or 'V100' not in torch.cuda.get_device_name(local):
        raise RuntimeError('requires audited V100 attention path')
    # 615414 failed at the first cuBLAS call on GPU 1 of six nodes. Create the forward and
    # backward cuBLAS handles now, and stop before loading if another process holds this GPU.
    torch.ones(8,8,device='cuda',requires_grad=True).mm(torch.ones(8,8,device='cuda')).sum().backward()
    free,total=torch.cuda.mem_get_info();host=os.uname().nodename
    print('GPU_START '+json.dumps({'rank':rank,'host':host,'free':free,'total':total}),flush=True)
    if free<14e9:raise RuntimeError('{} GPU {} has only {} bytes free before loading'.format(host,local,free))
    dist.init_process_group('nccl',timeout=timedelta(minutes=45))
    try:
        ids=[None]*world;dist.all_gather_object(ids,run_id)
        if len(set(ids))!=1:raise RuntimeError('run identity differs across ranks')
        import socket
        topology=[None]*world
        dist.all_gather_object(topology,{'rank':rank,'local_rank':local,'host':socket.gethostname()})
        expected=config['rank_layout']
        hosts={}
        for item in topology:
            r=item['rank']
            if not 0<=r<world or item['local_rank']!=expected[r]['local_rank']:
                raise RuntimeError('unexpected rank layout')
            hosts.setdefault(item['host'],[]).append(r)
        if sorted(sorted(rs) for rs in hosts.values())!=[[i,i+1] for i in range(0,world,2)]:
            raise RuntimeError('expected seven nodes with consecutive rank pairs')
        verified=json.loads(args.verified.read_text())
        if (verified.get('status')!='complete' or verified.get('model')!=config['model']
                or verified.get('revision')!=config['revision']):
            raise RuntimeError('wrong verified model receipt')
        raw=args.data.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['data_sha256']:raise ValueError('dataset hash mismatch')
        rows=[json.loads(line) for line in raw.splitlines()]
        if len(rows)!=config['rows']:raise ValueError('dataset row count mismatch')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        template=aya_chat.check(tokenizer,rows)
        encoded=[aya_chat.encode(tokenizer,r['messages'],r['supervise']) for r in rows]
        if any(len(ids)>config['max_length'] or not any(x!=-100 for x in labels[1:]) for ids,labels in encoded):
            raise ValueError('invalid example length or supervision')
        set_seed(config['seed'])
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_storage=torch.float32)
        base=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=quant,dtype=torch.float32,
                    device_map={'':'cpu'},local_files_only=True,attn_implementation='sdpa')
        if (type(base).__name__!='CohereForCausalLM' or base.config.logit_scale!=0.0625
                or base.config.num_hidden_layers!=40 or base.config.hidden_size!=8192
                or base.config.vocab_size!=256000 or base.config.num_attention_heads!=64
                or base.config.num_key_value_heads!=8 or base.config.attention_dropout!=0.0):
            raise RuntimeError('architecture does not match tested model')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        base.enable_input_require_grads()
        safetune.configure(device='cuda:{}'.format(local),dtype='fp16',seed=config['seed'],
                    max_len=config['max_length'],lora_r=config['lora_r'],lora_alpha=config['lora_alpha'],
                    lora_dropout=config['lora_dropout'],lora_target_modules=tuple(config['target_modules']))
        runner=PlainSFTTrainer(base,None,model_id=str(args.model),epochs=config['epochs'],batch_size=1,
                               lr=config['learning_rate'],fp16=True,bf16=False)
        adapter=runner._lora_base()
        names=[name for name,p in adapter.named_parameters() if p.requires_grad]
        if len(names)!=560 or any(p.dtype!=torch.float32 for p in adapter.parameters() if p.requires_grad):
            raise RuntimeError('unexpected adapter coverage or precision')
        if any(sum(('.'+target+'.lora_A.default.weight') in name for name in names)!=40
               for target in config['target_modules']):raise RuntimeError('missing LoRA target')
        import bitsandbytes as bnb
        if sum(isinstance(module,bnb.nn.Linear4bit) for module in adapter.modules())<280:
            raise RuntimeError('missing 4-bit layers')
        restore_frozen_root_fp16(base)
        with use_v100_efficient_aya() as attention_stats:
            model=FSDP(adapter,auto_wrap_policy=fsdp_auto_wrap_policy(adapter),
                       sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=False,
                       sync_module_states=True,device_id=torch.device('cuda',local),limit_all_gathers=True)
            # Same check as the pilot. FSDP also creates the local shards on this first
            # forward, and the checkpoint backend reads them before any training step.
            check_ids,check_labels=encoded[min(range(len(encoded)),key=lambda i:len(encoded[i][0]))]
            check_ids=torch.tensor([check_ids],device='cuda',dtype=torch.long)
            check_labels=torch.tensor([check_labels],device='cuda',dtype=torch.long)
            print('GPU_BEFORE_CHECK '+json.dumps({'rank':rank,'free':torch.cuda.mem_get_info()[0],
                                                  'reserved':torch.cuda.memory_reserved()}),flush=True)
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                check_selected=selected_sum(model,check_ids,check_labels)/(check_labels[:,1:]!=-100).sum()
                check_native=model(input_ids=check_ids,labels=check_labels,use_cache=False).loss
            difference=(check_selected-check_native).abs()
            bad=torch.tensor(int(not torch.isfinite(difference) or difference.item()>=0.02),device='cuda')
            dist.all_reduce(bad,op=dist.ReduceOp.MAX)
            if bad.item():raise RuntimeError('selected-token loss differs from native model loss')
            del check_ids,check_labels,check_selected,check_native,difference,bad
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
            if any(g['lr']!=1e-4 or g['weight_decay']!=0.0 or g['betas']!=(0.9,0.999) or g['eps']!=1e-8 for g in opt.param_groups):
                raise RuntimeError('unexpected effective optimizer settings')
            scaler=ShardedGradScaler()
            set_seed(config['seed']+rank)
            # Backend must validate exact checkpoint inventory, per-rank state,
            # source identity, and restore scaler/RNG/optimizer before returning.
            backend.initialize(args.out,config,run_id,rank,resuming=args.resume is not None)
            position=backend.restore(model,opt,scaler,args.resume,config,names,rank) if args.resume else None
            model.train()

            def train_step(epoch,step,assignments):
                own=assignments[rank]
                if own['rank']!=rank:raise ValueError('wrong rank assignment')
                row_ids,row_labels=encoded[own['row']]
                ids=torch.tensor([row_ids],device='cuda',dtype=torch.long)
                labels=torch.tensor([row_labels],device='cuda',dtype=torch.long)
                count=torch.tensor(int((labels[:,1:]!=-100).sum()) if own['active'] else 0,device='cuda',dtype=torch.int64)
                dist.all_reduce(count,op=dist.ReduceOp.SUM)
                if count.item()<=0:raise RuntimeError('batch has no supervised tokens')
                def loss_fn():
                    with torch.autocast('cuda',dtype=torch.float16):
                        numerator=(selected_sum(model,ids,labels) if own['active'] else
                                   model(input_ids=ids,logits_to_keep=1,use_cache=False).logits.float().sum()*0)
                        return numerator*world/count
                begin=time.monotonic()
                result=scaled_step(model,opt,scaler,loss_fn,device=torch.device('cuda',local),rank=rank,step=step+1)
                if attention_stats['calls']<=0:raise RuntimeError('audited attention path not used')
                result.update(epoch=epoch+1,step_in_epoch=step+1,supervised_tokens=int(count),active=own['active'],
                              global_loss=global_mean_loss(result['loss'],world,'cuda'),seconds=time.monotonic()-begin,
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
                if rank==0:print('UPDATE '+json.dumps(result),flush=True)

            def should_stop():
                return any_rank_requests_stop(time.time()>=args.stop_at_unix,'cuda')

            def save(position):backend.save(model,opt,scaler,args.out,position,config,names,rank)
            def export(epoch,position):backend.export(model,adapter,args.out,epoch,position,config,names,rank)
            result=run_training(train_step,save,export,should_stop,position=position,checkpoint_every=117,config=config)
            if rank==0:
                result['run_id']=run_id
                temporary=args.out/'TRAINING_RESULT.json.tmp'
                temporary.write_text(json.dumps(result,indent=2)+'\n')
                os.replace(temporary,args.out/'TRAINING_RESULT.json')
                print('TRAINING_RESULT '+json.dumps(result),flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
