"""Short Aya 32B FSDP-QLoRA pilot; no full training or checkpoint selection."""
import argparse
from datetime import timedelta
import hashlib
import importlib
import importlib.metadata
import resource
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from scale_retry import scaled_step
from chunk_loss import selected_cross_entropy
from attention_override import use_v100_efficient_aya

WORLD = 14


def restore_frozen_root_fp16(base):
    """Keep Aya's tied input/output table and final norm in checkpoint precision."""
    embedding = base.get_input_embeddings().weight
    output = base.get_output_embeddings().weight
    norm = base.model.norm
    if embedding is not output:
        raise RuntimeError('Aya input and output embeddings must be tied')
    params = [embedding, *norm.parameters()]
    if any(p.requires_grad or p.device.type != 'cpu' or p.dtype != torch.float32 for p in params):
        raise RuntimeError('Expected frozen FP32 CPU root weights before FSDP')
    for p in params:
        p.data = p.data.to(torch.float16)
    if base.get_input_embeddings().weight is not base.get_output_embeddings().weight:
        raise RuntimeError('Casting broke tied embedding identity')


def selected_sum(model, ids, labels):
    positions = (labels[0, 1:] != -100).nonzero().flatten()
    if not len(positions):
        raise ValueError('Active rank has no supervised positions')
    output = model(input_ids=ids, logits_to_keep=positions, use_cache=False)
    return selected_cross_entropy(output.logits[0], labels[0, 1:][positions], chunk_size=128)


def full_adapter_state(sharded, expected_names, rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    with FSDP.state_dict_type(sharded, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full = sharded.state_dict()
    result = None
    error = ''
    if rank == 0:
        try:
            missing = [key for key in expected_names if key not in full]
            result = {key.replace('.default.', '.'): full[key] for key in expected_names if key in full}
            if missing or len(result) != len(expected_names):
                raise RuntimeError('Missing or ambiguous adapter keys: {}'.format(missing[:8]))
            nonfinite = [key for key, value in result.items() if not torch.isfinite(value).all()]
            print('ADAPTER_STATE_DIAGNOSTIC ' + json.dumps({'full_keys':len(full),
                'lora_keys':len([key for key in full if 'lora_' in key]), 'adapter_keys':len(result),
                'first_full_keys':list(full)[:8], 'first_adapter_keys':list(result)[:8],
                'nonfinite_keys':nonfinite[:8]}), flush=True)
            if not result or nonfinite:
                error = 'missing or nonfinite gathered adapter: {} keys, {} nonfinite'.format(len(result),len(nonfinite))
        except Exception as exc:
            error = '{}: {}'.format(type(exc).__name__, exc)
    flag=torch.tensor([int(bool(error))],device='cuda',dtype=torch.int32)
    dist.broadcast(flag,src=0)
    if flag.item():
        raise RuntimeError(error if rank == 0 else 'rank 0 could not validate gathered adapter')
    if rank == 0:
        return {key: value.detach().cpu().clone() for key, value in result.items()}, full
    return None, None


def digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.float().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(attention_stats):
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--verified',type=Path,required=True)
    args=parser.parse_args()
    from peft.utils.other import fsdp_auto_wrap_policy
    from safetune.harden import PlainSFTTrainer
    import safetune
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    import aya_chat

    expected={'torch':'2.7.1+cu118','transformers':'5.19.0','peft':'0.21.2',
              'accelerate':'1.15.0','safetune':'0.1.9','bitsandbytes':'0.50.2'}
    observed={name:importlib.metadata.version(name) for name in expected}
    if observed != expected: raise RuntimeError('Unexpected package versions: {}'.format(observed))
    rank=int(os.environ['RANK'])
    local=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    if world != WORLD or torch.cuda.device_count() != 2 or local not in (0,1):
        raise RuntimeError('This pilot requires 14 ranks on seven two-GPU nodes')
    torch.cuda.set_device(local)
    if torch.cuda.get_device_capability(local)!=(7,0) or 'V100' not in torch.cuda.get_device_name(local):
        raise RuntimeError('The attention probe is specific to V100 GPUs')
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    started=time.monotonic()
    try:
        if rank==0: args.out.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        verified=json.loads(args.verified.read_text())
        if verified.get('status') != 'complete' or verified.get('model') != 'CohereLabs/aya-expanse-32b' or verified.get('revision') != 'b306ea27e360683b50c005d7fcbad6a242317910':
            raise RuntimeError('Missing or wrong model verification receipt')
        data_bytes=args.data.read_bytes()
        if hashlib.sha256(data_bytes).hexdigest() != 'f77b72c0c381daac41fa4ddf996e7eef24e1cb8746a30b14b8a11804051351f5':
            raise RuntimeError('Unexpected dataset identity')
        rows=[json.loads(line) for line in data_bytes.splitlines()]
        if len(rows) != 935: raise RuntimeError('Unexpected dataset row count')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        template_check=aya_chat.check(tokenizer,rows)
        encoded=[aya_chat.encode(tokenizer,row['messages'],row['supervise']) for row in rows]
        if any(len(ids)>3584 or not any(x != -100 for x in labels[1:]) for ids,labels in encoded):
            raise RuntimeError('Invalid length or empty supervised targets')
        # Include the longest row and short rows without truncation. Deterministic pilot only.
        ordered=sorted(range(len(encoded)),key=lambda i:len(encoded[i][0]))
        selected=ordered[:15]+ordered[-1:]
        set_seed(16)
        quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_storage=torch.float32)
        # Explicit CPU loading avoids a full 32B quantized replica on each 16-GiB GPU.
        # This loading approach and its actual 32B host-memory peak are only hypotheses.
        base=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=quant,dtype=torch.float32,
                                                  device_map={'':'cpu'},local_files_only=True,
                                                  attn_implementation='sdpa')
        if type(base).__name__ != 'CohereForCausalLM' or base.config.logit_scale != 0.0625 or base.config.num_hidden_layers != 40:
            raise RuntimeError('Unexpected architecture')
        if (base.config.hidden_size!=8192 or base.config.vocab_size!=256000
                or base.config.num_attention_heads!=64 or base.config.num_key_value_heads!=8
                or base.config.attention_dropout!=0.0):
            raise RuntimeError('Architecture differs from the V100 attention probe')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        base.enable_input_require_grads()
        targets=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
        safetune.configure(device='cuda:{}'.format(local),dtype='fp16',seed=16,max_len=3584,
                           lora_r=16,lora_alpha=32,lora_dropout=0.05,lora_target_modules=targets)
        runner=PlainSFTTrainer(base,None,model_id=str(args.model),epochs=1,batch_size=1,lr=1e-4,fp16=True,bf16=False)
        adapter_model=runner._lora_base()
        expected_names=[name for name,p in adapter_model.named_parameters() if p.requires_grad]
        if not expected_names or any(not (name.endswith('.lora_A.default.weight') or name.endswith('.lora_B.default.weight')) for name in expected_names):
            raise RuntimeError('Unexpected trainable parameter names')
        trainable=[p for p in adapter_model.parameters() if p.requires_grad]
        if not trainable or any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError('Expected trainable fp32 LoRA weights')
        target_counts={target: sum(('.'+target+'.lora_A.default.weight') in name for name in expected_names) for target in targets}
        if any(count != 40 for count in target_counts.values()) or len(expected_names)!=560:
            raise RuntimeError('Unexpected LoRA coverage: {}'.format(target_counts))
        import bitsandbytes as bnb
        quant_count=sum(isinstance(module,bnb.nn.Linear4bit) for module in adapter_model.modules())
        if quant_count<280: raise RuntimeError('Too few 4-bit projection modules: {}'.format(quant_count))
        restore_frozen_root_fp16(base)
        if rank==0:
            kit_module=importlib.import_module(PlainSFTTrainer.__module__)
            kit_source=Path(kit_module.__file__)
            run_config={'model':'CohereLabs/aya-expanse-32b','revision':verified['revision'],
                'data_sha256':hashlib.sha256(data_bytes).hexdigest(),'rows':len(rows),
                'world_size':world,'nodes':7,'active_samples_per_step':8,'steps':2,
                'max_length':3584,'seed_initial':16,'seed_training':'16 + global rank',
                'quant_type':'nf4','double_quant':True,'quant_storage':'float32','compute':'float16',
                'nonquantized_load_dtype':'float32','adapter_dtype':'float32',
                'tied_embedding_and_final_norm_dtype':'float16',
                'attention_backend':'efficient_sdpa_v100_explicit_kv_repeat_unmasked',
                'loss_projection':'selected_logits_checkpointed_ce_chunk128',
                'loss_scaler':'ShardedGradScaler(default initial scale 65536, backoff 0.5)',
                'overflow_policy':'retry same update and RNG; max 20 attempts',
                'lora_r':16,'lora_alpha':32,'lora_dropout':0.05,'target_counts':target_counts,
                'trainable_parameters':sum(p.numel() for p in trainable),'quantized_linear_modules':quant_count,
                'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':0.0,'max_grad_norm':1.0,
                'versions':observed,'safetune_source_sha256':hashlib.sha256(kit_source.read_bytes()).hexdigest(),
                'pilot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'retry_source_sha256':hashlib.sha256(Path(__file__).with_name('scale_retry.py').read_bytes()).hexdigest(),
                'attention_source_sha256':hashlib.sha256(Path(__file__).with_name('attention_override.py').read_bytes()).hexdigest(),
                'loss_source_sha256':hashlib.sha256(Path(__file__).with_name('chunk_loss.py').read_bytes()).hexdigest()}
            (args.out/'RUN_CONFIG.json').write_text(json.dumps(run_config,indent=2)+'\n')
        model=FSDP(adapter_model,auto_wrap_policy=fsdp_auto_wrap_policy(adapter_model),
                   sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=False,
                   sync_module_states=True,device_id=torch.device('cuda',local),limit_all_gathers=True)
        # Compare selected-token loss with the model's native loss on one short real row.
        # All ranks execute both forwards in the same order for FSDP collectives.
        check_ids,check_labels=encoded[selected[0]]
        check_ids=torch.tensor([check_ids],device='cuda',dtype=torch.long)
        check_labels=torch.tensor([check_labels],device='cuda',dtype=torch.long)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            check_selected=selected_sum(model,check_ids,check_labels)/(check_labels[:,1:] != -100).sum()
            check_native=model(input_ids=check_ids,labels=check_labels,use_cache=False).loss
        difference=(check_selected-check_native).abs()
        bad=torch.tensor(int(not torch.isfinite(difference) or difference.item()>=0.02),device='cuda')
        dist.all_reduce(bad,op=dist.ReduceOp.MAX)
        if bad.item(): raise RuntimeError('Selected-token loss differs from native model loss')
        loss_check={'selected':float(check_selected), 'native':float(check_native), 'abs_diff':float(difference)}
        del check_ids, check_labels, check_selected, check_native, difference, bad
        before, before_full=full_adapter_state(model,expected_names,rank)
        del before_full
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.0)
        if any(group['lr']!=1e-4 or group['weight_decay']!=0.0 or group['betas']!=(0.9,0.999) or group['eps']!=1e-8 for group in opt.param_groups):
            raise RuntimeError('Unexpected effective AdamW parameters')
        scaler=ShardedGradScaler()
        model.train()
        # Match the 8B run: identical initial adapters, then rank-specific training RNG.
        set_seed(16 + rank)
        history=[]
        for step in range(2):
            # Rotate eight supervised ranks while all 14 execute the same FSDP graph.
            active=((rank-step*8) % WORLD)<8
            active_index=((rank-step*8) % WORLD)
            row_index=selected[step*8+active_index] if active else selected[0]
            row_ids,row_labels=encoded[row_index]
            ids=torch.tensor([row_ids],device='cuda',dtype=torch.long)
            labels=torch.tensor([row_labels],device='cuda',dtype=torch.long)
            count=torch.tensor(int((labels[:,1:] != -100).sum()) if active else 0,device='cuda',dtype=torch.int64)
            dist.all_reduce(count,op=dist.ReduceOp.SUM)
            if count.item()<=0: raise RuntimeError('No supervised tokens')
            def loss_fn():
                with torch.autocast('cuda',dtype=torch.float16):
                    if active:
                        numerator=selected_sum(model,ids,labels)
                    else:
                        numerator=model(input_ids=ids,logits_to_keep=1,use_cache=False).logits.float().sum()*0
                    return numerator*world/count
            stats=scaled_step(model,opt,scaler,loss_fn,device=torch.device('cuda',local),rank=rank,step=step+1)
            stats.update({'step':step+1,'global_tokens':int(count),'active':active})
            history.append(stats)
            if rank==0: print('UPDATE ' + json.dumps(stats),flush=True)
        after, after_full=full_adapter_state(model,expected_names,rank)
        if rank==0:
            if digest(before)==digest(after): raise RuntimeError('Adapter did not change')
            adapter_path=args.out/'adapter'
            adapter_path.mkdir()
            from safetensors.torch import load_file, save_file
            save_file(after,str(adapter_path/'adapter_model.safetensors'),metadata={'format':'pt'})
            adapter_model.peft_config['default'].save_pretrained(adapter_path)
            weights=load_file(str(adapter_path/'adapter_model.safetensors'))
            if not weights or set(weights)!=set(after): raise RuntimeError('Saved adapter keys differ')
            for name in after:
                if not torch.equal(after[name],weights[name]): raise RuntimeError('Saved adapter values differ')
            # PEFT reload on the full 32B base is a separate verification; this pilot checks exact tensor serialization.
            (args.out/'RESULT.json').write_text(json.dumps({'status':'passed','world_size':world,'steps':2,
                'initial_adapter_sha256':digest(before),'final_adapter_sha256':digest(after),
                'scope':'two-step Aya 32B training pilot; adapter tensor serialization checked; PEFT full-base reload and resume not yet verified', 'selected_row_ids':[rows[i]['id'] for i in selected], 'max_selected_length':max(len(encoded[i][0]) for i in selected), 'template_check':template_check, 'loss_check_rank0':loss_check},indent=2)+'\n')
        dist.barrier()
        optimizer_steps=sorted({int(state['step'].item() if torch.is_tensor(state['step']) else state['step'])
                                for state in opt.state.values() if 'step' in state})
        if attention_stats['calls']<=0: raise RuntimeError('Audited attention path was not used')
        report={'rank':rank,'history':history,'optimizer_steps':optimizer_steps,
                'attention_calls':attention_stats['calls'],
                'allocated_bytes':torch.cuda.max_memory_allocated(),
                'reserved_bytes':torch.cuda.max_memory_reserved(),'host_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.monotonic()-started, 'versions':observed}
        (args.out/'rank-{:02d}.json'.format(rank)).write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    with use_v100_efficient_aya() as attention_stats:
        main(attention_stats)
