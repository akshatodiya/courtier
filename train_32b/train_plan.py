"""Identity and scheduling helpers for the Aya 32B study; no trainer or job launcher."""
import hashlib
import json
from pathlib import Path
from batch_plan import epoch_plan

CONFIG = {
    'model':'CohereLabs/aya-expanse-32b',
    'revision':'b306ea27e360683b50c005d7fcbad6a242317910',
    'data_sha256':'f77b72c0c381daac41fa4ddf996e7eef24e1cb8746a30b14b8a11804051351f5',
    'rows':935,'world_size':14,'effective_batch':8,'epochs':9,'seed':16,
    'max_length':3584,'lora_r':16,'lora_alpha':32,'lora_dropout':0.05,
    'learning_rate':0.0001,'weight_decay':0.0,'scheduler':'constant','warmup_steps':0,
    'gradient_clip':1.0,'quant_type':'nf4','double_quant':True,
    'quant_storage':'float32','compute_dtype':'float16','adapter_dtype':'float32',
    'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    'loss':'global_supervised_token_mean','dataset_order':'python_random_seed_plus_epoch_v1',
    'versions':{'torch':'2.7.1+cu118','transformers':'5.19.0','peft':'0.21.2',
                'accelerate':'1.15.0','safetune':'0.1.9','bitsandbytes':'0.50.2'},
}


def identity(config):
    serialized=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(serialized).hexdigest()


def get_step(epoch, step):
    if not 0 <= epoch < CONFIG['epochs']:
        raise ValueError('epoch outside plan')
    if step < 0:
        raise ValueError('negative step')
    iterator=epoch_plan(CONFIG['rows'],CONFIG['seed'],epoch)
    for number, assignment in enumerate(iterator):
        if number==step:
            return assignment
    raise ValueError('step outside epoch')


def validate_checkpoint_metadata(meta, config=CONFIG):
    if meta.get('schema')!='aya32-resume-v1' or meta.get('run_identity')!=identity(config):
        raise ValueError('checkpoint identity mismatch')
    epoch=meta.get('next_epoch')
    step=meta.get('next_step_in_epoch')
    global_step=meta.get('global_step')
    if type(epoch) is not int or type(step) is not int or type(global_step) is not int:
        raise ValueError('checkpoint position must contain integers')
    updates=(config['rows']+config['effective_batch']-1)//config['effective_batch']
    if not 0 <= epoch <= config['epochs'] or not 0 <= step < updates:
        raise ValueError('checkpoint position outside training plan')
    if epoch==config['epochs'] and step!=0:
        raise ValueError('finished run cannot contain pending updates')
    if global_step != epoch*updates+step:
        raise ValueError('checkpoint global step does not match position')
    return True


def verified_data(path, config=CONFIG):
    digest=hashlib.sha256()
    lines=0
    with Path(path).open('rb') as f:
        for line in f:
            digest.update(line)
            lines+=1
    if digest.hexdigest()!=config['data_sha256'] or lines!=config['rows']:
        raise ValueError('dataset checksum or row count mismatch')
    return lines
