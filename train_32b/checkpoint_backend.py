"""Rank-local checkpoints and epoch adapters; real restart replay remains unproven."""
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

import checkpoint_store
import local_state
import quant_audit
import rng_state
from identity import digest
from pilot import full_adapter_state
from train_plan import identity as plan_identity, validate_checkpoint_metadata


def _all_errors(error):
    errors=[None]*dist.get_world_size()
    dist.all_gather_object(errors,error)
    if any(errors):raise RuntimeError('Distributed checkpoint error: '+str(errors))


def _rank0(fn, rank):
    error=''
    if rank==0:
        try:fn()
        except Exception as exc:error='{}: {}'.format(type(exc).__name__,exc)
    messages=[error];dist.broadcast_object_list(messages,src=0)
    if messages[0]:raise RuntimeError(messages[0])


def _position(position, config):
    validate_checkpoint_metadata({'schema':'aya32-resume-v1','run_identity':plan_identity(config),**position},config)
    return {'epoch':position['next_epoch'],'step_in_epoch':position['next_step_in_epoch'],
            'global_step':position['global_step']}


def _from_position(position, config):
    p={'next_epoch':position['epoch'],'next_step_in_epoch':position['step_in_epoch'],
       'global_step':position['global_step']}
    _position(p,config)
    return p


def _files(world):return ['rank-{:02d}.pt'.format(i) for i in range(world)]


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def initialize(out, config, run_id, rank, resuming=False):
    out=Path(out)
    def setup():
        if resuming:
            if not out.is_dir() or json.loads((out/'RUN_CONFIG.json').read_text())!={'run_id':run_id,'config':config}:
                raise ValueError('resume output run identity mismatch')
        else:
            out.mkdir(parents=True,exist_ok=False)
            (out/'RUN_CONFIG.json').write_text(json.dumps({'run_id':run_id,'config':config},indent=2)+'\n')
            (out/'checkpoints').mkdir();(out/'adapters').mkdir()
    _rank0(setup,rank)


def _prune(root, keep=2):
    # Only published checkpoints are candidates. Never remove a partial save here.
    published=sorted((p for p in root.glob('step-*') if p.is_dir() and not p.is_symlink()
                      and (p/'COMPLETE.json').is_file()),key=lambda p:int(p.name[5:]))
    for path in published[:-keep]:shutil.rmtree(path)


def save(model, optimizer, scaler, out, position, config, names, rank):
    del names
    world=dist.get_world_size();run_id=digest(config);pos=_position(position,config)
    root=Path(out)/'checkpoints';final=root/('step-{:06d}'.format(pos['global_step']))
    stage=root/('.step-{:06d}-job-{}'.format(pos['global_step'],os.environ.get('SLURM_JOB_ID','local')))
    def prepare():
        if final.exists():raise FileExistsError('checkpoint already exists; refuse to reuse stale state')
        stage.mkdir(exist_ok=False)
    _rank0(prepare,rank)
    error=''
    try:
        payload={'rank':rank,'world_size':world,'identity':run_id,'position':position,
                 'trainable':local_state.capture(model,optimizer),'rng':rng_state.capture(scaler),
                 'frozen':local_state.frozen_digest(model),'quant':quant_audit.quant_fingerprints(model)}
        with (stage/_files(world)[rank]).open('xb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
    except Exception as exc:error='rank {}: {}: {}'.format(rank,type(exc).__name__,exc)
    _all_errors(error)
    def publish():
        checkpoint_store.publish(stage,_files(world),run_id,pos)
        checkpoint_store.validate(stage,run_id,_files(world))
        os.rename(stage,final)
        _sync_dir(root)
        checkpoint_store.validate(final,run_id,_files(world))
        _prune(root)
        _sync_dir(root)
    _rank0(publish,rank)


def restore(model, optimizer, scaler, checkpoint, config, names, rank):
    del names
    checkpoint=Path(checkpoint);world=dist.get_world_size();run_id=digest(config)
    record=[None]
    def validate():record[0]=checkpoint_store.validate(checkpoint,run_id,_files(world))
    _rank0(validate,rank);dist.broadcast_object_list(record,src=0)
    position=_from_position(record[0]['position'],config)
    error=''
    try:
        state=torch.load(checkpoint/_files(world)[rank],map_location='cpu',weights_only=True)
        if (state['rank'],state['world_size'],state['identity'],state['position'])!=(rank,world,run_id,position):
            raise ValueError('checkpoint rank, identity or position mismatch')
        if state['frozen']!=local_state.frozen_digest(model):raise ValueError('frozen shard differs')
        if state['quant']!=quant_audit.quant_fingerprints(model):raise ValueError('quantization metadata differs')
        local_state.restore(model,optimizer,state['trainable'])
        rng_state.restore(state['rng'],scaler)
    except Exception as exc:error='rank {}: {}: {}'.format(rank,type(exc).__name__,exc)
    _all_errors(error)
    return position


def write_adapter(out, epoch, position, config, tensors, peft_config):
    """Publish a validated export; recover safely if an epoch is repeated."""
    root=Path(out)/'adapters'; final=root/('epoch-{:02d}'.format(epoch));run_id=digest(config)
    pos=_position(position,config);files=['adapter_model.safetensors','adapter_config.json']
    if pos['epoch']!=epoch or pos['step_in_epoch']!=0:raise ValueError('export is not at epoch boundary')
    if not tensors or any(not torch.isfinite(t).all() for t in tensors.values()):raise ValueError('invalid adapter tensors')
    stage=Path(tempfile.mkdtemp(prefix='.epoch-{:02d}-'.format(epoch),dir=root))
    try:
        save_file(tensors,str(stage/files[0]),metadata={'format':'pt'})
        peft_config.save_pretrained(stage)
        reloaded=load_file(str(stage/files[0]))
        if set(reloaded)!=set(tensors) or any(not torch.equal(reloaded[k],v) for k,v in tensors.items()):
            raise ValueError('exported adapter does not match in-memory tensors')
        json.loads((stage/files[1]).read_text())
        checkpoint_store.publish(stage,files,run_id,pos)
        checkpoint_store.validate(stage,run_id,files)
        backup=root/('.epoch-{:02d}-previous'.format(epoch))
        if final.exists():
            marker=checkpoint_store.validate(final,run_id,files)
            if marker['position']!=pos:raise ValueError('existing adapter identity differs')
            if backup.exists():shutil.rmtree(backup)
            os.rename(final,backup)
        try:
            os.rename(stage,final)
            _sync_dir(root)
            checkpoint_store.validate(final,run_id,files)
        except Exception:
            if not final.exists() and backup.exists():os.rename(backup,final)
            raise
        if backup.exists():shutil.rmtree(backup)
        _sync_dir(root)
    finally:
        if stage.exists():shutil.rmtree(stage)


def export(model, adapter, out, epoch, position, config, names, rank):
    tensors,full=full_adapter_state(model,names,rank)
    del full
    _rank0(lambda:write_adapter(out,epoch,position,config,tensors,adapter.peft_config['default']),rank)
