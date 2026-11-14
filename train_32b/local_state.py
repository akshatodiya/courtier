"""Checkpoint same-topology FSDP trainable shards and optimizer state.

Frozen base weights and quantization metadata must be recreated from a verified
model and independently checked by the caller. This does not support reshaping
or changing world size. The caller validates rank, run identity, RNG and files.
"""
import hashlib
import json
import torch


def trainable(model):
    items=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not items or any('lora_' not in n for n,_ in items):
        raise ValueError('Only LoRA trainable shards are supported')
    if len({n for n,_ in items})!=len(items):raise ValueError('Duplicate parameter name')
    return items


def _schema(items):
    return [{'name':name,'shape':list(p.shape),'dtype':str(p.dtype)} for name,p in items]


def _cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value


def layout_identity(model,optimizer):
    items=trainable(model)
    index={id(p):i for i,(_,p) in enumerate(items)}
    groups=[]
    for group in optimizer.param_groups:
        try:groups.append([index[id(p)] for p in group['params']])
        except KeyError as exc:raise ValueError('Optimizer owns a non-LoRA parameter') from exc
    if sorted(i for g in groups for i in g)!=list(range(len(items))):
        raise ValueError('Optimizer must contain every trainable parameter exactly once')
    return {'schema':_schema(items),'optimizer_groups':groups}


def capture(model,optimizer):
    layout=layout_identity(model,optimizer)
    return {'layout':layout,'parameters':[p.detach().cpu().clone() for _,p in trainable(model)],
            'optimizer':_cpu(optimizer.state_dict())}


def restore(model,optimizer,state):
    if not isinstance(state,dict) or set(state)!={'layout','parameters','optimizer'}:
        raise ValueError('Invalid checkpoint structure')
    items=trainable(model)
    if state['layout']!=layout_identity(model,optimizer):raise ValueError('FSDP shard layout changed')
    saved=state['parameters']
    if not isinstance(saved,list) or len(saved)!=len(items):raise ValueError('Missing parameter shard')
    for (name,p),value in zip(items,saved):
        if not isinstance(value,torch.Tensor) or value.shape!=p.shape or value.dtype!=p.dtype:
            raise ValueError('Invalid parameter shard: '+name)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite parameter shard: '+name)
    # Verify optimizer identity before modifying model parameters.
    current=optimizer.state_dict()
    old_groups=state['optimizer'].get('param_groups',[])
    if len(old_groups)!=len(current['param_groups']):raise ValueError('Optimizer group count changed')
    for old,new in zip(old_groups,current['param_groups']):
        if old!=new:raise ValueError('Optimizer group settings or parameter mapping changed')
    with torch.no_grad():
        for (_,p),value in zip(items,saved):p.copy_(value.to(device=p.device))
    optimizer.load_state_dict(state['optimizer'])


def frozen_digest(model):
    """Fingerprint managed local FSDP storage, never stale unsharded views."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    modules=[(name,module) for name,module in model.named_modules() if isinstance(module,FSDP)]
    if modules:
        pairs=[]
        for name,module in modules:
            handle=getattr(module,'_handle',None)
            if handle is None:continue
            parameter=handle.flat_param
            if parameter.requires_grad:continue
            shard=getattr(parameter,'_local_shard',None)
            if not isinstance(shard,torch.Tensor):
                raise RuntimeError('FSDP local shard unavailable: '+name)
            pairs.append((name,shard))
    else:
        pairs=[(name,p) for name,p in model.named_parameters() if not p.requires_grad]
    h=hashlib.sha256(); count=0
    for name,p in pairs:
        h.update(json.dumps([name,str(p.dtype),list(p.shape)],separators=(',',':')).encode())
        # Some FSDP ranks own empty shards. No copy is necessary for them.
        if p.numel()==0:
            count+=1
            continue
        try:
            tensor=p.detach().cpu().contiguous()
        except RuntimeError as exc:
            raise RuntimeError('Frozen shard copy failed: name={} type={} shape={} dtype={} device={}'.format(
                name,type(p).__name__,tuple(p.shape),p.dtype,p.device)) from exc
        raw=memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        for pos in range(0,len(raw),1048576):h.update(raw[pos:pos+1048576])
        count+=1
    if count==0:raise ValueError('No frozen base shards')
    return {'sha256':h.hexdigest(),'parameter_count':count}
