"""Fingerprint runtime bitsandbytes 4-bit quantization metadata per module."""
import hashlib
import json
import torch


def quant_fingerprints(model,minimum=280):
    import bitsandbytes as bnb
    records={}
    for name,module in model.named_modules():
        if not isinstance(module,bnb.nn.Linear4bit):continue
        state=getattr(module.weight,'quant_state',None) or getattr(module,'quant_state',None)
        if state is None:raise ValueError('Missing 4-bit quantization state: '+name)
        fields=state.as_dict(packed=True)
        if not fields:raise ValueError('Empty 4-bit quantization state: '+name)
        digest=hashlib.sha256()
        for key,value in sorted(fields.items()):
            if not isinstance(value,torch.Tensor):raise ValueError('Unexpected non-tensor quant field: '+name+'.'+key)
            tensor=value.detach().cpu().contiguous()
            digest.update(json.dumps([key,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        records[name]=digest.hexdigest()
    if len(records)<minimum:raise ValueError('Too few 4-bit quantization states')
    return records
