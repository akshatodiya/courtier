"""Kit-backed epoch-7 GPU preflight on one V100: one synthetic prompt, no benchmark or judge calls.

Every alpha uses the benchmark actor's own fresh load and single SafeTune WiseFT
application. Alpha 0 must match the unmodified adapter and alpha 1 the base.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import benchmark_actor as actor
import safetune_repair as repair

PROMPT='Give one brief practical tip for organizing a desk.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','model-manifest','adapter','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError('receipt already exists')
    if not all(p.is_absolute() for p in (a.model_dir,a.model_manifest,a.adapter,a.out)):raise ValueError('paths must be absolute')
    actor.verify_inputs(a.model_dir,a.model_manifest,a.adapter)
    actor.check_runtime()
    import torch
    kit=repair.kit_identity();code=actor.code_identity()
    torch.manual_seed(16);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    state={}
    def measure(model):
        """Last-token logits and the layer-0 q-projection of one fixed probe."""
        q=next(m for n,m in model.named_modules() if n.endswith('model.layers.0.self_attn.q_proj'))
        if 'probe' not in state:
            state['probe']=torch.randn((1,2,q.in_features),generator=torch.Generator().manual_seed(16)).to('cuda:0',torch.float16)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            logits=model(**state['inputs'],use_cache=False).logits[:,-1,:].float().cpu()
            projection=q(state['probe']).float().cpu()
        if not (torch.isfinite(logits).all() and torch.isfinite(projection).all()):raise ValueError('nonfinite output')
        return logits,projection
    def released():
        gc.collect();torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(0)>2**30:raise ValueError('model memory was not released')
    close=lambda x,y:torch.allclose(x,y,rtol=.005,atol=.005)
    diff=lambda x,y:float((x-y).abs().max())
    # References from one load: bare base, unmodified adapter, disabled adapter.
    tok,model=actor.load_base(a.model_dir)
    text=tok.apply_chat_template([{'role':'user','content':PROMPT}],tokenize=False,add_generation_prompt=True)
    state['inputs']=tok(text,add_special_tokens=False,return_tensors='pt').to('cuda:0')
    bare,bare_p=measure(model)
    model=actor.attach_adapter(model,a.adapter)
    full,full_p=measure(model)
    with model.disable_adapter():base,base_p=measure(model)
    del model;released()
    if not (close(base,bare) and close(base_p,bare_p)):raise ValueError('disabled adapter differs from bare base')
    if close(full,base):raise ValueError('synthetic prompt cannot distinguish adapter and base')
    checks=[];first=None
    for alpha in actor.PREFLIGHT_ALPHAS:
        start=time.monotonic()
        tok,model=actor.load_repaired(a.model_dir,a.adapter,alpha)
        try:repair.apply_wiseft(model,alpha)
        except ValueError as exc:
            if 'fresh' not in str(exc):raise
        else:raise ValueError('second WiseFT application was not refused')
        logits,projection=measure(model)
        del model;released()
        expected=base_p+(1-alpha)*(full_p-base_p)
        error=float((projection-expected).norm()/expected.norm().clamp_min(1e-8))
        if error>.01:raise ValueError(f'projection interpolation mismatch for alpha {alpha}: {error}')
        if (alpha==0 and not close(logits,full)) or (alpha==1 and not close(logits,base)):
            raise ValueError(f'endpoint mismatch for alpha {alpha}')
        if alpha==.25:
            if first is None:first=logits
            else:
                repeat=diff(logits,first)
                if not close(logits,first):raise ValueError('a fresh reload changed the alpha 0.25 logits')
        checks.append({'alpha':alpha,'max_abs_diff_unmodified':diff(logits,full),'max_abs_diff_base':diff(logits,base),
                       'projection_relative_error':error,'second_application':'refused',
                       'load_and_apply_seconds':round(time.monotonic()-start,1)})
    if actor.code_identity()!=code or repair.kit_identity()!=kit:raise ValueError('code or kit changed during preflight')
    receipt={'status':'passed','method':actor.METHOD,'base_manifest_sha256':actor.MODEL_MANIFEST_SHA,
      'adapter_sha256':actor.ADAPTER_SHA,'adapter_config_sha256':actor.CONFIG_SHA,'checks':checks,
      'repeat_max_abs_diff':repeat,'full_vs_base_max_abs_diff':diff(full,base),
      'disabled_vs_bare_max_abs_diff':diff(base,bare),'prompt':PROMPT,'versions':actor.VERSIONS,
      'kit':kit,'code':code,'gpu':torch.cuda.get_device_name(0),
      'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),'slurm_job':os.environ.get('SLURM_JOB_ID')}
    with a.out.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt))


if __name__=='__main__':main()
