"""Use SafeTune WiseFTTrainer on the B factors of an ordinary PEFT LoRA.

The matching reference has the same A factors and zero B factors, so SafeTune
interpolating only B gives the effective-weight WiseFT interpolation. It mutates
only the loaded adapter; the original checkpoint files are never changed.
"""
import hashlib
import math
from pathlib import Path
import sys
import torch
from peft.tuners.lora.layer import LoraLayer
import safetune
from safetune.runner.recover import WiseFTTrainer
from adapter_scale import validate_config

# The safetune 0.1.9 wheel: every file matches its RECORD and checkout a869414.
KIT={'safetune_tree_sha256':'8cca63052c8ccf4b7590a147f714c5ec9a11956905bbc87dde898c003cb97598',
     'safetune_files':332,'safetune_version':'0.1.9'}
KIT_MODULES=('safetune.runner.recover._whole_model','safetune.interventions.recover.wise_ft',
             'safetune.interventions.recover._invariant')


def kit_identity(root=None):
    """Hash every file of the imported SafeTune package, excluding bytecode caches."""
    root=Path(root or Path(safetune.__file__).parent).resolve()
    paths=sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('SafeTune kit contains a symbolic link')
    files=[p for p in paths if p.is_file() and '__pycache__' not in p.relative_to(root).parts]
    h=hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode()+b'\0'+hashlib.sha256(p.read_bytes()).hexdigest().encode()+b'\n')
    found={'safetune_tree_sha256':h.hexdigest(),'safetune_files':len(files),'safetune_version':safetune.__version__}
    if found!=KIT: raise ValueError('SafeTune kit identity mismatch')
    if any(not Path(sys.modules[m].__file__).resolve().is_relative_to(root) for m in KIT_MODULES):
        raise ValueError('SafeTune modules loaded from another path')
    return dict(KIT)


def apply_wiseft(model, alpha, expected_layers=None):
    if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('alpha must be finite and between 0 and 1')
    kit_identity()
    if model.training or getattr(model,'_courtier_wise_applied',False):
        raise ValueError('requires a fresh evaluation-mode model')
    if set(getattr(model,'peft_config',{}))!={'default'} or list(model.active_adapters)!=['default']:
        raise ValueError('exactly one active default adapter required')
    validate_config(model.peft_config['default'])
    layers=[m for m in model.modules() if isinstance(m,LoraLayer)]
    if not layers or (expected_layers is not None and len(layers)!=expected_layers):
        raise ValueError('unexpected LoRA layer count')
    factors={}
    for i,layer in enumerate(layers):
        if (layer.training or layer.disable_adapters or layer.merged or list(layer.active_adapters)!=['default']
            or 'default' not in layer.lora_A or 'default' not in layer.lora_B
            or layer.use_dora.get('default',False) or layer.use_rslora.get('default',False)):
            raise ValueError('unsupported LoRA layer')
        expected=layer.lora_alpha['default']/layer.r['default']
        if not math.isclose(layer.scaling['default'],expected,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('adapter scaling is not original')
        p=layer.lora_B['default'].weight
        if p.dtype!=torch.float32 or not torch.isfinite(p).all().item():
            raise ValueError('invalid LoRA B tensor')
        factors[f'factor_{i:04d}']=p
    if len({id(p) for p in factors.values()})!=len(factors):
        raise ValueError('shared LoRA B tensor')
    proxy=torch.nn.ParameterDict(factors)
    reference=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.zeros(p.shape,dtype=p.dtype,device='cpu'),requires_grad=False)
                                      for k,p in factors.items()})
    before={k:p.detach().cpu().clone() for k,p in factors.items()}
    # A failed or partially applied operation must never be silently retried.
    model._courtier_wise_applied=True
    WiseFTTrainer(model=proxy,aligned_model=reference,alpha=float(alpha)).apply()
    for k,p in factors.items():
        target=(1-alpha)*before[k]
        if not torch.allclose(p.detach().cpu(),target,rtol=1e-6,atol=1e-7):
            raise ValueError('SafeTune WiseFT did not update factor '+k)
    return len(layers)
