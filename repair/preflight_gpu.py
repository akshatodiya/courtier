"""One-GPU, synthetic-input endpoint preflight; no benchmark or judge calls."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from adapter_scale import set_repair_strength, validate_config

MODEL_MANIFEST_SHA = '2e4ba00ef6e7e101f353d50000c36bfd18ec1ea308ed9a0d8526e63b6d1f668a'
ADAPTER_SHA = '37416288d21a286d7b3a77910884519ac1c2571789e8c6cff88b96b848a767a1'
CONFIG_SHA = '8770836e6a45463a795a1ab6cfb395d0472c3a1c630ddbdf8371cfabc5e93527'
VERSIONS = {'torch':'2.7.1+cu118','transformers':'5.19.0','peft':'0.21.2','bitsandbytes':'0.50.2'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify_inputs(base, manifest_path, adapter):
    if sha(manifest_path) != MODEL_MANIFEST_SHA: raise ValueError('base manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('repository')!='CohereLabs/aya-expanse-8b'
        or manifest.get('revision')!='5062468bf9bc0c6035fd64e06274333ec127d980'):
        raise ValueError('unexpected base identity')
    expected={'config.json','generation_config.json','model-00001-of-00004.safetensors',
      'model-00002-of-00004.safetensors','model-00003-of-00004.safetensors',
      'model-00004-of-00004.safetensors','model.safetensors.index.json',
      'special_tokens_map.json','tokenizer.json','tokenizer_config.json'}
    if set(manifest.get('files',{}))!=expected: raise ValueError('base manifest coverage mismatch')
    for name,meta in manifest['files'].items():
        path=base/name
        if path.stat().st_size!=meta['bytes'] or sha(path)!=meta['sha256']: raise ValueError('base mismatch: '+name)
    if sha(adapter/'adapter_model.safetensors')!=ADAPTER_SHA or sha(adapter/'adapter_config.json')!=CONFIG_SHA:
        raise ValueError('adapter mismatch')
    validate_config(json.loads((adapter/'adapter_config.json').read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--model-manifest',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists(): raise FileExistsError('receipt already exists')
    if not all(p.is_absolute() for p in (a.model_dir,a.model_manifest,a.adapter,a.out)):
        raise ValueError('paths must be absolute')
    verify_inputs(a.model_dir,a.model_manifest,a.adapter)
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if {k:importlib.metadata.version(k) for k in VERSIONS} != VERSIONS:
        raise ValueError('runtime versions differ')
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(7,0)
        or 'V100' not in torch.cuda.get_device_name(0)):
        raise ValueError('requires exactly one V100')
    torch.manual_seed(16);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,quantization_config=quant,dtype=torch.float16,
                                               device_map={'':0},attn_implementation='sdpa',local_files_only=True)
    if model.config.model_type!='cohere' or model.config.logit_scale!=.125: raise ValueError('unexpected model')
    text=tok.apply_chat_template([{'role':'user','content':'Give one brief practical tip for organizing a desk.'}],
                                 tokenize=False,add_generation_prompt=True)
    inputs=tok(text,add_special_tokens=False,return_tensors='pt').to('cuda:0')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        bare_logits=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
    if not torch.isfinite(bare_logits).all(): raise ValueError('nonfinite bare-base logits')
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False,local_files_only=True).eval()
    factors={n.replace('.lora_A.default.weight','.lora_A.weight').replace('.lora_B.default.weight','.lora_B.weight'):p
             for n,p in model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
    with safe_open(a.adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        if len(factors)!=448 or set(factors)!=set(f.keys()): raise ValueError('loaded LoRA key coverage mismatch')
        for name,param in factors.items():
            original=f.get_tensor(name)
            if (param.dtype!=original.dtype or not torch.equal(param.detach().cpu(),original)
                or not torch.isfinite(original).all()
                or (name.endswith('.lora_B.weight') and not torch.count_nonzero(original))):
                raise ValueError('loaded LoRA tensor mismatch: '+name)
    def logits():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            value=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
        if not torch.isfinite(value).all(): raise ValueError('nonfinite logits')
        return value
    layers=[(name,layer) for name,layer in model.named_modules() if isinstance(layer,LoraLayer)]
    qs=[layer for name,layer in layers if name.endswith('.q_proj')]
    if len(layers)!=224 or len(qs)!=32: raise ValueError('unexpected LoRA coverage')
    projection=qs[0]
    probe=torch.randn((1,2,projection.in_features),device='cuda:0',dtype=torch.float16)
    def project():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            value=projection(probe).float().cpu()
        if not torch.isfinite(value).all(): raise ValueError('nonfinite projection')
        return value
    # Capture both references before modifying any adapter scales.
    full_logits=logits(); full_projection=project()
    with model.disable_adapter(): base_logits=logits()
    with model.disable_adapter(): base_projection=project()
    if not torch.allclose(base_logits,bare_logits,rtol=.005,atol=.005):
        raise ValueError('disabled adapter differs from bare base')
    if torch.allclose(full_logits,base_logits,rtol=.005,atol=.005):
        raise ValueError('synthetic prompt cannot distinguish adapter and base')
    checks={}
    first_quarter=None
    for alpha in (0,.25,.5,.75,1,.25):
        if set_repair_strength(model,alpha,expected_layers=224)!=224: raise ValueError('layer count')
        for _,layer in layers:
            expected=(1-alpha)*layer.lora_alpha['default']/layer.r['default']
            if not math.isclose(layer.scaling['default'],expected,rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('incorrect LoRA layer scale')
        result=logits()
        p=project(); expected_projection=base_projection+(1-alpha)*(full_projection-base_projection)
        relative_error=float((p-expected_projection).norm() / expected_projection.norm().clamp_min(1e-8))
        if not math.isfinite(relative_error) or relative_error>.01:
            raise ValueError(f'projection interpolation mismatch for alpha {alpha}: {relative_error}')
        if alpha==.25:
            if first_quarter is None: first_quarter=result
            elif not torch.allclose(result,first_quarter,rtol=.005,atol=.005):
                raise ValueError('repeated alpha changes logits')
        if alpha in (0,1):
            ref=full_logits if alpha==0 else base_logits
            diff=float((result-ref).abs().max())
            if not math.isfinite(diff) or not torch.allclose(result,ref,rtol=.005,atol=.005):
                raise ValueError(f'endpoint mismatch for alpha {alpha}: {diff}')
            checks[str(alpha)]={'max_abs_diff':diff,'projection_relative_error':relative_error}
        else:
            checks[str(alpha)]={'finite':True,'projection_relative_error':relative_error}
    receipt={'status':'passed','base_manifest_sha256':MODEL_MANIFEST_SHA,'adapter_sha256':ADAPTER_SHA,
      'adapter_config_sha256':CONFIG_SHA,'alpha_toward_base':[0,.25,.5,.75,1],
      'endpoint_checks':checks,'prompt':'synthetic desk-organizing prompt','versions':VERSIONS,
      'full_vs_base_max_abs_diff':float((full_logits-base_logits).abs().max()),'repeated_alpha':.25,
      'gpu':torch.cuda.get_device_name(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),
      'preflight_sha256':sha(__file__),'scale_helper_sha256':sha(Path(__file__).with_name('adapter_scale.py'))}
    with a.out.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt))


if __name__=='__main__': main()
