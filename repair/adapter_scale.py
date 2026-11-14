"""Set a WiSE-style repair strength on one already-loaded ordinary PEFT LoRA.

The factor interpolates the effective LoRA delta relative to the exact same
base used by inference. It does not call SafeTune on the real model; direct
SafeTune interpolation of a PEFT wrapper against a bare base is unsafe.
"""
import math


def validate_config(config):
    def get(key, default=None):
        return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
    checks = {
        'peft_type': lambda x: str(x).split('.')[-1] == 'LORA',
        'bias': lambda x: x == 'none',
        'modules_to_save': lambda x: not x,
        'lora_bias': lambda x: not x,
        'use_dora': lambda x: not x,
        'use_rslora': lambda x: not x,
        'use_qalora': lambda x: not x,
        'target_parameters': lambda x: not x,
        'alora_invocation_tokens': lambda x: not x,
        'layer_replication': lambda x: not x,
        'rank_pattern': lambda x: not x,
        'alpha_pattern': lambda x: not x,
        'loftq_config': lambda x: not x,
        'corda_config': lambda x: not x,
        'use_bdlora': lambda x: not x,
        'velora_config': lambda x: not x,
        'arrow_config': lambda x: not x,
        'trainable_token_indices': lambda x: not x,
        'init_lora_weights': lambda x: x is True,
        'r': lambda x: isinstance(x, int) and x > 0,
        'lora_alpha': lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0,
    }
    for name, valid in checks.items():
        if not valid(get(name)):
            raise ValueError(f'unsupported adapter configuration: {name}')


def set_repair_strength(model, alpha, adapter='default', expected_layers=None):
    """Set alpha toward the common base without changing the adapter weights.

    Alpha 0 keeps the adapter; alpha 1 disables its additive contribution.
    The model must already be in eval mode with exactly one active adapter.
    Repeated calls replace rather than multiply the previous scale.
    """
    from peft.tuners.lora.layer import LoraLayer
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and between 0 and 1')
    if model.training:
        raise ValueError('model must be in evaluation mode')
    configs = getattr(model, 'peft_config', {})
    if set(configs) != {adapter} or list(model.active_adapters) != [adapter]:
        raise ValueError('exactly the requested adapter must be loaded and active')
    validate_config(configs[adapter])
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    if not layers or (expected_layers is not None and len(layers) != expected_layers):
        raise ValueError('unexpected LoRA layer count')
    for layer in layers:
        if (layer.training or layer.disable_adapters or layer.merged
                or list(layer.active_adapters) != [adapter]
                or adapter not in layer.lora_A or adapter not in layer.lora_B
                or adapter not in layer.scaling or layer.use_dora.get(adapter, False)
                or layer.use_rslora.get(adapter, False)):
            raise ValueError('unsupported or inactive LoRA layer')
    for layer in layers:
        layer.set_scale(adapter, 1.0 - alpha)
    return len(layers)
