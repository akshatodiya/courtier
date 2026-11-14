"""Constrain this Aya training pilot to the V100-probed SDPA path."""
from contextlib import contextmanager
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers.integrations.sdpa_attention as sdpa


@contextmanager
def use_v100_efficient_aya():
    original=sdpa.use_gqa_in_sdpa
    stats={'calls':0}
    if 'use_gqa_in_sdpa' not in sdpa.sdpa_attention_forward.__code__.co_names:
        raise RuntimeError('Unexpected Transformers SDPA implementation')

    def repeat_only(mask,key,value):
        if mask is not None:
            raise RuntimeError('Efficient Aya pilot requires unmasked, unpadded causal training')
        if key.ndim!=4 or key.shape[0]!=1 or key.shape[1]!=8 or key.shape[-1]!=128 or key.shape!=value.shape:
            raise RuntimeError('Unexpected Aya key/value shape')
        stats['calls']+=1
        return False

    sdpa.use_gqa_in_sdpa=repeat_only
    try:
        # This context covers forward AND backward: checkpointing re-runs
        # attention during backward, and fallback to the math kernel must fail.
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield stats
    finally:
        sdpa.use_gqa_in_sdpa=original
