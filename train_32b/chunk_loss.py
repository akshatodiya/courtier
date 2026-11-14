"""Exact summed cross entropy with checkpointed FP32 vocabulary chunks."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction='sum')


def selected_cross_entropy(logits, targets, chunk_size=128):
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError('Expected [positions, vocabulary] logits and [positions] labels')
    if logits.shape[0] == 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Expected nonempty positions and a positive integer chunk size')
    chunks=[]
    for start in range(0,logits.shape[0],chunk_size):
        x=logits[start:start+chunk_size]
        y=targets[start:start+chunk_size]
        if torch.is_grad_enabled() and x.requires_grad:
            chunks.append(checkpoint(_chunk,x,y,use_reentrant=False))
        else:
            chunks.append(_chunk(x,y))
    return torch.stack(chunks).sum()
