"""Per-rank RNG capture for a future resumable training implementation.

The code preserves Python, NumPy, PyTorch CPU, the current CUDA device and
GradScaler state. Checkpoint files must only be loaded from the trusted run.
"""
import random
import numpy as np
import torch


def capture(scaler=None):
    np_state=np.random.get_state()
    return {'python':random.getstate(),
            'numpy':{'kind':np_state[0], 'keys':np_state[1].tolist(), 'pos':int(np_state[2]),
                     'has_gauss':int(np_state[3]), 'cached_gaussian':float(np_state[4])},
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'scaler':scaler.state_dict() if scaler is not None else None}


def restore(state, scaler=None):
    if state['torch_cuda'] is not None and not torch.cuda.is_available():
        raise RuntimeError('CUDA RNG state present but CUDA is unavailable')
    if state['scaler'] is not None and scaler is None:
        raise ValueError('Scaler state present but no scaler supplied')
    random.setstate(state['python'])
    ns=state['numpy']
    np.random.set_state((ns['kind'],np.asarray(ns['keys'],dtype=np.uint32),ns['pos'],ns['has_gauss'],ns['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda'])
    if state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])
