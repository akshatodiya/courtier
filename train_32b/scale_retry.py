"""Retry the same distributed update when FP16 loss scaling overflows."""
import json
import math
import torch
import torch.distributed as dist


def distributed_any(flag):
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def local_invalid_gradients(optimizer, device):
    bad = torch.zeros((), dtype=torch.int32, device=device)
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None:
                bad |= (~torch.isfinite(parameter.grad).all()).to(dtype=torch.int32)
    return bad


def log_bad_gradients(model, *, rank, step, attempt, scale, log):
    """Report a small, non-sensitive sample of local invalid gradient tensors."""
    bad = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all().item():
            bad.append({'parameter': name, 'nan': int(torch.isnan(grad).sum().item()),
                        'inf': int(torch.isinf(grad).sum().item()), 'numel': grad.numel()})
            if len(bad) == 4:
                break
    if bad:
        log('BAD_GRADIENT ' + json.dumps({'rank': rank, 'step': step, 'attempt': attempt,
                                         'scale': scale, 'sample': bad}), flush=True)


def scaled_step(model, optimizer, scaler, loss_fn, *, device, rank, step,
                max_attempts=20, sync_any=distributed_any, log=print, after_backward=None):
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    if not scaler.is_enabled():
        raise ValueError('Gradient scaler must be enabled for safe overflow handling')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
    for attempt in range(max_attempts):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        invalid_loss = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
        if sync_any(invalid_loss):
            raise FloatingPointError('Nonfinite forward loss on at least one rank')
        old_scale = scaler.get_scale()
        if not math.isfinite(old_scale) or old_scale <= 0:
            raise FloatingPointError('Invalid gradient scale')
        scaler.scale(loss).backward()
        if after_backward is not None:
            after_backward(attempt)
        has_local_grad = any(p.grad is not None and p.grad.numel() > 0
                             for group in optimizer.param_groups for p in group['params'])
        if not sync_any(torch.tensor(int(has_local_grad), dtype=torch.int32, device=device)):
            raise FloatingPointError('No optimizer gradients on any rank')
        pre_unscale_bad = local_invalid_gradients(optimizer, device)
        has_pre_unscale_bad = sync_any(pre_unscale_bad.clone())
        if pre_unscale_bad.item():
            log_bad_gradients(model, rank=rank, step=step, attempt=attempt + 1,
                              scale=old_scale, log=log)
        scaler.unscale_(optimizer)
        post_unscale_bad = local_invalid_gradients(optimizer, device)
        has_post_unscale_bad = sync_any(post_unscale_bad.clone())
        if has_post_unscale_bad and not has_pre_unscale_bad:
            # No invalid scaled gradient was present when unscale_ checked, so
            # scaler.step could update parameters with invalid gradients.
            if post_unscale_bad.item():
                log_bad_gradients(model, rank=rank, step=step, attempt=attempt + 1,
                                  scale=old_scale, log=log)
            raise FloatingPointError('Nonfinite gradients appeared during unscale; no optimizer step')
        if has_pre_unscale_bad:
            # ShardedGradScaler syncs its found-inf state in unscale_. It skips
            # the optimizer step, then lowers its scale during update().
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if rank == 0:
                log('SCALE_RETRY ' + json.dumps({'step': step, 'attempt': attempt + 1,
                    'old_scale': old_scale, 'new_scale': new_scale, 'reason': 'nonfinite_gradient'}), flush=True)
            if not math.isfinite(new_scale) or not 0 < new_scale < old_scale:
                raise FloatingPointError('Scaler did not back off after an invalid gradient')
            continue
        norm = model.clip_grad_norm_(1.0)
        invalid_norm = (~torch.isfinite(norm.detach())).to(dtype=torch.int32)
        if sync_any(invalid_norm):
            raise FloatingPointError('Nonfinite gradient norm despite finite local gradients')
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if not math.isfinite(new_scale) or new_scale < old_scale:
            raise FloatingPointError('Unexpected skipped optimizer step')
        return {'loss': float(loss.detach()), 'grad_norm': float(norm.detach()),
                'attempts': attempt + 1, 'scale_before': old_scale, 'scale_after': new_scale}
    raise FloatingPointError('Gradient overflow persisted for {} attempts'.format(max_attempts))
