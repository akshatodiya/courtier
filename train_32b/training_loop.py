"""Deterministic epoch driver for the 14-rank Aya 32B trainer.

Checkpoint and export functions are supplied by a separately verified backend.
This driver does not itself prove the model or state can be restored.
"""
from batch_plan import epoch_plan
from train_plan import CONFIG, identity, validate_checkpoint_metadata


def run_training(train_step, save_checkpoint, export_epoch, should_stop, *,
                 position=None, checkpoint_every=117, config=CONFIG):
    if type(checkpoint_every) is not int or checkpoint_every<1:
        raise ValueError('checkpoint_every must be a positive integer')
    if config['world_size']!=14 or config['effective_batch']!=8:
        raise ValueError('schedule requires 14 ranks and batch 8')
    p=dict(position or {'next_epoch':0,'next_step_in_epoch':0,'global_step':0})
    if set(p)!={'next_epoch','next_step_in_epoch','global_step'}:
        raise ValueError('invalid resume position')
    validate_checkpoint_metadata({'schema':'aya32-resume-v1','run_identity':identity(config),**p},config)
    updates=(config['rows']+config['effective_batch']-1)//config['effective_batch']
    start_epoch,start_step=p['next_epoch'],p['next_step_in_epoch']
    for epoch in range(start_epoch,config['epochs']):
        for step, assignments in enumerate(epoch_plan(config['rows'],config['seed'],epoch)):
            if epoch==start_epoch and step<start_step: continue
            # train_step returns only after a real, finite optimizer update.
            train_step(epoch,step,assignments)
            epoch_done=step+1==updates
            p={'next_epoch':epoch+1 if epoch_done else epoch,
               'next_step_in_epoch':0 if epoch_done else step+1,'global_step':p['global_step']+1}
            # Export is idempotent: a crash between export and checkpoint can replay it.
            if epoch_done: export_epoch(epoch+1,dict(p))
            stopping=bool(should_stop())
            if epoch_done or p['global_step']%checkpoint_every==0 or stopping:
                save_checkpoint(dict(p))
            if stopping: return {'status':'stopped_at_boundary','position':p}
    return {'status':'complete','position':p}
