"""A completed Slurm job is insufficient: verify all nine published exports."""
import json
import os
from pathlib import Path
import sys
import checkpoint_store


def verify(root, slurm):
    root=Path(root)
    if slurm!='COMPLETED|0:0':raise ValueError('Slurm job did not complete successfully')
    result=json.loads((root/'TRAINING_RESULT.json').read_text())
    cfg=json.loads((root/'RUN_CONFIG.json').read_text())
    run_id=cfg['run_id']
    if result.get('run_id')!=run_id or result.get('status')!='complete' or result.get('position')!={
            'next_epoch':9,'next_step_in_epoch':0,'global_step':1053}:
        raise ValueError('training did not complete all 1053 updates')
    files=['adapter_model.safetensors','adapter_config.json']
    for epoch in range(1,10):
        marker=checkpoint_store.validate(root/'adapters'/('epoch-{:02d}'.format(epoch)),run_id,files)
        if marker['position']!={'epoch':epoch,'step_in_epoch':0,'global_step':epoch*117}:
            raise ValueError('adapter position mismatch')
    return {'status':'passed','updates':1053,'adapters':9,'run_id':run_id}


def main():
    root=Path(os.environ['AYA32_RUN_ROOT']);job=os.environ['AYA32_TRAIN_JOB'];slurm=os.environ['TRAIN_STATE']
    if not job.isdigit():raise ValueError('invalid job ID')
    report={'job_id':job,'slurm':slurm,'status':'failed_or_incomplete'}
    try:report.update(verify(root,slurm))
    except Exception as exc:report['detail']='{}: {}'.format(type(exc).__name__,exc)
    path=root.parent/('training-report-'+job+'.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=='__main__':main()
