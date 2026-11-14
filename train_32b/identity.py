"""Bind future checkpoints to the exact training recipe and source files."""
import hashlib
import json
from pathlib import Path
from train_plan import CONFIG

SOURCES=('train.py','training_loop.py','batch_plan.py','train_plan.py','identity.py',
         'checkpoint_backend.py','local_state.py','rng_state.py','checkpoint_store.py','quant_audit.py',
         'gate.py','probe_expected.json','train.sbatch','monitor.py','monitor.sbatch','pilot.py','scale_retry.py','attention_override.py','chunk_loss.py','aya_chat.py')


def build_config(stage):
    stage=Path(stage)
    sources={}
    for name in SOURCES:
        p=stage/name
        if not p.is_file() or p.is_symlink(): raise ValueError('missing or unsafe source: '+name)
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(CONFIG,
        tied_embedding_and_final_norm_dtype='float16',nonquantized_load_dtype='float32',
        attention_backend='efficient_sdpa_v100_explicit_kv_repeat_unmasked',
        loss_projection='selected_logits_checkpointed_ce_chunk128',
        loss_scaler='ShardedGradScaler(default initial scale 65536, backoff 0.5)',
        overflow_policy='retry same update and RNG; max 20 attempts',
        optimizer='AdamW',adam_betas=[0.9,0.999],adam_epsilon=1e-8,
        gradient_checkpointing='nonreentrant',fp16_autocast=True,
        source_sha256=sources)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
