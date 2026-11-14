"""SafeTune PlainSFT, with the pilot-proven Aya QLoRA loss and native HF checkpoints.

Project adaptations: quantized CUDA loading, exact turn masks, selected-position CE,
token-weighted distributed accumulation, bounded runtime and adapter-only saving.
No merge into NF4 weights. No automatic evaluation, selection or resubmission.
"""
import argparse
import dataclasses
from datetime import timedelta
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import torch.distributed as dist
from datasets import Dataset
import safetune
from safetune.harden import PlainSFTTrainer
from safetune.provenance import input_entry, write_provenance
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, set_seed

import aya_chat
from job_records import write_report
from train_control import digest, identity, validate_checkpoint

CONTEXT = {}


def selected_sum(model, inputs):
    labels = inputs['labels']
    if labels.shape[0] != 1:
        raise ValueError('selected logits require microbatch one')
    positions = (labels[0, 1:] != aya_chat.IGNORE).nonzero().flatten()
    if not len(positions):
        raise ValueError('row has no supervised prediction targets')
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'),
                    logits_to_keep=positions, use_cache=False)
    return torch.nn.functional.cross_entropy(outputs.logits[0].float(), labels[0, 1:][positions], reduction='sum')


class ProgressAndBudget(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        set_seed(args.seed + args.process_index)

    def on_step_end(self, args, state, control, **kwargs):
        native = CONTEXT['native']
        if native.accelerator.optimizer_step_was_skipped:
            raise RuntimeError('nonfinite gradients caused an optimizer update to be skipped')
        stop = torch.tensor(int(CONTEXT['signal_stop'] or time.time() >= CONTEXT['deadline']), device=args.device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            CONTEXT['budget_stopped'] = True
            control.should_training_stop = True
            control.should_save = True
        progress = {'status': 'budget_stopping' if CONTEXT['budget_stopped'] else 'running',
                    'global_step': state.global_step, 'epoch': state.epoch,
                    'seconds': time.time() - CONTEXT['started'],
                    'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                    'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9}
        write_report(CONTEXT['out'] / f'progress-rank-{args.process_index}.json', progress)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        dist.barrier()
        if args.process_index == 0:
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            marker = {'run_identity': CONTEXT['identity'], 'world_size': args.world_size,
                      'global_step': state.global_step, 'epoch': state.epoch}
            write_report(checkpoint / 'CHECKPOINT_COMPLETE.json', marker)
            validate_checkpoint(checkpoint, CONTEXT['identity'], args.world_size)
            write_report(CONTEXT['out'] / 'LATEST_CHECKPOINT.json', {**marker, 'path': str(checkpoint)})
        dist.barrier()
        return control


class SelectedTrainer(Trainer):
    loss_is_scaled_for_ga = True

    def __init__(self, *args, **kwargs):
        kwargs['callbacks'] = [ProgressAndBudget()]
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # native prefetch counts tokens across the accumulated global batch
        self._loss_shifts_labels = True
        CONTEXT['native'] = self
        if self.args.world_size != CONTEXT['config']['world_size']:
            raise ValueError('unexpected world size')
        if not self.args.fp16 or self.args.bf16 or self.args.train_batch_size != 1:
            raise ValueError('expected fp16 CUDA microbatch one')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is None or int(num_items_in_batch) <= 0:
            raise ValueError('native global supervised-token count is required')
        loss = selected_sum(model, inputs) * self.accelerator.num_processes / num_items_in_batch
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite selected-position loss')
        return (loss, {'loss': loss}) if return_outputs else loss

    def train(self, *args, **kwargs):
        if CONTEXT.get('resume'):
            kwargs['resume_from_checkpoint'] = str(CONTEXT['resume'])
        return super().train(*args, **kwargs)


class CourtierSFT(PlainSFTTrainer):
    def _training_args(self, out_dir):
        cfg = CONTEXT['config']
        return dataclasses.replace(super()._training_args(out_dir),
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'], weight_decay=cfg['weight_decay'],
            max_grad_norm=cfg['max_grad_norm'], lr_scheduler_type=cfg['scheduler'], warmup_steps=cfg['warmup_steps'],
            max_steps=-1, data_seed=cfg['seed'], save_strategy='steps', save_steps=cfg['save_steps'],
            save_total_limit=None, save_only_model=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}, ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False, average_tokens_across_devices=True, logging_nan_inf_filter=False,
            dataloader_drop_last=False, dataloader_pin_memory=False, torch_compile=False, push_to_hub=False)

    def _save_merged(self, model, out_dir):
        # The stock SafeTune saver merges LoRA. Preserve an adapter against the original NF4 base instead.
        adapter = Path(out_dir) / 'adapter'
        if dist.get_rank() == 0:
            adapter.mkdir(exist_ok=False)
            model.save_pretrained(adapter, save_embedding_layers=False)
            self.tok.save_pretrained(adapter)
            write_provenance(adapter, 'harden.PlainSFT', inputs=self.dataset_inputs,
                base_model=CONTEXT['config']['model'], params={**CONTEXT['config'],
                    'project_adaptations': 'selected-position CUDA loss; native HF distributed checkpoints; adapter-only save',
                    'run_identity': CONTEXT['identity'], 'budget_stopped': CONTEXT['budget_stopped']})
        dist.barrier()
        return str(adapter)


def adapter_digest(model):
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((args.root / 'config.json').read_text())
    run_identity = identity(args.root)
    for package, expected in cfg['versions'].items():
        if version(package) != expected:
            raise ValueError(f'Unexpected {package} version: {version(package)} != {expected}')
    if digest(args.root / 'data/train.jsonl') != cfg['data_sha256']:
        raise ValueError('training data changed')
    # Fail before submitting if the actual installed SafeTune runner no longer has the reviewed interface.
    kit_module = importlib.import_module(PlainSFTTrainer.__module__)
    if not hasattr(kit_module, 'Trainer') or not hasattr(PlainSFTTrainer, '_lora_base'):
        raise ValueError('unexpected SafeTune PlainSFT implementation')
    if args.preflight:
        print(json.dumps({'status': 'passed', 'run_identity': run_identity,
                          'versions': {p: version(p) for p in cfg['versions']},
                          'kit_source': str(Path(kit_module.__file__)), 'kit_source_sha256': digest(kit_module.__file__)}))
        return
    if args.out is None or not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise ValueError('full training requires --out and two CUDA workers on PARAM')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    started = time.time()
    report = {'status': 'starting', 'rank': rank, 'run_identity': run_identity, 'config': cfg}
    try:
        if args.resume:
            validate_checkpoint(args.resume, run_identity, cfg['world_size'])
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        CONTEXT.update(config=cfg, identity=run_identity, out=args.out, started=started,
                       deadline=float(os.environ['AYA_TRAIN_DEADLINE']), signal_stop=False, budget_stopped=False,
                       resume=args.resume)
        def stop(signum, frame):
            CONTEXT['signal_stop'] = True
        signal.signal(signal.SIGUSR1, stop)
        expected = json.loads((args.root / 'model_files.json').read_text())
        for name, record in expected['files'].items():
            if digest(args.model_dir / name) != record['sha256']:
                raise ValueError('model file changed: ' + name)
        set_seed(cfg['seed'])
        tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        rows = [json.loads(line) for line in (args.root / 'data/train.jsonl').read_text().splitlines()]
        if len(rows) != cfg['rows']:
            raise ValueError('wrong row count')
        report['template_check'] = aya_chat.check(tok, rows)
        encoded = [aya_chat.encode(tok, r['messages'], r['supervise']) for r in rows]
        if any(len(ids) > cfg['max_length'] or not any(y != -100 for y in labels[1:]) for ids, labels in encoded):
            raise ValueError('truncated or unsupervised row')
        dataset = Dataset.from_list([{'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}
                                     for ids, labels in encoded])
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(args.model_dir, quantization_config=quant, dtype=torch.float16,
            device_map={'': local_rank}, attn_implementation='sdpa', local_files_only=True)
        if type(base).__name__ != 'CohereForCausalLM' or base.config.logit_scale != 0.125:
            raise ValueError('unexpected Cohere architecture or logit scale')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
        safetune.configure(device=f'cuda:{local_rank}', dtype='fp16', seed=cfg['seed'], max_len=cfg['max_length'],
            lora_r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
            lora_target_modules=tuple(cfg['target_modules']))
        runner = CourtierSFT(base, tok, model_id=str(args.model_dir), epochs=cfg['epochs'],
            batch_size=cfg['microbatch'], lr=cfg['learning_rate'], fp16=True, bf16=False, logging_steps=1)
        runner.dataset_inputs = [input_entry('model', str(args.model_dir)),
                                input_entry('dataset', str(args.root / 'data/train.jsonl'))]
        # Use SafeTune's own LoRA wrapper exactly once, before the native trainer is constructed.
        model = runner._lora_base()
        params = [p for p in model.parameters() if p.requires_grad]
        if not params or any(p.dtype != torch.float32 for p in params):
            raise ValueError('expected fp32 LoRA parameters')
        runner._model = model
        runner._lora_base = lambda: model  # already wrapped above; never wrap a PEFT model twice
        initial = adapter_digest(model)
        initial_ranks = [None] * 2
        dist.all_gather_object(initial_ranks, initial)
        if len(set(initial_ranks)) != 1:
            raise ValueError('initial rank adapters differ')
        # Check the project loss against the real model once before training.
        index = min((i for i,r in enumerate(rows) if r['family']=='social_bc'), key=lambda i:len(encoded[i][0]))
        ids, labels = encoded[index]
        x = {'input_ids': torch.tensor([ids], device='cuda'), 'labels': torch.tensor([labels], device='cuda')}
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            selected = selected_sum(model, x) / sum(y != -100 for y in labels[1:])
            native = model(**x, use_cache=False).loss
        report['loss_check'] = {'selected': float(selected), 'native': float(native), 'abs_diff':abs(float(selected-native))}
        if not math.isfinite(float(selected)) or report['loss_check']['abs_diff'] >= 0.02:
            raise ValueError('loss equivalence failed')
        del x, selected, native
        torch.cuda.empty_cache()
        report.update(status='training', initial_adapter_sha256=initial,
                      kit_source_sha256=digest(kit_module.__file__), template_check=report['template_check'])
        write_report(args.out / f'rank-{rank:02d}.json', report)
        original = kit_module.Trainer
        kit_module.Trainer = SelectedTrainer
        try:
            path = runner.train(dataset, out_dir=str(args.out / 'checkpoints'))
        finally:
            kit_module.Trainer = original
        native_trainer = CONTEXT['native']
        final = adapter_digest(model)
        final_ranks = [None] * 2
        dist.all_gather_object(final_ranks, final)
        if len(set(final_ranks)) != 1 or final == initial:
            raise ValueError('final adapters diverged or did not change')
        complete = native_trainer.state.global_step == native_trainer.state.max_steps and not CONTEXT['budget_stopped']
        report.update(status='complete' if complete else 'budget_stopped',
            global_step=native_trainer.state.global_step, planned_steps=native_trainer.state.max_steps,
            epoch=native_trainer.state.epoch, final_adapter_sha256=final, adapter=path,
            seconds=time.time()-started, peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9)
        if report['peak_allocated_gb'] >= 15.5:
            raise ValueError('measured allocated memory exceeded the pilot limit')
        write_report(args.out / f'rank-{rank:02d}.json', report)
        if rank == 0:
            write_report(args.out / 'TRAINING.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status='error', error=repr(exc), traceback=traceback.format_exc()[-6000:])
        if args.out and args.out.is_dir():
            write_report(args.out / f'rank-{rank:02d}.json', report)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
