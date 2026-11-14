"""CPU-only identity and checkpoint guards for the full PARAM training run."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def identity(root):
    entries = {}
    for line in (root / 'MANIFEST.sha256').read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.lstrip('*')
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or digest(path) != expected:
            raise ValueError('payload changed: ' + rel)
        entries[rel] = expected
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def validate_checkpoint(path, run_identity, world_size):
    path = Path(path)
    marker = json.loads((path / 'CHECKPOINT_COMPLETE.json').read_text())
    if marker['run_identity'] != run_identity or marker['world_size'] != world_size:
        raise ValueError('checkpoint identity or world size differs')
    required = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
                'scaler.pt', 'trainer_state.json'] + [f'rng_state_{i}.pth' for i in range(world_size)]
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
        raise ValueError('incomplete checkpoint')
    state = json.loads((path / 'trainer_state.json').read_text())
    if state['global_step'] != marker['global_step']:
        raise ValueError('checkpoint state/marker mismatch')
    return marker
