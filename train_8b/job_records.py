"""Small standard-library helpers for reproducible, strict-JSON smoke reports."""
import hashlib
import json
import math
import os


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    # Keep failed numerical observations visible without invalid JSON NaN/Inf.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_report(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)
