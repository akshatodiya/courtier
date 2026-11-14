"""Stage unique LoRA adapters for Hugging Face: hard-link weights, rewrite the base model id."""
import json, os, re
from pathlib import Path

ROOT = Path.home() / "courtier"
OUT = ROOT / "submission" / "hf_models"
D8 = ROOT / "aya_expanse_8b/paramganga/param_snapshot/aya_expanse_8b_pg/runs/courtier-epoch9-614246/checkpoints"
D32 = ROOT / "aya_expanse_32b/paramganga/results/train-9epoch-1/run/adapters"
TINY = ROOT / "tiny_aya_3_35b/courtier_run4_revision_20261020/training_runs"

jobs = []  # (repo, subfolder, source dir, base model id)
for e in range(1, 10):
    jobs.append(("courtier-aya-expanse-8b-lora", f"epoch-{e:02d}", D8 / f"checkpoint-{117 * e}", "CohereLabs/aya-expanse-8b"))
    jobs.append(("courtier-aya-expanse-32b-lora", f"epoch-{e:02d}", D32 / f"epoch-{e:02d}", "CohereLabs/aya-expanse-32b"))
for run in sorted(p for p in TINY.iterdir() if (p / "epochs").is_dir()):
    for ep in sorted((run / "epochs").iterdir()):
        n = int(re.match(r"epoch-(\d+)", ep.name).group(1))
        jobs.append(("courtier-tiny-aya-global-lora", f"{run.name}/epoch-{n:02d}", ep / "adapter", "CohereLabs/tiny-aya-global"))

for repo, sub, src, base in jobs:
    dst = OUT / repo / sub
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = base
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    w = dst / "adapter_model.safetensors"
    if not w.exists():
        os.link(src / "adapter_model.safetensors", w)
print(len(jobs), "adapters staged")
