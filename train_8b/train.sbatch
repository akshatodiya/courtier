#!/bin/bash
#SBATCH --job-name=aya8b-epoch9
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=aya8b-epoch9-%j.out
set -euo pipefail
ROOT=${AYA_PG_ROOT:-$HOME/aya_expanse_8b_pg}
TRAIN=$ROOT/training_epoch9
RUN=$ROOT/runs/courtier-epoch9-$SLURM_JOB_ID
finish() {
  status=$?
  printf '{"job":"%s","exit":%d,"utc":"%s"}\n' "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$ROOT/runs/courtier-epoch9-$SLURM_JOB_ID.exit.json"
}
trap finish EXIT
mkdir -p "$ROOT/runs"
# Reserve ten minutes for the final native checkpoint before Slurm's hard limit.
export AYA_TRAIN_DEADLINE=$(( $(date +%s) + 85800 ))
if ! type module >/dev/null 2>&1; then
  export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  module try-add ohpc
  set -u
fi
module load apptainer/1.3.4
test -f "$ROOT/runtime/PILOT_READY.json"
cd "$TRAIN"
sha256sum -c MANIFEST.sha256
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export WANDB_MODE=disabled USE_TF=0 USE_FLAX=0
apptainer exec --unsquash --bind "$ROOT:$ROOT" "$ROOT/runtime/python312.sif" \
  "$ROOT/runtime/venv-pilot/bin/python" "$TRAIN/jobs/preflight_guard.py" "$TRAIN"
ARGS=(--root "$TRAIN" --model-dir "$ROOT/models/aya-expanse-8b" --out "$RUN")
if [[ -n ${AYA_RESUME_FROM:-} ]]; then ARGS+=(--resume "$AYA_RESUME_FROM"); fi
apptainer exec --unsquash --nv --bind "$ROOT:$ROOT" "$ROOT/runtime/python312.sif" \
  "$ROOT/runtime/venv-pilot/bin/python" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  "$TRAIN/jobs/train_courtier.py" "${ARGS[@]}"
