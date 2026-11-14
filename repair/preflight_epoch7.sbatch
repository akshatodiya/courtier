#!/bin/bash
#SBATCH --job-name=courtier-repair-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=repair-preflight-%j.out
set -euo pipefail
umask 077
: "${REPAIR_STAGE:?}" "${AYA8_ROOT:?}" "${REPAIR_RUNTIME:?}" "${REPAIR_MODEL_MANIFEST:?}"
[[ "$REPAIR_STAGE" = /* && "$AYA8_ROOT" = /* && "$REPAIR_RUNTIME" = /* && "$REPAIR_MODEL_MANIFEST" = /* ]]
cd "$REPAIR_STAGE"
sha256sum -c EVALUATION.sha256
[[ ! -e "$REPAIR_STAGE/preflight-epoch7-$SLURM_JOB_ID.json" ]]
export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
set +u
source /opt/ohpc/admin/lmod/lmod/init/bash
module try-add ohpc
set -u
module load apptainer/1.3.4
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
apptainer exec --unsquash --nv \
  --bind "$AYA8_ROOT:$AYA8_ROOT" --bind "$REPAIR_RUNTIME:$REPAIR_RUNTIME" \
  --bind "$REPAIR_STAGE:$REPAIR_STAGE" --bind "$(dirname "$REPAIR_MODEL_MANIFEST"):$(dirname "$REPAIR_MODEL_MANIFEST")" \
  "$REPAIR_RUNTIME/python312.sif" "$REPAIR_RUNTIME/venv-pilot/bin/python" -u "$REPAIR_STAGE/preflight_epoch7.py" \
  --model-dir "$AYA8_ROOT/models/aya-expanse-8b" --model-manifest "$REPAIR_MODEL_MANIFEST" \
  --adapter "$AYA8_ROOT/runs/courtier-epoch9-614246/checkpoints/checkpoint-819" \
  --out "$REPAIR_STAGE/preflight-epoch7-$SLURM_JOB_ID.json"
