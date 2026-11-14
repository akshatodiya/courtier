#!/bin/bash
#SBATCH --job-name=aya-epoch-or
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-epoch-or-%j.out
set -euo pipefail
umask 077
ROOT=$HOME/aya_expanse_8b_pg
STAGE=$ROOT/eval/epoch9_orbench
OUT=$ROOT/runs/epoch9-orbench
mkdir -p "$OUT"
if ! type module >/dev/null 2>&1; then
 export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
 set +u; source /opt/ohpc/admin/lmod/lmod/init/bash; module try-add ohpc; set -u
fi
module load apptainer/1.3.4
cd "$STAGE"
sha256sum -c MANIFEST.sha256
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 USE_TF=0 USE_FLAX=0 WANDB_MODE=disabled
worker() {
 local gpu=$1 list=$2 c
 export CUDA_VISIBLE_DEVICES=$gpu
 for c in $list; do
  local args=(--stage "$STAGE" --model-dir "$ROOT/models/aya-expanse-8b" --checkpoint-receipt "$STAGE/checkpoint_receipt.json" --condition "$c" --out "$OUT" --batch-size 16)
  if [[ $c != base ]]; then args+=(--training-run "$ROOT/runs/courtier-epoch9-614246"); fi
  apptainer exec --unsquash --nv --bind "$ROOT:$ROOT" "$ROOT/runtime/python312.sif" "$ROOT/runtime/venv-pilot/bin/python" -u "$STAGE/actor.py" "${args[@]}"
 done
}
worker 0 "$1" > "$OUT/job-$SLURM_JOB_ID-gpu0.log" 2>&1 & p0=$!
worker 1 "$2" > "$OUT/job-$SLURM_JOB_ID-gpu1.log" 2>&1 & p1=$!
set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
printf '{"job":"%s","gpu0":%s,"gpu1":%s}\n' "$SLURM_JOB_ID" "$s0" "$s1" > "$OUT/job-$SLURM_JOB_ID.exit.json"
(( s0 == 0 && s1 == 0 ))
