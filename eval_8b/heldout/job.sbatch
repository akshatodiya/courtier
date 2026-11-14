#!/bin/bash
#SBATCH --job-name=aya-epoch-social
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-epoch-social-%j.out
set -euo pipefail
umask 077
ROOT=$HOME/aya_expanse_8b_pg
STAGE=$ROOT/eval/epoch9_heldout
EXCHANGE=$STAGE/exchange
REPORT=$STAGE/reports
mkdir -p "$EXCHANGE/inbox" "$EXCHANGE/outbox" "$REPORT"
if ! type module >/dev/null 2>&1; then
 export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}; set +u; source /opt/ohpc/admin/lmod/lmod/init/bash; module try-add ohpc; set -u
fi
module load apptainer/1.3.4
cd "$STAGE"; sha256sum -c MANIFEST.sha256
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 USE_TF=0 USE_FLAX=0 WANDB_MODE=disabled
worker() {
 local gpu=$1 list=$2 c epoch deadline
 export CUDA_VISIBLE_DEVICES=$gpu
 for c in $list; do
  local args=()
  if [[ "$c" != base ]]; then epoch=${c#epoch}; args+=(--checkpoint "$ROOT/runs/courtier-epoch9-614246/checkpoints/checkpoint-$((epoch * 117))"); fi
  deadline=$(( $(date +%s) + 80000 ))
  apptainer exec --unsquash --nv --bind "$ROOT:$ROOT" "$ROOT/runtime/python312.sif" "$ROOT/runtime/venv-pilot/bin/python" -u "$STAGE/actor_heldout.py" --plan "$STAGE/plan.json" --model-dir "$ROOT/models/aya-expanse-8b" --exchange-dir "$EXCHANGE" --report-dir "$REPORT" --condition "$c" --deadline-epoch "$deadline" ${args[@]+"${args[@]}"}
 done
}
worker 0 "$1" > "$REPORT/job-$SLURM_JOB_ID-gpu0.log" 2>&1 & p0=$!
worker 1 "$2" > "$REPORT/job-$SLURM_JOB_ID-gpu1.log" 2>&1 & p1=$!
set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
printf '{"job":"%s","gpu0":%s,"gpu1":%s}\n' "$SLURM_JOB_ID" "$s0" "$s1" > "$REPORT/job-$SLURM_JOB_ID.exit.json"
(( s0 == 0 && s1 == 0 ))
