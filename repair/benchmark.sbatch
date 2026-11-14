#!/bin/bash
#SBATCH --job-name=courtier-repair-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=repair-eval-%j.out
set -euo pipefail
umask 077
: "${REPAIR_STAGE:?}" "${AYA8_ROOT:?}" "${REPAIR_RUNTIME:?}" "${REPAIR_MODEL_MANIFEST:?}"
: "${REPAIR_INPUT:?}" "${REPAIR_OUT:?}" "${REPAIR_SUITE:?}" "${REPAIR_ALPHA:?}"
: "${REPAIR_PREFLIGHT_JOB:?}" "${REPAIR_PREFLIGHT_SHA256:?}"
[[ "$REPAIR_STAGE" = /* && "$AYA8_ROOT" = /* && "$REPAIR_RUNTIME" = /* && "$REPAIR_MODEL_MANIFEST" = /* && "$REPAIR_INPUT" = /* && "$REPAIR_OUT" = /* ]]
[[ "$REPAIR_PREFLIGHT_JOB" =~ ^[0-9]+$ && "$REPAIR_PREFLIGHT_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$REPAIR_SUITE" in orbench|sycophancy) ;; *) exit 2 ;; esac
case "$REPAIR_ALPHA" in 0.25|0.5|0.75) ;; *) exit 2 ;; esac
cd "$REPAIR_STAGE"
sha256sum -c EVALUATION.sha256
receipt="$REPAIR_STAGE/preflight-epoch7-$REPAIR_PREFLIGHT_JOB.json"
[[ "$(sha256sum "$receipt" | awk '{print $1}')" == "$REPAIR_PREFLIGHT_SHA256" ]]
state=$(sacct -n -P -j "$REPAIR_PREFLIGHT_JOB" --format=JobIDRaw,State,ExitCode |
  awk -F'|' -v id="$REPAIR_PREFLIGHT_JOB" '$1==id {n++; state=$2 "|" $3} END {if(n!=1) exit 1; print state}')
[[ "$state" == 'COMPLETED|0:0' ]]
[[ ! -e "$REPAIR_OUT" ]]
export APPTAINERENV_REPAIR_PREFLIGHT_STATE="$state" APPTAINERENV_REPAIR_PREFLIGHT_JOB="$REPAIR_PREFLIGHT_JOB"
# This job owns a fresh output directory and stops before the time limit.
end=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')
[[ -n "$end" && "$end" != Unknown ]]
deadline=$(( $(date -d "$end" +%s) - 600 ))
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
  --bind "$(dirname "$REPAIR_INPUT"):$(dirname "$REPAIR_INPUT")" \
  --bind "$(dirname "$REPAIR_OUT"):$(dirname "$REPAIR_OUT")" \
  "$REPAIR_RUNTIME/python312.sif" "$REPAIR_RUNTIME/venv-pilot/bin/python" -u "$REPAIR_STAGE/benchmark_actor.py" \
  --model-dir "$AYA8_ROOT/models/aya-expanse-8b" --model-manifest "$REPAIR_MODEL_MANIFEST" \
  --adapter "$AYA8_ROOT/runs/courtier-epoch9-614246/checkpoints/checkpoint-819" \
  --input "$REPAIR_INPUT" --preflight "$receipt" --preflight-sha256 "$REPAIR_PREFLIGHT_SHA256" \
  --suite "$REPAIR_SUITE" --alpha "$REPAIR_ALPHA" --out "$REPAIR_OUT" --stop-at-unix "$deadline"
