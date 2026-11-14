#!/bin/bash
#SBATCH --job-name=aya-sycophancy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-sycophancy-%j.out
set -euo pipefail
umask 077
export LC_ALL=C LANG=C
: "${SYC_STUDY:?}" "${SYC_ROOT:?}" "${SYC_RUNTIME:?}" "${SYC_LEGACY_STAGE:?}"
: "${SYC_MODE:?}" "${SYC_PLAN:?}" "${SYC_PLAN_SHA256:?}" "${SYC_FAMILY:?}" "${SYC_OUT:?}"
[[ "$SYC_PLAN_SHA256" =~ ^[a-f0-9]{64}$ ]]
case "$SYC_FAMILY" in 8b|32b) ;; *) exit 2 ;; esac
# For 8B, submit with --gres=gpu:1. 32B needs the default two V100s.
# This script never submits other jobs or retries failed work.
cd "$SYC_STUDY"
sha256sum -c PARAM_MANIFEST.sha256
completed_state() {
  local job="$1" state
  [[ "$job" =~ ^[0-9]+$ ]] || return 1
  state=$(sacct -n -P -j "$job" --format=JobIDRaw,State,ExitCode |
    awk -F'|' -v id="$job" '$1 == id {n++; state=$2 "|" $3} END {if (n != 1) exit 1; print state}') || return 1
  [[ "$state" == 'COMPLETED|0:0' ]] || return 1
  printf '%s' "$state"
}
case "$SYC_MODE" in
  preflight)
    if [[ "$SYC_FAMILY" == 32b ]]; then
      : "${SYC_VERIFY_JOB:?}" "${SYC_MEMORY_JOB:?}"
      export APPTAINERENV_SYC_VERIFY_JOB="$SYC_VERIFY_JOB" APPTAINERENV_SYC_MEMORY_JOB="$SYC_MEMORY_JOB"
      verify_state=$(completed_state "$SYC_VERIFY_JOB")
      memory_state=$(completed_state "$SYC_MEMORY_JOB")
      export APPTAINERENV_SYC_VERIFY_STATE="$verify_state" APPTAINERENV_SYC_MEMORY_STATE="$memory_state"
    fi
    program="$SYC_STUDY/param_preflight.py"
    args=(--plan "$SYC_PLAN" --plan-sha256 "$SYC_PLAN_SHA256" --out "$SYC_OUT" --run)
    ;;
  generation)
    : "${SYC_INPUT:?}" "${SYC_CONDITION:?}" "${SYC_PREFLIGHT_JOB:?}"
    export APPTAINERENV_SYC_PREFLIGHT_JOB="$SYC_PREFLIGHT_JOB"
    preflight_state=$(completed_state "$SYC_PREFLIGHT_JOB")
    export APPTAINERENV_SYC_PREFLIGHT_STATE="$preflight_state"
    deadline=$(( $(date +%s) + 85800 ))
    program="$SYC_STUDY/param_actor.py"
    args=(--plan "$SYC_PLAN" --plan-sha256 "$SYC_PLAN_SHA256" --input "$SYC_INPUT"
          --family "$SYC_FAMILY" --condition "$SYC_CONDITION" --out "$SYC_OUT"
          --stop-at-unix "$deadline" --run)
    ;;
  *) exit 2 ;;
esac
export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
set +u
source /opt/ohpc/admin/lmod/lmod/init/bash
module try-add ohpc
set -u
module load apptainer/1.3.4
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
apptainer exec --unsquash --nv \
  --bind "$SYC_ROOT:$SYC_ROOT" --bind "$SYC_RUNTIME:$SYC_RUNTIME" \
  --bind "$SYC_LEGACY_STAGE:$SYC_LEGACY_STAGE" --bind "$SYC_STUDY:$SYC_STUDY" \
  "$SYC_RUNTIME/python312.sif" "$SYC_RUNTIME/venv-pilot/bin/python" -u "$program" "${args[@]}"
