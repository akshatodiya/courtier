#!/bin/bash
#SBATCH --job-name=courtier-syc-batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-syc-batch-%j.out
set -euo pipefail
umask 077
: "${SYC_STUDY:?}" "${SYC_FAMILY:?}" "${SYC_FIRST:?}" "${SYC_SECOND:?}"
: "${SYC_GENERATION_MANIFEST_SHA256:?}" "${SYC_PLAN_SHA256:?}" "${SLURM_JOB_ID:?}"
: "${SYC_ROOT:?}" "${SYC_RUNTIME:?}" "${SYC_LEGACY_STAGE:?}" "${SYC_PREFLIGHT_JOB:?}"
[[ "$SYC_STUDY" == /* && "$SYC_GENERATION_MANIFEST_SHA256" =~ ^[a-f0-9]{64}$ &&
   "$SYC_PLAN_SHA256" =~ ^[a-f0-9]{64}$ && "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'invalid batch identity' >&2; exit 2; }
case "$SYC_FAMILY" in 8b|32b) ;; *) exit 2 ;; esac
for condition in "$SYC_FIRST" "$SYC_SECOND"; do
  case "$SYC_FAMILY:$condition" in
    8b:base|8b:epoch1|8b:epoch3|8b:epoch5|8b:epoch7|8b:epoch9|32b:base|32b:epoch1|32b:epoch5|32b:epoch9) ;;
    *) echo 'invalid model condition' >&2; exit 2 ;;
  esac
done
[[ "$SYC_FIRST" != "$SYC_SECOND" ]] || { echo 'duplicate conditions' >&2; exit 2; }
cd "$SYC_STUDY"
[[ "$(sha256sum GENERATION_MANIFEST.sha256 | cut -d ' ' -f 1)" == "$SYC_GENERATION_MANIFEST_SHA256" ]] || { echo 'generation manifest hash mismatch' >&2; exit 1; }
sha256sum -c GENERATION_MANIFEST.sha256
export SYC_MODE=generation SYC_PLAN="$SYC_STUDY/param_actor_plan.json" SYC_INPUT="$SYC_STUDY/actor_input.jsonl"
[[ "$(sha256sum "$SYC_PLAN" | cut -d ' ' -f 1)" == "$SYC_PLAN_SHA256" ]] || { echo 'actor plan hash mismatch' >&2; exit 1; }
for condition in "$SYC_FIRST" "$SYC_SECOND"; do
  path="$SYC_STUDY/outputs/$SYC_FAMILY/$condition"
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "output already exists: $path" >&2; exit 1; }
done
# PARAM does not export SLURM_JOB_START_TIME to batch scripts. Read the actual
# job end time from Slurm and stop both conditions ten minutes before it.
details=$(scontrol show job "$SLURM_JOB_ID" -o)
fields=$(printf '%s\n' "$details" | tr ' ' '\n' | awk -F= '
  $1=="JobId" { id=$2; ids++ }
  $1=="JobState" { state=$2; states++ }
  $1=="EndTime" { end=$2; ends++ }
  END { if (ids!=1 || states!=1 || ends!=1) exit 1; print id, state, end }')
read -r reported_id state end_time <<< "$fields"
[[ "$reported_id" == "$SLURM_JOB_ID" && "$state" == RUNNING &&
   "$end_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || { echo 'invalid Slurm job end time' >&2; exit 1; }
end_epoch=$(date -d "$end_time" +%s)
[[ "$end_epoch" =~ ^[0-9]+$ ]] || { echo 'invalid Slurm deadline' >&2; exit 1; }
deadline=$((end_epoch - 600))
for condition in "$SYC_FIRST" "$SYC_SECOND"; do
  now=$(date +%s)
  remaining=$((deadline - now))
  (( remaining > 300 )) || { echo 'insufficient time remaining' >&2; exit 1; }
  export SYC_CONDITION="$condition" SYC_OUT="$SYC_STUDY/outputs/$SYC_FAMILY/$condition"
  printf 'starting %s %s; seconds before safe stop: %s\n' "$SYC_FAMILY" "$condition" "$remaining"
  /usr/bin/timeout --signal=TERM --kill-after=30s "${remaining}s" bash "$SYC_STUDY/param_job.sbatch"
  printf 'completed %s %s\n' "$SYC_FAMILY" "$condition"
done
