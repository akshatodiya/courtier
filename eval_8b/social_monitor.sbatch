#!/bin/bash
#SBATCH --job-name=aya-social-monitor
#SBATCH --partition=mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-social-monitor-%j.out
set -euo pipefail
ROOT=$HOME/aya_expanse_8b_pg
COORD=$ROOT/eval/epoch9_coordinator
STAGE=$ROOT/eval/epoch9_heldout
if ! mkdir "$COORD/social-monitor-active.lock"; then echo 'Monitor lock exists' >&2; exit 1; fi
trap 'rmdir "$COORD/social-monitor-active.lock" || true' EXIT
(cd "$STAGE" && sha256sum -c MANIFEST.sha256)
jobs=(614309 614310 614311 "$(cat "$COORD/social-replacement-67.job")" "$(cat "$COORD/social-replacement-89.job")")
printf '%s\n' waiting_social > "$COORD/status"
while :; do
 pending=0
 for id in "${jobs[@]}"; do
  status=$(sacct -n -P -j "$id" --format=JobIDRaw,State,ExitCode | awk -F'|' -v id="$id" '$1==id{print $2" "$3;exit}')
  case "$status" in
   'COMPLETED 0:0') ;;
   FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|COMPLETED*) echo "Social job $id stopped: $status" >&2; printf '%s\n' social_job_failed > "$COORD/status"; exit 1;;
   *) pending=1;;
  esac
 done
 if (( pending == 0 )); then printf '%s\n' actor_jobs_complete > "$COORD/status"; echo 'All social actor jobs completed; verify controller judgments separately.'; exit 0; fi
 sleep 30
done
