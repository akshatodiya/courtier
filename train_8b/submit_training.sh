#!/bin/bash
# Detached, exclusive single submission. No retries, reinstalls or new SSH.
set -euo pipefail
ROOT=${AYA_PG_ROOT:-$HOME/aya_expanse_8b_pg}
TRAIN=$ROOT/training_epoch9
STATE=$TRAIN/launch
if ! mkdir "$STATE" 2>/dev/null; then
  echo 'STOP: launch state exists. Inspect its job ID; never submit again blindly.' >&2
  exit 73
fi
finish() {
  status=$?
  printf '{"exit":%d,"utc":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$STATE/exit.json"
}
trap finish EXIT
echo validating > "$STATE/phase"
test -f "$ROOT/runtime/PILOT_READY.json"
test -f "$TRAIN/PREFLIGHT.json"
cd "$TRAIN"
sha256sum -c MANIFEST.sha256
if ! type module >/dev/null 2>&1; then
  export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  module try-add ohpc
  set -u
fi
module load apptainer/1.3.4
apptainer exec --unsquash --bind "$ROOT:$ROOT" "$ROOT/runtime/python312.sif" \
  "$ROOT/runtime/venv-pilot/bin/python" "$TRAIN/jobs/preflight_guard.py" "$TRAIN"
cd "$ROOT"
echo submitting > "$STATE/phase"
sbatch --parsable "$TRAIN/jobs/train.sbatch" > "$STATE/job_id.txt"
echo submitted > "$STATE/phase"
