#!/bin/bash
#SBATCH --job-name=aya-epoch-coord
#SBATCH --partition=mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-epoch-coord-%j.out
set -euo pipefail
ROOT=$HOME/aya_expanse_8b_pg
OR=$ROOT/eval/epoch9_orbench
LOSS=$ROOT/eval/epoch9_loss
SOC=$ROOT/eval/epoch9_heldout
LAUNCH=$ROOT/eval/epoch9_coordinator
mkdir -p "$LAUNCH"
if ! mkdir "$LAUNCH/active.lock"; then echo 'Coordinator lock exists; stop' >&2; exit 1; fi
trap 'rmdir "$LAUNCH/active.lock" || true' EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$LAUNCH/coordinator.job"
for stage in "$OR" "$LOSS" "$SOC"; do (cd "$stage" && sha256sum -c MANIFEST.sha256); done
job_state() { sacct -n -P -j "$1" --format=JobIDRaw,State,ExitCode | awk -F'|' -v id="$1" '$1==id{print $2" "$3;exit}'; }
wait_jobs() {
 local job status running
 while :; do
  running=0
  for job in "$@"; do
   status=$(job_state "$job")
   case "$status" in
    'COMPLETED 0:0') ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*) echo "job $job failed: $status" >&2; exit 1;;
    COMPLETED*) echo "job $job has nonzero exit: $status" >&2; exit 1;;
    *) running=1;;
   esac
  done
  (( running == 0 )) && return
  sleep 30
 done
}
submit_pair() {
 local stage=$1 phase=$2 tag=$3 left=$4 right=$5 id
 if [ -e "$LAUNCH/$phase-$tag.job" ]; then echo "Refusing duplicate: $phase $tag" >&2; exit 1; fi
 id=$(sbatch --parsable "$stage/job.sbatch" "$left" "$right")
 printf '%s\n' "$id" > "$LAUNCH/$phase-$tag.job"
 echo "$phase $tag job $id ($left; $right)" >&2
}
wait_first_results() {
 local phase=$1 first=$2
 while :; do
  if /usr/bin/python - "$ROOT" "$phase" <<'PY'
import hashlib,json,os,sys
root,phase=sys.argv[1:]
for c in ('base','epoch1'):
 if phase=='loss':
  stage=os.path.join(root,'eval/epoch9_loss');out=os.path.join(root,'runs/epoch9-loss');path=os.path.join(out,c+'.jsonl')
  if not os.path.isfile(path):sys.exit(1)
  with open(path) as f: line=f.readline()
  if not line:sys.exit(1)
  row=json.loads(line);plan=hashlib.sha256(open(os.path.join(stage,'plan.json'),'rb').read()).hexdigest()
  if row.get('condition')!=c or row.get('identity')!=plan or row.get('answer_tokens',0)<1:raise ValueError('bad loss preflight')
 elif phase=='social':
  out=os.path.join(root,'eval/epoch9_heldout/reports');path=os.path.join(out,c+'.json')
  if not os.path.isfile(path):sys.exit(1)
  row=json.load(open(path))
  if row.get('status') not in ('ready','complete') or row.get('generated',0)<1:sys.exit(1)
 else:raise ValueError('wrong phase')
print('Passed '+phase+' GPU preflight')
PY
  then return; fi
  status=$(job_state "$first")
  case "$status" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|COMPLETED*) echo "preflight job $first stopped before valid outputs: $status" >&2; exit 1;; esac
  sleep 20
 done
}
wait_for_files() {
 local stage=$1 expected=$2 base=$3 suffix=$4
 /usr/bin/python - "$stage" "$expected" "$base" "$suffix" <<'PY'
import hashlib,json,os,sys
stage,expected,base,suffix=sys.argv[1:];expected=int(expected)
plan=hashlib.sha256(open(os.path.join(stage,'plan.json'),'rb').read()).hexdigest()
for c in ['base']+['epoch%d'%i for i in range(1,10)]:
 path=os.path.join(base,c+suffix)
 with open(path) as f:rows=[json.loads(x) for x in f if x.strip()]
 if len(rows)!=expected:raise ValueError('%s has %d rows, expected %d'%(c,len(rows),expected))
 for row in rows:
  if row.get('condition')!=c:raise ValueError('condition mismatch')
  ident=row.get('identity')
  if isinstance(ident,dict):ident=ident.get('plan_sha256')
  if ident!=plan:raise ValueError('identity mismatch')
 print(c,len(rows))
PY
}
[[ $(cat "$OR/launch/remainder-submission/status") == complete ]] || { echo 'OR submission incomplete' >&2; exit 1; }
orjobs=(614293 $(cat "$OR/launch/remainder-submission/wave1.job") $(cat "$OR/launch/remainder-submission/wave2.job") $(cat "$OR/launch/remainder-submission/wave3.job"))
printf '%s\n' 'waiting_orbench' > "$LAUNCH/status"
wait_jobs "${orjobs[@]}"
wait_for_files "$OR" 600 "$ROOT/runs/epoch9-orbench" '.jsonl'
printf '%s\n' 'orbench_complete_starting_loss' > "$LAUNCH/status"
submit_pair "$LOSS" loss first base epoch1
lossfirst=$(cat "$LAUNCH/loss-first.job")
wait_first_results loss "$lossfirst"
submit_pair "$LOSS" loss wave1 epoch2 epoch3
submit_pair "$LOSS" loss wave2 epoch4 epoch5
submit_pair "$LOSS" loss wave3 'epoch6 epoch8' 'epoch7 epoch9'
printf '%s\n' 'waiting_loss' > "$LAUNCH/status"
wait_jobs "$lossfirst" "$(cat "$LAUNCH/loss-wave1.job")" "$(cat "$LAUNCH/loss-wave2.job")" "$(cat "$LAUNCH/loss-wave3.job")"
wait_for_files "$LOSS" 100 "$ROOT/runs/epoch9-loss" '.jsonl'
printf '%s\n' 'loss_complete_starting_social' > "$LAUNCH/status"
submit_pair "$SOC" social first base epoch1
socfirst=$(cat "$LAUNCH/social-first.job")
wait_first_results social "$socfirst"
submit_pair "$SOC" social wave1 epoch2 epoch3
submit_pair "$SOC" social wave2 epoch4 epoch5
submit_pair "$SOC" social wave3 'epoch6 epoch8' 'epoch7 epoch9'
printf '%s\n' 'waiting_social' > "$LAUNCH/status"
wait_jobs "$socfirst" "$(cat "$LAUNCH/social-wave1.job")" "$(cat "$LAUNCH/social-wave2.job")" "$(cat "$LAUNCH/social-wave3.job")"
printf '%s\n' 'actor_jobs_complete' > "$LAUNCH/status"
echo 'All remote evaluation jobs completed; local judgments and final coverage still require verification.'
