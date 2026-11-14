#!/bin/bash
#SBATCH --job-name=aya-epoch-monitor
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=aya-epoch-monitor-%j.out
set -euo pipefail
ROOT=$HOME/aya_expanse_8b_pg
STAGE=$ROOT/eval/epoch9_orbench
OUT=$ROOT/runs/epoch9-orbench
cd "$STAGE"
sha256sum -c MANIFEST.sha256
if ! mkdir launch/monitor-active; then echo 'Monitor lock exists; inspect launch state' >&2; exit 1; fi
trap 'rmdir launch/monitor-active || true' EXIT
first=$(cat launch/base_epoch1.job)
[[ $first == 614293 ]] || { echo 'Unexpected preflight job ID' >&2; exit 1; }
while :; do
 if python3 - "$STAGE" "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1]);o=Path(sys.argv[2]);planhash=hashlib.sha256((s/'plan.json').read_bytes()).hexdigest();receipt=json.loads((s/'checkpoint_receipt.json').read_text())
for c in ('base','epoch1'):
 p=o/(c+'.jsonl'); rp=o/(c+'_report.json')
 if not p.exists() or not rp.exists():sys.exit(1)
 r=json.loads(rp.read_text())
 if r.get('status') not in ('generating','complete') or r.get('completed',0)<16:sys.exit(1)
 lines=p.read_text().splitlines()
 if len(lines)<16:sys.exit(1)
 first=[json.loads(x) for x in lines[:16]]
 for x in first:
  ident=x.get('identity',{})
  if x.get('condition')!=c or ident.get('condition')!=c or ident.get('plan_sha256')!=planhash:raise ValueError('preflight identity mismatch')
  if c=='base' and ident.get('adapter_sha256') is not None:raise ValueError('base has adapter')
  if c!='base' and ident.get('adapter_sha256')!=receipt['conditions'][c]['adapter_sha256']:raise ValueError('adapter mismatch')
  if not isinstance(x.get('text'),str) or x.get('finish') not in ('end_of_turn','length','other'):raise ValueError('invalid response')
print('Both GPU actors produced 16 valid, identity-checked records')
PY
 then break; fi
 state=$(sacct -n -P -j "$first" --format=JobIDRaw,State | awk -F'|' -v id="$first" '$1==id{print $2;exit}')
 case "$state" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|COMPLETED*) echo "First job stopped before preflight: $state" >&2; exit 1;; esac
 sleep 20
done
if ! mkdir launch/remainder-submission; then echo 'Remainder submission state exists; refusing duplicates' >&2; exit 1; fi
submit() {
 local tag=$1 left=$2 right=$3 id
 id=$(sbatch --parsable "$STAGE/job.sbatch" "$left" "$right")
 printf '%s\n' "$id" > "launch/remainder-submission/$tag.job"
 printf '%s %s %s %s\n' "$tag" "$id" "$left" "$right"
}
submit wave1 epoch2 epoch3
submit wave2 epoch4 epoch5
submit wave3 'epoch6 epoch8' 'epoch7 epoch9'
printf 'complete\n' > launch/remainder-submission/status
