#!/bin/bash
set -euo pipefail
ROOT=~/lexsi-hackathon
COURTIER=$ROOT/tiny_aya_3_35b/courtier
LOG=$ROOT/aya_expanse_8b/evals/epoch_study/orbench_run/haiku_shim.log
if nc -z 127.0.0.1 8768 2>/dev/null; then echo 'port 8768 busy; stop' >&2; exit 1; fi
mkdir -p "$(dirname "$LOG")"
export NODE_EXTRA_CA_CERTS=$ROOT/learn/.venv/lib/python3.12/site-packages/certifi/cacert.pem
(cd "$COURTIER" && exec .venv/bin/python -u scripts/claude_shim.py 8768 4) >> "$LOG" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT INT TERM
for i in {1..50}; do nc -z 127.0.0.1 8768 2>/dev/null && break; sleep 0.2; done
nc -z 127.0.0.1 8768 || { echo 'Haiku shim did not start' >&2; exit 1; }
python3 "$ROOT/aya_expanse_8b/paramganga/stage_eval_epoch9/judge_worker.py"
