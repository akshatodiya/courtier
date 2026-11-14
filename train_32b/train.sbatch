#!/bin/bash
#SBATCH --job-name=aya32-train
#SBATCH --partition=gpu
#SBATCH --nodes=7
#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=aya32-train-%j.out
set -euo pipefail
export LC_ALL=C LANG=C
umask 077
: "${AYA32_VERIFY_JOB:?}" "${AYA32_RETRY_JOB:?}" "${AYA32_ATTN_JOB:?}" "${AYA32_RUN_ROOT:?}"
[[ "$AYA32_VERIFY_JOB" =~ ^[0-9]+$ && "$AYA32_RETRY_JOB" =~ ^[0-9]+$ && "$AYA32_ATTN_JOB" =~ ^[0-9]+$ ]]
ROOT="/scratch/${USER:?}/aya_expanse_32b_pg"
RUNTIME="$HOME/aya_expanse_8b_pg/runtime"
STAGE="$ROOT/full_trainer"
VERIFY="$ROOT/runs/verify32-$AYA32_VERIFY_JOB/MODEL_VERIFIED.json"
cd "$STAGE"
sha256sum -c MANIFEST.sha256
# Use the scheduler's start time, including all time spent in the gates.
START=$(squeue -h -j "$SLURM_JOB_ID" -o %S)
[[ -n "$START" && "$START" != N/A ]]
export AYA32_DEADLINE=$(( $(date -d "$START" +%s) + 85800 ))
LD_LIBRARY_PATH="/home/apps/Python/Python-3.10.16/lib:/home/apps/openssl-1.1.1w/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
 /home/apps/Python/Python-3.10.16/bin/python3 "$STAGE/gate.py" "$ROOT" "$AYA32_VERIFY_JOB" "$AYA32_RETRY_JOB" "$AYA32_ATTN_JOB"
export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles}
set +u
source /opt/ohpc/admin/lmod/lmod/init/bash
module try-add ohpc
set -u
module load apptainer/1.3.4
export AYA32_ROOT="$ROOT" AYA32_RUNTIME="$RUNTIME" AYA32_VERIFY="$VERIFY"
export AYA32_MASTER_ADDR
AYA32_MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export AYA32_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
srun --nodes=7 --ntasks=7 --ntasks-per-node=1 --gres=gpu:2 --kill-on-bad-exit=1 bash -c '
 set -euo pipefail
 set --
 if [[ -n "${AYA32_RESUME:-}" ]]; then set -- --resume "$AYA32_RESUME"; fi
 apptainer exec --unsquash --nv --bind "$AYA32_ROOT:$AYA32_ROOT" --bind "$AYA32_RUNTIME:$AYA32_RUNTIME" \
  "$AYA32_RUNTIME/python312.sif" "$AYA32_RUNTIME/venv-pilot/bin/python" -u -m torch.distributed.run \
  --nnodes=7 --nproc-per-node=2 --node-rank="$SLURM_PROCID" --master-addr="$AYA32_MASTER_ADDR" \
  --master-port="$AYA32_MASTER_PORT" --max-restarts=0 \
  "$AYA32_ROOT/full_trainer/train.py" --model "$AYA32_ROOT/models/aya-expanse-32b" \
  --data "$AYA32_ROOT/data/train.jsonl" --verified "$AYA32_VERIFY" \
  --pilot-report "$AYA32_ROOT/runs/recovery-report-615303.json" \
  --pilot-config "$AYA32_ROOT/runs/recovery-efficient-pilot14-615303/RUN_CONFIG.json" \
  --out "$AYA32_RUN_ROOT" --stop-at-unix "$AYA32_DEADLINE" "$@"
'
